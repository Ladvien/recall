//! Vector memory over Qdrant with a local ONNX embedder.
//!
//! A memory is an [`Item`]: a namespace (whose it is), a kind (what sort of thing), a key
//! (what makes it the same memory when written again), the text, a timestamp and a
//! private flag, plus free-form metadata. Its point id is a UUIDv5 over
//! `namespace\u{1f}kind\u{1f}key`, so re-ingesting is idempotent and a human can re-derive
//! an id from what they know about the memory.
//!
//! The embedder is a trait: [`Onnx`] runs `BAAI/bge-small-en-v1.5` (384-d, CLS pooling,
//! cosine) in process through ONNX Runtime, [`Ollama`] asks a running Ollama for
//! embeddings over HTTP. [`Store`] talks to Qdrant's REST API with a plain HTTP client —
//! no gRPC, no generated code — and creates its collection and payload indexes on open.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Which side of retrieval a text is on. bge's documented retrieval instruction is
/// prefixed to a query and never to a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Document,
    Query,
}

/// The bge retrieval instruction, from the model card.
const QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";

pub trait Embed: Send + Sync {
    fn dim(&self) -> usize;
    fn name(&self) -> &str;
    fn embed(&self, texts: &[String], role: Role) -> Result<Vec<Vec<f32>>>;
}

/// `BAAI/bge-small-en-v1.5` (or any BERT-shaped encoder with `input_ids`,
/// `attention_mask`, `token_type_ids` in and `last_hidden_state` out), CLS-pooled and
/// L2-normalised, in this process.
pub struct Onnx {
    session: parking_lot::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    name: String,
}

impl Onnx {
    /// 512 tokens is the model's window; longer texts are truncated, which for a memory
    /// is the right failure.
    const MAX_TOKENS: usize = 512;

    pub fn load(model: &Path, tokenizer: &Path, cuda: bool) -> Result<Self> {
        anyhow::ensure!(model.is_file(), "embedding model {} not found", model.display());
        anyhow::ensure!(tokenizer.is_file(), "tokenizer {} not found", tokenizer.display());
        let mut builder = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("ort session builder: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!("ort intra threads: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ort optimization level: {e}"))?;
        if cuda {
            #[cfg(feature = "cuda")]
            {
                builder = builder
                    .with_execution_providers([ort::ep::CUDA::default().with_device_id(0).build()])
                    .map_err(|e| anyhow::anyhow!("ort cuda provider: {e}"))?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("recall built without the `cuda` feature; embedding on the CPU");
            }
        }
        let session = builder
            .commit_from_file(model)
            .map_err(|e| anyhow::anyhow!("loading embedding model {}: {e}", model.display()))?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer)
            .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", tokenizer.display()))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: Self::MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;
        let name = model
            .parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "onnx".into());
        let mut this = Self { session: parking_lot::Mutex::new(session), tokenizer, dim: 0, name };
        // The dimension is what the graph says it is, learned from one warm-up pass,
        // which also pays the first-run cost before a turn does.
        let warm = this.embed(&["hello".to_string()], Role::Document)?;
        this.dim = warm.first().map(Vec::len).unwrap_or(0);
        anyhow::ensure!(this.dim > 0, "embedding model produced an empty vector");
        Ok(this)
    }

    fn run_one(&self, text: &str) -> Result<Vec<f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i64::from(i)).collect();
        let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i64::from(i)).collect();
        let types: Vec<i64> = enc.get_type_ids().iter().map(|&i| i64::from(i)).collect();
        let n = ids.len();
        let input_ids = ort::value::Tensor::<i64>::from_array(([1, n], ids.into_boxed_slice()))?;
        let attention = ort::value::Tensor::<i64>::from_array(([1, n], mask.into_boxed_slice()))?;
        let token_types = ort::value::Tensor::<i64>::from_array(([1, n], types.into_boxed_slice()))?;
        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention,
                "token_type_ids" => token_types,
            ])
            .map_err(|e| anyhow::anyhow!("embedding inference: {e}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("embedding output: {e}"))?;
        // `[1, tokens, dim]`: the CLS row is the first `dim` floats.
        let dim = *shape.last().context("embedding output has no shape")? as usize;
        anyhow::ensure!(data.len() >= dim && dim > 0, "embedding output too small");
        let mut v: Vec<f32> = data[..dim].to_vec();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(v)
    }
}

impl Embed for Onnx {
    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn embed(&self, texts: &[String], role: Role) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|t| match role {
                Role::Query => self.run_one(&format!("{QUERY_INSTRUCTION}{t}")),
                Role::Document => self.run_one(t),
            })
            .collect()
    }
}

/// A running Ollama's `/api/embed`. Blocking on purpose: `Embed` is synchronous so the
/// in-process embedder can be called from anywhere, and a caller that is on an async
/// runtime wraps either in `spawn_blocking`.
pub struct Ollama {
    endpoint: String,
    model: String,
    client: reqwest::blocking::Client,
    dim: parking_lot::Mutex<usize>,
}

impl Ollama {
    pub fn new(endpoint: &str, model: &str) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client");
        Self { endpoint: endpoint.trim_end_matches('/').to_string(), model: model.to_string(), client, dim: parking_lot::Mutex::new(0) }
    }
}

impl Embed for Ollama {
    fn dim(&self) -> usize {
        *self.dim.lock()
    }

    fn name(&self) -> &str {
        &self.model
    }

    fn embed(&self, texts: &[String], role: Role) -> Result<Vec<Vec<f32>>> {
        let input: Vec<String> = texts
            .iter()
            .map(|t| match role {
                Role::Query => format!("{QUERY_INSTRUCTION}{t}"),
                Role::Document => t.clone(),
            })
            .collect();
        let v: Value = self
            .client
            .post(format!("{}/api/embed", self.endpoint))
            .json(&json!({ "model": self.model, "input": input }))
            .send()?
            .error_for_status()?
            .json()?;
        let out: Vec<Vec<f32>> = serde_json::from_value(v["embeddings"].clone()).context("ollama embed answer")?;
        if let Some(first) = out.first() {
            *self.dim.lock() = first.len();
        }
        Ok(out)
    }
}

/// No embedder: for a store that is only counted or pruned (a CLI with no model loaded).
/// `search` and `upsert` fail; `open` on an absent collection fails too, because it
/// cannot know the dimension.
pub struct NoEmbed;

impl Embed for NoEmbed {
    fn dim(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "none"
    }

    fn embed(&self, _: &[String], _: Role) -> Result<Vec<Vec<f32>>> {
        anyhow::bail!("this store was opened without an embedder")
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub qdrant_url: String,
    pub collection: String,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Item {
    pub namespace: String,
    pub kind: String,
    pub key: String,
    pub text: String,
    pub ts_unix_ms: i64,
    pub private: bool,
    #[serde(default)]
    pub meta: BTreeMap<String, String>,
}

impl Item {
    /// The point id: UUIDv5 in `NAMESPACE_OID` over `namespace\u{1f}kind\u{1f}key`.
    pub fn id(&self) -> uuid::Uuid {
        uuid::Uuid::new_v5(
            &uuid::Uuid::NAMESPACE_OID,
            format!("{}\u{1f}{}\u{1f}{}", self.namespace, self.kind, self.key).as_bytes(),
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct Query {
    pub namespace: Option<String>,
    pub text: String,
    pub top_k: usize,
    pub min_score: f32,
    pub include_private: bool,
    pub kinds: Vec<String>,
    /// `ts_unix_ms` window, `[since, until)`. A question that names a time ("what did I
    /// tell you last weekend") is answered by the window, and similarity only ranks
    /// inside it.
    pub since_unix_ms: Option<i64>,
    pub until_unix_ms: Option<i64>,
    /// Only memories made on this device (`meta.device_id`).
    pub device_id: Option<String>,
}

impl Query {
    /// The filter clauses this query adds beyond its vector, in Qdrant's `must` shape.
    fn must(&self) -> Vec<Value> {
        let mut must: Vec<Value> = Vec::new();
        if let Some(ns) = &self.namespace {
            must.push(json!({ "key": "namespace", "match": { "value": ns } }));
        }
        if !self.include_private {
            must.push(json!({ "key": "private", "match": { "value": false } }));
        }
        if !self.kinds.is_empty() {
            must.push(json!({ "key": "kind", "match": { "any": self.kinds } }));
        }
        if let Some(d) = &self.device_id {
            must.push(json!({ "key": "device_id", "match": { "value": d } }));
        }
        if self.since_unix_ms.is_some() || self.until_unix_ms.is_some() {
            let mut range = serde_json::Map::new();
            if let Some(s) = self.since_unix_ms {
                range.insert("gte".into(), json!(s));
            }
            if let Some(u) = self.until_unix_ms {
                range.insert("lt".into(), json!(u));
            }
            must.push(json!({ "key": "ts_unix_ms", "range": range }));
        }
        must
    }
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub score: f32,
    pub item: Item,
}

#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub points: u64,
    pub dim: usize,
    pub collection: String,
    pub embedder: String,
}

/// The two Qdrant filters that decide what a retention prune may delete: made before the
/// window, and not recalled inside it. One function, so every caller of the curve agrees
/// about what "old" means — a `must`/`must_not` pair spelled at each call site is how the
/// household prune came to delete memories `server memory prune` spared (#136).
pub fn retention_filter(ts_unix_ms: i64) -> (Vec<Value>, Vec<Value>) {
    (
        vec![json!({ "key": "ts_unix_ms", "range": { "lt": ts_unix_ms } })],
        vec![json!({ "key": "last_recalled_ms", "range": { "gte": ts_unix_ms } })],
    )
}

/// An item read back from a point's payload: the six fixed fields, and every other
/// string-valued key as `meta`. Integer payload values a later write added (a recall
/// count, a stamp) are carried as their decimal text, so a reader can parse them back.
fn item_of(p: &Value) -> Option<Item> {
    let mut meta = BTreeMap::new();
    if let Some(o) = p.as_object() {
        for (k, v) in o {
            if !matches!(k.as_str(), "namespace" | "kind" | "key" | "text" | "ts_unix_ms" | "private") {
                match v {
                    Value::String(s) => {
                        meta.insert(k.clone(), s.clone());
                    }
                    Value::Number(n) => {
                        meta.insert(k.clone(), n.to_string());
                    }
                    _ => {}
                }
            }
        }
    }
    Some(Item {
        namespace: p["namespace"].as_str()?.to_string(),
        kind: p["kind"].as_str().unwrap_or("").to_string(),
        key: p["key"].as_str().unwrap_or("").to_string(),
        text: p["text"].as_str()?.to_string(),
        ts_unix_ms: p["ts_unix_ms"].as_i64().unwrap_or(0),
        private: p["private"].as_bool().unwrap_or(false),
        meta,
    })
}

/// Where a search's milliseconds went: the query embedding, then the Qdrant hop.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Phases {
    pub embed_ms: f64,
    pub hop_ms: f64,
}

impl Phases {
    pub fn total_ms(&self) -> f64 {
        self.embed_ms + self.hop_ms
    }

    /// Which half, in the fewest words a log field can carry: whichever is over half the
    /// total. `even` when neither is, which is the normal case and worth saying plainly so
    /// a reader is not left inferring it from two numbers.
    pub fn cost(&self) -> &'static str {
        let total = self.total_ms();
        match (self.embed_ms > total / 2.0, self.hop_ms > total / 2.0) {
            (true, _) => "embed",
            (_, true) => "qdrant",
            _ => "even",
        }
    }
}

pub struct Store {
    cfg: Config,
    client: reqwest::Client,
    embed: Arc<dyn Embed>,
}

impl Store {
    /// Create the collection (vector size from the embedder, cosine) and the payload
    /// indexes if absent, then hand back the store.
    pub async fn open(cfg: Config, embed: Arc<dyn Embed>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(cfg.timeout_ms))
            .build()
            .context("qdrant http client")?;
        let this = Self { cfg, client, embed };
        this.ensure_collection().await?;
        Ok(this)
    }

    /// One send, retried once on a transport error: a keep-alive connection Qdrant closed
    /// between two calls fails the next request before any byte is answered, and that is
    /// not the request's fault. A request that was answered is never retried.
    async fn send(&self, req: reqwest::RequestBuilder) -> Result<reqwest::Response> {
        let again = req.try_clone();
        match req.send().await {
            Ok(r) => Ok(r),
            Err(e) if (e.is_connect() || e.is_request()) && again.is_some() => {
                tracing::debug!("recall: retrying after a transport error: {e}");
                again.expect("checked").send().await.context("qdrant unreachable")
            }
            Err(e) => Err(e).context("qdrant unreachable"),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}/collections/{}{path}", self.cfg.qdrant_url.trim_end_matches('/'), self.cfg.collection)
    }

    async fn ensure_collection(&self) -> Result<()> {
        let exists = self.send(self.client.get(self.url(""))).await?.status().is_success();
        if !exists {
            let dim = self.embed.dim();
            anyhow::ensure!(dim > 0, "the embedder has no dimension yet");
            let resp = self
                .send(self.client.put(self.url("")).json(&json!({ "vectors": { "size": dim, "distance": "Cosine" } })))
                .await?;
            anyhow::ensure!(resp.status().is_success(), "creating collection: {}", resp.status());
            tracing::info!(collection = %self.cfg.collection, dim, "recall: collection created");
        }
        for (field, schema) in [
            ("namespace", "keyword"),
            ("kind", "keyword"),
            ("source", "keyword"),
            ("device_id", "keyword"),
            ("person", "keyword"),
            ("room", "keyword"),
            ("ts_unix_ms", "integer"),
            ("private", "bool"),
            ("last_recalled_ms", "integer"),
            ("recalled", "integer"),
        ] {
            // Idempotent: Qdrant answers success for an index that already exists.
            let _ = self
                .send(self.client.put(self.url("/index")).json(&json!({ "field_name": field, "field_schema": schema })))
                .await;
        }
        Ok(())
    }

    /// Upsert every item (embedding them as documents). Returns how many were sent.
    pub async fn upsert(&self, items: &[Item]) -> Result<usize> {
        if items.is_empty() {
            return Ok(0);
        }
        let texts: Vec<String> = items.iter().map(|i| i.text.clone()).collect();
        let embed = self.embed.clone();
        let vectors = tokio::task::spawn_blocking(move || embed.embed(&texts, Role::Document))
            .await
            .context("embedder panicked")??;
        let points: Vec<Value> = items
            .iter()
            .zip(vectors)
            .map(|(item, vector)| {
                let mut payload = json!({
                    "namespace": item.namespace,
                    "kind": item.kind,
                    "key": item.key,
                    "text": item.text,
                    "ts_unix_ms": item.ts_unix_ms,
                    "private": item.private,
                });
                for (k, v) in &item.meta {
                    payload[k] = json!(v);
                }
                json!({ "id": item.id().to_string(), "vector": vector, "payload": payload })
            })
            .collect();
        let resp = self
            .send(self.client.put(format!("{}?wait=true", self.url("/points"))).json(&json!({ "points": points })))
            .await?;
        anyhow::ensure!(resp.status().is_success(), "upsert: {} {}", resp.status(), resp.text().await.unwrap_or_default());
        Ok(items.len())
    }

    pub async fn search(&self, q: &Query) -> Result<Vec<Hit>> {
        self.search_timed(q).await.map(|(hits, _)| hits)
    }

    /// [`Self::search`], with **which half cost what**.
    ///
    /// The caller's deadline wraps both halves, so an expired recall could not say whether
    /// the query embedding or the Qdrant hop ate it — and only one of the two is fixed by a
    /// bigger number. Measured on this box with the shipped model: the embed is the whole
    /// cost (single-digit ms idle, tens of ms with the box busy), the hop is **1–3 ms**, and
    /// a query arriving behind a `Store::upsert` — which embeds a whole exchange under the
    /// same session — pays the writer's time as well. Returning the split is what makes a
    /// miss diagnosable in production rather than only in this crate's ignored test
    /// (`docs/SOTA.md the-half-of-a-recall-that-costs-is-neither-half`).
    pub async fn search_timed(&self, q: &Query) -> Result<(Vec<Hit>, Phases)> {
        let started = std::time::Instant::now();
        let embed = self.embed.clone();
        let text = q.text.clone();
        let vector = tokio::task::spawn_blocking(move || embed.embed(&[text], Role::Query))
            .await
            .context("embedder panicked")??
            .into_iter()
            .next()
            .context("no query vector")?;
        let embed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let hop_started = std::time::Instant::now();
        let body = json!({
            "vector": vector,
            "limit": q.top_k.max(1),
            "with_payload": true,
            "score_threshold": q.min_score,
            "filter": { "must": q.must() },
        });
        let resp = self.send(self.client.post(self.url("/points/search")).json(&body)).await?;
        anyhow::ensure!(resp.status().is_success(), "search: {}", resp.status());
        let v: Value = resp.json().await?;
        let hops = v["result"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|h| Some(Hit { score: h["score"].as_f64()? as f32, item: item_of(&h["payload"])? }))
                    .collect()
            })
            .unwrap_or_default();
        let phases = Phases { embed_ms, hop_ms: hop_started.elapsed().as_secs_f64() * 1000.0 };
        Ok((hops, phases))
    }

    /// The newest `limit` points matching `must`, newest first, with no vector involved:
    /// what "what do you remember from here" and "forget that" read. Sorted by the
    /// server over the `ts_unix_ms` index; a Qdrant too old for `order_by` answers an
    /// error, which is reported rather than silently unsorted.
    pub async fn scroll(&self, must: Vec<Value>, limit: usize) -> Result<Vec<Item>> {
        let body = json!({
            "limit": limit.max(1),
            "with_payload": true,
            "with_vector": false,
            "filter": { "must": must },
            "order_by": { "key": "ts_unix_ms", "direction": "desc" },
        });
        let resp = self.send(self.client.post(self.url("/points/scroll")).json(&body)).await?;
        anyhow::ensure!(resp.status().is_success(), "scroll: {} {}", resp.status(), resp.text().await.unwrap_or_default());
        let v: Value = resp.json().await?;
        Ok(v["result"]["points"]
            .as_array()
            .map(|a| a.iter().filter_map(|p| item_of(&p["payload"])).collect())
            .unwrap_or_default())
    }

    /// Delete these exact points; the count. `has_id` is a filter like any other, so this
    /// is the same round trip `delete_where` makes — with the one selector that cannot take
    /// a neighbour with it.
    ///
    /// What a **replacement** needs: a consolidating pass writes an exchange's new facts
    /// and then removes the old ones it did not re-state, so an interrupted pass leaves the
    /// old facts or the new ones and never neither (#137).
    pub async fn delete_ids(&self, ids: &[uuid::Uuid]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }
        let filter = json!({ "must": [{ "has_id": ids.iter().map(uuid::Uuid::to_string).collect::<Vec<_>>() }] });
        let resp = self
            .send(self.client.post(format!("{}?wait=true", self.url("/points/delete"))).json(&json!({ "filter": filter })))
            .await?;
        anyhow::ensure!(resp.status().is_success(), "delete by id: {}", resp.status());
        Ok(ids.len() as u64)
    }

    /// Delete every point matching `must` (and none of `must_not`); with `dry_run` only
    /// count them. The count is exact and taken first, so the number reported is the
    /// number deleted.
    pub async fn delete_where(&self, must: Vec<Value>, must_not: Vec<Value>, dry_run: bool) -> Result<u64> {
        let filter = json!({ "must": must, "must_not": must_not });
        let count: Value = self
            .send(self.client.post(self.url("/points/count")).json(&json!({ "filter": filter, "exact": true })))
            .await?
            .error_for_status()?
            .json()
            .await?;
        let n = count["result"]["count"].as_u64().unwrap_or(0);
        if dry_run || n == 0 {
            return Ok(n);
        }
        let resp = self
            .send(self.client.post(format!("{}?wait=true", self.url("/points/delete"))).json(&json!({ "filter": filter })))
            .await?;
        anyhow::ensure!(resp.status().is_success(), "delete: {}", resp.status());
        Ok(n)
    }

    /// Delete every memory made before `ts_unix_ms` that has **not** been recalled since.
    ///
    /// The forgetting curve's whole point is that reaching for a memory keeps it
    /// (`docs/SOTA.md a-forgetting-curve-is-a-prune-that-spares-what-was-recalled`), so
    /// this is the one place that decides what "old" means: a memory whose
    /// `last_recalled_ms` is inside the window is not old, however old it was made. One
    /// function and not a `must`/`must_not` pair built at each call site — the household
    /// prune built only the first half and deleted memories the curve was meant to spare
    /// (#136).
    pub async fn prune_before(&self, ts_unix_ms: i64, dry_run: bool) -> Result<u64> {
        let (must, must_not) = retention_filter(ts_unix_ms);
        self.delete_where(must, must_not, dry_run).await
    }

    /// Merge `payload` into one point's payload (existing keys are overwritten, the rest
    /// kept). The point's vector and id are untouched, so this is what a recall counter
    /// or a last-recalled stamp uses.
    pub async fn set_payload(&self, id: uuid::Uuid, payload: Value) -> Result<()> {
        let body = json!({ "points": [id.to_string()], "payload": payload });
        let resp = self
            .send(self.client.post(format!("{}?wait=true", self.url("/points/payload"))).json(&body))
            .await?;
        anyhow::ensure!(resp.status().is_success(), "set payload: {} {}", resp.status(), resp.text().await.unwrap_or_default());
        Ok(())
    }

    pub async fn stats(&self) -> Result<Stats> {
        let v: Value = self.send(self.client.get(self.url(""))).await?.error_for_status()?.json().await?;
        Ok(Stats {
            points: v["result"]["points_count"].as_u64().unwrap_or(0),
            dim: self.embed.dim(),
            collection: self.cfg.collection.clone(),
            embedder: self.embed.name().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_id_is_a_function_of_namespace_kind_and_key() {
        let a = Item { namespace: "jade".into(), kind: "fact".into(), key: "k".into(), text: "x".into(), ts_unix_ms: 0, private: false, meta: BTreeMap::new() };
        let mut b = a.clone();
        b.text = "a different text, same key".into();
        b.ts_unix_ms = 5;
        assert_eq!(a.id(), b.id(), "re-ingesting is idempotent");
        let mut c = a.clone();
        c.key = "other".into();
        assert_ne!(a.id(), c.id());
        assert_eq!(a.id().get_version(), Some(uuid::Version::Sha1));
    }

    /// Offline: the shipped model, when it is where homelab-voice keeps it. Skipped
    /// (not failed) elsewhere, because a unit test must not need a download.
    #[test]
    fn the_onnx_embedder_returns_unit_vectors_of_its_dimension() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models/bge-small-en-v1.5");
        if !dir.join("model.onnx").is_file() {
            eprintln!("skipping: {} has no model.onnx", dir.display());
            return;
        }
        let e = Onnx::load(&dir.join("model.onnx"), &dir.join("tokenizer.json"), false).expect("loads");
        assert_eq!(e.dim(), 384);
        let v = e.embed(&["the water filter was changed".into(), "unrelated".into()], Role::Document).expect("embeds");
        assert_eq!(v.len(), 2);
        for x in &v {
            assert_eq!(x.len(), 384);
            let norm = x.iter().map(|a| a * a).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "{norm}");
        }
        let q = e.embed(&["when was the water filter changed".into()], Role::Query).expect("embeds");
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(dot(&q[0], &v[0]) > dot(&q[0], &v[1]), "the related document scores higher");
    }

    /// `RECALL_TEST_QDRANT=http://127.0.0.1:6333 cargo test -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn round_trip_against_a_live_qdrant() {
        let url = std::env::var("RECALL_TEST_QDRANT").expect("RECALL_TEST_QDRANT");
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models/bge-small-en-v1.5");
        let e: Arc<dyn Embed> = Arc::new(Onnx::load(&dir.join("model.onnx"), &dir.join("tokenizer.json"), false).unwrap());
        let store = Store::open(Config { qdrant_url: url, collection: "recall_test".into(), timeout_ms: 5000 }, e).await.unwrap();
        let item = Item { namespace: "t".into(), kind: "fact".into(), key: "filter".into(), text: "the water filter was changed on the third of September".into(), ts_unix_ms: 1, private: false, meta: BTreeMap::new() };
        store.upsert(&[item.clone(), item.clone()]).await.unwrap();
        let hits = store.search(&Query { namespace: Some("t".into()), text: "when was the water filter changed".into(), top_k: 3, min_score: 0.3, include_private: true, ..Default::default() }).await.unwrap();
        assert_eq!(hits.len(), 1, "idempotent upsert, one point");
        assert!(hits[0].item.text.contains("water filter"));
        let mut later = item.clone();
        later.key = "later".into();
        later.ts_unix_ms = 9;
        later.meta.insert("device_id".into(), "office".into());
        store.upsert(&[later.clone()]).await.unwrap();
        // Newest first, and the window/device filters are the query's own.
        let ns = json!({ "key": "namespace", "match": { "value": "t" } });
        let scrolled = store.scroll(vec![ns.clone()], 10).await.unwrap();
        assert_eq!(scrolled.iter().map(|i| i.key.as_str()).collect::<Vec<_>>(), ["later", "filter"]);
        let windowed = store
            .search(&Query { namespace: Some("t".into()), text: "water filter".into(), top_k: 3, min_score: 0.0, include_private: true, since_unix_ms: Some(5), ..Default::default() })
            .await
            .unwrap();
        assert_eq!(windowed.len(), 1);
        assert_eq!(windowed[0].item.key, "later");
        let on_office = store
            .search(&Query { namespace: Some("t".into()), text: "water filter".into(), top_k: 3, min_score: 0.0, include_private: true, device_id: Some("office".into()), ..Default::default() })
            .await
            .unwrap();
        assert_eq!(on_office.len(), 1);
        // A payload merge keeps the point and adds the key, as text on the way back.
        store.set_payload(later.id(), json!({ "recalled": 3 })).await.unwrap();
        let back = store.scroll(vec![ns.clone(), json!({ "key": "key", "match": { "value": "later" } })], 1).await.unwrap();
        assert_eq!(back[0].meta.get("recalled").map(String::as_str), Some("3"));
        assert_eq!(store.prune_before(2, true).await.unwrap(), 1);
        assert_eq!(store.prune_before(2, false).await.unwrap(), 1);
        assert_eq!(store.delete_where(vec![ns], Vec::new(), false).await.unwrap(), 1);
        assert_eq!(store.stats().await.unwrap().points, 0);
    }

    /// **Where the recall deadline actually goes**: the query embedding or the Qdrant hop.
    ///
    /// Measured because a deadline miss is only fixable once the half that costs is named,
    /// and only one of the two halves moves with a bigger number: `[memory] deadline_ms` is
    /// **150** and this box's `voice_memory` answered a scroll in **6 ms** over loopback,
    /// so a 150 ms miss is not the network. `Onnx::load` builds its session with
    /// `with_intra_threads(2)` on a 12-core box, which is the knob this measures.
    ///
    /// `RECALL_TEST_QDRANT=http://127.0.0.1:6333 cargo test -p recall -- --ignored
    /// --nocapture the_phases_of_a_recall`.
    #[tokio::test]
    #[ignore]
    async fn the_phases_of_a_recall() {
        let url = std::env::var("RECALL_TEST_QDRANT").expect("RECALL_TEST_QDRANT");
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models/bge-small-en-v1.5");
        let e: Arc<dyn Embed> =
            Arc::new(Onnx::load(&dir.join("model.onnx"), &dir.join("tokenizer.json"), false).unwrap());
        let store = Store::open(Config { qdrant_url: url, collection: "recall_phases".into(), timeout_ms: 5000 }, e.clone())
            .await
            .unwrap();
        store.delete_where(Vec::new(), Vec::new(), true).await.unwrap();
        let items: Vec<Item> = (0..40)
            .map(|i| Item {
                namespace: "jade".into(),
                kind: "exchange".into(),
                key: format!("k{i}"),
                text: format!("They said: memory number {i} about the kettle and the filter — I answered: noted."),
                ts_unix_ms: 1_700_000_000_000 + i,
                private: false,
                meta: BTreeMap::new(),
            })
            .collect();
        store.upsert(&items).await.unwrap();

        let e2 = e.clone();
        let text = "what did I say about the kettle".to_string();
        let mut embed_ms = Vec::new();
        let mut hop_ms = Vec::new();
        for _ in 0..12 {
            let (t0, q) = (std::time::Instant::now(), text.clone());
            let vector = match e2.embed(&[q], Role::Query) {
                Ok(mut v) => {
                    embed_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                    v.remove(0)
                }
                Err(e) => panic!("embed: {e:#}"),
            };
            let t1 = std::time::Instant::now();
            let body = json!({ "vector": vector, "limit": 8, "with_payload": true, "score_threshold": 0.0 });
            let resp = store.client.post(store.url("/points/search")).json(&body).send().await.unwrap();
            assert!(resp.status().is_success());
            let _: Value = resp.json().await.unwrap();
            hop_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (emb, hop) = (med(&mut embed_ms), med(&mut hop_ms));
        println!("recall phases over 12 queries: embed p50 {emb:.1} ms, qdrant hop p50 {hop:.1} ms");
        // **Six milliseconds is not a 150 ms miss, so the cost is not one query's work.**
        // The session is a `parking_lot::Mutex<Session>` and every embed holds it, so the
        // suspect is *concurrency*: several recalls in flight at once queue on the same
        // session, and a turn that waits behind three others pays their sum. Measured with
        // four threads against the same embedder, which is the shape a turn has when the
        // memory write, the recall and the speaker model all touch it.
        let mut concurrent = Vec::new();
        for _ in 0..4 {
            let e3 = e.clone();
            let t = std::time::Instant::now();
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    let e3 = e3.clone();
                    std::thread::spawn(move || {
                        e3.embed(&[format!("a concurrent query number {i} about the kettle")], Role::Query).map(|_| ())
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap().unwrap();
            }
            concurrent.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let serial = med(&mut concurrent);
        println!("four concurrent embeds: p50 {serial:.1} ms (one at a time {emb:.1} ms)");
        // ...and under **load**, which is the state the live box was in: the same turn
        // carried `llm_ttft_ms=4711` because of a tool call, with whisper, Kokoro and the
        // search all busy. `Onnx::load` asks for two intra-op threads (`with_intra_threads(2)`)
        // on a twelve-core box, so a loaded machine gives this session a fraction of a core.
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let load: Vec<_> = (0..12)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut x = 1.0f64;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        x = (x * 1.0000001 + 0.5).sin().abs() + 1.0;
                    }
                    std::hint::black_box(x);
                })
            })
            .collect();
        let mut loaded = Vec::new();
        for _ in 0..8 {
            let t = std::time::Instant::now();
            e.embed(&[text.clone()], Role::Query).unwrap();
            loaded.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in load {
            h.join().unwrap();
        }
        let under_load = med(&mut loaded);
        println!("embed under full CPU load: p50 {under_load:.1} ms (idle {emb:.1} ms)");
        // **The write is the other caller, and it holds the same session.** `Store::upsert`
        // embeds every item as a document in one blocking call under this mutex, and a
        // memory write is the *whole exchange* — two points per turn plus the consolidator's
        // facts — while a recall is one sentence. This measures what a query pays when it
        // arrives behind a realistic write, which is the one number that can turn 6 ms into
        // 150 ms: not the query's cost, but the queue it stands in.
        let docs: Vec<String> = (0..8)
            .map(|i| format!("They said: a whole spoken exchange number {i}, about forty words long, the kettle and the filter and the chores — I answered: noted, here is what I have."))
            .collect();
        let mut behind = Vec::new();
        for _ in 0..6 {
            let writer = e.clone();
            let docs = docs.clone();
            let h = std::thread::spawn(move || writer.embed(&docs, Role::Document));
            let t = std::time::Instant::now();
            e.embed(&["what did I say about the kettle".to_string()], Role::Query).unwrap();
            behind.push(t.elapsed().as_secs_f64() * 1000.0);
            h.join().unwrap().unwrap();
        }
        let queued = med(&mut behind);
        println!("query arriving behind a write: p50 {queued:.1} ms (alone {emb:.1} ms)");
        // ...and the split the production path now logs, read off the real store rather
        // than a hand-timed embed: `search_timed` is what a recall calls, so this is the
        // number a deadline miss will carry (`Phases::cost`).
        let (_, phases) = store
            .search_timed(&Query { namespace: Some("jade".into()), text: "kettle".into(), top_k: 8, min_score: 0.0, include_private: true, ..Default::default() })
            .await
            .unwrap();
        println!(
            "search_timed reports: embed {:.1} ms, hop {:.1} ms, cost {}",
            phases.embed_ms,
            phases.hop_ms,
            phases.cost()
        );
        assert!(phases.total_ms() > 0.0);
        // The hop is a loopback round trip and the embed runs the model: the split has to
        // separate them, or a miss cannot say which half to fix.
        assert!(phases.embed_ms > 0.0 && phases.hop_ms > 0.0);
        // The measurement is the deliverable; the assertion is only that neither half is
        // unmeasurable, so this cannot pass by printing nothing.
        assert!(emb > 0.0 && hop > 0.0 && serial > 0.0 && under_load > 0.0 && queued > 0.0);
        store.delete_where(Vec::new(), Vec::new(), true).await.unwrap();
    }

    /// **A prune spares what was recalled, and both callers agree about it.**
    ///
    /// `prune_before` used to delete on `ts_unix_ms` alone, so the household-wide
    /// `server prune --apply` deleted memories the forgetting curve was meant to keep,
    /// while `server memory prune` — which builds the `must_not` half by hand — spared
    /// them: two counts for one window (#136). The filter is one function now, and this is
    /// the measurement that says so, against a throwaway collection:
    ///
    /// `RECALL_TEST_QDRANT=http://127.0.0.1:6333 cargo test -p recall -- --ignored
    /// --nocapture a_prune_spares_what_was_recalled`.
    #[tokio::test]
    #[ignore]
    async fn a_prune_spares_what_was_recalled() {
        /// A stub with a real dimension: this test is about the filter, not the vectors,
        /// and a store cannot open its collection without one.
        struct Fixed;
        impl Embed for Fixed {
            fn dim(&self) -> usize {
                4
            }
            fn name(&self) -> &str {
                "fixed"
            }
            fn embed(&self, texts: &[String], _: Role) -> Result<Vec<Vec<f32>>> {
                Ok(texts.iter().map(|t| vec![t.len() as f32, 0.0, 0.0, 1.0]).collect())
            }
        }

        let url = std::env::var("RECALL_TEST_QDRANT").expect("RECALL_TEST_QDRANT");
        let store = Store::open(
            Config { qdrant_url: url, collection: "recall_retention".into(), timeout_ms: 5000 },
            Arc::new(Fixed),
        )
        .await
        .unwrap();
        store.delete_where(Vec::new(), Vec::new(), true).await.unwrap();
        let old = 1_600_000_000_000i64;
        let cutoff = 1_700_000_000_000i64;
        let item = |key: &str, ts| Item {
            namespace: "jade".into(),
            kind: "exchange".into(),
            key: key.into(),
            text: format!("memory {key} about the kettle"),
            ts_unix_ms: ts,
            private: false,
            meta: BTreeMap::new(),
        };
        // Old and never reached for; old but recalled inside the window; new. The
        // `last_recalled_ms` stamp is a JSON **number**, which is what `Memory::touch`
        // writes through `set_payload` — as `meta` it would land as a string and the
        // range filter would miss it, which is the instrument's job to get right.
        store
            .upsert(&[item("forgotten", old), item("reached-for", old), item("fresh", cutoff + 1_000)])
            .await
            .unwrap();
        let reached = store.scroll(vec![json!({ "key": "key", "match": { "value": "reached-for" } })], 1).await.unwrap();
        assert_eq!(reached.len(), 1, "the recalled memory is in the store");
        store
            .set_payload(reached[0].id(), json!({ "recalled": 1, "last_recalled_ms": cutoff + 1_000 }))
            .await
            .unwrap();

        // The dry run is the count, and it must be one: only the memory nobody reached for.
        let counted = store.prune_before(cutoff, true).await.unwrap();
        assert_eq!(counted, 1, "only the memory that was never recalled is old");
        // ...and the same number twice, which is what "both commands agree" means.
        assert_eq!(store.prune_before(cutoff, true).await.unwrap(), 1);
        let deleted = store.prune_before(cutoff, false).await.unwrap();
        assert_eq!(deleted, 1);
        let left = store.scroll(vec![json!({ "key": "namespace", "match": { "value": "jade" } })], 10).await.unwrap();
        let keys: Vec<&str> = left.iter().map(|i| i.key.as_str()).collect();
        assert!(keys.contains(&"reached-for") && keys.contains(&"fresh"), "{keys:?}");
        assert!(!keys.contains(&"forgotten"), "{keys:?}");
        store.delete_where(Vec::new(), Vec::new(), true).await.unwrap();
    }

    /// **A replacement writes before it deletes, so an interrupted pass leaves one set or
    /// the other and never neither.**
    ///
    /// The consolidator used to delete a turn's facts as soon as its model stream ended,
    /// and write the replacements only once at the end of the whole pass: one failure after
    /// that point (a stream error on the next exchange, a store hiccup) left every
    /// processed turn with its facts gone (#137). `delete_ids` is the half that makes the
    /// safe order possible — the old ones are removed by id, after the new ones are in.
    ///
    /// `RECALL_TEST_QDRANT=http://127.0.0.1:6333 cargo test -p recall --features
    /// ort-binaries -- --ignored --nocapture a_replacement_writes_before_it_deletes`.
    #[tokio::test]
    #[ignore]
    async fn a_replacement_writes_before_it_deletes() {
        struct Fixed;
        impl Embed for Fixed {
            fn dim(&self) -> usize {
                4
            }
            fn name(&self) -> &str {
                "fixed"
            }
            fn embed(&self, texts: &[String], _: Role) -> Result<Vec<Vec<f32>>> {
                Ok(texts.iter().map(|t| vec![t.len() as f32, 0.0, 0.0, 1.0]).collect())
            }
        }
        let url = std::env::var("RECALL_TEST_QDRANT").expect("RECALL_TEST_QDRANT");
        let store = Store::open(
            Config { qdrant_url: url, collection: "recall_replace".into(), timeout_ms: 5000 },
            Arc::new(Fixed),
        )
        .await
        .unwrap();
        store.delete_where(Vec::new(), Vec::new(), true).await.unwrap();
        let fact = |key: &str, turn: &str| {
            let mut meta = BTreeMap::new();
            meta.insert("from_turn".to_string(), turn.to_string());
            Item {
                namespace: "jade".into(),
                kind: "fact".into(),
                key: key.into(),
                text: format!("fact {key}"),
                ts_unix_ms: 1_700_000_000_000,
                private: false,
                meta,
            }
        };
        let by_turn = |turn: &str| {
            vec![json!({ "key": "kind", "match": { "value": "fact" } }),
                 json!({ "key": "from_turn", "match": { "value": turn } })]
        };
        // An earlier pass's facts for one turn.
        store.upsert(&[fact("old-a", "t1"), fact("old-b", "t1")]).await.unwrap();
        assert_eq!(store.scroll(by_turn("t1"), 10).await.unwrap().len(), 2);
        // What the fixed pass does: write the replacements, then delete the old ones that
        // were not re-stated — in that order.
        store.upsert(&[fact("old-a", "t1"), fact("new-c", "t1")]).await.unwrap();
        let before = store.scroll(by_turn("t1"), 10).await.unwrap();
        let kept: Vec<uuid::Uuid> = vec![fact("old-a", "t1").id(), fact("new-c", "t1").id()];
        let stale: Vec<uuid::Uuid> =
            before.iter().map(Item::id).filter(|id| !kept.contains(id)).collect();
        assert_eq!(store.delete_ids(&stale).await.unwrap(), 1, "only old-b goes");
        let left = store.scroll(by_turn("t1"), 10).await.unwrap();
        let keys: Vec<&str> = left.iter().map(|i| i.key.as_str()).collect();
        assert!(keys.contains(&"old-a") && keys.contains(&"new-c"), "{keys:?}");
        // And an empty replacement (a model that answered `none`) removes the old facts
        // without writing anything: the pass that *did* complete replaces them with nothing.
        assert_eq!(store.delete_ids(&left.iter().map(Item::id).collect::<Vec<_>>()).await.unwrap(), 2);
        assert!(store.scroll(by_turn("t1"), 10).await.unwrap().is_empty());
        store.delete_where(Vec::new(), Vec::new(), true).await.unwrap();
    }
}
