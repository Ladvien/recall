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
            ("room", "keyword"),
            ("ts_unix_ms", "integer"),
            ("private", "bool"),
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
        let embed = self.embed.clone();
        let text = q.text.clone();
        let vector = tokio::task::spawn_blocking(move || embed.embed(&[text], Role::Query))
            .await
            .context("embedder panicked")??
            .into_iter()
            .next()
            .context("no query vector")?;
        let mut must: Vec<Value> = Vec::new();
        if let Some(ns) = &q.namespace {
            must.push(json!({ "key": "namespace", "match": { "value": ns } }));
        }
        if !q.include_private {
            must.push(json!({ "key": "private", "match": { "value": false } }));
        }
        if !q.kinds.is_empty() {
            must.push(json!({ "key": "kind", "match": { "any": q.kinds } }));
        }
        let body = json!({
            "vector": vector,
            "limit": q.top_k.max(1),
            "with_payload": true,
            "score_threshold": q.min_score,
            "filter": { "must": must },
        });
        let resp = self.send(self.client.post(self.url("/points/search")).json(&body)).await?;
        anyhow::ensure!(resp.status().is_success(), "search: {}", resp.status());
        let v: Value = resp.json().await?;
        let hits = v["result"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|h| {
                        let p = &h["payload"];
                        let mut meta = BTreeMap::new();
                        if let Some(o) = p.as_object() {
                            for (k, v) in o {
                                if !matches!(k.as_str(), "namespace" | "kind" | "key" | "text" | "ts_unix_ms" | "private") {
                                    if let Some(s) = v.as_str() {
                                        meta.insert(k.clone(), s.to_string());
                                    }
                                }
                            }
                        }
                        Some(Hit {
                            score: h["score"].as_f64()? as f32,
                            item: Item {
                                namespace: p["namespace"].as_str()?.to_string(),
                                kind: p["kind"].as_str().unwrap_or("").to_string(),
                                key: p["key"].as_str().unwrap_or("").to_string(),
                                text: p["text"].as_str()?.to_string(),
                                ts_unix_ms: p["ts_unix_ms"].as_i64().unwrap_or(0),
                                private: p["private"].as_bool().unwrap_or(false),
                                meta,
                            },
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(hits)
    }

    /// Delete every point with `ts_unix_ms` before `ts`; with `dry_run` only count them.
    pub async fn prune_before(&self, ts_unix_ms: i64, dry_run: bool) -> Result<u64> {
        let filter = json!({ "must": [{ "key": "ts_unix_ms", "range": { "lt": ts_unix_ms } }] });
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
        let hits = store.search(&Query { namespace: Some("t".into()), text: "when was the water filter changed".into(), top_k: 3, min_score: 0.3, include_private: true, kinds: vec![] }).await.unwrap();
        assert_eq!(hits.len(), 1, "idempotent upsert, one point");
        assert!(hits[0].item.text.contains("water filter"));
        assert_eq!(store.prune_before(2, true).await.unwrap(), 1);
        assert_eq!(store.prune_before(2, false).await.unwrap(), 1);
        assert_eq!(store.stats().await.unwrap().points, 0);
    }
}
