# recall

Vector memory over [Qdrant](https://qdrant.tech) with a local ONNX embedder.

- `Item { namespace, kind, key, text, ts_unix_ms, private, meta }` — a memory; its point id
  is a UUIDv5 over `namespace / kind / key`, so re-ingesting is idempotent and an id can be
  re-derived by hand.
- `Embed` — `Onnx` runs `BAAI/bge-small-en-v1.5` (384-d, CLS pooling, L2-normalised,
  bge's query instruction on the query side) in process through ONNX Runtime; `Ollama` asks a
  running Ollama over HTTP.
- `Store::open` creates the collection (cosine, the embedder's dimension) and the payload
  indexes if absent; `upsert`, `search`, `prune_before`, `stats` speak Qdrant's REST API with a
  plain HTTP client.

Built for [homelab-voice](https://github.com/Ladvien/homelab-voice), where it is a git
submodule; `ort` is pinned to the same `=2.0.0-rc.12` that server carries so one runtime is
linked. An application that already links ONNX Runtime needs no feature; the crate's own
tests want `--features ort-binaries`, and `RECALL_TEST_QDRANT=http://127.0.0.1:6333 cargo
test --features ort-binaries -- --ignored` runs the round trip against a live Qdrant.
