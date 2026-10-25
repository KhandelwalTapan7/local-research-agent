# Vector DB Research Agent

An agentic research assistant for vector-database selection and sizing. It plans a multi-step strategy, calls tools through its own agent loop (no agent framework), detects poor tool results, corrects itself, and returns an answer with inline `[S#]` citations. Reasoning runs on a local Ollama model, so no paid API is needed.

## Features

- Plan-and-Execute loop with per-step reflection and bounded, non-repeating retries (`agent/core.py`)
- Seven tools (`agent/tools.py`): `rag_search`, `web_search`, `summarise`, `code_executor`, `structured_output`, `memory_store`, `memory_recall`
- Hybrid retrieval: BM25 + dense embeddings fused with RRF, then cross-encoder reranking (`rag/`)
- FastAPI with SSE streaming; SQLite stores every trace, final answer and session fact (`agent/memory.py`)
- Plain HTML/JS frontend with the live reasoning trace, highlighted citations, tables and session history

## Model

Default: `qwen2.5:7b-instruct` (7.6B parameters, Q4_K_M, about 4.7 GB) through Ollama. It follows JSON instructions reliably at this size, leaves room for an 8k context on a single GPU, and is Apache-2.0 licensed. Change it with `OLLAMA_MODEL`. Embeddings (`BAAI/bge-small-en-v1.5`) and reranking (`Xenova/ms-marco-MiniLM-L-6-v2`) run on CPU through fastembed.

Requirements for the 7B model: Docker needs about 8 GB of free memory (on Windows, WSL2 must be allowed that much), or a GPU with 8 GB+ VRAM. On CPU each model call takes tens of seconds, so a full run takes several minutes.

Note on testing: the development machine has 8 GB of RAM, which is not enough for the 7B model in Docker, so the end-to-end runs and demo were done with `qwen2.5:3b-instruct`. The 7B default is configured but was not run on that machine. The 3B model works but produces weaker plans and occasionally needs the self-correction loop to fix its code.

To use the 3B model, change `qwen2.5:7b-instruct` to `qwen2.5:3b-instruct` in both places in `docker-compose.yml` (the `ollama-pull` entrypoint and `OLLAMA_MODEL`).

## Run with Docker

```
docker compose up --build
```

Open http://localhost:8000 (not `0.0.0.0`). The first start pulls the model, downloads the embedding and reranking models, and builds the index from `data/`. For GPU access add this to the `ollama` service:

```
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Run locally

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
uvicorn api.main:app --port 8000
```

Python 3.11+. `python -m rag.ingest` rebuilds the index after editing `data/`.

## API

| Method | Path | Description |
| --- | --- | --- |
| POST | `/run` | Body `{"query": str, "session_id": str}`. Streams SSE events: `plan`, `step`, `thought`, `tool_call`, `tool_result`, `reflection`, `final`, `error`, `done` |
| GET | `/history/{session_id}` | Every run in the session with its full event trace and final answer |
| GET | `/health` | Liveness check |

```
curl -N -X POST localhost:8000/run -H 'Content-Type: application/json' \
  -d '{"query":"How much raw memory do 100 million 128-dimensional float32 vectors need?","session_id":"demo"}'
```

## Demo queries

1. `How much raw memory do 100 million 128-dimensional float32 vectors need?` Retrieval, code execution (51.2 GB) and a cited answer. The model sometimes forgets `print()` on the first try, which shows the self-correction.
2. `Compare pgvector and Qdrant in a table covering indexing, filtering and best fit.` Retrieval and a table under the answer.
3. `Which one is best?` A vague question that scores low in retrieval and triggers a rewrite.
4. `Remember that I need self-hosted storage and have 20 million 384-dimensional vectors.` followed in the same session by `Which database fits me, and how much raw memory will I need?` Uses `memory_store` and the stored facts (30.72 GB).

## How self-correction works

A tool result is flagged poor when `rag_search` has a best rerank score at or below `RERANK_MIN` (default 0), `code_executor` errors, times out or prints nothing, `web_search` fails, or `structured_output` is malformed. The agent then asks the model to diagnose the failure and revise the query, tool or code. It does at most two retries per step and stops early if the revised call equals one already tried.

## Configuration

| Variable | Default |
| --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` |
| `RERANK_MIN` | `0` |
| `DB_PATH` | `runtime/agent.db` |
| `INDEX_DIR` / `DATA_DIR` | `runtime/index` / `data` |

## Notes and limits

- `code_executor` runs snippets in an isolated subprocess (`python -I`, empty environment, temporary directory, 10 s timeout). It is not a security sandbox; run the container with limited privileges.
- `web_search` scrapes DuckDuckGo HTML and needs internet access. The agent works without it from the local documents.
- `memory_store` is only planned when the question asks to remember something, and `memory_recall` only when the session has earlier turns.
- To change the knowledge base, add markdown files to `data/` (one `# Title`, `##` sections) and delete `runtime/index`.