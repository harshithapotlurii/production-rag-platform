# Grounded Retrieval Reference

A small, independently written Python retrieval service demonstrating document ingestion, deterministic local search, evidence extraction, and cited answers. It uses synthetic sample content and requires no external model credentials.

This is a **baseline portfolio project**, not a claim of production deployment or a copy of employer code. Its retrieval is lexical token overlap, not embeddings or hybrid vector search. It deliberately abstains when no passage matches.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e '.[dev]'
uvicorn rag_platform.api:app --reload
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/documents -H 'Content-Type: application/json' -d '{"source":"sample-guide","text":"The support team answers requests in English and Hindi. Escalations are tracked in the ticketing system."}'
curl -X POST http://127.0.0.1:8000/query -H 'Content-Type: application/json' -d '{"query":"Which languages does the support team use?"}'
```

Responses contain `answer` and `citations` with a source, excerpt and overlap score. Interactive API docs are at `/docs`. Set `RAG_DB_PATH` to choose the SQLite database file. Reingesting a source replaces its earlier passages atomically.

## Verify

```bash
pytest -q
```

## Design and limitations

- SQLite stores source identified passages; no external database is needed.
- Whole word token matching is deterministic but does not understand synonyms or morphology.
- The extractive answer copies source sentences and tags each with its citation index. It is not an LLM generated answer.
- Chunking is by word count, so a sentence may cross a chunk boundary.
- The in process SQLite connection is suitable for a local demo; concurrent multi worker deployment needs a separate persistence design, access control, observability and rate limits.

Next steps: add semantic embeddings, hybrid retrieval, reranking, a provider adapter for optional LLM synthesis, and an evaluation corpus with groundedness and latency metrics. Those features are planned, not implemented here.

No employer assets, secrets or internal data are included.
