# Sharded Multi-Agent Retrieval System

This repo now runs a single end-to-end flow matching the presentation architecture:

- Scout stage: discovers local videos + image metadata candidates.
- Analyst stage (Ray actors): processes videos and generates summaries.
- Storage stage: upserts vectors/payloads into a sharded Qdrant router.
- Retrieval stage: runs both targeted and global search.
- Report stage: writes one final project report JSON.

Primary entrypoint:
- `orchestrator/final_project.py`

## One-command run (Docker)

From the project root:

```bash
docker compose up --build
```

This starts:

- `qdrant_shard1`
- `qdrant_shard2`
- `qdrant_shard3`
- `shard_router`
- `ray-head`
- `ray-worker-1`
- `ray-worker-2`
- `orchestrator-runner`

## Outputs

- Final system report: `shared_data/outputs/final_project_report.json`
- Embedding log from analyst stage: `shared_data/outputs/embeddings.jsonl`

## ADK Integration

The ADK layer is wired into the same sharded retrieval backend used by the final project flow:

- `app/tools/rag_tools.py` uses the shard router (`/upsert`, `/search`, `/shards`) instead of a separate local Chroma store.
- `app/api_server.py` exposes:
  - `POST /final-project/run` to run `orchestrator.final_project`
  - `GET /final-project/report` to read `shared_data/outputs/final_project_report.json`
  - `GET /kb/stats`, `GET /kb/documents`, `POST /kb/search`, `POST /kb/upload` for router-backed knowledge-base operations.
- `frontend` API routes under `src/app/api/adk/` call those backend KB endpoints (no hardcoded `localhost:8003` dependency).

## Notes

- `orchestrator-runner` uses `MOCK_TRANSCRIPTION=1` by default so it can run without GPU/Whisper setup.
- If you also want standalone local execution, you can still run `python -m orchestrator.main` or `python -m orchestrator.final_project` after installing Python dependencies.
