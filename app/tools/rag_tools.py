"""RAG tools backed by the project's sharded router service."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.getenv("BASE_DIR", str(PROJECT_ROOT / "shared_data")))
KB_MANIFEST_PATH = Path(
    os.getenv("KB_MANIFEST_PATH", str(BASE_DIR / "outputs" / "kb_documents.jsonl"))
)
ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8000").rstrip("/")
FRAME_CLIP_DIM = int(os.getenv("FRAME_CLIP_DIM", "512"))
TEXT_EMBED_DIM = int(os.getenv("TEXT_EMBED_DIM", "768"))


def _now_ts() -> int:
    return int(time.time())


def _stable_id(prefix: str, value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}:{value}"))


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, method=method.upper(), headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=45) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {body}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _hash_embedding(text: str, dim: int) -> list[float]:
    import hashlib

    values: list[float] = []
    seed = text.encode("utf-8")
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(seed + counter.to_bytes(4, byteorder="big")).digest()
        for idx in range(0, len(digest), 4):
            chunk = digest[idx : idx + 4]
            if len(chunk) < 4:
                break
            number = int.from_bytes(chunk, byteorder="big", signed=False)
            values.append((number / 4294967295.0) * 2.0 - 1.0)
            if len(values) >= dim:
                break
        counter += 1
    return values


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def _append_manifest_entries(entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    KB_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KB_MANIFEST_PATH.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_manifest(limit: int | None = None) -> list[dict[str, Any]]:
    if not KB_MANIFEST_PATH.exists():
        return []

    rows: list[dict[str, Any]] = []
    with KB_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    rows.sort(key=lambda x: str(x.get("ingested_at", "")), reverse=True)
    if limit is not None and limit > 0:
        return rows[:limit]
    return rows


def _to_doc(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload") or {}
    score = float(item.get("score", 0.0))
    content = str(payload.get("kb_text") or payload.get("source_url") or "")
    source = str(payload.get("kb_source") or payload.get("source_url") or "unknown")
    doc_type = str(payload.get("kb_type") or "note")
    return {
        "doc_id": str(item.get("id", "")),
        "content": content,
        "metadata": payload,
        "score": round(score, 4),
        "source": source,
        "type": doc_type,
    }


def _upsert_chunks(
    *,
    chunks: list[str],
    source: str,
    doc_type: str,
    base_id: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not chunks:
        return {"status": "error", "error": "No content"}

    now_ts = _now_ts()
    extra_metadata = extra_metadata or {}

    points: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        point_id = _stable_id("kb_chunk", f"{base_id}:{idx}")
        payload = {
            "global_id": f"kb_{base_id}_{idx}",
            "video_id": f"kb_{base_id}",
            "platform": "kb",
            "content_type": "video",
            "orientation": "horizontal",
            "duration_sec": 120,
            "created_at": now_ts,
            "ingested_at": now_ts,
            "frame_ts_ms": idx * 1000,
            "source_url": source,
            "brand_tags": ["kb", doc_type],
            "embedding_model_version": "hash-v1",
            "kb_text": chunk,
            "kb_source": source,
            "kb_type": doc_type,
            "kb_doc_id": base_id,
            "kb_chunk_index": idx,
            **extra_metadata,
        }
        points.append(
            {
                "id": point_id,
                "vectors": {
                    "frame_clip": _hash_embedding(chunk, FRAME_CLIP_DIM),
                    "text_embed": _hash_embedding(chunk, TEXT_EMBED_DIM),
                },
                "payload": payload,
            }
        )
        manifest_entries.append(
            {
                "id": point_id,
                "doc_id": base_id,
                "type": doc_type,
                "source": source,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "preview": chunk[:260],
                "length": len(chunk),
                "ingested_at": now_ts,
            }
        )

    upsert_response = _http_json(
        "POST",
        f"{ROUTER_URL}/upsert",
        {"points": points, "batch_size": 64},
    )
    _append_manifest_entries(manifest_entries)

    return {
        "status": "success",
        "doc_id": base_id,
        "chunks_added": len(points),
        "upsert_response": upsert_response,
    }


def add_document(
    content: str,
    metadata: dict | None = None,
    doc_id: str | None = None,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> dict[str, Any]:
    chunks = _chunk_text(content, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {"status": "error", "error": "No content"}

    meta = metadata or {}
    base_id = doc_id or f"doc_{int(time.time() * 1000)}"
    source = str(meta.get("source", f"kb://{base_id}"))
    doc_type = str(meta.get("type", "note"))
    return _upsert_chunks(
        chunks=chunks,
        source=source,
        doc_type=doc_type,
        base_id=base_id,
        extra_metadata={"kb_metadata": meta},
    )


def ingest_text(content: str, metadata: dict | None = None, doc_id: str | None = None) -> dict[str, Any]:
    return add_document(content=content, metadata=metadata, doc_id=doc_id)


def ingest_file(file_path: str, metadata: dict | None = None) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return {"status": "error", "error": "File not found"}

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            content = json.dumps(
                json.loads(path.read_text(encoding="utf-8")),
                ensure_ascii=False,
                indent=2,
            )
        else:
            content = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}

    meta = {"source": str(path), "file_name": path.name, "type": "note"}
    if metadata:
        meta.update(metadata)
    return add_document(content=content, metadata=meta, doc_id=path.stem)


def ingest_directory(dir_path: str, glob_pattern: str = "**/*.txt") -> dict[str, Any]:
    base = Path(dir_path)
    if not base.exists() or not base.is_dir():
        return {"status": "error", "error": "Directory not found"}

    files = sorted(base.glob(glob_pattern))
    results = [ingest_file(str(path)) for path in files]
    return {
        "status": "success",
        "files_processed": len(files),
        "results": results,
    }


def vector_search(query: str, top_k: int = 3, filter_metadata: dict | None = None) -> dict[str, Any]:
    started = time.time()
    filters = {"platform": ["kb"]}
    if isinstance(filter_metadata, dict):
        filters.update(filter_metadata)

    payload = {
        "mode": "global",
        "vector_name": "text_embed",
        "query_vector": _hash_embedding(query, TEXT_EMBED_DIM),
        "top_k": int(top_k),
        "filters": filters,
        "with_payload": True,
    }
    result = _http_json("POST", f"{ROUTER_URL}/search", payload)
    docs = [_to_doc(item) for item in result.get("results", [])]
    return {
        "query": query,
        "documents": docs,
        "total_found": len(docs),
        "time_ms": round((time.time() - started) * 1000, 2),
        "shards_queried": result.get("shards_queried", []),
        "warnings": result.get("warnings", []),
    }


def list_collections() -> dict[str, Any]:
    try:
        shards_resp = _http_json("GET", f"{ROUTER_URL}/shards")
    except Exception as exc:  # noqa: BLE001
        return {"collections": [], "count": 0, "error": str(exc)}

    shards = shards_resp.get("shards", [])
    names = [str(s.get("name", "")) for s in shards if isinstance(s, dict)]
    return {"collections": names, "count": len(names), "shards": shards}


def list_documents(limit: int = 200) -> dict[str, Any]:
    rows = _read_manifest(limit=limit)
    docs = [
        {
            "id": str(row.get("id", "")),
            "type": str(row.get("type", "note")),
            "source": str(row.get("source", "unknown")),
            "timestamp": str(row.get("timestamp", "")),
            "preview": str(row.get("preview", "")),
            "length": int(row.get("length", 0) or 0),
        }
        for row in rows
    ]
    return {"documents": docs, "count": len(docs)}


def get_stats() -> dict[str, Any]:
    docs = _read_manifest(limit=None)
    by_type: dict[str, int] = {}
    for row in docs:
        key = str(row.get("type", "note"))
        by_type[key] = by_type.get(key, 0) + 1
    return {
        "totalDocs": len(docs),
        "collection": "router_multimodal_items",
        "model": "hash-v1",
        "byType": by_type,
    }
