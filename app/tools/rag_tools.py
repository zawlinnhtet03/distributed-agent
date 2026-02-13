"""RAG tools for ChromaDB retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import chromadb
from litellm import embedding
from app.config import get_settings


def _get_persist_dir() -> Path:
    settings = get_settings()
    base_dir = Path(settings.rag_persist_dir)
    if not base_dir.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        base_dir = repo_root / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _get_collection_name() -> str:
    return get_settings().rag_collection


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(_get_persist_dir()))


def _get_collection() -> chromadb.Collection:
    client = _get_client()
    name = _get_collection_name()
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    resp = embedding(model=settings.embedding_model, api_key=settings.mistral_api_key, input=texts)
    return [item["embedding"] for item in resp["data"]]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Chunk text into overlapping segments."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def add_document(content: str, metadata: dict | None = None, doc_id: str | None = None,
                 chunk_size: int = 1000, overlap: int = 150) -> dict[str, Any]:
    """Add document to ChromaDB with chunking."""
    metadata = metadata or {}
    chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {"status": "error", "error": "No content"}

    base_id = doc_id or f"doc_{int(time.time() * 1000)}"
    ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "chunk_index": i, "doc_id": base_id} for i in range(len(chunks))]
    embeddings = _embed_texts(chunks)

    collection = _get_collection()
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

    return {"status": "success", "doc_id": base_id, "chunks_added": len(chunks)}


def ingest_text(content: str, metadata: dict | None = None, doc_id: str | None = None) -> dict[str, Any]:
    """Ingest raw text into vector store."""
    return add_document(content=content, metadata=metadata, doc_id=doc_id)


def ingest_file(file_path: str, metadata: dict | None = None) -> dict[str, Any]:
    """Ingest a file into vector store."""
    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "error": "File not found"}

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            content = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=True, indent=2)
        else:
            content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    meta = {"source": str(path), "file_name": path.name}
    if metadata:
        meta.update(metadata)
    return add_document(content=content, metadata=meta, doc_id=path.stem)


def ingest_directory(dir_path: str, glob_pattern: str = "**/*.txt") -> dict[str, Any]:
    """Ingest multiple files from directory."""
    base = Path(dir_path)
    if not base.exists() or not base.is_dir():
        return {"status": "error", "error": "Directory not found"}

    files = list(base.glob(glob_pattern))
    results = [ingest_file(str(path)) for path in files]
    return {"status": "success", "files_processed": len(files), "results": results}


def vector_search(query: str, top_k: int = 3, filter_metadata: dict | None = None) -> dict[str, Any]:
    """Search vector store for relevant documents."""
    start = time.time()
    collection = _get_collection()
    query_embedding = _embed_texts([query])[0]

    response = collection.query(
        query_embeddings=[query_embedding], n_results=top_k, where=filter_metadata,
        include=["documents", "metadatas", "distances"],
    )

    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    ids = response.get("ids", [[]])[0]

    results = []
    for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
        score = max(0.0, min(1.0, 1.0 - float(dist)))
        results.append({"doc_id": doc_id, "content": doc, "metadata": meta or {}, "score": round(score, 4)})

    return {"query": query, "documents": results, "total_found": len(results),
            "time_ms": round((time.time() - start) * 1000, 2)}


def list_collections() -> dict[str, Any]:
    """List ChromaDB collections."""
    client = _get_client()
    collections = client.list_collections()
    return {"collections": [c.name for c in collections], "count": len(collections)}
