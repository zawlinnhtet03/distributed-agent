"""
Quick script to ingest project knowledge into ChromaDB for the RAG agent.

Usage:
    python ingest_knowledge.py
"""

import json
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from litellm import embedding
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / ".env.local")

# --- Config (matches app/config.py) ---
PERSIST_DIR = str(REPO_ROOT / "data" / "chroma")
COLLECTION_NAME = "default"
EMBEDDING_MODEL = "mistral/mistral-embed"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def _get_client():
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=PERSIST_DIR)


def _get_collection():
    client = _get_client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _embed_texts(texts):
    resp = embedding(model=EMBEDDING_MODEL, api_key=MISTRAL_API_KEY, input=texts)
    return [item["embedding"] for item in resp["data"]]


def chunk_text(text, chunk_size=1000, overlap=150):
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks, start = [], 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def ingest_text(content, metadata=None, doc_id=None):
    metadata = metadata or {}
    chunks = chunk_text(content)
    if not chunks:
        return {"status": "error", "error": "No content"}
    base_id = doc_id or f"doc_{int(time.time() * 1000)}"
    ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "chunk_index": i, "doc_id": base_id} for i in range(len(chunks))]
    embeddings = _embed_texts(chunks)
    collection = _get_collection()
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return {"status": "success", "doc_id": base_id, "chunks_added": len(chunks)}


def ingest_file(file_path, metadata=None):
    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "error": "File not found"}
    content = path.read_text(encoding="utf-8")
    meta = {"source": str(path), "file_name": path.name}
    if metadata:
        meta.update(metadata)
    return ingest_text(content, metadata=meta, doc_id=path.stem)


def list_collections():
    client = _get_client()
    cols = client.list_collections()
    return {"collections": [c.name for c in cols], "count": len(cols)}


def main():
    print("=" * 60)
    print("RAG Knowledge Ingestion")
    print("=" * 60)

    # 1. Ingest the README
    print("\n[1/4] Ingesting README.md...")
    result = ingest_file("README.md", metadata={"topic": "project", "type": "documentation"})
    print(f"  Status: {result['status']}, Chunks: {result.get('chunks_added', 0)}")

    # 2. Ingest architecture knowledge (current, accurate)
    print("\n[2/4] Ingesting architecture docs...")
    architecture_doc = """
# Multi Agent Intelligence Research Hub - Architecture

## Pipeline Flow
The system uses a 3-step sequential pipeline:
1. **Planner** (Step 1): Analyzes every user query and produces a structured execution plan. Writes the plan to session state.
2. **Aggregator** (Step 2): Reads the plan, delegates tasks to the appropriate specialist agents, and synthesizes all results into a single cohesive response.
3. **Guardian** (Step 3): Reviews the final output for safety, accuracy, and policy compliance before delivery.

## Agent Hierarchy
```
decision_pipeline (SequentialAgent) — ROOT
├── planner (LlmAgent) — writes plan to state["plan"]
├── aggregator (LlmAgent) — reads plan, delegates, synthesizes
│   ├── gathering_layer (ParallelAgent) — research layer
│   │   ├── scraper — web search via Tavily API
│   │   └── rag — knowledge retrieval via ChromaDB
│   ├── data — EDA, cleaning, feature engineering, baseline modeling
│   └── video — local video analysis via OpenCV + Ollama Moondream
└── guardian — safety and policy compliance gate
```

## Research Layer (gathering_layer)
The gathering_layer is a ParallelAgent that runs two agents simultaneously:
- **Scraper**: Uses Tavily API for live web search, content extraction, and AI-synthesized answers
- **RAG**: Searches the local ChromaDB knowledge base using Mistral embeddings for semantic similarity

## Data Agent
Operates independently from the research layer. Handles:
- Loading CSV/TSV/XLSX datasets
- Data profiling (columns, missing values, shape, duplicates)
- Cleaning (drop duplicates, fill missing, drop high-missing columns)
- Datetime feature engineering
- Small baseline model training (classification/regression with train/test split)

## Video Agent
Operates independently. Analyzes local video files using:
- OpenCV for frame extraction (3 key frames at 20%, 50%, 80%)
- Ollama Moondream for vision-language analysis
- In-memory processing (no disk I/O)

## Technology Stack
- **Framework**: Google ADK (Agent Development Kit)
- **LLM Provider**: Mistral via LiteLLM
- **Models**: mistral-medium-latest (default), mistral-small-latest (fast)
- **Embeddings**: Mistral via LiteLLM
- **Vector Store**: ChromaDB (persistent, cosine similarity)
- **Web Search**: Tavily API
- **Video Analysis**: OpenCV + Ollama Moondream
- **Data Processing**: Pandas, scikit-learn
"""
    result = ingest_text(
        content=architecture_doc,
        metadata={"topic": "architecture", "type": "documentation"},
        doc_id="architecture_overview",
    )
    print(f"  Status: {result['status']}, Chunks: {result.get('chunks_added', 0)}")

    # 3. Ingest agent details
    print("\n[3/4] Ingesting agent details...")
    agent_details = """
# Agent Details and Capabilities

## Planner Agent
- Temperature: 0.3 (precise planning)
- Output: Writes structured plan to state["plan"] via output_key
- Format: PLAN with steps, agent assignments, PARALLEL/SEQUENTIAL indicators
- Available agent names for assignment: gathering_layer, data, video, aggregator

## Aggregator Agent
- Temperature: 0.5 (balanced creativity and accuracy)
- Reads {plan} from state to determine routing
- Sub-agents: gathering_layer, data, video
- Synthesizes results from all delegated agents into one response

## Scraper Agent (inside gathering_layer)
- Temperature: 0.3
- Tier: fast (mistral-small-latest for speed)
- Tools: tavily_web_search, tavily_extract_content, tavily_get_answer
- Always cites sources with URLs

## RAG Agent (inside gathering_layer)
- Temperature: 0.2 (factual retrieval)
- Tools: vector_search, add_document, ingest_text, ingest_file, ingest_directory, list_collections
- Uses ChromaDB with Mistral embeddings
- Chunking: 1000 chars with 150 char overlap

## Data Agent
- Temperature: 0.2 (precise data analysis)
- Tools: load_dataset, profile_dataset, clean_dataset, add_datetime_features, train_small_model
- Supports CSV, TSV, TXT, XLSX formats
- Baseline model: Decision Tree with accuracy/MSE metrics

## Video Agent
- Temperature: 0.5
- Tools: analyze_video_locally, analyze_video, extract_frames, analyze_frame_with_moondream, cleanup_temp_files
- Extracts 3 key frames at 20%, 50%, 80% positions
- Uses Ollama Moondream for vision analysis

## Guardian Agent
- Temperature: 0.1 (strict safety review)
- Tier: fast
- Tools: check_content_safety, validate_sources, format_safe_response
- Decisions: APPROVE, APPROVE WITH DISCLAIMER, REVISE, REJECT
"""
    result = ingest_text(
        content=agent_details,
        metadata={"topic": "agents", "type": "documentation"},
        doc_id="agent_details",
    )
    print(f"  Status: {result['status']}, Chunks: {result.get('chunks_added', 0)}")

    # 4. Ingest tool reference
    print("\n[4/4] Ingesting tool reference...")
    tool_reference = """
# Tool Reference Guide

## Web Search Tools (Scraper Agent)
- **tavily_web_search**: Search the web with basic or advanced depth. Returns titles, URLs, content, scores. Max 10 results.
- **tavily_extract_content**: Extract full text content from specific URLs.
- **tavily_get_answer**: Get AI-synthesized answers with source citations.

## RAG Tools (RAG Agent)
- **vector_search**: Semantic search against ChromaDB. Parameters: query, top_k (default 3), filter_metadata, min_score.
- **add_document**: Add text content with optional metadata. Auto-chunks and embeds.
- **ingest_text**: Alias for add_document.
- **ingest_file**: Read and ingest a local .txt/.md/.json file.
- **ingest_directory**: Bulk ingest files matching a glob pattern (e.g., **/*.md).
- **list_collections**: Show all ChromaDB collections.

## Data Tools (Data Agent)
- **load_dataset**: Load CSV/TSV/XLSX and preview first 5 rows. Returns shape, columns, dtypes.
- **profile_dataset**: Column stats, missing values, unique counts, duplicate detection.
- **clean_dataset**: Drop duplicates, fill missing (median for numeric, mode for categorical), optionally drop high-missing columns.
- **add_datetime_features**: Extract year, month, day, day_of_week, hour from datetime columns.
- **train_small_model**: Train DecisionTree classifier or regressor. Auto-detects task type. Returns accuracy or MSE with feature importances.

## Video Tools (Video Agent)
- **analyze_video_locally**: Full analysis pipeline — extract 3 frames and analyze with Moondream.
- **analyze_video**: Alternative analysis function.
- **extract_frames**: Extract frames at specified positions.
- **analyze_frame_with_moondream**: Analyze a single frame with Ollama Moondream.
- **cleanup_temp_files**: Remove temporary files created during analysis.

## Safety Tools (Guardian Agent)
- **check_content_safety**: Scan for misinformation, bias, harmful content, missing citations. Returns risk level.
- **validate_sources**: Check URLs against trusted domain list (arxiv, github, nature, gov, edu).
- **format_safe_response**: Add disclaimers to long content before delivery.
"""
    result = ingest_text(
        content=tool_reference,
        metadata={"topic": "tools", "type": "reference"},
        doc_id="tool_reference",
    )
    print(f"  Status: {result['status']}, Chunks: {result.get('chunks_added', 0)}")

    # Show final state
    print("\n" + "=" * 60)
    print("Ingestion complete! Current collections:")
    collections = list_collections()
    print(f"  Collections: {collections['collections']}")
    print(f"  Count: {collections['count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
