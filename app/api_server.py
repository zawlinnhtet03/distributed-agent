import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Any
from pathlib import Path
import secrets

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.genai.types import Content, Part
import uvicorn

from app.agents.aggregator import root_agent
from app.agents.rag import rag_agent
from app.tools.rag_tools import add_document, get_stats, list_documents, vector_search

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SESSION_SERVICE = InMemorySessionService()

_SHARED_DATA_DIR = _PROJECT_ROOT / "shared_data"
_DATASETS_DIR = _PROJECT_ROOT / "datasets"
_VIDEO_UPLOAD_DIR = _SHARED_DATA_DIR / "videos" / "uploads"
_FINAL_REPORT_PATH = Path(
    os.getenv(
        "FINAL_PROJECT_REPORT_PATH",
        str(_SHARED_DATA_DIR / "outputs" / "final_project_report.json"),
    )
)
if not _FINAL_REPORT_PATH.is_absolute():
    _FINAL_REPORT_PATH = _PROJECT_ROOT / _FINAL_REPORT_PATH
_DEFAULT_ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8000")


# Serve generated chart HTML files (Plotly) and other dataset artifacts.
# This enables dashboards to iframe the interactive charts.
_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/datasets", StaticFiles(directory=str(_DATASETS_DIR), html=True), name="datasets")


def _posix(p: Path) -> str:
    return str(p).replace("\\", "/")


def _classify_upload(filename: str, kind: str) -> str:
    k = (kind or "").lower().strip()
    if k in {"video", "data"}:
        return k

    ext = Path(filename).suffix.lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return "video"
    if ext in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}:
        return "data"
    return "data"


def _extract_upload_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".csv", ".tsv", ".json"}:
        return content.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from io import BytesIO
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            pages = [(page.extract_text() or "") for page in reader.pages]
            return "\n".join(pages)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}") from exc
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")


def _get_agent_for_app(app_name: str):
    mapping = {
        "research_pipeline_agent": root_agent,
        "sharded-retrieval-system": root_agent,
        "rag_agent_shard": rag_agent,
    }
    if app_name not in mapping:
        raise HTTPException(status_code=404, detail=f"Unknown app_name: {app_name}")
    return mapping[app_name]


def _get_runner(app_name: str) -> Runner:
    agent = _get_agent_for_app(app_name)
    return Runner(agent=agent, app_name=app_name, session_service=_SESSION_SERVICE)


def _normalize_new_message(new_message: Any) -> Any:
    """Return a message in the shape Runner expects.

    - If caller provides an ADK-style message dict ({role, parts}), pass it through.
    - Otherwise fall back to a simple string.
    """

    if isinstance(new_message, dict) and isinstance(new_message.get("role"), str):
        parts = new_message.get("parts")
        if isinstance(parts, list):
            role = str(new_message.get("role"))
            msg_parts: list[Part] = []
            for p in parts:
                if isinstance(p, dict) and isinstance(p.get("text"), str) and p.get("text"):
                    msg_parts.append(Part(text=str(p.get("text"))))
            if not msg_parts:
                msg_parts = [Part(text="")]
            return Content(role=role, parts=msg_parts)
    if isinstance(new_message, str):
        return Content(role="user", parts=[Part(text=new_message)])
    if isinstance(new_message, dict):
        parts = new_message.get("parts")
        if isinstance(parts, list) and parts:
            part0 = parts[0]
            if isinstance(part0, dict) and isinstance(part0.get("text"), str):
                return part0["text"]
        if isinstance(new_message.get("text"), str):
            return new_message["text"]
    return str(new_message)


def _content_text(msg: Any) -> str:
    if isinstance(msg, Content):
        texts: list[str] = []
        for p in msg.parts or []:
            t = getattr(p, "text", None)
            if isinstance(t, str) and t:
                texts.append(t)
        return "\n".join(texts)
    if isinstance(msg, str):
        return msg
    return str(msg)


def _collect_text_values(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        if value:
            out.append(value)
        return
    if isinstance(value, dict):
        for v in value.values():
            _collect_text_values(v, out)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text_values(item, out)
        return


@app.get("/list-apps")
async def list_apps() -> list[str]:
    return ["research_pipeline_agent", "rag_agent_shard"]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), kind: str = Form("auto")):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    category = _classify_upload(file.filename, kind)
    if category == "video":
        target_dir = _VIDEO_UPLOAD_DIR
    else:
        target_dir = _DATASETS_DIR

    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    stem = Path(safe_name).stem
    ext = Path(safe_name).suffix
    suffix = secrets.token_hex(4)
    stored_name = f"{stem}-{suffix}{ext}"
    stored_path = target_dir / stored_name

    content = await file.read()
    stored_path.write_bytes(content)

    return JSONResponse(
        {
            "success": True,
            "kind": category,
            "original_filename": safe_name,
            "stored_filename": stored_name,
            "stored_path": _posix(stored_path),
        }
    )


@app.get("/kb/stats")
async def kb_stats():
    try:
        return JSONResponse(get_stats())
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"totalDocs": 0, "collection": "router_multimodal_items", "model": "hash-v1", "byType": {}, "error": str(exc)},
            status_code=200,
        )


@app.get("/kb/documents")
async def kb_documents(limit: int = 200):
    try:
        data = list_documents(limit=limit)
        return JSONResponse(data)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"documents": [], "count": 0, "error": str(exc)}, status_code=200)


@app.post("/kb/search")
async def kb_search(request: Request):
    body = await request.json()
    query = str(body.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    raw_top_k = body.get("top_k", body.get("nResults", 5))
    try:
        top_k = max(1, min(50, int(raw_top_k)))
    except Exception:
        top_k = 5

    try:
        data = vector_search(query=query, top_k=top_k)
        return JSONResponse(data)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"query": query, "documents": [], "total_found": 0, "error": str(exc)},
            status_code=200,
        )


@app.post("/kb/upload")
async def kb_upload(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    overlap: int = Form(100),
    doc_type: str = Form("note"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    raw = await file.read()
    text = _extract_upload_text(file.filename, raw)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file has no extractable text")

    safe_name = Path(file.filename).name
    source = f"upload://{safe_name}"
    doc_id = f"upload_{Path(safe_name).stem}_{secrets.token_hex(4)}"
    metadata = {"source": source, "type": doc_type, "filename": safe_name}
    result = add_document(
        content=text,
        metadata=metadata,
        doc_id=doc_id,
        chunk_size=max(100, min(4000, int(chunk_size))),
        overlap=max(0, min(1000, int(overlap))),
    )

    if result.get("status") != "success":
        return JSONResponse(
            {"success": False, "error": result.get("error", "Ingestion failed")},
            status_code=400,
        )

    return JSONResponse(
        {
            "success": True,
            "filename": safe_name,
            "doc_id": result.get("doc_id"),
            "chunks_created": result.get("chunks_added", 0),
            "total_chars": len(text),
            "upsert_response": result.get("upsert_response", {}),
        }
    )


@app.get("/final-project/report")
async def final_project_report():
    if not _FINAL_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Report not found at {_FINAL_REPORT_PATH}")

    try:
        data = json.loads(_FINAL_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to read report: {exc}") from exc
    return JSONResponse(data)


@app.post("/final-project/run")
async def run_final_project(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    router_url = str(body.get("router_url", _DEFAULT_ROUTER_URL)).strip()
    source = str(body.get("source", "api_server")).strip() or "api_server"

    try:
        limit_videos = int(body.get("limit_videos", 3))
    except Exception:
        limit_videos = 3
    try:
        limit_images = int(body.get("limit_images", 3))
    except Exception:
        limit_images = 3

    cmd = [
        sys.executable,
        "-m",
        "orchestrator.final_project",
        "--router-url",
        router_url,
        "--limit-videos",
        str(limit_videos),
        "--limit-images",
        str(limit_images),
        "--source",
        source,
    ]
    proc = await asyncio.to_thread(
        subprocess.run,
        cmd,
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    payload: dict[str, Any] = {
        "status": "ok" if proc.returncode == 0 else "error",
        "return_code": proc.returncode,
        "command": cmd,
        "stdout": (proc.stdout or "")[-4000:],
        "stderr": (proc.stderr or "")[-4000:],
        "report_path": _posix(_FINAL_REPORT_PATH),
    }

    if _FINAL_REPORT_PATH.exists():
        try:
            payload["report"] = json.loads(_FINAL_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            payload["report"] = None

    status = 200 if proc.returncode == 0 else 500
    return JSONResponse(payload, status_code=status)


@app.post("/apps/{app_name}/users/{user_id}/sessions")
async def create_session(app_name: str, user_id: str, request: Request):
    _get_agent_for_app(app_name)
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = body.get("id") if isinstance(body, dict) else None
    try:
        session = await _SESSION_SERVICE.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
        return {"id": session.id}
    except AlreadyExistsError:
        if session_id:
            return {"id": str(session_id)}
        raise
    except Exception as exc:  # noqa: BLE001
        # Some environments can surface AlreadyExistsError instances that don't match the imported symbol
        # due to namespace package/version quirks. Treat "already exists" as an idempotent success.
        cls_name = exc.__class__.__name__
        msg = str(exc)
        if session_id and (cls_name == "AlreadyExistsError" or "already exists" in msg.lower()):
            return {"id": str(session_id)}
        raise


@app.get("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
async def get_session(app_name: str, user_id: str, session_id: str):
    _get_agent_for_app(app_name)
    return {"id": session_id, "app_name": app_name, "user_id": user_id}


@app.post("/run")
async def run(request: Request):
    body = await request.json()
    app_name = body.get("app_name")
    user_id = body.get("user_id")
    session_id = body.get("session_id")
    new_message = body.get("new_message")

    if not app_name or not user_id or not session_id:
        raise HTTPException(status_code=400, detail="app_name, user_id, session_id are required")

    runner = _get_runner(str(app_name))
    normalized_message = _normalize_new_message(new_message)

    logger.info("/run app=%s user=%s session=%s message=%s", app_name, user_id, session_id, _content_text(normalized_message))

    responses: list[str] = []
    async for event in runner.run_async(
        user_id=str(user_id),
        session_id=str(session_id),
        new_message=normalized_message,
    ):
        if hasattr(event, "content") and event.content and hasattr(event.content, "parts"):
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    responses.append(part.text)

    return JSONResponse(
        {
            "response": "\n".join(responses) if responses else "",
            "session_id": session_id,
            "user_id": user_id,
            "app_name": app_name,
        }
    )


@app.post("/run_sse")
async def run_sse(request: Request):
    body = await request.json()
    app_name = body.get("app_name")
    user_id = body.get("user_id")
    session_id = body.get("session_id")
    new_message = body.get("new_message")

    if not app_name or not user_id or not session_id:
        raise HTTPException(status_code=400, detail="app_name, user_id, session_id are required")

    runner = _get_runner(str(app_name))
    normalized_message = _normalize_new_message(new_message)

    async def gen():
        full_text: list[str] = []
        try:
            async for event in runner.run_async(user_id=str(user_id), session_id=str(session_id), new_message=normalized_message):
                author = getattr(event, "author", None) or getattr(event, "agent_name", None) or "agent"

                try:
                    d = event.model_dump() if hasattr(event, "model_dump") else None
                except Exception:
                    d = None

                if isinstance(d, dict):
                    content = d.get("content")
                    if isinstance(content, dict):
                        parts = content.get("parts")
                        if isinstance(parts, list):
                            for p in parts:
                                if not isinstance(p, dict):
                                    continue
                                fc = p.get("function_call")
                                if isinstance(fc, dict) and isinstance(fc.get("name"), str) and fc.get("name"):
                                    payload = {
                                        "type": "tool_call",
                                        "author": str(author),
                                        "tool_name": str(fc.get("name")),
                                    }
                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                    await asyncio.sleep(0)
                                fr = p.get("function_response")
                                if isinstance(fr, dict) and isinstance(fr.get("name"), str) and fr.get("name"):
                                    payload = {
                                        "type": "tool_result",
                                        "author": str(author),
                                        "tool_name": str(fr.get("name")),
                                    }
                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                    await asyncio.sleep(0)

                                    # Emit tool response text so UI can parse artifacts like ADK_ARTIFACT.
                                    tool_texts: list[str] = []
                                    response = fr.get("response")
                                    content = fr.get("content")

                                    if isinstance(response, str) and response:
                                        tool_texts.append(response)
                                    elif isinstance(response, dict):
                                        r_content = response.get("content")
                                        if isinstance(r_content, dict):
                                            r_parts = r_content.get("parts")
                                            if isinstance(r_parts, list):
                                                for rp in r_parts:
                                                    if isinstance(rp, dict):
                                                        rt = rp.get("text")
                                                        if isinstance(rt, str) and rt:
                                                            tool_texts.append(rt)

                                    if isinstance(content, dict):
                                        c_parts = content.get("parts")
                                        if isinstance(c_parts, list):
                                            for cp in c_parts:
                                                if isinstance(cp, dict):
                                                    ct = cp.get("text")
                                                    if isinstance(ct, str) and ct:
                                                        tool_texts.append(ct)

                                    # Fallback: ADK response payload shapes can vary by runtime.
                                    # Recursively collect strings and keep likely tool output text.
                                    if not tool_texts:
                                        candidates: list[str] = []
                                        _collect_text_values(fr, candidates)
                                        for candidate in candidates:
                                            if "ADK_ARTIFACT:" in candidate:
                                                tool_texts.append(candidate)

                                        # Only use fallback candidates if they look like real output (not IDs/hashes).
                                        if not tool_texts:
                                            for candidate in candidates:
                                                s = candidate.strip()
                                                if not s or len(s) < 20:
                                                    continue
                                                # Skip if it looks like an ID/hash (mostly alphanumeric, no spaces)
                                                if len(s) < 100 and not any(c in s for c in [' ', '\n', '.', ',', ':', ';']):
                                                    continue
                                                if len(s) <= 6000:
                                                    tool_texts.append(s)
                                                    break

                                    if tool_texts:
                                        tool_payload = {
                                            "type": "agent_message",
                                            "author": str(author),
                                            "content": {"parts": [{"text": "\n".join(tool_texts)}]},
                                        }
                                        yield f"data: {json.dumps(tool_payload, ensure_ascii=False)}\n\n"
                                        await asyncio.sleep(0)

                if hasattr(event, "content") and event.content and hasattr(event.content, "parts"):
                    for part in event.content.parts:
                        text = getattr(part, "text", None)
                        if not text:
                            continue
                        full_text.append(text)
                        payload = {
                            "type": "agent_message",
                            "author": str(author),
                            "content": {"parts": [{"text": text}]},
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0)

            final_payload = {
                "type": "final",
                "author": "final",
                "is_final_response": True,
                "content": {"parts": [{"text": "\n".join(full_text)}]},
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            err_payload = {"type": "error", "author": "server", "error": str(exc)}
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    host = os.environ.get("ADK_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("ADK_BACKEND_PORT", "8001"))
    reload = os.environ.get("ADK_BACKEND_RELOAD", "0").lower() in {"1", "true", "yes"}
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=os.environ.get("ADK_BACKEND_LOG_LEVEL", "info"),
    )
