from common.models import ProcessingResult


def generate_summary(result: ProcessingResult, metadata: dict | None = None) -> str:
    source = (metadata or {}).get("source", "unknown")
    snippet = " ".join(result.transcript.split())
    excerpt = snippet[:220] + ("..." if len(snippet) > 220 else "")
    return f"[{result.video_id}] source={source}; transcript={excerpt}"
