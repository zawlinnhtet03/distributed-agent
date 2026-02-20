import ray
from services.video_service import process_video
from services.vector_service import store_embeddings
from services.rag_service import generate_summary
from common.models import VideoTask

@ray.remote
def pipeline(task: VideoTask):
    result = process_video(task)
    result = store_embeddings(result, metadata=task.metadata)
    summary = generate_summary(result, metadata=task.metadata)

    return {
        "video_id": task.video_id,
        "video_path": task.video_path,
        "transcript": result.transcript,
        "summary": summary,
        "embedding_size": len(result.embeddings),
    }
