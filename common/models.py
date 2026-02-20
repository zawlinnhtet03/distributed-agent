from pydantic import BaseModel
from typing import List

class VideoTask(BaseModel):
    video_id: str
    video_path: str
    metadata: dict

class ProcessingResult(BaseModel):
    video_id: str
    transcript: str
    embeddings: List[float]
