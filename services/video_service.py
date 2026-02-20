import ray

from common.models import ProcessingResult, VideoTask
from processors.video_processor import VideoProcessor

_PROCESSOR_ACTOR = None


def _get_processor_actor():
    global _PROCESSOR_ACTOR
    if _PROCESSOR_ACTOR is None:
        _PROCESSOR_ACTOR = VideoProcessor.remote()
    return _PROCESSOR_ACTOR


def process_video(task: VideoTask) -> ProcessingResult:
    processor = _get_processor_actor()
    return ray.get(processor.process.remote(task))
