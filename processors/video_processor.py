from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

import ray

from common.models import ProcessingResult, VideoTask

try:
    import ffmpeg
except ImportError:  # pragma: no cover - runtime dependency
    ffmpeg = None

try:
    import whisper
except ImportError:  # pragma: no cover - runtime dependency
    whisper = None


@ray.remote
class VideoProcessor:
    def __init__(self):
        self.mock_mode = os.getenv("MOCK_TRANSCRIPTION", "0") == "1" or ffmpeg is None or whisper is None
        self.model = None

        if not self.mock_mode:
            model_name = os.getenv("WHISPER_MODEL", "base")
            self.model = whisper.load_model(model_name)

    def _extract_audio(self, input_path: str) -> Optional[str]:
        if ffmpeg is None:
            return None

        output_audio = f"temp_audio_{uuid.uuid4().hex}.mp3"
        try:
            if os.path.exists(output_audio):
                os.remove(output_audio)

            (
                ffmpeg.input(input_path)
                .output(output_audio, acodec="libmp3lame", ac=1, ar="16k")
                .overwrite_output()
                .run(quiet=True)
            )
            return output_audio
        except Exception:
            return None

    def _mock_transcript(self, task: VideoTask) -> str:
        filename = Path(task.video_path).name
        return f"Mock transcript for {task.video_id} from file {filename}."

    def process(self, task: VideoTask) -> ProcessingResult:
        if not os.path.exists(task.video_path):
            raise FileNotFoundError(f"Video not found: {task.video_path}")

        if self.mock_mode:
            transcript = self._mock_transcript(task)
            return ProcessingResult(video_id=task.video_id, transcript=transcript, embeddings=[])

        audio_path = self._extract_audio(task.video_path)
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError("Audio extraction failed")

        result = self.model.transcribe(audio_path)
        os.remove(audio_path)

        return ProcessingResult(
            video_id=task.video_id,
            transcript=result["text"],
            embeddings=[],
        )
