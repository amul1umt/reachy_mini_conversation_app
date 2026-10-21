"""OpenVINO GenAI Whisper speech-to-text pipeline wrapper.

Verified live against openvino-genai==2026.2.1 on CPU (OpenVINO/whisper-tiny-fp16-ov):
`WhisperPipeline.generate(raw_speech_input: list[float])` returns a
`WhisperDecodedResults` object whose `.texts` is a `list[str]`; `.texts[0]` is
the transcript. Device selection (CPU/GPU/NPU/AUTO) and model download were
not exercised on real Core Ultra hardware during planning, only CPU.
"""

from pathlib import Path

import numpy as np


class WhisperSTT:
    """Wraps an OpenVINO GenAI Whisper pipeline for batch transcription."""

    def __init__(self, model_path: Path, device: str) -> None:
        """Load the Whisper pipeline from ``model_path`` onto ``device``."""
        import openvino_genai as ov_genai

        self._pipeline = ov_genai.WhisperPipeline(str(model_path), device=device)

    def transcribe(self, pcm16_audio: bytes, *, sample_rate: int) -> str:
        """Transcribe one complete utterance of 16-bit PCM audio."""
        samples = np.frombuffer(pcm16_audio, dtype=np.int16).astype(np.float32) / 32768.0
        result = self._pipeline.generate(samples)
        return str(result.texts[0]) if result.texts else ""
