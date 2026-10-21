"""Silero VAD (ONNX), replacing the energy-threshold VoiceActivityDetector when available.

Not part of the model catalog/settings UI - there's exactly one well-known
Silero VAD ONNX export (`_SILERO_VAD_REPO_ID` below), downloaded and used
transparently. Falls back to `VoiceActivityDetector` automatically if the
model can't be downloaded or loaded, so the realtime session's audio pipeline
keeps working either way (graceful degradation, not a hard dependency).

Input/output tensor names and the single-tensor recurrent state shape
(`(2, 1, 128)`) match Silero VAD v4+'s ONNX export as of plan time; re-verify
against the installed model on real hardware; Silero has changed its calling
convention across versions (v3 used separate `h`/`c` state tensors) and this
couldn't be exercised without a Core Ultra machine during planning.
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.local_openvino.vad import Transition, VoiceActivityDetector


logger = logging.getLogger(__name__)

SILERO_VAD_REPO_ID = "deepghs/silero-vad-onnx"

_CHUNK_SAMPLES = 512  # Silero VAD's recommended chunk size at 16 kHz
_SPEECH_THRESHOLD = 0.5


class SileroVAD:
    """Streaming voice-activity detector backed by Silero VAD's ONNX export."""

    def __init__(self, model_path: Path, *, sample_rate: int = 16000, silence_hold_s: float = 0.6) -> None:
        """Load the Silero VAD ONNX model."""
        import onnxruntime

        self._session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sample_rate = np.array(sample_rate, dtype=np.int64)
        self._buffer: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._silence_hold_samples = int(silence_hold_s * sample_rate)
        self._speaking = False
        self._silence_run_samples = 0

    def push(self, pcm16_chunk: bytes) -> Transition | None:
        """Feed one chunk of 16-bit PCM audio; return a transition, if any."""
        samples = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, samples])

        transition: Transition | None = None
        while self._buffer.shape[0] >= _CHUNK_SAMPLES:
            frame, self._buffer = self._buffer[:_CHUNK_SAMPLES], self._buffer[_CHUNK_SAMPLES:]
            probability = self._infer(frame)
            transition = self._update_state(probability, frame.shape[0]) or transition
        return transition

    def _infer(self, frame: NDArray[np.float32]) -> float:
        outputs = self._session.run(None, {"input": frame[None, :], "state": self._state, "sr": self._sample_rate})
        probability, self._state = outputs[0], outputs[1]
        return float(probability.reshape(-1)[0])

    def _update_state(self, probability: float, sample_count: int) -> Transition | None:
        if probability >= _SPEECH_THRESHOLD:
            self._silence_run_samples = 0
            if not self._speaking:
                self._speaking = True
                return "speech_started"
            return None
        if self._speaking:
            self._silence_run_samples += sample_count
            if self._silence_run_samples >= self._silence_hold_samples:
                self._speaking = False
                self._silence_run_samples = 0
                return "speech_stopped"
        return None


def load_vad(model_path: Path | None, *, sample_rate: int = 16000) -> "SileroVAD | VoiceActivityDetector":
    """Load Silero VAD if a model path is given and it loads cleanly; else the energy-based fallback."""
    if model_path is not None:
        try:
            return SileroVAD(model_path, sample_rate=sample_rate)
        except Exception as exc:
            logger.warning("Failed to load Silero VAD (%s); falling back to energy-threshold detection.", exc)
    return VoiceActivityDetector(sample_rate=sample_rate)
