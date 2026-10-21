"""Kokoro ONNX text-to-speech pipeline wrapper.

Kokoro's dynamic-rank ONNX graph is not compatible with the OpenVINO
execution provider (the `kokoro-onnx`/`pykokoro` projects document that an
OpenVINO provider request currently fails to initialize and falls back to
plain CPU), so unlike the STT/LLM roles, this one runs on ONNX Runtime's CPU
execution provider rather than a selected OpenVINO device. Re-verify the
`kokoro_onnx.Kokoro` constructor/`create()` call shape against the installed
version on real hardware, since this couldn't be exercised without a Core
Ultra machine during planning.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _resample_linear(samples: NDArray[np.float32], *, orig_rate: int, target_rate: int) -> NDArray[np.float32]:
    """Resample ``samples`` with basic linear interpolation."""
    if orig_rate == target_rate or samples.size == 0:
        return samples
    target_length = int(round(samples.shape[0] * target_rate / orig_rate))
    original_indices = np.arange(samples.shape[0])
    target_indices = np.linspace(0, samples.shape[0] - 1, num=target_length)
    return np.interp(target_indices, original_indices, samples).astype(np.float32)


class KokoroTTS:
    """Wraps a local Kokoro ONNX model for text-to-speech synthesis."""

    def __init__(self, model_path: Path, voices_path: Path, *, voice: str = "af_heart") -> None:
        """Load the Kokoro ONNX model and its voice-embedding table."""
        from kokoro_onnx import Kokoro

        self._pipeline = Kokoro(str(model_path), str(voices_path))
        self._voice = voice

    @classmethod
    def from_snapshot(cls, snapshot_dir: Path, *, voice: str = "af_heart") -> "KokoroTTS":
        """Locate the ONNX model and combined voices file inside a downloaded Kokoro snapshot.

        Expects the `fastrtc/kokoro-onnx` layout (`kokoro-v1.0.onnx` +
        `voices-v1.0.bin`, a single file covering all voices) - confirmed via
        its Hub file listing at plan time. Other Kokoro ONNX mirrors (e.g.
        `onnx-community/Kokoro-82M-*-ONNX`) instead ship one file per voice
        under `voices/<name>.bin` and are NOT compatible with this loader.
        """
        model_path = next(snapshot_dir.rglob("*.onnx"), None)
        voices_path = next(snapshot_dir.rglob("voices*.bin"), None)
        if model_path is None or voices_path is None:
            raise FileNotFoundError(
                f"Could not find a Kokoro model/combined-voices file under {snapshot_dir}; "
                "this loader needs the fastrtc/kokoro-onnx-style layout, see from_snapshot()'s docstring."
            )
        return cls(model_path, voices_path, voice=voice)

    def synthesize(self, text: str, *, output_sample_rate: int) -> bytes:
        """Return 16-bit PCM mono audio at ``output_sample_rate`` for ``text``."""
        samples, sample_rate = self._pipeline.create(text, voice=self._voice)
        samples = _resample_linear(
            np.asarray(samples, dtype=np.float32), orig_rate=sample_rate, target_rate=output_sample_rate
        )
        pcm16 = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
        return bytes(pcm16.tobytes())
