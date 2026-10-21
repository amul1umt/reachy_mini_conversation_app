"""Tests for the local Kokoro ONNX text-to-speech pipeline wrapper."""

import sys
import types
from pathlib import Path

import numpy as np

from reachy_mini_conversation_app.local_openvino.pipelines.tts import KokoroTTS, _resample_linear


def _stub_kokoro_onnx(monkeypatch, fake_kokoro_cls: type) -> None:
    """Stand in for the `kokoro_onnx` package, which isn't installed in tests."""
    fake_module = types.ModuleType("kokoro_onnx")
    fake_module.Kokoro = fake_kokoro_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_onnx", fake_module)


def test_resample_linear_is_a_noop_when_rates_match() -> None:
    """Resampling at the same rate should return the input unchanged."""
    samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    result = _resample_linear(samples, orig_rate=16000, target_rate=16000)

    assert np.array_equal(result, samples)


def test_resample_linear_changes_length_when_rates_differ() -> None:
    """Downsampling should shrink the sample count proportionally."""
    samples = np.linspace(0, 1, 24000, dtype=np.float32)

    result = _resample_linear(samples, orig_rate=24000, target_rate=16000)

    assert result.shape[0] == 16000


def test_from_snapshot_locates_model_and_voices_files(monkeypatch, tmp_path: Path) -> None:
    """from_snapshot should find the .onnx model and combined voices*.bin file."""
    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"")

    captured: dict[str, str] = {}

    class _FakeKokoro:
        def __init__(self, model_path: str, voices_path: str) -> None:
            captured["model_path"] = model_path
            captured["voices_path"] = voices_path

    _stub_kokoro_onnx(monkeypatch, _FakeKokoro)

    tts = KokoroTTS.from_snapshot(tmp_path, voice="af_heart")

    assert captured["model_path"] == str(tmp_path / "kokoro-v1.0.onnx")
    assert captured["voices_path"] == str(tmp_path / "voices-v1.0.bin")
    assert tts._voice == "af_heart"


def test_from_snapshot_raises_when_files_are_missing(tmp_path: Path) -> None:
    """A snapshot without the expected files should fail loudly, not silently."""
    try:
        KokoroTTS.from_snapshot(tmp_path)
    except FileNotFoundError as exc:
        assert "fastrtc/kokoro-onnx" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_synthesize_returns_resampled_pcm16_bytes(monkeypatch, tmp_path: Path) -> None:
    """synthesize() should resample to the requested rate and return 16-bit PCM bytes."""
    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"")

    class _FakeKokoro:
        def __init__(self, model_path: str, voices_path: str) -> None:
            pass

        def create(self, text: str, *, voice: str) -> tuple[np.ndarray, int]:
            assert text == "hello"
            assert voice == "af_heart"
            return np.full(2400, 0.5, dtype=np.float32), 24000

    _stub_kokoro_onnx(monkeypatch, _FakeKokoro)
    tts = KokoroTTS.from_snapshot(tmp_path)

    audio = tts.synthesize("hello", output_sample_rate=16000)

    assert isinstance(audio, bytes)
    sample_count = len(audio) // 2
    assert sample_count == 1600
    decoded = np.frombuffer(audio, dtype=np.int16)
    assert decoded[0] > 0
