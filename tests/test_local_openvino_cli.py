"""Tests for the local-openvino-server CLI's device-fallback and VAD download logic."""

import sys
import types
import argparse
from pathlib import Path

import pytest

from reachy_mini_conversation_app.local_openvino.cli import (
    _load_tts,
    _resolve_vad_model_path,
    _build_with_device_fallback,
)


def test_build_with_device_fallback_uses_the_requested_device_when_it_works() -> None:
    """A device that constructs cleanly should be used as-is, with no fallback."""
    calls: list[str] = []

    def build(device: str) -> str:
        calls.append(device)
        return f"pipeline-on-{device}"

    result = _build_with_device_fallback(build, "GPU", role="STT")

    assert result == "pipeline-on-GPU"
    assert calls == ["GPU"]


def test_build_with_device_fallback_retries_on_auto_when_the_device_fails() -> None:
    """A device that fails to construct (e.g. an NPU without a driver) should fall back to AUTO."""
    calls: list[str] = []

    def build(device: str) -> str:
        calls.append(device)
        if device == "NPU":
            raise RuntimeError("no NPU driver found")
        return f"pipeline-on-{device}"

    result = _build_with_device_fallback(build, "NPU", role="LLM")

    assert result == "pipeline-on-AUTO"
    assert calls == ["NPU", "AUTO"]


def test_build_with_device_fallback_reraises_when_auto_itself_fails() -> None:
    """If AUTO also fails, the original failure should propagate rather than loop forever."""

    def build(device: str) -> str:
        raise RuntimeError(f"no device available: {device}")

    with pytest.raises(RuntimeError, match="AUTO"):
        _build_with_device_fallback(build, "AUTO", role="STT")


def test_resolve_vad_model_path_returns_none_when_download_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed Silero VAD download should return None (triggering the energy-based fallback), not raise."""

    def fake_download(repo_id: str, *, on_progress: object = None) -> Path:
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino.cli.ensure_model_downloaded", fake_download)

    assert _resolve_vad_model_path() is None


def test_resolve_vad_model_path_returns_none_when_no_onnx_file_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A downloaded snapshot without any .onnx file should also fall back to None, not crash."""
    monkeypatch.setattr(
        "reachy_mini_conversation_app.local_openvino.cli.ensure_model_downloaded", lambda repo_id, **kwargs: tmp_path
    )

    assert _resolve_vad_model_path() is None


def test_resolve_vad_model_path_finds_the_onnx_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A successful download should return the path to its silero_vad.onnx file."""
    (tmp_path / "silero_vad.onnx").write_bytes(b"")
    monkeypatch.setattr(
        "reachy_mini_conversation_app.local_openvino.cli.ensure_model_downloaded", lambda repo_id, **kwargs: tmp_path
    )

    assert _resolve_vad_model_path() == tmp_path / "silero_vad.onnx"


def test_load_tts_warns_but_still_loads_when_device_is_not_cpu_or_auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Requesting a non-CPU device for TTS should warn (Kokoro is CPU-only) but still attempt to load it."""
    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"")
    monkeypatch.setattr(
        "reachy_mini_conversation_app.local_openvino.cli.ensure_model_downloaded", lambda repo_id, **kwargs: tmp_path
    )

    fake_module = types.ModuleType("kokoro_onnx")
    fake_module.Kokoro = lambda model_path, voices_path: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_onnx", fake_module)

    args = argparse.Namespace(tts_model="fastrtc/kokoro-onnx", tts_device="GPU")
    with caplog.at_level("WARNING"):
        result = _load_tts(args)

    assert result is not None
    assert "CPU" in caplog.text
