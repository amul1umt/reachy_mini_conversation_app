"""Tests for the Silero VAD wrapper and its fallback to the energy-threshold detector."""

import sys
import types
from pathlib import Path

import numpy as np

from reachy_mini_conversation_app.local_openvino.vad import VoiceActivityDetector
from reachy_mini_conversation_app.local_openvino.silero_vad import SileroVAD, load_vad


def _stub_onnxruntime(monkeypatch, probabilities: list[float]) -> None:
    """Stand in for onnxruntime, returning ``probabilities`` in order, then repeating the last one."""
    remaining = list(probabilities)
    last_value = probabilities[-1] if probabilities else 0.0

    class _FakeSession:
        def __init__(self, model_path: str, providers: list[str]) -> None:
            pass

        def run(self, output_names, inputs):
            probability = remaining.pop(0) if remaining else last_value
            state = inputs["state"]
            return [np.array([[probability]], dtype=np.float32), state]

    fake_module = types.ModuleType("onnxruntime")
    fake_module.InferenceSession = _FakeSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_module)


def test_load_vad_falls_back_to_energy_threshold_when_model_path_is_none() -> None:
    """Without a model path, load_vad should return the energy-threshold detector."""
    vad = load_vad(None)

    assert isinstance(vad, VoiceActivityDetector)


def test_load_vad_falls_back_when_construction_fails(monkeypatch, tmp_path: Path) -> None:
    """A model that fails to load (bad file, missing onnxruntime, ...) should fall back gracefully."""
    fake_module = types.ModuleType("onnxruntime")

    def _raise(*args, **kwargs):
        raise RuntimeError("invalid model")

    fake_module.InferenceSession = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_module)

    vad = load_vad(tmp_path / "silero_vad.onnx")

    assert isinstance(vad, VoiceActivityDetector)


def test_silero_vad_reports_speech_started_then_stopped(monkeypatch, tmp_path: Path) -> None:
    """A run of high speech probabilities followed by low ones should report start then stop."""
    # 1 loud frame, then enough silent frames to exceed the 0.6s hold at 512-sample chunks.
    silence_frames_needed = int(0.6 * 16000 / 512) + 2
    _stub_onnxruntime(monkeypatch, [0.9] + [0.0] * silence_frames_needed)
    vad = SileroVAD(tmp_path / "silero_vad.onnx")

    transitions = []
    chunk = np.zeros(512, dtype=np.int16).tobytes()
    for _ in range(1 + silence_frames_needed):
        transition = vad.push(chunk)
        if transition:
            transitions.append(transition)

    assert transitions == ["speech_started", "speech_stopped"]
