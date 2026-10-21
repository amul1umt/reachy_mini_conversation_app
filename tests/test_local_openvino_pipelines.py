"""Tests for the STT/LLM/VLM OpenVINO GenAI pipeline wrappers.

These lock in call shapes confirmed by live-testing against real
openvino-genai==2026.2.1 on real Intel Core Ultra hardware during planning
(OpenVINO/whisper-tiny-fp16-ov and OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov):
`WhisperPipeline.generate()` and `LLMPipeline.generate()` both return objects
with a `.texts: list[str]`, and `LLMPipeline.generate()` requires its input
wrapped in `openvino_genai.ChatHistory`, not a plain list of dicts.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from reachy_mini_conversation_app.local_openvino.session import ChatMessage
from reachy_mini_conversation_app.local_openvino.pipelines.llm import LocalLLM
from reachy_mini_conversation_app.local_openvino.pipelines.stt import WhisperSTT
from reachy_mini_conversation_app.local_openvino.pipelines.vlm import LocalVLM


class _FakeResults:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts


def _stub_openvino_genai(monkeypatch: pytest.MonkeyPatch, module_name: str, fake_module: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, module_name, fake_module)


def test_whisper_stt_extracts_the_first_text_from_decoded_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """WhisperSTT should return .texts[0], not a stringified results object."""

    class _FakeWhisperPipeline:
        def __init__(self, model_path: str, device: str) -> None:
            self.device = device

        def generate(self, samples: np.ndarray) -> _FakeResults:
            assert samples.dtype == np.float32
            return _FakeResults([" hello there"])

    fake_module = types.ModuleType("openvino_genai")
    fake_module.WhisperPipeline = _FakeWhisperPipeline  # type: ignore[attr-defined]
    _stub_openvino_genai(monkeypatch, "openvino_genai", fake_module)

    stt = WhisperSTT(tmp_path, "CPU")
    result = stt.transcribe(np.zeros(1600, dtype=np.int16).tobytes(), sample_rate=16000)

    assert result == " hello there"


def test_whisper_stt_returns_empty_string_for_no_texts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An empty .texts list should produce an empty transcript, not an IndexError."""

    class _FakeWhisperPipeline:
        def __init__(self, model_path: str, device: str) -> None:
            pass

        def generate(self, samples: np.ndarray) -> _FakeResults:
            return _FakeResults([])

    fake_module = types.ModuleType("openvino_genai")
    fake_module.WhisperPipeline = _FakeWhisperPipeline  # type: ignore[attr-defined]
    _stub_openvino_genai(monkeypatch, "openvino_genai", fake_module)

    stt = WhisperSTT(tmp_path, "CPU")
    result = stt.transcribe(np.zeros(1600, dtype=np.int16).tobytes(), sample_rate=16000)

    assert result == ""


def test_local_llm_wraps_messages_in_chat_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """LocalLLM must wrap plain dicts in ChatHistory before calling generate(), not pass a bare list."""
    captured: dict[str, object] = {}

    class _FakeChatHistory:
        def __init__(self, messages: list[dict[str, str]]) -> None:
            captured["messages"] = messages

    class _FakeLLMPipeline:
        def __init__(self, model_path: str, device: str) -> None:
            pass

        def generate(self, history: _FakeChatHistory) -> _FakeResults:
            captured["history"] = history
            return _FakeResults(["Hello! How can I help?"])

    fake_module = types.ModuleType("openvino_genai")
    fake_module.LLMPipeline = _FakeLLMPipeline  # type: ignore[attr-defined]
    fake_module.ChatHistory = _FakeChatHistory  # type: ignore[attr-defined]
    _stub_openvino_genai(monkeypatch, "openvino_genai", fake_module)

    llm = LocalLLM(tmp_path, "CPU")
    reply = llm.generate([ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")])

    assert reply == "Hello! How can I help?"
    assert captured["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert isinstance(captured["history"], _FakeChatHistory)


def test_local_llm_drops_images_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A text-only LocalLLM should ignore attached images rather than crash."""

    class _FakeChatHistory:
        def __init__(self, messages: list[dict[str, str]]) -> None:
            pass

    class _FakeLLMPipeline:
        def __init__(self, model_path: str, device: str) -> None:
            pass

        def generate(self, history: object) -> _FakeResults:
            return _FakeResults(["ok"])

    fake_module = types.ModuleType("openvino_genai")
    fake_module.LLMPipeline = _FakeLLMPipeline  # type: ignore[attr-defined]
    fake_module.ChatHistory = _FakeChatHistory  # type: ignore[attr-defined]
    _stub_openvino_genai(monkeypatch, "openvino_genai", fake_module)

    llm = LocalLLM(tmp_path, "CPU")
    with caplog.at_level("DEBUG"):
        reply = llm.generate([ChatMessage(role="user", content="", image=b"fake-jpeg")])

    assert reply == "ok"


def _sample_jpeg_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    image = Image.new("RGB", (4, 3), color=(10, 20, 30))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_local_vlm_flattens_prompt_and_converts_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """LocalVLM should render a flat prompt and pass images as openvino.Tensor, not raw bytes."""
    pytest.importorskip("PIL")
    pytest.importorskip("openvino")
    captured: dict[str, object] = {}

    class _FakeVLMPipeline:
        def __init__(self, model_path: str, device: str) -> None:
            pass

        def generate(self, prompt: str, *, images: list[object]) -> _FakeResults:
            captured["prompt"] = prompt
            captured["images"] = images
            return _FakeResults(["I see a red square."])

    fake_module = types.ModuleType("openvino_genai")
    fake_module.VLMPipeline = _FakeVLMPipeline  # type: ignore[attr-defined]
    _stub_openvino_genai(monkeypatch, "openvino_genai", fake_module)

    vlm = LocalVLM(tmp_path, "CPU")
    reply = vlm.generate(
        [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What do you see?"),
            ChatMessage(role="user", content="", image=_sample_jpeg_bytes()),
        ]
    )

    assert reply == "I see a red square."
    assert captured["prompt"] == "System: You are a helpful assistant.\nUser: What do you see?"
    images = captured["images"]
    assert isinstance(images, list) and len(images) == 1
    assert images[0].shape == [1, 3, 4, 3]


def test_local_vlm_without_images_still_generates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A VLM turn with no attached image should call generate() without the images kwarg."""
    calls: list[tuple[str, object]] = []

    class _FakeVLMPipeline:
        def __init__(self, model_path: str, device: str) -> None:
            pass

        def generate(self, prompt: str) -> _FakeResults:
            calls.append((prompt, None))
            return _FakeResults(["Sure, how can I help?"])

    fake_module = types.ModuleType("openvino_genai")
    fake_module.VLMPipeline = _FakeVLMPipeline  # type: ignore[attr-defined]
    _stub_openvino_genai(monkeypatch, "openvino_genai", fake_module)

    vlm = LocalVLM(tmp_path, "CPU")
    reply = vlm.generate([ChatMessage(role="user", content="hi")])

    assert reply == "Sure, how can I help?"
    assert calls == [("User: hi", None)]
