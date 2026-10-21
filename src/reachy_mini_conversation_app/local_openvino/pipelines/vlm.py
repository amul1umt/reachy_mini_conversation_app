"""OpenVINO GenAI VLM pipeline wrapper for vision-capable chat generation.

NOT live-tested (no VLM model was downloaded during planning; only STT and
the text-only LLM were verified against real openvino-genai==2026.2.1 calls
- see pipelines/stt.py and pipelines/llm.py). Confirmed via introspection
that `VLMPipeline.generate()` differs significantly from `LLMPipeline.
generate()`: it takes a single `prompt: str` plus `images: list[openvino.
Tensor]` (NOT raw JPEG bytes, and NOT a ChatHistory), and is normally used
statefully via `start_chat()`/`generate(prompt, images=...)`/`finish_chat()`.
This wrapper instead flattens the conversation into one prompt string per
call and stays stateless, matching how `pipelines/llm.py`'s `LocalLLM` is
called from `session.py`. Re-verify on real hardware: the prompt format, the
exact `images` tensor layout (assumed NHWC uint8 RGB, following common
OpenVINO GenAI VLM sample conventions), and whether the stateful
start_chat/generate/finish_chat sequence gives meaningfully better results
than this stateless one.
"""

import logging
from io import BytesIO
from pathlib import Path

import numpy as np

from reachy_mini_conversation_app.local_openvino.session import ChatMessage


logger = logging.getLogger(__name__)

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}


def _jpeg_bytes_to_tensor(jpeg_bytes: bytes) -> object:
    """Decode JPEG bytes into the NHWC uint8 RGB openvino.Tensor VLMPipeline expects."""
    import openvino as ov
    from PIL import Image

    image = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    return ov.Tensor(array[np.newaxis, :, :, :])


def _render_prompt(messages: list[ChatMessage]) -> str:
    """Flatten a conversation history into one plain-text prompt (images go in separately)."""
    lines = [
        f"{_ROLE_LABELS.get(message.role, message.role)}: {message.content}" for message in messages if message.content
    ]
    return "\n".join(lines)


class LocalVLM:
    """Wraps an OpenVINO GenAI VLM pipeline for chat generation, with optional vision."""

    def __init__(self, model_path: Path, device: str) -> None:
        """Load the VLM pipeline from ``model_path`` onto ``device``."""
        import openvino_genai as ov_genai

        self._pipeline = ov_genai.VLMPipeline(str(model_path), device=device)

    def generate(self, messages: list[ChatMessage]) -> str:
        """Generate a reply for a conversation history, using any attached images."""
        prompt = _render_prompt(messages)
        images = [_jpeg_bytes_to_tensor(message.image) for message in messages if message.image is not None]
        result = self._pipeline.generate(prompt, images=images) if images else self._pipeline.generate(prompt)
        return str(result.texts[0]) if result.texts else ""
