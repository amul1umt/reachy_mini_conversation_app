"""OpenVINO GenAI LLM pipeline wrapper for text-only chat generation.

Verified live against openvino-genai==2026.2.1 on CPU
(OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov):
`LLMPipeline.generate()` does NOT accept a plain `list[dict]` - it must be
wrapped in `openvino_genai.ChatHistory(messages)` (which itself accepts
exactly the `[{"role": ..., "content": ...}]` shape built here). The result
is a `DecodedResults` whose `.texts` is a `list[str]`; `.texts[0]` is the
reply. Device selection (CPU/GPU/NPU/AUTO) was not exercised on real Core
Ultra hardware during planning, only CPU.
"""

import logging
from pathlib import Path

from reachy_mini_conversation_app.local_openvino.session import ChatMessage


logger = logging.getLogger(__name__)


class LocalLLM:
    """Wraps an OpenVINO GenAI LLM pipeline for single-turn chat generation.

    Text-only: any `ChatMessage.image` is dropped (see `pipelines/vlm.py` for
    the vision-capable counterpart).
    """

    def __init__(self, model_path: Path, device: str) -> None:
        """Load the LLM pipeline from ``model_path`` onto ``device``."""
        import openvino_genai as ov_genai

        self._ov_genai = ov_genai
        self._pipeline = ov_genai.LLMPipeline(str(model_path), device=device)

    def generate(self, messages: list[ChatMessage]) -> str:
        """Generate a reply for a conversation history of messages, ignoring any images."""
        if any(message.image is not None for message in messages):
            logger.debug("Dropping attached image(s): this LLM is text-only, not a VLM.")
        prompt_messages = [{"role": message.role, "content": message.content} for message in messages]
        history = self._ov_genai.ChatHistory(prompt_messages)
        result = self._pipeline.generate(history)
        return str(result.texts[0]) if result.texts else ""
