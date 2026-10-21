"""Per-connection VAD -> STT -> LLM(+tools/vision)(+TTS) orchestration for the local OpenVINO realtime server.

Speaks the OpenAI-Realtime-API event subset that
`reachy_mini_conversation_app.huggingface_realtime.HuggingFaceRealtimeHandler`
already expects from a `local` backend, so the app's client code needs no
changes to talk to this server. See the local-OpenVINO plan's wire-protocol
section for the exact message shapes this module implements.

Tool-calling is prompt-driven, not dependent on any particular OpenVINO GenAI
API surface: available tools are rendered into a system instruction prepended
to the conversation (`_build_tool_instructions`), asking the model to answer
with a `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` block when
it wants to call one - the Hermes/Qwen-style convention many open chat models
already understand natively via their chat template. `_parse_tool_call` reads
that block back out of the model's raw text reply, also accepting a bare JSON
object with no wrapper tags: live-tested against
OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov, which reliably produced the
right JSON shape but sometimes dropped the `<tool_call>` wrapper - smaller
models don't always reproduce instructed formatting exactly.
"""

import re
import json
import uuid
import base64
import asyncio
import logging
from typing import Any, Protocol
from dataclasses import dataclass

from starlette.websockets import WebSocketDisconnect

from reachy_mini_conversation_app.local_openvino.vad import VoiceActivityDetector


logger = logging.getLogger(__name__)

_TOOL_CALL_WRAPPED_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ChatMessage:
    """One conversation turn: role, text, and an optional attached image (JPEG bytes)."""

    role: str
    content: str
    image: bytes | None = None


@dataclass(frozen=True)
class ParsedToolCall:
    """A tool call extracted from the LLM's raw text reply."""

    name: str
    arguments: dict[str, Any]


def _build_tool_instructions(tools: list[dict[str, Any]]) -> str:
    """Render available tools into a system instruction for prompt-driven tool calling."""
    tool_lines = [
        f"- {tool.get('name')}({json.dumps(tool.get('parameters', {}))}): {tool.get('description', '')}"
        for tool in tools
    ]
    return (
        "You can call tools when needed. Available tools:\n"
        + "\n".join(tool_lines)
        + "\n\nTo call a tool, respond with exactly one block in this form and nothing else:\n"
        '<tool_call>\n{"name": "<tool name>", "arguments": {...}}\n</tool_call>\n'
        "Only call a tool when it's actually needed to answer the user; otherwise reply normally."
    )


def _parse_tool_call(text: str) -> ParsedToolCall | None:
    """Extract a tool-call JSON object from the LLM's raw reply, if present.

    Prefers the instructed `<tool_call>{...}</tool_call>` wrapper, but also
    accepts a bare JSON object when the *entire* trimmed reply is one (see
    module docstring for why) - never inside a longer, otherwise-plain-text
    reply, to avoid mistaking an incidental brace for a tool call.
    """
    wrapped = _TOOL_CALL_WRAPPED_PATTERN.search(text)
    if wrapped is not None:
        candidate = wrapped.group(1)
    else:
        stripped = text.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return None
        candidate = stripped

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        if wrapped is not None:
            logger.warning("Malformed <tool_call> block from the local LLM: %s", candidate)
        return None
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        logger.warning("Malformed tool call payload: %s", payload)
        return None
    return ParsedToolCall(name=name, arguments=arguments)


def _decode_data_url_image(image_url: object) -> bytes | None:
    """Decode a base64 `data:` URL image, matching what huggingface_realtime.py sends for the camera tool."""
    if not isinstance(image_url, str) or not image_url.startswith("data:"):
        return None
    try:
        _, encoded = image_url.split(",", 1)
        return base64.b64decode(encoded)
    except (ValueError, TypeError) as exc:
        logger.warning("Failed to decode input_image data URL: %s", exc)
        return None


class RealtimeTransport(Protocol):
    """The subset of FastAPI's WebSocket this session depends on."""

    async def receive_json(self) -> dict[str, Any]:
        """Receive and JSON-decode the next client message."""
        ...

    async def send_json(self, data: dict[str, Any]) -> None:
        """JSON-encode and send a server event to the client."""
        ...


class SpeechToText(Protocol):
    """A batch speech-to-text pipeline."""

    def transcribe(self, pcm16_audio: bytes, *, sample_rate: int) -> str:
        """Transcribe one complete utterance of 16-bit PCM audio."""
        ...


class ConversationalLLM(Protocol):
    """A chat completion pipeline, optionally vision-capable (see `ChatMessage.image`)."""

    def generate(self, messages: list[ChatMessage]) -> str:
        """Generate a reply for a conversation history of messages."""
        ...


class TextToSpeech(Protocol):
    """A text-to-speech pipeline."""

    def synthesize(self, text: str, *, output_sample_rate: int) -> bytes:
        """Return 16-bit PCM mono audio at ``output_sample_rate`` for ``text``."""
        ...


class VadDetector(Protocol):
    """A streaming voice-activity detector (energy-threshold or Silero-backed)."""

    def push(self, pcm16_chunk: bytes) -> str | None:
        """Feed one chunk of 16-bit PCM audio; return "speech_started"/"speech_stopped", if any."""
        ...


class RealtimeSession:
    """One WebSocket connection's VAD+STT+LLM(+tools/vision)(+TTS) orchestration."""

    SAMPLE_RATE = 16000
    _AUDIO_CHUNK_BYTES = 3200  # 100 ms of 16-bit mono PCM at SAMPLE_RATE

    def __init__(
        self,
        transport: RealtimeTransport,
        *,
        stt: SpeechToText,
        llm: ConversationalLLM,
        tts: TextToSpeech | None = None,
        vad: VadDetector | None = None,
    ) -> None:
        """Initialize the session with injected pipelines (swappable for tests).

        ``tts`` is optional: without it, replies still play through as text
        (`response.output_text.*`/`response.output_audio_transcript.done`),
        just without synthesized audio — graceful degradation rather than a
        hard requirement to have a working TTS pipeline configured. Vision is
        likewise graceful: `ChatMessage.image` is only understood if ``llm``
        is a vision-capable pipeline; a text-only one just won't see it.
        ``vad`` defaults to the energy-threshold detector when not given (see
        `local_openvino/silero_vad.py` for the higher-quality alternative).
        """
        self._transport = transport
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._vad = vad if vad is not None else VoiceActivityDetector(sample_rate=self.SAMPLE_RATE)
        self._audio_buffer = bytearray()
        self._history: list[ChatMessage] = []
        self._tools: list[dict[str, Any]] = []
        self._response_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        """Consume client messages until the connection closes."""
        try:
            while True:
                try:
                    message = await self._transport.receive_json()
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    # An empty/non-JSON frame isn't a disconnect (that raises
                    # WebSocketDisconnect above); skip it and keep the session
                    # alive instead of letting it kill the whole connection.
                    logger.debug("Ignoring malformed (non-JSON) client message")
                    continue
                await self._handle_message(message)
        finally:
            if self._response_task is not None:
                self._response_task.cancel()

    async def _handle_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "session.update":
            self._handle_session_update(message.get("session") or {})
        elif message_type == "input_audio_buffer.append":
            await self._handle_audio_append(message.get("audio") or "")
        elif message_type == "conversation.item.create":
            self._handle_item_create(message.get("item") or {})
        elif message_type == "response.create":
            await self._trigger_response()
        else:
            logger.debug("Ignoring unsupported client message type: %s", message_type)

    def _handle_session_update(self, session: dict[str, Any]) -> None:
        tools = session.get("tools")
        if isinstance(tools, list):
            self._tools = tools
        # Voice selection lands in a later phase; instructions are accepted
        # but unused until the LLM prompt wiring needs them.
        logger.debug("session.update received: keys=%s", sorted(session))

    async def _handle_audio_append(self, audio_b64: str) -> None:
        if not audio_b64:
            return
        try:
            chunk = base64.b64decode(audio_b64)
        except (ValueError, TypeError) as exc:
            logger.warning("Dropping malformed input_audio_buffer.append payload: %s", exc)
            return

        self._audio_buffer.extend(chunk)
        try:
            transition = self._vad.push(chunk)
        except Exception:
            logger.exception("VAD failure; treating this chunk as silence")
            return
        if transition == "speech_started":
            await self._send({"type": "input_audio_buffer.speech_started"})
        elif transition == "speech_stopped":
            await self._send({"type": "input_audio_buffer.speech_stopped"})
            await self._commit_and_transcribe()

    async def _commit_and_transcribe(self) -> None:
        audio = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        if not audio:
            await self._send(
                {
                    "type": "error",
                    "error": {"code": "input_audio_buffer_commit_empty", "message": "No audio to commit."},
                }
            )
            return

        item_id = str(uuid.uuid4())
        try:
            transcript = await asyncio.to_thread(self._stt.transcribe, audio, sample_rate=self.SAMPLE_RATE)
        except Exception as exc:
            logger.exception("Local STT transcription failed")
            await self._send({"type": "error", "error": {"code": "stt_failed", "message": str(exc)}})
            return

        transcript = transcript.strip()
        await self._send(
            {"type": "conversation.item.input_audio_transcription.delta", "item_id": item_id, "delta": transcript}
        )
        await self._send(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "transcript": transcript,
            }
        )
        if not transcript:
            return
        self._history.append(ChatMessage(role="user", content=transcript))
        await self._trigger_response()

    def _handle_item_create(self, item: dict[str, Any]) -> None:
        item_type = item.get("type")
        if item_type == "function_call_output":
            self._handle_function_call_output(item)
            return
        if item_type != "message" or item.get("role") != "user":
            logger.debug("Ignoring unsupported conversation item: type=%s", item_type)
            return
        for part in item.get("content") or []:
            part_type = part.get("type")
            if part_type == "input_text":
                text = str(part.get("text") or "").strip()
                if text:
                    self._history.append(ChatMessage(role="user", content=text))
            elif part_type == "input_image":
                image = _decode_data_url_image(part.get("image_url"))
                if image is not None:
                    self._history.append(ChatMessage(role="user", content="", image=image))

    def _handle_function_call_output(self, item: dict[str, Any]) -> None:
        call_id = item.get("call_id")
        output = item.get("output")
        if not isinstance(call_id, str) or not isinstance(output, str):
            logger.warning("Ignoring malformed function_call_output item: %r", item)
            return
        self._history.append(ChatMessage(role="tool", content=output))

    async def _trigger_response(self) -> None:
        if self._response_task is not None and not self._response_task.done():
            await self._send(
                {
                    "type": "error",
                    "error": {
                        "code": "conversation_already_has_active_response",
                        "message": "A response is already in progress.",
                    },
                }
            )
            return
        self._response_task = asyncio.create_task(self._generate_and_emit_response())

    async def _generate_and_emit_response(self) -> None:
        await self._send({"type": "response.created"})
        messages = list(self._history)
        if self._tools:
            messages = [ChatMessage(role="system", content=_build_tool_instructions(self._tools)), *messages]
        try:
            reply = await asyncio.to_thread(self._llm.generate, messages)
        except Exception as exc:
            logger.exception("Local LLM generation failed")
            await self._send({"type": "error", "error": {"code": "llm_generation_failed", "message": str(exc)}})
            await self._send({"type": "response.done"})
            return

        tool_call = _parse_tool_call(reply)
        if tool_call is not None:
            self._history.append(ChatMessage(role="assistant", content=reply))
            await self._send(
                {
                    "type": "response.function_call_arguments.done",
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                    "call_id": str(uuid.uuid4()),
                }
            )
            await self._send({"type": "response.done"})
            return

        reply = reply.strip()
        if reply:
            self._history.append(ChatMessage(role="assistant", content=reply))
            await self._send({"type": "response.output_text.delta", "delta": reply})
            await self._send({"type": "response.output_text.done", "text": reply})
            if self._tts is not None:
                await self._synthesize_and_emit_audio(self._tts, reply)
            await self._send({"type": "response.output_audio_transcript.done", "transcript": reply})
        await self._send({"type": "response.done"})

    async def _synthesize_and_emit_audio(self, tts: TextToSpeech, text: str) -> None:
        try:
            pcm16_audio = await asyncio.to_thread(tts.synthesize, text, output_sample_rate=self.SAMPLE_RATE)
        except Exception as exc:
            logger.exception("Local TTS synthesis failed")
            await self._send({"type": "error", "error": {"code": "tts_failed", "message": str(exc)}})
            return

        for offset in range(0, len(pcm16_audio), self._AUDIO_CHUNK_BYTES):
            chunk = pcm16_audio[offset : offset + self._AUDIO_CHUNK_BYTES]
            await self._send({"type": "response.output_audio.delta", "delta": base64.b64encode(chunk).decode()})
        await self._send({"type": "response.output_audio.done"})

    async def _send(self, event: dict[str, Any]) -> None:
        await self._transport.send_json(event)
