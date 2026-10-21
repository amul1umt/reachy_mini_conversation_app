"""Protocol-compliance tests for the local OpenVINO realtime session orchestrator.

These are the highest-value tests in the local-OpenVINO backend: they prove the
server emits exactly the event sequence
`reachy_mini_conversation_app.huggingface_realtime.HuggingFaceRealtimeHandler`
expects, without needing real OpenVINO models or Core Ultra hardware — STT and
LLM are injected fakes (see `RealtimeSession`'s `SpeechToText`/`ConversationalLLM`
protocols).
"""

import json
import base64
import asyncio
import contextlib

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from reachy_mini_conversation_app.local_openvino.session import ChatMessage, RealtimeSession


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100 ms frames at 16 kHz


def _audio_append(amplitude: int) -> dict[str, object]:
    """Build an input_audio_buffer.append message for a constant-amplitude chunk."""
    samples = np.full(CHUNK_SAMPLES, amplitude, dtype=np.int16)
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(samples.tobytes()).decode()}


def _user_text_item(text: str) -> dict[str, object]:
    """Build a conversation.item.create message for a synthetic user text turn."""
    return {
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
    }


class FakeTransport:
    """A queued in-memory stand-in for FastAPI's WebSocket, for session tests.

    Behaves like an open connection with no more input once the queue is
    drained (mirroring a real client that's still connected), so response
    generation triggered by the last queued message isn't cancelled by a
    premature "disconnect".
    """

    def __init__(self, incoming: list[dict[str, object]]) -> None:
        """Queue ``incoming`` messages to be returned in order by receive_json()."""
        self._incoming = list(incoming)
        self.sent: list[dict[str, object]] = []

    async def receive_json(self) -> dict[str, object]:
        """Return the next queued message, or hang forever once the queue is empty."""
        if self._incoming:
            return self._incoming.pop(0)
        await asyncio.Future()
        raise AssertionError("unreachable")  # pragma: no cover

    async def send_json(self, data: dict[str, object]) -> None:
        """Record a server event instead of sending it over a real socket."""
        self.sent.append(data)


class _StubSTT:
    """A speech-to-text stub returning a fixed transcript."""

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.received_audio: bytes | None = None

    def transcribe(self, pcm16_audio: bytes, *, sample_rate: int) -> str:
        """Record the audio it was given and return the fixed transcript."""
        assert sample_rate == SAMPLE_RATE
        self.received_audio = pcm16_audio
        return self.transcript


class _FailingSTT:
    """A speech-to-text stub that always raises."""

    def transcribe(self, pcm16_audio: bytes, *, sample_rate: int) -> str:
        """Simulate an inference failure."""
        raise RuntimeError("model not loaded")


class _StubLLM:
    """A chat-completion stub returning a fixed reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.received_messages: list[list[ChatMessage]] = []

    def generate(self, messages: list[ChatMessage]) -> str:
        """Record the conversation history it was given and return the fixed reply."""
        self.received_messages.append(list(messages))
        return self.reply


class _FailingLLM:
    """A chat-completion stub that always raises."""

    def generate(self, messages: list[ChatMessage]) -> str:
        """Simulate an inference failure."""
        raise RuntimeError("out of memory")


class _BlockingLLM:
    """A chat-completion stub that blocks until the test releases it."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.release = asyncio.Event()
        self.calls = 0

    def generate(self, messages: list[ChatMessage]) -> str:
        """Block (from a worker thread) until the test calls `release.set()`."""
        self.calls += 1
        asyncio.run_coroutine_threadsafe(self.release.wait(), self._loop).result(timeout=5)
        return "reply"


class _StubTTS:
    """A text-to-speech stub returning fixed-length silence."""

    def __init__(self, sample_count: int = 4000) -> None:
        self.sample_count = sample_count
        self.received_text: str | None = None

    def synthesize(self, text: str, *, output_sample_rate: int) -> bytes:
        """Record the text it was given and return fixed-length 16-bit PCM silence."""
        self.received_text = text
        return bytes(self.sample_count * 2)


class _FailingTTS:
    """A text-to-speech stub that always raises."""

    def synthesize(self, text: str, *, output_sample_rate: int) -> bytes:
        """Simulate an inference failure."""
        raise RuntimeError("model not loaded")


async def _run_until_settled(session: RealtimeSession, transport: FakeTransport) -> None:
    """Run the session until a response.done or error event lands, then cancel it."""
    run_task = asyncio.create_task(session.run())
    try:
        for _ in range(500):
            if any(event["type"] in ("response.done", "error") for event in transport.sent):
                return
            await asyncio.sleep(0.01)
        raise AssertionError("timed out waiting for the session to settle")
    finally:
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task


@pytest.mark.asyncio
async def test_voice_turn_emits_the_expected_event_sequence() -> None:
    """A spoken turn should produce the exact server->client event sequence the real client expects."""
    messages = [_audio_append(3000)] * 3 + [_audio_append(0)] * 6
    transport = FakeTransport(messages)
    stt = _StubSTT("hello there")
    llm = _StubLLM("hi, how can I help?")
    session = RealtimeSession(transport, stt=stt, llm=llm)

    await _run_until_settled(session, transport)

    event_types = [event["type"] for event in transport.sent]
    assert event_types == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
        "response.created",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_audio_transcript.done",
        "response.done",
    ]
    completed = next(e for e in transport.sent if e["type"] == "conversation.item.input_audio_transcription.completed")
    assert completed["transcript"] == "hello there"
    transcript_event = next(e for e in transport.sent if e["type"] == "response.output_audio_transcript.done")
    assert transcript_event["transcript"] == "hi, how can I help?"
    assert llm.received_messages == [[ChatMessage(role="user", content="hello there")]]
    assert stt.received_audio is not None and len(stt.received_audio) > 0


@pytest.mark.asyncio
async def test_text_only_turn_via_response_create_matches_the_greeting_flow() -> None:
    """A conversation.item.create + response.create pair (say()/greeting) should reply without any audio input."""
    transport = FakeTransport([_user_text_item("Greet me warmly."), {"type": "response.create"}])
    llm = _StubLLM("Hello! Great to see you.")
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=llm)

    await _run_until_settled(session, transport)

    event_types = [event["type"] for event in transport.sent]
    assert event_types == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_audio_transcript.done",
        "response.done",
    ]
    assert llm.received_messages == [[ChatMessage(role="user", content="Greet me warmly.")]]


@pytest.mark.asyncio
async def test_text_turn_with_tts_streams_audio_chunks_before_the_transcript() -> None:
    """With a TTS pipeline configured, the reply should stream output_audio.delta chunks."""
    transport = FakeTransport([_user_text_item("hi"), {"type": "response.create"}])
    tts = _StubTTS(sample_count=4000)  # 2 full 3200-byte chunks + a partial third
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=_StubLLM("hello!"), tts=tts)

    await _run_until_settled(session, transport)

    event_types = [event["type"] for event in transport.sent]
    assert event_types == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_audio.delta",
        "response.output_audio.delta",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "response.done",
    ]
    assert tts.received_text == "hello!"
    audio_bytes = sum(
        len(base64.b64decode(event["delta"]))
        for event in transport.sent
        if event["type"] == "response.output_audio.delta"
    )
    assert audio_bytes == tts.sample_count * 2


@pytest.mark.asyncio
async def test_tts_failure_emits_an_error_but_still_completes_the_response() -> None:
    """A failing TTS pipeline should surface as a tts_failed error, then still send response.done."""
    transport = FakeTransport([_user_text_item("hi"), {"type": "response.create"}])
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=_StubLLM("hello!"), tts=_FailingTTS())

    await _run_until_settled(session, transport)

    event_types = [event["type"] for event in transport.sent]
    assert event_types == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.done",
        "error",
        "response.output_audio_transcript.done",
        "response.done",
    ]
    error_event = next(e for e in transport.sent if e["type"] == "error")
    assert error_event["error"]["code"] == "tts_failed"


@pytest.mark.asyncio
async def test_tool_call_emits_function_call_arguments_done_instead_of_text() -> None:
    """A <tool_call> block in the LLM's reply should become a function_call_arguments.done event, not text."""
    transport = FakeTransport(
        [
            {
                "type": "session.update",
                "session": {
                    "tools": [
                        {
                            "type": "function",
                            "name": "move_head",
                            "description": "Move the head.",
                            "parameters": {"type": "object", "properties": {"direction": {"type": "string"}}},
                        }
                    ]
                },
            },
            _user_text_item("look left"),
            {"type": "response.create"},
        ]
    )
    reply = '<tool_call>\n{"name": "move_head", "arguments": {"direction": "left"}}\n</tool_call>'
    llm = _StubLLM(reply)
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=llm)

    await _run_until_settled(session, transport)

    event_types = [event["type"] for event in transport.sent]
    assert event_types == ["response.created", "response.function_call_arguments.done", "response.done"]
    call_event = transport.sent[1]
    assert call_event["name"] == "move_head"
    assert json.loads(call_event["arguments"]) == {"direction": "left"}
    assert isinstance(call_event["call_id"], str) and call_event["call_id"]

    # The tools list from session.update should be rendered into a leading system message.
    sent_messages = llm.received_messages[0]
    assert sent_messages[0].role == "system"
    assert "move_head" in sent_messages[0].content
    assert sent_messages[-1] == ChatMessage(role="user", content="look left")


@pytest.mark.asyncio
async def test_malformed_tool_call_block_falls_back_to_plain_text_reply() -> None:
    """Invalid JSON inside <tool_call> tags should be treated as a normal text reply, not crash."""
    transport = FakeTransport([_user_text_item("hi"), {"type": "response.create"}])
    reply = "<tool_call>\nnot json\n</tool_call>"
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=_StubLLM(reply))

    await _run_until_settled(session, transport)

    event_types = [event["type"] for event in transport.sent]
    assert event_types == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_audio_transcript.done",
        "response.done",
    ]


@pytest.mark.asyncio
async def test_bare_json_tool_call_without_wrapper_tags_is_still_recognized() -> None:
    """A whole-reply bare JSON tool call (no <tool_call> wrapper) should still be recognized.

    Real small models don't always reproduce the instructed wrapper tags even
    when they get the JSON shape right (observed live against
    OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov during planning).
    """
    transport = FakeTransport([_user_text_item("look left"), {"type": "response.create"}])
    reply = '{"name": "move_head", "arguments": {"direction": "left"}}'
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=_StubLLM(reply))

    await _run_until_settled(session, transport)

    event_types = [event["type"] for event in transport.sent]
    assert event_types == ["response.created", "response.function_call_arguments.done", "response.done"]
    call_event = transport.sent[1]
    assert call_event["name"] == "move_head"
    assert json.loads(call_event["arguments"]) == {"direction": "left"}


@pytest.mark.asyncio
async def test_plain_text_reply_containing_braces_is_not_mistaken_for_a_tool_call() -> None:
    """Ordinary text that happens to mention JSON/braces should not be parsed as a tool call."""
    transport = FakeTransport([_user_text_item("what does a json object look like"), {"type": "response.create"}])
    reply = 'It looks like this: {"example": "value"} inside a sentence.'
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=_StubLLM(reply))

    await _run_until_settled(session, transport)

    event_types = [event["type"] for event in transport.sent]
    assert event_types == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_audio_transcript.done",
        "response.done",
    ]


@pytest.mark.asyncio
async def test_function_call_output_item_is_folded_into_history_as_tool_role() -> None:
    """A function_call_output item should appear in the next LLM call as a tool-role message."""
    transport = FakeTransport(
        [
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call-1", "output": '{"ok": true}'},
            },
            {"type": "response.create"},
        ]
    )
    llm = _StubLLM("Done!")
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=llm)

    await _run_until_settled(session, transport)

    assert llm.received_messages == [[ChatMessage(role="tool", content='{"ok": true}')]]


@pytest.mark.asyncio
async def test_input_image_item_is_attached_to_history_with_image_bytes() -> None:
    """A camera-tool input_image item should attach its decoded JPEG bytes to the history for a VLM to see."""
    jpeg_bytes = b"\xff\xd8\xff\xe0fakejpeg"
    data_url = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode()}"
    transport = FakeTransport(
        [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": data_url}],
                },
            },
            {"type": "response.create"},
        ]
    )
    llm = _StubLLM("I see a robot.")
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=llm)

    await _run_until_settled(session, transport)

    assert llm.received_messages == [[ChatMessage(role="user", content="", image=jpeg_bytes)]]


@pytest.mark.asyncio
async def test_overlapping_response_create_is_rejected_while_one_is_active() -> None:
    """A second response.create while one is in flight should get the real protocol's rejection error."""
    transport = FakeTransport([_user_text_item("hi"), {"type": "response.create"}, {"type": "response.create"}])
    llm = _BlockingLLM(asyncio.get_event_loop())
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=llm)

    run_task = asyncio.create_task(session.run())
    try:
        for _ in range(500):
            if any(event["type"] == "error" for event in transport.sent):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("timed out waiting for the rejection error")

        error_event = next(e for e in transport.sent if e["type"] == "error")
        assert error_event["error"]["code"] == "conversation_already_has_active_response"
        assert llm.calls == 1
    finally:
        llm.release.set()
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task


@pytest.mark.asyncio
async def test_stt_failure_emits_an_error_without_crashing_the_session() -> None:
    """A failing STT pipeline should surface as a stt_failed error event, not an exception."""
    messages = [_audio_append(3000)] * 3 + [_audio_append(0)] * 6
    transport = FakeTransport(messages)
    session = RealtimeSession(transport, stt=_FailingSTT(), llm=_StubLLM("unused"))

    await _run_until_settled(session, transport)

    error_event = next(e for e in transport.sent if e["type"] == "error")
    assert error_event["error"]["code"] == "stt_failed"


@pytest.mark.asyncio
async def test_llm_failure_emits_an_error_and_still_closes_the_response() -> None:
    """A failing LLM pipeline should surface as an llm_generation_failed error, followed by response.done."""
    transport = FakeTransport([_user_text_item("hi"), {"type": "response.create"}])
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=_FailingLLM())

    await _run_until_settled(session, transport)

    event_types = [event["type"] for event in transport.sent]
    assert event_types == ["response.created", "error", "response.done"]
    assert transport.sent[1]["error"]["code"] == "llm_generation_failed"


@pytest.mark.asyncio
async def test_malformed_audio_payload_is_dropped_without_crashing() -> None:
    """A non-base64 audio payload should be logged and dropped, not crash the session."""
    transport = FakeTransport([{"type": "input_audio_buffer.append", "audio": "not-valid-base64!!"}])
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=_StubLLM(""))

    run_task = asyncio.create_task(session.run())
    await asyncio.sleep(0.05)
    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run_task

    assert transport.sent == []


@pytest.mark.asyncio
async def test_disconnect_ends_the_session_cleanly() -> None:
    """A WebSocketDisconnect from the transport should end run() without raising."""

    class _DisconnectingTransport:
        async def receive_json(self) -> dict[str, object]:
            raise WebSocketDisconnect()

        async def send_json(self, data: dict[str, object]) -> None:
            raise AssertionError("should not send anything")  # pragma: no cover

    session = RealtimeSession(_DisconnectingTransport(), stt=_StubSTT(""), llm=_StubLLM(""))

    await asyncio.wait_for(session.run(), timeout=5)


@pytest.mark.asyncio
async def test_malformed_frame_is_skipped_not_fatal() -> None:
    """A non-JSON frame (e.g. an empty text frame) must not kill the whole session.

    Regression test: receive_json() can raise json.JSONDecodeError for a frame
    that isn't a real disconnect (that raises WebSocketDisconnect instead, see
    test_disconnect_ends_the_session_cleanly above); run() previously let that
    propagate uncaught, silently dropping the connection.
    """

    class _FlakyTransport:
        def __init__(self, subsequent: list[dict[str, object]]) -> None:
            self._raised = False
            self._subsequent = list(subsequent)
            self.sent: list[dict[str, object]] = []

        async def receive_json(self) -> dict[str, object]:
            if not self._raised:
                self._raised = True
                raise json.JSONDecodeError("Expecting value", "", 0)
            if self._subsequent:
                return self._subsequent.pop(0)
            await asyncio.Future()
            raise AssertionError("unreachable")  # pragma: no cover

        async def send_json(self, data: dict[str, object]) -> None:
            self.sent.append(data)

    transport = _FlakyTransport([_user_text_item("hello"), {"type": "response.create"}])
    session = RealtimeSession(transport, stt=_StubSTT(""), llm=_StubLLM("hi there"))

    run_task = asyncio.create_task(session.run())
    await asyncio.sleep(0.05)
    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run_task

    assert {"type": "response.output_text.done", "text": "hi there"} in transport.sent
