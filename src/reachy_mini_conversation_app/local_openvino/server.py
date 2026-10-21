"""FastAPI application exposing the local OpenVINO realtime WebSocket endpoint."""

import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from reachy_mini_conversation_app.local_openvino.session import (
    SpeechToText,
    TextToSpeech,
    RealtimeSession,
    ConversationalLLM,
)
from reachy_mini_conversation_app.local_openvino.silero_vad import load_vad


logger = logging.getLogger(__name__)


def create_app(
    *,
    stt: SpeechToText,
    llm: ConversationalLLM,
    tts: TextToSpeech | None = None,
    vad_model_path: Path | None = None,
) -> FastAPI:
    """Build the local OpenVINO realtime server, wiring the given pipelines into each session.

    ``vad`` is stateful per conversation turn, so unlike the other pipelines
    (loaded once and shared) a fresh one is built for each connection.
    """
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def _realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        vad = load_vad(vad_model_path, sample_rate=RealtimeSession.SAMPLE_RATE)
        try:
            await RealtimeSession(websocket, stt=stt, llm=llm, tts=tts, vad=vad).run()
        except WebSocketDisconnect:
            pass

    return app
