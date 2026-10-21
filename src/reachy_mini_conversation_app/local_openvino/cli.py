"""CLI entry point for the local OpenVINO realtime server (`local-openvino-server` subcommand)."""

import logging
import argparse
from typing import TypeVar
from pathlib import Path
from collections.abc import Callable

import uvicorn

from reachy_mini_conversation_app.local_openvino.models import ensure_model_downloaded
from reachy_mini_conversation_app.local_openvino.server import create_app
from reachy_mini_conversation_app.local_openvino.session import ConversationalLLM
from reachy_mini_conversation_app.local_openvino.silero_vad import SILERO_VAD_REPO_ID
from reachy_mini_conversation_app.local_openvino.pipelines.llm import LocalLLM
from reachy_mini_conversation_app.local_openvino.pipelines.stt import WhisperSTT
from reachy_mini_conversation_app.local_openvino.pipelines.tts import KokoroTTS
from reachy_mini_conversation_app.local_openvino.pipelines.vlm import LocalVLM


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _build_with_device_fallback(build: Callable[[str], _T], device: str, *, role: str) -> _T:
    """Try ``device``, falling back to AUTO if construction fails (e.g. an unavailable NPU)."""
    try:
        return build(device)
    except Exception as exc:
        if device == "AUTO":
            raise
        logger.warning(
            "Failed to initialize the %s pipeline on device %r (%s); falling back to AUTO.", role, device, exc
        )
        return build("AUTO")


def _load_tts(args: argparse.Namespace) -> KokoroTTS | None:
    """Download and load the configured TTS model, if any; None means text-only replies."""
    if not args.tts_model:
        return None
    if args.tts_device not in ("AUTO", "CPU"):
        logger.warning(
            "Kokoro TTS only runs on CPU (its ONNX graph is incompatible with the OpenVINO execution "
            "provider); ignoring --tts-device=%s.",
            args.tts_device,
        )
    tts_snapshot = ensure_model_downloaded(args.tts_model)
    try:
        return KokoroTTS.from_snapshot(tts_snapshot)
    except FileNotFoundError as exc:
        logger.warning("Local TTS unavailable, falling back to text-only responses: %s", exc)
        return None


def _resolve_vad_model_path() -> Path | None:
    """Download the Silero VAD model, if available; None triggers the energy-threshold fallback."""
    try:
        snapshot = ensure_model_downloaded(SILERO_VAD_REPO_ID)
    except Exception as exc:
        logger.warning("Could not download Silero VAD (%s); falling back to energy-threshold detection.", exc)
        return None
    model_path = next(snapshot.rglob("silero_vad.onnx"), None) or next(snapshot.rglob("*.onnx"), None)
    if model_path is None:
        logger.warning(
            "Silero VAD download at %s has no .onnx file; falling back to energy-threshold detection.", snapshot
        )
    return model_path


def run_local_openvino_server(args: argparse.Namespace) -> None:
    """Download (if needed) the configured models and serve the realtime WebSocket."""
    logger.info("Preparing local OpenVINO models (this can take a while on first run)...")
    stt_path = ensure_model_downloaded(args.stt_model)
    stt = _build_with_device_fallback(lambda device: WhisperSTT(stt_path, device), args.stt_device, role="STT")

    llm_path = ensure_model_downloaded(args.llm_model)
    llm_factory: Callable[[str], ConversationalLLM] = (
        (lambda device: LocalVLM(llm_path, device)) if args.vision else (lambda device: LocalLLM(llm_path, device))
    )
    llm = _build_with_device_fallback(llm_factory, args.llm_device, role="LLM")

    tts = _load_tts(args)
    vad_model_path = _resolve_vad_model_path()

    app = create_app(stt=stt, llm=llm, tts=tts, vad_model_path=vad_model_path)
    logger.info("Local OpenVINO realtime server listening on ws://127.0.0.1:%d/v1/realtime", args.port)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
