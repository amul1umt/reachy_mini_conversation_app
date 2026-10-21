"""Helpers for persisting the user's selected local-OpenVINO models and devices."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from dataclasses import dataclass


logger = logging.getLogger(__name__)

LOCAL_OPENVINO_SETTINGS_FILENAME = "local_openvino_settings.json"

DEFAULT_DEVICE = "AUTO"


@dataclass(frozen=True)
class LocalOpenvinoSettings:
    """Instance-local selection of models and target devices per pipeline role."""

    stt_model: str | None = None
    stt_device: str = DEFAULT_DEVICE
    llm_model: str | None = None
    llm_device: str = DEFAULT_DEVICE
    # Whether llm_model was picked from the catalog's "llm" or "vlm" role -
    # determines whether the spawned server gets --vision (see cli.py).
    llm_role: str = "llm"
    tts_model: str | None = None
    tts_device: str = DEFAULT_DEVICE


def _normalize_text(value: object, default: str) -> str:
    """Return a stripped string, or ``default`` for empty/non-string values."""
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()


def _normalize_optional_text(value: object) -> str | None:
    """Return a stripped string or None for empty/non-string values."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _settings_path(instance_path: str | Path | None) -> Path | None:
    """Return the local-OpenVINO settings JSON path for an instance directory."""
    if instance_path is None:
        return None
    return Path(instance_path) / LOCAL_OPENVINO_SETTINGS_FILENAME


def read_local_openvino_settings(instance_path: str | Path | None) -> LocalOpenvinoSettings:
    """Read the persisted local-OpenVINO model/device selection."""
    settings_path = _settings_path(instance_path)
    if settings_path is None or not settings_path.exists():
        return LocalOpenvinoSettings()

    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read local-OpenVINO settings from %s: %s", settings_path, exc)
        return LocalOpenvinoSettings()

    if not isinstance(payload, dict):
        logger.warning("Ignoring invalid local-OpenVINO settings payload from %s: %r", settings_path, payload)
        return LocalOpenvinoSettings()

    llm_role = payload.get("llm_role")
    return LocalOpenvinoSettings(
        stt_model=_normalize_optional_text(payload.get("stt_model")),
        stt_device=_normalize_text(payload.get("stt_device"), DEFAULT_DEVICE),
        llm_model=_normalize_optional_text(payload.get("llm_model")),
        llm_device=_normalize_text(payload.get("llm_device"), DEFAULT_DEVICE),
        llm_role=llm_role if llm_role in ("llm", "vlm") else "llm",
        tts_model=_normalize_optional_text(payload.get("tts_model")),
        tts_device=_normalize_text(payload.get("tts_device"), DEFAULT_DEVICE),
    )


def write_local_openvino_settings(instance_path: str | Path | None, settings: LocalOpenvinoSettings) -> None:
    """Persist the local-OpenVINO model/device selection."""
    settings_path = _settings_path(instance_path)
    if settings_path is None:
        return

    payload = {
        "stt_model": settings.stt_model,
        "stt_device": settings.stt_device,
        "llm_model": settings.llm_model,
        "llm_device": settings.llm_device,
        "llm_role": settings.llm_role,
        "tts_model": settings.tts_model,
        "tts_device": settings.tts_device,
    }
    settings_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
