"""OpenVINO-optimized model catalog and download management.

Unlike the rest of `local_openvino/`, this module only depends on
`huggingface_hub` (a core dependency of the whole app, installed regardless of
the `local-openvino` extra), so it is safe to import directly from the main
app process — no subprocess isolation needed here.
"""

import logging
from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

from huggingface_hub import HfApi, snapshot_download


logger = logging.getLogger(__name__)

Role = Literal["stt", "llm", "vlm", "tts"]

_OPENVINO_ORG = "OpenVINO"

# "tts" is deliberately absent: its catalog query searches by name instead
# (see refresh_catalog_from_hub), since it isn't drawn from the OpenVINO org.
_PIPELINE_TAGS: dict[Role, str] = {
    "stt": "automatic-speech-recognition",
    "llm": "text-generation",
    "vlm": "image-text-to-text",
}


@dataclass(frozen=True)
class ModelEntry:
    """One OpenVINO-optimized model catalog entry."""

    repo_id: str
    role: Role
    display_name: str


# Small curated fallback so the UI has something to show even offline or
# before a Hub query completes. Confirmed live (repo + file listing checked
# during planning) rather than guessed; the rest of the catalog is resolved
# live via refresh_catalog_from_hub() so it stays current without code changes.
_DEFAULT_CATALOG: dict[Role, list[ModelEntry]] = {
    "stt": [
        ModelEntry(repo_id="OpenVINO/whisper-large-v3-fp16-ov", role="stt", display_name="Whisper large-v3 (fp16)"),
    ],
    "llm": [
        ModelEntry(repo_id="OpenVINO/Qwen3-8B-int4-ov", role="llm", display_name="Qwen3 8B (int4)"),
    ],
    "vlm": [],
    # fastrtc/kokoro-onnx ships kokoro-v1.0.onnx + voices-v1.0.bin, the single
    # combined-file layout KokoroTTS.from_snapshot()/kokoro_onnx.Kokoro expect.
    # Other Kokoro ONNX mirrors (e.g. onnx-community's) split voices into one
    # file per voice instead and are NOT compatible with this pipeline.
    "tts": [
        ModelEntry(repo_id="fastrtc/kokoro-onnx", role="tts", display_name="Kokoro-82M (ONNX)"),
    ],
}


def default_catalog() -> dict[Role, list[ModelEntry]]:
    """Return a small curated catalog usable even without network access."""
    return {role: list(entries) for role, entries in _DEFAULT_CATALOG.items()}


def refresh_catalog_from_hub(role: Role) -> list[ModelEntry]:
    """Query the Hugging Face Hub for current models usable for ``role``."""
    try:
        if role == "tts":
            # KokoroTTS only understands the fastrtc/kokoro-onnx-style combined
            # voices file, not the general OpenVINO org catalog used by the
            # other roles, so search by name instead of pipeline_tag/author.
            models = HfApi().list_models(search="kokoro-onnx")
        else:
            models = HfApi().list_models(author=_OPENVINO_ORG, pipeline_tag=_PIPELINE_TAGS[role])
        entries = [
            ModelEntry(repo_id=model.id, role=role, display_name=model.id.split("/", 1)[-1]) for model in models
        ]
    except Exception as exc:
        logger.warning("Failed to query the Hugging Face Hub for %s models: %s", role, exc)
        return default_catalog()[role]
    return entries or default_catalog()[role]


def _models_cache_root() -> Path:
    """Return the on-disk root for downloaded OpenVINO model snapshots."""
    return Path.home() / ".cache" / "reachy_mini_conversation_app" / "openvino_models"


# Written only after a snapshot_download() completes without error, so a
# download interrupted partway through is correctly retried instead of being
# mistaken for a complete, usable snapshot.
_DOWNLOAD_COMPLETE_MARKER = ".download_complete"


def ensure_model_downloaded(repo_id: str, *, on_progress: Callable[[float], None] | None = None) -> Path:
    """Return ``repo_id``'s local snapshot path, downloading it only if not already cached."""
    destination = _models_cache_root() / repo_id
    if (destination / _DOWNLOAD_COMPLETE_MARKER).exists():
        logger.info("OpenVINO model %s already downloaded, skipping: %s", repo_id, destination)
        if on_progress is not None:
            on_progress(1.0)
        return destination

    if on_progress is not None:
        on_progress(0.0)
    # huggingface_hub's snapshot_download() doesn't expose a simple per-file
    # progress callback across versions, so progress is coarse (start/finish)
    # rather than tracking bytes transferred.
    downloaded_path = Path(snapshot_download(repo_id=repo_id, local_dir=destination))
    (downloaded_path / _DOWNLOAD_COMPLETE_MARKER).touch()
    if on_progress is not None:
        on_progress(1.0)
    return downloaded_path
