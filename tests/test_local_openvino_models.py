"""Tests for the local OpenVINO model catalog and download manager."""

from pathlib import Path

import pytest

from reachy_mini_conversation_app.local_openvino.models import (
    ModelEntry,
    default_catalog,
    ensure_model_downloaded,
    refresh_catalog_from_hub,
)


def test_default_catalog_has_seed_entries_per_role() -> None:
    """The curated fallback catalog should cover every declared role key."""
    catalog = default_catalog()

    assert set(catalog) == {"stt", "llm", "vlm", "tts"}
    assert catalog["stt"] == [
        ModelEntry(repo_id="OpenVINO/whisper-large-v3-fp16-ov", role="stt", display_name="Whisper large-v3 (fp16)")
    ]


def test_refresh_catalog_from_hub_returns_live_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful Hub query should be mapped into ModelEntry objects."""

    class _FakeModel:
        def __init__(self, model_id: str) -> None:
            self.id = model_id

    class _FakeHfApi:
        def list_models(self, *, author: str, pipeline_tag: str) -> list[_FakeModel]:
            assert author == "OpenVINO"
            assert pipeline_tag == "automatic-speech-recognition"
            return [_FakeModel("OpenVINO/whisper-tiny-fp16-ov")]

    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino.models.HfApi", _FakeHfApi)

    entries = refresh_catalog_from_hub("stt")

    assert entries == [
        ModelEntry(repo_id="OpenVINO/whisper-tiny-fp16-ov", role="stt", display_name="whisper-tiny-fp16-ov")
    ]


def test_refresh_catalog_from_hub_falls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed Hub query should fall back to the curated catalog instead of raising."""

    class _FailingHfApi:
        def list_models(self, *, author: str, pipeline_tag: str) -> list[object]:
            raise RuntimeError("network unavailable")

    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino.models.HfApi", _FailingHfApi)

    entries = refresh_catalog_from_hub("llm")

    assert entries == default_catalog()["llm"]


def test_ensure_model_downloaded_reports_start_and_finish_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Downloading a model should report 0.0 then 1.0 progress and return its local path."""
    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino.models._models_cache_root", lambda: tmp_path)

    def fake_snapshot_download(*, repo_id: str, local_dir: Path) -> str:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino.models.snapshot_download", fake_snapshot_download)
    progress_values: list[float] = []

    result = ensure_model_downloaded("OpenVINO/whisper-tiny-fp16-ov", on_progress=progress_values.append)

    assert progress_values == [0.0, 1.0]
    assert result == tmp_path / "OpenVINO/whisper-tiny-fp16-ov"
    assert (result / ".download_complete").exists()


def test_ensure_model_downloaded_skips_download_when_already_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A model with a completed prior download should not trigger snapshot_download at all."""
    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino.models._models_cache_root", lambda: tmp_path)
    destination = tmp_path / "OpenVINO/whisper-tiny-fp16-ov"
    destination.mkdir(parents=True)
    (destination / ".download_complete").touch()

    def fail_if_called(*, repo_id: str, local_dir: Path) -> str:
        raise AssertionError("snapshot_download should not be called for an already-downloaded model")

    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino.models.snapshot_download", fail_if_called)
    progress_values: list[float] = []

    result = ensure_model_downloaded("OpenVINO/whisper-tiny-fp16-ov", on_progress=progress_values.append)

    assert progress_values == [1.0]
    assert result == destination
