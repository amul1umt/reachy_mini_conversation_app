"""Tests for persisted local-OpenVINO model/device selection."""

from reachy_mini_conversation_app.local_openvino_settings import (
    DEFAULT_DEVICE,
    LocalOpenvinoSettings,
    read_local_openvino_settings,
    write_local_openvino_settings,
)


def test_write_and_read_round_trips(tmp_path) -> None:
    """A written selection should read back identically."""
    settings = LocalOpenvinoSettings(
        stt_model="OpenVINO/whisper-large-v3-fp16-ov",
        stt_device="CPU",
        llm_model="OpenVINO/Qwen3-8B-int4-ov",
        llm_device="GPU",
        tts_model=None,
        tts_device=DEFAULT_DEVICE,
    )

    write_local_openvino_settings(tmp_path, settings)

    assert read_local_openvino_settings(tmp_path) == settings


def test_read_missing_file_returns_defaults(tmp_path) -> None:
    """Reading before anything has been saved should return AUTO-device defaults."""
    assert read_local_openvino_settings(tmp_path) == LocalOpenvinoSettings()


def test_read_none_instance_path_returns_defaults() -> None:
    """A None instance path (no managed instance dir) should return defaults, not raise."""
    assert read_local_openvino_settings(None) == LocalOpenvinoSettings()


def test_read_invalid_payload_falls_back_to_defaults(tmp_path) -> None:
    """A corrupt settings file should log and fall back to defaults rather than raise."""
    (tmp_path / "local_openvino_settings.json").write_text("not json", encoding="utf-8")

    assert read_local_openvino_settings(tmp_path) == LocalOpenvinoSettings()
