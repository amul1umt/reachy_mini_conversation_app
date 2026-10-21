"""Tests for local OpenVINO device discovery."""

import sys
import types

from reachy_mini_conversation_app.local_openvino.device import DeviceInfo, list_available_devices


def test_list_available_devices_returns_full_names(monkeypatch) -> None:
    """Detected devices should include their OpenVINO-reported full names."""

    class _FakeCore:
        available_devices = ["CPU", "GPU"]

        def get_property(self, device_id: str, name: str) -> str:
            return f"{device_id} friendly name"

    fake_openvino = types.ModuleType("openvino")
    fake_openvino.Core = _FakeCore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openvino", fake_openvino)

    devices = list_available_devices()

    assert devices == [
        DeviceInfo(id="CPU", full_name="CPU friendly name"),
        DeviceInfo(id="GPU", full_name="GPU friendly name"),
    ]


def test_list_available_devices_falls_back_to_id_on_property_error(monkeypatch) -> None:
    """A device whose FULL_DEVICE_NAME lookup fails should still be reported, using its id."""

    class _BrokenCore:
        available_devices = ["NPU"]

        def get_property(self, device_id: str, name: str) -> str:
            raise RuntimeError("no such property")

    fake_openvino = types.ModuleType("openvino")
    fake_openvino.Core = _BrokenCore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openvino", fake_openvino)

    devices = list_available_devices()

    assert devices == [DeviceInfo(id="NPU", full_name="NPU")]
