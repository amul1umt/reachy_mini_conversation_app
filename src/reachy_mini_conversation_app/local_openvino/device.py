"""OpenVINO compute-device discovery, run out-of-process from the main app."""

import sys
import json
import logging
from dataclasses import asdict, dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """One OpenVINO-visible compute device (e.g. CPU, GPU, NPU)."""

    id: str
    full_name: str


def list_available_devices() -> list[DeviceInfo]:
    """Return the OpenVINO devices visible on this machine."""
    import openvino as ov

    core = ov.Core()
    devices = []
    for device_id in core.available_devices:
        try:
            full_name = str(core.get_property(device_id, "FULL_DEVICE_NAME"))
        except Exception as exc:
            logger.warning("Failed to read FULL_DEVICE_NAME for %s: %s", device_id, exc)
            full_name = device_id
        devices.append(DeviceInfo(id=device_id, full_name=full_name))
    return devices


def main() -> None:
    """Print detected devices as JSON on stdout, for the main process's subprocess probe."""
    devices = list_available_devices()
    json.dump([asdict(device) for device in devices], sys.stdout)


if __name__ == "__main__":
    main()
