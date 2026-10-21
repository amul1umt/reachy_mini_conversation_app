"""Tests for local-OpenVINO settings JSON-RPC methods."""

from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.local_openvino.models import ModelEntry
from reachy_mini_conversation_app.local_openvino_routes import register_openvino_methods
from reachy_mini_conversation_app.local_openvino_settings import read_local_openvino_settings


def _rpc_call(client: TestClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = websocket.receive_json()
        return response


def _mount_rpc(
    instance_path: Path | None = None,
    *,
    persist_local_connection: Any = None,
    restart_backend: Any = None,
) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()
    register_openvino_methods(
        rpc,
        instance_path=instance_path,
        persist_local_connection=persist_local_connection,
        restart_backend=restart_backend,
    )
    rpc.mount(app)
    return TestClient(app)


class _FakeProcess:
    """A minimal stand-in for asyncio.subprocess.Process."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminate_called = False

    def terminate(self) -> None:
        self.terminate_called = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


def _patch_subprocess(monkeypatch: pytest.MonkeyPatch, *, port_ready: bool = True) -> _FakeProcess:
    """Stub out subprocess spawning and port-readiness waiting for openvino.start tests."""
    fake_process = _FakeProcess()

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> _FakeProcess:
        return fake_process

    async def fake_wait_for_port(process: object, port: int, *, timeout_s: float) -> bool:
        return port_ready

    monkeypatch.setattr(
        "reachy_mini_conversation_app.local_openvino_routes.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino_routes._wait_for_port", fake_wait_for_port)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.local_openvino_routes._resolve_console_script", lambda: "fake-script"
    )
    return fake_process


def _stub_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ensure_model_downloaded a no-op, for tests that don't care about downloading."""
    monkeypatch.setattr(
        "reachy_mini_conversation_app.local_openvino_routes.ensure_model_downloaded",
        lambda repo_id, *, on_progress=None: None,
    )


def test_devices_reports_detected_hardware_when_extra_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """openvino.devices should surface the probe's detected devices."""

    async def fake_detect_devices() -> tuple[list[dict[str, str]], bool]:
        return [{"id": "CPU", "full_name": "Intel Core Ultra 3"}], True

    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino_routes._detect_devices", fake_detect_devices)
    client = _mount_rpc()

    response = _rpc_call(client, "openvino.devices")

    assert response["result"] == {
        "devices": [{"id": "CPU", "full_name": "Intel Core Ultra 3"}],
        "extra_installed": True,
    }


def test_devices_reports_extra_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """openvino.devices should report extra_installed=False when the probe can't run."""

    async def fake_detect_devices() -> tuple[list[dict[str, str]], bool]:
        return [], False

    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino_routes._detect_devices", fake_detect_devices)
    client = _mount_rpc()

    response = _rpc_call(client, "openvino.devices")

    assert response["result"] == {"devices": [], "extra_installed": False}


def test_devices_real_probe_reports_extra_not_installed() -> None:
    """Without the `local-openvino` extra installed, the real subprocess probe should fail gracefully."""
    client = _mount_rpc()

    response = _rpc_call(client, "openvino.devices")

    assert response["result"] == {"devices": [], "extra_installed": False}


def test_catalog_rejects_unknown_role() -> None:
    """openvino.catalog should reject a role outside stt/llm/vlm/tts."""
    client = _mount_rpc()

    response = _rpc_call(client, "openvino.catalog", {"role": "not-a-role"})

    assert response["error"]["data"]["reason"] == "invalid_role"


def test_catalog_returns_models_for_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """openvino.catalog should map catalog entries into the JSON-RPC shape."""
    monkeypatch.setattr(
        "reachy_mini_conversation_app.local_openvino_routes.refresh_catalog_from_hub",
        lambda role: [ModelEntry(repo_id="OpenVINO/whisper-large-v3-fp16-ov", role=role, display_name="Whisper")],
    )
    client = _mount_rpc()

    response = _rpc_call(client, "openvino.catalog", {"role": "stt"})

    assert response["result"] == {
        "models": [{"repo_id": "OpenVINO/whisper-large-v3-fp16-ov", "display_name": "Whisper"}]
    }


def test_status_reports_idle_with_persisted_selection(tmp_path: Path) -> None:
    """openvino.status should report idle state and the persisted selection."""
    client = _mount_rpc(tmp_path)

    response = _rpc_call(client, "openvino.status")

    assert response["result"] == {
        "state": "idle",
        "port": None,
        "error": None,
        "selection": {
            "stt_model": None,
            "stt_device": "AUTO",
            "llm_model": None,
            "llm_device": "AUTO",
            "llm_role": "llm",
            "tts_model": None,
            "tts_device": "AUTO",
        },
    }


def test_download_persists_selection_and_broadcasts_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """openvino.download should download the model, persist the selection, and report progress."""
    progress_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        "reachy_mini_conversation_app.local_openvino_routes.ensure_model_downloaded",
        lambda repo_id, *, on_progress=None: on_progress and (on_progress(0.0), on_progress(1.0)),
    )
    client = _mount_rpc(tmp_path)

    with client.websocket_connect("/rpc") as websocket:

        def _drain_progress() -> None:
            while True:
                message = websocket.receive_json()
                if message.get("method") == "openvino.progress":
                    progress_events.append(message["params"])
                    continue
                return message

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "openvino.download",
                "params": {"role": "stt", "repo_id": "OpenVINO/whisper-large-v3-fp16-ov", "device": "GPU"},
            }
        )
        response = _drain_progress()

    assert response["result"]["selection"]["stt_model"] == "OpenVINO/whisper-large-v3-fp16-ov"
    assert response["result"]["selection"]["stt_device"] == "GPU"
    assert progress_events == [
        {"role": "stt", "progress": 0.0},
        {"role": "stt", "progress": 1.0},
    ]
    assert read_local_openvino_settings(tmp_path).stt_model == "OpenVINO/whisper-large-v3-fp16-ov"


def test_download_rejects_missing_repo_id(tmp_path: Path) -> None:
    """openvino.download should reject a request without a repo_id."""
    client = _mount_rpc(tmp_path)

    response = _rpc_call(client, "openvino.download", {"role": "stt"})

    assert response["error"]["data"]["reason"] == "invalid_repo_id"


def test_start_downloads_models_spawns_process_and_wires_the_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """openvino.start should download models, start the server, and wire it into the app's connection."""
    _stub_downloads(monkeypatch)
    _patch_subprocess(monkeypatch)
    persisted: list[tuple[str, int]] = []
    restart_backend = AsyncMock()
    client = _mount_rpc(
        tmp_path,
        persist_local_connection=lambda host, port: persisted.append((host, port)),
        restart_backend=restart_backend,
    )

    response = _rpc_call(client, "openvino.start", {"stt_model": "stt-repo", "llm_model": "llm-repo"})

    assert response["result"]["state"] == "running"
    assert response["result"]["port"] == 8765
    assert persisted == [("127.0.0.1", 8765)]
    restart_backend.assert_awaited_once_with("local_openvino_started")

    status = _rpc_call(client, "openvino.status")
    assert status["result"]["state"] == "running"
    assert status["result"]["port"] == 8765


def test_start_persists_llm_role_for_vision_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """openvino.start should persist llm_role so the spawned server gets --vision for a VLM."""
    _stub_downloads(monkeypatch)
    _patch_subprocess(monkeypatch)
    client = _mount_rpc(tmp_path)

    _rpc_call(client, "openvino.start", {"stt_model": "stt-repo", "llm_model": "vlm-repo", "llm_role": "vlm"})

    assert read_local_openvino_settings(tmp_path).llm_role == "vlm"


def test_start_rejects_invalid_llm_role(tmp_path: Path) -> None:
    """openvino.start should reject an llm_role outside llm/vlm."""
    client = _mount_rpc(tmp_path)

    response = _rpc_call(
        client, "openvino.start", {"stt_model": "stt-repo", "llm_model": "llm-repo", "llm_role": "bogus"}
    )

    assert response["error"]["data"]["reason"] == "invalid_llm_role"


def test_start_rejects_when_already_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A second openvino.start while one is running should be rejected, not spawn a second process."""
    _stub_downloads(monkeypatch)
    _patch_subprocess(monkeypatch)
    client = _mount_rpc(tmp_path)
    _rpc_call(client, "openvino.start", {"stt_model": "stt-repo", "llm_model": "llm-repo"})

    response = _rpc_call(client, "openvino.start", {"stt_model": "stt-repo", "llm_model": "llm-repo"})

    assert response["error"]["data"]["reason"] == "already_running"


def test_start_reports_error_state_when_download_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed model download should leave the backend in an error state with a message."""

    def _fail(repo_id: str, *, on_progress: Any = None) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr("reachy_mini_conversation_app.local_openvino_routes.ensure_model_downloaded", _fail)
    client = _mount_rpc(tmp_path)

    response = _rpc_call(client, "openvino.start", {"stt_model": "stt-repo", "llm_model": "llm-repo"})

    assert response["error"]["data"]["reason"] == "download_failed"
    status = _rpc_call(client, "openvino.status")
    assert status["result"]["state"] == "error"
    assert status["result"]["error"] == "network down"


def test_start_reports_error_when_port_never_opens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If the spawned server never starts listening, openvino.start should fail with start_timeout."""
    _stub_downloads(monkeypatch)
    _patch_subprocess(monkeypatch, port_ready=False)
    client = _mount_rpc(tmp_path)

    response = _rpc_call(client, "openvino.start", {"stt_model": "stt-repo", "llm_model": "llm-repo"})

    assert response["error"]["data"]["reason"] == "start_timeout"
    status = _rpc_call(client, "openvino.status")
    assert status["result"]["state"] == "error"


def test_stop_terminates_the_process_and_resets_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """openvino.stop should terminate the running process and reset status back to idle."""
    _stub_downloads(monkeypatch)
    fake_process = _patch_subprocess(monkeypatch)
    client = _mount_rpc(tmp_path)
    _rpc_call(client, "openvino.start", {"stt_model": "stt-repo", "llm_model": "llm-repo"})

    response = _rpc_call(client, "openvino.stop")

    assert response["result"]["state"] == "idle"
    assert response["result"]["port"] is None
    assert response["result"]["error"] is None
    assert fake_process.terminate_called

    status = _rpc_call(client, "openvino.status")
    assert status["result"]["state"] == "idle"
    assert status["result"]["port"] is None


def test_stop_when_nothing_is_running_is_a_no_op(tmp_path: Path) -> None:
    """openvino.stop with no running process should just report idle, not raise."""
    client = _mount_rpc(tmp_path)

    response = _rpc_call(client, "openvino.stop")

    assert response["result"]["state"] == "idle"
