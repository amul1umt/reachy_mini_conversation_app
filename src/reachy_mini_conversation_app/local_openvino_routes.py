"""JSON-RPC methods for the local OpenVINO realtime backend (settings UI)."""

import sys
import json
import shutil
import asyncio
import logging
from typing import Any, Literal
from pathlib import Path
from dataclasses import replace, dataclass
from collections.abc import Callable, Awaitable

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.local_openvino.models import (
    Role,
    ensure_model_downloaded,
    refresh_catalog_from_hub,
)
from reachy_mini_conversation_app.local_openvino_settings import (
    DEFAULT_DEVICE,
    LocalOpenvinoSettings,
    read_local_openvino_settings,
    write_local_openvino_settings,
)


logger = logging.getLogger(__name__)

_DEVICE_PROBE_TIMEOUT_S = 15.0
_STARTUP_TIMEOUT_S = 120.0
_SHUTDOWN_TIMEOUT_S = 10.0
_DEFAULT_PORT = 8765

# The UI groups vision under the LLM role; both share the same settings fields.
_ROLE_SETTINGS_FIELDS: dict[Role, tuple[str, str]] = {
    "stt": ("stt_model", "stt_device"),
    "llm": ("llm_model", "llm_device"),
    "vlm": ("llm_model", "llm_device"),
    "tts": ("tts_model", "tts_device"),
}

BackendState = Literal["idle", "downloading", "starting", "running", "error"]

PersistLocalConnection = Callable[[str, int], None]
RestartBackend = Callable[[str], Awaitable[None]]


@dataclass
class _ProcessState:
    """Mutable state for the spawned `local-openvino-server` subprocess."""

    process: "asyncio.subprocess.Process | None" = None
    state: BackendState = "idle"
    port: int | None = None
    error: str | None = None


async def _detect_devices() -> tuple[list[dict[str, str]], bool]:
    """Return detected OpenVINO devices and whether the extra is installed.

    Runs the probe in a short-lived subprocess so the main app process never
    imports `openvino` itself (a missing or broken install must not affect it).
    """
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "reachy_mini_conversation_app.local_openvino.device",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=_DEVICE_PROBE_TIMEOUT_S)
    except (OSError, asyncio.TimeoutError) as exc:
        logger.info("OpenVINO device probe unavailable: %s", exc)
        return [], False

    if process.returncode != 0:
        logger.debug(
            "OpenVINO device probe exited %s: %s", process.returncode, stderr.decode("utf-8", errors="replace")
        )
        return [], False

    try:
        devices: list[dict[str, str]] = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("OpenVINO device probe returned invalid JSON: %s", exc)
        return [], False
    return devices, True


def _resolve_console_script() -> str:
    """Return the path to the installed `reachy-mini-conversation-app` console script."""
    found = shutil.which("reachy-mini-conversation-app")
    if found:
        return found
    # Fall back to the same venv/bin (or Scripts, on Windows) directory as the
    # running interpreter, in case the venv itself isn't on PATH.
    for candidate_name in ("reachy-mini-conversation-app", "reachy-mini-conversation-app.exe"):
        candidate = Path(sys.executable).parent / candidate_name
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("Could not locate the reachy-mini-conversation-app console script")


async def _wait_for_port(process: "asyncio.subprocess.Process", port: int, *, timeout_s: float) -> bool:
    """Poll until something accepts TCP connections on ``port``, or the process exits/times out."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if process.returncode is not None:
            return False
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.5)
            continue
        writer.close()
        await writer.wait_closed()
        return True
    return False


def _settings_payload(settings: LocalOpenvinoSettings) -> dict[str, object]:
    """Return the JSON-RPC-friendly shape of a persisted model/device selection."""
    return {
        "stt_model": settings.stt_model,
        "stt_device": settings.stt_device,
        "llm_model": settings.llm_model,
        "llm_device": settings.llm_device,
        "llm_role": settings.llm_role,
        "tts_model": settings.tts_model,
        "tts_device": settings.tts_device,
    }


def _status_payload(process_state: _ProcessState, settings: LocalOpenvinoSettings) -> dict[str, object]:
    """Return the JSON-RPC-friendly shape of the backend's current lifecycle state."""
    return {
        "state": process_state.state,
        "port": process_state.port,
        "error": process_state.error,
        "selection": _settings_payload(settings),
    }


def _validated_role(params: dict[str, Any]) -> Role:
    """Return the requested role, or raise a JSON-RPC error for an unknown one."""
    role = params.get("role")
    if role not in _ROLE_SETTINGS_FIELDS:
        raise JsonRpcError(f"invalid role: {role!r}", reason="invalid_role", code=-32602)
    return role  # type: ignore[no-any-return]


def _required_str(params: dict[str, Any], key: str) -> str:
    """Return a required non-empty string param, or raise a JSON-RPC error."""
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JsonRpcError(f"{key} is required", reason=f"invalid_{key}", code=-32602)
    return value


def register_openvino_methods(
    rpc: JsonRpcServer,
    *,
    instance_path: str | Path | None = None,
    persist_local_connection: PersistLocalConnection | None = None,
    restart_backend: RestartBackend | None = None,
) -> None:
    """Register local-OpenVINO settings methods (device discovery, model catalog, lifecycle).

    ``persist_local_connection``/``restart_backend`` wire a successfully
    started local server into the app's existing `local` connection mode
    (see `console.py`'s `_persist_hf_direct_connection`/`request_backend_restart`);
    without them, `openvino.start` can still download/select models but
    cannot make the main conversation loop actually use the local backend.
    """
    process_state = _ProcessState()

    async def _devices(_params: dict[str, Any]) -> dict[str, object]:
        devices, extra_installed = await _detect_devices()
        return {"devices": devices, "extra_installed": extra_installed}

    async def _catalog(params: dict[str, Any]) -> dict[str, object]:
        role = _validated_role(params)
        entries = await asyncio.to_thread(refresh_catalog_from_hub, role)
        return {"models": [{"repo_id": entry.repo_id, "display_name": entry.display_name} for entry in entries]}

    async def _status(_params: dict[str, Any]) -> dict[str, object]:
        settings = await asyncio.to_thread(read_local_openvino_settings, instance_path)
        return _status_payload(process_state, settings)

    async def _download(params: dict[str, Any]) -> dict[str, object]:
        role = _validated_role(params)
        repo_id = _required_str(params, "repo_id")
        device = str(params.get("device") or DEFAULT_DEVICE).strip() or DEFAULT_DEVICE

        def _on_progress(progress: float) -> None:
            rpc.broadcast_threadsafe("openvino.progress", {"role": role, "progress": progress})

        try:
            await asyncio.to_thread(ensure_model_downloaded, repo_id, on_progress=_on_progress)
        except Exception as exc:
            logger.exception("Failed to download OpenVINO model %s", repo_id)
            raise JsonRpcError(f"failed to download {repo_id}: {exc}", reason="download_failed") from exc

        current = await asyncio.to_thread(read_local_openvino_settings, instance_path)
        if role == "stt":
            updated = replace(current, stt_model=repo_id, stt_device=device)
        elif role in ("llm", "vlm"):
            updated = replace(current, llm_model=repo_id, llm_device=device, llm_role=role)
        else:
            updated = replace(current, tts_model=repo_id, tts_device=device)
        await asyncio.to_thread(write_local_openvino_settings, instance_path, updated)

        return {"role": role, "repo_id": repo_id, "selection": _settings_payload(updated)}

    async def _download_all(settings: LocalOpenvinoSettings) -> None:
        """Download every model referenced by ``settings``, broadcasting per-role progress."""
        roles_and_models = [("stt", settings.stt_model), (settings.llm_role, settings.llm_model)]
        if settings.tts_model:
            roles_and_models.append(("tts", settings.tts_model))

        for role, model in roles_and_models:
            if not model:
                raise JsonRpcError(f"{role} model is required", reason=f"invalid_{role}_model", code=-32602)

            def _on_progress(progress: float, role: str = role) -> None:
                rpc.broadcast_threadsafe("openvino.progress", {"role": role, "progress": progress})

            await asyncio.to_thread(ensure_model_downloaded, model, on_progress=_on_progress)

    def _build_server_command(settings: LocalOpenvinoSettings, port: int) -> list[str]:
        command = [
            _resolve_console_script(),
            "local-openvino-server",
            "--stt-model",
            str(settings.stt_model),
            "--stt-device",
            settings.stt_device,
            "--llm-model",
            str(settings.llm_model),
            "--llm-device",
            settings.llm_device,
            "--port",
            str(port),
        ]
        if settings.llm_role == "vlm":
            command.append("--vision")
        if settings.tts_model:
            command += ["--tts-model", settings.tts_model, "--tts-device", settings.tts_device]
        return command

    async def _start(params: dict[str, Any]) -> dict[str, object]:
        if process_state.process is not None and process_state.process.returncode is None:
            raise JsonRpcError("Local OpenVINO backend is already running", reason="already_running")

        current = await asyncio.to_thread(read_local_openvino_settings, instance_path)
        llm_role = params.get("llm_role") or "llm"
        if llm_role not in ("llm", "vlm"):
            raise JsonRpcError(f"invalid llm_role: {llm_role!r}", reason="invalid_llm_role", code=-32602)
        settings = replace(
            current,
            stt_model=_required_str(params, "stt_model"),
            stt_device=str(params.get("stt_device") or DEFAULT_DEVICE),
            llm_model=_required_str(params, "llm_model"),
            llm_device=str(params.get("llm_device") or DEFAULT_DEVICE),
            llm_role=llm_role,
            tts_model=(params.get("tts_model") or None),
            tts_device=str(params.get("tts_device") or DEFAULT_DEVICE),
        )
        await asyncio.to_thread(write_local_openvino_settings, instance_path, settings)

        process_state.state = "downloading"
        process_state.error = None
        try:
            await _download_all(settings)
        except Exception as exc:
            process_state.state = "error"
            process_state.error = str(exc)
            if isinstance(exc, JsonRpcError):
                raise
            logger.exception("Failed to download local OpenVINO models")
            raise JsonRpcError(f"failed to download models: {exc}", reason="download_failed") from exc

        process_state.state = "starting"
        port = _DEFAULT_PORT
        command = _build_server_command(settings, port)
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as exc:
            process_state.state = "error"
            process_state.error = str(exc)
            raise JsonRpcError(f"failed to start the local backend: {exc}", reason="start_failed") from exc

        process_state.process = process
        if not await _wait_for_port(process, port, timeout_s=_STARTUP_TIMEOUT_S):
            process_state.state = "error"
            process_state.error = "Local backend did not start listening in time."
            raise JsonRpcError(process_state.error, reason="start_timeout")

        process_state.state = "running"
        process_state.port = port

        if persist_local_connection is not None:
            persist_local_connection("127.0.0.1", port)
        if restart_backend is not None:
            await restart_backend("local_openvino_started")

        return _status_payload(process_state, settings)

    async def _stop(_params: dict[str, Any]) -> dict[str, object]:
        process = process_state.process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=_SHUTDOWN_TIMEOUT_S)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        process_state.process = None
        process_state.state = "idle"
        process_state.port = None
        process_state.error = None
        settings = await asyncio.to_thread(read_local_openvino_settings, instance_path)
        return _status_payload(process_state, settings)

    rpc.register("openvino.devices", _devices)
    rpc.register("openvino.catalog", _catalog)
    rpc.register("openvino.status", _status)
    rpc.register("openvino.download", _download)
    rpc.register("openvino.start", _start)
    rpc.register("openvino.stop", _stop)
