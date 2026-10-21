"""Local OpenVINO realtime backend (optional; requires the `local-openvino` extra).

Nothing in this package is imported by the main app process. It is only ever
run out-of-process, either as the `local-openvino-server` CLI or via a short
subprocess probe (see `local_openvino_routes.py`), so a missing or broken
OpenVINO install can never affect the main conversation app.
"""
