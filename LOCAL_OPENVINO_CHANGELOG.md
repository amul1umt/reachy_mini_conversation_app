# Local OpenVINO Backend — Implementation Changelog

> **Read this before touching anything under `src/reachy_mini_conversation_app/local_openvino/`,
> `local_openvino_routes.py`, `local_openvino_settings.py`, or the "Local OpenVINO backend" UI
> section.** This file is the durable record of what was built, why, and — most importantly —
> the real bugs that were found and fixed along the way, so a future session doesn't have to
> rediscover them the hard way. It is written for an agent with no memory of this conversation.
>
> The original design plan (written before implementation, less detailed on real-world findings)
> is at `/home/robot/.claude/plans/analyze-this-project-especially-warm-bachman.md` on the machine
> this was built on — read it for the wire-protocol spec in full; this file focuses on what
> changed *during* implementation and why.

## What this feature is, in one paragraph

`reachy_mini_conversation_app` normally delegates all AI (LLM, STT, TTS, VAD, vision) to one
remote multimodal endpoint over a WebSocket, via `HuggingFaceRealtimeHandler`
(`src/reachy_mini_conversation_app/huggingface_realtime.py`). This feature adds a **second,
independent realtime server** (`local_openvino/`) that speaks the *exact same* wire protocol, runs
entirely on-device via OpenVINO (Intel Core Ultra CPU/GPU/NPU), and plugs into the app through the
connection mode that already existed for exactly this purpose (`HF_REALTIME_CONNECTION_MODE=local`
+ `HF_REALTIME_WS_URL`). **`huggingface_realtime.py` and `conversation_handler.py` were never
modified** — that is a load-bearing design constraint, not an accident. Do not change it without
a strong reason; the entire value of this design is that the existing client, and all its tests,
keep working untouched.

## Why this exists / why it's a personal fork, not upstream

One month before this work started, the maintainers deliberately deleted ~3,000 lines of
multi-backend abstraction (commit `5b8d974`, "Consolidate the app around the default backend",
2026-07-02) to simplify the app to a single Hugging Face backend. Adding local multi-model
inference goes directly against that decision. **The user explicitly confirmed this is for a
personal fork**, not an upstream contribution, when asked directly — optimizing for "get local
OpenVINO inference working well" over "minimal footprint / match upstream review conventions." If
this is ever revisited for upstreaming, that conversation needs to happen with the maintainers
first (per `AGENTS.md`'s "issue first" rule) — don't assume it's wanted.

## Order of work (7 phases, each gated on ruff + mypy + pytest before moving on)

1. **Scaffolding** — OpenVINO device detection (`local_openvino/device.py`), run out-of-process
   via a subprocess probe so the main app never imports `openvino` itself. RPC (`openvino.devices`)
   and a read-only device table in Settings.
2. **Model catalog + downloads** — `local_openvino/models.py` queries the Hugging Face Hub live
   (`HfApi().list_models(...)`) for current OpenVINO-org models per role, with a tiny hardcoded
   fallback catalog. `ensure_model_downloaded()` wraps `snapshot_download`. RPC
   (`openvino.catalog`, `openvino.download`) and per-role model/device pickers in Settings.
3. **Protocol bring-up** — `local_openvino/session.py`'s `RealtimeSession` is the heart of this
   feature: it implements the *exact* client↔server event sequence
   `HuggingFaceRealtimeHandler` expects (see the plan file for the full spec). STT/LLM pipelines
   wired in behind small `Protocol` interfaces so `session.py` never imports `openvino_genai`
   directly and is fully unit-testable without hardware.
4. **TTS** — see "Kokoro / TTS pivot" below; this phase changed direction mid-stream based on user
   input.
5. **Tool-calling + vision** — prompt-driven tool-calling (not dependent on any specific
   OpenVINO GenAI API — see "Tool-calling design" below), plus a `ChatMessage(role, content,
   image)` dataclass replacing plain dicts in the conversation history, so images can flow through
   to a VLM.
6. **Process lifecycle** — `openvino.start`/`openvino.stop` RPC methods spawn/manage the
   `local-openvino-server` subprocess and wire it into the app's existing `local` connection mode
   (`console.py`'s `_persist_hf_direct_connection`/`request_backend_restart`, threaded through as
   callables).
7. **Polish** — Silero VAD (replacing the energy-threshold one, with automatic fallback),
   per-device construction fallback (retry on `AUTO` if a requested device fails), README/
   `.env.example` docs. **This phase is also where most of the real bugs below were found**, because
   real OpenVINO/OpenVINO-GenAI packages got installed for validation (see next section).

## Real hardware validation: what was found, and why it matters

Partway through Phase 7, `openvino`/`openvino-genai`/`kokoro-onnx`/`onnxruntime`/`pillow` were
installed for real (`uv sync --extra local-openvino`) to validate the pipeline wrapper code instead
of leaving it as pure guesswork. **This machine turned out to have genuine Intel Core Ultra X7
358H hardware with an Arc B390 iGPU and an AI Boost NPU** — exactly the target hardware class. This
was not expected going in, and it surfaced real, would-have-shipped-broken bugs. **Do not assume
the pipeline wrapper code in `local_openvino/pipelines/` is untested guesswork — most of it is
now verified against real inference on real CPU/GPU/NPU.** The exceptions are called out explicitly
in "What's still unverified" below.

### Bugs found and fixed via live testing

1. **`LLMPipeline.generate()` does not accept a plain `list[dict]`.** The original code (written
   before real testing) passed `[{"role": ..., "content": ...}]` directly to `generate()`. The real
   API only accepts `str | list[str] | ChatHistory | Tensor | TokenizedInputs`. This would have
   **crashed on first use on real hardware**. Fix: wrap in `openvino_genai.ChatHistory(messages)`
   before calling `generate()`. See `pipelines/llm.py`. `ChatHistory`'s constructor conveniently
   accepts exactly the `[{"role": ..., "content": ...}]` shape already being built, so the fix was
   small.
2. **`WhisperPipeline.generate()`/`LLMPipeline.generate()` return a results object, not a string.**
   `str(result)` happens to work by accident (the pybind11 binding defines `__str__` to return the
   primary transcript/reply text) but is not the documented, robust way to extract it. Fixed to use
   `.texts[0]` explicitly (`WhisperDecodedResults.texts: list[str]`, same on `DecodedResults` for
   `LLMPipeline`). If you see `str(pipeline.generate(...))` anywhere, it's a smell — use `.texts[0]`.
3. **`VLMPipeline.generate()` has a completely different signature from `LLMPipeline.generate()`.**
   It takes `prompt: str` + `images: list[openvino.Tensor]` — not a `ChatHistory`, and images are
   **not raw JPEG bytes**, they must be decoded to an NHWC uint8 RGB `openvino.Tensor`. `VLMPipeline`
   is also normally used *statefully* via `start_chat()`/`generate(prompt, images=...)`/
   `finish_chat()`, which conflicts with this app's stateless "rebuild the full prompt every call"
   design. `pipelines/vlm.py`'s `LocalVLM` was rewritten to flatten the `ChatMessage` history into
   one prompt string per call (stateless, matching `LocalLLM`) and decode JPEG→NHWC tensor via
   Pillow. **This path was NOT live-tested with a real VLM model** (too large/slow to download
   during this session) — only the tensor-construction mechanics were verified
   (`openvino.Tensor` shape `[1, H, W, 3]`, dtype uint8, confirmed via a synthetic red-square JPEG).
   If you're picking this up: download a small VLM (e.g. search the `OpenVINO` HF org for an
   `image-text-to-text` model) and actually run `LocalVLM.generate()` with a real image before
   trusting this path.
4. **A locally-downloaded 0.5B model (`OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov`) did not
   reliably reproduce the instructed `<tool_call>...</tool_call>` wrapper tags**, even though it
   got the JSON payload itself right (`{"name": "look", "arguments": {"direction": "left"}}` with no
   wrapper — it also got the tool name wrong, "look" instead of "move_head", which is just small-model
   weakness, not a protocol bug). Fixed `_parse_tool_call()` in `session.py` to also accept a bare
   JSON object when it's the model's *entire* trimmed reply (not accepted mid-sentence, to avoid
   false-positiving on ordinary text that happens to contain braces). **Expect this kind of
   instruction-following flakiness to scale with model size** — a bigger, more capable model
   should follow the wrapper format more reliably, but the fallback parsing should stay regardless.
5. **NPU compilation genuinely failed for the LLM** (`OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov`
   on real NPU hardware): `StopLocationVerifierPass ... Found 80 duplicated names`, then
   `ZE_RESULT_ERROR_INVALID_ARGUMENT` from the Level Zero driver. **STT (Whisper) worked fine on
   the same NPU.** This is real evidence that device support is genuinely model-dependent, not a
   simple "does this hardware have an NPU" question — confirms the value of the
   `_build_with_device_fallback()` helper in `cli.py` (retries on `AUTO` if the requested device's
   construction raises), which was validated to correctly recover from exactly this failure live.
   GPU worked for both STT and this LLM (with a benign-looking `vpux-compiler` warning in the logs
   that did not actually block execution).
6. **`optimum-intel[openvino]` was a dead dependency.** It was added speculatively in Phase 1
   ("might be needed for model conversion") and never once imported anywhere in the code — all
   model loading goes through `huggingface_hub.snapshot_download` on already-converted OpenVINO IR
   repos, then straight into `openvino_genai`. It was pulling in `torch`, `transformers`,
   `scikit-learn`, and a pile of CUDA wheels (irrelevant on this Intel-only target) into `uv.lock`
   for nothing. **Removed.** If you think you need `optimum-intel`, check first whether the model
   you want is already published pre-converted under the `OpenVINO` HF org — it almost certainly
   is, and you don't need it.
7. **A module-level `from PIL import Image` in `pipelines/vlm.py` and in its test file would have
   broken test collection in the default (no-extras) CI environment.** `Pillow` is only installed
   under the `local-openvino` extra; `pytest tests/` in the *default* environment must never hard-
   `import` anything from that extra at module scope. Fixed by deferring the `PIL` import inside
   `_jpeg_bytes_to_tensor()` (matching the existing pattern in every other `pipelines/*.py` file —
   `openvino_genai`/`kokoro_onnx` are all deferred inside `__init__`/methods, never at module top),
   and gating the two PIL-dependent tests with `pytest.importorskip("PIL")` /
   `pytest.importorskip("openvino")`. **Rule of thumb reinforced by this bug: everything under
   `local_openvino/` must be importable with zero extras installed; only calling into a pipeline
   class's methods should ever touch the optional dependencies.**
8. **`RealtimeSession.run()` (`session.py`) crashed the whole connection on a malformed inbound
   frame, found via live testing with a real, long-running app↔backend connection (not synthetic
   tests).** `receive_json()` can raise `json.JSONDecodeError` for a frame that isn't a real
   disconnect (a real disconnect raises `WebSocketDisconnect`, which *was* already handled); the
   loop only caught `WebSocketDisconnect`, so a `JSONDecodeError` propagated uncaught out of the
   FastAPI websocket route, and the ASGI server dropped the connection ungracefully. Client-side
   this showed up as `HuggingFaceRealtimeHandler` logging "Realtime websocket closed unexpectedly
   ... no close frame received or sent" roughly every 60-90s during a real session with continuous
   mic audio streaming — **this is a very plausible contributor to "voice commands aren't being
   detected reliably"** on a real deployment, since the app↔backend connection would silently die
   and reconnect on this cadence. The exact origin of the empty/malformed frame wasn't root-caused
   (not a real WS close frame, not from the `openai` SDK's own ping handling as far as could be
   determined); the fix doesn't need to know why — `run()` now catches `JSONDecodeError` the same
   way as `WebSocketDisconnect`, logs it at debug level, and continues the loop instead of dying.
   Verified live: before the fix, a real app↔backend session died on this schedule every time;
   after the fix, the same setup ran 20+ minutes with continuous mic streaming and multiple LLM
   responses with zero disconnects.

### What's still unverified (be skeptical, verify before relying on it)

- **`LocalVLM` end-to-end with a real VLM + real image.** Only the tensor-construction mechanics
  were checked; the actual `VLMPipeline.generate(prompt, images=[...])` call, and whether the
  flattened-prompt/stateless approach produces sane replies, were never run.
- **Silero VAD (`local_openvino/silero_vad.py`)**, tensor names/shapes (`input`/`state`/`sr`,
  single combined `(2, 1, 128)` state matching the v4+ ONNX export) — based on public
  documentation of Silero VAD's calling convention, not live-tested (no `onnxruntime` session was
  actually run against a downloaded `deepghs/silero-vad-onnx` model). It has automatic fallback to
  the energy-threshold `VoiceActivityDetector` if construction or inference fails, so a wrong
  guess here degrades gracefully rather than breaking the session — but the "improved VAD quality"
  goal of this phase is unconfirmed.
- **Device selection for TTS.** Kokoro runs on plain ONNX Runtime CPU only (see below); the
  `--tts-device` flag is accepted but a non-CPU/AUTO value just logs a warning and is ignored.
  This was a deliberate scope decision, not a bug, but it means TTS never actually got a "does the
  device fallback matter here" test the way STT/LLM did.
- **Audio quality/latency at the human-listening level** — TTS output was checked for correct byte
  length and shape, not listened to. STT was only tested against silence (returned `" you"`, a
  known Whisper artifact on silence — not a real transcript test). Real speech input was never
  fed through any pipeline in this session.

## Kokoro / TTS pivot (Phase 4) — why it doesn't use OpenVINO GenAI

The original plan's Phase 4 spike task was to check whether `openvino_genai.Text2SpeechPipeline`
(SpeechT5-based) existed and use it. **The user redirected this mid-session**: Kokoro (a small,
high-quality TTS model) should be used instead over an ONNX pipeline, because OpenVINO has limited
TTS model support. Investigation confirmed: Kokoro's dynamic-rank ONNX graph is **documented as
incompatible with the OpenVINO execution provider** (falls back to CPU or fails to initialize,
per the `kokoro-onnx`/`pykokoro` project issue trackers). So TTS deliberately runs on plain ONNX
Runtime CPU (`pipelines/tts.py`'s `KokoroTTS`, via the `kokoro-onnx` PyPI package), *not* through
an OpenVINO device — this is intentional, not a limitation to "fix" by forcing it onto GPU/NPU.

**Model repo matters here and was verified by inspecting file listings, not guessed**:
`fastrtc/kokoro-onnx` ships `kokoro-v1.0.onnx` + `voices-v1.0.bin` (one combined file for all
voices) — the exact layout the `kokoro_onnx.Kokoro(model_path, voices_path)` constructor expects.
`onnx-community/Kokoro-82M-*-ONNX` (an easy repo to reach for instead) ships one file **per voice**
under `voices/<name>.bin` and is **not** compatible with this loader. If the model catalog or
default ever changes, keep pointing at `fastrtc/kokoro-onnx` (or verify any replacement's file
layout the same way — `huggingface_hub.HfApi().list_repo_files(repo_id)` — before trusting it).

## Tool-calling design — why prompt-driven, not a native API

`openvino_genai.ChatHistory` does have a native `set_tools()`/`get_tools()` mechanism (discovered
during live testing), which could let the model's own chat template render tool schemas instead of
a hand-written prompt instruction. **This was deliberately not used.** The current design
(`_build_tool_instructions()` renders available tools into a system message; `_parse_tool_call()`
reads a `<tool_call>{...}</tool_call>` block or bare JSON back out of the raw reply — see
`session.py`) works regardless of which model is selected from the catalog, is fully unit-testable
without any real pipeline, and was validated end-to-end (full round trip: tool call → client
executes → `function_call_output` → follow-up reply) against both a scripted stub and the real
`openai` SDK client. Switching to `ChatHistory.set_tools()` would likely produce better-formatted
output *for Qwen-family models specifically* (their template already uses the same `<tool_call>`
convention) but would tie correctness to a chat-template detail of whatever model is picked. If
revisiting this, keep `_parse_tool_call()`'s lenient bare-JSON fallback either way — see bug #4
above for why.

## DLStreamer — deliberately not used, don't re-litigate this

Early in planning, GStreamer/DLStreamer came up as a possible vision-pipeline approach. **This
was investigated and rejected**: the app's `camera` tool (`tools/camera.py`) does one on-demand
JPEG snapshot per question, not a continuous video stream — there is no GStreamer pipeline in this
repo to extend (the only GStreamer code lives inside the separately-installed `reachy_mini` SDK,
which is off-limits to fork/vendor per `AGENTS.md`). DLStreamer would also bring real ABI conflicts
(`openvino`/`openvino-tokenizers`/`openvino-genai` must stay version-lockstepped; DLStreamer pins
its own matched set per release and isn't a normal `uv`-resolvable pip package). Vision in this
feature is handled entirely by feeding the captured JPEG into a VLM directly (see `pipelines/vlm.py`)
— no GStreamer/DLStreamer involvement, and none should be added unless the *app* grows a genuine
continuous-video-analytics feature outside of this local-OpenVINO work (in which case DLStreamer
should run as an isolated sidecar process, never sharing a Python environment with `local_openvino/`).

## Phase 8 — skip re-downloading already-cached models

`ensure_model_downloaded()` (`models.py`) previously called `snapshot_download()` unconditionally
every time, including on every "Download & start local backend" click and every
`local-openvino-server` startup — `snapshot_download()` still re-verifies every file's ETag against
the Hub over the network even when nothing needs downloading, which is slow and requires network
access on every restart. Fixed by writing a `.download_complete` marker file into the destination
directory only after a snapshot finishes successfully; if the marker is already there,
`ensure_model_downloaded()` returns the cached path immediately with no network call at all. The
marker is written *after* success specifically so an interrupted download is retried rather than
mistaken for complete. Models downloaded before this fix won't have the marker on first run after
upgrading (one real `snapshot_download()` pass, same as before) but will skip on every run after
that. Verified live: a fresh server startup with all four models already marked complete (STT, LLM,
TTS, Silero VAD) logged `already downloaded, skipping` for each with no HTTP calls, versus dozens of
`HEAD https://huggingface.co/...` requests taking 30-90s+ per model before the fix.

## Phase 9 — text chat mode (no-microphone fallback)

Added a text input to the web UI's Talk view (`static/js/views/talk.js`) as a second, non-voice way
into the same conversation session — for setups without a working microphone, or just to type
instead of speak. **No backend or protocol changes were needed for this**: the app already had a
complete text-injection path end-to-end before this session — `ConversationHandler.say()`
(`conversation_handler.py`, implemented in `huggingface_realtime.py`) creates a `conversation.item.create`
user message and triggers `response.create`, exposed over `/rpc` as `conversation.say`
(`console.py`) — and `local_openvino/session.py`'s `_handle_item_create`/`_trigger_response` already
handled that exact same event shape identically to the Hugging Face backend, since it was built to
mirror the same wire protocol from Phase 3 onward. The only gap was that nothing in the UI called
`conversation.say`. Added `sayText()` to `api.js` and a chat log + form to the Talk view, subscribing
to the existing `conversation.transcript` broadcast to render replies (and, for voice turns, the
user's spoken transcript too — both surfaces share one chat log). Verified live end-to-end over the
real `/rpc` WebSocket against the local OpenVINO backend: `conversation.say` → `response_created` →
`assistant_audio_delta` → `assistant_transcript_done` → `conversation.transcript` with the correct
reply text, matching what the frontend listens for.

**Frontend was not visually verified in a browser this session** (browser automation was
unavailable) — the JS was checked with `node --check` (valid ESM syntax) and the RPC/event contract
it depends on was verified directly over the wire, but nobody has looked at the rendered chat
bubbles. If picking this up: open `--ui`, go to Talk, and confirm the chat box actually renders and
scrolls sensibly before trusting it further.

## Frontend gap found and fixed during Phase 6

The Settings UI's "LLM (+ vision)" row originally only fetched the `llm` catalog role
(`getOpenvinoCatalog("llm")`), never `vlm` — meaning **there was no way to actually pick a vision
model through the UI**, even though the backend (`pipelines/vlm.py`, `--vision` CLI flag) fully
supported it. Fixed in `static/js/views/settings.js`: the LLM row now fetches both `llm` and `vlm`
catalogs and merges them into one dropdown, tagging each option with its source role (shown as a
`(vlm)` suffix) so `openvino.download`/`openvino.start` get the correct role. `LocalOpenvinoSettings`
gained an `llm_role: str = "llm"` field to persist which one was picked, since `--vision` needs to
be known at server-spawn time (`cli.py`'s `run_local_openvino_server` picks `LocalVLM` vs `LocalLLM`
based on it).

## Current file map

```
src/reachy_mini_conversation_app/
  local_openvino_routes.py       # RPC: openvino.devices/catalog/status/download/start/stop
  local_openvino_settings.py     # persisted model/device selection (local_openvino_settings.json)
  local_openvino/
    __init__.py
    device.py                    # OpenVINO Core().available_devices probe (run via -m, subprocess-only)
    models.py                    # HF Hub catalog query + snapshot_download wrapper
    vad.py                       # energy-threshold VoiceActivityDetector (default/fallback)
    silero_vad.py                # SileroVAD (ONNX) + load_vad() fallback chooser — UNVERIFIED, see above
    session.py                   # RealtimeSession: the wire-protocol implementation, ChatMessage,
                                  # tool-call prompt/parse, all Protocol interfaces (SpeechToText,
                                  # ConversationalLLM, TextToSpeech, VadDetector, RealtimeTransport)
    server.py                    # FastAPI app, /v1/realtime websocket route
    cli.py                       # `local-openvino-server` subcommand entry point, device fallback,
                                  # --vision flag, Silero VAD download
    pipelines/
      stt.py                     # WhisperSTT — LIVE VALIDATED (CPU/GPU/NPU)
      llm.py                     # LocalLLM — LIVE VALIDATED (CPU/GPU; NPU fails for this model, by design of the model/driver, not a bug)
      vlm.py                     # LocalVLM — NOT live validated, see "What's still unverified"
      tts.py                     # KokoroTTS — LIVE VALIDATED (real audio output shape/length)
tests/
  test_local_openvino_device.py
  test_local_openvino_models.py
  test_local_openvino_settings.py
  test_local_openvino_routes.py
  test_local_openvino_session.py       # highest-value: full protocol-compliance suite
  test_local_openvino_pipelines.py     # locks in the ChatHistory/.texts[0]/tensor bugs fixed above
  test_local_openvino_tts.py
  test_local_openvino_silero_vad.py
  test_local_openvino_cli.py           # device-fallback + VAD-download-failure tests
```

Modified (not new) files: `pyproject.toml` (`local-openvino` extra), `uv.lock`, `console.py`
(wires `register_openvino_methods` with `persist_local_connection`/`restart_backend`), `main.py` +
`utils.py` (the `local-openvino-server` subcommand), `static/js/api.js` + `static/js/views/
settings.js` (Settings UI section) + `static/js/views/talk.js` (chat text input, Phase 9),
`README.md`, `.env.example`.

## How to validate changes on real hardware

```bash
# Install the extra (heavy: openvino, openvino-genai, kokoro-onnx, onnxruntime, pillow)
uv sync --extra local-openvino

# Confirm device enumeration
uv run python -c "
import openvino as ov
core = ov.Core()
for d in core.available_devices:
    print(d, core.get_property(d, 'FULL_DEVICE_NAME'))
"

# Run the server standalone (downloads models on first run)
reachy-mini-conversation-app local-openvino-server \
  --stt-model OpenVINO/whisper-large-v3-fp16-ov \
  --llm-model OpenVINO/Qwen3-8B-int4-ov \
  --tts-model fastrtc/kokoro-onnx

# Restore the clean baseline before running the test gate / committing —
# CI runs with no extras installed, and one test
# (test_devices_real_probe_reports_extra_not_installed) specifically asserts that state:
uv sync
```

**Always run the full gate in the clean (no-extras) environment before considering work done**:
`ruff check . --fix && ruff format . && mypy --pretty --show-error-codes && pytest tests/ -v`. As
of the end of this session: 370 passed, 1 skipped (the PIL/openvino-gated VLM tensor test, which
only runs when the extra is installed).

## Outstanding / good next steps for a future session

- Live-test `LocalVLM` end-to-end with a real small VLM and a real image (see bug #3).
- Live-test `SileroVAD` against a real downloaded `deepghs/silero-vad-onnx` model with real speech
  audio, not silence (see "What's still unverified").
- Human-listen to actual Kokoro TTS output for quality/intelligibility; only byte-shape was checked.
- Flagged but not actioned: `docs/scheme.mmd`/`docs/assets/conversation_app_arch.svg` may want a
  new box for the local OpenVINO backend. Regenerate from `scheme.mmd`, per `AGENTS.md` — don't
  hand-edit the SVG.
- If picking a different default LLM than `OpenVINO/Qwen3-8B-int4-ov`, re-check tool-call
  wrapper-tag reliability (bug #4) and NPU compilation success (bug #5) for that specific model —
  both were observed to be model-specific, not general OpenVINO/NPU properties.
- Visually verify the Phase 9 chat text input in an actual browser (see Phase 9 section) — only
  verified over the raw `/rpc` wire this session, browser automation wasn't available.
- The full tool-instructions system prompt (all ~16 default tools) noticeably slows small local
  LLMs — a fresh `OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov` on CPU took ~60s to answer a trivial
  greeting with the full toolset in context, vs ~9s with no tools. Not investigated further this
  session (out of scope), but worth knowing if a response feels "stuck": it may just be slow, not
  broken — give a small local model at least a couple of minutes before assuming it hung.
