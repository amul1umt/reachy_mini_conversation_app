/** Settings view: Hugging Face connection, voice, and runtime status. */

import {
  applyVoice,
  describeError,
  getCurrentVoice,
  getStatus,
  listVoices,
  saveBackendConfig,
  subscribe,
  untilReady,
  getOpenvinoStatus,
  getOpenvinoCatalog,
  getOpenvinoDevices,
  stopOpenvinoBackend,
  startOpenvinoBackend,
  downloadOpenvinoModel,
} from "../api.js";
import { h } from "../ui.js";

const HF_CONNECTION_MODES = Object.freeze({
  DEPLOYED: "deployed",
  LOCAL: "local",
});

const DEFAULT_HF_HOST = "localhost";
const DEFAULT_HF_PORT = 8765;

const HF_MODE_HINTS = Object.freeze({
  [HF_CONNECTION_MODES.DEPLOYED]: "Uses the hosted Hugging Face backend. No API key required.",
  [HF_CONNECTION_MODES.LOCAL]: "Connects directly to the host and port below.",
});

const OPENVINO_ROLES = Object.freeze([
  { role: "stt", label: "Speech-to-text", modelKey: "stt_model", deviceKey: "stt_device" },
  { role: "llm", label: "LLM (+ vision)", modelKey: "llm_model", deviceKey: "llm_device" },
  { role: "tts", label: "Text-to-speech", modelKey: "tts_model", deviceKey: "tts_device" },
]);

export async function mountSettingsView({ outlet, signal }) {
  const connectionSection = buildConnectionSection({
    onSaved: () =>
      Promise.all([
        refreshStatus({ statusSection, connectionSection, signal }),
        refreshVoices({ voiceSection, signal }),
      ]),
  });
  const voiceSection = buildVoiceSection();
  const statusSection = buildStatusSection();
  const openvinoSection = buildOpenvinoSection();

  const view = h(
    "section",
    { class: "view view--settings" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Settings"),
      h("p", { class: "view-subtitle" }, "Connection, voice, and runtime state for Reachy Mini.")
    ),
    connectionSection.element,
    voiceSection.element,
    statusSection.element,
    openvinoSection.element
  );
  outlet.replaceChildren(view);

  const unsubscribeProgress = subscribe("openvino.progress", (params) => {
    openvinoSection.setProgress(params?.role, params?.progress);
  });
  signal.addEventListener("abort", unsubscribeProgress, { once: true });

  await Promise.all([
    refreshStatus({ statusSection, connectionSection, signal }),
    refreshVoices({ voiceSection, signal }),
    refreshOpenvinoSection({ openvinoSection, signal }),
  ]);
}

function buildConnectionSection({ onSaved } = {}) {
  const hfModeSelect = h(
    "select",
    { class: "settings-select", name: "hf_mode" },
    h("option", { value: HF_CONNECTION_MODES.DEPLOYED }, "Hosted"),
    h("option", { value: HF_CONNECTION_MODES.LOCAL }, "Local")
  );
  const hfHostInput = h("input", {
    type: "text",
    name: "hf_host",
    autocomplete: "off",
    placeholder: DEFAULT_HF_HOST,
    value: DEFAULT_HF_HOST,
    class: "settings-input",
  });
  const hfPortInput = h("input", {
    type: "number",
    name: "hf_port",
    min: "1",
    max: "65535",
    step: "1",
    inputmode: "numeric",
    value: String(DEFAULT_HF_PORT),
    class: "settings-input",
  });
  const hfLocalFields = h(
    "div",
    { class: "settings-field-row", "data-role": "hf-local-fields" },
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "Host/IP"),
      hfHostInput
    ),
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "Port"),
      hfPortInput
    )
  );
  const hint = h("p", { class: "settings-hint" }, "");
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const submitButton = h("button", { type: "submit", class: "btn btn--primary" }, "Save connection");

  const form = h(
    "form",
    { class: "settings-form" },
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "Hugging Face connection"),
      hfModeSelect
    ),
    hfLocalFields,
    hint,
    h("div", { class: "settings-actions" }, submitButton),
    status
  );

  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Connection"),
    form
  );

  function syncLocalFields() {
    const isLocal = hfModeSelect.value === HF_CONNECTION_MODES.LOCAL;
    hfLocalFields.style.display = isLocal ? "" : "none";
    hfHostInput.disabled = !isLocal;
    hfPortInput.disabled = !isLocal;
    hfHostInput.required = isLocal;
    hfPortInput.required = isLocal;
    hint.textContent = HF_MODE_HINTS[hfModeSelect.value] || "";
  }

  hfModeSelect.addEventListener("change", syncLocalFields);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitButton.disabled) return;
    submitButton.disabled = true;
    hfModeSelect.disabled = true;
    hfHostInput.disabled = true;
    hfPortInput.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.classList.remove("is-error");
    status.textContent = "Saving…";
    try {
      const payload = { hf_mode: hfModeSelect.value };
      if (hfModeSelect.value === HF_CONNECTION_MODES.LOCAL) {
        payload.hf_host = hfHostInput.value.trim();
        if (hfPortInput.value) {
          payload.hf_port = Number.parseInt(hfPortInput.value, 10);
        }
      }
      const result = await saveBackendConfig(payload);
      status.textContent =
        result?.message || (result?.requires_restart ? "Saved. Restart the app to apply." : "Saved.");
      await onSaved?.();
    } catch (error) {
      status.textContent = `Failed to save: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = false;
      hfModeSelect.disabled = false;
      syncLocalFields();
      form.removeAttribute("aria-busy");
    }
  });

  syncLocalFields();

  return {
    element,
    syncFromStatus(payload) {
      if (Object.values(HF_CONNECTION_MODES).includes(payload?.hf_connection_mode)) {
        hfModeSelect.value = payload.hf_connection_mode;
      }
      if (payload?.hf_direct_host) {
        hfHostInput.value = payload.hf_direct_host;
      }
      if (payload?.hf_direct_port != null) {
        hfPortInput.value = String(payload.hf_direct_port);
      }
      syncLocalFields();
    },
  };
}

function buildVoiceSection() {
  const select = h(
    "select",
    { class: "settings-select", name: "voice", disabled: "disabled" },
    h("option", { value: "" }, "Loading voices…")
  );
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const submitButton = h(
    "button",
    { type: "submit", class: "btn btn--primary", disabled: "disabled" },
    "Apply voice"
  );
  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Voice"), select),
    h("div", { class: "settings-actions" }, submitButton),
    status
  );

  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Voice"),
    form
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitButton.disabled || !select.value) return;
    submitButton.disabled = true;
    select.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.classList.remove("is-error");
    status.textContent = "Applying…";
    try {
      const result = await applyVoice(select.value);
      status.textContent = result?.status || "Voice applied.";
    } catch (error) {
      status.textContent = `Failed to apply: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = !select.value;
      select.disabled = !select.value;
      form.removeAttribute("aria-busy");
    }
  });

  return {
    element,
    setOptions(voices, current) {
      select.replaceChildren();
      if (!voices.length) {
        select.appendChild(h("option", { value: "" }, "No voices available"));
        select.disabled = true;
        submitButton.disabled = true;
        status.textContent = "Voices are unavailable right now.";
        return;
      }
      for (const v of voices) {
        const opt = h("option", { value: v }, v);
        if (v === current) opt.selected = true;
        select.appendChild(opt);
      }
      select.disabled = false;
      submitButton.disabled = false;
      status.textContent = "";
    },
  };
}

function buildStatusSection() {
  const list = h(
    "dl",
    { class: "settings-status-grid" },
    statusRow("Backend", "Loading…")
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Current state"),
    list
  );

  return {
    element,
    render(payload) {
      list.replaceChildren();
      list.appendChild(statusRow("HF connection", formatHfMode(payload.hf_connection_mode)));
      if (payload.hf_connection_mode === HF_CONNECTION_MODES.LOCAL) {
        list.appendChild(statusRow("HF target", formatHfTarget(payload)));
      }
      list.appendChild(
        statusRow(
          "Configuration",
          payload.has_hf_connection ? "Ready" : "Missing",
          payload.has_hf_connection ? "ok" : "warn"
        )
      );
      const backendState = payload.backend_connected
        ? "connected"
        : payload.backend_connection_state || "not_started";
      const backendLabels = {
        connected: "Connected",
        connecting: "Connecting…",
        disconnected: "Disconnected",
        not_started: "Not started",
        restart_required: "Restart required",
        waiting_for_config: "Waiting for configuration",
      };
      list.appendChild(
        statusRow(
          "Backend",
          backendLabels[backendState] || "Unavailable",
          backendState === "connected" ? "ok" : backendState === "not_started" ? undefined : "warn"
        )
      );
      if (payload.backend_error) {
        list.appendChild(statusRow("Backend error", payload.backend_error, "warn"));
      }
      if (payload.requires_restart) {
        list.appendChild(statusRow("Restart", "Required to apply changes", "warn"));
      }
    },
    renderUnavailable(error) {
      list.replaceChildren(statusRow("Backend", `Unavailable: ${describeError(error)}`, "warn"));
    },
  };
}

function buildOpenvinoRoleRow({ role, label }) {
  const modelSelect = h(
    "select",
    { class: "settings-select", name: `openvino_${role}_model`, disabled: "disabled" },
    h("option", { value: "" }, "Loading models…")
  );
  const deviceSelect = h(
    "select",
    { class: "settings-select", name: `openvino_${role}_device` },
    h("option", { value: "AUTO" }, "AUTO (recommended)")
  );
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const downloadButton = h(
    "button",
    { type: "button", class: "btn btn--secondary", disabled: "disabled" },
    "Download"
  );

  // Populated by setModels(); for the "llm" row this mixes text-only and
  // vision-capable ("vlm") catalog entries in one dropdown, so each option's
  // actual role has to be tracked separately from the row's nominal role.
  let modelRoles = new Map();

  downloadButton.addEventListener("click", async () => {
    if (!modelSelect.value) return;
    downloadButton.disabled = true;
    status.classList.remove("is-error");
    status.textContent = "Downloading…";
    try {
      const modelRole = modelRoles.get(modelSelect.value) || role;
      const result = await downloadOpenvinoModel(modelRole, modelSelect.value, deviceSelect.value);
      status.textContent = `Ready: ${result.repo_id}`;
    } catch (error) {
      status.textContent = `Failed to download: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      downloadButton.disabled = false;
    }
  });

  const element = h(
    "div",
    { class: "settings-field-row" },
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, `${label} model`),
      modelSelect
    ),
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Device"), deviceSelect),
    h("div", { class: "settings-actions" }, downloadButton),
    status
  );

  return {
    element,
    setModels(models) {
      modelSelect.replaceChildren();
      modelRoles = new Map(models.map((model) => [model.repo_id, model.role || role]));
      if (!models.length) {
        modelSelect.appendChild(h("option", { value: "" }, "No models found"));
        modelSelect.disabled = true;
        downloadButton.disabled = true;
        return;
      }
      for (const model of models) {
        const roleSuffix = model.role && model.role !== role ? ` (${model.role})` : "";
        modelSelect.appendChild(h("option", { value: model.repo_id }, `${model.display_name}${roleSuffix}`));
      }
      modelSelect.disabled = false;
      downloadButton.disabled = false;
    },
    setDevices(devices) {
      deviceSelect.replaceChildren(h("option", { value: "AUTO" }, "AUTO (recommended)"));
      for (const device of devices) {
        deviceSelect.appendChild(h("option", { value: device.id }, `${device.id} — ${device.full_name}`));
      }
    },
    setSelection(model, device) {
      if (model) modelSelect.value = model;
      if (device) deviceSelect.value = device;
    },
    setProgress(progress) {
      status.classList.remove("is-error");
      status.textContent = `Downloading… ${Math.round(progress * 100)}%`;
    },
    getSelection() {
      return {
        model: modelSelect.value || null,
        device: deviceSelect.value || "AUTO",
        role: modelRoles.get(modelSelect.value) || role,
      };
    },
  };
}

const OPENVINO_BACKEND_STATE_LABELS = Object.freeze({
  idle: "Idle",
  downloading: "Downloading models…",
  starting: "Starting…",
  running: "Running",
  error: "Error",
});

function buildOpenvinoSection() {
  const deviceList = h("dl", { class: "settings-status-grid" }, statusRow("Devices", "Loading…"));
  const roleRows = new Map(OPENVINO_ROLES.map((entry) => [entry.role, buildOpenvinoRoleRow(entry)]));

  const backendStatusList = h("dl", { class: "settings-status-grid" }, statusRow("State", "Loading…"));
  const lifecycleStatus = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const startButton = h(
    "button",
    { type: "button", class: "btn btn--primary" },
    "Download & start local backend"
  );
  const stopButton = h(
    "button",
    { type: "button", class: "btn btn--secondary", disabled: "disabled" },
    "Stop local backend"
  );

  startButton.addEventListener("click", async () => {
    startButton.disabled = true;
    lifecycleStatus.classList.remove("is-error");
    lifecycleStatus.textContent = "Starting…";
    try {
      const stt = roleRows.get("stt").getSelection();
      const llm = roleRows.get("llm").getSelection();
      const tts = roleRows.get("tts").getSelection();
      if (!stt.model || !llm.model) {
        throw new Error("Pick a speech-to-text and LLM model first.");
      }
      const payload = {
        stt_model: stt.model,
        stt_device: stt.device,
        llm_model: llm.model,
        llm_device: llm.device,
        llm_role: llm.role,
      };
      if (tts.model) {
        payload.tts_model = tts.model;
        payload.tts_device = tts.device;
      }
      const result = await startOpenvinoBackend(payload);
      lifecycleStatus.textContent = `Running on port ${result.port}.`;
    } catch (error) {
      lifecycleStatus.textContent = `Failed to start: ${describeError(error)}`;
      lifecycleStatus.classList.add("is-error");
    } finally {
      startButton.disabled = false;
    }
  });

  stopButton.addEventListener("click", async () => {
    stopButton.disabled = true;
    lifecycleStatus.classList.remove("is-error");
    lifecycleStatus.textContent = "Stopping…";
    try {
      await stopOpenvinoBackend();
      lifecycleStatus.textContent = "Stopped.";
    } catch (error) {
      lifecycleStatus.textContent = `Failed to stop: ${describeError(error)}`;
      lifecycleStatus.classList.add("is-error");
    } finally {
      stopButton.disabled = false;
    }
  });

  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Local OpenVINO backend"),
    h(
      "p",
      { class: "settings-hint" },
      "Run the voice pipeline locally with OpenVINO-optimized models. Install with " +
        "`uv sync --extra local-openvino` to enable it."
    ),
    deviceList,
    ...OPENVINO_ROLES.map((entry) => roleRows.get(entry.role).element),
    backendStatusList,
    h("div", { class: "settings-actions" }, startButton, stopButton),
    lifecycleStatus
  );

  return {
    element,
    renderDevices({ devices, extra_installed: extraInstalled }) {
      deviceList.replaceChildren();
      if (!extraInstalled) {
        deviceList.appendChild(statusRow("Devices", "local-openvino extra not installed", "warn"));
      } else if (!devices.length) {
        deviceList.appendChild(statusRow("Devices", "No OpenVINO devices detected", "warn"));
      } else {
        for (const device of devices) {
          deviceList.appendChild(statusRow(device.id, device.full_name, "ok"));
        }
      }
      for (const row of roleRows.values()) row.setDevices(devices);
    },
    renderDevicesUnavailable(error) {
      deviceList.replaceChildren(statusRow("Devices", `Unavailable: ${describeError(error)}`, "warn"));
    },
    setCatalog(role, models) {
      roleRows.get(role)?.setModels(models);
    },
    setSelection(selection) {
      for (const { role, modelKey, deviceKey } of OPENVINO_ROLES) {
        roleRows.get(role)?.setSelection(selection?.[modelKey], selection?.[deviceKey]);
      }
    },
    setProgress(role, progress) {
      if (typeof progress === "number") roleRows.get(role)?.setProgress(progress);
    },
    renderBackendStatus({ state, port, error }) {
      backendStatusList.replaceChildren();
      backendStatusList.appendChild(
        statusRow("State", OPENVINO_BACKEND_STATE_LABELS[state] || state, state === "running" ? "ok" : state === "error" ? "warn" : undefined)
      );
      if (port != null) backendStatusList.appendChild(statusRow("Port", String(port)));
      if (error) backendStatusList.appendChild(statusRow("Error", error, "warn"));
      const busy = state === "downloading" || state === "starting";
      startButton.disabled = busy || state === "running";
      stopButton.disabled = busy || state !== "running";
    },
  };
}

function statusRow(label, value, tone) {
  return h(
    "div",
    { class: ["settings-status-row", tone && `is-${tone}`] },
    h("dt", { class: "settings-status-label" }, label),
    h("dd", { class: "settings-status-value" }, value)
  );
}

function formatHfMode(mode) {
  if (mode === HF_CONNECTION_MODES.LOCAL) return "Local";
  if (mode === HF_CONNECTION_MODES.DEPLOYED) return "Hosted";
  return "-";
}

function formatHfTarget(payload) {
  const host = payload?.hf_direct_host;
  const port = payload?.hf_direct_port;
  if (!host) return "-";
  return `${host}:${port || DEFAULT_HF_PORT}`;
}

async function refreshStatus({ statusSection, connectionSection, signal }) {
  try {
    const payload = await untilReady(getStatus, signal);
    if (signal.aborted) return;
    statusSection.render(payload);
    connectionSection.syncFromStatus(payload);
  } catch (error) {
    if (signal.aborted) return;
    statusSection.renderUnavailable(error);
  }
}

async function refreshVoices({ voiceSection, signal }) {
  let voices = [];
  let current = "";
  try {
    voices = await untilReady(listVoices, signal);
  } catch {
    voices = [];
  }
  if (signal.aborted) return;
  try {
    const data = await getCurrentVoice();
    current = data?.voice || "";
  } catch {
    current = "";
  }
  if (signal.aborted) return;
  voiceSection.setOptions(voices, current);
}

async function refreshOpenvinoSection({ openvinoSection, signal }) {
  try {
    const payload = await untilReady(getOpenvinoDevices, signal);
    if (signal.aborted) return;
    openvinoSection.renderDevices(payload);
  } catch (error) {
    if (signal.aborted) return;
    openvinoSection.renderDevicesUnavailable(error);
  }

  await Promise.all(
    OPENVINO_ROLES.map(async ({ role }) => {
      // The "llm" row offers both text-only and vision-capable models in one
      // dropdown, so it merges the "llm" and "vlm" catalogs; other roles only
      // ever query their own.
      const catalogRoles = role === "llm" ? ["llm", "vlm"] : [role];
      try {
        const results = await Promise.all(catalogRoles.map((catalogRole) => getOpenvinoCatalog(catalogRole)));
        if (signal.aborted) return;
        const models = results.flatMap((result, index) =>
          (result.models || []).map((model) => ({ ...model, role: catalogRoles[index] }))
        );
        openvinoSection.setCatalog(role, models);
      } catch {
        if (signal.aborted) return;
        openvinoSection.setCatalog(role, []);
      }
    })
  );

  try {
    const { selection, state, port, error } = await getOpenvinoStatus();
    if (signal.aborted) return;
    openvinoSection.setSelection(selection);
    openvinoSection.renderBackendStatus({ state, port, error });
  } catch {
    // Selection/backend status are convenience defaults; leave the pickers/status at fallback values.
  }
}
