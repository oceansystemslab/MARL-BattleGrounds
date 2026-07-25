import {
  acquireCapabilityToken,
  acquireClientId,
  DebuggerApiError,
  extractFrame,
  extractNotice,
  getCurrentFrame,
  postCommand,
} from "./api.js";
import { bindBattlefieldControls, keyboardCommand } from "./controls.js";
import { DebuggerPanels } from "./panels.js";
import { BattlefieldRenderer } from "./scene.js";

/**
 * @param {string} id
 * @returns {any}
 */
function requiredElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Browser shell is missing required element #${id}.`);
  }
  return element;
}

const elements = {
  connectionStatus: requiredElement("connection-status"),
  audienceBadge: requiredElement("audience-badge"),
  terminalBadge: requiredElement("terminal-badge"),
  scenarioSelect: requiredElement("scenario-select"),
  viewSelect: requiredElement("view-select"),
  presetSelect: requiredElement("preset-select"),
  revisionValue: requiredElement("revision-value"),
  stepValue: requiredElement("step-value"),
  transitionValue: requiredElement("transition-value"),
  reconnectButton: requiredElement("reconnect-button"),
  helpButton: requiredElement("help-button"),
  exitButton: requiredElement("exit-button"),
  resetButton: requiredElement("reset-button"),
  notice: requiredElement("notice"),
  scenarioDescription: requiredElement("scenario-description"),
  battlefieldShell: requiredElement("battlefield-shell"),
  battlefield: requiredElement("battlefield"),
  battlefieldEmpty: requiredElement("battlefield-empty"),
  commandDeck: document.querySelector(".command-deck"),
  roster: requiredElement("roster"),
  rosterCount: requiredElement("roster-count"),
  selectionCard: requiredElement("selection-card"),
  pendingCard: requiredElement("pending-card"),
  acceptedCard: requiredElement("accepted-card"),
  acceptedAnnouncement: requiredElement("accepted-announcement"),
  eventFeed: requiredElement("event-feed"),
  eventCount: requiredElement("event-count"),
  diagnosticsCard: requiredElement("diagnostics-card"),
  helpDialog: requiredElement("help-dialog"),
};

/**
 * @type {{
 *   token: string | null,
 *   clientId: string,
 *   frame: Record<string, any> | null,
 *   busy: boolean,
 *   offline: boolean,
 *   resyncRequired: boolean,
 *   shuttingDown: boolean,
 *   notice: string | null,
 *   noticeLevel: string,
 * }}
 */
const state = {
  token: acquireCapabilityToken(),
  clientId: acquireClientId(),
  frame: null,
  busy: false,
  offline: false,
  resyncRequired: false,
  shuttingDown: false,
  notice: null,
  noticeLevel: "info",
};

const battlefieldRenderer = new BattlefieldRenderer({
  battlefield: elements.battlefield,
  empty: elements.battlefieldEmpty,
});

const panels = new DebuggerPanels({
  roster: elements.roster,
  rosterCount: elements.rosterCount,
  selectionCard: elements.selectionCard,
  pendingCard: elements.pendingCard,
  acceptedCard: elements.acceptedCard,
  acceptedAnnouncement: elements.acceptedAnnouncement,
  eventFeed: elements.eventFeed,
  eventCount: elements.eventCount,
  diagnosticsCard: elements.diagnosticsCard,
  onCommand: dispatchCommand,
});

let lastBattlefieldSizeKey = "";
/** @type {number | null} */
let pendingResizeFrame = null;

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * @param {unknown} value
 * @returns {any[]}
 */
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * @param {unknown} value
 * @param {number} fallback
 */
function integer(value, fallback = 0) {
  return Number.isInteger(value) ? value : fallback;
}

/**
 * @param {unknown} frame
 * @returns {Record<string, any> | null}
 */
function frameScene(frame) {
  if (!isRecord(frame)) {
    return null;
  }
  return isRecord(frame.scene)
    ? frame.scene
    : isRecord(frame.battlefield_scene)
      ? frame.battlefield_scene
      : null;
}

/**
 * @param {unknown} frame
 * @returns {Record<string, any>}
 */
function scenarioRecord(frame) {
  return isRecord(frame) && isRecord(frame.scenario) ? frame.scenario : {};
}

/**
 * @param {unknown} frame
 */
function scenarioName(frame) {
  const scenario = scenarioRecord(frame);
  const record = isRecord(frame) ? frame : {};
  return typeof scenario.name === "string"
    ? scenario.name
    : typeof record.scenario_name === "string"
      ? record.scenario_name
      : "";
}

/**
 * @param {unknown} frame
 */
function scenarioDescription(frame) {
  const scenario = scenarioRecord(frame);
  const record = isRecord(frame) ? frame : {};
  return typeof scenario.description === "string"
    ? scenario.description
    : typeof record.scenario_description === "string"
      ? record.scenario_description
      : "Authoritative debugger frame received.";
}

/**
 * @param {unknown} frame
 */
function isTerminal(frame) {
  const record = isRecord(frame) ? frame : {};
  if (typeof record.terminal === "boolean") {
    return record.terminal;
  }
  if (isRecord(record.terminal)) {
    return Boolean(record.terminal.terminated || record.terminal.truncated);
  }
  return Boolean(record.terminated || record.truncated);
}

function currentRevision() {
  return integer(state.frame?.revision, 0);
}

/**
 * @param {string} message
 * @param {string} level
 */
function setNotice(message, level = "info") {
  state.notice = message;
  state.noticeLevel = level;
}

/**
 * @param {unknown} value
 */
function humanize(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

/**
 * @template {keyof HTMLElementTagNameMap} K
 * @param {K} tagName
 * @param {string | null} className
 * @param {string | null} text
 * @returns {HTMLElementTagNameMap[K]}
 */
function htmlElement(tagName, className = null, text = null) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== null) {
    element.textContent = text;
  }
  return element;
}

function renderConnection() {
  let label = "Online";
  let status = "online";
  if (state.shuttingDown) {
    label = "Shutting down";
    status = "busy";
  } else if (state.busy) {
    label = "Command in flight";
    status = "busy";
  } else if (state.resyncRequired) {
    label = "Resync required";
    status = "offline";
  } else if (state.offline) {
    label = "Offline";
    status = "offline";
  } else if (!state.frame) {
    label = "Connecting";
    status = "loading";
  }
  elements.connectionStatus.textContent = label;
  elements.connectionStatus.dataset.state = status;

  elements.notice.hidden = !state.notice;
  elements.notice.textContent = state.notice ?? "";
  elements.notice.dataset.level = state.noticeLevel;
  elements.battlefieldShell.setAttribute("aria-busy", String(state.busy));
}

function renderSessionToolbar() {
  const frame = state.frame;
  const scene = frameScene(frame);
  const disabled =
    state.busy || !frame || state.shuttingDown || state.resyncRequired || state.offline;

  elements.revisionValue.textContent = frame ? String(frame.revision ?? "—") : "—";
  elements.stepValue.textContent = frame
    ? String(frame.simulator_step ?? frame.step ?? "—")
    : "—";
  elements.transitionValue.textContent = frame
    ? String(frame.transition_id ?? "—")
    : "—";

  elements.audienceBadge.textContent =
    typeof scene?.audience_badge === "string"
      ? scene.audience_badge
      : typeof frame?.view_mode === "string"
        ? humanize(frame.view_mode)
        : "View unavailable";
  elements.audienceBadge.dataset.audience =
    scene?.audience === "researcher" || scene?.audience === "agent_pov"
      ? scene.audience
      : "unavailable";
  document.documentElement.dataset.audience = elements.audienceBadge.dataset.audience;
  document.documentElement.dataset.preset =
    frame?.preset === "presentation" ||
    frame?.preset === "analysis" ||
    frame?.preset === "debug"
      ? frame.preset
      : "unavailable";

  const terminal = isTerminal(frame);
  elements.terminalBadge.hidden = !terminal;
  elements.terminalBadge.textContent = terminal
    ? "Terminal · submissions blocked by Python"
    : "Terminal";

  elements.scenarioDescription.textContent = frame
    ? scenarioDescription(frame)
    : "Waiting for the Python debugger service.";

  renderScenarioOptions(frame);
  const viewMode = frame?.view_mode === "agent_pov" ? "pov" : frame?.view_mode;
  if (typeof viewMode === "string") {
    elements.viewSelect.value = viewMode;
  }
  if (typeof frame?.preset === "string") {
    elements.presetSelect.value = frame.preset;
  }
  elements.scenarioSelect.disabled = disabled;
  elements.viewSelect.disabled = disabled;
  elements.presetSelect.disabled = disabled;
  elements.reconnectButton.disabled = state.busy || state.shuttingDown;
  elements.exitButton.disabled = disabled;
  elements.resetButton.disabled = disabled;
  if (elements.commandDeck) {
    const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
      elements.commandDeck.querySelectorAll("button[data-key]")
    );
    for (const button of buttons) {
      button.disabled = disabled;
    }
  }
}

/**
 * @param {Record<string, any> | null} frame
 */
function renderScenarioOptions(frame) {
  const currentName = scenarioName(frame);
  const available = asArray(frame?.available_scenarios);
  const normalized = available
    .map((entry) => {
      if (typeof entry === "string") {
        return { name: entry, title: entry };
      }
      if (isRecord(entry) && typeof entry.name === "string") {
        return {
          name: entry.name,
          title:
            typeof entry.title === "string" && entry.title.trim()
              ? entry.title
              : entry.name,
        };
      }
      return null;
    })
    .filter((entry) => entry !== null);

  if (currentName && !normalized.some((entry) => entry.name === currentName)) {
    normalized.unshift({ name: currentName, title: currentName });
  }

  elements.scenarioSelect.replaceChildren();
  if (normalized.length === 0) {
    const option = htmlElement("option", null, frame ? "No scenarios" : "Loading…");
    option.value = "";
    elements.scenarioSelect.append(option);
    return;
  }
  for (const entry of normalized) {
    const option = htmlElement("option", null, entry.title);
    option.value = entry.name;
    option.selected = entry.name === currentName;
    elements.scenarioSelect.append(option);
  }
}

function render() {
  renderConnection();
  renderSessionToolbar();
  battlefieldRenderer.render(state.frame, { offline: state.offline });
  lastBattlefieldSizeKey = battlefieldSizeKey();
  panels.render(state.frame, {
    busy: state.busy,
    shuttingDown: state.shuttingDown,
    resyncRequired: state.resyncRequired,
    offline: state.offline,
  });
}

function battlefieldSizeKey() {
  return `${Math.round(elements.battlefield.clientWidth)}x${Math.round(elements.battlefield.clientHeight)}`;
}

function scheduleBattlefieldResize() {
  if (pendingResizeFrame !== null) {
    return;
  }
  pendingResizeFrame = window.requestAnimationFrame(() => {
    pendingResizeFrame = null;
    const sizeKey = battlefieldSizeKey();
    if (sizeKey === lastBattlefieldSizeKey) {
      return;
    }
    lastBattlefieldSizeKey = sizeKey;
    battlefieldRenderer.render(state.frame, { offline: state.offline });
  });
}

/**
 * @param {Record<string, unknown>} command
 */
function commandRequest(command) {
  return {
    schema_version: 1,
    client_id: state.clientId,
    command_id: window.crypto.randomUUID(),
    base_revision: currentRevision(),
    command,
  };
}

/**
 * @param {Record<string, unknown>} command
 */
async function dispatchCommand(command) {
  if (state.busy || state.shuttingDown) {
    setNotice("A command is already in flight; no second command was sent.", "warning");
    renderConnection();
    return;
  }
  if (state.resyncRequired || state.offline) {
    setNotice(
      "Reconnect to install the latest authoritative frame before sending another command.",
      "warning",
    );
    renderConnection();
    return;
  }
  if (!state.frame) {
    setNotice(
      "No authoritative frame is available. Reconnect before sending commands.",
      "error",
    );
    renderConnection();
    return;
  }

  state.busy = true;
  state.offline = false;
  setNotice("Waiting for the authoritative Python response…", "info");
  render();

  try {
    const payload = await postCommand(state.token, commandRequest(command));
    const frame = extractFrame(payload);
    if (!frame) {
      throw new DebuggerApiError("Command response did not contain a debugger frame.");
    }
    state.frame = frame;
    state.offline = false;
    const notice = extractNotice(payload);
    setNotice(
      notice ??
        (payload?.result === "duplicate"
          ? "Duplicate command recognized; it was not applied again."
          : "Authoritative frame updated."),
      payload?.result === "duplicate" ? "warning" : "success",
    );
    if (command.command_type === "exit") {
      state.shuttingDown = true;
      setNotice("Exit accepted. The local debugger server is shutting down.", "info");
    }
  } catch (error) {
    if (error instanceof DebuggerApiError && error.status === 409) {
      const latest = extractFrame(error.payload);
      if (latest) {
        state.frame = latest;
      }
      state.offline = false;
      state.resyncRequired = false;
      const errorCode = isRecord(error.payload) ? error.payload.error_code : null;
      setNotice(
        errorCode === "command_id_conflict"
          ? "The service rejected a command-ID conflict. The latest frame was installed and nothing was retried."
          : "This tab was stale. The latest frame was installed; the command was not retried.",
        "warning",
      );
    } else {
      const status = error instanceof DebuggerApiError ? error.status : 0;
      state.offline = status === 0 || status === 401 || status === 403;
      state.resyncRequired = true;
      setNotice(
        status === 401 || status === 403
          ? "Debugger capability is invalid. Reopen the exact URL printed by the Python launcher."
          : error instanceof Error
            ? `${error.message} Reconnect before sending another command.`
            : "Debugger command failed. Reconnect before sending another command.",
        "error",
      );
    }
  } finally {
    state.busy = false;
    render();
  }
}

async function loadCurrentFrame() {
  if (state.busy || state.shuttingDown) {
    return;
  }
  state.busy = true;
  setNotice("Fetching the current authoritative frame…", "info");
  renderConnection();
  try {
    const payload = await getCurrentFrame(state.token);
    const frame = extractFrame(payload);
    if (!frame) {
      throw new DebuggerApiError("Frame response did not contain a debugger frame.");
    }
    state.frame = frame;
    state.offline = false;
    state.resyncRequired = false;
    setNotice(extractNotice(payload) ?? "Connected to the local debugger.", "success");
  } catch (error) {
    const status = error instanceof DebuggerApiError ? error.status : 0;
    state.offline = status === 0 || status === 401 || status === 403;
    state.resyncRequired = true;
    setNotice(
      status === 401 || status === 403
        ? "Debugger capability is invalid. Reopen the exact URL printed by the Python launcher."
        : error instanceof Error
          ? error.message
          : "Could not load debugger frame.",
      "error",
    );
  } finally {
    state.busy = false;
    render();
  }
}

bindBattlefieldControls({
  battlefield: elements.battlefield,
  toWorldPoint: (point) => battlefieldRenderer.toWorldPoint(point),
  onCommand: dispatchCommand,
  onHelp: () => elements.helpDialog.showModal(),
  onReleaseFocus: () => {
    const firstCommand = /** @type {HTMLButtonElement | null} */ (
      elements.commandDeck?.querySelector("button:not([disabled])") ?? null
    );
    const focusTarget = firstCommand ?? elements.helpButton;
    focusTarget.focus({ preventScroll: true });
  },
});

elements.scenarioSelect.addEventListener("change", () => {
  if (!elements.scenarioSelect.value) {
    return;
  }
  dispatchCommand({
    command_type: "scenario_switch",
    scenario_name: elements.scenarioSelect.value,
  });
});

elements.viewSelect.addEventListener("change", () => {
  dispatchCommand({
    command_type: "set_view",
    view_mode: elements.viewSelect.value,
  });
});

elements.presetSelect.addEventListener("change", () => {
  dispatchCommand({
    command_type: "set_preset",
    preset: elements.presetSelect.value,
  });
});

elements.resetButton.addEventListener("click", () => {
  dispatchCommand({ command_type: "reset" });
});

elements.exitButton.addEventListener("click", () => {
  dispatchCommand({ command_type: "exit" });
});

elements.reconnectButton.addEventListener("click", loadCurrentFrame);

elements.helpButton.addEventListener("click", () => {
  elements.helpDialog.showModal();
});

if (elements.commandDeck) {
  const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
    elements.commandDeck.querySelectorAll("button[data-key]")
  );
  for (const button of buttons) {
    button.addEventListener("click", () => {
      dispatchCommand(
        keyboardCommand(button.dataset.key ?? "", {
          shiftKey: button.dataset.shift === "true",
        }),
      );
    });
  }
}

const battlefieldResizeObserver = new ResizeObserver(scheduleBattlefieldResize);
battlefieldResizeObserver.observe(elements.battlefieldShell);

render();
loadCurrentFrame();
