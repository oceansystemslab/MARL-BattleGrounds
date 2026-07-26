import {
  acquireCapabilityToken,
  acquireClientId,
  DebuggerApiError,
  extractFrame,
  extractNotice,
  getCurrentFrame,
  postCommand,
} from "./api.js";
import { CombatChoreographer, ConsumedTransitionLedger } from "./choreography.js";
import { SvgChoreographyPainter } from "./choreography-painter.js";
import { isSubmissionCommand } from "./choreography-plan.js";
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
  motionPauseButton: requiredElement("motion-pause-button"),
  motionSkipButton: requiredElement("motion-skip-button"),
  motionStatus: requiredElement("motion-status"),
  motionRateButtons: /** @type {NodeListOf<HTMLButtonElement>} */ (
    document.querySelectorAll("[data-motion-rate]")
  ),
  notice: requiredElement("notice"),
  scenarioDescription: requiredElement("scenario-description"),
  battlefieldShell: requiredElement("battlefield-shell"),
  battlefield: requiredElement("battlefield"),
  battlefieldEmpty: requiredElement("battlefield-empty"),
  commandDeck: document.querySelector(".command-deck"),
  roster: requiredElement("roster"),
  rosterCount: requiredElement("roster-count"),
  selectionCard: requiredElement("selection-card"),
  pendingHeading: requiredElement("pending-heading"),
  pendingCount: requiredElement("pending-count"),
  pendingScope: requiredElement("pending-scope"),
  pendingCard: requiredElement("pending-card"),
  stayButton: requiredElement("stay-button"),
  commandTargetSelect: requiredElement("command-target-select"),
  noCombatButton: requiredElement("no-combat-button"),
  basicButton: requiredElement("basic-button"),
  ultimateButton: requiredElement("ultimate-button"),
  submitTurnButton: requiredElement("submit-turn-button"),
  advanceScriptButton: requiredElement("advance-script-button"),
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

const reducedMotionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
/** @type {Storage | null} */
let presentationStorage = null;
try {
  presentationStorage = window.sessionStorage;
} catch {
  // Storage is an optional presentation convenience, never an authority input.
}

const choreographer = new CombatChoreographer({
  painter: new SvgChoreographyPainter(),
  ledger: new ConsumedTransitionLedger({ storage: presentationStorage }),
  motionMode: reducedMotionPreference.matches ? "reduced" : "normal",
  onStateChange: () => {
    renderMotionControls();
    renderCommandAvailability();
  },
});

const panels = new DebuggerPanels({
  roster: elements.roster,
  rosterCount: elements.rosterCount,
  selectionCard: elements.selectionCard,
  pendingHeading: elements.pendingHeading,
  pendingCount: elements.pendingCount,
  pendingScope: elements.pendingScope,
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

const DRAFT_KEYS = new Set([
  "w",
  "a",
  "s",
  "d",
  "q",
  "e",
  "z",
  "c",
  "x",
  "arrowup",
  "arrowdown",
  "arrowleft",
  "arrowright",
  "0",
  "1",
  "2",
]);

/**
 * @param {Record<string, unknown>} command
 * @returns {"draft" | "interactive-submit" | "script-advance" | null}
 */
function commandRole(command) {
  if (command.command_type !== "keyboard" || typeof command.key !== "string") {
    return null;
  }
  const key = command.key.toLowerCase();
  if (DRAFT_KEYS.has(key)) {
    return "draft";
  }
  if (key === "enter" || key === "return" || key === " " || key === "spacebar") {
    return "interactive-submit";
  }
  return key === "n" ? "script-advance" : null;
}

/**
 * Browser-side mode filtering is advisory UX. Python repeats the same
 * authorization decision before it can mutate the session.
 *
 * @param {Record<string, unknown>} command
 * @param {Record<string, any> | null} frame
 */
function modeAvailability(command, frame) {
  const role = commandRole(command);
  if (!role) {
    return { allowed: true, notice: null };
  }
  const scenario = scenarioRecord(frame);
  const scripted = scenario.mode === "scripted";
  if (isTerminal(frame)) {
    return {
      allowed: false,
      notice: "The episode is terminal; reset or switch scenario to continue.",
    };
  }
  if (scripted && role !== "script-advance") {
    return {
      allowed: false,
      notice:
        "Scripted playback is inspection-only. Press N to advance the registered frame.",
    };
  }
  if (!scripted && role === "script-advance") {
    return {
      allowed: false,
      notice: "N advances scripted playback only.",
    };
  }
  if (scripted && role === "script-advance" && scenario.script_complete === true) {
    return {
      allowed: false,
      notice: "The scripted trajectory is complete; reset or switch scenario.",
    };
  }
  return { allowed: true, notice: null };
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
  renderCommandAvailability();
}

/**
 * @param {HTMLButtonElement} button
 * @param {boolean} selected
 */
function setDraftSelection(button, selected) {
  button.setAttribute("aria-pressed", String(selected));
  button.dataset.selected = String(selected);
}

/**
 * Keep masked controls focusable so their explanation remains inspectable.
 * Python repeats the exact mask check before any pending row can change.
 *
 * @param {HTMLButtonElement} button
 * @param {boolean} available
 * @param {string} explanation
 */
function setAuthoritativeAvailability(button, available, explanation) {
  button.setAttribute("aria-disabled", String(!available));
  button.dataset.authoritativeAvailable = String(available);
  button.dataset.tooltipKind = "legality";
  button.dataset.tooltipText = explanation;
}

/**
 * @param {Record<string, any>} hud
 * @param {number} targetAction
 * @returns {Record<string, any> | null}
 */
function candidateLegality(hud, targetAction) {
  return (
    asArray(hud.candidate_legalities).find(
      (candidate) =>
        isRecord(candidate) && Number(candidate.target_action) === targetAction,
    ) ?? null
  );
}

/**
 * Rebuild target choices from the current controlled actor's authorized rows.
 * The select is a view over Python facts, never a client-owned pending model.
 *
 * @param {Record<string, any>} hud
 * @param {Record<string, any>} pending
 */
function renderCommandTargets(hud, pending) {
  const candidates = asArray(hud.candidate_legalities).filter(isRecord);
  const selectedSlot = isRecord(pending.target) ? pending.target.global_slot : null;
  const fragment = document.createDocumentFragment();
  for (const candidate of candidates) {
    const target = isRecord(candidate.target) ? candidate.target : {};
    const option = document.createElement("option");
    const basic = candidate.basic_available === true ? "B ✓" : "B ×";
    const ultimate = candidate.ultimate_available === true ? "U ✓" : "U ×";
    if (target.disclosure === "public" && Number.isInteger(target.global_slot)) {
      option.value = String(target.global_slot);
      option.textContent = `id_${target.global_slot} · ${basic} · ${ultimate}`;
      option.selected = Number(target.global_slot) === Number(selectedSlot);
    } else if (target.disclosure === "target_none") {
      option.value = "";
      option.textContent = `No target · ${basic} · ${ultimate}`;
      option.selected = selectedSlot === null || selectedSlot === undefined;
    } else {
      continue;
    }
    fragment.append(option);
  }
  if (fragment.childNodes.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No authorized targets";
    fragment.append(option);
  }
  elements.commandTargetSelect.replaceChildren(fragment);
}

/**
 * @param {Record<string, any>} hud
 */
function renderDraftState(hud) {
  const pending = isRecord(hud.pending_action) ? hud.pending_action : {};
  const pendingMove = Number(pending.move_action);
  const movementRows = new Map(
    asArray(hud.movement_legalities)
      .filter(isRecord)
      .map((row) => [Number(row.move_action), row.available]),
  );
  const movementButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
    elements.commandDeck?.querySelectorAll("button[data-move-action]") ?? []
  );
  for (const button of movementButtons) {
    const moveAction = Number(button.dataset.moveAction);
    const available = movementRows.get(moveAction);
    setDraftSelection(button, moveAction === pendingMove);
    setAuthoritativeAvailability(
      button,
      available === true,
      available === true
        ? `${button.getAttribute("aria-label") ?? "Movement"} is legal in the current authoritative movement mask.`
        : `${button.getAttribute("aria-label") ?? "Movement"} is unavailable in the current authoritative movement mask.`,
    );
  }

  renderCommandTargets(hud, pending);
  const targetAction = Number.isInteger(pending.target_action)
    ? Number(pending.target_action)
    : 0;
  const candidate = candidateLegality(hud, targetAction);
  const basicAvailable = candidate?.basic_available === true;
  const ultimateAvailable = candidate?.ultimate_available === true;
  const noCombatSelected =
    pending.armed_lane === null || (pending.armed_lane === 0 && targetAction === 0);
  setDraftSelection(elements.noCombatButton, noCombatSelected);
  setDraftSelection(elements.basicButton, pending.armed_lane === 0 && targetAction > 0);
  setDraftSelection(elements.ultimateButton, pending.armed_lane === 1);
  setAuthoritativeAvailability(
    elements.noCombatButton,
    true,
    "No combat is always a valid staged choice; movement can still be submitted.",
  );
  setAuthoritativeAvailability(
    elements.basicButton,
    basicAvailable,
    basicAvailable
      ? "Basic is legal for the currently staged target."
      : "Basic is unavailable for the currently staged target.",
  );
  setAuthoritativeAvailability(
    elements.ultimateButton,
    ultimateAvailable,
    ultimateAvailable
      ? "Ultimate is legal for the currently staged target."
      : "Ultimate is unavailable for the currently staged target.",
  );
}

function renderCommandAvailability() {
  const disabled =
    state.busy ||
    !state.frame ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline;
  const presentation = choreographer.snapshot();
  const scenario = scenarioRecord(state.frame);
  const scripted = scenario.mode === "scripted";
  const hud = isRecord(state.frame?.hud) ? state.frame.hud : {};
  renderDraftState(hud);
  elements.commandTargetSelect.disabled =
    disabled || scripted || isTerminal(state.frame);
  elements.submitTurnButton.textContent = scripted
    ? "Manual submit unavailable"
    : hud.pending_submission_scope === "controlled_actor"
      ? "Submit controlled actor"
      : "Submit joint turn";
  elements.advanceScriptButton.textContent =
    scripted && scenario.script_complete === true
      ? "Script complete"
      : "Advance scripted frame";
  if (elements.commandDeck) {
    const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
      elements.commandDeck.querySelectorAll("button[data-key]")
    );
    for (const button of buttons) {
      const command = keyboardCommand(button.dataset.key ?? "", {
        shiftKey: button.dataset.shift === "true",
      });
      const mode = modeAvailability(command, state.frame);
      button.disabled =
        disabled ||
        !mode.allowed ||
        (presentation.submissionBlocked && isSubmissionCommand(command));
    }
  }
}

function togglePresentationPause() {
  const presentation = choreographer.snapshot();
  if (
    state.shuttingDown ||
    presentation.motionMode === "off" ||
    presentation.animationCount === 0
  ) {
    return;
  }
  choreographer.togglePaused();
}

function renderMotionControls() {
  const presentation = choreographer.snapshot();
  const hasActiveClock = presentation.animationCount > 0;
  const presentationDisabled = state.shuttingDown;

  document.documentElement.dataset.motionMode = presentation.motionMode;
  document.documentElement.dataset.motionPaused = String(presentation.paused);
  document.documentElement.dataset.motionRate = String(presentation.playbackRate);
  document.documentElement.dataset.submissionBlocked = String(
    presentation.submissionBlocked,
  );

  elements.motionPauseButton.disabled =
    presentationDisabled || presentation.motionMode === "off" || !hasActiveClock;
  elements.motionPauseButton.setAttribute("aria-pressed", String(presentation.paused));
  elements.motionPauseButton.textContent = presentation.paused ? "Resume" : "Pause";
  elements.motionSkipButton.disabled =
    presentationDisabled || (!hasActiveClock && !presentation.submissionBlocked);

  for (const button of elements.motionRateButtons) {
    const value = button.dataset.motionRate;
    const selected =
      value === "off"
        ? presentation.motionMode === "off"
        : presentation.motionMode !== "off" &&
          Number(value) === presentation.playbackRate;
    button.disabled = presentationDisabled;
    button.setAttribute("aria-pressed", String(selected));
  }

  if (presentation.motionMode === "off") {
    elements.motionStatus.textContent = "Motion off";
    return;
  }
  const prefix =
    presentation.motionMode === "reduced"
      ? "Reduced motion"
      : presentation.paused
        ? "Paused"
        : presentation.submissionBlocked
          ? "Explaining"
          : "Motion";
  elements.motionStatus.textContent = `${prefix} ${presentation.playbackRate}×`;
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
  try {
    choreographer.presentFrame(state.frame, battlefieldRenderer.choreographySurface());
  } catch (error) {
    choreographer.clear("presentation_error");
    setNotice(
      error instanceof Error
        ? `Combat presentation failed: ${error.message} The authoritative frame remains available.`
        : "Combat presentation failed. The authoritative frame remains available.",
      "error",
    );
    renderConnection();
  }
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
    try {
      choreographer.reproject(state.frame, battlefieldRenderer.choreographySurface());
    } catch (error) {
      choreographer.clear("resize_projection_error");
      setNotice(
        error instanceof Error
          ? `Combat presentation resize failed: ${error.message}`
          : "Combat presentation resize failed.",
        "error",
      );
      renderConnection();
    }
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
  const mode = modeAvailability(command, state.frame);
  if (!mode.allowed) {
    setNotice(
      mode.notice ?? "That command is unavailable in the current mode.",
      "warning",
    );
    renderConnection();
    renderCommandAvailability();
    return;
  }
  if (choreographer.snapshot().submissionBlocked && isSubmissionCommand(command)) {
    setNotice(
      "The current transition is still being explained. Wait briefly or choose Skip; no submission was sent.",
      "warning",
    );
    renderConnection();
    renderSessionToolbar();
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
  onPresentationKey: (command) => {
    if (command === "toggle-pause") {
      togglePresentationPause();
    }
  },
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

elements.commandTargetSelect.addEventListener("change", () => {
  const value = elements.commandTargetSelect.value;
  if (value === "") {
    dispatchCommand(keyboardCommand("Escape"));
    return;
  }
  const globalSlot = Number(value);
  if (!Number.isInteger(globalSlot)) {
    return;
  }
  dispatchCommand({
    command_type: "roster_selection",
    role: "target",
    global_slot: globalSlot,
  });
});

elements.exitButton.addEventListener("click", () => {
  dispatchCommand({ command_type: "exit" });
});

elements.reconnectButton.addEventListener("click", loadCurrentFrame);

elements.helpButton.addEventListener("click", () => {
  elements.helpDialog.showModal();
});

elements.motionPauseButton.addEventListener("click", () => {
  togglePresentationPause();
});

elements.motionSkipButton.addEventListener("click", () => {
  choreographer.skip();
});

for (const button of elements.motionRateButtons) {
  button.addEventListener("click", () => {
    const value = button.dataset.motionRate;
    if (value === "off") {
      choreographer.setMotionMode("off");
      return;
    }
    const rate = Number(value);
    if (!Number.isFinite(rate) || rate <= 0) {
      return;
    }
    if (choreographer.snapshot().motionMode === "off") {
      choreographer.setMotionMode("normal");
    }
    choreographer.setPlaybackRate(rate);
  });
}

if (elements.commandDeck) {
  const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
    elements.commandDeck.querySelectorAll("button[data-key]")
  );
  for (const button of buttons) {
    button.addEventListener("click", () => {
      if (button.getAttribute("aria-disabled") === "true") {
        setNotice(
          button.dataset.tooltipText ??
            "That pending choice is unavailable in the current authoritative mask.",
          "warning",
        );
        renderConnection();
        return;
      }
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
