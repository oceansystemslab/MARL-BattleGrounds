import {
  acquireCapabilityToken,
  acquireClientId,
  DebuggerApiError,
  extractFrame,
  extractNotice,
  extractReplayTimeline,
  getCurrentFrame,
  getReplayTimeline,
  postCommand,
  postReplayCommand,
} from "./api.js";
import { CombatChoreographer, ConsumedTransitionLedger } from "./choreography.js";
import { SvgChoreographyPainter } from "./choreography-painter.js";
import { isSubmissionCommand } from "./choreography-plan.js";
import {
  bindBattlefieldControls,
  keyboardCommand,
  targetSelectionCommand,
} from "./controls.js";
import { formatDisplayNumber } from "./display.js";
import {
  liveDebuggerFrameIsScripted,
  liveDebuggerScenarioControlsAvailable,
} from "./frame-normalizer.js";
import { DebuggerPanels } from "./panels.js";
import {
  bindReplayTimelineControls,
  ReplayPlaybackController,
  renderReplayTimelineControls,
  replayCommandRequest,
  validateReplayCommandOutcome,
} from "./replay-controls.js";
import {
  joinReplayFrameAndTimeline,
  validateReplayFrameContinuity,
} from "./replay-frame-normalizer.js";
import { BattlefieldRenderer } from "./scene.js";
import { createTooltipController, registerTooltipOwner } from "./tooltip.js";

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
  scenarioControl: requiredElement("scenario-control"),
  scenarioSelect: requiredElement("scenario-select"),
  viewSelect: requiredElement("view-select"),
  presetSelect: requiredElement("preset-select"),
  movementScaleInput: requiredElement("movement-scale-input"),
  movementScaleValue: requiredElement("movement-scale-value"),
  movementScaleTenthButton: requiredElement("movement-scale-tenth-button"),
  movementScaleDefaultButton: requiredElement("movement-scale-default-button"),
  revisionValue: requiredElement("revision-value"),
  stepValue: requiredElement("step-value"),
  transitionValue: requiredElement("transition-value"),
  replayTimeline: requiredElement("replay-timeline"),
  replayArtifactReference: requiredElement("replay-artifact-reference"),
  replayCompletionBadge: requiredElement("replay-completion-badge"),
  replayProcessingBadge: requiredElement("replay-processing-badge"),
  replayEndReason: requiredElement("replay-end-reason"),
  replayIncomingValue: requiredElement("replay-incoming-value"),
  replayFirstButton: requiredElement("replay-first-button"),
  replayPreviousButton: requiredElement("replay-previous-button"),
  replayPlayPauseButton: requiredElement("replay-play-pause-button"),
  replayNextButton: requiredElement("replay-next-button"),
  replayLastButton: requiredElement("replay-last-button"),
  replayFrameSlider: requiredElement("replay-frame-slider"),
  replayFramePosition: requiredElement("replay-frame-position"),
  replayRangesButton: requiredElement("replay-ranges-button"),
  replayVerbosityButton: requiredElement("replay-verbosity-button"),
  replayClearReferenceButton: requiredElement("replay-clear-reference-button"),
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
  commandControlledActor: requiredElement("command-controlled-actor"),
  roster: requiredElement("roster"),
  rosterCount: requiredElement("roster-count"),
  selectionCard: requiredElement("selection-card"),
  selectionHeading: requiredElement("selection-heading"),
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
  visualTooltip: requiredElement("visual-tooltip"),
  visualTooltipTitle: requiredElement("visual-tooltip-title"),
  visualTooltipDetails: requiredElement("visual-tooltip-details"),
  helpDialog: requiredElement("help-dialog"),
  battlefieldInstructions: requiredElement("battlefield-instructions"),
  liveOnly: /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll("[data-live-only]")
  ),
  replayOnly: /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll("[data-replay-only]")
  ),
};

/**
 * @type {{
 *   token: string | null,
 *   clientId: string,
 *   frame: Record<string, any> | null,
 *   timeline: Record<string, any> | null,
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
  timeline: null,
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
  onStateChange: (presentation) => {
    renderMotionControls();
    renderCommandAvailability();
    syncCompactActiveCombatPriority(presentation);
  },
});

const replayTimelineElements = {
  root: elements.replayTimeline,
  firstButton: elements.replayFirstButton,
  previousButton: elements.replayPreviousButton,
  playPauseButton: elements.replayPlayPauseButton,
  nextButton: elements.replayNextButton,
  lastButton: elements.replayLastButton,
  slider: elements.replayFrameSlider,
  position: elements.replayFramePosition,
};

const replayPlayback = new ReplayPlaybackController({
  request: sendReplayCommand,
  waitForPresentation: () => choreographer.whenSettled(),
  getMotionMode: () => choreographer.snapshot().motionMode,
  onStateChange: (playback) => {
    renderReplayTimelineControls(replayTimelineElements, playback);
  },
  onError: (error) => {
    if (!state.notice || state.noticeLevel !== "error") {
      setNotice(
        error instanceof Error ? error.message : "Replay navigation failed.",
        "error",
      );
      renderConnection();
    }
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
  onCommand: dispatchPanelCommand,
});

const tooltipController = createTooltipController({
  root: document.body,
  tooltip: elements.visualTooltip,
  title: elements.visualTooltipTitle,
  details: elements.visualTooltipDetails,
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

function isReplayMode() {
  return state.frame?.viewer_mode === "replay";
}

function renderViewerBoundary() {
  const replay = isReplayMode();
  document.documentElement.dataset.viewerMode = replay ? "replay" : "live";
  for (const element of elements.liveOnly) {
    element.toggleAttribute("hidden", replay);
  }
  for (const element of elements.replayOnly) {
    element.toggleAttribute("hidden", !replay);
  }
  elements.replayTimeline.toggleAttribute("hidden", !replay);
  elements.battlefield.setAttribute("role", replay ? "img" : "application");
  elements.battlefield.tabIndex = replay ? -1 : 0;
  elements.battlefield.setAttribute(
    "aria-label",
    replay
      ? "Read-only replay battlefield snapshot."
      : "Interactive battlefield. Press Help for keyboard controls.",
  );
  elements.battlefieldInstructions.textContent = replay
    ? "Replay transport changes only the selected recorded frame. The battlefield cannot submit actions or advance the simulator."
    : "The battlefield owns debugger keyboard commands while it has focus. Tab and Shift Tab cycle controlled actors here. Escape clears the target and moves focus to the command deck, or to Help while commands are unavailable. Tab behaves normally in the side panel.";
  elements.selectionHeading.textContent = replay
    ? state.frame?.replay_audience === "researcher"
      ? "Reference"
      : "Replay recipient"
    : "Controlled and selected";
}

function renderReplayMetadata() {
  const frame = state.frame;
  if (!isReplayMode() || !frame) {
    return;
  }
  const reference = isRecord(frame.artifact_summary?.replay_reference)
    ? frame.artifact_summary.replay_reference
    : {};
  const completion = isRecord(frame.completion) ? frame.completion : {};
  const processing = isRecord(frame.processing) ? frame.processing : {};
  const scene = frameScene(frame);
  elements.replayArtifactReference.textContent = String(
    reference.artifact_id ?? "Unavailable",
  );
  elements.replayArtifactReference.title = String(
    reference.canonical_digest_sha256 ?? "",
  );
  elements.replayCompletionBadge.textContent = humanize(
    completion.completion_state ?? "unavailable",
  );
  elements.replayProcessingBadge.textContent =
    processing.disclosure === "not_available_in_actor_pov"
      ? "Not available in actor POV"
      : humanize(processing.status ?? "unavailable");
  elements.replayEndReason.textContent = String(
    completion.public_end_or_failure_reason ??
      completion.end_or_failure_reason ??
      (asArray(completion.completion_bases).length > 0
        ? asArray(completion.completion_bases).map(humanize).join(" + ")
        : "Captured prefix"),
  );
  elements.replayIncomingValue.textContent = frame.transition_id
    ? String(frame.transition_id)
    : "Initial frame";
  elements.replayRangesButton.setAttribute(
    "aria-pressed",
    String(frame.show_ranges === true),
  );
  elements.replayRangesButton.disabled =
    state.busy || frame.replay_audience !== "researcher";
  elements.replayVerbosityButton.setAttribute(
    "aria-pressed",
    String(frame.verbose === true),
  );
  elements.replayVerbosityButton.disabled = state.busy;
  const selectedSlot = isRecord(scene?.selection)
    ? scene.selection.selected_global_slot
    : null;
  elements.replayClearReferenceButton.disabled =
    state.busy ||
    frame.replay_audience !== "researcher" ||
    !Number.isInteger(selectedSlot);
  renderReplayTimelineControls(replayTimelineElements, replayPlayback.snapshot());
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
 * @returns {number}
 */
function integer(value, fallback = 0) {
  return Number.isInteger(value) ? Number(value) : fallback;
}

/**
 * @param {unknown} frame
 * @returns {Record<string, any> | null}
 */
function frameScene(frame) {
  if (!isRecord(frame)) {
    return null;
  }
  return isRecord(frame.scene) ? frame.scene : null;
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
  if (record.viewer_mode === "replay" && isRecord(record.cursor)) {
    return record.cursor.frame_index === record.cursor.final_frame_index;
  }
  if (typeof record.terminal === "boolean") {
    return record.terminal;
  }
  if (isRecord(record.terminal)) {
    return Boolean(
      record.terminal.is_sealed ||
        record.terminal.terminated ||
        record.terminal.truncated ||
        record.terminal.reached_declared_horizon,
    );
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
  const scripted = liveDebuggerFrameIsScripted(frame);
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
  renderViewerBoundary();
  const replay = isReplayMode();
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
  if (terminal && replay) {
    const completion = isRecord(frame?.completion) ? frame.completion : {};
    const bases = asArray(completion.completion_bases);
    elements.terminalBadge.textContent =
      completion.completion_state !== "complete"
        ? "End of captured prefix"
        : bases.includes("task_terminal") && bases.includes("declared_horizon")
          ? "Task terminal · declared horizon"
          : bases.includes("task_terminal")
            ? "Task terminal"
            : bases.includes("declared_horizon")
              ? "Declared horizon"
              : "Complete replay";
  } else {
    elements.terminalBadge.textContent = terminal
      ? "Terminal · submissions blocked by Python"
      : "Terminal";
  }

  elements.scenarioDescription.textContent = frame
    ? replay
      ? `${humanize(frame.replay_audience ?? "replay")} · recorded frame ${frame.cursor?.frame_index ?? "—"} of ${frame.cursor?.final_frame_index ?? "—"}`
      : scenarioDescription(frame)
    : "Waiting for the Python debugger service.";

  renderScenarioOptions(frame);
  const viewMode = frame?.view_mode === "agent_pov" ? "pov" : frame?.view_mode;
  if (typeof viewMode === "string") {
    elements.viewSelect.value = viewMode;
  }
  if (typeof frame?.preset === "string") {
    elements.presetSelect.value = frame.preset;
  }
  const scenario = scenarioRecord(frame);
  const movementScale = Number(scenario.ordinary_movement_distance_scale);
  const movementScaleMinimum = Number(scenario.movement_scale_minimum);
  const movementScaleMaximum = Number(scenario.movement_scale_maximum);
  const movementScaleStep = Number(scenario.movement_scale_step);
  if (
    Number.isFinite(movementScaleMinimum) &&
    Number.isFinite(movementScaleMaximum) &&
    movementScaleMinimum <= movementScaleMaximum
  ) {
    elements.movementScaleInput.min = formatDisplayNumber(movementScaleMinimum, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    elements.movementScaleInput.max = formatDisplayNumber(movementScaleMaximum, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  if (Number.isFinite(movementScaleStep) && movementScaleStep > 0) {
    elements.movementScaleInput.step = formatDisplayNumber(movementScaleStep, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  if (Number.isFinite(movementScale)) {
    const displayedMovementScale = formatDisplayNumber(movementScale, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    elements.movementScaleInput.value = displayedMovementScale;
    elements.movementScaleValue.value = displayedMovementScale;
    elements.movementScaleValue.textContent = displayedMovementScale;
  }
  const movementScaleDisabled =
    disabled ||
    frame?.view_mode !== "researcher" ||
    !Number.isFinite(movementScale) ||
    !Number.isFinite(movementScaleMinimum) ||
    !Number.isFinite(movementScaleMaximum) ||
    !Number.isFinite(movementScaleStep);
  elements.movementScaleInput.disabled = movementScaleDisabled;
  elements.movementScaleTenthButton.disabled =
    movementScaleDisabled ||
    (Number.isFinite(movementScale) && Math.abs(movementScale - 0.1) < 1e-9);
  elements.movementScaleDefaultButton.disabled =
    movementScaleDisabled || scenario.movement_scale_overridden !== true;
  const scenarioControlsAvailable =
    !replay && (!state.frame || liveDebuggerScenarioControlsAvailable(state.frame));
  elements.scenarioControl.toggleAttribute("hidden", !scenarioControlsAvailable);
  elements.scenarioSelect.disabled = disabled || !scenarioControlsAvailable;
  elements.scenarioSelect.setAttribute(
    "aria-disabled",
    String(disabled || !scenarioControlsAvailable),
  );
  elements.scenarioSelect.title = scenarioControlsAvailable
    ? "Choose a registered live researcher scenario."
    : "Scenario controls are unavailable in recipient POV.";
  elements.viewSelect.disabled = disabled;
  elements.presetSelect.disabled = disabled;
  elements.reconnectButton.disabled = state.busy || state.shuttingDown;
  elements.exitButton.disabled = disabled;
  elements.exitButton.textContent = replay ? "Exit replay viewer" : "Exit analyzer";
  elements.resetButton.disabled = disabled;
  renderReplayMetadata();
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
  registerTooltipOwner(button, {
    kind: "legality",
    id: `command:${button.id || button.dataset.key || "choice"}`,
    title:
      button.getAttribute("aria-label") ??
      button.textContent?.trim() ??
      "Pending choice",
    details: [explanation],
    anchor: "element",
  });
}

/**
 * @param {Record<string, any>} hud
 * @param {number} targetAction
 * @returns {Record<string, any> | null}
 */
function candidateLegality(hud, targetAction) {
  return (
    asArray(hud.candidate_legalities).find((candidate) => {
      if (!isRecord(candidate)) {
        return false;
      }
      const candidateTarget = isRecord(candidate.target) ? candidate.target : null;
      return (
        Number(candidateTarget?.target_action ?? candidate.target_action) ===
        targetAction
      );
    }) ?? null
  );
}

/**
 * Rebuild target choices from the current controlled actor's authorized rows.
 * The select is a view over Python facts, never a client-owned pending model.
 *
 * @param {Record<string, any>} hud
 * @param {Record<string, any>} pending
 * @param {boolean} pov
 */
function renderCommandTargets(hud, pending, pov) {
  const candidates = asArray(hud.candidate_legalities).filter(isRecord);
  const pendingTarget = isRecord(pending.target) ? pending.target : {};
  const selectedSlot = pendingTarget.global_slot;
  const selectedTargetAction = Number(
    pendingTarget.target_action ?? pending.target_action,
  );
  const fragment = document.createDocumentFragment();
  for (const candidate of candidates) {
    const target = isRecord(candidate.target) ? candidate.target : {};
    const option = document.createElement("option");
    const basic = candidate.basic_available === true ? "B ✓" : "B ×";
    const ultimate = candidate.ultimate_available === true ? "U ✓" : "U ×";
    if (pov && Number.isInteger(target.target_action)) {
      option.value = `pov-target-action:${target.target_action}`;
      option.textContent = `${target.public_agent_id ?? "No target"} · action ${target.target_action} · ${basic} · ${ultimate}`;
      option.selected = Number(target.target_action) === selectedTargetAction;
    } else if (target.disclosure === "public" && Number.isInteger(target.global_slot)) {
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
 * @param {Record<string, any> | null} frame
 */
function renderDraftState(hud, frame) {
  const pending = isRecord(hud.pending_action) ? hud.pending_action : {};
  const pov = frame?.frame_kind === "actor_pov_live_debugger";
  const controlledSlot = Number(hud.controlled_global_slot);
  const controlledIdentity = pov
    ? hud.controlled_public_agent_id
    : Number.isInteger(controlledSlot)
      ? `id_${controlledSlot}`
      : null;
  elements.commandControlledActor.textContent = controlledIdentity
    ? `Actor · ${controlledIdentity}`
    : "Actor · unavailable";
  elements.commandControlledActor.setAttribute(
    "aria-label",
    controlledIdentity
      ? `Controlled actor ${controlledIdentity}`
      : "Controlled actor unavailable",
  );
  elements.commandControlledActor.dataset.controlledSlot = Number.isInteger(
    controlledSlot,
  )
    ? String(controlledSlot)
    : "";
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

  renderCommandTargets(hud, pending, pov);
  const pendingTarget = isRecord(pending.target) ? pending.target : {};
  const rawTargetAction = pendingTarget.target_action ?? pending.target_action;
  const targetAction = Number.isInteger(rawTargetAction) ? Number(rawTargetAction) : 0;
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
  if (isReplayMode()) {
    elements.commandTargetSelect.disabled = true;
    if (elements.commandDeck) {
      for (const button of elements.commandDeck.querySelectorAll("button")) {
        /** @type {HTMLButtonElement} */ (button).disabled = true;
      }
    }
    return;
  }
  const scenario = scenarioRecord(state.frame);
  const scripted = liveDebuggerFrameIsScripted(state.frame);
  const hud = isRecord(state.frame?.hud) ? state.frame.hud : {};
  renderDraftState(hud, state.frame);
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

function applyReplayReferenceSemantics() {
  if (state.frame?.replay_audience !== "researcher") {
    return;
  }
  const scene = frameScene(state.frame);
  const selectedSlot = isRecord(scene?.selection)
    ? scene.selection.selected_global_slot
    : null;
  if (!Number.isInteger(selectedSlot)) {
    return;
  }
  const reference = elements.battlefield.querySelector(
    `.agent[data-slot="${selectedSlot}"]`,
  );
  if (!(reference instanceof Element)) {
    return;
  }
  const agent = asArray(scene?.agents).find(
    (candidate) => isRecord(candidate) && candidate.global_slot === selectedSlot,
  );
  if (!isRecord(agent) || typeof agent.public_agent_id !== "string") {
    return;
  }
  const publicAgentId = agent.public_agent_id;
  const ariaLabel = reference.getAttribute("aria-label") ?? `Agent ID ${publicAgentId}`;
  reference.setAttribute(
    "aria-label",
    ariaLabel.replace(/selected target/giu, "Reference"),
  );
  registerTooltipOwner(reference, {
    kind: "agent",
    id: `replay-reference:${publicAgentId}`,
    title: `Agent ID ${publicAgentId}`,
    details: ["Researcher Reference", "Inspector and highlight only"],
    anchor: "element",
  });
}

function render() {
  renderConnection();
  renderSessionToolbar();
  battlefieldRenderer.render(state.frame, { offline: state.offline });
  applyReplayReferenceSemantics();
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
  tooltipController.refresh();
}

/**
 * On the supported minimum battlefield, accepted combat truth temporarily
 * outranks analysis decoration. The choreography controller remains the only
 * presentation clock; this adapter merely mirrors its active-animation state
 * into the durable SVG renderer and reprojects retained effects when the set of
 * protected rectangles changes.
 *
 * @param {ReturnType<CombatChoreographer["snapshot"]>} presentation
 */
function syncCompactActiveCombatPriority(presentation) {
  const changed = battlefieldRenderer.setCompactActiveCombat(
    presentation.active && presentation.animationCount > 0,
  );
  if (!changed || !presentation.active || !state.frame) {
    return;
  }
  try {
    choreographer.reproject(state.frame, battlefieldRenderer.choreographySurface());
  } catch (error) {
    setNotice(
      error instanceof Error
        ? `Compact combat presentation failed to reproject: ${error.message}`
        : "Compact combat presentation failed to reproject.",
      "error",
    );
    renderConnection();
  }
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
    tooltipController.refresh();
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
 * Send one replay command through the replay-only route. This function is the
 * controller's single request boundary; it installs the authoritative frame
 * before resolving so presentation settling can begin immediately.
 *
 * @param {Readonly<Record<string, any>>} command
 */
async function sendReplayCommand(command) {
  if (!isReplayMode() || !state.frame) {
    throw new DebuggerApiError("Replay controls require an installed replay frame.");
  }
  if (state.busy || state.shuttingDown) {
    throw new DebuggerApiError("A replay request is already in flight.");
  }
  if (state.resyncRequired || state.offline) {
    throw new DebuggerApiError("Reconnect before sending another replay command.");
  }
  state.busy = true;
  setNotice("Waiting for the read-only replay response…", "info");
  render();
  try {
    const previousFrame = state.frame;
    const previousCursor = previousFrame.cursor;
    const payload = await postReplayCommand(
      state.token,
      replayCommandRequest({
        clientId: state.clientId,
        commandId: window.crypto.randomUUID(),
        baseRevision: currentRevision(),
        command,
      }),
    );
    const frame = extractFrame(payload);
    if (frame?.viewer_mode !== "replay") {
      throw new DebuggerApiError("Replay response did not contain a replay frame.");
    }
    validateReplayFrameContinuity(previousFrame, frame, payload.result);
    validateReplayCommandOutcome(command, payload, previousCursor);
    let timeline = state.timeline;
    if (
      frame.timeline_id !== previousFrame.timeline_id ||
      timeline?.timeline_id !== frame.timeline_id
    ) {
      timeline = extractReplayTimeline(await getReplayTimeline(state.token));
    }
    if (!timeline) {
      throw new TypeError("Replay response has no audience timeline candidate.");
    }
    const joinedTimeline = joinReplayFrameAndTimeline(frame, timeline);
    state.frame = frame;
    state.timeline = joinedTimeline;
    state.offline = false;
    state.resyncRequired = false;
    replayPlayback.setConnected(true);
    const notice = extractNotice(payload);
    setNotice(
      notice ??
        (payload?.result === "duplicate"
          ? "Duplicate replay command recognized; it was not applied again."
          : payload?.result === "no_op"
            ? "Replay already matched that request."
            : "Read-only replay frame updated."),
      payload?.result === "duplicate" || payload?.result === "no_op"
        ? "warning"
        : "success",
    );
    if (command.command_type === "exit") {
      state.shuttingDown = true;
      setNotice("Exit accepted. The local replay viewer is shutting down.", "info");
    }
    return payload;
  } catch (error) {
    let replayError = null;
    if (error instanceof DebuggerApiError && isRecord(error.payload)) {
      // postReplayCommand's decode boundary has already strictly normalized
      // this envelope. Its latest frame is an internal settled replay frame,
      // not raw wire data, so it must never cross the raw normalizer twice.
      replayError = error.payload;
    }
    if (error instanceof DebuggerApiError && error.status === 409) {
      const latest = replayError?.latest_frame ?? null;
      if (latest?.viewer_mode !== "replay") {
        state.offline = false;
        state.resyncRequired = true;
        replayPlayback.setConnected(false);
        setNotice(
          "The replay service reported stale state without a valid latest frame. Reconnect is required.",
          "error",
        );
        throw error;
      }
      try {
        validateReplayFrameContinuity(state.frame, latest, "stale_resync");
        const latestTimeline = joinReplayFrameAndTimeline(
          latest,
          extractReplayTimeline(await getReplayTimeline(state.token)),
        );
        state.frame = latest;
        state.timeline = latestTimeline;
      } catch (candidateError) {
        state.offline = false;
        state.resyncRequired = true;
        replayPlayback.setConnected(false);
        setNotice(
          candidateError instanceof Error
            ? `The stale replay candidate failed validation: ${candidateError.message}`
            : "The stale replay candidate failed validation.",
          "error",
        );
        throw candidateError;
      }
      state.offline = false;
      state.resyncRequired = false;
      replayPlayback.setConnected(true);
      const errorCode = replayError?.error_code ?? null;
      setNotice(
        errorCode === "command_id_conflict"
          ? "The replay service rejected a command-ID conflict. Its latest frame was installed; nothing was retried."
          : "This replay tab was stale. The latest frame was installed; the command was not retried.",
        "warning",
      );
      return Object.freeze({ handled_resync: true, frame: latest });
    } else {
      const status = error instanceof DebuggerApiError ? error.status : 0;
      state.offline = status === 0 || status === 401 || status === 403;
      state.resyncRequired = true;
      replayPlayback.setConnected(false);
      setNotice(
        status === 401 || status === 403
          ? "Replay capability is invalid. Reopen the exact URL printed by the Python launcher."
          : error instanceof Error
            ? `${error.message} Reconnect before sending another replay command.`
            : "Replay command failed. Reconnect before sending another command.",
        "error",
      );
    }
    throw error;
  } finally {
    state.busy = false;
    render();
  }
}

/** @param {Readonly<Record<string, any>>} command */
async function dispatchReplayCommand(command) {
  if (
    state.frame?.replay_audience !== "researcher" &&
    (command.command_type === "select_agent" || command.command_type === "set_ranges")
  ) {
    setNotice(
      "Reference and range controls are unavailable in actor POV replay.",
      "warning",
    );
    renderConnection();
    return null;
  }
  replayPlayback.pause("user_command");
  try {
    const payload = await sendReplayCommand(command);
    const frame =
      payload?.handled_resync === true ? payload.frame : extractFrame(payload);
    if (frame) {
      replayPlayback.installCursor(frame.cursor);
    }
    return payload;
  } catch {
    return null;
  }
}

/** @param {Record<string, unknown>} command */
function dispatchPanelCommand(command) {
  if (!isReplayMode()) {
    return dispatchCommand(command);
  }
  if (
    command.command_type === "roster_selection" &&
    Number.isInteger(command.global_slot)
  ) {
    if (command.role === "target") {
      return dispatchReplayCommand({
        command_type: "select_agent",
        selected_global_slot: command.global_slot,
      });
    }
    if (command.role === "control") {
      return dispatchReplayCommand({
        command_type: "set_pov_actor",
        global_slot: command.global_slot,
      });
    }
  }
  setNotice("That live debugger action is unavailable in replay.", "warning");
  renderConnection();
  return Promise.resolve(null);
}

/**
 * @param {Record<string, unknown>} command
 */
async function dispatchCommand(command) {
  if (isReplayMode()) {
    setNotice("Live debugger commands are unavailable in read-only replay.", "warning");
    renderConnection();
    return;
  }
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
      setNotice("Exit accepted. The local analyzer server is shutting down.", "info");
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
  if (isReplayMode()) {
    replayPlayback.pause("reconnect");
  }
  setNotice("Fetching the current authoritative frame…", "info");
  renderConnection();
  try {
    const payload = await getCurrentFrame(state.token);
    const frame = extractFrame(payload);
    if (!frame) {
      throw new DebuggerApiError("Frame response did not contain a debugger frame.");
    }
    if (state.frame?.viewer_mode === "replay" && frame.viewer_mode !== "replay") {
      throw new DebuggerApiError(
        "A replay viewer cannot reconnect to a live debugger frame. Reopen the replay URL to resynchronize.",
      );
    }
    let timeline = null;
    if (frame.viewer_mode === "replay") {
      if (state.frame?.viewer_mode === "replay") {
        validateReplayFrameContinuity(state.frame, frame, "stale_resync");
      }
      timeline = joinReplayFrameAndTimeline(
        frame,
        extractReplayTimeline(await getReplayTimeline(state.token)),
      );
    }
    state.frame = frame;
    state.timeline = timeline;
    if (frame.viewer_mode === "replay") {
      replayPlayback.installCursor(frame.cursor);
      replayPlayback.setConnected(true);
    }
    state.offline = false;
    state.resyncRequired = false;
    setNotice(extractNotice(payload) ?? "Connected to the local analyzer.", "success");
  } catch (error) {
    const status = error instanceof DebuggerApiError ? error.status : 0;
    state.offline = status === 0 || status === 401 || status === 403;
    state.resyncRequired = true;
    if (isReplayMode()) {
      replayPlayback.setConnected(false);
    }
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
  isInteractive: () => !isReplayMode(),
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
  const command = {
    command_type: "set_view",
    view_mode: elements.viewSelect.value,
  };
  if (isReplayMode()) {
    void dispatchReplayCommand(command);
  } else {
    void dispatchCommand(command);
  }
});

elements.presetSelect.addEventListener("change", () => {
  const command = {
    command_type: "set_preset",
    preset: elements.presetSelect.value,
  };
  if (isReplayMode()) {
    void dispatchReplayCommand(command);
  } else {
    void dispatchCommand(command);
  }
});

elements.resetButton.addEventListener("click", () => {
  dispatchCommand({ command_type: "reset" });
});

elements.movementScaleInput.addEventListener("input", () => {
  const movementScale = Number(elements.movementScaleInput.value);
  if (!Number.isFinite(movementScale)) {
    return;
  }
  const displayedMovementScale = formatDisplayNumber(movementScale, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  elements.movementScaleValue.value = displayedMovementScale;
  elements.movementScaleValue.textContent = displayedMovementScale;
});

elements.movementScaleInput.addEventListener("change", () => {
  const movementScale = Number(elements.movementScaleInput.value);
  if (!Number.isFinite(movementScale)) {
    return;
  }
  dispatchCommand({
    command_type: "set_movement_scale",
    movement_scale: movementScale,
  });
});

elements.movementScaleTenthButton.addEventListener("click", () => {
  dispatchCommand({
    command_type: "set_movement_scale",
    movement_scale: 0.1,
  });
});

elements.movementScaleDefaultButton.addEventListener("click", () => {
  dispatchCommand({
    command_type: "set_movement_scale",
    movement_scale: null,
  });
});

elements.commandTargetSelect.addEventListener("change", () => {
  const command = targetSelectionCommand(elements.commandTargetSelect.value, {
    actorPov: state.frame?.frame_kind === "actor_pov_live_debugger",
  });
  if (!command) {
    return;
  }
  dispatchCommand(command);
});

elements.exitButton.addEventListener("click", () => {
  if (isReplayMode()) {
    void dispatchReplayCommand({ command_type: "exit" });
  } else {
    void dispatchCommand({ command_type: "exit" });
  }
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

bindReplayTimelineControls(replayTimelineElements, replayPlayback);

elements.replayRangesButton.addEventListener("click", () => {
  if (!isReplayMode() || state.frame?.replay_audience !== "researcher") {
    return;
  }
  void dispatchReplayCommand({
    command_type: "set_ranges",
    show_ranges: state.frame.show_ranges !== true,
  });
});

elements.replayVerbosityButton.addEventListener("click", () => {
  if (!isReplayMode()) {
    return;
  }
  void dispatchReplayCommand({
    command_type: "set_verbosity",
    verbose: state.frame?.verbose !== true,
  });
});

elements.replayClearReferenceButton.addEventListener("click", () => {
  if (!isReplayMode() || state.frame?.replay_audience !== "researcher") {
    return;
  }
  void dispatchReplayCommand({
    command_type: "select_agent",
    selected_global_slot: null,
  });
});

document.addEventListener("visibilitychange", () => {
  replayPlayback.setHidden(document.hidden);
});

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
replayPlayback.setHidden(document.hidden);

render();
loadCurrentFrame();
