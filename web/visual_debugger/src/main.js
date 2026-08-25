import {
  acquireCapabilityToken,
  acquireClientId,
  DebuggerApiError,
  extractFrame,
  extractJoinedFrame,
  extractNotice,
  getCurrentFrameAndPresentation,
  getCurrentPresentation,
  getReplayMetricReport,
  getReplayTimeline,
  postCommand,
  postReplayCommand,
} from "./api.js";
import {
  authorizedOracleCommandSlotForPresentationKey,
  authorizedOracleCommandSlotForPublicAgentId,
  authorizedPresentationAudience,
  authorizedPresentationInspectionState,
  authorizedPresentationPreferenceKey,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
  sameAuthorizedPresentationPreferenceKey,
} from "./authorized-presentation-adapter.js";
import {
  isJoinedTransportAndAuthorizedPresentationV1,
  isPresentationJoinRace,
  joinReplayTransportAndTimelineV1,
  validateReplayTransportContinuityV1,
} from "./authorized-presentation-normalizer.js";
import { CombatChoreographer, ConsumedTransitionLedger } from "./choreography.js";
import { SvgChoreographyPainter } from "./choreography-painter.js";
import { isSubmissionCommand } from "./choreography-plan.js";
import {
  bindBattlefieldControls,
  commandResponseSchedulesShutdown,
  keyboardCommand,
  presentationRequiresSubmissionSettle,
  recordingCommandDecision,
  recordingReviewHandoffRequired,
  recordingSaveAsCommand,
  targetSelectionCommand,
} from "./controls.js";
import { explainAgent, explainLegality, explainTechnicalFact } from "./explanations.js";
import {
  authorizedInspectorView,
  DebuggerPanels,
  disclosurePanelInitiallyOpen,
  renderSemanticInspector,
} from "./panels.js";
import {
  pendingPresentationSurfaceView,
  resolveInstalledPresentationAuthorityV1,
} from "./presentation-authority-view.js";
import { PresentationInstallCoordinator } from "./presentation-install.js";
import {
  bindReplayTimelineControls,
  ReplayPlaybackController,
  REPLAY_TRANSPORT_STATES,
  renderReplayTimelineControls,
  replayCommandRequest,
  replayTimelineSimulatorStep,
  validateReplayCommandOutcome,
} from "./replay-controls.js";
import { captureReplayBattlefieldPngV1 } from "./replay-export.js";
import { BattlefieldRenderer } from "./scene.js";
import {
  createSemanticDescriptor,
  createTooltipController,
  registerTooltipOwner,
} from "./tooltip.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualFilterEnabled,
  reduceVisualFilterState,
  VISUAL_FILTER_REGISTRY,
} from "./visual-filters.js";

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
  appTitle: requiredElement("app-title"),
  connectionStatus: requiredElement("connection-status"),
  audienceBadge: requiredElement("audience-badge"),
  terminalBadge: requiredElement("terminal-badge"),
  recordingBadge: requiredElement("recording-badge"),
  viewSelect: requiredElement("view-select"),
  stepValue: requiredElement("step-value"),
  transitionValue: requiredElement("transition-value"),
  recordingPanel: requiredElement("recording-panel"),
  recordingLifecycle: requiredElement("recording-lifecycle"),
  recordingProgress: requiredElement("recording-progress"),
  recordingCompletion: requiredElement("recording-completion"),
  recordingPersistenceFact: requiredElement("recording-persistence-fact"),
  recordingPersistenceError: requiredElement("recording-persistence-error"),
  recordingStatusNote: requiredElement("recording-status-note"),
  recordingFinishButton: requiredElement("recording-finish-button"),
  recordingReviewButton: requiredElement("recording-review-button"),
  recordingRetryButton: requiredElement("recording-retry-button"),
  recordingSaveAsControl: requiredElement("recording-save-as-control"),
  recordingSaveAsInput: requiredElement("recording-save-as-input"),
  recordingSaveAsButton: requiredElement("recording-save-as-button"),
  recordingDiscardDialog: requiredElement("recording-discard-dialog"),
  recordingDiscardIntent: requiredElement("recording-discard-intent"),
  recordingDiscardCancelButton: requiredElement("recording-discard-cancel-button"),
  recordingDiscardConfirmButton: requiredElement("recording-discard-confirm-button"),
  replayTimeline: requiredElement("replay-timeline"),
  replayArtifactReference: requiredElement("replay-artifact-reference"),
  replayCompletionBadge: requiredElement("replay-completion-badge"),
  replayProcessingBadge: requiredElement("replay-processing-badge"),
  replayEndReason: requiredElement("replay-end-reason"),
  replayFirstButton: requiredElement("replay-first-button"),
  replayBackTenButton: requiredElement("replay-back-ten-button"),
  replayPreviousButton: requiredElement("replay-previous-button"),
  replayPlayPauseButton: requiredElement("replay-play-pause-button"),
  replayNextButton: requiredElement("replay-next-button"),
  replayForwardTenButton: requiredElement("replay-forward-ten-button"),
  replayLastButton: requiredElement("replay-last-button"),
  replayFrameSlider: requiredElement("replay-frame-slider"),
  replayFramePosition: requiredElement("replay-frame-position"),
  replayPlaybackRate: requiredElement("replay-playback-rate"),
  replayTransportStatus: requiredElement("replay-transport-status"),
  replayArtifactActions: requiredElement("replay-artifact-actions"),
  replayExportPngButton: requiredElement("replay-export-png-button"),
  replayDownloadMetricsButton: requiredElement("replay-download-metrics-button"),
  replayRangesButton: requiredElement("replay-ranges-button"),
  replayClearReferenceButton: requiredElement("replay-clear-reference-button"),
  reconnectButton: requiredElement("reconnect-button"),
  helpButton: requiredElement("help-button"),
  exitButton: requiredElement("exit-button"),
  resetButton: requiredElement("reset-button"),
  liveRangesButton: requiredElement("live-ranges-button"),
  notice: requiredElement("notice"),
  workspace: requiredElement("workspace"),
  scenarioDescription: requiredElement("scenario-description"),
  battlefieldShell: requiredElement("battlefield-shell"),
  battlefield: requiredElement("battlefield"),
  battlefieldEmpty: requiredElement("battlefield-empty"),
  commandDeck: document.querySelector(".command-deck"),
  commandControlledActor: requiredElement("command-controlled-actor"),
  roster: requiredElement("roster"),
  rosterCount: requiredElement("roster-count"),
  agentDetails: requiredElement("agent-details"),
  visualFilters: requiredElement("visual-filters"),
  visualFilterOptions: requiredElement("visual-filter-options"),
  visualFilterCount: requiredElement("visual-filter-count"),
  enableAllVisualFiltersButton: requiredElement("enable-all-visual-filters-button"),
  disableAllVisualFiltersButton: requiredElement("disable-all-visual-filters-button"),
  visualKey: requiredElement("visual-key"),
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
  commandCommitTitle: requiredElement("command-commit-title"),
  commandCommitSummary: requiredElement("command-commit-summary"),
  acceptedCard: requiredElement("accepted-card"),
  acceptedAnnouncement: requiredElement("accepted-announcement"),
  diagnosticsCard: requiredElement("diagnostics-card"),
  visualTooltip: requiredElement("visual-tooltip"),
  visualTooltipTitle: requiredElement("visual-tooltip-title"),
  visualTooltipDetails: requiredElement("visual-tooltip-details"),
  helpDialog: requiredElement("help-dialog"),
  helpHeading: requiredElement("help-heading"),
  battlefieldInstructions: requiredElement("battlefield-instructions"),
  liveOnly: /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll("[data-live-only]")
  ),
  liveHelpModes: /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll("[data-live-help-mode]")
  ),
  replayOnly: /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll("[data-replay-only]")
  ),
};

const EXPECTED_VISUAL_FILTER_COUNT = 18;

/**
 * Build the fixed page-local filter surface from the shared paint registry.
 * Replacing the empty markup container also prevents browser form restoration
 * from overriding the all-enabled state on a genuine document load.
 */
function installVisualFilterControls() {
  const registeredIds = VISUAL_FILTER_REGISTRY.map(({ id }) => id);
  if (
    registeredIds.length !== EXPECTED_VISUAL_FILTER_COUNT ||
    new Set(registeredIds).size !== EXPECTED_VISUAL_FILTER_COUNT
  ) {
    throw new TypeError(
      `Visual Filters requires exactly ${EXPECTED_VISUAL_FILTER_COUNT} unique entries.`,
    );
  }
  const fragment = document.createDocumentFragment();
  for (const { id, label, defaultEnabled } of VISUAL_FILTER_REGISTRY) {
    const enabled = isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, id);
    if (enabled !== defaultEnabled) {
      throw new TypeError(`Visual filter ${id} disagrees with the default state.`);
    }
    const option = document.createElement("label");
    option.className = "visual-filters__option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `visual-filter-${id.replaceAll("_", "-")}`;
    input.value = id;
    input.dataset.visualFilterId = id;
    input.setAttribute("autocomplete", "off");
    input.setAttribute("aria-describedby", "visual-filters-help");
    input.defaultChecked = enabled;
    input.checked = enabled;
    const text = document.createElement("span");
    text.textContent = label;
    option.append(input, text);
    fragment.append(option);
  }
  elements.visualFilterOptions.replaceChildren(fragment);
}

/**
 * Page-lifetime presentation state. It is replaced atomically, never stored,
 * and deliberately survives transport, authority, audience, and episode
 * changes until the document itself reloads.
 */
let visualFilterState = DEFAULT_VISUAL_FILTER_STATE;
installVisualFilterControls();

/**
 * @type {{
 *   token: string | null,
 *   clientId: string,
 *   authority: Readonly<Record<string, any>> | null,
 *   frame: Record<string, any> | null,
 *   presentation: Readonly<Record<string, any>> | null,
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
  authority: null,
  frame: null,
  presentation: null,
  timeline: null,
  busy: false,
  offline: false,
  resyncRequired: false,
  shuttingDown: false,
  notice: null,
  noticeLevel: "info",
};

/**
 * One live request may retain one fresh Enter intent while a coherent draft
 * response is installed. This is page-local latency state, not simulator,
 * action, or replay state.
 *
 * @type {{
 *   allowsDeferredSubmit: boolean,
 *   deferredSubmit: Readonly<Record<string, unknown>> | null,
 * } | null}
 */
let activeLiveCommandTransaction = null;

/**
 * One page-local read-only artifact transaction. It is an async ownership
 * fence, not replay transport state, and is invalidated synchronously whenever
 * installed presentation authority is cleared.
 *
 * @type {Readonly<{
 *   kind: "export_png" | "download_metrics",
 *   authority: Readonly<Record<string, any>>,
 *   transport: Readonly<Record<string, any>>,
 *   presentation: Readonly<Record<string, any>>,
 *   playbackGeneration: number,
 *   localInspectedPresentationKey: string | null,
 *   showRanges: boolean,
 *   visualFilters: typeof visualFilterState,
 * }> | null}
 */
let replayArtifactActionTransaction = null;

const SCIENTIFIC_DISCLOSURE_BODY_IDS = Object.freeze({
  "command-deck": "command-deck-body",
  "roster-details": "roster-details-body",
  "agent-details": "agent-details-body",
  "pending-turn-details": "pending-turn-details-body",
  "latest-transition-details": "latest-transition-details-body",
  "technical-frame-details": "technical-frame-details-body",
});

const scientificDisclosures = Object.freeze(
  Object.entries(SCIENTIFIC_DISCLOSURE_BODY_IDS).map(([panelId, bodyId]) => {
    const panel = requiredElement(panelId);
    const body = requiredElement(bodyId);
    if (!(panel instanceof HTMLDetailsElement) || !(body instanceof HTMLElement)) {
      throw new TypeError(`Scientific disclosure ${panelId} has invalid markup.`);
    }
    return Object.freeze({ panelId, panel, body });
  }),
);

/**
 * The only retained scientific preference record. This is deliberately one
 * active record rather than a cache keyed by authority: crossing A -> B
 * replaces A, so a later B -> A boundary receives fresh defaults.
 *
 * @type {{
 *   authorityKey: Readonly<{tuple: readonly unknown[], serialized: string}>,
 *   disclosures: Record<string, {open: boolean, scrollTop: number}>,
 *   agentDetailsAutoOpenAllowed: boolean,
 *   primaryFocus: null | {surface: "roster" | "battlefield", presentationKey: string},
 *   localInspection: {owned: false} | {owned: true, presentationKey: string | null},
 *   localRanges: {owned: false} | {owned: true, visible: boolean},
 * } | null}
 */
let activePresentationPreference = null;

let presentationPreferenceGeneration = 0;
let presentationPreferenceNeedsContentRender = false;
let consumedReplayRestartGeneration = -1;
let suppressPlaybackStateRender = false;
/** @type {number | null} */
let pendingPresentationPreferenceRestoreFrame = null;
/** @type {Element | null} */
let pendingPrimaryFocusRestoreAnchor = null;
/** @type {string | null} */
let workspaceMinimumHeightBeforeAuthorityInstall = null;

/** @type {WeakMap<HTMLDetailsElement, Readonly<{open: boolean}>>} */
const expectedDisclosureToggles = new WeakMap();

const PRODUCT_TITLES = Object.freeze({
  combat_debugger: "MARL-BattleGrounds Combat Debugger",
  replay_viewer: "MARL-BattleGrounds Replay Viewer",
});
const PRODUCT_HANDOFF_COMMANDS = new Set([
  "finish_and_review",
  "review_replay",
  "save_as",
]);
const SCRIPTED_INSPECTION_RECORDING_COMMANDS = new Set([
  "finish_and_review",
  "review_replay",
  "retry_save",
  "save_as",
  "confirm_discard_and_replace",
  "exit",
]);

class ProductIdentityMismatchError extends Error {}
class ProductReviewHandoff extends Error {}

/** @type {Readonly<{schema_version: 1, product_kind: "combat_debugger" | "replay_viewer"}> | null} */
let productIdentity = null;
/** @type {string | null} */
let startupProductIdentityError = null;

/**
 * Install one validated route-owned identity. Startup and the future explicit
 * recording-review handoff must both cross this boundary; frame facts are not
 * product-identity authority.
 *
 * @param {unknown} rawIdentity
 */
function applyProductIdentity(rawIdentity) {
  if (!isRecord(rawIdentity)) {
    throw new TypeError("Product bootstrap must be an object.");
  }
  const keys = Object.keys(rawIdentity).sort();
  if (keys.length !== 2 || keys[0] !== "product_kind" || keys[1] !== "schema_version") {
    throw new TypeError("Product bootstrap has an invalid shape.");
  }
  if (rawIdentity.schema_version !== 1) {
    throw new TypeError("Product bootstrap schema version is unsupported.");
  }
  if (
    rawIdentity.product_kind !== "combat_debugger" &&
    rawIdentity.product_kind !== "replay_viewer"
  ) {
    throw new TypeError("Product bootstrap kind is unsupported.");
  }
  productIdentity = Object.freeze({
    schema_version: 1,
    product_kind: rawIdentity.product_kind,
  });
  const title = PRODUCT_TITLES[productIdentity.product_kind];
  document.title = title;
  elements.appTitle.textContent = title;
  elements.helpHeading.textContent =
    productIdentity.product_kind === "replay_viewer"
      ? "Replay Viewer help"
      : "Combat Debugger help";
  document.documentElement.dataset.productKind = productIdentity.product_kind;
}

/** @param {Record<string, any>} frame */
function assertFrameMatchesProductIdentity(frame) {
  if (productIdentity === null) {
    throw new ProductIdentityMismatchError(
      "No validated product identity is installed.",
    );
  }
  const frameIsReplay = frame.viewer_mode === "replay";
  const identityIsReplay = productIdentity.product_kind === "replay_viewer";
  if (frameIsReplay !== identityIsReplay) {
    throw new ProductIdentityMismatchError(
      "The authoritative frame does not match this route's product identity.",
    );
  }
}

try {
  applyProductIdentity(Reflect.get(globalThis, "__MARL_DEBUGGER_BOOTSTRAP__"));
} catch (error) {
  startupProductIdentityError =
    error instanceof Error ? error.message : "Product bootstrap is invalid.";
}

const CONTROL_HELP = Object.freeze([
  [
    "#replay-timeline",
    "Replay timeline",
    "Use the read-only transport and presentation controls to inspect recorded frames.",
    "composite",
  ],
  [
    "#view-select",
    "Audience view",
    "Switch between Oracle View and recipient-authorized views.",
  ],
  [
    "#reconnect-button",
    "Reconnect",
    "Fetch and atomically install the latest authoritative frame.",
  ],
  ["#help-button", "Help", "Open the keyboard, recording, and replay controls guide."],
  ["#help-close-button", "Close help", "Close the product help dialog."],
  ["#exit-button", "Exit", "Ask the local Python service to close safely."],
  [
    "#recording-finish-button",
    "Finish and review",
    "Finalize the captured prefix, save it, and enter read-only review.",
  ],
  [
    "#recording-review-button",
    "Review replay",
    "Enter read-only review for the saved replay.",
  ],
  [
    "#recording-retry-button",
    "Retry save",
    "Retry publishing the same immutable replay bytes.",
  ],
  [
    "#recording-save-as-input",
    "Save As basename",
    "Enter a basename only; paths and overwrites are rejected.",
  ],
  [
    "#recording-save-as-button",
    "Save As",
    "Publish the same immutable replay bytes under the entered basename.",
  ],
  [
    "#recording-discard-cancel-button",
    "Keep recording",
    "Cancel replacement and preserve the captured prefix.",
  ],
  [
    "#recording-discard-confirm-button",
    "Discard and replace",
    "Confirm permanent loss of the unpublished prefix and start its named replacement.",
  ],
  ["#replay-first-button", "Start replay tick", "Seek to settled replay tick zero."],
  [
    "#replay-back-ten-button",
    "Back ten ticks",
    "Seek ten ticks backward with one clamped request.",
  ],
  [
    "#replay-previous-button",
    "Previous replay frame",
    "Seek one captured frame backward.",
  ],
  [
    "#replay-play-pause-button",
    "Replay playback",
    "Start or pause serialized read-only autoplay.",
  ],
  [
    "#replay-next-button",
    "Next replay frame",
    "Advance exactly one captured replay frame.",
  ],
  [
    "#replay-forward-ten-button",
    "Forward ten ticks",
    "Seek ten ticks forward with one clamped request.",
  ],
  ["#replay-last-button", "End replay tick", "Seek to the end of the captured prefix."],
  [
    "#replay-frame-slider",
    "Replay tick",
    "Preview locally without a request, then commit one exact captured-tick seek.",
  ],
  [
    "#replay-playback-rate",
    "Replay playback speed",
    "Scale the complete replay presentation clock without changing artifact authority.",
  ],
  [
    "#command-target-select",
    "Selected target",
    "Stage an authorized target for the controlled actor.",
  ],
  [
    "#submit-turn-button",
    "Apply authorized action",
    "Submit an editable draft or advance an inspection-only scripted frame through the authoritative Python service.",
  ],
  [
    "#reset-button",
    "Reset",
    "Start a deterministic fresh episode; recorded prefixes require confirmation.",
  ],
  [
    "#visual-key > summary",
    "Visual Key",
    "Explain the non-color visual grammar used on the battlefield.",
  ],
  [
    "#visual-filters > summary",
    "Visual Filters",
    "Show or hide individual battlefield presentation layers without changing scientific authority.",
  ],
  [
    "#enable-all-visual-filters-button",
    "Enable All",
    "Turn on all 18 local visual filters. The separate Ranges control is unchanged.",
  ],
  [
    "#disable-all-visual-filters-button",
    "Disable All",
    "Turn off all 18 local visual filters. The separate Ranges control is unchanged.",
  ],
  [
    ".diagnostics > summary",
    "Technical frame",
    "Inspect authorized wire and diagnostic details.",
  ],
  [
    "[data-key='Escape']",
    "Clear target",
    "Clear the selected target and leave battlefield command focus.",
  ],
]);

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
    exposePresentationState(presentation);
    renderCommandAvailability();
  },
});

const replayTimelineElements = {
  root: elements.replayTimeline,
  keyboardTarget: document,
  keyboardEnabled: () => installedPresentationAuthority() !== null && !state.busy,
  firstButton: elements.replayFirstButton,
  backTenButton: elements.replayBackTenButton,
  previousButton: elements.replayPreviousButton,
  playPauseButton: elements.replayPlayPauseButton,
  nextButton: elements.replayNextButton,
  forwardTenButton: elements.replayForwardTenButton,
  lastButton: elements.replayLastButton,
  slider: elements.replayFrameSlider,
  position: elements.replayFramePosition,
  rateSelect: elements.replayPlaybackRate,
  status: elements.replayTransportStatus,
  tickForFrameIndex: (/** @type {number} */ frameIndex) =>
    installedPresentationAuthority() === null
      ? null
      : replayTimelineSimulatorStep(state.timeline, frameIndex),
  incomingTransitionForFrameIndex: (/** @type {number} */ frameIndex) => {
    const installed = installedPresentationAuthority();
    if (installed === null || installed.transport.cursor?.frame_index !== frameIndex) {
      return null;
    }
    return authorizedIncomingTransitionId(installed.presentation);
  },
};

/**
 * Keep retained replay authority visible while fencing timeline navigation
 * behind the request that is installing its complete successor.
 *
 * @param {ReturnType<ReplayPlaybackController["snapshot"]>} playback
 */
function replayTimelineRenderState(playback) {
  if (!state.busy || playback.transportState === REPLAY_TRANSPORT_STATES.OFFLINE) {
    return playback;
  }
  return Object.freeze({
    ...playback,
    transportState: REPLAY_TRANSPORT_STATES.ADVANCING,
  });
}

const replayPlayback = new ReplayPlaybackController({
  request: sendReplayTransportCommand,
  waitForPresentation: () => choreographer.whenSettled(),
  getMotionMode: () => choreographer.snapshot().motionMode,
  onStateChange: (playback) => {
    if (
      replayArtifactActionTransaction !== null &&
      playback.generation !== replayArtifactActionTransaction.playbackGeneration
    ) {
      invalidateReplayArtifactAction();
    }
    choreographer.setPlaybackRate(playback.playbackRate);
    const timeline =
      installedPresentationAuthority() === null
        ? pendingPresentationSurfaceView(playback).replay.timeline
        : playback;
    renderReplayTimelineControls(
      replayTimelineElements,
      replayTimelineRenderState(timeline),
    );
    renderReplayArtifactActions(installedPresentationAuthority());
    if (
      installedPresentationAuthority() !== null &&
      !state.busy &&
      !suppressPlaybackStateRender &&
      playback.transportState !== "ADVANCING"
    ) {
      render();
    }
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
  diagnosticsCard: elements.diagnosticsCard,
  onCommand: dispatchPanelCommand,
});

const tooltipController = createTooltipController({
  root: document.body,
  tooltip: elements.visualTooltip,
  title: elements.visualTooltipTitle,
  details: elements.visualTooltipDetails,
  onInspect: showSemanticInspector,
});

let lastBattlefieldSizeKey = "";
/** @type {number | null} */
let pendingResizeFrame = null;
/** @type {Readonly<Record<string, unknown>> | null} */
let pendingRecordingReplacement = null;
let productHandoffOutcomeUnknown = false;
let productHandoffReloadRequested = false;

/**
 * Make a persistent DOM node unable to resolve a descriptor from the previous
 * authority. Descriptor values live in a WeakMap; removing the owner marker is
 * the public lookup fence, while the ARIA fields remove its durable projection.
 * A later authorized render registers a fresh descriptor on the same node.
 *
 * @param {Element} element
 */
function clearPresentationTooltipOwner(element) {
  element.removeAttribute("data-tooltip-owner");
  element.removeAttribute("data-tooltip-kind");
  element.removeAttribute("data-tooltip-text");
  element.removeAttribute("data-tooltip-tone");
  element.removeAttribute("data-tooltip-accent");
  element.removeAttribute("aria-description");
}

/**
 * Reapply the same fail-closed operational chrome at synchronous clear time
 * and on every later render until one coherent branded pair is installed.
 * The retained transport is protocol bookkeeping only.
 */
function renderPendingPresentationChrome() {
  const pending = pendingPresentationSurfaceView(replayPlayback.snapshot());
  clearPresentationTooltipOwner(elements.replayArtifactReference);
  clearPresentationTooltipOwner(elements.replayCompletionBadge);
  clearPresentationTooltipOwner(elements.replayProcessingBadge);
  elements.replayArtifactReference.removeAttribute("title");
  elements.replayArtifactReference.textContent = pending.replay.artifactReference;
  elements.replayCompletionBadge.textContent = pending.replay.completion;
  elements.replayProcessingBadge.textContent = pending.replay.processing;
  elements.replayEndReason.textContent = pending.replay.endReason;
  elements.replayRangesButton.setAttribute("aria-pressed", "false");
  elements.replayRangesButton.disabled = true;
  elements.replayClearReferenceButton.disabled = true;
  renderReplayTimelineControls(replayTimelineElements, pending.replay.timeline);
  renderReplayArtifactActions(null);

  elements.terminalBadge.hidden = pending.terminal.hidden;
  elements.terminalBadge.textContent = pending.terminal.text;
  elements.viewSelect.value = pending.viewMode;
  elements.viewSelect.disabled = true;
  elements.scenarioDescription.textContent = pending.scenarioDescription;

  elements.recordingPanel.toggleAttribute("hidden", pending.recording.hidden);
  elements.recordingBadge.toggleAttribute("hidden", pending.recording.hidden);
  elements.recordingBadge.textContent = pending.recording.badgeText;
  delete elements.recordingBadge.dataset.lifecycle;
  delete document.documentElement.dataset.recordingLifecycle;
  elements.recordingLifecycle.textContent = pending.recording.lifecycle;
  elements.recordingProgress.textContent = pending.recording.progress;
  elements.recordingCompletion.textContent = pending.recording.completion;
  elements.recordingPersistenceFact.toggleAttribute("hidden", true);
  elements.recordingPersistenceError.textContent = pending.recording.persistence;
  elements.recordingStatusNote.textContent = pending.recording.status;
  for (const control of [
    elements.recordingFinishButton,
    elements.recordingReviewButton,
    elements.recordingRetryButton,
  ]) {
    control.toggleAttribute("hidden", true);
    control.disabled = true;
  }
  elements.recordingSaveAsControl.toggleAttribute("hidden", true);
  elements.recordingSaveAsInput.disabled = true;
  elements.recordingSaveAsButton.disabled = true;
  elements.recordingDiscardIntent.textContent = pending.recording.status;
  elements.recordingDiscardConfirmButton.disabled = true;
  if (elements.recordingDiscardDialog.open) {
    elements.recordingDiscardDialog.close();
  }
  pendingRecordingReplacement = null;
}

/**
 * Remove every presentation-owned surface synchronously. The raw transport
 * epoch stays installed for diagnostics and command-outcome accounting until a
 * complete successor pair is ready; it is never used to keep the battlefield
 * or scientific panels populated while authority is pending.
 *
 * @param {string} reason
 */
function clearPresentationAuthority(reason) {
  invalidateReplayArtifactAction();
  holdWorkspaceHeightDuringAuthorityInstall();
  savePresentationPreferenceBeforeClear();
  enterPendingPresentationPreferenceState();
  state.authority = null;
  state.presentation = null;
  tooltipController.hide();
  choreographer.clear(reason);
  battlefieldRenderer.render(null, {
    offline: true,
    visualFilterState,
  });
  panels.render(null, {
    busy: true,
    shuttingDown: state.shuttingDown,
    resyncRequired: state.resyncRequired,
    offline: true,
  });
  elements.pendingCard.removeAttribute("data-submission-scope");
  elements.pendingCard.removeAttribute("data-inspection-state");
  elements.pendingCard.removeAttribute("data-pending-count");
  elements.pendingHeading.textContent = "Inspection unavailable";
  elements.pendingCount.textContent = "0 actors";
  elements.pendingScope.textContent = "Waiting for authorized action details.";
  const pendingLabel = elements.pendingCard.querySelector(".action-card__label");
  if (pendingLabel) {
    pendingLabel.textContent = "NO AUTHORIZED INSPECTION";
  }
  elements.stepValue.textContent = "—";
  elements.transitionValue.textContent = "—";
  elements.audienceBadge.textContent = "View unavailable";
  elements.audienceBadge.dataset.audience = "unavailable";
  document.documentElement.dataset.audience = "unavailable";
  renderPendingPresentationChrome();
  elements.liveRangesButton.disabled = true;
  elements.resetButton.disabled = true;
  elements.exitButton.disabled = true;
  for (const control of [
    elements.recordingFinishButton,
    elements.recordingReviewButton,
    elements.recordingRetryButton,
    elements.recordingSaveAsInput,
    elements.recordingSaveAsButton,
    elements.recordingDiscardConfirmButton,
  ]) {
    control.disabled = true;
  }
  if (elements.recordingDiscardDialog.open) {
    elements.recordingDiscardDialog.close();
  }
  pendingRecordingReplacement = null;
  elements.commandControlledActor.textContent = "Actor · unavailable";
  elements.commandControlledActor.removeAttribute("aria-label");
  delete elements.commandControlledActor.dataset.controlledSlot;
  const emptyTarget = document.createElement("option");
  emptyTarget.value = "";
  emptyTarget.textContent = "No authorized targets";
  elements.commandTargetSelect.replaceChildren(emptyTarget);
  elements.commandTargetSelect.disabled = true;
  if (elements.commandDeck) {
    for (const button of elements.commandDeck.querySelectorAll("button")) {
      /** @type {HTMLButtonElement} */ (button).disabled = true;
      if (button.hasAttribute("data-authoritative-available")) {
        clearPresentationTooltipOwner(button);
        button.removeAttribute("data-authoritative-available");
        button.removeAttribute("data-selected");
        button.setAttribute("aria-disabled", "true");
        button.setAttribute("aria-pressed", "false");
      }
    }
  }
  elements.agentDetails.removeAttribute("data-tone");
  elements.agentDetails.removeAttribute("data-accent");
  elements.selectionHeading.textContent = AUTHORIZED_INSPECTOR_TITLE;
  elements.battlefield.removeAttribute("aria-activedescendant");
  document.documentElement.dataset.presentationAuthority = "pending";
}

/**
 * Begin one authority request under an explicit visual pending policy.
 * Ordinary same-audience work keeps only the last complete certified pair;
 * cross-audience changes clear synchronously, and later identity failures
 * fail closed before the next render.
 *
 * @param {string} reason
 * @param {"retain_last_authorized" | "clear"} pendingPolicy
 */
function beginPresentationAuthorityAttempt(reason, pendingPolicy) {
  if (
    pendingPolicy === "retain_last_authorized" &&
    installedPresentationAuthority() !== null
  ) {
    retainPresentationPreferenceAcrossBusyRender();
    tooltipController.hide();
    document.documentElement.dataset.presentationAuthority = "retained";
    return;
  }
  clearPresentationAuthority(reason);
}

/**
 * Keep the two-column workspace from collapsing while an authoritative
 * command synchronously clears its old scientific content. Without this
 * temporary layout floor, a focused native roster button near the document
 * boundary can move the page when the pending state removes HUD rows.
 */
function holdWorkspaceHeightDuringAuthorityInstall() {
  if (workspaceMinimumHeightBeforeAuthorityInstall !== null) {
    return;
  }
  const height = elements.workspace.getBoundingClientRect().height;
  if (!Number.isFinite(height) || height <= 0) {
    return;
  }
  workspaceMinimumHeightBeforeAuthorityInstall = elements.workspace.style.minHeight;
  elements.workspace.style.minHeight = `${Math.ceil(height)}px`;
}

/** Release the temporary pending-authority layout floor. */
function releaseWorkspaceHeightAfterAuthorityInstall() {
  if (workspaceMinimumHeightBeforeAuthorityInstall === null) {
    return;
  }
  const previous = workspaceMinimumHeightBeforeAuthorityInstall;
  workspaceMinimumHeightBeforeAuthorityInstall = null;
  if (previous.length === 0) {
    elements.workspace.style.removeProperty("min-height");
  } else {
    elements.workspace.style.minHeight = previous;
  }
}

/**
 * Install one already-joined, fully prepared authority candidate. The
 * coordinator guarantees generation freshness; this boundary guarantees the
 * display root still carries the strict normalizer's unforgeable brand.
 *
 * @param {Readonly<Record<string, any>>} joined
 */
function installJoinedAuthority(joined) {
  if (
    !isJoinedTransportAndAuthorizedPresentationV1(joined) ||
    !isRecord(joined.transport) ||
    !isAuthorizedPresentationFrame(joined.presentation) ||
    (joined.transport.viewer_mode === "replay" && !isRecord(joined.timeline))
  ) {
    throw new TypeError("Joined browser authority is incomplete or unbranded.");
  }
  assertFrameMatchesProductIdentity(joined.transport);
  const preferenceKey = authorizedPresentationPreferenceKey(joined.presentation);
  if (preferenceKey === null) {
    throw new TypeError("Authorized presentation has no certified preference key.");
  }
  installActivePresentationPreference(joined.presentation, preferenceKey);
  state.authority = joined;
  state.frame = joined.transport;
  state.presentation = joined.presentation;
  state.timeline = isRecord(joined.timeline) ? joined.timeline : null;
  presentationPreferenceNeedsContentRender = true;
  document.documentElement.dataset.presentationAuthority = "installed";
}

/**
 * Complete the transport/presentation pair with its audience-safe replay
 * timeline before it is eligible for the single install callback. A timeline
 * race is classified by A/B and therefore enters the same one-GET resync
 * budget; malformed timeline bytes fail without retry.
 *
 * @param {unknown} joined
 * @param {{
 *   previousAuthority?: Readonly<Record<string, any>> | null,
 *   continuityResult?: unknown,
 * }} options
 */
async function prepareJoinedAuthority(
  joined,
  { previousAuthority = null, continuityResult = "stale_resync" } = {},
) {
  if (!isJoinedTransportAndAuthorizedPresentationV1(joined)) {
    throw new TypeError(
      "Browser authority pair is not certified by the join boundary.",
    );
  }
  const certified = /** @type {Readonly<Record<string, any>>} */ (joined);
  const frame = certified.transport;
  assertFrameMatchesProductIdentity(frame);
  if (
    frame.viewer_mode === "replay" &&
    previousAuthority?.transport?.viewer_mode === "replay"
  ) {
    validateReplayTransportContinuityV1(previousAuthority, certified, continuityResult);
  }
  if (frame.viewer_mode !== "replay") {
    return certified;
  }
  const rawTimeline = await getReplayTimeline(state.token);
  return joinReplayTransportAndTimelineV1(certified, rawTimeline);
}

const presentationInstallation = new PresentationInstallCoordinator({
  onAttemptBegin: beginPresentationAuthorityAttempt,
  install: installJoinedAuthority,
  isJoinRace: isPresentationJoinRace,
});

const AUTHORIZED_INSPECTOR_TITLE = "Comprehensive Agent Class Details";

function applyAuthorizedInspectorChrome() {
  const inspector = authorizedInspectorView(
    state.presentation,
    installedLocalInspectedPresentationKey(state.presentation),
  );
  elements.selectionHeading.textContent =
    inspector?.title ?? AUTHORIZED_INSPECTOR_TITLE;
  elements.agentDetails.removeAttribute("data-tone");
  if (typeof inspector?.owner_class_accent === "string") {
    elements.agentDetails.dataset.accent = inspector.owner_class_accent;
  } else {
    elements.agentDetails.removeAttribute("data-accent");
  }
}

function reloadForProductHandoff() {
  if (productHandoffReloadRequested) {
    return;
  }
  productHandoffReloadRequested = true;
  window.location.reload();
}

/**
 * Recognize only an installed authorized compact agent activation owner. This
 * is the central authority fence behind each owner's non-inspectable tooltip
 * registration; other scientific owners retain generic full inspection.
 *
 * @param {Element} owner
 */
function isAuthorizedCompactAgentOwner(owner) {
  const presentation = state.presentation;
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    !installedAuthorityIsCoherent() ||
    (!owner.matches(".agent[data-presentation-key]") &&
      !owner.matches('[data-action="activate-agent"][data-presentation-key]'))
  ) {
    return false;
  }
  const presentationKey = owner.getAttribute("data-presentation-key");
  return (
    typeof presentationKey === "string" &&
    authorizedAgentForPresentationKey(presentation, presentationKey) !== null
  );
}

/**
 * @param {unknown} descriptor
 * @param {{owner: Element, trigger: Element | null}} context
 */
function showSemanticInspector(descriptor, context) {
  if (installedActivePresentationPreference(state.presentation) === null) {
    return;
  }
  const normalized = createSemanticDescriptor(descriptor);
  if (normalized.kind === "agent" && isAuthorizedCompactAgentOwner(context.owner)) {
    return;
  }
  if (
    isReplayMode() &&
    authorizedPresentationAudience(state.presentation) !== "researcher" &&
    normalized.kind === "agent"
  ) {
    const ownerKey = context.owner
      .closest("[data-presentation-key]")
      ?.getAttribute("data-presentation-key");
    const scene = installedAuthorizedPresentationSceneView(state.presentation);
    const inspectedKey = isRecord(scene?.selection)
      ? scene.selection.inspection_owner_presentation_key
      : null;
    if (typeof ownerKey !== "string" || ownerKey !== inspectedKey) {
      return;
    }
  }
  renderSemanticInspector(elements.selectionCard, normalized);
  applyAuthorizedInspectorChrome();
  openAgentDetails();
}

function registerControlHelp() {
  for (const [selector, title, summary, kind = "control"] of CONTROL_HELP) {
    for (const control of document.querySelectorAll(selector)) {
      registerControlHelpOwner(control, selector, title, summary, kind);
    }
  }
}

/**
 * @param {Element} control
 * @param {string} selector
 * @param {string} title
 * @param {string} summary
 * @param {string} [kind]
 */
function registerControlHelpOwner(control, selector, title, summary, kind = "control") {
  registerTooltipOwner(
    control,
    createSemanticDescriptor({
      kind,
      id: `control:${selector}:${title}`,
      title,
      tone: "information",
      accent: "none",
      summary,
      rows: [],
      sections: [],
      metadata: { compact: true, full: false },
      anchor: selector === "#battlefield" ? "pointer" : "element",
    }),
    { inspectable: false },
  );
}

function registerAuthorityAwareUtilityHelp() {
  const presentation = state.presentation;
  const installed =
    isAuthorizedPresentationFrame(presentation) && installedAuthorityIsCoherent();
  const audience = installed ? authorizedPresentationAudience(presentation) : null;
  const agentPov = audience === "agent_pov";
  const researcher = audience === "researcher";
  registerControlHelpOwner(
    elements.liveRangesButton,
    "#live-ranges-button",
    "Ranges",
    agentPov
      ? "Show or hide locally authorized range overlays. This sends no command and does not change the fixed recipient."
      : researcher
        ? "Toggle server-authored Oracle View range presentation."
        : "Range presentation is unavailable until one coherent authorized live frame is installed.",
  );
  registerControlHelpOwner(
    elements.replayRangesButton,
    "#replay-ranges-button",
    "Replay ranges",
    agentPov
      ? "Show or hide locally authorized inspected-agent range overlays. This sends no replay command and does not change the fixed recipient."
      : researcher
        ? "Toggle recorded Oracle View range presentation."
        : "Replay range presentation is unavailable until one coherent authorized frame is installed.",
  );
  registerControlHelpOwner(
    elements.replayClearReferenceButton,
    "#replay-clear-reference-button",
    "Clear Selection",
    agentPov
      ? "Clear the local inspected-agent highlight, details, and ranges. This sends no replay command and does not change the fixed recipient."
      : researcher
        ? "Clear the selected Oracle View agent and its inspection highlight."
        : "Selection cannot be cleared until one coherent authorized replay frame is installed.",
  );
  for (const [selector, title, oracleSummary] of [
    [
      "[data-key='Tab']:not([data-shift])",
      "Next actor",
      "Move Oracle View control to the next active actor.",
    ],
    [
      "[data-key='Tab'][data-shift='true']",
      "Previous actor",
      "Move Oracle View control to the previous active actor.",
    ],
  ]) {
    for (const control of document.querySelectorAll(selector)) {
      const oracleActorCyclingAvailable =
        researcher && control instanceof HTMLButtonElement && !control.disabled;
      registerControlHelpOwner(
        control,
        selector,
        title,
        agentPov
          ? "Actor cycling is unavailable in Agent POV. Tab and Shift+Tab keep native browser focus and never change the fixed recipient."
          : oracleActorCyclingAvailable
            ? oracleSummary
            : researcher
              ? "Actor cycling is unavailable in the current Oracle View state. Tab and Shift+Tab retain native browser focus navigation."
              : "Actor cycling is unavailable until one coherent authorized live frame is installed.",
      );
    }
  }
}

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function cancelPendingPresentationPreferenceRestore() {
  if (pendingPresentationPreferenceRestoreFrame === null) {
    return;
  }
  window.cancelAnimationFrame(pendingPresentationPreferenceRestoreFrame);
  pendingPresentationPreferenceRestoreFrame = null;
}

/**
 * Native details toggle events are task-queued and may coalesce. Record the
 * expected final state before assigning `.open`; a synchronous boolean would
 * be cleared before the browser delivers the event.
 *
 * @param {HTMLDetailsElement} panel
 * @param {boolean} open
 */
function setProgrammaticDisclosureOpen(panel, open) {
  if (panel.open === open) {
    return;
  }
  expectedDisclosureToggles.set(panel, Object.freeze({ open }));
  panel.open = open;
}

/**
 * @param {boolean} available
 * @param {{preserveOpen?: boolean}} [options]
 */
function setScientificDisclosureAvailability(available, { preserveOpen = false } = {}) {
  for (const { panel, body } of scientificDisclosures) {
    const summary = panel.querySelector(":scope > summary");
    if (!available) {
      if (!preserveOpen) {
        setProgrammaticDisclosureOpen(panel, false);
      }
      panel.setAttribute("inert", "");
      body.setAttribute("inert", "");
      if (summary instanceof HTMLElement) {
        summary.setAttribute("aria-disabled", "true");
        summary.setAttribute("tabindex", "-1");
      }
      continue;
    }
    panel.removeAttribute("inert");
    body.removeAttribute("inert");
    if (summary instanceof HTMLElement) {
      summary.removeAttribute("aria-disabled");
      summary.removeAttribute("tabindex");
    }
  }
}

/**
 * @param {unknown} presentation
 * @returns {typeof activePresentationPreference}
 */
function installedActivePresentationPreference(presentation) {
  if (
    activePresentationPreference === null ||
    !isAuthorizedPresentationFrame(presentation) ||
    !installedAuthorityIsCoherent() ||
    state.presentation !== presentation
  ) {
    return null;
  }
  const installedKey = authorizedPresentationPreferenceKey(presentation);
  return installedKey !== null &&
    sameAuthorizedPresentationPreferenceKey(
      activePresentationPreference.authorityKey,
      installedKey,
    )
    ? activePresentationPreference
    : null;
}

/**
 * @param {Readonly<Record<string, any>>} presentation
 * @param {Readonly<{tuple: readonly unknown[], serialized: string}>} authorityKey
 * @returns {NonNullable<typeof activePresentationPreference>}
 */
function defaultPresentationPreference(presentation, authorityKey) {
  /** @type {Record<string, {open: boolean, scrollTop: number}>} */
  const disclosures = {};
  const replay = presentation.product_kind === "replay_viewer";
  for (const { panelId } of scientificDisclosures) {
    disclosures[panelId] = {
      open: disclosurePanelInitiallyOpen(panelId, replay),
      scrollTop: 0,
    };
  }
  const audience = authorizedPresentationAudience(presentation);
  const ownsLocalInspection =
    audience === "agent_pov" ||
    authorizedPresentationInspectionState(presentation).state_kind === "live_scripted";
  const recipientKey =
    audience === "agent_pov" ? presentation.authority.recipient_presentation_key : null;
  const inspectedPresentationKey =
    typeof recipientKey === "string" &&
    authorizedAgentForPresentationKey(presentation, recipientKey) !== null
      ? recipientKey
      : null;
  return {
    authorityKey,
    disclosures,
    agentDetailsAutoOpenAllowed: true,
    primaryFocus: null,
    localInspection: ownsLocalInspection
      ? { owned: true, presentationKey: inspectedPresentationKey }
      : { owned: false },
    localRanges: ownsLocalInspection
      ? { owned: true, visible: true }
      : { owned: false },
  };
}

/**
 * Re-resolve retained opaque keys exclusively through the new certified scene.
 * A disappeared local selection becomes explicit null and never falls back to
 * the recipient merely because the authority tuple is unchanged.
 *
 * @param {NonNullable<typeof activePresentationPreference>} preference
 * @param {Readonly<Record<string, any>>} presentation
 */
function reconcilePresentationPreference(preference, presentation) {
  if (
    preference.localInspection.owned &&
    preference.localInspection.presentationKey !== null &&
    authorizedAgentForPresentationKey(
      presentation,
      preference.localInspection.presentationKey,
    ) === null
  ) {
    invalidateReplayArtifactAction();
    preference.localInspection.presentationKey = null;
    preference.disclosures["agent-details"].open = false;
  }
  if (
    preference.primaryFocus !== null &&
    authorizedAgentForPresentationKey(
      presentation,
      preference.primaryFocus.presentationKey,
    ) === null
  ) {
    preference.primaryFocus = null;
  }
}

/**
 * @param {Readonly<Record<string, any>>} presentation
 * @param {Readonly<{tuple: readonly unknown[], serialized: string}>} authorityKey
 */
function installActivePresentationPreference(presentation, authorityKey) {
  presentationPreferenceGeneration += 1;
  cancelPendingPresentationPreferenceRestore();
  if (
    activePresentationPreference === null ||
    !sameAuthorizedPresentationPreferenceKey(
      activePresentationPreference.authorityKey,
      authorityKey,
    )
  ) {
    activePresentationPreference = defaultPresentationPreference(
      presentation,
      authorityKey,
    );
  } else {
    reconcilePresentationPreference(activePresentationPreference, presentation);
  }
}

/**
 * @param {Readonly<Record<string, any>>} presentation
 * @returns {null | {surface: "roster" | "battlefield", presentationKey: string}}
 */
function focusedAuthorizedPrimaryAction(presentation) {
  const active = document.activeElement;
  if (!(active instanceof Element)) {
    return null;
  }
  const rosterAction = active.closest(
    "#roster .roster-primary-action[data-presentation-key]",
  );
  const battlefieldAction = active.closest(
    "#battlefield .agent[data-presentation-key]",
  );
  const action = rosterAction ?? battlefieldAction;
  const presentationKey = action?.getAttribute("data-presentation-key");
  if (
    typeof presentationKey !== "string" ||
    authorizedAgentForPresentationKey(presentation, presentationKey) === null
  ) {
    return null;
  }
  return {
    surface: rosterAction !== null ? "roster" : "battlefield",
    presentationKey,
  };
}

/**
 * @param {NonNullable<typeof activePresentationPreference>} preference
 * @param {Readonly<Record<string, any>>} presentation
 */
function capturePresentationPreference(preference, presentation) {
  for (const { panelId, panel, body } of scientificDisclosures) {
    const saved = preference.disclosures[panelId];
    const expected = expectedDisclosureToggles.get(panel);
    const userClosedBeforeToggle =
      saved?.open === true && !panel.open && expected?.open !== false;
    if (panelId === "agent-details" && userClosedBeforeToggle) {
      preference.agentDetailsAutoOpenAllowed = false;
    }
    preference.disclosures[panelId] = {
      open: panel.open,
      scrollTop:
        panel.open || userClosedBeforeToggle ? body.scrollTop : (saved?.scrollTop ?? 0),
    };
  }
  preference.primaryFocus = focusedAuthorizedPrimaryAction(presentation);
}

function savePresentationPreferenceBeforeClear() {
  const presentation = state.presentation;
  const preference = installedActivePresentationPreference(presentation);
  if (preference === null || !isAuthorizedPresentationFrame(presentation)) {
    return;
  }
  capturePresentationPreference(preference, presentation);
  const active = document.activeElement;
  const presentationOwnedFocus =
    active instanceof Element &&
    (elements.battlefield.contains(active) ||
      scientificDisclosures.some(({ panel }) => panel.contains(active)));
  if (!presentationOwnedFocus) {
    pendingPrimaryFocusRestoreAnchor = null;
    return;
  }
  elements.helpButton.focus({ preventScroll: true });
  pendingPrimaryFocusRestoreAnchor =
    preference.primaryFocus === null ? null : elements.helpButton;
}

function enterPendingPresentationPreferenceState() {
  presentationPreferenceGeneration += 1;
  presentationPreferenceNeedsContentRender = false;
  cancelPendingPresentationPreferenceRestore();
  setScientificDisclosureAvailability(false);
}

function capturePresentationPreferenceBeforeRender() {
  if (
    presentationPreferenceNeedsContentRender ||
    pendingPresentationPreferenceRestoreFrame !== null
  ) {
    return;
  }
  const presentation = state.presentation;
  const preference = installedActivePresentationPreference(presentation);
  if (preference === null || !isAuthorizedPresentationFrame(presentation)) {
    return;
  }
  capturePresentationPreference(preference, presentation);
  pendingPrimaryFocusRestoreAnchor =
    preference.primaryFocus === null ? null : document.activeElement;
}

/**
 * Preserve the exact keyboard-owned primary action while a same-authority
 * request repaints the retained scene into its inert busy state. The eventual
 * complete successor (or retained failure state) restores that focus only if
 * the user has not moved it elsewhere in the meantime.
 */
function retainPresentationPreferenceAcrossBusyRender() {
  const presentation = state.presentation;
  const preference = installedActivePresentationPreference(presentation);
  if (preference === null || !isAuthorizedPresentationFrame(presentation)) {
    return;
  }
  capturePresentationPreference(preference, presentation);
  if (preference.primaryFocus === null) {
    pendingPrimaryFocusRestoreAnchor = null;
  } else {
    elements.helpButton.focus({ preventScroll: true });
    pendingPrimaryFocusRestoreAnchor = elements.helpButton;
  }
  presentationPreferenceNeedsContentRender = true;
}

/**
 * @param {NonNullable<typeof activePresentationPreference>} preference
 */
function schedulePresentationPreferenceRestore(preference) {
  cancelPendingPresentationPreferenceRestore();
  const generation = presentationPreferenceGeneration;
  const authorityKey = preference.authorityKey;
  pendingPresentationPreferenceRestoreFrame = window.requestAnimationFrame(() => {
    pendingPresentationPreferenceRestoreFrame = null;
    const installed = installedActivePresentationPreference(state.presentation);
    if (
      generation !== presentationPreferenceGeneration ||
      installed === null ||
      !sameAuthorizedPresentationPreferenceKey(installed.authorityKey, authorityKey)
    ) {
      return;
    }
    try {
      for (const { panelId, panel, body } of scientificDisclosures) {
        const saved = installed.disclosures[panelId];
        if (panel.open && saved?.open === true) {
          body.scrollTop = saved.scrollTop;
        }
      }
      const focus = installed.primaryFocus;
      const anchor = pendingPrimaryFocusRestoreAnchor;
      pendingPrimaryFocusRestoreAnchor = null;
      if (focus === null || anchor === null) {
        return;
      }
      const active = document.activeElement;
      const focusWasNotMovedByUser =
        active === anchor || (!anchor.isConnected && active === document.body);
      if (
        !focusWasNotMovedByUser ||
        authorizedAgentForPresentationKey(state.presentation, focus.presentationKey) ===
          null
      ) {
        return;
      }
      const root = focus.surface === "roster" ? elements.roster : elements.battlefield;
      const selector =
        focus.surface === "roster"
          ? `.roster-primary-action[data-presentation-key="${CSS.escape(focus.presentationKey)}"]`
          : `.agent[data-presentation-key="${CSS.escape(focus.presentationKey)}"]`;
      const target = root.querySelector(selector);
      if (
        !(target instanceof HTMLElement || target instanceof SVGElement) ||
        target.getAttribute("aria-disabled") === "true" ||
        target.getAttribute("tabindex") === "-1" ||
        (target instanceof HTMLButtonElement && target.disabled)
      ) {
        return;
      }
      target.focus({ preventScroll: true });
    } finally {
      releaseWorkspaceHeightAfterAuthorityInstall();
    }
  });
}

function restorePresentationPreferenceAfterRender() {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null) {
    setScientificDisclosureAvailability(false);
    if (!state.busy) {
      releaseWorkspaceHeightAfterAuthorityInstall();
    }
    return;
  }
  if (state.busy) {
    setScientificDisclosureAvailability(false, { preserveOpen: true });
    return;
  }
  setScientificDisclosureAvailability(true);
  for (const { panelId, panel } of scientificDisclosures) {
    setProgrammaticDisclosureOpen(
      panel,
      preference.disclosures[panelId]?.open === true,
    );
  }
  presentationPreferenceNeedsContentRender = false;
  schedulePresentationPreferenceRestore(preference);
}

function openAgentDetails() {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null || !preference.agentDetailsAutoOpenAllowed) {
    return false;
  }
  preference.disclosures["agent-details"].open = true;
  setProgrammaticDisclosureOpen(elements.agentDetails, true);
  return true;
}

function closeAgentDetailsWithoutLatching() {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null) {
    return;
  }
  preference.disclosures["agent-details"].open = false;
  setProgrammaticDisclosureOpen(elements.agentDetails, false);
}

function isReplayMode() {
  return productIdentity?.product_kind === "replay_viewer";
}

/** @param {typeof visualFilterState} snapshot */
function renderVisualFilterControls(snapshot) {
  const inputs = /** @type {NodeListOf<HTMLInputElement>} */ (
    elements.visualFilterOptions.querySelectorAll(
      'input[type="checkbox"][data-visual-filter-id]',
    )
  );
  if (inputs.length !== EXPECTED_VISUAL_FILTER_COUNT) {
    throw new TypeError(
      `Visual Filters requires exactly ${EXPECTED_VISUAL_FILTER_COUNT} checkboxes.`,
    );
  }
  let enabledCount = 0;
  for (const input of inputs) {
    const enabled = isVisualFilterEnabled(snapshot, input.dataset.visualFilterId);
    input.checked = enabled;
    enabledCount += enabled ? 1 : 0;
  }
  elements.visualFilterCount.textContent = `${enabledCount} enabled`;
  elements.enableAllVisualFiltersButton.disabled =
    enabledCount === EXPECTED_VISUAL_FILTER_COUNT;
  elements.disableAllVisualFiltersButton.disabled = enabledCount === 0;
}

/**
 * Replace the whole local snapshot before exposing it to either painter. Replay
 * playback is paused locally so the selected tick cannot advance underneath a
 * presentation-only filter change.
 *
 * @param {unknown} action
 * @returns {boolean} Whether the local snapshot changed.
 */
function applyVisualFilterAction(action) {
  const next = reduceVisualFilterState(visualFilterState, action);
  if (next === visualFilterState) {
    return false;
  }
  invalidateReplayArtifactAction();
  visualFilterState = next;
  if (isReplayMode()) {
    replayPlayback.pause("visual_filter_changed");
  }
  render();
  return true;
}

/**
 * Derive the presentation-only choreography policy from the coherent installed
 * product and transport pair. Scientific presentation data never selects how
 * its own events animate.
 *
 * @param {unknown} presentation
 * @param {typeof visualFilterState} visualFilters
 * @param {{consumeAnimatedRestart?: boolean}} [options]
 * @returns {Readonly<{
 *   renderPolicy: "live_once" | "replay_animated" | "replay_static",
 *   visualFilters: typeof visualFilterState,
 *   restartAnimated?: true,
 * }>}
 */
function installedChoreographyControl(
  presentation,
  visualFilters,
  { consumeAnimatedRestart = false } = {},
) {
  const authority = state.authority;
  const coherentInstalledPair =
    isAuthorizedPresentationFrame(presentation) &&
    isJoinedTransportAndAuthorizedPresentationV1(authority) &&
    authority.presentation === presentation &&
    authority.transport === state.frame;
  const replayInstalled =
    coherentInstalledPair &&
    isReplayMode() &&
    authority.transport.viewer_mode === "replay";
  if (!replayInstalled) {
    return Object.freeze({
      renderPolicy: "live_once",
      visualFilters,
    });
  }

  const playback = replayPlayback.snapshot();
  const intent = playback.presentationIntent;
  const currentIntent =
    isRecord(intent) &&
    replayCursorsMatch(playback.cursor, authority.transport.cursor) &&
    playback.transportState !== "OFFLINE";
  const renderPolicy = currentIntent ? intent.renderPolicy : "replay_static";
  const restartAnimated =
    consumeAnimatedRestart &&
    renderPolicy === "replay_animated" &&
    intent?.restartAnimated === true &&
    intent.generation > consumedReplayRestartGeneration;
  if (restartAnimated) {
    consumedReplayRestartGeneration = intent.generation;
  }
  return Object.freeze({
    renderPolicy,
    visualFilters,
    ...(restartAnimated ? { restartAnimated: true } : {}),
  });
}

/** @param {unknown} left @param {unknown} right */
function replayCursorsMatch(left, right) {
  if (!isRecord(left) || !isRecord(right)) {
    return false;
  }
  return (
    left.schema_version === right.schema_version &&
    left.frame_index === right.frame_index &&
    left.final_frame_index === right.final_frame_index &&
    left.cursor_generation === right.cursor_generation &&
    left.choreography_generation === right.choreography_generation
  );
}

function installedPresentationAuthority() {
  return resolveInstalledPresentationAuthorityV1(
    state.authority,
    state.frame,
    state.presentation,
  );
}

function installedAuthorityIsCoherent() {
  return installedPresentationAuthority() !== null;
}

/**
 * Resolve an opaque presentation key only through the certified scene that is
 * installed now. DOM order, raw slots, and a previous presentation are never
 * identity authority.
 *
 * @param {unknown} presentation
 * @param {unknown} presentationKey
 * @returns {Readonly<Record<string, any>> | null}
 */
function authorizedAgentForPresentationKey(presentation, presentationKey) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    typeof presentationKey !== "string"
  ) {
    return null;
  }
  return (
    asArray(authorizedPresentationSceneView(presentation)?.agents).find(
      (candidate) =>
        isRecord(candidate) && candidate.presentation_key === presentationKey,
    ) ?? null
  );
}

/**
 * @param {unknown} presentation
 * @returns {Readonly<{
 *   inspectedPresentationKey: string | null,
 *   rangesVisible: boolean,
 * }> | null}
 */
function installedLocalPresentationPreference(presentation) {
  if (!isAuthorizedPresentationFrame(presentation)) {
    return null;
  }
  const preference = installedActivePresentationPreference(presentation);
  return preference?.localInspection.owned && preference.localRanges.owned
    ? Object.freeze({
        inspectedPresentationKey: preference.localInspection.presentationKey,
        rangesVisible: preference.localRanges.visible,
      })
    : null;
}

/**
 * Undefined preserves the accepted Oracle view; null is Agent POV's explicit
 * no-selection state.
 *
 * @param {unknown} presentation
 * @returns {string | null | undefined}
 */
function installedLocalInspectedPresentationKey(presentation) {
  return installedLocalPresentationPreference(presentation)?.inspectedPresentationKey;
}

/** @param {unknown} presentation */
function installedAuthorizedPresentationSceneView(presentation) {
  return authorizedPresentationSceneView(
    presentation,
    installedLocalInspectedPresentationKey(presentation),
  );
}

/**
 * Update only the active authority's inert local selection. A stale or
 * disappeared key is rejected without falling back to the recipient or
 * retaining old facts.
 *
 * @param {string | null} presentationKey
 * @returns {boolean}
 */
function setLocalInspectedPresentationKey(presentationKey) {
  const presentation = state.presentation;
  const preference = installedActivePresentationPreference(presentation);
  if (
    preference === null ||
    !preference.localInspection.owned ||
    (presentationKey !== null &&
      authorizedAgentForPresentationKey(presentation, presentationKey) === null)
  ) {
    return false;
  }
  invalidateReplayArtifactAction();
  preference.localInspection.presentationKey = presentationKey;
  return true;
}

/** @returns {boolean} */
function toggleAgentLocalRanges() {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null || !preference.localRanges.owned) {
    return false;
  }
  invalidateReplayArtifactAction();
  preference.localRanges.visible = !preference.localRanges.visible;
  return true;
}

/** @param {unknown} presentation */
function installedPresentationRangesVisible(presentation) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    !installedAuthorityIsCoherent() ||
    state.presentation !== presentation
  ) {
    return false;
  }
  const localPreference = installedLocalPresentationPreference(presentation);
  return authorizedPresentationAudience(presentation) === "researcher"
    ? state.frame?.show_ranges === true
    : localPreference?.rangesVisible === true;
}

/**
 * Project the exact public capability state from one coherent authority and
 * one CP8 controller snapshot. Artifact actions belong to the local researcher
 * tool, so visual POV changes do not remove them.
 *
 * @param {{
 *   replayProduct: boolean,
 *   coherentAuthority: boolean,
 *   audience: string | null,
 *   transportState: string,
 *   connected: boolean,
 *   hidden: boolean,
 *   playing: boolean,
 *   requestPending: boolean,
 *   presentationPending: boolean,
 *   renderPolicy: string | null,
 *   cursorMatches: boolean,
 *   operationallyBlocked: boolean,
 *   actionPending: boolean,
 *   battlefieldReady: boolean,
 * }} input
 */
function replayArtifactActionCapabilities(input) {
  const settled =
    input.replayProduct &&
    input.coherentAuthority &&
    (input.audience === "researcher" || input.audience === "agent_pov") &&
    input.transportState === "SETTLED" &&
    input.connected &&
    !input.hidden &&
    !input.playing &&
    !input.requestPending &&
    !input.presentationPending &&
    input.renderPolicy === "replay_static" &&
    input.cursorMatches &&
    !input.operationallyBlocked &&
    !input.actionPending;
  return Object.freeze({
    exportPng: settled && input.battlefieldReady,
    downloadMetrics: settled,
  });
}

function replayBattlefieldReady() {
  return (
    elements.battlefield.id === "battlefield" &&
    elements.battlefield.dataset.renderPolicy === "replay_static" &&
    elements.battlefieldShell.getAttribute("aria-busy") === "false" &&
    Number.isInteger(elements.battlefield.clientWidth) &&
    elements.battlefield.clientWidth > 0 &&
    Number.isInteger(elements.battlefield.clientHeight) &&
    elements.battlefield.clientHeight > 0
  );
}

/**
 * @param {ReturnType<typeof installedPresentationAuthority>} installed
 * @param {{ignoreActionPending?: boolean}} [options]
 */
function currentReplayArtifactActionCapabilities(
  installed,
  { ignoreActionPending = false } = {},
) {
  const playback = replayPlayback.snapshot();
  return replayArtifactActionCapabilities({
    replayProduct: isReplayMode(),
    coherentAuthority: installed !== null,
    audience:
      installed === null
        ? null
        : (authorizedPresentationAudience(installed.presentation) ?? null),
    transportState: playback.transportState,
    connected: playback.connected,
    hidden: playback.hidden,
    playing: playback.playing,
    requestPending: playback.requestPending,
    presentationPending: playback.presentationPending,
    renderPolicy: playback.presentationIntent?.renderPolicy ?? null,
    cursorMatches:
      installed !== null &&
      replayCursorsMatch(playback.cursor, installed.transport.cursor),
    operationallyBlocked:
      state.busy || state.shuttingDown || state.resyncRequired || state.offline,
    actionPending: !ignoreActionPending && replayArtifactActionTransaction !== null,
    battlefieldReady: replayBattlefieldReady(),
  });
}

/** @param {ReturnType<typeof installedPresentationAuthority>} installed */
function renderReplayArtifactActions(installed) {
  const capabilities = currentReplayArtifactActionCapabilities(installed);
  const pending = replayArtifactActionTransaction;
  elements.replayExportPngButton.disabled = !capabilities.exportPng;
  elements.replayDownloadMetricsButton.disabled = !capabilities.downloadMetrics;
  elements.replayExportPngButton.setAttribute(
    "aria-busy",
    String(pending?.kind === "export_png"),
  );
  elements.replayDownloadMetricsButton.setAttribute(
    "aria-busy",
    String(pending?.kind === "download_metrics"),
  );
}

/**
 * Capture one exact action epoch before any asynchronous export or GET work.
 *
 * @param {"export_png" | "download_metrics"} kind
 */
function beginReplayArtifactAction(kind) {
  if (replayArtifactActionTransaction !== null) {
    return null;
  }
  const installed = installedPresentationAuthority();
  const capabilities = currentReplayArtifactActionCapabilities(installed);
  if (
    installed === null ||
    (kind === "export_png" ? !capabilities.exportPng : !capabilities.downloadMetrics)
  ) {
    return null;
  }
  const authority = state.authority;
  if (
    !isJoinedTransportAndAuthorizedPresentationV1(authority) ||
    authority.transport !== installed.transport ||
    authority.presentation !== installed.presentation
  ) {
    return null;
  }
  const localInspectedPresentationKey =
    installedLocalInspectedPresentationKey(installed.presentation) ?? null;
  const transaction = Object.freeze({
    kind,
    authority,
    transport: installed.transport,
    presentation: installed.presentation,
    playbackGeneration: replayPlayback.snapshot().generation,
    localInspectedPresentationKey,
    showRanges: installedPresentationRangesVisible(installed.presentation),
    visualFilters: visualFilterState,
  });
  replayArtifactActionTransaction = transaction;
  renderReplayArtifactActions(installed);
  return transaction;
}

/**
 * @param {NonNullable<typeof replayArtifactActionTransaction>} transaction
 */
function replayArtifactActionIsCurrent(transaction) {
  if (replayArtifactActionTransaction !== transaction) {
    return false;
  }
  const installed = installedPresentationAuthority();
  if (
    installed === null ||
    state.authority !== transaction.authority ||
    installed.transport !== transaction.transport ||
    installed.presentation !== transaction.presentation
  ) {
    return false;
  }
  const capabilities = currentReplayArtifactActionCapabilities(installed, {
    ignoreActionPending: true,
  });
  if (
    replayPlayback.snapshot().generation !== transaction.playbackGeneration ||
    (transaction.kind === "export_png"
      ? !capabilities.exportPng
      : !capabilities.downloadMetrics)
  ) {
    return false;
  }
  if (transaction.kind === "export_png") {
    return (
      visualFilterState === transaction.visualFilters &&
      (installedLocalInspectedPresentationKey(installed.presentation) ?? null) ===
        transaction.localInspectedPresentationKey &&
      installedPresentationRangesVisible(installed.presentation) ===
        transaction.showRanges
    );
  }
  return true;
}

/** @param {NonNullable<typeof replayArtifactActionTransaction>} transaction */
function finishReplayArtifactAction(transaction) {
  if (replayArtifactActionTransaction !== transaction) {
    return;
  }
  replayArtifactActionTransaction = null;
  renderReplayArtifactActions(installedPresentationAuthority());
}

function invalidateReplayArtifactAction() {
  replayArtifactActionTransaction = null;
}

/** @param {Blob} blob @param {string} filename */
function downloadReplayArtifact(blob, filename) {
  if (
    !(blob instanceof Blob) ||
    typeof filename !== "string" ||
    filename.length < 1 ||
    filename.length > 240 ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(filename)
  ) {
    throw new TypeError("Replay download artifact is invalid.");
  }
  const objectUrl = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = filename;
    anchor.rel = "noopener";
    anchor.click();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

/** @param {unknown} error */
function replayMetricDownloadError(error) {
  if (error instanceof DebuggerApiError && error.status === 404) {
    return Object.freeze({
      message: "No metric report is available for this replay.",
      level: "warning",
    });
  }
  if (error instanceof DebuggerApiError && error.status === 403) {
    return Object.freeze({
      message: "Metric download is not authorized for this replay session.",
      level: "error",
    });
  }
  return Object.freeze({
    message:
      error instanceof Error
        ? `Metric download failed: ${error.message}`
        : "Metric download failed.",
    level: "error",
  });
}

async function exportReplayBattlefieldPng() {
  const transaction = beginReplayArtifactAction("export_png");
  if (transaction === null) {
    return;
  }
  try {
    const playback = replayPlayback.snapshot();
    const artifact = await captureReplayBattlefieldPngV1({
      battlefield: elements.battlefield,
      installedAuthority: transaction.authority,
      isCurrent: () => replayArtifactActionIsCurrent(transaction),
      transportState: playback.transportState,
      renderPolicy: playback.presentationIntent?.renderPolicy ?? null,
      showRanges: transaction.showRanges,
      localInspectedPresentationKey: transaction.localInspectedPresentationKey,
      visualFilters: transaction.visualFilters,
    });
    if (!replayArtifactActionIsCurrent(transaction)) {
      return;
    }
    if (artifact.schemaVersion !== 1 || artifact.blob.type !== "image/png") {
      throw new TypeError("Replay PNG builder returned an invalid artifact.");
    }
    downloadReplayArtifact(artifact.blob, artifact.filename);
    setNotice(`Exported ${artifact.filename}.`, "success");
    renderConnection();
  } catch (error) {
    if (!replayArtifactActionIsCurrent(transaction)) {
      return;
    }
    setNotice(
      error instanceof Error
        ? `PNG export failed: ${error.message}`
        : "PNG export failed.",
      "error",
    );
    renderConnection();
  } finally {
    finishReplayArtifactAction(transaction);
  }
}

async function downloadReplayMetricReport() {
  const transaction = beginReplayArtifactAction("download_metrics");
  if (transaction === null) {
    return;
  }
  try {
    const report = await getReplayMetricReport(state.token);
    if (!replayArtifactActionIsCurrent(transaction)) {
      return;
    }
    downloadReplayArtifact(
      new Blob([report.bytes], { type: "application/json; charset=utf-8" }),
      report.filename,
    );
    setNotice(`Downloaded ${report.filename}.`, "success");
    renderConnection();
  } catch (error) {
    if (!replayArtifactActionIsCurrent(transaction)) {
      return;
    }
    const failure = replayMetricDownloadError(error);
    setNotice(failure.message, failure.level);
    renderConnection();
  } finally {
    finishReplayArtifactAction(transaction);
  }
}

/**
 * Recognize only the installed branded scripted-live pair. Every passive
 * interaction fence shares this one predicate so raw transport metadata or an
 * unjoined presentation can never activate scripted controls.
 */
function liveScriptedInspectionOnly() {
  return (
    installedAuthorityIsCoherent() &&
    (state.frame?.frame_kind === "researcher_live_debugger" ||
      state.frame?.frame_kind === "actor_pov_live_debugger") &&
    authorizedPresentationInspectionState(state.presentation).state_kind ===
      "live_scripted"
  );
}

/** @param {Record<string, unknown>} command */
function allowedDuringLiveScriptedInspection(command) {
  if (command.command_type === "set_view") {
    return true;
  }
  if (SCRIPTED_INSPECTION_RECORDING_COMMANDS.has(String(command.command_type))) {
    return recordingCommandDecision(state.frame ?? {}, command).action === "allow";
  }
  if (
    command.command_type !== "keyboard" ||
    (command.key !== "n" && command.key !== "g")
  ) {
    return false;
  }
  return (
    command.shift_key === false &&
    command.ctrl_key === false &&
    command.alt_key === false &&
    command.meta_key === false &&
    command.repeat === false
  );
}

/**
 * Select the presentation variant's authorized incoming-transition identity.
 * Frame zero has no latest-events branch and therefore no incoming identity.
 *
 * @param {unknown} presentation
 * @returns {string | null}
 */
function authorizedIncomingTransitionId(presentation) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    !isRecord(presentation.latest_events)
  ) {
    return null;
  }
  const incomingTransitionId =
    authorizedPresentationAudience(presentation) === "researcher"
      ? presentation.latest_events.incoming_transition_id
      : presentation.latest_events.incoming_recipient_transition_id;
  return typeof incomingTransitionId === "string" && incomingTransitionId.length > 0
    ? incomingTransitionId
    : null;
}

function recordingStatus() {
  const frame = installedPresentationAuthority()?.transport;
  return isRecord(frame?.recording) ? frame.recording : null;
}

function recordingScientificControlsFenced() {
  const recording = recordingStatus();
  return recording !== null && recording.lifecycle !== "recording";
}

function recordingRestartControlsBlocked() {
  const recording = recordingStatus();
  return (
    recording !== null &&
    recording.restart_fenced === true &&
    recording.discard_available !== true
  );
}

const SCRIPTED_BATTLEFIELD_LABEL =
  "Inspection-only scripted battlefield. Authorized bodies can be inspected; use Advance scripted frame for the next authorized step.";
const SCRIPTED_BATTLEFIELD_INSTRUCTIONS =
  "Scripted live view is inspection-only. Activate an authorized body to inspect current facts; use Advance scripted frame for the next authorized step.";
const FENCED_LIVE_BATTLEFIELD_LABEL =
  "Read-only live battlefield. Simulator and actor activation controls are unavailable.";
const FENCED_LIVE_BATTLEFIELD_INSTRUCTIONS =
  "Live battlefield interaction is unavailable while authority is pending, offline, resynchronizing, shutting down, or terminal.";
const READ_ONLY_LIVE_SCIENTIFIC_LABEL =
  "Read-only live battlefield. Scientific facts can be inspected; simulator and actor activation controls are unavailable.";
const READ_ONLY_LIVE_SCIENTIFIC_INSTRUCTIONS =
  "Scientific tooltip facts remain inspectable. Live simulator and actor activation controls are unavailable while recording is closing, the session is offline or resynchronizing, or the frame is terminal.";
const AGENT_CLOSEOUT_BATTLEFIELD_LABEL =
  "Agent POV recording closeout battlefield. Authorized bodies can be inspected; simulator controls are unavailable.";
const AGENT_CLOSEOUT_BATTLEFIELD_INSTRUCTIONS =
  "Agent POV keeps one fixed recipient. Activate an authorized visible body to inspect current facts; recording closeout has fenced simulator and pending-action controls.";
const TERMINAL_REPLAY_BATTLEFIELD_LABEL =
  "Read-only terminal replay battlefield snapshot.";
const TERMINAL_REPLAY_BATTLEFIELD_INSTRUCTIONS =
  "This replay frame is terminal. Activate an agent to inspect current facts, or use the timeline to review another frame.";
const TERMINAL_REPLAY_AGENT_BATTLEFIELD_INSTRUCTIONS =
  "This replay frame is terminal. Activate a visible body or choose any agent in the roster to switch the fog-of-war recipient; use the timeline to review another frame.";

function liveBattlefieldCommandsInteractive() {
  return (
    !isReplayMode() &&
    isAuthorizedPresentationFrame(state.presentation) &&
    installedAuthorityIsCoherent() &&
    !liveScriptedInspectionOnly() &&
    !isTerminal(state.frame) &&
    !state.busy &&
    !state.shuttingDown &&
    !state.resyncRequired &&
    !state.offline &&
    !recordingScientificControlsFenced()
  );
}

function scriptedBattlefieldLocallyActionable() {
  return (
    liveScriptedInspectionOnly() &&
    !isTerminal(state.frame) &&
    !state.busy &&
    !state.shuttingDown &&
    !state.resyncRequired &&
    !state.offline &&
    !recordingScientificControlsFenced()
  );
}

function liveAgentCloseoutInspectionActionable() {
  return (
    !isReplayMode() &&
    authorizedPresentationAudience(state.presentation) === "agent_pov" &&
    isAuthorizedPresentationFrame(state.presentation) &&
    installedAuthorityIsCoherent() &&
    recordingScientificControlsFenced() &&
    !isTerminal(state.frame) &&
    !state.busy &&
    !state.shuttingDown &&
    !state.resyncRequired &&
    !state.offline
  );
}

function installedLiveScientificInspectionAvailable() {
  return (
    !isReplayMode() &&
    isAuthorizedPresentationFrame(state.presentation) &&
    installedAuthorityIsCoherent()
  );
}

function applyBattlefieldBoundaryCopy() {
  const installed = installedPresentationAuthority();
  const replay = isReplayMode();
  const terminal = installed !== null && isTerminal(installed.transport);
  const scientificFenced = recordingScientificControlsFenced();
  const audience = authorizedPresentationAudience(installed?.presentation);
  const scriptedInspection = scriptedBattlefieldLocallyActionable();
  const agentCloseoutInspection = liveAgentCloseoutInspectionActionable();
  const liveInteractive = liveBattlefieldCommandsInteractive();
  if (installed === null) {
    elements.battlefield.setAttribute("role", "img");
    elements.battlefield.tabIndex = -1;
    elements.battlefield.setAttribute("aria-label", FENCED_LIVE_BATTLEFIELD_LABEL);
    elements.battlefieldInstructions.textContent = FENCED_LIVE_BATTLEFIELD_INSTRUCTIONS;
  } else if (scriptedInspection && !scientificFenced) {
    elements.battlefield.setAttribute("role", "group");
    elements.battlefield.tabIndex = -1;
    elements.battlefield.setAttribute("aria-label", SCRIPTED_BATTLEFIELD_LABEL);
    elements.battlefieldInstructions.textContent = SCRIPTED_BATTLEFIELD_INSTRUCTIONS;
  } else if (agentCloseoutInspection) {
    elements.battlefield.setAttribute("role", "group");
    elements.battlefield.tabIndex = -1;
    elements.battlefield.setAttribute("aria-label", AGENT_CLOSEOUT_BATTLEFIELD_LABEL);
    elements.battlefieldInstructions.textContent =
      AGENT_CLOSEOUT_BATTLEFIELD_INSTRUCTIONS;
  } else if (replay) {
    elements.battlefield.setAttribute("role", "group");
    elements.battlefield.tabIndex = -1;
    elements.battlefield.setAttribute(
      "aria-label",
      terminal
        ? TERMINAL_REPLAY_BATTLEFIELD_LABEL
        : "Read-only replay battlefield snapshot. Authorized agents can be inspected.",
    );
    elements.battlefieldInstructions.textContent = terminal
      ? audience === "agent_pov"
        ? TERMINAL_REPLAY_AGENT_BATTLEFIELD_INSTRUCTIONS
        : TERMINAL_REPLAY_BATTLEFIELD_INSTRUCTIONS
      : audience === "agent_pov"
        ? "Replay Agent POV is read-only. Activate a visible body or choose any agent in the roster to switch to that agent's fog-of-war view at the same replay tick."
        : "Replay is read-only. Upcoming Transition shows the authorized recorded joint action out of this frame; activate an agent to inspect current facts, or use the timeline to change frames.";
  } else if (liveInteractive) {
    elements.battlefield.setAttribute("role", "application");
    elements.battlefield.tabIndex = 0;
    elements.battlefield.setAttribute(
      "aria-label",
      "Interactive battlefield. Press Help for keyboard controls.",
    );
    elements.battlefieldInstructions.textContent =
      audience === "agent_pov"
        ? "Live Agent POV keeps one fixed recipient. Bodies are passive inspection targets; use the authorized draft controls to prepare that recipient's action."
        : "Live Oracle View is interactive. Activate an authorized actor to control it; Shift-click selects an authorized target; Escape clears the target and leaves battlefield focus. Battlefield keyboard commands apply only while this surface has focus.";
  } else if (installedLiveScientificInspectionAvailable()) {
    elements.battlefield.setAttribute("role", "group");
    elements.battlefield.tabIndex = -1;
    elements.battlefield.setAttribute("aria-label", READ_ONLY_LIVE_SCIENTIFIC_LABEL);
    elements.battlefieldInstructions.textContent =
      READ_ONLY_LIVE_SCIENTIFIC_INSTRUCTIONS;
  } else {
    elements.battlefield.setAttribute("role", "img");
    elements.battlefield.tabIndex = -1;
    elements.battlefield.setAttribute(
      "aria-label",
      scientificFenced
        ? "Read-only recording closeout battlefield snapshot."
        : FENCED_LIVE_BATTLEFIELD_LABEL,
    );
    elements.battlefieldInstructions.textContent = scientificFenced
      ? "Recording closeout has fenced simulator and pending-action controls. Presentation, recovery, review, and Exit controls remain available."
      : FENCED_LIVE_BATTLEFIELD_INSTRUCTIONS;
  }
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
  const inspectionState = authorizedPresentationInspectionState(state.presentation);
  const audience = authorizedPresentationAudience(state.presentation);
  const liveHelpMode = !isAuthorizedPresentationFrame(state.presentation)
    ? "unavailable"
    : inspectionState.state_kind === "live_scripted"
      ? "scripted"
      : audience === "researcher"
        ? "oracle"
        : audience === "agent_pov"
          ? "agent"
          : "unavailable";
  for (const element of elements.liveHelpModes) {
    element.toggleAttribute(
      "hidden",
      replay || element.dataset.liveHelpMode !== liveHelpMode,
    );
  }
  elements.replayTimeline.toggleAttribute("hidden", !replay);
  applyBattlefieldBoundaryCopy();
  applyAuthorizedInspectorChrome();
}

/** @param {ReturnType<typeof installedPresentationAuthority>} installed */
function renderReplayMetadata(installed) {
  if (!isReplayMode()) {
    return;
  }
  if (installed === null) {
    return;
  }
  const frame = installed.transport;
  const presentation = installed.presentation;
  const artifactFacts =
    authorizedPresentationAudience(presentation) === "researcher"
      ? {
          artifact_summary: frame.artifact_summary,
          completion: frame.completion,
          processing: frame.processing,
        }
      : frame.artifact_facts;
  const summary = isRecord(artifactFacts?.artifact_summary)
    ? artifactFacts.artifact_summary
    : {};
  const reference = isRecord(summary.replay_reference) ? summary.replay_reference : {};
  const completion = isRecord(artifactFacts?.completion)
    ? artifactFacts.completion
    : {};
  elements.replayArtifactReference.textContent = String(
    reference.artifact_id ?? "Unavailable",
  );
  elements.replayArtifactReference.removeAttribute("title");
  registerTooltipOwner(
    elements.replayArtifactReference,
    createSemanticDescriptor({
      kind: "control",
      id: "replay-artifact-reference",
      title: "Replay artifact",
      tone: "information",
      accent: "none",
      summary: "Canonical identity for the loaded immutable replay artifact.",
      rows: [
        {
          label: "Artifact",
          value: String(reference.artifact_id ?? "Unavailable"),
          metadata: { compact: true, full: true },
        },
        {
          label: "Canonical digest",
          value: String(reference.canonical_digest_sha256 ?? "Unavailable"),
          metadata: { compact: false, full: true },
        },
      ],
      sections: [],
      metadata: { compact: true, full: true },
      anchor: "element",
    }),
  );
  elements.replayCompletionBadge.textContent = humanize(
    completion.completion_state ?? "unavailable",
  );
  elements.replayProcessingBadge.textContent = "Authorized replay";
  registerTooltipOwner(
    elements.replayCompletionBadge,
    explainTechnicalFact("completion"),
    { inspectable: false },
  );
  registerTooltipOwner(
    elements.replayProcessingBadge,
    explainTechnicalFact("processing"),
    { inspectable: false },
  );
  elements.replayEndReason.textContent = String(
    completion.public_end_or_failure_reason ??
      completion.end_or_failure_reason ??
      (asArray(completion.completion_bases).length > 0
        ? asArray(completion.completion_bases).map(humanize).join(" + ")
        : "Captured prefix"),
  );
  elements.replayRangesButton.setAttribute(
    "aria-pressed",
    String(installedPresentationRangesVisible(presentation)),
  );
  elements.replayRangesButton.disabled =
    state.busy || state.shuttingDown || state.resyncRequired || state.offline;
  const scene = installedAuthorizedPresentationSceneView(presentation);
  const selectedOwnerKey = isRecord(scene?.selection)
    ? scene.selection.inspection_owner_presentation_key
    : null;
  elements.replayClearReferenceButton.disabled =
    state.busy ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline ||
    typeof selectedOwnerKey !== "string";
  renderReplayTimelineControls(
    replayTimelineElements,
    replayTimelineRenderState(replayPlayback.snapshot()),
  );
}

/** @type {Readonly<Record<string, string>>} */
const recordingPersistenceLabels = Object.freeze({
  target_unavailable: "Destination unavailable",
  publication_failed: "Publication failed",
  verification_failed: "Publication verification failed",
});

/** @param {ReturnType<typeof installedPresentationAuthority>} installed */
function renderRecordingControls(installed) {
  if (installed === null) {
    return;
  }
  const recording = recordingStatus();
  const unavailable = isReplayMode() || recording === null;
  elements.recordingPanel.toggleAttribute("hidden", unavailable);
  elements.recordingBadge.toggleAttribute("hidden", unavailable);
  if (unavailable) {
    delete document.documentElement.dataset.recordingLifecycle;
    if (elements.recordingDiscardDialog.open) {
      elements.recordingDiscardDialog.close();
    }
    pendingRecordingReplacement = null;
    return;
  }

  document.documentElement.dataset.recordingLifecycle = recording.lifecycle;
  elements.recordingBadge.dataset.lifecycle = recording.lifecycle;
  elements.recordingBadge.textContent =
    recording.lifecycle === "recording"
      ? `Recording ${recording.captured_transition_count} / ${recording.expected_transition_count}`
      : `Recording · ${humanize(recording.lifecycle)}`;
  elements.recordingLifecycle.textContent = humanize(recording.lifecycle);
  elements.recordingProgress.textContent = `${recording.captured_transition_count} / ${recording.expected_transition_count} transitions`;
  elements.recordingCompletion.textContent =
    recording.completion_state === null
      ? "Capture in progress"
      : recording.completion_reason === null
        ? humanize(recording.completion_state)
        : `${humanize(recording.completion_state)} · ${humanize(recording.completion_reason)}`;

  const persistenceLabel =
    recordingPersistenceLabels[recording.persistence_error_code] ?? null;
  elements.recordingPersistenceFact.toggleAttribute(
    "hidden",
    persistenceLabel === null,
  );
  elements.recordingPersistenceError.textContent =
    persistenceLabel ?? "No persistence error";

  const interactionDisabled =
    state.busy || state.shuttingDown || state.resyncRequired || state.offline;
  const actionAvailability = [
    [elements.recordingFinishButton, recording.finish_available === true],
    [elements.recordingReviewButton, recording.review_available === true],
    [elements.recordingRetryButton, recording.retry_available === true],
  ];
  for (const [element, available] of actionAvailability) {
    element.toggleAttribute("hidden", !available);
    element.disabled = interactionDisabled || !available;
  }
  const saveAsAvailable = recording.save_as_available === true;
  elements.recordingSaveAsControl.toggleAttribute("hidden", !saveAsAvailable);
  elements.recordingSaveAsInput.disabled = interactionDisabled || !saveAsAvailable;
  elements.recordingSaveAsButton.disabled = interactionDisabled || !saveAsAvailable;

  elements.recordingStatusNote.textContent =
    recording.lifecycle === "recording" && recording.discard_available === true
      ? "Capture is active. Reset requires confirmation because it replaces this recorded prefix."
      : recording.lifecycle === "recording"
        ? "Capture is active. Scientific controls remain authoritative in Python."
        : recording.lifecycle === "persistence_failed"
          ? "The exact canonical replay bytes remain cached. Retry the same destination or choose a basename-only Save As target."
          : recording.lifecycle === "saved"
            ? "The replay and metric report were saved and publicly verified. Review opens the same local session in read-only mode."
            : recording.lifecycle === "reviewing"
              ? "The local service is switching this session to read-only replay review."
              : "Recording closeout has fenced scientific controls while the canonical artifact is finalized.";

  if (recording.discard_available !== true && elements.recordingDiscardDialog.open) {
    elements.recordingDiscardDialog.close();
    pendingRecordingReplacement = null;
  }
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
 * @param {unknown} publicAgentId
 * @returns {string}
 */
function agentIdentity(publicAgentId) {
  return typeof publicAgentId === "string" && publicAgentId.length > 0
    ? `Agent ID ${publicAgentId}`
    : "Agent unavailable";
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

const DEFERRED_SUBMIT_PREPARATION_COMMAND_TYPES = new Set([
  "battlefield_pointer",
  "roster_selection",
  "actor_pov_target_action",
]);

/** @param {Record<string, unknown>} command */
function commandPreparesDeferredSubmit(command) {
  if (DEFERRED_SUBMIT_PREPARATION_COMMAND_TYPES.has(String(command.command_type))) {
    return true;
  }
  if (command.command_type !== "keyboard" || typeof command.key !== "string") {
    return false;
  }
  const key = command.key.toLowerCase();
  return DRAFT_KEYS.has(key) || key === "tab" || key === "escape";
}

/** @param {Record<string, unknown>} command */
function isFreshUnmodifiedEnter(command) {
  return (
    command.command_type === "keyboard" &&
    typeof command.key === "string" &&
    command.key.toLowerCase() === "enter" &&
    command.shift_key === false &&
    command.ctrl_key === false &&
    command.alt_key === false &&
    command.meta_key === false &&
    command.repeat === false
  );
}

/**
 * Consume live Enter while a request owns the battlefield. Only one fresh,
 * unmodified edge may be retained, and only behind a draft-preparation
 * transaction. Replay and every non-request fence retain native Enter.
 *
 * @param {Record<string, unknown>} command
 */
function retainFencedEnter(command) {
  const transaction = activeLiveCommandTransaction;
  if (
    isReplayMode() ||
    !state.busy ||
    transaction === null ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline
  ) {
    return false;
  }
  if (
    transaction.allowsDeferredSubmit &&
    transaction.deferredSubmit === null &&
    isFreshUnmodifiedEnter(command)
  ) {
    transaction.deferredSubmit = Object.freeze({ ...command });
    setNotice(
      "Submit queued; waiting for the staged action to be installed first…",
      "info",
    );
    renderConnection();
  }
  return true;
}

/**
 * @param {Record<string, unknown>} command
 * @returns {"draft" | "interactive-submit" | null}
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
  return null;
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
  if (isTerminal(frame)) {
    return {
      allowed: false,
      notice: "The episode is terminal; reset to continue.",
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

/** @param {ProductIdentityMismatchError} error */
function failClosedProductIdentity(error) {
  state.frame = null;
  state.timeline = null;
  clearPresentationAuthority("product_identity_mismatch");
  state.offline = true;
  state.resyncRequired = true;
  replayPlayback.setConnected(false);
  setNotice(error.message, "error");
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
 * Translate internal audience discriminators into public product language.
 * Raw values remain available to authorization and replay normalization only.
 *
 * @param {unknown} value
 */
function publicAudienceLabel(value) {
  if (value === "researcher") {
    return "Oracle View";
  }
  if (value === "actor_pov" || value === "agent_pov" || value === "pov") {
    return "Agent POV";
  }
  return "View unavailable";
}

function renderConnection() {
  let label = "Online";
  let status = "online";
  if (state.shuttingDown) {
    label = "Shutting down";
    status = "busy";
  } else if (state.busy) {
    label = activeLiveCommandTransaction?.deferredSubmit
      ? "Submit queued"
      : "Command in flight";
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

/** @param {ReturnType<typeof installedPresentationAuthority>} [installed] */
function renderSessionToolbar(installed = installedPresentationAuthority()) {
  const frame = installed?.transport ?? null;
  const presentation = installed?.presentation ?? null;
  renderViewerBoundary();
  if (installed === null) {
    renderPendingPresentationChrome();
  }
  const replay = isReplayMode();
  const disabled =
    state.busy || !frame || state.shuttingDown || state.resyncRequired || state.offline;
  const restartControlsBlocked = recordingRestartControlsBlocked();

  elements.stepValue.textContent = presentation
    ? String(presentation.simulator_step_count ?? "—")
    : "—";
  const incomingTransition = authorizedIncomingTransitionId(presentation);
  elements.transitionValue.textContent = incomingTransition
    ? String(incomingTransition)
    : "—";

  const audience = authorizedPresentationAudience(presentation) ?? "unavailable";
  elements.audienceBadge.textContent = publicAudienceLabel(audience);
  elements.audienceBadge.dataset.audience = audience;
  document.documentElement.dataset.audience = elements.audienceBadge.dataset.audience;
  document.documentElement.dataset.preset = "analysis";

  const terminal = frame !== null && isTerminal(frame);
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

  const retainedAuthority =
    document.documentElement.dataset.presentationAuthority === "retained";
  const retainedState = state.shuttingDown
    ? "session shutting down"
    : state.busy
      ? "update pending"
      : "reconnect required";
  elements.scenarioDescription.textContent = presentation
    ? retainedAuthority
      ? replay
        ? `Last confirmed ${publicAudienceLabel(authorizedPresentationAudience(presentation))} recorded frame ${frame?.cursor?.frame_index ?? "—"} of ${frame?.cursor?.final_frame_index ?? "—"} · ${retainedState}`
        : `Last confirmed ${publicAudienceLabel(authorizedPresentationAudience(presentation))} battlefield · ${retainedState}`
      : replay
        ? `${publicAudienceLabel(authorizedPresentationAudience(presentation))} · recorded frame ${frame?.cursor?.frame_index ?? "—"} of ${frame?.cursor?.final_frame_index ?? "—"}`
        : `${publicAudienceLabel(authorizedPresentationAudience(presentation))} · authorized live presentation`
    : "Waiting for an authorized presentation.";

  const viewMode = frame?.view_mode === "agent_pov" ? "pov" : frame?.view_mode;
  if (typeof viewMode === "string") {
    elements.viewSelect.value = viewMode;
  } else {
    elements.viewSelect.value = "";
  }
  elements.viewSelect.disabled = disabled;
  elements.reconnectButton.disabled =
    state.busy || state.shuttingDown || productIdentity === null;
  elements.exitButton.disabled = disabled;
  elements.exitButton.textContent =
    productIdentity?.product_kind === "replay_viewer"
      ? "Exit replay viewer"
      : productIdentity?.product_kind === "combat_debugger"
        ? "Exit combat debugger"
        : "Exit";
  elements.resetButton.disabled =
    disabled || restartControlsBlocked || liveScriptedInspectionOnly();
  const authorizedLive =
    !replay &&
    (authorizedPresentationAudience(presentation) === "researcher" ||
      authorizedPresentationAudience(presentation) === "agent_pov");
  elements.liveRangesButton.hidden = !authorizedLive;
  elements.liveRangesButton.setAttribute(
    "aria-pressed",
    String(authorizedLive && installedPresentationRangesVisible(presentation)),
  );
  elements.liveRangesButton.disabled = disabled || !authorizedLive;
  renderRecordingControls(installed);
  renderReplayMetadata(installed);
  renderCommandAvailability();
  registerAuthorityAwareUtilityHelp();
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
 * @param {unknown} [descriptor]
 */
function setAuthoritativeAvailability(
  button,
  available,
  explanation,
  descriptor = null,
) {
  button.setAttribute("aria-disabled", String(!available));
  button.dataset.authoritativeAvailable = String(available);
  button.dataset.tooltipKind = "legality";
  button.dataset.tooltipText = explanation;
  registerTooltipOwner(
    button,
    descriptor ?? {
      kind: "legality",
      id: `command:${button.id || button.dataset.key || "choice"}`,
      title:
        button.getAttribute("aria-label") ??
        button.textContent?.trim() ??
        "Pending choice",
      tone: available ? "positive" : "warning",
      accent: "none",
      summary: explanation,
      rows: [
        {
          label: "Status",
          value: available ? "True" : "False",
          metadata: { compact: true, full: true },
        },
      ],
      sections: [],
      metadata: { compact: true, full: true },
      anchor: "element",
    },
  );
}

/**
 * Fail closed when a command lane has no exact authorized owner. The generic
 * control-help fallback is intentionally bypassed because legality is an
 * owner-bound scientific fact.
 *
 * @param {HTMLButtonElement} button
 */
function clearOwnerBoundLegalityAvailability(button) {
  button.setAttribute("aria-disabled", "true");
  button.dataset.authoritativeAvailable = "false";
  clearPresentationTooltipOwner(button);
}

/**
 * Rebuild target choices from the exact authorized decision mask. Oracle
 * command slots come only from the validated presentation identity directory;
 * Agent POV uses its recipient-local target-action axis and never gains a slot.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @param {Readonly<Record<string, any>>} inspection
 */
function renderCommandTargets(presentation, inspection) {
  const mask = isRecord(inspection.decision_mask) ? inspection.decision_mask : {};
  const action = isRecord(inspection.draft_action) ? inspection.draft_action : {};
  const selectedTargetAction = Number(action.target_action);
  const audience = authorizedPresentationAudience(presentation);
  const fragment = document.createDocumentFragment();
  for (const target of asArray(mask.target_actions).filter(isRecord)) {
    const targetAction = Number(target.target_action);
    if (!Number.isInteger(targetAction)) {
      continue;
    }
    const pairMask = asArray(mask.target_use_ultimate_joint_mask)[targetAction];
    const option = document.createElement("option");
    const basic = targetAction > 0 && asArray(pairMask)[0] === true ? "B ✓" : "B ×";
    const ultimate = asArray(pairMask)[1] === true ? "U ✓" : "U ×";
    if (target.target_kind === "no_target") {
      option.value = "";
      option.textContent = `${target.display_name ?? "No target"} · ${basic} · ${ultimate}`;
    } else if (audience === "agent_pov") {
      option.value = `pov-target-action:${targetAction}`;
      option.textContent = `${agentIdentity(target.target_public_agent_id)} · action ${targetAction} · ${basic} · ${ultimate}`;
    } else if (audience === "researcher") {
      const commandSlot = authorizedOracleCommandSlotForPublicAgentId(
        presentation,
        target.target_public_agent_id,
      );
      if (!Number.isInteger(commandSlot)) {
        continue;
      }
      option.value = String(commandSlot);
      option.textContent = `${agentIdentity(target.target_public_agent_id)} · ${basic} · ${ultimate}`;
    } else {
      continue;
    }
    if (typeof target.target_presentation_key === "string") {
      option.dataset.presentationKey = target.target_presentation_key;
    }
    option.selected = targetAction === selectedTargetAction;
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
 * Render live command facts only from the authorized editable-draft branch.
 *
 * @param {Readonly<Record<string, any>>} presentation
 */
function renderDraftState(presentation) {
  const inspectionState = authorizedPresentationInspectionState(presentation);
  const inspection = isRecord(inspectionState.inspection)
    ? inspectionState.inspection
    : {};
  const mask = isRecord(inspection.decision_mask) ? inspection.decision_mask : {};
  const pending = isRecord(inspection.draft_action) ? inspection.draft_action : {};
  const controlledCandidate = authorizedAgentForPresentationKey(
    presentation,
    inspection.actor_presentation_key,
  );
  const controlledOwner =
    controlledCandidate !== null &&
    controlledCandidate.public_agent_id === inspection.actor_public_agent_id
      ? controlledCandidate
      : null;
  const controlledPublicAgentId = controlledOwner?.public_agent_id;
  const controlledIdentity =
    typeof controlledPublicAgentId === "string"
      ? agentIdentity(controlledPublicAgentId)
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
  const controlledSlot =
    controlledOwner === null
      ? null
      : authorizedOracleCommandSlotForPresentationKey(
          presentation,
          controlledOwner.presentation_key,
        );
  if (Number.isInteger(controlledSlot)) {
    elements.commandControlledActor.dataset.controlledSlot = String(controlledSlot);
  } else {
    delete elements.commandControlledActor.dataset.controlledSlot;
  }
  const pendingMove = Number(pending.move_action);
  const movementMask = asArray(mask.movement_action_mask);
  const movementButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
    elements.commandDeck?.querySelectorAll("button[data-move-action]") ?? []
  );
  for (const button of movementButtons) {
    const moveAction = Number(button.dataset.moveAction);
    const available = movementMask[moveAction];
    setDraftSelection(button, moveAction === pendingMove);
    setAuthoritativeAvailability(
      button,
      available === true,
      available === true
        ? `${button.getAttribute("aria-label") ?? "Movement"} is legal in the current authoritative movement mask.`
        : `${button.getAttribute("aria-label") ?? "Movement"} is unavailable in the current authoritative movement mask.`,
    );
  }

  renderCommandTargets(presentation, inspection);
  const targetAction = Number.isInteger(pending.target_action)
    ? Number(pending.target_action)
    : 0;
  const pairMask = asArray(mask.target_use_ultimate_joint_mask)[targetAction];
  const basicAvailable = targetAction > 0 && asArray(pairMask)[0] === true;
  const ultimateAvailable = asArray(pairMask)[1] === true;
  const noCombatSelected =
    pending.armed_lane === "none" ||
    (pending.armed_lane === "basic" && targetAction === 0);
  setDraftSelection(elements.noCombatButton, noCombatSelected);
  setDraftSelection(
    elements.basicButton,
    pending.armed_lane === "basic" && targetAction > 0,
  );
  setDraftSelection(elements.ultimateButton, pending.armed_lane === "ultimate");
  setAuthoritativeAvailability(
    elements.noCombatButton,
    true,
    "No combat is always a valid staged choice; movement can still be submitted.",
  );
  const legality =
    controlledOwner === null
      ? null
      : {
          owner_presentation_key: controlledOwner.presentation_key,
          owner_public_agent_id: controlledOwner.public_agent_id,
          lane_0_available: asArray(pairMask)[0] === true,
          lane_1_available: ultimateAvailable,
          basic_available: basicAvailable,
          ultimate_available: ultimateAvailable,
        };
  const basicLegality =
    legality === null ? null : explainLegality(legality, 0, controlledOwner);
  const ultimateLegality =
    legality === null ? null : explainLegality(legality, 1, controlledOwner);
  if (basicLegality === null) {
    clearOwnerBoundLegalityAvailability(elements.basicButton);
  } else {
    setAuthoritativeAvailability(
      elements.basicButton,
      basicAvailable,
      `Basic ability is ${basicAvailable ? "" : "not "}available this tick.`,
      basicLegality,
    );
  }
  if (ultimateLegality === null) {
    clearOwnerBoundLegalityAvailability(elements.ultimateButton);
  } else {
    setAuthoritativeAvailability(
      elements.ultimateButton,
      ultimateAvailable,
      `Ultimate ability is ${ultimateAvailable ? "" : "not "}available this tick.`,
      ultimateLegality,
    );
  }
}

function renderCommandAvailability() {
  const presentation = state.presentation;
  const inspectionState = authorizedPresentationInspectionState(presentation);
  const editableDraft = inspectionState.state_kind === "live_editable";
  const scriptedAdvance = liveScriptedInspectionOnly();
  const fixedAgentRecipient =
    authorizedPresentationAudience(presentation) === "agent_pov";
  const disabled =
    state.busy ||
    !state.frame ||
    !isAuthorizedPresentationFrame(presentation) ||
    !installedAuthorityIsCoherent() ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline;
  const scientificFenced = recordingScientificControlsFenced();
  elements.submitTurnButton.dataset.key = scriptedAdvance ? "n" : "Enter";
  elements.submitTurnButton.textContent = scriptedAdvance
    ? "Advance scripted frame"
    : inspectionState.submission_scope === "controlled_actor"
      ? "Submit controlled actor"
      : "Submit joint turn";
  elements.commandCommitTitle.textContent = scriptedAdvance
    ? "Advance the registered script"
    : "Submit the staged joint turn";
  elements.commandCommitSummary.textContent = scriptedAdvance
    ? "One authoritative scripted transition"
    : "One authoritative transition";
  if (isReplayMode()) {
    elements.commandTargetSelect.disabled = true;
    if (elements.commandDeck) {
      for (const button of elements.commandDeck.querySelectorAll("button")) {
        /** @type {HTMLButtonElement} */ (button).disabled = true;
      }
    }
    return;
  }
  if (editableDraft && presentation) {
    renderDraftState(presentation);
  } else {
    elements.commandControlledActor.textContent = "Actor · unavailable";
    delete elements.commandControlledActor.dataset.controlledSlot;
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No authorized targets";
    elements.commandTargetSelect.replaceChildren(option);
  }
  elements.commandTargetSelect.disabled =
    disabled || !editableDraft || scientificFenced || isTerminal(state.frame);
  if (elements.commandDeck) {
    const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
      elements.commandDeck.querySelectorAll("button[data-key]")
    );
    for (const button of buttons) {
      const command = keyboardCommand(button.dataset.key ?? "", {
        shiftKey: button.dataset.shift === "true",
      });
      const mode = modeAvailability(command, state.frame);
      const recordingDecision = recordingCommandDecision(state.frame ?? {}, command);
      const enabledByInspection =
        editableDraft || (scriptedAdvance && button === elements.submitTurnButton);
      button.disabled =
        disabled ||
        scientificFenced ||
        !enabledByInspection ||
        (fixedAgentRecipient && button.dataset.key === "Tab") ||
        (scriptedAdvance && isTerminal(state.frame)) ||
        recordingDecision.action === "block" ||
        !mode.allowed;
    }
  }
}

/**
 * Preserve presentation state for diagnostics and deterministic E2E waits
 * without exposing motion-editing controls in the product shell.
 *
 * @param {ReturnType<CombatChoreographer["snapshot"]>} presentation
 */
function exposePresentationState(presentation) {
  document.documentElement.dataset.motionMode = presentation.motionMode;
  document.documentElement.dataset.motionPaused = String(presentation.paused);
  if (presentation.renderPolicy === null) {
    document.documentElement.removeAttribute("data-render-policy");
  } else {
    document.documentElement.dataset.renderPolicy = presentation.renderPolicy;
  }
  document.documentElement.dataset.submissionBlocked = String(
    presentation.submissionBlocked,
  );
}

function applyReplayReferenceSemantics() {
  if (!isReplayMode()) {
    return;
  }
  const presentation = state.presentation;
  if (authorizedPresentationAudience(presentation) !== "researcher") {
    return;
  }
  const scene = installedAuthorizedPresentationSceneView(presentation);
  const selectedKey = isRecord(scene?.selection)
    ? scene.selection.inspection_owner_presentation_key
    : null;
  if (typeof selectedKey !== "string") {
    return;
  }
  const reference = elements.battlefield.querySelector(
    `.agent[data-presentation-key="${CSS.escape(selectedKey)}"]`,
  );
  if (!(reference instanceof Element)) {
    return;
  }
  const agent = asArray(scene?.agents).find(
    (candidate) => isRecord(candidate) && candidate.presentation_key === selectedKey,
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
  registerTooltipOwner(reference, explainAgent(agent, { audience: "researcher" }), {
    inspectable: false,
  });
}

/**
 * Resolve one opaque key to the only action authorized by the installed
 * product/audience. The returned effect is a browser decision, never a value
 * supplied by a DOM data-role or raw slot.
 *
 * @param {unknown} presentationKey
 * @returns {Readonly<Record<string, any>> | null}
 */
function authorizedAgentActivation(presentationKey) {
  const presentation = state.presentation;
  if (
    typeof presentationKey !== "string" ||
    !isAuthorizedPresentationFrame(presentation) ||
    !installedAuthorityIsCoherent() ||
    state.busy ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline
  ) {
    return null;
  }
  const agent = authorizedAgentForPresentationKey(presentation, presentationKey);
  if (agent === null) {
    return null;
  }
  const audience = authorizedPresentationAudience(presentation);
  const inspectionState = authorizedPresentationInspectionState(presentation);
  const replayPovSwitch = isReplayMode() && audience === "agent_pov";
  if (isTerminal(state.frame) && !isReplayMode()) {
    return null;
  }
  if (audience === "researcher" && recordingScientificControlsFenced()) {
    return null;
  }
  if (replayPovSwitch) {
    return Object.freeze({
      effect: "replay_pov_switch",
      presentationKey,
      agent,
      audience,
    });
  }
  if (audience === "agent_pov" || inspectionState.state_kind === "live_scripted") {
    return Object.freeze({
      effect: "local_inspection",
      presentationKey,
      agent,
      audience,
    });
  }
  if (audience !== "researcher") {
    return null;
  }
  const commandSlot = authorizedOracleCommandSlotForPresentationKey(
    presentation,
    presentationKey,
  );
  if (!Number.isInteger(commandSlot)) {
    return null;
  }
  if (isReplayMode()) {
    return Object.freeze({
      effect: "replay_select",
      presentationKey,
      commandSlot,
      agent,
      audience,
    });
  }
  return inspectionState.state_kind === "live_editable"
    ? Object.freeze({
        effect: "live_control",
        presentationKey,
        commandSlot,
        agent,
        audience,
      })
    : null;
}

/** @param {unknown} target */
function authorizedAgentActivationFromTarget(target) {
  if (!(target instanceof Element)) {
    return null;
  }
  const element = target.closest(".agent[data-presentation-key]");
  if (!(element instanceof SVGElement)) {
    return null;
  }
  const nestedTooltipOwner = target.closest("[data-tooltip-owner]");
  if (nestedTooltipOwner !== null && nestedTooltipOwner !== element) {
    return null;
  }
  const activation = authorizedAgentActivation(element.dataset.presentationKey);
  return activation === null
    ? null
    : Object.freeze({
        ...activation,
        presentationKey: String(activation.presentationKey),
        element,
      });
}

function installAuthorizedAgentActivation() {
  const presentation = state.presentation;
  const scene = installedAuthorizedPresentationSceneView(presentation);
  const agents = asArray(scene?.agents);
  const audience = authorizedPresentationAudience(presentation);
  for (const element of elements.battlefield.querySelectorAll(
    ".agent[data-presentation-key]",
  )) {
    const presentationKey = element.dataset.presentationKey;
    const agent = agents.find(
      (candidate) =>
        isRecord(candidate) && candidate.presentation_key === presentationKey,
    );
    if (!isRecord(agent)) {
      element.setAttribute("role", "img");
      element.setAttribute("tabindex", "-1");
      continue;
    }
    const activation = authorizedAgentActivation(presentationKey);
    if (activation === null) {
      element.setAttribute("role", "img");
      element.setAttribute("tabindex", "-1");
    } else {
      element.setAttribute("tabindex", "0");
      element.setAttribute("role", "button");
      const identity = agentIdentity(agent.public_agent_id);
      element.setAttribute(
        "aria-label",
        activation.effect === "live_control"
          ? `${identity}. Control and inspect this authorized agent.`
          : activation.effect === "replay_pov_switch"
            ? `${identity}. Switch replay Agent POV to this authorized agent.`
            : `${identity}. Inspect this authorized agent.`,
      );
    }
    registerTooltipOwner(
      element,
      explainAgent(agent, { audience: audience ?? "agent_pov" }),
      { inspectable: false },
    );
  }
}

/**
 * Apply the already-resolved single effect. Local Agent state is installed
 * before rendering; Oracle waits for the server successor and never receives
 * an optimistic scientific selection.
 *
 * @param {string} presentationKey
 * @returns {boolean}
 */
function activateAuthorizedAgent(presentationKey) {
  const activation = authorizedAgentActivation(presentationKey);
  if (activation === null) {
    return false;
  }
  if (activation.effect === "local_inspection") {
    if (!setLocalInspectedPresentationKey(presentationKey)) {
      return false;
    }
    render();
    openAgentDetails();
    return true;
  }
  openAgentDetails();
  if (activation.effect === "replay_pov_switch") {
    void dispatchReplayCommand({
      command_type: "set_pov_actor",
      presentation_key: activation.presentationKey,
    });
  } else if (activation.effect === "replay_select") {
    void dispatchReplayCommand({
      command_type: "select_agent",
      selected_global_slot: activation.commandSlot,
    });
  } else if (activation.effect === "live_control") {
    void dispatchCommand({
      command_type: "roster_selection",
      role: "control",
      global_slot: activation.commandSlot,
    });
  }
  return true;
}

/** @param {string} reason */
function pauseReplayAfterPresentationFailure(reason) {
  suppressPlaybackStateRender = true;
  try {
    replayPlayback.pause(reason);
  } finally {
    suppressPlaybackStateRender = false;
  }
}

function render() {
  capturePresentationPreferenceBeforeRender();
  const installed = installedPresentationAuthority();
  const presentationFrame = installed?.presentation ?? null;
  const transportFrame = installed?.transport ?? null;
  const visualFilterSnapshot = visualFilterState;
  const choreographyControl = installedChoreographyControl(
    presentationFrame,
    visualFilterSnapshot,
    { consumeAnimatedRestart: true },
  );
  renderVisualFilterControls(visualFilterSnapshot);
  renderConnection();
  renderSessionToolbar(installed);
  battlefieldRenderer.render(presentationFrame, {
    offline: state.offline,
    showRanges: installedPresentationRangesVisible(presentationFrame),
    localInspectedPresentationKey:
      installedLocalInspectedPresentationKey(presentationFrame),
    visualFilterState: visualFilterSnapshot,
    renderPolicy: choreographyControl.renderPolicy,
  });
  applyBattlefieldBoundaryCopy();
  installAuthorizedAgentActivation();
  applyReplayReferenceSemantics();
  try {
    choreographer.presentFrame(
      presentationFrame,
      battlefieldRenderer.choreographySurface(),
      choreographyControl,
    );
  } catch (error) {
    pauseReplayAfterPresentationFailure("presentation_error");
    choreographer.clear("presentation_error");
    setNotice(
      error instanceof Error
        ? `Combat presentation failed: ${error.message} The authoritative frame remains available.`
        : "Combat presentation failed. The authoritative frame remains available.",
      "error",
    );
    renderConnection();
  }
  renderReplayArtifactActions(installed);
  lastBattlefieldSizeKey = battlefieldSizeKey();
  panels.render(presentationFrame, {
    busy: state.busy,
    shuttingDown: state.shuttingDown,
    resyncRequired: state.resyncRequired,
    offline: state.offline,
    activationDisabled:
      (transportFrame !== null && isTerminal(transportFrame) && !isReplayMode()) ||
      (authorizedPresentationAudience(presentationFrame) === "researcher" &&
        recordingScientificControlsFenced()),
    localInspectedPresentationKey:
      installedLocalInspectedPresentationKey(presentationFrame),
  });
  tooltipController.refresh();
  restorePresentationPreferenceAfterRender();
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
    capturePresentationPreferenceBeforeRender();
    const presentationFrame = state.presentation;
    const visualFilterSnapshot = visualFilterState;
    const choreographyControl = installedChoreographyControl(
      presentationFrame,
      visualFilterSnapshot,
    );
    battlefieldRenderer.render(presentationFrame, {
      offline: state.offline,
      showRanges: installedPresentationRangesVisible(presentationFrame),
      localInspectedPresentationKey:
        installedLocalInspectedPresentationKey(presentationFrame),
      visualFilterState: visualFilterSnapshot,
      renderPolicy: choreographyControl.renderPolicy,
    });
    applyBattlefieldBoundaryCopy();
    installAuthorizedAgentActivation();
    applyReplayReferenceSemantics();
    try {
      choreographer.reproject(
        presentationFrame,
        battlefieldRenderer.choreographySurface(),
        choreographyControl,
      );
    } catch (error) {
      pauseReplayAfterPresentationFailure("resize_projection_error");
      choreographer.clear("resize_projection_error");
      setNotice(
        error instanceof Error
          ? `Combat presentation resize failed: ${error.message}`
          : "Combat presentation resize failed.",
        "error",
      );
      renderConnection();
    }
    renderReplayArtifactActions(installedPresentationAuthority());
    tooltipController.refresh();
    restorePresentationPreferenceAfterRender();
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

/** @param {Readonly<Record<string, unknown>>} replacement */
function recordingReplacementLabel(replacement) {
  if (replacement.command_type === "reset") {
    return "Reset the current episode";
  }
  if (replacement.command_type === "scenario_switch") {
    return `Switch to scenario ${String(replacement.scenario_name)}`;
  }
  return "Replace the current episode";
}

/** @param {Readonly<Record<string, unknown>>} replacement */
function requestRecordingDiscardConfirmation(replacement) {
  pendingRecordingReplacement = replacement;
  elements.recordingDiscardConfirmButton.disabled =
    state.busy ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline ||
    !isAuthorizedPresentationFrame(state.presentation);
  renderSessionToolbar();
  elements.recordingDiscardIntent.textContent = recordingReplacementLabel(replacement);
  if (!elements.recordingDiscardDialog.open) {
    elements.recordingDiscardDialog.showModal();
  }
  elements.recordingDiscardCancelButton.focus({ preventScroll: true });
}

/**
 * Send one replay command through the replay-only route. This function is the
 * controller's single request boundary; it installs the authoritative frame
 * before resolving so presentation settling can begin immediately.
 *
 * @param {Readonly<Record<string, any>>} command
 * @param {{deferFinalRender?: boolean}} [options]
 */
async function sendReplayCommand(command, { deferFinalRender = false } = {}) {
  if (!isReplayMode() || !state.frame || !state.presentation) {
    throw new DebuggerApiError("Replay controls require an installed replay frame.");
  }
  if (state.busy || state.shuttingDown) {
    throw new DebuggerApiError("A replay request is already in flight.");
  }
  if (state.resyncRequired || state.offline) {
    throw new DebuggerApiError("Reconnect before sending another replay command.");
  }
  invalidateReplayArtifactAction();
  state.busy = true;
  setNotice("Waiting for the read-only replay response…", "info");
  const previousAuthority = state.authority;
  const previousFrame = state.frame;
  const previousCursor = previousFrame.cursor;
  const request = replayCommandRequest({
    clientId: state.clientId,
    commandId: window.crypto.randomUUID(),
    baseRevision: currentRevision(),
    command,
  });
  /** @type {{current: {kind: "success" | "stale", payload: any} | null}} */
  const commandOutcome = { current: null };
  try {
    const installPromise = presentationInstallation.installFromCommand({
      reason:
        command.command_type === "set_view"
          ? "replay_audience_change"
          : "replay_command",
      pendingPolicy: "retain_last_authorized",
      sendCommand: async () => {
        try {
          const payload = await postReplayCommand(state.token, request);
          commandOutcome.current = { kind: "success", payload };
          return commandOutcome.current;
        } catch (error) {
          if (error instanceof DebuggerApiError && error.status === 409) {
            commandOutcome.current = { kind: "stale", payload: error.payload };
            return commandOutcome.current;
          }
          throw error;
        }
      },
      joinCommandResult: async (outcome) => {
        const joined = await extractJoinedFrame(
          outcome.payload,
          await getCurrentPresentation(state.token),
        );
        if (joined === null) {
          throw new TypeError("Replay response has no joinable transport candidate.");
        }
        if (joined.transport.viewer_mode !== "replay") {
          throw new DebuggerApiError(
            "Replay response did not contain replay transport authority.",
          );
        }
        if (outcome.kind === "success") {
          validateReplayCommandOutcome(
            request.command,
            outcome.payload,
            previousCursor,
          );
        }
        return prepareJoinedAuthority(joined, {
          previousAuthority,
          continuityResult:
            outcome.kind === "success" ? outcome.payload.result : "stale_resync",
        });
      },
      getJoined: async () =>
        prepareJoinedAuthority(await getCurrentFrameAndPresentation(state.token), {
          previousAuthority,
          continuityResult: "stale_resync",
        }),
    });
    render();
    const installOutcome = await installPromise;
    if (installOutcome.status === "superseded") {
      throw new DebuggerApiError(
        "Replay response was superseded by a newer authority request.",
      );
    }
    state.offline = false;
    state.resyncRequired = false;
    replayPlayback.setConnected(true);
    const payload = commandOutcome.current?.payload;
    const stale = commandOutcome.current?.kind === "stale";
    const notice = extractNotice(payload);
    setNotice(
      stale
        ? payload?.error_code === "command_id_conflict"
          ? "The replay service rejected a command-ID conflict. Its coherent latest pair was installed; the command was not retried."
          : "This replay tab was stale. Its coherent latest pair was installed; the command was not retried."
        : installOutcome.resynchronized
          ? "The command completed once, its mixed presentation candidate was discarded, and one fresh GET pair was installed."
          : (notice ??
            (payload?.result === "duplicate"
              ? "Duplicate replay command recognized; it was not applied again."
              : payload?.result === "no_op"
                ? "Replay already matched that request."
                : "Read-only replay frame updated.")),
      stale ||
        installOutcome.resynchronized ||
        payload?.result === "duplicate" ||
        payload?.result === "no_op"
        ? "warning"
        : "success",
    );
    if (commandResponseSchedulesShutdown(request.command, payload)) {
      state.shuttingDown = true;
      setNotice("Exit accepted. The local replay viewer is shutting down.", "info");
    }
    return stale || installOutcome.resynchronized
      ? Object.freeze({ handled_resync: true, frame: state.frame })
      : payload;
  } catch (error) {
    if (error instanceof ProductIdentityMismatchError) {
      failClosedProductIdentity(error);
      throw error;
    }
    const status = error instanceof DebuggerApiError ? error.status : 0;
    if (isPresentationJoinRace(error)) {
      clearPresentationAuthority("replay_presentation_identity_mismatch");
    } else if (status === 401 || status === 403) {
      clearPresentationAuthority("replay_authorization_failure");
    }
    const payload = commandOutcome.current?.payload;
    if (commandResponseSchedulesShutdown(request.command, payload)) {
      state.shuttingDown = true;
      setNotice(
        "Exit was accepted, but no coherent successor presentation was installed while the local replay viewer shuts down.",
        "info",
      );
    } else {
      state.offline =
        error instanceof DebuggerApiError &&
        (status === 0 || status === 401 || status === 403);
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
    if (!deferFinalRender || state.resyncRequired || state.shuttingDown) {
      render();
    }
  }
}

/**
 * Let the playback controller validate and publish its successor intent before
 * the newly installed authority receives its first non-pending render.
 *
 * @param {Readonly<Record<string, any>>} command
 */
function sendReplayTransportCommand(command) {
  return sendReplayCommand(command, { deferFinalRender: true });
}

/** @param {Readonly<Record<string, any>>} command */
async function dispatchReplayCommand(command) {
  if (
    authorizedPresentationAudience(state.presentation) !== "researcher" &&
    (command.command_type === "select_agent" || command.command_type === "set_ranges")
  ) {
    setNotice(
      "Reference and range controls are unavailable in actor POV replay.",
      "warning",
    );
    renderConnection();
    return null;
  }
  invalidateReplayArtifactAction();
  replayPlayback.pause("user_command");
  try {
    const payload = await sendReplayCommand(command);
    const frame = state.frame;
    if (frame) {
      replayPlayback.installCursor(frame.cursor);
    }
    if (
      command.command_type === "set_pov_actor" &&
      !state.resyncRequired &&
      !state.offline &&
      authorizedPresentationAudience(state.presentation) === "agent_pov"
    ) {
      elements.battlefield.focus({ preventScroll: true });
    }
    return payload;
  } catch {
    return null;
  }
}

/** @param {Record<string, unknown>} command */
function dispatchPanelCommand(command) {
  if (
    command.command_type === "activate_authorized_agent" &&
    typeof command.presentation_key === "string"
  ) {
    activateAuthorizedAgent(command.presentation_key);
    return Promise.resolve(null);
  }
  if (
    command.command_type === "activate_replay_pov_agent" &&
    Number.isInteger(command.global_slot) &&
    Number(command.global_slot) >= 0 &&
    Number(command.global_slot) < 10 &&
    isReplayMode()
  ) {
    openAgentDetails();
    return dispatchReplayCommand({
      command_type: "set_pov_actor",
      global_slot: command.global_slot,
    });
  }
  if (
    command.command_type === "roster_selection" &&
    Number.isInteger(command.global_slot)
  ) {
    openAgentDetails();
    if (!isReplayMode()) {
      return dispatchCommand(command);
    }
    if (command.role === "target") {
      return dispatchReplayCommand({
        command_type: "select_agent",
        selected_global_slot: command.global_slot,
      });
    }
  }
  if (!isReplayMode()) {
    return dispatchCommand(command);
  }
  setNotice("That live debugger action is unavailable in replay.", "warning");
  renderConnection();
  return Promise.resolve(null);
}

/**
 * @param {Record<string, unknown>} command
 * @param {{allowDeferredSubmit?: boolean}} options
 */
async function dispatchCommand(command, { allowDeferredSubmit = true } = {}) {
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
  if (!state.frame || !state.presentation) {
    setNotice(
      "No coherent transport/presentation pair is available. Reconnect before sending commands.",
      "error",
    );
    renderConnection();
    return;
  }
  if (
    command.command_type === "keyboard" &&
    typeof command.key === "string" &&
    authorizedPresentationAudience(state.presentation) === "agent_pov"
  ) {
    if (command.key === "Tab") {
      return;
    }
    if (command.key.toLowerCase() === "g") {
      if (
        command.shift_key === false &&
        command.ctrl_key === false &&
        command.alt_key === false &&
        command.meta_key === false &&
        command.repeat === false &&
        toggleAgentLocalRanges()
      ) {
        render();
      }
      return;
    }
  }
  if (liveScriptedInspectionOnly() && !allowedDuringLiveScriptedInspection(command)) {
    setNotice(
      "Scripted live view is inspection-only. Use Advance scripted frame for the next authorized step.",
      "warning",
    );
    renderConnection();
    renderCommandAvailability();
    return;
  }
  if (
    command.command_type === "battlefield_pointer" &&
    authorizedPresentationAudience(state.presentation) !== "researcher"
  ) {
    setNotice(
      "Battlefield bodies are passive in Agent POV. Use the authorized draft controls.",
      "warning",
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
  const recordingDecision = recordingCommandDecision(state.frame, command);
  if (recordingDecision.action === "block") {
    setNotice(
      recordingDecision.notice ??
        "That command is fenced by the current recording lifecycle.",
      "warning",
    );
    renderConnection();
    renderSessionToolbar();
    return;
  }
  if (recordingDecision.action === "confirm") {
    if (recordingDecision.replacement) {
      requestRecordingDiscardConfirmation(recordingDecision.replacement);
    }
    return;
  }
  if (
    isSubmissionCommand(command) &&
    presentationRequiresSubmissionSettle(choreographer.snapshot())
  ) {
    // Submit owns this synchronous edge: settle the current (even paused)
    // explanation, then send the same current draft through the normal fence.
    choreographer.skip();
  }

  /** @type {{
   *   allowsDeferredSubmit: boolean,
   *   deferredSubmit: Readonly<Record<string, unknown>> | null,
   * }} */
  const liveCommandTransaction = {
    allowsDeferredSubmit: allowDeferredSubmit && commandPreparesDeferredSubmit(command),
    deferredSubmit: null,
  };
  activeLiveCommandTransaction = liveCommandTransaction;
  state.busy = true;
  state.offline = false;
  setNotice("Waiting for the authoritative Python response…", "info");
  let reviewHandoff = false;
  /** @type {Readonly<Record<string, unknown>> | null} */
  let deferredSubmitToDispatch = null;
  const previousAuthority = state.authority;
  const request = commandRequest(command);
  /** @type {{current: {kind: "success" | "stale", payload: any} | null}} */
  const commandOutcome = { current: null };
  try {
    const installPromise = presentationInstallation.installFromCommand({
      reason:
        command.command_type === "set_view" ? "live_audience_change" : "live_command",
      pendingPolicy:
        command.command_type === "set_view" ? "clear" : "retain_last_authorized",
      sendCommand: async () => {
        try {
          const payload = await postCommand(state.token, request);
          commandOutcome.current = { kind: "success", payload };
          return commandOutcome.current;
        } catch (error) {
          if (error instanceof DebuggerApiError && error.status === 409) {
            commandOutcome.current = { kind: "stale", payload: error.payload };
            return commandOutcome.current;
          }
          throw error;
        }
      },
      joinCommandResult: async (outcome) => {
        const transportCandidate = extractFrame(outcome.payload);
        if (transportCandidate && recordingReviewHandoffRequired(transportCandidate)) {
          reviewHandoff = true;
          throw new ProductReviewHandoff(
            "Recording review handoff requires a route reload.",
          );
        }
        const joined = await extractJoinedFrame(
          outcome.payload,
          await getCurrentPresentation(state.token),
        );
        if (joined === null) {
          throw new TypeError("Live response has no joinable transport candidate.");
        }
        return prepareJoinedAuthority(joined, {
          previousAuthority,
          continuityResult: "stale_resync",
        });
      },
      getJoined: async () =>
        prepareJoinedAuthority(await getCurrentFrameAndPresentation(state.token), {
          previousAuthority,
          continuityResult: "stale_resync",
        }),
    });
    render();
    const installOutcome = await installPromise;
    if (installOutcome.status === "superseded") {
      return;
    }
    state.offline = false;
    state.resyncRequired = false;
    const payload = commandOutcome.current?.payload;
    const stale = commandOutcome.current?.kind === "stale";
    const notice = extractNotice(payload);
    setNotice(
      stale
        ? payload?.error_code === "command_id_conflict"
          ? "The service rejected a command-ID conflict. Its coherent latest pair was installed; the command was not retried."
          : "This tab was stale. Its coherent latest pair was installed; the command was not retried."
        : installOutcome.resynchronized
          ? "The command completed once, its mixed presentation candidate was discarded, and one fresh GET pair was installed."
          : (notice ??
            (payload?.result === "duplicate"
              ? "Duplicate command recognized; it was not applied again."
              : "Authoritative frame updated.")),
      stale || installOutcome.resynchronized || payload?.result === "duplicate"
        ? "warning"
        : "success",
    );
    if (commandResponseSchedulesShutdown(command, payload)) {
      state.shuttingDown = true;
      setNotice("Exit accepted. The local product server is shutting down.", "info");
    }
    if (
      liveCommandTransaction.allowsDeferredSubmit &&
      liveCommandTransaction.deferredSubmit !== null &&
      !stale &&
      !installOutcome.resynchronized &&
      (payload?.result === "applied" || payload?.result === "no_op") &&
      !state.shuttingDown &&
      !state.resyncRequired &&
      !state.offline &&
      isAuthorizedPresentationFrame(state.presentation) &&
      installedAuthorityIsCoherent() &&
      !isTerminal(state.frame)
    ) {
      deferredSubmitToDispatch = liveCommandTransaction.deferredSubmit;
    }
  } catch (error) {
    if (error instanceof ProductReviewHandoff) {
      state.offline = false;
      state.resyncRequired = false;
      setNotice("Recording review accepted. Opening the replay route…", "info");
    } else if (error instanceof ProductIdentityMismatchError) {
      failClosedProductIdentity(error);
    } else {
      const status = error instanceof DebuggerApiError ? error.status : 0;
      const payload = commandOutcome.current?.payload;
      if (isPresentationJoinRace(error)) {
        clearPresentationAuthority("live_presentation_identity_mismatch");
      } else if (status === 401 || status === 403) {
        clearPresentationAuthority("live_authorization_failure");
      }
      if (commandResponseSchedulesShutdown(command, payload)) {
        state.shuttingDown = true;
        setNotice(
          "Exit was accepted, but no coherent successor presentation was installed while the local product server shuts down.",
          "info",
        );
        return;
      }
      if (
        (status === 0 && PRODUCT_HANDOFF_COMMANDS.has(String(command.command_type))) ||
        (status === 404 && productIdentity?.product_kind === "combat_debugger")
      ) {
        productHandoffOutcomeUnknown = true;
      }
      state.offline =
        error instanceof DebuggerApiError &&
        (status === 0 || status === 401 || status === 403);
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
    if (activeLiveCommandTransaction === liveCommandTransaction) {
      activeLiveCommandTransaction = null;
    }
    if (!reviewHandoff || state.shuttingDown || state.resyncRequired) {
      render();
    }
  }
  if (reviewHandoff && !state.shuttingDown && !state.resyncRequired) {
    reloadForProductHandoff();
    return;
  }
  if (deferredSubmitToDispatch !== null) {
    await dispatchCommand(deferredSubmitToDispatch);
  }
}

/** @param {{reviewHandoff?: boolean}} options */
async function loadCurrentFrame({ reviewHandoff = false } = {}) {
  if (reviewHandoff) {
    reloadForProductHandoff();
    return;
  }
  if (productIdentity === null) {
    state.offline = true;
    state.resyncRequired = true;
    setNotice(
      `Startup blocked: ${startupProductIdentityError ?? "Product bootstrap is unavailable."} Reopen the exact URL printed by the Python launcher.`,
      "error",
    );
    render();
    return;
  }
  if (state.busy || state.shuttingDown) {
    return;
  }
  invalidateReplayArtifactAction();
  state.busy = true;
  if (isReplayMode()) {
    replayPlayback.pause("reconnect");
  }
  setNotice("Fetching the current transport and authorized presentation…", "info");
  const previousFrame = state.frame;
  const previousAuthority = state.authority;
  const retainLastAuthorized = installedPresentationAuthority() !== null;
  let focusReplayTimeline = false;
  try {
    const installPromise = presentationInstallation.installFromGet({
      reason: previousFrame === null ? "initial_authority" : "reconnect_authority",
      pendingPolicy: retainLastAuthorized ? "retain_last_authorized" : "clear",
      getJoined: async () =>
        prepareJoinedAuthority(await getCurrentFrameAndPresentation(state.token), {
          previousAuthority,
          continuityResult: "stale_resync",
        }),
    });
    render();
    const outcome = await installPromise;
    if (outcome.status === "superseded") {
      return;
    }
    const frame = outcome.joined.transport;
    focusReplayTimeline =
      frame.viewer_mode === "replay" && previousFrame?.viewer_mode !== "replay";
    if (focusReplayTimeline) {
      choreographer.clear("replay_handoff");
    }
    if (frame.viewer_mode === "replay") {
      replayPlayback.installCursor(frame.cursor);
      replayPlayback.setConnected(true);
    }
    state.offline = false;
    state.resyncRequired = false;
    setNotice(
      outcome.resynchronized
        ? "A mixed authority GET was discarded; one fresh pair was installed."
        : focusReplayTimeline
          ? "Read-only replay review is ready."
          : "Connected to the local product service.",
      "success",
    );
  } catch (error) {
    if (error instanceof ProductIdentityMismatchError) {
      failClosedProductIdentity(error);
    }
    const status = error instanceof DebuggerApiError ? error.status : 0;
    if (error instanceof TypeError) {
      clearPresentationAuthority("reconnect_invalid_presentation_identity");
    } else if (isPresentationJoinRace(error)) {
      clearPresentationAuthority("reconnect_presentation_identity_mismatch");
    } else if (status === 401 || status === 403) {
      clearPresentationAuthority("reconnect_authorization_failure");
    }
    state.offline = status === 0 || status === 401 || status === 403;
    state.resyncRequired = true;
    if (isReplayMode()) {
      replayPlayback.setConnected(false);
    }
    setNotice(
      status === 401 || status === 403
        ? "Debugger capability is invalid. Reopen the exact URL printed by the Python launcher."
        : error instanceof Error
          ? `${error.message} Reconnect to request one fresh authority pair.`
          : "Could not load debugger frame.",
      "error",
    );
  } finally {
    state.busy = false;
    render();
  }
  if (focusReplayTimeline && !state.resyncRequired) {
    elements.replayTimeline.focus({ preventScroll: true });
  }
}

/**
 * Move pointer-originated draft flow back to the battlefield while preserving
 * native keyboard-control ownership across the serialized request. Keyboard
 * activation uses the battlefield only as a safe temporary focus anchor and
 * may not turn a second Enter into Submit.
 *
 * @param {Record<string, unknown>} command
 * @param {{
 *   restoreFocusTo?: HTMLElement | null,
 *   allowDeferredSubmit?: boolean,
 * }} options
 */
function dispatchCommandFromDraftControl(
  command,
  { restoreFocusTo = null, allowDeferredSubmit = true } = {},
) {
  elements.battlefield.focus({ preventScroll: true });
  const completion = dispatchCommand(command, { allowDeferredSubmit });
  if (restoreFocusTo === null) {
    void completion;
    return;
  }
  void completion.finally(() => {
    if (
      document.activeElement === elements.battlefield &&
      restoreFocusTo.isConnected &&
      !restoreFocusTo.matches(":disabled") &&
      restoreFocusTo.getAttribute("aria-disabled") !== "true"
    ) {
      restoreFocusTo.focus({ preventScroll: true });
    }
  });
}

bindBattlefieldControls({
  battlefield: elements.battlefield,
  toWorldPoint: (point) => battlefieldRenderer.toWorldPoint(point),
  onCommand: (command) => dispatchCommand(command),
  onPointerCommand: (target, command) => {
    if (
      command.button === "primary" &&
      target instanceof Element &&
      target.closest(".agent[data-presentation-key]") !== null
    ) {
      openAgentDetails();
    }
  },
  onHelp: () => elements.helpDialog.showModal(),
  isInteractive: liveBattlefieldCommandsInteractive,
  onFencedEnter: retainFencedEnter,
  onReleaseFocus: () => {
    const firstCommand = /** @type {HTMLButtonElement | null} */ (
      elements.commandDeck?.querySelector("button:not([disabled])") ?? null
    );
    const focusTarget = firstCommand ?? elements.helpButton;
    focusTarget.focus({ preventScroll: true });
  },
});

elements.battlefield.addEventListener(
  "pointerdown",
  (/** @type {PointerEvent} */ event) => {
    if (event.target instanceof Element) {
      const tooltipOwner = event.target.closest("[data-tooltip-owner]");
      const agent = event.target.closest(".agent[data-presentation-key]");
      if (tooltipOwner !== null && tooltipOwner !== agent) {
        event.stopImmediatePropagation();
        return;
      }
    }
    if (
      event.button !== 0 ||
      event.shiftKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.altKey
    ) {
      return;
    }
    const activation = authorizedAgentActivationFromTarget(event.target);
    if (activation === null) {
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
    elements.battlefield.focus({ preventScroll: true });
    activateAuthorizedAgent(activation.presentationKey);
  },
  true,
);

elements.battlefield.addEventListener(
  "keydown",
  (/** @type {KeyboardEvent} */ event) => {
    if (
      event.key === "Tab" &&
      event.target === elements.battlefield &&
      authorizedPresentationAudience(state.presentation) === "agent_pov"
    ) {
      // Preserve native forward/backward focus navigation while preventing the
      // fixed Agent POV recipient from being cycled through the live service.
      event.stopImmediatePropagation();
      return;
    }
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    if (event.target instanceof Element) {
      const tooltipOwner = event.target.closest("[data-tooltip-owner]");
      const agent = event.target.closest(".agent[data-presentation-key]");
      if (tooltipOwner !== null && tooltipOwner !== agent) {
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
    }
    const activation = authorizedAgentActivationFromTarget(event.target);
    if (activation !== null) {
      event.preventDefault();
      event.stopImmediatePropagation();
      activateAuthorizedAgent(activation.presentationKey);
    }
  },
  true,
);

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

elements.resetButton.addEventListener("click", () => {
  dispatchCommand({ command_type: "reset" });
});

/** @type {"pointer" | "keyboard" | null} */
let commandTargetSelectionModality = null;
elements.commandTargetSelect.addEventListener("pointerdown", () => {
  commandTargetSelectionModality = "pointer";
});
elements.commandTargetSelect.addEventListener("pointercancel", () => {
  commandTargetSelectionModality = null;
});
elements.commandTargetSelect.addEventListener("keydown", () => {
  commandTargetSelectionModality = "keyboard";
});
elements.commandTargetSelect.addEventListener("change", () => {
  const pointerOriginated = commandTargetSelectionModality === "pointer";
  commandTargetSelectionModality = null;
  const command = targetSelectionCommand(elements.commandTargetSelect.value, {
    actorPov: authorizedPresentationAudience(state.presentation) === "agent_pov",
  });
  if (!command) {
    return;
  }
  dispatchCommandFromDraftControl(command, {
    restoreFocusTo: pointerOriginated ? null : elements.commandTargetSelect,
    allowDeferredSubmit: pointerOriginated,
  });
});

elements.recordingFinishButton.addEventListener("click", () => {
  void dispatchCommand({ command_type: "finish_and_review" });
});

elements.recordingReviewButton.addEventListener("click", () => {
  void dispatchCommand({ command_type: "review_replay" });
});

elements.recordingRetryButton.addEventListener("click", () => {
  void dispatchCommand({ command_type: "retry_save" });
});

function dispatchRecordingSaveAs() {
  const command = recordingSaveAsCommand(elements.recordingSaveAsInput.value);
  if (!command) {
    setNotice(
      "Save As requires a basename ending in .marlbg-replay.json; paths, spaces, and hidden filenames are not accepted.",
      "warning",
    );
    renderConnection();
    elements.recordingSaveAsInput.focus({ preventScroll: true });
    return;
  }
  void dispatchCommand(command);
}

elements.recordingSaveAsButton.addEventListener("click", dispatchRecordingSaveAs);
elements.recordingSaveAsInput.addEventListener(
  "keydown",
  (/** @type {KeyboardEvent} */ event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    dispatchRecordingSaveAs();
  },
);

elements.recordingDiscardDialog.addEventListener("close", () => {
  pendingRecordingReplacement = null;
});

elements.recordingDiscardDialog.addEventListener("cancel", () => {
  pendingRecordingReplacement = null;
});

elements.recordingDiscardConfirmButton.addEventListener("click", () => {
  const replacement = pendingRecordingReplacement;
  if (!replacement || recordingStatus()?.discard_available !== true) {
    pendingRecordingReplacement = null;
    elements.recordingDiscardDialog.close();
    setNotice(
      "The recording lifecycle changed before discard confirmation. No episode replacement was sent.",
      "warning",
    );
    renderConnection();
    return;
  }
  pendingRecordingReplacement = null;
  elements.recordingDiscardDialog.close();
  void dispatchCommand({
    command_type: "confirm_discard_and_replace",
    replacement,
  });
});

elements.exitButton.addEventListener("click", () => {
  invalidateReplayArtifactAction();
  if (isReplayMode()) {
    void dispatchReplayCommand({ command_type: "exit" });
  } else {
    void dispatchCommand({ command_type: "exit" });
  }
});

elements.reconnectButton.addEventListener("click", () => {
  invalidateReplayArtifactAction();
  if (productHandoffOutcomeUnknown) {
    reloadForProductHandoff();
    return;
  }
  void loadCurrentFrame();
});

elements.helpButton.addEventListener("click", () => {
  elements.helpDialog.showModal();
});

bindReplayTimelineControls(replayTimelineElements, replayPlayback);

/** @param {Event} event */
const handleVisualFilterChange = (event) => {
  const input = event.target;
  if (
    !(input instanceof HTMLInputElement) ||
    input.type !== "checkbox" ||
    !input.dataset.visualFilterId
  ) {
    return;
  }
  applyVisualFilterAction({
    type: "set",
    filterId: input.dataset.visualFilterId,
    enabled: input.checked,
  });
};

elements.visualFilterOptions.addEventListener("change", handleVisualFilterChange);

elements.enableAllVisualFiltersButton.addEventListener("click", () => {
  applyVisualFilterAction({ type: "enable_all" });
});

elements.disableAllVisualFiltersButton.addEventListener("click", () => {
  applyVisualFilterAction({ type: "disable_all" });
});

elements.replayExportPngButton.addEventListener("click", () => {
  void exportReplayBattlefieldPng();
});

elements.replayDownloadMetricsButton.addEventListener("click", () => {
  void downloadReplayMetricReport();
});

elements.replayRangesButton.addEventListener("click", () => {
  if (!isReplayMode() || elements.replayRangesButton.disabled) {
    return;
  }
  if (authorizedPresentationAudience(state.presentation) === "agent_pov") {
    if (toggleAgentLocalRanges()) {
      render();
    }
    return;
  }
  if (authorizedPresentationAudience(state.presentation) !== "researcher") {
    return;
  }
  const frame = state.frame;
  if (!frame) {
    return;
  }
  void dispatchReplayCommand({
    command_type: "set_ranges",
    show_ranges: frame.show_ranges !== true,
  });
});

elements.replayClearReferenceButton.addEventListener("click", () => {
  if (!isReplayMode() || elements.replayClearReferenceButton.disabled) {
    return;
  }
  if (authorizedPresentationAudience(state.presentation) === "agent_pov") {
    if (setLocalInspectedPresentationKey(null)) {
      closeAgentDetailsWithoutLatching();
      render();
      elements.battlefield.removeAttribute("aria-activedescendant");
    }
    return;
  }
  if (authorizedPresentationAudience(state.presentation) !== "researcher") {
    return;
  }
  void dispatchReplayCommand({
    command_type: "select_agent",
    selected_global_slot: null,
  });
});

elements.liveRangesButton.addEventListener("click", () => {
  if (isReplayMode() || elements.liveRangesButton.disabled) {
    return;
  }
  if (authorizedPresentationAudience(state.presentation) === "agent_pov") {
    if (toggleAgentLocalRanges()) {
      render();
    }
    return;
  }
  if (authorizedPresentationAudience(state.presentation) !== "researcher") {
    return;
  }
  void dispatchCommand(keyboardCommand("g"));
});

document.addEventListener("visibilitychange", () => {
  replayPlayback.setHidden(document.hidden);
});

if (elements.commandDeck) {
  const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
    elements.commandDeck.querySelectorAll("button[data-key]")
  );
  for (const button of buttons) {
    button.addEventListener("click", (event) => {
      if (button.getAttribute("aria-disabled") === "true") {
        setNotice(
          button.dataset.tooltipText ??
            "That pending choice is unavailable in the current authoritative mask.",
          "warning",
        );
        renderConnection();
        return;
      }
      const restoreKeyboardFocus = event.detail === 0;
      dispatchCommandFromDraftControl(
        keyboardCommand(button.dataset.key ?? "", {
          shiftKey: button.dataset.shift === "true",
        }),
        {
          restoreFocusTo: restoreKeyboardFocus ? button : null,
          allowDeferredSubmit: !restoreKeyboardFocus,
        },
      );
    });
  }
}

const battlefieldResizeObserver = new ResizeObserver(scheduleBattlefieldResize);
battlefieldResizeObserver.observe(elements.battlefieldShell);
replayPlayback.setHidden(document.hidden);

for (const { panelId, panel, body } of scientificDisclosures) {
  /** @param {Event} event */
  const blockPendingSummaryActivation = (event) => {
    const summary = panel.querySelector(":scope > summary");
    if (
      !(event.target instanceof Element) ||
      !(summary instanceof Element) ||
      (event.target !== summary && !summary.contains(event.target))
    ) {
      return;
    }
    if (installedActivePresentationPreference(state.presentation) !== null) {
      expectedDisclosureToggles.delete(panel);
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
  };
  panel.addEventListener("click", blockPendingSummaryActivation, true);
  panel.addEventListener(
    "keydown",
    (event) => {
      if (event.key === "Enter" || event.key === " ") {
        blockPendingSummaryActivation(event);
      }
    },
    true,
  );
  panel.addEventListener("toggle", () => {
    const expected = expectedDisclosureToggles.get(panel);
    if (expected?.open === panel.open) {
      expectedDisclosureToggles.delete(panel);
      return;
    }
    expectedDisclosureToggles.delete(panel);
    const preference = installedActivePresentationPreference(state.presentation);
    if (preference === null) {
      setProgrammaticDisclosureOpen(panel, false);
      return;
    }
    const saved = preference.disclosures[panelId];
    preference.disclosures[panelId] = {
      open: panel.open,
      scrollTop: panel.open ? (saved?.scrollTop ?? 0) : body.scrollTop,
    };
    if (panel.open) {
      body.scrollTop = preference.disclosures[panelId].scrollTop;
      return;
    }
    if (panelId === "agent-details") {
      preference.agentDetailsAutoOpenAllowed = false;
    }
    const active = document.activeElement;
    const summary = panel.querySelector(":scope > summary");
    if (
      active instanceof Element &&
      active !== summary &&
      panel.contains(active) &&
      summary instanceof HTMLElement
    ) {
      summary.focus({ preventScroll: true });
    }
  });
}

elements.visualFilters.addEventListener("toggle", () => {
  if (elements.visualFilters.open) {
    return;
  }
  const active = document.activeElement;
  const summary = elements.visualFilters.querySelector(":scope > summary");
  if (
    active instanceof Element &&
    active !== summary &&
    elements.visualFilters.contains(active) &&
    summary instanceof HTMLElement
  ) {
    summary.focus({ preventScroll: true });
  }
});

elements.visualKey.addEventListener("toggle", () => {
  if (elements.visualKey.open) {
    return;
  }
  const active = document.activeElement;
  const summary = elements.visualKey.querySelector(":scope > summary");
  if (
    active instanceof Element &&
    active !== summary &&
    elements.visualKey.contains(active) &&
    summary instanceof HTMLElement
  ) {
    summary.focus({ preventScroll: true });
  }
});

registerControlHelp();
if (productIdentity === null) {
  state.offline = true;
  state.resyncRequired = true;
  setNotice(
    `Startup blocked: ${startupProductIdentityError ?? "Product bootstrap is unavailable."} Reopen the exact URL printed by the Python launcher.`,
    "error",
  );
  render();
} else {
  render();
  void loadCurrentFrame();
}
