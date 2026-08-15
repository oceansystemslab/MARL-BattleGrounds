import {
  acquireCapabilityToken,
  acquireClientId,
  DebuggerApiError,
  extractFrame,
  extractJoinedFrame,
  extractNotice,
  getCurrentFrameAndPresentation,
  getCurrentPresentation,
  getReplayTimeline,
  postCommand,
  postReplayCommand,
} from "./api.js";
import {
  authorizedOracleCommandSlotForPresentationKey,
  authorizedOracleCommandSlotForPublicAgentId,
  authorizedPresentationAudience,
  authorizedPresentationInspectionState,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
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
import { explainAgent, explainLegality } from "./explanations.js";
import {
  DebuggerPanels,
  disclosurePanelInitiallyOpen,
  panelDisclosureAuthorityKey,
} from "./panels.js";
import { PresentationInstallCoordinator } from "./presentation-install.js";
import {
  bindReplayTimelineControls,
  ReplayPlaybackController,
  renderReplayTimelineControls,
  replayCommandRequest,
  replayTimelineSimulatorStep,
  validateReplayCommandOutcome,
} from "./replay-controls.js";
import { BattlefieldRenderer } from "./scene.js";
import {
  createSemanticDescriptor,
  createTooltipController,
  registerTooltipOwner,
  renderSemanticDescriptor,
} from "./tooltip.js";

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
  revisionValue: requiredElement("revision-value"),
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
  replayIncomingValue: requiredElement("replay-incoming-value"),
  replayFirstButton: requiredElement("replay-first-button"),
  replayBackTenButton: requiredElement("replay-back-ten-button"),
  replayPreviousButton: requiredElement("replay-previous-button"),
  replayPlayPauseButton: requiredElement("replay-play-pause-button"),
  replayNextButton: requiredElement("replay-next-button"),
  replayForwardTenButton: requiredElement("replay-forward-ten-button"),
  replayLastButton: requiredElement("replay-last-button"),
  replayFrameSlider: requiredElement("replay-frame-slider"),
  replayFramePosition: requiredElement("replay-frame-position"),
  replayRangesButton: requiredElement("replay-ranges-button"),
  replayClearReferenceButton: requiredElement("replay-clear-reference-button"),
  reconnectButton: requiredElement("reconnect-button"),
  helpButton: requiredElement("help-button"),
  exitButton: requiredElement("exit-button"),
  resetButton: requiredElement("reset-button"),
  liveRangesButton: requiredElement("live-ranges-button"),
  notice: requiredElement("notice"),
  scenarioDescription: requiredElement("scenario-description"),
  battlefieldShell: requiredElement("battlefield-shell"),
  battlefield: requiredElement("battlefield"),
  battlefieldEmpty: requiredElement("battlefield-empty"),
  commandDeck: document.querySelector(".command-deck"),
  commandControlledActor: requiredElement("command-controlled-actor"),
  roster: requiredElement("roster"),
  rosterCount: requiredElement("roster-count"),
  agentDetails: requiredElement("agent-details"),
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
  eventFeed: requiredElement("event-feed"),
  eventCount: requiredElement("event-count"),
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
  replayOnly: /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll("[data-replay-only]")
  ),
  disclosurePanels: /** @type {NodeListOf<HTMLDetailsElement>} */ (
    document.querySelectorAll(
      "details.command-deck, #roster-details, #agent-details, #pending-turn-details, #latest-transition-details, #events-details, #visual-key, #technical-frame-details",
    )
  ),
};

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
    "#battlefield",
    "Battlefield Commands",
    "Inspect the authoritative scene. Researcher live view supports pointer control and targeting; Agent POV bodies remain passive and draft controls own actions.",
    "composite",
  ],
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
  ["#replay-first-button", "First replay tick", "Seek to settled replay tick zero."],
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
  [
    "#replay-last-button",
    "Last replay tick",
    "Seek to the end of the captured prefix.",
  ],
  [
    "#replay-frame-slider",
    "Replay tick",
    "Seek to an exact captured tick after a short debounce.",
  ],
  [
    "#replay-ranges-button",
    "Replay ranges",
    "Toggle recorded Oracle View range presentation.",
  ],
  [
    "#replay-clear-reference-button",
    "Clear Reference",
    "Clear the Oracle View inspector highlight; the range anchor is unchanged.",
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
    "#live-ranges-button",
    "Ranges",
    "Toggle server-authored Oracle View range presentation.",
  ],
  [
    "#reset-button",
    "Reset",
    "Start a deterministic fresh episode; recorded prefixes require confirmation.",
  ],
  [
    "#visual-key > summary",
    "Visual key",
    "Explain the non-color visual grammar used on the battlefield.",
  ],
  [
    ".diagnostics > summary",
    "Technical frame",
    "Inspect authorized wire and diagnostic details.",
  ],
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
    syncCompactActiveCombatPriority(presentation);
  },
});

const replayTimelineElements = {
  root: elements.replayTimeline,
  firstButton: elements.replayFirstButton,
  backTenButton: elements.replayBackTenButton,
  previousButton: elements.replayPreviousButton,
  playPauseButton: elements.replayPlayPauseButton,
  nextButton: elements.replayNextButton,
  forwardTenButton: elements.replayForwardTenButton,
  lastButton: elements.replayLastButton,
  slider: elements.replayFrameSlider,
  position: elements.replayFramePosition,
  tickForFrameIndex: (/** @type {number} */ frameIndex) =>
    replayTimelineSimulatorStep(state.timeline, frameIndex),
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
 * Remove every presentation-owned surface synchronously. The raw transport
 * epoch stays installed for diagnostics and command-outcome accounting until a
 * complete successor pair is ready; it is never used to keep the battlefield
 * or scientific panels populated while authority is pending.
 *
 * @param {string} reason
 */
function clearPresentationAuthority(reason) {
  state.authority = null;
  state.presentation = null;
  disclosureAuthorityKey = null;
  tooltipController.hide();
  choreographer.clear(reason);
  battlefieldRenderer.render(null, { offline: true });
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
  elements.pendingScope.textContent = "Waiting for an authorized outgoing inspection.";
  const pendingLabel = elements.pendingCard.querySelector(".action-card__label");
  if (pendingLabel) {
    pendingLabel.textContent = "NO AUTHORIZED INSPECTION";
  }
  elements.stepValue.textContent = "—";
  elements.transitionValue.textContent = "—";
  elements.audienceBadge.textContent = "View unavailable";
  elements.audienceBadge.dataset.audience = "unavailable";
  document.documentElement.dataset.audience = "unavailable";
  elements.scenarioDescription.textContent = "Waiting for an authorized presentation.";
  clearPresentationTooltipOwner(elements.replayArtifactReference);
  elements.replayArtifactReference.textContent =
    "Unavailable while authority is pending";
  elements.replayProcessingBadge.textContent = "Unavailable while authority is pending";
  elements.replayIncomingValue.textContent = "Unavailable while authority is pending";
  elements.replayRangesButton.disabled = true;
  elements.replayClearReferenceButton.disabled = true;
  elements.liveRangesButton.disabled = true;
  elements.viewSelect.disabled = true;
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
  for (const panel of elements.disclosurePanels) {
    if (panel.id !== "visual-key") {
      panel.open = false;
    }
  }
  elements.agentDetails.open = false;
  elements.agentDetails.removeAttribute("data-tone");
  elements.agentDetails.removeAttribute("data-accent");
  elements.selectionHeading.textContent = "Agent Details";
  elements.battlefield.removeAttribute("aria-activedescendant");
  document.documentElement.dataset.presentationAuthority = "pending";
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
  state.authority = joined;
  state.frame = joined.transport;
  state.presentation = joined.presentation;
  state.timeline = isRecord(joined.timeline) ? joined.timeline : null;
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
  clear: clearPresentationAuthority,
  install: installJoinedAuthority,
  isJoinRace: isPresentationJoinRace,
});

function reloadForProductHandoff() {
  if (productHandoffReloadRequested) {
    return;
  }
  productHandoffReloadRequested = true;
  window.location.reload();
}

/**
 * @param {unknown} descriptor
 * @param {{owner: Element, trigger: Element | null}} _context
 */
function showSemanticInspector(descriptor, _context) {
  const normalized = createSemanticDescriptor(descriptor);
  if (
    isReplayMode() &&
    authorizedPresentationAudience(state.presentation) !== "researcher" &&
    normalized.kind === "agent"
  ) {
    const ownerKey = _context.owner
      .closest("[data-presentation-key]")
      ?.getAttribute("data-presentation-key");
    if (
      typeof ownerKey !== "string" ||
      ownerKey !== state.presentation?.recipient_presentation_key
    ) {
      return;
    }
  }
  renderSemanticDescriptor({
    descriptor: normalized,
    title: elements.selectionHeading,
    details: elements.selectionCard,
    surface: "full",
  });
  elements.agentDetails.dataset.tone = normalized.tone;
  elements.agentDetails.dataset.accent = normalized.accent;
  elements.agentDetails.open = true;
}

function registerControlHelp() {
  for (const [selector, title, summary, kind = "control"] of CONTROL_HELP) {
    for (const control of document.querySelectorAll(selector)) {
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
  }
}

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** @type {string | null} */
let disclosureAuthorityKey = null;

function syncDisclosurePanels() {
  const key = panelDisclosureAuthorityKey(state.presentation);
  if (key === null || key === disclosureAuthorityKey) {
    return;
  }
  disclosureAuthorityKey = key;
  const replay = isReplayMode();
  for (const panel of elements.disclosurePanels) {
    panel.open = disclosurePanelInitiallyOpen(panel.id, replay);
  }
}

function openAgentDetails() {
  elements.agentDetails.open = true;
}

function isReplayMode() {
  return state.frame?.viewer_mode === "replay";
}

/**
 * Carry replay animation intent beside, never inside, the authorized scientific
 * presentation. Only the currently installed branded pair may authorize a
 * transient replay of its incoming transition.
 *
 * @param {unknown} presentation
 * @returns {{animateIncoming: boolean}}
 */
function installedChoreographyControl(presentation) {
  const authority = state.authority;
  return {
    animateIncoming:
      isAuthorizedPresentationFrame(presentation) &&
      isJoinedTransportAndAuthorizedPresentationV1(authority) &&
      authority.presentation === presentation &&
      authority.transport === state.frame &&
      authority.transport.viewer_mode === "replay" &&
      authority.transport.animate_incoming === true,
  };
}

function installedAuthorityIsCoherent() {
  const authority = state.authority;
  return (
    isJoinedTransportAndAuthorizedPresentationV1(authority) &&
    authority.transport === state.frame &&
    authority.presentation === state.presentation
  );
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
  return isRecord(state.frame?.recording) ? state.frame.recording : null;
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
  "Inspection-only scripted battlefield. Use Advance scripted frame for the next authorized step.";
const SCRIPTED_BATTLEFIELD_INSTRUCTIONS =
  "Scripted live view is inspection-only. Battlefield bodies are passive and cannot submit actions. Use the single Advance scripted frame button to apply the next authorized script step; it is disabled when the script is complete.";

function applyScriptedBattlefieldBoundaryCopy() {
  elements.battlefield.setAttribute("role", "img");
  elements.battlefield.tabIndex = -1;
  elements.battlefield.setAttribute("aria-label", SCRIPTED_BATTLEFIELD_LABEL);
  elements.battlefieldInstructions.textContent = SCRIPTED_BATTLEFIELD_INSTRUCTIONS;
}

function renderViewerBoundary() {
  const replay = isReplayMode();
  const scientificFenced = recordingScientificControlsFenced();
  const audience = authorizedPresentationAudience(state.presentation);
  const scriptedInspection = liveScriptedInspectionOnly();
  document.documentElement.dataset.viewerMode = replay ? "replay" : "live";
  for (const element of elements.liveOnly) {
    element.toggleAttribute("hidden", replay);
  }
  for (const element of elements.replayOnly) {
    element.toggleAttribute("hidden", !replay);
  }
  elements.replayTimeline.toggleAttribute("hidden", !replay);
  if (scriptedInspection && !scientificFenced) {
    applyScriptedBattlefieldBoundaryCopy();
  } else {
    elements.battlefield.setAttribute(
      "role",
      replay ? "group" : scientificFenced ? "img" : "application",
    );
    elements.battlefield.tabIndex = replay || scientificFenced ? -1 : 0;
    elements.battlefield.setAttribute(
      "aria-label",
      replay
        ? "Read-only replay battlefield snapshot. Authorized agents can be inspected."
        : scientificFenced
          ? "Read-only recording closeout battlefield snapshot."
          : "Interactive battlefield. Press Help for keyboard controls.",
    );
    elements.battlefieldInstructions.textContent = replay
      ? "Replay transport changes only the selected recorded frame. Activate an authorized agent to inspect it; the battlefield cannot submit actions or advance the simulator."
      : scientificFenced
        ? "Recording closeout has fenced simulator and pending-action controls. Presentation, recovery, review, and Exit controls remain available."
        : audience === "agent_pov"
          ? "Agent POV bodies are inspectable and passive; they cannot change control or submit actions. Use the command draft controls or battlefield keyboard shortcuts to prepare actions. Escape moves focus to the command deck, or to Help while commands are unavailable. Tab behaves normally in the side panel."
          : "The battlefield owns debugger keyboard commands while it has focus. Left click controls an authorized actor, Shift plus left click selects an authorized target, and right click clears the target. Tab and Shift Tab cycle controlled actors here. Escape clears the target and moves focus to the command deck, or to Help while commands are unavailable. Tab behaves normally in the side panel.";
  }
  elements.selectionHeading.textContent = "Agent Details";
}

function renderReplayMetadata() {
  const frame = state.frame;
  const presentation = state.presentation;
  if (!isReplayMode() || !frame) {
    return;
  }
  const presentationInstalled = isAuthorizedPresentationFrame(presentation);
  const source =
    presentationInstalled && isRecord(presentation.source) ? presentation.source : {};
  const completion = isRecord(frame.completion) ? frame.completion : {};
  elements.replayArtifactReference.textContent = String(
    presentationInstalled
      ? (source.source_artifact_id ?? "Unavailable in this audience")
      : "Unavailable while authority is pending",
  );
  elements.replayArtifactReference.removeAttribute("title");
  if (presentationInstalled) {
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
            value: String(source.source_artifact_id ?? "Unavailable in this audience"),
            metadata: { compact: true, full: true },
          },
          {
            label: "Canonical digest",
            value: String(
              source.source_artifact_digest_sha256 ?? "Unavailable in this audience",
            ),
            metadata: { compact: false, full: true },
          },
        ],
        sections: [],
        metadata: { compact: true, full: true },
        anchor: "element",
      }),
    );
  } else {
    clearPresentationTooltipOwner(elements.replayArtifactReference);
  }
  elements.replayCompletionBadge.textContent = humanize(
    completion.completion_state ?? "unavailable",
  );
  elements.replayProcessingBadge.textContent = !presentationInstalled
    ? "Unavailable while authority is pending"
    : authorizedPresentationAudience(presentation) !== "researcher"
      ? "Not available in actor POV"
      : "Authorized replay";
  elements.replayEndReason.textContent = String(
    completion.public_end_or_failure_reason ??
      completion.end_or_failure_reason ??
      (asArray(completion.completion_bases).length > 0
        ? asArray(completion.completion_bases).map(humanize).join(" + ")
        : "Captured prefix"),
  );
  const incomingTransitionId = authorizedIncomingTransitionId(presentation);
  elements.replayIncomingValue.textContent = !presentationInstalled
    ? "Unavailable while authority is pending"
    : incomingTransitionId
      ? String(incomingTransitionId)
      : "Initial frame";
  elements.replayRangesButton.setAttribute(
    "aria-pressed",
    String(frame.show_ranges === true),
  );
  elements.replayRangesButton.disabled =
    state.busy || authorizedPresentationAudience(presentation) !== "researcher";
  const inspection = authorizedPresentationInspectionState(presentation).inspection;
  elements.replayClearReferenceButton.disabled =
    state.busy ||
    authorizedPresentationAudience(presentation) !== "researcher" ||
    !isRecord(inspection);
  renderReplayTimelineControls(replayTimelineElements, replayPlayback.snapshot());
}

/** @type {Readonly<Record<string, string>>} */
const recordingPersistenceLabels = Object.freeze({
  target_unavailable: "Destination unavailable",
  publication_failed: "Publication failed",
  verification_failed: "Publication verification failed",
});

function renderRecordingControls() {
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
  const presentation = state.presentation;
  renderViewerBoundary();
  const replay = isReplayMode();
  const disabled =
    state.busy || !frame || state.shuttingDown || state.resyncRequired || state.offline;
  const restartControlsBlocked = recordingRestartControlsBlocked();

  elements.revisionValue.textContent = frame ? String(frame.revision ?? "—") : "—";
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

  elements.scenarioDescription.textContent = presentation
    ? replay
      ? `${publicAudienceLabel(authorizedPresentationAudience(presentation))} · recorded frame ${frame?.cursor?.frame_index ?? "—"} of ${frame?.cursor?.final_frame_index ?? "—"}`
      : `${publicAudienceLabel(authorizedPresentationAudience(presentation))} · authorized live presentation`
    : "Waiting for the Python debugger service.";

  const viewMode = frame?.view_mode === "agent_pov" ? "pov" : frame?.view_mode;
  if (typeof viewMode === "string") {
    elements.viewSelect.value = viewMode;
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
  const researcherLive =
    !replay && authorizedPresentationAudience(presentation) === "researcher";
  elements.liveRangesButton.hidden = !researcherLive;
  elements.liveRangesButton.setAttribute(
    "aria-pressed",
    String(researcherLive && frame?.show_ranges === true),
  );
  elements.liveRangesButton.disabled = disabled || !researcherLive;
  renderRecordingControls();
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
    const basic = asArray(pairMask)[0] === true ? "B ✓" : "B ×";
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
  const controlledPublicAgentId = inspection.actor_public_agent_id;
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
  const controlledSlot = authorizedOracleCommandSlotForPresentationKey(
    presentation,
    inspection.actor_presentation_key,
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
  const basicAvailable = asArray(pairMask)[0] === true;
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
  setAuthoritativeAvailability(
    elements.basicButton,
    basicAvailable,
    `Basic ability is ${basicAvailable ? "" : "not "}available this tick.`,
    explainLegality({ lane_0_available: basicAvailable }, 0),
  );
  setAuthoritativeAvailability(
    elements.ultimateButton,
    ultimateAvailable,
    `Ultimate ability is ${ultimateAvailable ? "" : "not "}available this tick.`,
    explainLegality({ lane_1_available: ultimateAvailable }, 1),
  );
}

function renderCommandAvailability() {
  const presentation = state.presentation;
  const inspectionState = authorizedPresentationInspectionState(presentation);
  const editableDraft = inspectionState.state_kind === "live_editable";
  const scriptedAdvance = liveScriptedInspectionOnly();
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
  document.documentElement.dataset.submissionBlocked = String(
    presentation.submissionBlocked,
  );
}

function applyReplayReferenceSemantics() {
  const presentation = state.presentation;
  if (authorizedPresentationAudience(presentation) !== "researcher") {
    return;
  }
  const scene = authorizedPresentationSceneView(presentation);
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
  const classMechanics = asArray(scene?.class_mechanics).find(
    (candidate) => isRecord(candidate) && candidate.class_id === agent.class_id,
  );
  const ariaLabel = reference.getAttribute("aria-label") ?? `Agent ID ${publicAgentId}`;
  reference.setAttribute(
    "aria-label",
    ariaLabel.replace(/selected target/giu, "Reference"),
  );
  registerTooltipOwner(
    reference,
    explainAgent(
      agent,
      { reference: true, audience: "researcher" },
      isRecord(classMechanics) ? classMechanics : null,
      asArray(scene?.agents),
    ),
  );
}

/**
 * Return an authorized replay-inspection target from the current rendered SVG.
 * Researcher replay may inspect every authorized scene agent; recipient replay
 * exposes activation only for its own recorded actor.
 *
 * @param {unknown} target
 */
function replayAgentActivation(target) {
  if (!isReplayMode() || !(target instanceof Element)) {
    return null;
  }
  const element = target.closest(".agent[data-presentation-key]");
  if (!(element instanceof SVGElement)) {
    return null;
  }
  const presentationKey = element.dataset.presentationKey;
  const presentation = state.presentation;
  const agent = asArray(authorizedPresentationSceneView(presentation)?.agents).find(
    (candidate) =>
      isRecord(candidate) && candidate.presentation_key === presentationKey,
  );
  if (typeof presentationKey !== "string" || !isRecord(agent)) {
    return null;
  }
  if (authorizedPresentationAudience(presentation) === "researcher") {
    const commandSlot = authorizedOracleCommandSlotForPresentationKey(
      presentation,
      presentationKey,
    );
    return Number.isInteger(commandSlot)
      ? { element, presentationKey, commandSlot, agent, researcher: true }
      : null;
  }
  return presentationKey === presentation?.recipient_presentation_key
    ? { element, presentationKey, commandSlot: null, agent, researcher: false }
    : null;
}

function installReplayAgentActivation() {
  if (!isReplayMode()) {
    return;
  }
  const scene = authorizedPresentationSceneView(state.presentation);
  const agents = asArray(scene?.agents);
  const selection = isRecord(scene?.selection) ? scene.selection : {};
  for (const element of elements.battlefield.querySelectorAll(
    ".agent[data-presentation-key]",
  )) {
    const activation = replayAgentActivation(element);
    if (activation === null) {
      element.removeAttribute("tabindex");
      element.removeAttribute("role");
      element.removeAttribute("aria-description");
      continue;
    }
    element.setAttribute("tabindex", "0");
    element.setAttribute("role", "button");
    const identity = agentIdentity(activation.agent.public_agent_id);
    element.setAttribute(
      "aria-label",
      `${identity}. Activate to open authorized Agent Details${activation.researcher ? " and set replay Reference" : ""}.`,
    );
    const classMechanics = asArray(scene?.class_mechanics).find(
      (candidate) =>
        isRecord(candidate) && candidate.class_id === activation.agent.class_id,
    );
    registerTooltipOwner(
      element,
      explainAgent(
        activation.agent,
        {
          audience: activation.researcher ? "researcher" : "agent_pov",
          controlled:
            selection.controlled_presentation_key === activation.presentationKey,
          selected: selection.selected_presentation_key === activation.presentationKey,
        },
        isRecord(classMechanics) ? classMechanics : null,
        agents,
      ),
    );
  }
}

/** @param {unknown} target */
function activateReplayAgent(target) {
  const activation = replayAgentActivation(target);
  if (activation === null) {
    return false;
  }
  openAgentDetails();
  if (activation.researcher) {
    void dispatchReplayCommand({
      command_type: "select_agent",
      selected_global_slot: activation.commandSlot,
    });
  }
  return true;
}

function render() {
  const presentationFrame = state.presentation;
  renderConnection();
  renderSessionToolbar();
  syncDisclosurePanels();
  battlefieldRenderer.render(presentationFrame, {
    offline: state.offline,
    showRanges: state.frame?.show_ranges === true,
  });
  if (liveScriptedInspectionOnly() && !recordingScientificControlsFenced()) {
    applyScriptedBattlefieldBoundaryCopy();
  }
  installReplayAgentActivation();
  applyReplayReferenceSemantics();
  try {
    choreographer.presentFrame(
      presentationFrame,
      battlefieldRenderer.choreographySurface(),
      installedChoreographyControl(presentationFrame),
    );
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
  panels.render(presentationFrame, {
    busy: state.busy,
    shuttingDown:
      state.shuttingDown ||
      recordingScientificControlsFenced() ||
      liveScriptedInspectionOnly(),
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
  if (!changed || !presentation.active || !state.presentation) {
    return;
  }
  try {
    choreographer.reproject(
      state.presentation,
      battlefieldRenderer.choreographySurface(),
      installedChoreographyControl(state.presentation),
    );
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
    const presentationFrame = state.presentation;
    battlefieldRenderer.render(presentationFrame, {
      offline: state.offline,
      showRanges: state.frame?.show_ranges === true,
    });
    if (liveScriptedInspectionOnly() && !recordingScientificControlsFenced()) {
      applyScriptedBattlefieldBoundaryCopy();
    }
    try {
      choreographer.reproject(
        presentationFrame,
        battlefieldRenderer.choreographySurface(),
        installedChoreographyControl(presentationFrame),
      );
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
 */
async function sendReplayCommand(command) {
  if (!isReplayMode() || !state.frame || !state.presentation) {
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
        command.command_type === "set_view" || command.command_type === "set_pov_actor"
          ? "replay_audience_change"
          : "replay_command",
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
    const payload = commandOutcome.current?.payload;
    if (commandResponseSchedulesShutdown(request.command, payload)) {
      state.shuttingDown = true;
      setNotice(
        "Exit was accepted, but no coherent successor presentation was installed while the local replay viewer shuts down.",
        "info",
      );
    } else {
      const status = error instanceof DebuggerApiError ? error.status : 0;
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
    render();
  }
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
  replayPlayback.pause("user_command");
  try {
    const payload = await sendReplayCommand(command);
    const frame = state.frame;
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
    if (command.role === "control") {
      return dispatchReplayCommand({
        command_type: "set_pov_actor",
        global_slot: command.global_slot,
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
  if (!state.frame || !state.presentation) {
    setNotice(
      "No coherent transport/presentation pair is available. Reconnect before sending commands.",
      "error",
    );
    renderConnection();
    return;
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

  state.busy = true;
  state.offline = false;
  setNotice("Waiting for the authoritative Python response…", "info");
  let reviewHandoff = false;
  const previousAuthority = state.authority;
  const request = commandRequest(command);
  /** @type {{current: {kind: "success" | "stale", payload: any} | null}} */
  const commandOutcome = { current: null };
  try {
    const installPromise = presentationInstallation.installFromCommand({
      reason:
        command.command_type === "set_view" ? "live_audience_change" : "live_command",
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
    if (!reviewHandoff || state.shuttingDown || state.resyncRequired) {
      render();
    }
  }
  if (reviewHandoff && !state.shuttingDown && !state.resyncRequired) {
    reloadForProductHandoff();
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
  state.busy = true;
  if (isReplayMode()) {
    replayPlayback.pause("reconnect");
  }
  setNotice("Fetching the current transport and authorized presentation…", "info");
  const previousFrame = state.frame;
  const previousAuthority = state.authority;
  let focusReplayTimeline = false;
  try {
    const installPromise = presentationInstallation.installFromGet({
      reason: previousFrame === null ? "initial_authority" : "reconnect_authority",
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
  isInteractive: () =>
    !isReplayMode() &&
    isAuthorizedPresentationFrame(state.presentation) &&
    !liveScriptedInspectionOnly() &&
    !state.busy &&
    !state.resyncRequired &&
    !state.offline &&
    !recordingScientificControlsFenced(),
  onReleaseFocus: () => {
    const firstCommand = /** @type {HTMLButtonElement | null} */ (
      elements.commandDeck?.querySelector("button:not([disabled])") ?? null
    );
    const focusTarget = firstCommand ?? elements.helpButton;
    focusTarget.focus({ preventScroll: true });
  },
});

elements.battlefield.addEventListener("click", (/** @type {MouseEvent} */ event) => {
  if (event.button === 0 && activateReplayAgent(event.target)) {
    event.preventDefault();
  }
});

elements.battlefield.addEventListener(
  "keydown",
  (/** @type {KeyboardEvent} */ event) => {
    if (
      (event.key === "Enter" || event.key === " ") &&
      activateReplayAgent(event.target)
    ) {
      event.preventDefault();
      event.stopPropagation();
    }
  },
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

elements.commandTargetSelect.addEventListener("change", () => {
  const command = targetSelectionCommand(elements.commandTargetSelect.value, {
    actorPov: authorizedPresentationAudience(state.presentation) === "agent_pov",
  });
  if (!command) {
    return;
  }
  dispatchCommand(command);
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
  if (isReplayMode()) {
    void dispatchReplayCommand({ command_type: "exit" });
  } else {
    void dispatchCommand({ command_type: "exit" });
  }
});

elements.reconnectButton.addEventListener("click", () => {
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

elements.replayRangesButton.addEventListener("click", () => {
  if (
    !isReplayMode() ||
    authorizedPresentationAudience(state.presentation) !== "researcher"
  ) {
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
  if (
    !isReplayMode() ||
    authorizedPresentationAudience(state.presentation) !== "researcher"
  ) {
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

for (const panel of elements.disclosurePanels) {
  panel.addEventListener("toggle", () => {
    if (panel.open) {
      return;
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
