/**
 * @file Start the Combat Debugger or Replay Viewer page and connect its controls.
 * The launcher supplies product identity and a capability token. This module binds
 * the required HTML elements, installs page-lifetime listeners, and loads a matched
 * transport frame and authorized presentation. It paints only installed authority,
 * serializes commands, and manages local focus, filters, replay playback, and
 * downloads. Open the launcher URL; this module is not a standalone Node program.
 */

import {
  acquireCapabilityToken,
  acquireClientId,
  DebuggerApiError,
  extractFrame,
  extractJoinedFrame,
  extractNotice,
  getCurrentFrameAndPresentation,
  getCurrentPresentation,
  getReplayEpisodeDetails,
  getReplayMetricCatalog,
  getReplayMetrics,
  getReplayTimeline,
  postCommand,
  postReplayCommand,
} from "./api.js";
import {
  authorizedOracleCommandSlotForPresentationKey,
  authorizedOracleCommandSlotForPublicAgentId,
  authorizedPresentationAgentDisplayId,
  authorizedPresentationAudience,
  authorizedPresentationIdentityRows,
  authorizedPresentationInspectionState,
  authorizedPresentationLatestTransitionId,
  authorizedPresentationPreferenceKey,
  authorizedPresentationResearcherInspectionState,
  authorizedPresentationResearcherSceneView,
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
import { renderMatchSummary } from "./match-summary.js";
import {
  buildMetricSearchIndex,
  renderMetricDefinition,
  renderMetricNavigation,
  renderMetricRows,
  renderMetricSearchResults,
  searchMeasurements,
} from "./metrics-panel.js";
import {
  authorizedInspectorView,
  DebuggerPanels,
  disclosurePanelInitiallyOpen,
} from "./panels.js";
import {
  pendingPresentationSurfaceView,
  resolveInstalledPresentationAuthorityV1,
} from "./presentation-authority-view.js";
import { PresentationInstallCoordinator } from "./presentation-install.js";
import {
  bindReplayTimelineControls,
  REPLAY_TRANSPORT_STATES,
  ReplayPlaybackController,
  renderReplayTimelineControls,
  replayCommandRequest,
  replayTimelineSimulatorStep,
  validateReplayCommandOutcome,
} from "./replay-controls.js";
import { captureReplayBattlefieldPngV1 } from "./replay-export.js";
import { isReplayAgentRecipientRotation } from "./replay-recipient-rotation.js";
import { BattlefieldRenderer } from "./scene.js";
import {
  requiresSharedObs as isReactiveController,
  isSystemController,
  isTeamController,
} from "./system-controls.js";
import {
  createSemanticDescriptor,
  createTooltipController,
  registerTooltipOwner,
} from "./tooltip.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualFilterEnabled,
  reduceVisualFilterState,
  setVisualFilterEnabled,
  VISUAL_FILTER_REGISTRY,
} from "./visual-filters.js";

/**
 * Find a required page element by its HTML ID. Return the element, or throw
 * Error when the page template is missing it; startup must not continue with
 * a partly connected interface.
 *
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
  matchScoreboard: requiredElement("match-scoreboard"),
  matchTask: requiredElement("match-task"),
  matchTeamA: requiredElement("match-team-a"),
  matchTeamB: requiredElement("match-team-b"),
  taskSelect: requiredElement("devclient-task-select"),
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
  recordingMetricsProgress: requiredElement("recording-metrics-progress"),
  recordingMetricsHelp: requiredElement("recording-metrics-help"),
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
  replayEpisodeDetailsButton: requiredElement("replay-episode-details-button"),
  replayMetricsPreparation: requiredElement("replay-metrics-preparation"),
  replayMetricsPreparationText: requiredElement("replay-metrics-preparation-text"),
  metricPanel: requiredElement("evaluation-metrics"),
  metricScope: requiredElement("metric-scope"),
  metricSelection: requiredElement("metric-selection"),
  metricView: requiredElement("metric-view"),
  metricViewField: requiredElement("metric-view-field"),
  metricSearchArea: requiredElement("metric-search-area"),
  metricSearchContent: requiredElement("metric-search-content"),
  metricSearch: requiredElement("metric-search"),
  metricSearchStatus: requiredElement("metric-search-status"),
  metricSearchResults: requiredElement("metric-search-results"),
  metricSearchMore: requiredElement("metric-search-more"),
  metricSearchDefinition: requiredElement("metric-search-definition"),
  metricStatus: requiredElement("metric-status"),
  metricProgress: requiredElement("metric-progress"),
  metricRows: requiredElement("metric-rows"),
  metricDescription: requiredElement("metric-description"),
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
  defaultVisualFiltersButton: requiredElement("default-visual-filters-button"),
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

/**
 * Short help for every Visual Filters option, keyed by filter ID.
 * title and text fill the option's hover and focus tooltip (through CONTROL_HELP).
 * text also fills a hidden element with ID id, which the checkbox lists after
 * visual-filters-help in its aria-describedby, so screen readers read both.
 * Each ID must be unique in the page.
 */
const VISUAL_FILTER_OPTION_HELP = Object.freeze({
  aura_fields: Object.freeze({
    id: "visual-filter-aura-fields-help",
    title: "Aura Fields",
    text: "Show the areas around Mages and Warriors where their auras help teammates.",
  }),
  aura_modifier_badges: Object.freeze({
    id: "visual-filter-aura-modifier-badges-help",
    title: "Aura Modifier Badges",
    text: "Show badges for the extra damage or damage protection an agent gets from auras.",
  }),
  duration_status_badges: Object.freeze({
    id: "visual-filter-duration-status-badges-help",
    title: "Duration Status Badges",
    text:
      "Show badges for active effects, such as slows and stuns. Show their remaining " +
      "ticks when known, including the In Combat countdown.",
  }),
  spawn_shield: Object.freeze({
    id: "visual-filter-spawn-shield-help",
    title: "Spawn Shield",
    text: "Show the shield around an agent while it is protected after spawning.",
  }),
  target_selection_visuals: Object.freeze({
    id: "visual-filter-target-selection-visuals-help",
    title: "Target Selection Visuals",
    text: "Mark the selected target and show whether the chosen ability can target it.",
  }),
  basic_ability_effects: Object.freeze({
    id: "visual-filter-basic-ability-effects-help",
    title: "Basic Ability Effects",
    text: "Show effects when agents use Basic abilities, including attack and healing effects.",
  }),
  ultimate_ability_effects: Object.freeze({
    id: "visual-filter-ultimate-ability-effects-help",
    title: "Ultimate Ability Effects",
    text: "Show effects when agents use Ultimate abilities, including their paths and impacts.",
  }),
  regeneration_effects: Object.freeze({
    id: "visual-filter-regeneration-effects-help",
    title: "Regeneration Effects",
    text: "Show the recovery effect when an agent regains health outside combat.",
  }),
  cooldown_effects: Object.freeze({
    id: "visual-filter-cooldown-effects-help",
    title: "Cooldown Effects",
    text:
      "Show how many ticks remain before an agent can use its Ultimate again. " +
      "Show a signal when it is ready.",
  }),
  status_application: Object.freeze({
    id: "visual-filter-status-application-help",
    title: "Status Application",
    text: "Show a short effect when an agent gets a slow, stun or other status, or that status is applied again.",
  }),
  natural_status_expiry: Object.freeze({
    id: "visual-filter-natural-status-expiry-help",
    title: "Natural Status Expiry",
    text: "Show a short effect when a timed status ends on its own, including when an agent leaves combat.",
  }),
  freezing_trap_break: Object.freeze({
    id: "visual-filter-freezing-trap-break-help",
    title: "Freezing Trap Break",
    text: "Show a shattering effect when damage breaks a Hunter's Freezing Trap.",
  }),
  status_clear_on_death: Object.freeze({
    id: "visual-filter-status-clear-on-death-help",
    title: "Status Clear on Death",
    text: "Show a short effect when an agent's death removes its active statuses.",
  }),
  death_effects: Object.freeze({
    id: "visual-filter-death-effects-help",
    title: "Death Effects",
    text: "Show a ring around an agent when it dies.",
  }),
  respawn_wave: Object.freeze({
    id: "visual-filter-respawn-wave-help",
    title: "Respawn Wave",
    text: "Show a team message when it is time for its dead agents to return.",
  }),
  resurrection_effects: Object.freeze({
    id: "visual-filter-resurrection-effects-help",
    title: "Resurrection Effects",
    text: "Show a ring around an agent when it returns at its spawn pad.",
  }),
  spawn_shield_expiry: Object.freeze({
    id: "visual-filter-spawn-shield-expiry-help",
    title: "Spawn-Shield Expiry",
    text: "Show a short effect when an agent's spawn protection ends.",
  }),
  scrolling_battle_text: Object.freeze({
    id: "visual-filter-scrolling-battle-text-help",
    title: "Scrolling Battle Text",
    text: "Show health-change numbers beside agents, including health regained outside combat. Combat numbers show the final health change after damage and healing are combined.",
  }),
  death_announcer: Object.freeze({
    id: "visual-filter-death-announcer-help",
    title: "Death Announcer",
    text:
      "Show which team got a kill and which agents died. Each victim's info box " +
      "lists who helped with the kill, when those details were recorded.",
  }),
  red_zone_floors: Object.freeze({
    id: "visual-filter-red-zone-floors-help",
    title: "Red Zone Floors",
    text:
      "Tint each team's Red Zone floor deep red. When an agent dies inside its own " +
      "team's Red Zone, the enemy team gets 2 points.",
  }),
});

/**
 * Build the page checkboxes from the shared visual-filter registry and defaults.
 * Replace the container so browser form restoration cannot override a fresh
 * page load. Each option label carries data-visual-filter-option with its filter
 * ID; an option listed in VISUAL_FILTER_OPTION_HELP also gets its hidden help
 * element (after the label, so it never joins the checkbox's name) and that
 * element's ID in aria-describedby. Throw TypeError for duplicate IDs or
 * disagreement between registry and default values; this changes only the DOM.
 */
function installVisualFilterControls() {
  const registeredIds = VISUAL_FILTER_REGISTRY.map(({ id }) => id);
  if (new Set(registeredIds).size !== registeredIds.length) {
    throw new TypeError("Visual Filters requires unique registry entries.");
  }
  const fragment = document.createDocumentFragment();
  for (const { id, label, defaultEnabled } of VISUAL_FILTER_REGISTRY) {
    const enabled = isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, id);
    if (enabled !== defaultEnabled) {
      throw new TypeError(`Visual filter ${id} disagrees with the default state.`);
    }
    const option = document.createElement("label");
    option.className = "visual-filters__option";
    option.dataset.visualFilterOption = id;
    const help = Object.hasOwn(VISUAL_FILTER_OPTION_HELP, id)
      ? VISUAL_FILTER_OPTION_HELP[
          /** @type {keyof typeof VISUAL_FILTER_OPTION_HELP} */ (id)
        ]
      : null;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `visual-filter-${id.replaceAll("_", "-")}`;
    input.value = id;
    input.dataset.visualFilterId = id;
    input.setAttribute("autocomplete", "off");
    input.setAttribute(
      "aria-describedby",
      help ? `visual-filters-help ${help.id}` : "visual-filters-help",
    );
    input.defaultChecked = enabled;
    input.checked = enabled;
    const text = document.createElement("span");
    text.textContent = label;
    option.append(input, text);
    fragment.append(option);
    if (help) {
      const description = document.createElement("span");
      description.id = help.id;
      description.className = "sr-only";
      description.textContent = help.text;
      fragment.append(description);
    }
  }
  elements.visualFilterOptions.replaceChildren(fragment);
}

/**
 * Page-lifetime presentation state. It is replaced atomically, never stored,
 * and deliberately survives transport, authority, audience, and episode
 * changes until the document itself reloads.
 */
let visualFilterState = DEFAULT_VISUAL_FILTER_STATE;
let agentLocalRangesVisible = false;
let agentLocalRangesInitialized = false;
let confirmedResearcherRangesVisible = false;
let visualFilterProductDefaultsApplied = false;
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
 * One live request may retain one fresh draft intent and one following Enter
 * while a coherent response is installed. This is page-local latency state,
 * not simulator, action, or replay state.
 *
 * @type {{
 *   allowsDeferredSubmit: boolean,
 *   deferredDraft: Readonly<Record<string, unknown>> | null,
 *   deferredSubmit: Readonly<Record<string, unknown>> | null,
 *   releaseFocusAfterSettlement: boolean,
 *   recordingPreparation: "metrics" | "possible" | null,
 * } | null}
 */
let activeLiveCommandTransaction = null;

/**
 * One page-local read-only artifact transaction. It is an async ownership
 * fence, not replay transport state, and is invalidated synchronously whenever
 * installed presentation authority is cleared.
 *
 * @type {Readonly<{
 *   kind: "export_png" | "download_metrics" | "download_episode_details",
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
 * active record rather than a cache keyed by authority: unrelated A -> B
 * crossings replace A, so a later B -> A boundary receives fresh defaults.
 * One certified Agent recipient rotation retains only the current
 * user-interface preferences; local inspection is still rebuilt from the new
 * authority.
 *
 * @type {{
 *   authorityKey: Readonly<{tuple: readonly unknown[], serialized: string}>,
 *   disclosures: Record<string, {open: boolean, scrollTop: number}>,
 *   agentDetailsAutoOpenAllowed: boolean,
 *   primaryFocus: null | {
 *     surface: "roster" | "battlefield",
 *     presentationKey: string,
 *     publicAgentId: string,
 *   },
 *   localInspection: {owned: false} | {owned: true, presentationKey: string | null},
 * } | null}
 */
let activePresentationPreference = null;

/**
 * One Replay Agent activation is staged until its certified successor is
 * installed. This keeps the previous Agent Details content stable while the
 * retained battlefield is busy and carries no geometry or opaque authority
 * data into the successor.
 *
 * @type {null | Readonly<{
 *   sourcePreferenceGeneration: number,
 *   recipientPublicAgentId: string,
 *   autoOpenAgentDetails: boolean,
 * }>}
 */
let pendingReplayRecipientActivation = null;

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
  combat_debugger: "MARL-BattleGrounds DevClient",
  replay_viewer: "MARL-BattleGrounds Replay Viewer",
});
const REPLAY_SAVE_MESSAGE = "Saving replay. Longer recordings can take longer.";
const METRIC_PREPARATION_MESSAGE = "Preparing metrics. Longer replays can take longer.";
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

/**
 * Mark a frame that belongs to a different product than this page.
 * This Error subclass lets request handlers clear displayed authority and require
 * a reconnect instead of showing a live frame in the Replay Viewer or vice versa.
 */
class ProductIdentityMismatchError extends Error {}
/**
 * Signal that accepted recording review requires a page reload.
 * This Error subclass is an internal control-flow signal caught by the live
 * command handler; it does not mean the recording command failed.
 */
class ProductReviewHandoff extends Error {}

/** @type {Readonly<{schema_version: 1, product_kind: "combat_debugger" | "replay_viewer", authoring_available: boolean}> | null} */
let productIdentity = null;
/** @type {string | null} */
let startupProductIdentityError = null;

/**
 * Validate and install the launcher product identity before accepting frames.
 * Accept schema version 1, a known product, boolean authoring permission, and an
 * optional boolean initial range choice; Replay Viewer cannot enable authoring.
 * Throw TypeError for invalid input. Update page labels and product state, and
 * apply initial range/filter defaults only on their first installation.
 *
 * @param {unknown} rawIdentity
 * @returns {Readonly<{schema_version: 1, product_kind: "combat_debugger" | "replay_viewer", authoring_available: boolean}>}
 */
function applyBootstrap(rawIdentity) {
  if (!isRecord(rawIdentity)) {
    throw new TypeError("Product bootstrap must be an object.");
  }
  const hasInitialRanges = Object.hasOwn(rawIdentity, "initial_show_ranges");
  if (hasInitialRanges && typeof rawIdentity.initial_show_ranges !== "boolean") {
    throw new TypeError("Initial range visibility must be a Boolean.");
  }
  const keys = Object.keys(rawIdentity)
    .filter((key) => key !== "initial_show_ranges")
    .sort();
  if (
    keys.length !== 3 ||
    keys[0] !== "authoring_available" ||
    keys[1] !== "product_kind" ||
    keys[2] !== "schema_version"
  ) {
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
  if (typeof rawIdentity.authoring_available !== "boolean") {
    throw new TypeError("Product bootstrap authoring capability is invalid.");
  }
  if (rawIdentity.product_kind === "replay_viewer" && rawIdentity.authoring_available) {
    throw new TypeError("Replay Viewer cannot receive authoring authority.");
  }
  if (!agentLocalRangesInitialized && hasInitialRanges) {
    agentLocalRangesVisible = rawIdentity.initial_show_ranges;
    agentLocalRangesInitialized = true;
  }
  productIdentity = Object.freeze({
    schema_version: 1,
    product_kind: rawIdentity.product_kind,
    authoring_available: rawIdentity.authoring_available,
  });
  if (!visualFilterProductDefaultsApplied) {
    visualFilterState =
      productIdentity.product_kind === "replay_viewer"
        ? setVisualFilterEnabled(
            DEFAULT_VISUAL_FILTER_STATE,
            "target_selection_visuals",
            false,
          )
        : DEFAULT_VISUAL_FILTER_STATE;
    visualFilterProductDefaultsApplied = true;
  }
  const title = PRODUCT_TITLES[productIdentity.product_kind];
  document.title = title;
  elements.appTitle.textContent = title;
  elements.helpHeading.textContent =
    productIdentity.product_kind === "replay_viewer"
      ? "Replay Viewer help"
      : "DevClient help";
  document.documentElement.dataset.productKind = productIdentity.product_kind;
  return productIdentity;
}

/**
 * Check that a frame agrees with the product installed from the launcher.
 * Throw ProductIdentityMismatchError when identity is absent or live/replay modes
 * differ. A frame cannot choose which product this page is allowed to become.
 *
 * @param {Record<string, any>} frame
 */
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
  const startupIdentity = applyBootstrap(
    Reflect.get(globalThis, "__MARL_DEBUGGER_BOOTSTRAP__"),
  );
  if (
    startupIdentity.product_kind === "combat_debugger" &&
    startupIdentity.authoring_available
  ) {
    void import("./dev-client.js").then(() => {
      publishInstalledCombatConfiguration(state.frame);
    });
  }
} catch (error) {
  startupProductIdentityError =
    error instanceof Error ? error.message : "Product bootstrap is invalid.";
}

const CONTROL_HELP = Object.freeze([
  [
    "[data-devclient-area]",
    "DevClient Area",
    "Switch between the Combat Debugger, reusable Maps, and task-controlled Scenarios.",
  ],
  [
    "#devclient-scenario-select",
    "Saved Debug Asset",
    "Choose an execution-valid scenario or a map to inspect through the explicit default 5v5 TDM preview.",
  ],
  [
    "#devclient-scenario-load",
    "Load Debug Asset",
    "Reopen, compile, and revalidate the selected scenario or map preview before replacing the current Debug session.",
  ],
  [
    "#devclient-task-select",
    "Task",
    "TDM is the available task for authored matches. Load a saved scenario or map to use its captured configuration.",
  ],
  [
    "#metric-scope",
    "Evaluation Metric Scope",
    "Up to Current Tick includes only captured transitions through this tick. Entire Episode includes the entire captured replay.",
  ],
  [
    "#metric-selection",
    "Evaluation Topic",
    "Choose a topic, then its Totals or By Recipient view when available. Metrics are researcher information and do not enter agent observations.",
  ],
  [
    "#replay-episode-details-button",
    "Episode Details",
    "Download the recorded episode identity, policies, configuration, completion and runtime details as JSON.",
  ],
  [
    "#devclient-team-a-controller",
    "Team A Controller",
    "Choose manual control, the scripted Team Deathmatch policy, or Random, then restart from the same snapshot.",
  ],
  [
    "#devclient-team-b-controller",
    "Team B Controller",
    "Choose manual control, the scripted Team Deathmatch policy, or Random, then restart from the same snapshot.",
  ],
  [
    "#devclient-information-mode",
    "Information Mode",
    "Choose SharedObs or NoSharedObs, then restart from the same scenario snapshot and seed.",
  ],
  [
    "#authoring-saved-draft-select",
    "Saved Draft",
    "Choose an exact saved revision for the active Map or Scenario Author tab.",
  ],
  [
    "#authoring-new-scenario-mode",
    "Scenario Starting Point",
    "Start blank, copy a saved map, or duplicate a saved scenario.",
  ],
  [
    "#authoring-new-scenario-source",
    "Scenario Source Asset",
    "Choose the exact saved revision to copy into a new independent scenario draft.",
  ],
  ["#authoring-new", "New Draft", "Create a new local map or scenario draft."],
  ["#authoring-open", "Open Draft", "Open an exact saved draft revision."],
  [
    "#authoring-delete-saved",
    "Delete Saved Asset",
    "After confirmation, permanently delete the selected saved map or scenario and all of its revisions.",
  ],
  [
    "#authoring-save",
    "Save Draft",
    "Atomically save the complete draft at its current revision.",
  ],
  [
    "#authoring-save-as",
    "Save Draft As",
    "Save the complete draft under a new safe asset identity.",
  ],
  [
    "#authoring-validate",
    "Validate Draft",
    "Compile the current draft with host-authoritative rules and show linked problems.",
  ],
  [
    "#authoring-open-debug",
    "Open in Debug",
    "Compile and revalidate an immutable scenario snapshot or an explicit default 5v5 TDM map preview, then atomically replace the Combat Debugger session.",
  ],
  [
    "[data-authoring-add]",
    "Add Obstacle",
    "Append a wall or pillar to the map's ordered active obstacle prefix.",
  ],
  [
    "#authoring-reset",
    "Reset Draft",
    "Restore the latest new, opened, or successfully saved draft baseline. The reset is undoable.",
  ],
  [
    "#authoring-recenter",
    "Recenter Map",
    "Fit the complete authored map in the canvas without changing draft content.",
  ],
  ["#authoring-undo", "Undo", "Undo the most recent browser-local edit."],
  ["#authoring-redo", "Redo", "Redo the most recently undone browser-local edit."],
  [
    "#authoring-duplicate",
    "Duplicate Obstacle",
    "Append a copy of the selected wall or pillar.",
  ],
  [
    "#authoring-delete",
    "Delete Obstacle",
    "Delete the selected wall or pillar and compact the ordered obstacle prefix.",
  ],
  [
    "#authoring-order-up",
    "Move Obstacle Up",
    "Move the selected obstacle earlier in fixed-slot order.",
  ],
  [
    "#authoring-order-down",
    "Move Obstacle Down",
    "Move the selected obstacle later in fixed-slot order.",
  ],
  [
    "#replay-timeline",
    "Replay Timeline",
    "Use the read-only transport and presentation controls to inspect recorded frames.",
    "composite",
  ],
  [
    "#view-select",
    "Audience View",
    "Switch between Oracle View and recipient-authorized views.",
  ],
  [
    "#reconnect-button",
    "Reconnect",
    "Fetch and atomically install the latest authoritative frame.",
  ],
  ["#help-button", "Help", "Open the keyboard, recording, and replay controls guide."],
  ["#help-close-button", "Close Help", "Close the product help dialog."],
  ["#exit-button", "Exit", "Ask the local Python service to close safely."],
  [
    "#recording-finish-button",
    "Finish and Review",
    "Finalize the captured prefix, save it, and enter read-only review.",
  ],
  [
    "#recording-review-button",
    "Review Replay",
    "Enter read-only review for the saved replay.",
  ],
  [
    "#recording-retry-button",
    "Retry Save",
    "Retry publishing the same immutable replay bytes.",
  ],
  [
    "#recording-save-as-input",
    "Save As Basename",
    "Enter a basename only; paths and overwrites are rejected.",
  ],
  [
    "#recording-save-as-button",
    "Save As",
    "Publish the same immutable replay bytes under the entered basename.",
  ],
  [
    "#recording-discard-cancel-button",
    "Keep Recording",
    "Cancel replacement and preserve the captured prefix.",
  ],
  [
    "#recording-discard-confirm-button",
    "Discard and Replace",
    "Confirm permanent loss of the unpublished prefix and start its named replacement.",
  ],
  ["#replay-first-button", "Start Replay Tick", "Seek to settled replay tick zero."],
  [
    "#replay-back-ten-button",
    "Back Ten Ticks",
    "Seek ten ticks backward with one clamped request.",
  ],
  [
    "#replay-previous-button",
    "Previous Replay Frame",
    "Seek one captured frame backward.",
  ],
  [
    "#replay-play-pause-button",
    "Replay Playback",
    "Start or pause serialized read-only autoplay.",
  ],
  [
    "#replay-next-button",
    "Next Replay Frame",
    "Advance exactly one captured replay frame.",
  ],
  [
    "#replay-forward-ten-button",
    "Forward Ten Ticks",
    "Seek ten ticks forward with one clamped request.",
  ],
  ["#replay-last-button", "End Replay Tick", "Seek to the end of the captured prefix."],
  [
    "#replay-frame-slider",
    "Replay Tick",
    "Preview locally without a request, then commit one exact captured-tick seek.",
  ],
  [
    "#replay-playback-rate",
    "Replay Playback Speed",
    "Scale the complete replay presentation clock without changing artifact authority.",
  ],
  [
    "#command-target-select",
    "Selected Target",
    "Stage an authorized target for the controlled actor.",
  ],
  [
    "#submit-turn-button",
    "Apply Authorized Action",
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
    "Turn on all local visual filters and the active Ranges overlay.",
  ],
  [
    "#disable-all-visual-filters-button",
    "Disable All",
    "Turn off all local visual filters and the active Ranges overlay.",
  ],
  [
    "#default-visual-filters-button",
    "Default Configuration",
    "Restore the eleven default effects, including Cooldown Effects, Death Announcer and Red Zone Floors, and turn off Ranges.",
  ],
  ...Object.entries(VISUAL_FILTER_OPTION_HELP).map(
    ([id, help]) =>
      /** @type {[string, string, string]} */ ([
        `[data-visual-filter-option='${id}']`,
        help.title,
        help.text,
      ]),
  ),
  [
    ".diagnostics > summary",
    "Technical Frame",
    "Inspect authorized wire and diagnostic details.",
  ],
  [
    "[data-key='Escape']",
    "Clear Target",
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
  /**
   * Reflect a new choreography snapshot in page state and command controls.
   * The presentation argument is the choreographer's current snapshot. This
   * synchronous callback updates display/availability without issuing a command.
   */
  onStateChange: (presentation) => {
    exposePresentationState(presentation);
    renderCommandAvailability();
  },
});

const replayTimelineElements = {
  root: elements.replayTimeline,
  keyboardTarget: document,
  /**
   * Return whether a matched frame and presentation are currently installed.
   * This lets keyboard handling distinguish an installed view from a loading gap;
   * it does not decide whether another command may start.
   */
  keyboardAuthorityInstalled: () => installedPresentationAuthority() !== null,
  /**
   * Return whether installed presentation authority exists and the page is idle.
   * Replay keyboard commands remain fenced while a page request is busy.
   */
  keyboardEnabled: () => installedPresentationAuthority() !== null && !state.busy,
  /**
   * Clear the current replay inspection selection through its shared handler.
   * The handler may send the supported selection command; playback position is
   * not changed by this callback itself.
   */
  clearSelection: () => clearReplaySelection(),
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
  /**
   * Return the recorded simulator tick at frameIndex, or null if unavailable.
   * The numeric frame index is a captured-frame position, not a simulator tick.
   * Require installed authority before consulting the recorded timeline.
   */
  tickForFrameIndex: (/** @type {number} */ frameIndex) =>
    installedPresentationAuthority() === null
      ? null
      : replayTimelineSimulatorStep(state.timeline, frameIndex),
  /**
   * Return the incoming transition ID for the currently installed frame only.
   * The numeric frameIndex must match the installed cursor. Return null for another
   * frame, missing authority, or a frame with no authorized incoming transition.
   */
  incomingTransitionForFrameIndex: (/** @type {number} */ frameIndex) => {
    const installed = installedPresentationAuthority();
    if (installed === null || installed.transport.cursor?.frame_index !== frameIndex) {
      return null;
    }
    return authorizedIncomingTransitionId(installed.presentation);
  },
};

/**
 * Return the playback snapshot used to draw timeline controls.
 * While a page request is busy, return a frozen copy marked ADVANCING unless the
 * controller is OFFLINE; otherwise return the original snapshot. This does not
 * change playback state or start a request.
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
  /**
   * Return a promise that resolves without a value when choreography settles.
   * Playback uses this barrier before requesting the next recorded frame.
   */
  waitForPresentation: () => choreographer.whenSettled(),
  /**
   * Return the choreographer's current normal/reduced motion preference.
   * Read a snapshot without starting an animation or changing playback.
   */
  getMotionMode: () => choreographer.snapshot().motionMode,
  /**
   * Apply the supplied playback snapshot to timeline and artifact controls.
   * Invalidate an export when its playback generation changed, synchronize animation
   * rate, and render the page only when the installed view is idle and rendering
   * is not suppressed. This callback does not send a replay request.
   */
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
  /**
   * Show a playback failure unless an error notice is already visible.
   * Use error.message for an Error, otherwise the generic navigation message.
   * Update connection presentation without replacing an existing error notice.
   */
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
});

/** @type {string | null} */
let activatedAgentPublicId = null;

let lastBattlefieldSizeKey = "";
/** @type {number | null} */
let pendingResizeFrame = null;
/** @type {Readonly<Record<string, unknown>> | null} */
let pendingRecordingReplacement = null;
let productHandoffOutcomeUnknown = false;
let productHandoffReloadRequested = false;

/**
 * Remove the element attributes that expose a presentation-owned tooltip.
 * Also remove its accessible description. Old weak-map entries may still exist,
 * but the missing owner attribute prevents them from being selected.
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
 * Clear scientific labels and disable controls while no presentation is installed.
 * Reset replay, recording, selection, and tooltip surfaces, and close any pending
 * recording-discard dialog. This paints an unavailable state without guessing
 * facts from an unjoined transport frame.
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
 * Remove the installed scientific presentation for the supplied reason.
 * Save permitted display preferences, hold the workspace height during replacement,
 * clear scenes and panels, and invalidate pending display/download work. Raw
 * transport may remain for diagnostics and request accounting; it cannot authorize
 * what the page displays.
 *
 * @param {string} reason
 */
function clearPresentationAuthority(reason) {
  renderInstalledMatch(null);
  elements.metricProgress.hidden = true;
  elements.metricRows.setAttribute("aria-busy", "false");
  elements.recordingMetricsProgress.hidden = true;
  elements.recordingMetricsHelp.hidden = true;
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
 * Prepare the page for an authority request with the supplied reason and policy.
 * With retain_last_authorized and an installed pair, keep its display preferences
 * and mark the retained surface pending. Otherwise clear presentation authority.
 * This does not send the request or install its result.
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
 * Keep the current workspace height while its scientific content is replaced.
 * Save the previous inline minimum height once, then use the measured positive
 * height in CSS pixels. Missing or nonpositive measurements leave the style alone.
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

/**
 * Restore the workspace minimum-height style saved before authority replacement.
 * Do nothing when no height is held; clear the saved value after restoration.
 */
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
 * Install a validated transport/presentation pair and its replay timeline.
 * Require the trusted join marker, matching product, and a usable preference key;
 * throw on an invalid pair before treating it as authority. Reconcile local
 * preferences, publish confirmed combat settings, and request a content repaint.
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
  installActivePresentationPreference(joined.presentation, preferenceKey, {
    previousAuthority: state.authority,
    nextAuthority: joined,
  });
  if (
    authorizedPresentationAudience(joined.presentation) === "researcher" &&
    typeof joined.transport.show_ranges === "boolean"
  ) {
    confirmedResearcherRangesVisible = joined.transport.show_ranges;
  }
  if (
    !agentLocalRangesInitialized &&
    typeof joined.transport.show_ranges === "boolean"
  ) {
    agentLocalRangesVisible = joined.transport.show_ranges;
    agentLocalRangesInitialized = true;
  }
  state.authority = joined;
  state.frame = joined.transport;
  state.presentation = joined.presentation;
  state.timeline = isRecord(joined.timeline) ? joined.timeline : null;
  publishInstalledCombatConfiguration(joined.transport);
  presentationPreferenceNeedsContentRender = true;
  document.documentElement.dataset.presentationAuthority = "installed";
}

/**
 * Prepare a trusted joined pair for installation without installing it.
 * Check product identity and, when a previous replay authority is supplied, replay
 * continuity. Replay pairs also fetch and join their timeline; live pairs return
 * directly. Previous authority defaults to null and continuity to stale_resync.
 * Validation and request failures reject the returned promise.
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

/**
 * Set the inspector heading and accent for the activated authorized agent.
 * Reset them when no agent is activated. Read identity through the authorized
 * inspector view rather than a stale DOM row.
 */
function applyAuthorizedInspectorChrome() {
  if (activatedAgentPublicId === null) {
    elements.selectionHeading.textContent = AUTHORIZED_INSPECTOR_TITLE;
    elements.agentDetails.removeAttribute("data-tone");
    elements.agentDetails.removeAttribute("data-accent");
    return;
  }
  const inspector = authorizedInspectorView(
    state.presentation,
    installedLocalInspectedPresentationKey(state.presentation),
    activatedAgentPublicId,
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

/**
 * Reload the page once after a live-to-replay product handoff.
 * A page flag suppresses duplicate reload attempts from later callbacks.
 */
function reloadForProductHandoff() {
  if (productHandoffReloadRequested) {
    return;
  }
  productHandoffReloadRequested = true;
  window.location.reload();
}

/**
 * Attach the static control-help registry to matching page elements.
 * Entries without a kind use control; missing matching elements need no handler.
 */
function registerControlHelp() {
  for (const [selector, title, summary, kind = "control"] of CONTROL_HELP) {
    for (const control of document.querySelectorAll(selector)) {
      registerControlHelpOwner(control, selector, title, summary, kind);
    }
  }
}

/**
 * Attach a non-inspectable help tooltip to one control.
 * Use the supplied selector as its stable ID and the supplied title and summary
 * as prose; kind defaults to control. Battlefield help follows the pointer, while
 * other controls anchor the tooltip to their element.
 *
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

/**
 * Refresh utility help for the currently installed product and audience.
 * Explain range, selection, and Tab behavior using the same authority and
 * availability checks as their controls, including page-local Agent POV choices.
 */
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
      ? "Show or hide fog-authorized range overlays for the current Agent POV. This sends no command."
      : researcher
        ? "Toggle server-authored Oracle View range presentation."
        : "Range presentation is unavailable until one coherent authorized live frame is installed.",
  );
  registerControlHelpOwner(
    elements.replayRangesButton,
    "#replay-ranges-button",
    "Replay Ranges",
    agentPov
      ? "Show or hide locally authorized inspected-agent range overlays. This sends no replay command and does not switch Agent POV."
      : researcher
        ? "Toggle recorded Oracle View range presentation."
        : "Replay range presentation is unavailable until one coherent authorized frame is installed.",
  );
  registerControlHelpOwner(
    elements.replayClearReferenceButton,
    "#replay-clear-reference-button",
    "Clear Selection",
    agentPov
      ? "Clear the local inspected-agent highlight, details, and ranges. This sends no replay command and does not switch Agent POV."
      : researcher
        ? "Clear the selected Oracle View agent and its inspection highlight."
        : "Selection cannot be cleared until one coherent authorized replay frame is installed.",
  );
  for (const [selector, title, oracleSummary] of [
    [
      "[data-key='Tab']:not([data-shift])",
      "Next Actor",
      "Move Oracle View control to the next active actor.",
    ],
    [
      "[data-key='Tab'][data-shift='true']",
      "Previous Actor",
      "Move Oracle View control to the previous active actor.",
    ],
  ]) {
    for (const control of document.querySelectorAll(selector)) {
      const oracleActorCyclingAvailable =
        researcher && control instanceof HTMLButtonElement && !control.disabled;
      const agentActorCyclingAvailable =
        agentPov && control instanceof HTMLButtonElement && !control.disabled;
      registerControlHelpOwner(
        control,
        selector,
        title,
        agentActorCyclingAvailable
          ? `${oracleSummary.replace("Oracle View", "Agent POV")} Staged drafts are preserved.`
          : agentPov
            ? "Actor cycling is unavailable in the current Agent POV state. Tab and Shift+Tab retain native browser focus navigation."
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
 * Return whether a value is a non-null object that is not an array.
 * This is a shape guard only; it does not validate a protocol or certify authority.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Cancel the queued animation-frame preference restore, if present.
 * Clear its handle so a later render can schedule a fresh restore.
 */
function cancelPendingPresentationPreferenceRestore() {
  if (pendingPresentationPreferenceRestoreFrame === null) {
    return;
  }
  window.cancelAnimationFrame(pendingPresentationPreferenceRestoreFrame);
  pendingPresentationPreferenceRestoreFrame = null;
}

/**
 * Set a details panel open or closed without treating that change as user intent.
 * Record the expected final toggle state before assigning open, because browsers
 * queue and may combine toggle events. Do nothing when the state already matches.
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
 * Enable or disable the scientific details panels and their keyboard summaries.
 * When unavailable, close them unless preserveOpen is true; that option defaults
 * to false. Use inert and accessibility attributes to block interaction while
 * presentation authority is missing or a request is busy.
 *
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
 * Return the active local preference only for this exact installed presentation.
 * Return null for an untrusted presentation, an incoherent installed pair, a
 * different object, or a mismatched authority key.
 *
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
 * Create fresh local display preferences for a presentation and authority key.
 * Initialize disclosure openness and zero scroll offsets, allow automatic inspector
 * opening, and start without keyboard focus. Agent POV and scripted live scenes
 * own local inspection; select the recipient only if its key exists in this scene.
 *
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
  };
}

/**
 * Recheck retained selection and focus against a replacement authorized scene.
 * If a selected key disappeared, clear selection, close Agent Details, and
 * invalidate downloads. Rebind focus only through the new scene; do not silently
 * replace a missing selection with the recipient.
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
  if (preference.primaryFocus !== null) {
    preference.primaryFocus = rebindAuthorizedPrimaryFocus(
      presentation,
      preference.primaryFocus,
    );
  }
}

/**
 * Resolve saved keyboard focus in the new authorized roster or battlefield.
 * Match the opaque presentation key; optional public-identity fallback defaults
 * to false and is used only for an approved recipient rotation. Return the new
 * focus record or null. Roster lookup uses the researcher directory, not fogged SVG.
 *
 * @param {unknown} presentation
 * @param {Readonly<{
 *   surface: "roster" | "battlefield",
 *   presentationKey: string,
 *   publicAgentId: string,
 * }>} focus
 * @param {Readonly<{allowPublicIdentityFallback?: boolean}>} [options]
 * @returns {null | {surface: "roster" | "battlefield", presentationKey: string, publicAgentId: string}}
 */
function rebindAuthorizedPrimaryFocus(
  presentation,
  focus,
  { allowPublicIdentityFallback = false } = {},
) {
  const scene =
    focus.surface === "roster"
      ? authorizedPresentationResearcherSceneView(presentation)
      : authorizedPresentationSceneView(presentation);
  const authorizedAgent = asArray(scene?.agents).find(
    (candidate) =>
      isRecord(candidate) &&
      (candidate.presentation_key === focus.presentationKey ||
        (allowPublicIdentityFallback &&
          candidate.public_agent_id === focus.publicAgentId)),
  );
  if (!isRecord(authorizedAgent)) {
    return null;
  }
  return {
    surface: focus.surface,
    presentationKey: String(authorizedAgent.presentation_key),
    publicAgentId: String(authorizedAgent.public_agent_id),
  };
}

/**
 * Return whether two preference keys describe a recipient change in one live
 * Agent POV episode. Require matching product, session, episode, and information
 * mode, with changed recipient and view identities. Other boundaries get fresh
 * preferences instead of borrowing old opaque keys.
 *
 * @param {Readonly<{tuple: readonly unknown[]}>} previous
 * @param {Readonly<{tuple: readonly unknown[]}>} next
 */
function isLiveAgentRecipientRotation(previous, next) {
  const left = previous.tuple;
  const right = next.tuple;
  const sameLiveAgentMode =
    right[3] === left[3] &&
    right[5] === left[5] &&
    ((left[3] === "live_no_shared_obs_agent_pov" && left[5] === "no_shared_obs") ||
      (left[3] === "live_shared_obs_agent_pov" &&
        left[5] === "shared_obs_visual_union"));
  return (
    left[0] === "combat_debugger" &&
    right[0] === left[0] &&
    right[1] === left[1] &&
    right[2] === left[2] &&
    sameLiveAgentMode &&
    left[4] === "agent_pov" &&
    right[4] === left[4] &&
    left[6] === null &&
    right[6] === null &&
    typeof left[7] === "string" &&
    typeof right[7] === "string" &&
    right[7] !== left[7] &&
    right[8] !== left[8]
  );
}

/**
 * Return whether an allowed live recipient rotation selects the saved focused agent.
 * Use the focus public ID only after the surrounding authority keys establish
 * that this is the same live Agent POV episode.
 *
 * @param {Readonly<{tuple: readonly unknown[]}>} previous
 * @param {Readonly<{tuple: readonly unknown[]}>} next
 * @param {Readonly<{publicAgentId: string}>} focus
 */
function isLiveAgentRecipientFocusRotation(previous, next, focus) {
  return (
    isLiveAgentRecipientRotation(previous, next) &&
    next.tuple[7] === focus.publicAgentId
  );
}

/**
 * Install or reconcile display preferences for a new presentation authority.
 * Cancel older restores and advance the preference generation. New authority keys
 * get fresh defaults, with narrowly allowed live/replay recipient rotations retaining
 * disclosures and rebinding focus. Optional previous and next authorities default
 * to null; no preference may authorize scientific facts.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @param {Readonly<{tuple: readonly unknown[], serialized: string}>} authorityKey
 * @param {{
 *   previousAuthority?: Readonly<Record<string, any>> | null,
 *   nextAuthority?: Readonly<Record<string, any>> | null,
 * }} [options]
 */
function installActivePresentationPreference(
  presentation,
  authorityKey,
  { previousAuthority = null, nextAuthority = null } = {},
) {
  presentationPreferenceGeneration += 1;
  cancelPendingPresentationPreferenceRestore();
  if (
    activePresentationPreference === null ||
    !sameAuthorizedPresentationPreferenceKey(
      activePresentationPreference.authorityKey,
      authorityKey,
    )
  ) {
    const previousPreference = activePresentationPreference;
    const previousAuthorityKey = previousPreference?.authorityKey ?? null;
    const previousPrimaryFocus = previousPreference?.primaryFocus ?? null;
    const retainLiveAgentPreferences =
      previousPreference !== null &&
      isLiveAgentRecipientRotation(previousPreference.authorityKey, authorityKey);
    const retainReplayAgentPreferences =
      previousPreference !== null &&
      isReplayAgentRecipientRotation(previousAuthority, nextAuthority);
    activePresentationPreference = defaultPresentationPreference(
      presentation,
      authorityKey,
    );
    if (
      (retainLiveAgentPreferences || retainReplayAgentPreferences) &&
      previousPreference !== null
    ) {
      activePresentationPreference.disclosures = Object.fromEntries(
        Object.entries(previousPreference.disclosures).map(([panelId, saved]) => [
          panelId,
          { ...saved },
        ]),
      );
      activePresentationPreference.agentDetailsAutoOpenAllowed =
        previousPreference.agentDetailsAutoOpenAllowed;
    }
    if (
      retainReplayAgentPreferences &&
      pendingReplayRecipientActivation !== null &&
      pendingReplayRecipientActivation.sourcePreferenceGeneration ===
        presentationPreferenceGeneration - 1 &&
      pendingReplayRecipientActivation.recipientPublicAgentId === authorityKey.tuple[7]
    ) {
      activatedAgentPublicId = pendingReplayRecipientActivation.recipientPublicAgentId;
      if (
        pendingReplayRecipientActivation.autoOpenAgentDetails &&
        activePresentationPreference.agentDetailsAutoOpenAllowed
      ) {
        activePresentationPreference.disclosures["agent-details"].open = true;
      }
      pendingReplayRecipientActivation = null;
    }
    if (
      previousPrimaryFocus !== null &&
      previousAuthorityKey !== null &&
      isLiveAgentRecipientFocusRotation(
        previousAuthorityKey,
        authorityKey,
        previousPrimaryFocus,
      )
    ) {
      activePresentationPreference.primaryFocus = rebindAuthorizedPrimaryFocus(
        presentation,
        previousPrimaryFocus,
        { allowPublicIdentityFallback: true },
      );
    }
  } else {
    reconcilePresentationPreference(activePresentationPreference, presentation);
  }
}

/**
 * Read focused roster or battlefield agent identity through the supplied presentation.
 * Return its surface, opaque key, and public ID, or null when focus is elsewhere
 * or the identity is absent. DOM order and slot numbers do not establish identity.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @returns {null | {
 *   surface: "roster" | "battlefield",
 *   presentationKey: string,
 *   publicAgentId: string,
 * }}
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
  const authorizedAgent =
    rosterAction !== null
      ? authorizedResearcherAgentForPresentationKey(presentation, presentationKey)
      : authorizedAgentForPresentationKey(presentation, presentationKey);
  if (typeof presentationKey !== "string" || authorizedAgent === null) {
    return null;
  }
  return {
    surface: rosterAction !== null ? "roster" : "battlefield",
    presentationKey,
    publicAgentId: String(authorizedAgent.public_agent_id),
  };
}

/**
 * Save current disclosure openness, scroll positions, and authorized focus.
 * A user closing Agent Details disables later automatic opening; expected
 * programmatic toggles do not. Mutate the supplied active preference in place.
 *
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

/**
 * Save preferences before removing the current authorized content.
 * If focus belongs to that content, move it to Help and retain a restore anchor
 * only for a recognized primary agent action. Do nothing without matching authority.
 */
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

/**
 * Invalidate older preference restores and disable scientific disclosures.
 * Advance the generation and clear the content-render flag so stale callbacks
 * cannot reopen panels while authority is pending.
 */
function enterPendingPresentationPreferenceState() {
  presentationPreferenceGeneration += 1;
  presentationPreferenceNeedsContentRender = false;
  cancelPendingPresentationPreferenceRestore();
  setScientificDisclosureAvailability(false);
}

/**
 * Save current display choices before a normal content repaint.
 * Skip capture while new content or an earlier restore is pending; those states
 * would overwrite saved user choices with temporary DOM state.
 */
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
 * Save focus and disclosure choices before a busy repaint of the same authority.
 * Move a recognized primary focus to Help temporarily, and mark content for a
 * restore. Later restoration keeps focus only if the user has not moved elsewhere.
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
 * Queue scroll and primary-focus restoration for the next animation frame.
 * Cancel the older callback and require the same preference generation and
 * authority key. Restore focus only to a current enabled authorized element while
 * the saved focus anchor still owns focus; release the held workspace height.
 *
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
      const reboundFocus = rebindAuthorizedPrimaryFocus(state.presentation, focus);
      const active = document.activeElement;
      const focusWasNotMovedByUser =
        active === anchor || (!anchor.isConnected && active === document.body);
      if (!focusWasNotMovedByUser || reboundFocus === null) {
        return;
      }
      installed.primaryFocus = reboundFocus;
      const root =
        reboundFocus.surface === "roster" ? elements.roster : elements.battlefield;
      const selector =
        reboundFocus.surface === "roster"
          ? `.roster-primary-action[data-presentation-key="${CSS.escape(reboundFocus.presentationKey)}"]`
          : `.agent[data-presentation-key="${CSS.escape(reboundFocus.presentationKey)}"]`;
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

/**
 * Restore disclosure choices after painting a coherent non-busy presentation.
 * Disable panels when authority is missing; retain their openness while busy.
 * Otherwise reopen saved panels and schedule guarded scroll/focus restoration.
 */
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

/**
 * Open Agent Details only when the active preference permits automatic opening.
 * Return true after updating the saved and DOM states, or false when authority
 * is absent or the user previously disabled automatic opening by closing it.
 */
function openAgentDetails() {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null || !preference.agentDetailsAutoOpenAllowed) {
    return false;
  }
  preference.disclosures["agent-details"].open = true;
  setProgrammaticDisclosureOpen(elements.agentDetails, true);
  return true;
}

/**
 * Remember inspector activation until the requested replay recipient is installed.
 * For the already installed recipient, activate and open details locally. Without
 * active preferences clear pending activation; otherwise save its source generation
 * and whether a successful handoff should open the panel.
 *
 * @param {string} publicAgentId
 */
function stageReplayRecipientActivation(publicAgentId) {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null) {
    pendingReplayRecipientActivation = null;
    return;
  }
  if (state.presentation?.authority?.recipient_public_agent_id === publicAgentId) {
    pendingReplayRecipientActivation = null;
    activatedAgentPublicId = publicAgentId;
    openAgentDetails();
    return;
  }
  pendingReplayRecipientActivation = Object.freeze({
    sourcePreferenceGeneration: presentationPreferenceGeneration,
    recipientPublicAgentId: publicAgentId,
    autoOpenAgentDetails:
      preference.agentDetailsAutoOpenAllowed &&
      preference.disclosures["agent-details"].open !== true,
  });
}

/**
 * Close Agent Details in both saved preferences and the DOM when authority exists.
 * This programmatic close preserves permission for a later automatic opening.
 */
function closeAgentDetailsWithoutLatching() {
  const preference = installedActivePresentationPreference(state.presentation);
  if (preference === null) {
    return;
  }
  preference.disclosures["agent-details"].open = false;
  setProgrammaticDisclosureOpen(elements.agentDetails, false);
}

/**
 * Return whether the installed launcher product is the Replay Viewer.
 * Use product identity, not frame metadata, to select the page mode.
 */
function isReplayMode() {
  return productIdentity?.product_kind === "replay_viewer";
}

/**
 * Return the display label for a controller value.
 * Known reactive and random controllers have explicit names; other values use
 * Manual. This label helper does not validate the requested configuration.
 *
 * @param {unknown} controller
 */
function combatControllerLabel(controller) {
  if (isSystemController(controller)) return String(controller).slice(7);
  if (controller === "reactive_tdm") {
    return "Reactive TDM ALPHA";
  }
  if (controller === "random_valid") {
    return "Random";
  }
  if (controller === "scenario_5") {
    return "Reactive TDM BETA";
  }
  if (controller === "tdm_gamma") {
    return "Reactive TDM GAMMA";
  }
  return "Manual";
}

/**
 * Read and validate the three public combat settings from a frame-like object.
 * Require known controllers and information mode. Both teams accept the same
 * built-in controllers and host-declared Systems. Reactive controllers
 * (reactive_tdm, scenario_5 and tdm_gamma) and Systems require SharedObs. Return a frozen copy of
 * those settings, or null for malformed or unsupported input.
 *
 * @param {unknown} frame
 */
function combatConfigurationFromFrame(frame) {
  if (!isRecord(frame)) {
    return null;
  }
  const candidate = frame.combat_configuration;
  if (
    !isRecord(candidate) ||
    !isTeamController(candidate.team_a_controller) ||
    !isTeamController(candidate.team_b_controller) ||
    (candidate.execution_information_mode !== "shared_obs" &&
      candidate.execution_information_mode !== "no_shared_obs") ||
    ((isReactiveController(candidate.team_a_controller) ||
      isReactiveController(candidate.team_b_controller)) &&
      candidate.execution_information_mode !== "shared_obs")
  ) {
    return null;
  }
  return Object.freeze({
    team_a_controller: candidate.team_a_controller,
    team_b_controller: candidate.team_b_controller,
    execution_information_mode: candidate.execution_information_mode,
  });
}

/**
 * Publish confirmed combat settings to DevClient through a document event.
 * Only the Combat Debugger sends this event, and only for a valid configuration.
 * No event is sent for replay or missing settings.
 *
 * @param {unknown} frame
 */
function publishInstalledCombatConfiguration(frame) {
  if (productIdentity?.product_kind !== "combat_debugger") {
    return;
  }
  const configuration = combatConfigurationFromFrame(frame);
  if (configuration === null) {
    return;
  }
  document.dispatchEvent(
    new CustomEvent("marl-devclient-combat-configuration-installed", {
      detail: {
        ...configuration,
        system_choices: isRecord(frame) ? (frame.system_choices ?? []) : [],
      },
    }),
  );
}

/**
 * Build a frozen combat-configuration command from a requested settings object.
 * Return null when either requested or installed settings are invalid, or all
 * three settings already match. This prepares a command without sending it.
 *
 * @param {unknown} value
 */
function requestedCombatConfigurationCommand(value) {
  if (!isRecord(value)) {
    return null;
  }
  const requested = combatConfigurationFromFrame({ combat_configuration: value });
  const installed = combatConfigurationFromFrame(state.frame);
  if (
    requested === null ||
    installed === null ||
    (requested.team_a_controller === installed.team_a_controller &&
      requested.team_b_controller === installed.team_b_controller &&
      requested.execution_information_mode === installed.execution_information_mode)
  ) {
    return null;
  }
  return Object.freeze({
    command_type: "set_combat_configuration",
    team_a_controller: requested.team_a_controller,
    team_b_controller: requested.team_b_controller,
    execution_information_mode: requested.execution_information_mode,
  });
}

/**
 * Paint checkbox and bulk-toggle state from one local filter snapshot.
 * Include the active range choice in the enabled count. Throw TypeError if the
 * DOM checkbox count differs from the shared registry; do not change the snapshot.
 *
 * @param {typeof visualFilterState} snapshot
 */
function renderVisualFilterControls(snapshot) {
  const inputs = /** @type {NodeListOf<HTMLInputElement>} */ (
    elements.visualFilterOptions.querySelectorAll(
      'input[type="checkbox"][data-visual-filter-id]',
    )
  );
  if (inputs.length !== VISUAL_FILTER_REGISTRY.length) {
    throw new TypeError(
      `Visual Filters requires exactly ${VISUAL_FILTER_REGISTRY.length} checkboxes.`,
    );
  }
  let enabledCount = 0;
  for (const input of inputs) {
    const enabled = isVisualFilterEnabled(snapshot, input.dataset.visualFilterId);
    input.checked = enabled;
    enabledCount += enabled ? 1 : 0;
  }
  const rangesEnabled = installedPresentationRangesVisible(state.presentation);
  const visibleControlCount = VISUAL_FILTER_REGISTRY.length + 1;
  const enabledControlCount = enabledCount + (rangesEnabled ? 1 : 0);
  elements.visualFilterCount.textContent = `${enabledControlCount} enabled`;
  elements.enableAllVisualFiltersButton.disabled =
    enabledControlCount === visibleControlCount;
  elements.disableAllVisualFiltersButton.disabled = enabledControlCount === 0;
}

/**
 * Apply a local filter action and return whether its snapshot changed.
 * On change, invalidate downloads, replace the whole snapshot, pause replay, and
 * repaint. An unchanged reducer result returns false without these side effects.
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
 * Request the boolean range visibility for the current authorized audience.
 * Agent POV changes page-local state; Oracle View sends its existing server command.
 * Return whether a local change or request started. Return false while unavailable,
 * busy, disconnected, or already at the requested value.
 *
 * @param {boolean} visible
 * @returns {boolean} Whether a local change or one service request was started.
 */
function setActiveRangesVisible(visible) {
  const presentation = state.presentation;
  if (
    typeof visible !== "boolean" ||
    !isAuthorizedPresentationFrame(presentation) ||
    state.busy ||
    state.shuttingDown ||
    state.resyncRequired ||
    state.offline ||
    installedPresentationRangesVisible(presentation) === visible
  ) {
    return false;
  }
  const audience = authorizedPresentationAudience(presentation);
  if (audience === "agent_pov") {
    if (!setAgentLocalRangesVisible(visible)) {
      return false;
    }
    render();
    return true;
  }
  if (audience !== "researcher") {
    return false;
  }
  if (isReplayMode()) {
    void dispatchReplayCommand({
      command_type: "set_ranges",
      show_ranges: visible,
    });
  } else {
    void dispatchCommand(keyboardCommand("g"));
  }
  return true;
}

/**
 * Enable or disable every local filter, then request the same range visibility.
 * The range change keeps its normal local or server-confirmed authority path;
 * the two calls may repaint or send a request through their existing handlers.
 *
 * @param {boolean} enabled
 */
function applyAllVisualControls(enabled) {
  applyVisualFilterAction({ type: enabled ? "enable_all" : "disable_all" });
  setActiveRangesVisible(enabled);
}

/**
 * Choose animation policy from the matched installed transport and playback state.
 * Live uses live_once; replay uses the current cursor intent or replay_static.
 * Return a frozen policy carrying the supplied filters. consumeAnimatedRestart
 * defaults to false; when true, consume each requested animated restart once.
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

/**
 * Compare the five cursor version, index, and generation fields of two records.
 * Return false for non-records. This equality helper assumes separate protocol
 * validation; matching missing fields alone does not certify a cursor.
 *
 * @param {unknown} left @param {unknown} right
 */
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

/**
 * Resolve the installed transport and presentation into one coherent authority.
 * Return the trusted pair or null when its objects or join marker do not agree.
 */
function installedPresentationAuthority() {
  return resolveInstalledPresentationAuthorityV1(
    state.authority,
    state.frame,
    state.presentation,
  );
}

/**
 * Return whether the currently installed transport and presentation form a
 * trusted matching pair. This is the common gate for scientific UI actions.
 */
function installedAuthorityIsCoherent() {
  return installedPresentationAuthority() !== null;
}

/**
 * Find an opaque agent key in the supplied trusted presentation scene.
 * Return the matching authorized agent or null for invalid input or no match.
 * Do not infer identity from DOM order, raw slots, or an earlier presentation.
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
 * Find an opaque agent key in a trusted presentation researcher directory.
 * Return its agent or null. This lookup serves non-battlefield roster controls;
 * SVG painting and hit testing must use the audience-specific scene instead.
 *
 * @param {unknown} presentation
 * @param {unknown} presentationKey
 * @returns {Readonly<Record<string, any>> | null}
 */
function authorizedResearcherAgentForPresentationKey(presentation, presentationKey) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    typeof presentationKey !== "string"
  ) {
    return null;
  }
  return (
    asArray(authorizedPresentationResearcherSceneView(presentation)?.agents).find(
      (candidate) =>
        isRecord(candidate) && candidate.presentation_key === presentationKey,
    ) ?? null
  );
}

/**
 * Return a frozen local selection/range view for matching installed authority.
 * Return null when the presentation is untrusted or does not own local inspection.
 * The returned choices affect display only and cannot add scientific facts.
 *
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
  return preference?.localInspection.owned
    ? Object.freeze({
        inspectedPresentationKey: preference.localInspection.presentationKey,
        rangesVisible: agentLocalRangesVisible,
      })
    : null;
}

/**
 * Return the active local selection key, explicit null, or undefined.
 * Null means locally cleared selection; undefined leaves the server-confirmed
 * Oracle selection unchanged.
 *
 * @param {unknown} presentation
 * @returns {string | null | undefined}
 */
function installedLocalInspectedPresentationKey(presentation) {
  return installedLocalPresentationPreference(presentation)?.inspectedPresentationKey;
}

/**
 * Build the supplied presentation scene with its current allowed local selection.
 * Pass undefined when this authority does not own local inspection so the accepted
 * Oracle selection remains in force.
 *
 * @param {unknown} presentation
 */
function installedAuthorizedPresentationSceneView(presentation) {
  return authorizedPresentationSceneView(
    presentation,
    installedLocalInspectedPresentationKey(presentation),
  );
}

/**
 * Set the active local selection to an authorized opaque key or explicit null.
 * Return false without mutation when local inspection is unavailable or the key
 * is absent. Otherwise invalidate downloads, store the selection, and return true;
 * callers decide when to repaint.
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

/**
 * Set local range visibility for an authority that owns local inspection.
 * Require a boolean value different from the current choice. Return false for
 * an unavailable or unchanged request; otherwise invalidate downloads and return
 * true after updating page state. This helper does not repaint.
 *
 * @param {boolean} visible @returns {boolean}
 */
function setAgentLocalRangesVisible(visible) {
  const preference = installedActivePresentationPreference(state.presentation);
  if (
    preference === null ||
    !preference.localInspection.owned ||
    typeof visible !== "boolean" ||
    agentLocalRangesVisible === visible
  ) {
    return false;
  }
  invalidateReplayArtifactAction();
  agentLocalRangesVisible = visible;
  return true;
}

/**
 * Toggle the current local range choice through its normal availability gate.
 * Return whether that helper changed the value; this does not send a server command.
 *
 * @returns {boolean}
 */
function toggleAgentLocalRanges() {
  return setAgentLocalRangesVisible(!agentLocalRangesVisible);
}

/**
 * Return range visibility for this exact installed authorized presentation.
 * Oracle uses the server-confirmed choice; other audiences use allowed local
 * preferences. Invalid or different presentation objects return false.
 *
 * @param {unknown} presentation
 */
function installedPresentationRangesVisible(presentation) {
  if (
    !isAuthorizedPresentationFrame(presentation) ||
    state.presentation !== presentation
  ) {
    return false;
  }
  const localPreference = installedLocalPresentationPreference(presentation);
  return authorizedPresentationAudience(presentation) === "researcher"
    ? confirmedResearcherRangesVisible
    : localPreference?.rangesVisible === true;
}

/**
 * Compute replay download availability from a complete UI state snapshot.
 * Require settled static replay, matching authority/cursor, a connected visible
 * page, and no pending work. PNG additionally requires a ready battlefield; metric
 * downloads remain researcher tools even when the visible scene is Agent POV.
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

/**
 * Return whether the replay SVG has a settled static render and positive integer
 * CSS-pixel dimensions. This is a download readiness check, not a schema validator.
 */
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
 * Collect playback, page, cursor, and battlefield facts for artifact controls.
 * Use the supplied installed pair or null. ignoreActionPending defaults to false;
 * an existing transaction uses true to check that its other prerequisites remain
 * valid without blocking itself.
 *
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

/**
 * Paint PNG, metric, and episode-details download buttons from current capabilities.
 * Mark the matching transaction busy and show metric preparation only while the
 * metric download is pending with installed authority.
 *
 * @param {ReturnType<typeof installedPresentationAuthority>} installed
 */
function renderReplayArtifactActions(installed) {
  const capabilities = currentReplayArtifactActionCapabilities(installed);
  const pending = replayArtifactActionTransaction;
  elements.replayExportPngButton.disabled = !capabilities.exportPng;
  elements.replayDownloadMetricsButton.disabled = !capabilities.downloadMetrics;
  elements.replayEpisodeDetailsButton.disabled = !capabilities.downloadMetrics;
  elements.replayExportPngButton.setAttribute(
    "aria-busy",
    String(pending?.kind === "export_png"),
  );
  elements.replayDownloadMetricsButton.setAttribute(
    "aria-busy",
    String(pending?.kind === "download_metrics"),
  );
  elements.replayEpisodeDetailsButton.setAttribute(
    "aria-busy",
    String(pending?.kind === "download_episode_details"),
  );
  elements.replayMetricsPreparation.hidden =
    installed === null || pending?.kind !== "download_metrics";
  elements.replayMetricsPreparationText.textContent = METRIC_PREPARATION_MESSAGE;
}

/**
 * Start one allowed PNG, metric, or episode-details download transaction.
 * Return null if another action is pending or the current surface is unavailable.
 * Otherwise freeze the exact authority, playback generation, local selection,
 * ranges, and filters, store the transaction, and repaint its controls.
 *
 * @param {"export_png" | "download_metrics" | "download_episode_details"} kind
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
 * Return whether a captured download transaction still owns the current UI state.
 * Require the same authority objects, playback generation, and download capability.
 * PNG also requires unchanged filters, range choice, and local selection.
 *
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

/**
 * Clear a completed download only if it is still the active transaction.
 * Refresh controls from current authority; an older completion cannot clear a
 * newer transaction.
 *
 * @param {NonNullable<typeof replayArtifactActionTransaction>} transaction
 */
function finishReplayArtifactAction(transaction) {
  if (replayArtifactActionTransaction !== transaction) {
    return;
  }
  replayArtifactActionTransaction = null;
  renderReplayArtifactActions(installedPresentationAuthority());
}

/**
 * Discard the active download identity and hide metric-preparation indicators.
 * This makes later asynchronous results stale; it does not abort their underlying
 * network request or image work.
 */
function invalidateReplayArtifactAction() {
  replayArtifactActionTransaction = null;
  elements.replayMetricsPreparation.hidden = true;
  elements.replayDownloadMetricsButton.setAttribute("aria-busy", "false");
  elements.replayEpisodeDetailsButton.setAttribute("aria-busy", "false");
}

/**
 * Trigger a browser download for a Blob and a safe basename.
 * Require 1–240 filename characters using letters, digits, dots, underscores,
 * and hyphens, beginning with a letter or digit; throw TypeError otherwise.
 * Create an object URL, click a temporary anchor, and revoke the URL in finally.
 *
 * @param {Blob} blob @param {string} filename
 */
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

/**
 * Convert a download failure into a frozen user notice and severity.
 * Use specific messages for missing reports and denied authorization; preserve
 * an Error message for other failures. This does not display the notice.
 *
 * @param {unknown} error
 */
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

/**
 * Capture and download the current settled replay battlefield as a PNG.
 * Return without work when unavailable. Check transaction identity after capture
 * and validate its artifact type; ignore stale results, display current failures,
 * and always release this transaction. No simulator command is sent.
 */
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
      /**
       * Return whether this PNG transaction still matches the installed view.
       * The exporter calls this during asynchronous work to reject a changed frame,
       * selection, filter configuration or playback generation.
       */
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

/**
 * Download metrics CSV, or episode-details JSON when details is true.
 * details defaults to false. Capture one settled replay transaction, request its
 * current metric scope/frame when needed, and discard stale replies. Display
 * current request failures and always release the transaction.
 *
 * @param {boolean} [details]
 */
async function downloadReplayMetricReport(details = false) {
  const transaction = beginReplayArtifactAction(
    details ? "download_episode_details" : "download_metrics",
  );
  if (transaction === null) {
    return;
  }
  try {
    const context = replayMetricContext();
    if (!details && context === null) return;
    const report =
      !details && context !== null
        ? await getReplayMetrics(state.token, context.frameIndex, context.scope, "csv")
        : await getReplayEpisodeDetails(state.token);
    if (
      !replayArtifactActionIsCurrent(transaction) ||
      (!details && replayMetricContext()?.key !== context?.key)
    ) {
      return;
    }
    downloadReplayArtifact(
      new Blob([report.bytes], {
        type: details ? "application/json; charset=utf-8" : "text/csv; charset=utf-8",
      }),
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

/** @type {{key: string, summary: Record<string, any>} | null} */
let replayMetricSummary = null;
/** @type {string | null} */
let replayMetricRequest = null;
/** @type {{key: string, message: string} | null} */
let replayMetricFailure = null;
let replayMetricRenderKey = "";
let replayMetricLoadedKey = "";
let replayMetricGeneration = 0;
/** @type {Record<string, any> | null} */
let replayMetricCatalog = null;
/** @type {ReturnType<typeof buildMetricSearchIndex>} */
let replayMetricSearchIndex = buildMetricSearchIndex([]);
let replayMetricCatalogPending = false;
let replayMetricCatalogFailure = "";
let replayMetricSearchLimit = 20;
/** @type {HTMLButtonElement | null} */
let replayMetricSearchActive = null;
/** @type {string | null} */
let replayMetricFocus = null;
/** @type {number | null} */
let replayMetricHighlightTimer = null;

/**
 * Cancel the metric-row highlight timer and remove all current highlight classes.
 * This does not change metric selection or request state.
 */
function clearReplayMetricHighlight() {
  if (replayMetricHighlightTimer !== null) {
    window.clearTimeout(replayMetricHighlightTimer);
    replayMetricHighlightTimer = null;
  }
  for (const row of elements.metricRows.querySelectorAll(".metric-search-match")) {
    row.classList.remove("metric-search-match");
  }
}

/**
 * Set the active search-result button for keyboard navigation, or clear it with null.
 * Update aria-selected on old/new buttons and the search input active-descendant.
 *
 * @param {HTMLButtonElement | null} button
 */
function setActiveMetricSearchResult(button) {
  replayMetricSearchActive?.setAttribute("aria-selected", "false");
  replayMetricSearchActive = button;
  if (button) {
    button.setAttribute("aria-selected", "true");
    elements.metricSearch.setAttribute("aria-activedescendant", button.id);
  } else {
    elements.metricSearch.removeAttribute("aria-activedescendant");
  }
}

/**
 * Show or hide search suggestions, requiring a nonempty trimmed search value.
 * Update aria-expanded and clear active keyboard selection whenever the list closes.
 *
 * @param {boolean} open
 */
function setReplayMetricSearchOpen(open) {
  elements.metricSearchContent.hidden = !open || !elements.metricSearch.value.trim();
  elements.metricSearch.setAttribute(
    "aria-expanded",
    String(!elements.metricSearchContent.hidden),
  );
  if (elements.metricSearchContent.hidden) setActiveMetricSearchResult(null);
}

/**
 * Render catalog search matches up to the current result limit.
 * Clear the active result, update status and the More control, and wire each row
 * to measurement selection. Do nothing until a catalog is available.
 */
function renderReplayMetricSearch() {
  if (replayMetricCatalog === null) return;
  setActiveMetricSearchResult(null);
  const query = elements.metricSearch.value.trim();
  const { matches, message } = searchMeasurements(replayMetricSearchIndex, query);
  elements.metricSearchResults.hidden = !query;
  elements.metricSearchMore.hidden = matches.length <= replayMetricSearchLimit;
  elements.metricSearchStatus.textContent = query
    ? (message ??
      `${Math.min(matches.length, replayMetricSearchLimit)} of ${matches.length} measurements found.`)
    : "Search includes every numerical CSV column, including those that do not apply to this roster.";
  renderMetricSearchResults(
    elements.metricSearchResults,
    matches,
    replayMetricCatalog.topics,
    replayMetricSearchLimit,
    selectReplayMeasurement,
  );
}

/**
 * Handle selection of a catalog measurement row.
 * For an inapplicable row, show its definition. Otherwise select its topic/view,
 * remember the row to focus, and refresh metrics. Clear old search focus/highlights;
 * do nothing without a loaded catalog.
 *
 * @param {Record<string, any>} row
 */
function selectReplayMeasurement(row) {
  if (replayMetricCatalog === null) return;
  setActiveMetricSearchResult(null);
  clearReplayMetricHighlight();
  replayMetricFocus = null;
  if (!row.applicable) {
    renderMetricDefinition(
      elements.metricSearchDefinition,
      row,
      replayMetricCatalog.topics,
    );
    return;
  }
  elements.metricSearchDefinition.hidden = true;
  elements.metricSelection.value = row.primary_topic;
  renderMetricNavigation(
    elements.metricSelection,
    elements.metricView,
    elements.metricViewField,
    replayMetricCatalog.topics,
  );
  elements.metricView.value = row.primary_view;
  replayMetricFocus = row.name;
  renderReplayMetrics();
}

/**
 * Focus and scroll to the requested rendered metric row when it exists.
 * Highlight it for 3 seconds, then clear the highlight. Keep a missing row pending
 * so a later metric render can complete the focus request.
 */
function focusReplayMeasurement() {
  if (replayMetricFocus === null) return;
  const row = elements.metricRows.querySelector(
    `[data-metric="${CSS.escape(replayMetricFocus)}"]`,
  );
  if (row) {
    clearReplayMetricHighlight();
    row.classList.add("metric-search-match");
    replayMetricHighlightTimer = window.setTimeout(() => {
      row.classList.remove("metric-search-match");
      replayMetricHighlightTimer = null;
    }, 3000);
    row.scrollIntoView({ block: "nearest" });
    row.querySelector(".metric-measure").focus({ preventScroll: true });
    replayMetricFocus = null;
  }
}

/**
 * Return the installed replay digest, frame, scope, and request/cache keys.
 * Use final scope only when selected, otherwise cursor scope; final data keys omit
 * the changing cursor. Return null without a replay pair, string digest, or safe
 * integer frame index.
 */
function replayMetricContext() {
  const installed = installedPresentationAuthority();
  if (!isReplayMode() || installed === null) return null;
  const frame = installed.transport;
  const summary = frame.artifact_facts?.artifact_summary ?? frame.artifact_summary;
  const digest = summary?.replay_reference?.canonical_digest_sha256;
  const frameIndex = frame.cursor?.frame_index;
  const scope = elements.metricScope.value === "final" ? "final" : "cursor";
  if (typeof digest !== "string" || !Number.isSafeInteger(frameIndex)) return null;
  return {
    digest,
    frameIndex,
    scope: /** @type {"cursor" | "final"} */ (scope),
    catalogKey: `${installed.presentation.source.source_session_id}:${digest}`,
    key: `${installed.presentation.source.source_session_id}:${digest}:${scope}:${scope === "final" ? "final" : frameIndex}`,
  };
}

/**
 * Render metrics and start the catalog or summary GETs that the current scope needs.
 * Reset cached UI state when the replay changes, reuse current results, and label
 * loading or failed requests. Check generation and replay digest before accepting
 * async replies so an old artifact cannot populate the new page.
 */
function renderReplayMetrics() {
  const context = replayMetricContext();
  const loadedKey = context?.catalogKey ?? "";
  if (loadedKey !== replayMetricLoadedKey) {
    clearReplayMetricHighlight();
    replayMetricLoadedKey = loadedKey;
    replayMetricGeneration += 1;
    replayMetricSummary = null;
    replayMetricRequest = null;
    replayMetricFailure = null;
    replayMetricRenderKey = "";
    replayMetricCatalog = null;
    replayMetricSearchIndex = buildMetricSearchIndex([]);
    replayMetricCatalogPending = false;
    replayMetricCatalogFailure = "";
    replayMetricFocus = null;
    replayMetricSearchLimit = 20;
    elements.metricSearch.value = "";
    elements.metricSearch.disabled = true;
    setReplayMetricSearchOpen(false);
    elements.metricSearchResults.replaceChildren();
    elements.metricSearchResults.hidden = true;
    elements.metricSearchMore.hidden = true;
    elements.metricSearchDefinition.hidden = true;
  }
  elements.metricProgress.hidden = true;
  elements.metricRows.setAttribute("aria-busy", "false");
  elements.metricPanel.hidden =
    context === null ||
    installedPresentationAuthority()?.presentation.match_summary?.task_mode !== 1;
  if (context === null || elements.metricPanel.hidden || !elements.metricPanel.open)
    return;
  const generation = replayMetricGeneration;
  if (
    replayMetricCatalog === null &&
    !replayMetricCatalogPending &&
    !replayMetricCatalogFailure
  ) {
    replayMetricCatalogPending = true;
    elements.metricSearchStatus.textContent = "Preparing the measurement catalog.";
    void getReplayMetricCatalog(state.token)
      .then((catalog) => {
        if (generation !== replayMetricGeneration) return;
        if (catalog.source_replay_digest !== context.digest) {
          throw new TypeError("Measurement catalog belongs to another replay.");
        }
        replayMetricCatalog = catalog;
        replayMetricSearchIndex = buildMetricSearchIndex(
          catalog.measurements,
          catalog.topics,
          catalog.agents,
          catalog.class_names,
        );
        elements.metricSearch.disabled = false;
        renderMetricNavigation(
          elements.metricSelection,
          elements.metricView,
          elements.metricViewField,
          catalog.topics,
        );
        renderReplayMetricSearch();
      })
      .catch((error) => {
        if (generation !== replayMetricGeneration) return;
        replayMetricCatalogFailure =
          error instanceof Error ? error.message : "Catalog request failed.";
        elements.metricSearchStatus.textContent = `Search unavailable: ${replayMetricCatalogFailure}`;
      })
      .finally(() => {
        if (generation !== replayMetricGeneration) return;
        replayMetricCatalogPending = false;
        renderReplayMetrics();
      });
  }
  if (replayMetricSummary?.key === context.key) {
    const summary = replayMetricSummary.summary;
    renderMetricNavigation(
      elements.metricSelection,
      elements.metricView,
      elements.metricViewField,
      summary.topics,
    );
    const renderKey = `${context.key}:${elements.metricSelection.value}:${elements.metricView.value}`;
    if (renderKey !== replayMetricRenderKey) {
      renderMetricRows(
        elements.metricRows,
        elements.metricSelection.value,
        elements.metricView.value,
        summary,
        elements.metricDescription,
      );
      replayMetricRenderKey = renderKey;
    }
    elements.metricStatus.textContent = `${summary.scope === "final" ? "Entire Episode" : "Up to Current Tick"} · tick ${summary.simulator_step_count}`;
    focusReplayMeasurement();
    return;
  }
  elements.metricRows.replaceChildren();
  replayMetricRenderKey = "";
  if (replayMetricFailure?.key === context.key) {
    elements.metricStatus.textContent = replayMetricFailure.message;
    return;
  }
  elements.metricProgress.hidden = false;
  elements.metricRows.setAttribute("aria-busy", "true");
  elements.metricStatus.textContent = `${METRIC_PREPARATION_MESSAGE} Playback and POV controls remain available.`;
  if (replayMetricRequest !== null) return;
  replayMetricRequest = context.key;
  void getReplayMetrics(state.token, context.frameIndex, context.scope)
    .then((summary) => {
      if (generation !== replayMetricGeneration) return;
      if (summary.source_replay_digest !== context.digest) {
        throw new TypeError("Metric analysis belongs to another replay.");
      }
      replayMetricSummary = { key: context.key, summary };
      replayMetricFailure = null;
    })
    .catch((error) => {
      if (generation !== replayMetricGeneration) return;
      replayMetricFailure = {
        key: context.key,
        message:
          error instanceof Error
            ? `Metrics unavailable: ${error.message}`
            : "Metrics unavailable.",
      };
    })
    .finally(() => {
      if (generation !== replayMetricGeneration) return;
      replayMetricRequest = null;
      renderReplayMetrics();
    });
}

/**
 * Return whether the installed trusted live presentation is scripted inspection.
 * Require a supported live transport kind and coherent pair; raw frame metadata
 * or an unjoined presentation cannot enable these controls.
 */
function liveScriptedInspectionOnly() {
  return (
    installedAuthorityIsCoherent() &&
    (state.frame?.frame_kind === "researcher_live_debugger" ||
      state.frame?.frame_kind === "actor_pov_live_debugger" ||
      state.frame?.frame_kind === "shared_obs_agent_pov_live_debugger") &&
    authorizedPresentationInspectionState(state.presentation).state_kind ===
      "live_scripted"
  );
}

/**
 * Return whether a command is allowed by the scripted-live UI gate.
 * Allow view changes, recording commands accepted by their lifecycle guard, and
 * fresh unmodified N/G keys. This browser check does not replace server validation.
 *
 * @param {Record<string, unknown>} command
 */
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
 * Return the incoming transition ID from the trusted presentation event branch.
 * Use the researcher or recipient-scoped field for its audience; return null for
 * untrusted input, frame zero without events, or a missing/empty string.
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

/**
 * Return recording metadata from the installed coherent transport, or null.
 * A stale or unjoined raw frame cannot control recording UI availability.
 */
function recordingStatus() {
  const frame = installedPresentationAuthority()?.transport;
  return isRecord(frame?.recording) ? frame.recording : null;
}

/**
 * Return whether installed recording metadata has left the recording lifecycle.
 * No recording metadata means this particular closeout gate is inactive.
 */
function recordingScientificControlsFenced() {
  const recording = recordingStatus();
  return recording !== null && recording.lifecycle !== "recording";
}

/**
 * Return whether recording metadata blocks restart without an available discard.
 * Require both restart_fenced and no explicit discard_available permission.
 */
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
  "Authorized visible bodies can still be inspected locally; recording closeout has fenced POV switching, simulator input, and pending-action controls.";
const TERMINAL_REPLAY_BATTLEFIELD_LABEL =
  "Read-only terminal replay battlefield snapshot.";
const TERMINAL_REPLAY_BATTLEFIELD_INSTRUCTIONS =
  "This replay frame is terminal. Activate an agent to inspect current facts, or use the timeline to review another frame.";
const TERMINAL_REPLAY_AGENT_BATTLEFIELD_INSTRUCTIONS =
  "This replay frame is terminal. Activate a visible body or choose any agent in the roster to switch the fog-of-war recipient; use the timeline to review another frame.";

/**
 * Return whether the live battlefield may submit manual commands now.
 * Require a coherent nonterminal editable presentation and an idle connected page,
 * with no scripted-inspection or recording-closeout gate.
 */
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

/**
 * Return whether the installed scripted-live battlefield may accept local inspection.
 * Require a nonterminal frame, no recording-closeout gate, and an idle connected
 * page. This does not permit manual simulator action editing.
 */
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

/**
 * Return whether local Agent POV inspection remains available during recording closeout.
 * Require a coherent live nonterminal Agent presentation and an idle connected page.
 * Simulator and pending-action controls remain governed by their separate gates.
 */
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

/**
 * Return whether any coherent authorized live presentation is installed.
 * This broad display check does not imply commands are idle, legal, or editable.
 */
function installedLiveScientificInspectionAvailable() {
  return (
    !isReplayMode() &&
    isAuthorizedPresentationFrame(state.presentation) &&
    installedAuthorityIsCoherent()
  );
}

/**
 * Set battlefield accessibility role, focusability, and instructions for current authority.
 * Distinguish replay, editable live, scripted inspection, Agent closeout inspection,
 * and unavailable surfaces using the same interaction gates as the controls.
 */
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
        ? "Live Agent POV is interactive. Activate a visible authorized actor or choose any active actor in Roster to control it and switch POV; Shift-click selects a visible authorized target; Escape clears the target and leaves battlefield focus."
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

/**
 * Show product-specific page sections and audience-specific live help.
 * Update the root viewer-mode marker, timeline visibility, battlefield instructions,
 * and inspector heading from installed product and presentation state.
 */
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

/**
 * Paint replay identity, completion, audience, reference, range, and timeline controls.
 * Use the supplied coherent pair and the authorized artifact branch for its audience.
 * Do nothing outside replay or without a pair; availability follows current busy,
 * connection, selection, and playback state.
 *
 * @param {ReturnType<typeof installedPresentationAuthority>} installed
 */
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
      title: "Replay Artifact",
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
          label: "Canonical Digest",
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

/**
 * Paint live recording lifecycle, persistence facts, actions, and progress hints.
 * Use the supplied installed pair and recording metadata; hide replay/unavailable
 * recording surfaces and close obsolete discard dialogs. Progress text describes
 * current request state, not a fabricated percentage.
 *
 * @param {ReturnType<typeof installedPresentationAuthority>} installed
 */
function renderRecordingControls(installed) {
  elements.recordingMetricsProgress.hidden = true;
  elements.recordingMetricsHelp.hidden = true;
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
  const preparation =
    state.busy && !state.offline && !state.resyncRequired
      ? activeLiveCommandTransaction?.recordingPreparation
      : null;
  if (preparation) {
    elements.recordingMetricsProgress.hidden = false;
    elements.recordingMetricsHelp.hidden = false;
    elements.recordingStatusNote.textContent =
      preparation === "metrics"
        ? REPLAY_SAVE_MESSAGE
        : "Processing the recorded game. If the episode ends, its metrics will be prepared before saving. Longer replays can take longer.";
  }

  if (recording.discard_available !== true && elements.recordingDiscardDialog.open) {
    elements.recordingDiscardDialog.close();
    pendingRecordingReplacement = null;
  }
}

/**
 * Return an array unchanged, or a fresh empty array for any other input.
 * This does not validate array elements.
 *
 * @param {unknown} value
 * @returns {any[]}
 */
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * Return an integer numeric input unchanged, otherwise fallback, which defaults to 0.
 * Numeric strings are not coerced into accepted integers.
 *
 * @param {unknown} value
 * @param {number} fallback
 * @returns {number}
 */
function integer(value, fallback = 0) {
  return Number.isInteger(value) ? Number(value) : fallback;
}

/**
 * Build a readable agent label using the current authorized display ID when available.
 * A nonempty public ID falls back to itself; invalid IDs return Agent unavailable.
 *
 * @param {unknown} publicAgentId
 * @returns {string}
 */
function agentIdentity(publicAgentId) {
  const displayId = authorizedPresentationAgentDisplayId(
    state.presentation,
    publicAgentId,
  );
  return typeof publicAgentId === "string" && publicAgentId.length > 0
    ? `Agent ID ${displayId ?? publicAgentId}`
    : "Agent unavailable";
}

/**
 * Read whether transport is at its terminal or final-frame boundary.
 * Replay compares current and final frame indices; live accepts the supported
 * boolean/object terminal forms and legacy terminated/truncated flags. This is
 * a compatibility reader, not a validator of an arbitrary frame.
 *
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
]);

/**
 * Return the selected live editable actor team and automatic controller, or null.
 * Resolve the actor through the authorized researcher directory and confirmed
 * combat settings. Manual teams, missing identity, and noneditable modes return null.
 */
function policyControlledActor() {
  const configuration = combatConfigurationFromFrame(state.frame);
  if (configuration === null) {
    return null;
  }
  const inspectionState = authorizedPresentationResearcherInspectionState(
    state.presentation,
  );
  if (inspectionState.state_kind !== "live_editable") {
    return null;
  }
  const inspection = isRecord(inspectionState.inspection)
    ? inspectionState.inspection
    : null;
  if (inspection === null) {
    return null;
  }
  const actor = authorizedResearcherAgentForPresentationKey(
    state.presentation,
    inspection.actor_presentation_key,
  );
  if (actor === null || actor.public_agent_id !== inspection.actor_public_agent_id) {
    return null;
  }
  let team;
  let controller;
  if (actor.team_id === 1) {
    team = "Team A";
    controller = configuration.team_a_controller;
  } else if (actor.team_id === 2) {
    team = "Team B";
    controller = configuration.team_b_controller;
  } else {
    return null;
  }
  return controller === "manual" ? null : Object.freeze({ team, controller });
}

/**
 * Return whether a command would edit an automatically controlled actor action.
 * Block battlefield pointers, target selection, draft keys, and Escape while such
 * an actor is selected. Other commands keep their normal availability checks.
 *
 * @param {Record<string, unknown>} command
 */
function policyControllerBlocksActionEdit(command) {
  if (policyControlledActor() === null) {
    return false;
  }
  if (command.command_type === "battlefield_pointer") {
    return true;
  }
  if (command.command_type === "roster_selection" && command.role === "target") {
    return true;
  }
  if (command.command_type !== "keyboard" || typeof command.key !== "string") {
    return false;
  }
  const key = command.key.toLowerCase();
  return DRAFT_KEYS.has(key) || key === "escape";
}

/**
 * Return whether a command can stage a draft before a later Enter.
 * Recognize the preparation command set plus draft, Tab, and Escape keys; this
 * does not check freshness, modifiers, or current command availability.
 *
 * @param {Record<string, unknown>} command
 */
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

/**
 * Return whether a keyboard command belongs to the draft-preparation key group.
 * Other command types and non-string keys return false.
 *
 * @param {Record<string, unknown>} command
 */
function isDeferredDraftKey(command) {
  return (
    command.command_type === "keyboard" &&
    typeof command.key === "string" &&
    commandPreparesDeferredSubmit(command)
  );
}

/**
 * Return whether a draft-preparation key can be retained during a pending request.
 * Require explicit false Ctrl/Alt/Meta/repeat flags; Shift is allowed only for Tab.
 *
 * @param {Record<string, unknown>} command
 */
function isFreshDeferredDraft(command) {
  if (!isDeferredDraftKey(command)) {
    return false;
  }
  const key = String(command.key).toLowerCase();
  return (
    command.ctrl_key === false &&
    command.alt_key === false &&
    command.meta_key === false &&
    command.repeat === false &&
    (command.shift_key === false || key === "tab")
  );
}

/**
 * Return whether a command is a fresh Enter with every modifier and repeat false.
 * Only the keyboard command with a string Enter key qualifies.
 *
 * @param {Record<string, unknown>} command
 */
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
 * Return whether a non-null command is the Escape keyboard command.
 * This recognizes key identity only; modifier and repeat checks belong elsewhere.
 *
 * @param {Readonly<Record<string, unknown>> | null} command
 */
function isEscapeKeyboardCommand(command) {
  return (
    command !== null &&
    command.command_type === "keyboard" &&
    typeof command.key === "string" &&
    command.key.toLowerCase() === "escape"
  );
}

/**
 * Consume eligible live keyboard input while one request owns the battlefield.
 * Retain at most one fresh draft and a following Enter for ordered dispatch after
 * a coherent successor. Escape also saves focus-release intent. Return whether
 * the input was consumed; replay and other unavailable states keep native behavior.
 *
 * @param {Record<string, unknown>} command
 */
function retainFencedCommand(command) {
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
  if (isEscapeKeyboardCommand(command)) {
    transaction.releaseFocusAfterSettlement = true;
  }
  if (isDeferredDraftKey(command)) {
    if (
      transaction.deferredDraft === null &&
      transaction.deferredSubmit === null &&
      isFreshDeferredDraft(command)
    ) {
      transaction.deferredDraft = Object.freeze({ ...command });
      setNotice(
        "Input queued; waiting for the current battlefield update first…",
        "info",
      );
      renderConnection();
    }
    return true;
  }
  if (
    command.command_type !== "keyboard" ||
    typeof command.key !== "string" ||
    command.key.toLowerCase() !== "enter"
  ) {
    return false;
  }
  if (
    transaction.deferredSubmit === null &&
    (transaction.allowsDeferredSubmit || transaction.deferredDraft !== null) &&
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
 * Classify keyboard commands as draft, interactive-submit, or null.
 * Recognize draft keys and Enter/Return/Space aliases without deciding legality.
 *
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
 * Return the browser advisory availability and notice for a command/frame pair.
 * Reject draft and submit roles at a terminal boundary; other commands pass this
 * particular gate. Python repeats the real authorization before changing state.
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

/**
 * Read the installed raw transport revision, falling back to 0 if it is not an integer.
 * Callers use it as the command base revision; it does not establish authority.
 */
function currentRevision() {
  return integer(state.frame?.revision, 0);
}

/**
 * Store a notice message and level for the next connection render.
 * The level defaults to info; this helper does not repaint or send a request.
 *
 * @param {string} message
 * @param {string} level
 */
function setNotice(message, level = "info") {
  state.notice = message;
  state.noticeLevel = level;
}

/**
 * Clear frames and presentation after a product-identity mismatch.
 * Mark the page offline and in need of resynchronization, disconnect replay playback,
 * and store the error notice so no incompatible scientific surface remains active.
 *
 * @param {ProductIdentityMismatchError} error
 */
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
 * Convert a value to a display string with underscores replaced by spaces
 * and word starts capitalized. This is a label formatter, not protocol normalization.
 *
 * @param {unknown} value
 */
function humanize(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

/**
 * Map internal audience names to Oracle View or Agent POV.
 * Unknown values return View unavailable; protocol discriminators remain unchanged.
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

/**
 * Paint connection status, pending-input status, notice text, and battlefield busy state.
 * Read page state only; rendering a status does not reconnect or retry a command.
 */
function renderConnection() {
  let label = "Online";
  let status = "online";
  if (state.shuttingDown) {
    label = "Shutting down";
    status = "busy";
  } else if (state.busy) {
    label = activeLiveCommandTransaction?.deferredSubmit
      ? "Submit queued"
      : activeLiveCommandTransaction?.deferredDraft
        ? "Input queued"
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

/**
 * Paint match, episode, view, recording, and command toolbar controls.
 * The optional installed pair defaults to the currently resolved authority. Missing
 * authority clears scientific chrome; all controls respect busy, connection,
 * terminal, recording, and audience gates.
 *
 * @param {ReturnType<typeof installedPresentationAuthority>} [installed]
 */
function renderSessionToolbar(installed = installedPresentationAuthority()) {
  const frame = installed?.transport ?? null;
  const presentation = installed?.presentation ?? null;
  renderInstalledMatch(presentation);
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
  const incomingTransition = authorizedPresentationLatestTransitionId(presentation);
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
 * Render match title, team summaries, and task selection from a presentation.
 * Delegate to the shared match-summary renderer with this page DOM targets.
 *
 * @param {unknown} presentation
 */
function renderInstalledMatch(presentation) {
  renderMatchSummary(
    {
      root: elements.matchScoreboard,
      task: elements.matchTask,
      teams: [elements.matchTeamA, elements.matchTeamB],
      taskSelect: elements.taskSelect,
    },
    presentation,
  );
}

/**
 * Mark a draft button selected or unselected in aria-pressed and its dataset.
 * This changes presentation state only, not the pending simulator action.
 *
 * @param {HTMLButtonElement} button
 * @param {boolean} selected
 */
function setDraftSelection(button, selected) {
  button.setAttribute("aria-pressed", String(selected));
  button.dataset.selected = String(selected);
}

/**
 * Paint an action choice availability while keeping its explanation focusable.
 * Set aria-disabled and attach the supplied descriptor, or build a legality tooltip
 * from explanation when descriptor is null (the default). Python still checks
 * the exact action mask before accepting an edit.
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
 * Mark a command button unavailable and remove its old scientific tooltip.
 * Use this when no exact authorized owner exists; generic help must not preserve
 * or invent an owner-specific legality claim.
 *
 * @param {HTMLButtonElement} button
 */
function clearOwnerBoundLegalityAvailability(button) {
  button.setAttribute("aria-disabled", "true");
  button.dataset.authoritativeAvailable = "false";
  clearPresentationTooltipOwner(button);
}

/**
 * Rebuild target options from the authorized editable decision mask and draft.
 * Resolve target identities through the researcher directory and command-slot
 * mapping. Show Basic/Ultimate pair availability and current choice; fall back
 * to No authorized targets when no option can be built.
 *
 * @param {Readonly<Record<string, any>>} presentation
 * @param {Readonly<Record<string, any>>} inspection
 */
function renderCommandTargets(presentation, inspection) {
  const mask = isRecord(inspection.decision_mask) ? inspection.decision_mask : {};
  const action = isRecord(inspection.draft_action) ? inspection.draft_action : {};
  const selectedTargetAction = Number(action.target_action);
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
    } else {
      const commandSlot = authorizedOracleCommandSlotForPublicAgentId(
        presentation,
        target.target_public_agent_id,
      );
      if (!Number.isInteger(commandSlot)) {
        continue;
      }
      option.value = String(commandSlot);
      option.textContent = `${agentIdentity(target.target_public_agent_id)} · ${basic} · ${ultimate}`;
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
 * Paint the pending live action and its decision-mask explanations.
 * Use only the authorized editable researcher branch and a matching actor identity.
 * Update movement, stance, ability, and target controls; clear owner-bound facts
 * where their exact authorized owner cannot be found.
 *
 * @param {Readonly<Record<string, any>>} presentation
 */
function renderDraftState(presentation) {
  const inspectionState = authorizedPresentationResearcherInspectionState(presentation);
  const inspection = isRecord(inspectionState.inspection)
    ? inspectionState.inspection
    : {};
  const mask = isRecord(inspection.decision_mask) ? inspection.decision_mask : {};
  const pending = isRecord(inspection.draft_action) ? inspection.draft_action : {};
  const controlledCandidate = authorizedResearcherAgentForPresentationKey(
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
  if (
    authorizedPresentationAudience(presentation) === "researcher" &&
    Number.isInteger(controlledSlot)
  ) {
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

/**
 * Apply current live/replay, authority, lifecycle, and controller gates to commands.
 * Paint a valid editable draft when available, otherwise clear its controls.
 * Automatic-controller action edits stay disabled while allowed submission and
 * inspection commands keep their own gates.
 */
function renderCommandAvailability() {
  const presentation = state.presentation;
  const inspectionState = authorizedPresentationResearcherInspectionState(presentation);
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
  const policyControlled = policyControlledActor();
  const policyControllerReadOnly = policyControlled !== null;
  elements.commandDeck?.setAttribute(
    "data-policy-controller-read-only",
    String(policyControllerReadOnly),
  );
  if (policyControlled !== null) {
    const controllerLabel = combatControllerLabel(policyControlled.controller);
    elements.commandControlledActor.textContent += ` · ${controllerLabel} (read-only)`;
    elements.commandControlledActor.setAttribute(
      "aria-label",
      `${elements.commandControlledActor.getAttribute("aria-label") ?? `Controlled ${policyControlled.team} actor`}. ${controllerLabel} supplies this actor's action.`,
    );
  }
  elements.commandTargetSelect.disabled =
    disabled ||
    !editableDraft ||
    scientificFenced ||
    isTerminal(state.frame) ||
    policyControllerReadOnly;
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
        policyControllerBlocksActionEdit(command) ||
        (scriptedAdvance && isTerminal(state.frame)) ||
        recordingDecision.action === "block" ||
        !mode.allowed;
    }
  }
}

/**
 * Publish choreography status as root DOM data attributes for diagnostics.
 * Expose motion, pause, render policy, and submission blocking from the supplied
 * snapshot without adding motion-editing controls to the product.
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

/**
 * Label the selected Oracle replay agent as a reference rather than a live target.
 * Resolve the selected key from the installed authorized scene, update its SVG
 * accessible label, and attach a non-inspectable agent tooltip. Other views do nothing.
 */
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
 * Resolve an opaque scene key to the single action allowed in the current view.
 * Return a frozen local-inspection, replay-selection, replay-POV, or live-control
 * effect, or null when unavailable. Require coherent current authority; DOM role
 * attributes and raw slot values cannot choose the effect.
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
  const battlefieldInspectionState =
    authorizedPresentationInspectionState(presentation);
  const researcherInspectionState =
    authorizedPresentationResearcherInspectionState(presentation);
  const replayPovSwitch = isReplayMode() && audience === "agent_pov";
  if (audience === "agent_pov" && agent.life_state === "corpse") {
    return Object.freeze({
      effect: "local_inspection",
      presentationKey,
      agent,
      audience,
    });
  }
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
  if (
    audience === "agent_pov" &&
    (battlefieldInspectionState.state_kind === "live_scripted" ||
      recordingScientificControlsFenced())
  ) {
    return Object.freeze({
      effect: "local_inspection",
      presentationKey,
      agent,
      audience,
    });
  }
  if (audience === "agent_pov") {
    const commandSlot = authorizedOracleCommandSlotForPublicAgentId(
      presentation,
      agent.public_agent_id,
    );
    return researcherInspectionState.state_kind === "live_editable" &&
      Number.isInteger(commandSlot)
      ? Object.freeze({
          effect: "live_control",
          presentationKey,
          commandSlot,
          agent,
          audience,
        })
      : null;
  }
  if (battlefieldInspectionState.state_kind === "live_scripted") {
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
  return researcherInspectionState.state_kind === "live_editable"
    ? Object.freeze({
        effect: "live_control",
        presentationKey,
        commandSlot,
        agent,
        audience,
      })
    : null;
}

/**
 * Resolve an event target to an authorized SVG agent activation.
 * Reject non-elements, non-SVG agent owners, and nested tooltip controls. Return
 * the effect plus its element/key, or null; do not perform the activation yet.
 *
 * @param {unknown} target
 * @returns {(NonNullable<ReturnType<typeof authorizedAgentActivation>> & {presentationKey: string, element: SVGElement}) | null}
 */
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

/**
 * Assign each painted agent its current accessible action, focus state, and tooltip.
 * Use only the installed authorized scene and activation gate. Unavailable agents
 * remain images without keyboard activation; active agents become labeled buttons.
 */
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
 * Apply the currently authorized effect for one opaque agent key.
 * Return false when unavailable, otherwise update local inspection or start the
 * proper live/replay request. pointerOriginated defaults to false; pointer control
 * activation hands focus to the battlefield, while keyboard roster activation
 * keeps its native focus route. Server-owned scientific state waits for its reply.
 *
 * @param {string} presentationKey
 * @param {Readonly<{pointerOriginated?: boolean}>} context
 * @returns {boolean}
 */
function activateAuthorizedAgent(presentationKey, { pointerOriginated = false } = {}) {
  const activation = authorizedAgentActivation(presentationKey);
  if (activation === null) {
    return false;
  }
  if (activation.effect === "local_inspection") {
    const localSelectionInstalled = setLocalInspectedPresentationKey(presentationKey);
    if (!localSelectionInstalled && activation.agent.life_state !== "corpse") {
      return false;
    }
    activatedAgentPublicId = String(activation.agent.public_agent_id);
    render();
    openAgentDetails();
    return true;
  }
  if (activation.effect === "replay_pov_switch") {
    stageReplayRecipientActivation(String(activation.agent.public_agent_id));
    void dispatchReplayCommand({
      command_type: "set_pov_actor",
      presentation_key: activation.presentationKey,
    });
  } else if (activation.effect === "replay_select") {
    activatedAgentPublicId = String(activation.agent.public_agent_id);
    openAgentDetails();
    void dispatchReplayCommand({
      command_type: "select_agent",
      selected_global_slot: activation.commandSlot,
    });
  } else if (activation.effect === "live_control") {
    activatedAgentPublicId = String(activation.agent.public_agent_id);
    openAgentDetails();
    const controlCommand = {
      command_type: "roster_selection",
      role: "control",
      global_slot: activation.commandSlot,
    };
    if (pointerOriginated) {
      dispatchCommandFromDraftControl(controlCommand);
    } else {
      void dispatchCommand(controlCommand);
    }
  }
  return true;
}

/**
 * Pause replay for a presentation error while suppressing recursive state renders.
 * Always restore the render-callback flag, including when pause throws.
 *
 * @param {string} reason
 */
function pauseReplayAfterPresentationFailure(reason) {
  suppressPlaybackStateRender = true;
  try {
    replayPlayback.pause(reason);
  } finally {
    suppressPlaybackStateRender = false;
  }
}

/** @type {number | null} */
let systemPollTimer = null;

/** Show pending work and poll its explicit completion command without a new turn. */
function renderSystemOperation() {
  const button = document.getElementById("cancel-system-button");
  const status = document.getElementById("system-operation-status");
  const operation = isRecord(state.frame) ? state.frame.system_operation : null;
  if (button instanceof HTMLButtonElement) {
    button.hidden = !operation || isReplayMode();
    button.disabled = state.busy || !["loading", "thinking"].includes(operation?.state);
    button.onclick = () => {
      if (operation)
        void dispatchCommand({
          command_type: "cancel_system",
          operation_id: operation.operation_id,
        });
    };
  }
  if (status) {
    /** @type {Record<string, string>} */
    const labels = {
      loading: "Loading",
      thinking: "Choosing Actions",
      finishing: "Finishing",
      cancelling: "Cancelling",
      failed: "Cleanup Failed",
    };
    status.textContent = operation ? `System: ${labels[operation.state]}` : "";
  }
  if (systemPollTimer !== null) window.clearTimeout(systemPollTimer);
  systemPollTimer = null;
  if (
    operation &&
    operation.state !== "failed" &&
    !isReplayMode() &&
    !state.shuttingDown &&
    !state.offline &&
    !state.resyncRequired
  ) {
    systemPollTimer = window.setTimeout(() => {
      systemPollTimer = null;
      if (!state.busy)
        void dispatchCommand({
          command_type: "finish_system",
          operation_id: operation.operation_id,
        });
      else renderSystemOperation();
    }, 250);
  }
}

/**
 * Paint the complete page from one coherent installed transport/presentation pair.
 * Share a single visual-filter snapshot across scene and choreography, refresh
 * panels, metrics, controls, and tooltips, and preserve allowed focus/disclosure
 * choices. Presentation failures pause replay and clear choreography with a notice.
 */
function render() {
  renderSystemOperation();
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
  renderReplayMetrics();
  panels.render(presentationFrame, {
    busy: state.busy,
    shuttingDown: state.shuttingDown,
    resyncRequired: state.resyncRequired,
    offline: state.offline,
    activationDisabled:
      (transportFrame !== null && isTerminal(transportFrame) && !isReplayMode()) ||
      (!isReplayMode() && recordingScientificControlsFenced()),
    localInspectedPresentationKey:
      installedLocalInspectedPresentationKey(presentationFrame),
    activatedAgentPublicId,
  });
  tooltipController.refresh();
  restorePresentationPreferenceAfterRender();
}

/**
 * Return the rounded battlefield width and height as a CSS-pixel size key.
 * Resize handling uses this key to skip unchanged dimensions.
 */
function battlefieldSizeKey() {
  return `${Math.round(elements.battlefield.clientWidth)}x${Math.round(elements.battlefield.clientHeight)}`;
}

/**
 * Coalesce resize notifications into one animation-frame repaint.
 * Skip unchanged rounded dimensions, repaint the scene, and reproject existing
 * choreography without advancing simulation. Refresh artifact controls/tooltips
 * and preserve local preferences; presentation failures pause replay.
 */
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
 * Wrap a live command with schema version, client ID, new UUID, and current revision.
 * Return the request without sending it. Each call creates a fresh command identity.
 *
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
 * Describe the episode replacement that a recording-discard dialog would permit.
 * Recognize reset, scenario switch, and combat configuration; use a general
 * replacement label for other commands. This does not perform the replacement.
 *
 * @param {Readonly<Record<string, unknown>>} replacement
 */
function recordingReplacementLabel(replacement) {
  if (replacement.command_type === "reset") {
    return "Reset the current episode";
  }
  if (replacement.command_type === "scenario_switch") {
    return `Switch to scenario ${String(replacement.scenario_name)}`;
  }
  if (replacement.command_type === "set_combat_configuration") {
    const teamAController = combatControllerLabel(replacement.team_a_controller);
    const teamBController = combatControllerLabel(replacement.team_b_controller);
    const information =
      replacement.execution_information_mode === "shared_obs"
        ? "SharedObs"
        : "NoSharedObs";
    return `Restart with Team A ${teamAController}, Team B ${teamBController}, and ${information}`;
  }
  return "Replace the current episode";
}

/**
 * Store a proposed replacement and open its recording-discard confirmation dialog.
 * Refresh toolbar availability, describe the replacement, and focus Cancel.
 * No destructive command is sent until a later explicit confirmation.
 *
 * @param {Readonly<Record<string, unknown>>} replacement
 */
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
 * Send one replay command and install its matching authorized successor.
 * Reject unavailable, busy, or disconnected use. Never retry the POST: a stale
 * response or join race may install a fresh GET pair and return handled_resync.
 * Update connection, shutdown, and notice state; rethrow failures after fencing
 * authority as needed. deferFinalRender defaults to false for ordinary commands.
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
  /**
   * Resolve the replay-command completion barrier after its finally cleanup.
   * Start as a no-op, then replace it with the Promise resolver before the request;
   * view-switch handlers await this barrier before sending another command.
   */
  let finishCommand = () => {};
  replayCommandCompletion = new Promise((resolve) => {
    finishCommand = resolve;
  });
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
      /**
       * Post the captured replay request and remember its response.
       * Resolve to a success outcome, or a stale outcome for HTTP 409. Other API and
       * transport failures reject; the request is not retried inside this callback.
       */
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
      /**
       * Join outcome.payload with the current presentation and prepare installation.
       * The outcome comes from this transaction's send callback. Require replay
       * transport and validate successful command continuity. Missing joins, wrong
       * products, invalid outcomes and network failures reject the promise.
       */
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
      /**
       * Fetch a fresh matched frame/presentation for replay resynchronization.
       * Prepare it against previousAuthority with stale_resync continuity. The
       * promise rejects on fetch, join or preparation failure; no command is repeated.
       */
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
    finishCommand();
  }
}

/**
 * Send a replay command with its final render deferred to the playback controller.
 * Return the request promise so playback can validate its successor intent before
 * the newly installed frame receives a non-pending render. Failures propagate.
 *
 * @param {Readonly<Record<string, any>>} command
 */
function sendReplayTransportCommand(command) {
  return sendReplayCommand(command, { deferFinalRender: true });
}

/** @type {Promise<void>} */
let replayCommandCompletion = Promise.resolve();

/**
 * Run a user replay command through audience gates and playback coordination.
 * Pause first; view/recipient changes wait for the current request and may resume
 * previous playback if their generation is still current. Return the payload or
 * null for blocked, superseded, or handled failed requests; clear pending recipient
 * activation after settlement.
 *
 * @param {Readonly<Record<string, any>>} command
 */
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
  const stagedRecipientActivation =
    command.command_type === "set_pov_actor" ? pendingReplayRecipientActivation : null;
  const changingView = ["set_view", "set_pov_actor"].includes(command.command_type);
  const wasPlaying = replayPlayback.snapshot().playing;
  invalidateReplayArtifactAction();
  const suspended = replayPlayback.pause("user_command");
  try {
    if (changingView) {
      await replayCommandCompletion;
      if (replayPlayback.snapshot().generation !== suspended.generation) {
        return null;
      }
    }
    const payload = await sendReplayCommand(command);
    const frame = state.frame;
    const resume =
      changingView &&
      wasPlaying &&
      replayPlayback.snapshot().generation === suspended.generation;
    if (frame) {
      replayPlayback.installCursor(frame.cursor);
      if (resume && !replayPlayback.snapshot().atEnd) {
        replayPlayback.play({ restartCurrent: false });
      }
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
  } finally {
    if (pendingReplayRecipientActivation === stagedRecipientActivation) {
      pendingReplayRecipientActivation = null;
    }
  }
}

/**
 * Route a panel action to local activation, live command, or replay command.
 * Validate public identity/slot pairs against authorized identity rows. Pointer
 * activation defaults to false and controls focus handoff. Return a promise for
 * the routed result, or resolved null when the action is local or unavailable.
 *
 * @param {Record<string, unknown>} command
 * @param {Readonly<{pointerOriginated?: boolean}>} context
 */
function dispatchPanelCommand(command, { pointerOriginated = false } = {}) {
  if (
    command.command_type === "activate_authorized_agent" &&
    typeof command.presentation_key === "string"
  ) {
    activateAuthorizedAgent(command.presentation_key, { pointerOriginated });
    return Promise.resolve(null);
  }
  if (
    command.command_type === "activate_replay_pov_agent" &&
    Number.isInteger(command.global_slot) &&
    Number(command.global_slot) >= 0 &&
    Number(command.global_slot) < 10 &&
    typeof command.public_agent_id === "string" &&
    authorizedPresentationIdentityRows(state.presentation).some(
      (identity) =>
        identity.public_agent_id === command.public_agent_id &&
        identity.command_global_slot === command.global_slot,
    ) &&
    isReplayMode()
  ) {
    stageReplayRecipientActivation(command.public_agent_id);
    return dispatchReplayCommand({
      command_type: "set_pov_actor",
      global_slot: command.global_slot,
    });
  }
  if (
    command.command_type === "activate_live_pov_agent" &&
    Number.isInteger(command.global_slot) &&
    Number(command.global_slot) >= 0 &&
    Number(command.global_slot) < 10 &&
    typeof command.public_agent_id === "string" &&
    authorizedPresentationIdentityRows(state.presentation).some(
      (identity) =>
        identity.public_agent_id === command.public_agent_id &&
        identity.command_global_slot === command.global_slot,
    ) &&
    !isReplayMode() &&
    authorizedPresentationAudience(state.presentation) === "agent_pov"
  ) {
    activatedAgentPublicId = command.public_agent_id;
    openAgentDetails();
    const controlCommand = {
      command_type: "roster_selection",
      role: "control",
      global_slot: command.global_slot,
    };
    if (pointerOriginated) {
      dispatchCommandFromDraftControl(controlCommand);
      return Promise.resolve(null);
    }
    return dispatchCommand(controlCommand);
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
 * Serialize one live command through authority, policy, terminal, and recording gates.
 * Optional deferredSubmit defaults to null and may retain one fresh Enter after
 * draft preparation. Send the POST once, install a coherent successor, and only
 * then dispatch retained input. Show handled failures, require resync when needed,
 * and reload on accepted recording review; normal completion resolves without a value.
 *
 * @param {Record<string, unknown>} command
 * @param {{deferredSubmit?: Readonly<Record<string, unknown>> | null}} options
 */
async function dispatchCommand(command, { deferredSubmit = null } = {}) {
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
  const policyControlled = policyControlledActor();
  if (policyControllerBlocksActionEdit(command)) {
    const team = policyControlled?.team ?? "This team";
    const controller = combatControllerLabel(policyControlled?.controller);
    setNotice(
      `${team} is controlled by ${controller}. Its actions are read-only; Submit remains available.`,
      "warning",
    );
    renderConnection();
    renderCommandAvailability();
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

  const recordingLifecycle = recordingStatus()?.lifecycle;
  /** @type {{
   *   allowsDeferredSubmit: boolean,
   *   deferredDraft: Readonly<Record<string, unknown>> | null,
   *   deferredSubmit: Readonly<Record<string, unknown>> | null,
   *   releaseFocusAfterSettlement: boolean,
   *   recordingPreparation: "metrics" | "possible" | null,
   * }} */
  const liveCommandTransaction = {
    allowsDeferredSubmit: commandPreparesDeferredSubmit(command),
    deferredDraft: null,
    deferredSubmit:
      commandPreparesDeferredSubmit(command) &&
      deferredSubmit !== null &&
      isFreshUnmodifiedEnter(deferredSubmit)
        ? deferredSubmit
        : null,
    releaseFocusAfterSettlement: false,
    recordingPreparation:
      ["recording", "sealed"].includes(recordingLifecycle) &&
      ["finish_and_review", "exit"].includes(String(command.command_type))
        ? "metrics"
        : null,
  };
  activeLiveCommandTransaction = liveCommandTransaction;
  state.busy = true;
  state.offline = false;
  const recordingHintTimer =
    recordingLifecycle === "recording" && isSubmissionCommand(command)
      ? window.setTimeout(() => {
          if (activeLiveCommandTransaction === liveCommandTransaction && state.busy) {
            liveCommandTransaction.recordingPreparation = "possible";
            renderRecordingControls(installedPresentationAuthority());
          }
        }, 2000)
      : null;
  setNotice("Waiting for the authoritative Python response…", "info");
  let reviewHandoff = false;
  /** @type {Readonly<Record<string, unknown>> | null} */
  let deferredDraftToDispatch = null;
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
      /**
       * Post the captured live request and remember its response.
       * Resolve to success or, for HTTP 409, a stale outcome. Other API/transport
       * errors reject. The transaction's join/recovery callbacks own resynchronization.
       */
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
      /**
       * Prepare the supplied live command outcome for authority installation.
       * If it switches into recording review, mark the handoff and reject with
       * ProductReviewHandoff so the caller reloads that route. Otherwise fetch the
       * presentation and require a valid join; API/join/preparation errors reject.
       */
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
      /**
       * Fetch and prepare current authority after a stale live command.
       * Return a promise for a matched frame/presentation prepared against the previous
       * view. Fetch/join/preparation failures reject; this does not resubmit the action.
       */
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
      (liveCommandTransaction.deferredDraft !== null ||
        liveCommandTransaction.deferredSubmit !== null) &&
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
      deferredDraftToDispatch = liveCommandTransaction.deferredDraft;
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
    if (recordingHintTimer !== null) window.clearTimeout(recordingHintTimer);
    state.busy = false;
    if (activeLiveCommandTransaction === liveCommandTransaction) {
      activeLiveCommandTransaction = null;
    }
    if (!reviewHandoff || state.shuttingDown || state.resyncRequired) {
      render();
    }
    if (liveCommandTransaction.releaseFocusAfterSettlement) {
      releaseBattlefieldFocus();
    }
  }
  if (reviewHandoff && !state.shuttingDown && !state.resyncRequired) {
    reloadForProductHandoff();
    return;
  }
  if (deferredDraftToDispatch !== null) {
    await dispatchCommand(deferredDraftToDispatch, {
      deferredSubmit: deferredSubmitToDispatch,
    });
  } else if (deferredSubmitToDispatch !== null) {
    await dispatchCommand(deferredSubmitToDispatch);
  }
}

/**
 * Load and install the current matched transport, presentation, and replay timeline.
 * reviewHandoff defaults to false; when true, reload the route instead. Skip busy
 * or shutting-down pages, preserve permitted old display while loading, and show
 * handled failures with resync required. A successful first replay load focuses
 * the timeline; this never retries a simulator command.
 *
 * @param {{reviewHandoff?: boolean}} options
 */
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
      /**
       * Fetch current matched authority for the initial load or reconnect.
       * Prepare it with stale_resync continuity relative to the retained previous view.
       * The promise rejects if fetching, matching or preparation fails.
       */
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
 * Send a draft-control command after handing keyboard ownership to the battlefield.
 * restoreFocusTo defaults to null. When supplied, restore that connected enabled
 * control after completion only if the battlefield still owns focus; do not steal
 * focus the user moved elsewhere. The request runs asynchronously with no return value.
 *
 * @param {Record<string, unknown>} command
 * @param {{
 *   restoreFocusTo?: HTMLElement | null,
 * }} options
 */
function dispatchCommandFromDraftControl(command, { restoreFocusTo = null } = {}) {
  elements.battlefield.focus({ preventScroll: true });
  const completion = dispatchCommand(command);
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

/**
 * Move focus to the first enabled command button, or Help when none exists.
 * Use preventScroll so releasing battlefield keyboard control does not move the page.
 */
function releaseBattlefieldFocus() {
  const firstCommand = /** @type {HTMLButtonElement | null} */ (
    elements.commandDeck?.querySelector("button:not([disabled])") ?? null
  );
  const focusTarget = firstCommand ?? elements.helpButton;
  focusTarget.focus({ preventScroll: true });
}

bindBattlefieldControls({
  battlefield: elements.battlefield,
  /**
   * Convert the supplied SVG-local point into battlefield world coordinates.
   * Delegate the coordinate transform to the installed renderer. Return its point
   * or null when its current view cannot provide a valid transform.
   */
  toWorldPoint: (point) => battlefieldRenderer.toWorldPoint(point),
  /**
   * Dispatch the supplied normalized live battlefield command.
   * Return the shared dispatch promise. That handler owns action authorization,
   * request fencing, server calls and error notices.
   */
  onCommand: (command) => dispatchCommand(command),
  /**
   * Open the existing keyboard-help dialog modally without a server request.
   * Browser dialog errors propagate, including an incompatible nonmodal open state.
   */
  onHelp: () => elements.helpDialog.showModal(),
  isInteractive: liveBattlefieldCommandsInteractive,
  onFencedCommand: retainFencedCommand,
  /**
   * Return whether live control owns Space while a command is fenced.
   * Replay mode leaves its Space handling to the replay keyboard controller.
   */
  ownsFencedSpaceDefault: () => !isReplayMode(),
  onReleaseFocus: releaseBattlefieldFocus,
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
    if (event.button !== 0) {
      return;
    }
    const activation = authorizedAgentActivationFromTarget(event.target);
    const inspectionOnlyCorpse =
      activation?.effect === "local_inspection" &&
      activation.agent.life_state === "corpse";
    if (
      !inspectionOnlyCorpse &&
      (event.shiftKey || event.ctrlKey || event.metaKey || event.altKey)
    ) {
      return;
    }
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
  const command = targetSelectionCommand(elements.commandTargetSelect.value);
  if (!command) {
    return;
  }
  dispatchCommandFromDraftControl(command, {
    restoreFocusTo: pointerOriginated ? null : elements.commandTargetSelect,
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

/**
 * Validate the Save As field and send its recording command when valid.
 * On an invalid basename, show the filename rule and focus the input. The host
 * performs persistence; this handler does not write a browser-local replay file.
 */
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

if (!isReplayMode()) {
  document.addEventListener("marl-devclient-combat-configuration", (event) => {
    const command = requestedCombatConfigurationCommand(
      event instanceof CustomEvent ? event.detail : null,
    );
    if (command !== null) {
      void dispatchCommand(command);
    } else {
      publishInstalledCombatConfiguration(state.frame);
    }
  });
  document.addEventListener("marl-devclient-debug-session-replaced", () => {
    void loadCurrentFrame();
  });
}

elements.helpButton.addEventListener("click", () => {
  elements.helpDialog.showModal();
});

bindReplayTimelineControls(replayTimelineElements, replayPlayback);

/**
 * Apply a checkbox change from the visual-filter container to local filter state.
 * Ignore other event targets or inputs without a filter ID. The shared action
 * handler validates the ID, pauses replay on change, and repaints.
 *
 * @param {Event} event
 */
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
  applyAllVisualControls(true);
});

elements.disableAllVisualFiltersButton.addEventListener("click", () => {
  applyAllVisualControls(false);
});

elements.defaultVisualFiltersButton.addEventListener("click", () => {
  applyVisualFilterAction({ type: "restore_defaults" });
  setActiveRangesVisible(false);
});

elements.replayExportPngButton.addEventListener("click", () => {
  void exportReplayBattlefieldPng();
});

elements.replayDownloadMetricsButton.addEventListener("click", () => {
  void downloadReplayMetricReport();
});
elements.replayEpisodeDetailsButton.addEventListener("click", () => {
  void downloadReplayMetricReport(true);
});
for (const control of [
  elements.metricPanel,
  elements.metricScope,
  elements.metricSelection,
  elements.metricView,
]) {
  control.addEventListener(
    control === elements.metricPanel ? "toggle" : "change",
    () => {
      replayMetricFailure = null;
      replayMetricCatalogFailure = "";
      replayMetricFocus = null;
      elements.metricSearchDefinition.hidden = true;
      if (
        control === elements.metricScope &&
        replayArtifactActionTransaction?.kind === "download_metrics"
      ) {
        invalidateReplayArtifactAction();
        renderReplayArtifactActions(installedPresentationAuthority());
      }
      renderReplayMetrics();
    },
  );
}

elements.metricSearch.addEventListener("input", () => {
  replayMetricSearchLimit = 20;
  elements.metricSearchDefinition.hidden = true;
  renderReplayMetricSearch();
  setReplayMetricSearchOpen(true);
});
for (const event of ["focus", "click"]) {
  elements.metricSearch.addEventListener(event, () => setReplayMetricSearchOpen(true));
}
elements.metricSearchArea.addEventListener(
  "focusout",
  (/** @type {FocusEvent} */ event) => {
    if (
      event.relatedTarget !== null &&
      !elements.metricSearchArea.contains(event.relatedTarget)
    ) {
      setReplayMetricSearchOpen(false);
    }
  },
);
document.addEventListener(
  "pointerdown",
  (event) => {
    if (!elements.metricSearchArea.contains(event.target)) {
      setReplayMetricSearchOpen(false);
    }
  },
  true,
);
elements.metricSearchArea.addEventListener(
  "keydown",
  (/** @type {KeyboardEvent} */ event) => {
    if (event.isComposing) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      elements.metricSearch.focus({ preventScroll: true });
      setReplayMetricSearchOpen(false);
      return;
    }
    if (
      event.target !== elements.metricSearch ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey
    ) {
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const options = [...elements.metricSearchResults.querySelectorAll("button")];
      if (!options.length) return;
      event.preventDefault();
      event.stopPropagation();
      setReplayMetricSearchOpen(true);
      const index = options.indexOf(replayMetricSearchActive);
      const down = event.key === "ArrowDown";
      const next =
        index < 0
          ? down
            ? 0
            : options.length - 1
          : Math.max(0, Math.min(options.length - 1, index + (down ? 1 : -1)));
      const button = options[next];
      setActiveMetricSearchResult(button);
      button.scrollIntoView({ block: "nearest" });
    } else if (event.key === "Enter" && replayMetricSearchActive) {
      event.preventDefault();
      event.stopPropagation();
      replayMetricSearchActive.click();
    }
  },
);
elements.metricSearchMore.addEventListener("click", () => {
  replayMetricSearchLimit += 20;
  renderReplayMetricSearch();
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

/**
 * Clear the current replay reference through its permitted authority path.
 * Agent POV clears local inspection and closes details without disabling future
 * automatic opening; Oracle sends select_agent with null. Do nothing outside
 * replay or while the Clear control is disabled.
 */
function clearReplaySelection() {
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
}

elements.replayClearReferenceButton.addEventListener("click", clearReplaySelection);

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
        },
      );
    });
  }
}

const battlefieldResizeObserver = new ResizeObserver(scheduleBattlefieldResize);
battlefieldResizeObserver.observe(elements.battlefieldShell);
replayPlayback.setHidden(document.hidden);

for (const { panelId, panel, body } of scientificDisclosures) {
  /**
   * Block this disclosure summary click or keyboard activation while authority is pending.
   * Ignore events outside the summary. With a matching active preference, clear the
   * expected programmatic-toggle marker and allow the user action; otherwise prevent
   * default behavior and stop later listeners from opening unavailable content.
   *
   * @param {Event} event
   */
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
