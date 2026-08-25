import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { joinTransportAndAuthorizedPresentationV1 } from "../src/authorized-presentation-normalizer.js";
import {
  pendingPresentationSurfaceView,
  resolveInstalledPresentationAuthorityV1,
} from "../src/presentation-authority-view.js";

const mainUrl = new URL("../src/main.js", import.meta.url);
const indexUrl = new URL("../index.html", import.meta.url);
const controlsUrl = new URL("../src/controls.js", import.meta.url);
const panelsUrl = new URL("../src/panels.js", import.meta.url);
const fixtureUrl = new URL(
  "./fixtures/authorized-presentations-v1.json",
  import.meta.url,
);

/**
 * @param {string} source
 * @param {string} earlier
 * @param {string} later
 */
function assertSourceOrder(source, earlier, later) {
  const earlierIndex = source.indexOf(earlier);
  const laterIndex = source.indexOf(later);
  assert.notEqual(earlierIndex, -1, `missing earlier source marker: ${earlier}`);
  assert.notEqual(laterIndex, -1, `missing later source marker: ${later}`);
  assert.ok(earlierIndex < laterIndex, `${earlier} must precede ${later}`);
}

/**
 * Import one dependency-free pure helper directly from the production source
 * without evaluating the browser module's DOM bootstrap.
 *
 * @param {string} source
 * @param {string} name
 * @param {string} nextName
 */
async function importPureMainHelper(source, name, nextName) {
  const start = source.indexOf(`function ${name}(`);
  const end = source.indexOf(`function ${nextName}(`, start);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  const moduleSource = `export ${source.slice(start, end)}`;
  return import(
    `data:text/javascript;charset=utf-8,${encodeURIComponent(moduleSource)}`
  );
}

test("main keeps raw transport and certified presentation authority separate", async () => {
  const source = await readFile(mainUrl, "utf8");

  assert.match(source, /frame: Record<string, any> \| null,/u);
  assert.match(source, /presentation: Readonly<Record<string, any>> \| null,/u);
  assert.match(source, /authority: Readonly<Record<string, any>> \| null,/u);
  assert.match(source, /const presentationFrame = state\.presentation;/u);
  assert.match(source, /battlefieldRenderer\.render\(presentationFrame,/u);
  assert.match(source, /choreographer\.presentFrame\(\s*presentationFrame,/u);
  assert.match(source, /panels\.render\(presentationFrame,/u);
  assert.doesNotMatch(source, /analysisPresentationFrame/u);
  assert.doesNotMatch(
    source,
    /(?:battlefieldRenderer\.render|choreographer\.presentFrame|panels\.render)\(\s*state\.frame/u,
  );
  assert.doesNotMatch(source, /state\.frame\?\.hud|state\.frame\.hud/u);
  assert.doesNotMatch(source, /frame\?\.replay_audience|frame\.replay_audience/u);
  assert.match(
    source,
    /const audience = authorizedPresentationAudience\(presentation\) \?\? "unavailable";/u,
  );
});

test("main has one branded state install boundary and no legacy half installs", async () => {
  const source = await readFile(mainUrl, "utf8");
  const frameAssignments = [
    ...source.matchAll(/state\.frame\s*=(?!=)\s*([^;]+);/gu),
  ].map((match) => match[1].trim());
  const timelineAssignments = [
    ...source.matchAll(/state\.timeline\s*=(?!=)\s*([^;]+);/gu),
  ].map((match) => match[1].trim());
  const presentationAssignments = [
    ...source.matchAll(/state\.presentation\s*=(?!=)\s*([^;]+);/gu),
  ].map((match) => match[1].trim());
  const authorityAssignments = [
    ...source.matchAll(/state\.authority\s*=(?!=)\s*([^;]+);/gu),
  ].map((match) => match[1].trim());

  assert.deepEqual(frameAssignments, ["joined.transport", "null"]);
  assert.deepEqual(timelineAssignments, [
    "isRecord(joined.timeline) ? joined.timeline : null",
    "null",
  ]);
  assert.deepEqual(presentationAssignments, ["null", "joined.presentation"]);
  assert.deepEqual(authorityAssignments, ["null", "joined"]);
  assert.match(source, /isJoinedTransportAndAuthorizedPresentationV1\(joined\)/u);
  assert.match(source, /isAuthorizedPresentationFrame\(joined\.presentation\)/u);
  assert.match(source, /validateReplayTransportContinuityV1\(/u);
  assert.doesNotMatch(source, /validateReplayFrameContinuity/u);
});

test("main derives product shell mode only from validated route identity", async () => {
  const source = await readFile(mainUrl, "utf8");
  const helperStart = source.indexOf("function isReplayMode()");
  const helperEnd = source.indexOf("/**", helperStart);

  assert.notEqual(helperStart, -1);
  assert.notEqual(helperEnd, -1);
  const helperSource = source.slice(helperStart, helperEnd);
  assert.match(helperSource, /productIdentity\?\.product_kind === "replay_viewer"/u);
  assert.doesNotMatch(helperSource, /state\.frame|viewer_mode/u);

  assert.match(source, /function assertFrameMatchesProductIdentity\(frame\)/u);
  assert.match(source, /const frameIsReplay = frame\.viewer_mode === "replay";/u);
  assert.match(
    source,
    /const identityIsReplay = productIdentity\.product_kind === "replay_viewer";/u,
  );
});

test("main applies explicit retain-or-clear policy before bounded installation", async () => {
  const source = await readFile(mainUrl, "utf8");
  const beginStart = source.indexOf("function beginPresentationAuthorityAttempt(");
  const beginEnd = source.indexOf("/**\n * Keep the two-column workspace", beginStart);

  assert.notEqual(beginStart, -1);
  assert.notEqual(beginEnd, -1);
  const beginSource = source.slice(beginStart, beginEnd);

  assert.match(
    source,
    /new PresentationInstallCoordinator\(\{[\s\S]*onAttemptBegin: beginPresentationAuthorityAttempt,[\s\S]*install: installJoinedAuthority,[\s\S]*isJoinRace: isPresentationJoinRace,/u,
  );
  assert.match(
    beginSource,
    /pendingPolicy === "retain_last_authorized" &&[\s\S]*installedPresentationAuthority\(\) !== null[\s\S]*presentationAuthority = "retained"[\s\S]*return;/u,
  );
  assert.match(beginSource, /clearPresentationAuthority\(reason\);/u);
  assert.doesNotMatch(
    beginSource,
    /state\.(?:authority|frame|presentation|timeline)\s*=/u,
  );
  assert.match(
    source,
    /command\.command_type === "set_view" \? "clear" : "retain_last_authorized"/u,
  );
  assert.match(
    source,
    /const retainLastAuthorized = installedPresentationAuthority\(\) !== null;[\s\S]*pendingPolicy: retainLastAuthorized \? "retain_last_authorized" : "clear"/u,
  );
  const replayCommandStart = source.indexOf("async function sendReplayCommand(");
  const replayCommandEnd = source.indexOf(
    "async function dispatchReplayCommand(",
    replayCommandStart,
  );
  const liveCommandStart = source.indexOf("async function dispatchCommand(");
  const liveCommandEnd = source.indexOf(
    "async function loadCurrentFrame(",
    liveCommandStart,
  );
  assert.notEqual(replayCommandStart, -1);
  assert.notEqual(replayCommandEnd, -1);
  assert.notEqual(liveCommandStart, -1);
  assert.notEqual(liveCommandEnd, -1);
  const replayCommandSource = source.slice(replayCommandStart, replayCommandEnd);
  const liveCommandSource = source.slice(liveCommandStart, liveCommandEnd);
  assert.match(
    replayCommandSource,
    /command\.command_type === "set_view"[\s\S]*\? "replay_audience_change"[\s\S]*: "replay_command"[\s\S]*pendingPolicy: "retain_last_authorized"/u,
  );
  assert.match(
    replayCommandSource,
    /const status = error instanceof DebuggerApiError \? error\.status : 0;[\s\S]*isPresentationJoinRace\(error\)[\s\S]*clearPresentationAuthority\("replay_presentation_identity_mismatch"\)[\s\S]*status === 401 \|\| status === 403[\s\S]*clearPresentationAuthority\("replay_authorization_failure"\)[\s\S]*commandResponseSchedulesShutdown\(request\.command, payload\)/u,
  );
  assert.match(
    liveCommandSource,
    /const status = error instanceof DebuggerApiError \? error\.status : 0;[\s\S]*isPresentationJoinRace\(error\)[\s\S]*clearPresentationAuthority\("live_presentation_identity_mismatch"\)[\s\S]*status === 401 \|\| status === 403[\s\S]*clearPresentationAuthority\("live_authorization_failure"\)[\s\S]*commandResponseSchedulesShutdown\(command, payload\)/u,
  );
  assert.match(source, /state\.presentation = null;/u);
  assert.match(source, /state\.authority = null;/u);
  assert.match(
    source,
    /battlefieldRenderer\.render\(null,\s*\{\s*offline: true,\s*visualFilterState,\s*\}\);/u,
  );
  assert.match(source, /panels\.render\(null,/u);
  assert.match(source, /tooltipController\.hide\(\);/u);
  assert.match(source, /choreographer\.clear\(reason\);/u);
  assert.match(
    source,
    /clearPresentationTooltipOwner\(elements\.replayArtifactReference\);/u,
  );
  assert.match(
    source,
    /clearPresentationTooltipOwner\(elements\.replayCompletionBadge\);/u,
  );
  assert.match(
    source,
    /clearPresentationTooltipOwner\(elements\.replayProcessingBadge\);/u,
  );
  assert.match(
    source,
    /elements\.commandTargetSelect\.replaceChildren\(emptyTarget\);/u,
  );
  assert.match(source, /elements\.commandTargetSelect\.disabled = true;/u);
  assert.match(
    source,
    /elements\.commandControlledActor\.removeAttribute\("aria-label"\);/u,
  );
  assert.match(
    source,
    /elements\.pendingCard\.removeAttribute\("data-inspection-state"\);/u,
  );
  assert.match(source, /pendingLabel\.textContent = "NO AUTHORIZED INSPECTION";/u);
  assert.match(
    source,
    /elements\.recordingReviewButton,[\s\S]*elements\.recordingSaveAsButton,/u,
  );
  assert.match(source, /enterPendingPresentationPreferenceState\(\);/u);
  assert.match(
    source,
    /elements\.battlefield\.removeAttribute\("aria-activedescendant"\);/u,
  );
  assert.match(
    source,
    /document\.documentElement\.dataset\.presentationAuthority = "pending";/u,
  );
  assert.equal(
    [...source.matchAll(/postCommand\(state\.token, request\)/gu)].length,
    1,
  );
  assert.equal(
    [...source.matchAll(/postReplayCommand\(state\.token, request\)/gu)].length,
    1,
  );
});

test("clear then pending render cannot republish retained Oracle or successor POV transport", async () => {
  const [source, fixtureText] = await Promise.all([
    readFile(mainUrl, "utf8"),
    readFile(fixtureUrl, "utf8"),
  ]);
  const fixture = JSON.parse(fixtureText);
  const oraclePair = fixture.continuity_pairs.oracle;
  const povPair = fixture.continuity_pairs.shared_obs;
  const oracle = await joinTransportAndAuthorizedPresentationV1(
    oraclePair.transport,
    oraclePair.presentation,
  );
  const pov = await joinTransportAndAuthorizedPresentationV1(
    povPair.transport,
    povPair.presentation,
  );
  /** @type {{
   *   authority: Readonly<Record<string, any>> | null,
   *   frame: Record<string, any> | null,
   *   presentation: Readonly<Record<string, any>> | null,
   * }} */
  const runtimeState = {
    authority: oracle,
    frame: oracle.transport,
    presentation: oracle.presentation,
  };

  assert.deepEqual(
    resolveInstalledPresentationAuthorityV1(
      runtimeState.authority,
      runtimeState.frame,
      runtimeState.presentation,
    ),
    { transport: oracle.transport, presentation: oracle.presentation },
  );

  runtimeState.authority = null;
  runtimeState.presentation = null;
  assert.equal(
    resolveInstalledPresentationAuthorityV1(
      runtimeState.authority,
      runtimeState.frame,
      runtimeState.presentation,
    ),
    null,
  );
  const pending = pendingPresentationSurfaceView({
    transportState: "PLAYING",
    generation: 41,
    presentationIntent: {
      generation: 17,
      renderPolicy: "replay_animated",
      restartAnimated: true,
    },
    connected: true,
    hidden: false,
    playbackRate: 1.75,
    cursor: {
      schema_version: 1,
      frame_index: 7,
      final_frame_index: 9,
      cursor_generation: 4,
      choreography_generation: 3,
    },
    playing: true,
    pauseReason: null,
    requestPending: false,
    presentationPending: false,
    atStart: false,
    atEnd: false,
  });
  assert.equal(pending.presentation, null);
  assert.equal(pending.transport, null);
  assert.deepEqual(
    [
      pending.replay.artifactReference,
      pending.replay.completion,
      pending.replay.processing,
      pending.replay.endReason,
      pending.recording.lifecycle,
      pending.recording.progress,
      pending.recording.completion,
      pending.recording.persistence,
    ],
    Array(8).fill("Unavailable while authority is pending"),
  );
  assert.deepEqual(pending.terminal, { hidden: true, text: "Terminal" });
  assert.equal(pending.viewMode, "");
  assert.equal(pending.replay.timeline.cursor, null);
  assert.equal(pending.replay.timeline.transportState, "OFFLINE");
  assert.equal(pending.replay.timeline.generation, 0);
  assert.equal(pending.replay.timeline.presentationIntent, null);
  assert.equal(pending.replay.timeline.connected, false);
  assert.equal(pending.replay.timeline.playbackRate, 1.75);
  assert.equal(pending.replay.timeline.playing, false);
  assert.equal(pending.replay.timeline.requestPending, false);
  assert.equal(pending.replay.timeline.presentationPending, false);

  runtimeState.frame = pov.transport;
  runtimeState.presentation = pov.presentation;
  assert.equal(
    resolveInstalledPresentationAuthorityV1(
      runtimeState.authority,
      runtimeState.frame,
      runtimeState.presentation,
    ),
    null,
  );
  runtimeState.authority = oracle;
  assert.equal(
    resolveInstalledPresentationAuthorityV1(
      runtimeState.authority,
      runtimeState.frame,
      runtimeState.presentation,
    ),
    null,
  );
  assert.equal(
    resolveInstalledPresentationAuthorityV1(pov, pov.transport, {
      ...pov.presentation,
    }),
    null,
  );

  const replayStart = source.indexOf("function renderReplayMetadata(installed)");
  const replayEnd = source.indexOf("const recordingPersistenceLabels", replayStart);
  const toolbarStart = source.indexOf("function renderSessionToolbar(");
  const toolbarEnd = source.indexOf("function setDraftSelection(", toolbarStart);
  const renderStart = source.indexOf("function render()");
  const renderEnd = source.indexOf("function battlefieldSizeKey()", renderStart);
  const boundaryStart = source.indexOf("function applyBattlefieldBoundaryCopy()");
  const boundaryEnd = source.indexOf("function renderViewerBoundary()", boundaryStart);
  for (const boundary of [
    replayStart,
    replayEnd,
    toolbarStart,
    toolbarEnd,
    renderStart,
    renderEnd,
  ]) {
    assert.notEqual(boundary, -1);
  }
  const replaySource = source.slice(replayStart, replayEnd);
  const toolbarSource = source.slice(toolbarStart, toolbarEnd);
  const renderSource = source.slice(renderStart, renderEnd);
  const boundarySource = source.slice(boundaryStart, boundaryEnd);
  assert.doesNotMatch(replaySource, /state\.frame|state\.presentation/u);
  assert.doesNotMatch(toolbarSource, /state\.frame|state\.presentation/u);
  assert.match(toolbarSource, /renderPendingPresentationChrome\(\);/u);
  assert.match(toolbarSource, /renderRecordingControls\(installed\);/u);
  assert.match(toolbarSource, /renderReplayMetadata\(installed\);/u);
  assert.match(renderSource, /const installed = installedPresentationAuthority\(\);/u);
  assert.match(
    renderSource,
    /const presentationFrame = installed\?\.presentation \?\? null;/u,
  );
  assert.match(renderSource, /panels\.render\(presentationFrame,/u);
  assert.match(
    boundarySource,
    /if \(installed === null\)[\s\S]*FENCED_LIVE_BATTLEFIELD_INSTRUCTIONS/u,
  );
  assertSourceOrder(boundarySource, "if (installed === null)", "else if (replay)");
  assert.match(
    source,
    /function recordingStatus\(\)[\s\S]*installedPresentationAuthority\(\)\?\.transport/u,
  );
  assert.match(
    source,
    /onStateChange: \(playback\) => \{[\s\S]*installedPresentationAuthority\(\) === null[\s\S]*pendingPresentationSurfaceView\(playback\)\.replay\.timeline/u,
  );
  assert.match(
    source,
    /tickForFrameIndex:[\s\S]*installedPresentationAuthority\(\) === null[\s\S]*replayTimelineSimulatorStep\(state\.timeline, frameIndex\)/u,
  );
});

test("main scopes one non-caching preference record to the certified authority tuple", async () => {
  const source = await readFile(mainUrl, "utf8");
  const controllerStart = source.indexOf("let activePresentationPreference = null;");
  const controllerEnd = source.indexOf("function isReplayMode()", controllerStart);
  const controller = source.slice(controllerStart, controllerEnd);
  const installStart = source.indexOf("function installJoinedAuthority(joined)");
  const installEnd = source.indexOf(
    "async function prepareJoinedAuthority(",
    installStart,
  );
  const install = source.slice(installStart, installEnd);

  assert.notEqual(controllerStart, -1);
  assert.notEqual(controllerEnd, -1);
  assert.notEqual(installStart, -1);
  assert.notEqual(installEnd, -1);
  assert.doesNotMatch(controller, /\bnew Map\s*\(/u);
  assert.match(
    install,
    /authorizedPresentationPreferenceKey\(joined\.presentation\)[\s\S]*installActivePresentationPreference\(joined\.presentation, preferenceKey\)/u,
  );
  assert.match(controller, /sameAuthorizedPresentationPreferenceKey\(/u);
  assert.match(
    controller,
    /activePresentationPreference = defaultPresentationPreference\(/u,
  );
  assert.doesNotMatch(source, /localPresentationPreference|disclosureAuthorityKey/u);
});

test("main saves before pending close and restores only after coherent content render", async () => {
  const source = await readFile(mainUrl, "utf8");
  const clearStart = source.indexOf("function clearPresentationAuthority(reason)");
  const clearEnd = source.indexOf(
    "function installJoinedAuthority(joined)",
    clearStart,
  );
  const clear = source.slice(clearStart, clearEnd);
  const pendingStart = source.indexOf(
    "function enterPendingPresentationPreferenceState()",
  );
  const pendingEnd = source.indexOf(
    "function capturePresentationPreferenceBeforeRender()",
    pendingStart,
  );
  const pending = source.slice(pendingStart, pendingEnd);
  const renderStart = source.indexOf("function render()");
  const renderEnd = source.indexOf(
    "function syncCompactActiveCombatPriority(",
    renderStart,
  );
  const render = source.slice(renderStart, renderEnd);

  assertSourceOrder(
    clear,
    "savePresentationPreferenceBeforeClear();",
    "enterPendingPresentationPreferenceState();",
  );
  assertSourceOrder(
    clear,
    "enterPendingPresentationPreferenceState();",
    "state.authority = null;",
  );
  assert.match(pending, /setScientificDisclosureAvailability\(false\);/u);
  assertSourceOrder(
    render,
    "battlefieldRenderer.render(presentationFrame,",
    "restorePresentationPreferenceAfterRender();",
  );
  assertSourceOrder(
    render,
    "panels.render(presentationFrame,",
    "restorePresentationPreferenceAfterRender();",
  );
  assertSourceOrder(
    render,
    "tooltipController.refresh();",
    "restorePresentationPreferenceAfterRender();",
  );
});

test("main distinguishes delayed programmatic details toggles from user intent", async () => {
  const source = await readFile(mainUrl, "utf8");
  const setterStart = source.indexOf("function setProgrammaticDisclosureOpen(");
  const setterEnd = source.indexOf(
    "function setScientificDisclosureAvailability(",
    setterStart,
  );
  const setter = source.slice(setterStart, setterEnd);
  const captureStart = source.indexOf("function capturePresentationPreference(");
  const captureEnd = source.indexOf(
    "function savePresentationPreferenceBeforeClear()",
    captureStart,
  );
  const capture = source.slice(captureStart, captureEnd);

  assert.match(source, /const expectedDisclosureToggles = new WeakMap\(\);/u);
  assertSourceOrder(
    setter,
    "expectedDisclosureToggles.set(panel",
    "panel.open = open;",
  );
  assert.match(
    source,
    /installedActivePresentationPreference\(state\.presentation\) !== null[\s\S]*expectedDisclosureToggles\.delete\(panel\);/u,
  );
  assert.match(capture, /const userClosedBeforeToggle =/u);
  assert.match(capture, /expected\?\.open !== false/u);
  assert.match(capture, /preference\.agentDetailsAutoOpenAllowed = false;/u);
  assert.match(
    capture,
    /panel\.open \|\| userClosedBeforeToggle[\s\S]*body\.scrollTop/u,
  );
  assert.equal(
    [...source.matchAll(/schedulePresentationPreferenceRestore\(preference\)/gu)]
      .length,
    2,
  );
  assert.match(
    source,
    /if \(panel\.open\) \{\s*body\.scrollTop = preference\.disclosures\[panelId\]\.scrollTop;/u,
  );
  assert.doesNotMatch(source, /elements\.agentDetails\.open\s*=/u);
});

test("main shares the Agent Details latch and rejects stale local keys and focus", async () => {
  const source = await readFile(mainUrl, "utf8");
  const showStart = source.indexOf("function showSemanticInspector(");
  const showEnd = source.indexOf("function registerControlHelp()", showStart);
  const show = source.slice(showStart, showEnd);
  const openStart = source.indexOf("function openAgentDetails()");
  const openEnd = source.indexOf(
    "function closeAgentDetailsWithoutLatching()",
    openStart,
  );
  const open = source.slice(openStart, openEnd);
  const reconcileStart = source.indexOf("function reconcilePresentationPreference(");
  const reconcileEnd = source.indexOf(
    "function installActivePresentationPreference(",
    reconcileStart,
  );
  const reconcile = source.slice(reconcileStart, reconcileEnd);
  const activationStart = source.indexOf("function activateAuthorizedAgent(");
  const activationEnd = source.indexOf("function render()", activationStart);
  const activation = source.slice(activationStart, activationEnd);
  const restoreStart = source.indexOf(
    "function schedulePresentationPreferenceRestore(",
  );
  const restoreEnd = source.indexOf(
    "function restorePresentationPreferenceAfterRender()",
    restoreStart,
  );
  const restore = source.slice(restoreStart, restoreEnd);

  assert.match(show, /openAgentDetails\(\);/u);
  assert.match(activation, /openAgentDetails\(\);/u);
  assert.match(open, /!preference\.agentDetailsAutoOpenAllowed/u);
  assert.match(reconcile, /preference\.localInspection\.presentationKey = null;/u);
  assert.match(reconcile, /preference\.disclosures\["agent-details"\]\.open = false;/u);
  assert.match(reconcile, /preference\.primaryFocus = null;/u);
  assert.doesNotMatch(reconcile, /recipient/u);
  assert.match(
    restore,
    /generation !== presentationPreferenceGeneration[\s\S]*sameAuthorizedPresentationPreferenceKey\(installed\.authorityKey, authorityKey\)/u,
  );
  assert.match(restore, /CSS\.escape\(focus\.presentationKey\)/u);
  assert.match(restore, /focusWasNotMovedByUser/u);
});

test("main excludes the global Visual Key from scientific preference and inert state", async () => {
  const source = await readFile(mainUrl, "utf8");
  const idsStart = source.indexOf("const SCIENTIFIC_DISCLOSURE_BODY_IDS");
  const idsEnd = source.indexOf("const scientificDisclosures", idsStart);
  const scientificIds = source.slice(idsStart, idsEnd);

  assert.doesNotMatch(scientificIds, /visual-key/u);
  assert.match(source, /elements\.visualKey\.addEventListener\("toggle"/u);
});

test("main rejects battlefield pointer commands unless Oracle authority is installed", async () => {
  const source = await readFile(mainUrl, "utf8");
  const dispatchStart = source.indexOf("async function dispatchCommand(");
  const dispatchEnd = source.indexOf(
    "/** @param {{reviewHandoff?: boolean}} options */",
    dispatchStart,
  );

  assert.notEqual(dispatchStart, -1);
  assert.notEqual(dispatchEnd, -1);
  const dispatchSource = source.slice(dispatchStart, dispatchEnd);
  assert.match(
    dispatchSource,
    /command\.command_type === "battlefield_pointer"[\s\S]*authorizedPresentationAudience\(state\.presentation\) !== "researcher"[\s\S]*return;/u,
  );
  assertSourceOrder(
    dispatchSource,
    'command.command_type === "battlefield_pointer"',
    "const mode = modeAvailability(command, state.frame)",
  );
});

test("main resolves one certified activation to Oracle, live-local, or Replay POV behavior", async () => {
  const source = await readFile(mainUrl, "utf8");
  const resolverStart = source.indexOf("function authorizedAgentActivation(");
  const resolverEnd = source.indexOf(
    "function authorizedAgentActivationFromTarget(",
    resolverStart,
  );
  const panelStart = source.indexOf("function dispatchPanelCommand(");
  const panelEnd = source.indexOf("async function dispatchCommand(", panelStart);

  assert.notEqual(resolverStart, -1);
  assert.notEqual(resolverEnd, -1);
  assert.notEqual(panelStart, -1);
  assert.notEqual(panelEnd, -1);
  const resolver = source.slice(resolverStart, resolverEnd);
  const panel = source.slice(panelStart, panelEnd);
  assert.match(resolver, /installedAuthorityIsCoherent\(\)/u);
  assert.match(resolver, /authorizedAgentForPresentationKey\(/u);
  assert.match(
    resolver,
    /state\.busy[\s\S]*state\.resyncRequired[\s\S]*state\.offline/u,
  );
  assert.match(resolver, /isTerminal\(state\.frame\)/u);
  assert.match(resolver, /isTerminal\(state\.frame\) && !isReplayMode\(\)/u);
  assert.match(
    resolver,
    /const replayPovSwitch = isReplayMode\(\) && audience === "agent_pov"/u,
  );
  assert.match(resolver, /effect: "replay_pov_switch"/u);
  assert.match(resolver, /effect: "local_inspection"/u);
  assert.match(resolver, /effect: "replay_select"/u);
  assert.match(resolver, /effect: "live_control"/u);
  assert.match(
    panel,
    /command\.command_type === "activate_authorized_agent"[\s\S]*activateAuthorizedAgent\(command\.presentation_key\)/u,
  );
  assert.match(
    panel,
    /command\.command_type === "activate_replay_pov_agent"[\s\S]*command_type: "set_pov_actor"[\s\S]*global_slot: command\.global_slot/u,
  );
  assert.match(
    source,
    /effect === "replay_pov_switch"[\s\S]*command_type: "set_pov_actor"[\s\S]*presentation_key: activation\.presentationKey/u,
  );
  assert.match(
    source,
    /setLocalInspectedPresentationKey\(presentationKey\)[\s\S]*render\(\);/u,
  );
  assert.match(
    source,
    /"pointerdown"[\s\S]*authorizedAgentActivationFromTarget\(event\.target\)[\s\S]*stopImmediatePropagation\(\)[\s\S]*true,/u,
  );
  assert.match(
    source,
    /elements\.battlefield\.focus\(\{ preventScroll: true \}\);[\s\S]*activateAuthorizedAgent\(activation\.presentationKey\);/u,
  );
  assert.doesNotMatch(source, /activation\.element\.focus\(/u);
  assert.match(
    source,
    /event\.key !== "Enter" && event\.key !== " "[\s\S]*activateAuthorizedAgent\(activation\.presentationKey\)/u,
  );
});

test("command legality requires one exact authorized owner and has no generic fallback", async () => {
  const source = await readFile(mainUrl, "utf8");
  const renderStart = source.indexOf("function renderDraftState(");
  const renderEnd = source.indexOf("function renderCommandAvailability()", renderStart);
  const clearStart = source.indexOf("function clearOwnerBoundLegalityAvailability(");
  const clearEnd = source.indexOf("function renderCommandTargets(", clearStart);
  assert.notEqual(renderStart, -1);
  assert.notEqual(renderEnd, -1);
  assert.notEqual(clearStart, -1);
  assert.notEqual(clearEnd, -1);
  const renderDraft = source.slice(renderStart, renderEnd);
  const clearOwner = source.slice(clearStart, clearEnd);

  assert.match(renderDraft, /authorizedAgentForPresentationKey\(/u);
  assert.match(
    renderDraft,
    /controlledCandidate\.public_agent_id === inspection\.actor_public_agent_id/u,
  );
  assert.match(
    renderDraft,
    /owner_presentation_key: controlledOwner\.presentation_key[\s\S]*owner_public_agent_id: controlledOwner\.public_agent_id/u,
  );
  assert.match(
    renderDraft,
    /basicAvailable = targetAction > 0 && asArray\(pairMask\)\[0\] === true/u,
  );
  assert.match(
    renderDraft,
    /lane_0_available: asArray\(pairMask\)\[0\] === true[\s\S]*basic_available: basicAvailable/u,
  );
  assert.match(renderDraft, /explainLegality\(legality, 0, controlledOwner\)/u);
  assert.match(renderDraft, /explainLegality\(legality, 1, controlledOwner\)/u);
  assert.match(
    renderDraft,
    /basicLegality === null[\s\S]*clearOwnerBoundLegalityAvailability\(elements\.basicButton\)/u,
  );
  assert.match(
    renderDraft,
    /ultimateLegality === null[\s\S]*clearOwnerBoundLegalityAvailability\(elements\.ultimateButton\)/u,
  );
  assert.doesNotMatch(renderDraft, /explainLegality\(\{\s*lane_[01]_available/u);
  assert.match(clearOwner, /clearPresentationTooltipOwner\(button\)/u);
  assert.doesNotMatch(clearOwner, /registerTooltipOwner|setAuthoritativeAvailability/u);
});

test("only certified activation installs agent documentation while compact owners stay help-only", async () => {
  const [source, panelsSource] = await Promise.all([
    readFile(mainUrl, "utf8"),
    readFile(panelsUrl, "utf8"),
  ]);
  const fenceStart = source.indexOf("function isAuthorizedCompactAgentOwner(");
  const fenceEnd = source.indexOf("function registerControlHelp()", fenceStart);
  const fence = source.slice(fenceStart, fenceEnd);
  const activationStart = source.indexOf("function activateAuthorizedAgent(");
  const activationEnd = source.indexOf("function render()", activationStart);
  const activation = source.slice(activationStart, activationEnd);
  const renderStart = source.indexOf("function render()", activationEnd);
  const renderEnd = source.indexOf(
    "/**\n * On the supported minimum battlefield",
    renderStart,
  );
  const render = source.slice(renderStart, renderEnd);
  const rosterStart = panelsSource.indexOf("renderAuthorizedRoster(");
  const rosterEnd = panelsSource.indexOf(
    "/**\n   * @param {Record<string, any>} presentation\n   * @param {string | null | undefined} [localInspectedPresentationKey]",
    rosterStart,
  );
  const roster = panelsSource.slice(rosterStart, rosterEnd);

  assert.match(fence, /isAuthorizedPresentationFrame\(presentation\)/u);
  assert.match(fence, /installedAuthorityIsCoherent\(\)/u);
  assert.match(fence, /authorizedAgentForPresentationKey\(/u);
  assert.match(
    fence,
    /normalized\.kind === "agent"[\s\S]*isAuthorizedCompactAgentOwner\(context\.owner\)[\s\S]*return;/u,
  );
  assertSourceOrder(
    fence,
    "isAuthorizedCompactAgentOwner(context.owner)",
    "renderSemanticInspector(",
  );
  assertSourceOrder(activation, "setLocalInspectedPresentationKey", "render();");
  assert.match(render, /panels\.render\(presentationFrame,/u);
  assert.match(render, /localInspectedPresentationKey:/u);
  assert.match(
    roster,
    /registerTooltipOwner\(\s*row\.primaryButton,[\s\S]*?\{ inspectable: false \},\s*\);/u,
  );
});

test("battlefield delegation leaves nested scientific owners and terminal frames inert", async () => {
  const [mainSource, controlsSource] = await Promise.all([
    readFile(mainUrl, "utf8"),
    readFile(controlsUrl, "utf8"),
  ]);
  const installStart = mainSource.indexOf(
    "function installAuthorizedAgentActivation()",
  );
  const installEnd = mainSource.indexOf(
    "/**\n * Apply the already-resolved single effect.",
    installStart,
  );
  const installSource = mainSource.slice(installStart, installEnd);

  assert.match(
    controlsSource,
    /if \(event\.target !== battlefield \|\| !isDebuggerKey\(event\)\) \{\s*return;/u,
  );
  assert.match(
    controlsSource,
    /if \(!isInteractive\(\)\) \{[\s\S]*?if \(event\.key === " "\) \{[\s\S]*?event\.preventDefault\(\);[\s\S]*?return;/u,
  );
  assert.match(
    mainSource,
    /function liveBattlefieldCommandsInteractive\(\)[\s\S]*!isTerminal\(state\.frame\)/u,
  );
  assert.match(mainSource, /isInteractive: liveBattlefieldCommandsInteractive/u);
  assert.match(
    mainSource,
    /const tooltipOwner = event\.target\.closest\("\[data-tooltip-owner\]"\);[\s\S]*tooltipOwner !== agent[\s\S]*stopImmediatePropagation\(\)/u,
  );
  assert.notEqual(installStart, -1);
  assert.notEqual(installEnd, -1);
  assert.match(installSource, /\{ inspectable: false \}/u);
});

test("main derives replay animation only from the current controller intent", async () => {
  const source = await readFile(mainUrl, "utf8");
  const helperStart = source.indexOf("function installedChoreographyControl(");
  const helperEnd = source.indexOf(
    "function installedPresentationAuthority()",
    helperStart,
  );
  const resizeStart = source.indexOf("function scheduleBattlefieldResize()");
  const resizeEnd = source.indexOf("function commandRequest(", resizeStart);

  assert.notEqual(helperStart, -1);
  assert.notEqual(helperEnd, -1);
  assert.notEqual(resizeStart, -1);
  assert.notEqual(resizeEnd, -1);
  const helperSource = source.slice(helperStart, helperEnd);
  const resizeSource = source.slice(resizeStart, resizeEnd);
  assert.match(
    helperSource,
    /isJoinedTransportAndAuthorizedPresentationV1\(authority\)/u,
  );
  assert.match(helperSource, /authority\.presentation === presentation/u);
  assert.match(helperSource, /authority\.transport === state\.frame/u);
  assert.match(
    helperSource,
    /authority\.transport\.viewer_mode === "replay"[\s\S]*replayPlayback\.snapshot\(\)[\s\S]*replayCursorsMatch\(playback\.cursor, authority\.transport\.cursor\)/u,
  );
  assert.match(helperSource, /playback\.transportState !== "OFFLINE"/u);
  assert.match(
    helperSource,
    /intent\.generation > consumedReplayRestartGeneration[\s\S]*consumedReplayRestartGeneration = intent\.generation/u,
  );
  assert.doesNotMatch(helperSource, /animate_incoming/u);
  assert.match(
    source,
    /const choreographyControl = installedChoreographyControl\(\s*presentationFrame,\s*visualFilterSnapshot,\s*\{ consumeAnimatedRestart: true \},?\s*\);[\s\S]*choreographer\.presentFrame\(\s*presentationFrame,\s*battlefieldRenderer\.choreographySurface\(\),\s*choreographyControl,?\s*\)/u,
  );
  assert.equal(
    [...source.matchAll(/const choreographyControl = installedChoreographyControl\(/gu)]
      .length,
    2,
  );
  assert.match(
    source,
    /choreographer\.reproject\(\s*presentationFrame,\s*battlefieldRenderer\.choreographySurface\(\),\s*choreographyControl,?\s*\)/u,
  );
  assert.doesNotMatch(resizeSource, /consumeAnimatedRestart/u);
  assert.match(
    source,
    /incomingTransitionForFrameIndex:[\s\S]*installed\.transport\.cursor\?\.frame_index !== frameIndex[\s\S]*return authorizedIncomingTransitionId\(installed\.presentation\);/u,
  );
  assert.doesNotMatch(
    source.slice(
      source.indexOf("incomingTransitionForFrameIndex:"),
      source.indexOf(
        "const replayPlayback",
        source.indexOf("incomingTransitionForFrameIndex:"),
      ),
    ),
    /previousTick|currentTick|\u2192/u,
  );
});

test("replay artifact capabilities remain available across visual POVs", async () => {
  const source = await readFile(mainUrl, "utf8");
  const module = await importPureMainHelper(
    source,
    "replayArtifactActionCapabilities",
    "replayBattlefieldReady",
  );
  const capabilities = module.replayArtifactActionCapabilities;
  const base = {
    replayProduct: true,
    coherentAuthority: true,
    audience: "researcher",
    transportState: "SETTLED",
    connected: true,
    hidden: false,
    playing: false,
    requestPending: false,
    presentationPending: false,
    renderPolicy: "replay_static",
    cursorMatches: true,
    operationallyBlocked: false,
    actionPending: false,
    battlefieldReady: true,
  };

  assert.deepEqual(
    { ...capabilities(base) },
    {
      exportPng: true,
      downloadMetrics: true,
    },
  );
  const oracleMissing = {
    ...base,
    metricReportAvailability: "missing",
  };
  assert.equal(capabilities(oracleMissing).downloadMetrics, true);
  assert.deepEqual(
    { ...capabilities({ ...base, audience: "agent_pov" }) },
    { exportPng: true, downloadMetrics: true },
  );
  for (const transportState of ["OFFLINE", "PLAYING", "ADVANCING"]) {
    assert.deepEqual(
      { ...capabilities({ ...base, transportState }) },
      { exportPng: false, downloadMetrics: false },
    );
  }
  for (const override of [
    { replayProduct: false },
    { coherentAuthority: false },
    { audience: null },
    { connected: false },
    { hidden: true },
    { playing: true },
    { requestPending: true },
    { presentationPending: true },
    { renderPolicy: "replay_animated" },
    { cursorMatches: false },
    { operationallyBlocked: true },
    { actionPending: true },
  ]) {
    assert.deepEqual(
      { ...capabilities({ ...base, ...override }) },
      { exportPng: false, downloadMetrics: false },
    );
  }
  assert.deepEqual(
    { ...capabilities({ ...base, battlefieldReady: false }) },
    { exportPng: false, downloadMetrics: true },
  );
  const helperStart = source.indexOf("function replayArtifactActionCapabilities(");
  const helperEnd = source.indexOf("function replayBattlefieldReady()", helperStart);
  const helperSource = source.slice(helperStart, helperEnd);
  assert.doesNotMatch(helperSource, /sidecar|metric.*(?:present|missing|available)/iu);
});

test("replay artifact actions snapshot once, fence every await, and never drive transport", async () => {
  const source = await readFile(mainUrl, "utf8");
  const exportStart = source.indexOf("async function exportReplayBattlefieldPng()");
  const exportEnd = source.indexOf(
    "async function downloadReplayMetricReport()",
    exportStart,
  );
  const metricEnd = source.indexOf(
    "/**\n * Recognize only the installed branded scripted-live pair.",
    exportEnd,
  );
  const metricErrorStart = source.indexOf("function replayMetricDownloadError(");
  const metricErrorEnd = exportStart;
  const clearStart = source.indexOf("function clearPresentationAuthority(reason)");
  const clearEnd = source.indexOf(
    "function holdWorkspaceHeightDuringAuthorityInstall()",
    clearStart,
  );
  const filterStart = source.indexOf("function applyVisualFilterAction(action)");
  const filterEnd = source.indexOf(
    "function installedChoreographyControl(",
    filterStart,
  );
  const localStart = source.indexOf("function setLocalInspectedPresentationKey(");
  const localEnd = source.indexOf(
    "function installedPresentationRangesVisible(",
    localStart,
  );
  for (const boundary of [
    exportStart,
    exportEnd,
    metricEnd,
    clearStart,
    clearEnd,
    filterStart,
    filterEnd,
    localStart,
    localEnd,
    metricErrorStart,
    metricErrorEnd,
  ]) {
    assert.notEqual(boundary, -1);
  }
  const exportSource = source.slice(exportStart, exportEnd);
  const metricSource = source.slice(exportEnd, metricEnd);
  const metricErrorSource = source.slice(metricErrorStart, metricErrorEnd);
  const clearSource = source.slice(clearStart, clearEnd);
  const filterSource = source.slice(filterStart, filterEnd);
  const localSource = source.slice(localStart, localEnd);

  assert.match(
    source,
    /currentReplayArtifactActionCapabilities\(installed\)[\s\S]*isJoinedTransportAndAuthorizedPresentationV1\(authority\)[\s\S]*authority\.transport !== installed\.transport[\s\S]*authority\.presentation !== installed\.presentation/u,
  );
  assert.match(
    exportSource,
    /await captureReplayBattlefieldPngV1\(\{[\s\S]*battlefield: elements\.battlefield,[\s\S]*installedAuthority: transaction\.authority,[\s\S]*isCurrent: \(\) => replayArtifactActionIsCurrent\(transaction\),[\s\S]*localInspectedPresentationKey: transaction\.localInspectedPresentationKey,[\s\S]*visualFilters: transaction\.visualFilters,/u,
  );
  assertSourceOrder(
    exportSource,
    "await captureReplayBattlefieldPngV1({",
    "if (!replayArtifactActionIsCurrent(transaction))",
  );
  assertSourceOrder(
    metricSource,
    "await getReplayMetricReport(state.token)",
    "if (!replayArtifactActionIsCurrent(transaction))",
  );
  assertSourceOrder(
    metricSource,
    "await getReplayMetricReport(state.token)",
    "downloadReplayArtifact(",
  );
  assert.equal(
    [...metricSource.matchAll(/getReplayMetricReport\(state\.token\)/gu)].length,
    1,
  );
  for (const actionSource of [exportSource, metricSource]) {
    assert.doesNotMatch(
      actionSource,
      /postReplayCommand|getCurrentFrame|getCurrentPresentation|getReplayTimeline|sendReplayCommand|dispatchReplayCommand|replayPlayback\.(?:play|pause|seek|next|previous|first|last)|\brender\(\)/u,
    );
  }
  assert.match(
    metricSource,
    /new Blob\(\[report\.bytes\], \{ type: "application\/json; charset=utf-8" \}\)/u,
  );
  const metricCatchStart = metricSource.indexOf("} catch (error) {");
  const metricFinallyStart = metricSource.indexOf("} finally {", metricCatchStart);
  assert.notEqual(metricCatchStart, -1);
  assert.notEqual(metricFinallyStart, -1);
  assert.doesNotMatch(
    metricSource.slice(metricCatchStart, metricFinallyStart),
    /downloadReplayArtifact\(/u,
  );
  assert.match(
    metricErrorSource,
    /error instanceof DebuggerApiError && error\.status === 404[\s\S]*No metric report is available for this replay\.[\s\S]*level: "warning"/u,
  );
  assert.match(clearSource, /invalidateReplayArtifactAction\(\);/u);
  assert.match(
    filterSource,
    /invalidateReplayArtifactAction\(\);[\s\S]*visualFilterState = next;/u,
  );
  assert.equal(
    [...localSource.matchAll(/invalidateReplayArtifactAction\(\);/gu)].length,
    2,
  );
  assert.match(
    source,
    /onStateChange: \(playback\) => \{[\s\S]*playback\.generation !== replayArtifactActionTransaction\.playbackGeneration[\s\S]*invalidateReplayArtifactAction\(\);/u,
  );
  assert.match(
    source,
    /replayArtifactActionIsCurrent\(transaction\)[\s\S]*state\.authority !== transaction\.authority[\s\S]*installed\.transport !== transaction\.transport[\s\S]*installed\.presentation !== transaction\.presentation/u,
  );
  assert.match(
    source,
    /function reconcilePresentationPreference\([\s\S]*invalidateReplayArtifactAction\(\);[\s\S]*preference\.localInspection\.presentationKey = null;/u,
  );
  assert.match(
    source,
    /async function sendReplayCommand\([\s\S]*invalidateReplayArtifactAction\(\);[\s\S]*state\.busy = true;/u,
  );
  assert.match(
    source,
    /async function dispatchReplayCommand\([\s\S]*invalidateReplayArtifactAction\(\);[\s\S]*replayPlayback\.pause\("user_command"\);/u,
  );
  assert.match(
    source,
    /async function loadCurrentFrame\([\s\S]*invalidateReplayArtifactAction\(\);[\s\S]*replayPlayback\.pause\("reconnect"\);/u,
  );
  assert.match(
    source,
    /elements\.exitButton\.addEventListener\("click", \(\) => \{\s*invalidateReplayArtifactAction\(\);[\s\S]*dispatchReplayCommand\(\{ command_type: "exit" \}\)/u,
  );
  assert.match(
    source,
    /elements\.reconnectButton\.addEventListener\("click", \(\) => \{\s*invalidateReplayArtifactAction\(\);[\s\S]*loadCurrentFrame\(\)/u,
  );
});

test("playback controller owns the first installed-successor render", async () => {
  const source = await readFile(mainUrl, "utf8");
  const controllerStart = source.indexOf(
    "const replayPlayback = new ReplayPlaybackController(",
  );
  const controllerEnd = source.indexOf(
    "const panels = new DebuggerPanels(",
    controllerStart,
  );
  const sendStart = source.indexOf("async function sendReplayCommand(");
  const sendEnd = source.indexOf("async function dispatchReplayCommand(", sendStart);
  const dispatchEnd = source.indexOf("function dispatchPanelCommand(", sendEnd);

  for (const boundary of [
    controllerStart,
    controllerEnd,
    sendStart,
    sendEnd,
    dispatchEnd,
  ]) {
    assert.notEqual(boundary, -1);
  }
  const controllerSource = source.slice(controllerStart, controllerEnd);
  const sendSource = source.slice(sendStart, sendEnd);
  const dispatchSource = source.slice(sendEnd, dispatchEnd);

  assert.match(controllerSource, /request: sendReplayTransportCommand/u);
  assert.match(
    controllerSource,
    /!state\.busy[\s\S]*!suppressPlaybackStateRender[\s\S]*playback\.transportState !== "ADVANCING"[\s\S]*render\(\);/u,
  );
  assert.match(
    source,
    /keyboardEnabled: \(\) => installedPresentationAuthority\(\) !== null && !state\.busy/u,
  );
  assert.match(
    source,
    /function replayTimelineRenderState\(playback\)[\s\S]*!state\.busy[\s\S]*REPLAY_TRANSPORT_STATES\.OFFLINE[\s\S]*transportState: REPLAY_TRANSPORT_STATES\.ADVANCING/u,
  );
  assert.match(
    sendSource,
    /function sendReplayTransportCommand\(command\) \{\s*return sendReplayCommand\(command, \{ deferFinalRender: true \}\);/u,
  );
  assert.match(
    sendSource,
    /async function sendReplayCommand\(command, \{ deferFinalRender = false \} = \{\}\)/u,
  );
  assert.match(
    sendSource,
    /finally \{\s*state\.busy = false;\s*if \(!deferFinalRender \|\| state\.resyncRequired \|\| state\.shuttingDown\) \{\s*render\(\);/u,
  );
  assert.match(dispatchSource, /const payload = await sendReplayCommand\(command\);/u);
  assert.doesNotMatch(dispatchSource, /deferFinalRender/u);
  assert.equal([...source.matchAll(/deferFinalRender: true/gu)].length, 1);
  assert.match(
    source,
    /function pauseReplayAfterPresentationFailure\(reason\)[\s\S]*suppressPlaybackStateRender = true;[\s\S]*replayPlayback\.pause\(reason\);[\s\S]*suppressPlaybackStateRender = false;/u,
  );
  assertSourceOrder(
    source,
    'pauseReplayAfterPresentationFailure("presentation_error")',
    'choreographer.clear("presentation_error")',
  );
  assertSourceOrder(
    source,
    'pauseReplayAfterPresentationFailure("resize_projection_error")',
    'choreographer.clear("resize_projection_error")',
  );
});

test("main renders ranges and inspector chrome only from installed presentation authority", async () => {
  const source = await readFile(mainUrl, "utf8");
  const visibilityStart = source.indexOf(
    "function installedPresentationRangesVisible(",
  );
  const visibilityEnd = source.indexOf(
    "function liveScriptedInspectionOnly()",
    visibilityStart,
  );
  const inspectorStart = source.indexOf("function applyAuthorizedInspectorChrome()");
  const inspectorEnd = source.indexOf(
    "function reloadForProductHandoff()",
    inspectorStart,
  );
  const replayReferenceStart = source.indexOf(
    "function applyReplayReferenceSemantics()",
  );
  const replayReferenceEnd = source.indexOf(
    "function authorizedAgentActivation(",
    replayReferenceStart,
  );

  assert.notEqual(visibilityStart, -1);
  assert.notEqual(visibilityEnd, -1);
  assert.notEqual(inspectorStart, -1);
  assert.notEqual(inspectorEnd, -1);
  assert.notEqual(replayReferenceStart, -1);
  assert.notEqual(replayReferenceEnd, -1);
  const visibilitySource = source.slice(visibilityStart, visibilityEnd);
  const inspectorSource = source.slice(inspectorStart, inspectorEnd);
  const replayReferenceSource = source.slice(replayReferenceStart, replayReferenceEnd);
  assert.match(visibilitySource, /isAuthorizedPresentationFrame\(presentation\)/u);
  assert.match(visibilitySource, /installedAuthorityIsCoherent\(\)/u);
  assert.match(visibilitySource, /state\.presentation !== presentation/u);
  assert.match(
    visibilitySource,
    /authorizedPresentationAudience\(presentation\) === "researcher"[\s\S]*state\.frame\?\.show_ranges === true[\s\S]*localPreference\?\.rangesVisible === true/u,
  );
  assert.equal(
    [
      ...source.matchAll(
        /showRanges: installedPresentationRangesVisible\(presentationFrame\)/gu,
      ),
    ].length,
    2,
  );
  assert.match(
    inspectorSource,
    /authorizedInspectorView\(\s*state\.presentation,\s*installedLocalInspectedPresentationKey\(state\.presentation\),\s*\)/u,
  );
  assert.match(inspectorSource, /AUTHORIZED_INSPECTOR_TITLE/u);
  assert.match(inspectorSource, /inspector\.owner_class_accent/u);
  assert.match(
    source,
    /renderSemanticInspector\(elements\.selectionCard, normalized\);\s*applyAuthorizedInspectorChrome\(\);/u,
  );
  assert.doesNotMatch(source, /selectionHeading\.textContent = "Agent Details"/u);
  assert.match(replayReferenceSource, /if \(!isReplayMode\(\)\) \{\s*return;/u);
  assert.equal(
    [...source.matchAll(/installAuthorizedAgentActivation\(\);/gu)].length,
    2,
  );
  assert.equal([...source.matchAll(/applyReplayReferenceSemantics\(\);/gu)].length, 2);
});

test("main selects incoming transition IDs from the exact presentation variant", async () => {
  const source = await readFile(mainUrl, "utf8");
  const helperStart = source.indexOf("function authorizedIncomingTransitionId(");
  const helperEnd = source.indexOf("function recordingStatus()", helperStart);

  assert.notEqual(helperStart, -1);
  assert.notEqual(helperEnd, -1);
  const helperSource = source.slice(helperStart, helperEnd);
  assert.match(helperSource, /!isRecord\(presentation\.latest_events\)/u);
  assert.match(
    helperSource,
    /authorizedPresentationAudience\(presentation\) === "researcher"\s*\? presentation\.latest_events\.incoming_transition_id\s*:\s*presentation\.latest_events\.incoming_recipient_transition_id/u,
  );
  assert.doesNotMatch(helperSource, /state\.frame|transport/u);
  assert.equal(
    [...source.matchAll(/authorizedIncomingTransitionId\(presentation\)/gu)].length,
    2,
  );
  assert.doesNotMatch(source, /requiredElement\("revision-value"\)/u);
  assert.doesNotMatch(source, /requiredElement\("replay-incoming-value"\)/u);
  assert.doesNotMatch(source, /replayIncomingValue|revisionValue/u);
  assert.match(
    source,
    /registerTooltipOwner\(\s*elements\.replayCompletionBadge,\s*explainTechnicalFact\("completion"\)/u,
  );
  assert.match(
    source,
    /registerTooltipOwner\(\s*elements\.replayProcessingBadge,\s*explainTechnicalFact\("processing"\)/u,
  );
});

test("main describes Agent bodies as passive while preserving researcher guidance", async () => {
  const source = await readFile(mainUrl, "utf8");
  const boundaryStart = source.indexOf("function applyBattlefieldBoundaryCopy()");
  const boundaryEnd = source.indexOf(
    "function renderReplayMetadata(installed)",
    boundaryStart,
  );

  assert.notEqual(boundaryStart, -1);
  assert.notEqual(boundaryEnd, -1);
  const boundarySource = source.slice(boundaryStart, boundaryEnd);
  assert.match(boundarySource, /audience === "agent_pov"/u);
  assert.match(
    boundarySource,
    /Live Agent POV keeps one fixed recipient\. Bodies are passive inspection targets; use the authorized draft controls to prepare that recipient's action\./u,
  );
  assert.match(
    boundarySource,
    /Live Oracle View is interactive\. Activate an authorized actor to control it; Shift-click selects an authorized target; Escape clears the target and leaves battlefield focus\. Battlefield keyboard commands apply only while this surface has focus\./u,
  );
  assert.match(
    boundarySource,
    /Replay is read-only\. Upcoming Transition shows the authorized recorded joint action out of this frame; activate an agent to inspect current facts, or use the timeline to change frames\./u,
  );
  assert.match(
    boundarySource,
    /Replay Agent POV is read-only\. Activate a visible body or choose any agent in the roster to switch to that agent's fog-of-war view at the same replay tick\./u,
  );
  assert.match(
    source,
    /Scripted live view is inspection-only\. Activate an authorized body to inspect current facts; use Advance scripted frame for the next authorized step\./u,
  );
  assert.match(
    source,
    /function liveAgentCloseoutInspectionActionable\(\)[\s\S]*authorizedPresentationAudience\(state\.presentation\) === "agent_pov"[\s\S]*recordingScientificControlsFenced\(\)/u,
  );
  assert.match(
    source,
    /function installedLiveScientificInspectionAvailable\(\)[\s\S]*isAuthorizedPresentationFrame\(state\.presentation\)[\s\S]*installedAuthorityIsCoherent\(\)/u,
  );
  assert.match(
    source,
    /READ_ONLY_LIVE_SCIENTIFIC_LABEL[\s\S]*Scientific facts can be inspected/u,
  );
  assert.match(
    source,
    /This replay frame is terminal\. Activate a visible body or choose any agent in the roster to switch the fog-of-war recipient; use the timeline to review another frame\./u,
  );
});

test("main leaves the battlefield to visible instructions rather than a tooltip owner", async () => {
  const source = await readFile(mainUrl, "utf8");
  const helpStart = source.indexOf("const CONTROL_HELP = Object.freeze([");
  const helpEnd = source.indexOf("]);", helpStart);
  const utilityStart = source.indexOf("function registerAuthorityAwareUtilityHelp()");
  const utilityEnd = source.indexOf("function isRecord(value)", utilityStart);

  assert.notEqual(helpStart, -1);
  assert.notEqual(helpEnd, -1);
  assert.notEqual(utilityStart, -1);
  assert.notEqual(utilityEnd, -1);
  const helpSource = source.slice(helpStart, helpEnd);
  const utilitySource = source.slice(utilityStart, utilityEnd);
  assert.doesNotMatch(helpSource, /#battlefield/u);
  assert.doesNotMatch(helpSource, /Battlefield Commands/u);
  assert.match(utilitySource, /#live-ranges-button/u);
  assert.match(utilitySource, /#replay-ranges-button/u);
  assert.match(utilitySource, /#replay-clear-reference-button/u);
  assert.match(utilitySource, /sends no (?:replay )?command/u);
  assert.match(utilitySource, /Oracle View range presentation/u);
});

test("main reuses only Submit for coherent scripted-live advancement", async () => {
  const [source, index] = await Promise.all([
    readFile(mainUrl, "utf8"),
    readFile(indexUrl, "utf8"),
  ]);
  const helperStart = source.indexOf("function installedAuthorityIsCoherent()");
  const helperEnd = source.indexOf(
    "function authorizedIncomingTransitionId(",
    helperStart,
  );
  const availabilityStart = source.indexOf("function renderCommandAvailability()");
  const availabilityEnd = source.indexOf(
    "function exposePresentationState(",
    availabilityStart,
  );
  const dispatchStart = source.indexOf("async function dispatchCommand(");
  const dispatchEnd = source.indexOf(
    "/** @param {{reviewHandoff?: boolean}} options */",
    dispatchStart,
  );

  assert.notEqual(helperStart, -1);
  assert.notEqual(helperEnd, -1);
  assert.notEqual(availabilityStart, -1);
  assert.notEqual(availabilityEnd, -1);
  assert.notEqual(dispatchStart, -1);
  assert.notEqual(dispatchEnd, -1);
  const helperSource = source.slice(helperStart, helperEnd);
  const availabilitySource = source.slice(availabilityStart, availabilityEnd);
  const dispatchSource = source.slice(dispatchStart, dispatchEnd);
  assert.match(
    source,
    /function installedPresentationAuthority\(\)[\s\S]*resolveInstalledPresentationAuthorityV1\([\s\S]*state\.authority,[\s\S]*state\.frame,[\s\S]*state\.presentation,/u,
  );
  assert.match(helperSource, /return installedPresentationAuthority\(\) !== null;/u);
  assert.match(
    helperSource,
    /installedAuthorityIsCoherent\(\)[\s\S]*state\.frame\?\.frame_kind === "researcher_live_debugger"[\s\S]*state\.frame\?\.frame_kind === "actor_pov_live_debugger"[\s\S]*authorizedPresentationInspectionState\(state\.presentation\)\.state_kind ===\s*"live_scripted"/u,
  );
  assert.match(helperSource, /command\.command_type === "set_view"/u);
  assert.match(
    source,
    /const SCRIPTED_INSPECTION_RECORDING_COMMANDS = new Set\(\[\s*"finish_and_review",\s*"review_replay",\s*"retry_save",\s*"save_as",\s*"confirm_discard_and_replace",\s*"exit",\s*\]\);/u,
  );
  assert.match(
    helperSource,
    /SCRIPTED_INSPECTION_RECORDING_COMMANDS\.has\(String\(command\.command_type\)\)[\s\S]*recordingCommandDecision\(state\.frame \?\? \{\}, command\)\.action === "allow"/u,
  );
  assert.match(
    helperSource,
    /command\.command_type !== "keyboard" \|\|\s*\(command\.key !== "n" && command\.key !== "g"\)/u,
  );
  assert.match(
    helperSource,
    /command\.shift_key === false &&[\s\S]*command\.ctrl_key === false &&[\s\S]*command\.alt_key === false &&[\s\S]*command\.meta_key === false &&[\s\S]*command\.repeat === false/u,
  );
  assert.match(
    availabilitySource,
    /const scriptedAdvance = liveScriptedInspectionOnly\(\);/u,
  );
  assert.match(availabilitySource, /!installedAuthorityIsCoherent\(\)/u);
  assert.match(
    availabilitySource,
    /elements\.submitTurnButton\.dataset\.key = scriptedAdvance \? "n" : "Enter";/u,
  );
  assert.match(
    availabilitySource,
    /scriptedAdvance\s*\? "Advance scripted frame"[\s\S]*\? "Submit controlled actor"\s*:\s*"Submit joint turn"/u,
  );
  assert.match(
    availabilitySource,
    /editableDraft \|\| \(scriptedAdvance && button === elements\.submitTurnButton\)/u,
  );
  assert.match(
    availabilitySource,
    /scientificFenced \|\|[\s\S]*!enabledByInspection \|\|[\s\S]*\(scriptedAdvance && isTerminal\(state\.frame\)\)/u,
  );
  assert.match(
    availabilitySource,
    /scriptedAdvance\s*\? "Advance the registered script"\s*:\s*"Submit the staged joint turn"/u,
  );
  assert.match(
    availabilitySource,
    /scriptedAdvance\s*\? "One authoritative scripted transition"\s*:\s*"One authoritative transition"/u,
  );
  assert.doesNotMatch(source, /advance-script-button/u);
  assert.match(
    source,
    /"Apply authorized action",\s*"Submit an editable draft or advance an inspection-only scripted frame through the authoritative Python service\."/u,
  );
  assert.match(
    source,
    /authorizedPresentationInspectionState\(state\.presentation\)\.state_kind ===\s*"live_scripted"/u,
  );
  assert.match(
    source,
    /Inspection-only scripted battlefield\. Authorized bodies can be inspected; use Advance scripted frame for the next authorized step\./u,
  );
  assert.match(
    source,
    /Scripted live view is inspection-only\. Activate an authorized body to inspect current facts; use Advance scripted frame for the next authorized step\./u,
  );
  assert.match(
    source,
    /elements\.battlefield\.setAttribute\("role", "group"\);[\s\S]*elements\.battlefield\.tabIndex = -1;/u,
  );
  assert.equal(
    [...source.matchAll(/\n\s*applyBattlefieldBoundaryCopy\(\);/gu)].length,
    3,
  );
  assert.match(source, /shuttingDown:\s*state\.shuttingDown,/u);
  assert.match(
    source,
    /activationDisabled:[\s\S]*isTerminal\(transportFrame\)[\s\S]*!isReplayMode\(\)[\s\S]*recordingScientificControlsFenced\(\)/u,
  );
  assert.match(source, /isInteractive: liveBattlefieldCommandsInteractive/u);
  assert.match(
    source,
    /function liveBattlefieldCommandsInteractive\(\)[\s\S]*installedAuthorityIsCoherent\(\)[\s\S]*!isTerminal\(state\.frame\)[\s\S]*!state\.shuttingDown/u,
  );
  assert.match(
    source,
    /FENCED_LIVE_BATTLEFIELD_LABEL[\s\S]*Read-only live battlefield/u,
  );
  assert.match(
    dispatchSource,
    /liveScriptedInspectionOnly\(\) &&\s*!allowedDuringLiveScriptedInspection\(command\)[\s\S]*Use Advance scripted frame[\s\S]*return;/u,
  );
  assertSourceOrder(
    dispatchSource,
    "!allowedDuringLiveScriptedInspection(command)",
    "const mode = modeAvailability(command, state.frame)",
  );
  assert.match(
    source,
    /elements\.resetButton\.disabled =\s*disabled \|\| restartControlsBlocked \|\| liveScriptedInspectionOnly\(\);/u,
  );
  assert.match(
    index,
    /<strong id="command-commit-title">Submit the staged joint turn<\/strong>/u,
  );
  assert.match(
    index,
    /<small id="command-commit-summary">One authoritative transition<\/small>/u,
  );
});
