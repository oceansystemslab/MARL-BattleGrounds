import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const mainUrl = new URL("../src/main.js", import.meta.url);
const indexUrl = new URL("../index.html", import.meta.url);

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
  const frameAssignments = [...source.matchAll(/state\.frame\s*=\s*([^;]+);/gu)].map(
    (match) => match[1].trim(),
  );
  const timelineAssignments = [
    ...source.matchAll(/state\.timeline\s*=\s*([^;]+);/gu),
  ].map((match) => match[1].trim());
  const presentationAssignments = [
    ...source.matchAll(/state\.presentation\s*=\s*([^;]+);/gu),
  ].map((match) => match[1].trim());
  const authorityAssignments = [
    ...source.matchAll(/state\.authority\s*=\s*([^;]+);/gu),
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

test("main clears presentation before requests and delegates bounded retry policy", async () => {
  const source = await readFile(mainUrl, "utf8");

  assert.match(
    source,
    /new PresentationInstallCoordinator\(\{[\s\S]*clear: clearPresentationAuthority,[\s\S]*install: installJoinedAuthority,[\s\S]*isJoinRace: isPresentationJoinRace,/u,
  );
  assert.match(source, /state\.presentation = null;/u);
  assert.match(source, /state\.authority = null;/u);
  assert.match(source, /battlefieldRenderer\.render\(null, \{ offline: true \}\);/u);
  assert.match(source, /panels\.render\(null,/u);
  assert.match(source, /tooltipController\.hide\(\);/u);
  assert.match(source, /choreographer\.clear\(reason\);/u);
  assert.match(
    source,
    /clearPresentationTooltipOwner\(elements\.replayArtifactReference\);/u,
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
  assert.match(source, /elements\.agentDetails\.open = false;/u);
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

test("main rejects battlefield pointer commands unless Oracle authority is installed", async () => {
  const source = await readFile(mainUrl, "utf8");
  const dispatchStart = source.indexOf("async function dispatchCommand(command)");
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
  assert.ok(
    dispatchSource.indexOf('command.command_type === "battlefield_pointer"') <
      dispatchSource.indexOf("const mode = modeAvailability(command, state.frame)"),
  );
});

test("main carries replay animation intent only beside the installed presentation", async () => {
  const source = await readFile(mainUrl, "utf8");
  const helperStart = source.indexOf("function installedChoreographyControl(");
  const helperEnd = source.indexOf(
    "function authorizedIncomingTransitionId(",
    helperStart,
  );

  assert.notEqual(helperStart, -1);
  assert.notEqual(helperEnd, -1);
  const helperSource = source.slice(helperStart, helperEnd);
  assert.match(
    helperSource,
    /isJoinedTransportAndAuthorizedPresentationV1\(authority\)/u,
  );
  assert.match(helperSource, /authority\.presentation === presentation/u);
  assert.match(helperSource, /authority\.transport === state\.frame/u);
  assert.match(
    helperSource,
    /authority\.transport\.viewer_mode === "replay"[\s\S]*authority\.transport\.animate_incoming === true/u,
  );
  assert.doesNotMatch(helperSource, /presentation\.animate_incoming/u);
  assert.match(
    source,
    /choreographer\.presentFrame\(\s*presentationFrame,\s*battlefieldRenderer\.choreographySurface\(\),\s*installedChoreographyControl\(presentationFrame\),\s*\)/u,
  );
  assert.equal(
    [
      ...source.matchAll(
        /choreographer\.reproject\([\s\S]*?installedChoreographyControl\(/gu,
      ),
    ].length,
    2,
  );
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
    3,
  );
});

test("main describes Agent bodies as passive while preserving researcher guidance", async () => {
  const source = await readFile(mainUrl, "utf8");
  const boundaryStart = source.indexOf("function renderViewerBoundary()");
  const boundaryEnd = source.indexOf("function renderReplayMetadata()", boundaryStart);

  assert.notEqual(boundaryStart, -1);
  assert.notEqual(boundaryEnd, -1);
  const boundarySource = source.slice(boundaryStart, boundaryEnd);
  assert.match(boundarySource, /audience === "agent_pov"/u);
  assert.match(
    boundarySource,
    /Agent POV bodies are inspectable and passive; they cannot change control or submit actions\./u,
  );
  assert.match(
    boundarySource,
    /Left click controls an authorized actor, Shift plus left click selects an authorized target, and right click clears the target\./u,
  );
  assert.match(
    source,
    /Researcher live view supports pointer control and targeting; Agent POV bodies remain passive and draft controls own actions\./u,
  );
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
  const dispatchStart = source.indexOf("async function dispatchCommand(command)");
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
    helperSource,
    /isJoinedTransportAndAuthorizedPresentationV1\(authority\)[\s\S]*authority\.transport === state\.frame[\s\S]*authority\.presentation === state\.presentation/u,
  );
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
    /Inspection-only scripted battlefield\. Use Advance scripted frame for the next authorized step\./u,
  );
  assert.match(
    source,
    /Scripted live view is inspection-only\. Battlefield bodies are passive and cannot submit actions\. Use the single Advance scripted frame button/u,
  );
  assert.match(
    source,
    /elements\.battlefield\.setAttribute\("role", "img"\);[\s\S]*elements\.battlefield\.tabIndex = -1;/u,
  );
  assert.equal(
    [...source.matchAll(/\n\s*applyScriptedBattlefieldBoundaryCopy\(\);/gu)].length,
    3,
  );
  assert.match(
    source,
    /shuttingDown:\s*state\.shuttingDown \|\|\s*recordingScientificControlsFenced\(\) \|\|\s*liveScriptedInspectionOnly\(\)/u,
  );
  assert.match(
    source,
    /isInteractive: \(\) =>[\s\S]*!isReplayMode\(\)[\s\S]*!liveScriptedInspectionOnly\(\)/u,
  );
  assert.match(
    dispatchSource,
    /liveScriptedInspectionOnly\(\) &&\s*!allowedDuringLiveScriptedInspection\(command\)[\s\S]*Use Advance scripted frame[\s\S]*return;/u,
  );
  assert.ok(
    dispatchSource.indexOf("!allowedDuringLiveScriptedInspection(command)") <
      dispatchSource.indexOf("const mode = modeAvailability(command, state.frame)"),
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
