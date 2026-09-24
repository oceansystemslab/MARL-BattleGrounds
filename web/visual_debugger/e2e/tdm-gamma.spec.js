/**
 * @file Check the Reactive TDM GAMMA controller (tdm_gamma) on Team B from
 * selection to saved replay in a real recording DevClient. The page starts with
 * Team A and Team B on Manual under SharedObs, and the Team A list also offers
 * GAMMA, enabled under SharedObs. Choosing GAMMA for Team B must install
 * exactly Team A Manual, Team B tdm_gamma and SharedObs, and lock NoSharedObs.
 * A selected Team B actor must show "Reactive TDM GAMMA" as its read-only
 * controller. One real Submit must record one transition, and Finish
 * (finish_and_review) must save the replay and hand the page to review with
 * Team B named "Reactive TDM GAMMA". The saved file must record every Team B
 * row as policy kind tdm_gamma, algorithm
 * reactive-team-deathmatch-gamma-controller, deterministic, with the GAMMA
 * pressure protocol digest. A separate Replay Viewer opened on that file must
 * name Team B "Reactive TDM GAMMA" with tdm_gamma policy IDs. The recording
 * debugger, which owns the file's folder, stops only after that viewer stops.
 * Waits have no time limit.
 */
import { expect, test } from "@playwright/test";

import {
  readJsonArtifact,
  startRecordingDebugger,
  stopRecordingDebugger,
} from "./support/recording-handoff.js";
import { startReplayViewer, stopDebugger } from "./support/replay-viewer.js";

/** @type {Awaited<ReturnType<typeof startRecordingDebugger>> | null} */
let recording = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let replayViewer = null;

test.afterEach(async () => {
  const viewer = replayViewer;
  const started = recording;
  replayViewer = null;
  recording = null;
  /** @type {unknown[]} */
  const cleanupErrors = [];
  try {
    // The Replay Viewer reads the saved file, so it stops before the recording
    // debugger removes that file's folder.
    await stopDebugger(viewer?.process ?? null);
  } catch (error) {
    cleanupErrors.push(error);
  }
  try {
    await stopRecordingDebugger(started);
  } catch (error) {
    cleanupErrors.push(error);
  }
  if (cleanupErrors.length > 0) {
    throw new AggregateError(cleanupErrors, "TDM-GAMMA E2E cleanup failed.");
  }
});

const GAMMA_LABEL = "Reactive TDM GAMMA";
const GAMMA_CONFIGURATION = Object.freeze({
  team_a_controller: "manual",
  team_b_controller: "tdm_gamma",
  execution_information_mode: "shared_obs",
});

/** @param {import("@playwright/test").Page} page */
function collectBrowserErrors(page) {
  /** @type {string[]} */
  const errors = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
  return errors;
}

/** @param {import("@playwright/test").Page} page
 * @param {"/api/frame" | "/api/presentation/frame"} path
 * @returns {Promise<Record<string, any>>} */
function authenticatedGet(page, path) {
  return page.evaluate(async (requestPath) => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Debugger capability token is unavailable.");
    }
    const response = await fetch(requestPath, {
      cache: "no-store",
      credentials: "omit",
      headers: { "X-MARL-Debugger-Token": token },
      redirect: "error",
    });
    if (!response.ok) {
      throw new Error(`${requestPath} failed with HTTP ${response.status}.`);
    }
    return response.json();
  }, path);
}

/** @param {import("@playwright/test").Page} page
 * @param {() => Promise<unknown>} activate */
async function applyLiveCommand(page, activate) {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await activate();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  return response;
}

/** @param {import("@playwright/test").Page} page */
async function expectGammaScoreboard(page) {
  const teamB = page.locator("#match-team-b .match-scoreboard__policy");
  await expect(teamB).toHaveText(GAMMA_LABEL);
  await expect(teamB).toHaveAttribute(
    "title",
    /^debugger-action-source:tdm_gamma:slot:\d+(?:\ndebugger-action-source:tdm_gamma:slot:\d+)*$/u,
  );
  await expect(page.locator("#match-team-a .match-scoreboard__policy")).toHaveText(
    "Manual",
  );
  const presentation = await authenticatedGet(page, "/api/presentation/frame");
  const [teamASummary, teamBSummary] = presentation.match_summary.teams;
  expect(teamASummary).toMatchObject({ team_id: 1, display_name: "Manual" });
  expect(teamBSummary).toMatchObject({ team_id: 2, display_name: GAMMA_LABEL });
  expect(teamBSummary.policy_ids.length).toBeGreaterThan(0);
  for (const policyId of teamBSummary.policy_ids) {
    expect(policyId).toMatch(/^debugger-action-source:tdm_gamma:slot:\d+$/u);
  }
}

/** @param {Record<string, any>} context */
function expectGammaContext(context) {
  const keys = Object.fromEntries(
    context.aggregation_keys.map((/** @type {{name: string, value: string}} */ row) => [
      row.name,
      row.value,
    ]),
  );
  expect(keys).toMatchObject({
    team_a_controller: "manual",
    team_b_controller: "tdm_gamma",
    information_regime: "shared_obs",
    pressure_protocol: "reactive-team-deathmatch-gamma-controller@2",
  });
  expect(keys.pressure_protocol_digest).toMatch(/^[0-9a-f]{64}$/u);
  const teamBySlot = new Map(
    context.roster.map(
      (/** @type {{global_slot: number, configured_team_id: number}} */ row) => [
        row.global_slot,
        row.configured_team_id,
      ],
    ),
  );
  const assigned = context.policy_assignments.filter(
    (/** @type {Record<string, any>} */ row) => typeof row.policy_kind === "string",
  );
  const teamBRows = assigned.filter(
    (/** @type {Record<string, any>} */ row) => teamBySlot.get(row.global_slot) === 2,
  );
  const teamARows = assigned.filter(
    (/** @type {Record<string, any>} */ row) => teamBySlot.get(row.global_slot) === 1,
  );
  expect(teamBRows.length).toBeGreaterThan(0);
  expect(teamARows.length).toBeGreaterThan(0);
  for (const row of teamBRows) {
    expect(row).toMatchObject({
      policy_kind: "tdm_gamma",
      policy_id: `debugger-action-source:tdm_gamma:slot:${row.global_slot}`,
      algorithm_id: "reactive-team-deathmatch-gamma-controller",
      execution_mode: "deterministic",
      policy_content_digest: keys.pressure_protocol_digest,
    });
  }
  for (const row of teamARows) {
    expect(row).toMatchObject({ policy_kind: "manual" });
  }
}

test("Team B GAMMA records one real step and reopens with its identity", async ({
  page,
  context,
}) => {
  const started = await startRecordingDebugger({ stem: "tdm-gamma" });
  recording = started;
  const errors = collectBrowserErrors(page);
  await page.goto(started.url);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-product-kind",
    "combat_debugger",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "recording",
  );
  await expect(page.locator("#devclient-combat-config")).toBeVisible();

  const teamA = page.locator("#devclient-team-a-controller");
  const teamB = page.locator("#devclient-team-b-controller");
  const information = page.locator("#devclient-information-mode");
  const gammaOption = page.locator("#devclient-tdm-gamma-controller-option");
  await expect(teamA).toHaveValue("manual");
  await expect(teamB).toHaveValue("manual");
  await expect(information).toHaveValue("shared_obs");
  await expect(gammaOption).toHaveText(GAMMA_LABEL);
  await expect(gammaOption).toHaveAttribute("value", "tdm_gamma");
  await expect(gammaOption).toHaveJSProperty("disabled", false);
  const teamAGammaOption = teamA.locator('option[value="tdm_gamma"]');
  await expect(teamAGammaOption).toHaveAttribute(
    "id",
    "devclient-team-a-tdm-gamma-option",
  );
  await expect(teamAGammaOption).toHaveText(GAMMA_LABEL);
  await expect(teamAGammaOption).toHaveJSProperty("disabled", false);
  await expect(page.locator("#devclient-no-shared-option")).toHaveJSProperty(
    "disabled",
    false,
  );

  const configurationResponse = await applyLiveCommand(page, () =>
    teamB.selectOption("tdm_gamma"),
  );
  expect(configurationResponse.request().postDataJSON().command).toEqual({
    command_type: "set_combat_configuration",
    ...GAMMA_CONFIGURATION,
  });
  await expect(teamB).toHaveValue("tdm_gamma");
  await expect(teamA).toHaveValue("manual");
  await expect(information).toHaveValue("shared_obs");
  await expect(page.locator("html")).toHaveAttribute(
    "data-team-b-controller",
    "tdm_gamma",
  );
  await expect(page.locator("#devclient-no-shared-option")).toHaveJSProperty(
    "disabled",
    true,
  );
  const installed = await authenticatedGet(page, "/api/frame");
  expect(installed.combat_configuration).toEqual(GAMMA_CONFIGURATION);
  expect(installed.recording).toMatchObject({
    lifecycle: "recording",
    captured_transition_count: 0,
  });
  await expect(page.locator("#step-value")).toHaveText("0");
  await expectGammaScoreboard(page);

  const priestB = page.locator('#roster .roster-row[data-team="team-b"]').filter({
    has: page.locator(".roster-class", { hasText: "Priest" }),
  });
  await applyLiveCommand(page, () => priestB.locator(".roster-primary-action").click());
  await expect(page.locator("#command-deck")).toHaveAttribute(
    "data-policy-controller-read-only",
    "true",
  );
  await expect(page.locator("#command-controlled-actor")).toContainText(
    `${GAMMA_LABEL} (read-only)`,
  );
  await expect(page.locator("#command-controlled-actor")).toHaveAttribute(
    "aria-label",
    new RegExp(`${GAMMA_LABEL} supplies this actor's action\\.$`, "u"),
  );
  await expect(page.locator("#command-target-select")).toBeDisabled();

  await expect(page.locator("#submit-turn-button")).toBeEnabled();
  await applyLiveCommand(page, () => page.locator("#submit-turn-button").click());
  await expect(page.locator("#step-value")).toHaveText("1");
  await expect(page.locator("#recording-progress")).toHaveText(
    /^1 \/ [1-9]\d* transitions$/u,
  );
  const stepped = await authenticatedGet(page, "/api/frame");
  expect(stepped.combat_configuration).toEqual(GAMMA_CONFIGURATION);
  expect(stepped.recording).toMatchObject({
    lifecycle: "recording",
    captured_transition_count: 1,
  });

  const finishResponse = await applyLiveCommand(page, () =>
    page.locator("#recording-finish-button").click(),
  );
  expect(finishResponse.request().postDataJSON().command).toMatchObject({
    command_type: "finish_and_review",
  });
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
  await expect(page.locator("html")).toHaveAttribute(
    "data-product-kind",
    "replay_viewer",
  );
  await expectGammaScoreboard(page);

  const saved = await readJsonArtifact(started.replayPath);
  expect(saved.value).toMatchObject({
    schema_id: "marl_battlegrounds.evaluation.replay_artifact",
    schema_version: 3,
    header: { recorded_transition_count: 1 },
  });
  expectGammaContext(saved.value.header.context);

  const viewer = await startReplayViewer({ replayPath: started.replayPath });
  replayViewer = viewer;
  const reviewPage = await context.newPage();
  const reviewErrors = collectBrowserErrors(reviewPage);
  await reviewPage.goto(viewer.url);
  await expect(reviewPage).toHaveTitle("MARL-BattleGrounds Replay Viewer");
  await expect(reviewPage.locator("#connection-status")).toHaveText("Online");
  await expect(reviewPage.locator("html")).toHaveAttribute(
    "data-product-kind",
    "replay_viewer",
  );
  await expectGammaScoreboard(reviewPage);
  expect(reviewErrors).toEqual([]);
  await reviewPage.close();
  expect(errors).toEqual([]);
});
