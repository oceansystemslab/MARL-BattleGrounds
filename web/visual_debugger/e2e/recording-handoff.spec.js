import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { expect, test } from "@playwright/test";
import { finishControllerClock } from "./support/choreography.js";
import {
  createReplayTargetRace,
  metricReportPathForReplay,
  readJsonArtifact,
  startRecordingDebugger,
  stopRecordingDebugger,
  waitForRecordingDebuggerExit,
} from "./support/recording-handoff.js";
import { expectVisibleInteractiveHelpInventory } from "./support/visual-regression.js";

test.describe.configure({ mode: "serial" });

/** @type {Awaited<ReturnType<typeof startRecordingDebugger>> | null} */
let recording = null;

/** @type {WeakMap<import("@playwright/test").Page, string[]>} */
const browserErrors = new WeakMap();

const RECORDING_STATUS_KEYS = Object.freeze([
  "captured_transition_count",
  "completion_reason",
  "completion_state",
  "discard_available",
  "expected_transition_count",
  "finish_available",
  "lifecycle",
  "persistence_error_code",
  "restart_fenced",
  "retry_available",
  "review_available",
  "save_as_available",
  "schema_version",
]);

test.beforeEach(async () => {
  recording = await startRecordingDebugger();
});

test.afterEach(async () => {
  const started = recording;
  recording = null;
  await stopRecordingDebugger(started);
});

function requiredRecording() {
  if (!recording) {
    throw new Error("The recording debugger did not start.");
  }
  return recording;
}

/** @param {import("@playwright/test").Page} page */
function collectBrowserErrors(page) {
  if (browserErrors.has(page)) {
    return;
  }
  /** @type {string[]} */
  const errors = [];
  browserErrors.set(page, errors);
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 */
async function openRecording(page, url) {
  collectBrowserErrors(page);
  await page.goto(url);
  await expect(page.locator("#connection-status")).toHaveText("Online", {
    timeout: 30_000,
  });
  await expect(page).toHaveTitle("MARL-BattleGrounds Combat Debugger");
  await expect(page.locator("html")).toHaveAttribute(
    "data-product-kind",
    "combat_debugger",
  );
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "recording",
  );
  await expect(page.locator("#recording-panel")).toBeVisible();
  await expect(page.locator("#recording-progress")).toContainText("0 /");
  expect(page.url()).not.toContain("token=");
}

/** @param {import("@playwright/test").Page} page */
function expectNoBrowserErrors(page) {
  expect(browserErrors.get(page) ?? []).toEqual([]);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} path
 */
async function authenticatedGet(page, path) {
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

/** @param {import("@playwright/test").Page} page */
function currentFrame(page) {
  return authenticatedGet(page, "/api/frame");
}

/** @param {import("@playwright/test").Page} page */
function currentTimeline(page) {
  return authenticatedGet(page, "/api/replay/timeline");
}

/**
 * Prove the real wire status uses the exact path-free root shared by both live
 * audiences, not merely the subset rendered by the page.
 *
 * @param {Record<string, any>} frame
 * @param {number} count
 */
function expectExactWireRecording(frame, count) {
  expect(frame.recording).not.toBeNull();
  expect(Object.keys(frame.recording).sort()).toEqual(RECORDING_STATUS_KEYS);
  expect(frame.recording).toMatchObject({
    schema_version: 1,
    captured_transition_count: count,
    expected_transition_count: expect.any(Number),
  });
  expect(frame.recording.expected_transition_count).toBeGreaterThan(0);
  expect(frame.recording.captured_transition_count).toBe(frame.frame_index);
  for (const forbidden of [
    "path",
    "replay_path",
    "metric_report_path",
    "destination",
    "detail",
    "last_io_error_code",
  ]) {
    expect(frame.recording).not.toHaveProperty(forbidden);
    expect(frame).not.toHaveProperty(forbidden);
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} count
 */
async function expectRecordingCount(page, count) {
  await expect(page.locator("#recording-progress")).toHaveText(
    new RegExp(`^${count} / [1-9]\\d* transitions$`),
    { timeout: 120_000 },
  );
  const frame = await currentFrame(page);
  expectExactWireRecording(frame, count);
  expect(frame.frame_index).toBe(count);
  return frame;
}

/**
 * Submit one live joint turn and settle its local presentation without adding
 * a second simulator command.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} expectedCount
 */
async function captureNextTransition(page, expectedCount) {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
    { timeout: 120_000 },
  );
  await page.locator("#submit-turn-button").click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const frame = await expectRecordingCount(page, expectedCount);
  const activeChoreography = page.locator(
    '[data-layer="transient-events"] > .combat-choreography[data-state="active"]',
  );
  if ((await activeChoreography.count()) > 0) {
    await finishControllerClock(page, "cleanup");
    await expect(activeChoreography).toHaveCount(0);
  }
  return frame;
}

/** @param {import("@playwright/test").Page} page */
function captureOneTransition(page) {
  return captureNextTransition(page, 1);
}

/** @param {import("@playwright/test").Page} page */
async function expectSettledReplayHandoff(page, audience = "researcher") {
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay", {
    timeout: 120_000,
  });
  await expect(page).toHaveTitle("MARL-BattleGrounds Replay Viewer");
  await expect(page.locator("html")).toHaveAttribute(
    "data-product-kind",
    "replay_viewer",
  );
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#replay-timeline")).toBeVisible();
  await expect(page.locator("#replay-timeline")).toBeFocused();
  await expect(page.locator("#replay-frame-slider")).toHaveValue("0");
  await expect(page.locator("#replay-frame-position")).toContainText("Tick 0 /");
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("[data-live-only]:not([hidden])")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Submit joint turn" })).toHaveCount(0);
  await expect(
    page.locator(
      "#motion-pause-button, #motion-off-button, #motion-skip-button, #motion-status",
    ),
  ).toHaveCount(0);

  const [frame, timeline, boundary] = await Promise.all([
    currentFrame(page),
    currentTimeline(page),
    page.evaluate(() => {
      const liveRoots = [...document.querySelectorAll("[data-live-only]")];
      const focusable = [
        ...document.querySelectorAll(
          'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      ].filter((element) => {
        if (!(element instanceof HTMLElement)) {
          return false;
        }
        const style = getComputedStyle(element);
        return (
          !element.hidden &&
          !element.hasAttribute("disabled") &&
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          element.getClientRects().length > 0
        );
      });
      return {
        everyLiveRootHidden: liveRoots.every(
          (element) => element instanceof HTMLElement && element.hidden,
        ),
        focusableInsideLiveRoot: focusable
          .filter((element) => element.closest("[data-live-only]") !== null)
          .map((element) => element.id || element.textContent?.trim()),
      };
    }),
  ]);
  const researcher = audience === "researcher";
  expect(frame).toMatchObject({
    schema_version: 1,
    frame_kind: researcher ? "researcher_replay_viewer" : "actor_pov_replay_viewer",
    view_mode: researcher ? "researcher" : "pov",
    cursor: { frame_index: 0 },
    timeline_id: timeline.timeline_id,
  });
  if (researcher) {
    expect(frame).toMatchObject({
      incoming_transition_index: null,
      incoming_transition_id: null,
    });
    expect(timeline).toMatchObject({ timeline_kind: "researcher" });
    expect(timeline.rows[0].frame_id).toBe(frame.frame_id);
  } else {
    expect(frame).toMatchObject({ incoming_pov_transition_id: null });
    expect(timeline).toMatchObject({
      timeline_kind: "actor_pov",
      pov_global_slot: frame.pov_global_slot,
      public_agent_id: frame.public_agent_id,
    });
    expect(timeline.rows[0].pov_frame_id).toBe(frame.pov_frame_id);
  }
  expect(boundary).toEqual({
    everyLiveRootHidden: true,
    focusableInsideLiveRoot: [],
  });
  const transientPresentation = await page
    .locator("#battlefield")
    .evaluate((battlefield) => ({
      activeRootCount: battlefield.querySelectorAll(
        '[data-layer="transient-events"] > .combat-choreography[data-state="active"]',
      ).length,
      ownedAnimationIds: battlefield
        .getAnimations({ subtree: true })
        .map(({ id }) => id)
        .filter((id) => id.startsWith("mbg:")),
    }));
  expect(transientPresentation).toEqual({
    activeRootCount: 0,
    ownedAnimationIds: [],
  });
  return { frame, timeline };
}

/**
 * Assert the two path-independent artifacts materialized by the real recorder.
 *
 * @param {string} replayPath
 * @param {string} metricPath
 * @param {number} transitionCount
 */
async function expectSavedArtifacts(replayPath, metricPath, transitionCount) {
  const [replay, metric] = await Promise.all([
    readJsonArtifact(replayPath),
    readJsonArtifact(metricPath),
  ]);
  expect(replay.bytes.byteLength).toBeGreaterThan(0);
  expect(metric.bytes.byteLength).toBeGreaterThan(0);
  expect(replay.value).toMatchObject({
    schema_id: "marl_battlegrounds.evaluation.replay_artifact",
    schema_version: 1,
    header: {
      recorded_transition_count: transitionCount,
      recorded_frame_count: transitionCount + 1,
    },
    completion: { validated_transition_count: transitionCount },
  });
  expect(replay.value.frames).toHaveLength(transitionCount + 1);
  expect(replay.value.transitions).toHaveLength(transitionCount);
  expect(metric.value).toMatchObject({
    schema_id: "marl_battlegrounds.evaluation.metric_report_artifact",
    schema_version: 1,
    report: { completion: { validated_transition_count: transitionCount } },
  });
  expect(metric.value.report.completion).toEqual(replay.value.completion);
  expect(metric.value.source_trajectory.episode_id).toBe(
    replay.value.header.context.identity.episode_id,
  );
  return { replay, metric };
}

test("confirmed prefix discard restarts capture and Finish opens settled frame-zero replay", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  const stableUrl = page.url();
  const stableToken = await page.evaluate(() =>
    window.sessionStorage.getItem("marl-battlegrounds.debugger-token"),
  );
  expect(stableToken).toMatch(/^[A-Za-z0-9_-]+$/u);
  const initial = await currentFrame(page);
  expectExactWireRecording(initial, 0);
  expect(initial.recording).toMatchObject({
    schema_version: 1,
    lifecycle: "recording",
    captured_transition_count: 0,
    finish_available: true,
    restart_fenced: false,
    discard_available: false,
  });
  expect(initial.recording).not.toHaveProperty("path");
  await expect(page.locator("#recording-finish-button")).toBeEnabled();
  await expect(page.locator("#recording-review-button")).toBeHidden();
  await expect(page.locator("#recording-retry-button")).toBeHidden();
  await expect(page.locator("#recording-save-as-control")).toBeHidden();
  await expect(page.locator("body")).not.toContainText(started.outputDirectory);
  await expect(page.locator("body")).not.toContainText(started.replayPath);

  await page.locator("#view-select").selectOption("pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  const pov = await currentFrame(page);
  expect(pov).toMatchObject({
    schema_version: 2,
    frame_kind: "actor_pov_live_debugger",
    view_mode: "pov",
  });
  expectExactWireRecording(pov, 0);
  expect(pov.recording).toEqual(initial.recording);
  await expect(page.locator("#recording-panel")).toBeVisible();
  await expect(page.locator("body")).not.toContainText(started.outputDirectory);
  await page.locator("#view-select").selectOption("researcher");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");

  const captured = await captureOneTransition(page);
  expect(captured.recording).toMatchObject({
    lifecycle: "recording",
    captured_transition_count: 1,
    restart_fenced: true,
    discard_available: true,
  });
  await expect(page.locator("#reset-button")).toBeEnabled();
  await expect(page.locator("#scenario-select")).toHaveCount(0);
  await expect(page.locator("#movement-scale-input")).toHaveCount(0);

  await page.locator("#reset-button").click();
  await expect(page.locator("#recording-discard-dialog")).toBeVisible();
  await expect(page.locator("#recording-discard-intent")).toHaveText(
    "Reset the current episode",
  );
  await expect(page.locator("#recording-discard-cancel-button")).toBeFocused();
  await expectRecordingCount(page, 1);
  const openDiscardHelp = await expectVisibleInteractiveHelpInventory(page);
  expect(openDiscardHelp.registered).toContain("#recording-discard-cancel-button");
  expect(openDiscardHelp.registered).toContain("#recording-discard-confirm-button");
  await page.locator("#recording-discard-cancel-button").click();
  await expect(page.locator("#recording-discard-dialog")).toBeHidden();
  const closedDiscardHelp = await expectVisibleInteractiveHelpInventory(page);
  expect(closedDiscardHelp.registered).not.toContain(
    "#recording-discard-cancel-button",
  );
  expect(closedDiscardHelp.registered).not.toContain(
    "#recording-discard-confirm-button",
  );
  await expectRecordingCount(page, 1);

  await page.locator("#reset-button").click();
  const resetResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await page.locator("#recording-discard-confirm-button").click();
  expect((await resetResponse).status()).toBe(200);
  const restarted = await expectRecordingCount(page, 0);
  expect(restarted.episode_id).not.toBe(captured.episode_id);
  expect(restarted.recording).toMatchObject({
    restart_fenced: false,
    discard_available: false,
  });

  await captureOneTransition(page);
  const finishResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
    { timeout: 120_000 },
  );
  await page.locator("#recording-finish-button").click();
  const response = await finishResponse;
  expect(response.status()).toBe(200);
  await expectSettledReplayHandoff(page);
  expect(page.url()).toBe(stableUrl);
  expect(
    await page.evaluate(() =>
      window.sessionStorage.getItem("marl-battlegrounds.debugger-token"),
    ),
  ).toBe(stableToken);
  await expectSavedArtifacts(started.replayPath, started.metricReportPath, 1);
  expectNoBrowserErrors(page);
});

test("Exit persists an interrupted prefix before clean process shutdown", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureOneTransition(page);
  expectNoBrowserErrors(page);
  const exitPromise = waitForRecordingDebuggerExit(started.process);
  const responsePromise = page
    .waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname === "/api/command",
      { timeout: 120_000 },
    )
    .then(async (response) => ({
      payload: await response.json(),
      status: response.status(),
    }));
  await page.locator("#exit-button").click();
  const response = await responsePromise;
  expect(response.status).toBe(200);
  expect(response.payload).toMatchObject({
    schema_version: 2,
    result: "shutdown_scheduled",
    frame: {
      recording: {
        lifecycle: "saved",
        captured_transition_count: 1,
        completion_state: "interrupted",
        completion_reason: "user_exit",
      },
    },
  });
  const { replay } = await expectSavedArtifacts(
    started.replayPath,
    started.metricReportPath,
    1,
  );
  expect(replay.value.completion).toMatchObject({
    completion_state: "interrupted",
    end_or_failure_reason: "user_exit",
  });
  expect(await exitPromise).toEqual({ exitCode: 0, signalCode: null });
});

test("a second live tab cannot advance after Finish and Reconnect adopts replay", async ({
  context,
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  const stalePage = await context.newPage();
  await openRecording(stalePage, started.url);
  await captureOneTransition(page);

  await page.locator("#recording-finish-button").click();
  await expectSettledReplayHandoff(page);
  await expect(stalePage.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  await expect(stalePage.locator("#recording-progress")).toContainText("0 /");

  const rejectedPromise = stalePage.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
    { timeout: 120_000 },
  );
  await stalePage.locator("#submit-turn-button").click();
  const rejected = await rejectedPromise;
  expect(rejected.status()).toBe(404);
  await expect(stalePage.locator("#connection-status")).toHaveText("Resync required");
  await expect(stalePage.locator("#step-value")).toHaveText("0");

  await stalePage.locator("#reconnect-button").click();
  const { frame } = await expectSettledReplayHandoff(stalePage);
  expect(frame.cursor).toMatchObject({ frame_index: 0, final_frame_index: 1 });
  const saved = await readJsonArtifact(started.replayPath);
  expect(saved.value.transitions).toHaveLength(1);
  expectNoBrowserErrors(page);
  expect(browserErrors.get(stalePage) ?? []).toEqual([
    "console: Failed to load resource: the server responded with a status of 404 (Not Found)",
  ]);
  await stalePage.close();
});

test("actor POV handoff retains battlefield fog and artifact-wide facts", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await page.locator("#view-select").selectOption("pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  const initialPov = await currentFrame(page);
  expect(initialPov).toMatchObject({
    schema_version: 2,
    frame_kind: "actor_pov_live_debugger",
    view_mode: "pov",
  });
  expectExactWireRecording(initialPov, 0);
  for (const forbidden of [
    "scenario",
    "available_scenarios",
    "incoming_transition_id",
    "incoming_transition_index",
  ]) {
    expect(initialPov).not.toHaveProperty(forbidden);
  }
  await expect(page.locator("#recording-panel")).toBeVisible();

  await captureOneTransition(page);
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
    { timeout: 120_000 },
  );
  await page.locator("#recording-finish-button").click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const { frame, timeline } = await expectSettledReplayHandoff(page, "pov");
  expect(frame).toMatchObject({
    artifact_summary: {
      metric_report_availability: "not_available_in_actor_pov",
    },
    processing_disclosure: { disclosure: "not_available_in_actor_pov" },
    artifact_facts: {
      schema_version: 1,
      artifact_summary: {
        metric_report_availability: "available",
        recorded_transition_count: 1,
        recorded_frame_count: 2,
      },
      completion: { validated_transition_count: 1 },
      processing: {
        status: "succeeded",
        processed_transition_count: 1,
      },
    },
  });
  expect(frame.artifact_facts.artifact_summary.replay_reference).toEqual(
    frame.artifact_summary.replay_reference,
  );
  await expect(page.locator("#replay-artifact-reference")).toHaveText(
    frame.artifact_facts.artifact_summary.replay_reference.artifact_id,
  );
  await expect(page.locator("#replay-processing-badge")).toHaveText(
    "Authorized replay",
  );
  await expect(page.locator("#replay-download-metrics-button")).toBeEnabled();
  const forbiddenKeys = new Set([
    "processing",
    "processing_status",
    "processed_transition_count",
    "failure_stage",
    "failure_code",
    "attempted_transition_index",
    "metric_report_reference",
  ]);
  /** @type {Set<string>} */
  const leaked = new Set();
  /** @param {unknown} value */
  const visit = (value) => {
    if (Array.isArray(value)) {
      for (const child of value) visit(child);
      return;
    }
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      if (forbiddenKeys.has(key)) leaked.add(key);
      visit(child);
    }
  };
  const battlefieldTransport = { ...frame };
  delete battlefieldTransport.artifact_facts;
  visit(battlefieldTransport);
  visit(timeline);
  expect([...leaked].sort()).toEqual([]);
  await expectSavedArtifacts(started.replayPath, started.metricReportPath, 1);
  expectNoBrowserErrors(page);
});

test("target race remains Online and fenced until Save As recovers cached artifacts", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureOneTransition(page);
  const sentinel = await createReplayTargetRace(started);

  await page.locator("#recording-finish-button").click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "persistence_failed",
    { timeout: 120_000 },
  );
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#recording-persistence-error")).toHaveText(
    "Destination unavailable",
  );
  await expect(page.locator("#recording-retry-button")).toBeEnabled();
  await expect(page.locator("#recording-save-as-input")).toBeEnabled();
  await expect(page.locator("#recording-save-as-button")).toBeEnabled();
  await expect(page.locator("#reset-button")).toBeDisabled();
  await expect(page.locator("#scenario-select")).toHaveCount(0);
  await expect(page.locator("#movement-scale-input")).toHaveCount(0);
  await expect(page.locator("#advance-script-button")).toHaveCount(0);
  await expect(page.locator("#view-select")).toBeEnabled();
  await expect(page.locator("#preset-select")).toHaveCount(0);
  await expect(page.locator("#exit-button")).toBeEnabled();
  const oracleCloseoutLabel =
    "Read-only live battlefield. Scientific facts can be inspected; simulator and actor activation controls are unavailable.";
  await expect(page.getByRole("group", { name: oracleCloseoutLabel })).toBeVisible();
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Scientific tooltip facts remain inspectable. Live simulator and actor activation controls are unavailable while recording is closing, the session is offline or resynchronizing, or the frame is terminal.",
  );
  await expect(page.locator('[data-key="g"]')).toBeEnabled();
  await expect(page.locator('[data-key="v"]')).toHaveCount(0);
  await expect(page.locator('[data-key="n"]')).toHaveCount(0);
  await expect(page.locator('[data-key="w"]')).toBeDisabled();
  await expect(page.locator("#command-target-select")).toBeDisabled();
  const rosterButtons = page.locator("#roster button");
  await expect.poll(() => rosterButtons.count()).toBeGreaterThan(0);
  expect(
    await rosterButtons.evaluateAll((buttons) =>
      buttons.every((button) => button instanceof HTMLButtonElement && button.disabled),
    ),
  ).toBe(true);
  const scientificOwner = page
    .locator('#battlefield [data-tooltip-owner][tabindex="0"]')
    .first();
  await expect(scientificOwner).toBeVisible();
  /** @type {unknown[]} */
  const scientificOwnerRequests = [];
  /** @param {import("@playwright/test").Request} request */
  const recordScientificOwnerRequest = (request) => {
    if (
      request.method() === "POST" &&
      ["/api/command", "/api/replay/command"].includes(new URL(request.url()).pathname)
    ) {
      scientificOwnerRequests.push(request.postDataJSON());
    }
  };
  page.on("request", recordScientificOwnerRequest);
  await scientificOwner.focus();
  await scientificOwner.press("Enter");
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  page.off("request", recordScientificOwnerRequest);
  expect(scientificOwnerRequests).toEqual([]);
  const focusFence = await page.evaluate(() => {
    const blockedSelectors = [
      "#battlefield",
      "#reset-button",
      "#command-target-select",
      '[data-key="w"]',
      "#roster button",
    ];
    /** @param {Element | null} element */
    const tabFocusable = (element) =>
      element instanceof HTMLElement &&
      !element.hidden &&
      !element.hasAttribute("disabled") &&
      element.tabIndex >= 0 &&
      getComputedStyle(element).display !== "none" &&
      getComputedStyle(element).visibility !== "hidden" &&
      element.getClientRects().length > 0;
    const blockedTabLeaks = [];
    for (const selector of blockedSelectors) {
      for (const element of document.querySelectorAll(selector)) {
        if (tabFocusable(element)) {
          blockedTabLeaks.push(selector);
        }
      }
    }
    const allowedSelectors = ["#view-select", '[data-key="g"]', "#exit-button"];
    const allowedFocus = [];
    for (const selector of allowedSelectors) {
      const element = document.querySelector(selector);
      if (tabFocusable(element)) {
        allowedFocus.push(selector);
      }
    }
    return { allowedFocus, blockedTabLeaks };
  });
  expect(focusFence).toEqual({
    allowedFocus: ["#view-select", '[data-key="g"]', "#exit-button"],
    blockedTabLeaks: [],
  });

  await page.locator("#view-select").selectOption("pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  const agentCloseoutLabel =
    "Agent POV recording closeout battlefield. Authorized bodies can be inspected; simulator controls are unavailable.";
  await expect(page.getByRole("group", { name: agentCloseoutLabel })).toBeVisible();
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Authorized visible bodies can still be inspected locally; recording closeout has fenced POV switching, simulator input, and pending-action controls.",
  );
  const agentCloseoutBodies = page.locator("#battlefield .agent[role='button']");
  await expect.poll(() => agentCloseoutBodies.count()).toBeGreaterThan(0);
  const agentCloseoutRows = page.locator("#roster .roster-primary-action");
  await expect.poll(() => agentCloseoutRows.count()).toBeGreaterThan(1);
  await expect(page.locator('#roster [data-visibility="visible"]')).toBeVisible();
  await expect(page.locator('#roster [data-visibility="not-visible"]')).toBeVisible();
  await expect(page.locator("#roster .roster-team[data-team-id]")).toHaveCount(2);
  await expect(page.locator("#roster .roster-team[data-team-id]:visible")).toHaveCount(
    0,
  );
  expect(
    await agentCloseoutRows.evaluateAll((buttons) =>
      buttons.every((button) => button instanceof HTMLButtonElement && button.disabled),
    ),
  ).toBe(true);
  /** @type {unknown[]} */
  const localActivationRequests = [];
  /** @param {import("@playwright/test").Request} request */
  const recordLocalActivation = (request) => {
    if (
      request.method() === "POST" &&
      ["/api/command", "/api/replay/command"].includes(new URL(request.url()).pathname)
    ) {
      localActivationRequests.push(request.postDataJSON());
    }
  };
  page.on("request", recordLocalActivation);
  await agentCloseoutBodies.nth(1).click();
  await expect(agentCloseoutBodies.nth(1)).toHaveAttribute("data-selected", "true");
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  page.off("request", recordLocalActivation);
  expect(localActivationRequests).toEqual([]);

  const failedFrame = await currentFrame(page);
  expect(failedFrame.recording).toMatchObject({
    lifecycle: "persistence_failed",
    persistence_error_code: "target_unavailable",
    retry_available: true,
    save_as_available: true,
  });
  await expect(page.locator("body")).not.toContainText(started.outputDirectory);
  await expect(page.locator("body")).not.toContainText(started.replayPath);
  const recoveryHelp = await expectVisibleInteractiveHelpInventory(page);
  expect(recoveryHelp.disabled.length).toBeGreaterThan(0);

  await page.locator("#recording-retry-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "persistence_failed",
  );

  await page.locator("#exit-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online", {
    timeout: 120_000,
  });
  await expect(page.locator("#connection-status")).not.toHaveText("Shutting down");
  await expect(page.locator("#notice")).toContainText("remains open for recovery");

  await page.locator("#recording-save-as-input").fill("../escape.marlbg-replay.json");
  await page.locator("#recording-save-as-button").click();
  await expect(page.locator("#notice")).toContainText(
    "Save As requires a basename ending in .marlbg-replay.json",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-recording-lifecycle",
    "persistence_failed",
  );

  const recoveredName = "recovered-episode.marlbg-replay.json";
  const recoveredReplayPath = join(started.outputDirectory, recoveredName);
  const recoveredMetricPath = metricReportPathForReplay(recoveredReplayPath);
  await page.locator("#recording-save-as-input").fill(recoveredName);
  await page.locator("#recording-save-as-button").click();
  await expectSettledReplayHandoff(page, "pov");
  await expectSavedArtifacts(recoveredReplayPath, recoveredMetricPath, 1);
  expect(await readFile(started.replayPath)).toEqual(Buffer.from(sentinel));
  expectNoBrowserErrors(page);
});

test("Reconnect completes a lost Finish response without retrying publication", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureOneTransition(page);
  let finishRequests = 0;
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    if (payload?.command?.command_type !== "finish_and_review") {
      await route.continue();
      return;
    }
    finishRequests += 1;
    const response = await route.fetch();
    expect(response.status()).toBe(200);
    await route.abort("failed");
  });

  await page.locator("#recording-finish-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Resync required", {
    timeout: 120_000,
  });
  await expect(page.locator("#notice")).toContainText(
    "Command outcome is unknown because the connection failed",
  );
  expect(finishRequests).toBe(1);

  await page.unroute("**/api/command");
  await page.locator("#reconnect-button").click();
  await expectSettledReplayHandoff(page);
  expect(finishRequests).toBe(1);
  await expectSavedArtifacts(started.replayPath, started.metricReportPath, 1);

  const errors = browserErrors.get(page) ?? [];
  expect(errors).toEqual(["console: Failed to load resource: net::ERR_FAILED"]);
});
