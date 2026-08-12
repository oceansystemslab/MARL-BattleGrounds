import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { expect, test } from "@playwright/test";

import {
  createReplayTargetRace,
  metricReportPathForReplay,
  readJsonArtifact,
  startRecordingDebugger,
  stopRecordingDebugger,
  waitForRecordingDebuggerExit,
} from "./support/recording-handoff.js";

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
 * Advance the exact registered scripted trajectory once and settle its local
 * presentation without adding a second simulator command.
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
  await page.locator("#advance-script-button").click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const frame = await expectRecordingCount(page, expectedCount);
  const skip = page.locator("#motion-skip-button");
  if (await skip.isEnabled()) {
    await skip.click();
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
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#replay-timeline")).toBeVisible();
  await expect(page.locator("#replay-timeline")).toBeFocused();
  await expect(page.locator("#replay-frame-slider")).toHaveValue("0");
  await expect(page.locator("#replay-frame-position")).toContainText("Frame 0 /");
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("[data-live-only]:not([hidden])")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Submit joint turn" })).toHaveCount(0);
  await expect(page.locator("#motion-skip-button")).toBeDisabled();

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

test("T0 Finish publishes an exact zero-transition partial replay", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
    { timeout: 120_000 },
  );
  await page.locator("#recording-finish-button").click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  expect(await response.json()).toMatchObject({
    schema_version: 2,
    result: "applied",
    frame: {
      frame_index: 0,
      recording: {
        lifecycle: "reviewing",
        captured_transition_count: 0,
        completion_state: "partial",
        completion_reason: "user_finish_and_review",
      },
    },
  });
  const { frame, timeline } = await expectSettledReplayHandoff(page);
  expect(frame.cursor).toMatchObject({ frame_index: 0, final_frame_index: 0 });
  expect(timeline.rows).toHaveLength(1);
  const { replay } = await expectSavedArtifacts(
    started.replayPath,
    started.metricReportPath,
    0,
  );
  expect(replay.value.completion).toMatchObject({
    completion_state: "partial",
    validated_transition_count: 0,
    end_or_failure_reason: "user_finish_and_review",
  });
  expectNoBrowserErrors(page);
});

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
  await expect(page.locator("#scenario-select")).toBeEnabled();
  await expect(page.locator("#movement-scale-input")).toBeEnabled();

  await page.locator("#reset-button").click();
  await expect(page.locator("#recording-discard-dialog")).toBeVisible();
  await expect(page.locator("#recording-discard-intent")).toHaveText(
    "Reset the current scenario to a fresh episode",
  );
  await expect(page.locator("#recording-discard-cancel-button")).toBeFocused();
  await expectRecordingCount(page, 1);
  await page.locator("#recording-discard-cancel-button").click();
  await expect(page.locator("#recording-discard-dialog")).toBeHidden();
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
  const payload = await response.json();
  expect(payload).toMatchObject({
    schema_version: 2,
    result: "applied",
    frame: {
      recording: {
        lifecycle: "reviewing",
        captured_transition_count: 1,
      },
    },
  });
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

test("scripted endpoint auto-saves complete capture and Review opens frame zero", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureNextTransition(page, 1);
  const endpoint = await captureNextTransition(page, 2);
  expect(endpoint.recording).toMatchObject({
    lifecycle: "saved",
    captured_transition_count: 2,
    completion_state: "complete",
    completion_reason: null,
    finish_available: false,
    review_available: true,
  });
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#recording-review-button")).toBeEnabled();
  await expect(page.locator("#recording-finish-button")).toBeHidden();
  await expect(page.locator("#advance-script-button")).toBeDisabled();
  const saved = await expectSavedArtifacts(
    started.replayPath,
    started.metricReportPath,
    2,
  );
  expect(saved.replay.value.completion).toMatchObject({
    completion_state: "complete",
    validated_transition_count: 2,
    end_or_failure_reason: null,
  });

  const reviewResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
    { timeout: 120_000 },
  );
  await page.locator("#recording-review-button").click();
  expect((await reviewResponse).status()).toBe(200);
  const { frame } = await expectSettledReplayHandoff(page);
  expect(frame.cursor).toMatchObject({ frame_index: 0, final_frame_index: 2 });
  expectNoBrowserErrors(page);
});

test("Exit persists an interrupted prefix before clean process shutdown", async ({
  page,
}) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureOneTransition(page);
  const exitPromise = waitForRecordingDebuggerExit(started.process);
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
    { timeout: 120_000 },
  );
  await page.locator("#exit-button").click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  expect(await response.json()).toMatchObject({
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
  expectNoBrowserErrors(page);
});

test("Ctrl-C persists an interrupted prefix and exits cleanly", async ({ page }) => {
  const started = requiredRecording();
  await openRecording(page, started.url);
  await captureOneTransition(page);
  expectNoBrowserErrors(page);
  await page.close();

  const exitPromise = waitForRecordingDebuggerExit(started.process);
  expect(started.process.kill("SIGINT")).toBe(true);
  expect(await exitPromise).toEqual({ exitCode: 0, signalCode: null });
  const { replay } = await expectSavedArtifacts(
    started.replayPath,
    started.metricReportPath,
    1,
  );
  expect(replay.value.completion).toMatchObject({
    completion_state: "interrupted",
    end_or_failure_reason: "keyboard_interrupt",
  });
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
  await stalePage.locator("#advance-script-button").click();
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

test("actor POV recording and handoff retain exact status and non-disclosure roots", async ({
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
  expect(await response.json()).toMatchObject({
    frame: {
      frame_kind: "actor_pov_live_debugger",
      view_mode: "pov",
      recording: { lifecycle: "reviewing", captured_transition_count: 1 },
    },
  });
  const { frame, timeline } = await expectSettledReplayHandoff(page, "pov");
  expect(frame).toMatchObject({
    artifact_summary: {
      metric_report_availability: "not_available_in_actor_pov",
    },
    processing_disclosure: { disclosure: "not_available_in_actor_pov" },
  });
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
  visit(frame);
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
  await expect(page.locator("#scenario-select")).toBeDisabled();
  await expect(page.locator("#movement-scale-input")).toBeDisabled();
  await expect(page.locator("#advance-script-button")).toBeDisabled();
  await expect(page.locator("#view-select")).toBeEnabled();
  await expect(page.locator("#preset-select")).toBeEnabled();
  await expect(page.locator("#exit-button")).toBeEnabled();
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator('[data-key="g"]')).toBeEnabled();
  await expect(page.locator('[data-key="v"]')).toBeEnabled();
  await expect(page.locator('[data-key="n"]')).toBeDisabled();
  await expect(page.locator('[data-key="w"]')).toBeDisabled();
  await expect(page.locator("#command-target-select")).toBeDisabled();
  const rosterButtons = page.locator("#roster button");
  await expect.poll(() => rosterButtons.count()).toBeGreaterThan(0);
  expect(
    await rosterButtons.evaluateAll((buttons) =>
      buttons.every((button) => button instanceof HTMLButtonElement && button.disabled),
    ),
  ).toBe(true);
  const focusFence = await page.evaluate(() => {
    const blockedSelectors = [
      "#battlefield",
      "#reset-button",
      "#scenario-select",
      "#movement-scale-input",
      "#advance-script-button",
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
    const allowedSelectors = [
      "#view-select",
      "#preset-select",
      '[data-key="g"]',
      '[data-key="v"]',
      "#exit-button",
    ];
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
    allowedFocus: [
      "#view-select",
      "#preset-select",
      '[data-key="g"]',
      '[data-key="v"]',
      "#exit-button",
    ],
    blockedTabLeaks: [],
  });
  const failedFrame = await currentFrame(page);
  expect(failedFrame.recording).toMatchObject({
    lifecycle: "persistence_failed",
    persistence_error_code: "target_unavailable",
    retry_available: true,
    save_as_available: true,
  });
  await expect(page.locator("body")).not.toContainText(started.outputDirectory);
  await expect(page.locator("body")).not.toContainText(started.replayPath);

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
  await expectSettledReplayHandoff(page);
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
