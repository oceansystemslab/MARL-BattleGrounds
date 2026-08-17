import { readFileSync } from "node:fs";

import { expect, test } from "@playwright/test";
import {
  currentReplayFrame,
  currentReplayTimeline,
  expectReplayFrameIndex,
  exportReplayArtifacts,
  removeReplayArtifacts,
  startReplayViewer,
  stopDebugger,
} from "./support/replay-viewer.js";
import {
  expectVisibleInteractiveHelpInventory,
  waitForStablePresentation,
} from "./support/visual-regression.js";

test.describe.configure({ mode: "serial" });

/** @type {Awaited<ReturnType<typeof exportReplayArtifacts>> | null} */
let artifacts = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let completeViewer = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let partialViewer = null;
/** @type {Record<string, any> | null} */
let liveFrameCandidate = null;

/** @type {WeakMap<import("@playwright/test").Page, string[]>} */
const browserErrors = new WeakMap();

test.beforeAll(async () => {
  artifacts = await exportReplayArtifacts();
  /** @type {import("node:child_process").ChildProcess[]} */
  const startedProcesses = [];
  try {
    const fixture = JSON.parse(
      readFileSync(
        new URL("../tests/fixtures/authorized-presentations-v1.json", import.meta.url),
        "utf8",
      ),
    );
    liveFrameCandidate = structuredClone(fixture.pairs.live_oracle.transport);
    completeViewer = await startReplayViewer({ replayPath: artifacts.complete });
    startedProcesses.push(completeViewer.process);
    partialViewer = await startReplayViewer({
      replayPath: artifacts.partial,
      frameIndex: 2,
    });
    startedProcesses.push(partialViewer.process);
  } catch (error) {
    const stopResults = await Promise.allSettled(
      startedProcesses.map((process) => stopDebugger(process)),
    );
    /** @type {unknown[]} */
    const cleanupErrors = stopResults.flatMap((result) =>
      result.status === "rejected" ? [result.reason] : [],
    );
    completeViewer = null;
    partialViewer = null;
    liveFrameCandidate = null;
    try {
      await removeReplayArtifacts(artifacts.outputDirectory);
    } catch (cleanupError) {
      cleanupErrors.push(cleanupError);
    }
    artifacts = null;
    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        [error, ...cleanupErrors],
        "Replay E2E startup and cleanup both failed.",
      );
    }
    throw error;
  }
});

test.afterAll(async () => {
  const processes = [completeViewer?.process ?? null, partialViewer?.process ?? null];
  completeViewer = null;
  partialViewer = null;
  liveFrameCandidate = null;
  const stopResults = await Promise.allSettled(
    processes.map((process) => stopDebugger(process)),
  );
  /** @type {unknown[]} */
  const cleanupErrors = stopResults.flatMap((result) =>
    result.status === "rejected" ? [result.reason] : [],
  );
  try {
    await removeReplayArtifacts(artifacts?.outputDirectory);
  } catch (error) {
    cleanupErrors.push(error);
  }
  artifacts = null;
  if (cleanupErrors.length > 0) {
    throw new AggregateError(cleanupErrors, "Replay E2E cleanup failed.");
  }
});

/** @param {import("@playwright/test").Page} page */
function captureBrowserErrors(page) {
  if (!browserErrors.has(page)) {
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
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 */
async function openReplay(page, url) {
  captureBrowserErrors(page);
  await page.goto(url);
  await expect(page.locator("#connection-status")).toHaveText("Online", {
    timeout: 30_000,
  });
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
  await expect(page.locator("#replay-timeline")).toBeVisible();
  expect(page.url()).not.toContain("token=");
}

/** @param {import("@playwright/test").Page} page */
function expectNoBrowserErrors(page) {
  expect(browserErrors.get(page) ?? []).toEqual([]);
}

/**
 * Chromium reports an HTTP failure at the network-console layer even when the
 * application intentionally handles the exact replay conflict response.
 *
 * @param {import("@playwright/test").Page} page
 * @param {import("@playwright/test").Response} response
 */
function expectOnlyHandledReplayConflictConsole(page, response) {
  expect(response.status()).toBe(409);
  expect(response.request().method()).toBe("POST");
  expect(new URL(response.url()).pathname).toBe("/api/replay/command");
  expect(browserErrors.get(page) ?? []).toEqual([
    "console: Failed to load resource: the server responded with a status of 409 (Conflict)",
  ]);
}

/**
 * @param {import("@playwright/test").Page} page
 */
function nextReplayResponse(page) {
  return page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/replay/command",
    { timeout: 30_000 },
  );
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} selector
 */
async function clickReplayCommand(page, selector) {
  const responsePromise = nextReplayResponse(page);
  await page.locator(selector).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  return response.json();
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} key
 */
async function pressReplayCommand(page, key) {
  await expect(page.locator("#replay-frame-slider")).toBeEnabled({
    timeout: 30_000,
  });
  const responsePromise = nextReplayResponse(page);
  await page.locator("#replay-timeline").focus();
  await page.keyboard.press(key);
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  return response.json();
}

/**
 * @param {Record<string, any>} frame
 * @param {Record<string, any>} timeline
 * @param {number} frameIndex
 */
function expectResearcherJoin(frame, timeline, frameIndex) {
  const episodeId = frame.artifact_summary.replay_reference.episode_id;
  const row = timeline.rows[frameIndex];
  expect(frame).toMatchObject({
    schema_version: 1,
    frame_kind: "researcher_replay_viewer",
    view_mode: "researcher",
    timeline_id: timeline.timeline_id,
    frame_id: `${episodeId}:frame:${frameIndex}`,
    cursor: {
      frame_index: frameIndex,
      final_frame_index: timeline.final_frame_index,
    },
  });
  expect(row).toMatchObject({
    frame_index: frameIndex,
    frame_id: frame.frame_id,
    simulator_step_count: frame.simulator_step_count,
    incoming_transition_id: frame.incoming_transition_id,
  });
  expect(frame.projection.scene).toMatchObject({
    episode_id: episodeId,
    frame_index: frameIndex,
    frame_id: frame.frame_id,
    simulator_step_count: frame.simulator_step_count,
    incoming_transition_id: frame.incoming_transition_id,
  });
  if (frame.incoming_transition_id === null) {
    expect(frame.projection.incoming_events).toBeNull();
  } else {
    expect(frame.projection.incoming_events.transition_id).toBe(
      frame.incoming_transition_id,
    );
  }
}

/**
 * Install frame zero through the user-facing replay keyboard boundary.
 *
 * @param {import("@playwright/test").Page} page
 */
async function installFirstFrame(page) {
  const slider = page.locator("#replay-frame-slider");
  if ((await slider.inputValue()) !== "0") {
    await pressReplayCommand(page, "Home");
  }
  await expectReplayFrameIndex(page, 0);
}

/**
 * Prove the public presentation clock has no active replay animation. Motion
 * editing controls are intentionally absent from the product shell.
 *
 * @param {import("@playwright/test").Page} page
 */
async function expectReplayChoreographySettled(page) {
  await expect(
    page.locator("#battlefield .combat-choreography[data-state=playing]"),
  ).toHaveCount(0);
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );
}

/**
 * Freeze and prove the complete replay transport at one supported viewport.
 * The element screenshot is intentionally paired with its semantic ordering,
 * cursor, range, and overflow contract so a visually plausible but incomplete
 * transport cannot refresh the baseline.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{width: number, height: number}} viewport
 */
async function captureReplayTransportBaseline(page, viewport) {
  await page.setViewportSize(viewport);
  await waitForStablePresentation(page);

  const transport = page.locator("#replay-timeline .replay-timeline__transport");
  await expect(transport).toBeVisible();
  expect(
    await transport.locator(":scope > button").evaluateAll((buttons) =>
      buttons.map((button) => ({
        id: button.id,
        label: button.textContent?.trim(),
      })),
    ),
  ).toEqual([
    { id: "replay-first-button", label: "First" },
    { id: "replay-back-ten-button", label: "−10" },
    { id: "replay-previous-button", label: "−1" },
    { id: "replay-play-pause-button", label: "Play" },
    { id: "replay-next-button", label: "+1" },
    { id: "replay-forward-ten-button", label: "+10" },
    { id: "replay-last-button", label: "Last" },
  ]);
  await expect(page.locator("#replay-frame-slider")).toHaveAttribute("min", "0");
  await expect(page.locator("#replay-frame-slider")).toHaveAttribute("max", "5");
  await expect(page.locator("#replay-frame-slider")).toHaveValue("0");
  await expect(page.locator("#replay-frame-slider")).toBeEnabled();
  for (const selector of [
    "#replay-first-button",
    "#replay-back-ten-button",
    "#replay-previous-button",
  ]) {
    await expect(page.locator(selector)).toBeDisabled();
  }
  for (const selector of [
    "#replay-play-pause-button",
    "#replay-next-button",
    "#replay-forward-ten-button",
    "#replay-last-button",
  ]) {
    await expect(page.locator(selector)).toBeEnabled();
  }
  await expect(page.locator("#replay-play-pause-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await expect(page.locator("#replay-frame-position")).toHaveText("Tick 0 / 5");
  const layout = await transport.evaluate((element) => ({
    horizontalOverflow: element.scrollWidth - element.clientWidth,
    viewportEscape: {
      left: Math.max(0, -element.getBoundingClientRect().left),
      right: Math.max(
        0,
        element.getBoundingClientRect().right - document.documentElement.clientWidth,
      ),
    },
  }));
  expect(layout.horizontalOverflow).toBeLessThanOrEqual(1);
  expect(layout.viewportEscape).toEqual({ left: 0, right: 0 });

  await expect(transport).toHaveScreenshot(
    `replay-transport-${viewport.width}x${viewport.height}.png`,
    { animations: "disabled" },
  );
}

/** @param {number} milliseconds */
function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

/**
 * @param {Awaited<ReturnType<typeof startReplayViewer>> | null} viewer
 * @param {string} name
 * @returns {Awaited<ReturnType<typeof startReplayViewer>>}
 */
function requiredViewer(viewer, name) {
  if (!viewer) {
    throw new Error(`${name} replay viewer was not started.`);
  }
  return viewer;
}

/** @returns {Record<string, any>} */
function requiredLiveFrameCandidate() {
  if (!liveFrameCandidate) {
    throw new Error("The validated live-frame reconnect candidate was not loaded.");
  }
  return liveFrameCandidate;
}

test("canonical complete and partial artifacts join their frame-zero and captured endpoints", async ({
  page,
  context,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  const partial = requiredViewer(partialViewer, "partial");
  await openReplay(page, complete.url);

  const [frame, timeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  expect(timeline).toMatchObject({
    schema_version: 1,
    timeline_kind: "researcher",
    final_frame_index: 5,
    completion: { completion_state: "complete" },
  });
  expect(timeline.rows).toHaveLength(6);
  expect(timeline.rows[5].endpoint_kind).toBe("declared_horizon");
  expectResearcherJoin(frame, timeline, 0);
  expect(frame.incoming_transition_index).toBeNull();
  expect(frame.incoming_transition_id).toBeNull();
  expect(frame.artifact_summary).toMatchObject({
    expected_transition_count: 5,
    recorded_transition_count: 5,
    recorded_frame_count: 6,
    metric_report_availability: "available",
  });
  await expect(page.locator("#replay-frame-position")).toHaveText("Tick 0 / 5");
  expect(
    await page
      .locator("#replay-timeline .replay-timeline__transport")
      .evaluate((transport) =>
        [...transport.querySelectorAll(":scope > button")].map((button) => button.id),
      ),
  ).toEqual([
    "replay-first-button",
    "replay-back-ten-button",
    "replay-previous-button",
    "replay-play-pause-button",
    "replay-next-button",
    "replay-forward-ten-button",
    "replay-last-button",
  ]);
  await expect(
    page.locator("#battlefield-utilities > #replay-ranges-button"),
  ).toHaveCount(1);
  await expect(
    page.locator("#battlefield-utilities > #replay-clear-reference-button"),
  ).toHaveCount(1);
  await expect(page.locator("#replay-clear-reference-button")).toHaveText(
    "Clear Selection",
  );
  await expect(page.locator("#replay-timeline #replay-ranges-button")).toHaveCount(0);
  await expect(
    page.locator("#replay-timeline #replay-clear-reference-button"),
  ).toHaveCount(0);
  await expect(page.locator("#battlefield-shell #replay-ranges-button")).toHaveCount(0);
  await expect(page.locator("#replay-visual-key > dt")).toHaveText([
    "Team A",
    "Team B",
    "Selected agent",
  ]);
  await expect(page.locator("#live-visual-key")).toHaveAttribute("hidden", "");
  await expect(page.locator("#replay-visual-key")).not.toHaveAttribute("hidden", "");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  await expect(page.locator("#events-details")).toHaveAttribute("open", "");
  await expect(page.locator("#event-feed .event-item")).toHaveCount(0);
  for (const selector of [
    "#agent-details",
    "#visual-key",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
  }
  await expect(page.locator("#replay-completion-badge")).toHaveText("Complete");
  await expect(page.locator("#replay-processing-badge")).toHaveText(
    "Authorized replay",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  const researcherHelp = await expectVisibleInteractiveHelpInventory(page);
  expect(researcherHelp.disabled.length).toBeGreaterThan(0);
  expect(researcherHelp.registered).toContain("#replay-timeline");
  await expect(page.locator("#replay-timeline")).toHaveAttribute(
    "aria-description",
    "Use the read-only transport and presentation controls to inspect recorded frames.",
  );

  const replayBoundary = await page.evaluate(() => {
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
  });
  expect(replayBoundary.everyLiveRootHidden).toBe(true);
  expect(replayBoundary.focusableInsideLiveRoot).toEqual([]);
  await expect(page.getByRole("button", { name: "Submit joint turn" })).toHaveCount(0);

  const frameIdentityBeforeResize = frame.frame_id;
  await page.setViewportSize({ width: 960, height: 600 });
  await page.evaluate(
    () =>
      new Promise((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      }),
  );
  const resized = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(resized.scrollWidth).toBeLessThanOrEqual(resized.clientWidth);
  expect((await currentReplayFrame(page)).frame_id).toBe(frameIdentityBeforeResize);
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  expect((await currentReplayFrame(page)).frame_id).toBe(frameIdentityBeforeResize);

  const partialPage = await context.newPage();
  await openReplay(partialPage, partial.url);
  const [partialFrame, partialTimeline] = await Promise.all([
    currentReplayFrame(partialPage),
    currentReplayTimeline(partialPage),
  ]);
  expect(partialTimeline).toMatchObject({
    timeline_kind: "researcher",
    final_frame_index: 2,
    completion: {
      completion_state: "partial",
      end_or_failure_reason: "browser_test_capture_stopped",
      validated_transition_count: 2,
    },
  });
  expect(partialTimeline.rows).toHaveLength(3);
  expect(partialTimeline.rows[2].endpoint_kind).toBe("captured_prefix");
  expectResearcherJoin(partialFrame, partialTimeline, 2);
  await expect(partialPage.locator("#replay-frame-position")).toHaveText("Tick 2 / 2");
  await expect(partialPage.locator("#terminal-badge")).toHaveText(
    "End of captured prefix",
  );
  await expect(partialPage.locator("#replay-completion-badge")).toHaveText("Partial");
  await expect(partialPage.locator("#replay-end-reason")).toHaveText(
    "browser_test_capture_stopped",
  );
  expectNoBrowserErrors(page);
  expectNoBrowserErrors(partialPage);
  await partialPage.close();
});

test("replay transport has permanent visual proof at the minimum supported viewport", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);

  const [frame, timeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  expectResearcherJoin(frame, timeline, 0);
  expect(timeline.final_frame_index).toBe(5);
  await captureReplayTransportBaseline(page, { width: 960, height: 600 });
  expectNoBrowserErrors(page);
});

test("replay reconnect rejects a valid live frame without crossing the viewer boundary", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  const liveFrame = requiredLiveFrameCandidate();
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  const [installedFrame, installedTimeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  expectResearcherJoin(installedFrame, installedTimeline, 0);
  expect(liveFrame).toMatchObject({
    schema_version: 2,
    frame_kind: "researcher_live_debugger",
  });
  const installedDom = {
    framePosition: await page.locator("#replay-frame-position").textContent(),
    view: await page.locator("#view-select").inputValue(),
  };

  let frameRequestCount = 0;
  let timelineRequestCount = 0;
  /** @param {import("@playwright/test").Request} request */
  const recordTimelineRequest = (request) => {
    if (
      request.method() === "GET" &&
      new URL(request.url()).pathname === "/api/replay/timeline"
    ) {
      timelineRequestCount += 1;
    }
  };
  page.on("request", recordTimelineRequest);
  await page.route("**/api/frame", async (route) => {
    frameRequestCount += 1;
    await route.fulfill({ json: structuredClone(liveFrame), status: 200 });
  });
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect(page.locator("#connection-status")).toHaveAttribute(
    "data-state",
    "offline",
  );
  await expect(page.locator("#notice")).toContainText(
    "Raw transport and presentation kinds raced between GET responses",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
  await expect(page.locator("#replay-timeline")).toBeVisible();
  await expect(page.locator("#replay-artifact-reference")).toHaveText(
    "Unavailable while authority is pending",
  );
  await expect(page.locator("#replay-frame-position")).toHaveText("Tick — / —");
  await expect(page.locator("#view-select")).toHaveValue("");
  await expect(page.locator("#view-select")).toBeDisabled();
  await expect(page.locator("[data-live-only]:not([hidden])")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Submit joint turn" })).toHaveCount(0);
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-label",
    "Read-only live battlefield. Simulator and actor activation controls are unavailable.",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  expect(frameRequestCount).toBe(2);
  expect(timelineRequestCount).toBe(0);

  await page.unroute("**/api/frame");
  page.off("request", recordTimelineRequest);
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(page, 0);
  await expect(page.locator("#replay-frame-position")).toHaveText(
    installedDom.framePosition ?? "",
  );
  await expect(page.locator("#view-select")).toHaveValue(installedDom.view);
  await expect(page.locator("#view-select")).toBeEnabled();
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expectReplayChoreographySettled(page);
  const recoveredFrame = await currentReplayFrame(page);
  expect(recoveredFrame.frame_id).toBe(installedFrame.frame_id);
  expect(recoveredFrame.timeline_id).toBe(installedTimeline.timeline_id);
  expectNoBrowserErrors(page);
});

test("real replay next animates while previous and absolute seek settle", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);

  const playingChoreography = page.locator(
    '#battlefield [data-layer="transient-events"] > .combat-choreography[data-state="playing"]',
  );
  const next = await clickReplayCommand(page, "#replay-next-button");
  expect(next).toMatchObject({
    result: "applied",
    animate_incoming: true,
    frame: { cursor: { frame_index: 1 } },
  });
  await expectReplayFrameIndex(page, 1);
  await expect(playingChoreography).toHaveCount(1);
  await expect(page.locator("html")).toHaveAttribute("data-submission-blocked", "true");

  const previous = await clickReplayCommand(page, "#replay-previous-button");
  expect(previous).toMatchObject({
    result: "applied",
    animate_incoming: false,
    frame: { cursor: { frame_index: 0 } },
  });
  await expectReplayFrameIndex(page, 0);
  await expectReplayChoreographySettled(page);
  await expect(page.locator("#replay-frame-slider")).toBeEnabled();

  const seekRequestPromise = page.waitForRequest(
    (request) =>
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command",
    { timeout: 30_000 },
  );
  const seekResponsePromise = nextReplayResponse(page);
  await page.locator("#replay-frame-slider").evaluate((element) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new TypeError("Replay slider is unavailable.");
    }
    element.value = "4";
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const [seekRequest, seekResponse] = await Promise.all([
    seekRequestPromise,
    seekResponsePromise,
  ]);
  expect(seekRequest.postDataJSON().command).toEqual({
    command_type: "absolute_seek",
    frame_index: 4,
  });
  expect(seekResponse.status()).toBe(200);
  const seek = await seekResponse.json();
  expect(seek).toMatchObject({
    result: "applied",
    animate_incoming: false,
    frame: { cursor: { frame_index: 4 } },
  });
  await expectReplayFrameIndex(page, 4);
  await expectReplayChoreographySettled(page);
  await expect(
    page.locator('#battlefield [data-layer="transient-events"] > .combat-choreography'),
  ).toHaveAttribute("data-state", "settled");
  expectNoBrowserErrors(page);
});

test("a stale cross-tab command atomically installs the latest audience without retry or animation", async ({
  page,
  context,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  if ((await page.locator("#view-select").inputValue()) !== "researcher") {
    const researcherResponse = nextReplayResponse(page);
    await page.locator("#view-select").selectOption("researcher");
    await researcherResponse;
  }

  const secondPage = await context.newPage();
  await openReplay(secondPage, complete.url);
  const audienceChangeResponse = nextReplayResponse(secondPage);
  await secondPage.locator("#view-select").selectOption("pov");
  expect((await audienceChangeResponse).status()).toBe(200);
  await expect(secondPage.locator("#view-select")).toHaveValue("pov");

  let stalePostCount = 0;
  /** @param {import("@playwright/test").Request} request */
  const recordStalePost = (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      stalePostCount += 1;
    }
  };
  page.on("request", recordStalePost);
  const staleResponsePromise = page.waitForResponse(
    (response) =>
      response.status() === 409 &&
      new URL(response.url()).pathname === "/api/replay/command",
    { timeout: 30_000 },
  );
  const matchingTimelinePromise = page.waitForResponse(
    (response) =>
      response.status() === 200 &&
      response.request().method() === "GET" &&
      new URL(response.url()).pathname === "/api/replay/timeline",
    { timeout: 30_000 },
  );
  await page.locator("#replay-next-button").click();
  const staleResponse = await staleResponsePromise;
  const matchingTimelineResponse = await matchingTimelinePromise;
  const [stalePayload, matchingTimeline] = await Promise.all([
    staleResponse.json(),
    matchingTimelineResponse.json(),
  ]);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expect(page.locator("#notice")).toContainText(
    "coherent latest pair was installed; the command was not retried",
  );
  await expect(page.locator("#replay-play-pause-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await expectReplayChoreographySettled(page);
  expect(stalePayload.error_code).toBe("stale_revision");
  expect(stalePayload).not.toHaveProperty("animate_incoming");
  expect(matchingTimeline.timeline_id).toBe(stalePayload.latest_frame.timeline_id);
  expect(stalePostCount).toBe(1);
  page.off("request", recordStalePost);

  const [latestFrame, latestTimeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  expect(latestFrame.frame_kind).toBe("actor_pov_replay_viewer");
  expect(latestTimeline.timeline_kind).toBe("actor_pov");
  expect(latestTimeline.timeline_id).toBe(latestFrame.timeline_id);

  const restoreResponse = nextReplayResponse(page);
  await page.locator("#view-select").selectOption("researcher");
  expect((await restoreResponse).status()).toBe(200);
  await expect(page.locator("#view-select")).toHaveValue("researcher");
  expectOnlyHandledReplayConflictConsole(page, staleResponse);
  expectNoBrowserErrors(secondPage);
  await secondPage.close();
});

test("accessible playback pauses on hidden/error/endpoint and keeps one request in flight", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  const timeline = page.locator("#replay-timeline");
  const play = page.locator("#replay-play-pause-button");

  await timeline.focus();
  await page.keyboard.press("Space");
  await expect(play).toHaveAttribute("aria-label", "Pause replay");
  await expect(play).toHaveAttribute("aria-pressed", "true");
  await page.keyboard.press("Space");
  await expect(play).toHaveAttribute("aria-label", "Play replay");
  await expect(play).toHaveAttribute("aria-pressed", "false");

  await timeline.focus();
  await page.keyboard.press("Space");
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(play).toHaveAttribute("aria-pressed", "false");
  await page.evaluate(() => {
    Reflect.deleteProperty(document, "hidden");
    document.dispatchEvent(new Event("visibilitychange"));
  });

  await page.route("**/api/replay/command", async (route) => {
    await route.abort("connectionfailed");
  });
  await play.click();
  await expect(page.locator("#connection-status")).toHaveText("Resync required", {
    timeout: 10_000,
  });
  await expect(page.locator("#connection-status")).toHaveAttribute(
    "data-state",
    "offline",
  );
  await expect(play).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator("#notice")).toContainText(
    "Reconnect before sending another replay command",
  );
  await page.unroute("**/api/replay/command");
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(page, 0);

  let activeRequests = 0;
  let maximumActiveRequests = 0;
  /** @type {Record<string, any>[]} */
  const playbackRequests = [];
  /** @type {Record<string, any>[]} */
  const playbackResponses = [];
  await page.route("**/api/replay/command", async (route) => {
    activeRequests += 1;
    maximumActiveRequests = Math.max(maximumActiveRequests, activeRequests);
    playbackRequests.push(route.request().postDataJSON());
    await delay(225);
    const response = await route.fetch();
    playbackResponses.push(await response.json());
    activeRequests -= 1;
    await route.fulfill({ response });
  });
  await play.click();
  await expect(play).toHaveAttribute("aria-label", "Pause replay");
  await expectReplayFrameIndex(page, 5);
  await expect(play).toHaveAttribute("aria-label", "Play replay");
  await expect(play).toHaveAttribute("aria-pressed", "false");
  await expect(play).toBeDisabled();
  await expect(page.locator("#terminal-badge")).toHaveText("Declared horizon");
  await page.unroute("**/api/replay/command");

  expect(maximumActiveRequests).toBe(1);
  expect(playbackRequests).toHaveLength(5);
  expect(playbackRequests.map((request) => request.command.command_type)).toEqual(
    Array(5).fill("next_frame"),
  );
  expect(playbackResponses).toHaveLength(5);
  expect(playbackResponses.every((response) => response.animate_incoming)).toBe(true);
  expect(browserErrors.get(page) ?? []).toEqual([
    "console: Failed to load resource: net::ERR_CONNECTION_FAILED",
  ]);
});

test("Actor POV all-surface scan excludes researcher authority and host secrets", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  const artifactManifest = artifacts;
  if (!artifactManifest) {
    throw new Error("Replay artifacts are unavailable for the POV isolation scan.");
  }

  await openReplay(page, complete.url);
  await installFirstFrame(page);
  if ((await page.locator("#view-select").inputValue()) !== "researcher") {
    const researcherResponse = nextReplayResponse(page);
    await page.locator("#view-select").selectOption("researcher");
    expect((await researcherResponse).status()).toBe(200);
  }
  const next = await clickReplayCommand(page, "#replay-next-button");
  expect(next.frame.cursor.frame_index).toBe(1);
  await expectReplayFrameIndex(page, 1);
  await expectReplayChoreographySettled(page);
  const researcherFrame = await currentReplayFrame(page);
  expect(researcherFrame.frame_kind).toBe("researcher_replay_viewer");

  const povResponse = nextReplayResponse(page);
  await page.locator("#view-select").selectOption("pov");
  expect((await povResponse).status()).toBe(200);
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expectReplayFrameIndex(page, 1);
  await expect(page.locator("#preset-select")).toHaveCount(0);
  await expect(page.locator("html")).toHaveAttribute("data-preset", "analysis");
  await page.locator("#technical-frame-details").evaluate((details) => {
    if (!(details instanceof HTMLDetailsElement)) {
      throw new TypeError("Technical frame details are unavailable.");
    }
    details.open = true;
  });

  const [povFrame, povTimeline, capabilityToken] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
    page.evaluate(() =>
      window.sessionStorage.getItem("marl-battlegrounds.debugger-token"),
    ),
  ]);
  expect(povFrame.frame_kind).toBe("actor_pov_replay_viewer");
  expect(povTimeline.timeline_kind).toBe("actor_pov");
  expect(capabilityToken).toMatch(/^[A-Za-z0-9_-]+$/u);

  /** @param {unknown} value @param {string} field @returns {unknown[]} */
  const recursiveFieldValues = (value, field) => {
    /** @type {unknown[]} */
    const found = [];
    /** @param {unknown} candidate */
    const visit = (candidate) => {
      if (Array.isArray(candidate)) {
        candidate.forEach(visit);
        return;
      }
      if (!candidate || typeof candidate !== "object") {
        return;
      }
      for (const [key, child] of Object.entries(candidate)) {
        if (key === field) {
          found.push(child);
        }
        visit(child);
      }
    };
    visit(value);
    return found;
  };
  /** @param {unknown} value @returns {Set<string>} */
  const recursiveKeys = (value) => {
    /** @type {Set<string>} */
    const found = new Set();
    /** @param {unknown} candidate */
    const visit = (candidate) => {
      if (Array.isArray(candidate)) {
        candidate.forEach(visit);
        return;
      }
      if (!candidate || typeof candidate !== "object") {
        return;
      }
      for (const [key, child] of Object.entries(candidate)) {
        found.add(key);
        visit(child);
      }
    };
    visit(value);
    return found;
  };

  const authorizedPublicIds = new Set(
    recursiveFieldValues([povFrame, povTimeline], "public_agent_id").filter(
      (value) => typeof value === "string",
    ),
  );
  expect(authorizedPublicIds.has(povFrame.public_agent_id)).toBe(true);
  /** @type {Array<Record<string, any>>} */
  const researcherAgents = researcherFrame.projection.scene.agents;
  const hiddenAgents = researcherAgents.filter(
    (agent) => !authorizedPublicIds.has(agent.public_agent_id),
  );
  expect(hiddenAgents.length).toBeGreaterThan(0);
  const hiddenSlots = new Set(hiddenAgents.map((agent) => agent.global_slot));
  const hiddenPublicIds = hiddenAgents.map((agent) => agent.public_agent_id);
  const researcherEventIds = recursiveFieldValues(
    researcherFrame.projection.incoming_events,
    "event_id",
  ).filter((value) => typeof value === "string");
  expect(researcherEventIds.length).toBeGreaterThan(0);

  const selfRoster = page.locator("#roster .roster-primary-action").first();
  await expect(selfRoster).toBeVisible();
  await selfRoster.hover();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  const tooltipSnapshot = await page.locator("#visual-tooltip").evaluate((root) => ({
    text: root.textContent ?? "",
    attributes: [...root.querySelectorAll("*")].flatMap((element) =>
      [...element.attributes].map((attribute) => ({
        name: attribute.name,
        value: attribute.value,
      })),
    ),
  }));
  const selfAgent = page.locator("#battlefield .agent").first();
  await expect(page.locator('#battlefield .agent[role="button"]')).toHaveCount(
    await page.locator("#battlefield .agent").count(),
  );
  await expect(selfAgent).toHaveAttribute("role", "button");
  await expect(selfAgent).toHaveAttribute("tabindex", "0");
  expect(await selfAgent.getAttribute("data-slot")).toBeNull();
  await expect(selfAgent).toHaveAttribute("data-presentation-key", /.+/u);
  await selfAgent.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await expect(page.locator("#event-feed .event-item")).not.toHaveCount(0);

  const surfaces = await page.evaluate(() => {
    /** @param {Element} element */
    const isVisible = (element) => {
      if (element.closest("[hidden], [aria-hidden='true']")) return false;
      const style = getComputedStyle(element);
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        element.getClientRects().length > 0
      );
    };
    /** @param {Element} root */
    const snapshot = (root) => ({
      text: root.textContent ?? "",
      attributes: [root, ...root.querySelectorAll("*")].flatMap((element) =>
        [...element.attributes].map((attribute) => ({
          name: attribute.name,
          value: attribute.value,
        })),
      ),
    });
    /** @param {string} selector @returns {Element} */
    const requiredRoot = (selector) => {
      const root = document.querySelector(selector);
      if (!root) {
        throw new Error(`Required isolation surface is missing: ${selector}`);
      }
      return root;
    };
    const visibleElements = [
      document.body,
      ...document.body.querySelectorAll("*"),
    ].filter(isVisible);
    const visibleAttributes = visibleElements.flatMap((element) =>
      [...element.attributes].map((attribute) => ({
        name: attribute.name,
        value: attribute.value,
      })),
    );
    const descriptorText = visibleElements
      .filter((element) => element.hasAttribute("data-tooltip-owner"))
      .flatMap((element) => [
        element.getAttribute("aria-label") ?? "",
        element.getAttribute("aria-description") ?? "",
      ])
      .join("\n");
    const technicalRoot = requiredRoot("#diagnostics-card");
    return {
      normalText: document.body.innerText,
      visibleAttributes,
      descriptorText,
      feed: snapshot(requiredRoot("#event-feed")),
      inspector: snapshot(requiredRoot("#agent-details")),
      technical: snapshot(technicalRoot),
      technicalFacts: [
        ...technicalRoot.querySelectorAll(".fact[data-technical-fact]"),
      ].map((node) => ({
        id: node.getAttribute("data-technical-fact"),
        label: node.querySelector("span")?.textContent ?? "",
        value: node.querySelector("strong")?.textContent ?? "",
      })),
      technicalJson: [...document.querySelectorAll(".technical-json")].map(
        (node) => node.textContent ?? "",
      ),
    };
  });

  const allAttributes = [
    ...surfaces.visibleAttributes,
    ...tooltipSnapshot.attributes,
    ...surfaces.feed.attributes,
    ...surfaces.inspector.attributes,
    ...surfaces.technical.attributes,
  ];
  const semanticSurfaceText = [
    surfaces.normalText,
    surfaces.descriptorText,
    tooltipSnapshot.text,
    surfaces.feed.text,
    surfaces.inspector.text,
    surfaces.technical.text,
    ...allAttributes
      .filter(({ name }) => name.startsWith("aria-") || name === "title")
      .map(({ value }) => value),
  ].join("\n");
  const completeSurfaceBytes = JSON.stringify({
    surfaces,
    tooltipSnapshot,
  });
  const povWireBytes = JSON.stringify([povFrame, povTimeline]);

  for (const authorizedId of authorizedPublicIds) {
    if (authorizedId === povFrame.public_agent_id) {
      expect(completeSurfaceBytes).toContain(authorizedId);
    }
  }
  for (const forbiddenValue of [
    ...hiddenPublicIds,
    ...researcherEventIds,
    artifactManifest.outputDirectory,
    artifactManifest.complete,
    artifactManifest.missingMetric,
    capabilityToken,
  ]) {
    if (typeof forbiddenValue === "string" && forbiddenValue.length > 0) {
      expect(completeSurfaceBytes).not.toContain(forbiddenValue);
      expect(povWireBytes).not.toContain(forbiddenValue);
    }
  }
  for (const slot of hiddenSlots) {
    expect(semanticSurfaceText).not.toMatch(
      new RegExp(`\\b(?:global[ _-]?slot|slot|id_)\\s*[:#= -]?\\s*${slot}\\b`, "iu"),
    );
  }
  const hiddenSlotAttributes = allAttributes.filter(
    ({ name, value }) =>
      /slot/iu.test(name) && /^\d+$/u.test(value) && hiddenSlots.has(Number(value)),
  );
  expect(hiddenSlotAttributes).toEqual([]);

  for (const researcherOnlyPhrase of [
    "PRIVILEGED RESEARCHER",
    "Exact Class Mechanics",
    "Observer visibility",
    "Status source evidence",
    "Researcher selection",
  ]) {
    expect(semanticSurfaceText).not.toContain(researcherOnlyPhrase);
  }
  const actorKeys = recursiveKeys([povFrame, povTimeline]);
  for (const researcherOnlyKey of [
    "agent_phase_trajectories",
    "aura_fields",
    "class_mechanics",
    "configured_active_by_global_slot",
    "event_id",
    "incoming_event_ids",
    "metric_report_reference",
    "observer_visibility",
    "processing",
    "public_agent_id_by_global_slot",
    "status_source_evidence",
  ]) {
    expect(actorKeys.has(researcherOnlyKey), researcherOnlyKey).toBe(false);
  }
  expect(surfaces.technicalFacts).toEqual([
    {
      id: "frame",
      label: "Frame",
      value: String(povFrame.cursor.frame_index),
    },
    {
      id: "simulator_step",
      label: "Simulator step",
      value: String(povFrame.simulator_step_count),
    },
  ]);
  expect(completeSurfaceBytes).not.toContain("technical_kind");
  expect(completeSurfaceBytes).not.toContain("replay_no_shared_obs_technical_frame");
  expect(surfaces.normalText).toContain("Agent POV");
  expect(completeSurfaceBytes).not.toContain("actor_pov_replay_viewer");
  expect(surfaces.technicalJson).toEqual([]);
  expect(completeSurfaceBytes).not.toMatch(/(?:token|secret|password)\s*[:=]/iu);
  expectNoBrowserErrors(page);
});

test("Exit flushes its replay response before clean server shutdown", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  const exitPromise = new Promise((resolve, reject) => {
    const process = complete.process;
    if (process.exitCode !== null || process.signalCode !== null) {
      resolve(undefined);
      return;
    }
    const timeout = setTimeout(
      () => reject(new Error("Replay viewer did not exit after its exit response.")),
      5_000,
    );
    process.once("exit", () => {
      clearTimeout(timeout);
      resolve(undefined);
    });
  });
  const response = await clickReplayCommand(page, "#exit-button");
  expect(response).toMatchObject({
    result: "shutdown_scheduled",
    animate_incoming: false,
  });
  await exitPromise;
  expect(complete.process.exitCode).toBe(0);
  expect(complete.process.signalCode).toBeNull();
});
