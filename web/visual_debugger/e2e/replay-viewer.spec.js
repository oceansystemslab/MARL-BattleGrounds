import { expect, test } from "@playwright/test";
import {
  loadRendererFixture,
  syntheticDebuggerWireFrame,
} from "./support/renderer-fixture.js";
import {
  currentReplayFrame,
  currentReplayTimeline,
  expectReplayFrameIndex,
  exportReplayArtifacts,
  removeReplayArtifacts,
  startReplayViewer,
  stopDebugger,
} from "./support/replay-viewer.js";
import { expectVisibleInteractiveHelpInventory } from "./support/visual-regression.js";

test.describe.configure({ mode: "serial" });

/** @type {Awaited<ReturnType<typeof exportReplayArtifacts>> | null} */
let artifacts = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let completeViewer = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let partialViewer = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let missingMetricViewer = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let sharedViewer = null;
/** @type {Record<string, any> | null} */
let liveFrameCandidate = null;

/** @type {WeakMap<import("@playwright/test").Page, string[]>} */
const browserErrors = new WeakMap();

test.beforeAll(async () => {
  artifacts = await exportReplayArtifacts();
  /** @type {import("node:child_process").ChildProcess[]} */
  const startedProcesses = [];
  try {
    liveFrameCandidate = syntheticDebuggerWireFrame(
      await loadRendererFixture("crowded_teamfight"),
    );
    completeViewer = await startReplayViewer({ replayPath: artifacts.complete });
    startedProcesses.push(completeViewer.process);
    partialViewer = await startReplayViewer({
      replayPath: artifacts.partial,
      frameIndex: 2,
    });
    startedProcesses.push(partialViewer.process);
    missingMetricViewer = await startReplayViewer({
      replayPath: artifacts.missingMetric,
      view: "pov",
      povSlot: 0,
    });
    startedProcesses.push(missingMetricViewer.process);
    sharedViewer = await startReplayViewer({
      replayPath: artifacts.shared,
      view: "pov",
      povSlot: 0,
    });
    startedProcesses.push(sharedViewer.process);
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
    missingMetricViewer = null;
    sharedViewer = null;
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
  const processes = [
    completeViewer?.process ?? null,
    partialViewer?.process ?? null,
    missingMetricViewer?.process ?? null,
    sharedViewer?.process ?? null,
  ];
  completeViewer = null;
  partialViewer = null;
  missingMetricViewer = null;
  sharedViewer = null;
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

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 */
async function openReplay(page, url) {
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
  await expect(page.locator("#replay-frame-position")).toHaveText("Frame 0 / 5");
  await expect(page.locator("#replay-completion-badge")).toHaveText("Complete");
  await expect(page.locator("#replay-processing-badge")).toHaveText("Succeeded");
  await expect(page.locator("#replay-incoming-value")).toHaveText("Initial frame");
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
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
  await expect(partialPage.locator("#replay-frame-position")).toHaveText("Frame 2 / 2");
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
    artifact: await page.locator("#replay-artifact-reference").textContent(),
    framePosition: await page.locator("#replay-frame-position").textContent(),
    revision: await page.locator("#revision-value").textContent(),
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
    "A replay viewer cannot reconnect to a live debugger frame",
  );
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
  await expect(page.locator("#replay-timeline")).toBeVisible();
  await expect(page.locator("#replay-artifact-reference")).toHaveText(
    installedDom.artifact ?? "",
  );
  await expect(page.locator("#replay-frame-position")).toHaveText(
    installedDom.framePosition ?? "",
  );
  await expect(page.locator("#revision-value")).toHaveText(
    installedDom.revision ?? String(installedFrame.revision),
  );
  await expect(page.locator("#view-select")).toHaveValue(installedDom.view);
  await expect(page.locator("[data-live-only]:not([hidden])")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Submit joint turn" })).toHaveCount(0);
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  expect(frameRequestCount).toBe(1);
  expect(timelineRequestCount).toBe(0);

  await page.unroute("**/api/frame");
  page.off("request", recordTimelineRequest);
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(page, 0);
  await expect(page.locator("#motion-skip-button")).toBeDisabled();
  const recoveredFrame = await currentReplayFrame(page);
  expect(recoveredFrame.frame_id).toBe(installedFrame.frame_id);
  expect(recoveredFrame.timeline_id).toBe(installedTimeline.timeline_id);
  expectNoBrowserErrors(page);
});

test("buttons, slider debounce, keyboard, and researcher Reference preserve exact cursor semantics", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  const initial = await currentReplayFrame(page);

  const next = await clickReplayCommand(page, "#replay-next-button");
  expect(next).toMatchObject({
    result: "applied",
    animate_incoming: true,
    frame: { cursor: { frame_index: 1 } },
  });
  expect(next.frame.cursor.cursor_generation).toBe(
    initial.cursor.cursor_generation + 1,
  );
  expect(next.frame.cursor.choreography_generation).toBe(
    initial.cursor.choreography_generation + 1,
  );
  await expectReplayFrameIndex(page, 1);
  await expect(page.locator("#motion-skip-button")).toBeEnabled();

  const previous = await clickReplayCommand(page, "#replay-previous-button");
  expect(previous).toMatchObject({
    result: "applied",
    animate_incoming: false,
    frame: { cursor: { frame_index: 0 } },
  });
  expect(previous.frame.cursor.choreography_generation).toBe(
    next.frame.cursor.choreography_generation,
  );
  await expect(page.locator("#motion-skip-button")).toBeDisabled();

  const last = await clickReplayCommand(page, "#replay-last-button");
  expect(last).toMatchObject({
    animate_incoming: false,
    frame: { cursor: { frame_index: 5 } },
  });
  await expect(page.locator("#replay-last-button")).toBeDisabled();
  const first = await clickReplayCommand(page, "#replay-first-button");
  expect(first).toMatchObject({
    animate_incoming: false,
    frame: { cursor: { frame_index: 0 } },
  });
  await expect(page.locator("#replay-first-button")).toBeDisabled();

  /** @type {Record<string, any>[]} */
  const sliderRequests = [];
  /** @param {import("@playwright/test").Request} request */
  const recordSliderRequest = (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      sliderRequests.push(request.postDataJSON());
    }
  };
  page.on("request", recordSliderRequest);
  const sliderResponsePromise = nextReplayResponse(page);
  await page.locator("#replay-frame-slider").evaluate((element) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new TypeError("Replay slider is unavailable.");
    }
    for (const value of ["1", "3", "4"]) {
      element.value = value;
      element.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });
  const sliderResponse = await sliderResponsePromise;
  const sliderPayload = await sliderResponse.json();
  await expectReplayFrameIndex(page, 4);
  await delay(250);
  page.off("request", recordSliderRequest);
  expect(sliderRequests).toHaveLength(1);
  expect(sliderRequests[0].command).toEqual({
    command_type: "absolute_seek",
    frame_index: 4,
  });
  expect(sliderPayload).toMatchObject({
    animate_incoming: false,
    frame: { cursor: { frame_index: 4 } },
  });
  await expect(page.locator("#motion-skip-button")).toBeDisabled();

  expect((await pressReplayCommand(page, "Home")).frame.cursor.frame_index).toBe(0);
  expect((await pressReplayCommand(page, "End")).frame.cursor.frame_index).toBe(5);
  expect((await pressReplayCommand(page, "ArrowLeft")).frame.cursor.frame_index).toBe(
    4,
  );
  const arrowForward = await pressReplayCommand(page, "ArrowRight");
  expect(arrowForward.frame.cursor.frame_index).toBe(5);
  expect(arrowForward.animate_incoming).toBe(true);

  await pressReplayCommand(page, "Home");
  const referenceButton = page.locator(
    '#roster button[aria-label="Use Agent ID agent-slot-1 as Reference"]',
  );
  const povActorButton = page.locator(
    '#roster button[aria-label="Choose Agent ID agent-slot-1 as POV actor"]',
  );
  await referenceButton.hover();
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "control",
  );
  await expect(page.locator("#visual-tooltip-title")).toHaveText("Reference");
  await referenceButton.focus();
  await expect(referenceButton).toHaveAttribute("aria-describedby", "visual-tooltip");
  await expect(page.locator("#visual-tooltip-title")).toHaveText("Reference");
  await povActorButton.hover();
  await expect(page.locator("#visual-tooltip-title")).toHaveText("POV actor");
  await povActorButton.focus();
  await expect(povActorButton).toHaveAttribute("aria-describedby", "visual-tooltip");
  await expect(page.locator("#visual-tooltip-title")).toHaveText("POV actor");
  const reference = await clickReplayCommand(
    page,
    '#roster button[aria-label="Use Agent ID agent-slot-1 as Reference"]',
  );
  expect(reference.frame.projection.scene.selection.selected_global_slot).toBe(1);
  await expect(page.locator("#selection-heading")).toHaveText("Reference");
  await expect(
    page.locator('.comparison-agent[data-role="reference"][data-slot="1"]'),
  ).toBeVisible();
  await expect(page.locator('.roster-row[data-slot="1"]')).toHaveAttribute(
    "data-reference",
    "true",
  );
  const rosterReference = page.locator('.roster-row[data-slot="1"]');
  await rosterReference.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#semantic-inspector")).toBeVisible();
  await expect(page.locator("#semantic-inspector")).toContainText("Reference");
  await expect(page.locator("#semantic-inspector")).toContainText(
    "Exact Class Mechanics",
  );
  await expect(page.locator("#semantic-inspector")).not.toContainText(
    "Selected target",
  );
  await page.keyboard.press("Escape");
  await expect(page.locator("#semantic-inspector")).toBeHidden();
  await expect(rosterReference).toBeFocused();
  const cleared = await clickReplayCommand(page, "#replay-clear-reference-button");
  expect(cleared.frame.projection.scene.selection.selected_global_slot).toBeNull();
  await expect(page.locator("#replay-clear-reference-button")).toBeDisabled();
  expectNoBrowserErrors(page);
});

test("invalid response and audience-timeline candidates never replace the installed replay pair", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  if ((await page.locator("#view-select").inputValue()) !== "researcher") {
    const researcherResponse = nextReplayResponse(page);
    await page.locator("#view-select").selectOption("researcher");
    await researcherResponse;
  }
  const installedFrame = await currentReplayFrame(page);
  const installedRevision = await page.locator("#revision-value").textContent();
  const installedTimeline = await currentReplayTimeline(page);

  let commandRequestCount = 0;
  await page.route("**/api/replay/command", async (route) => {
    commandRequestCount += 1;
    const response = await route.fetch();
    const payload = await response.json();
    payload.frame.timeline_id = `${payload.frame.timeline_id}:invalid`;
    await route.fulfill({ response, json: payload });
  });
  await page.locator("#replay-next-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect(page.locator("#replay-frame-slider")).toHaveValue("0");
  await expect(page.locator("#revision-value")).toHaveText(
    installedRevision ?? String(installedFrame.revision),
  );
  await expect(page.locator("#view-select")).toHaveValue("researcher");
  expect(commandRequestCount).toBe(1);
  expect((await currentReplayFrame(page)).cursor.frame_index).toBe(1);
  await page.unroute("**/api/replay/command");

  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(page, 1);
  const frameBeforeBadTimeline = await currentReplayFrame(page);
  const revisionBeforeBadTimeline = await page.locator("#revision-value").textContent();

  let timelineRequestCount = 0;
  await page.route("**/api/replay/timeline", async (route) => {
    timelineRequestCount += 1;
    await route.fulfill({
      contentType: "application/json",
      json: installedTimeline,
      status: 200,
    });
  });
  const povResponse = nextReplayResponse(page);
  await page.locator("#view-select").selectOption("pov");
  expect((await povResponse).status()).toBe(200);
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect(page.locator("#view-select")).toHaveValue("researcher");
  await expect(page.locator("#revision-value")).toHaveText(
    revisionBeforeBadTimeline ?? String(frameBeforeBadTimeline.revision),
  );
  await expectReplayFrameIndex(page, 1);
  expect(timelineRequestCount).toBe(1);
  await page.unroute("**/api/replay/timeline");

  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#view-select")).toHaveValue("pov");
  const [povFrame, povTimeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  expect(povFrame.frame_kind).toBe("actor_pov_replay_viewer");
  expect(povTimeline.timeline_kind).toBe("actor_pov");
  expect(povTimeline.timeline_id).toBe(povFrame.timeline_id);
  const povHelp = await expectVisibleInteractiveHelpInventory(page);
  expect(povHelp.disabled.length).toBeGreaterThan(0);

  const restoreResponse = nextReplayResponse(page);
  await page.locator("#view-select").selectOption("researcher");
  expect((await restoreResponse).status()).toBe(200);
  await expect(page.locator("#view-select")).toHaveValue("researcher");
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
    "latest frame was installed; the command was not retried",
  );
  await expect(page.locator("#replay-play-pause-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await expect(page.locator("#motion-skip-button")).toBeDisabled();
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

test("a real command-ID conflict installs the validated latest frame without retry or animation", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  if ((await page.locator("#view-select").inputValue()) !== "researcher") {
    const researcherResponse = nextReplayResponse(page);
    await page.locator("#view-select").selectOption("researcher");
    await researcherResponse;
  }

  let postCount = 0;
  /** @type {string | null} */
  let firstCommandId = null;
  await page.route("**/api/replay/command", async (route) => {
    postCount += 1;
    const requestPayload = route.request().postDataJSON();
    if (postCount === 1) {
      firstCommandId = requestPayload.command_id;
      const response = await route.fetch();
      await route.fulfill({ response });
      return;
    }
    if (!firstCommandId) {
      throw new Error("The first replay command ID was not captured.");
    }
    const conflictingPayload = {
      ...requestPayload,
      command_id: firstCommandId,
    };
    const response = await route.fetch({
      postData: JSON.stringify(conflictingPayload),
    });
    await route.fulfill({ response });
  });

  const applied = await clickReplayCommand(page, "#replay-next-button");
  expect(applied).toMatchObject({
    result: "applied",
    animate_incoming: true,
    frame: { cursor: { frame_index: 1 } },
  });
  await expectReplayFrameIndex(page, 1);
  const skip = page.locator("#motion-skip-button");
  if (await skip.isEnabled()) {
    await skip.click();
  }
  await expect(skip).toBeDisabled();

  const conflictResponsePromise = page.waitForResponse(
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
  await page.locator("#replay-previous-button").click();
  const [conflictResponse, matchingTimelineResponse] = await Promise.all([
    conflictResponsePromise,
    matchingTimelinePromise,
  ]);
  const [conflictPayload, matchingTimeline] = await Promise.all([
    conflictResponse.json(),
    matchingTimelineResponse.json(),
  ]);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#notice")).toHaveAttribute("data-level", "warning");
  await expect(page.locator("#notice")).toContainText(
    "command-ID conflict. Its latest frame was installed; nothing was retried",
  );
  await expectReplayFrameIndex(page, 1);
  await expect(page.locator("#motion-skip-button")).toBeDisabled();
  await expect(page.locator("#replay-play-pause-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  expect(conflictPayload.error_code).toBe("command_id_conflict");
  expect(conflictPayload).not.toHaveProperty("animate_incoming");
  expect(matchingTimeline.timeline_id).toBe(conflictPayload.latest_frame.timeline_id);
  await delay(250);
  expect(postCount).toBe(2);
  await page.unroute("**/api/replay/command");
  expectOnlyHandledReplayConflictConsole(page, conflictResponse);
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

test("reduced-motion and Motion Off playback use their slower serialized cadences", async ({
  page,
  browser,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  await page.locator("#motion-off-button").click();
  await expect(page.locator("html")).toHaveAttribute("data-motion-mode", "off");
  await expect(page.locator("#motion-status")).toHaveText("Motion off");

  /** @type {number[]} */
  const offRequestTimes = [];
  /** @param {import("@playwright/test").Request} request */
  const recordOffRequest = (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      offRequestTimes.push(performance.now());
    }
  };
  page.on("request", recordOffRequest);
  await page.locator("#replay-play-pause-button").click();
  await expectReplayFrameIndex(page, 2);
  await page.locator("#replay-play-pause-button").click();
  page.off("request", recordOffRequest);
  expect(offRequestTimes.length).toBeGreaterThanOrEqual(2);
  expect(offRequestTimes[1] - offRequestTimes[0]).toBeGreaterThanOrEqual(850);

  const reducedContext = await browser.newContext({ reducedMotion: "reduce" });
  const reducedPage = await reducedContext.newPage();
  await openReplay(reducedPage, complete.url);
  await installFirstFrame(reducedPage);
  await expect(reducedPage.locator("html")).toHaveAttribute(
    "data-motion-mode",
    "reduced",
  );
  await expect(reducedPage.locator("#motion-status")).toContainText("Reduced motion");
  /** @type {number[]} */
  const reducedRequestTimes = [];
  /** @param {import("@playwright/test").Request} request */
  const recordReducedRequest = (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      reducedRequestTimes.push(performance.now());
    }
  };
  reducedPage.on("request", recordReducedRequest);
  await reducedPage.locator("#replay-play-pause-button").click();
  await expectReplayFrameIndex(reducedPage, 2);
  await reducedPage.locator("#replay-play-pause-button").click();
  reducedPage.off("request", recordReducedRequest);
  expect(reducedRequestTimes.length).toBeGreaterThanOrEqual(2);
  expect(reducedRequestTimes[1] - reducedRequestTimes[0]).toBeGreaterThanOrEqual(650);
  expectNoBrowserErrors(page);
  expectNoBrowserErrors(reducedPage);
  await reducedContext.close();
});

test("exact NoSharedObs POV hides metric/processing truth and SharedObs stays source-material-only", async ({
  page,
  context,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  const missingMetric = requiredViewer(missingMetricViewer, "missing-metric");
  const shared = requiredViewer(sharedViewer, "shared");
  await openReplay(page, missingMetric.url);
  const [missingFrame, missingTimeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  const publicAgentId = missingFrame.public_agent_id;
  const episodeId = missingFrame.artifact_summary.replay_reference.episode_id;
  expect(missingFrame).toMatchObject({
    schema_version: 1,
    frame_kind: "actor_pov_replay_viewer",
    view_mode: "pov",
    pov_global_slot: 0,
    pov_frame_id: `${episodeId}:actor-pov:${publicAgentId}:frame:0`,
    incoming_pov_transition_id: null,
    artifact_summary: {
      metric_report_availability: "not_available_in_actor_pov",
    },
    processing_disclosure: { disclosure: "not_available_in_actor_pov" },
  });
  expect(missingTimeline).toMatchObject({
    timeline_kind: "actor_pov",
    timeline_id: missingFrame.timeline_id,
    pov_global_slot: 0,
    public_agent_id: publicAgentId,
  });
  expect(missingTimeline.rows[0].pov_frame_id).toBe(missingFrame.pov_frame_id);
  expect(missingTimeline.rows[0].incoming_pov_transition_id).toBeNull();
  const forbiddenKeys = new Set([
    "processing",
    "processing_status",
    "processed_transition_count",
    "failure_stage",
    "failure_code",
    "attempted_transition_index",
    "metric_report_reference",
  ]);
  const observedForbiddenKeys = page.evaluate(
    ([framePayload, timelinePayload, forbidden]) => {
      const found = new Set();
      const visit = (/** @type {any} */ value) => {
        if (Array.isArray(value)) {
          for (const child of value) visit(child);
          return;
        }
        if (!value || typeof value !== "object") return;
        for (const [key, child] of Object.entries(value)) {
          if (forbidden.includes(key)) found.add(key);
          visit(child);
        }
      };
      visit(framePayload);
      visit(timelinePayload);
      return [...found].sort();
    },
    [missingFrame, missingTimeline, [...forbiddenKeys]],
  );
  expect(await observedForbiddenKeys).toEqual([]);
  await expect(page.locator("#replay-processing-badge")).toHaveText(
    "Not available in actor POV",
  );
  await expect(page.locator("#selection-heading")).toHaveText("Replay recipient");
  await expect(page.locator("#replay-ranges-button")).toBeDisabled();
  await expect(
    page.getByRole("button", { name: /as Reference|as POV actor/ }),
  ).toHaveCount(0);

  const availablePage = await context.newPage();
  await openReplay(availablePage, complete.url);
  await installFirstFrame(availablePage);
  const povResponse = nextReplayResponse(availablePage);
  await availablePage.locator("#view-select").selectOption("pov");
  await povResponse;
  const [availableFrame, availableTimeline] = await Promise.all([
    currentReplayFrame(availablePage),
    currentReplayTimeline(availablePage),
  ]);
  /** @param {Record<string, any>} payload */
  const normalizeSessionFields = (payload) => {
    const normalized = structuredClone(payload);
    if (Object.hasOwn(normalized, "viewer_session_id")) {
      normalized.viewer_session_id = "session";
    }
    if (Object.hasOwn(normalized, "revision")) {
      normalized.revision = 0;
    }
    if (normalized.cursor) {
      normalized.cursor.cursor_generation = 0;
      normalized.cursor.choreography_generation = 0;
    }
    return normalized;
  };
  expect(normalizeSessionFields(availableFrame)).toEqual(
    normalizeSessionFields(missingFrame),
  );
  expect(availableTimeline).toEqual(missingTimeline);
  expectNoBrowserErrors(availablePage);
  await availablePage.close();

  const sharedPage = await context.newPage();
  await openReplay(sharedPage, shared.url);
  const [sharedFrame, sharedTimeline] = await Promise.all([
    currentReplayFrame(sharedPage),
    currentReplayTimeline(sharedPage),
  ]);
  expect(sharedFrame).toMatchObject({
    frame_kind: "shared_obs_source_material_replay_viewer",
    view_mode: "pov",
    observation_materialization: "source_material_only",
    selected_global_slot: 0,
  });
  expect(sharedTimeline).toMatchObject({
    timeline_kind: "shared_obs_source_material",
    timeline_id: sharedFrame.timeline_id,
    observation_materialization: "source_material_only",
    selected_global_slot: 0,
  });
  const sourceProjection = sharedFrame.projection;
  /** @type {number[]} */
  const allyGlobalSlots = sourceProjection.ally_observation_row_global_slot_by_id;
  /** @type {number[]} */
  const enemyGlobalSlots = sourceProjection.enemy_observation_row_global_slot_by_id;
  /** @type {string[]} */
  const allyPublicIds =
    sourceProjection.axis_mapping.ally_observation_row_public_agent_id_by_id;
  /** @type {string[]} */
  const enemyPublicIds =
    sourceProjection.axis_mapping.enemy_observation_row_public_agent_id_by_id;
  expect(allyGlobalSlots).toEqual([0, 1, 2, 3, 4]);
  expect(enemyGlobalSlots).toEqual([5, 6, 7, 8, 9]);
  expect(allyPublicIds).toEqual(allyGlobalSlots.map((slot) => `agent-slot-${slot}`));
  expect(enemyPublicIds).toEqual(enemyGlobalSlots.map((slot) => `agent-slot-${slot}`));

  /** @type {Array<Record<string, any>>} */
  const availabilityRows = sourceProjection.sensor_source_availability;
  expect(availabilityRows).toHaveLength(10);
  expect(availabilityRows.map((row) => row.sensor_source_global_slot)).toEqual([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
  ]);
  for (const row of availabilityRows) {
    const axisGlobalSlots =
      row.base_sensor_relation_axis === "ally" ? allyGlobalSlots : enemyGlobalSlots;
    const axisPublicIds =
      row.base_sensor_relation_axis === "ally" ? allyPublicIds : enemyPublicIds;
    expect(row.sensor_source_global_slot).toBe(
      axisGlobalSlots[row.base_sensor_observation_row],
    );
    expect(row.sensor_source_public_agent_id).toBe(
      axisPublicIds[row.base_sensor_observation_row],
    );
  }
  const selfAvailabilityRows = availabilityRows.filter(
    (row) => row.relation_to_recipient === "self",
  );
  expect(selfAvailabilityRows).toHaveLength(1);
  expect(selfAvailabilityRows[0]).toMatchObject({
    sensor_source_global_slot: sharedFrame.selected_global_slot,
    sensor_source_public_agent_id: sharedFrame.public_agent_id,
    base_sensor_relation_axis: "ally",
    base_sensor_observation_row: 0,
  });
  expect(sourceProjection.base_sensor_frame).toMatchObject({
    public_agent_id: sharedFrame.public_agent_id,
    source_material_frame_id: sharedFrame.source_material_frame_id,
    source_frame_id: sharedFrame.source_frame_id,
  });
  expect(sourceProjection.base_sensor_scene.self_actor).toMatchObject({
    global_slot: sharedFrame.selected_global_slot,
    public_agent_id: sharedFrame.public_agent_id,
  });
  expect(sourceProjection.exact_actor_input_export_available).toBe(false);

  const forbiddenMaterializedFields = new Set([
    "actor_input",
    "composed_actor_input",
    "materialized_actor_input",
    "materialized_shared_obs",
    "materialized_shared_observation",
    "shared_obs",
    "shared_obs_actor_input",
    "shared_observation",
  ]);
  /** @type {string[]} */
  const leakedMaterializedFields = [];
  const inspectMaterializationBoundary = (
    /** @type {unknown} */ value,
    /** @type {string} */ path,
  ) => {
    if (Array.isArray(value)) {
      value.forEach((child, index) => {
        inspectMaterializationBoundary(child, `${path}[${index}]`);
      });
      return;
    }
    if (!value || typeof value !== "object") {
      return;
    }
    for (const [key, child] of Object.entries(value)) {
      const childPath = path ? `${path}.${key}` : key;
      if (forbiddenMaterializedFields.has(key)) {
        leakedMaterializedFields.push(childPath);
      }
      inspectMaterializationBoundary(child, childPath);
    }
  };
  inspectMaterializationBoundary(sharedFrame, "frame");
  inspectMaterializationBoundary(sharedTimeline, "timeline");
  expect(leakedMaterializedFields).toEqual([]);
  await expect(sharedPage.locator("#scenario-description")).toContainText(
    "Shared Obs Source Material",
  );

  const installedSourceDom = {
    framePosition: await sharedPage.locator("#replay-frame-position").textContent(),
    revision: await sharedPage.locator("#revision-value").textContent(),
    view: await sharedPage.locator("#view-select").inputValue(),
  };
  let sourceFrameRequestCount = 0;
  let sourceTimelineRequestCount = 0;
  /** @type {string | null} */
  let interceptedSourceFrameId = null;
  /** @param {import("@playwright/test").Request} request */
  const recordSourceTimelineRequest = (request) => {
    if (
      request.method() === "GET" &&
      new URL(request.url()).pathname === "/api/replay/timeline"
    ) {
      sourceTimelineRequestCount += 1;
    }
  };
  sharedPage.on("request", recordSourceTimelineRequest);
  await sharedPage.route("**/api/frame", async (route) => {
    sourceFrameRequestCount += 1;
    const response = await route.fetch();
    const payload = await response.json();
    interceptedSourceFrameId = payload.source_material_frame_id;
    payload.projection.base_sensor_scene.self_actor.researcher_hidden_truth = {
      must_not_cross_audience_boundary: true,
    };
    await route.fulfill({ response, json: payload });
  });
  await sharedPage.locator("#reconnect-button").click();
  await expect(sharedPage.locator("#connection-status")).toHaveText("Resync required");
  await expect(sharedPage.locator("#notice")).toHaveAttribute("data-level", "error");
  await expect(sharedPage.locator("#notice")).toContainText(
    "unknown or missing fields",
  );
  await expect(sharedPage.locator("html")).toHaveAttribute(
    "data-viewer-mode",
    "replay",
  );
  await expect(sharedPage.locator("#replay-frame-position")).toHaveText(
    installedSourceDom.framePosition ?? "",
  );
  await expect(sharedPage.locator("#revision-value")).toHaveText(
    installedSourceDom.revision ?? String(sharedFrame.revision),
  );
  await expect(sharedPage.locator("#view-select")).toHaveValue(installedSourceDom.view);
  await expect(sharedPage.locator("#scenario-description")).toContainText(
    "Shared Obs Source Material",
  );
  await expect(sharedPage.locator("[data-live-only]:not([hidden])")).toHaveCount(0);
  expect(interceptedSourceFrameId).toBe(sharedFrame.source_material_frame_id);
  expect(sourceFrameRequestCount).toBe(1);
  expect(sourceTimelineRequestCount).toBe(0);

  await sharedPage.unroute("**/api/frame");
  sharedPage.off("request", recordSourceTimelineRequest);
  await sharedPage.locator("#reconnect-button").click();
  await expect(sharedPage.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(sharedPage, 0);
  await expect(sharedPage.locator("#motion-skip-button")).toBeDisabled();
  const recoveredSourceFrame = await currentReplayFrame(sharedPage);
  expect(recoveredSourceFrame.source_material_frame_id).toBe(
    sharedFrame.source_material_frame_id,
  );
  expect(recoveredSourceFrame.timeline_id).toBe(sharedTimeline.timeline_id);
  expectNoBrowserErrors(page);
  expectNoBrowserErrors(sharedPage);
  await sharedPage.close();
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
  expectNoBrowserErrors(page);
});
