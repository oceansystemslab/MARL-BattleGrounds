import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  exportReplayArtifacts,
  removeReplayArtifacts,
  startReplayViewer,
} from "./support/replay-viewer.js";

test.describe.configure({ mode: "serial" });

/** @type {Awaited<ReturnType<typeof exportReplayArtifacts>> | null} */
let artifacts = null;
/** @type {Awaited<ReturnType<typeof startDebugger>> | null} */
let liveDebugger = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let noSharedReplay = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let sharedReplay = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let deathReplay = null;

/** @type {WeakMap<import("@playwright/test").Page, string[]>} */
const browserErrors = new WeakMap();

test.beforeAll(async () => {
  artifacts = await exportReplayArtifacts();
  /** @type {import("node:child_process").ChildProcess[]} */
  const startedProcesses = [];
  try {
    liveDebugger = await startDebugger();
    startedProcesses.push(liveDebugger.process);
    noSharedReplay = await startReplayViewer({ replayPath: artifacts.complete });
    startedProcesses.push(noSharedReplay.process);
    sharedReplay = await startReplayViewer({
      replayPath: artifacts.shared,
      view: "pov",
      povSlot: 0,
    });
    startedProcesses.push(sharedReplay.process);
    deathReplay = await startReplayViewer({
      sampleReplay: "death-respawn-shield",
    });
    startedProcesses.push(deathReplay.process);
  } catch (error) {
    await Promise.allSettled(startedProcesses.map((child) => stopDebugger(child)));
    await removeReplayArtifacts(artifacts.outputDirectory);
    artifacts = null;
    throw error;
  }
});

test.afterAll(async () => {
  const processes = [
    liveDebugger?.process ?? null,
    noSharedReplay?.process ?? null,
    sharedReplay?.process ?? null,
    deathReplay?.process ?? null,
  ];
  liveDebugger = null;
  noSharedReplay = null;
  sharedReplay = null;
  deathReplay = null;
  const stopResults = await Promise.allSettled(
    processes.map((child) => stopDebugger(child)),
  );
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
    throw new AggregateError(
      cleanupErrors,
      "Authorized-presentation E2E cleanup failed.",
    );
  }
});

/** @param {import("@playwright/test").Page} page */
function captureBrowserErrors(page) {
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
 * @param {"live" | "replay"} mode
 */
async function openProduct(page, url, mode) {
  captureBrowserErrors(page);
  await page.goto(url);
  await expect(page.locator("#connection-status")).toHaveText("Online", {
    timeout: 30_000,
  });
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", mode);
  expect(page.url()).not.toContain("token=");
}

/**
 * Read one exact authenticated product resource without bypassing the real
 * loopback service or hand-authoring an authority payload.
 *
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

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} transportKind
 * @param {string} presentationKind
 */
async function expectInstalledLeaf(page, transportKind, presentationKind) {
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const [transport, presentation] = await Promise.all([
    authenticatedGet(page, "/api/frame"),
    authenticatedGet(page, "/api/presentation/frame"),
  ]);
  expect(transport.frame_kind).toBe(transportKind);
  expect(presentation.presentation_kind).toBe(presentationKind);
  expect(presentation.source.source_session_id).toBe(
    transport.session_id ?? transport.viewer_session_id,
  );
  await expect(page.locator("#battlefield .agent")).not.toHaveCount(0);
  const bodyKeys = await page
    .locator("#battlefield .agent")
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  expect(bodyKeys.every((key) => typeof key === "string" && key.length > 0)).toBe(true);
  return { transport, presentation };
}

/**
 * Prove both metadata surfaces use the installed presentation variant's exact
 * incoming-transition identity. The authoritative frame-zero representation
 * has no latest-events branch.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 */
async function expectAuthorizedIncomingTransitionDom(page, presentation) {
  const latestEvents = presentation.latest_events;
  const oracle =
    presentation.presentation_kind === "live_oracle" ||
    presentation.presentation_kind === "replay_oracle";
  const expected =
    latestEvents === null
      ? null
      : oracle
        ? latestEvents.incoming_transition_id
        : latestEvents.incoming_recipient_transition_id;
  if (latestEvents !== null) {
    expect(typeof expected).toBe("string");
    expect(expected.length).toBeGreaterThan(0);
  }
  await expect(page.locator("#transition-value")).toHaveText(expected ?? "—");
  if (presentation.viewer_mode === "replay") {
    await expect(page.locator("#replay-incoming-value")).toHaveText(
      expected ?? "Initial frame",
    );
  }
}

/**
 * Snapshot every browser-owned DOM string and attribute at the authority
 * boundary. Source JavaScript is intentionally outside this scan.
 *
 * @param {import("@playwright/test").Page} page
 */
async function authorityDomSnapshot(page) {
  return page.evaluate(() => ({
    attributes: [document.body, ...document.body.querySelectorAll("*")].flatMap(
      (element) =>
        [...element.attributes].map((attribute) => ({
          name: attribute.name,
          value: attribute.value,
        })),
    ),
    text: document.body.textContent ?? "",
  }));
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 * @param {string[]} forbiddenValues
 */
async function expectAgentAuthoritySurface(page, presentation, forbiddenValues) {
  expect(presentation.authority.authority_kind).toBe("agent_pov");
  const recipientKey = presentation.authority.recipient_presentation_key;
  const bodyKeys = await page.locator("#battlefield .agent").evaluateAll((agents) =>
    agents.map((agent) => ({
      key: agent.getAttribute("data-presentation-key"),
      role: agent.getAttribute("role"),
      slot: agent.getAttribute("data-slot"),
    })),
  );
  expect(bodyKeys.every(({ slot }) => slot === null)).toBe(true);
  expect(
    bodyKeys
      .filter(({ role }) => role === "button")
      .every(({ key }) => key === recipientKey),
  ).toBe(true);

  const snapshot = await authorityDomSnapshot(page);
  const slotAttributes = snapshot.attributes.filter(({ name }) =>
    /(?:^|[-_])(?:global[-_]?slot|source[-_]?slot|target[-_]?slot|recipient[-_]?slot|actor[-_]?slot|controlled[-_]?slot)(?:$|[-_])/iu.test(
      name,
    ),
  );
  expect(slotAttributes).toEqual([]);
  const surfaceBytes = JSON.stringify(snapshot);
  for (const forbidden of forbiddenValues) {
    if (forbidden.length > 0 && !JSON.stringify(presentation).includes(forbidden)) {
      expect(surfaceBytes).not.toContain(forbidden);
    }
  }
  for (const forbiddenPhrase of [
    "shared_obs_source_material",
    "source_artifact_digest_sha256",
    "source_trajectory_content_digest_sha256",
    "source_context_digest_sha256",
    "global_slot",
  ]) {
    expect(surfaceBytes).not.toContain(forbiddenPhrase);
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} frameIndex
 */
async function seekReplay(page, frameIndex) {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/replay/command",
    { timeout: 30_000 },
  );
  await page.locator("#replay-frame-slider").evaluate((element, value) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new TypeError("Replay slider is unavailable.");
    }
    element.value = String(value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
  }, frameIndex);
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  await expect(page.locator("#replay-frame-slider")).toHaveValue(String(frameIndex));
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
}

/**
 * Prove the synchronous clear boundary while one real authority response is
 * deliberately held after the server has accepted the audience command.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} oldScientificSentinel
 * @param {string} oldPresentationKey
 */
async function expectPendingAuthorityIsEmpty(
  page,
  oldScientificSentinel,
  oldPresentationKey,
) {
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  await expect(page.locator("#battlefield .agent")).toHaveCount(0);
  await expect(page.locator("[data-presentation-key]")).toHaveCount(0);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#visual-tooltip")).toBeHidden();
  const scientificSurfaces = [
    "#battlefield",
    "#roster",
    "#agent-details",
    "#pending-card",
    "#accepted-card",
    "#event-feed",
    "#diagnostics-card",
  ];
  const descendantsWith = (/** @type {string} */ attribute) =>
    scientificSurfaces.map((surface) => `${surface} [${attribute}]`).join(",");
  await expect(page.locator(descendantsWith("data-tooltip-owner"))).toHaveCount(0);
  await expect(page.locator(descendantsWith("aria-describedby"))).toHaveCount(0);
  await expect(page.locator(descendantsWith("aria-description"))).toHaveCount(0);
  await expect(page.locator("#replay-artifact-reference")).not.toHaveAttribute(
    "data-tooltip-owner",
  );
  await expect(page.locator("#replay-artifact-reference")).not.toHaveAttribute(
    "aria-description",
  );
  await expect(page.locator("[data-authoritative-available]")).toHaveCount(0);
  await expect(page.locator("#battlefield")).not.toHaveAttribute(
    "aria-activedescendant",
  );
  await expect(page.locator("#command-controlled-actor")).not.toHaveAttribute(
    "aria-label",
  );
  await expect(page.locator("#command-controlled-actor")).not.toHaveAttribute(
    "data-controlled-slot",
  );
  await expect(page.locator("#agent-details")).not.toHaveAttribute("data-tone");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("data-accent");
  await expect(page.locator("#selection-heading")).toHaveText("Agent Details");
  for (const selector of [
    "#command-deck",
    "#roster-details",
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#events-details",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
  }
  for (const selector of [
    "#roster",
    "#event-feed",
    "#pending-card",
    "#accepted-card",
    "#diagnostics-card",
    "#visual-tooltip",
  ]) {
    await expect(page.locator(selector)).not.toContainText(oldScientificSentinel);
  }
  const pendingDomBytes = JSON.stringify(await authorityDomSnapshot(page));
  expect(pendingDomBytes).not.toContain(oldScientificSentinel);
  expect(pendingDomBytes).not.toContain(oldPresentationKey);
  await expect(page.locator(".combat-effect")).toHaveCount(0);
  await expect(page.locator(".combat-choreography-routes > *")).toHaveCount(0);
  for (const selector of [
    "#recording-finish-button",
    "#recording-review-button",
    "#recording-retry-button",
    "#recording-save-as-input",
    "#recording-save-as-button",
    "#recording-discard-confirm-button",
    "#view-select",
  ]) {
    await expect(page.locator(selector)).toBeDisabled();
  }
}

test("all five real service leaves install and live authority clears atomically", async ({
  page,
}) => {
  if (!liveDebugger || !noSharedReplay) {
    throw new Error("Authorized-presentation test services are unavailable.");
  }

  await openProduct(page, liveDebugger.url, "live");
  const liveOracle = await expectInstalledLeaf(
    page,
    "researcher_live_debugger",
    "live_oracle",
  );
  const oldScientificSentinel = `Agent ID ${liveOracle.presentation.current_endpoint.scene.agents[0].public_agent_id}`;
  const oldPresentationKey =
    liveOracle.presentation.current_endpoint.scene.agents[0].presentation_key;
  const firstAgent = page.locator("#battlefield .agent").first();
  await firstAgent.hover();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await page.locator("#agent-details > summary").click();
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  const installedScientificBytes = JSON.stringify(await authorityDomSnapshot(page));
  expect(installedScientificBytes).toContain(oldScientificSentinel);
  expect(installedScientificBytes).toContain(oldPresentationKey);

  let releasePresentation = () => {};
  let markPresentationHeld = () => {};
  const presentationHeld = new Promise((resolve) => {
    markPresentationHeld = () => resolve(undefined);
  });
  const presentationRelease = new Promise((resolve) => {
    releasePresentation = () => resolve(undefined);
  });
  let holdNextPresentation = true;
  await page.route("**/api/presentation/frame", async (route) => {
    if (!holdNextPresentation) {
      await route.continue();
      return;
    }
    holdNextPresentation = false;
    const response = await route.fetch();
    markPresentationHeld();
    await presentationRelease;
    await route.fulfill({ response });
  });

  const switchAction = page.locator("#view-select").selectOption("pov");
  await presentationHeld;
  await expectPendingAuthorityIsEmpty(page, oldScientificSentinel, oldPresentationKey);
  releasePresentation();
  await switchAction;
  await page.unroute("**/api/presentation/frame");

  const liveAgent = await expectInstalledLeaf(
    page,
    "actor_pov_live_debugger",
    "live_no_shared_obs_agent_pov",
  );
  await expectAgentAuthoritySurface(page, liveAgent.presentation, [oldPresentationKey]);
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-description",
    "Inspect the authoritative scene. Researcher live view supports pointer control and targeting; Agent POV bodies remain passive and draft controls own actions.",
  );
  await expect(page.locator("#battlefield-instructions")).toContainText(
    "Agent POV bodies are inspectable and passive",
  );
  await expect(page.locator("#battlefield-instructions")).toContainText(
    "command draft controls",
  );

  const agentRecipientKey = liveAgent.presentation.authority.recipient_presentation_key;
  const agentBodyKeys = await page
    .locator("#battlefield .agent")
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  const passiveAgentIndex = agentBodyKeys.findIndex(
    (presentationKey) => presentationKey !== agentRecipientKey,
  );
  expect(passiveAgentIndex).toBeGreaterThanOrEqual(0);
  let agentPointerPostCount = 0;
  const countAgentPointerPosts = (
    /** @type {import("@playwright/test").Request} */ request,
  ) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/command"
    ) {
      agentPointerPostCount += 1;
    }
  };
  page.on("request", countAgentPointerPosts);
  await page.locator("#battlefield .agent").nth(passiveAgentIndex).click();
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  page.off("request", countAgentPointerPosts);
  expect(agentPointerPostCount).toBe(0);
  const agentPresentationAfterClick = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  expect(agentPresentationAfterClick.authority.recipient_presentation_key).toBe(
    agentRecipientKey,
  );
  const agentKeyboardRequest = page.waitForRequest(
    (request) =>
      request.method() === "POST" && new URL(request.url()).pathname === "/api/command",
  );
  const agentKeyboardResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await page.locator("#battlefield").focus();
  await page.keyboard.press("x");
  const [keyboardRequest, keyboardResponse] = await Promise.all([
    agentKeyboardRequest,
    agentKeyboardResponse,
  ]);
  expect(keyboardRequest.postDataJSON().command).toMatchObject({
    command_type: "keyboard",
    key: "x",
  });
  expect(keyboardResponse.status()).toBe(200);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const agentPresentationAfterKeyboard = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  expect(agentPresentationAfterKeyboard.authority.recipient_presentation_key).toBe(
    agentRecipientKey,
  );

  await page.locator("#view-select").selectOption("researcher");
  const restoredOracle = await expectInstalledLeaf(
    page,
    "researcher_live_debugger",
    "live_oracle",
  );
  const oracleCommandRequest = page.waitForRequest(
    (request) =>
      request.method() === "POST" && new URL(request.url()).pathname === "/api/command",
  );
  const oracleCommandResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await page.locator("#battlefield .agent").first().click();
  const [oracleRequest, oracleResponse] = await Promise.all([
    oracleCommandRequest,
    oracleCommandResponse,
  ]);
  expect(oracleRequest.postDataJSON().command).toMatchObject({
    command_type: "battlefield_pointer",
    button: "primary",
  });
  expect(oracleResponse.status()).toBe(200);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const stalePresentation = restoredOracle.presentation;
  let commandCount = 0;
  let frameGetCount = 0;
  let presentationGetCount = 0;
  const countRaceRequests = (
    /** @type {import("@playwright/test").Request} */ request,
  ) => {
    const path = new URL(request.url()).pathname;
    if (request.method() === "POST" && path === "/api/command") {
      commandCount += 1;
    }
    if (request.method() === "GET" && path === "/api/frame") {
      frameGetCount += 1;
    }
  };
  page.on("request", countRaceRequests);
  await page.route("**/api/presentation/frame", async (route) => {
    presentationGetCount += 1;
    const response = await route.fetch();
    if (presentationGetCount === 1) {
      await route.fulfill({ response, json: stalePresentation });
      return;
    }
    await route.fulfill({ response });
  });
  await page.locator("#live-ranges-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#notice")).toContainText("command completed once");
  expect(commandCount).toBe(1);
  expect(frameGetCount).toBe(1);
  expect(presentationGetCount).toBe(2);
  await page.unroute("**/api/presentation/frame");
  page.off("request", countRaceRequests);

  await openProduct(page, noSharedReplay.url, "replay");
  const replayOracle = await expectInstalledLeaf(
    page,
    "researcher_replay_viewer",
    "replay_oracle",
  );
  await expectAuthorizedIncomingTransitionDom(page, replayOracle.presentation);
  expect(replayOracle.presentation.latest_events).toBeNull();
  await seekReplay(page, 1);
  const replayOracleMiddle = await authenticatedGet(page, "/api/presentation/frame");
  expect(replayOracleMiddle.presentation_kind).toBe("replay_oracle");
  await expectAuthorizedIncomingTransitionDom(page, replayOracleMiddle);
  expect(replayOracleMiddle.latest_events.incoming_transition_id).toBe(
    await page.locator("#transition-value").textContent(),
  );
  await seekReplay(page, 0);
  const oracleOnlyValues = [
    replayOracle.presentation.source.source_artifact_id,
    replayOracle.presentation.source.source_artifact_digest_sha256,
    replayOracle.presentation.source.source_trajectory_content_digest_sha256,
    replayOracle.presentation.source.source_context_digest_sha256,
    ...replayOracle.presentation.current_endpoint.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => agent.presentation_key,
    ),
  ].filter((value) => typeof value === "string");

  await page.locator("#view-select").selectOption("pov");
  const replayAgent = await expectInstalledLeaf(
    page,
    "actor_pov_replay_viewer",
    "replay_no_shared_obs_agent_pov",
  );
  await expectAgentAuthoritySurface(page, replayAgent.presentation, oracleOnlyValues);
  expect(replayAgent.presentation.source.source_frame_index).toBe(0);
  await expectAuthorizedIncomingTransitionDom(page, replayAgent.presentation);
  for (const frameIndex of [
    1,
    replayAgent.presentation.source.source_final_frame_index,
  ]) {
    await seekReplay(page, frameIndex);
    const presentation = await authenticatedGet(page, "/api/presentation/frame");
    expect(presentation.presentation_kind).toBe("replay_no_shared_obs_agent_pov");
    expect(presentation.source.source_frame_index).toBe(frameIndex);
    await expectAuthorizedIncomingTransitionDom(page, presentation);
    expect(presentation.latest_events.incoming_recipient_transition_id).toBe(
      await page.locator("#transition-value").textContent(),
    );
  }
  expect(browserErrors.get(page) ?? []).toEqual([]);
});

test("real Shared replay installs frame zero, middle, final, then rejects retired diagnostics", async ({
  page,
}) => {
  if (!sharedReplay) {
    throw new Error("Shared replay service is unavailable.");
  }
  await openProduct(page, sharedReplay.url, "replay");

  const installed = [];
  for (const frameIndex of [0, 1, 2]) {
    if (frameIndex !== 0) {
      await seekReplay(page, frameIndex);
    }
    const leaf = await expectInstalledLeaf(
      page,
      "shared_obs_agent_pov_replay_viewer",
      "replay_shared_obs_agent_pov",
    );
    expect(leaf.transport.cursor).toMatchObject({
      frame_index: frameIndex,
      final_frame_index: 2,
    });
    expect(leaf.presentation.source.source_frame_index).toBe(frameIndex);
    expect(leaf.presentation.source.source_final_frame_index).toBe(2);
    expect(leaf.presentation.source.source_recipient_frame_id).toBe(
      leaf.transport.recipient_frame_id,
    );
    await expectAgentAuthoritySurface(page, leaf.presentation, []);
    await expectAuthorizedIncomingTransitionDom(page, leaf.presentation);
    installed.push(leaf);
  }

  expect(installed[0].presentation.latest_events).toBeNull();
  expect(installed[0].presentation.latest_transition).toBeNull();
  expect(installed[1].presentation.latest_events).not.toBeNull();
  expect(installed[1].presentation.latest_transition).not.toBeNull();
  expect(installed[1].presentation.replay_inspection).not.toBeNull();
  expect(installed[2].presentation.latest_events).not.toBeNull();
  expect(installed[2].presentation.latest_transition).not.toBeNull();
  expect(installed[2].presentation.replay_inspection).toBeNull();

  await seekReplay(page, 1);
  const middlePresentation = await authenticatedGet(page, "/api/presentation/frame");
  const incomingCount =
    middlePresentation.latest_events.events?.length ??
    middlePresentation.latest_events.deltas?.length ??
    0;
  await expect(page.locator("#event-feed .event-item")).toHaveCount(incomingCount);
  await expect(page.locator("#accepted-card .accepted-action-row")).toHaveCount(
    middlePresentation.latest_transition.action_rows.length,
  );
  await expect(page.locator("#accepted-card")).toHaveAttribute(
    "data-transition-id",
    middlePresentation.latest_transition.incoming_transition_id,
  );

  let frameRequestCount = 0;
  let presentationRequestCount = 0;
  const countRetiredRequests = (
    /** @type {import("@playwright/test").Request} */ request,
  ) => {
    const path = new URL(request.url()).pathname;
    if (request.method() === "GET" && path === "/api/frame") {
      frameRequestCount += 1;
    }
    if (request.method() === "GET" && path === "/api/presentation/frame") {
      presentationRequestCount += 1;
    }
  };
  page.on("request", countRetiredRequests);
  await page.route("**/api/frame", async (route) => {
    const response = await route.fetch();
    const current = await response.json();
    await route.fulfill({
      response,
      json: {
        schema_version: 1,
        frame_kind: "shared_obs_source_material_replay_viewer",
        revision: current.revision,
        projection: {
          researcher_hidden_truth: "retired-diagnostic-sentinel",
        },
      },
    });
  });
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  expect(frameRequestCount).toBe(1);
  expect(presentationRequestCount).toBe(1);
  await expect(page.locator("[data-presentation-key]")).toHaveCount(0);
  await expect(page.locator("#replay-timeline")).toBeVisible();
  await expect(page.locator("#replay-artifact-reference")).toHaveText(
    "Unavailable while authority is pending",
  );
  for (const selector of [
    "#replay-first-button",
    "#replay-back-ten-button",
    "#replay-previous-button",
    "#replay-play-pause-button",
    "#replay-next-button",
    "#replay-forward-ten-button",
    "#replay-last-button",
    "#replay-frame-slider",
  ]) {
    await expect(page.locator(selector)).toBeDisabled();
  }
  await expect(page.locator("body")).not.toContainText("retired-diagnostic-sentinel");
  await expect(page.locator("body")).not.toContainText(
    /Shared\s*Obs.*Source Material/iu,
  );
  await page.unroute("**/api/frame");
  page.off("request", countRetiredRequests);
  expect(browserErrors.get(page) ?? []).toEqual([]);
});

test("real death and respawn keep one opaque Oracle body identity", async ({
  page,
}) => {
  if (!deathReplay) {
    throw new Error("Death replay service is unavailable.");
  }
  await openProduct(page, deathReplay.url, "replay");
  const initial = await expectInstalledLeaf(
    page,
    "researcher_replay_viewer",
    "replay_oracle",
  );
  const subject = initial.presentation.current_endpoint.scene.agents.find(
    (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
  );
  if (!subject) {
    throw new Error("Death replay subject is absent from frame zero.");
  }
  const subjectSelector = `.agent[data-presentation-key="${subject.presentation_key}"]`;

  await seekReplay(page, 1);
  const corpse = await authenticatedGet(page, "/api/presentation/frame");
  expect(
    corpse.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
    ).presentation_key,
  ).toBe(subject.presentation_key);
  await expect(page.locator(subjectSelector)).toHaveAttribute("data-alive", "false");

  await seekReplay(page, 3);
  const respawn = await authenticatedGet(page, "/api/presentation/frame");
  expect(
    respawn.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
    ).presentation_key,
  ).toBe(subject.presentation_key);
  await expect(page.locator(subjectSelector)).toHaveAttribute("data-alive", "true");
  await expect(page.locator(subjectSelector)).toHaveAttribute(
    "data-spawn-shield-remaining",
    "3",
  );
  expect(browserErrors.get(page) ?? []).toEqual([]);
});
