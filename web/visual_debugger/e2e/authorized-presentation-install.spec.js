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
 * Hold only the initial authorized-presentation response after the synchronous
 * route bootstrap has established product identity. The pending shell must
 * remain product-correct without exposing any previously or partially joined
 * scientific presentation.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 * @param {"live" | "replay"} mode
 * @param {"combat_debugger" | "replay_viewer"} productKind
 */
async function openProductWithHeldInitialPresentation(page, url, mode, productKind) {
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

  const opening = openProduct(page, url, mode);
  try {
    await presentationHeld;
    await expect(page.locator("html")).toHaveAttribute(
      "data-product-kind",
      productKind,
    );
    await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", mode);
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "pending",
    );
    await expect(page.locator("#battlefield .agent")).toHaveCount(0);
    await expect(page.locator("[data-presentation-key]")).toHaveCount(0);
    await expect(page.locator("#step-value")).toHaveText("—");
    await expect(page.locator("#transition-value")).toHaveText("—");
    await expect(page.locator("#view-select")).toBeDisabled();
    await expect(page.locator("#exit-button")).toBeDisabled();

    const oppositeSelector =
      productKind === "replay_viewer" ? "[data-live-only]" : "[data-replay-only]";
    const visibleOppositeRoots = await page
      .locator(oppositeSelector)
      .evaluateAll((elements) =>
        elements
          .filter((element) => getComputedStyle(element).display !== "none")
          .map((element) => element.id || element.className),
      );
    expect(visibleOppositeRoots).toEqual([]);

    if (productKind === "replay_viewer") {
      await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
      await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
      await expect(page.locator("#replay-timeline")).toBeVisible();
      await expect(page.locator("#replay-ranges-button")).toBeVisible();
      await expect(page.locator("#replay-ranges-button")).toBeDisabled();
      await expect(page.locator("#replay-clear-reference-button")).toBeVisible();
      await expect(page.locator("#replay-clear-reference-button")).toBeDisabled();
      expect(
        await page
          .locator("#replay-timeline button")
          .evaluateAll((buttons) =>
            buttons.every(
              (button) => button instanceof HTMLButtonElement && button.disabled,
            ),
          ),
      ).toBe(true);
      await expect(page.locator("#battlefield-instructions")).toHaveText(
        "Replay is read-only. Activate an authorized agent to inspect current facts and its recorded outgoing action; use the timeline to change frames.",
      );
    } else {
      await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
      await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
      await expect(page.locator("#command-deck")).toBeVisible();
      await expect(page.locator("#live-ranges-button")).toBeDisabled();
      expect(
        await page
          .locator("#command-deck button")
          .evaluateAll((buttons) =>
            buttons.every(
              (button) => button instanceof HTMLButtonElement && button.disabled,
            ),
          ),
      ).toBe(true);
      await expect(page.locator("#battlefield-instructions")).toHaveText(
        "Live battlefield interaction is unavailable while authority is pending, offline, resynchronizing, shutting down, or terminal.",
      );
    }
  } finally {
    releasePresentation();
    await Promise.allSettled([opening]);
    await page.unroute("**/api/presentation/frame");
  }
  await opening;
}

/**
 * Prove one replay utility activation emits one exact command and installs the
 * response's successor revision before this helper returns.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} selector
 * @param {Readonly<Record<string, unknown>>} expectedCommand
 */
async function expectSingleReplayUtilityCommand(page, selector, expectedCommand) {
  /** @type {import("@playwright/test").Request[]} */
  const commandRequests = [];
  const recordCommand = (/** @type {import("@playwright/test").Request} */ request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      commandRequests.push(request);
    }
  };
  page.on("request", recordCommand);
  let responsePayload;
  try {
    const responsePromise = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname === "/api/replay/command",
      { timeout: 30_000 },
    );
    await page.locator(selector).click();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    responsePayload = await response.json();
    await expect(page.locator("#revision-value")).toHaveText(
      String(responsePayload.frame.revision),
    );
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
  } finally {
    page.off("request", recordCommand);
  }
  expect(commandRequests).toHaveLength(1);
  expect(commandRequests[0].postDataJSON().command).toEqual(expectedCommand);
  return responsePayload;
}

/**
 * Run one native activation and prove it crosses exactly one product command
 * boundary with the exact existing command body.
 *
 * @param {import("@playwright/test").Page} page
 * @param {"/api/command" | "/api/replay/command"} path
 * @param {() => Promise<void>} activate
 * @param {Readonly<Record<string, unknown>>} expectedCommand
 */
async function expectSingleActivationCommand(page, path, activate, expectedCommand) {
  /** @type {import("@playwright/test").Request[]} */
  const requests = [];
  const record = (/** @type {import("@playwright/test").Request} */ request) => {
    if (
      request.method() === "POST" &&
      ["/api/command", "/api/replay/command"].includes(new URL(request.url()).pathname)
    ) {
      requests.push(request);
    }
  };
  page.on("request", record);
  try {
    const responsePromise = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname === path,
      { timeout: 30_000 },
    );
    await activate();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
    await page.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
  } finally {
    page.off("request", record);
  }
  expect(requests).toHaveLength(1);
  expect(new URL(requests[0].url()).pathname).toBe(path);
  expect(requests[0].postDataJSON().command).toEqual(expectedCommand);
}

/**
 * Run one local interaction through a real browser event and prove neither
 * product command route was touched.
 *
 * @param {import("@playwright/test").Page} page
 * @param {() => Promise<void>} activate
 */
async function expectZeroCommandInteraction(page, activate) {
  /** @type {import("@playwright/test").Request[]} */
  const requests = [];
  const record = (/** @type {import("@playwright/test").Request} */ request) => {
    if (
      request.method() === "POST" &&
      ["/api/command", "/api/replay/command"].includes(new URL(request.url()).pathname)
    ) {
      requests.push(request);
    }
  };
  page.on("request", record);
  try {
    await activate();
    await page.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
  } finally {
    page.off("request", record);
  }
  expect(requests).toEqual([]);
}

/**
 * Exercise the one authorized activation through both public surfaces and all
 * three native activation gestures. Oracle cells cross one exact command
 * boundary; local inspection cells cross none.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{
 *   body: import("@playwright/test").Locator,
 *   row: import("@playwright/test").Locator,
 *   path: "/api/command" | "/api/replay/command" | null,
 *   command: Readonly<Record<string, unknown>> | null,
 * }} options
 */
async function expectNativeAgentActivationMatrix(page, { body, row, path, command }) {
  const scrollY = () => page.evaluate(() => window.scrollY);
  const cells = [
    () => row.click(),
    async () => {
      await row.focus();
      await row.press("Enter");
    },
    async () => {
      await row.focus();
      const before = await scrollY();
      await row.press(" ");
      expect(await scrollY()).toBe(before);
    },
    () => body.click(),
    async () => {
      await body.focus();
      await body.press("Enter");
    },
    async () => {
      await body.focus();
      const before = await scrollY();
      await body.press(" ");
      expect(await scrollY()).toBe(before);
    },
  ];
  for (const activate of cells) {
    if (path === null || command === null) {
      await expectZeroCommandInteraction(page, activate);
    } else {
      await expectSingleActivationCommand(page, path, activate, command);
    }
  }
}

/**
 * Terminal presentation bodies and roster activators remain descriptive but
 * cannot cross either product command route, even under forced DOM events.
 *
 * @param {import("@playwright/test").Page} page
 */
async function expectTerminalAgentActivationInert(page) {
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-label",
    "Read-only terminal replay battlefield snapshot. Agent inspection controls are unavailable.",
  );
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "This replay frame is terminal. Agent inspection controls are unavailable; use the timeline to review another frame.",
  );
  const bodies = page.locator("#battlefield .agent");
  const rows = page.locator("#roster .roster-primary-action");
  expect(await bodies.count()).toBeGreaterThan(0);
  await expect(bodies.first()).toHaveAttribute("role", "img");
  await expect(bodies.first()).toHaveAttribute("tabindex", "-1");
  expect(
    await rows.evaluateAll((buttons) =>
      buttons.every((button) => button instanceof HTMLButtonElement && button.disabled),
    ),
  ).toBe(true);
  const localState = () =>
    page.evaluate(() => ({
      selectedKeys: [
        ...document.querySelectorAll('#battlefield .agent[data-selected="true"]'),
      ].map((agent) => agent.getAttribute("data-presentation-key")),
      pressedRows: [...document.querySelectorAll("#roster .roster-primary-action")].map(
        (button) => button.getAttribute("aria-pressed"),
      ),
      ranges: [...document.querySelectorAll("#battlefield .range-ring")].map(
        (range) => ({
          kind: range.getAttribute("data-kind"),
          owner: range.getAttribute("data-presentation-key"),
        }),
      ),
      detailsOpen: document.querySelector("#agent-details")?.hasAttribute("open"),
      detailsAccent: document
        .querySelector("#agent-details")
        ?.getAttribute("data-accent"),
      selectionText: document.querySelector("#selection-card")?.textContent,
    }));
  const before = await localState();
  await expectZeroCommandInteraction(page, () => bodies.first().click({ force: true }));
  await expectZeroCommandInteraction(page, () =>
    bodies.first().dispatchEvent("keydown", { key: "Enter" }),
  );
  await expectZeroCommandInteraction(page, () =>
    bodies.first().dispatchEvent("keydown", { key: " " }),
  );
  expect(await localState()).toEqual(before);
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
 * Prove current selected-owner facts remain separate from the optional outgoing
 * inspection overlay. This reuses the real service trajectory already owned by
 * the five-leaf test; it adds no fixture server or viewport loop.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 */
async function expectReplayInspectionDom(page, presentation) {
  expect(presentation.product_kind).toBe("replay_viewer");
  const inspection = presentation.replay_inspection;
  const endpoint = presentation.current_endpoint;
  const axis = endpoint.action_axis;
  const scene = endpoint.scene ?? endpoint.parts?.scene;
  expect(Array.isArray(scene?.agents)).toBe(true);
  const ownerKey =
    inspection?.actor_presentation_key ?? axis?.owner_presentation_key ?? null;
  const ownerPublicId =
    inspection?.actor_public_agent_id ?? axis?.owner_public_agent_id ?? null;
  const owner =
    typeof ownerKey === "string"
      ? (scene.agents.find(
          (/** @type {Record<string, any>} */ agent) =>
            agent.presentation_key === ownerKey &&
            agent.public_agent_id === ownerPublicId,
        ) ?? null)
      : null;

  await expect(page.locator("#selection-heading")).toHaveText(
    "Comprehensive Agent Details",
  );
  await expect(page.locator('[data-layer="selection-legality"]')).toHaveAttribute(
    "aria-label",
    "Selection and exact actor-owned legality",
  );
  if (owner === null) {
    await expect(page.locator('#battlefield .agent[data-selected="true"]')).toHaveCount(
      0,
    );
    await expect(page.locator("#agent-details")).not.toHaveAttribute("data-accent");
    await expect(page.locator("#selection-card")).toContainText(
      "No authorized agent details are available.",
    );
    await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
    await expect(page.locator("#selection-card .selected-legality__lane")).toHaveCount(
      0,
    );
    await expect(page.locator("#selection-card .selected-outgoing-target")).toHaveCount(
      0,
    );
    await expect(page.locator("#battlefield .legality-dock")).toHaveCount(0);
    await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
    return;
  }

  const ownerBody = page.locator(
    `#battlefield .agent[data-presentation-key="${owner.presentation_key}"]`,
  );
  await expect(ownerBody).toHaveCount(1);
  await expect(ownerBody).toHaveAttribute("data-selected", "true");
  const classAccent = await ownerBody.getAttribute("data-class");
  expect(classAccent).not.toBeNull();
  await expect(page.locator("#agent-details")).toHaveAttribute(
    "data-accent",
    String(classAccent),
  );
  await expect(page.locator("#selection-card")).toContainText(
    `Agent ID ${owner.public_agent_id}`,
  );
  const expectedRangeKinds = [
    ["observation", owner.observation_radius],
    ["basic", owner.basic_interaction_radius],
    ["ultimate", owner.ultimate_interaction_radius],
  ]
    .filter(([, radius]) => typeof radius === "number" && radius > 0)
    .map(([kind]) => kind);
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual(
    expectedRangeKinds.map((kind) => ({
      kind,
      owner: owner.presentation_key,
    })),
  );

  if (inspection === null) {
    await expect(page.locator("#selection-card .selected-outgoing-target")).toHaveCount(
      0,
    );
    await expect(page.locator("#selection-card .selected-legality__lane")).toHaveCount(
      0,
    );
    await expect(page.locator("#battlefield .legality-dock")).toHaveCount(0);
    await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
    return;
  }

  const targetAction = inspection.accepted_action.target_action;
  const exactRow =
    inspection.decision_mask.target_use_ultimate_joint_mask[targetAction];
  const outgoingTarget = page.locator("#selection-card .selected-outgoing-target");
  await expect(outgoingTarget).toHaveCount(1);
  await expect(outgoingTarget).toHaveAttribute(
    "data-target-kind",
    inspection.accepted_target.target_kind,
  );
  await expect(outgoingTarget).toContainText(inspection.accepted_target.display_name);
  await expect(outgoingTarget).toContainText(
    inspection.accepted_target.target_kind === "no_target"
      ? "No target"
      : `Agent ID ${inspection.accepted_target.target_public_agent_id}`,
  );
  await expect(page.locator("#selection-card .selected-legality__lane h3")).toHaveText([
    `Basic Legality · Agent ID ${owner.public_agent_id}`,
    `Ultimate Legality · Agent ID ${owner.public_agent_id}`,
  ]);
  const legalityDock = page.locator(
    `#battlefield .legality-dock[data-presentation-key="${owner.presentation_key}"]`,
  );
  await expect(legalityDock).toHaveCount(1);
  await expect(legalityDock).toHaveAttribute(
    "aria-label",
    `Exact actor-owned legality for Agent ID ${owner.public_agent_id}`,
  );
  expect(
    await legalityDock.locator(".legality-pill").evaluateAll((pills) =>
      pills.map((pill) => ({
        available: pill.getAttribute("data-available"),
        lane: pill.getAttribute("data-lane"),
      })),
    ),
  ).toEqual([
    { available: String(exactRow[0]), lane: "0" },
    { available: String(exactRow[1]), lane: "1" },
  ]);
  const routeExpected =
    inspection.accepted_target.target_kind === "visible_authorized_agent" &&
    (inspection.combat_lane === "basic" || inspection.combat_lane === "ultimate");
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(
    routeExpected ? 1 : 0,
  );
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
 * @param {boolean} [activationEnabled]
 */
async function expectAgentAuthoritySurface(
  page,
  presentation,
  forbiddenValues,
  activationEnabled = true,
) {
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
  expect(bodyKeys.every(({ key }) => typeof key === "string")).toBe(true);
  expect(
    bodyKeys.every(({ role }) => role === (activationEnabled ? "button" : "img")),
  ).toBe(true);
  await expect(page.locator("#roster .roster-row--authorized")).toHaveCount(
    bodyKeys.length,
  );
  await expect(page.locator("#roster .roster-primary-action")).toHaveCount(
    bodyKeys.length,
  );
  if (activationEnabled) {
    await expect(page.locator("#roster .roster-primary-action").first()).toBeEnabled();
  } else {
    expect(
      await page
        .locator("#roster .roster-primary-action")
        .evaluateAll((buttons) =>
          buttons.every(
            (button) => button instanceof HTMLButtonElement && button.disabled,
          ),
        ),
    ).toBe(true);
  }
  await expect(page.locator("#roster .roster-actions")).toHaveCount(0);
  await expect(page.locator("#roster [data-role]")).toHaveCount(0);
  expect(bodyKeys.some(({ key }) => key === recipientKey)).toBe(true);

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
  const agentUtilitySelectors =
    presentation.product_kind === "replay_viewer"
      ? ["#replay-ranges-button", "#replay-clear-reference-button"]
      : ["#live-ranges-button"];
  for (const selector of agentUtilitySelectors) {
    await expect(page.locator(selector)).toHaveAttribute(
      "aria-description",
      /locally authorized|local inspected-agent/u,
    );
    await expect(page.locator(selector)).toHaveAttribute(
      "aria-description",
      /sends no (?:replay )?command/u,
    );
    await expect(page.locator(selector)).toHaveAttribute(
      "aria-description",
      /fixed recipient/u,
    );
    await expect(page.locator(selector)).not.toHaveAttribute(
      "aria-description",
      /Oracle View|server-authored/u,
    );
  }
}

/** @param {import("@playwright/test").Page} page */
async function expectAuthorizedRosterColors(page) {
  const colors = await page.locator("#roster").evaluate((roster) => {
    const resolvedColor = (/** @type {string} */ variable) => {
      const probe = document.createElement("span");
      probe.style.color = `var(${variable})`;
      document.body.append(probe);
      const color = getComputedStyle(probe).color;
      probe.remove();
      return color;
    };
    const classColors = Object.fromEntries(
      [...roster.querySelectorAll(".roster-id[data-class]")].map((element) => [
        element.getAttribute("data-class"),
        getComputedStyle(element).color,
      ]),
    );
    const expectedClassColors = Object.fromEntries(
      ["mage", "warrior", "hunter", "rogue", "priest"].map((className) => [
        className,
        resolvedColor(`--class-${className}`),
      ]),
    );
    const teamBorders = Object.fromEntries(
      [...roster.querySelectorAll(".roster-row--authorized[data-team]")].map(
        (element) => [
          element.getAttribute("data-team"),
          getComputedStyle(element).borderLeftColor,
        ],
      ),
    );
    return {
      classColors,
      expectedClassColors,
      teamBorders,
      expectedTeamBorders: {
        "team-a": resolvedColor("--team-a"),
        "team-b": resolvedColor("--team-b"),
      },
      mutedClassLabels: [
        ...roster.querySelectorAll(".roster-row--authorized .roster-class"),
      ].map((element) => getComputedStyle(element).color),
      expectedMuted: resolvedColor("--text-muted"),
    };
  });
  expect(colors.classColors).toEqual(colors.expectedClassColors);
  expect(colors.teamBorders).toEqual(colors.expectedTeamBorders);
  expect(new Set(colors.mutedClassLabels)).toEqual(new Set([colors.expectedMuted]));
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
  const pendingTooltip = await page.locator("#visual-tooltip").evaluate((tooltip) => ({
    hidden: tooltip instanceof HTMLElement && tooltip.hidden,
    kind: tooltip.getAttribute("data-tooltip-kind"),
  }));
  expect(pendingTooltip.hidden || pendingTooltip.kind === "control").toBe(true);
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
  await expect(page.locator("#selection-heading")).toHaveText(
    "Comprehensive Agent Details",
  );
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

  await openProductWithHeldInitialPresentation(
    page,
    liveDebugger.url,
    "live",
    "combat_debugger",
  );
  const liveOracle = await expectInstalledLeaf(
    page,
    "researcher_live_debugger",
    "live_oracle",
  );
  await expectAuthorizedRosterColors(page);
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Live Oracle View is interactive. Activate an authorized actor to control it; Shift-click selects an authorized target; right-click clears the target. Battlefield keyboard commands apply only while this surface has focus.",
  );
  await expect(page.locator("#pending-scope")).toHaveText(
    "This panel shows only the authorized pending draft for the next submission.",
  );
  await expect(
    page.locator("#battlefield-utilities > #live-ranges-button"),
  ).toHaveCount(1);
  await expect(page.locator("#live-ranges-button")).toHaveAttribute(
    "aria-description",
    "Toggle server-authored Oracle View range presentation.",
  );
  await expect(
    page.locator('#command-deck button[data-key="Tab"]:not([data-shift])'),
  ).toHaveAttribute(
    "aria-description",
    "Move Oracle View control to the next active actor.",
  );
  await expect(page.locator("#command-deck #live-ranges-button")).toHaveCount(0);
  await expect(page.locator("#battlefield-shell #live-ranges-button")).toHaveCount(0);
  await expect(page.locator("#live-visual-key > dt")).toHaveText([
    "Team A",
    "Team B",
    "Controlled",
    "Selected target",
  ]);
  await expect(page.locator("#live-visual-key")).not.toHaveAttribute("hidden", "");
  await expect(page.locator("#replay-visual-key")).toHaveAttribute("hidden", "");
  const oldScientificSentinel = `Agent ID ${liveOracle.presentation.current_endpoint.scene.agents[0].public_agent_id}`;
  const oldPresentationKey =
    liveOracle.presentation.current_endpoint.scene.agents[0].presentation_key;
  const controlledAgent = page.locator('#battlefield .agent[data-controlled="true"]');
  await expect(controlledAgent).toHaveCount(1);
  await controlledAgent.hover();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await expect(page.locator("#visual-tooltip")).toContainText("Controlled actor");
  await expect(page.locator("#visual-tooltip")).not.toContainText("Reference");
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
  await expect(page.locator("#battlefield")).not.toHaveAttribute(
    "aria-description",
    /.+/u,
  );
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-describedby",
    "battlefield-instructions",
  );
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Live Agent POV keeps one fixed recipient. Bodies are passive inspection targets; use the authorized draft controls to prepare that recipient's action.",
  );
  await page.locator("#help-button").click();
  await expect(page.locator("#help-dialog")).toBeVisible();
  const visibleLiveHelp = page.locator(
    "#help-dialog [data-live-help-mode]:not([hidden])",
  );
  await expect(visibleLiveHelp).toHaveCount(1);
  await expect(visibleLiveHelp).toHaveAttribute("data-live-help-mode", "agent");
  const agentHelpText = await visibleLiveHelp.innerText();
  expect(agentHelpText).toContain("fixed recipient");
  expect(agentHelpText).toContain("Toggle locally authorized ranges without a command");
  expect(agentHelpText).not.toContain("Cycle the controlled actor");
  expect(agentHelpText).not.toContain("Control the clicked authorized actor");
  expect(agentHelpText).not.toContain("Toggle Oracle View ranges");
  expect(agentHelpText).not.toContain("drafts persist when you cycle");
  await page.locator("#help-close-button").click();
  await expect(page.locator("#help-dialog")).toBeHidden();

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
  const passiveAgentKey = agentBodyKeys[passiveAgentIndex];
  expect(typeof passiveAgentKey).toBe("string");
  const liveAgentScene =
    liveAgent.presentation.current_endpoint.scene ??
    liveAgent.presentation.current_endpoint.parts?.scene;
  const passiveAgent = liveAgentScene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key === passiveAgentKey,
  );
  expect(passiveAgent).toBeTruthy();
  const passiveAgentBody = page.locator(
    `#battlefield .agent[data-presentation-key="${passiveAgentKey}"]`,
  );
  const passiveRowButton = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${passiveAgentKey}"]`,
  );
  await expectNativeAgentActivationMatrix(page, {
    body: passiveAgentBody,
    row: passiveRowButton,
    path: null,
    command: null,
  });
  await expect(passiveAgentBody).toHaveAttribute("data-selected", "true");
  await expect(passiveRowButton).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#selection-card")).toContainText(
    `Agent ID ${passiveAgent.public_agent_id}`,
  );
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual([
    { kind: "observation", owner: passiveAgentKey },
    { kind: "basic", owner: passiveAgentKey },
    { kind: "ultimate", owner: passiveAgentKey },
  ]);
  await expect(page.locator("#selection-card .selected-legality__lane")).toHaveCount(0);
  await expect(page.locator("#selection-card .selected-outgoing-target")).toHaveCount(
    0,
  );
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
  await expectAgentAuthoritySurface(page, liveAgent.presentation, [oldPresentationKey]);
  const recipientBody = page.locator(
    `#battlefield .agent[data-presentation-key="${agentRecipientKey}"]`,
  );
  await recipientBody.focus();
  await expectZeroCommandInteraction(page, () => recipientBody.press("Enter"));
  await expect(recipientBody).toHaveAttribute("data-selected", "true");
  const agentTabButtons = page.locator('#command-deck button[data-key="Tab"]');
  await expect(agentTabButtons).toHaveCount(2);
  expect(
    await agentTabButtons.evaluateAll((buttons) =>
      buttons.every((button) => button instanceof HTMLButtonElement && button.disabled),
    ),
  ).toBe(true);
  for (const button of await agentTabButtons.all()) {
    await expect(button).toHaveAttribute(
      "aria-description",
      /unavailable in Agent POV[\s\S]*native browser focus[\s\S]*fixed recipient/u,
    );
    await expect(button).not.toHaveAttribute(
      "aria-description",
      /Move Oracle View control/u,
    );
  }
  await page.locator("#battlefield").focus();
  await expectZeroCommandInteraction(page, () => page.keyboard.press("Tab"));
  await expect(page.locator("#battlefield")).not.toBeFocused();
  for (const button of await agentTabButtons.all()) {
    await expectZeroCommandInteraction(page, () =>
      button.evaluate((element) => {
        if (!(element instanceof HTMLButtonElement)) {
          throw new TypeError("Agent actor-cycle control is not a button.");
        }
        element.click();
      }),
    );
  }
  expect(
    (await authenticatedGet(page, "/api/presentation/frame")).authority
      .recipient_presentation_key,
  ).toBe(agentRecipientKey);
  const liveAgentRanges = page.locator("#live-ranges-button");
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");
  await expectZeroCommandInteraction(page, () => liveAgentRanges.click());
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expectZeroCommandInteraction(page, () => liveAgentRanges.click());
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");
  const currentAgentRanges = await page
    .locator('[data-layer="debug-range"] .range-ring')
    .evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    );
  expect(currentAgentRanges.length).toBeGreaterThan(0);
  expect(currentAgentRanges.every(({ owner }) => owner === agentRecipientKey)).toBe(
    true,
  );
  await page.locator("#battlefield").focus();
  await expectZeroCommandInteraction(page, () => page.keyboard.press("Shift+g"));
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual(currentAgentRanges);
  await expectZeroCommandInteraction(page, () => page.keyboard.press("g"));
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expectZeroCommandInteraction(page, () => page.keyboard.press("g"));
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual(currentAgentRanges);
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
  const oracleBody = page.locator("#battlefield .agent").first();
  const oracleKey = await oracleBody.getAttribute("data-presentation-key");
  const oracleAgent = restoredOracle.presentation.current_endpoint.scene.agents.find(
    (/** @type {Record<string, any>} */ agent) => agent.presentation_key === oracleKey,
  );
  const oracleIdentity =
    restoredOracle.presentation.current_endpoint.identity_directory.identities.find(
      (/** @type {Record<string, any>} */ identity) =>
        identity.public_agent_id === oracleAgent.public_agent_id,
    );
  const oracleSlot = (oracleIdentity.team_id - 1) * 5 + oracleIdentity.team_local_slot;
  const expectedOracleActivation = {
    command_type: "roster_selection",
    role: "control",
    global_slot: oracleSlot,
  };
  const oracleRow = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${oracleKey}"]`,
  );
  await expectNativeAgentActivationMatrix(page, {
    body: oracleBody,
    row: oracleRow,
    path: "/api/command",
    command: expectedOracleActivation,
  });
  const beforeScientificOwnerInput = await authenticatedGet(page, "/api/frame");
  const selectedKeysBeforeScientificOwnerInput = await page
    .locator('#battlefield .agent[data-selected="true"]')
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  const rangeOwner = page.locator("#battlefield .range-ring-owner").first();
  await expect(rangeOwner).toBeVisible();
  const scientificOwnerPoint = await page.locator("#battlefield").evaluate(() => {
    const owners = [
      ...document.querySelectorAll("#battlefield [data-tooltip-owner]"),
    ].filter((owner) => owner.closest(".agent") !== owner);
    for (const owner of owners) {
      const bounds = owner.getBoundingClientRect();
      for (const xRatio of [0.1, 0.25, 0.5, 0.75, 0.9]) {
        for (const yRatio of [0.1, 0.25, 0.5, 0.75, 0.9]) {
          const x = bounds.left + bounds.width * xRatio;
          const y = bounds.top + bounds.height * yRatio;
          const topOwner = document
            .elementsFromPoint(x, y)
            .map((element) => element.closest("[data-tooltip-owner]"))
            .find((candidate) => candidate !== null);
          if (topOwner === owner) {
            return { x, y };
          }
        }
      }
    }
    return null;
  });
  expect(scientificOwnerPoint).not.toBeNull();
  if (scientificOwnerPoint === null) {
    throw new Error("No exposed scientific tooltip owner was hit-testable.");
  }
  await expectZeroCommandInteraction(page, () =>
    page.mouse.click(scientificOwnerPoint.x, scientificOwnerPoint.y),
  );
  await rangeOwner.focus();
  await expectZeroCommandInteraction(page, () => rangeOwner.press("Enter"));
  await expectZeroCommandInteraction(page, () => rangeOwner.press(" "));
  await rangeOwner.focus();
  await expectZeroCommandInteraction(page, () => rangeOwner.press("Tab"));
  expect(
    await page
      .locator('#battlefield .agent[data-selected="true"]')
      .evaluateAll((agents) =>
        agents.map((agent) => agent.getAttribute("data-presentation-key")),
      ),
  ).toEqual(selectedKeysBeforeScientificOwnerInput);
  const afterScientificOwnerInput = await authenticatedGet(page, "/api/frame");
  expect(afterScientificOwnerInput.revision).toBe(beforeScientificOwnerInput.revision);
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

  await openProductWithHeldInitialPresentation(
    page,
    noSharedReplay.url,
    "replay",
    "replay_viewer",
  );
  const replayOracle = await expectInstalledLeaf(
    page,
    "researcher_replay_viewer",
    "replay_oracle",
  );
  await expect(page.locator("#events-details")).toHaveAttribute("open", "");
  await expect(page.locator("#event-feed .event-item")).toHaveCount(0);
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Replay is read-only. Activate an authorized agent to inspect current facts and its recorded outgoing action; use the timeline to change frames.",
  );
  await expect(
    page.locator("#battlefield-utilities > #replay-ranges-button"),
  ).toHaveCount(1);
  await expect(page.locator("#replay-ranges-button")).toHaveAttribute(
    "aria-description",
    "Toggle recorded Oracle View range presentation.",
  );
  await expect(
    page.locator("#battlefield-utilities > #replay-clear-reference-button"),
  ).toHaveCount(1);
  await expect(page.locator("#replay-clear-reference-button")).toHaveAttribute(
    "aria-description",
    "Clear the selected Oracle View agent and its inspection highlight.",
  );
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

  const replaySelectedBody = page.locator('#battlefield .agent[role="button"]').first();
  const replaySelectedKey = await replaySelectedBody.getAttribute(
    "data-presentation-key",
  );
  const replaySelectedAgent =
    replayOracle.presentation.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) =>
        agent.presentation_key === replaySelectedKey,
    );
  const replaySelectedIdentity =
    replayOracle.presentation.current_endpoint.identity_directory.identities.find(
      (/** @type {Record<string, any>} */ identity) =>
        identity.public_agent_id === replaySelectedAgent.public_agent_id,
    );
  const replaySelectedSlot =
    (replaySelectedIdentity.team_id - 1) * 5 + replaySelectedIdentity.team_local_slot;
  const replaySelectedRow = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${replaySelectedKey}"]`,
  );
  await expectNativeAgentActivationMatrix(page, {
    body: replaySelectedBody,
    row: replaySelectedRow,
    path: "/api/replay/command",
    command: {
      command_type: "select_agent",
      selected_global_slot: replaySelectedSlot,
    },
  });
  await expect(page.locator("#replay-clear-reference-button")).toBeEnabled();
  const replayOracleSelected = await authenticatedGet(page, "/api/presentation/frame");
  await expectReplayInspectionDom(page, replayOracleSelected);
  await page.locator('#battlefield .agent[data-selected="true"]').hover();
  await expect(page.locator("#visual-tooltip")).toContainText("Reference");
  await expect(page.locator("#visual-tooltip")).not.toContainText("Controlled actor");

  await expectAuthorizedIncomingTransitionDom(page, replayOracle.presentation);
  expect(replayOracle.presentation.latest_events).toBeNull();
  await page.locator("#events-details > summary").click();
  await expect(page.locator("#events-details")).not.toHaveAttribute("open", "");
  await page.locator("#technical-frame-details > summary").click();
  await expect(page.locator("#technical-frame-details")).toHaveAttribute("open", "");
  if ((await page.locator("#agent-details").getAttribute("open")) === "") {
    await page.locator("#agent-details > summary").click();
  }
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await replaySelectedRow.focus();
  let replaySeekRequestCount = 0;
  const countReplaySeek = (
    /** @type {import("@playwright/test").Request} */ request,
  ) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      replaySeekRequestCount += 1;
    }
  };
  page.on("request", countReplaySeek);
  try {
    await seekReplay(page, 1);
  } finally {
    page.off("request", countReplaySeek);
  }
  expect(replaySeekRequestCount).toBe(1);
  await expect(page.locator("#events-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#technical-frame-details")).toHaveAttribute("open", "");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect
    .poll(() =>
      page.evaluate(() => ({
        surface: document.activeElement?.closest("#roster")?.id ?? null,
        presentationKey:
          document.activeElement
            ?.closest("[data-presentation-key]")
            ?.getAttribute("data-presentation-key") ?? null,
      })),
    )
    .toEqual({ surface: "roster", presentationKey: replaySelectedKey });
  const replayOracleMiddle = await authenticatedGet(page, "/api/presentation/frame");
  expect(replayOracleMiddle.presentation_kind).toBe("replay_oracle");
  await expectAuthorizedIncomingTransitionDom(page, replayOracleMiddle);
  await expectReplayInspectionDom(page, replayOracleMiddle);
  expect(replayOracleMiddle.latest_events.incoming_transition_id).toBe(
    await page.locator("#transition-value").textContent(),
  );
  await seekReplay(page, replayOracle.presentation.source.source_final_frame_index);
  const replayOracleFinal = await authenticatedGet(page, "/api/presentation/frame");
  expect(replayOracleFinal.replay_inspection).toBeNull();
  await expectReplayInspectionDom(page, replayOracleFinal);
  await expectTerminalAgentActivationInert(page);
  const nextShowRanges = replayOracle.transport.show_ranges !== true;
  await expectSingleReplayUtilityCommand(page, "#replay-ranges-button", {
    command_type: "set_ranges",
    show_ranges: nextShowRanges,
  });
  await expect(page.locator("#replay-ranges-button")).toHaveAttribute(
    "aria-pressed",
    String(nextShowRanges),
  );
  await expectSingleReplayUtilityCommand(page, "#replay-clear-reference-button", {
    command_type: "select_agent",
    selected_global_slot: null,
  });
  const replayOracleFinalUnselected = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  await expectReplayInspectionDom(page, replayOracleFinalUnselected);
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
  await expectReplayInspectionDom(page, replayAgent.presentation);
  await page.locator('#battlefield .agent[data-selected="true"]').hover();
  await expect(page.locator("#visual-tooltip")).toContainText("Inspected agent");
  await expect(page.locator("#visual-tooltip")).not.toContainText("Selected target");
  await expect(page.locator("#visual-tooltip")).not.toContainText("Reference");
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Replay Agent POV is read-only and keeps one fixed recipient. Activate an authorized visible body to inspect current facts; the replay authority does not change.",
  );
  const replayAgentRecipientKey =
    replayAgent.presentation.authority.recipient_presentation_key;
  const replayAgentBodies = page.locator("#battlefield .agent[role=button]");
  const replayAgentBodyKeys = await replayAgentBodies.evaluateAll((agents) =>
    agents.map((agent) => agent.getAttribute("data-presentation-key")),
  );
  const replayAgentLocalKey = replayAgentBodyKeys.find(
    (key) => key !== replayAgentRecipientKey,
  );
  expect(typeof replayAgentLocalKey).toBe("string");
  const replayAgentScene =
    replayAgent.presentation.current_endpoint.scene ??
    replayAgent.presentation.current_endpoint.parts?.scene;
  const replayAgentLocal = replayAgentScene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key === replayAgentLocalKey,
  );
  expect(replayAgentLocal).toBeTruthy();
  const replayAgentLocalRow = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${replayAgentLocalKey}"]`,
  );
  const replayAgentLocalBody = page.locator(
    `#battlefield .agent[data-presentation-key="${replayAgentLocalKey}"]`,
  );
  await expectNativeAgentActivationMatrix(page, {
    body: replayAgentLocalBody,
    row: replayAgentLocalRow,
    path: null,
    command: null,
  });
  await expect(replayAgentLocalBody).toHaveAttribute("data-selected", "true");
  await expect(replayAgentLocalRow).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#selection-card")).toContainText(
    `Agent ID ${replayAgentLocal.public_agent_id}`,
  );
  await expect(page.locator("#selection-card .selected-legality__lane")).toHaveCount(0);
  await expect(page.locator("#selection-card .selected-outgoing-target")).toHaveCount(
    0,
  );
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual([
    { kind: "observation", owner: replayAgentLocalKey },
    { kind: "basic", owner: replayAgentLocalKey },
    { kind: "ultimate", owner: replayAgentLocalKey },
  ]);
  await expectAgentAuthoritySurface(page, replayAgent.presentation, oracleOnlyValues);
  const replayAgentRanges = page.locator("#replay-ranges-button");
  await expect(replayAgentRanges).toHaveAttribute("aria-pressed", "true");
  await expectZeroCommandInteraction(page, () => replayAgentRanges.click());
  await expect(replayAgentRanges).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expectZeroCommandInteraction(page, () => replayAgentRanges.click());
  await expectZeroCommandInteraction(page, () =>
    page.locator("#replay-clear-reference-button").click(),
  );
  await expect(page.locator('#battlefield .agent[data-selected="true"]')).toHaveCount(
    0,
  );
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#replay-clear-reference-button")).toBeDisabled();
  const replayAgentPresentationAfterLocalActions = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  expect(
    replayAgentPresentationAfterLocalActions.authority.recipient_presentation_key,
  ).toBe(replayAgentRecipientKey);
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expectZeroCommandInteraction(page, () =>
    page
      .locator(
        `#roster .roster-primary-action[data-presentation-key="${replayAgentRecipientKey}"]`,
      )
      .click(),
  );
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
    await expectReplayInspectionDom(page, presentation);
    expect(presentation.latest_events.incoming_recipient_transition_id).toBe(
      await page.locator("#transition-value").textContent(),
    );
    if (frameIndex === replayAgent.presentation.source.source_final_frame_index) {
      await expectTerminalAgentActivationInert(page);
    }
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
    await expectAgentAuthoritySurface(page, leaf.presentation, [], frameIndex < 2);
    await expectAuthorizedIncomingTransitionDom(page, leaf.presentation);
    await expectReplayInspectionDom(page, leaf.presentation);
    if (frameIndex === 0) {
      await expect(page.locator("#events-details")).toHaveAttribute("open", "");
      await expect(page.locator("#event-feed .event-item")).toHaveCount(0);
      const recipientKey = leaf.presentation.authority.recipient_presentation_key;
      const scene =
        leaf.presentation.current_endpoint.scene ??
        leaf.presentation.current_endpoint.parts?.scene;
      const localAgent = scene.agents.find(
        (/** @type {Record<string, any>} */ agent) =>
          agent.presentation_key !== recipientKey,
      );
      expect(localAgent).toBeTruthy();
      const localKey = localAgent.presentation_key;
      const localBody = page.locator(
        `#battlefield .agent[data-presentation-key="${localKey}"]`,
      );
      const localRow = page.locator(
        `#roster .roster-primary-action[data-presentation-key="${localKey}"]`,
      );
      await expectNativeAgentActivationMatrix(page, {
        body: localBody,
        row: localRow,
        path: null,
        command: null,
      });
      await expect(localBody).toHaveAttribute("data-selected", "true");
      await expect(localRow).toHaveAttribute("aria-pressed", "true");
      await expect(page.locator("#selection-card")).toContainText(
        `Agent ID ${localAgent.public_agent_id}`,
      );
      await expect(
        page.locator("#selection-card .selected-legality__lane"),
      ).toHaveCount(0);
      await expect(
        page.locator("#selection-card .selected-outgoing-target"),
      ).toHaveCount(0);
      await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
      expect(
        await page
          .locator('[data-layer="debug-range"] .range-ring')
          .evaluateAll((ranges) =>
            ranges.map((range) => ({
              kind: range.getAttribute("data-kind"),
              owner: range.getAttribute("data-presentation-key"),
            })),
          ),
      ).toEqual([
        { kind: "observation", owner: localKey },
        { kind: "basic", owner: localKey },
        { kind: "ultimate", owner: localKey },
      ]);
      await expectAgentAuthoritySurface(page, leaf.presentation, []);
      const rangesButton = page.locator("#replay-ranges-button");
      await expectZeroCommandInteraction(page, () => rangesButton.click());
      await expect(rangesButton).toHaveAttribute("aria-pressed", "false");
      await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(
        0,
      );
      await expectZeroCommandInteraction(page, () => rangesButton.click());
      await expect(rangesButton).toHaveAttribute("aria-pressed", "true");
      expect(
        await page
          .locator('[data-layer="debug-range"] .range-ring')
          .evaluateAll((ranges) =>
            ranges.map((range) => ({
              kind: range.getAttribute("data-kind"),
              owner: range.getAttribute("data-presentation-key"),
            })),
          ),
      ).toEqual([
        { kind: "observation", owner: localKey },
        { kind: "basic", owner: localKey },
        { kind: "ultimate", owner: localKey },
      ]);
      await expectZeroCommandInteraction(page, () =>
        page.locator("#replay-clear-reference-button").click(),
      );
      await expect(
        page.locator('#battlefield .agent[data-selected="true"]'),
      ).toHaveCount(0);
      await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(
        0,
      );
      await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
      await expect(page.locator("#replay-clear-reference-button")).toBeDisabled();
      await expectZeroCommandInteraction(page, () =>
        page
          .locator(
            `#roster .roster-primary-action[data-presentation-key="${recipientKey}"]`,
          )
          .click(),
      );
      const afterLocalActions = await authenticatedGet(page, "/api/presentation/frame");
      expect(afterLocalActions.authority.recipient_presentation_key).toBe(recipientKey);
    }
    if (frameIndex === 2) {
      await expectTerminalAgentActivationInert(page);
    }
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
