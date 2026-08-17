import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";

const SCIENTIFIC_DISCLOSURES = Object.freeze([
  "#command-deck",
  "#roster-details",
  "#agent-details",
  "#pending-turn-details",
  "#latest-transition-details",
  "#events-details",
  "#technical-frame-details",
]);

test.beforeAll(async () => {
  const started = await startDebugger();
  serverProcess = started.process;
  debuggerUrl = started.url;
});

test.afterAll(async () => {
  const child = serverProcess;
  serverProcess = null;
  await stopDebugger(child);
});

/**
 * @param {import("@playwright/test").Page} page
 * @returns {Promise<number>}
 */
async function currentStep(page) {
  return Number(await page.locator("#step-value").textContent());
}

/**
 * Hold exactly one real authorized-presentation response after the browser has
 * synchronously crossed into pending authority.
 *
 * @param {import("@playwright/test").Page} page
 */
async function holdNextPresentation(page) {
  let markHeld = () => {};
  let releaseResponse = () => {};
  let released = false;
  const held = new Promise((resolve) => {
    markHeld = () => resolve(undefined);
  });
  const release = new Promise((resolve) => {
    releaseResponse = () => resolve(undefined);
  });
  const handler = async (/** @type {import("@playwright/test").Route} */ route) => {
    const response = await route.fetch();
    markHeld();
    await release;
    await route.fulfill({ response });
  };
  await page.route("**/api/presentation/frame", handler, { times: 1 });
  return {
    held,
    release() {
      if (!released) {
        released = true;
        releaseResponse();
      }
    },
    async dispose() {
      if (!released) {
        released = true;
        releaseResponse();
      }
      await page.unroute("**/api/presentation/frame", handler);
    },
  };
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} selector
 * @param {boolean} open
 */
async function setDisclosureOpen(page, selector, open) {
  const disclosure = page.locator(selector);
  if ((await disclosure.getAttribute("open")) !== (open ? "" : null)) {
    await disclosure.locator(":scope > summary").click();
  }
  if (open) {
    await expect(disclosure).toHaveAttribute("open", "");
  } else {
    await expect(disclosure).not.toHaveAttribute("open", "");
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{frameZeroEvents?: boolean}} [options]
 */
async function expectExactDisclosureDefaults(page, { frameZeroEvents = false } = {}) {
  await expect(page.locator("#command-deck")).toHaveAttribute("open", "");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  await expect(page.locator("#events-details")).toHaveAttribute("open", "");
  for (const selector of [
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
  }
  if (frameZeroEvents) {
    await expect(page.locator("#step-value")).toHaveText("0");
    await expect(page.locator("#event-count")).toHaveText("0");
    await expect(page.locator("#event-feed .event-item")).toHaveCount(0);
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} previousPresentationKey
 */
async function expectPendingDisclosureAuthority(page, previousPresentationKey) {
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  for (const selector of SCIENTIFIC_DISCLOSURES) {
    const disclosure = page.locator(selector);
    await expect(disclosure).not.toHaveAttribute("open", "");
    await expect(disclosure).toHaveAttribute("inert", "");
    await expect(disclosure.locator(":scope > summary")).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    await expect(disclosure.locator(":scope > summary")).toHaveAttribute(
      "tabindex",
      "-1",
    );
  }
  await expect(page.locator("[data-presentation-key]")).toHaveCount(0);
  await expect(page.locator("[data-authoritative-available]")).toHaveCount(0);
  await expect(page.locator("#battlefield [data-tooltip-owner]")).toHaveCount(0);
  await expect(page.locator("#roster [data-tooltip-owner]")).toHaveCount(0);
  expect(
    await page
      .locator("body")
      .evaluate(
        (body, oldKey) => !body.innerHTML.includes(String(oldKey)),
        previousPresentationKey,
      ),
  ).toBe(true);
  expect(
    await page.evaluate(
      () => document.activeElement?.closest("[data-presentation-key]") === null,
    ),
  ).toBe(true);
}

test("authorized draft edit and rapid Submit install exactly one successor", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commands = [];
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    commands.push(payload?.command ?? {});
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-inspection-state",
    "live_editable",
  );
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-submission-scope",
    "joint_turn",
  );
  await expect(page.locator("#command-controlled-actor")).toContainText(
    "Actor · Agent ID ",
  );
  await expect(
    page.locator("#battlefield .agent[data-presentation-key]"),
  ).not.toHaveCount(0);

  const draftLabel = page.locator("#pending-card .action-card__label");
  const draftBefore = await draftLabel.textContent();
  if (draftBefore === null || draftBefore.length === 0) {
    throw new Error("Authorized draft label is unavailable.");
  }
  const movement = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"]:not([aria-pressed="true"])',
    )
    .first();
  await expect(movement).toBeVisible();
  const movementAction = await movement.getAttribute("data-move-action");
  if (movementAction === null) {
    throw new Error("Authorized movement control did not publish its action identity.");
  }
  const selectedMovement = page.locator(
    `#command-deck button[data-move-action=${JSON.stringify(movementAction)}]`,
  );
  await movement.click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(selectedMovement).toHaveAttribute("aria-pressed", "true");
  await expect(draftLabel).not.toHaveText(draftBefore);
  await expect.poll(() => commands.length).toBe(1);

  const successorStep = await currentStep(page);
  const transitionBefore = await page.locator("#transition-value").textContent();
  await page.locator("#submit-turn-button").evaluate((button) => {
    if (!(button instanceof HTMLButtonElement)) {
      throw new TypeError("Submit button is unavailable.");
    }
    button.click();
    button.click();
  });
  await expect(page.locator("#step-value")).toHaveText(String(successorStep + 1), {
    timeout: 120_000,
  });
  await expect.poll(() => commands.length).toBe(2);
  expect(
    commands.filter(
      (command) =>
        command.command_type === "keyboard" &&
        String(command.key).toLowerCase() === "enter",
    ),
  ).toHaveLength(1);

  const transitionId = await page.locator("#transition-value").textContent();
  expect(transitionId).not.toBeNull();
  expect(transitionId).not.toBe("—");
  expect(transitionId).not.toBe(transitionBefore);
  await expect(page.locator("#accepted-card")).not.toHaveAttribute(
    "data-transition-id",
    /.+/u,
  );
  await expect(page.locator("#accepted-card .accepted-action-row")).not.toHaveCount(0);
  await expect(page.locator("#accepted-card")).toContainText("Submitted");
  await expect(page.locator("#accepted-card")).toContainText("Accepted");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await page.unroute("**/api/command");
});

test("a stale tab adopts the latest ranges state without replaying its command", async ({
  context,
  page,
}) => {
  await page.goto(debuggerUrl);
  const stalePage = await context.newPage();
  await stalePage.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(stalePage.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(stalePage.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const ranges = page.locator("#live-ranges-button");
  const staleRanges = stalePage.locator("#live-ranges-button");
  const initialRangesPressed = await ranges.getAttribute("aria-pressed");
  expect(["true", "false"]).toContain(initialRangesPressed);
  if (initialRangesPressed === null) {
    throw new Error("Live range control did not publish pressed state.");
  }

  await ranges.click();
  await expect(ranges).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed === "true" ? "false" : "true",
  );
  await expect(staleRanges).toHaveAttribute("aria-pressed", initialRangesPressed);

  await staleRanges.click();
  await expect(stalePage.locator("#notice")).toContainText("stale");
  await expect(staleRanges).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed === "true" ? "false" : "true",
  );
  await staleRanges.click();
  await expect(staleRanges).toHaveAttribute("aria-pressed", initialRangesPressed);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#live-ranges-button")).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed,
  );
  await stalePage.close();
});

test("a lost applied response requires GET resync and never replays submit", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-inspection-state",
    "live_editable",
  );
  const baseStep = await currentStep(page);
  let interceptedCommands = 0;
  let appliedStatus = 0;
  await page.route("**/api/command", async (route) => {
    interceptedCommands += 1;
    const response = await route.fetch();
    appliedStatus = response.status();
    await route.abort("failed");
  });

  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect.poll(() => appliedStatus).toBe(200);
  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect.poll(() => interceptedCommands).toBe(1);

  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect.poll(() => interceptedCommands).toBe(1);
  await page.unroute("**/api/command");

  const ranges = page.locator("#live-ranges-button");
  const rangesBefore = await ranges.getAttribute("aria-pressed");
  await ranges.click();
  await expect(ranges).toHaveAttribute(
    "aria-pressed",
    rangesBefore === "true" ? "false" : "true",
  );
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
});

test("native panels preserve user state only within exact authority", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#view-select")).toHaveValue("researcher");
  await expectExactDisclosureDefaults(page, {
    frameZeroEvents: (await currentStep(page)) === 0,
  });

  const firstActivation = page
    .locator(
      '#roster .roster-primary-action[data-presentation-key]:not([aria-pressed="true"])',
    )
    .first();
  await expect(firstActivation).toBeVisible();
  const firstCommand = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await firstActivation.press("Enter");
  expect((await firstCommand).status()).toBe(200);
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");

  await setDisclosureOpen(page, "#agent-details", false);
  const ranges = page.locator("#live-ranges-button");
  if ((await ranges.getAttribute("aria-pressed")) !== "true") {
    await ranges.click();
    await expect(ranges).toHaveAttribute("aria-pressed", "true");
  }
  await expect(page.locator("#battlefield .range-ring-owner").first()).toBeVisible();
  await page.locator("#battlefield .range-ring-owner").first().dispatchEvent("click");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");

  await setDisclosureOpen(page, "#events-details", false);
  await setDisclosureOpen(page, "#technical-frame-details", true);
  const rosterBody = page.locator("#roster-details-body");
  let savedScrollTop = await rosterBody.evaluate((body) => {
    body.style.height = "56px";
    body.style.maxHeight = "56px";
    body.style.overflowY = "scroll";
    body.scrollTop = 24;
    return body.scrollTop;
  });
  expect(savedScrollTop).toBeGreaterThan(0);

  const focusedAction = page
    .locator(
      '#roster .roster-primary-action[data-presentation-key]:not([aria-pressed="true"])',
    )
    .first();
  const focusedPresentationKey = await focusedAction.getAttribute(
    "data-presentation-key",
  );
  if (focusedPresentationKey === null) {
    throw new Error("Authorized roster action has no presentation key.");
  }
  await focusedAction.focus();
  savedScrollTop = await rosterBody.evaluate((body) => body.scrollTop);

  /** @type {import("@playwright/test").Request[]} */
  const commandRequests = [];
  const recordCommands = (
    /** @type {import("@playwright/test").Request} */ request,
  ) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/command"
    ) {
      commandRequests.push(request);
    }
  };
  page.on("request", recordCommands);
  const heldCommandPresentation = await holdNextPresentation(page);
  try {
    await focusedAction.press("Enter");
    await heldCommandPresentation.held;
    await expectPendingDisclosureAuthority(page, focusedPresentationKey);
    heldCommandPresentation.release();
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
  } finally {
    heldCommandPresentation.release();
    await heldCommandPresentation.dispose();
    page.off("request", recordCommands);
  }
  expect(commandRequests).toHaveLength(1);
  expect(commandRequests[0].postDataJSON().command).toMatchObject({
    command_type: "roster_selection",
    role: "control",
  });
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  await expect(page.locator("#events-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#technical-frame-details")).toHaveAttribute("open", "");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect
    .poll(() => rosterBody.evaluate((body) => body.scrollTop))
    .toBe(savedScrollTop);
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
    .toEqual({ surface: "roster", presentationKey: focusedPresentationKey });

  const heldRefresh = await holdNextPresentation(page);
  try {
    await page.locator("#reconnect-button").evaluate((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        throw new TypeError("Reconnect button is unavailable.");
      }
      button.click();
    });
    await heldRefresh.held;
    await expectPendingDisclosureAuthority(page, focusedPresentationKey);
    await page.locator("#help-button").click();
    await expect(page.locator("#help-dialog")).toHaveAttribute("open", "");
    heldRefresh.release();
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
  } finally {
    heldRefresh.release();
    await heldRefresh.dispose();
  }
  await expect
    .poll(() =>
      page.evaluate(() => document.activeElement?.closest("dialog")?.id ?? null),
    )
    .toBe("help-dialog");
  await page.locator("#help-close-button").click();
  await expect(page.locator("#help-dialog")).not.toHaveAttribute("open", "");
  await expect
    .poll(() => rosterBody.evaluate((body) => body.scrollTop))
    .toBe(savedScrollTop);

  await setDisclosureOpen(page, "#roster-details", false);
  await page.locator("#view-select").selectOption("pov");
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expectExactDisclosureDefaults(page);
  await expect.poll(() => rosterBody.evaluate((body) => body.scrollTop)).toBe(0);
  await expect
    .poll(() =>
      page.evaluate(
        (oldKey) =>
          document.activeElement
            ?.closest("[data-presentation-key]")
            ?.getAttribute("data-presentation-key") !== oldKey,
        focusedPresentationKey,
      ),
    )
    .toBe(true);

  await setDisclosureOpen(page, "#events-details", false);
  await setDisclosureOpen(page, "#technical-frame-details", true);
  await page.locator("#view-select").selectOption("researcher");
  await expect(page.locator("#view-select")).toHaveValue("researcher");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expectExactDisclosureDefaults(page);
  await expect.poll(() => rosterBody.evaluate((body) => body.scrollTop)).toBe(0);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
});
