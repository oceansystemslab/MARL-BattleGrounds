import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";

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
async function currentRevision(page) {
  return Number(await page.locator("#revision-value").textContent());
}

/**
 * @param {import("@playwright/test").Page} page
 * @returns {Promise<number>}
 */
async function currentStep(page) {
  return Number(await page.locator("#step-value").textContent());
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
  const editRevision = await currentRevision(page);
  await movement.click();
  await expect(page.locator("#revision-value")).toHaveText(String(editRevision + 1));
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(selectedMovement).toHaveAttribute("aria-pressed", "true");
  await expect(draftLabel).not.toHaveText(draftBefore);
  await expect.poll(() => commands.length).toBe(1);

  const successorRevision = await currentRevision(page);
  const successorStep = await currentStep(page);
  const transitionBefore = await page.locator("#transition-value").textContent();
  await page.locator("#submit-turn-button").evaluate((button) => {
    if (!(button instanceof HTMLButtonElement)) {
      throw new TypeError("Submit button is unavailable.");
    }
    button.click();
    button.click();
  });
  await expect(page.locator("#revision-value")).toHaveText(
    String(successorRevision + 1),
    { timeout: 120_000 },
  );
  await expect(page.locator("#step-value")).toHaveText(String(successorStep + 1));
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
  await expect(page.locator("#accepted-card")).toHaveAttribute(
    "data-transition-id",
    transitionId ?? "",
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
  const baseRevision = await currentRevision(page);
  await expect(stalePage.locator("#revision-value")).toHaveText(String(baseRevision));

  const ranges = page.locator("#live-ranges-button");
  const staleRanges = stalePage.locator("#live-ranges-button");
  const initialRangesPressed = await ranges.getAttribute("aria-pressed");
  expect(["true", "false"]).toContain(initialRangesPressed);
  if (initialRangesPressed === null) {
    throw new Error("Live range control did not publish pressed state.");
  }

  await ranges.click();
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 1));
  await expect(ranges).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed === "true" ? "false" : "true",
  );
  await expect(staleRanges).toHaveAttribute("aria-pressed", initialRangesPressed);

  await staleRanges.click();
  await expect(stalePage.locator("#revision-value")).toHaveText(
    String(baseRevision + 1),
  );
  await expect(stalePage.locator("#notice")).toContainText("stale");
  await expect(staleRanges).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed === "true" ? "false" : "true",
  );
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 1));

  await staleRanges.click();
  await expect(stalePage.locator("#revision-value")).toHaveText(
    String(baseRevision + 2),
  );
  await expect(staleRanges).toHaveAttribute("aria-pressed", initialRangesPressed);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 2));
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
  const baseRevision = await currentRevision(page);
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
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 1));
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect.poll(() => interceptedCommands).toBe(1);
  await page.unroute("**/api/command");

  await page.locator("#live-ranges-button").click();
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 2));
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
});
