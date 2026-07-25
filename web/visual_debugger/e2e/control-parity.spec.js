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

test("battlefield commands preserve UI and simulator revision boundaries", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText("0");
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(page.locator("#battlefield .agent")).toHaveCount(10);
  await expect(page.locator("#roster .roster-row")).toHaveCount(10);
  expect(page.url()).not.toContain("token=");

  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Move northwest" })).toBeFocused();
  await expect(page.locator("#revision-value")).toHaveText("0");
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("button", { name: "Move north", exact: true }),
  ).toBeFocused();

  await battlefield.evaluate((element) => {
    const agent = element.querySelector('.agent[data-slot="0"]');
    const transientLayer = element.querySelector('[data-layer="transient-events"]');
    if (!agent || !transientLayer) {
      throw new Error("Retained battlefield layers were not installed.");
    }
    agent.setAttribute("data-retained-probe", "agent-0");
    const probe = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    probe.setAttribute("data-retained-probe", "transient-event");
    transientLayer.append(probe);
  });
  await battlefield.focus();
  await page.keyboard.press("d");
  await expect(page.locator("#revision-value")).toHaveText("1");
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(
    page.locator('#battlefield .agent[data-retained-probe="agent-0"]'),
  ).toHaveCount(1);
  await expect(
    page.locator(
      '#battlefield [data-layer="transient-events"] [data-retained-probe="transient-event"]',
    ),
  ).toHaveCount(1);

  await battlefield.evaluate((element) => {
    element.dispatchEvent(
      new KeyboardEvent("keydown", {
        bubbles: true,
        cancelable: true,
        key: "Enter",
        repeat: true,
      }),
    );
  });
  await expect(page.locator("#notice")).toContainText(
    "Repeated submission input ignored.",
  );
  await expect(page.locator("#revision-value")).toHaveText("1");
  await expect(page.locator("#step-value")).toHaveText("0");

  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#revision-value")).toHaveText("2", {
    timeout: 120_000,
  });
  await expect(page.locator("#step-value")).toHaveText("1");
  await expect(page.locator("#transition-value")).toHaveText("1");

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText("2");
  await expect(page.locator("#step-value")).toHaveText("1");
  expect(page.url()).not.toContain("token=");
});

test("pointer, roster, toolbar, and command-deck controls use the live service", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  let revision = await currentRevision(page);

  const targetButton = page.getByRole("button", { name: "Target id_6" });
  await expect(targetButton).toBeVisible();
  await targetButton.evaluate((element) => {
    element.setAttribute("data-retained-probe", "target-6");
  });
  await targetButton.click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(targetButton).toHaveAttribute("data-retained-probe", "target-6");
  await expect(targetButton).toBeFocused();
  await expect(targetButton).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator('.roster-row[data-selected="true"]')).toHaveAttribute(
    "aria-label",
    /^Agent id_6,/,
  );
  await expect(page.locator("#battlefield .legality-pill")).toHaveCount(2);
  await expect(page.locator('#battlefield .legality-dock[data-slot="6"]')).toHaveCount(
    1,
  );
  await expect(
    page.locator('#battlefield .agent[data-slot="6"] .agent-id-tag'),
  ).toHaveCSS("opacity", "1");
  await expect(page.locator(".candidate-legality-row")).toHaveCount(0);
  await expect(page.locator("#diagnostics-card")).not.toContainText(
    "candidate_legalities",
  );

  await page.locator("#preset-select").selectOption("debug");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#preset-select")).toHaveValue("debug");
  await expect(page.locator("html")).toHaveAttribute("data-preset", "debug");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  await expect(page.locator(".candidate-legality-row")).toHaveCount(11);
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(10);

  await page.locator("#view-select").selectOption("pov");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(0);
  const povRosterSlots = await page
    .locator("#roster .roster-row")
    .evaluateAll((rows) =>
      rows.map((row) => Number(row.getAttribute("data-slot"))).sort((a, b) => a - b),
    );
  const povCandidateRows = await page
    .locator(".candidate-legality-row")
    .evaluateAll((rows) =>
      rows.map((row) => ({
        targetAction: Number(row.getAttribute("data-target-action")),
        targetSlot: row.hasAttribute("data-target-slot")
          ? Number(row.getAttribute("data-target-slot"))
          : null,
      })),
    );
  expect(povCandidateRows).toHaveLength(povRosterSlots.length + 1);
  expect(
    povCandidateRows
      .map(({ targetSlot }) => targetSlot)
      .filter((slot) => slot !== null)
      .sort((a, b) => Number(a) - Number(b)),
  ).toEqual(povRosterSlots);
  expect(
    povCandidateRows.filter(({ targetAction }) => targetAction === 0),
  ).toHaveLength(1);
  expect(povRosterSlots).not.toContain(5);
  await expect(
    page.locator('.candidate-legality-row[data-target-slot="5"]'),
  ).toHaveCount(0);

  await page.locator("#view-select").selectOption("researcher");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#view-select")).toHaveValue("researcher");

  const controlButton = page.getByRole("button", { name: "Control id_1" });
  await controlButton.evaluate((element) => {
    element.setAttribute("data-retained-probe", "control-1");
  });
  await controlButton.click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(controlButton).toHaveAttribute("data-retained-probe", "control-1");
  await expect(controlButton).toBeFocused();
  await expect(controlButton).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator('.roster-row[data-controlled="true"]')).toHaveAttribute(
    "aria-label",
    /^Agent id_1,/,
  );

  await page.setViewportSize({ width: 960, height: 600 });
  await page
    .locator('#battlefield .agent[data-slot="2"] .agent-body')
    .click({ modifiers: ["Shift"] });
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator('.roster-row[data-controlled="true"]')).toHaveAttribute(
    "aria-label",
    /^Agent id_2,/,
  );

  await page.locator('#battlefield .agent[data-slot="7"] .agent-body').click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator('.roster-row[data-selected="true"]')).toHaveAttribute(
    "aria-label",
    /^Agent id_7,/,
  );
  await expect(page.locator("#battlefield .legality-pill")).toHaveCount(2);
  await expect(page.locator('#battlefield .legality-dock[data-slot="7"]')).toHaveCount(
    1,
  );

  await page.locator("#scenario-select").selectOption("basic_support");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#scenario-select")).toHaveValue("basic_support");
  await expect(page.locator("#step-value")).toHaveText("0");

  await page.getByRole("button", { name: "Reset" }).click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#step-value")).toHaveText("0");

  await page.getByRole("button", { name: "Move east" }).click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#step-value")).toHaveText("0");
});

test("a stale tab adopts the latest frame without replaying its command", async ({
  context,
  page,
}) => {
  await page.goto(debuggerUrl);
  const stalePage = await context.newPage();
  await stalePage.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(stalePage.locator("#connection-status")).toHaveText("Online");
  const baseRevision = await currentRevision(page);
  await expect(stalePage.locator("#revision-value")).toHaveText(String(baseRevision));

  await page.getByRole("button", { name: "Ranges" }).click();
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 1));

  await stalePage.getByRole("button", { name: "Verbosity" }).click();
  await expect(stalePage.locator("#revision-value")).toHaveText(
    String(baseRevision + 1),
  );
  await expect(stalePage.locator("#notice")).toContainText("stale");
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 1));

  await stalePage.getByRole("button", { name: "Verbosity" }).click();
  await expect(stalePage.locator("#revision-value")).toHaveText(
    String(baseRevision + 2),
  );

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 2));
  await stalePage.close();
});

test("a lost applied response requires GET resync and never replays submit", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
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
  await page.keyboard.press("v");
  await expect.poll(() => interceptedCommands).toBe(1);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Help" })).toBeFocused();

  await page.unroute("**/api/command");
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 1));
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));

  await battlefield.focus();
  await page.keyboard.press("g");
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 2));
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
});

test("Exit button reaches the authenticated shutdown path", async ({ page }) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await page.getByRole("button", { name: "Exit debugger" }).click();
  await expect(page.locator("#connection-status")).toHaveText("Shutting down");
  await expect(page.locator("#notice")).toContainText("Exit accepted");
});
