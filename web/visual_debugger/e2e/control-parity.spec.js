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

  await expect(page.getByRole("button", { name: "Move east" })).toBeDisabled();
  await expect(page.locator("#submit-turn-button")).toBeDisabled();
  await expect(page.locator("#advance-script-button")).toBeEnabled();
  await expect(page.locator("#step-value")).toHaveText("0");
});

test("movement scale previews locally and resets one authoritative epoch on commit", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  let revision = await currentRevision(page);
  if ((await page.locator("#view-select").inputValue()) !== "researcher") {
    await page.locator("#view-select").selectOption("researcher");
    revision += 1;
    await expect(page.locator("#revision-value")).toHaveText(String(revision));
  }
  if ((await page.locator("#scenario-select").inputValue()) !== "arena_5v5") {
    await page.locator("#scenario-select").selectOption("arena_5v5");
    revision += 1;
    await expect(page.locator("#revision-value")).toHaveText(String(revision));
  }
  await page.locator("#reset-button").click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#step-value")).toHaveText("0");

  await page.getByRole("button", { name: "Move east" }).click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(
    page.locator('.pending-action-row[data-actor-slot="0"]'),
  ).toHaveAttribute("data-move-action", "3");

  let movementScaleCommands = 0;
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    if (payload?.command?.command_type === "set_movement_scale") {
      movementScaleCommands += 1;
    }
    await route.continue();
  });

  const scaleInput = page.locator("#movement-scale-input");
  await scaleInput.evaluate((element) => {
    const input = /** @type {HTMLInputElement} */ (element);
    input.value = "0.37";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#movement-scale-value")).toHaveText("0.37");
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  expect(movementScaleCommands).toBe(0);

  await scaleInput.dispatchEvent("change");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#movement-scale-value")).toHaveText("0.37");
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(
    page.locator('.pending-action-row[data-actor-slot="0"]'),
  ).toHaveAttribute("data-move-action", "0");
  expect(movementScaleCommands).toBe(1);

  await page.locator("#movement-scale-tenth-button").click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#movement-scale-value")).toHaveText("0.10");
  await expect(page.locator("#step-value")).toHaveText("0");
  expect(movementScaleCommands).toBe(2);

  await page.locator("#reset-button").click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#movement-scale-value")).toHaveText("0.10");
  await expect(page.locator("#step-value")).toHaveText("0");
  expect(movementScaleCommands).toBe(2);

  await page.locator("#movement-scale-default-button").click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#movement-scale-value")).toHaveText("1.00");
  await expect(page.locator("#movement-scale-default-button")).toBeDisabled();
  await expect(page.locator("#step-value")).toHaveText("0");
  expect(movementScaleCommands).toBe(3);
  await page.unroute("**/api/command");
});

test("joint turn drafts survive actor cycling and submit exactly once", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  let revision = await currentRevision(page);
  if ((await page.locator("#scenario-select").inputValue()) !== "arena_5v5") {
    await page.locator("#scenario-select").selectOption("arena_5v5");
    revision += 1;
    await expect(page.locator("#revision-value")).toHaveText(String(revision));
  }
  await page.locator("#reset-button").click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-submission-scope",
    "joint_turn",
  );
  await expect(page.locator(".pending-action-row")).toHaveCount(10);
  await expect(page.locator('.pending-action-row[data-controlled="true"]')).toHaveCount(
    1,
  );

  const battlefield = page.locator("#battlefield");
  await expect(page.locator("#basic-button")).toHaveAttribute("aria-disabled", "true");
  await expect(page.locator("#basic-button")).not.toHaveAttribute("disabled");
  await page.locator("#basic-button").hover();
  await page.locator("#basic-button").dispatchEvent("click");
  await expect(page.locator("#notice")).toContainText(
    "Basic is unavailable for the currently staged target.",
  );
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await battlefield.focus();
  await page.keyboard.press("1");
  await expect(page.locator("#notice")).toContainText("canonical no-combat tuple");
  await expect(page.locator("#revision-value")).toHaveText(String(revision));

  await battlefield.focus();
  await page.getByRole("button", { name: "Move east" }).click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await page.locator("#command-target-select").selectOption("6");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  const actor0 = page.locator('.pending-action-row[data-actor-slot="0"]');
  await actor0.evaluate((element) => {
    element.setAttribute("data-retained-probe", "joint-g0");
  });

  await battlefield.focus();
  await page.keyboard.press("Tab");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await page.getByRole("button", { name: "Move north", exact: true }).click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await page.locator("#command-target-select").selectOption("5");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  const actor1 = page.locator('.pending-action-row[data-actor-slot="1"]');
  await actor1.evaluate((element) => {
    element.setAttribute("data-retained-probe", "joint-g1");
  });

  await battlefield.focus();
  await page.keyboard.press("Shift+Tab");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(actor0).toHaveAttribute("data-retained-probe", "joint-g0");
  await expect(actor1).toHaveAttribute("data-retained-probe", "joint-g1");
  await expect(actor0).toHaveAttribute("data-controlled", "true");
  await expect(actor0).toHaveAttribute("data-move-action", "3");
  await expect(actor0).toHaveAttribute("data-target-slot", "6");
  await expect(actor1).toHaveAttribute("data-controlled", "false");
  await expect(actor1).toHaveAttribute("data-move-action", "1");
  await expect(actor1).toHaveAttribute("data-target-slot", "5");

  for (let actorSlot = 2; actorSlot < 10; actorSlot += 1) {
    await page.getByRole("button", { name: `Control id_${actorSlot}` }).click();
    revision += 1;
    await expect(page.locator("#revision-value")).toHaveText(String(revision));
    const movementName = actorSlot < 5 ? "Move east" : "Move west";
    await page.getByRole("button", { name: movementName, exact: true }).click();
    revision += 1;
    await expect(page.locator("#revision-value")).toHaveText(String(revision));
    await expect(
      page.locator(
        `.pending-action-row[data-actor-slot="${actorSlot}"][data-controlled="true"]`,
      ),
    ).toHaveAttribute("data-move-action", actorSlot < 5 ? "3" : "4");
  }

  await page.getByRole("button", { name: "Control id_0" }).click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.getByRole("button", { name: "Move east" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator("#command-target-select")).toHaveValue("6");
  await expect(page.locator("#no-combat-button")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  const stagedMoves = await page
    .locator(".pending-action-row")
    .evaluateAll((rows) =>
      rows.map((row) => Number(row.getAttribute("data-move-action"))),
    );
  expect(stagedMoves).toEqual([3, 1, 3, 3, 3, 4, 4, 4, 4, 4]);

  let enterCommands = 0;
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    if (
      payload?.command?.command_type === "keyboard" &&
      String(payload.command.key).toLowerCase() === "enter"
    ) {
      enterCommands += 1;
    }
    await route.continue();
  });
  const beforeSubmitRevision = await currentRevision(page);
  const beforeSubmitStep = await currentStep(page);
  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#revision-value")).toHaveText(
    String(beforeSubmitRevision + 1),
    { timeout: 120_000 },
  );
  await expect(page.locator("#step-value")).toHaveText(String(beforeSubmitStep + 1));
  await expect(page.locator("#transition-value")).toHaveText(
    String(beforeSubmitStep + 1),
  );
  await expect(page.locator("#accepted-card .action-result")).toHaveCount(10);
  await expect.poll(() => enterCommands).toBe(1);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText(
    String(beforeSubmitRevision + 1),
  );
  await expect(page.locator("#step-value")).toHaveText(String(beforeSubmitStep + 1));
  await page.unroute("**/api/command");

  await page.setViewportSize({ width: 960, height: 600 });
  const horizontalOverflow = await page
    .locator("#pending-card")
    .evaluate((element) => element.scrollWidth - element.clientWidth);
  expect(horizontalOverflow).toBeLessThanOrEqual(1);
});

test("scripted playback exposes only N and never sends manual submit", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  let revision = await currentRevision(page);
  if ((await page.locator("#scenario-select").inputValue()) !== "basic_support") {
    await page.locator("#scenario-select").selectOption("basic_support");
    revision += 1;
    await expect(page.locator("#revision-value")).toHaveText(String(revision));
  }
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-submission-scope",
    "scripted_playback",
  );
  await expect(page.locator("#pending-card .action-card__label")).toHaveText(
    "PLAYBACK / INSPECTION ONLY",
  );
  await expect(page.locator("#submit-turn-button")).toBeDisabled();
  await expect(page.locator("#submit-turn-button")).toHaveText(
    "Manual submit unavailable",
  );
  await expect(page.getByRole("button", { name: "Move east" })).toBeDisabled();
  await expect(page.locator("#stay-button")).toBeDisabled();
  await expect(page.locator("#advance-script-button")).toBeEnabled();

  let manualSubmitCommands = 0;
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    if (
      payload?.command?.command_type === "keyboard" &&
      ["enter", " "].includes(String(payload.command.key).toLowerCase())
    ) {
      manualSubmitCommands += 1;
    }
    await route.continue();
  });
  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#notice")).toContainText(
    "Scripted playback is inspection-only",
  );
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#step-value")).toHaveText("0");
  expect(manualSubmitCommands).toBe(0);

  await page.locator("#advance-script-button").click();
  await expect(page.locator("#revision-value")).toHaveText(String(revision + 1), {
    timeout: 120_000,
  });
  await expect(page.locator("#step-value")).toHaveText("1");
  expect(manualSubmitCommands).toBe(0);
  await page.unroute("**/api/command");
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
  let setupRevision = await currentRevision(page);
  if ((await page.locator("#scenario-select").inputValue()) !== "arena_5v5") {
    await page.locator("#scenario-select").selectOption("arena_5v5");
    setupRevision += 1;
    await expect(page.locator("#revision-value")).toHaveText(String(setupRevision));
  }
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-submission-scope",
    "joint_turn",
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
  await page.getByRole("button", { name: "Exit analyzer" }).click();
  await expect(page.locator("#connection-status")).toHaveText("Shutting down");
  await expect(page.locator("#notice")).toContainText("Exit accepted");
});
