import { expect, test } from "@playwright/test";

import { startScriptedDebugger, stopDebugger } from "./support/live-debugger.js";

/** @type {Awaited<ReturnType<typeof startScriptedDebugger>> | null} */
let scriptedDebugger = null;

/** @type {string[]} */
const browserErrors = [];

test.beforeAll(async () => {
  scriptedDebugger = await startScriptedDebugger();
});

test.afterAll(async () => {
  const child = scriptedDebugger?.process ?? null;
  scriptedDebugger = null;
  await stopDebugger(child);
});

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

test("scripted live Submit advances once, installs T0, and seals at completion", async ({
  page,
}) => {
  if (!scriptedDebugger) {
    throw new Error("Scripted DebuggerService harness is unavailable.");
  }
  browserErrors.length = 0;
  page.on("pageerror", (error) => browserErrors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(`console: ${message.text()}`);
    }
  });

  await page.goto(scriptedDebugger.url);
  await expect(page.locator("#connection-status")).toHaveText("Online", {
    timeout: 30_000,
  });
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(page.locator("#transition-value")).toHaveText("—");
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-submission-scope",
    "scripted_playback",
  );
  await expect(page.locator("#scenario-select, #advance-script-button")).toHaveCount(0);

  const submit = page.locator("#submit-turn-button");
  await expect(submit).toHaveText("Advance scripted frame");
  await expect(submit).toHaveAttribute("data-key", "n");
  await expect(submit).toHaveAttribute(
    "aria-description",
    "Submit an editable draft or advance an inspection-only scripted frame through the authoritative Python service.",
  );
  await expect(submit).toBeEnabled();
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-label",
    "Inspection-only scripted battlefield. Authorized bodies can be inspected; use Advance scripted frame for the next authorized step.",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Scripted live view is inspection-only. Activate an authorized body to inspect current facts; use Advance scripted frame for the next authorized step.",
  );
  await expect(page.locator("#command-commit-title")).toHaveText(
    "Advance the registered script",
  );
  await expect(page.locator("#command-commit-summary")).toHaveText(
    "One authoritative scripted transition",
  );
  await submit.hover();
  await expect(page.locator("#visual-tooltip-title")).toHaveText(
    "Apply authorized action",
  );
  await expect(page.locator("#reset-button")).toBeDisabled();
  await expect(page.locator("#command-target-select")).toBeDisabled();
  expect(
    await page
      .locator('[data-command-role="draft"]')
      .evaluateAll((buttons) =>
        buttons.every(
          (button) => button instanceof HTMLButtonElement && button.disabled,
        ),
      ),
  ).toBe(true);
  expect(
    await page.locator("#command-deck button[data-key]").evaluateAll((buttons) =>
      buttons
        .filter((button) => button instanceof HTMLButtonElement && !button.disabled)
        .map((button) => ({
          id: button.id,
          key: /** @type {HTMLElement} */ (button).dataset.key,
          label: button.textContent?.trim(),
        })),
    ),
  ).toEqual([
    {
      id: "submit-turn-button",
      key: "n",
      label: "Advance scripted frame",
    },
  ]);

  const rosterRows = page.locator("#roster .roster-row");
  const rosterActions = page.locator("#roster .roster-primary-action");
  expect(await rosterRows.count()).toBeGreaterThan(0);
  await expect(rosterRows.first()).not.toHaveAttribute("tabindex", /.+/u);
  await expect(rosterActions).toHaveCount(await rosterRows.count());
  await expect(
    page.getByRole("group", {
      name: "Inspection-only scripted battlefield. Authorized bodies can be inspected; use Advance scripted frame for the next authorized step.",
    }),
  ).toHaveCount(1);
  await expect(page.locator('#battlefield .agent[role="button"]')).toHaveCount(
    await rosterRows.count(),
  );
  await expect(
    page.locator("#roster .roster-actions, #roster [data-role]"),
  ).toHaveCount(0);
  await expect(rosterActions.first()).toBeEnabled();
  await expect(rosterActions.first()).toHaveAttribute(
    "aria-label",
    /Inspect Agent ID/u,
  );
  await expect(rosterActions.first()).not.toHaveAttribute(
    "aria-label",
    /Control|Target|Reference|POV actor/u,
  );

  /** @type {import("@playwright/test").Request[]} */
  const commandPosts = [];
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      ["/api/command", "/api/replay/command"].includes(new URL(request.url()).pathname)
    ) {
      commandPosts.push(request);
    }
  });

  const baselineFrame = await authenticatedGet(page, "/api/frame");
  expect(await rosterActions.count()).toBeGreaterThan(1);
  const localAction = rosterActions.nth(1);
  const localKey = await localAction.getAttribute("data-presentation-key");
  const localIdentity = (await localAction.getAttribute("aria-label"))?.replace(
    /^Inspect /u,
    "",
  );
  expect(typeof localKey).toBe("string");
  expect(typeof localIdentity).toBe("string");
  await localAction.click();
  await expect(page.locator("#selection-card")).toContainText(String(localIdentity));
  await expect(localAction).toHaveAttribute("aria-pressed", "true");
  await localAction.focus();
  const scrollBeforeSpace = await page.evaluate(() => window.scrollY);
  await localAction.press("Enter");
  await localAction.press(" ");
  expect(await page.evaluate(() => window.scrollY)).toBe(scrollBeforeSpace);
  const localBody = page.locator(`.agent[data-presentation-key="${localKey}"]`);
  await expect(localBody).toHaveAttribute("data-selected", "true");
  await expect(page.locator("#battlefield .range-ring-owner")).toHaveCount(3);
  await localBody.click();
  await localBody.focus();
  await localBody.press("Enter");
  await localBody.press(" ");
  const localFactChip = page
    .locator(
      `#roster .roster-row[data-presentation-key="${localKey}"] .roster-fact-token[data-tooltip-owner]`,
    )
    .first();
  await expect(localFactChip).toBeVisible();
  const selectionBeforeFactInspection = await page
    .locator('#battlefield .agent[data-selected="true"]')
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  const pressedRowsBeforeFactInspection = await rosterActions.evaluateAll((buttons) =>
    buttons.map((button) => button.getAttribute("aria-pressed")),
  );
  await localFactChip.hover();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await localFactChip.click();
  const factTitle = await page.locator("#visual-tooltip-title").textContent();
  expect(factTitle).not.toBeNull();
  await expect(page.locator("#selection-card")).toContainText(String(factTitle));
  expect(
    await page
      .locator('#battlefield .agent[data-selected="true"]')
      .evaluateAll((agents) =>
        agents.map((agent) => agent.getAttribute("data-presentation-key")),
      ),
  ).toEqual(selectionBeforeFactInspection);
  expect(
    await rosterActions.evaluateAll((buttons) =>
      buttons.map((button) => button.getAttribute("aria-pressed")),
    ),
  ).toEqual(pressedRowsBeforeFactInspection);
  await page.locator("#battlefield").evaluate((battlefield) => {
    battlefield.dispatchEvent(
      new KeyboardEvent("keydown", { bubbles: true, key: "Tab" }),
    );
  });
  const actorCycleButtons = page.locator('#command-deck button[data-key="Tab"]');
  await expect(actorCycleButtons).toHaveCount(2);
  for (const button of await actorCycleButtons.all()) {
    await expect(button).toBeDisabled();
    await expect(button).toHaveAttribute(
      "aria-description",
      /Actor cycling is unavailable in the current Oracle View state.*native browser focus navigation/,
    );
  }
  await actorCycleButtons.first().evaluate((button) => {
    button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  expect(commandPosts).toHaveLength(0);
  const frameAfterPassiveInputs = await authenticatedGet(page, "/api/frame");
  expect(frameAfterPassiveInputs.revision).toBe(baselineFrame.revision);
  expect(frameAfterPassiveInputs.simulator_step_count).toBe(
    baselineFrame.simulator_step_count,
  );

  await page.setViewportSize({ width: 967, height: 731 });
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-label",
    "Inspection-only scripted battlefield. Authorized bodies can be inspected; use Advance scripted frame for the next authorized step.",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Scripted live view is inspection-only. Activate an authorized body to inspect current facts; use Advance scripted frame for the next authorized step.",
  );

  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
    { timeout: 30_000 },
  );
  await submit.click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  expect(commandPosts).toHaveLength(1);
  expect(commandPosts[0].postDataJSON().command).toEqual({
    command_type: "keyboard",
    key: "n",
    shift_key: false,
    ctrl_key: false,
    alt_key: false,
    meta_key: false,
    repeat: false,
  });
  const commandPayload = await response.json();
  expect(commandPayload).toMatchObject({
    result: "applied",
    frame: {
      scenario: {
        completed_frame_count: 1,
        frame_count: 1,
        script_complete: true,
      },
      simulator_step_count: 1,
      terminal: {
        is_sealed: true,
        reached_declared_horizon: true,
        reason: "declared_horizon",
      },
    },
  });

  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#step-value")).toHaveText("1");
  const presentation = await authenticatedGet(page, "/api/presentation/frame");
  const transitionId = presentation.latest_events.incoming_transition_id;
  expect(typeof transitionId).toBe("string");
  expect(transitionId.length).toBeGreaterThan(0);
  expect(presentation.latest_transition.incoming_transition_id).toBe(transitionId);
  await expect(page.locator("#transition-value")).toHaveText(transitionId);
  await expect(page.locator("#accepted-card")).toHaveAttribute(
    "data-transition-id",
    transitionId,
  );
  await expect(page.locator("#accepted-card .accepted-action-row")).toHaveCount(
    presentation.latest_transition.action_rows.length,
  );

  await expect(submit).toHaveText("Advance scripted frame");
  await expect(submit).toHaveAttribute("data-key", "n");
  await expect(submit).toBeDisabled();
  expect(
    await page
      .locator("#roster .roster-primary-action")
      .evaluateAll((buttons) =>
        buttons.every(
          (button) => button instanceof HTMLButtonElement && button.disabled,
        ),
      ),
  ).toBe(true);
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-label",
    "Read-only live battlefield. Scientific facts can be inspected; simulator and actor activation controls are unavailable.",
  );
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Scientific tooltip facts remain inspectable. Live simulator and actor activation controls are unavailable while recording is closing, the session is offline or resynchronizing, or the frame is terminal.",
  );
  expect(
    await page
      .locator("#battlefield .agent")
      .evaluateAll((agents) =>
        agents.every(
          (agent) =>
            agent.getAttribute("role") === "img" &&
            agent.getAttribute("tabindex") === "-1",
        ),
      ),
  ).toBe(true);
  const terminalDetails = page.locator("#agent-details");
  if ((await terminalDetails.getAttribute("open")) !== null) {
    await page.locator("#agent-details > summary").click();
  }
  await page.locator("#battlefield .agent").first().click({ force: true });
  await page
    .locator("#battlefield .agent")
    .first()
    .dispatchEvent("keydown", { key: "Enter" });
  await expect(terminalDetails).not.toHaveAttribute("open", "");
  expect(commandPosts).toHaveLength(1);
  await expect(page.locator("#command-commit-title")).toHaveText(
    "Advance the registered script",
  );
  await expect(page.locator("#command-commit-summary")).toHaveText(
    "One authoritative scripted transition",
  );
  const terminalRevision = commandPayload.frame.revision;
  await submit.evaluate((button) => {
    if (!(button instanceof HTMLButtonElement)) {
      throw new TypeError("Scripted advance button is unavailable.");
    }
    button.click();
  });
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  expect(commandPosts).toHaveLength(1);
  const terminalFrame = await authenticatedGet(page, "/api/frame");
  expect(terminalFrame.revision).toBe(terminalRevision);
  expect(terminalFrame.simulator_step_count).toBe(1);
  expect(terminalFrame.scenario.script_complete).toBe(true);
  expect(browserErrors).toEqual([]);
});
