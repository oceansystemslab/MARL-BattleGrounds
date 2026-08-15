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
    "Inspection-only scripted battlefield. Use Advance scripted frame for the next authorized step.",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Scripted live view is inspection-only. Battlefield bodies are passive and cannot submit actions. Use the single Advance scripted frame button to apply the next authorized script step; it is disabled when the script is complete.",
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
  const rosterTargets = page.locator(
    '#roster button[data-role="target"]:not([hidden])',
  );
  const rosterControls = page.locator(
    '#roster button[data-role="control"]:not([hidden])',
  );
  expect(await rosterRows.count()).toBeGreaterThan(0);
  await expect(rosterRows.first()).toHaveAttribute("tabindex", "0");
  expect(await rosterTargets.count()).toBeGreaterThan(0);
  expect(await rosterControls.count()).toBeGreaterThan(0);
  await expect(rosterTargets.first()).toBeDisabled();
  await expect(rosterControls.first()).toBeDisabled();

  /** @type {import("@playwright/test").Request[]} */
  const commandPosts = [];
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/command"
    ) {
      commandPosts.push(request);
    }
  });

  const baselineFrame = await authenticatedGet(page, "/api/frame");
  await page.locator(".agent[data-presentation-key]").first().click({ force: true });
  await page.locator("#battlefield").evaluate((battlefield) => {
    battlefield.dispatchEvent(
      new KeyboardEvent("keydown", { bubbles: true, key: "Tab" }),
    );
  });
  for (const button of [rosterTargets.first(), rosterControls.first()]) {
    await button.evaluate((element) => {
      element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  }
  await page
    .locator('#command-deck button[data-key="Tab"]')
    .first()
    .evaluate((button) => {
      button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  await page.waitForTimeout(150);
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
    "Inspection-only scripted battlefield. Use Advance scripted frame for the next authorized step.",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Scripted live view is inspection-only. Battlefield bodies are passive and cannot submit actions. Use the single Advance scripted frame button to apply the next authorized script step; it is disabled when the script is complete.",
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
