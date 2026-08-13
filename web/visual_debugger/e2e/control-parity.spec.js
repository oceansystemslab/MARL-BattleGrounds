import { expect, test } from "@playwright/test";

import { finishControllerClock } from "./support/choreography.js";
import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  DENSE_BASELINE_MAX_DIFF_PIXEL_RATIO,
  waitForStablePresentation,
} from "./support/visual-regression.js";

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

/**
 * Read the authenticated V2 wire frame used by the browser tab.
 *
 * @param {import("@playwright/test").Page} page
 * @returns {Promise<Record<string, any>>}
 */
async function currentWireFrame(page) {
  return page.evaluate(async () => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Debugger capability token is unavailable.");
    }
    const response = await fetch("/api/frame", {
      cache: "no-store",
      credentials: "omit",
      headers: { "X-MARL-Debugger-Token": token },
      redirect: "error",
    });
    if (!response.ok) {
      throw new Error(`Frame request failed with HTTP ${response.status}.`);
    }
    return response.json();
  });
}

/**
 * Establish the one live interactive authority required by isolated tests.
 * The serial server deliberately retains audience and scenario between cases.
 *
 * @param {import("@playwright/test").Page} page
 */
async function installResearcherArena(page) {
  if ((await page.locator("#view-select").inputValue()) !== "researcher") {
    await page.locator("#view-select").selectOption("researcher");
    await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  }
  if ((await page.locator("#scenario-select").inputValue()) !== "arena_5v5") {
    await page.locator("#scenario-select").selectOption("arena_5v5");
    await expect(page.locator("#scenario-select")).toHaveValue("arena_5v5");
  }
}

/**
 * Assert the full V2 transition identity rendered for the current researcher frame.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} transitionIndex
 * @returns {Promise<string>}
 */
async function expectCurrentResearcherTransition(page, transitionIndex) {
  const frame = await currentWireFrame(page);
  expect(frame).toMatchObject({
    schema_version: 2,
    frame_kind: "researcher_live_debugger",
    incoming_transition_index: transitionIndex,
  });
  const transitionId = `${frame.episode_id}:transition:${transitionIndex}`;
  expect(frame.incoming_transition_id).toBe(transitionId);
  expect(frame.projection.scene.incoming_transition_id).toBe(transitionId);
  expect(frame.projection.incoming_events.transition_id).toBe(transitionId);
  expect(frame.hud.latest_transition.transition_id).toBe(transitionId);
  await expect(page.locator("#transition-value")).toHaveText(transitionId);
  return transitionId;
}

/**
 * Freeze and prove the expanded comprehensive Agent Details panel at one
 * supported viewport. The visual assertion is downstream of the complete
 * semantic inventory and explicit layout bounds.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{width: number, height: number}} viewport
 */
async function captureAgentDetailsBaseline(page, viewport) {
  await page.setViewportSize(viewport);
  await waitForStablePresentation(page);

  const details = page.locator("#agent-details");
  await expect(details).toBeVisible();
  await expect(details).toHaveAttribute("open", "");
  await expect(details.locator(":scope > summary")).toHaveText("Agent Details");
  for (const fact of [
    "Identity",
    "Class Role",
    "Role",
    "Strengths",
    "Limitations",
    "Teamwork",
    "Counterplay",
    "Exact Class Mechanics",
    "Current State",
    "Persistent Statuses",
    "Aggregate Aura Modifiers",
    "Ultimate Status",
    "Current legality",
    "Basic Legality",
    "Ultimate Legality",
  ]) {
    await expect(details).toContainText(fact);
  }
  await expect(details.locator(".selected-legality")).toHaveCount(1);
  await expect(details.locator(".selected-legality__lane")).toHaveCount(2);
  const layout = await details.evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    const card = element.querySelector("#selection-card");
    const semanticDetails = element.querySelector(".semantic-inspector__details");
    if (!(card instanceof HTMLElement) || !(semanticDetails instanceof HTMLElement)) {
      throw new Error("Agent Details semantic layout is unavailable.");
    }
    const cardBounds = card.getBoundingClientRect();
    const semanticBounds = semanticDetails.getBoundingClientRect();
    const semanticStyle = getComputedStyle(semanticDetails);
    return {
      horizontalOverflow: element.scrollWidth - element.clientWidth,
      semanticColumn: {
        end: semanticStyle.gridColumnEnd,
        start: semanticStyle.gridColumnStart,
        widthRatio: semanticBounds.width / cardBounds.width,
      },
      viewportEscape: {
        left: Math.max(0, -bounds.left),
        right: Math.max(0, bounds.right - document.documentElement.clientWidth),
      },
    };
  });
  expect(layout.horizontalOverflow).toBeLessThanOrEqual(1);
  expect(layout.semanticColumn.start).toBe("1");
  expect(layout.semanticColumn.end).toBe("-1");
  expect(layout.semanticColumn.widthRatio).toBeGreaterThan(0.99);
  expect(layout.viewportEscape).toEqual({ left: 0, right: 0 });

  const hud = page.locator(".hud-panel");
  await hud.evaluate((element, detailsId) => {
    const details = element.querySelector(`#${detailsId}`);
    if (!(details instanceof HTMLElement)) {
      throw new Error("Agent Details panel is unavailable in the HUD.");
    }
    element.scrollTop = details.offsetTop;
  }, "agent-details");
  await expect(details.locator(":scope > summary")).toBeInViewport();
  await expect(details.getByText("Identity", { exact: true })).toBeInViewport();
  await expect(details.getByText("Class Role", { exact: true })).toBeInViewport();
  const visibleHud = await hud.evaluate((element) => ({
    clientHeight: element.clientHeight,
    scrollHeight: element.scrollHeight,
    scrollTop: element.scrollTop,
    viewportHeight: document.documentElement.clientHeight,
  }));
  expect(visibleHud.scrollTop).toBeGreaterThan(0);
  expect(visibleHud.scrollHeight).toBeGreaterThan(visibleHud.clientHeight);
  expect(visibleHud.clientHeight).toBeLessThanOrEqual(visibleHud.viewportHeight);

  await expect(hud).toHaveScreenshot(
    `agent-details-expanded-${viewport.width}x${viewport.height}.png`,
    { animations: "disabled" },
  );
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

  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 960, height: 600 },
  ]) {
    await page.setViewportSize(viewport);
    await page.evaluate(
      () =>
        new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        }),
    );
    const consoleLayout = await page.locator(".command-deck").evaluate((deck) => {
      const bounds = deck.getBoundingClientRect();
      const directSections = [...deck.children].map((child) => child.className);
      const submit = deck.querySelector("#submit-turn-button");
      const advance = deck.querySelector("#advance-script-button");
      if (
        !(submit instanceof HTMLButtonElement) ||
        !(advance instanceof HTMLButtonElement)
      ) {
        throw new Error("Command console commit controls are unavailable.");
      }
      const submitStyle = getComputedStyle(submit);
      const advanceStyle = getComputedStyle(advance);
      return {
        directSections,
        horizontalOverflow: deck.scrollWidth - deck.clientWidth,
        viewportEscape: {
          left: Math.max(0, -bounds.left),
          right: Math.max(0, bounds.right - document.documentElement.clientWidth),
        },
        submitDistinct:
          submitStyle.backgroundImage !== advanceStyle.backgroundImage &&
          submitStyle.color !== advanceStyle.color,
      };
    });
    expect(consoleLayout.directSections).toEqual([
      "command-deck__header",
      "command-deck__composer",
      "command-commit",
      "command-deck__utilities",
    ]);
    expect(consoleLayout.horizontalOverflow).toBeLessThanOrEqual(1);
    expect(consoleLayout.viewportEscape.left).toBe(0);
    expect(consoleLayout.viewportEscape.right).toBe(0);
    expect(consoleLayout.submitDistinct).toBe(true);
    await expect(page.locator(".command-deck")).toHaveScreenshot(
      `command-console-${viewport.width}x${viewport.height}.png`,
    );
  }

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
  const transitionId = await expectCurrentResearcherTransition(page, 0);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText("2");
  await expect(page.locator("#step-value")).toHaveText("1");
  await expect(page.locator("#transition-value")).toHaveText(transitionId);
  expect(page.url()).not.toContain("token=");
});

test("pointer, roster, toolbar, and command-deck controls use the live service", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await installResearcherArena(page);
  const resetRevision = await currentRevision(page);
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.locator("#revision-value")).toHaveText(String(resetRevision + 1));
  await expect(page.locator("#step-value")).toHaveText("0");
  let revision = resetRevision + 1;

  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await page.locator("#battlefield").click({ position: { x: 8, y: 8 } });
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");

  const targetButton = page.getByRole("button", { name: "Target Agent ID 6" });
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
    /^Agent ID 6,/,
  );
  await expect(page.locator("#battlefield .legality-pill")).toHaveCount(2);
  await expect(page.locator('#battlefield .legality-dock[data-slot="6"]')).toHaveCount(
    1,
  );
  await expect(page.locator("#battlefield .agent-id-tag")).toHaveCount(0);
  await expect(page.locator('#battlefield .agent[data-slot="6"]')).toHaveAttribute(
    "aria-label",
    /Agent ID 6/,
  );
  await expect(page.locator(".candidate-legality-row")).toHaveCount(0);
  await expect(page.locator("#diagnostics-card")).not.toContainText(
    "candidate_legalities",
  );
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await expect(page.locator("#agent-details")).toContainText("Agent ID 6");
  for (const fact of [
    "Identity",
    "Class Role",
    "Role",
    "Strengths",
    "Limitations",
    "Teamwork",
    "Counterplay",
    "Exact Class Mechanics",
    "Current State",
    "Persistent Statuses",
    "Aggregate Aura Modifiers",
    "Ultimate Status",
    "Current legality",
  ]) {
    await expect(page.locator("#agent-details")).toContainText(fact);
  }
  await expect(page.locator("#preset-select option")).toHaveCount(2);
  await expect(page.locator('#preset-select option[value="debug"]')).toHaveCount(0);
  await expect(page.locator("#live-verbosity-button")).toHaveCount(0);
  await expect(page.locator("#movement-scale-input")).toHaveCount(0);
  await expect(page.locator("#graphics-speed-input")).toHaveCount(0);
  await expect(page.locator("html")).toHaveAttribute(
    "data-preset",
    /analysis|presentation/,
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-protected-zone")).toHaveCount(0);

  await page.locator("#view-select").selectOption("pov");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  const povFrame = await currentWireFrame(page);
  expect(povFrame).toMatchObject({
    schema_version: 2,
    frame_kind: "actor_pov_live_debugger",
    view_mode: "pov",
  });
  const povRosterSlots = await page
    .locator("#roster .roster-row")
    .evaluateAll((rows) =>
      rows.map((row) => Number(row.getAttribute("data-slot"))).sort((a, b) => a - b),
    );
  expect(povRosterSlots).toEqual([povFrame.projection.scene.self_actor.global_slot]);
  await expect(page.locator(".candidate-legality-row")).toHaveCount(0);
  expect(povRosterSlots).not.toContain(5);
  await expect(
    page.locator('.candidate-legality-row[data-target-slot="5"]'),
  ).toHaveCount(0);

  await page.locator("#view-select").selectOption("researcher");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#view-select")).toHaveValue("researcher");

  const controlButton = page.getByRole("button", { name: "Control Agent ID 1" });
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
    /^Agent ID 1,/,
  );

  await page.setViewportSize({ width: 960, height: 600 });
  await page
    .locator('#battlefield .agent[data-slot="7"] .agent-body')
    .click({ modifiers: ["Shift"] });
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator('.roster-row[data-selected="true"]')).toHaveAttribute(
    "aria-label",
    /^Agent ID 7,/,
  );

  await page
    .locator('#battlefield .agent[data-slot="7"] .agent-body')
    .click({ button: "right" });
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator('.roster-row[data-selected="true"]')).toHaveAttribute(
    "aria-label",
    /^Agent ID 1,/,
  );
  await expect(
    page.locator('.pending-action-row[data-controlled="true"]'),
  ).toHaveAttribute("data-target-disclosure", "target_none");

  await page.locator('#battlefield .agent[data-slot="2"] .agent-body').click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await expect(page.locator('.roster-row[data-controlled="true"]')).toHaveAttribute(
    "aria-label",
    /^Agent ID 2,/,
  );
  await expect(page.locator('.roster-row[data-selected="true"]')).toHaveAttribute(
    "aria-label",
    /^Agent ID 2,/,
  );
  await expect(page.locator("#battlefield .legality-pill")).toHaveCount(2);
  await expect(page.locator('#battlefield .legality-dock[data-slot="7"]')).toHaveCount(
    0,
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

test("expanded Agent Details has permanent visual proof at both supported viewports", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await installResearcherArena(page);
  const resetRevision = await currentRevision(page);
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.locator("#revision-value")).toHaveText(String(resetRevision + 1));
  await expect(page.locator("#step-value")).toHaveText("0");

  await page.getByRole("button", { name: "Target Agent ID 6" }).click();
  await expect(page.locator("#revision-value")).toHaveText(String(resetRevision + 2));
  await expect(page.locator('.roster-row[data-selected="true"]')).toHaveAttribute(
    "aria-label",
    /^Agent ID 6,/,
  );
  await captureAgentDetailsBaseline(page, { width: 1440, height: 900 });
  await captureAgentDetailsBaseline(page, { width: 960, height: 600 });
});

test("command composer keeps the exact controlled actor visible at both viewports", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  if ((await page.locator("#scenario-select").inputValue()) !== "arena_5v5") {
    await page.locator("#scenario-select").selectOption("arena_5v5");
  }
  await page.locator("#reset-button").click();

  const controlledActor = page.locator("#command-controlled-actor");
  /**
   * @param {{width: number, height: number}} viewport
   * @param {number} slot
   */
  const assertComposerIdentity = async (viewport, slot) => {
    await page.setViewportSize(viewport);
    await page.evaluate(
      () =>
        new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        }),
    );
    await expect(controlledActor).toHaveText(`Actor · Agent ID ${slot}`);
    await expect(controlledActor).toHaveAttribute(
      "aria-label",
      `Controlled actor Agent ID ${slot}`,
    );
    await expect(controlledActor).toHaveAttribute("data-controlled-slot", String(slot));
    await expect(controlledActor).toBeVisible();
    const visual = await controlledActor.evaluate((label) => {
      const bounds = label.getBoundingClientRect();
      const heading = label.closest(".command-group__heading");
      if (!(heading instanceof HTMLElement)) {
        throw new Error("Movement-stage heading is unavailable.");
      }
      const headingBounds = heading.getBoundingClientRect();
      const style = getComputedStyle(label);
      return {
        clipped:
          bounds.left < headingBounds.left - 0.5 ||
          bounds.right > headingBounds.right + 0.5 ||
          bounds.top < headingBounds.top - 0.5 ||
          bounds.bottom > headingBounds.bottom + 0.5,
        opacity: Number.parseFloat(style.opacity),
        visibility: style.visibility,
      };
    });
    expect(visual).toEqual({
      clipped: false,
      opacity: 1,
      visibility: "visible",
    });
  };

  await assertComposerIdentity({ width: 1440, height: 900 }, 0);
  await page.getByRole("button", { name: "Control Agent ID 1" }).click();
  await expect(controlledActor).toHaveText("Actor · Agent ID 1");
  await assertComposerIdentity({ width: 960, height: 600 }, 1);
  await page.locator('#battlefield .agent[data-slot="2"] .agent-body').click();
  await expect(controlledActor).toHaveText("Actor · Agent ID 2");
});

test("native panels preserve user state within one authority and reset at its boundary", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#command-deck")).toHaveAttribute("open", "");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  for (const selector of [
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#events-details",
    "#visual-key",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
  }

  const rosterButton = page.locator("#roster button").first();
  await rosterButton.focus();
  await page.locator("#roster-details").evaluate((details) => {
    if (!(details instanceof HTMLDetailsElement)) {
      throw new TypeError("Roster disclosure is unavailable.");
    }
    details.open = false;
  });
  await expect(page.locator("#roster-details > summary")).toBeFocused();
  await page.locator("#roster-details > summary").click();
  await page.locator("#events-details > summary").click();
  await page.locator("#technical-frame-details > summary").click();

  const revision = await currentRevision(page);
  await page.locator("#live-ranges-button").click();
  await expect(page.locator("#revision-value")).toHaveText(String(revision + 1));
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  await expect(page.locator("#events-details")).toHaveAttribute("open", "");
  await expect(page.locator("#technical-frame-details")).toHaveAttribute("open", "");

  await page.locator("#view-select").selectOption("pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  await expect(page.locator("#command-deck")).toHaveAttribute("open", "");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  await expect(page.locator("#events-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#technical-frame-details")).not.toHaveAttribute(
    "open",
    "",
  );
});

test("product movement scale is read-only canonical metadata", async ({ page }) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await installResearcherArena(page);
  const frame = await currentWireFrame(page);
  expect(frame.scenario.ordinary_movement_distance_scale).toBe(1);
  expect(frame.scenario.description).toBe(
    "Interactive LOS, visibility, range, relation, and mask inspection.",
  );
  await expect(page.locator("#scenario-description")).toBeEmpty();
  await expect(page.locator("#movement-scale-input")).toHaveCount(0);
  await expect(page.locator("#movement-scale-tenth-button")).toHaveCount(0);
  await expect(page.locator("#movement-scale-default-button")).toHaveCount(0);
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
  const controlledActor = page.locator("#command-controlled-actor");
  if ((await controlledActor.getAttribute("data-controlled-slot")) !== "0") {
    await page.getByRole("button", { name: "Control Agent ID 0" }).click();
    revision += 1;
    await expect(page.locator("#revision-value")).toHaveText(String(revision));
  }
  await expect(controlledActor).toHaveAttribute("data-controlled-slot", "0");
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-submission-scope",
    "joint_turn",
  );
  await expect(page.locator(".pending-action-row")).toHaveCount(10);
  await expect(
    page.locator(".pending-action-row__facts").filter({
      hasText: "Target target-none",
    }),
  ).toHaveCount(0);
  await expect(
    page.locator(".pending-action-row__facts").filter({
      hasText: "Lane 0/B",
    }),
  ).toHaveCount(0);
  await expect(
    page.locator('.pending-action-row[data-actor-slot="0"] .pending-action-row__facts'),
  ).toContainText("Target · None");
  await expect(
    page.locator('.pending-action-row[data-actor-slot="0"] .pending-action-row__facts'),
  ).toContainText("Action · No combat");
  await expect(page.locator('.pending-action-row[data-controlled="true"]')).toHaveCount(
    1,
  );

  const battlefield = page.locator("#battlefield");
  await expect(page.locator("#basic-button")).toHaveAttribute("aria-disabled", "true");
  await expect(page.locator("#basic-button")).not.toHaveAttribute("disabled");
  await expect
    .poll(() =>
      page.locator("#basic-button").evaluate((button) => {
        const style = getComputedStyle(button);
        const probe = document.createElement("span");
        probe.style.color = "color-mix(in srgb, var(--danger) 70%, var(--text-muted))";
        probe.style.borderColor =
          "color-mix(in srgb, var(--danger) 70%, var(--border))";
        document.body.append(probe);
        const probeStyle = getComputedStyle(probe);
        const matches =
          style.borderColor === probeStyle.borderColor &&
          style.color === probeStyle.color;
        probe.remove();
        return matches;
      }),
    )
    .toBe(true);
  const unavailableBasicOpacity = await page
    .locator("#basic-button")
    .evaluate((button) => {
      const style = getComputedStyle(button);
      return Number.parseFloat(style.opacity);
    });
  expect(unavailableBasicOpacity).toBeLessThan(1);
  await page.setViewportSize({ width: 960, height: 600 });
  await waitForStablePresentation(page);
  const commandDeckOverflow = await page
    .locator(".command-deck")
    .evaluate((element) => element.scrollWidth - element.clientWidth);
  expect(commandDeckOverflow).toBeLessThanOrEqual(1);
  await page.locator("#basic-button").hover();
  await page.locator("#basic-button").dispatchEvent("click");
  await expect(page.locator("#notice")).toContainText(
    "Basic ability is not available this tick.",
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
  await expect(actor0.locator(".pending-action-row__facts")).toContainText(
    "Target · Agent ID 6 (action 7)",
  );
  await expect(actor1).toHaveAttribute("data-controlled", "false");
  await expect(actor1).toHaveAttribute("data-move-action", "1");
  await expect(actor1).toHaveAttribute("data-target-slot", "5");
  await expect(actor1.locator(".pending-action-row__facts")).toContainText(
    "Target · Agent ID 5 (action 6)",
  );

  for (let actorSlot = 2; actorSlot < 10; actorSlot += 1) {
    await page.getByRole("button", { name: `Control Agent ID ${actorSlot}` }).click();
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

  await page.getByRole("button", { name: "Control Agent ID 0" }).click();
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
  const stagedInventory = await page
    .locator(".pending-action-row")
    .evaluateAll((rows) =>
      rows.map((row) => ({
        actorSlot: Number(row.getAttribute("data-actor-slot")),
        armedLane: row.getAttribute("data-armed-lane"),
        controlled: row.getAttribute("data-controlled"),
        moveAction: Number(row.getAttribute("data-move-action")),
        targetSlot: row.hasAttribute("data-target-slot")
          ? Number(row.getAttribute("data-target-slot"))
          : null,
        facts: [...row.querySelectorAll(".pending-action-chip")].map(
          (chip) => chip.textContent,
        ),
      })),
    );
  const stagedMoveActions = [3, 1, 3, 3, 3, 4, 4, 4, 4, 4];
  expect(stagedInventory).toEqual(
    stagedMoveActions.map((moveAction, actorSlot) => {
      const targetSlot = actorSlot === 0 ? 6 : actorSlot === 1 ? 5 : null;
      const targetAction = targetSlot === null ? 0 : targetSlot + 1;
      const movement = moveAction === 1 ? "North" : moveAction === 3 ? "East" : "West";
      return {
        actorSlot,
        armedLane: actorSlot < 2 ? null : "0",
        controlled: String(actorSlot === 0),
        moveAction,
        targetSlot,
        facts: [
          `Movement · ${movement} (${moveAction}) · Available`,
          targetSlot === null
            ? "Target · None"
            : `Target · Agent ID ${targetSlot} (action ${targetAction})`,
          "Action · No combat",
          "Legality · Not applicable",
        ],
      };
    }),
  );
  // The ten exact pending rows exceed the supported viewport height. Move the
  // real authoritative card—not a reconstructed copy—into a labelled, tall
  // review host so every staged tuple is visible in one evidence artifact.
  // Responsive runtime behavior remains asserted separately at 960×600.
  await page.setViewportSize({ width: 1440, height: 1600 });
  await page.evaluate(() => {
    const card = document.querySelector("#pending-card");
    if (!(card instanceof HTMLElement) || !card.parentElement) {
      throw new Error("Authoritative pending card is unavailable.");
    }
    const placeholder = document.createElement("span");
    placeholder.id = "joint-turn-evidence-placeholder";
    card.parentElement.insertBefore(placeholder, card);
    const host = document.createElement("section");
    host.id = "joint-turn-evidence";
    host.setAttribute("aria-label", "Authoritative ten-actor pending turn inventory");
    host.style.cssText =
      "position:fixed;left:16px;top:16px;z-index:1000;width:520px;" +
      "padding:14px;border:1px solid #334155;border-radius:10px;" +
      "background:#0b1020;color:#f4f7fb";
    const heading = document.createElement("h2");
    heading.textContent = "AUTHORITATIVE 10-ACTOR PENDING TURN · REVIEW INVENTORY";
    heading.style.cssText =
      "margin:0 0 10px;color:#22d3ee;font:700 14px/1.2 sans-serif;" +
      "letter-spacing:.06em";
    card.style.marginTop = "0";
    host.append(heading, card);
    document.body.append(host);
  });
  await waitForStablePresentation(page);
  await expect(page.locator("#joint-turn-evidence")).toHaveScreenshot(
    "joint-turn-ten-agent-pending-inventory-1440x1600.png",
    { maxDiffPixelRatio: DENSE_BASELINE_MAX_DIFF_PIXEL_RATIO },
  );
  await page.evaluate(() => {
    const card = document.querySelector("#pending-card");
    const placeholder = document.querySelector("#joint-turn-evidence-placeholder");
    const host = document.querySelector("#joint-turn-evidence");
    if (!(card instanceof HTMLElement) || !placeholder?.parentElement) {
      throw new Error("Pending evidence host cannot be restored.");
    }
    card.style.removeProperty("margin-top");
    placeholder.parentElement.insertBefore(card, placeholder);
    placeholder.remove();
    host?.remove();
  });

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
  const transitionId = await expectCurrentResearcherTransition(page, beforeSubmitStep);
  await expect(page.locator("#accepted-card .action-result")).toHaveCount(10);
  await expect.poll(() => enterCommands).toBe(1);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText(
    String(beforeSubmitRevision + 1),
  );
  await expect(page.locator("#step-value")).toHaveText(String(beforeSubmitStep + 1));
  await expect(page.locator("#transition-value")).toHaveText(transitionId);
  await page.unroute("**/api/command");

  await page.setViewportSize({ width: 960, height: 600 });
  const horizontalOverflow = await page
    .locator("#pending-card")
    .evaluate((element) => element.scrollWidth - element.clientWidth);
  expect(horizontalOverflow).toBeLessThanOrEqual(1);
});

test("rapid Submit settles active and paused explanations and sends each current draft once", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await installResearcherArena(page);
  const resetRevision = await currentRevision(page);
  await page.locator("#reset-button").click();
  await expect(page.locator("#revision-value")).toHaveText(String(resetRevision + 1));

  let enterCommands = 0;
  /** @type {number[]} */
  const enterBaseRevisions = [];
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    if (
      payload?.command?.command_type === "keyboard" &&
      String(payload.command.key).toLowerCase() === "enter"
    ) {
      enterCommands += 1;
      enterBaseRevisions.push(Number(payload.base_revision));
    }
    await route.continue();
  });

  const battlefield = page.locator("#battlefield");
  const armRevision = await currentRevision(page);
  await page.locator("#ultimate-button").click();
  await expect(page.locator("#revision-value")).toHaveText(String(armRevision + 1));
  const firstRevision = await currentRevision(page);
  const firstStep = await currentStep(page);
  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#revision-value")).toHaveText(String(firstRevision + 1), {
    timeout: 120_000,
  });
  await expect(page.locator("#step-value")).toHaveText(String(firstStep + 1));
  await expect.poll(() => enterCommands).toBe(1);
  await expect(page.locator("#motion-pause-button")).toBeEnabled();

  await finishControllerClock(page, "gate");
  await expect(page.locator("html")).not.toHaveAttribute("data-motion-paused", "true");
  await expect(page.locator("#motion-skip-button")).toBeEnabled();
  const activeControlRevision = await currentRevision(page);
  await page.getByRole("button", { name: "Control Agent ID 4" }).click();
  await expect(page.locator("#revision-value")).toHaveText(
    String(activeControlRevision + 1),
  );
  const activeDraftRevision = await currentRevision(page);
  await page.getByRole("button", { name: "Move east", exact: true }).click();
  await expect(page.locator("#revision-value")).toHaveText(
    String(activeDraftRevision + 1),
  );
  await expect(
    page.locator('.pending-action-row[data-controlled="true"]'),
  ).toHaveAttribute("data-move-action", "3");
  const activeTargetRevision = await currentRevision(page);
  await page.getByRole("button", { name: "Target Agent ID 3" }).click();
  await expect(page.locator("#revision-value")).toHaveText(
    String(activeTargetRevision + 1),
  );
  const activeArmRevision = await currentRevision(page);
  await page.locator("#basic-button").click();
  await expect(page.locator("#revision-value")).toHaveText(
    String(activeArmRevision + 1),
  );
  const activeRapidRevision = await currentRevision(page);
  const activeRapidStep = await currentStep(page);
  await page.locator("#submit-turn-button").evaluate((button) => {
    if (!(button instanceof HTMLButtonElement)) {
      throw new TypeError("Submit button is unavailable.");
    }
    button.click();
    button.click();
  });
  await expect(page.locator("#revision-value")).toHaveText(
    String(activeRapidRevision + 1),
    { timeout: 120_000 },
  );
  await expect(page.locator("#step-value")).toHaveText(String(activeRapidStep + 1));
  await expect.poll(() => enterCommands).toBe(2);
  await expect(
    page
      .locator(
        '.action-result[data-actor-slot="4"] .action-tuple[data-kind="submitted"] .fact',
      )
      .filter({ hasText: "Move action" })
      .locator("strong"),
  ).toHaveText("3");

  await page.locator("#motion-pause-button").click();
  await expect(page.locator("html")).toHaveAttribute("data-motion-paused", "true");

  const draftRevision = await currentRevision(page);
  await page.getByRole("button", { name: "Move north", exact: true }).click();
  await expect(page.locator("#revision-value")).toHaveText(String(draftRevision + 1));
  await expect(
    page.locator('.pending-action-row[data-controlled="true"]'),
  ).toHaveAttribute("data-move-action", "1");
  await expect(page.locator("html")).toHaveAttribute("data-motion-paused", "true");

  const rapidRevision = await currentRevision(page);
  const rapidStep = await currentStep(page);
  await page.locator("#submit-turn-button").evaluate((button) => {
    if (!(button instanceof HTMLButtonElement)) {
      throw new TypeError("Submit button is unavailable.");
    }
    button.click();
    button.click();
  });
  await expect(page.locator("#revision-value")).toHaveText(String(rapidRevision + 1), {
    timeout: 120_000,
  });
  await expect(page.locator("#step-value")).toHaveText(String(rapidStep + 1));
  await expect.poll(() => enterCommands).toBe(3);
  expect(enterBaseRevisions).toEqual([
    firstRevision,
    activeRapidRevision,
    rapidRevision,
  ]);
  const submittedMove = page
    .locator(
      '.action-result[data-actor-slot="4"] .action-tuple[data-kind="submitted"] .fact',
    )
    .filter({ hasText: "Move action" })
    .locator("strong");
  await expect(submittedMove).toHaveText("1");
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await page.unroute("**/api/command");
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

  const ranges = page.locator("#live-ranges-button");
  const staleRanges = stalePage.locator("#live-ranges-button");
  const initialRangesPressed = await ranges.getAttribute("aria-pressed");
  expect(["true", "false"]).toContain(initialRangesPressed);
  if (initialRangesPressed === null) {
    throw new Error("Live toggle controls did not publish pressed state.");
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
