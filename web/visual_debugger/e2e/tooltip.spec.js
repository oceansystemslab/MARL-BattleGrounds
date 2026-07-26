import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  loadRendererFixture,
  syntheticDebuggerFrame,
} from "./support/renderer-fixture.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";
/** @type {Record<string, any>} */
let crowdedFrame = {};
/** @type {Record<string, any>} */
let povFrame = {};
/** @type {Record<string, any>} */
let vocabularyFrame = {};

test.use({ viewport: { width: 960, height: 600 } });

test.beforeAll(async () => {
  const [started, crowded, pov, vocabulary] = await Promise.all([
    startDebugger(),
    loadRendererFixture("crowded_teamfight"),
    loadRendererFixture("pov_redaction"),
    loadRendererFixture("visual_vocabulary"),
  ]);
  serverProcess = started.process;
  debuggerUrl = started.url;
  crowdedFrame = syntheticDebuggerFrame(crowded);
  povFrame = syntheticDebuggerFrame(pov);
  vocabularyFrame = syntheticDebuggerFrame(vocabulary);
});

test.afterAll(async () => {
  const child = serverProcess;
  serverProcess = null;
  await stopDebugger(child);
});

/**
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} frame
 */
async function installFrame(page, frame) {
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: frame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    await route.abort("blockedbyclient");
  });
}

test("one tooltip arbitrates dense battlefield and keyboard explanations", async ({
  page,
}) => {
  const frame = structuredClone(crowdedFrame);
  frame.scene.agents[2].ultimate_cooldown = 30;
  await installFrame(page, frame);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const tooltip = page.locator("#visual-tooltip");
  const title = page.locator("#visual-tooltip-title");
  const details = page.locator("#visual-tooltip-details");
  const agent = page.locator('#battlefield .agent[data-slot="0"]');
  await agent.locator(".agent-body").hover();
  await expect(tooltip).toHaveAttribute("data-tooltip-kind", "agent");
  await expect(title).toContainText("id_0");

  const statusCells = page.locator("#battlefield .status-cell");
  expect(await statusCells.count()).toBeGreaterThan(1);
  await statusCells.nth(0).hover();
  const firstStatusTitle = await title.textContent();
  await expect(tooltip).toHaveAttribute("data-tooltip-kind", "status");
  await statusCells.nth(1).hover();
  await expect(title).not.toHaveText(firstStatusTitle ?? "");

  const overflow = page
    .locator("#battlefield .status-overflow, #battlefield .modifier-overflow")
    .first();
  await expect(overflow).toBeVisible();
  const hiddenCount = Number(await overflow.getAttribute("data-hidden-count"));
  await overflow.hover();
  await expect(tooltip).toHaveAttribute(
    "data-tooltip-kind",
    /^(status|modifier)-overflow$/,
  );
  const listedFacts = (await details.textContent())
    ?.split("\n")
    .filter((line) => line.trim() !== "");
  expect(listedFacts?.length).toBe(hiddenCount);

  const cooldown = page.locator("#battlefield .cooldown-cell").first();
  await cooldown.hover();
  await expect(tooltip).toHaveAttribute("data-tooltip-kind", "cooldown");
  await expect(details).toContainText(/tick/);

  const targetButton = page.locator(
    '#roster .roster-row[data-slot="0"] button[data-role="target"]',
  );
  await page.locator("#scenario-select").focus();
  await targetButton.focus();
  await expect(targetButton).toHaveAttribute("aria-describedby", "visual-tooltip");
  await expect(title).toContainText("id_0");
  await expect(page.locator('#roster .roster-row[data-slot="0"]')).not.toHaveAttribute(
    "aria-describedby",
    /visual-tooltip/,
  );

  await agent.locator(".agent-body").hover();
  await expect(targetButton).not.toHaveAttribute("aria-describedby", /visual-tooltip/);
  await page.locator("body").dispatchEvent("pointerleave");
  await expect(targetButton).toHaveAttribute("aria-describedby", "visual-tooltip");
  await expect(title).toContainText("id_0");

  await page.keyboard.press("Escape");
  await expect(tooltip).toBeHidden();
  await expect(targetButton).not.toHaveAttribute("aria-describedby", /visual-tooltip/);

  await page.locator("#scenario-select").focus();
  await targetButton.focus();
  await expect(tooltip).toBeVisible();
  const bounds = await tooltip.boundingBox();
  expect(bounds).not.toBeNull();
  expect(bounds?.x).toBeGreaterThanOrEqual(8);
  expect(bounds?.y).toBeGreaterThanOrEqual(8);
  expect((bounds?.x ?? 0) + (bounds?.width ?? 0)).toBeLessThanOrEqual(952);
  expect((bounds?.y ?? 0) + (bounds?.height ?? 0)).toBeLessThanOrEqual(592);

  await page.locator('#roster .roster-row[data-slot="0"]').evaluate((row) => {
    row.remove();
  });
  await expect(tooltip).toBeHidden();
});

test("range hits are inspectable and POV explanations retain redaction", async ({
  page,
}) => {
  await installFrame(page, vocabularyFrame);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const basicRangeHit = page.locator('#battlefield .range-ring-hit[data-kind="basic"]');
  await basicRangeHit.scrollIntoViewIfNeeded();
  const rangeProbe = await basicRangeHit.evaluate((circle) => {
    if (!(circle instanceof SVGCircleElement)) {
      throw new Error("Expected an SVG range hit circle.");
    }
    const matrix = circle.getScreenCTM();
    if (!matrix) {
      throw new Error("Range hit circle has no screen transform.");
    }
    const candidates = [];
    for (const angle of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) {
      const radius = Math.max(circle.r.baseVal.value - 2, 0);
      const point = new DOMPoint(
        circle.cx.baseVal.value + Math.cos(angle) * radius,
        circle.cy.baseVal.value + Math.sin(angle) * radius,
      ).matrixTransform(matrix);
      const elements = document.elementsFromPoint(point.x, point.y);
      candidates.push({
        classes: elements.map((element) => element.getAttribute("class")),
        includesHitCircle: elements.includes(circle),
        x: point.x,
        y: point.y,
      });
      if (
        elements.includes(circle) &&
        !elements.some((element) =>
          element.closest(
            ".agent, .status-cell, .modifier-cell, .cooldown-cell, .legality-pill, .obstacle",
          ),
        )
      ) {
        return {
          candidates,
          point: { x: point.x, y: point.y },
          pointerEvents: getComputedStyle(circle).pointerEvents,
          stroke: getComputedStyle(circle).stroke,
          strokeWidth: getComputedStyle(circle).strokeWidth,
        };
      }
    }
    return {
      candidates,
      point: null,
      pointerEvents: getComputedStyle(circle).pointerEvents,
      stroke: getComputedStyle(circle).stroke,
      strokeWidth: getComputedStyle(circle).strokeWidth,
    };
  });
  expect(rangeProbe.point, JSON.stringify(rangeProbe)).not.toBeNull();
  await page.mouse.move(rangeProbe.point?.x ?? 0, rangeProbe.point?.y ?? 0);
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    /^range-/,
  );

  await page.unroute("**/api/frame");
  await page.unroute("**/api/command");
  await installFrame(page, povFrame);
  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const activation = page.locator(
    '#event-feed .event-item[data-event-type="accepted_activation"]',
  );
  await activation.hover();
  await expect(page.locator("#visual-tooltip-details")).toContainText(
    "Target endpoint not disclosed in this view",
  );
  await expect(page.locator("#visual-tooltip-details")).not.toContainText(/Target id_/);
  const tooltipText = await page.locator("#visual-tooltip").textContent();
  expect(tooltipText).not.toContain("target_anchor");
  expect(tooltipText).not.toMatch(/\b(?:37|30)\b/);
});
