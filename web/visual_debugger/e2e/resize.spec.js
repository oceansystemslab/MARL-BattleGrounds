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
 * Wait for font metrics, ResizeObserver delivery, and its requestAnimationFrame
 * redraw to settle.
 *
 * @param {import("@playwright/test").Page} page
 */
async function settleResponsiveLayout(page) {
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise((resolve) => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          requestAnimationFrame(resolve);
        });
      });
    });
  });
}

/**
 * Measure the supported split layout and the SVG size consumed by the real
 * renderer after ResizeObserver delivery.
 *
 * @param {import("@playwright/test").Page} page
 */
async function responsiveSnapshot(page) {
  return page.evaluate(() => {
    const workspace = document.querySelector(".workspace");
    const battlefieldPanel = document.querySelector(".battlefield-panel");
    const battlefield = document.querySelector("#battlefield");
    const hud = document.querySelector(".hud-panel");
    if (
      !(workspace instanceof HTMLElement) ||
      !(battlefieldPanel instanceof HTMLElement) ||
      !(battlefield instanceof SVGSVGElement) ||
      !(hud instanceof HTMLElement)
    ) {
      throw new Error("Supported resize measurement targets are unavailable.");
    }
    const battlefieldPanelBounds = battlefieldPanel.getBoundingClientRect();
    const hudBounds = hud.getBoundingClientRect();
    const viewBox = battlefield.viewBox.baseVal;
    return {
      battlefieldClientHeight: battlefield.clientHeight,
      battlefieldClientWidth: battlefield.clientWidth,
      battlefieldPanelRight: battlefieldPanelBounds.right,
      battlefieldPanelTop: battlefieldPanelBounds.top,
      documentClientWidth: document.documentElement.clientWidth,
      documentScrollWidth: document.documentElement.scrollWidth,
      gridColumns: getComputedStyle(workspace).gridTemplateColumns,
      hudLeft: hudBounds.left,
      hudTop: hudBounds.top,
      viewBoxHeight: viewBox.height,
      viewBoxWidth: viewBox.width,
    };
  });
}

test("supported resize preserves real installed authority at 960px", async ({
  page,
}) => {
  let commandRequests = 0;
  const authorityGets = new Set();
  page.on("request", (request) => {
    const pathname = new URL(request.url()).pathname;
    if (
      request.method() === "GET" &&
      (pathname === "/api/frame" || pathname === "/api/presentation/frame")
    ) {
      authorityGets.add(pathname);
    }
    if (request.method() === "POST" && pathname === "/api/command") {
      commandRequests += 1;
    }
  });

  await page.setViewportSize({ width: 1024, height: 768 });
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await settleResponsiveLayout(page);
  expect([...authorityGets].sort()).toEqual(["/api/frame", "/api/presentation/frame"]);

  const agentKeysBefore = await page
    .locator("#battlefield .agent")
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  expect(agentKeysBefore.length).toBeGreaterThan(0);
  expect(
    agentKeysBefore.every((key) => typeof key === "string" && key.length > 0),
  ).toBe(true);
  const revisionBefore = await page.locator("#revision-value").textContent();
  const initial = await responsiveSnapshot(page);
  expect(Math.abs(initial.viewBoxWidth - initial.battlefieldClientWidth)).toBeLessThan(
    1,
  );
  expect(
    Math.abs(initial.viewBoxHeight - initial.battlefieldClientHeight),
  ).toBeLessThan(1);

  const focusedControl = page
    .locator('#roster button[data-role="control"]:visible')
    .first();
  await expect(focusedControl).toBeEnabled();
  await focusedControl.focus();
  await expect(focusedControl).toBeFocused();

  await page.setViewportSize({ width: 960, height: 600 });
  await settleResponsiveLayout(page);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const resized = await responsiveSnapshot(page);
  expect(resized.documentScrollWidth).toBeLessThanOrEqual(resized.documentClientWidth);
  expect(resized.gridColumns.split(" ")).toHaveLength(2);
  expect(resized.hudLeft).toBeGreaterThanOrEqual(resized.battlefieldPanelRight);
  expect(Math.abs(resized.hudTop - resized.battlefieldPanelTop)).toBeLessThan(2);
  expect(Math.abs(resized.viewBoxWidth - resized.battlefieldClientWidth)).toBeLessThan(
    1,
  );
  expect(
    Math.abs(resized.viewBoxHeight - resized.battlefieldClientHeight),
  ).toBeLessThan(1);
  expect(
    Math.abs(resized.battlefieldClientWidth - initial.battlefieldClientWidth) > 1 ||
      Math.abs(resized.battlefieldClientHeight - initial.battlefieldClientHeight) > 1,
  ).toBe(true);

  expect(
    await page
      .locator("#battlefield .agent")
      .evaluateAll((agents) =>
        agents.map((agent) => agent.getAttribute("data-presentation-key")),
      ),
  ).toEqual(agentKeysBefore);
  await expect(page.locator("#revision-value")).toHaveText(revisionBefore ?? "");
  await expect(focusedControl).toBeFocused();
  expect(commandRequests).toBe(0);
});
