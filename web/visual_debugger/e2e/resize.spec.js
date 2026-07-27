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
let syntheticFrame = {};

test.beforeAll(async () => {
  const [started, fixture] = await Promise.all([
    startDebugger(),
    loadRendererFixture("viewport_matrix"),
  ]);
  serverProcess = started.process;
  debuggerUrl = started.url;
  syntheticFrame = syntheticDebuggerFrame(fixture);
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

test("resize observer preserves authoritative DOM across split and stacked layouts", async ({
  page,
}) => {
  let commandRequests = 0;
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: syntheticFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    commandRequests += 1;
    await route.abort("blockedbyclient");
  });

  await page.goto(debuggerUrl);
  await settleResponsiveLayout(page);
  const expectedStatusCount = syntheticFrame.scene.agents.reduce(
    /**
     * @param {number} total
     * @param {{statuses: unknown[]}} agent
     */
    (total, agent) => total + agent.statuses.length,
    0,
  );
  await expect(page.locator("#roster .roster-row")).toHaveCount(10);
  await expect(page.locator("#roster .roster-fact-token--status")).toHaveCount(
    expectedStatusCount,
  );
  const focusedControl = page.getByRole("button", { name: "Control id_0" });
  await focusedControl.focus();
  await expect(focusedControl).toBeFocused();
  await page.locator("#battlefield").evaluate((battlefield) => {
    const agent = battlefield.querySelector('.agent[data-slot="0"]');
    const transient = battlefield.querySelector('[data-layer="transient-events"]');
    if (!agent || !transient) {
      throw new Error("Retained resize probes could not be installed.");
    }
    agent.setAttribute("data-resize-probe", "agent");
    const cue = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    cue.setAttribute("data-resize-probe", "transient");
    transient.append(cue);
  });

  const cases = [
    { width: 1601, height: 900, layout: "split" },
    { width: 1440, height: 900, layout: "split" },
    { width: 1024, height: 768, layout: "split" },
    { width: 960, height: 600, layout: "split" },
    { width: 959, height: 600, layout: "stacked" },
    { width: 800, height: 900, layout: "stacked" },
  ];
  for (const viewport of cases) {
    await page.setViewportSize({
      width: viewport.width,
      height: viewport.height,
    });
    await settleResponsiveLayout(page);
    const measurement = await page.evaluate(() => {
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
        throw new Error("Responsive measurement targets are unavailable.");
      }
      const battlefieldPanelBounds = battlefieldPanel.getBoundingClientRect();
      const hudBounds = hud.getBoundingClientRect();
      const viewBox = (battlefield.getAttribute("viewBox") ?? "")
        .split(/\s+/)
        .map(Number);
      return {
        battlefieldClientHeight: battlefield.clientHeight,
        battlefieldClientWidth: battlefield.clientWidth,
        battlefieldPanel: {
          bottom: battlefieldPanelBounds.bottom,
          left: battlefieldPanelBounds.left,
          right: battlefieldPanelBounds.right,
          top: battlefieldPanelBounds.top,
        },
        bodyScrollHeight: document.body.scrollHeight,
        documentClientWidth: document.documentElement.clientWidth,
        documentScrollHeight: document.documentElement.scrollHeight,
        documentScrollWidth: document.documentElement.scrollWidth,
        gridColumns: getComputedStyle(workspace).gridTemplateColumns,
        hud: {
          bottom: hudBounds.bottom,
          clientHeight: hud.clientHeight,
          left: hudBounds.left,
          overflowY: getComputedStyle(hud).overflowY,
          right: hudBounds.right,
          scrollHeight: hud.scrollHeight,
          top: hudBounds.top,
        },
        viewBox,
      };
    });

    expect(
      measurement.documentScrollWidth,
      `${viewport.width} viewport must not overflow horizontally`,
    ).toBeLessThanOrEqual(measurement.documentClientWidth);
    expect(
      measurement.documentScrollHeight,
      `${viewport.width} viewport HUD overflow must not inflate document height`,
    ).toBeLessThanOrEqual(measurement.bodyScrollHeight + 1);
    expect(measurement.hud.overflowY).toBe("auto");
    expect(measurement.hud.scrollHeight).toBeGreaterThan(measurement.hud.clientHeight);
    if (viewport.layout === "split") {
      expect(measurement.gridColumns.split(" ")).toHaveLength(2);
      expect(measurement.hud.left).toBeGreaterThanOrEqual(
        measurement.battlefieldPanel.right,
      );
      expect(
        Math.abs(measurement.hud.top - measurement.battlefieldPanel.top),
      ).toBeLessThan(2);
    } else {
      expect(measurement.gridColumns.split(" ")).toHaveLength(1);
      expect(measurement.hud.top).toBeGreaterThanOrEqual(
        measurement.battlefieldPanel.bottom,
      );
    }
    expect(measurement.viewBox).toHaveLength(4);
    expect(
      Math.abs(measurement.viewBox[2] - measurement.battlefieldClientWidth),
    ).toBeLessThan(1);
    expect(
      Math.abs(measurement.viewBox[3] - measurement.battlefieldClientHeight),
    ).toBeLessThan(1);
    await expect(page.locator("#revision-value")).toHaveText("0");
    await expect(page.locator("#roster .roster-row")).toHaveCount(10);
    await expect(page.locator("#roster .roster-fact-token--status")).toHaveCount(
      expectedStatusCount,
    );
    await expect(
      page.locator('#battlefield .agent[data-resize-probe="agent"]'),
    ).toHaveCount(1);
    await expect(
      page.locator(
        '#battlefield [data-layer="transient-events"] [data-resize-probe="transient"]',
      ),
    ).toHaveCount(1);
    await expect(focusedControl).toBeFocused();
  }
  expect(commandRequests).toBe(0);
});

test("toolbar wrapping preserves DOM, visual, and keyboard focus order", async ({
  page,
}) => {
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: syntheticFrame,
      status: 200,
    });
  });
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");

  for (const viewport of [
    { width: 1601, height: 900 },
    { width: 1440, height: 900 },
    { width: 960, height: 600 },
  ]) {
    await page.setViewportSize(viewport);
    await settleResponsiveLayout(page);
    const measurement = await page.evaluate(() => {
      const toolbar = document.querySelector(".session-toolbar");
      const facts = document.querySelector(".session-facts");
      const motion = document.querySelector(".motion-controls");
      const actions = document.querySelector(".toolbar-actions");
      if (
        !(toolbar instanceof HTMLElement) ||
        !(facts instanceof HTMLElement) ||
        !(motion instanceof HTMLElement) ||
        !(actions instanceof HTMLElement)
      ) {
        throw new Error("Toolbar ordering targets are unavailable.");
      }
      /**
       * @param {Element} element
       */
      const bounds = (element) => {
        const box = element.getBoundingClientRect();
        return {
          bottom: box.bottom,
          left: box.left,
          right: box.right,
          top: box.top,
        };
      };
      const focusable = [...toolbar.querySelectorAll("select, input, button")].filter(
        (element) =>
          (element instanceof HTMLButtonElement ||
            element instanceof HTMLInputElement ||
            element instanceof HTMLSelectElement) &&
          !element.disabled,
      );
      return {
        actions: bounds(actions),
        documentClientWidth: document.documentElement.clientWidth,
        documentScrollWidth: document.documentElement.scrollWidth,
        facts: bounds(facts),
        flexDirection: getComputedStyle(toolbar).flexDirection,
        flexWrap: getComputedStyle(toolbar).flexWrap,
        focusKeys: focusable.map(
          (element) =>
            element.id ||
            `motion-rate:${element.getAttribute("data-motion-rate") ?? "missing"}`,
        ),
        motion: bounds(motion),
        orders: [facts, motion, actions].map(
          (element) => getComputedStyle(element).order,
        ),
      };
    });
    /**
     * @param {{bottom: number, left: number, right: number, top: number}} first
     * @param {{bottom: number, left: number, right: number, top: number}} second
     */
    const visuallyPrecedes = (first, second) =>
      first.bottom <= second.top + 1 ||
      (Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top) > 1 &&
        first.left < second.left);

    expect(measurement.flexDirection).toBe("row");
    expect(measurement.flexWrap).toBe("wrap");
    expect(measurement.orders).toEqual(["0", "0", "0"]);
    expect(
      visuallyPrecedes(measurement.facts, measurement.motion),
      `${viewport.width}px facts must precede motion visually`,
    ).toBe(true);
    expect(
      visuallyPrecedes(measurement.motion, measurement.actions),
      `${viewport.width}px motion must precede actions visually`,
    ).toBe(true);
    expect(measurement.documentScrollWidth).toBeLessThanOrEqual(
      measurement.documentClientWidth,
    );

    const toolbarControls = page.locator(
      ".session-toolbar select:not(:disabled), " +
        ".session-toolbar input:not(:disabled), " +
        ".session-toolbar button:not(:disabled)",
    );
    await toolbarControls.first().focus();
    const actualFocusKeys = [];
    for (let index = 0; index < measurement.focusKeys.length; index += 1) {
      actualFocusKeys.push(
        await page.evaluate(() => {
          const active = document.activeElement;
          if (!(active instanceof HTMLElement)) {
            return "missing";
          }
          return (
            active.id ||
            `motion-rate:${active.getAttribute("data-motion-rate") ?? "missing"}`
          );
        }),
      );
      if (index < measurement.focusKeys.length - 1) {
        await page.keyboard.press("Tab");
      }
    }
    expect(actualFocusKeys).toEqual(measurement.focusKeys);
  }
});
