import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import { startReplayViewer } from "./support/replay-viewer.js";

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

/**
 * Measure the unit grid installed by the actual Debugger service and perform a
 * browser hit test at one line midpoint.
 *
 * @param {import("@playwright/test").Page} page
 */
async function debuggerGridSnapshot(page) {
  return page.locator("#battlefield").evaluate((battlefield) => {
    if (!(battlefield instanceof SVGSVGElement)) {
      throw new Error("Debugger battlefield is unavailable.");
    }
    const mapLayer = battlefield.querySelector('[data-layer="map"]');
    const obstacleLayer = battlefield.querySelector('[data-layer="obstacle"]');
    const bodyLayer = battlefield.querySelector('[data-layer="body"]');
    const boundary = battlefield.querySelector(".map-boundary");
    const vertical = [...battlefield.querySelectorAll(".map-grid-line--vertical")];
    const horizontal = [...battlefield.querySelectorAll(".map-grid-line--horizontal")];
    if (
      !(mapLayer instanceof SVGGElement) ||
      !(obstacleLayer instanceof SVGGElement) ||
      !(bodyLayer instanceof SVGGElement) ||
      !(boundary instanceof SVGRectElement) ||
      !(vertical[0] instanceof SVGLineElement)
    ) {
      throw new Error("Debugger grid layers are unavailable.");
    }
    /** @param {Element} element @param {string} name */
    const numberAttribute = (element, name) => {
      const value = Number(element.getAttribute(name));
      if (!Number.isFinite(value)) {
        throw new Error(`Grid ${name} is not finite.`);
      }
      return value;
    };
    const boundaryBox = {
      x: numberAttribute(boundary, "x"),
      y: numberAttribute(boundary, "y"),
      width: numberAttribute(boundary, "width"),
      height: numberAttribute(boundary, "height"),
    };
    const first = vertical[0];
    const midpoint = battlefield.createSVGPoint();
    midpoint.x = numberAttribute(first, "x1");
    midpoint.y = (numberAttribute(first, "y1") + numberAttribute(first, "y2")) / 2;
    const matrix = first.getScreenCTM();
    if (matrix === null) {
      throw new Error("Debugger grid has no screen transform.");
    }
    const screenMidpoint = midpoint.matrixTransform(matrix);
    const hit = document.elementFromPoint(screenMidpoint.x, screenMidpoint.y);
    /** @param {Element} layer */
    const followsMap = (layer) =>
      Boolean(
        mapLayer.compareDocumentPosition(layer) & Node.DOCUMENT_POSITION_FOLLOWING,
      );
    return {
      vertical: vertical.map((line) => ({
        x: numberAttribute(line, "x1"),
        y1: numberAttribute(line, "y1"),
        y2: numberAttribute(line, "y2"),
      })),
      horizontal: horizontal.map((line) => ({
        y: numberAttribute(line, "y1"),
        x1: numberAttribute(line, "x1"),
        x2: numberAttribute(line, "x2"),
      })),
      boundary: boundaryBox,
      mapAriaHidden: mapLayer.getAttribute("aria-hidden"),
      mapIsFirstLayer: mapLayer === battlefield.querySelector("[data-layer]"),
      mapPrecedesObstacleAndBody: followsMap(obstacleLayer) && followsMap(bodyLayer),
      lineAccessibility: [...vertical, ...horizontal].map((line) => ({
        role: line.getAttribute("role"),
        label: line.getAttribute("aria-label"),
        tabindex: line.getAttribute("tabindex"),
        pointerEvents: getComputedStyle(line).pointerEvents,
      })),
      hitIsGrid: hit?.closest(".map-grid-line") !== null,
    };
  });
}

/**
 * Prove that real disclosure bodies, rather than the shared HUD or document,
 * own vertical scrolling at one supported viewport.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{width: number, height: number}} viewport
 */
async function expectIndependentDisclosureLayout(page, viewport) {
  await page.setViewportSize(viewport);
  await settleResponsiveLayout(page);
  await expect(page.locator(".session-toolbar")).toBeVisible();

  const initial = await page.evaluate(() => {
    const toolbar = document.querySelector(".session-toolbar");
    const roster = document.querySelector("#roster-details-body");
    const command = document.querySelector("#command-deck-body");
    const hud = document.querySelector(".hud-panel");
    if (
      !(toolbar instanceof HTMLElement) ||
      !(roster instanceof HTMLElement) ||
      !(command instanceof HTMLElement) ||
      !(hud instanceof HTMLElement)
    ) {
      throw new Error("Independent disclosure layout targets are unavailable.");
    }
    const toolbarBounds = toolbar.getBoundingClientRect();
    return {
      commandClientHeight: command.clientHeight,
      commandOverflowY: getComputedStyle(command).overflowY,
      commandScrollHeight: command.scrollHeight,
      commandScrollTop: command.scrollTop,
      documentClientWidth: document.documentElement.clientWidth,
      documentScrollTop: document.scrollingElement?.scrollTop ?? -1,
      documentScrollWidth: document.documentElement.scrollWidth,
      hudScrollTop: hud.scrollTop,
      rosterClientHeight: roster.clientHeight,
      rosterOverflowY: getComputedStyle(roster).overflowY,
      rosterScrollHeight: roster.scrollHeight,
      toolbarBottom: toolbarBounds.bottom,
      toolbarLeft: toolbarBounds.left,
      toolbarRight: toolbarBounds.right,
      toolbarTop: toolbarBounds.top,
    };
  });
  expect(initial.toolbarTop).toBeGreaterThanOrEqual(0);
  expect(initial.toolbarLeft).toBeGreaterThanOrEqual(0);
  expect(initial.toolbarRight).toBeLessThanOrEqual(viewport.width);
  expect(initial.toolbarBottom).toBeLessThanOrEqual(viewport.height);
  expect(initial.rosterOverflowY).toBe("auto");
  expect(initial.commandOverflowY).toBe("auto");
  expect(initial.rosterScrollHeight).toBeGreaterThan(initial.rosterClientHeight);
  expect(initial.commandScrollHeight).toBeGreaterThan(initial.commandClientHeight);
  expect(initial.documentScrollWidth).toBeLessThanOrEqual(initial.documentClientWidth);

  const rosterBody = page.locator("#roster-details-body");
  await rosterBody.evaluate((body) => {
    body.scrollTop = Math.min(48, body.scrollHeight - body.clientHeight);
  });
  const independentlyScrolled = await page.evaluate(() => {
    const roster = document.querySelector("#roster-details-body");
    const command = document.querySelector("#command-deck-body");
    const hud = document.querySelector(".hud-panel");
    if (
      !(roster instanceof HTMLElement) ||
      !(command instanceof HTMLElement) ||
      !(hud instanceof HTMLElement)
    ) {
      throw new Error("Disclosure scroll targets disappeared.");
    }
    return {
      commandScrollTop: command.scrollTop,
      documentScrollTop: document.scrollingElement?.scrollTop ?? -1,
      hudScrollTop: hud.scrollTop,
      rosterScrollTop: roster.scrollTop,
    };
  });
  expect(independentlyScrolled.rosterScrollTop).toBeGreaterThan(0);
  expect(independentlyScrolled.commandScrollTop).toBe(initial.commandScrollTop);
  expect(independentlyScrolled.hudScrollTop).toBe(initial.hudScrollTop);
  expect(independentlyScrolled.documentScrollTop).toBe(initial.documentScrollTop);

  await rosterBody.evaluate((body) => {
    body.scrollTop = body.scrollHeight;
  });
  const rosterBoundary = await rosterBody.evaluate((body) => body.scrollTop);
  await rosterBody.hover();
  await page.mouse.wheel(0, 600);
  await settleResponsiveLayout(page);
  expect(await rosterBody.evaluate((body) => body.scrollTop)).toBe(rosterBoundary);
  expect(
    await page.locator("#command-deck-body").evaluate((body) => body.scrollTop),
  ).toBe(initial.commandScrollTop);
  expect(await page.locator(".hud-panel").evaluate((hud) => hud.scrollTop)).toBe(
    initial.hudScrollTop,
  );
  expect(await page.evaluate(() => document.scrollingElement?.scrollTop ?? -1)).toBe(
    initial.documentScrollTop,
  );

  const lastRosterAction = page
    .locator("#roster .roster-primary-action:visible")
    .last();
  await lastRosterAction.focus();
  await expect(lastRosterAction).toBeFocused();
  const focusedBounds = await page.evaluate(() => {
    const body = document.querySelector("#roster-details-body");
    const active = document.activeElement;
    if (!(body instanceof HTMLElement) || !(active instanceof HTMLElement)) {
      throw new Error("Focused roster action is unavailable.");
    }
    const bodyBounds = body.getBoundingClientRect();
    const activeBounds = active.getBoundingClientRect();
    return {
      activeBottom: activeBounds.bottom,
      activeTop: activeBounds.top,
      bodyBottom: bodyBounds.bottom,
      bodyTop: bodyBounds.top,
    };
  });
  expect(focusedBounds.activeTop).toBeGreaterThanOrEqual(focusedBounds.bodyTop - 1);
  expect(focusedBounds.activeBottom).toBeLessThanOrEqual(focusedBounds.bodyBottom + 1);

  const followingTopBefore = await page
    .locator("#agent-details")
    .evaluate((panel) => panel.getBoundingClientRect().top);
  await page.locator("#roster-details").evaluate((panel) => {
    if (!(panel instanceof HTMLDetailsElement)) {
      throw new TypeError("Roster disclosure is unavailable.");
    }
    panel.open = false;
  });
  await expect(page.locator("#roster-details > summary")).toBeFocused();
  await settleResponsiveLayout(page);
  expect(
    await page
      .locator("#agent-details")
      .evaluate((panel) => panel.getBoundingClientRect().top),
  ).toBeLessThan(followingTopBefore);
  const documentScrollBeforeSpace = await page.evaluate(
    () => document.scrollingElement?.scrollTop ?? -1,
  );
  await page.locator("#roster-details > summary").press("Space");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  expect(await page.evaluate(() => document.scrollingElement?.scrollTop ?? -1)).toBe(
    documentScrollBeforeSpace,
  );
  await settleResponsiveLayout(page);

  const containment = await page.evaluate(() => {
    const visibleSections = [...document.querySelectorAll(".hud-panel > details")]
      .filter((panel) => panel.getClientRects().length > 0)
      .map((panel) => panel.getBoundingClientRect())
      .sort((left, right) => left.top - right.top);
    const bodies = [...document.querySelectorAll("details[open] > .disclosure-body")]
      .filter((body) => body.getClientRects().length > 0)
      .map((body) => {
        const owner = body.parentElement;
        if (!(owner instanceof HTMLDetailsElement)) {
          throw new Error("Disclosure body lost its owning details element.");
        }
        const bodyBounds = body.getBoundingClientRect();
        const ownerBounds = owner.getBoundingClientRect();
        return {
          bodyLeft: bodyBounds.left,
          bodyRight: bodyBounds.right,
          bodyScrollWidth: body.scrollWidth,
          bodyClientWidth: body.clientWidth,
          ownerLeft: ownerBounds.left,
          ownerRight: ownerBounds.right,
          ownerTop: ownerBounds.top,
          ownerBottom: ownerBounds.bottom,
          bodyTop: bodyBounds.top,
          bodyBottom: bodyBounds.bottom,
        };
      });
    return {
      bodies,
      sections: visibleSections.map((bounds) => ({
        bottom: bounds.bottom,
        top: bounds.top,
      })),
    };
  });
  for (let index = 1; index < containment.sections.length; index += 1) {
    expect(containment.sections[index].top).toBeGreaterThanOrEqual(
      containment.sections[index - 1].bottom - 1,
    );
  }
  for (const bounds of containment.bodies) {
    expect(bounds.bodyTop).toBeGreaterThanOrEqual(bounds.ownerTop - 1);
    expect(bounds.bodyBottom).toBeLessThanOrEqual(bounds.ownerBottom + 1);
    expect(bounds.bodyLeft).toBeGreaterThanOrEqual(bounds.ownerLeft - 1);
    expect(bounds.bodyRight).toBeLessThanOrEqual(bounds.ownerRight + 1);
    expect(bounds.bodyScrollWidth).toBeLessThanOrEqual(bounds.bodyClientWidth + 1);
  }
}

test("real Debugger grid is exact, lowest, and click-transparent", async ({ page }) => {
  let commandRequests = 0;
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/command"
    ) {
      commandRequests += 1;
    }
  });
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const stepBefore = await page.locator("#step-value").textContent();
  const transitionBefore = await page.locator("#transition-value").textContent();

  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 960, height: 600 },
  ]) {
    await page.setViewportSize(viewport);
    await settleResponsiveLayout(page);
    const snapshot = await debuggerGridSnapshot(page);
    expect(snapshot.vertical).toHaveLength(19);
    expect(snapshot.horizontal).toHaveLength(9);
    expect(snapshot.mapAriaHidden).toBe("true");
    expect(snapshot.mapIsFirstLayer).toBe(true);
    expect(snapshot.mapPrecedesObstacleAndBody).toBe(true);
    expect(snapshot.hitIsGrid).toBe(false);
    expect(
      snapshot.lineAccessibility.every(
        ({ role, label, tabindex, pointerEvents }) =>
          role === null &&
          label === null &&
          tabindex === null &&
          pointerEvents === "none",
      ),
    ).toBe(true);
    for (const [index, line] of snapshot.vertical.entries()) {
      expect((line.x - snapshot.boundary.x) / snapshot.boundary.width).toBeCloseTo(
        (index + 1) / 20,
        8,
      );
      expect([line.y1, line.y2].sort((left, right) => left - right)).toEqual([
        snapshot.boundary.y,
        snapshot.boundary.y + snapshot.boundary.height,
      ]);
    }
    for (const [index, line] of snapshot.horizontal
      .sort((left, right) => left.y - right.y)
      .entries()) {
      expect((line.y - snapshot.boundary.y) / snapshot.boundary.height).toBeCloseTo(
        (index + 1) / 10,
        8,
      );
      expect([line.x1, line.x2].sort((left, right) => left - right)).toEqual([
        snapshot.boundary.x,
        snapshot.boundary.x + snapshot.boundary.width,
      ]);
    }
  }
  await expect(page.locator("#step-value")).toHaveText(stepBefore ?? "");
  await expect(page.locator("#transition-value")).toHaveText(transitionBefore ?? "");
  expect(commandRequests).toBe(0);
});

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
  const stepBefore = await page.locator("#step-value").textContent();
  const transitionBefore = await page.locator("#transition-value").textContent();
  const initial = await responsiveSnapshot(page);
  expect(Math.abs(initial.viewBoxWidth - initial.battlefieldClientWidth)).toBeLessThan(
    1,
  );
  expect(
    Math.abs(initial.viewBoxHeight - initial.battlefieldClientHeight),
  ).toBeLessThan(1);

  const focusedControl = page.locator("#roster .roster-primary-action:visible").first();
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
  await expect(page.locator("#step-value")).toHaveText(stepBefore ?? "");
  await expect(page.locator("#transition-value")).toHaveText(transitionBefore ?? "");
  await expect(focusedControl).toBeFocused();
  expect(commandRequests).toBe(0);
});

test("disclosure bodies scroll independently at both supported viewports", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 960, height: 600 },
  ]) {
    await expectIndependentDisclosureLayout(page, viewport);
  }

  const replay = await startReplayViewer({
    sampleReplay: "death-respawn-shield",
  });
  try {
    for (const viewport of [
      { width: 1440, height: 900 },
      { width: 960, height: 600 },
    ]) {
      await page.setViewportSize(viewport);
      await page.goto(replay.url);
      await expect(page.locator("#connection-status")).toHaveText("Online");
      await expect(page.locator("html")).toHaveAttribute(
        "data-presentation-authority",
        "installed",
      );
      await settleResponsiveLayout(page);
      await expect(page.locator(".replay-timeline__transport")).toBeVisible();
      const transport = await page
        .locator(".replay-timeline__transport")
        .evaluate((element) => {
          const bounds = element.getBoundingClientRect();
          return {
            bottom: bounds.bottom,
            clientWidth: element.clientWidth,
            left: bounds.left,
            right: bounds.right,
            scrollWidth: element.scrollWidth,
            top: bounds.top,
          };
        });
      expect(transport.top).toBeGreaterThanOrEqual(0);
      expect(transport.left).toBeGreaterThanOrEqual(0);
      expect(transport.right).toBeLessThanOrEqual(viewport.width);
      expect(transport.bottom).toBeLessThanOrEqual(viewport.height);
      expect(transport.scrollWidth).toBeLessThanOrEqual(transport.clientWidth + 1);
      expect(
        await page.evaluate(
          () =>
            document.documentElement.scrollWidth <=
            document.documentElement.clientWidth,
        ),
      ).toBe(true);
    }
  } finally {
    await stopDebugger(replay.process);
  }
});
