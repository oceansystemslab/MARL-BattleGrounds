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
  await expect(page.locator("#revision-value")).toHaveText(revisionBefore ?? "");
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
