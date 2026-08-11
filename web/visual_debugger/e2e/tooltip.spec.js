import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  loadRendererFixture,
  syntheticDebuggerFrame,
} from "./support/renderer-fixture.js";
import { waitForStablePresentation } from "./support/visual-regression.js";

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

/**
 * Return a canonical researcher frame-zero envelope with no incoming events.
 * Mutating normalized aliases would be ignored by the strict API boundary;
 * intercepted tests therefore edit the raw wire projection itself.
 *
 * @param {Record<string, any>} frame
 */
function withoutIncomingResearcherEvents(frame) {
  const initial = structuredClone(frame);
  initial.frame_index = 0;
  initial.frame_id = `${initial.episode_id}:frame:0`;
  initial.simulator_step_count = 0;
  initial.incoming_transition_index = null;
  initial.incoming_transition_id = null;
  initial.hud.latest_transition = null;
  initial.projection.scene.frame_index = 0;
  initial.projection.scene.frame_id = initial.frame_id;
  initial.projection.scene.simulator_step_count = 0;
  initial.projection.scene.incoming_transition_id = null;
  initial.projection.scene.incoming_event_ids = [];
  initial.projection.incoming_events = null;
  return initial;
}

/**
 * Find a stable pointer coordinate where one agent remains the foreground
 * semantic owner over a broad aura or range field.
 *
 * @param {import("@playwright/test").Locator} agent
 */
async function overlappingFieldPoint(agent) {
  return agent.evaluate((element) => {
    const body = element.querySelector(".agent-body");
    if (!(body instanceof SVGGraphicsElement)) {
      throw new Error("Agent body is unavailable.");
    }
    const bounds = body.getBoundingClientRect();
    const samples = [
      [0.5, 0.5],
      [0.35, 0.5],
      [0.65, 0.5],
      [0.5, 0.35],
      [0.5, 0.65],
    ];
    for (const [xRatio, yRatio] of samples) {
      const x = bounds.left + bounds.width * xRatio;
      const y = bounds.top + bounds.height * yRatio;
      const hits = document.elementsFromPoint(x, y);
      const ownsAgent = hits.some((hit) => hit.closest(".agent") === element);
      const field = hits.find((hit) => hit.matches(".aura-field, .range-ring-hit"));
      if (ownsAgent && field) {
        return {
          fieldClass:
            field instanceof SVGElement ? field.className.baseVal : field.className,
          x,
          y,
        };
      }
    }
    return null;
  });
}

/**
 * Wait for font and tooltip placement work, then prove the singleton remains
 * wholly inside the supported viewport gutter.
 *
 * @param {import("@playwright/test").Page} page
 */
async function expectStableViewportTooltip(page) {
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve)),
    );
  });
  await expect(page.locator('[role="tooltip"]:visible')).toHaveCount(1);
  const tooltip = page.locator("#visual-tooltip");
  await expect(tooltip).toBeVisible();
  const bounds = await tooltip.boundingBox();
  expect(bounds).not.toBeNull();
  expect(bounds?.x).toBeGreaterThanOrEqual(8);
  expect(bounds?.y).toBeGreaterThanOrEqual(8);
  expect((bounds?.x ?? 0) + (bounds?.width ?? 0)).toBeLessThanOrEqual(952);
  expect((bounds?.y ?? 0) + (bounds?.height ?? 0)).toBeLessThanOrEqual(592);
  return bounds;
}

/**
 * Find a pointer coordinate where a rendered semantic owner participates in
 * the browser's actual hit stack. Exclusive probes reject points where another
 * registered semantic owner could win arbitration.
 *
 * @param {import("@playwright/test").Locator} owner
 * @param {{exclusive?: boolean}} [options]
 */
async function registeredOwnerPoint(owner, options = {}) {
  const exclusive = options.exclusive === true;
  return owner.evaluate((element, requireExclusive) => {
    const points = [];
    const geometry =
      element instanceof SVGGeometryElement
        ? element
        : element.querySelector(
            ".combat-route__hit, .pending-route-hit, .range-ring-hit",
          );
    if (geometry instanceof SVGGeometryElement) {
      const matrix = geometry.getScreenCTM();
      const length = geometry.getTotalLength();
      if (matrix && Number.isFinite(length) && length > 0) {
        for (const ratio of [0.08, 0.18, 0.32, 0.5, 0.68, 0.82, 0.92]) {
          const local = geometry.getPointAtLength(length * ratio);
          const screen = new DOMPoint(local.x, local.y).matrixTransform(matrix);
          points.push({ x: screen.x, y: screen.y });
        }
      }
    }
    const bounds = element.getBoundingClientRect();
    for (const [xRatio, yRatio] of [
      [0.5, 0.5],
      [0.2, 0.2],
      [0.8, 0.2],
      [0.2, 0.8],
      [0.8, 0.8],
      [0.35, 0.65],
      [0.65, 0.35],
    ]) {
      points.push({
        x: bounds.left + bounds.width * xRatio,
        y: bounds.top + bounds.height * yRatio,
      });
    }
    for (const point of points) {
      const owners = [
        ...new Set(
          document
            .elementsFromPoint(point.x, point.y)
            .map((hit) => hit.closest("[data-tooltip-owner]"))
            .filter((candidate) => candidate instanceof Element),
        ),
      ];
      if (owners.includes(element) && (!requireExclusive || owners.length === 1)) {
        return {
          ownerCount: owners.length,
          x: point.x,
          y: point.y,
        };
      }
    }
    return null;
  }, exclusive);
}

/**
 * Move through the singleton delegated controller and prove the expected
 * semantic descriptor—not a native title or direct event listener—wins.
 *
 * @param {import("@playwright/test").Page} page
 * @param {import("@playwright/test").Locator} owner
 * @param {string | RegExp} expectedKind
 * @param {{exclusive?: boolean}} [options]
 */
async function expectDelegatedOwner(page, owner, expectedKind, options = {}) {
  await expect(owner).toHaveAttribute("data-tooltip-owner", "");
  const point = await registeredOwnerPoint(owner, options);
  expect(
    point,
    `No inspectable point found for ${String(expectedKind)}`,
  ).not.toBeNull();
  await page.mouse.move(point?.x ?? 0, point?.y ?? 0);
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    expectedKind,
  );
  return point;
}

test("one tooltip arbitrates dense battlefield and keyboard explanations", async ({
  page,
}) => {
  const frame = structuredClone(crowdedFrame);
  frame.projection.scene.agents[2].ultimate_cooldown_remaining = 30;
  await installFrame(page, frame);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await waitForStablePresentation(page);

  const tooltip = page.locator("#visual-tooltip");
  const title = page.locator("#visual-tooltip-title");
  const details = page.locator("#visual-tooltip-details");
  const agent = page.locator('#battlefield .agent[data-slot="0"]');
  const overlapPoint = await overlappingFieldPoint(agent);
  expect(overlapPoint).not.toBeNull();
  expect(overlapPoint?.fieldClass).toMatch(/^(?:aura-field|range-ring-hit)$/);
  await page.mouse.move(overlapPoint?.x ?? 0, overlapPoint?.y ?? 0);
  await expect(tooltip).toHaveAttribute("data-tooltip-kind", "agent");
  await expect(title).toContainText("id_0");
  await expectStableViewportTooltip(page);
  await expect(page).toHaveScreenshot("agent-tooltip-overlapping-fields-960x600.png", {
    animations: "disabled",
  });

  const compactStatusSummaries = page.locator("#battlefield .status-overflow");
  expect(await compactStatusSummaries.count()).toBeGreaterThan(1);
  const firstStatusOwner = await compactStatusSummaries
    .nth(0)
    .getAttribute("data-slot");
  const secondStatusOwner = await compactStatusSummaries
    .nth(1)
    .getAttribute("data-slot");
  expect(firstStatusOwner).not.toBe(secondStatusOwner);
  await compactStatusSummaries.nth(0).hover();
  await expect(tooltip).toHaveAttribute("data-tooltip-kind", "status-overflow");
  await compactStatusSummaries.nth(1).hover();
  await expect(tooltip).toHaveAttribute("data-tooltip-kind", "status-overflow");

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

  const statusOverflows = page.locator("#battlefield .status-overflow");
  expect(await statusOverflows.count()).toBeGreaterThan(0);
  const edgeOverflowIndex = await statusOverflows.evaluateAll((nodes) => {
    let bestIndex = 0;
    let bestRight = Number.NEGATIVE_INFINITY;
    let bestBottom = Number.NEGATIVE_INFINITY;
    for (const [index, node] of nodes.entries()) {
      const bounds = node.getBoundingClientRect();
      if (
        bounds.right > bestRight ||
        (bounds.right === bestRight && bounds.bottom > bestBottom)
      ) {
        bestIndex = index;
        bestRight = bounds.right;
        bestBottom = bounds.bottom;
      }
    }
    return bestIndex;
  });
  const edgeOverflow = statusOverflows.nth(edgeOverflowIndex);
  const neutralColors = await edgeOverflow.evaluate((element) => {
    const probe = document.createElement("span");
    probe.style.color = "var(--text-muted)";
    document.body.append(probe);
    const overflowColor = getComputedStyle(element).color;
    const label = element.querySelector(".status-overflow__label");
    const labelFill = label ? getComputedStyle(label).fill : "";
    const expected = getComputedStyle(probe).color;
    probe.remove();
    return { expected, labelFill, overflowColor };
  });
  expect(neutralColors.overflowColor).toBe(neutralColors.expected);
  expect(neutralColors.labelFill).toBe(neutralColors.expected);
  const edgeHiddenCount = Number(await edgeOverflow.getAttribute("data-hidden-count"));
  await edgeOverflow.hover();
  await expect(tooltip).toHaveAttribute("data-tooltip-kind", "status-overflow");
  const overflowRecipientSlot = await edgeOverflow.getAttribute("data-slot");
  await expect(title).toHaveText(
    `id_${overflowRecipientSlot} · ${edgeHiddenCount} hidden statuses`,
  );
  await expect(tooltip).toHaveAttribute(
    "data-tooltip-placement",
    /^(?:right|left)-(?:below|above)$/,
  );
  const edgeListedFacts = (await details.textContent())
    ?.split("\n")
    .filter((line) => line.trim() !== "");
  expect(edgeListedFacts).toHaveLength(edgeHiddenCount);
  for (const fact of edgeListedFacts ?? []) {
    expect(fact).toContain("source ");
    expect(fact).toMatch(/(?:duration unavailable|\d+ ticks?)$/);
  }
  const edgeTooltipBounds = await expectStableViewportTooltip(page);
  const edgeOwnerBounds = await edgeOverflow.boundingBox();
  expect(edgeOwnerBounds).not.toBeNull();
  const tooltipAvoidsOwner =
    (edgeTooltipBounds?.x ?? 0) + (edgeTooltipBounds?.width ?? 0) <=
      (edgeOwnerBounds?.x ?? 0) ||
    (edgeOwnerBounds?.x ?? 0) + (edgeOwnerBounds?.width ?? 0) <=
      (edgeTooltipBounds?.x ?? 0) ||
    (edgeTooltipBounds?.y ?? 0) + (edgeTooltipBounds?.height ?? 0) <=
      (edgeOwnerBounds?.y ?? 0) ||
    (edgeOwnerBounds?.y ?? 0) + (edgeOwnerBounds?.height ?? 0) <=
      (edgeTooltipBounds?.y ?? 0);
  expect(tooltipAvoidsOwner).toBe(true);
  await expect(page).toHaveScreenshot(
    "neutral-status-overflow-tooltip-edge-clamped-960x600.png",
    { animations: "disabled" },
  );

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

test("status overflow explanation names its recipient and retains every item", async ({
  page,
}) => {
  await installFrame(page, structuredClone(crowdedFrame));
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await waitForStablePresentation(page);

  const overflow = page.locator("#battlefield .status-overflow").first();
  await expect(overflow).toBeVisible();
  const slot = await overflow.getAttribute("data-slot");
  const hiddenCount = Number(await overflow.getAttribute("data-hidden-count"));
  await overflow.hover();

  const tooltip = page.locator("#visual-tooltip");
  await expect(tooltip).toHaveAttribute("data-tooltip-kind", "status-overflow");
  await expect(page.locator("#visual-tooltip-title")).toHaveText(
    `id_${slot} · ${hiddenCount} hidden statuses`,
  );
  await expect(tooltip).toHaveAttribute(
    "data-tooltip-placement",
    /^(?:right|left)-(?:below|above)$/,
  );
  const details = (await page.locator("#visual-tooltip-details").textContent())
    ?.split("\n")
    .filter((line) => line.trim() !== "");
  expect(details).toHaveLength(hiddenCount);
  for (const detail of details ?? []) {
    expect(detail).toContain("source ");
    expect(detail).toMatch(/(?:duration unavailable|\d+ ticks?)$/);
  }
});

test("durable semantic cues all use the delegated tooltip controller", async ({
  page,
}) => {
  const frame = withoutIncomingResearcherEvents(crowdedFrame);
  frame.projection.scene.agents = frame.projection.scene.agents.map(
    /** @param {Record<string, any>} agent */ (agent) => ({
      ...agent,
      statuses: [],
      ultimate_cooldown_remaining: 0,
    }),
  );
  const pendingAction = {
    ...frame.hud.pending_action,
    label: "Basic Action Route",
    actor_global_slot: 0,
    target_action: 6,
    armed_lane: 0,
    arm_origin: "manual",
    target: { disclosure: "public", global_slot: 5 },
    pair_mask_value: true,
    summary: "STAY + BASIC → Agent ID 5",
  };
  frame.hud.pending_action = pendingAction;
  frame.hud.pending_actions = [pendingAction];
  await installFrame(page, frame);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await waitForStablePresentation(page);

  for (const [selector, expectedKind] of [
    ["#battlefield .modifier-cell", "modifier"],
    ["#battlefield .obstacle", "obstacle"],
  ]) {
    const owner = page.locator(`${selector}:visible`).first();
    await expect(owner).toHaveAttribute("data-tooltip-owner", "");
    await owner.hover();
    await expect(page.locator("#visual-tooltip")).toHaveAttribute(
      "data-tooltip-kind",
      expectedKind,
    );
  }
  await expectDelegatedOwner(
    page,
    page.locator("#battlefield .aura-field:visible").first(),
    "aura",
    { exclusive: true },
  );
  await expectDelegatedOwner(
    page,
    page.locator("#battlefield .pending-route-hit:visible").first(),
    "pending-route",
  );

  frame.projection.scene.next_decision_selected_legality = {
    armed_lane: 0,
    armed_pair_legal: true,
    controlled_global_slot: 0,
    lane_0_available: true,
    lane_1_available: false,
    target_action: 6,
    target_global_slot: 5,
  };
  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await waitForStablePresentation(page);

  const legality = page.locator("#battlefield .legality-pill:visible").first();
  await expect(legality).toHaveAttribute("data-tooltip-owner", "");
  await legality.hover();
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "legality",
  );
});

test("an accepted route yields to an agent body at the same pointer coordinate", async ({
  page,
}) => {
  const frame = structuredClone(vocabularyFrame);
  const activation = frame.projection.incoming_events.events.find(
    /** @param {Record<string, any>} event */ (event) =>
      event.event_type === "ability_activated" && event.source_global_slot === 0,
  );
  const targetTrajectory =
    frame.projection.incoming_events.agent_phase_trajectories.find(
      /** @param {Record<string, any>} trajectory */ (trajectory) =>
        trajectory.global_slot === 4,
    );
  expect(activation).toBeTruthy();
  expect(targetTrajectory).toBeTruthy();
  activation.recipient_global_slot = 4;
  activation.recipient_anchor = targetTrajectory.transition_start;
  const activationId = String(activation.event_id);

  await installFrame(page, frame);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await waitForStablePresentation(page);

  const routeOwner = page.locator(
    `#battlefield .combat-route-effect[data-event-id="${activationId}"]`,
  );
  const route = routeOwner.locator(".combat-route__hit");
  const emptyRoutePoint = await registeredOwnerPoint(routeOwner, {
    exclusive: true,
  });
  expect(emptyRoutePoint).not.toBeNull();
  await page.mouse.move(emptyRoutePoint?.x ?? 0, emptyRoutePoint?.y ?? 0);
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "accepted-route",
  );

  const overlapPoint = await route.evaluate((path) => {
    if (!(path instanceof SVGGeometryElement)) {
      throw new Error("Accepted route hit owner is not SVG geometry.");
    }
    const routeOwner = path.closest("[data-tooltip-owner]");
    if (!(routeOwner instanceof Element)) {
      throw new Error("Accepted route has no delegated tooltip owner.");
    }
    const matrix = path.getScreenCTM();
    const length = path.getTotalLength();
    if (!matrix || !Number.isFinite(length) || length <= 0) {
      return null;
    }
    for (let index = 0; index <= 160; index += 1) {
      const local = path.getPointAtLength((length * index) / 160);
      const screen = new DOMPoint(local.x, local.y).matrixTransform(matrix);
      const hits = document.elementsFromPoint(screen.x, screen.y);
      const hasRoute = hits.some(
        (hit) => hit.closest("[data-tooltip-owner]") === routeOwner,
      );
      const agent = hits
        .map((hit) => hit.closest('.agent[data-slot="2"]'))
        .find((candidate) => candidate instanceof Element);
      if (hasRoute && agent) {
        return { x: screen.x, y: screen.y };
      }
    }
    return null;
  });
  expect(overlapPoint).not.toBeNull();
  await page.mouse.move(overlapPoint?.x ?? 0, overlapPoint?.y ?? 0);
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "agent",
  );
  await expect(page.locator("#visual-tooltip-title")).toContainText("id_2");
});

test("range hits are inspectable and POV explanations retain redaction", async ({
  page,
}) => {
  await installFrame(page, vocabularyFrame);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");

  // Compact active choreography deliberately yields range inspection to
  // accepted combat truth. Settle the presentation before probing the range.
  await page.locator("#motion-skip-button").evaluate((button) => {
    if (button instanceof HTMLButtonElement && !button.disabled) {
      button.click();
    }
  });
  await waitForStablePresentation(page);

  const basicRangeHit = page.locator(
    '#battlefield .range-ring-hit[data-kind="basic"][data-slot="0"]',
  );
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

  const actionOutcome = page.locator(
    '#event-feed .event-item[data-event-type="own_action_outcome"]',
  );
  await actionOutcome.hover();
  await expect(page.locator("#visual-tooltip-details")).not.toContainText(/target/i);
  await expect(page.locator("#visual-tooltip-details")).not.toContainText(/id_5/);
  const tooltipText = await page.locator("#visual-tooltip").textContent();
  expect(tooltipText).not.toContain("target_anchor");
  expect(tooltipText).not.toMatch(/\b(?:37|30)\b/);
});
