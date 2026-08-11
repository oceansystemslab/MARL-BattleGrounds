import { expect, test } from "@playwright/test";

import {
  assertBoundedChoreography,
  CHOREOGRAPHY_ROOT,
  CHOREOGRAPHY_ROUTE_ROOT,
  choreographySnapshot,
  installWaapiAutopause,
  pauseAtLogicalTime,
} from "./support/choreography.js";
import { startDebugger, stopDebugger } from "./support/live-debugger.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";

test.describe.configure({ mode: "serial" });

test.beforeAll(async () => {
  const started = await startDebugger({
    scenario: "team_focus_crossfire",
    extraArgs: ["--include-stress"],
  });
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
 */
function trackCommandPosts(page) {
  let count = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/command")) {
      count += 1;
    }
  });
  return Object.freeze({
    count: () => count,
  });
}

/**
 * Install a deterministic page and reset or switch the shared live service to
 * the requested scenario.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} scenario
 */
async function loadScenario(page, scenario) {
  await installWaapiAutopause(page);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const revisionBefore = Number(await page.locator("#revision-value").textContent());
  if ((await page.locator("#scenario-select").inputValue()) === scenario) {
    await page.getByRole("button", { name: "Reset" }).click();
  } else {
    await page.locator("#scenario-select").selectOption(scenario);
  }

  await expect(page.locator("#scenario-select")).toHaveValue(scenario);
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect
    .poll(async () => Number(await page.locator("#revision-value").textContent()), {
      timeout: 120_000,
    })
    .toBeGreaterThan(revisionBefore);
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(0);
}

/**
 * Submit one scripted frame and seek its shared presentation clock.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} transitionId
 * @param {number} [logicalMs]
 * @returns {Promise<Record<string, any>>}
 */
async function advanceAnimatedFrame(page, transitionId, logicalMs = 520) {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" && response.url().endsWith("/api/command"),
  );
  await page.locator("#battlefield").focus();
  await page.keyboard.press("n");
  const response = await responsePromise;
  await expect(page.locator("#transition-value")).toHaveText(
    new RegExp(`:transition:${transitionId - 1}$`),
    { timeout: 120_000 },
  );
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
  await pauseAtLogicalTime(page, logicalMs);
  const payload = await response.json();
  if (
    typeof payload !== "object" ||
    payload === null ||
    typeof payload.frame !== "object" ||
    payload.frame === null
  ) {
    throw new Error("Command response did not contain an authoritative frame.");
  }
  return payload.frame;
}

/**
 * Prove each rendered Charge ownership pill stays inside the map and outside
 * durable bodies/docks, recipient NET text, and every other ownership pill.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} expectedCount
 */
async function assertChargeOwnershipLayout(page, expectedCount) {
  const labels = page.locator(
    `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="warrior_charge"] .combat-route__ownership[data-spatial-disposition="rendered"]`,
  );
  await expect(labels).toHaveCount(expectedCount);
  const violations = await page.evaluate(() => {
    const tolerance = 0.75;
    const battlefield = document.querySelector("#battlefield");
    const mapBoundary = battlefield?.querySelector(".map-boundary");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(mapBoundary instanceof SVGElement)
    ) {
      throw new Error("Battlefield map boundary is unavailable.");
    }
    const mapBounds = mapBoundary.getBoundingClientRect();
    /**
     * @param {DOMRect} left
     * @param {DOMRect} right
     */
    const overlap = (left, right) => ({
      x: Math.min(left.right, right.right) - Math.max(left.left, right.left),
      y: Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top),
    });
    /**
     * @param {Element} element
     */
    const isPainted = (element) => {
      const bounds = element.getBoundingClientRect();
      if (bounds.width <= 0 || bounds.height <= 0) {
        return false;
      }
      /** @type {Element | null} */
      let current = element;
      while (current) {
        const style = getComputedStyle(current);
        if (
          style.display === "none" ||
          style.visibility === "hidden" ||
          Number.parseFloat(style.opacity || "1") <= 0.001
        ) {
          return false;
        }
        if (current === battlefield) {
          break;
        }
        current = current.parentElement;
      }
      return true;
    };
    const protectedElements = [
      ...battlefield.querySelectorAll(
        [
          ".agent-team-ring",
          ".controlled-halo:not([hidden])",
          ".selected-reticle:not([hidden])",
          ".status-cell__box",
          ".modifier-cell__box",
          ".cooldown-cell__box",
          ".legality-pill__box",
          '.agent-id-tag[data-layout-suppressed="false"] .agent-id-tag-box',
          '.combat-effect--net-health[data-spatial-disposition="rendered"] .combat-net__recipient',
          '.combat-effect--net-health[data-spatial-disposition="rendered"] .combat-net__label',
        ].join(","),
      ),
    ].filter(isPainted);
    const protectedRects = protectedElements.map((element) => ({
      bounds: element.getBoundingClientRect(),
      selector:
        [...element.classList].map((name) => `.${name}`).join("") ||
        element.tagName.toLowerCase(),
    }));
    const records = [
      ...battlefield.querySelectorAll(
        '.combat-route-effect--activation[data-token-id="warrior_charge"] .combat-route__ownership',
      ),
    ].map((ownership) => {
      const box = ownership.querySelector(".combat-route__ownership-box");
      return {
        box: box?.getBoundingClientRect() ?? null,
        collisionFree: ownership.getAttribute("data-layout-collision-free"),
        disposition: ownership.getAttribute("data-spatial-disposition"),
        painted: isPainted(ownership),
        source: ownership.getAttribute("data-source-slot"),
        target: ownership.getAttribute("data-target-slot"),
      };
    });
    const result = [];
    for (const record of records) {
      const owner = `${record.source}->${record.target}`;
      if (
        record.collisionFree !== "true" ||
        record.disposition !== "rendered" ||
        !record.painted ||
        record.box === null ||
        record.box.width <= 0 ||
        record.box.height <= 0
      ) {
        result.push({ owner, reason: "ownership pill is not measurably rendered" });
        continue;
      }
      if (
        record.box.left < mapBounds.left - tolerance ||
        record.box.top < mapBounds.top - tolerance ||
        record.box.right > mapBounds.right + tolerance ||
        record.box.bottom > mapBounds.bottom + tolerance
      ) {
        result.push({ owner, reason: "ownership pill escapes the map" });
      }
      for (const protectedRect of protectedRects) {
        const depth = overlap(record.box, protectedRect.bounds);
        if (depth.x > tolerance && depth.y > tolerance) {
          result.push({
            owner,
            overlap: depth,
            protectedSelector: protectedRect.selector,
            reason: "ownership pill overlaps protected geometry",
          });
        }
      }
    }
    for (let index = 0; index < records.length; index += 1) {
      for (let other = index + 1; other < records.length; other += 1) {
        const leftBox = records[index].box;
        const rightBox = records[other].box;
        if (leftBox === null || rightBox === null) {
          continue;
        }
        const depth = overlap(leftBox, rightBox);
        if (depth.x > tolerance && depth.y > tolerance) {
          result.push({
            owner: `${records[index].source}->${records[index].target}`,
            overlap: depth,
            protectedSelector: `${records[other].source}->${records[other].target}`,
            reason: "Charge ownership pills overlap",
          });
        }
      }
    }
    return result;
  });
  expect(violations).toEqual([]);
}

/**
 * Release the current presentation gate when one exists.
 *
 * @param {import("@playwright/test").Page} page
 */
async function skipIfAvailable(page) {
  const skip = page.locator("#motion-skip-button");
  if (!(await skip.isDisabled())) {
    await skip.click();
  }
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} slot
 * @param {string} tokenId
 */
function durableRosterStatus(page, slot, tokenId) {
  return page.locator(
    `.roster-row[data-slot="${slot}"] .roster-fact-token--status[data-token-id="${tokenId}"]`,
  );
}

/**
 * @param {import("@playwright/test").Page} page
 * @returns {Promise<Map<number, {x: number, y: number, radius: number}>>}
 */
async function battlefieldCenters(page) {
  return new Map(
    await page.locator("#battlefield .agent").evaluateAll((agents) =>
      agents.map((agent) => {
        const body = agent.querySelector(".agent-body");
        if (!(body instanceof SVGCircleElement)) {
          throw new Error("Agent body is not measurable.");
        }
        return [
          Number(agent.getAttribute("data-slot")),
          {
            x: body.cx.baseVal.value,
            y: body.cy.baseVal.value,
            radius: body.r.baseVal.value,
          },
        ];
      }),
    ),
  );
}

/**
 * @param {Map<number, {x: number, y: number, radius: number}>} centers
 * @param {number} slot
 */
function centerAt(centers, slot) {
  const center = centers.get(slot);
  if (!center) {
    throw new Error(`Battlefield center for id_${slot} is unavailable.`);
  }
  return center;
}

/**
 * @param {string | null} transform
 */
function translatedPoint(transform) {
  const match = transform?.match(
    /^translate\((-?(?:\d+\.?\d*|\.\d+)) (-?(?:\d+\.?\d*|\.\d+))\)(?: scale\(-?(?:\d+\.?\d*|\.\d+)\))?$/,
  );
  if (!match) {
    throw new Error(`Unexpected impact transform: ${transform}`);
  }
  return { x: Number(match[1]), y: Number(match[2]) };
}

/**
 * @param {{x: number, y: number}} first
 * @param {{x: number, y: number}} second
 */
function pointDistance(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

/**
 * Assert that every activation uses the explicit transition-start anchors
 * shipped in the canonical V2 phase trajectory. Movement is a later phase and
 * must never rewrite combat causality onto the durable successor bodies.
 *
 * @param {Record<string, any>} frame
 */
function assertCanonicalActivationStartAnchors(frame) {
  const batch = frame.projection?.incoming_events;
  expect(batch).toBeTruthy();
  const startAnchors = new Map(
    batch.agent_phase_trajectories.map(
      /** @param {Record<string, any>} trajectory */ (trajectory) => [
        Number(trajectory.global_slot),
        trajectory.transition_start,
      ],
    ),
  );
  for (const event of batch.events.filter(
    /** @param {Record<string, any>} event */ (event) =>
      event.event_type === "ability_activated",
  )) {
    expect(event.source_anchor).toEqual(
      startAnchors.get(Number(event.source_global_slot)),
    );
    if (Number.isInteger(event.recipient_global_slot)) {
      expect(event.recipient_anchor).toEqual(
        startAnchors.get(Number(event.recipient_global_slot)),
      );
    } else {
      expect(event.recipient_anchor).toBeNull();
    }
  }
}

/**
 * Prove the live feed and each choreography subset retain the authoritative
 * canonical event order. No browser-side sorting is permitted here: a sorted
 * comparison would hide a presentation reorder of simultaneous CP2 facts.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} frame
 */
async function assertCanonicalEventOrder(page, frame) {
  const events = frame.projection?.incoming_events?.events;
  if (!Array.isArray(events)) {
    throw new Error("Live frame is missing its canonical V2 event rows.");
  }
  const canonicalIds = events.map((event) => event.event_id);
  expect(
    await page
      .locator("#event-feed [data-event-id]")
      .evaluateAll((items) => items.map((item) => item.getAttribute("data-event-id"))),
  ).toEqual(canonicalIds);

  for (const [eventType, selector] of [
    ["ability_activated", `${CHOREOGRAPHY_ROOT} .combat-effect--activation`],
    ["recipient_health_resolution", `${CHOREOGRAPHY_ROOT} .combat-effect--net-health`],
  ]) {
    const expectedIds = events
      .filter((event) => event.event_type === eventType)
      .map((event) => event.event_id);
    expect(
      await page
        .locator(selector)
        .evaluateAll((items) =>
          items.map((item) => item.getAttribute("data-event-id")),
        ),
    ).toEqual(expectedIds);
  }
}

test("focus crossfire retriggers events and keeps presentation controls local", async ({
  page,
}) => {
  const commandPosts = trackCommandPosts(page);
  await loadScenario(page, "team_focus_crossfire");

  const beforeFirstSubmit = commandPosts.count();
  await page.locator("#battlefield").focus();
  await page.keyboard.press("n");
  await expect(page.locator("#transition-value")).toHaveText(/:transition:0$/, {
    timeout: 120_000,
  });
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
  expect(commandPosts.count()).toBe(beforeFirstSubmit + 1);

  const afterFirstSubmit = commandPosts.count();
  await page.locator("#battlefield").focus();
  await page.keyboard.press("p");
  await expect(page.locator("html")).toHaveAttribute("data-motion-paused", "true");
  expect(commandPosts.count()).toBe(afterFirstSubmit);

  await pauseAtLogicalTime(page, 520);
  await page.getByRole("button", { name: "0.5×", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-motion-rate", "0.5");
  expect(commandPosts.count()).toBe(afterFirstSubmit);

  await page.locator("#battlefield").focus();
  await page.keyboard.press("n");
  await expect(page.locator("#notice")).toContainText("still being explained");
  expect(commandPosts.count()).toBe(afterFirstSubmit);

  const root = page.locator(CHOREOGRAPHY_ROOT);
  await root.evaluate((element) => {
    element.setAttribute("data-retained-probe", "same-transition");
  });
  const revisionBeforeRangeToggle = Number(
    await page.locator("#revision-value").textContent(),
  );
  const beforeRangeToggle = await choreographySnapshot(page);
  await page.locator("#battlefield").focus();
  await page.keyboard.press("g");
  await expect(page.locator("#revision-value")).toHaveText(
    String(revisionBeforeRangeToggle + 1),
  );
  expect(commandPosts.count()).toBe(afterFirstSubmit + 1);
  const afterRangeToggle = await choreographySnapshot(page);
  await expect(root).toHaveAttribute("data-retained-probe", "same-transition");
  expect(afterRangeToggle.epochKey).toBe(beforeRangeToggle.epochKey);
  expect(afterRangeToggle.effectIds).toEqual(beforeRangeToggle.effectIds);
  expect(afterRangeToggle.animationIds).toEqual(beforeRangeToggle.animationIds);

  const afterRangeTogglePost = commandPosts.count();
  const firstEventIds = new Set(beforeRangeToggle.effectIds);
  await page.locator("#motion-skip-button").click();
  expect(commandPosts.count()).toBe(afterRangeTogglePost);

  await advanceAnimatedFrame(page, 2);
  const second = await choreographySnapshot(page);
  expect(second.effectIds.every((eventId) => !firstEventIds.has(eventId))).toBe(true);
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  const thirdFrame = await advanceAnimatedFrame(page, 3);
  await assertCanonicalEventOrder(page, thirdFrame);
  const activations = page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`);
  await expect(activations).toHaveCount(7);
  expect(
    await activations.evaluateAll((nodes) =>
      nodes
        .map((node) => ({
          source: Number(node.getAttribute("data-source-slot")),
          target: Number(node.getAttribute("data-target-slot")),
        }))
        .sort((left, right) => left.source - right.source),
    ),
  ).toEqual([
    { source: 0, target: 5 },
    { source: 1, target: 5 },
    { source: 2, target: 5 },
    { source: 3, target: 5 },
    { source: 6, target: 5 },
    { source: 7, target: 5 },
    { source: 8, target: 5 },
  ]);
  const net = page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--net-health`);
  await expect(net).toHaveCount(1);
  await expect(net).toHaveAttribute("data-recipient-slot", "5");
  await assertBoundedChoreography(page);
});

test("moving Basics retain transition-start combat anchors before successor movement", async ({
  page,
}) => {
  await loadScenario(page, "moving_basic_crossfire");
  const initial = await battlefieldCenters(page);

  const firstFrame = await advanceAnimatedFrame(page, 1);
  assertCanonicalActivationStartAnchors(firstFrame);
  const firstSuccessor = await battlefieldCenters(page);
  const firstActivations = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--activation`,
  );
  await expect(firstActivations).toHaveCount(10);
  const firstRecords = await firstActivations.evaluateAll((nodes) =>
    nodes.map((node) => ({
      eventId: node.getAttribute("data-event-id"),
      source: Number(node.getAttribute("data-source-slot")),
      target: Number(node.getAttribute("data-target-slot")),
      impactTransform: node.querySelector(".combat-impact")?.getAttribute("transform"),
    })),
  );
  expect(new Set(firstRecords.map(({ source }) => source))).toEqual(
    new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
  );
  for (const slot of initial.keys()) {
    expect(
      pointDistance(centerAt(initial, slot), centerAt(firstSuccessor, slot)),
    ).toBeGreaterThan(1);
  }
  for (const record of firstRecords) {
    const targetBody = centerAt(initial, record.target);
    const impact = translatedPoint(record.impactTransform ?? null);
    const startDistance = pointDistance(impact, targetBody);
    expect(startDistance).toBeLessThanOrEqual(targetBody.radius + 3.01);
  }
  await skipIfAvailable(page);

  const secondFrame = await advanceAnimatedFrame(page, 2);
  assertCanonicalActivationStartAnchors(secondFrame);
  const secondSuccessor = await battlefieldCenters(page);
  const secondActivations = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--activation`,
  );
  await expect(secondActivations).toHaveCount(10);
  const secondRecords = await secondActivations.evaluateAll((nodes) =>
    nodes.map((node) => ({
      eventId: node.getAttribute("data-event-id"),
      target: Number(node.getAttribute("data-target-slot")),
      impactTransform: node.querySelector(".combat-impact")?.getAttribute("transform"),
    })),
  );
  const firstEventIds = new Set(firstRecords.map(({ eventId }) => eventId));
  expect(secondRecords.every(({ eventId }) => !firstEventIds.has(eventId))).toBe(true);
  for (const slot of firstSuccessor.keys()) {
    expect(
      pointDistance(centerAt(firstSuccessor, slot), centerAt(secondSuccessor, slot)),
    ).toBeGreaterThan(1);
  }
  for (const record of secondRecords) {
    const targetBody = centerAt(firstSuccessor, record.target);
    const impact = translatedPoint(record.impactTransform ?? null);
    const startDistance = pointDistance(impact, targetBody);
    expect(startDistance).toBeLessThanOrEqual(targetBody.radius + 3.01);
  }
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await page.setViewportSize({ width: 960, height: 600 });
  await loadScenario(page, "moving_focus_crossfire");
  const focusInitial = await battlefieldCenters(page);
  const focusFrame = await advanceAnimatedFrame(page, 1);
  assertCanonicalActivationStartAnchors(focusFrame);
  const focusSuccessor = await battlefieldCenters(page);
  for (const slot of [0, 1, 2, 3, 5, 6, 7, 8]) {
    expect(
      pointDistance(centerAt(focusInitial, slot), centerAt(focusSuccessor, slot)),
    ).toBeGreaterThan(1);
  }
  const focusActivations = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--activation`,
  );
  await expect(focusActivations).toHaveCount(7);
  await expect(
    focusActivations.locator(".combat-impact__semantic--damage"),
  ).toHaveCount(4);
  await expect(
    focusActivations.locator(".combat-impact__semantic--healing"),
  ).toHaveCount(3);
  const focusRecords = await focusActivations.evaluateAll((nodes) =>
    nodes.map((node) => ({
      target: Number(node.getAttribute("data-target-slot")),
      impactTransform: node.querySelector(".combat-impact")?.getAttribute("transform"),
    })),
  );
  expect(focusRecords.every(({ target }) => target === 5)).toBe(true);
  // The canonical anchor equality above proves the causal epoch. Dense
  // crossfire deliberately fans seven impact ports around that one start body,
  // so no single body-radius threshold is truthful for this layout case.
  expect(
    focusRecords.every(({ impactTransform }) =>
      Number.isFinite(translatedPoint(impactTransform ?? null).x),
    ),
  ).toBe(true);
  const focusRoutes = await page
    .locator(
      `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation .combat-route__path`,
    )
    .evaluateAll((paths) =>
      paths.map((path) => {
        const owner = path.closest(".combat-route-effect--activation");
        if (!(path instanceof SVGGeometryElement)) {
          throw new Error("Combat route is not measurable.");
        }
        const sourceSlot = Number(owner?.getAttribute("data-source-slot"));
        const targetSlot = Number(owner?.getAttribute("data-target-slot"));
        const length = path.getTotalLength();
        const endpoint = path.getPointAtLength(length);
        const beforeEndpoint = path.getPointAtLength(
          Math.max(0, length - Math.min(2, length * 0.1)),
        );
        const tangent = {
          x: endpoint.x - beforeEndpoint.x,
          y: endpoint.y - beforeEndpoint.y,
        };
        return {
          endTangentDegrees: (Math.atan2(tangent.y, tangent.x) * 180) / Math.PI,
          path: path.getAttribute("d"),
          source: sourceSlot,
          target: targetSlot,
        };
      }),
    );
  expect(focusRoutes).toHaveLength(7);
  expect(new Set(focusRoutes.map(({ path }) => path)).size).toBe(7);
  expect(new Set(focusRoutes.map(({ source }) => source))).toEqual(
    new Set([0, 1, 2, 3, 6, 7, 8]),
  );
  expect(focusRoutes.every(({ target }) => target === 5)).toBe(true);
  const minimumTangentSeparation = focusRoutes.reduce(
    (minimum, route, index) =>
      focusRoutes.slice(index + 1).reduce((pairMinimum, other) => {
        const rawDifference = Math.abs(
          route.endTangentDegrees - other.endTangentDegrees,
        );
        const angularDifference = Math.min(rawDifference, 360 - rawDifference);
        return Math.min(pairMinimum, angularDifference);
      }, minimum),
    Number.POSITIVE_INFINITY,
  );
  expect(minimumTangentSeparation).toBeGreaterThan(0);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--net-health`),
  ).toHaveAttribute("data-recipient-slot", "5");
  const netPlacement = await page
    .locator(`${CHOREOGRAPHY_ROOT} .combat-effect--net-health`)
    .evaluate((net) => {
      const labels = [
        net.querySelector(".combat-net__recipient"),
        net.querySelector(".combat-net__label"),
      ].filter((element) => element instanceof SVGGraphicsElement);
      const icons = [...document.querySelectorAll("#battlefield .agent-class-icon")];
      /**
       * @param {DOMRect} left
       * @param {DOMRect} right
       */
      const overlaps = (left, right) =>
        Math.min(left.right, right.right) > Math.max(left.left, right.left) &&
        Math.min(left.bottom, right.bottom) > Math.max(left.top, right.top);
      return labels.map((label) => {
        const bounds = label.getBoundingClientRect();
        return {
          insideViewport:
            bounds.left >= 0 &&
            bounds.top >= 0 &&
            bounds.right <= window.innerWidth &&
            bounds.bottom <= window.innerHeight,
          overlapsCrest: icons.some((icon) =>
            overlaps(bounds, icon.getBoundingClientRect()),
          ),
          visible: bounds.width > 0 && bounds.height > 0,
        };
      });
    });
  expect(netPlacement).toEqual([
    { insideViewport: true, overlapsCrest: false, visible: true },
    { insideViewport: true, overlapsCrest: false, visible: true },
  ]);
  await assertBoundedChoreography(page);
});

test("mirrored Ultimates keep activation identity separate from durable consequences", async ({
  page,
}) => {
  await loadScenario(page, "mirrored_ultimates");

  /** @type {Array<{
   *   tokenId: string,
   *   local: boolean,
   *   grammarSelector: string,
   *   grammarCount: number,
   *   routeStyle?: {strokeWidth: number, dashed: boolean},
   *   statuses: Array<[number, string]>,
   * }>} */
  const frames = [
    {
      tokenId: "mage_burst",
      local: true,
      grammarSelector: ".combat-burst__wave",
      grammarCount: 4,
      statuses: [
        [0, "mage_burst"],
        [5, "mage_burst"],
      ],
    },
    {
      tokenId: "warrior_charge",
      local: false,
      grammarSelector: ".combat-charge__impact",
      grammarCount: 2,
      routeStyle: { strokeWidth: 5, dashed: true },
      statuses: [
        [1, "stun_warrior_charge"],
        [1, "slow_warrior_charge"],
        [6, "stun_warrior_charge"],
        [6, "slow_warrior_charge"],
      ],
    },
    {
      tokenId: "hunter_trap",
      local: false,
      grammarSelector: ".combat-trap__lattice",
      grammarCount: 2,
      routeStyle: { strokeWidth: 3, dashed: true },
      statuses: [
        [2, "stun_hunter_trap"],
        [7, "stun_hunter_trap"],
      ],
    },
    {
      tokenId: "rogue_poison",
      local: false,
      grammarSelector: ".combat-poison__splash",
      grammarCount: 6,
      routeStyle: { strokeWidth: 2, dashed: true },
      statuses: [
        [3, "stun_rogue_poison"],
        [3, "slow_rogue_poison"],
        [3, "anti_heal_rogue_poison"],
        [8, "stun_rogue_poison"],
        [8, "slow_rogue_poison"],
        [8, "anti_heal_rogue_poison"],
      ],
    },
    {
      tokenId: "holy_word",
      local: false,
      grammarSelector: ".combat-holy__pulse",
      grammarCount: 4,
      routeStyle: { strokeWidth: 5, dashed: false },
      statuses: [],
    },
  ];

  for (const [index, frame] of frames.entries()) {
    const transitionId = index + 1;
    await advanceAnimatedFrame(page, transitionId, 120);
    const activations = page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="${frame.tokenId}"]`,
    );
    await expect(activations).toHaveCount(2);
    await expect(
      page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`),
    ).toHaveCount(2);
    await expect(activations.locator(".combat-local")).toHaveCount(frame.local ? 2 : 0);
    await expect(
      page.locator(
        `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="${frame.tokenId}"] .combat-route__path`,
      ),
    ).toHaveCount(frame.local ? 0 : 2);
    await expect(
      page.locator(
        `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="${frame.tokenId}"]`,
      ),
    ).toHaveCount(frame.local ? 0 : 2);
    if (frame.routeStyle) {
      const routeStyles = await page
        .locator(
          `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="${frame.tokenId}"] .combat-route__path`,
        )
        .evaluateAll((paths) =>
          paths.map((path) => {
            const style = getComputedStyle(path);
            return {
              strokeWidth: Number.parseFloat(style.strokeWidth),
              dashed: style.strokeDasharray !== "" && style.strokeDasharray !== "none",
            };
          }),
        );
      expect(routeStyles).toEqual([frame.routeStyle, frame.routeStyle]);
    }
    await expect(page.locator(frame.grammarSelector)).toHaveCount(frame.grammarCount);

    if (frame.tokenId === "warrior_charge") {
      // V2 owns Charge displacement as a later, independent phase rather than
      // folding it into the ability-activation route.
      await pauseAtLogicalTime(page, 400);
      await expect(
        page.locator(
          `${CHOREOGRAPHY_ROOT} [data-event-type="charge_phase_displacement"]`,
        ),
      ).toHaveCount(2);
    }
    for (const [slot, tokenId] of frame.statuses) {
      await expect(durableRosterStatus(page, slot, tokenId)).toHaveCount(1);
    }
    await assertBoundedChoreography(page);
    await skipIfAvailable(page);
    for (const [slot, tokenId] of frame.statuses) {
      await expect(durableRosterStatus(page, slot, tokenId)).toHaveCount(1);
    }
  }
});

test("converging Charge routes reproject and displacement settles without replay", async ({
  page,
}) => {
  await loadScenario(page, "charge_convergence");
  await advanceAnimatedFrame(page, 1, 120);

  const activations = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="warrior_charge"]`,
  );
  await expect(activations).toHaveCount(3);
  expect(
    await activations.evaluateAll((nodes) =>
      nodes
        .map((node) => [
          Number(node.getAttribute("data-source-slot")),
          Number(node.getAttribute("data-target-slot")),
        ])
        .sort((left, right) => left[0] - right[0]),
    ),
  ).toEqual([
    [0, 5],
    [1, 5],
    [5, 0],
  ]);
  const routePaths = await page
    .locator(
      `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="warrior_charge"] .combat-route__path`,
    )
    .evaluateAll((paths) => paths.map((path) => path.getAttribute("d")));
  expect(new Set(routePaths).size).toBe(3);
  const ownershipLabels = page.locator(
    `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="warrior_charge"] .combat-route__ownership`,
  );
  await expect(ownershipLabels).toHaveCount(3);
  expect(
    await ownershipLabels.evaluateAll((labels) =>
      labels
        .map((label) => ({
          source: Number(label.getAttribute("data-source-slot")),
          target: Number(label.getAttribute("data-target-slot")),
          text: label.textContent?.trim(),
        }))
        .sort((left, right) => left.source - right.source),
    ),
  ).toEqual([
    { source: 0, target: 5, text: "id_0 → id_5" },
    { source: 1, target: 5, text: "id_1 → id_5" },
    { source: 5, target: 0, text: "id_5 → id_0" },
  ]);
  await assertChargeOwnershipLayout(page, 3);
  const mitigation = page.locator(
    '#roster .roster-row[data-slot="0"] .roster-fact-token--modifier[data-token-id="warrior_mitigation"]',
  );
  await expect(mitigation).toContainText("×0.72");
  expect(await mitigation.getAttribute("data-multiplier")).toMatch(/\.\d{3,}/);
  const humanFacingValues = await page
    .locator(
      "#battlefield .modifier-cell__value, #roster .roster-fact-token--modifier, .comparison-agent .fact strong, #event-feed .event-item",
    )
    .allTextContents();
  expect(humanFacingValues.every((value) => !/\.\d{3,}/.test(value))).toBe(true);
  await assertBoundedChoreography(page);

  const root = page.locator(CHOREOGRAPHY_ROOT);
  await root.evaluate((element) => {
    element.setAttribute("data-retained-probe", "charge-resize");
  });
  const beforeResize = await choreographySnapshot(page);
  await page.setViewportSize({ width: 960, height: 600 });
  await expect
    .poll(async () => root.getAttribute("data-viewport-key"))
    .not.toBe(beforeResize.viewportKey);
  const afterResize = await choreographySnapshot(page);
  await expect(root).toHaveAttribute("data-retained-probe", "charge-resize");
  expect(afterResize.epochKey).toBe(beforeResize.epochKey);
  expect(afterResize.animationIds).toEqual(beforeResize.animationIds);
  expect(afterResize.effectIds).toEqual(beforeResize.effectIds);
  await assertChargeOwnershipLayout(page, 3);

  // The exact V2 displacement events begin only after the activation route
  // phase, and remain independently addressable by canonical event type.
  await pauseAtLogicalTime(page, 400);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} [data-event-type="charge_phase_displacement"]`),
  ).toHaveCount(3);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROUTE_ROOT} [data-event-type="charge_phase_displacement"] .combat-charge__path`,
    ),
  ).toHaveCount(3);

  await page.locator("#motion-skip-button").click();
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`),
  ).toHaveCount(0);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--charge-displacement`),
  ).toHaveCount(3);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-charge__path`),
  ).toHaveCount(3);
  expect((await choreographySnapshot(page)).animationIds).toEqual([]);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--charge-displacement`),
  ).toHaveCount(3);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect:not(.combat-effect--charge-displacement)`,
    ),
  ).toHaveCount(0);
  expect((await choreographySnapshot(page)).animationIds).toEqual([]);
  await assertBoundedChoreography(page);

  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(0);
});

test("Trap lifecycle preserves each canonical apply, break, and expiry event", async ({
  page,
}) => {
  await loadScenario(page, "trap_lifecycle");

  await advanceAnimatedFrame(page, 1);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="hunter_trap"]`,
    ),
  ).toHaveCount(4);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-event-type="status_applied"][data-token-id="stun_hunter_trap"][data-lifecycle="applied"]`,
    ),
  ).toHaveCount(4);
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await advanceAnimatedFrame(page, 2);
  const exactBreak = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-event-type="status_broken_by_damage"][data-token-id="stun_hunter_trap"][data-lifecycle="trap_broken"]`,
  );
  await expect(exactBreak).toHaveCount(1);
  await expect(exactBreak).toHaveAttribute("data-recipient-slot", "5");
  await expect(exactBreak.locator(".combat-lifecycle__shard")).toHaveCount(6);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} [data-lifecycle="decremented"]`),
  ).toHaveCount(0);
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await page.locator("#battlefield").focus();
  await page.keyboard.press("n");
  await expect(page.locator("#transition-value")).toHaveText(/:transition:2$/, {
    timeout: 120_000,
  });
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
  const expiredBasicSlow = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-event-type="status_aged_to_zero"][data-token-id="slow_hunter_basic"][data-lifecycle="expired"]`,
  );
  await expect(expiredBasicSlow).toHaveCount(1);
  await expect(expiredBasicSlow).toHaveAttribute("data-recipient-slot", "5");
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-token-id="stun_hunter_trap"]`,
    ),
  ).toHaveCount(0);
  await expect(
    page.locator('#event-feed [data-event-type="status_aged_to_zero"]'),
  ).toHaveCount(1);
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await advanceAnimatedFrame(page, 4);
  const broken = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-event-type="status_broken_by_damage"][data-token-id="stun_hunter_trap"]`,
  );
  const reapplied = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-event-type="status_applied"][data-token-id="stun_hunter_trap"]`,
  );
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="hunter_trap"] .combat-impact__semantic--damage`,
    ),
  ).toHaveCount(1);
  await expect(broken).toHaveCount(1);
  await expect(reapplied).toHaveCount(1);
  await expect(broken).toHaveAttribute("data-recipient-slot", "6");
  await expect(reapplied).toHaveAttribute("data-recipient-slot", "6");
  await expect(broken).toHaveAttribute("data-application-event-ids", "[]");
  await expect(reapplied).toHaveAttribute("data-application-event-ids", "[]");
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await advanceAnimatedFrame(page, 5);
  const expired = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-event-type="status_aged_to_zero"][data-token-id="stun_hunter_trap"][data-lifecycle="expired"]`,
  );
  // The V1 adapter's "cleared_unclassified" label was inferred by joining
  // unrelated lifecycle records. Canonical V2 truth contains two independent
  // status_aged_to_zero events, which this assertion preserves without guessing.
  await expect(expired).toHaveCount(2);
  expect(
    await expired.evaluateAll((nodes) =>
      nodes
        .map((node) => Number(node.getAttribute("data-recipient-slot")))
        .sort((left, right) => left - right),
    ),
  ).toEqual([7, 8]);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-lifecycle__shard`),
  ).toHaveCount(0);
  await assertBoundedChoreography(page);
});

test("maximum status stack keeps nine durable channels and independent source evidence", async ({
  page,
}) => {
  await loadScenario(page, "max_status_stack");
  await advanceAnimatedFrame(page, 1, 600);

  const expectedTokens = [
    "stun_warrior_charge",
    "stun_hunter_trap",
    "stun_rogue_poison",
    "slow_warrior_charge",
    "slow_hunter_basic",
    "slow_rogue_poison",
    "anti_heal_rogue_poison",
    "priest_freedom",
    "mage_burst",
  ];
  const durable = page.locator('#battlefield .status-cell[data-slot="0"]');
  await expect(durable).toHaveCount(9);
  await expect(
    page.locator('#battlefield .status-dock[data-slot="0"]'),
  ).toHaveAttribute("data-expanded", "true");
  expect(
    await durable.evaluateAll((nodes) =>
      nodes.map((node) => node.getAttribute("data-token-id")),
    ),
  ).toEqual(expectedTokens);
  const cooldowns = await page
    .locator("#battlefield .cooldown-dock")
    .evaluateAll((docks) =>
      docks
        .map((dock) => ({
          slot: Number(dock.getAttribute("data-slot")),
          ticks: Number(
            dock.querySelector(".cooldown-cell")?.getAttribute("data-ticks"),
          ),
        }))
        .sort((left, right) => left.slot - right.slot),
    );
  expect(cooldowns).toEqual([
    { slot: 0, ticks: 30 },
    { slot: 5, ticks: 30 },
    { slot: 6, ticks: 30 },
    { slot: 8, ticks: 30 },
  ]);
  const durableLayer = page.locator(
    '#battlefield [data-layer="durable-status-modifier"]',
  );
  await expect(durableLayer).toHaveAttribute("data-suppressed-status-slots", "");
  await expect(durableLayer).toHaveAttribute("data-suppressed-cooldown-slots", "");

  const activations = page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`);
  await expect(activations).toHaveCount(6);
  const activationIds = new Set(
    await activations.evaluateAll((nodes) =>
      nodes.map((node) => node.getAttribute("data-event-id")),
    ),
  );
  expect(activationIds.size).toBe(6);
  expect(
    [...activationIds].every(
      (eventId) =>
        typeof eventId === "string" && /:transition:0:event:\d{4}$/.test(eventId),
    ),
  ).toBe(true);
  const lifecycle = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-event-type="status_applied"][data-recipient-slot="0"][data-lifecycle="applied"]`,
  );
  await expect(lifecycle).toHaveCount(9);
  const records = await lifecycle.evaluateAll((nodes) =>
    nodes.map((node) => ({
      eventId: node.getAttribute("data-event-id"),
      applicationIds: JSON.parse(
        node.getAttribute("data-application-event-ids") ?? "[]",
      ),
      sourceSlot: Number(node.getAttribute("data-source-slot")),
      tokenId: node.getAttribute("data-token-id"),
      statusIcon: node
        .querySelector(".combat-lifecycle__status-icon")
        ?.getAttribute("data-icon"),
      changeIcon: node
        .querySelector(".combat-lifecycle__change-icon")
        ?.getAttribute("data-icon"),
      collisionFree: node
        .querySelector(".combat-lifecycle")
        ?.getAttribute("data-layout-collision-free"),
      transform: node.querySelector(".combat-lifecycle")?.getAttribute("transform"),
    })),
  );
  // This checks lifecycle membership only. Canonical source/DOM event order is
  // already proved directly by the live focus case above; durable dock display
  // order is asserted separately here without sorting.
  expect(new Set(records.map(({ tokenId }) => tokenId))).toEqual(
    new Set(expectedTokens),
  );
  expect(new Set(records.map(({ transform }) => transform)).size).toBe(9);
  expect(records.every(({ statusIcon }) => statusIcon?.startsWith("status-"))).toBe(
    true,
  );
  expect(records.every(({ changeIcon }) => changeIcon === "lifecycle-applied")).toBe(
    true,
  );
  expect(records.every(({ collisionFree }) => collisionFree === "true")).toBe(true);
  expect(new Set(records.map(({ eventId }) => eventId)).size).toBe(9);
  expect(
    records.every(({ eventId }) => /:transition:0:event:\d{4}$/.test(eventId ?? "")),
  ).toBe(true);
  // V2 status applications carry their own direct source evidence. They are
  // never joined back to independent ability events by a guessed identifier.
  expect(records.every(({ applicationIds }) => applicationIds.length === 0)).toBe(true);
  /** @type {Map<string, number>} */
  const sourceCounts = new Map();
  for (const { sourceSlot } of records) {
    const sourceKey = String(sourceSlot);
    sourceCounts.set(sourceKey, (sourceCounts.get(sourceKey) ?? 0) + 1);
  }
  expect([...sourceCounts.values()].sort((left, right) => left - right)).toEqual([
    1, 1, 1, 1, 2, 3,
  ]);
  await assertBoundedChoreography(page);

  await skipIfAvailable(page);
  await expect(durable).toHaveCount(9);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle`),
  ).toHaveCount(0);
});
