import { expect, test } from "@playwright/test";

import { statusTokenIdFromCatalogId } from "../src/vocabulary.js";
import {
  assertBoundedChoreography,
  assertTransientSlotsAuthorized,
  CHOREOGRAPHY_ROOT,
  CHOREOGRAPHY_ROUTE_ROOT,
  choreographySnapshot,
  pauseInsideEventWindow,
} from "./support/choreography.js";
import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  loadRendererFixture,
  syntheticDebuggerPresentationFrame,
  syntheticDebuggerWireFrame,
} from "./support/renderer-fixture.js";
import {
  ABILITY_EVENT_TYPE,
  advanceScriptTo,
  assertCurrentEventIds,
  assertDurableDockFlags,
  assertFrameIdentity,
  assertHudStoryLabels,
  assertStablePresentationFrame,
  assertTransientNumberLayout,
  CHARGE_EVENT_TYPE,
  captureBaseline,
  DENSE_BASELINE_MAX_DIFF_PIXEL_RATIO,
  DESKTOP_VIEWPORT,
  expectActivationPairs,
  expectRosterSlots,
  expectRosterStatuses,
  HEALTH_RESOLUTION_EVENT_TYPE,
  installSyntheticVisualCase,
  loadLiveVisualCase,
  MINIMUM_VIEWPORT,
  POV_HEALTH_EVENT_TYPE,
  STATUS_EVENT_TYPE,
  waitForStablePresentation,
} from "./support/visual-regression.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";
/** @type {Record<string, any>} */
let crowdedFrame = {};
/** @type {Record<string, any>} */
let crowdedWireFrame = {};
/** @type {Record<string, any>} */
let durableControlsWireFrame = {};
/** @type {Record<string, any>} */
let povFrame = {};
/** @type {Record<string, any>} */
let povWireFrame = {};
/** @type {Record<string, any>} */
let vocabularyWireFrame = {};

/**
 * Read the authenticated wire frame that authored the current browser view.
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

/** @param {Record<string, any>} frame */
function incomingEvents(frame) {
  const events = frame.projection?.incoming_events?.events;
  expect(Array.isArray(events)).toBe(true);
  return /** @type {Record<string, any>[]} */ (events);
}

/** @param {Record<string, any>} frame */
function eventKindCountsFromFrame(frame) {
  const kinds = incomingEvents(frame).map((event) => String(event.event_type));
  return Object.fromEntries(
    [...new Set(kinds)]
      .sort()
      .map((kind) => [kind, kinds.filter((candidate) => candidate === kind).length]),
  );
}

/**
 * Derive exact durable status values from the served researcher Scene V2.
 * Stable token identity remains a renderer-vocabulary assertion; numeric
 * duration truth stays owned by the simulator frame.
 *
 * @param {Record<string, any>} frame
 * @param {number} slot
 */
function sceneStatuses(frame, slot) {
  const agent = frame.projection?.scene?.agents?.find(
    /** @param {Record<string, any>} row */ (row) => row.global_slot === slot,
  );
  expect(agent).toBeTruthy();
  return /** @type {Record<string, any>[]} */ (agent.statuses).map((status) => ({
    duration: Number(status.remaining_duration),
    tokenId: statusTokenIdFromCatalogId(status.status_id),
  }));
}

/** @param {Record<string, any>} frame */
function sceneCooldowns(frame) {
  return /** @type {Record<string, any>[]} */ (frame.projection.scene.agents)
    .filter((agent) => Number(agent.ultimate_cooldown_remaining) > 0)
    .map((agent) => ({
      slot: Number(agent.global_slot),
      ticks: Number(agent.ultimate_cooldown_remaining),
    }))
    .sort((left, right) => left.slot - right.slot);
}

/**
 * Read one volatile duration from the catalog-normalized Scene V2 root.
 *
 * @param {Record<string, any>} frame
 * @param {string} statusId
 */
function sceneStatusDuration(frame, statusId) {
  const mechanics = /** @type {Record<string, any>[]} */ (
    frame.projection.scene.class_mechanics
  ).flatMap((row) => row.status_mechanics);
  const mechanic = mechanics.find((row) => row.status_id === statusId);
  expect(mechanic).toBeTruthy();
  return Number(mechanic.duration_steps);
}

/** @param {Record<string, any>} frame */
function statusEventsFromFrame(frame) {
  const lifecycleByType = /** @type {Readonly<Record<string, string>>} */ (
    Object.freeze({
      status_aged_to_zero: "expired",
      status_applied: "applied",
      status_broken_by_damage: "trap_broken",
      status_cleared_by_new_death: "cleared_by_death",
      status_refreshed_or_extended: "refreshed",
    })
  );
  return incomingEvents(frame)
    .filter((event) => Object.hasOwn(lifecycleByType, event.event_type))
    .map((event) => ({
      eventType: String(event.event_type),
      lifecycle: lifecycleByType[event.event_type],
      recipient: Number(event.recipient_global_slot),
      source:
        event.event_type === "status_applied" ? Number(event.source_global_slot) : null,
      tokenId: statusTokenIdFromCatalogId(event.status_id),
    }));
}

/**
 * Preserve the authored Trap causal story while letting its expiry epoch move
 * with the Scene V2 catalog duration.
 *
 * @param {Record<string, any>} frame
 * @param {2 | 4 | 5} transition
 */
function expectedTrapLifecycle(frame, transition) {
  const duration = sceneStatusDuration(frame, "hunter_trap_stun");
  /**
   * @param {"status_aged_to_zero" | "status_applied" | "status_broken_by_damage"} eventType
   * @param {number} recipient
   * @param {number | null} source
   */
  const row = (eventType, recipient, source = null) => ({
    eventType,
    lifecycle:
      eventType === "status_applied"
        ? "applied"
        : eventType === "status_broken_by_damage"
          ? "trap_broken"
          : "expired",
    recipient,
    source,
    tokenId: "stun_hunter_trap",
  });

  if (transition === 2) {
    return duration === 1
      ? [5, 6, 7, 8].map((recipient) => row("status_aged_to_zero", recipient))
      : [row("status_broken_by_damage", 5)];
  }
  if (transition === 4) {
    if (duration === 3) {
      return [
        row("status_aged_to_zero", 6),
        row("status_applied", 6, 4),
        row("status_aged_to_zero", 7),
        row("status_aged_to_zero", 8),
      ];
    }
    return duration >= 4
      ? [row("status_broken_by_damage", 6), row("status_applied", 6, 4)]
      : [row("status_applied", 6, 4)];
  }
  if (duration === 1) {
    return [row("status_aged_to_zero", 6)];
  }
  if (duration === 4) {
    return [row("status_aged_to_zero", 7), row("status_aged_to_zero", 8)];
  }
  return duration >= 5 ? [row("status_broken_by_damage", 7)] : [];
}

/** @param {Record<string, any>} frame */
function healthResolutionCount(frame) {
  return incomingEvents(frame).filter(
    (event) => event.event_type === "recipient_health_resolution",
  ).length;
}

// Screenshot-only CSS hides run-specific identity values while retaining their
// stable labels and layout. The production CSP is exercised by every ordinary
// browser/server test; this suite bypasses it only so Playwright can inject the
// reviewed visual-snapshot stylesheet.
test.use({ bypassCSP: true });

test.beforeAll(async () => {
  const [
    started,
    crowdedFixture,
    durableControlsFixture,
    povFixture,
    vocabularyFixture,
  ] = await Promise.all([
    startDebugger({
      scenario: "team_focus_crossfire",
      extraArgs: ["--include-stress"],
    }),
    loadRendererFixture("crowded_teamfight"),
    loadRendererFixture("durable_controls"),
    loadRendererFixture("pov_redaction"),
    loadRendererFixture("visual_vocabulary"),
  ]);
  serverProcess = started.process;
  debuggerUrl = started.url;
  crowdedFrame = syntheticDebuggerPresentationFrame(crowdedFixture);
  crowdedWireFrame = syntheticDebuggerWireFrame(crowdedFixture);
  durableControlsWireFrame = syntheticDebuggerWireFrame(durableControlsFixture);
  povFrame = syntheticDebuggerPresentationFrame(povFixture);
  povWireFrame = syntheticDebuggerWireFrame(povFixture);
  vocabularyWireFrame = syntheticDebuggerWireFrame(vocabularyFixture);
});

test.afterAll(async () => {
  const child = serverProcess;
  serverProcess = null;
  await stopDebugger(child);
});

/**
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, number>} expected
 */
async function expectEventKindCounts(page, expected) {
  const kinds = await page
    .locator("#event-feed .event-item")
    .evaluateAll((items) =>
      items.map((item) => item.getAttribute("data-event-type") ?? "missing"),
    );
  const counts = Object.fromEntries(
    [...new Set(kinds)]
      .sort()
      .map((kind) => [kind, kinds.filter((candidate) => candidate === kind).length]),
  );
  expect(counts).toEqual(expected);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} expected
 */
async function expectActivationRouteCount(page, expected) {
  await expect(
    page.locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation`),
  ).toHaveCount(expected);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} expected
 */
async function expectNetCount(page, expected) {
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--net-health`),
  ).toHaveCount(expected);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number[]} expected
 */
async function expectNetRecipients(page, expected) {
  const rawRecipients = await page
    .locator(`${CHOREOGRAPHY_ROOT} .combat-effect--net-health`)
    .evaluateAll((effects) =>
      effects.map((effect) => effect.getAttribute("data-recipient-slot")),
    );
  expect(rawRecipients).not.toContain(null);
  for (const rawRecipient of rawRecipients) {
    expect(rawRecipient).toMatch(/^(0|[1-9]\d*)$/);
  }
  const recipients = rawRecipients.map(Number).sort((left, right) => left - right);
  expect(recipients).toEqual([...expected].sort((left, right) => left - right));
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} slot
 * @param {Array<{tokenId: string, duration: number}>} expected
 */
async function expectBattlefieldStatuses(page, slot, expected) {
  const observed = await page
    .locator(`#battlefield .status-cell[data-slot="${slot}"]`)
    .evaluateAll((cells) =>
      cells.map((cell) => ({
        duration: Number(
          cell.querySelector(".status-cell__value")?.textContent ?? Number.NaN,
        ),
        tokenId: cell.getAttribute("data-token-id"),
      })),
    );
  expect(observed).toEqual(expected);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {Array<{slot: number, ticks: number}>} expected
 */
async function expectBattlefieldCooldowns(page, expected) {
  const observed = await page
    .locator("#battlefield .cooldown-cell")
    .evaluateAll((cells) =>
      cells
        .map((cell) => ({
          slot: Number(cell.getAttribute("data-slot")),
          ticks: Number(cell.getAttribute("data-ticks")),
        }))
        .sort((left, right) => left.slot - right.slot),
    );
  expect(observed).toEqual(expected);
}

/**
 * Prove each authoritative CP2 status event remains independently present in
 * the structured feed. V2 deliberately has no inferred duration-decrement or
 * composite break-and-reapply record.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Array<{
 *   recipient: number,
 *   eventType: string,
 *   source: number | null,
 * }>} expected
 */
async function expectStatusFeed(page, expected) {
  const observed = await page
    .locator('#event-feed .event-item[data-event-type^="status_"]')
    .evaluateAll((items) =>
      items.map((item) => ({
        eventType: item.getAttribute("data-event-type"),
        recipient: Number(item.getAttribute("data-recipient-slot")),
        source:
          item.getAttribute("data-source-slot") === null
            ? null
            : Number(item.getAttribute("data-source-slot")),
      })),
    );
  expect(observed).toEqual(expected);
}

/**
 * Prove every spatial status cue retains the exact V2 source event type,
 * recipient, optional direct source, status vocabulary, and visual grammar.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Array<{
 *   tokenId: string,
 *   recipient: number,
 *   eventType: string,
 *   lifecycle: string,
 *   source: number | null,
 * }>} expected
 */
async function expectRenderedStatusEvents(page, expected) {
  const observed = await page
    .locator(`${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle`)
    .evaluateAll((effects) =>
      effects.map((effect) => ({
        eventType: effect.getAttribute("data-event-type"),
        lifecycle: effect.getAttribute("data-lifecycle"),
        recipient: Number(effect.getAttribute("data-recipient-slot")),
        source:
          effect.getAttribute("data-source-slot") === null
            ? null
            : Number(effect.getAttribute("data-source-slot")),
        tokenId: effect.getAttribute("data-token-id"),
      })),
    );
  expect(observed).toEqual(expected);
}

/**
 * Prove every authoritative status is either visible in its owning dock or
 * counted by that slot's explicit overflow/suppression policy.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Array<{global_slot: number, statuses: unknown[]}>} sceneAgents
 */
async function assertStatusDockCoverage(page, sceneAgents) {
  const docks = await page.locator("#battlefield .status-dock").evaluateAll((nodes) =>
    nodes.map((dock) => ({
      hidden: dock.getAttribute("data-hidden-count"),
      slot: dock.getAttribute("data-slot"),
      visible: dock.getAttribute("data-visible-count"),
    })),
  );
  for (const dock of docks) {
    expect(dock.hidden).toMatch(/^(0|[1-9]\d*)$/);
    expect(dock.slot).toMatch(/^(0|[1-9]\d*)$/);
    expect(dock.visible).toMatch(/^(0|[1-9]\d*)$/);
  }
  const parsedDocks = docks.map(({ hidden, slot, visible }) => ({
    hidden: Number(hidden),
    slot: Number(slot),
    visible: Number(visible),
  }));
  for (const dock of parsedDocks) {
    const authoritative = sceneAgents.find((agent) => agent.global_slot === dock.slot);
    expect(dock.visible + dock.hidden).toBe(authoritative?.statuses.length);
  }
  const rawSuppressedSlots = (
    (await page
      .locator('#battlefield [data-layer="durable-status-modifier"]')
      .getAttribute("data-suppressed-status-slots")) ?? ""
  )
    .split(",")
    .filter(Boolean);
  for (const rawSlot of rawSuppressedSlots) {
    expect(rawSlot).toMatch(/^(0|[1-9]\d*)$/);
  }
  const suppressedSlots = rawSuppressedSlots.map(Number);
  const expectedSlots = sceneAgents
    .filter((agent) => agent.statuses.length > 0)
    .map((agent) => agent.global_slot);
  expect(new Set([...parsedDocks.map(({ slot }) => slot), ...suppressedSlots])).toEqual(
    new Set(expectedSlots),
  );
}

/**
 * At the supported minimum viewport, active combat must retain every exact
 * analysis owner while painting only the higher-priority accepted story.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number[]} expectedStatusSlots
 */
async function assertCompactActiveCombatPriority(page, expectedStatusSlots) {
  const evidence = await page.evaluate(() => {
    const battlefield = document.querySelector("#battlefield");
    if (!(battlefield instanceof SVGSVGElement)) {
      throw new Error("Battlefield SVG is unavailable.");
    }
    const analysisOwners = [
      battlefield.querySelector('[data-layer="range-cues"]'),
      battlefield.querySelector('[data-layer="pending-route"]'),
      battlefield.querySelector('[data-layer="legality-cues"]'),
      ...battlefield.querySelectorAll(".status-dock"),
    ].filter((owner) => owner instanceof SVGElement);
    const paintedAccepted = [
      ...battlefield.querySelectorAll(
        [
          ".agent",
          ".controlled-halo:not([hidden])",
          ".selected-reticle:not([hidden])",
          ".combat-route-effect--activation",
          '.combat-effect--net-health[data-spatial-disposition="rendered"]',
        ].join(","),
      ),
    ].filter((owner) => {
      const bounds = owner.getBoundingClientRect();
      return (
        getComputedStyle(owner).display !== "none" &&
        getComputedStyle(owner).visibility !== "hidden" &&
        bounds.width > 0 &&
        bounds.height > 0
      );
    });
    return {
      active: battlefield.dataset.compactActiveCombat,
      paintedAcceptedCount: paintedAccepted.length,
      statusSummaries: [...battlefield.querySelectorAll(".status-dock")].map(
        (dock) => ({
          count: dock.getAttribute("data-hidden-count"),
          owner: dock.querySelector(".status-overflow__owner")?.textContent?.trim(),
          slot: Number(dock.getAttribute("data-slot")),
        }),
      ),
      suppressedFacts: battlefield.dataset.compactActiveSuppressedFacts,
      suppressedOwners: analysisOwners.map((owner) => ({
        display: getComputedStyle(owner).display,
        suppressed: owner.getAttribute("data-compact-active-suppressed"),
      })),
    };
  });
  expect(evidence.active).toBe("true");
  expect(evidence.suppressedFacts).toBe(
    "ranges,pending-route,selected-legality,status-summaries",
  );
  expect(
    evidence.suppressedOwners.every(
      ({ display, suppressed }) => display === "none" && suppressed === "true",
    ),
  ).toBe(true);
  expect(evidence.paintedAcceptedCount).toBeGreaterThanOrEqual(30);
  expect(
    evidence.statusSummaries
      .map(({ slot }) => slot)
      .sort((left, right) => left - right),
  ).toEqual([...expectedStatusSlots].sort((left, right) => left - right));
  expect(
    evidence.statusSummaries.every(
      ({ count, owner, slot }) => count === "9" && owner === `Agent ID ${slot}`,
    ),
  ).toBe(true);
}

/**
 * Recursively reject hidden slots, IDs, and dynamic coordinates from the
 * synthetic safe envelope. Python's protocol allowlist remains the primary
 * production proof; this keeps the browser fixture honest as it evolves.
 *
 * @param {Record<string, any>} frame
 */
function expectPovPayloadRedacted(frame) {
  const agents = /** @type {Array<Record<string, any>>} */ (frame.scene.agents);
  const authorizedSlots = new Set(agents.map((agent) => Number(agent.global_slot)));
  const sourceScene = /** @type {Record<string, any>} */ (
    frame.projection?.scene ?? {}
  );
  const pointOwners = [
    ...agents,
    ...(Array.isArray(frame.scene.observed_bodies) ? frame.scene.observed_bodies : []),
    ...(Array.isArray(frame.scene.visible_bodies) ? frame.scene.visible_bodies : []),
    ...(Array.isArray(frame.scene.spawn_pads) ? frame.scene.spawn_pads : []),
    ...(sourceScene.self_actor ? [sourceScene.self_actor] : []),
    ...(Array.isArray(sourceScene.visible_bodies) ? sourceScene.visible_bodies : []),
    ...(Array.isArray(sourceScene.spawn_pads) ? sourceScene.spawn_pads : []),
  ];
  const authorizedPoints = new Set(
    pointOwners
      .map((owner) => owner?.position)
      .filter(
        (point) =>
          Array.isArray(point) && point.length === 2 && point.every(Number.isFinite),
      )
      .map((point) => JSON.stringify(point)),
  );
  /** @type {string[]} */
  const violations = [];

  /**
   * @param {unknown} value
   * @param {string} path
   * @param {boolean} [authorizedOwnPositionEndpoint]
   */
  const visit = (value, path, authorizedOwnPositionEndpoint = false) => {
    const field = path.split(".").at(-1) ?? "";
    if (Array.isArray(value)) {
      if (
        /(?:anchor|position|start|end)$/.test(field) &&
        value.length === 2 &&
        value.every(Number.isFinite) &&
        !authorizedOwnPositionEndpoint &&
        !authorizedPoints.has(JSON.stringify(value))
      ) {
        violations.push(`${path} exposes an unauthorized dynamic point.`);
      }
      for (const [index, item] of value.entries()) {
        visit(item, `${path}.${index}`);
      }
      return;
    }
    if (typeof value === "string") {
      for (const match of value.matchAll(/\bid_(\d+)\b/g)) {
        if (!authorizedSlots.has(Number(match[1]))) {
          violations.push(`${path} names unauthorized ${match[0]}.`);
        }
      }
      return;
    }
    if (!value || typeof value !== "object") {
      return;
    }
    const record = /** @type {Record<string, any>} */ (value);
    const isAuthorizedOwnPositionRecord =
      record.cue_type === "own_position_changed" ||
      record.event_type === "own_position_changed";
    if (
      record.target_disclosure === "redacted" &&
      (record.target_global_slot !== null || record.target_anchor !== null)
    ) {
      violations.push(`${path} retains a redacted target identity or anchor.`);
    }
    if (
      record.event_type === "action_rejected" &&
      (Object.hasOwn(record, "target_global_slot") ||
        Object.hasOwn(record, "target_anchor"))
    ) {
      violations.push(`${path} adds non-contract target data to an action rejection.`);
    }
    for (const [key, item] of Object.entries(record)) {
      if (
        /global_slot$/.test(key) &&
        Number.isInteger(item) &&
        !authorizedSlots.has(Number(item))
      ) {
        violations.push(`${path}.${key} exposes unauthorized slot ${item}.`);
      }
      const isAuthorizedOwnPositionEndpoint =
        isAuthorizedOwnPositionRecord &&
        (key === "start_position" || key === "successor_position");
      visit(item, `${path}.${key}`, isAuthorizedOwnPositionEndpoint);
    }
  };

  expect(frame.scene.observer_visibility).toEqual([]);
  visit(frame, "frame");
  expect(violations).toEqual([]);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{
 *   scenario: string,
 *   transition: number,
 *   roster: number[],
 *   stableEventKinds: Record<string, number>,
 *   activations: Array<{tokenId: string, source: number, target: number | null}>,
 *   routeCount: number,
 *   netCount: number,
 * }} expected
 * @returns {Promise<Record<string, any>>}
 */
async function assertLiveCanonicalFrame(page, expected) {
  const frame = await currentWireFrame(page);
  const events = incomingEvents(frame);
  const eventKindCounts = eventKindCountsFromFrame(frame);
  await assertFrameIdentity(page, {
    scenario: expected.scenario,
    simulatorStep: expected.transition,
    transitionId: expected.transition,
    view: "researcher",
    preset: "analysis",
    badge: "PRIVILEGED RESEARCHER VIEW · CANONICAL EVALUATION",
  });
  await expectRosterSlots(page, expected.roster);
  await assertHudStoryLabels(page, {
    pending: "PLAYBACK / INSPECTION ONLY",
    accepted: "LATEST ACCEPTED RESULT",
  });
  await assertCurrentEventIds(page, events.length);
  await expectEventKindCounts(page, eventKindCounts);
  for (const [eventKind, count] of Object.entries(expected.stableEventKinds)) {
    expect(eventKindCounts[eventKind] ?? 0).toBe(count);
  }
  await expectActivationPairs(page, expected.activations);
  await expectActivationRouteCount(page, expected.routeCount);
  await expectNetCount(page, expected.netCount);
  await assertDurableDockFlags(page);
  await assertBoundedChoreography(page);
  await assertTransientSlotsAuthorized(page);
  return frame;
}

/**
 * @param {Array<[string, number]>} pairs
 */
function statuses(pairs) {
  return pairs.map(([tokenId, duration]) => ({ tokenId, duration }));
}

/**
 * Seek the exact V2 health-resolution phase and prove every numeric outcome
 * is painted and collision-free before a later causal phase is captured.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} expectedCount
 */
async function assertHealthResolutionPhase(page, expectedCount) {
  await pauseInsideEventWindow(page, HEALTH_RESOLUTION_EVENT_TYPE);
  await waitForStablePresentation(page);
  await assertTransientNumberLayout(page, expectedCount);
}

test("idle 5v5 Analysis battlefield remains readable", async ({ page }) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "arena_5v5",
  });
  await assertFrameIdentity(page, {
    scenario: "arena_5v5",
    simulatorStep: 0,
    transitionId: null,
    view: "researcher",
    preset: "analysis",
    badge: "PRIVILEGED RESEARCHER VIEW · CANONICAL EVALUATION",
  });
  await expectRosterSlots(page, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  await assertHudStoryLabels(page, {
    pending: "PENDING / WILL SUBMIT",
    accepted: "No transition yet.",
  });
  await assertCurrentEventIds(page, 0);
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(0);
  await expect(page.locator(CHOREOGRAPHY_ROUTE_ROOT)).toHaveCount(0);
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-submission-scope",
    "joint_turn",
  );
  await expect(page.locator(".pending-action-row")).toHaveCount(10);
  await expect(page.locator("#pending-count")).toHaveText("10 actors");

  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
  });
});

test("visual vocabulary presents every class and combat grammar", async ({ page }) => {
  const commandPosts = await installSyntheticVisualCase(
    page,
    debuggerUrl,
    vocabularyWireFrame,
    { viewport: DESKTOP_VIEWPORT },
  );
  await assertFrameIdentity(page, {
    scenario: "visual_vocabulary",
    simulatorStep: 1,
    transitionId: 1,
    view: "researcher",
    preset: "analysis",
    badge: /PRIVILEGED RESEARCHER VIEW.*SYNTHETIC VISUAL VOCABULARY/,
  });
  expect(await page.title()).toBe("MARL-BattleGrounds Visual Debugger and Analyzer");
  await expect(page.locator("h1")).toHaveText("Visual Debugger and Analyzer");
  await expect(page.locator("#scenario-description")).toContainText("SYNTHETIC:");
  await expectRosterSlots(page, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  await assertHudStoryLabels(page, {
    pending: "PLAYBACK / INSPECTION ONLY",
    accepted: "No transition yet.",
  });
  await assertCurrentEventIds(page, 12);
  await expectEventKindCounts(page, {
    ability_activated: 10,
    recipient_health_resolution: 2,
  });
  await expectActivationPairs(page, [
    { tokenId: "basic_damage", source: 0, target: 5 },
    { tokenId: "basic_damage", source: 1, target: 6 },
    { tokenId: "basic_damage", source: 2, target: 7 },
    { tokenId: "basic_damage", source: 3, target: 8 },
    { tokenId: "basic_heal", source: 4, target: 4 },
    { tokenId: "mage_burst", source: 0, target: null },
    { tokenId: "warrior_charge", source: 1, target: 6 },
    { tokenId: "hunter_trap", source: 2, target: 7 },
    { tokenId: "rogue_poison", source: 3, target: 8 },
    { tokenId: "holy_word", source: 4, target: 4 },
  ]);
  await expectActivationRouteCount(page, 9);
  await expectNetCount(page, 2);
  await expectNetRecipients(page, [5, 9]);
  await expectBattlefieldCooldowns(
    page,
    [1, 2, 3, 4, 5].map((ticks, slot) => ({ slot, ticks })),
  );
  await assertDurableDockFlags(page);
  await assertBoundedChoreography(page);
  await assertTransientSlotsAuthorized(page);

  const expectedBasicRangeColors = [
    ["mage", "rgb(34, 211, 238)"],
    ["warrior", "rgb(209, 139, 71)"],
    ["hunter", "rgb(132, 204, 22)"],
    ["rogue", "rgb(250, 204, 21)"],
    ["priest", "rgb(244, 114, 182)"],
  ];
  const basicRangeStyles = await page
    .locator('#battlefield .range-ring[data-kind="basic"]')
    .evaluateAll((ranges) =>
      ranges.map((range) => ({
        className: range.getAttribute("data-class"),
        dasharray: getComputedStyle(range).strokeDasharray,
        stroke: getComputedStyle(range).stroke,
      })),
    );
  expect(basicRangeStyles).toEqual(
    expectedBasicRangeColors.map(([className, stroke]) => ({
      className,
      dasharray: "8px, 5px",
      stroke,
    })),
  );

  const priestIcon = page.locator(
    '#battlefield .agent[data-slot="4"] .agent-class-icon[data-icon="class-priest"]',
  );
  await expect(priestIcon).toHaveCount(1);
  await expect(priestIcon.locator("rect")).toHaveCount(2);
  expect(await priestIcon.evaluate((icon) => getComputedStyle(icon).color)).toBe(
    "rgb(244, 114, 182)",
  );

  const impactGrammar = await page
    .locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="basic_damage"] .combat-impact__semantic--damage, ${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="basic_heal"] .combat-impact__semantic--healing`,
    )
    .evaluateAll((impacts) =>
      impacts.map((impact) => ({
        kind: impact.classList.contains("combat-impact__semantic--damage")
          ? "damage"
          : "healing",
        lines: [...impact.querySelectorAll("line")].map((line) => ({
          x1: line.getAttribute("x1"),
          x2: line.getAttribute("x2"),
          y1: line.getAttribute("y1"),
          y2: line.getAttribute("y2"),
        })),
        stroke: getComputedStyle(impact).stroke,
      })),
    );
  expect(impactGrammar).toEqual([
    ...Array.from({ length: 4 }, () => ({
      kind: "damage",
      lines: [{ x1: "-6", x2: "6", y1: "0", y2: "0" }],
      stroke: "rgb(251, 113, 133)",
    })),
    {
      kind: "healing",
      lines: [
        { x1: "-6", x2: "6", y1: "0", y2: "0" },
        { x1: "0", x2: "0", y1: "-6", y2: "6" },
      ],
      stroke: "rgb(52, 211, 153)",
    },
  ]);

  for (const [tokenId, flareClass] of [
    ["mage_burst", "combat-burst__flare"],
    ["warrior_charge", "combat-charge__flare"],
    ["hunter_trap", "combat-trap__flare"],
    ["rogue_poison", "combat-poison__flare"],
    ["holy_word", "combat-holy__flare"],
  ]) {
    const flare = page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="${tokenId}"] .combat-ultimate__flare.${flareClass}`,
    );
    await expect(flare).toHaveCount(1);
    const flareStyle = await flare.evaluate((element) => ({
      path: element.getAttribute("d"),
      stroke: getComputedStyle(element).stroke,
      strokeWidth: Number.parseFloat(getComputedStyle(element).strokeWidth),
    }));
    expect(flareStyle.path).toBeTruthy();
    expect(flareStyle.stroke).not.toBe("none");
    expect(flareStyle.strokeWidth).toBeGreaterThanOrEqual(2);
  }

  const auraStyles = await page
    .locator("#battlefield .aura-field")
    .evaluateAll((auras) =>
      auras.map((aura) => ({
        source: aura.getAttribute("data-source-slot"),
        stroke: getComputedStyle(aura).stroke,
      })),
    );
  expect(auraStyles).toEqual([
    { source: "0", stroke: "none" },
    { source: "1", stroke: "none" },
  ]);

  await assertHealthResolutionPhase(page, 2);
  await captureBaseline(
    page,
    "visual-vocabulary-synthetic-ability-phase-1440x900.png",
    {
      commandPosts,
      expectedTransientCount: 0,
      eventWindow: { eventType: ABILITY_EVENT_TYPE, part: "route", progress: 0.2 },
    },
    { maxDiffPixelRatio: DENSE_BASELINE_MAX_DIFF_PIXEL_RATIO },
  );
});

test("durable controls use one stun and one slow glyph with source accents", async ({
  page,
}) => {
  const commandPosts = await installSyntheticVisualCase(
    page,
    debuggerUrl,
    durableControlsWireFrame,
    { viewport: DESKTOP_VIEWPORT },
  );
  await assertFrameIdentity(page, {
    scenario: "durable_controls",
    simulatorStep: 0,
    transitionId: null,
    view: "researcher",
    preset: "analysis",
    badge: /PRIVILEGED RESEARCHER VIEW.*SYNTHETIC DURABLE CONTROLS/,
  });
  await expectRosterSlots(page, [0, 5]);
  const controls = await page
    .locator("#battlefield .status-cell")
    .evaluateAll((cells) =>
      cells.map((cell) => ({
        icon: cell.querySelector(".status-cell__icon")?.getAttribute("data-icon"),
        sourceClass: cell.getAttribute("data-source-class"),
        tokenId: cell.getAttribute("data-token-id"),
        value: cell.querySelector(".status-cell__value")?.textContent,
      })),
    );
  expect(controls).toEqual([
    {
      icon: "status-stun",
      sourceClass: "warrior",
      tokenId: "stun_warrior_charge",
      value: "3",
    },
    {
      icon: "status-stun",
      sourceClass: "hunter",
      tokenId: "stun_hunter_trap",
      value: "3",
    },
    {
      icon: "status-stun",
      sourceClass: "rogue",
      tokenId: "stun_rogue_poison",
      value: "3",
    },
    {
      icon: "status-slow",
      sourceClass: "warrior",
      tokenId: "slow_warrior_charge",
      value: "2",
    },
    {
      icon: "status-slow",
      sourceClass: "hunter",
      tokenId: "slow_hunter_basic",
      value: "2",
    },
    {
      icon: "status-slow",
      sourceClass: "rogue",
      tokenId: "slow_rogue_poison",
      value: "2",
    },
  ]);
  await assertDurableDockFlags(page);
  await captureBaseline(
    page,
    "durable-control-vocabulary-synthetic-settled-1440x900.png",
    {
      commandPosts,
      expectedTransientCount: 0,
    },
  );
});

test("focus fire and healing remain traceable at one shared impact phase", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "team_focus_crossfire",
  });
  await advanceScriptTo(page, 3);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "team_focus_crossfire",
    transition: 3,
    roster: [0, 1, 2, 3, 5, 6, 7, 8],
    stableEventKinds: {
      ability_activated: 7,
      combat_countdown_reset: 8,
      recipient_health_resolution: 1,
      source_damage_output: 4,
      source_healing_output: 3,
      status_applied: 4,
    },
    activations: [
      { tokenId: "basic_damage", source: 0, target: 5 },
      { tokenId: "basic_damage", source: 1, target: 5 },
      { tokenId: "basic_damage", source: 2, target: 5 },
      { tokenId: "basic_damage", source: 3, target: 5 },
      { tokenId: "basic_heal", source: 6, target: 5 },
      { tokenId: "basic_heal", source: 7, target: 5 },
      { tokenId: "basic_heal", source: 8, target: 5 },
    ],
    routeCount: 7,
    netCount: 1,
  });
  const focusStatuses = sceneStatuses(frame, 5);
  expect(focusStatuses.map(({ tokenId }) => tokenId)).toEqual([
    "slow_hunter_basic",
    "priest_freedom",
  ]);
  await expectRosterStatuses(page, 5, focusStatuses);

  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: healthResolutionCount(frame),
    eventWindow: { eventType: HEALTH_RESOLUTION_EVENT_TYPE },
  });
});

test("moving Basic crossfire preserves combat before successor movement", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "moving_basic_crossfire",
  });
  await advanceScriptTo(page, 1);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "moving_basic_crossfire",
    transition: 1,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 10,
      combat_countdown_reset: 8,
      ordinary_movement_phase_displacement: 10,
      recipient_health_resolution: 8,
      source_damage_output: 8,
      source_healing_output: 2,
      status_applied: 4,
    },
    activations: [
      { tokenId: "basic_damage", source: 0, target: 5 },
      { tokenId: "basic_damage", source: 1, target: 6 },
      { tokenId: "basic_damage", source: 2, target: 7 },
      { tokenId: "basic_damage", source: 3, target: 8 },
      { tokenId: "basic_heal", source: 4, target: 0 },
      { tokenId: "basic_damage", source: 5, target: 0 },
      { tokenId: "basic_damage", source: 6, target: 1 },
      { tokenId: "basic_damage", source: 7, target: 2 },
      { tokenId: "basic_damage", source: 8, target: 3 },
      { tokenId: "basic_heal", source: 9, target: 5 },
    ],
    routeCount: 10,
    netCount: 8,
  });
  await expectNetRecipients(page, [0, 1, 2, 3, 5, 6, 7, 8]);
  await expectRosterStatuses(page, 0, sceneStatuses(frame, 0));
  await expectRosterStatuses(page, 2, sceneStatuses(frame, 2));
  await expectRosterStatuses(page, 5, sceneStatuses(frame, 5));
  await expectRosterStatuses(page, 7, sceneStatuses(frame, 7));

  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: healthResolutionCount(frame),
    eventWindow: { eventType: HEALTH_RESOLUTION_EVENT_TYPE },
  });
});

test("moving focus fire and healing remain readable at minimum size", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "moving_focus_crossfire",
    viewport: MINIMUM_VIEWPORT,
  });
  await advanceScriptTo(page, 1);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "moving_focus_crossfire",
    transition: 1,
    roster: [0, 1, 2, 3, 5, 6, 7, 8],
    stableEventKinds: {
      ability_activated: 7,
      combat_countdown_reset: 5,
      ordinary_movement_phase_displacement: 8,
      recipient_health_resolution: 1,
      source_damage_output: 4,
      source_healing_output: 3,
      status_applied: 4,
    },
    activations: [
      { tokenId: "basic_damage", source: 0, target: 5 },
      { tokenId: "basic_damage", source: 1, target: 5 },
      { tokenId: "basic_damage", source: 2, target: 5 },
      { tokenId: "basic_damage", source: 3, target: 5 },
      { tokenId: "basic_heal", source: 6, target: 5 },
      { tokenId: "basic_heal", source: 7, target: 5 },
      { tokenId: "basic_heal", source: 8, target: 5 },
    ],
    routeCount: 7,
    netCount: 1,
  });
  await expectNetRecipients(page, [5]);
  const focusStatuses = sceneStatuses(frame, 5);
  expect(focusStatuses.map(({ tokenId }) => tokenId)).toEqual([
    "slow_hunter_basic",
    "priest_freedom",
  ]);
  await expectRosterStatuses(page, 5, focusStatuses);

  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: healthResolutionCount(frame),
    eventWindow: { eventType: HEALTH_RESOLUTION_EVENT_TYPE },
  });
});

test("mirrored Mage Burst separates activation from persistence", async ({ page }) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 1);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 1,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 2,
      cooldown_started: 2,
      ordinary_movement_phase_displacement: 2,
      status_applied: 2,
    },
    activations: [
      { tokenId: "mage_burst", source: 0, target: null },
      { tokenId: "mage_burst", source: 5, target: null },
    ],
    routeCount: 0,
    netCount: 0,
  });
  await expect(page.locator(".combat-burst__wave")).toHaveCount(4);
  expect(sceneStatuses(frame, 0).map(({ tokenId }) => tokenId)).toEqual(["mage_burst"]);
  await expectRosterStatuses(page, 0, sceneStatuses(frame, 0));
  await expectRosterStatuses(page, 5, sceneStatuses(frame, 5));

  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    eventWindow: { eventType: ABILITY_EVENT_TYPE, part: "group", progress: 0.2 },
  });
});

test("mirrored Warrior Charge keeps reciprocal routes and consequences distinct", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 2);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 2,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 2,
      charge_phase_displacement: 2,
      combat_countdown_reset: 2,
      cooldown_started: 2,
      recipient_health_resolution: 2,
      source_damage_output: 2,
      status_applied: 4,
    },
    activations: [
      { tokenId: "warrior_charge", source: 1, target: 6 },
      { tokenId: "warrior_charge", source: 6, target: 1 },
    ],
    routeCount: 2,
    netCount: 2,
  });
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--charge-displacement`),
  ).toHaveCount(2);
  expect(sceneStatuses(frame, 1).map(({ tokenId }) => tokenId)).toEqual([
    "stun_warrior_charge",
    "slow_warrior_charge",
  ]);
  await expectRosterStatuses(page, 1, sceneStatuses(frame, 1));
  await expectRosterStatuses(page, 6, sceneStatuses(frame, 6));

  await assertHealthResolutionPhase(page, healthResolutionCount(frame));
  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    eventWindow: { eventType: CHARGE_EVENT_TYPE },
  });
});

test("mirrored Hunter Freezing Trap keeps delivery separate from durable control", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 3);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 3,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 2,
      combat_countdown_reset: 2,
      cooldown_started: 2,
      ordinary_movement_phase_displacement: 2,
      recipient_health_resolution: 2,
      source_damage_output: 2,
      status_applied: 2,
    },
    activations: [
      { tokenId: "hunter_trap", source: 2, target: 7 },
      { tokenId: "hunter_trap", source: 7, target: 2 },
    ],
    routeCount: 2,
    netCount: 2,
  });
  await expect(page.locator(".combat-trap__lattice")).toHaveCount(2);
  expect(sceneStatuses(frame, 2).map(({ tokenId }) => tokenId)).toContain(
    "stun_hunter_trap",
  );
  await expectRosterStatuses(page, 2, sceneStatuses(frame, 2));
  await expectRosterStatuses(page, 7, sceneStatuses(frame, 7));

  await assertHealthResolutionPhase(page, healthResolutionCount(frame));
  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    eventWindow: { eventType: STATUS_EVENT_TYPE },
  });
});

test("mirrored Rogue Crippling Poison keeps route identity and three consequences legible", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 4);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 4,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 2,
      combat_countdown_reset: 2,
      cooldown_started: 2,
      ordinary_movement_phase_displacement: 2,
      recipient_health_resolution: 2,
      source_damage_output: 2,
      status_applied: 6,
    },
    activations: [
      { tokenId: "rogue_poison", source: 3, target: 8 },
      { tokenId: "rogue_poison", source: 8, target: 3 },
    ],
    routeCount: 2,
    netCount: 2,
  });
  await expect(page.locator(".combat-poison__splash")).toHaveCount(6);
  const poisonStatuses = sceneStatuses(frame, 3);
  expect(poisonStatuses.map(({ tokenId }) => tokenId)).toEqual([
    "stun_rogue_poison",
    "slow_rogue_poison",
    "anti_heal_rogue_poison",
  ]);
  await expectRosterStatuses(page, 3, poisonStatuses);
  await expectRosterStatuses(page, 8, poisonStatuses);

  await assertHealthResolutionPhase(page, healthResolutionCount(frame));
  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    eventWindow: { eventType: STATUS_EVENT_TYPE },
  });
});

test("mirrored Holy Word: Salvation and Crippling Poison lifecycle remains distinct across causal phases", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 5);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 5,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 2,
      combat_countdown_reset: 4,
      cooldown_started: 2,
      ordinary_movement_phase_displacement: 2,
      recipient_health_resolution: 2,
      source_healing_output: 2,
    },
    activations: [
      { tokenId: "holy_word", source: 4, target: 3 },
      { tokenId: "holy_word", source: 9, target: 8 },
    ],
    routeCount: 2,
    netCount: 2,
  });
  await expect(page.locator(".combat-holy__pulse")).toHaveCount(4);
  const lifecycle = statusEventsFromFrame(frame);
  const poisonStunLifecycle = lifecycle.filter(
    ({ tokenId }) => tokenId === "stun_rogue_poison",
  );
  expect(poisonStunLifecycle).toEqual(
    sceneStatusDuration(frame, "rogue_poison_stun") === 1
      ? [3, 8].map((recipient) => ({
          eventType: "status_aged_to_zero",
          lifecycle: "expired",
          recipient,
          source: null,
          tokenId: "stun_rogue_poison",
        }))
      : [],
  );
  await expectRenderedStatusEvents(page, lifecycle);
  await expectStatusFeed(
    page,
    lifecycle.map(({ eventType, recipient, source }) => ({
      eventType,
      recipient,
      source,
    })),
  );
  const remainingPoisonStatuses = sceneStatuses(frame, 3);
  await expectRosterStatuses(page, 3, remainingPoisonStatuses);
  await expectRosterStatuses(page, 8, remainingPoisonStatuses);

  await assertHealthResolutionPhase(page, healthResolutionCount(frame));
  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    // This transition has no status event family. Dynamic choreography omits
    // that dead window, so prove the post-health settled frame directly.
    settle: true,
  });
});

test("converging Charge preserves three directions without numeric collisions", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "charge_convergence",
  });
  await advanceScriptTo(page, 1);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "charge_convergence",
    transition: 1,
    roster: [0, 1, 5],
    stableEventKinds: {
      ability_activated: 3,
      charge_phase_displacement: 3,
      combat_countdown_reset: 3,
      cooldown_started: 3,
      recipient_health_resolution: 2,
      source_damage_output: 3,
      status_applied: 6,
    },
    activations: [
      { tokenId: "warrior_charge", source: 0, target: 5 },
      { tokenId: "warrior_charge", source: 1, target: 5 },
      { tokenId: "warrior_charge", source: 5, target: 0 },
    ],
    routeCount: 3,
    netCount: 2,
  });
  const routePaths = await page
    .locator(
      `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation .combat-route__path`,
    )
    .evaluateAll((paths) => paths.map((path) => path.getAttribute("d")));
  expect(routePaths).not.toContain(null);
  expect(new Set(routePaths).size).toBe(3);
  const chargeStatuses = sceneStatuses(frame, 0);
  expect(chargeStatuses.map(({ tokenId }) => tokenId)).toEqual([
    "stun_warrior_charge",
    "slow_warrior_charge",
  ]);
  await expectRosterStatuses(page, 0, chargeStatuses);
  await expectRosterStatuses(page, 5, chargeStatuses);

  await assertHealthResolutionPhase(page, healthResolutionCount(frame));
  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    eventWindow: { eventType: CHARGE_EVENT_TYPE },
  });
});

test("Freezing Trap lifecycle t1 proves four exact applications", async ({ page }) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "trap_lifecycle",
  });
  await advanceScriptTo(page, 1);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "trap_lifecycle",
    transition: 1,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 4,
      combat_countdown_reset: 8,
      cooldown_started: 4,
      recipient_health_resolution: 4,
      source_damage_output: 4,
      status_applied: 4,
    },
    activations: [
      { tokenId: "hunter_trap", source: 0, target: 5 },
      { tokenId: "hunter_trap", source: 1, target: 6 },
      { tokenId: "hunter_trap", source: 2, target: 7 },
      { tokenId: "hunter_trap", source: 3, target: 8 },
    ],
    routeCount: 4,
    netCount: 4,
  });
  const lifecycle = statusEventsFromFrame(frame);
  expect(lifecycle).toHaveLength(4);
  expect(lifecycle.every(({ tokenId }) => tokenId === "stun_hunter_trap")).toBe(true);
  await expectRenderedStatusEvents(page, lifecycle);
  await expectStatusFeed(
    page,
    lifecycle.map(({ eventType, recipient, source }) => ({
      eventType,
      recipient,
      source,
    })),
  );
  for (const recipient of [5, 6, 7, 8]) {
    await expectRosterStatuses(page, recipient, sceneStatuses(frame, recipient));
  }

  await assertHealthResolutionPhase(page, healthResolutionCount(frame));
  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    eventWindow: { eventType: STATUS_EVENT_TYPE },
  });
});

test("Freezing Trap lifecycle t2 proves authoritative records without overclaiming decrements", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "trap_lifecycle",
  });
  await advanceScriptTo(page, 2);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "trap_lifecycle",
    transition: 2,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 1,
      combat_countdown_reset: 2,
      recipient_health_resolution: 1,
      source_damage_output: 1,
      status_applied: 1,
    },
    activations: [{ tokenId: "basic_damage", source: 0, target: 5 }],
    routeCount: 1,
    netCount: 1,
  });
  await expectNetRecipients(page, [5]);
  const lifecycle = statusEventsFromFrame(frame);
  expect(lifecycle.filter(({ tokenId }) => tokenId === "stun_hunter_trap")).toEqual(
    expectedTrapLifecycle(frame, 2),
  );
  expect(lifecycle).toContainEqual({
    eventType: "status_applied",
    lifecycle: "applied",
    recipient: 5,
    source: 0,
    tokenId: "slow_hunter_basic",
  });
  await expectRenderedStatusEvents(page, lifecycle);
  await expectStatusFeed(page, [
    ...lifecycle.map(({ eventType, recipient, source }) => ({
      eventType,
      recipient,
      source,
    })),
  ]);
  // Former V1 decrement rows had no CP2 event identity. Their absence here is
  // now proved by durable successor status values instead of inferred events.
  await expectRosterStatuses(page, 5, sceneStatuses(frame, 5));
  for (const recipient of [6, 7, 8]) {
    await expectRosterStatuses(page, recipient, sceneStatuses(frame, recipient));
  }

  await assertHealthResolutionPhase(page, healthResolutionCount(frame));
  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    eventWindow: { eventType: STATUS_EVENT_TYPE },
  });
});

test("Freezing Trap lifecycle t4 proves authoritative lifecycle and durable successor values", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "trap_lifecycle",
  });
  await advanceScriptTo(page, 4);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "trap_lifecycle",
    transition: 4,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 1,
      combat_countdown_reset: 2,
      cooldown_started: 1,
      recipient_health_resolution: 1,
      source_damage_output: 1,
      status_applied: 1,
    },
    activations: [{ tokenId: "hunter_trap", source: 4, target: 6 }],
    routeCount: 1,
    netCount: 1,
  });
  const atomicLifecycle = statusEventsFromFrame(frame);
  expect(
    atomicLifecycle.filter(({ tokenId }) => tokenId === "stun_hunter_trap"),
  ).toEqual(expectedTrapLifecycle(frame, 4));
  expect(atomicLifecycle).toContainEqual({
    eventType: "status_applied",
    lifecycle: "applied",
    recipient: 6,
    source: 4,
    tokenId: "stun_hunter_trap",
  });
  const renderedLifecycle = atomicLifecycle
    .filter(
      (row) =>
        !(
          row.tokenId === "stun_hunter_trap" &&
          row.recipient === 6 &&
          (row.eventType === "status_broken_by_damage" ||
            row.eventType === "status_applied")
        ),
    )
    .concat({
      eventType: "status_applied",
      lifecycle: "trap_broken_and_reapplied",
      recipient: 6,
      source: 4,
      tokenId: "stun_hunter_trap",
    });
  await expectRenderedStatusEvents(page, renderedLifecycle);
  await expectStatusFeed(
    page,
    atomicLifecycle.map(({ eventType, recipient, source }) => ({
      eventType,
      recipient,
      source,
    })),
  );
  await expectRosterStatuses(page, 6, sceneStatuses(frame, 6));
  for (const recipient of [7, 8]) {
    await expectRosterStatuses(page, recipient, sceneStatuses(frame, recipient));
  }

  await assertHealthResolutionPhase(page, healthResolutionCount(frame));
  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    eventWindow: { eventType: STATUS_EVENT_TYPE },
  });
});

test("Freezing Trap lifecycle t5 retains independent authoritative lifecycle records", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "trap_lifecycle",
  });
  await advanceScriptTo(page, 5);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "trap_lifecycle",
    transition: 5,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    stableEventKinds: {
      ability_activated: 1,
      combat_countdown_reset: 2,
      recipient_health_resolution: 1,
      source_damage_output: 1,
      status_applied: 1,
    },
    activations: [{ tokenId: "basic_damage", source: 2, target: 7 }],
    routeCount: 1,
    netCount: 1,
  });
  await expectNetRecipients(page, [7]);
  const lifecycle = statusEventsFromFrame(frame);
  expect(lifecycle.filter(({ tokenId }) => tokenId === "stun_hunter_trap")).toEqual(
    expectedTrapLifecycle(frame, 5),
  );
  expect(lifecycle).toContainEqual({
    eventType: "status_applied",
    lifecycle: "applied",
    recipient: 7,
    source: 2,
    tokenId: "slow_hunter_basic",
  });
  await expectRenderedStatusEvents(page, lifecycle);
  await expectStatusFeed(
    page,
    lifecycle.map(({ eventType, recipient, source }) => ({
      eventType,
      recipient,
      source,
    })),
  );
  await expectRosterStatuses(page, 6, sceneStatuses(frame, 6));
  await expectRosterStatuses(page, 7, sceneStatuses(frame, 7));
  await expectRosterStatuses(page, 8, sceneStatuses(frame, 8));
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-lifecycle__shard`),
  ).toHaveCount(0);

  await assertHealthResolutionPhase(page, healthResolutionCount(frame));
  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    eventWindow: { eventType: STATUS_EVENT_TYPE },
  });
});

test("maximum status density remains complete after the explanation settles", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "max_status_stack",
  });
  await advanceScriptTo(page, 1);
  const frame = await assertLiveCanonicalFrame(page, {
    scenario: "max_status_stack",
    transition: 1,
    roster: [0, 1, 5, 6, 7, 8],
    stableEventKinds: {
      ability_activated: 6,
      charge_phase_displacement: 1,
      combat_countdown_reset: 5,
      cooldown_started: 4,
      recipient_health_resolution: 1,
      source_damage_output: 4,
      source_healing_output: 1,
      status_applied: 9,
    },
    activations: [
      { tokenId: "mage_burst", source: 0, target: null },
      { tokenId: "basic_heal", source: 1, target: 0 },
      { tokenId: "warrior_charge", source: 5, target: 0 },
      { tokenId: "hunter_trap", source: 6, target: 0 },
      { tokenId: "basic_damage", source: 7, target: 0 },
      { tokenId: "rogue_poison", source: 8, target: 0 },
    ],
    routeCount: 5,
    netCount: 1,
  });
  const fullStatusStack = sceneStatuses(frame, 0);
  expect(fullStatusStack.map(({ tokenId }) => tokenId)).toEqual([
    "stun_warrior_charge",
    "stun_hunter_trap",
    "stun_rogue_poison",
    "slow_warrior_charge",
    "slow_hunter_basic",
    "slow_rogue_poison",
    "anti_heal_rogue_poison",
    "priest_freedom",
    "mage_burst",
  ]);
  await expectRosterStatuses(page, 0, fullStatusStack);
  await expectBattlefieldStatuses(page, 0, fullStatusStack);
  await expectBattlefieldCooldowns(page, sceneCooldowns(frame));
  const durableLayer = page.locator(
    '#battlefield [data-layer="durable-status-modifier"]',
  );
  await expect(durableLayer).toHaveAttribute("data-suppressed-status-slots", "");
  await expect(durableLayer).toHaveAttribute("data-suppressed-cooldown-slots", "");
  await expect(
    page.locator('#battlefield .status-dock[data-slot="0"]'),
  ).toHaveAttribute("data-expanded", "true");
  await expect(page.locator("#battlefield .cooldown-dock")).toHaveCount(4);

  await assertStablePresentationFrame(page, {
    commandPosts,
    expectedTransientCount: 0,
    settle: true,
    afterSettle: async () => {
      const settled = await choreographySnapshot(page);
      expect(settled.animationIds).toEqual([]);
      await expect(
        page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`),
      ).toHaveCount(0);
      await expect(
        page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle`),
      ).toHaveCount(0);
      // Charge displacement is the only V2 event whose exact path persists
      // after the animation settles, preserving spatial truth through UI-only
      // activity without retaining unrelated transient cues.
      await expect(
        page.locator(
          `${CHOREOGRAPHY_ROOT} .combat-effect--charge-displacement[data-event-type="charge_phase_displacement"]`,
        ),
      ).toHaveCount(1);
      await expect(
        page.locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-charge__path`),
      ).toHaveCount(1);
    },
  });
});

test("crowded synthetic renderer fixture remains bounded at the minimum viewport", async ({
  page,
}) => {
  const commandPosts = await installSyntheticVisualCase(
    page,
    debuggerUrl,
    crowdedWireFrame,
    { viewport: MINIMUM_VIEWPORT },
  );
  await assertFrameIdentity(page, {
    scenario: "crowded_teamfight",
    simulatorStep: 1,
    transitionId: 1,
    view: "researcher",
    preset: "analysis",
    badge: /PRIVILEGED RESEARCHER VIEW.*SYNTHETIC FIXTURE/,
  });
  await expect(page.locator("#scenario-description")).toHaveText(
    "SYNTHETIC: dense V2 status, aura, range, selection, legality, and simultaneous-event pressure.",
  );
  await expectRosterSlots(page, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  await assertHudStoryLabels(page, {
    pending: "PLAYBACK / INSPECTION ONLY",
    accepted: "No transition yet.",
  });
  await assertCurrentEventIds(page, 32);
  await expectEventKindCounts(page, {
    ability_activated: 10,
    charge_phase_displacement: 2,
    recipient_health_resolution: 8,
    status_applied: 12,
  });
  await expectActivationPairs(page, [
    { tokenId: "basic_damage", source: 0, target: 5 },
    { tokenId: "warrior_charge", source: 1, target: 6 },
    { tokenId: "hunter_trap", source: 2, target: 7 },
    { tokenId: "rogue_poison", source: 3, target: 8 },
    { tokenId: "holy_word", source: 4, target: 4 },
    { tokenId: "basic_damage", source: 5, target: 0 },
    { tokenId: "warrior_charge", source: 6, target: 1 },
    { tokenId: "hunter_trap", source: 7, target: 2 },
    { tokenId: "rogue_poison", source: 8, target: 3 },
    { tokenId: "holy_word", source: 9, target: 9 },
  ]);
  await expectActivationRouteCount(page, 10);
  await expectNetCount(page, 8);
  await expectNetRecipients(page, [0, 1, 3, 4, 5, 6, 8, 9]);
  const netAssociations = await page
    .locator(`${CHOREOGRAPHY_ROOT} .combat-effect--net-health`)
    .evaluateAll((effects) =>
      effects.map((effect) => {
        const recipient = effect.getAttribute("data-recipient-slot");
        return {
          recipient,
          recipientLabel: effect
            .querySelector(".combat-net__recipient")
            ?.textContent?.trim(),
          hasAnchor: Boolean(effect.querySelector(".combat-net__recipient-anchor")),
          hasLeader: Boolean(effect.querySelector(".combat-cue__leader")),
        };
      }),
    );
  expect(netAssociations).toEqual(
    [0, 1, 3, 4, 5, 6, 8, 9].map((recipient) => ({
      recipient: String(recipient),
      recipientLabel: `Agent ID ${recipient}`,
      hasAnchor: true,
      hasLeader: true,
    })),
  );
  const battlefieldBounds = await page.locator("#battlefield").boundingBox();
  expect(battlefieldBounds).not.toBeNull();
  expect(battlefieldBounds?.y).toBeLessThan(300);
  expect(
    Math.min(
      battlefieldBounds?.height ?? 0,
      MINIMUM_VIEWPORT.height - (battlefieldBounds?.y ?? MINIMUM_VIEWPORT.height),
    ),
  ).toBeGreaterThan(300);
  const canonicalStack = statuses([
    ["stun_warrior_charge", 3],
    ["stun_hunter_trap", 3],
    ["stun_rogue_poison", 3],
    ["slow_warrior_charge", 2],
    ["slow_hunter_basic", 2],
    ["slow_rogue_poison", 2],
    ["anti_heal_rogue_poison", 3],
    ["priest_freedom", 2],
    ["mage_burst", 3],
  ]);
  for (const slot of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
    await expectRosterStatuses(page, slot, canonicalStack);
  }
  for (const slot of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
    const dock = page.locator(`#battlefield .status-dock[data-slot="${slot}"]`);
    await expect(dock).toHaveAttribute("data-expanded", "false");
    await expect(dock).toHaveAttribute("data-visible-count", "0");
    await expect(dock).toHaveAttribute("data-hidden-count", "9");
  }
  await assertStatusDockCoverage(
    page,
    /** @type {Array<{global_slot: number, statuses: unknown[]}>} */ (
      crowdedFrame.scene.agents
    ),
  );
  await assertDurableDockFlags(page);
  await assertBoundedChoreography(page);
  await assertTransientSlotsAuthorized(page);

  await captureBaseline(
    page,
    "crowded-teamfight-synthetic-health-resolution-phase-960x600.png",
    {
      afterSettle: async () => {
        await assertCompactActiveCombatPriority(page, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      },
      commandPosts,
      expectedTransientCount: 8,
      // All eight authoritative NET outcomes remain visible. Lifecycle cues
      // reserve only same-phase peers, so every later status record retains a
      // collision-free disposition instead of competing with health labels
      // that no longer coexist on screen.
      expectedSuppressedLifecycleCount: 0,
      eventWindow: { eventType: HEALTH_RESOLUTION_EVENT_TYPE },
    },
  );
});

test("synthetic POV fixture omits hidden agents and spatial endpoints", async ({
  page,
}) => {
  const servedFrame = /** @type {Record<string, any>} */ ({
    ...povFrame,
    preset: "analysis",
  });
  const servedWireFrame = /** @type {Record<string, any>} */ ({
    ...povWireFrame,
    preset: "analysis",
  });
  expectPovPayloadRedacted(servedFrame);
  expect(
    /** @type {Array<{global_slot: number}>} */ (servedFrame.scene.agents).map(
      (agent) => agent.global_slot,
    ),
  ).toEqual([0]);
  expect(servedFrame.scene.observer_visibility).toEqual([]);
  const ownActionOutcome = /** @type {Array<Record<string, any>>} */ (
    servedFrame.event_batch.events
  ).find((event) => event.event_type === "own_action_outcome");
  expect(ownActionOutcome).toMatchObject({ outcome: "accepted" });
  expect(ownActionOutcome).not.toHaveProperty("source_global_slot");
  expect(ownActionOutcome).not.toHaveProperty("recipient_global_slot");
  expect(ownActionOutcome).not.toHaveProperty("target_global_slot");
  const commandPosts = await installSyntheticVisualCase(
    page,
    debuggerUrl,
    servedWireFrame,
    { viewport: DESKTOP_VIEWPORT },
  );
  await assertFrameIdentity(page, {
    scenario: null,
    simulatorStep: 1,
    transitionId: 1,
    view: "pov",
    preset: "analysis",
    badge: /AGENT POV.*SYNTHETIC FIXTURE/,
  });
  await expect(page.locator("#scenario-description")).toContainText(
    "Authoritative debugger frame received.",
  );
  await expectRosterSlots(page, [0]);
  await assertHudStoryLabels(page, {
    pending: "PLAYBACK / INSPECTION ONLY",
    accepted: "No transition yet.",
  });
  await assertCurrentEventIds(page, 6);
  await expectEventKindCounts(page, {
    own_action_outcome: 1,
    own_health_changed: 1,
    own_position_changed: 1,
    own_status_changed: 3,
  });
  await expectActivationPairs(page, []);
  await expectActivationRouteCount(page, 0);
  await expectNetCount(page, 1);
  // Decode only the recipient-authorized duration columns. Source identity is
  // deliberately absent even though the V1 feature layout names each effect.
  await expectRosterStatuses(
    page,
    0,
    statuses([
      ["slow_warrior_charge", 1],
      ["slow_hunter_basic", 1],
      ["slow_rogue_poison", 1],
    ]),
  );
  await expect(page.locator('#battlefield .agent[data-slot="5"]')).toHaveCount(0);
  await expect(page.locator('#roster .roster-row[data-slot="5"]')).toHaveCount(0);
  const observedTrap = page.locator(
    '.pov-observed-status[data-token-id="stun_hunter_trap"]',
  );
  await expect(observedTrap).toHaveCount(1);
  await expect(observedTrap).toHaveAttribute("data-duration", "2");
  await expect(observedTrap).toHaveAttribute("data-effect-class", "hunter");
  expect(
    await observedTrap.evaluate((element) => getComputedStyle(element).color),
  ).toBe("rgb(132, 204, 22)");
  await expect(
    page.locator('.pov-observed-body[data-observation-key="ally:1"]'),
  ).toHaveAttribute("aria-label", /Hunter Freezing Trap stun, 2 ticks/u);
  await observedTrap.hover();
  await expect(page.locator("#visual-tooltip-title")).toHaveText(
    "Hunter (Ultimate: Freezing Trap) Stun",
  );
  await expect(page.locator("#visual-tooltip-details")).toContainText(
    "Source agent identity is not disclosed",
  );
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-protected-zone")).toHaveCount(0);
  await expect(
    page.locator('#battlefield .agent[data-public-agent-id="5"]'),
  ).toHaveCount(0);
  await expect(
    page.locator('#roster .roster-row[data-public-agent-id="5"]'),
  ).toHaveCount(0);
  await expect(
    page.locator('[data-source-slot="5"], [data-target-slot="5"]'),
  ).toHaveCount(0);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-target-slot]`),
  ).toHaveCount(0);
  await assertDurableDockFlags(page);
  await assertBoundedChoreography(page);
  await assertTransientSlotsAuthorized(page);

  await captureBaseline(
    page,
    "pov-redaction-synthetic-analysis-successor-observation-phase-1440x900.png",
    {
      commandPosts,
      expectedTransientCount: 1,
      // Authorized successor-only POV deltas share one non-causal observation
      // phase; this capture must not imply movement-before-health ordering.
      eventWindow: { eventType: POV_HEALTH_EVENT_TYPE },
    },
  );
});
