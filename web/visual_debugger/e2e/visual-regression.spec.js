import { expect, test } from "@playwright/test";

import {
  assertBoundedChoreography,
  assertTransientSlotsAuthorized,
  CHOREOGRAPHY_ROOT,
  CHOREOGRAPHY_ROUTE_ROOT,
  choreographySnapshot,
} from "./support/choreography.js";
import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  loadRendererFixture,
  syntheticDebuggerFrame,
} from "./support/renderer-fixture.js";
import {
  advanceScriptTo,
  assertCurrentEventIds,
  assertDurableDockFlags,
  assertFrameIdentity,
  assertHudStoryLabels,
  captureBaseline,
  DESKTOP_VIEWPORT,
  expectActivationPairs,
  expectRosterSlots,
  expectRosterStatuses,
  installSyntheticVisualCase,
  loadLiveVisualCase,
  MID_IMPACT_MS,
  MINIMUM_VIEWPORT,
} from "./support/visual-regression.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";
/** @type {Record<string, any>} */
let crowdedFrame = {};
/** @type {Record<string, any>} */
let durableControlsFrame = {};
/** @type {Record<string, any>} */
let povFrame = {};
/** @type {Record<string, any>} */
let vocabularyFrame = {};

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
  crowdedFrame = syntheticDebuggerFrame(crowdedFixture);
  durableControlsFrame = syntheticDebuggerFrame(durableControlsFixture);
  povFrame = syntheticDebuggerFrame(povFixture);
  vocabularyFrame = syntheticDebuggerFrame(vocabularyFixture);
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
 * Prove the exact authoritative lifecycle records remain present in the
 * structured event feed, including non-spatial duration decrements.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Array<{
 *   label: string,
 *   recipient: number,
 *   change: string,
 *   before: number,
 *   after: number,
 * }>} expected
 */
async function expectLifecycleFeed(page, expected) {
  const observed = await page
    .locator('#event-feed .event-item[data-event-type="status_lifecycle"]')
    .evaluateAll((items) =>
      items.map((item) => ({
        recipient: Number(item.getAttribute("data-recipient-slot")),
        summary: item.textContent?.trim() ?? "",
      })),
    );
  expect(observed).toEqual(
    expected.map(({ label, recipient, change, before, after }) => ({
      recipient,
      summary: `${label} · id_${recipient} · ${change} · duration ${before} → ${after}`,
    })),
  );
}

/**
 * Prove every defensible spatial lifecycle classification carries the exact
 * recipient, status, duration epochs, and application-link cardinality.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Array<{
 *   tokenId: string,
 *   recipient: number,
 *   change: string,
 *   before: number,
 *   after: number,
 *   applicationCount: number,
 * }>} expected
 */
async function expectRenderedLifecycle(page, expected) {
  const observed = await page
    .locator(`${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle`)
    .evaluateAll((effects) =>
      effects.map((effect) => ({
        after: Number(effect.getAttribute("data-duration-after")),
        applicationCount: JSON.parse(
          effect.getAttribute("data-application-event-ids") ?? "[]",
        ).length,
        before: Number(effect.getAttribute("data-duration-before")),
        change: effect.getAttribute("data-lifecycle"),
        recipient: Number(effect.getAttribute("data-recipient-slot")),
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
      ({ count, owner, slot }) => count === "9" && owner === `id_${slot}`,
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
  const authorizedPoints = new Set(
    agents.map((agent) => JSON.stringify(agent.position)),
  );
  /** @type {string[]} */
  const violations = [];

  /**
   * @param {unknown} value
   * @param {string} path
   */
  const visit = (value, path) => {
    const field = path.split(".").at(-1) ?? "";
    if (Array.isArray(value)) {
      if (
        /(?:anchor|position|start|end)$/.test(field) &&
        value.length === 2 &&
        value.every(Number.isFinite) &&
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
    if (
      record.target_disclosure === "redacted" &&
      (record.target_global_slot !== null || record.target_anchor !== null)
    ) {
      violations.push(`${path} retains a redacted target identity or anchor.`);
    }
    if (
      record.event_type === "rejected_action" &&
      record.target_disclosure === "redacted"
    ) {
      violations.push(`${path} discloses rejection of a redacted target pair.`);
    }
    for (const [key, item] of Object.entries(record)) {
      if (
        /global_slot$/.test(key) &&
        Number.isInteger(item) &&
        !authorizedSlots.has(Number(item))
      ) {
        violations.push(`${path}.${key} exposes unauthorized slot ${item}.`);
      }
      visit(item, `${path}.${key}`);
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
 *   eventCount: number,
 *   eventKinds: Record<string, number>,
 *   activations: Array<{tokenId: string, source: number, target: number | null}>,
 *   routeCount: number,
 *   netCount: number,
 * }} expected
 */
async function assertLiveMidImpactFrame(page, expected) {
  await assertFrameIdentity(page, {
    scenario: expected.scenario,
    simulatorStep: expected.transition,
    transitionId: expected.transition,
    view: "researcher",
    preset: "analysis",
    badge: "PRIVILEGED RESEARCHER VIEW",
  });
  await expectRosterSlots(page, expected.roster);
  await assertHudStoryLabels(page, {
    pending: "PLAYBACK / INSPECTION ONLY",
    accepted: "LATEST ACCEPTED RESULT",
  });
  await assertCurrentEventIds(page, expected.eventCount);
  await expectEventKindCounts(page, expected.eventKinds);
  await expectActivationPairs(page, expected.activations);
  await expectActivationRouteCount(page, expected.routeCount);
  await expectNetCount(page, expected.netCount);
  await assertDurableDockFlags(page);
  await assertBoundedChoreography(page);
  await assertTransientSlotsAuthorized(page);
}

/**
 * @param {Array<[string, number]>} pairs
 */
function statuses(pairs) {
  return pairs.map(([tokenId, duration]) => ({ tokenId, duration }));
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
    badge: "PRIVILEGED RESEARCHER VIEW",
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

  await captureBaseline(page, "arena-5v5-idle-analysis-1440x900.png", {
    commandPosts,
    expectedTransientCount: 0,
  });
});

test("visual vocabulary presents every class and combat grammar", async ({ page }) => {
  const commandPosts = await installSyntheticVisualCase(
    page,
    debuggerUrl,
    vocabularyFrame,
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
    accepted: "SYNTHETIC EVENT BATCH",
  });
  await assertCurrentEventIds(page, 12);
  await expectEventKindCounts(page, {
    accepted_activation: 10,
    net_health: 2,
  });
  await expectActivationPairs(page, [
    { tokenId: "basic_damage", source: 0, target: 5 },
    { tokenId: "basic_damage", source: 1, target: 6 },
    { tokenId: "basic_damage", source: 2, target: 7 },
    { tokenId: "basic_damage", source: 3, target: 8 },
    { tokenId: "basic_heal", source: 4, target: 9 },
    { tokenId: "mage_burst", source: 0, target: null },
    { tokenId: "warrior_charge", source: 1, target: 6 },
    { tokenId: "hunter_trap", source: 2, target: 7 },
    { tokenId: "rogue_poison", source: 3, target: 8 },
    { tokenId: "holy_word", source: 4, target: 9 },
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

  await captureBaseline(page, "visual-vocabulary-synthetic-mid-impact-1440x900.png", {
    commandPosts,
    expectedTransientCount: 2,
    logicalMs: MID_IMPACT_MS,
  });
});

test("durable controls use one stun and one slow glyph with source accents", async ({
  page,
}) => {
  const commandPosts = await installSyntheticVisualCase(
    page,
    debuggerUrl,
    durableControlsFrame,
    { viewport: DESKTOP_VIEWPORT },
  );
  await assertFrameIdentity(page, {
    scenario: "durable_controls",
    simulatorStep: 0,
    transitionId: 0,
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
  await assertLiveMidImpactFrame(page, {
    scenario: "team_focus_crossfire",
    transition: 3,
    roster: [0, 1, 2, 3, 5, 6, 7, 8],
    eventCount: 10,
    eventKinds: {
      accepted_activation: 7,
      net_health: 1,
      status_lifecycle: 2,
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
  await expectRosterStatuses(
    page,
    5,
    statuses([
      ["slow_hunter_basic", 1],
      ["priest_freedom", 1],
    ]),
  );

  await captureBaseline(page, "team-focus-crossfire-t3-mid-impact-1440x900.png", {
    commandPosts,
    expectedTransientCount: 1,
    logicalMs: MID_IMPACT_MS,
  });
});

test("moving Basic crossfire follows all ten successor bodies", async ({ page }) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "moving_basic_crossfire",
  });
  await advanceScriptTo(page, 1);
  await assertLiveMidImpactFrame(page, {
    scenario: "moving_basic_crossfire",
    transition: 1,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 22,
    eventKinds: {
      accepted_activation: 10,
      net_health: 8,
      status_lifecycle: 4,
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
  await expectRosterStatuses(page, 0, statuses([["priest_freedom", 1]]));
  await expectRosterStatuses(page, 2, statuses([["slow_hunter_basic", 1]]));
  await expectRosterStatuses(page, 5, statuses([["priest_freedom", 1]]));
  await expectRosterStatuses(page, 7, statuses([["slow_hunter_basic", 1]]));

  await captureBaseline(page, "moving-basic-crossfire-t1-mid-impact-1440x900.png", {
    commandPosts,
    expectedTransientCount: 8,
    logicalMs: MID_IMPACT_MS,
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
  await assertLiveMidImpactFrame(page, {
    scenario: "moving_focus_crossfire",
    transition: 1,
    roster: [0, 1, 2, 3, 5, 6, 7, 8],
    eventCount: 10,
    eventKinds: {
      accepted_activation: 7,
      net_health: 1,
      status_lifecycle: 2,
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
  await expectRosterStatuses(
    page,
    5,
    statuses([
      ["slow_hunter_basic", 1],
      ["priest_freedom", 1],
    ]),
  );

  await captureBaseline(page, "moving-focus-crossfire-t1-mid-impact-960x600.png", {
    commandPosts,
    expectedTransientCount: 1,
    logicalMs: MID_IMPACT_MS,
  });
});

test("mirrored Mage Burst separates activation from persistence", async ({ page }) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 1);
  await assertLiveMidImpactFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 1,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 4,
    eventKinds: {
      accepted_activation: 2,
      status_lifecycle: 2,
    },
    activations: [
      { tokenId: "mage_burst", source: 0, target: null },
      { tokenId: "mage_burst", source: 5, target: null },
    ],
    routeCount: 0,
    netCount: 0,
  });
  await expect(page.locator(".combat-burst__wave")).toHaveCount(4);
  await expectRosterStatuses(page, 0, statuses([["mage_burst", 5]]));
  await expectRosterStatuses(page, 5, statuses([["mage_burst", 5]]));

  await captureBaseline(
    page,
    "mirrored-ultimates-mage-burst-t1-mid-impact-1440x900.png",
    {
      commandPosts,
      expectedTransientCount: 0,
      logicalMs: MID_IMPACT_MS,
    },
  );
});

test("mirrored Warrior Charge keeps reciprocal routes and consequences distinct", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 2);
  await assertLiveMidImpactFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 2,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 12,
    eventKinds: {
      accepted_activation: 2,
      charge_displacement: 2,
      net_health: 2,
      status_lifecycle: 6,
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
  await expectRosterStatuses(
    page,
    1,
    statuses([
      ["stun_warrior_charge", 1],
      ["slow_warrior_charge", 5],
    ]),
  );
  await expectRosterStatuses(
    page,
    6,
    statuses([
      ["stun_warrior_charge", 1],
      ["slow_warrior_charge", 5],
    ]),
  );

  await captureBaseline(
    page,
    "mirrored-ultimates-warrior-charge-t2-mid-impact-1440x900.png",
    {
      commandPosts,
      expectedTransientCount: 2,
      logicalMs: MID_IMPACT_MS,
    },
  );
});

test("mirrored Hunter Trap keeps delivery separate from durable control", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 3);
  await assertLiveMidImpactFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 3,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 12,
    eventKinds: {
      accepted_activation: 2,
      net_health: 2,
      status_lifecycle: 8,
    },
    activations: [
      { tokenId: "hunter_trap", source: 2, target: 7 },
      { tokenId: "hunter_trap", source: 7, target: 2 },
    ],
    routeCount: 2,
    netCount: 2,
  });
  await expect(page.locator(".combat-trap__lattice")).toHaveCount(2);
  await expectRosterStatuses(page, 2, statuses([["stun_hunter_trap", 4]]));
  await expectRosterStatuses(page, 7, statuses([["stun_hunter_trap", 4]]));

  await captureBaseline(
    page,
    "mirrored-ultimates-hunter-trap-t3-mid-impact-1440x900.png",
    {
      commandPosts,
      expectedTransientCount: 2,
      logicalMs: MID_IMPACT_MS,
    },
  );
});

test("mirrored Rogue Poison keeps route identity and three consequences legible", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 4);
  await assertLiveMidImpactFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 4,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 16,
    eventKinds: {
      accepted_activation: 2,
      net_health: 2,
      status_lifecycle: 12,
    },
    activations: [
      { tokenId: "rogue_poison", source: 3, target: 8 },
      { tokenId: "rogue_poison", source: 8, target: 3 },
    ],
    routeCount: 2,
    netCount: 2,
  });
  await expect(page.locator(".combat-poison__splash")).toHaveCount(6);
  const poisonStatuses = statuses([
    ["stun_rogue_poison", 1],
    ["slow_rogue_poison", 5],
    ["anti_heal_rogue_poison", 4],
  ]);
  await expectRosterStatuses(page, 3, poisonStatuses);
  await expectRosterStatuses(page, 8, poisonStatuses);

  await captureBaseline(
    page,
    "mirrored-ultimates-rogue-poison-t4-mid-impact-1440x900.png",
    {
      commandPosts,
      expectedTransientCount: 2,
      logicalMs: MID_IMPACT_MS,
    },
  );
});

test("mirrored Holy Word remains readable beside truthful Poison expiry cues", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "mirrored_ultimates",
  });
  await advanceScriptTo(page, 5);
  await assertLiveMidImpactFrame(page, {
    scenario: "mirrored_ultimates",
    transition: 5,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 16,
    eventKinds: {
      accepted_activation: 2,
      net_health: 2,
      status_lifecycle: 12,
    },
    activations: [
      { tokenId: "holy_word", source: 4, target: 3 },
      { tokenId: "holy_word", source: 9, target: 8 },
    ],
    routeCount: 2,
    netCount: 2,
  });
  await expect(page.locator(".combat-holy__pulse")).toHaveCount(4);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-token-id="stun_rogue_poison"][data-lifecycle="expired"]`,
    ),
  ).toHaveCount(2);
  const remainingPoisonStatuses = statuses([
    ["slow_rogue_poison", 4],
    ["anti_heal_rogue_poison", 3],
  ]);
  await expectRosterStatuses(page, 3, remainingPoisonStatuses);
  await expectRosterStatuses(page, 8, remainingPoisonStatuses);

  await captureBaseline(
    page,
    "mirrored-ultimates-holy-word-t5-mid-impact-1440x900.png",
    {
      commandPosts,
      expectedTransientCount: 2,
      logicalMs: MID_IMPACT_MS,
    },
  );
});

test("converging Charge preserves three directions without numeric collisions", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "charge_convergence",
  });
  await advanceScriptTo(page, 1);
  await assertLiveMidImpactFrame(page, {
    scenario: "charge_convergence",
    transition: 1,
    roster: [0, 1, 5],
    eventCount: 12,
    eventKinds: {
      accepted_activation: 3,
      charge_displacement: 3,
      net_health: 2,
      status_lifecycle: 4,
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
  const chargeStatuses = statuses([
    ["stun_warrior_charge", 1],
    ["slow_warrior_charge", 5],
  ]);
  await expectRosterStatuses(page, 0, chargeStatuses);
  await expectRosterStatuses(page, 5, chargeStatuses);

  await captureBaseline(page, "charge-convergence-t1-mid-impact-1440x900.png", {
    commandPosts,
    expectedTransientCount: 2,
    logicalMs: MID_IMPACT_MS,
  });
});

test("Trap lifecycle t1 proves four exact applications", async ({ page }) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "trap_lifecycle",
  });
  await advanceScriptTo(page, 1);
  await assertLiveMidImpactFrame(page, {
    scenario: "trap_lifecycle",
    transition: 1,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 12,
    eventKinds: {
      accepted_activation: 4,
      net_health: 4,
      status_lifecycle: 4,
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
  const lifecycle = [5, 6, 7, 8].map((recipient) => ({
    after: 4,
    applicationCount: 1,
    before: 0,
    change: "applied",
    recipient,
    tokenId: "stun_hunter_trap",
  }));
  await expectRenderedLifecycle(page, lifecycle);
  await expectLifecycleFeed(
    page,
    lifecycle.map(({ recipient }) => ({
      after: 4,
      before: 0,
      change: "Applied",
      label: "Trap",
      recipient,
    })),
  );
  for (const recipient of [5, 6, 7, 8]) {
    await expectRosterStatuses(page, recipient, statuses([["stun_hunter_trap", 4]]));
  }

  await captureBaseline(page, "trap-lifecycle-t1-applied-1440x900.png", {
    commandPosts,
    expectedTransientCount: 4,
    logicalMs: MID_IMPACT_MS,
  });
});

test("Trap lifecycle t2 proves one exact break without overclaiming decrements", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "trap_lifecycle",
  });
  await advanceScriptTo(page, 2);
  await assertLiveMidImpactFrame(page, {
    scenario: "trap_lifecycle",
    transition: 2,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 7,
    eventKinds: {
      accepted_activation: 1,
      net_health: 1,
      status_lifecycle: 5,
    },
    activations: [{ tokenId: "basic_damage", source: 0, target: 5 }],
    routeCount: 1,
    netCount: 1,
  });
  await expectNetRecipients(page, [5]);
  await expectRenderedLifecycle(page, [
    {
      after: 1,
      applicationCount: 1,
      before: 0,
      change: "applied",
      recipient: 5,
      tokenId: "slow_hunter_basic",
    },
    {
      after: 0,
      applicationCount: 0,
      before: 4,
      change: "trap_broken",
      recipient: 5,
      tokenId: "stun_hunter_trap",
    },
  ]);
  await expectLifecycleFeed(page, [
    {
      after: 1,
      before: 0,
      change: "Applied",
      label: "Hunter slow",
      recipient: 5,
    },
    {
      after: 0,
      before: 4,
      change: "Trap Broken",
      label: "Trap",
      recipient: 5,
    },
    ...[6, 7, 8].map((recipient) => ({
      after: 3,
      before: 4,
      change: "Decremented",
      label: "Trap",
      recipient,
    })),
  ]);
  await expectRosterStatuses(page, 5, statuses([["slow_hunter_basic", 1]]));
  for (const recipient of [6, 7, 8]) {
    await expectRosterStatuses(page, recipient, statuses([["stun_hunter_trap", 3]]));
  }

  await captureBaseline(page, "trap-lifecycle-t2-broken-1440x900.png", {
    commandPosts,
    expectedTransientCount: 1,
    logicalMs: MID_IMPACT_MS,
  });
});

test("Trap lifecycle t4 proves exact refresh and durable successor values", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "trap_lifecycle",
  });
  await advanceScriptTo(page, 4);
  await assertLiveMidImpactFrame(page, {
    scenario: "trap_lifecycle",
    transition: 4,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 5,
    eventKinds: {
      accepted_activation: 1,
      net_health: 1,
      status_lifecycle: 3,
    },
    activations: [{ tokenId: "hunter_trap", source: 4, target: 6 }],
    routeCount: 1,
    netCount: 1,
  });
  await expectRenderedLifecycle(page, [
    {
      after: 4,
      applicationCount: 1,
      before: 2,
      change: "refreshed",
      recipient: 6,
      tokenId: "stun_hunter_trap",
    },
  ]);
  await expectLifecycleFeed(page, [
    {
      after: 4,
      before: 2,
      change: "Refreshed",
      label: "Trap",
      recipient: 6,
    },
    ...[7, 8].map((recipient) => ({
      after: 1,
      before: 2,
      change: "Decremented",
      label: "Trap",
      recipient,
    })),
  ]);
  await expectRosterStatuses(page, 6, statuses([["stun_hunter_trap", 4]]));
  for (const recipient of [7, 8]) {
    await expectRosterStatuses(page, recipient, statuses([["stun_hunter_trap", 1]]));
  }

  await captureBaseline(page, "trap-lifecycle-t4-refreshed-1440x900.png", {
    commandPosts,
    expectedTransientCount: 1,
    logicalMs: MID_IMPACT_MS,
  });
});

test("Trap lifecycle t5 separates ambiguous ending from natural expiry", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "trap_lifecycle",
  });
  await advanceScriptTo(page, 5);
  await assertLiveMidImpactFrame(page, {
    scenario: "trap_lifecycle",
    transition: 5,
    roster: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    eventCount: 6,
    eventKinds: {
      accepted_activation: 1,
      net_health: 1,
      status_lifecycle: 4,
    },
    activations: [{ tokenId: "basic_damage", source: 2, target: 7 }],
    routeCount: 1,
    netCount: 1,
  });
  await expectNetRecipients(page, [7]);
  await expectRenderedLifecycle(page, [
    {
      after: 1,
      applicationCount: 1,
      before: 0,
      change: "applied",
      recipient: 7,
      tokenId: "slow_hunter_basic",
    },
    {
      after: 0,
      applicationCount: 0,
      before: 1,
      change: "cleared_unclassified",
      recipient: 7,
      tokenId: "stun_hunter_trap",
    },
    {
      after: 0,
      applicationCount: 0,
      before: 1,
      change: "expired",
      recipient: 8,
      tokenId: "stun_hunter_trap",
    },
  ]);
  await expectLifecycleFeed(page, [
    {
      after: 1,
      before: 0,
      change: "Applied",
      label: "Hunter slow",
      recipient: 7,
    },
    {
      after: 3,
      before: 4,
      change: "Decremented",
      label: "Trap",
      recipient: 6,
    },
    {
      after: 0,
      before: 1,
      change: "Cleared Unclassified",
      label: "Trap",
      recipient: 7,
    },
    {
      after: 0,
      before: 1,
      change: "Expired",
      label: "Trap",
      recipient: 8,
    },
  ]);
  await expectRosterStatuses(page, 6, statuses([["stun_hunter_trap", 3]]));
  await expectRosterStatuses(page, 7, statuses([["slow_hunter_basic", 1]]));
  await expectRosterStatuses(page, 8, []);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-lifecycle__shard`),
  ).toHaveCount(0);

  await captureBaseline(page, "trap-lifecycle-t5-ambiguous-and-expired-1440x900.png", {
    commandPosts,
    expectedTransientCount: 1,
    logicalMs: MID_IMPACT_MS,
  });
});

test("maximum status density remains complete after the explanation settles", async ({
  page,
}) => {
  const commandPosts = await loadLiveVisualCase(page, debuggerUrl, {
    scenario: "max_status_stack",
  });
  await advanceScriptTo(page, 1);
  await assertLiveMidImpactFrame(page, {
    scenario: "max_status_stack",
    transition: 1,
    roster: [0, 1, 5, 6, 7, 8],
    eventCount: 17,
    eventKinds: {
      accepted_activation: 6,
      charge_displacement: 1,
      net_health: 1,
      status_lifecycle: 9,
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
  const fullStatusStack = statuses([
    ["stun_warrior_charge", 1],
    ["stun_hunter_trap", 4],
    ["stun_rogue_poison", 1],
    ["slow_warrior_charge", 5],
    ["slow_hunter_basic", 1],
    ["slow_rogue_poison", 5],
    ["anti_heal_rogue_poison", 4],
    ["priest_freedom", 1],
    ["mage_burst", 5],
  ]);
  await expectRosterStatuses(page, 0, fullStatusStack);
  await expectBattlefieldStatuses(page, 0, fullStatusStack);
  await expectBattlefieldCooldowns(page, [
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
  await expect(
    page.locator('#battlefield .status-dock[data-slot="0"]'),
  ).toHaveAttribute("data-expanded", "true");
  await expect(page.locator("#battlefield .cooldown-dock")).toHaveCount(4);

  await captureBaseline(page, "max-status-stack-t1-settled-1440x900.png", {
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
      await expect(
        page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--charge-displacement`),
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
    crowdedFrame,
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
  await expect(page.locator("#scenario-description")).toContainText("SYNTHETIC:");
  await expectRosterSlots(page, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  await assertHudStoryLabels(page, {
    pending: "PLAYBACK / INSPECTION ONLY",
    accepted: "SYNTHETIC EVENT BATCH",
  });
  await assertCurrentEventIds(page, 32);
  await expectEventKindCounts(page, {
    accepted_activation: 10,
    charge_displacement: 2,
    net_health: 8,
    status_lifecycle: 12,
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
      recipientLabel: `id_${recipient}`,
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

  await captureBaseline(page, "crowded-teamfight-synthetic-mid-impact-960x600.png", {
    afterSettle: async () => {
      await assertCompactActiveCombatPriority(page, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    },
    commandPosts,
    expectedTransientCount: 8,
    // All eight authoritative NET outcomes remain visible. Compact active
    // combat suppresses eight lifecycle decorations that still cannot claim a
    // collision-free lane; four meaningful application cues remain visible
    // after lower-priority analysis decoration leaves the battlefield.
    expectedSuppressedLifecycleCount: 8,
    expectedSuppressedLifecycleIds: [
      "synthetic:crowded_teamfight:status-0",
      "synthetic:crowded_teamfight:status-1",
      "synthetic:crowded_teamfight:status-4",
      "synthetic:crowded_teamfight:status-6",
      "synthetic:crowded_teamfight:status-7",
      "synthetic:crowded_teamfight:status-8",
      "synthetic:crowded_teamfight:status-10",
      "synthetic:crowded_teamfight:status-11",
    ],
    logicalMs: MID_IMPACT_MS,
  });
});

test("synthetic POV fixture omits hidden agents and spatial endpoints", async ({
  page,
}) => {
  const servedFrame = /** @type {Record<string, any>} */ ({
    ...povFrame,
    preset: "debug",
  });
  expectPovPayloadRedacted(servedFrame);
  expect(
    /** @type {Array<{global_slot: number}>} */ (servedFrame.scene.agents).map(
      (agent) => agent.global_slot,
    ),
  ).toEqual([0, 1]);
  expect(servedFrame.scene.observer_visibility).toEqual([]);
  const rawActivation = /** @type {Array<Record<string, any>>} */ (
    servedFrame.event_batch.events
  ).find((event) => event.event_type === "accepted_activation");
  expect(rawActivation).toMatchObject({
    target_anchor: null,
    target_disclosure: "redacted",
    target_global_slot: null,
  });
  const commandPosts = await installSyntheticVisualCase(
    page,
    debuggerUrl,
    servedFrame,
    { viewport: DESKTOP_VIEWPORT },
  );
  await assertFrameIdentity(page, {
    scenario: "pov_redaction",
    simulatorStep: 1,
    transitionId: 1,
    view: "pov",
    preset: "debug",
    badge: /AGENT POV.*SYNTHETIC FIXTURE/,
  });
  await expect(page.locator("#scenario-description")).toContainText("SYNTHETIC:");
  await expectRosterSlots(page, [0, 1]);
  await assertHudStoryLabels(page, {
    pending: "PLAYBACK / INSPECTION ONLY",
    accepted: "SYNTHETIC EVENT BATCH",
  });
  await assertCurrentEventIds(page, 5);
  await expectEventKindCounts(page, {
    accepted_activation: 1,
    net_health: 1,
    status_lifecycle: 3,
  });
  await expectActivationPairs(page, [
    { tokenId: "basic_damage", source: 0, target: null },
  ]);
  await expectActivationRouteCount(page, 0);
  await expectNetCount(page, 1);
  await expectRosterStatuses(
    page,
    0,
    statuses([
      ["stun_rogue_poison", 3],
      ["slow_rogue_poison", 2],
      ["anti_heal_rogue_poison", 3],
    ]),
  );
  await expect(page.locator('#battlefield .agent[data-slot="5"]')).toHaveCount(0);
  await expect(page.locator('#roster .roster-row[data-slot="5"]')).toHaveCount(0);
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-protected-zone")).toHaveCount(2);
  await expect(page.locator("body")).not.toContainText("id_5");
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-target-slot]`),
  ).toHaveCount(0);
  await assertDurableDockFlags(page);
  await assertBoundedChoreography(page);
  await assertTransientSlotsAuthorized(page);

  await captureBaseline(page, "pov-redaction-synthetic-debug-mid-impact-1440x900.png", {
    commandPosts,
    expectedTransientCount: 1,
    logicalMs: MID_IMPACT_MS,
  });
});
