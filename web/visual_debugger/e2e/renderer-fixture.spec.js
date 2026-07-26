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
/** @type {Record<string, any>} */
let syntheticPovFrame = {};

test.beforeAll(async () => {
  const [started, fixture, povFixture] = await Promise.all([
    startDebugger(),
    loadRendererFixture("crowded_teamfight"),
    loadRendererFixture("pov_redaction"),
  ]);
  serverProcess = started.process;
  debuggerUrl = started.url;
  syntheticFrame = syntheticDebuggerFrame(fixture);
  syntheticPovFrame = syntheticDebuggerFrame(povFixture);
});

test.afterAll(async () => {
  const child = serverProcess;
  serverProcess = null;
  await stopDebugger(child);
});

test("test-only fixture interception installs an explicitly synthetic scene", async ({
  page,
}) => {
  const layoutStressFrame = structuredClone(syntheticFrame);
  layoutStressFrame.scene.agents[4].modifiers[0].multiplier = 123456.789;
  layoutStressFrame.scene.agents[0].statuses[0].duration = 123456789;
  layoutStressFrame.scene.agents[7].statuses.slice(0, 5).forEach(
    /**
     * @param {Record<string, any>} status
     * @param {number} index
     */
    (status, index) => {
      status.duration = index + 1;
    },
  );
  layoutStressFrame.scene.agents[2].ultimate_cooldown = 30;
  let commandRequests = 0;
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: layoutStressFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    commandRequests += 1;
    await route.abort("blockedbyclient");
  });

  await page.goto(debuggerUrl);

  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#terminal-badge")).toBeHidden();
  await expect(page.locator("#scenario-description")).toContainText("SYNTHETIC:");
  await expect(page.locator("#battlefield .agent")).toHaveCount(10);
  await expect(page.locator("#battlefield .agent-class-icon")).toHaveCount(10);
  await expect(page.locator("#battlefield .agent-class-letter")).toHaveCount(10);
  await expect(page.locator('#battlefield .agent[data-class="mage"]')).toHaveCount(2);
  await expect(
    page.locator('#battlefield .agent[data-team="team-b"] .agent-team-ring'),
  ).toHaveCount(5);
  const fallbackLetterOffsets = await page
    .locator("#battlefield .agent")
    .evaluateAll((agents) =>
      agents.map((agent) => {
        const body = agent.querySelector(".agent-body");
        const letter = agent.querySelector(".agent-class-letter");
        if (
          !(body instanceof SVGCircleElement) ||
          !(letter instanceof SVGTextElement)
        ) {
          throw new Error("Agent body and fallback letter must be measurable.");
        }
        return (
          Number(letter.getAttribute("y")) -
          (body.cy.baseVal.value + Math.min(body.r.baseVal.value * 0.5, 11))
        );
      }),
    );
  expect(fallbackLetterOffsets.every((offset) => Math.abs(offset - 2) <= 1e-3)).toBe(
    true,
  );
  await expect(page.locator("#battlefield .controlled-halo:not([hidden])")).toHaveCount(
    1,
  );
  await expect(
    page.locator("#battlefield .selected-reticle:not([hidden])"),
  ).toHaveCount(1);
  const expectedStatusOrder = /** @type {Array<{token_id: string}>} */ (
    syntheticFrame.scene.agents[0].statuses
  ).map((status) => status.token_id);
  expect(
    await page.locator("#battlefield .status-dock").count(),
  ).toBeGreaterThanOrEqual(2);
  expect(await page.locator("#battlefield .modifier-dock").count()).toBeGreaterThan(0);
  await expect(page.locator("#battlefield .cooldown-dock")).toHaveCount(1);
  const cooldown = page.locator(
    '#battlefield .cooldown-dock[data-slot="2"] .cooldown-cell',
  );
  await expect(cooldown).toHaveAttribute("data-class", "hunter");
  await expect(cooldown).toHaveAttribute("data-token-id", "hunter_trap");
  await expect(cooldown).toHaveAttribute("data-ticks", "30");
  await expect(cooldown).toHaveAttribute("data-numeric-layout", "compartments");
  await expect(cooldown).toHaveAttribute(
    "aria-label",
    /Hunter Trap.*cooldown, 30 ticks remaining, id_2/i,
  );
  await expect(cooldown.locator(".cooldown-cell__value")).toHaveText("30");
  await expect(cooldown.locator(".cooldown-cell__icon:not([hidden])")).toHaveCount(1);
  await expect(
    page.locator('#battlefield [data-layer="durable-status-modifier"]'),
  ).toHaveAttribute("data-suppressed-cooldown-slots", "");
  const legalityPillCount = await page.locator("#battlefield .legality-pill").count();
  expect([0, 2]).toContain(legalityPillCount);
  if (legalityPillCount === 0) {
    await expect(
      page.locator('#battlefield [data-layer="legality-cues"]'),
    ).toHaveAttribute("data-suppressed-slot", "7");
  } else {
    await expect(
      page.locator('#battlefield .legality-dock[data-slot="7"]'),
    ).toHaveCount(1);
    await expect(
      page.locator('#battlefield .legality-pill[data-lane="0"]'),
    ).toHaveAttribute("data-available", "true");
    await expect(
      page.locator('#battlefield .legality-pill[data-lane="1"]'),
    ).toHaveAttribute("data-available", "false");
    await expect(
      page.locator('#battlefield .legality-pill[data-lane="1"]'),
    ).toHaveAttribute("data-armed", "true");
    await expect(
      page.locator('#battlefield .legality-pill[data-lane="1"]'),
    ).toHaveAttribute("data-pair-legal", "false");
  }
  for (const slot of [0, 7]) {
    const dock = page.locator(`#battlefield .status-dock[data-slot="${slot}"]`);
    await expect(dock).toHaveAttribute("data-expanded", "true");
    await expect(dock).toHaveAttribute("data-visible-count", "9");
    await expect(dock).toHaveAttribute("data-hidden-count", "0");
    expect(
      await dock
        .locator(".status-cell")
        .evaluateAll((cells) =>
          cells.map((cell) => cell.getAttribute("data-token-id")),
        ),
    ).toEqual(expectedStatusOrder);
  }
  const dockConservation = await page
    .locator("#battlefield .status-dock")
    .evaluateAll((docks) =>
      docks.map((dock) => ({
        hidden: Number(dock.getAttribute("data-hidden-count")),
        slot: Number(dock.getAttribute("data-slot")),
        visible: Number(dock.getAttribute("data-visible-count")),
      })),
    );
  for (const dock of dockConservation) {
    expect(dock.visible + dock.hidden).toBe(
      /** @type {Array<{global_slot: number, statuses: unknown[]}>} */ (
        syntheticFrame.scene.agents
      ).find((agent) => agent.global_slot === dock.slot)?.statuses.length,
    );
  }
  for (const dock of dockConservation.filter(({ hidden }) => hidden > 0)) {
    await expect(
      page.locator(
        `#battlefield .status-dock[data-slot="${dock.slot}"] .status-overflow`,
      ),
    ).toHaveAttribute("aria-label", /hidden status cues.*duration/);
  }
  const suppressedStatusSlots = (
    (await page
      .locator('#battlefield [data-layer="durable-status-modifier"]')
      .getAttribute("data-suppressed-status-slots")) ?? ""
  )
    .split(",")
    .filter(Boolean)
    .map(Number);
  expect(
    new Set([...dockConservation.map(({ slot }) => slot), ...suppressedStatusSlots]),
  ).toEqual(
    new Set(
      /** @type {Array<{global_slot: number}>} */ (syntheticFrame.scene.agents).map(
        (agent) => agent.global_slot,
      ),
    ),
  );
  const geometryViolations = await page.evaluate(() => {
    const tolerance = 0.75;
    const battlefield = document.querySelector("#battlefield");
    if (!(battlefield instanceof SVGSVGElement)) {
      throw new Error("Battlefield SVG is unavailable.");
    }
    const viewport = battlefield.getBoundingClientRect();
    /**
     * @typedef {{bottom: number, left: number, right: number, top: number}} Bounds
     */
    /**
     * @param {DOMRect[]} rectangles
     * @returns {Bounds}
     */
    const union = (rectangles) => ({
      bottom: Math.max(...rectangles.map(({ bottom }) => bottom)),
      left: Math.min(...rectangles.map(({ left }) => left)),
      right: Math.max(...rectangles.map(({ right }) => right)),
      top: Math.min(...rectangles.map(({ top }) => top)),
    });
    /**
     * @param {Element} root
     * @param {string} selector
     * @returns {Bounds}
     */
    const boundsFor = (root, selector) => {
      const rectangles = [...root.querySelectorAll(selector)]
        .map((element) => element.getBoundingClientRect())
        .filter(({ height, width }) => height > 0 && width > 0);
      if (rectangles.length === 0) {
        throw new Error(`No measurable zones matched ${selector}.`);
      }
      return union(rectangles);
    };
    /**
     * @param {Bounds} first
     * @param {Bounds} second
     */
    const intersects = (first, second) =>
      Math.min(first.right, second.right) - Math.max(first.left, second.left) >
        tolerance &&
      Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top) >
        tolerance;
    const protectedBodies = [...battlefield.querySelectorAll(".agent")].map((agent) => {
      const visibleZones = [
        agent.querySelector(".agent-team-ring"),
        battlefield.querySelector(
          `.agent-selection[data-slot="${agent.getAttribute("data-slot")}"] .controlled-halo:not([hidden])`,
        ),
        battlefield.querySelector(
          `.agent-selection[data-slot="${agent.getAttribute("data-slot")}"] .selected-reticle:not([hidden])`,
        ),
      ].filter((element) => element instanceof SVGElement);
      return {
        bounds: union(visibleZones.map((element) => element.getBoundingClientRect())),
        slot: agent.getAttribute("data-slot"),
      };
    });
    const docks = [
      ...battlefield.querySelectorAll(".status-dock"),
      ...battlefield.querySelectorAll(".cooldown-dock"),
      ...battlefield.querySelectorAll(".modifier-dock"),
      ...battlefield.querySelectorAll(".legality-dock"),
    ].map((dock) => ({
      bounds: boundsFor(
        dock,
        ".status-cell__box, .cooldown-cell__box, .modifier-cell__box, .legality-pill__box",
      ),
      collisionFree: dock.getAttribute("data-collision-free"),
      kind: dock.getAttribute("class"),
      slot: dock.getAttribute("data-slot"),
    }));
    const identities = [...battlefield.querySelectorAll(".agent-id-tag")]
      .filter((tag) => Number.parseFloat(getComputedStyle(tag).opacity) > 0.5)
      .map((tag) => ({
        bounds: tag.getBoundingClientRect(),
        slot: tag.parentElement?.getAttribute("data-slot"),
      }));
    const violations = [];
    for (const dock of docks) {
      if (dock.collisionFree !== "true") {
        violations.push(`${dock.kind} id_${dock.slot} reports a collision`);
      }
      if (
        dock.bounds.left < viewport.left - tolerance ||
        dock.bounds.top < viewport.top - tolerance ||
        dock.bounds.right > viewport.right + tolerance ||
        dock.bounds.bottom > viewport.bottom + tolerance
      ) {
        violations.push(`${dock.kind} id_${dock.slot} escapes the viewport`);
      }
      for (const body of protectedBodies) {
        if (intersects(dock.bounds, body.bounds)) {
          violations.push(`${dock.kind} id_${dock.slot} overlaps body id_${body.slot}`);
        }
      }
      for (const identity of identities) {
        if (intersects(dock.bounds, identity.bounds)) {
          violations.push(
            `${dock.kind} id_${dock.slot} overlaps identity id_${identity.slot}`,
          );
        }
      }
    }
    for (let index = 0; index < docks.length; index += 1) {
      for (let other = index + 1; other < docks.length; other += 1) {
        if (intersects(docks[index].bounds, docks[other].bounds)) {
          violations.push(
            `${docks[index].kind} id_${docks[index].slot} overlaps ${docks[other].kind} id_${docks[other].slot}`,
          );
        }
      }
    }
    return violations;
  });
  expect(geometryViolations).toEqual([]);
  await expect(page.locator("#battlefield path.pending-route")).toHaveAttribute(
    "data-route-kind",
    /curve|local_arc/,
  );
  await expect(page.locator('#battlefield .agent[data-slot="0"]')).toHaveAttribute(
    "aria-label",
    /id_0, Mage, Team A, health 82 of 100, alive, controlled actor/,
  );
  await expect(page.locator('#battlefield .agent[data-slot="7"]')).toHaveAttribute(
    "aria-label",
    /selected target/,
  );
  const teamStrokePatterns = await page.evaluate(() => {
    const teamA = document.querySelector(
      '#battlefield .agent[data-team="team-a"] .agent-team-ring',
    );
    const teamB = document.querySelector(
      '#battlefield .agent[data-team="team-b"] .agent-team-ring',
    );
    if (!teamA || !teamB) {
      throw new Error("Both team perimeter styles must be present.");
    }
    return {
      teamA: getComputedStyle(teamA).strokeDasharray,
      teamB: getComputedStyle(teamB).strokeDasharray,
    };
  });
  expect(teamStrokePatterns.teamA).toBe("none");
  expect(teamStrokePatterns.teamB).toBe("none");
  await expect(
    page.locator('#battlefield .agent[data-team="team-b"] .agent-team-marker:visible'),
  ).toHaveCount(5);
  await expect(
    page.locator('#battlefield .agent[data-team="team-a"] .agent-team-marker:visible'),
  ).toHaveCount(0);
  const hunterSurfaceColors = await page.evaluate(() => {
    const agent = document.querySelector('#battlefield .agent[data-class="hunter"]');
    const status = document.querySelector(
      '#battlefield .status-cell[data-source-class="hunter"]',
    );
    const cooldownIcon = document.querySelector(
      '#battlefield .cooldown-cell[data-class="hunter"] .cooldown-cell__icon',
    );
    if (!agent || !status || !cooldownIcon) {
      throw new Error("Hunter-owned browser surfaces are incomplete.");
    }
    return {
      agent: getComputedStyle(agent).color,
      cooldownIcon: getComputedStyle(cooldownIcon).color,
      customProperty: getComputedStyle(document.documentElement)
        .getPropertyValue("--class-hunter")
        .trim()
        .toUpperCase(),
      status: getComputedStyle(status).color,
    };
  });
  expect(hunterSurfaceColors).toEqual({
    agent: "rgb(132, 204, 22)",
    cooldownIcon: "rgb(132, 204, 22)",
    customProperty: "#84CC16",
    status: "rgb(132, 204, 22)",
  });
  expect(
    await page
      .locator("#battlefield .aura-field")
      .evaluateAll((auras) =>
        auras.every((aura) => getComputedStyle(aura).stroke === "none"),
      ),
  ).toBe(true);
  const rangeStrokePatterns = await page
    .locator("#battlefield .range-ring")
    .evaluateAll((ranges) =>
      Object.fromEntries(
        ranges.map((range) => [
          range.getAttribute("data-kind"),
          {
            classKey: range.getAttribute("data-class"),
            dash: Array.from(
              getComputedStyle(range).strokeDasharray.matchAll(/\d+(?:\.\d+)?/g),
              (match) => Number(match[0]),
            ),
            stroke: getComputedStyle(range).stroke,
          },
        ]),
      ),
    );
  expect(rangeStrokePatterns.observation.stroke).toBe("rgb(244, 247, 251)");
  expect(rangeStrokePatterns.observation.dash).toEqual([1, 5]);
  expect(rangeStrokePatterns.basic.classKey).toBe("mage");
  expect(rangeStrokePatterns.basic.stroke).toBe("rgb(34, 211, 238)");
  expect(rangeStrokePatterns.basic.dash).toEqual([8, 5]);
  expect(rangeStrokePatterns.ultimate.stroke).toBe("rgb(167, 139, 250)");
  expect(rangeStrokePatterns.ultimate.dash).toEqual([10, 4, 2, 4]);
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  });
  const numericDockInternalOverlaps = await page
    .locator(
      "#battlefield .status-cell, #battlefield .cooldown-cell, #battlefield .modifier-cell",
    )
    .evaluateAll((cells) =>
      cells.flatMap((cell) => {
        const kind = cell.classList.contains("status-cell")
          ? "status"
          : cell.classList.contains("cooldown-cell")
            ? "cooldown"
            : "modifier";
        const box = cell.querySelector(`.${kind}-cell__box`);
        const icon = cell.querySelector(`.${kind}-cell__icon`);
        const value = cell.querySelector(`.${kind}-cell__value`);
        if (
          !(box instanceof SVGGraphicsElement) ||
          !(icon instanceof SVGGraphicsElement) ||
          !(value instanceof SVGGraphicsElement)
        ) {
          return [`missing measurable ${kind} cue`];
        }
        const boxBounds = box.getBoundingClientRect();
        const iconBounds = icon.getBoundingClientRect();
        const valueBounds = value.getBoundingClientRect();
        const violations = [];
        if (iconBounds.width > 0 && iconBounds.right > valueBounds.left) {
          violations.push("icon/value overlap");
        }
        if (
          valueBounds.left < boxBounds.left - 0.5 ||
          valueBounds.right > boxBounds.right + 0.5
        ) {
          violations.push("value escapes numeric dock pill");
        }
        return violations;
      }),
    );
  expect(numericDockInternalOverlaps).toEqual([]);
  const supportedStatusCompartments = await page
    .locator(
      '#battlefield .status-cell[data-slot="7"][data-index="0"], ' +
        '#battlefield .status-cell[data-slot="7"][data-index="1"], ' +
        '#battlefield .status-cell[data-slot="7"][data-index="2"], ' +
        '#battlefield .status-cell[data-slot="7"][data-index="3"], ' +
        '#battlefield .status-cell[data-slot="7"][data-index="4"]',
    )
    .evaluateAll((cells) =>
      cells.map((cell) => {
        const icon = cell.querySelector(".status-cell__icon");
        const value = cell.querySelector(".status-cell__value");
        const iconCompartment = cell.querySelector(".status-cell__icon-compartment");
        const valueCompartment = cell.querySelector(".status-cell__value-compartment");
        if (
          !(icon instanceof SVGGraphicsElement) ||
          !(value instanceof SVGGraphicsElement) ||
          !(iconCompartment instanceof SVGGraphicsElement) ||
          !(valueCompartment instanceof SVGGraphicsElement)
        ) {
          throw new Error("Supported status compartments are incomplete.");
        }
        const iconBounds = icon.getBoundingClientRect();
        const valueBounds = value.getBoundingClientRect();
        const iconReserved = iconCompartment.getBoundingClientRect();
        const valueReserved = valueCompartment.getBoundingClientRect();
        return {
          fallback: cell.hasAttribute("data-numeric-fallback"),
          glyphVisible: !icon.hasAttribute("hidden") && iconBounds.width > 0,
          iconInside:
            iconBounds.left >= iconReserved.left - 0.5 &&
            iconBounds.right <= iconReserved.right + 0.5,
          layout: cell.getAttribute("data-numeric-layout"),
          separated: iconBounds.right + 2 <= valueBounds.left,
          value: value.textContent,
          valueInside:
            valueBounds.left >= valueReserved.left - 0.5 &&
            valueBounds.right <= valueReserved.right + 0.5,
        };
      }),
    );
  expect(supportedStatusCompartments).toEqual(
    ["1", "2", "3", "4", "5"].map((value) => ({
      fallback: false,
      glyphVisible: true,
      iconInside: true,
      layout: "compartments",
      separated: true,
      value,
      valueInside: true,
    })),
  );
  const overflowPalette = await page
    .locator("#battlefield .status-overflow, #battlefield .modifier-overflow")
    .evaluateAll((overflows) =>
      overflows.map((overflow) => {
        const box = overflow.querySelector(".status-cell__box, .modifier-cell__box");
        const label = overflow.querySelector(
          ".status-overflow__label, .modifier-overflow__label",
        );
        if (!box || !label) {
          throw new Error("Overflow cue is missing its monochrome surfaces.");
        }
        return {
          box: getComputedStyle(box).stroke,
          color: getComputedStyle(overflow).color,
          label: getComputedStyle(label).fill,
        };
      }),
    );
  expect(overflowPalette.length).toBeGreaterThan(0);
  expect(
    overflowPalette.every(
      ({ box, color, label }) =>
        box === "rgb(154, 167, 184)" &&
        color === "rgb(154, 167, 184)" &&
        label === "rgb(154, 167, 184)",
    ),
  ).toBe(true);
  await expect(
    page.locator(
      '#battlefield .modifier-cell[data-slot="4"][data-index="0"][data-icon-suppressed="true"]',
    ),
  ).toHaveCount(1);
  const stressedModifier = page.locator(
    '#battlefield .modifier-cell[data-slot="4"][data-index="0"]',
  );
  await expect(stressedModifier.locator(".modifier-cell__value")).toHaveText(
    "×123456.79",
  );
  await expect(stressedModifier).toHaveAttribute("aria-label", /multiplier 123456\.79/);
  await expect(stressedModifier).toHaveAttribute("data-numeric-fallback", "true");
  expect(await stressedModifier.evaluate((cell) => getComputedStyle(cell).color)).toBe(
    "rgb(154, 167, 184)",
  );
  await expect(stressedModifier.locator("title")).toHaveText(/multiplier 123456\.79/);
  const stressedStatus = page.locator(
    '#battlefield .status-cell[data-slot="0"][data-index="0"]',
  );
  await expect(stressedStatus).toHaveAttribute("data-icon-suppressed", "true");
  await expect(stressedStatus).toHaveAttribute("data-numeric-fallback", "true");
  await expect(stressedStatus.locator(".status-cell__value")).toHaveText("123456789");
  await expect(stressedStatus).toHaveAttribute("aria-label", /duration 123456789/);
  expect(await stressedStatus.evaluate((cell) => getComputedStyle(cell).color)).toBe(
    "rgb(154, 167, 184)",
  );
  const visibleIdentityTags = await page
    .locator("#battlefield .agent-id-tag")
    .evaluateAll(
      (tags) =>
        tags.filter((tag) => Number.parseFloat(getComputedStyle(tag).opacity) > 0.5)
          .length,
    );
  expect(visibleIdentityTags).toBeLessThanOrEqual(1);
  expect(
    await page.evaluate(() => document.fonts.check('16px "Atkinson Hyperlegible"')),
  ).toBe(true);
  await expect(page.locator("#roster .roster-row")).toHaveCount(10);
  await expect(page.locator("#event-count")).not.toHaveText("0");
  expect(commandRequests).toBe(0);
});

test("structured HUD keeps exact roster, intent, and accepted-result facts distinct", async ({
  page,
}) => {
  const transitionId = Number(syntheticFrame.transition_id);
  const rejectedEvent = {
    event_type: "rejected_action",
    event_id: "synthetic:crowded_teamfight:rejected-panel-proof",
    transition_id: transitionId,
    actor_global_slot: 0,
    component: "combat",
    actor_anchor: [6.4, 5.7],
    target_global_slot: 7,
    target_anchor: [7.15, 6.15],
    target_disclosure: "public",
    lane: 1,
    movement_mask_value: true,
    pair_mask_value: false,
  };
  /** @type {Record<string, any>} */
  let structuredFrame = {
    ...syntheticFrame,
    event_batch: {
      ...syntheticFrame.event_batch,
      events: [...syntheticFrame.event_batch.events, rejectedEvent],
    },
    hud: {
      roster_global_slots: syntheticFrame.scene.agents.map(
        /** @param {{global_slot: number}} agent */ (agent) => agent.global_slot,
      ),
      controlled_global_slot: 0,
      selected_global_slot: 7,
      pending_action: {
        label: "PENDING / WILL SUBMIT",
        actor_global_slot: 0,
        move_action: 0,
        target_action: 8,
        armed_lane: 1,
        arm_origin: "explicit",
        target: { disclosure: "public", global_slot: 7 },
        movement_mask_value: true,
        pair_mask_value: false,
        summary: "Stay + Burst → id_7",
      },
      latest_transition: {
        label: "LATEST ACCEPTED RESULT",
        transition_id: transitionId,
        submission_kind: "interactive",
        actors: [
          {
            actor_global_slot: 0,
            submitted: {
              move_action: 0,
              target_action: 8,
              use_ultimate_action: 1,
              target: { disclosure: "public", global_slot: 7 },
              summary: "Stay + Burst → id_7",
            },
            accepted: {
              move_action: 0,
              target_action: 0,
              use_ultimate_action: 0,
              target: { disclosure: "target_none", global_slot: null },
              summary: "Stay + NO COMBAT",
            },
            movement_mask_value: true,
            pair_mask_value: false,
            movement_accepted: true,
            combat_result: "rejected",
          },
        ],
      },
      candidate_legalities: [],
      diagnostics: [],
    },
  };
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: structuredFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    const request = route.request().postDataJSON();
    if (request.command?.command_type !== "set_preset") {
      await route.abort("blockedbyclient");
      return;
    }
    structuredFrame = {
      ...structuredFrame,
      preset: request.command.preset,
      revision: Number(structuredFrame.revision) + 1,
    };
    await route.fulfill({
      contentType: "application/json",
      json: {
        schema_version: 1,
        result: "applied",
        frame: structuredFrame,
        notice: null,
      },
      status: 200,
    });
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#roster .roster-team")).toHaveCount(2);
  await expect(
    page.locator('#roster .roster-team[data-team="team-a"] .roster-row'),
  ).toHaveCount(5);
  await expect(
    page.locator('#roster .roster-team[data-team="team-b"] .roster-row'),
  ).toHaveCount(5);
  await expect(page.locator("#roster .roster-row")).toHaveCount(10);
  const agentZero = syntheticFrame.scene.agents.find(
    /** @param {{global_slot: number}} agent */ (agent) => agent.global_slot === 0,
  );
  if (!agentZero) {
    throw new Error("Synthetic crowded fixture is missing id_0.");
  }
  await expect(
    page.locator('#roster .roster-row[data-slot="0"] .roster-fact-token--status'),
  ).toHaveCount(agentZero.statuses.length);
  expect(
    await page
      .locator('#roster .roster-row[data-slot="0"] .roster-fact-token--status')
      .evaluateAll((tokens) =>
        tokens.map((token) => ({
          duration: Number(token.getAttribute("data-duration")),
          tokenId: token.getAttribute("data-token-id"),
        })),
      ),
  ).toEqual(
    agentZero.statuses.map(
      /** @param {{duration: number, token_id: string}} status */ (status) => ({
        duration: status.duration,
        tokenId: status.token_id,
      }),
    ),
  );
  await expect(
    page.locator('#roster .roster-row[data-slot="0"] .roster-fact-token--modifier'),
  ).toHaveCount(agentZero.modifiers.length);
  expect(
    await page
      .locator('#roster .roster-row[data-slot="0"] .roster-fact-token--modifier')
      .evaluateAll((tokens) =>
        tokens.map((token) => ({
          multiplier: Number(token.getAttribute("data-multiplier")),
          tokenId: token.getAttribute("data-token-id"),
        })),
      ),
  ).toEqual(
    agentZero.modifiers.map(
      /** @param {{multiplier: number, token_id: string}} modifier */ (modifier) => ({
        multiplier: modifier.multiplier,
        tokenId: modifier.token_id,
      }),
    ),
  );
  await expect(
    page.locator('.comparison-agent[data-role="controlled"]'),
  ).toHaveAttribute("data-slot", "0");
  await expect(page.locator('.comparison-agent[data-role="selected"]')).toHaveAttribute(
    "data-slot",
    "7",
  );
  await expect(page.locator(".selected-legality")).toContainText("Basic lane");
  await expect(page.locator(".selected-legality")).toContainText("Ultimate lane");
  await expect(page.locator("#pending-card .action-card__label")).toHaveText(
    "PENDING / WILL SUBMIT",
  );
  await expect(page.locator("#pending-card")).toContainText("Stay + Burst → id_7");
  await expect(page.locator("#accepted-card .action-card__label")).toHaveText(
    "LATEST ACCEPTED RESULT",
  );
  await expect(
    page.locator('#accepted-card .action-tuple[data-kind="submitted"]'),
  ).toContainText("Stay + Burst → id_7");
  await expect(
    page.locator('#accepted-card .action-tuple[data-kind="accepted"]'),
  ).toContainText("Stay + NO COMBAT");
  await expect(page.locator("#accepted-card")).toContainText("Combat result");
  await expect(page.locator("#accepted-card")).toContainText("Rejected");
  await expect(page.locator("#accepted-announcement")).toContainText(
    `Transition ${transitionId}`,
  );
  await expect(page.locator("#event-count")).toHaveText(
    String(structuredFrame.event_batch.events.length),
  );
  expect(
    await page.locator("#event-feed .event-item").evaluateAll((items) =>
      items.map((item) => ({
        eventId: item.getAttribute("data-event-id"),
        eventType: item.getAttribute("data-event-type"),
      })),
    ),
  ).toEqual(
    structuredFrame.event_batch.events.map(
      /** @param {{event_id: string, event_type: string}} event */ (event) => ({
        eventId: event.event_id,
        eventType: event.event_type,
      }),
    ),
  );
  for (const activation of await page
    .locator('#event-feed .event-item[data-event-type="accepted_activation"]')
    .all()) {
    await expect(activation).not.toContainText(/NET|HP|amount/i);
  }
  for (const netHealth of await page
    .locator('#event-feed .event-item[data-event-type="net_health"]')
    .all()) {
    await expect(netHealth).toContainText("NET");
    await expect(netHealth).toHaveAttribute("data-recipient-slot", /\d+/);
    await expect(netHealth).not.toHaveAttribute("data-source-slot", /.+/);
  }
  const rejected = page.locator(
    '#event-feed .event-item[data-event-type="rejected_action"]',
  );
  await expect(rejected).toHaveAttribute("data-actor-slot", "0");
  await expect(rejected).toHaveAttribute("data-target-slot", "7");
  await expect(rejected).toContainText("movement mask 1");
  await expect(rejected).toContainText("pair mask 0");
  await expect(rejected).not.toContainText(/range|cooldown|line of sight|LOS/);
  await expect(page.locator("#visual-key")).toContainText("never a per-source amount");
  await expect(page.locator("#visual-key")).not.toHaveAttribute("open", "");
  await expect(page.locator("details.diagnostics")).not.toHaveAttribute("open", "");

  const firstEvent = page.locator("#event-feed .event-item").first();
  await firstEvent.evaluate((element) => {
    element.setAttribute("data-retained-probe", "event");
  });
  await page.locator("details.diagnostics").evaluate((details) => {
    details.setAttribute("open", "");
  });
  await page.locator("#accepted-announcement").evaluate((element) => {
    element.setAttribute("data-mutation-count", "0");
    const observer = new MutationObserver(() => {
      const count = Number(element.getAttribute("data-mutation-count") ?? "0");
      element.setAttribute("data-mutation-count", String(count + 1));
    });
    observer.observe(element, { childList: true, characterData: true, subtree: true });
  });
  await page.locator("#preset-select").selectOption("debug");
  await expect(page.locator("#revision-value")).toHaveText("1");
  await expect(firstEvent).toHaveAttribute("data-retained-probe", "event");
  await expect(page.locator("details.diagnostics")).toHaveAttribute("open", "");
  await expect(page.locator("#accepted-announcement")).toHaveAttribute(
    "data-mutation-count",
    "0",
  );
});

test("presets omit irrelevant DOM while retaining canonical facts and audience", async ({
  page,
}) => {
  /** @type {Record<string, any>} */
  let servedFrame = {
    ...syntheticFrame,
    preset: "analysis",
    scene: {
      ...syntheticFrame.scene,
      agents: syntheticFrame.scene.agents.map(
        /**
         * @param {Record<string, any>} agent
         * @param {number} index
         */
        (agent, index) => ({
          ...agent,
          ultimate_cooldown: index === 2 ? 30 : 0,
        }),
      ),
      selected_legality: {
        ...syntheticFrame.scene.selected_legality,
        armed_lane: null,
        armed_pair_legal: false,
      },
    },
  };
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: servedFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    const request = route.request().postDataJSON();
    if (request.command?.command_type !== "set_preset") {
      await route.abort("blockedbyclient");
      return;
    }
    servedFrame = {
      ...servedFrame,
      preset: request.command.preset,
      revision: Number(servedFrame.revision) + 1,
    };
    await route.fulfill({
      contentType: "application/json",
      json: {
        schema_version: 1,
        result: "applied",
        frame: servedFrame,
        notice: null,
      },
      status: 200,
    });
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("html")).toHaveAttribute("data-preset", "analysis");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  await expect(page.locator("#audience-badge")).toContainText(
    "PRIVILEGED RESEARCHER VIEW",
  );
  await expect(page.locator("#battlefield .agent")).toHaveCount(10);
  expect(await page.locator("#battlefield .status-cell").count()).toBeGreaterThan(0);
  await expect(page.locator("#battlefield .cooldown-cell")).toHaveCount(1);
  expect(await page.locator("#battlefield .aura-field").count()).toBeGreaterThan(0);
  expect(await page.locator("#battlefield .range-ring").count()).toBeGreaterThan(0);
  expect(await page.locator("#battlefield .modifier-cell").count()).toBeGreaterThan(0);
  await expect(page.locator(".selected-legality")).toHaveCount(1);
  await expect(
    page
      .locator(".selected-legality .fact")
      .filter({ hasText: "Armed pair" })
      .locator("strong"),
  ).toHaveText("Not applicable");
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-protected-zone")).toHaveCount(0);
  await expect(page.locator(".candidate-legality-row")).toHaveCount(0);
  await expect(page.locator("#diagnostics-card")).not.toContainText(
    "candidate_legalities",
  );

  await page.locator("#preset-select").selectOption("debug");
  await expect(page.locator("html")).toHaveAttribute("data-preset", "debug");
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(10);
  await expect(page.locator("#battlefield .debug-protected-zone")).toHaveCount(10);
  await expect(page.locator("#battlefield .cooldown-cell")).toHaveCount(1);
  for (const aura of await page.locator("#battlefield .aura-field").all()) {
    await expect(aura).toHaveAttribute("data-source-slot", /\d+/);
    await expect(aura).not.toHaveAttribute("data-multiplier", /.+/);
  }
  for (const modifier of await page.locator("#battlefield .modifier-cell").all()) {
    await expect(modifier).not.toHaveAttribute("data-source-slot", /.+/);
    await expect(modifier).toHaveAttribute("aria-label", /multiplier/);
  }

  await page.locator("#preset-select").selectOption("presentation");
  await expect(page.locator("html")).toHaveAttribute("data-preset", "presentation");
  await expect(page.locator("#audience-badge")).toContainText(
    "PRIVILEGED RESEARCHER VIEW",
  );
  await expect(page.locator("#battlefield .agent")).toHaveCount(10);
  expect(await page.locator("#battlefield .status-cell").count()).toBeGreaterThan(0);
  await expect(page.locator("#battlefield .cooldown-cell")).toHaveCount(1);
  await expect(page.locator("#battlefield .aura-field")).toHaveCount(0);
  await expect(page.locator("#battlefield .range-ring")).toHaveCount(0);
  await expect(page.locator("#battlefield .modifier-cell")).toHaveCount(0);
  await expect(page.locator("#battlefield .legality-pill")).toHaveCount(0);
  await expect(page.locator(".selected-legality")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-protected-zone")).toHaveCount(0);
  await expect(page.locator(".candidate-legality-row")).toHaveCount(0);
  await expect(page.locator("#roster")).toHaveAttribute("data-compact", "true");
  await expect(page.locator("#roster .roster-row")).toHaveCount(10);
  await expect(page.locator("#roster .roster-fact-token")).toHaveCount(0);
  await expect(page.locator("#selection-card").locator("..")).toBeHidden();
  await expect(page.locator("#diagnostics-card").locator("..")).toBeHidden();
  await expect(page.locator("#pending-card")).toBeVisible();
  await expect(page.locator("#accepted-card")).toBeVisible();
  await expect(page.locator("#event-feed")).toBeVisible();
});

test("debug POV omits hidden agents and researcher-only visibility DOM", async ({
  page,
}) => {
  const servedFrame = { ...syntheticPovFrame, preset: "debug" };
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: servedFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    await route.abort("blockedbyclient");
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("html")).toHaveAttribute("data-preset", "debug");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  await expect(page.locator("#audience-badge")).toContainText("AGENT POV");
  await expect(page.locator("#battlefield .agent")).toHaveCount(2);
  await expect(page.locator("#roster .roster-row")).toHaveCount(2);
  await expect(page.locator('#battlefield .agent[data-slot="5"]')).toHaveCount(0);
  await expect(page.locator('#roster .roster-row[data-slot="5"]')).toHaveCount(0);
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-protected-zone")).toHaveCount(2);
  await expect(page.locator("body")).not.toContainText("id_5");
  expect(
    await page.evaluate(() => {
      const authorized = new Set(
        [...document.querySelectorAll("#roster .roster-row")].map((row) =>
          Number(row.getAttribute("data-slot")),
        ),
      );
      return [...document.querySelectorAll("*")]
        .flatMap((element) =>
          [...element.attributes]
            .filter((attribute) => /^data-(?:slot|.*-slot)$/.test(attribute.name))
            .map((attribute) => Number(attribute.value)),
        )
        .filter((slot) => Number.isInteger(slot) && !authorized.has(slot));
    }),
  ).toEqual([]);
});
