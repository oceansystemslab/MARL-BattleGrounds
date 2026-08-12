import { expect, test } from "@playwright/test";

import { formatDisplayNumber } from "../src/display.js";
import {
  CHOREOGRAPHY_ROOT,
  installWaapiAutopause,
  pauseAtLogicalTime,
} from "./support/choreography.js";
import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  loadCatalogPropagationFixture,
  loadRendererFixture,
  syntheticDebuggerPresentationFrame,
  syntheticDebuggerWireFrame,
} from "./support/renderer-fixture.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";
/** @type {Record<string, any>} */
let syntheticFrame = {};
/** @type {Record<string, any>} */
let syntheticWireFrame = {};
/** @type {Record<string, any>} */
let syntheticPovWireFrame = {};
/** @type {Record<string, any>} */
let canonicalGrammarFrame = {};
/** @type {Record<string, any>} */
let canonicalGrammarWireFrame = {};
/** @type {{live_frame: Record<string, any>, expected: Record<string, number>}} */
let catalogPropagationFixture = { live_frame: {}, expected: {} };

/**
 * Replace one catalog-owned display fragment while preserving every stable
 * word around it for the semantic snapshot. A missing or repeated fragment is
 * evidence drift, not permission to normalize an arbitrary string.
 *
 * @param {string | undefined} value
 * @param {string} fragment
 * @param {string} placeholder
 * @param {string} label
 */
function replaceCatalogFragmentOnce(value, fragment, placeholder, label) {
  if (typeof value !== "string") {
    throw new Error(`${label} has no semantic value to snapshot.`);
  }
  const first = value.indexOf(fragment);
  const repeated = first >= 0 && value.indexOf(fragment, first + fragment.length) >= 0;
  if (first < 0 || repeated) {
    throw new Error(
      `${label} must contain exactly one catalog fragment ${JSON.stringify(fragment)}.`,
    );
  }
  return `${value.slice(0, first)}${placeholder}${value.slice(first + fragment.length)}`;
}

/**
 * Keep deliberately extreme synthetic presentation values within the same
 * serialized mechanics authority that the strict browser boundary validates.
 *
 * @param {Record<string, any>} scene
 * @param {number} agentIndex
 * @param {number} statusIndex
 * @param {number} duration
 */
function authorizeSyntheticStatusDuration(scene, agentIndex, statusIndex, duration) {
  const status = scene.agents[agentIndex]?.statuses?.[statusIndex];
  const mechanic = scene.class_mechanics
    .flatMap(/** @param {Record<string, any>} row */ (row) => row.status_mechanics)
    .find(
      /** @param {Record<string, any>} row */ (row) =>
        row.status_channel === status?.status_channel,
    );
  if (!status || !mechanic) {
    throw new Error("Synthetic status stress value has no serialized mechanic.");
  }
  status.remaining_duration = duration;
  mechanic.duration_steps = Math.max(mechanic.duration_steps, duration);
}

/**
 * @param {Record<string, any>} scene
 * @param {number} agentIndex
 * @param {number} ticks
 */
function authorizeSyntheticCooldown(scene, agentIndex, ticks) {
  const agent = scene.agents[agentIndex];
  const mechanic = scene.class_mechanics.find(
    /** @param {Record<string, any>} row */
    (row) => row.class_id === agent?.class_id,
  );
  if (!agent || !mechanic) {
    throw new Error("Synthetic cooldown stress value has no serialized mechanic.");
  }
  agent.ultimate_cooldown_remaining = ticks;
  mechanic.ultimate_cooldown_steps = Math.max(mechanic.ultimate_cooldown_steps, ticks);
}

/**
 * @param {Record<string, any>} scene
 * @param {number} agentIndex
 * @param {number} modifierIndex
 * @param {number} multiplier
 */
function authorizeSyntheticAuraModifier(scene, agentIndex, modifierIndex, multiplier) {
  const modifier = scene.agents[agentIndex]?.aura_modifiers?.[modifierIndex];
  const mechanic = scene.class_mechanics
    .flatMap(/** @param {Record<string, any>} row */ (row) => row.aura_mechanics)
    .find(
      /** @param {Record<string, any>} row */ (row) =>
        row.aura_id === modifier?.aura_id,
    );
  if (!modifier || !mechanic || mechanic.clamp_kind !== "ceiling") {
    throw new Error("Synthetic aura stress value has no ceiling mechanic.");
  }
  modifier.multiplier = multiplier;
  mechanic.clamp_value = multiplier;
  for (const field of scene.aura_fields.filter(
    /** @param {Record<string, any>} row */
    (row) => row.aura_id === modifier.aura_id,
  )) {
    field.clamp_value = multiplier;
  }
}

test.beforeAll(async () => {
  const [started, fixture, povFixture, grammarFixture, catalogFixture] =
    await Promise.all([
      startDebugger(),
      loadRendererFixture("crowded_teamfight"),
      loadRendererFixture("pov_redaction"),
      loadRendererFixture("canonical_event_vocabulary"),
      loadCatalogPropagationFixture(),
    ]);
  serverProcess = started.process;
  debuggerUrl = started.url;
  syntheticFrame = syntheticDebuggerPresentationFrame(fixture);
  syntheticWireFrame = syntheticDebuggerWireFrame(fixture);
  syntheticPovWireFrame = syntheticDebuggerWireFrame(povFixture);
  canonicalGrammarFrame = syntheticDebuggerPresentationFrame(grammarFixture);
  canonicalGrammarWireFrame = syntheticDebuggerWireFrame(grammarFixture);
  catalogPropagationFixture = catalogFixture;
  expect(syntheticFrame).toMatchObject({
    schema_version: 2,
    frame_kind: "researcher_live_debugger",
  });
  expect(syntheticWireFrame).not.toHaveProperty("scene");
  expect(syntheticWireFrame).not.toHaveProperty("event_batch");
  expect(syntheticWireFrame).not.toHaveProperty("simulator_step");
  expect(syntheticWireFrame).not.toHaveProperty("transition_id");
  expect(syntheticFrame).toHaveProperty("scene");
  expect(syntheticFrame).toHaveProperty("event_batch");
  expect(syntheticWireFrame.projection).toMatchObject({ schema_version: 2 });
});

test.afterAll(async () => {
  const child = serverProcess;
  serverProcess = null;
  await stopDebugger(child);
});

test("test-only fixture interception installs an explicitly synthetic scene", async ({
  page,
}) => {
  const layoutStressFrame = structuredClone(syntheticWireFrame);
  const layoutStressScene = layoutStressFrame.projection.scene;
  authorizeSyntheticAuraModifier(layoutStressScene, 4, 0, 123456.789);
  authorizeSyntheticStatusDuration(layoutStressScene, 0, 0, 123456789);
  layoutStressScene.agents[7].statuses.slice(0, 5).forEach(
    /**
     * @param {Record<string, any>} _status
     * @param {number} index
     */
    (_status, index) => {
      authorizeSyntheticStatusDuration(layoutStressScene, 7, index, index + 1);
    },
  );
  authorizeSyntheticCooldown(layoutStressScene, 2, 30);
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
  const skipPresentation = page.locator("#motion-skip-button");
  if (await skipPresentation.isEnabled()) {
    await skipPresentation.click();
    await expect(
      page.locator(
        '#battlefield [data-layer="transient-events"] > .combat-choreography',
      ),
    ).toHaveAttribute("data-state", "settled");
  }
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
    /Accepted Hunter Trap activation cooldown.*30 ticks remaining.*Agent ID 2/i,
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
  await expect(page.locator('#battlefield .agent[data-slot="0"]')).toHaveAttribute(
    "aria-label",
    /Agent ID 0, Mage, Team A, health 82 of 100, alive, controlled actor/,
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
  await expect(stressedModifier.locator(".modifier-cell__value")).toHaveText("×123K");
  await expect(stressedModifier).toHaveAttribute("data-multiplier", "123456.789");
  await expect(stressedModifier).toHaveAttribute("aria-label", /multiplier 123456\.79/);
  await expect(stressedModifier).toHaveAttribute("data-numeric-fallback", "true");
  await expect(stressedModifier).toHaveAttribute(
    "data-visible-value-abbreviated",
    "true",
  );
  expect(await stressedModifier.evaluate((cell) => getComputedStyle(cell).color)).toBe(
    "rgb(154, 167, 184)",
  );
  await expect(stressedModifier).toHaveAttribute("data-tooltip-owner", "");
  const stressedStatus = page.locator(
    '#battlefield .status-cell[data-slot="0"][data-index="0"]',
  );
  await expect(stressedStatus).toHaveAttribute("data-icon-suppressed", "true");
  await expect(stressedStatus).toHaveAttribute("data-numeric-fallback", "true");
  await expect(stressedStatus).toHaveAttribute(
    "data-numeric-layout",
    "compact-measured-fallback",
  );
  await expect(stressedStatus).toHaveAttribute(
    "data-visible-value-abbreviated",
    "true",
  );
  await expect(stressedStatus).toHaveAttribute("data-duration", "123456789");
  await expect(stressedStatus.locator(".status-cell__value")).toHaveText("123M");
  await expect(stressedStatus).toHaveAttribute("aria-label", /duration 123456789/);
  expect(await stressedStatus.evaluate((cell) => getComputedStyle(cell).color)).toBe(
    "rgb(154, 167, 184)",
  );
  await expect(page.locator("#battlefield .agent-id-tag")).toHaveCount(0);
  expect(
    await page.evaluate(() => document.fonts.check('16px "Atkinson Hyperlegible"')),
  ).toBe(true);
  await expect(page.locator("#roster .roster-row")).toHaveCount(10);
  await expect(page.locator("#event-count")).not.toHaveText("0");
  await expect(page).toHaveScreenshot(
    "human-number-formatting-synthetic-1440x900.png",
    { animations: "disabled" },
  );
  expect(commandRequests).toBe(0);
});

test("serialized mechanics tuning flows through the full semantic inspector", async ({
  page,
}) => {
  const catalogMutationFrame = catalogPropagationFixture.live_frame;
  const expected = catalogPropagationFixture.expected;
  const mechanics = catalogMutationFrame.projection.scene.class_mechanics.find(
    /** @param {Record<string, any>} row */ (row) => row.class_id === 1,
  );
  if (!mechanics) {
    throw new Error("The Python catalog mutation has no Mage mechanics row.");
  }
  const burst = mechanics.status_mechanics.find(
    /** @param {Record<string, any>} row */ (row) =>
      row.status_id === "mage_burst_damage_amplification",
  );
  const mageAura = mechanics.aura_mechanics.find(
    /** @param {Record<string, any>} row */ (row) =>
      row.aura_id === "mage_damage_amplification",
  );
  if (!burst || !mageAura) {
    throw new Error(
      "The Python catalog mutation has no Mage authored status/aura mechanics.",
    );
  }
  expect(mechanics.basic_raw_damage).toBe(expected.basic_raw_damage);
  expect(burst.duration_steps).toBe(expected.burst_duration_steps);
  expect(burst.magnitude).toBe(expected.burst_multiplier);
  expect(mageAura.radius).toBe(expected.aura_radius);
  expect(mageAura.per_emitter_multiplier).toBe(expected.aura_multiplier);

  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: catalogMutationFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    await route.abort("blockedbyclient");
  });
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const mageComparison = page.locator(
    '.comparison-agent[data-role="controlled"][data-slot="0"]',
  );
  await expect(mageComparison).toBeVisible();
  await mageComparison.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#semantic-inspector")).toBeVisible();
  const mechanicsSection = page
    .locator("#semantic-inspector .semantic-explanation__section")
    .filter({ hasText: "Exact Class Mechanics" });
  const damageTerm = mechanicsSection.locator("dt", {
    hasText: "Basic Raw Damage",
  });
  await expect(damageTerm).toHaveCount(1);
  await expect(damageTerm.locator("xpath=following-sibling::dd[1]")).toHaveText(
    String(expected.basic_raw_damage),
  );
  const statusSection = page
    .locator("#semantic-inspector .semantic-explanation__section")
    .filter({ hasText: "Authored Status Mechanics" });
  await expect(statusSection).toContainText(
    "Mage (Ultimate: Burst) Damage Amplification",
  );
  await expect(statusSection).toContainText(`${expected.burst_duration_steps} Ticks`);
  await expect(statusSection).toContainText(
    `${Math.round((expected.burst_multiplier - 1) * 100)}% more damage dealt`,
  );
  const passiveSection = page
    .locator("#semantic-inspector .semantic-explanation__section")
    .filter({ hasText: "Authored Passive Mechanics" });
  await expect(passiveSection).toContainText("Sorcerer's Aura Field");
  await expect(passiveSection).toContainText(`Radius ${expected.aura_radius}`);
  await expect(passiveSection).toContainText(
    `${Math.round((expected.aura_multiplier - 1) * 100)}% more damage dealt`,
  );

  const semanticInspector = await page
    .locator("#semantic-inspector")
    .evaluate((node) => ({
      rows: [...node.querySelectorAll("dt")].map((term) => ({
        label: term.textContent?.trim(),
        value: term.nextElementSibling?.textContent?.trim(),
      })),
      sections: [...node.querySelectorAll("h3")].map((heading) =>
        heading.textContent?.trim(),
      ),
    }));
  expect(semanticInspector.sections).toContain("Exact Class Mechanics");
  expect(semanticInspector.rows).toContainEqual({
    label: "Basic Raw Damage",
    value: String(expected.basic_raw_damage),
  });
  const snapshotLabels = [
    "Basic Raw Damage",
    "Mage (Ultimate: Burst) Damage Amplification",
    "Sorcerer's Aura Field (Mage Damage Amplification Aura)",
  ];
  const semanticSnapshot = {
    sections: semanticInspector.sections.filter((section) =>
      [
        "Exact Class Mechanics",
        "Authored Status Mechanics",
        "Authored Passive Mechanics",
      ].includes(section ?? ""),
    ),
    rows: semanticInspector.rows.filter((row) =>
      snapshotLabels.includes(row.label ?? ""),
    ),
  };
  expect(semanticSnapshot.sections).toHaveLength(3);
  expect(semanticSnapshot.rows).toHaveLength(3);
  const burstPercent = formatDisplayNumber(
    Math.abs(expected.burst_multiplier - 1) * 100,
  );
  const auraPercent = formatDisplayNumber(Math.abs(expected.aura_multiplier - 1) * 100);
  const tuningIndependentSnapshot = {
    sections: semanticSnapshot.sections,
    rows: semanticSnapshot.rows.map((row) => {
      if (row.label === "Basic Raw Damage") {
        return {
          label: row.label,
          value: replaceCatalogFragmentOnce(
            row.value,
            formatDisplayNumber(expected.basic_raw_damage),
            "<catalog-number>",
            row.label,
          ),
        };
      }
      if (row.label === "Mage (Ultimate: Burst) Damage Amplification") {
        let value = replaceCatalogFragmentOnce(
          row.value,
          `${formatDisplayNumber(expected.burst_duration_steps)} Ticks`,
          "<catalog-duration> Ticks",
          row.label,
        );
        value = replaceCatalogFragmentOnce(
          value,
          `${burstPercent}% more damage dealt`,
          "<catalog-percent>% more damage dealt",
          row.label,
        );
        value = replaceCatalogFragmentOnce(
          value,
          `×${formatDisplayNumber(expected.burst_multiplier)}`,
          "×<catalog-multiplier>",
          row.label,
        );
        return { label: row.label, value };
      }
      let value = replaceCatalogFragmentOnce(
        row.value,
        `Radius ${formatDisplayNumber(expected.aura_radius)}`,
        "Radius <catalog-radius>",
        row.label ?? "Authored aura mechanics",
      );
      value = replaceCatalogFragmentOnce(
        value,
        `${auraPercent}% more damage dealt per recorded emitter`,
        "<catalog-percent>% more damage dealt per recorded emitter",
        row.label ?? "Authored aura mechanics",
      );
      value = replaceCatalogFragmentOnce(
        value,
        `×${formatDisplayNumber(mageAura.clamp_value)}`,
        "×<catalog-clamp>",
        row.label ?? "Authored aura mechanics",
      );
      return { label: row.label, value };
    }),
  };
  expect(`${JSON.stringify(tuningIndependentSnapshot, null, 2)}\n`).toMatchSnapshot(
    "catalog-mechanics-semantic-snapshot.json",
  );
});

test("compact active combat yields analysis decoration and restores it after Skip", async ({
  page,
}) => {
  await page.setViewportSize({ width: 960, height: 600 });
  await installWaapiAutopause(page);
  let commandRequests = 0;
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: syntheticWireFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    commandRequests += 1;
    await route.abort("blockedbyclient");
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
  await pauseAtLogicalTime(page, 520);

  const battlefield = page.locator("#battlefield");
  await expect(battlefield).toHaveAttribute("data-compact-active-combat", "true");
  await expect(battlefield).toHaveAttribute(
    "data-compact-active-suppressed-facts",
    "ranges,pending-route,selected-legality,status-summaries",
  );
  await expect(page.locator('#battlefield [data-layer="range-cues"]')).toHaveAttribute(
    "data-compact-active-suppressed",
    "true",
  );
  await expect(
    page.locator('#battlefield [data-layer="pending-route"]'),
  ).toHaveAttribute("data-compact-active-suppressed", "true");
  await expect(
    page.locator('#battlefield [data-layer="legality-cues"]'),
  ).toHaveAttribute("data-compact-active-suppressed", "true");
  await expect(page.locator("#battlefield .status-dock")).toHaveCount(10);
  expect(
    await page
      .locator("#battlefield .status-dock")
      .evaluateAll((docks) =>
        docks.every(
          (dock) =>
            dock.getAttribute("data-compact-active-suppressed") === "true" &&
            getComputedStyle(dock).display === "none",
        ),
      ),
  ).toBe(true);
  expect(
    await page
      .locator(
        '#battlefield [data-layer="range-cues"], #battlefield [data-layer="pending-route"], #battlefield [data-layer="legality-cues"]',
      )
      .evaluateAll((owners) =>
        owners.every((owner) => getComputedStyle(owner).display === "none"),
      ),
  ).toBe(true);

  // Suppression is battlefield-only: the settled V2 scene facts and structured
  // inspection surfaces remain present while accepted combat takes priority.
  await expect(page.locator("#battlefield .range-ring")).toHaveCount(3);
  await expect(
    page.locator("#roster .roster-row .roster-fact-token--status"),
  ).toHaveCount(90);
  await expect(page.locator("#selection-card .selected-legality")).toContainText(
    "Selected target legality",
  );
  await expect(page.locator("#event-feed .event-item")).toHaveCount(32);
  await expect(
    page.locator(
      '#battlefield [data-layer="transient-route"] .combat-route-effect--activation',
    ),
  ).toHaveCount(10);
  await expect(
    page.locator(
      '#battlefield [data-layer="transient-events"] .combat-effect--net-health',
    ),
  ).toHaveCount(8);
  await expect(page.locator("#battlefield .agent")).toHaveCount(10);
  await expect(
    page.locator("#battlefield .selected-reticle:not([hidden])"),
  ).toHaveCount(1);

  const skip = page.locator("#motion-skip-button");
  await expect(skip).toBeEnabled();
  await skip.click();
  await expect(battlefield).toHaveAttribute("data-compact-active-combat", "false");
  await expect(battlefield).not.toHaveAttribute(
    "data-compact-active-suppressed-facts",
    /.*/,
  );
  expect(
    await page
      .locator(
        '#battlefield [data-layer="range-cues"], #battlefield [data-layer="pending-route"], #battlefield [data-layer="legality-cues"], #battlefield .status-dock',
      )
      .evaluateAll((owners) =>
        owners.every(
          (owner) =>
            owner.getAttribute("data-compact-active-suppressed") === "false" &&
            getComputedStyle(owner).display !== "none",
        ),
      ),
  ).toBe(true);
  await expect(
    page.locator('#battlefield .status-dock[data-slot="0"] .status-overflow'),
  ).toHaveAttribute("aria-label", /hidden status cues for Agent ID 0/i);
  await expect(page.locator("#battlefield .range-ring")).toHaveCount(3);
  expect(commandRequests).toBe(0);
});

test("extreme dock values use a readable visual abbreviation without losing truth", async ({
  page,
}) => {
  const extremeFrame = structuredClone(syntheticWireFrame);
  authorizeSyntheticStatusDuration(extremeFrame.projection.scene, 0, 0, 123456789);
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: extremeFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    await route.abort("blockedbyclient");
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const cue = page.locator('#battlefield .status-cell[data-slot="0"][data-index="0"]');
  await expect(cue).toHaveAttribute("data-duration", "123456789");
  await expect(cue).toHaveAttribute("data-numeric-layout", "compact-measured-fallback");
  await expect(cue.locator(".status-cell__value")).toHaveText("123M");
  await expect(cue).toHaveAttribute("aria-label", /duration 123456789/);
  expect(
    await cue.evaluate((cell) => {
      const box = cell.querySelector(".status-cell__box");
      const value = cell.querySelector(".status-cell__value");
      if (
        !(box instanceof SVGGraphicsElement) ||
        !(value instanceof SVGGraphicsElement)
      ) {
        throw new Error("Extreme status cue is not measurable.");
      }
      return value.getBBox().width <= box.getBBox().width - 6;
    }),
  ).toBe(true);

  await cue.hover();
  await expect(page.locator("#visual-tooltip")).toContainText("123456789");

  const rosterCue = page
    .locator('#roster .roster-row[data-slot="0"] .roster-fact-token--status')
    .first();
  await expect(rosterCue).toHaveAttribute("data-duration", "123456789");
  await expect(rosterCue).toHaveAttribute("data-visible-value-abbreviated", "true");
  await expect(rosterCue.locator(".roster-fact-token__duration")).toHaveText("123M");
  await expect(rosterCue).toHaveAttribute("aria-label", /duration 123456789/);
  await rosterCue.hover();
  await expect(page.locator("#visual-tooltip")).toContainText("123456789");
});

test("required fallback tooltip avoids the inspected agent and its local docks", async ({
  page,
}) => {
  const denseRequiredFrame = structuredClone(
    syntheticDebuggerWireFrame(await loadRendererFixture("required_dock_fallback")),
  );
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: denseRequiredFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    await route.abort("blockedbyclient");
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const fallback = page
    .locator('#battlefield .required-dock-fallback[data-kind="cooldown"]')
    .first();
  const slot = await fallback.getAttribute("data-slot");
  expect(slot).not.toBeNull();
  await fallback.hover();
  const tooltip = page.locator("#visual-tooltip");
  await expect(tooltip).toBeVisible();
  const overlapsLocalEnvelope = await page.evaluate((globalSlot) => {
    const tooltipElement = document.querySelector("#visual-tooltip");
    const battlefield = document.querySelector("#battlefield");
    if (
      !(tooltipElement instanceof HTMLElement) ||
      !(battlefield instanceof SVGElement)
    ) {
      throw new Error("Tooltip placement targets are unavailable.");
    }
    const tooltipBounds = tooltipElement.getBoundingClientRect();
    const localElements = [...battlefield.querySelectorAll("[data-slot]")].filter(
      (element) =>
        element.getAttribute("data-slot") === globalSlot &&
        [
          "agent",
          "status-dock",
          "cooldown-dock",
          "required-dock-fallback-dock",
          "modifier-dock",
          "legality-dock",
        ].some((className) => element.classList.contains(className)),
    );
    return localElements.some((element) => {
      const bounds = element.getBoundingClientRect();
      const width =
        Math.min(tooltipBounds.right, bounds.right) -
        Math.max(tooltipBounds.left, bounds.left);
      const height =
        Math.min(tooltipBounds.bottom, bounds.bottom) -
        Math.max(tooltipBounds.top, bounds.top);
      return width > 0.5 && height > 0.5;
    });
  }, slot);
  expect(overlapsLocalEnvelope).toBe(false);
});

test("near-dense required truth keeps every compact fallback individually associated", async ({
  page,
}) => {
  const denseRequiredFrame = structuredClone(
    syntheticDebuggerWireFrame(await loadRendererFixture("required_dock_fallback")),
  );
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: denseRequiredFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    await route.abort("blockedbyclient");
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const durableLayer = page.locator(
    '#battlefield [data-layer="durable-status-modifier"]',
  );
  await expect(durableLayer).toHaveAttribute("data-suppressed-cooldown-slots", "");
  await expect(durableLayer).toHaveAttribute(
    "data-compacted-required-docks",
    /cooldown:/,
  );
  const suppressedStatusSlots = (
    (await durableLayer.getAttribute("data-suppressed-status-slots")) ?? ""
  )
    .split(",")
    .filter(Boolean)
    .map(Number);
  expect(suppressedStatusSlots).not.toContain(0);
  expect(suppressedStatusSlots).not.toContain(7);

  const representedKeys = await page.evaluate(() => {
    const keys = [
      ...document.querySelectorAll("#battlefield .required-dock-fallback"),
    ].map((node) => node.getAttribute("data-layout-key"));
    for (const node of document.querySelectorAll("#battlefield .cooldown-dock")) {
      keys.push(`cooldown:${node.getAttribute("data-slot")}`);
    }
    for (const node of document.querySelectorAll("#battlefield .status-dock")) {
      const slot = Number(node.getAttribute("data-slot"));
      if (slot === 0 || slot === 7) {
        keys.push(`status:${slot}`);
      }
    }
    return keys.filter((key) => key !== null).sort();
  });
  expect(representedKeys).toEqual(
    [
      ...Array.from({ length: 10 }, (_, slot) => `cooldown:${slot}`),
      "status:0",
      "status:7",
    ].sort(),
  );

  const cooldownFallbacks = page.locator(
    '#battlefield .required-dock-fallback[data-kind="cooldown"]',
  );
  expect(await cooldownFallbacks.count()).toBeGreaterThan(0);
  for (let index = 0; index < (await cooldownFallbacks.count()); index += 1) {
    const fallback = cooldownFallbacks.nth(index);
    await expect(fallback).toHaveAttribute("data-ticks", "30");
    await expect(fallback).toHaveAttribute(
      "aria-label",
      /Cooldown Remaining: 30 Ticks/i,
    );
    const slot = await fallback.getAttribute("data-slot");
    await expect(fallback).toHaveAttribute("data-owner-label", `Agent ID ${slot}`);
    await expect(fallback.locator(".required-dock-fallback__owner")).toHaveText(
      `Agent ID ${slot}`,
    );
    await expect(fallback.locator(".required-dock-fallback__value")).toHaveText("U30");
  }

  const geometry = await page.evaluate(() => {
    const battlefield = document.querySelector("#battlefield");
    if (!(battlefield instanceof SVGSVGElement)) {
      throw new Error("Battlefield SVG is unavailable.");
    }
    /** @param {string} selector */
    const bounds = (selector) =>
      [...battlefield.querySelectorAll(selector)].map((node) => {
        if (!(node instanceof SVGGraphicsElement)) {
          throw new Error(`Expected SVG graphics element for ${selector}.`);
        }
        const box = node.getBBox();
        return {
          left: box.x,
          top: box.y,
          right: box.x + box.width,
          bottom: box.y + box.height,
        };
      });
    /**
     * @param {{left: number, top: number, right: number, bottom: number}} first
     * @param {{left: number, top: number, right: number, bottom: number}} second
     */
    const overlaps = (first, second) =>
      Math.min(first.right, second.right) - Math.max(first.left, second.left) > 0.5 &&
      Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top) > 0.5;
    const fallbackBoxes = bounds(".required-dock-fallback__box");
    const bodyBoxes = bounds(".agent-body");
    const fullDockBoxes = bounds(".status-cell__box, .cooldown-cell__box");
    const selectionBoxes = bounds(
      ".controlled-halo:not([hidden]), .selected-reticle:not([hidden])",
    );
    const fallbackAssociations = [
      ...battlefield.querySelectorAll(".required-dock-fallback-dock"),
    ].map((dock) => {
      const slot = Number(dock.getAttribute("data-slot"));
      const body = battlefield.querySelector(`.agent[data-slot="${slot}"] .agent-body`);
      const marker = dock.querySelector(".required-dock-fallback__box");
      const leader = dock.querySelector(".required-dock-fallback__leader");
      const owner = dock.querySelector(".required-dock-fallback__owner");
      if (
        !(body instanceof SVGCircleElement) ||
        !(marker instanceof SVGGraphicsElement) ||
        !(leader instanceof SVGLineElement) ||
        !(owner instanceof SVGElement)
      ) {
        throw new Error(`Fallback ownership geometry is incomplete for slot ${slot}.`);
      }
      const center = {
        x: Number(body.getAttribute("cx")),
        y: Number(body.getAttribute("cy")),
      };
      const radius = Number(body.getAttribute("r"));
      const start = {
        x: Number(leader.getAttribute("x1")),
        y: Number(leader.getAttribute("y1")),
      };
      const end = {
        x: Number(leader.getAttribute("x2")),
        y: Number(leader.getAttribute("y2")),
      };
      const markerBounds = marker.getBBox();
      const otherCenters = [...battlefield.querySelectorAll(".agent")]
        .filter((agent) => Number(agent.getAttribute("data-slot")) !== slot)
        .map((agent) => {
          const otherBody = agent.querySelector(".agent-body");
          if (!(otherBody instanceof SVGCircleElement)) {
            throw new Error("Fallback comparison body is unavailable.");
          }
          return {
            x: Number(otherBody.getAttribute("cx")),
            y: Number(otherBody.getAttribute("cy")),
          };
        });
      /**
       * @param {{x: number, y: number}} first
       * @param {{x: number, y: number}} second
       */
      const distance = (first, second) =>
        Math.hypot(first.x - second.x, first.y - second.y);
      const ownerDistance = distance(start, center);
      return {
        endTouchesMarker:
          end.x >= markerBounds.x - 0.5 &&
          end.x <= markerBounds.x + markerBounds.width + 0.5 &&
          end.y >= markerBounds.y - 0.5 &&
          end.y <= markerBounds.y + markerBounds.height + 0.5,
        nearestOtherDistance: Math.min(
          ...otherCenters.map((otherCenter) => distance(start, otherCenter)),
        ),
        nearestStartIsOwner: otherCenters.every(
          (otherCenter) => ownerDistance < distance(start, otherCenter),
        ),
        ownerDistance,
        ownerLabel: owner.textContent,
        radius,
        slot,
        startsAtOwner: ownerDistance <= radius + 20,
      };
    });
    const viewBox = battlefield.viewBox.baseVal;
    return {
      fallbackAssociations,
      fallbackCount: fallbackBoxes.length,
      overlapsBody: fallbackBoxes.some((fallback) =>
        bodyBoxes.some((body) => overlaps(fallback, body)),
      ),
      overlapsFullDock: fallbackBoxes.some((fallback) =>
        fullDockBoxes.some((dock) => overlaps(fallback, dock)),
      ),
      overlapsFallback: fallbackBoxes.some((fallback, index) =>
        fallbackBoxes.slice(index + 1).some((other) => overlaps(fallback, other)),
      ),
      overlapsSelection: fallbackBoxes.some((fallback) =>
        selectionBoxes.some((selection) => overlaps(fallback, selection)),
      ),
      outsideViewport: fallbackBoxes.some(
        (fallback) =>
          fallback.left < viewBox.x - 0.5 ||
          fallback.top < viewBox.y - 0.5 ||
          fallback.right > viewBox.x + viewBox.width + 0.5 ||
          fallback.bottom > viewBox.y + viewBox.height + 0.5,
      ),
    };
  });
  expect(geometry).toMatchObject({
    outsideViewport: false,
    overlapsBody: false,
    overlapsFallback: false,
    overlapsFullDock: false,
    overlapsSelection: false,
  });
  expect(geometry.fallbackCount).toBeGreaterThan(0);
  expect(geometry.fallbackAssociations).toHaveLength(geometry.fallbackCount);
  expect(
    geometry.fallbackAssociations.filter(
      ({ endTouchesMarker, nearestStartIsOwner, ownerLabel, slot, startsAtOwner }) =>
        !endTouchesMarker ||
        !nearestStartIsOwner ||
        ownerLabel !== `Agent ID ${slot}` ||
        !startsAtOwner,
    ),
  ).toEqual([]);

  const firstFallback = cooldownFallbacks.first();
  await firstFallback.hover();
  const tooltip = page.locator("#visual-tooltip");
  await expect(tooltip).toBeVisible();
  await expect(tooltip).toContainText(/Cooldown Remaining\s*30 Ticks/);
  await firstFallback.focus();
  expect(
    await firstFallback.evaluate((node) => {
      const style = getComputedStyle(node);
      return `${style.outlineStyle} ${style.outlineWidth}`;
    }),
  ).toBe("solid 3px");
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve)),
    );
  });
  await expect(page).toHaveScreenshot(
    "required-dock-fallback-focus-synthetic-1440x900.png",
    { animations: "disabled" },
  );

  const rosterFact = page.locator('.roster-fact-token[tabindex="0"]').first();
  await rosterFact.focus();
  expect(
    await rosterFact.evaluate((node) => {
      const style = getComputedStyle(node);
      return `${style.outlineStyle} ${style.outlineWidth}`;
    }),
  ).toBe("solid 3px");
});

test("live pending routes use authorized researcher and POV public identities", async ({
  page,
}) => {
  const researcher = structuredClone(syntheticWireFrame);
  const researcherLegality =
    researcher.projection.scene.next_decision_selected_legality;
  const researcherTarget = researcher.projection.scene.agents.find(
    /** @param {Record<string, any>} agent */
    (agent) => agent.global_slot === researcherLegality.target_global_slot,
  );
  if (
    !researcherTarget ||
    researcherLegality.lane_0_available !== true ||
    researcherLegality.lane_1_available !== false
  ) {
    throw new Error(
      "Synthetic researcher pending-route proof requires its exact selected legality.",
    );
  }
  const researcherPending = {
    ...researcher.hud.pending_action,
    target_action: researcherLegality.target_action,
    armed_lane: 1,
    arm_origin: "explicit",
    target: {
      disclosure: "public",
      global_slot: researcherTarget.global_slot,
    },
    pair_mask_value: false,
    summary: "STAY + ULTIMATE",
  };
  researcher.hud.pending_action = researcherPending;
  researcher.hud.pending_actions = [researcherPending];
  const legalResearcher = structuredClone(researcher);
  const legalResearcherPending = {
    ...researcherPending,
    armed_lane: 0,
    pair_mask_value: true,
    summary: "STAY + BASIC",
  };
  legalResearcher.hud.pending_action = legalResearcherPending;
  legalResearcher.hud.pending_actions = [legalResearcherPending];

  const pov = structuredClone(syntheticPovWireFrame);
  const povTarget = pov.projection.scene.visible_bodies[0];
  const povTargetAction = pov.hud.candidate_legalities.find(
    /** @param {Record<string, any>} candidate */
    (candidate) => candidate.target.public_agent_id === povTarget.public_agent_id,
  ).target.target_action;
  pov.projection.next_decision_action_mask.select_target[povTargetAction] = true;
  pov.projection.next_decision_action_mask.select_target_use_ultimate_joint[
    povTargetAction
  ] = [true, false];
  const povCandidate = pov.hud.candidate_legalities.find(
    /** @param {Record<string, any>} candidate */
    (candidate) => candidate.target.target_action === povTargetAction,
  );
  povCandidate.lane_0_available = true;
  povCandidate.basic_available = true;
  pov.hud.pending_action = {
    ...pov.hud.pending_action,
    target: {
      target_action: povTargetAction,
      public_agent_id: povTarget.public_agent_id,
    },
    armed_lane: 0,
    arm_origin: "explicit",
    pair_mask_value: true,
    summary: "STAY + BASIC",
  };

  let servedFrame = researcher;
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
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);

  servedFrame = legalResearcher;
  await page.reload();
  const researcherRoute = page.locator("#battlefield .pending-route");
  await expect(researcherRoute).toHaveCount(1);
  await expect(researcherRoute).toHaveAttribute(
    "data-source-agent-id",
    legalResearcher.projection.scene.agents.find(
      /** @param {Record<string, any>} agent */
      (agent) =>
        agent.global_slot === legalResearcher.hud.pending_action.actor_global_slot,
    ).public_agent_id,
  );
  await expect(researcherRoute).toHaveAttribute(
    "data-target-agent-id",
    researcherTarget.public_agent_id,
  );
  await expect(researcherRoute).toHaveAttribute("data-lane", "0");
  await expect(researcherRoute).toHaveAttribute("data-legal", "true");

  servedFrame = pov;
  await page.reload();
  const povRoute = page.locator("#battlefield .pending-route");
  await expect(povRoute).toHaveCount(1);
  await expect(povRoute).toHaveAttribute(
    "data-source-agent-id",
    pov.projection.scene.self_actor.public_agent_id,
  );
  await expect(povRoute).toHaveAttribute(
    "data-target-agent-id",
    povTarget.public_agent_id,
  );
  await expect(povRoute).not.toHaveAttribute("data-source-slot", /.+/u);
  await expect(povRoute).not.toHaveAttribute("data-target-slot", /.+/u);
  await expect(povRoute).toHaveAttribute("data-lane", "0");
  await expect(povRoute).toHaveAttribute("data-legal", "true");
  const routePointer = await page
    .locator("#battlefield .pending-route-hit")
    .evaluate((path) => {
      if (!(path instanceof SVGPathElement)) {
        throw new Error("Pending route hit owner must be an SVG path.");
      }
      const matrix = path.getScreenCTM();
      if (matrix === null) {
        throw new Error("Pending route has no screen transform.");
      }
      const local = path.getPointAtLength(path.getTotalLength() / 2);
      const screen = new DOMPoint(local.x, local.y).matrixTransform(matrix);
      return { x: screen.x, y: screen.y };
    });
  await page.mouse.move(routePointer.x, routePointer.y);
  await expect(page.locator("#visual-tooltip-title")).toHaveText("Basic Action Route");
  await expect(page.locator("#visual-tooltip-details")).toContainText(
    "Selected Target",
  );
  await expect(page.locator("#visual-tooltip-details")).toContainText(
    `Agent ID ${povTarget.public_agent_id}`,
  );
  await expect(page.locator("#visual-tooltip-details")).not.toContainText("id_");
});

test("canonical V2 death clear is distinct in cue, feed, and accessible copy", async ({
  page,
}) => {
  await installWaapiAutopause(page);
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: canonicalGrammarWireFrame,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    await route.abort("blockedbyclient");
  });

  await page.goto(debuggerUrl);
  await pauseAtLogicalTime(page, 680);

  const deathClear = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect[data-event-type="status_cleared_by_new_death"][data-lifecycle="cleared_by_death"]`,
  );
  const expiry = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect[data-event-type="status_aged_to_zero"][data-lifecycle="expired"]`,
  );
  const damageBreak = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect[data-event-type="status_broken_by_damage"][data-lifecycle="trap_broken"]`,
  );
  await expect(deathClear).toHaveCount(1);
  await expect(deathClear.locator(".combat-lifecycle__death-sweep")).toHaveCount(1);
  await expect(expiry.locator(".combat-lifecycle__death-sweep")).toHaveCount(0);
  await expect(damageBreak.locator(".combat-lifecycle__death-sweep")).toHaveCount(0);
  await expect(damageBreak.locator(".combat-lifecycle__shard")).toHaveCount(6);

  const deathClearFeed = page.locator(
    '#event-feed .event-item[data-event-type="status_cleared_by_new_death"]',
  );
  await expect(deathClearFeed).toContainText("cleared by new death");
  await deathClearFeed.focus();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await expect(page.locator("#visual-tooltip")).toContainText(
    /Status Cleared By New Death/u,
  );

  const rejection = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect[data-event-type="action_rejected"]`,
  );
  const charge = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect[data-event-type="charge_phase_displacement"]`,
  );
  await expect(rejection).toHaveCSS("color", "rgb(251, 113, 133)");
  await expect(charge).toHaveCSS("color", "rgb(209, 139, 71)");
  expect(canonicalGrammarFrame.event_batch.events).toHaveLength(23);
});

test("structured HUD keeps exact roster, intent, and accepted-result facts distinct", async ({
  page,
}) => {
  const transitionId = String(syntheticFrame.transition_id);
  const sourceEvents = syntheticWireFrame.projection.incoming_events.events;
  const firstSourceEvent = sourceEvents[0];
  if (!firstSourceEvent) {
    throw new Error(
      "Synthetic crowded fixture requires at least one transition event.",
    );
  }
  const rejectedEvent = {
    event_type: "action_rejected",
    event_id: firstSourceEvent.event_id,
    transition_id: transitionId,
    ordinal: firstSourceEvent.ordinal,
    phase_rank: 10,
    actor_global_slot: 0,
    actor_public_agent_id: "0",
    actor_configured_active: true,
    rejection_component: "combat_pair",
    submitted_move_action: 0,
    submitted_select_target_action: 8,
    submitted_use_ultimate_action: 1,
    actor_anchor: {
      phase: "transition_start",
      global_slot: 0,
      public_agent_id: "0",
      position:
        syntheticWireFrame.projection.incoming_events.agent_phase_trajectories.find(
          /** @param {Record<string, any>} trajectory */
          (trajectory) => trajectory.global_slot === 0,
        ).transition_start.position,
    },
  };
  /** @type {Record<string, any>} */
  let structuredFrame = structuredClone(syntheticWireFrame);
  structuredFrame.projection.incoming_events.events = [
    rejectedEvent,
    ...structuredFrame.projection.incoming_events.events.slice(1),
  ];
  structuredFrame.projection.scene.incoming_event_ids =
    structuredFrame.projection.incoming_events.events.map(
      /** @param {{event_id: string}} event */ (event) => event.event_id,
    );
  const pendingAction = {
    label: "PENDING / WILL SUBMIT",
    actor_global_slot: 0,
    move_action: 0,
    target_action: 8,
    armed_lane: 1,
    arm_origin: "explicit",
    target: { disclosure: "public", global_slot: 7 },
    movement_mask_value: true,
    pair_mask_value: false,
    summary: "Stay + Burst → Agent ID 7",
  };
  structuredFrame.hud = {
    ...structuredFrame.hud,
    roster_global_slots: syntheticFrame.scene.agents.map(
      /** @param {{global_slot: number}} agent */ (agent) => agent.global_slot,
    ),
    controlled_global_slot: 0,
    selected_global_slot: 7,
    pending_submission_scope: "controlled_actor",
    pending_actions: [pendingAction],
    pending_action: pendingAction,
    latest_transition: {
      label: "LATEST ACCEPTED RESULT",
      transition_index: 0,
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
            summary: "Stay + Burst → Agent ID 7",
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
  };
  const structuredEvents = structuredFrame.projection.incoming_events.events;
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
        schema_version: 2,
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
    throw new Error("Synthetic crowded fixture is missing global slot 0.");
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
          icon: token.getAttribute("data-icon"),
          renderedIcon: token
            .querySelector(".roster-fact-token__icon")
            ?.getAttribute("data-icon"),
          sourceClass: token.getAttribute("data-source-class"),
          tokenId: token.getAttribute("data-token-id"),
          separated: (() => {
            const icon = token.querySelector(".roster-fact-token__icon");
            const duration = token.querySelector(".roster-fact-token__duration");
            if (
              !(icon instanceof SVGSVGElement) ||
              !(duration instanceof HTMLElement)
            ) {
              return false;
            }
            return (
              icon.getBoundingClientRect().right + 1 <=
              duration.getBoundingClientRect().left
            );
          })(),
        })),
      ),
  ).toEqual(
    agentZero.statuses.map(
      /** @param {{duration: number, source_class_id: number, token_id: string}} status */ (
        status,
      ) => ({
        duration: status.duration,
        icon: status.token_id.startsWith("stun_")
          ? "status-stun"
          : status.token_id.startsWith("slow_")
            ? "status-slow"
            : {
                anti_heal_rogue_poison: "status-anti-heal",
                mage_burst: "status-burst",
                priest_freedom: "status-freedom",
              }[status.token_id],
        renderedIcon: status.token_id.startsWith("stun_")
          ? "status-stun"
          : status.token_id.startsWith("slow_")
            ? "status-slow"
            : {
                anti_heal_rogue_poison: "status-anti-heal",
                mage_burst: "status-burst",
                priest_freedom: "status-freedom",
              }[status.token_id],
        sourceClass: ["unknown", "mage", "warrior", "hunter", "rogue", "priest"][
          status.source_class_id
        ],
        tokenId: status.token_id,
        separated: true,
      }),
    ),
  );
  const durableControlCells = await page
    .locator('#battlefield .status-cell[data-slot="0"]')
    .evaluateAll((cells) =>
      cells.slice(0, 6).map((cell) => ({
        icon: cell.querySelector(".status-cell__icon")?.getAttribute("data-icon"),
        sourceClass: cell.getAttribute("data-source-class"),
      })),
    );
  expect(durableControlCells).toEqual([
    { icon: "status-stun", sourceClass: "warrior" },
    { icon: "status-stun", sourceClass: "hunter" },
    { icon: "status-stun", sourceClass: "rogue" },
    { icon: "status-slow", sourceClass: "warrior" },
    { icon: "status-slow", sourceClass: "hunter" },
    { icon: "status-slow", sourceClass: "rogue" },
  ]);
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
  await expect(
    page
      .locator(".selected-legality__lane")
      .filter({ has: page.getByRole("heading", { name: "Basic Legality" }) })
      .locator(".semantic-explanation__value"),
  ).toHaveText("True");
  await expect(
    page
      .locator(".selected-legality__lane")
      .filter({ has: page.getByRole("heading", { name: "Ultimate Legality" }) })
      .locator(".semantic-explanation__value"),
  ).toHaveText("False");
  await expect(page.locator("#pending-card .action-card__label")).toHaveText(
    "PENDING / WILL SUBMIT",
  );
  await expect(page.locator("#pending-card")).toContainText(
    "Stay + Burst → Agent ID 7",
  );
  await expect(page.locator("#accepted-card .action-card__label")).toHaveText(
    "LATEST ACCEPTED RESULT",
  );
  await expect(
    page.locator('#accepted-card .action-tuple[data-kind="submitted"]'),
  ).toContainText("Stay + Burst → Agent ID 7");
  await expect(
    page.locator('#accepted-card .action-tuple[data-kind="accepted"]'),
  ).toContainText("Stay + NO COMBAT");
  await expect(page.locator("#accepted-card")).toContainText("Combat result");
  await expect(page.locator("#accepted-card")).toContainText("Rejected");
  await expect(page.locator("#accepted-announcement")).toContainText(
    `Transition ${transitionId}`,
  );
  await expect(page.locator("#event-count")).toHaveText(
    String(structuredEvents.length),
  );
  expect(
    await page.locator("#event-feed .event-item").evaluateAll((items) =>
      items.map((item) => ({
        eventId: item.getAttribute("data-event-id"),
        eventType: item.getAttribute("data-event-type"),
      })),
    ),
  ).toEqual(
    structuredEvents.map(
      /** @param {{event_id: string, event_type: string}} event */ (event) => ({
        eventId: event.event_id,
        eventType: event.event_type,
      }),
    ),
  );
  for (const activation of await page
    .locator('#event-feed .event-item[data-event-type="ability_activated"]')
    .all()) {
    await expect(activation).not.toContainText(/NET|HP|amount/i);
  }
  for (const netHealth of await page
    .locator('#event-feed .event-item[data-event-type="recipient_health_resolution"]')
    .all()) {
    await expect(netHealth).toContainText("Net combat health");
    await expect(netHealth).toHaveAttribute("data-recipient-slot", /\d+/);
    await expect(netHealth).not.toHaveAttribute("data-source-slot", /.+/);
  }
  const rejected = page.locator(
    '#event-feed .event-item[data-event-type="action_rejected"]',
  );
  await expect(rejected).toHaveAttribute("data-actor-slot", "0");
  await expect(rejected).not.toHaveAttribute("data-target-slot", /.+/);
  await expect(rejected).toContainText("recorded component Combat Pair");
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
  let servedFrame = structuredClone(syntheticWireFrame);
  servedFrame.preset = "analysis";
  servedFrame.projection.scene.agents.forEach(
    /** @param {Record<string, any>} agent */ (agent) => {
      agent.ultimate_cooldown_remaining = 0;
    },
  );
  authorizeSyntheticCooldown(servedFrame.projection.scene, 2, 30);
  servedFrame.projection.scene.next_decision_selected_legality = {
    ...servedFrame.projection.scene.next_decision_selected_legality,
    armed_lane: null,
    armed_pair_legal: false,
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
        schema_version: 2,
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
  const authorizedAuraCount = await page.locator("#battlefield .aura-field").count();
  const authorizedRangeCount = await page.locator("#battlefield .range-ring").count();
  const authorizedModifierCount = await page
    .locator("#battlefield .modifier-cell")
    .count();
  expect(authorizedAuraCount).toBeGreaterThan(0);
  expect(authorizedRangeCount).toBeGreaterThan(0);
  expect(authorizedModifierCount).toBeGreaterThan(0);
  await expect(page.locator(".selected-legality")).toHaveCount(1);
  await expect(
    page
      .locator(".selected-legality__lane")
      .filter({ has: page.getByRole("heading", { name: "Basic Legality" }) })
      .locator(".semantic-explanation__value"),
  ).toHaveText("True");
  await expect(
    page
      .locator(".selected-legality__lane")
      .filter({ has: page.getByRole("heading", { name: "Ultimate Legality" }) })
      .locator(".semantic-explanation__value"),
  ).toHaveText("False");
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
  await expect(page.locator("#battlefield .aura-field")).toHaveCount(
    authorizedAuraCount,
  );
  await expect(page.locator("#battlefield .range-ring")).toHaveCount(
    authorizedRangeCount,
  );
  await expect(page.locator("#battlefield .modifier-cell")).toHaveCount(
    authorizedModifierCount,
  );
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

  await page.locator("#battlefield .aura-field").first().focus();
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "aura",
  );
  await expect(page.locator("#visual-tooltip-title")).toContainText("Aura Field");
  await page.locator("#battlefield .modifier-cell").first().hover();
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "modifier",
  );
  await expect(page.locator("#visual-tooltip-title")).toContainText("Aura");
});

test("debug POV omits hidden agents and researcher-only visibility DOM", async ({
  page,
}) => {
  const servedFrame = structuredClone(syntheticPovWireFrame);
  servedFrame.preset = "debug";
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
  await expect(page.locator("#battlefield .pov-observed-body")).toHaveCount(1);
  await expect(page.locator("#roster .roster-row")).toHaveCount(1);
  await expect(page.locator('#battlefield .agent[data-slot="5"]')).toHaveCount(0);
  await expect(page.locator('#roster .roster-row[data-slot="5"]')).toHaveCount(0);
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-visibility-cue")).toHaveCount(0);
  await expect(page.locator("#battlefield .debug-protected-zone")).toHaveCount(2);
  await expect(page.locator("#scenario-control")).toBeHidden();
  await expect(page.locator("#scenario-select")).toBeDisabled();
  await expect(page.locator("#scenario-select")).not.toHaveAttribute("title", /.+/u);
  await expect(page.locator("#scenario-select")).toHaveAttribute(
    "data-tooltip-owner",
    "",
  );
  await expect(page.locator("#scenario-select")).toHaveAttribute(
    "aria-description",
    "Choose a registered live episode setup.",
  );
  await expect(
    page.locator('#battlefield .agent[data-public-agent-id="5"]'),
  ).toHaveCount(0);
  await expect(
    page.locator('#roster .roster-row[data-public-agent-id="5"]'),
  ).toHaveCount(0);
  await expect(
    page.locator('[data-source-slot="5"], [data-target-slot="5"]'),
  ).toHaveCount(0);
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
