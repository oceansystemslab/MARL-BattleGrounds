import { fileURLToPath } from "node:url";

import { expect } from "@playwright/test";

import {
  CHOREOGRAPHY_ROOT,
  CHOREOGRAPHY_ROUTE_ROOT,
  choreographySnapshot,
  installWaapiAutopause,
  pauseAtLogicalTime,
} from "./choreography.js";

export const DESKTOP_VIEWPORT = Object.freeze({ width: 1440, height: 900 });
export const MINIMUM_VIEWPORT = Object.freeze({ width: 960, height: 600 });
export const ABILITY_PHASE_MS = 120;
export const HEALTH_RESOLUTION_PHASE_MS = 210;
export const CHARGE_PHASE_MS = 400;
export const STATUS_PHASE_MS = 680;
// Both recipient-safe successor deltas begin together at 520 ms. Sample just
// inside that shared, explicitly non-causal observation phase so both are
// visibly painted without inventing an order between them.
export const POV_SUCCESSOR_OBSERVATION_PHASE_MS = 600;

const SNAPSHOT_STYLE_PATH = fileURLToPath(
  new URL("./visual-snapshot.css", import.meta.url),
);

/**
 * @typedef {{count: () => number}} CommandPostCounter
 */

/**
 * Count authoritative command requests without intercepting them.
 *
 * @param {import("@playwright/test").Page} page
 * @returns {CommandPostCounter}
 */
export function trackCommandPosts(page) {
  let count = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/command")) {
      count += 1;
    }
  });
  return Object.freeze({ count: () => count });
}

/**
 * @param {import("@playwright/test").Page} page
 */
async function currentRevision(page) {
  return Number(await page.locator("#revision-value").textContent());
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} previous
 */
async function expectRevisionAdvance(page, previous) {
  await expect
    .poll(() => currentRevision(page), { timeout: 120_000 })
    .toBeGreaterThan(previous);
  await expect(page.locator("#connection-status")).toHaveText("Online");
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} selector
 * @param {string} value
 */
async function selectAuthoritativeValue(page, selector, value) {
  const control = page.locator(selector);
  if ((await control.inputValue()) === value) {
    return;
  }
  const revisionBefore = await currentRevision(page);
  await control.selectOption(value);
  await expectRevisionAdvance(page, revisionBefore);
  await expect(control).toHaveValue(value);
}

/**
 * Install a deterministic browser page and reset or switch the shared live
 * service to one exact visual case.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} debuggerUrl
 * @param {{
 *   scenario: string,
 *   view?: "researcher" | "pov",
 *   preset?: "presentation" | "analysis" | "debug",
 *   viewport?: {width: number, height: number},
 * }} options
 * @returns {Promise<CommandPostCounter>}
 */
export async function loadLiveVisualCase(
  page,
  debuggerUrl,
  { scenario, view = "researcher", preset = "analysis", viewport = DESKTOP_VIEWPORT },
) {
  await page.setViewportSize(viewport);
  await installWaapiAutopause(page);
  const commandPosts = trackCommandPosts(page);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const revisionBefore = await currentRevision(page);
  if ((await page.locator("#scenario-select").inputValue()) === scenario) {
    await page.getByRole("button", { name: "Reset" }).click();
  } else {
    await page.locator("#scenario-select").selectOption(scenario);
  }
  await expectRevisionAdvance(page, revisionBefore);
  await expect(page.locator("#scenario-select")).toHaveValue(scenario);
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(page.locator("#transition-value")).toHaveText("—");
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(0);
  await expect(page.locator(CHOREOGRAPHY_ROUTE_ROOT)).toHaveCount(0);

  await selectAuthoritativeValue(page, "#view-select", view);
  await selectAuthoritativeValue(page, "#preset-select", preset);
  return commandPosts;
}

/**
 * Advance from a freshly reset scripted scenario to one exact transition.
 * Intermediate explanations are skipped locally; the requested transition is
 * left installed for deterministic seeking by captureBaseline.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} targetTransition
 */
export async function advanceScriptTo(page, targetTransition) {
  if (!Number.isInteger(targetTransition) || targetTransition < 1) {
    throw new RangeError("targetTransition must be a positive integer.");
  }
  for (let transition = 1; transition <= targetTransition; transition += 1) {
    await page.locator("#battlefield").focus();
    await page.keyboard.press("n");
    await expect(page.locator("#transition-value")).toHaveText(
      new RegExp(`:transition:${transition - 1}$`),
      { timeout: 120_000 },
    );
    await expect(page.locator("#step-value")).toHaveText(String(transition));
    await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
    await expect(page.locator(CHOREOGRAPHY_ROUTE_ROOT)).toHaveCount(1);
    if (transition === targetTransition) {
      return;
    }
    const skip = page.locator("#motion-skip-button");
    await expect(skip).toBeEnabled();
    await skip.click();
    await expect(page.locator("html")).toHaveAttribute(
      "data-submission-blocked",
      "false",
    );
  }
}

/**
 * Install an explicitly synthetic renderer-only frame. No live command may
 * mutate or replace it.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} debuggerUrl
 * @param {Record<string, any>} frame
 * @param {{viewport?: {width: number, height: number}}} [options]
 * @returns {Promise<CommandPostCounter>}
 */
export async function installSyntheticVisualCase(
  page,
  debuggerUrl,
  frame,
  { viewport = DESKTOP_VIEWPORT } = {},
) {
  await page.setViewportSize(viewport);
  await installWaapiAutopause(page);
  const commandPosts = trackCommandPosts(page);
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
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  return commandPosts;
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{
 *   scenario: string | null,
 *   simulatorStep: number,
 *   transitionId: number | null,
 *   view: "researcher" | "pov",
 *   preset: "presentation" | "analysis" | "debug",
 *   badge: string | RegExp,
 * }} expected
 */
export async function assertFrameIdentity(page, expected) {
  const scenario = page.locator("#scenario-select");
  if (expected.scenario === null) {
    await expect(scenario).toHaveValue("");
    await expect(scenario).toBeDisabled();
  } else {
    await expect(scenario).toHaveValue(expected.scenario);
  }
  await expect(page.locator("#step-value")).toHaveText(String(expected.simulatorStep));
  await expect(page.locator("#transition-value")).toHaveText(
    expected.transitionId === null
      ? "—"
      : new RegExp(`:transition:${expected.transitionId - 1}$`),
  );
  await expect(page.locator("#view-select")).toHaveValue(expected.view);
  await expect(page.locator("#preset-select")).toHaveValue(expected.preset);
  await expect(page.locator("#audience-badge")).toHaveText(expected.badge);
  await expect(page.locator("html")).toHaveAttribute(
    "data-audience",
    expected.view === "pov" ? "agent_pov" : "researcher",
  );
  await expect(page.locator("html")).toHaveAttribute("data-preset", expected.preset);
  const revision = await currentRevision(page);
  expect(Number.isInteger(revision) && revision >= 0).toBe(true);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number[]} expectedSlots
 */
export async function expectRosterSlots(page, expectedSlots) {
  const rawSlots = await page
    .locator("#roster .roster-row")
    .evaluateAll((rows) => rows.map((row) => row.getAttribute("data-slot")));
  expect(rawSlots).not.toContain(null);
  for (const rawSlot of rawSlots) {
    expect(rawSlot).toMatch(/^(0|[1-9]\d*)$/);
  }
  const slots = rawSlots
    .map((rawSlot) => Number(rawSlot))
    .sort((left, right) => left - right);
  expect(slots).toEqual([...expectedSlots].sort((left, right) => left - right));
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} slot
 * @param {Array<{tokenId: string, duration: number}>} expectedStatuses
 */
export async function expectRosterStatuses(page, slot, expectedStatuses) {
  const statuses = await page
    .locator(`#roster .roster-row[data-slot="${slot}"] .roster-fact-token--status`)
    .evaluateAll((tokens) =>
      tokens.map((token) => ({
        duration: Number(token.getAttribute("data-duration")),
        tokenId: token.getAttribute("data-token-id"),
      })),
    );
  expect(statuses).toEqual(expectedStatuses);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {Array<{tokenId: string, source: number, target: number | null}>} expected
 */
export async function expectActivationPairs(page, expected) {
  const rawActivations = await page
    .locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`)
    .evaluateAll((nodes) =>
      nodes.map((node) => ({
        source: node.getAttribute("data-source-slot"),
        target: node.getAttribute("data-target-slot"),
        tokenId: node.getAttribute("data-token-id"),
      })),
    );
  for (const activation of rawActivations) {
    expect(activation.source).toMatch(/^(0|[1-9]\d*)$/);
    if (activation.target !== null) {
      expect(activation.target).toMatch(/^(0|[1-9]\d*)$/);
    }
    expect(activation.tokenId).toMatch(/^[a-z0-9_]+$/);
  }
  const activations = rawActivations
    .map(({ source, target, tokenId }) => ({
      source: Number(source),
      target: target === null ? null : Number(target),
      tokenId,
    }))
    .sort(
      (left, right) =>
        left.source - right.source ||
        (left.target ?? -1) - (right.target ?? -1) ||
        String(left.tokenId).localeCompare(String(right.tokenId)),
    );
  const normalizedExpected = [...expected].sort(
    (left, right) =>
      left.source - right.source ||
      (left.target ?? -1) - (right.target ?? -1) ||
      left.tokenId.localeCompare(right.tokenId),
  );
  expect(activations).toEqual(normalizedExpected);
}

/**
 * Prove the event feed and every rendered effect use unique IDs from the
 * current authoritative batch.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} expectedCount
 */
export async function assertCurrentEventIds(page, expectedCount) {
  await expect(page.locator("#event-count")).toHaveText(String(expectedCount));
  const feedIds = await page
    .locator("#event-feed [data-event-id]")
    .evaluateAll((items) => items.map((item) => item.getAttribute("data-event-id")));
  expect(feedIds).not.toContain(null);
  expect(feedIds).toHaveLength(expectedCount);
  expect(new Set(feedIds).size).toBe(feedIds.length);

  const feedIdSet = new Set(feedIds);
  const renderedIds =
    (await page.locator(CHOREOGRAPHY_ROOT).count()) === 0
      ? []
      : [
          ...(await choreographySnapshot(page)).effectIds,
          ...(await choreographySnapshot(page)).routeEffectIds,
        ];
  expect(renderedIds).not.toContain(null);
  for (const eventId of renderedIds) {
    expect(feedIdSet.has(eventId)).toBe(true);
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{pending: string, accepted: string}} expected
 */
export async function assertHudStoryLabels(page, expected) {
  await expect(page.locator("#pending-card .action-card__label")).toHaveText(
    expected.pending,
  );
  if (expected.accepted === "No transition yet.") {
    await expect(page.locator("#accepted-card .empty-copy")).toHaveText(
      expected.accepted,
    );
  } else {
    await expect(page.locator("#accepted-card .action-card__label")).toHaveText(
      expected.accepted,
    );
  }
}

/**
 * Every durable dock that survived the deterministic suppression policy must
 * explicitly report collision-free placement.
 *
 * @param {import("@playwright/test").Page} page
 */
export async function assertDurableDockFlags(page) {
  const flags = await page
    .locator(
      "#battlefield .status-dock, #battlefield .modifier-dock, #battlefield .cooldown-dock, #battlefield .legality-dock",
    )
    .evaluateAll((docks) =>
      docks.map((dock) => ({
        collisionFree: dock.getAttribute("data-collision-free"),
        kind: [...dock.classList].find((name) => name.endsWith("-dock")) ?? "dock",
        slot: dock.getAttribute("data-slot"),
      })),
    );
  expect(flags.filter(({ collisionFree }) => collisionFree !== "true")).toEqual([]);
}

/**
 * Wait until the actual captured geometry, not merely elapsed RAF count, is
 * unchanged across two consecutive frame-to-frame comparisons.
 *
 * @param {import("@playwright/test").Page} page
 */
export async function waitForStablePresentation(page) {
  await page.evaluate(async () => {
    await document.fonts.ready;
    const scrolling = document.scrollingElement;
    if (scrolling) {
      scrolling.scrollTop = 0;
      scrolling.scrollLeft = 0;
    }
    const hud = document.querySelector(".hud-panel");
    if (hud instanceof HTMLElement) {
      hud.scrollTop = 0;
      hud.scrollLeft = 0;
    }
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }

    /**
     * @param {Element} element
     */
    const elementFingerprint = (element) => {
      const bounds = element.getBoundingClientRect();
      const identity = [
        element.id,
        element.getAttribute("data-slot"),
        element.getAttribute("data-event-id"),
        element.getAttribute("data-token-id"),
        element.getAttribute("data-anchor"),
      ].join(":");
      return [
        element instanceof SVGElement ? element.className.baseVal : element.className,
        identity,
        bounds.left.toFixed(3),
        bounds.top.toFixed(3),
        bounds.width.toFixed(3),
        bounds.height.toFixed(3),
        element instanceof HTMLElement ? element.scrollWidth : 0,
        element instanceof HTMLElement ? element.scrollHeight : 0,
      ].join("|");
    };

    const measure = () => {
      const battlefield = document.querySelector("#battlefield");
      if (!(battlefield instanceof SVGSVGElement)) {
        throw new Error("Battlefield SVG is unavailable.");
      }
      const battlefieldBounds = battlefield.getBoundingClientRect();
      const geometry = [
        ...battlefield.querySelectorAll(
          ".agent, .status-dock, .modifier-dock, .cooldown-dock, .legality-dock, .combat-effect, .combat-route-effect",
        ),
      ]
        .map(elementFingerprint)
        .sort();
      const interfaceGeometry = [
        ...document.querySelectorAll(
          [
            ".app-header",
            ".session-toolbar",
            ".session-facts",
            ".motion-controls",
            ".toolbar-actions",
            ".workspace",
            ".battlefield-panel",
            ".panel-heading",
            ".battlefield-shell",
            ".command-deck",
            ".hud-panel",
            ".hud-section",
            ".roster-row",
            ".roster-fact-token",
            ".action-card",
            ".event-item",
          ].join(","),
        ),
      ]
        .map(elementFingerprint)
        .sort();
      const viewportKeys = [
        battlefield
          .querySelector(".combat-choreography")
          ?.getAttribute("data-viewport-key") ?? "rootless",
        battlefield
          .querySelector(".combat-choreography-routes")
          ?.getAttribute("data-viewport-key") ?? "rootless",
      ];
      return JSON.stringify({
        battlefield: [
          battlefieldBounds.left.toFixed(3),
          battlefieldBounds.top.toFixed(3),
          battlefieldBounds.width.toFixed(3),
          battlefieldBounds.height.toFixed(3),
          battlefield.getAttribute("viewBox"),
        ],
        geometry,
        interfaceGeometry,
        viewportKeys,
      });
    };

    let previous = "";
    let stableComparisons = 0;
    for (let attempt = 0; attempt < 16; attempt += 1) {
      await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
      const current = measure();
      if (current === previous) {
        stableComparisons += 1;
        if (stableComparisons >= 2) {
          return;
        }
      } else {
        stableComparisons = 0;
        previous = current;
      }
    }
    throw new Error(
      "Page shell, HUD, roster, battlefield, dock, and choreography geometry did not stabilize.",
    );
  });
}

/**
 * Prove every transient recipient-labelled NET cue stays inside the
 * battlefield and outside the protected visual zones. Leader lines are
 * association cues, not protected geometry; only their ownership and finite
 * endpoints are validated.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} expectedCount
 */
export async function assertTransientNumberLayout(page, expectedCount) {
  const result = await page.evaluate(() => {
    const tolerance = 0.75;
    const battlefield = document.querySelector("#battlefield");
    const mapBoundary = battlefield?.querySelector(".map-boundary");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(mapBoundary instanceof SVGElement)
    ) {
      throw new Error("Battlefield map boundary is unavailable.");
    }

    /**
     * @typedef {{
     *   bottom: number,
     *   left: number,
     *   right: number,
     *   top: number,
     * }} Bounds
     */
    /**
     * @param {DOMRect} rect
     * @param {number} padding
     * @returns {Bounds}
     */
    const expand = (rect, padding) => ({
      bottom: rect.bottom + padding,
      left: rect.left - padding,
      right: rect.right + padding,
      top: rect.top - padding,
    });
    /**
     * @param {Bounds} first
     * @param {Bounds} second
     */
    const overlap = (first, second) => ({
      x: Math.min(first.right, second.right) - Math.max(first.left, second.left),
      y: Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top),
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
          ".combat-impact__icon",
          ".combat-local__icon",
          ".combat-lifecycle__status-icon",
          ".combat-lifecycle__change-icon",
          ".combat-lifecycle__reapply-icon",
        ].join(","),
      ),
    ].filter(isPainted);
    const protectedRects = protectedElements.map((element) => ({
      bounds: expand(element.getBoundingClientRect(), 0),
      selector:
        [...element.classList].map((name) => `.${name}`).join("") ||
        element.tagName.toLowerCase(),
    }));
    const mapBounds = expand(mapBoundary.getBoundingClientRect(), 0);
    const labelRecords = [
      ...battlefield.querySelectorAll(
        '.combat-effect--net-health[data-spatial-disposition="rendered"] .combat-net__label',
      ),
    ].map((label) => {
      const effect = label.closest(".combat-effect--net-health");
      if (!(effect instanceof SVGElement)) {
        throw new Error("NET label is detached from its semantic event.");
      }
      const recipientLabel = effect.querySelector(".combat-net__recipient");
      const strokeWidth = Number.parseFloat(getComputedStyle(label).strokeWidth);
      const padding =
        Number.isFinite(strokeWidth) && strokeWidth > 0 ? strokeWidth / 2 : 2;
      const rect = label.getBoundingClientRect();
      const recipientRect = recipientLabel?.getBoundingClientRect() ?? null;
      const cueRect =
        recipientRect === null
          ? rect
          : {
              bottom: Math.max(rect.bottom, recipientRect.bottom),
              left: Math.min(rect.left, recipientRect.left),
              right: Math.max(rect.right, recipientRect.right),
              top: Math.min(rect.top, recipientRect.top),
            };
      return {
        bounds: {
          bottom: cueRect.bottom + padding,
          left: cueRect.left - padding,
          right: cueRect.right + padding,
          top: cueRect.top - padding,
        },
        eventId: effect.getAttribute("data-event-id"),
        painted: isPainted(label),
        recipientLabelPainted:
          recipientLabel instanceof Element && isPainted(recipientLabel),
        recipientLabelText: recipientLabel?.textContent ?? null,
        recipientRect,
        recipientSlot: effect.getAttribute("data-recipient-slot"),
        rect,
      };
    });

    const paintedLabelRecords = labelRecords.filter(
      (record) => record.painted || record.recipientLabelPainted,
    );
    const violations = [];
    for (const record of paintedLabelRecords) {
      if (
        !record.eventId ||
        !/^(0|[1-9]\d*)$/.test(record.recipientSlot ?? "") ||
        !record.painted ||
        !record.recipientLabelPainted ||
        record.recipientLabelText !== `id_${record.recipientSlot}` ||
        record.recipientRect === null ||
        ![
          record.rect.left,
          record.rect.top,
          record.rect.width,
          record.rect.height,
          record.recipientRect?.left,
          record.recipientRect?.top,
          record.recipientRect?.width,
          record.recipientRect?.height,
        ].every(Number.isFinite) ||
        record.rect.width <= 0 ||
        record.rect.height <= 0 ||
        (record.recipientRect?.width ?? 0) <= 0 ||
        (record.recipientRect?.height ?? 0) <= 0
      ) {
        violations.push({
          eventId: record.eventId,
          recipientSlot: record.recipientSlot,
          protectedSelector: "self",
          reason: "NET label has incomplete identity or non-measurable geometry",
        });
        continue;
      }
      const ownTextOverlap = overlap(record.rect, record.recipientRect);
      if (ownTextOverlap.x > tolerance && ownTextOverlap.y > tolerance) {
        violations.push({
          eventId: record.eventId,
          recipientSlot: record.recipientSlot,
          protectedSelector: ".combat-net__recipient",
          overlap: ownTextOverlap,
          reason: "NET value overlaps its recipient identity",
        });
      }
      const effect = battlefield.querySelector(
        `.combat-effect--net-health[data-event-id="${CSS.escape(record.eventId)}"]`,
      );
      if (effect?.getAttribute("data-layout-collision-free") !== "true") {
        violations.push({
          eventId: record.eventId,
          recipientSlot: record.recipientSlot,
          protectedSelector: "self",
          reason: "NET event reports a layout collision",
        });
      }
      if (
        record.bounds.left < mapBounds.left - tolerance ||
        record.bounds.top < mapBounds.top - tolerance ||
        record.bounds.right > mapBounds.right + tolerance ||
        record.bounds.bottom > mapBounds.bottom + tolerance
      ) {
        violations.push({
          eventId: record.eventId,
          recipientSlot: record.recipientSlot,
          protectedSelector: ".map-boundary",
          reason: "NET label escapes the map",
        });
      }
      for (const protectedRect of protectedRects) {
        const depth = overlap(record.bounds, protectedRect.bounds);
        if (depth.x > tolerance && depth.y > tolerance) {
          violations.push({
            eventId: record.eventId,
            recipientSlot: record.recipientSlot,
            protectedSelector: protectedRect.selector,
            overlap: depth,
            reason: "NET label overlaps protected geometry",
          });
        }
      }

      const effectGroup = battlefield.querySelector(
        `.combat-effect--net-health[data-event-id="${CSS.escape(record.eventId)}"]`,
      );
      const leader = effectGroup?.querySelector(".combat-cue__leader");
      if (leader && getComputedStyle(leader).visibility !== "hidden") {
        if (leader.closest(".combat-effect--net-health") !== effectGroup) {
          violations.push({
            eventId: record.eventId,
            recipientSlot: record.recipientSlot,
            protectedSelector: ".combat-cue__leader",
            reason: "NET leader belongs to another event",
          });
        }
        const rawEndpoints = ["x1", "y1", "x2", "y2"].map((name) =>
          leader.getAttribute(name),
        );
        const endpoints = rawEndpoints.map(Number);
        if (
          rawEndpoints.some((value) => value === null || value.trim() === "") ||
          !endpoints.every(Number.isFinite)
        ) {
          violations.push({
            eventId: record.eventId,
            recipientSlot: record.recipientSlot,
            protectedSelector: ".combat-cue__leader",
            reason: "Visible NET leader has non-finite endpoints",
          });
        }
      }
    }

    for (let index = 0; index < paintedLabelRecords.length; index += 1) {
      for (let other = index + 1; other < paintedLabelRecords.length; other += 1) {
        const depth = overlap(
          paintedLabelRecords[index].bounds,
          paintedLabelRecords[other].bounds,
        );
        if (depth.x > tolerance && depth.y > tolerance) {
          violations.push({
            eventId: paintedLabelRecords[index].eventId,
            recipientSlot: paintedLabelRecords[index].recipientSlot,
            protectedSelector: `NET ${paintedLabelRecords[other].eventId}`,
            overlap: depth,
            reason: "NET labels overlap",
          });
        }
      }
    }
    return { paintedCount: paintedLabelRecords.length, violations };
  });
  expect(result.paintedCount).toBe(expectedCount);
  expect(result.violations).toEqual([]);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--net-health[data-spatial-disposition="rendered"][data-layout-collision-free="false"]`,
    ),
  ).toHaveCount(0);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-spatial-disposition="rendered"] .combat-lifecycle[data-layout-collision-free="false"]`,
    ),
  ).toHaveCount(0);
}

/**
 * Collision suppression remains explicit, retained, and hidden. Curated cases
 * default to no suppressed lifecycle effects; NET suppression is never allowed
 * in an accepted visual baseline.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{lifecycle?: number, lifecycleIds?: string[] | null, net?: number}} [expected]
 */
export async function assertOutcomeSuppression(
  page,
  { lifecycle = 0, lifecycleIds = null, net = 0 } = {},
) {
  const suppressedNet = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--net-health[data-spatial-disposition="suppressed-collision"]`,
  );
  const suppressedLifecycle = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-spatial-disposition="suppressed-collision"]`,
  );
  if (lifecycleIds !== null) {
    const observedIds = (
      await suppressedLifecycle.evaluateAll((groups) =>
        groups.map((group) => group.getAttribute("data-event-id")),
      )
    ).sort();
    expect(observedIds).toEqual([...lifecycleIds].sort());
  }
  await expect(suppressedNet).toHaveCount(net);
  await expect(suppressedLifecycle).toHaveCount(lifecycle);
  const visibleSuppressions = await page
    .locator(`${CHOREOGRAPHY_ROOT} [data-spatial-disposition="suppressed-collision"]`)
    .evaluateAll((groups) =>
      groups
        .filter((group) => getComputedStyle(group).visibility !== "hidden")
        .map((group) => group.getAttribute("data-event-id")),
    );
  expect(visibleSuppressions).toEqual([]);
}

/**
 * Human-facing floating-point labels are capped at two decimal places.
 * Collapsed technical JSON remains outside this presentation assertion.
 *
 * @param {import("@playwright/test").Page} page
 */
export async function assertVisibleDecimalPrecision(page) {
  const overPrecise = await page.locator("body").evaluate((body) => {
    const text = /** @type {HTMLElement} */ (body).innerText;
    return [...text.matchAll(/(?:^|[^\d])\d+\.(\d{3,})/g)].map((match) =>
      match[0].trim(),
    );
  });
  expect(overPrecise).toEqual([]);
}

/**
 * Seek or settle one case, prove presentation-only work sent no command, wait
 * for observable geometry stability, run collision checks, and compare the
 * viewport against its reviewed baseline.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} snapshotName
 * @param {{
 *   commandPosts: CommandPostCounter,
 *   expectedTransientCount: number,
 *   expectedSuppressedLifecycleCount?: number,
 *   expectedSuppressedLifecycleIds?: string[] | null,
 *   logicalMs?: number,
 *   settle?: boolean,
 *   afterSettle?: () => Promise<void>,
 * }} options
 */
export async function captureBaseline(
  page,
  snapshotName,
  {
    commandPosts,
    expectedTransientCount,
    expectedSuppressedLifecycleCount = 0,
    expectedSuppressedLifecycleIds = null,
    logicalMs,
    settle = false,
    afterSettle,
  },
) {
  const commandCountBeforePresentation = commandPosts.count();
  if (logicalMs !== undefined) {
    await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
    await pauseAtLogicalTime(page, logicalMs);
  } else if (settle) {
    const skip = page.locator("#motion-skip-button");
    await expect(skip).toBeEnabled();
    await skip.click();
    await expect(page.locator("html")).toHaveAttribute(
      "data-submission-blocked",
      "false",
    );
    await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveAttribute(
      "data-state",
      "settled",
    );
  }

  await waitForStablePresentation(page);
  if (afterSettle) {
    await afterSettle();
    await waitForStablePresentation(page);
  }
  expect(commandPosts.count()).toBe(commandCountBeforePresentation);
  await assertDurableDockFlags(page);
  await assertTransientNumberLayout(page, expectedTransientCount);
  await assertOutcomeSuppression(page, {
    lifecycle: expectedSuppressedLifecycleCount,
    lifecycleIds: expectedSuppressedLifecycleIds,
  });
  await assertVisibleDecimalPrecision(page);
  const revision = await currentRevision(page);
  expect(Number.isInteger(revision) && revision >= 0).toBe(true);
  await expect(page).toHaveScreenshot(snapshotName, {
    animations: "allow",
    stylePath: SNAPSHOT_STYLE_PATH,
  });
  expect(commandPosts.count()).toBe(commandCountBeforePresentation);
}
