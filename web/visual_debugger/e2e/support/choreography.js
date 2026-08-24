import { expect } from "@playwright/test";

export const CHOREOGRAPHY_ROOT =
  '#battlefield [data-layer="transient-events"] > .combat-choreography';
export const CHOREOGRAPHY_ROUTE_ROOT =
  '#battlefield [data-layer="transient-route"] > .combat-choreography-routes';

/**
 * Pause CP5-owned Web Animations at creation time so browser assertions can
 * seek one deterministic shared presentation clock without production hooks.
 *
 * @param {import("@playwright/test").Page} page
 */
export async function installWaapiAutopause(page) {
  await page.addInitScript(() => {
    const nativeAnimate = Element.prototype.animate;
    Element.prototype.animate = function animateAndPause(keyframes, options) {
      const animation = nativeAnimate.call(this, keyframes, options);
      if (this.closest(".combat-choreography, .combat-choreography-routes")) {
        animation.pause();
      }
      return animation;
    };
  });
}

/**
 * Synchronize the visible controller state and seek every owned animation to
 * the same logical time.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} logicalMs
 */
export async function pauseAtLogicalTime(page, logicalMs) {
  const root = page.locator(CHOREOGRAPHY_ROOT);
  const routeRoot = page.locator(CHOREOGRAPHY_ROUTE_ROOT);
  await Promise.all([
    root.waitFor({ state: "attached" }),
    routeRoot.waitFor({ state: "attached" }),
  ]);
  if ((await page.locator("html").getAttribute("data-motion-paused")) !== "true") {
    await page.locator("#motion-pause-button").click();
  }
  await root.evaluate(async (element, time) => {
    const battlefield = element.closest("svg");
    if (!battlefield) {
      throw new Error("Missing choreography battlefield.");
    }
    const animations = battlefield
      .getAnimations({ subtree: true })
      .filter(({ id }) => id.startsWith("mbg:"));
    await document.fonts.ready;
    await Promise.all(animations.map(({ ready }) => ready));
    for (const animation of animations) {
      animation.pause();
      animation.currentTime = time;
    }
    await Promise.all(animations.map(({ ready }) => ready));
    for (let frame = 0; frame < 3; frame += 1) {
      await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    }
    for (const animation of animations) {
      if (animation.playState !== "paused" || Number(animation.currentTime) !== time) {
        throw new Error(`Animation ${animation.id} did not settle at ${time} ms.`);
      }
    }
  }, logicalMs);
}

/**
 * Resolve one readable event-owned animation window. Some event rows are
 * intentionally durable-only and therefore own no transient animation. Walk
 * the rendered rows in DOM order and select the first row with the requested
 * animation part instead of assuming the first row is animated.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{eventType?: string, part: "auto" | "group" | "route", progress: number}} request
 */
async function resolveReadableEventWindow(page, request) {
  const root = page.locator(CHOREOGRAPHY_ROOT);
  await root.waitFor({ state: "attached" });
  return root.evaluate((element, requested) => {
    const battlefield = element.closest("svg");
    const epochKey = element.getAttribute("data-epoch-key");
    if (!battlefield || !epochKey) {
      throw new Error("Event animation authority is unavailable.");
    }

    const parts = requested.part === "auto" ? ["group", "route"] : [requested.part];
    const animations = battlefield.getAnimations({ subtree: true });
    const effects = [
      ...element.querySelectorAll(".combat-effect[data-event-type][data-event-id]"),
    ].filter(
      (effect) =>
        requested.eventType === undefined ||
        effect.getAttribute("data-event-type") === requested.eventType,
    );

    for (const effect of effects) {
      const eventId = effect.getAttribute("data-event-id");
      const eventType = effect.getAttribute("data-event-type");
      if (!eventId || !eventType) {
        continue;
      }
      for (const part of parts) {
        const expectedId = `mbg:${epochKey}:${eventId}:${part}`;
        const animation = animations.find(({ id }) => id === expectedId);
        const timing = animation?.effect?.getTiming();
        const delay = Number(timing?.delay);
        const duration = Number(timing?.duration);
        if (Number.isFinite(delay) && duration > 0) {
          return {
            eventId,
            eventType,
            logicalMs: delay + duration * requested.progress,
            part,
          };
        }
      }
    }

    const eventDescription =
      requested.eventType === undefined
        ? "any rendered event"
        : `event type ${requested.eventType}`;
    throw new Error(
      `Missing readable ${parts.join(" or ")} animation window for ${eventDescription}.`,
    );
  }, request);
}

/**
 * Seek inside the installed animation for one exact event family. Dynamic
 * choreography omits absent families, so browser proofs derive their sample
 * point from the authored WAAPI window instead of duplicating one static
 * transition schedule.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} eventType
 * @param {{part?: "auto" | "group" | "route", progress?: number}} [options]
 */
export async function pauseInsideEventWindow(
  page,
  eventType,
  { part = "auto", progress = 0.5 } = {},
) {
  if (typeof eventType !== "string" || eventType.length === 0) {
    throw new TypeError("eventType must be a non-empty string.");
  }
  if (!new Set(["auto", "group", "route"]).has(part)) {
    throw new TypeError("part must be auto, group, or route.");
  }
  if (!(typeof progress === "number" && progress > 0 && progress < 1)) {
    throw new RangeError("progress must lie strictly between zero and one.");
  }
  const resolved = await resolveReadableEventWindow(page, {
    eventType,
    part,
    progress,
  });
  await pauseAtLogicalTime(page, resolved.logicalMs);
  return resolved.logicalMs;
}

/**
 * Seek the first rendered event that owns a readable animation window.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{part?: "auto" | "group" | "route", progress?: number}} [options]
 */
export async function pauseInsideFirstEventWindow(
  page,
  { part = "auto", progress = 0.5 } = {},
) {
  if (!new Set(["auto", "group", "route"]).has(part)) {
    throw new TypeError("part must be auto, group, or route.");
  }
  if (!(typeof progress === "number" && progress > 0 && progress < 1)) {
    throw new RangeError("progress must lie strictly between zero and one.");
  }
  const resolved = await resolveReadableEventWindow(page, { part, progress });
  await pauseAtLogicalTime(page, resolved.logicalMs);
  return resolved.logicalMs;
}

/**
 * Finish one controller-owned sentinel clock without wall-clock waiting.
 *
 * @param {import("@playwright/test").Page} page
 * @param {"gate" | "cleanup"} suffix
 */
export async function finishControllerClock(page, suffix) {
  await page.locator(CHOREOGRAPHY_ROOT).evaluate((root, clockSuffix) => {
    const clock = root
      .getAnimations({ subtree: true })
      .find(({ id }) => id.endsWith(`:${clockSuffix}`));
    if (!clock) {
      throw new Error(`Missing ${clockSuffix} clock.`);
    }
    clock.finish();
  }, suffix);
}

/**
 * @param {import("@playwright/test").Page} page
 */
export async function choreographySnapshot(page) {
  return page.locator(CHOREOGRAPHY_ROOT).evaluate((root) => ({
    animationIds: (root.closest("svg") ?? root)
      .getAnimations({ subtree: true })
      .filter(({ id }) => id.startsWith("mbg:"))
      .map(({ currentTime, id, playbackRate, playState }) => ({
        currentTime: Number(currentTime),
        id,
        playbackRate,
        playState,
      }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    authorizationKey: root.getAttribute("data-authorization-key"),
    effectIds: [...root.querySelectorAll(".combat-effect")].map((effect) =>
      effect.getAttribute("data-event-id"),
    ),
    routeEffectIds: [
      ...(root
        .closest("svg")
        ?.querySelectorAll(
          '[data-layer="transient-route"] > .combat-choreography-routes > .combat-route-effect',
        ) ?? []),
    ].map((effect) => effect.getAttribute("data-event-id")),
    epochKey: root.getAttribute("data-epoch-key"),
    fingerprint: root.getAttribute("data-event-fingerprint"),
    state: root.getAttribute("data-state"),
    viewportKey: root.getAttribute("data-viewport-key"),
  }));
}

/**
 * Prove the retained subtree and its Web Animations remain within the public
 * per-batch limits. This complements, rather than duplicates, planner units.
 *
 * @param {import("@playwright/test").Page} page
 */
export async function assertBoundedChoreography(page) {
  const roots = page.locator(CHOREOGRAPHY_ROOT);
  const routeRoots = page.locator(CHOREOGRAPHY_ROUTE_ROOT);
  await expect(roots).toHaveCount(1);
  await expect(routeRoots).toHaveCount(1);
  const snapshot = await choreographySnapshot(page);
  const effectIds = snapshot.effectIds.filter((eventId) => eventId !== null);
  expect(new Set(effectIds).size).toBe(effectIds.length);
  expect(snapshot.routeEffectIds).not.toContain(null);
  expect(new Set(snapshot.routeEffectIds).size).toBe(snapshot.routeEffectIds.length);
  for (const routeEffectId of snapshot.routeEffectIds) {
    expect(effectIds).toContain(routeEffectId);
  }

  const nodeCount = await roots.evaluate(
    (root) =>
      root.querySelectorAll("*").length +
      1 +
      (root
        .closest("svg")
        ?.querySelector('[data-layer="transient-route"] > .combat-choreography-routes')
        ?.querySelectorAll("*").length ?? 0) +
      1,
  );
  expect(nodeCount).toBeLessThanOrEqual(Math.min(effectIds.length * 28 + 2, 512));
  expect(snapshot.animationIds.length).toBeLessThanOrEqual(
    Math.min(effectIds.length * 3 + 2, 512),
  );
  expect(new Set(snapshot.animationIds.map(({ id }) => id)).size).toBe(
    snapshot.animationIds.length,
  );
}

/**
 * Every transient slot reference must belong to the currently authorized
 * roster. Absence remains valid for redacted endpoints.
 *
 * @param {import("@playwright/test").Page} page
 */
export async function assertTransientSlotsAuthorized(page) {
  const rawRosterSlots = await page
    .locator("#roster .roster-row")
    .evaluateAll((rows) => rows.map((row) => row.getAttribute("data-slot")));
  expect(rawRosterSlots).not.toContain(null);
  for (const rawSlot of rawRosterSlots) {
    expect(rawSlot).toMatch(/^(0|[1-9]\d*)$/);
  }
  const rosterSlots = rawRosterSlots.map(Number);
  const transientSlots = await page.locator(CHOREOGRAPHY_ROOT).evaluate((root) => {
    const values = [];
    const battlefield = root.closest("svg");
    const elements = battlefield?.querySelectorAll(
      ".combat-effect, .combat-route-effect",
    );
    for (const element of elements ?? []) {
      for (const name of element.getAttributeNames()) {
        if (name === "data-slot" || /^data-.+-slot$/.test(name)) {
          const rawValue = element.getAttribute(name);
          if (!/^(0|[1-9]\d*)$/.test(rawValue ?? "")) {
            throw new Error(`Invalid transient slot attribute ${name}.`);
          }
          values.push(Number(rawValue));
        }
      }
    }
    return values;
  });
  for (const slot of transientSlots) {
    expect(rosterSlots).toContain(slot);
  }
}
