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
      if (this.closest(".combat-choreography")) {
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
  await root.waitFor();
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
    await Promise.all(animations.map(({ ready }) => ready));
    for (const animation of animations) {
      animation.pause();
      animation.currentTime = time;
    }
    await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
  }, logicalMs);
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
  const eventCount = Number(await page.locator("#event-count").textContent());
  const snapshot = await choreographySnapshot(page);
  const effectIds = snapshot.effectIds.filter((eventId) => eventId !== null);
  expect(new Set(effectIds).size).toBe(effectIds.length);
  expect(effectIds.length).toBeLessThanOrEqual(eventCount);

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
  expect(nodeCount).toBeLessThanOrEqual(Math.min(eventCount * 28 + 2, 512));
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
  const rosterSlots = await page
    .locator("#roster .roster-row")
    .evaluateAll((rows) => rows.map((row) => Number(row.getAttribute("data-slot"))));
  const transientSlots = await page.locator(CHOREOGRAPHY_ROOT).evaluate((root) => {
    const values = [];
    const battlefield = root.closest("svg");
    const elements = battlefield?.querySelectorAll(
      ".combat-effect, .combat-route-effect",
    );
    for (const element of elements ?? []) {
      for (const name of element.getAttributeNames()) {
        if (name === "data-slot" || /^data-.+-slot$/.test(name)) {
          const value = Number(element.getAttribute(name));
          if (Number.isInteger(value)) {
            values.push(value);
          }
        }
      }
    }
    return values;
  });
  for (const slot of transientSlots) {
    expect(rosterSlots).toContain(slot);
  }
}
