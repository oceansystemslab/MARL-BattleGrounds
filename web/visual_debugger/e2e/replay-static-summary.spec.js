import { expect, test } from "@playwright/test";

import { VISUAL_FILTER_IDS } from "../src/visual-filters.js";
import { CHOREOGRAPHY_ROOT, CHOREOGRAPHY_ROUTE_ROOT } from "./support/choreography.js";
import {
  expectReplayFrameIndex,
  startReplayViewer,
  stopDebugger,
} from "./support/replay-viewer.js";
import { DESKTOP_VIEWPORT, MINIMUM_VIEWPORT } from "./support/visual-regression.js";

test.describe.configure({ mode: "serial" });
test.use({ screenshot: "off" });

const FILTER_INPUT = 'input[type="checkbox"][data-visual-filter-id]';
const SELECTIVE_FILTER = "status_refresh_extension";
const SPATIAL_HIT_SELECTOR = [
  ".combat-impact__hit",
  ".combat-local__hit",
  ".combat-net__hit",
  ".combat-regeneration__hit",
  ".combat-charge__endpoint",
  ".combat-lifecycle__hit",
  ".combat-semantic-pulse__hit",
  ".combat-lifecycle-ring__hit",
  ".combat-respawn-wave__panel",
  ".combat-rejection__hit",
].join(", ");
const TRANSIENT_EVENT_FAMILIES = Object.freeze([
  [
    ".combat-effect--activation[data-tooltip-owner]",
    ".combat-impact__hit",
    "activation",
  ],
  [".combat-effect--net-health[data-tooltip-owner]", ".combat-net__hit", "impact"],
  [
    ".combat-effect--regeneration[data-tooltip-owner]",
    ".combat-regeneration__hit",
    "event",
  ],
  [
    ".combat-effect--semantic-pulse[data-tooltip-owner]",
    ".combat-semantic-pulse__hit",
    "event",
  ],
  [
    ".combat-effect--status-lifecycle[data-tooltip-owner]",
    ".combat-lifecycle__hit",
    "event",
  ],
]);

/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let replay = null;

test.beforeAll(async () => {
  replay = await startReplayViewer({
    sampleReplay: "recovery-status-lifecycle",
    frameIndex: 2,
    view: "researcher",
    preset: "analysis",
    ranges: false,
  });
});

test.afterAll(async () => {
  const child = replay?.process ?? null;
  replay = null;
  await stopDebugger(child);
});

/** @param {import("@playwright/test").Page} page */
function captureBrowserErrors(page) {
  /** @type {string[]} */
  const errors = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
  return errors;
}

/** @param {import("@playwright/test").Page} page */
function captureApiRequests(page) {
  /** @type {{method: string, path: string}[]} */
  const requests = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.startsWith("/api/")) {
      requests.push({ method: request.method(), path });
    }
  });
  return requests;
}

/** @param {import("@playwright/test").Page} page */
async function settleStaticSummary(page) {
  await page.evaluate(async () => {
    await document.fonts.ready;
    for (let frame = 0; frame < 3; frame += 1) {
      await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    }
  });
  await expect(page.locator("html")).toHaveAttribute(
    "data-render-policy",
    "replay_static",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "data-render-policy",
    "replay_static",
  );
  for (const selector of [CHOREOGRAPHY_ROOT, CHOREOGRAPHY_ROUTE_ROOT]) {
    const root = page.locator(selector);
    await expect(root).toHaveCount(1);
    await expect(root).toHaveAttribute("role", "group");
    await expect(root).toHaveAttribute("aria-label", /\S/u);
    await expect(root).toHaveAttribute("data-render-policy", "replay_static");
    await expect(root).toHaveAttribute("data-state", "settled");
  }
  expect(
    await page.evaluate(() =>
      document
        .getAnimations()
        .map(({ id }) => id)
        .filter((id) => id.startsWith("mbg:")),
    ),
  ).toEqual([]);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} path
 */
async function authenticatedText(page, path) {
  return page.evaluate(async (requestPath) => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Replay capability token is unavailable.");
    }
    const response = await fetch(requestPath, {
      cache: "no-store",
      credentials: "omit",
      headers: { "X-MARL-Debugger-Token": token },
      redirect: "error",
    });
    if (!response.ok) {
      throw new Error(`${requestPath} failed with HTTP ${response.status}.`);
    }
    return response.text();
  }, path);
}

/** @param {import("@playwright/test").Page} page */
async function scientificSignature(page) {
  const paths = ["/api/frame", "/api/presentation/frame", "/api/replay/timeline"];
  const api = Object.fromEntries(
    await Promise.all(
      paths.map(async (path) => [path, await authenticatedText(page, path)]),
    ),
  );
  return {
    api,
    eventFeed: await page.locator("#event-feed").innerHTML(),
  };
}

/** @param {import("@playwright/test").Page} page */
async function openVisualFilters(page) {
  await page.locator("#visual-filters").evaluate((details) => {
    if (!(details instanceof HTMLDetailsElement)) {
      throw new TypeError("Visual Filters disclosure is unavailable.");
    }
    details.open = true;
  });
  await expect(page.locator("#visual-filter-options")).toBeVisible();
}

/**
 * Derive the exact enabled spatial identity from the same pure planner used by
 * the product. Geometry uses a roomy synthetic surface because this helper
 * audits identity/filter ownership; the real DOM below owns viewport geometry.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} rawPresentation
 * @param {string[]} disabledFilters
 */
async function expectedPlanSignature(page, rawPresentation, disabledFilters = []) {
  return page.evaluate(
    async ({ disabled, raw }) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { authorizedPresentationIncomingRows } = await import(
        `${moduleRoot}/authorized-presentation-adapter.js`
      );
      const { buildChoreographyPlan } = await import(
        `${moduleRoot}/choreography-plan.js`
      );
      const { VISUAL_FILTER_IDS: rawFilterIds } = await import(
        `${moduleRoot}/visual-filters.js`
      );
      const filterIds = /** @type {string[]} */ (rawFilterIds);
      const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
      const filters = Object.fromEntries(
        filterIds.map((id) => [id, !disabled.includes(id)]),
      );
      const surface = Object.freeze({
        /** @param {{x: number, y: number} | number[]} point */
        worldToScreen: (point) => {
          const x = Array.isArray(point) ? point[0] : point.x;
          const y = Array.isArray(point) ? point[1] : point.y;
          return { x: Number(x) * 40 + 512, y: Number(y) * 40 + 512 };
        },
        /** @param {number} length */
        worldLengthToScreen: (length) => Number(length) * 40,
        viewportBounds: Object.freeze({
          left: 0,
          top: 0,
          right: 4096,
          bottom: 4096,
          width: 4096,
          height: 4096,
        }),
        protectedRects: Object.freeze([]),
      });
      const rows = /** @type {Array<Record<string, any>>} */ (
        authorizedPresentationIncomingRows(presentation)
      );
      const plan = buildChoreographyPlan(
        presentation,
        /** @type {any} */ (surface),
        filters,
        "replay_static",
      );
      if (!plan) {
        throw new Error("Registered replay produced no choreography plan.");
      }
      /** @param {Record<string, any>} event */
      const atoms = (event) =>
        Array.isArray(event.atomicEventIds) ? event.atomicEventIds : [event.eventId];
      const events = /** @type {Array<Record<string, any>>} */ (plan.events);
      const layoutKeys = events
        .flatMap((event) =>
          Object.entries(event)
            .filter(
              ([name, value]) =>
                name.endsWith("LayoutKey") &&
                typeof value === "string" &&
                value.length > 0,
            )
            .map(([, value]) => String(value)),
        )
        .sort((left, right) => left.localeCompare(right));
      return {
        authorizationKey: plan.authorizationKey,
        epochKey: plan.epochKey,
        fingerprint: plan.fingerprint,
        orderedIds: rows.map(({ id }) => id),
        plannedIds: events.flatMap(atoms),
        spatialIds: events.filter(({ spatial }) => spatial).flatMap(atoms),
        layoutKeys,
      };
    },
    { disabled: disabledFilters, raw: rawPresentation },
  );
}

/** @param {import("@playwright/test").Page} page */
async function staticDomSignature(page) {
  return page.evaluate(
    ({ eventRootSelector, routeRootSelector }) => {
      const eventRoot = document.querySelector(eventRootSelector);
      const routeRoot = document.querySelector(routeRootSelector);
      const map = document.querySelector("#battlefield .map-boundary");
      if (!(eventRoot instanceof SVGElement) || !(routeRoot instanceof SVGElement)) {
        throw new Error("Static choreography roots are unavailable.");
      }
      if (!(map instanceof SVGRectElement)) {
        throw new Error("Battlefield map bounds are unavailable.");
      }
      /**
       * @param {Element} element
       * @param {string} name
       */
      const numberAttribute = (element, name) => {
        const raw = element.getAttribute(name);
        if (raw === null) {
          return null;
        }
        const value = Number(raw);
        return Number.isFinite(value) ? value : null;
      };
      /** @param {Element} element */
      const rectangle = (element) => ({
        left: numberAttribute(element, "data-layout-left"),
        top: numberAttribute(element, "data-layout-top"),
        right: numberAttribute(element, "data-layout-right"),
        bottom: numberAttribute(element, "data-layout-bottom"),
      });
      const effects = [...eventRoot.querySelectorAll(".combat-effect")].map(
        (effect) => {
          const eventId = effect.getAttribute("data-event-id");
          const rawAtomicIds = effect.getAttribute("data-atomic-event-ids");
          return {
            eventId,
            eventType: effect.getAttribute("data-event-type"),
            atomicIds: rawAtomicIds === null ? [eventId] : JSON.parse(rawAtomicIds),
          };
        },
      );
      const placements = [
        ...document.querySelectorAll(
          `${eventRootSelector} [data-layout-key], ${routeRootSelector} [data-layout-key]`,
        ),
      ]
        .map((element) => {
          const route = element.matches(".combat-route-effect");
          const path = route
            ? element.querySelector(
                ".combat-route__path, .combat-charge__path, .combat-rejection__route",
              )
            : null;
          return {
            key: element.getAttribute("data-layout-key"),
            className: element.getAttribute("class"),
            route,
            bounds: route ? null : rectangle(element),
            collisionFree: element.getAttribute("data-layout-collision-free"),
            disposition: element.getAttribute("data-layout-disposition"),
            lane: route ? numberAttribute(element, "data-lane") : null,
            path: path?.getAttribute("d") ?? null,
            bridges: [...element.querySelectorAll(".combat-route__bridge-backplate")]
              .map((bridge) => ({
                ariaHidden: bridge.getAttribute("aria-hidden"),
                withLayoutKey: bridge.getAttribute("data-bridge-with-layout-key"),
                gap: numberAttribute(bridge, "data-bridge-gap"),
                cx: numberAttribute(bridge, "cx"),
                cy: numberAttribute(bridge, "cy"),
              }))
              .sort((left, right) =>
                String(left.withLayoutKey).localeCompare(String(right.withLayoutKey)),
              ),
          };
        })
        .sort((left, right) => String(left.key).localeCompare(String(right.key)));
      const protectedRects = [
        ...document.querySelectorAll(
          [
            "#battlefield .status-cell__box",
            "#battlefield .cooldown-cell__box",
            "#battlefield .modifier-cell__box",
            "#battlefield .legality-pill__box",
            "#battlefield .required-dock-fallback__box",
          ].join(", "),
        ),
      ].map((element) => {
        const left = numberAttribute(element, "x");
        const top = numberAttribute(element, "y");
        const width = numberAttribute(element, "width");
        const height = numberAttribute(element, "height");
        return {
          left,
          top,
          right: left === null || width === null ? null : left + width,
          bottom: top === null || height === null ? null : top + height,
        };
      });
      for (const body of document.querySelectorAll("#battlefield .agent-body")) {
        const cx = numberAttribute(body, "cx");
        const cy = numberAttribute(body, "cy");
        const radius = numberAttribute(body, "r");
        protectedRects.push({
          left: cx === null || radius === null ? null : cx - radius,
          top: cy === null || radius === null ? null : cy - radius,
          right: cx === null || radius === null ? null : cx + radius,
          bottom: cy === null || radius === null ? null : cy + radius,
        });
      }
      const rootState = [eventRoot, routeRoot].map((root) => ({
        authorizationKey: root.getAttribute("data-authorization-key"),
        epochKey: root.getAttribute("data-epoch-key"),
        fingerprint: root.getAttribute("data-event-fingerprint"),
        paintKey: root.getAttribute("data-paint-key"),
        renderPolicy: root.getAttribute("data-render-policy"),
        state: root.getAttribute("data-state"),
        viewportKey: root.getAttribute("data-viewport-key"),
      }));
      return {
        rootState,
        effects,
        atomicIds: effects.flatMap(({ atomicIds }) => atomicIds),
        placements,
        layoutKeys: placements.map(({ key }) => key),
        protectedRects,
        mapBounds: {
          left: numberAttribute(map, "x"),
          top: numberAttribute(map, "y"),
          right: Number(map.getAttribute("x")) + Number(map.getAttribute("width")),
          bottom: Number(map.getAttribute("y")) + Number(map.getAttribute("height")),
        },
        leaders: [
          ...document.querySelectorAll(
            `${eventRootSelector} .combat-cue__leader, ${eventRootSelector} .combat-route__ownership-leader`,
          ),
        ].map((leader) => ({
          ariaHidden: leader.getAttribute("aria-hidden"),
          tagName: leader.localName,
          points: /** @type {Array<{x: number, y: number}>} */ (
            JSON.parse(leader.getAttribute("data-leader-points") ?? "[]")
          ),
        })),
        routeHitCount: routeRoot.querySelectorAll(".combat-route__hit").length,
      };
    },
    {
      eventRootSelector: CHOREOGRAPHY_ROOT,
      routeRootSelector: CHOREOGRAPHY_ROUTE_ROOT,
    },
  );
}

/** @param {import("@playwright/test").Page} page */
async function spatialContractEvidence(page) {
  return page.evaluate(
    async ({ eventRootSelector, routeRootSelector, hitSelector }) => {
      await document.fonts.ready;
      const battlefield = document.querySelector("#battlefield");
      const eventRoot = document.querySelector(eventRootSelector);
      const routeRoot = document.querySelector(routeRootSelector);
      const map = battlefield?.querySelector(".map-boundary") ?? null;
      if (
        !(battlefield instanceof SVGSVGElement) ||
        !(eventRoot instanceof SVGElement) ||
        !(routeRoot instanceof SVGElement) ||
        !(map instanceof SVGRectElement)
      ) {
        throw new Error("Static spatial contract roots are unavailable.");
      }
      const mapMatrix = map.getScreenCTM();
      if (mapMatrix === null) {
        throw new Error("Static map screen transform is unavailable.");
      }
      /** @param {DOMRect | DOMRectReadOnly} bounds */
      const rectangle = (bounds) => ({
        left: bounds.left,
        top: bounds.top,
        right: bounds.right,
        bottom: bounds.bottom,
      });
      /** @param {Element} element @param {string} name */
      const numberAttribute = (element, name) => {
        const raw = element.getAttribute(name);
        if (raw === null) return null;
        const value = Number(raw);
        return Number.isFinite(value) ? value : null;
      };
      /** @param {{x: number, y: number}} point @param {DOMMatrix} matrix */
      const transformPoint = (point, matrix) => {
        const transformed = new DOMPoint(point.x, point.y).matrixTransform(matrix);
        return { x: transformed.x, y: transformed.y };
      };
      /** @param {ReadonlyArray<{x: number, y: number}>} points */
      const pointsBounds = (points) => ({
        left: Math.min(...points.map(({ x }) => x)),
        top: Math.min(...points.map(({ y }) => y)),
        right: Math.max(...points.map(({ x }) => x)),
        bottom: Math.max(...points.map(({ y }) => y)),
      });
      /** @param {Element} element */
      const publishedBounds = (element) => {
        const left = numberAttribute(element, "data-layout-left");
        const top = numberAttribute(element, "data-layout-top");
        const right = numberAttribute(element, "data-layout-right");
        const bottom = numberAttribute(element, "data-layout-bottom");
        if (left === null || top === null || right === null || bottom === null) {
          return null;
        }
        return pointsBounds(
          [
            { x: left, y: top },
            { x: right, y: top },
            { x: right, y: bottom },
            { x: left, y: bottom },
          ].map((point) => transformPoint(point, mapMatrix)),
        );
      };
      /** @param {ReadonlyArray<Element>} elements */
      const paintedBounds = (elements) => {
        const rectangles = elements
          .filter((element) => {
            const style = getComputedStyle(element);
            return style.display !== "none" && style.visibility !== "hidden";
          })
          .map((element) => rectangle(element.getBoundingClientRect()))
          .filter(
            ({ left, top, right, bottom }) =>
              [left, top, right, bottom].every(Number.isFinite) &&
              (right > left || bottom > top),
          );
        return rectangles.length === 0
          ? null
          : {
              left: Math.min(...rectangles.map(({ left }) => left)),
              top: Math.min(...rectangles.map(({ top }) => top)),
              right: Math.max(...rectangles.map(({ right }) => right)),
              bottom: Math.max(...rectangles.map(({ bottom }) => bottom)),
            };
      };
      /** @param {Element} element */
      const layoutPaintElements = (element) => {
        if (element.matches(".combat-route__ownership")) {
          return [
            ...element.querySelectorAll(
              ".combat-route__ownership-box, .combat-route__ownership-label",
            ),
          ];
        }
        if (element.matches(".combat-effect--net-health")) {
          return [
            ...element.querySelectorAll(
              ".combat-net__hit, .combat-net__recipient, .combat-net__label",
            ),
          ];
        }
        if (element.matches(".combat-rejection__ring")) {
          const owner = element.closest(".combat-effect");
          return owner
            ? [
                ...owner.querySelectorAll(
                  ".combat-rejection__ring, .combat-rejection__hit:not(.combat-route__hit)",
                ),
              ]
            : [element];
        }
        return [element];
      };
      const layoutElements = [
        ...document.querySelectorAll(
          `${eventRootSelector} [data-layout-key], ${routeRootSelector} [data-layout-key]`,
        ),
      ].filter((element) => !element.matches(".combat-route-effect"));
      const layoutFootprints = layoutElements.map((element) => {
        const texts = [
          ...element.querySelectorAll(
            ".combat-net__recipient, .combat-net__label, .combat-route__ownership-label",
          ),
        ].map((text) => ({
          className: text.getAttribute("class"),
          bounds: rectangle(text.getBoundingClientRect()),
        }));
        return {
          key: element.getAttribute("data-layout-key"),
          className: element.getAttribute("class"),
          eventId: element
            .closest(".combat-effect, .combat-route-effect")
            ?.getAttribute("data-event-id"),
          published: publishedBounds(element),
          painted: paintedBounds(layoutPaintElements(element)),
          texts,
        };
      });
      const effects = [...eventRoot.querySelectorAll(".combat-effect")].map(
        (effect) => {
          const hits = [...effect.querySelectorAll(hitSelector)];
          return {
            eventId: effect.getAttribute("data-event-id"),
            eventType: effect.getAttribute("data-event-type"),
            tooltipOwner: effect.hasAttribute("data-tooltip-owner"),
            role: effect.getAttribute("role"),
            tabindex: effect.getAttribute("tabindex"),
            ariaLabel: effect.getAttribute("aria-label"),
            ariaDescription: effect.getAttribute("aria-description"),
            hitCount: hits.length,
            interactiveHitCount: hits.filter(
              (hit) => getComputedStyle(hit).pointerEvents !== "none",
            ).length,
          };
        },
      );
      const routeOwners = [...routeRoot.querySelectorAll(".combat-route-effect")].map(
        (route) => ({
          eventId: route.getAttribute("data-event-id"),
          tooltipOwner: route.hasAttribute("data-tooltip-owner"),
          ariaHidden: route.getAttribute("aria-hidden"),
          role: route.getAttribute("role"),
          tabindex: route.getAttribute("tabindex"),
          ariaLabel: route.getAttribute("aria-label"),
        }),
      );
      const protectedRects = [
        ...battlefield.querySelectorAll(
          [
            ".status-cell__box",
            ".cooldown-cell__box",
            ".modifier-cell__box",
            ".legality-pill__box",
            ".required-dock-fallback__box",
            ".agent-body",
          ].join(", "),
        ),
      ].map((element) => ({
        kind: element.matches(".agent-body") ? "body" : "durable",
        ownerPresentationKey:
          element
            .closest("[data-presentation-key]")
            ?.getAttribute("data-presentation-key") ?? null,
        bounds: rectangle(element.getBoundingClientRect()),
      }));
      const cueRects = layoutElements.map((element) => ({
        key: element.getAttribute("data-layout-key"),
        bounds: publishedBounds(element),
      }));
      /** @param {Element} leader @param {{x: number, y: number}} end */
      const leaderLayoutKey = (leader, end) => {
        const directOwner = leader.closest("[data-layout-key]");
        if (directOwner) return directOwner.getAttribute("data-layout-key");
        const event = leader.closest(".combat-effect, .combat-route-effect");
        if (!event) return null;
        const candidates = [
          ...(event.hasAttribute("data-layout-key") ? [event] : []),
          ...event.querySelectorAll("[data-layout-key]"),
        ];
        const matches = candidates
          .map((candidate) => ({
            key: candidate.getAttribute("data-layout-key"),
            bounds: publishedBounds(candidate),
          }))
          .filter(
            ({ bounds }) =>
              bounds !== null &&
              end.x >= bounds.left - 1.5 &&
              end.x <= bounds.right + 1.5 &&
              end.y >= bounds.top - 1.5 &&
              end.y <= bounds.bottom + 1.5,
          )
          .sort((left, right) => String(left.key).localeCompare(String(right.key)));
        return matches[0]?.key ?? null;
      };
      /** @param {Element} leader */
      const allowedBodyOwners = (leader) => {
        const event = leader.closest(".combat-effect, .combat-route-effect");
        if (!(event instanceof HTMLElement || event instanceof SVGElement)) return [];
        const historicalChargeImpact =
          leader.matches(".combat-cue__leader--impact") &&
          event.getAttribute("data-token-id") === "warrior_charge";
        const attribute = leader.matches(".combat-cue__leader--source")
          ? "data-source-presentation-key"
          : leader.matches(".combat-cue__leader--impact")
            ? historicalChargeImpact
              ? null
              : "data-target-presentation-key"
            : leader.matches(".combat-cue__leader--semantic")
              ? "data-agent-presentation-key"
              : leader.matches(".combat-cue__leader--rejection")
                ? "data-actor-presentation-key"
                : leader.matches(".combat-cue__leader--charge-end")
                  ? "data-source-presentation-key"
                  : leader.matches(
                        ".combat-route__ownership-leader, .combat-cue__leader--charge-start",
                      )
                    ? null
                    : event.hasAttribute("data-recipient-presentation-key")
                      ? "data-recipient-presentation-key"
                      : "data-agent-presentation-key";
        const value = attribute === null ? null : event.getAttribute(attribute);
        return value ? [value] : [];
      };
      const leaders = [
        ...document.querySelectorAll(
          `${eventRootSelector} .combat-cue__leader, ${eventRootSelector} .combat-route__ownership-leader`,
        ),
      ]
        .filter((leader) => getComputedStyle(leader).visibility !== "hidden")
        .map((leader) => {
          if (!(leader instanceof SVGGraphicsElement)) {
            throw new TypeError("Published cue leader is not SVG geometry.");
          }
          const rawPoints = /** @type {Array<{x: number, y: number}>} */ (
            JSON.parse(leader.getAttribute("data-leader-points") ?? "[]")
          );
          const matrix = leader.getScreenCTM();
          if (matrix === null) {
            throw new Error("Published cue leader has no screen transform.");
          }
          const points = rawPoints.map((point) => transformPoint(point, matrix));
          return {
            eventId: leader
              .closest(".combat-effect, .combat-route-effect")
              ?.getAttribute("data-event-id"),
            className: leader.getAttribute("class"),
            ariaHidden: leader.getAttribute("aria-hidden"),
            points,
            ownLayoutKey:
              points.length >= 2
                ? leaderLayoutKey(
                    leader,
                    /** @type {{x: number, y: number}} */ (points.at(-1)),
                  )
                : null,
            allowedBodyOwners: allowedBodyOwners(leader),
          };
        });
      const routeGeometry = [...routeRoot.querySelectorAll(".combat-route-effect")].map(
        (route) => ({
          eventId: route.getAttribute("data-event-id"),
          visiblePaths: [
            ...route.querySelectorAll(
              ".combat-route__path, .combat-charge__path, .combat-rejection__route",
            ),
          ].map((path) => rectangle(path.getBoundingClientRect())),
          markers: [
            ...route.querySelectorAll(
              ".combat-route__arrow, .combat-charge__direction",
            ),
          ].map((marker) => rectangle(marker.getBoundingClientRect())),
        }),
      );
      return {
        mapBounds: rectangle(map.getBoundingClientRect()),
        effects,
        routeOwners,
        layoutFootprints,
        protectedRects,
        cueRects,
        leaders,
        routeGeometry,
      };
    },
    {
      eventRootSelector: CHOREOGRAPHY_ROOT,
      routeRootSelector: CHOREOGRAPHY_ROUTE_ROOT,
      hitSelector: SPATIAL_HIT_SELECTOR,
    },
  );
}

/** @param {Awaited<ReturnType<typeof spatialContractEvidence>>} evidence */
function expectSpatialContracts(evidence) {
  const tolerance = 0.5;
  /**
   * @param {{left: number, top: number, right: number, bottom: number} | null} bounds
   * @returns {bounds is {left: number, top: number, right: number, bottom: number}}
   */
  const finiteBounds = (bounds) =>
    bounds !== null &&
    [bounds.left, bounds.top, bounds.right, bounds.bottom].every(Number.isFinite);
  /**
   * @param {{x: number, y: number}} point
   * @param {{left: number, top: number, right: number, bottom: number}} bounds
   */
  const pointInside = (point, bounds) =>
    point.x >= bounds.left - 0.001 &&
    point.x <= bounds.right + 0.001 &&
    point.y >= bounds.top - 0.001 &&
    point.y <= bounds.bottom + 0.001;
  /**
   * @param {{x: number, y: number}} first
   * @param {{x: number, y: number}} second
   * @param {{x: number, y: number}} third
   */
  const orientation = (first, second, third) =>
    (second.x - first.x) * (third.y - first.y) -
    (second.y - first.y) * (third.x - first.x);
  /**
   * @param {{x: number, y: number}} firstStart
   * @param {{x: number, y: number}} firstEnd
   * @param {{x: number, y: number}} secondStart
   * @param {{x: number, y: number}} secondEnd
   */
  const segmentsIntersect = (firstStart, firstEnd, secondStart, secondEnd) => {
    const firstSide = orientation(firstStart, firstEnd, secondStart);
    const secondSide = orientation(firstStart, firstEnd, secondEnd);
    const thirdSide = orientation(secondStart, secondEnd, firstStart);
    const fourthSide = orientation(secondStart, secondEnd, firstEnd);
    return (
      ((firstSide > 0 && secondSide < 0) || (firstSide < 0 && secondSide > 0)) &&
      ((thirdSide > 0 && fourthSide < 0) || (thirdSide < 0 && fourthSide > 0))
    );
  };
  /**
   * @param {{x: number, y: number}} start
   * @param {{x: number, y: number}} end
   * @param {{left: number, top: number, right: number, bottom: number}} bounds
   */
  const segmentIntersects = (start, end, bounds) => {
    if (pointInside(start, bounds) || pointInside(end, bounds)) return true;
    const corners = [
      { x: bounds.left, y: bounds.top },
      { x: bounds.right, y: bounds.top },
      { x: bounds.right, y: bounds.bottom },
      { x: bounds.left, y: bounds.bottom },
    ];
    return corners.some((corner, index) =>
      segmentsIntersect(start, end, corner, corners[(index + 1) % corners.length]),
    );
  };
  /**
   * @param {{left: number, top: number, right: number, bottom: number} | null} actual
   * @param {{left: number, top: number, right: number, bottom: number} | null} reserved
   * @param {string} label
   */
  const assertContained = (actual, reserved, label) => {
    if (!finiteBounds(actual) || !finiteBounds(reserved)) {
      throw new Error(`${label} has non-finite paint or reservation bounds.`);
    }
    expect(actual.left, `${label} escaped left`).toBeGreaterThanOrEqual(
      reserved.left - tolerance,
    );
    expect(actual.top, `${label} escaped top`).toBeGreaterThanOrEqual(
      reserved.top - tolerance,
    );
    expect(actual.right, `${label} escaped right`).toBeLessThanOrEqual(
      reserved.right + tolerance,
    );
    expect(actual.bottom, `${label} escaped bottom`).toBeLessThanOrEqual(
      reserved.bottom + tolerance,
    );
  };

  expect(evidence.effects.length).toBeGreaterThan(0);
  for (const effect of evidence.effects) {
    expect(effect.tooltipOwner, `${effect.eventId} has no tooltip owner`).toBe(true);
    expect(effect.role, String(effect.eventId)).toBe("img");
    expect(effect.tabindex, String(effect.eventId)).toBe("0");
    expect(effect.ariaLabel, String(effect.eventId)).toMatch(/\S/u);
    expect(effect.ariaDescription, String(effect.eventId)).toMatch(/\S/u);
    expect(
      effect.hitCount,
      `${effect.eventId} has no spatial hit surface`,
    ).toBeGreaterThan(0);
    expect(
      effect.interactiveHitCount,
      `${effect.eventId} has no pointer-inspectable hit surface`,
    ).toBeGreaterThan(0);
  }
  const effectIds = new Set(evidence.effects.map(({ eventId }) => eventId));
  for (const route of evidence.routeOwners) {
    expect(
      effectIds.has(route.eventId),
      `${route.eventId} has no visible event owner`,
    ).toBe(true);
    expect(route.tooltipOwner, `${route.eventId} route has no tooltip owner`).toBe(
      true,
    );
    expect(route.ariaHidden, String(route.eventId)).toBe("true");
    expect(route.role, String(route.eventId)).toBeNull();
    expect(route.tabindex, String(route.eventId)).toBeNull();
    expect(route.ariaLabel, String(route.eventId)).toBeNull();
  }
  expect(evidence.layoutFootprints.length).toBeGreaterThan(0);
  for (const footprint of evidence.layoutFootprints) {
    assertContained(footprint.painted, footprint.published, String(footprint.key));
    for (const text of footprint.texts) {
      assertContained(
        text.bounds,
        footprint.published,
        `${String(footprint.key)} ${String(text.className)}`,
      );
    }
  }
  const netLayouts = evidence.layoutFootprints.filter(({ className }) =>
    String(className).includes("combat-effect--net-health"),
  );
  if (netLayouts.length > 0) {
    expect(netLayouts.every(({ texts }) => texts.length > 0)).toBe(true);
  }
  const ownershipLayouts = evidence.layoutFootprints.filter(({ className }) =>
    String(className).includes("combat-route__ownership"),
  );
  if (ownershipLayouts.length > 0) {
    expect(ownershipLayouts.every(({ texts }) => texts.length === 1)).toBe(true);
  }
  if (!finiteBounds(evidence.mapBounds)) {
    throw new Error("Static map client bounds are not finite.");
  }
  for (const route of evidence.routeGeometry) {
    expect(
      route.visiblePaths.length,
      `${route.eventId} has no visible route`,
    ).toBeGreaterThan(0);
    for (const bounds of [...route.visiblePaths, ...route.markers]) {
      assertContained(bounds, evidence.mapBounds, `${route.eventId} visible route`);
    }
  }
  expect(evidence.leaders.length).toBeGreaterThan(0);
  for (const leader of evidence.leaders) {
    expect(leader.ariaHidden, String(leader.eventId)).toBe("true");
    expect(leader.points.length, String(leader.eventId)).toBeGreaterThanOrEqual(2);
    expect(leader.ownLayoutKey, `${leader.eventId} has no owning cue`).not.toBeNull();
    for (const point of leader.points) {
      expect([point.x, point.y].every(Number.isFinite), String(leader.eventId)).toBe(
        true,
      );
      expect(point.x, String(leader.eventId)).toBeGreaterThanOrEqual(
        evidence.mapBounds.left - tolerance,
      );
      expect(point.x, String(leader.eventId)).toBeLessThanOrEqual(
        evidence.mapBounds.right + tolerance,
      );
      expect(point.y, String(leader.eventId)).toBeGreaterThanOrEqual(
        evidence.mapBounds.top - tolerance,
      );
      expect(point.y, String(leader.eventId)).toBeLessThanOrEqual(
        evidence.mapBounds.bottom + tolerance,
      );
    }
    for (let index = 1; index < leader.points.length; index += 1) {
      const start = leader.points[index - 1];
      const end = leader.points[index];
      for (const protectedRect of evidence.protectedRects) {
        if (!finiteBounds(protectedRect.bounds)) {
          throw new Error("Durable protected client bounds are not finite.");
        }
        const allowedBody =
          protectedRect.kind === "body" &&
          protectedRect.ownerPresentationKey !== null &&
          leader.allowedBodyOwners.includes(protectedRect.ownerPresentationKey);
        if (!allowedBody) {
          expect(
            segmentIntersects(start, end, protectedRect.bounds),
            `${leader.eventId} leader intersects nonowner durable geometry`,
          ).toBe(false);
        }
      }
      for (const cue of evidence.cueRects) {
        if (cue.key === leader.ownLayoutKey) continue;
        if (!finiteBounds(cue.bounds)) {
          throw new Error(`Cue ${String(cue.key)} has non-finite client bounds.`);
        }
        expect(
          segmentIntersects(start, end, cue.bounds),
          `${leader.eventId} leader intersects cue ${String(cue.key)}`,
        ).toBe(false);
      }
    }
  }
}

/** @param {Awaited<ReturnType<typeof staticDomSignature>>} signature */
function expectBoundedNonOverlap(signature) {
  /**
   * @param {{left: number | null, top: number | null, right: number | null, bottom: number | null} | null} rectangle
   * @returns {rectangle is {left: number, top: number, right: number, bottom: number}}
   */
  const finiteRectangle = (rectangle) =>
    rectangle !== null &&
    [rectangle.left, rectangle.top, rectangle.right, rectangle.bottom].every(
      Number.isFinite,
    );
  /**
   * @param {{left: number, top: number, right: number, bottom: number}} left
   * @param {{left: number, top: number, right: number, bottom: number}} right
   * @param {number} [tolerance]
   */
  const overlaps = (left, right, tolerance = 0.001) =>
    Math.min(left.right, right.right) - Math.max(left.left, right.left) > tolerance &&
    Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top) > tolerance;
  const mapBounds = signature.mapBounds;
  if (!finiteRectangle(mapBounds)) {
    throw new Error("Static summary map bounds are not finite.");
  }
  const cues = signature.placements.filter(({ route }) => !route);
  expect(cues.length).toBeGreaterThan(0);
  for (const cue of cues) {
    expect(cue.key).toMatch(/^\["event",/u);
    expect(cue.collisionFree).toBe("true");
    if (!finiteRectangle(cue.bounds)) {
      throw new Error(`Cue ${String(cue.key)} has non-finite bounds.`);
    }
    expect(cue.bounds.left).toBeGreaterThanOrEqual(mapBounds.left - 0.001);
    expect(cue.bounds.top).toBeGreaterThanOrEqual(mapBounds.top - 0.001);
    expect(cue.bounds.right).toBeLessThanOrEqual(mapBounds.right + 0.001);
    expect(cue.bounds.bottom).toBeLessThanOrEqual(mapBounds.bottom + 0.001);
  }
  for (let leftIndex = 0; leftIndex < cues.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < cues.length; rightIndex += 1) {
      const leftBounds = cues[leftIndex].bounds;
      const rightBounds = cues[rightIndex].bounds;
      if (!finiteRectangle(leftBounds) || !finiteRectangle(rightBounds)) {
        throw new Error("Cue pair has non-finite bounds.");
      }
      expect(overlaps(leftBounds, rightBounds)).toBe(false);
    }
  }
  for (const cue of cues) {
    if (!finiteRectangle(cue.bounds)) {
      throw new Error(`Cue ${String(cue.key)} has non-finite bounds.`);
    }
    for (const protectedRect of signature.protectedRects) {
      if (!finiteRectangle(protectedRect)) {
        throw new Error("Durable protected geometry is not finite.");
      }
      expect(overlaps(cue.bounds, protectedRect)).toBe(false);
    }
  }
  const routes = signature.placements.filter(({ route }) => route);
  expect(routes.length).toBeGreaterThan(0);
  expect(signature.routeHitCount).toBe(routes.length);
  for (const route of routes) {
    expect(route.lane).toBeGreaterThanOrEqual(0);
    expect(route.path).toMatch(/^M /u);
    expect(route.path).not.toMatch(/NaN|Infinity/u);
    for (const bridge of route.bridges) {
      expect(bridge).toMatchObject({ ariaHidden: "true" });
      expect(bridge.withLayoutKey).toMatch(/^\["event",/u);
      expect(bridge.gap).toBeGreaterThan(0);
      expect(Number.isFinite(bridge.cx)).toBe(true);
      expect(Number.isFinite(bridge.cy)).toBe(true);
    }
  }
  expect(signature.leaders.length).toBeGreaterThan(0);
  for (const leader of signature.leaders) {
    expect(leader.ariaHidden).toBe("true");
    expect(["line", "path"]).toContain(leader.tagName);
    expect(leader.points.length).toBeGreaterThanOrEqual(2);
    expect(
      leader.points.every(({ x, y }) => Number.isFinite(x) && Number.isFinite(y)),
    ).toBe(true);
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{method: string, path: string}[]} apiRequests
 * @param {string} filterId
 * @param {boolean} enabled
 */
async function setFilter(page, apiRequests, filterId, enabled) {
  const mark = apiRequests.length;
  const input = page.locator(`${FILTER_INPUT}[data-visual-filter-id="${filterId}"]`);
  if (enabled) {
    await input.check();
  } else {
    await input.uncheck();
  }
  await settleStaticSummary(page);
  expect(apiRequests.slice(mark), `${filterId} caused an API request`).toEqual([]);
}

/** @param {import("@playwright/test").Page} page */
async function expectTransientHoverHelp(page) {
  const routeOwners = page.locator(
    `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect[data-tooltip-owner]`,
  );
  expect(await routeOwners.count()).toBeGreaterThan(0);
  for (const owner of await routeOwners.all()) {
    await expect(owner).toHaveAttribute("aria-hidden", "true");
    await expect(owner).not.toHaveAttribute("role", /.+/u);
    await expect(owner).not.toHaveAttribute("tabindex", /.+/u);
    await expect(owner).not.toHaveAttribute("aria-label", /.+/u);
  }
  for (const [ownerSelector, hitSelector, tooltipKind] of TRANSIENT_EVENT_FAMILIES) {
    const owner = page.locator(`${CHOREOGRAPHY_ROOT} ${ownerSelector}`).first();
    await expect(owner).toHaveCount(1);
    await expect(owner).toHaveAttribute("role", "img");
    await expect(owner).toHaveAttribute("tabindex", "0");
    await expect(owner).toHaveAttribute("aria-label", /\S/u);
    await expect(owner).toHaveAttribute("aria-description", /\S/u);
    const hit = owner.locator(hitSelector).first();
    await expect(hit).toHaveCount(1);
    await hit.hover();
    await expect(page.locator("#visual-tooltip")).toBeVisible();
    await expect(page.locator("#visual-tooltip")).toHaveAttribute(
      "data-tooltip-kind",
      tooltipKind,
    );
    await expect(page.locator("#visual-tooltip-title")).not.toHaveText("");
    await expect(page.locator("#visual-tooltip-details")).not.toHaveText("");
    await page.mouse.move(1, 1);
    await expect(page.locator("#visual-tooltip")).toBeHidden();
  }
}

test("paused replay installs a complete deterministic static summary at both supported viewports", async ({
  page,
}) => {
  if (!replay) {
    throw new Error("Registered replay viewer was not started.");
  }
  const browserErrors = captureBrowserErrors(page);
  const apiRequests = captureApiRequests(page);
  await page.setViewportSize(MINIMUM_VIEWPORT);
  await page.goto(replay.url);
  await expect(page.locator("#connection-status")).toHaveText("Online", {
    timeout: 30_000,
  });
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expectReplayFrameIndex(page, 2);
  await settleStaticSummary(page);

  const rawPresentation = JSON.parse(
    await authenticatedText(page, "/api/presentation/frame"),
  );
  expect(rawPresentation.presentation_kind).toBe("replay_oracle");
  expect(rawPresentation.source.source_frame_index).toBe(2);
  expect(rawPresentation.latest_events.incoming_transition_index).toBe(1);
  const allOnExpected = await expectedPlanSignature(page, rawPresentation);
  expect(allOnExpected.orderedIds).toHaveLength(24);
  expect(allOnExpected.plannedIds).toEqual(allOnExpected.orderedIds);
  expect(
    await page
      .locator("#event-feed .event-item")
      .evaluateAll((rows) => rows.map((row) => row.getAttribute("data-event-id"))),
  ).toEqual(allOnExpected.orderedIds);
  const science = await scientificSignature(page);
  await openVisualFilters(page);

  /** @type {Awaited<ReturnType<typeof staticDomSignature>> | null} */
  let minimumSignature = null;
  for (const viewport of [MINIMUM_VIEWPORT, DESKTOP_VIEWPORT]) {
    await page.setViewportSize(viewport);
    await settleStaticSummary(page);
    const allOn = await staticDomSignature(page);
    expect(allOn.atomicIds).toEqual(allOnExpected.spatialIds);
    expect(allOn.layoutKeys).toEqual(allOnExpected.layoutKeys);
    expect(new Set(allOn.layoutKeys).size).toBe(allOn.layoutKeys.length);
    expect(allOn.rootState[0]).toMatchObject({
      authorizationKey: allOnExpected.authorizationKey,
      epochKey: allOnExpected.epochKey,
      fingerprint: allOnExpected.fingerprint,
      renderPolicy: "replay_static",
      state: "settled",
    });
    expect(allOn.rootState[1]).toEqual(allOn.rootState[0]);
    expectBoundedNonOverlap(allOn);
    expectSpatialContracts(await spatialContractEvidence(page));
    await expectTransientHoverHelp(page);
    expect(await scientificSignature(page)).toEqual(science);

    const filteredExpected = await expectedPlanSignature(page, rawPresentation, [
      SELECTIVE_FILTER,
    ]);
    await setFilter(page, apiRequests, SELECTIVE_FILTER, false);
    const filtered = await staticDomSignature(page);
    expect(filtered.atomicIds).toEqual(filteredExpected.spatialIds);
    expect(filtered.layoutKeys).toEqual(filteredExpected.layoutKeys);
    expect(filtered.layoutKeys).toEqual(allOn.layoutKeys);
    expect(filtered.rootState[0]).toMatchObject({
      authorizationKey: allOnExpected.authorizationKey,
      epochKey: allOnExpected.epochKey,
      fingerprint: allOnExpected.fingerprint,
      renderPolicy: "replay_static",
      state: "settled",
    });
    expectBoundedNonOverlap(filtered);
    expect(await scientificSignature(page)).toEqual(science);

    await setFilter(page, apiRequests, SELECTIVE_FILTER, true);
    expect(await staticDomSignature(page)).toEqual(allOn);

    const allOffMark = apiRequests.length;
    await page
      .locator(`#visual-filter-options ${FILTER_INPUT}`)
      .evaluateAll((inputs) => {
        for (const input of inputs) {
          if (input instanceof HTMLInputElement && input.checked) {
            input.click();
          }
        }
      });
    await settleStaticSummary(page);
    expect(apiRequests.slice(allOffMark), "all-off caused an API request").toEqual([]);
    await expect(page.locator(FILTER_INPUT)).toHaveCount(VISUAL_FILTER_IDS.length);
    await expect(page.locator(`${FILTER_INPUT}:checked`)).toHaveCount(0);
    await expect(page.locator(`${CHOREOGRAPHY_ROOT} > *`)).toHaveCount(0);
    await expect(page.locator(`${CHOREOGRAPHY_ROUTE_ROOT} > *`)).toHaveCount(0);
    await expect(page.locator("#battlefield [data-layout-key]")).toHaveCount(0);
    expect(await scientificSignature(page)).toEqual(science);

    const restoreMark = apiRequests.length;
    await page.locator("#restore-all-visual-filters-button").click();
    await settleStaticSummary(page);
    expect(apiRequests.slice(restoreMark), "Restore All caused an API request").toEqual(
      [],
    );
    await expect(page.locator(`${FILTER_INPUT}:checked`)).toHaveCount(
      VISUAL_FILTER_IDS.length,
    );
    expect(await staticDomSignature(page)).toEqual(allOn);
    expect(await scientificSignature(page)).toEqual(science);

    if (viewport === MINIMUM_VIEWPORT) {
      minimumSignature = allOn;
    }
  }

  await page.setViewportSize(MINIMUM_VIEWPORT);
  await settleStaticSummary(page);
  expect(await staticDomSignature(page)).toEqual(minimumSignature);
  expect(await scientificSignature(page)).toEqual(science);
  expect(browserErrors).toEqual([]);
});
