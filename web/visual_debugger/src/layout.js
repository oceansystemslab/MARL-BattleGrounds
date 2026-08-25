import {
  createPolylineRouteGeometry,
  createRouteGeometry,
  routeMarkerPose,
} from "./routes.js";

const EPSILON = 1e-9;
const STATUS_DOCK_SEARCH_LIMIT = 100_000;
const REQUIRED_DOCK_SEARCH_LIMIT = 5_000;
const REQUIRED_DOCK_JOINT_SEARCH_MAX_REQUESTS = 6;
const REQUIRED_DOCK_FALLBACK_OPTIONS = Object.freeze({
  cellWidth: 32,
  cellHeight: 16,
  cellGap: 0,
  dockGap: 3,
  tangentStep: 6,
  maxTangentShift: 24,
});
const CROSS_PHASE_ROUTE_SAMPLES = 32;
// SVG route paths serialize to four decimal places. Keep fallback waypoints a
// full presentation pixel outside protected rectangles so serialization cannot
// round a collision-free detour back onto a durable boundary.
const CROSS_PHASE_ROUTE_GRAPH_MARGIN = 1;
const CROSS_PHASE_ROUTE_GRAPH_NODE_LIMIT = 384;
const CROSS_PHASE_CUE_PAINT_PADDING = 1;
const CROSS_PHASE_RECIPIENT_COMPACTION_RADIAL_OFFSET = 4;
const CROSS_PHASE_RECIPIENT_COMPACTION_RADIAL_STEP = 8;
const CROSS_PHASE_RECIPIENT_COMPACTION_ANGLE_OFFSET_DEGREES = 4;
const CROSS_PHASE_RECIPIENT_COMPACTION_ANGLE_STEP_DEGREES = 12;
const CROSS_PHASE_RECIPIENT_REFINEMENT_ANGLE_OFFSET_DEGREES = 10;

export const DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS = Object.freeze({
  clearance: 3,
  cueGap: 8,
  stackGap: 5,
  routeLaneSpacing: 18,
  routeLaneSearch: 8,
  bridgeGap: 10,
});

/** @type {ReadonlyArray<"north" | "east" | "west" | "south">} */
export const STATUS_DOCK_ANCHORS = Object.freeze(["north", "east", "west", "south"]);

export const STATUS_DOCK_CAPACITY = 9;

export const DEFAULT_STATUS_DOCK_OPTIONS = Object.freeze({
  bodyPadding: 3,
  selectionAllowance: 5,
  cellWidth: 28,
  cellHeight: 20,
  cellGap: 3,
  dockGap: 7,
  tangentStep: 8,
  maxTangentShift: 24,
  ordinaryVisibleLimit: STATUS_DOCK_CAPACITY,
  requiredVisibleLimit: STATUS_DOCK_CAPACITY,
});

/**
 * @typedef {{x: number, y: number}} Point
 * @typedef {{
 *   left: number,
 *   top: number,
 *   right: number,
 *   bottom: number,
 *   width: number,
 *   height: number,
 * }} Rectangle
 * @typedef {{top: number, right: number, bottom: number, left: number}} Insets
 * @typedef {{
 *   worldWidth: number,
 *   worldHeight: number,
 *   viewportWidth: number,
 *   viewportHeight: number,
 *   padding?: number | Partial<Insets>,
 * }} ViewportTransformInput
 * @typedef {{
 *   scale: number,
 *   worldWidth: number,
 *   worldHeight: number,
 *   viewportBounds: Rectangle,
 *   mapBounds: Rectangle,
 *   worldToScreen: (point: Point | readonly [number, number]) => Point,
 *   screenToWorld: (point: Point | readonly [number, number]) => Point,
 *   worldLengthToScreen: (length: number) => number,
 * }} ViewportTransform
 * @typedef {{
 *   center: Point | readonly [number, number],
 *   radius: number,
 *   controlled?: boolean,
 *   selected?: boolean,
 * }} ProtectedBodyInput
 * @typedef {{
 *   bodyPadding?: number,
 *   selectionAllowance?: number,
 * }} ProtectedBodyOptions
 * @typedef {{
 *   globalSlot: number,
 *   center: Point | readonly [number, number],
 *   radius: number,
 *   statuses: ReadonlyArray<unknown>,
 *   required?: boolean,
 *   controlled?: boolean,
 *   selected?: boolean,
 * }} StatusDockAgent
 * @typedef {{
 *   bodyPadding?: number,
 *   selectionAllowance?: number,
 *   cellWidth?: number,
 *   cellHeight?: number,
 *   cellGap?: number,
 *   dockGap?: number,
 *   tangentStep?: number,
 *   maxTangentShift?: number,
 *   ordinaryVisibleLimit?: number,
 *   requiredVisibleLimit?: number,
 * }} StatusDockOptions
 * @typedef {{
 *   agents: ReadonlyArray<StatusDockAgent>,
 *   viewport: Rectangle,
 *   reservedRects?: ReadonlyArray<Rectangle>,
 * }} StatusDockLayoutInput
 * @typedef {{
 *   layoutKey: string,
 *   globalSlot: number,
 *   statuses: ReadonlyArray<unknown>,
 *   dockOptions?: StatusDockOptions,
 *   fallbackDockOptions?: StatusDockOptions,
 *   priority?: number,
 * }} RequiredDockRequest
 * @typedef {{
 *   agents: ReadonlyArray<StatusDockAgent>,
 *   requests: ReadonlyArray<RequiredDockRequest>,
 *   viewport: Rectangle,
 *   reservedRects?: ReadonlyArray<Rectangle>,
 * }} RequiredDockLayoutInput
 * @typedef {{
 *   viewportOverflow: number,
 *   bodyOrReservedIntersection: number,
 *   priorDockIntersection: number,
 *   displacement: number,
 *   anchorIndex: number,
 * }} StatusDockScore
 * @typedef {{
 *   start: Point,
 *   end: Point,
 * }} LeaderLine
 * @typedef {{
 *   globalSlot: number,
 *   priorityIndex: number,
 *   required: boolean,
 *   controlled: boolean,
 *   selected: boolean,
 *   anchor: "north" | "east" | "west" | "south",
 *   tangentShift: number,
 *   bounds: Rectangle,
 *   leader: LeaderLine,
 *   columns: number,
 *   rows: number,
 *   expanded: boolean,
 *   collisionFree: boolean,
 *   visibleStatuses: ReadonlyArray<unknown>,
 *   hiddenStatuses: ReadonlyArray<unknown>,
 *   visibleCount: number,
 *   hiddenCount: number,
 *   totalCount: number,
 *   overflowLabel: string | null,
 *   score: StatusDockScore,
 * }} StatusDockPlacement
 * @typedef {{
 *   docks: ReadonlyArray<StatusDockPlacement>,
 *   protectedBodies: ReadonlyArray<{globalSlot: number, bounds: Rectangle}>,
 *   placementOrder: ReadonlyArray<number>,
 *   suppressedGlobalSlots: ReadonlyArray<number>,
 * }} StatusDockLayout
 * @typedef {StatusDockPlacement & {
 *   layoutKey: string,
 *   compactFallback?: boolean,
 * }} RequiredDockPlacement
 * @typedef {{
 *   docks: ReadonlyArray<RequiredDockPlacement>,
 *   protectedBodies: ReadonlyArray<{globalSlot: number, bounds: Rectangle}>,
 *   placementOrder: ReadonlyArray<string>,
 *   compactedLayoutKeys: ReadonlyArray<string>,
 *   suppressedLayoutKeys: ReadonlyArray<string>,
 * }} RequiredDockLayout
 * @typedef {"recipient_cue" | "perimeter_callout" | "route"} CrossPhaseKind
 * @typedef {{layoutKey: string, bounds?: Rectangle} & Partial<Rectangle>}
 *   CrossPhaseProtectedRegionInput
 * @typedef {{
 *   layoutKey: string,
 *   kind: CrossPhaseKind,
 *   enabled?: boolean,
 *   priority?: number,
 *   stableOrder: number,
 *   anchor?: Point | readonly [number, number],
 *   anchorRadius?: number,
 *   recipientKey?: string,
 *   width?: number,
 *   height?: number,
 *   source?: Point | readonly [number, number],
 *   target?: Point | readonly [number, number],
 *   sourceRadius?: number,
 *   targetRadius?: number,
 *   sourceEndpointGap?: number,
 *   targetEndpointGap?: number,
 *   pathPadding?: number,
 *   markerPadding?: number,
 *   compactMarkerPadding?: number,
 *   markerProgress?: number,
 *   allowProtectedKeys?: ReadonlyArray<string>,
 *   sourceProtectedKey?: string,
 *   targetProtectedKey?: string,
 * }} CrossPhaseRequest
 * @typedef {{
 *   viewport: Rectangle,
 *   protectedRects?: ReadonlyArray<CrossPhaseProtectedRegionInput>,
 *   requests: ReadonlyArray<CrossPhaseRequest>,
 * }} CrossPhaseLayoutInput
 * @typedef {Readonly<{
 *   layoutKey: string,
 *   kind: "recipient_cue" | "perimeter_callout",
 *   priority: number,
 *   stableOrder: number,
 *   anchor: Point,
 *   anchorRadius: number,
 *   recipientKey: string,
 *   width: number,
 *   height: number,
 *   allowProtectedKeys: ReadonlyArray<string>,
 * }>} NormalizedCrossPhaseCueRequest
 * @typedef {Readonly<{
 *   layoutKey: string,
 *   kind: "route",
 *   priority: number,
 *   stableOrder: number,
 *   source: Point,
 *   target: Point,
 *   sourceRadius: number,
 *   targetRadius: number,
 *   sourceEndpointGap: number,
 *   targetEndpointGap: number,
 *   pathPadding: number,
 *   markerPadding: number,
 *   compactMarkerPadding: number | null,
 *   markerProgress: number | null,
 *   allowProtectedKeys: ReadonlyArray<string>,
 *   sourceProtectedKey: string | null,
 *   targetProtectedKey: string | null,
 * }>} NormalizedCrossPhaseRouteRequest
 * @typedef {NormalizedCrossPhaseCueRequest | NormalizedCrossPhaseRouteRequest}
 *   NormalizedCrossPhaseRequest
 * @typedef {Readonly<{center: Point, bounds: Rectangle}>} CrossPhaseCueCandidate
 * @typedef {ReturnType<typeof createRouteGeometry> & Readonly<{
 *   layoutKey: string,
 *   priority: number,
 *   stableOrder: number,
 *   lane: number,
 *   markerVariant: "full" | "compact",
 *   markerPadding: number,
 *   bridgeGaps: ReadonlyArray<{
 *     withLayoutKey: string,
 *     at: Point,
 *     gap: number,
 *   }>,
 * }>} CrossPhaseRoutePlacement
 */

/**
 * Build the sole fitted world-to-screen transform for the battlefield.
 * World Y increases upward; screen Y increases downward.
 *
 * @param {ViewportTransformInput} input
 * @returns {ViewportTransform}
 */
export function createViewportTransform(input) {
  if (!isRecord(input)) {
    throw new TypeError("viewport transform input must be an object.");
  }
  const worldWidth = positiveFinite(input.worldWidth, "worldWidth");
  const worldHeight = positiveFinite(input.worldHeight, "worldHeight");
  const viewportWidth = positiveFinite(input.viewportWidth, "viewportWidth");
  const viewportHeight = positiveFinite(input.viewportHeight, "viewportHeight");
  const padding = normalizeInsets(input.padding);
  const availableWidth = viewportWidth - padding.left - padding.right;
  const availableHeight = viewportHeight - padding.top - padding.bottom;
  if (availableWidth <= 0 || availableHeight <= 0) {
    throw new RangeError("viewport padding must leave a positive map rectangle.");
  }

  const scale = Math.min(availableWidth / worldWidth, availableHeight / worldHeight);
  const fittedWidth = worldWidth * scale;
  const fittedHeight = worldHeight * scale;
  const left = padding.left + (availableWidth - fittedWidth) / 2;
  const top = padding.top + (availableHeight - fittedHeight) / 2;
  const viewportBounds = rectangle(0, 0, viewportWidth, viewportHeight);
  const mapBounds = rectangle(left, top, left + fittedWidth, top + fittedHeight);

  return Object.freeze({
    scale,
    worldWidth,
    worldHeight,
    viewportBounds,
    mapBounds,
    worldToScreen(point) {
      const world = normalizePoint(point, "world point");
      return frozenPoint(
        mapBounds.left + world.x * scale,
        mapBounds.top + (worldHeight - world.y) * scale,
      );
    },
    screenToWorld(point) {
      const screen = normalizePoint(point, "screen point");
      return frozenPoint(
        (screen.x - mapBounds.left) / scale,
        worldHeight - (screen.y - mapBounds.top) / scale,
      );
    },
    worldLengthToScreen(length) {
      return nonNegativeFinite(length, "world length") * scale;
    },
  });
}

/**
 * Return the immutable screen-space rectangle reserved for one agent body,
 * its durable rings, and any selected/controlled focus decoration.
 *
 * @param {ProtectedBodyInput} body
 * @param {ProtectedBodyOptions} [options]
 * @returns {Rectangle}
 */
export function protectedBodyRect(body, options = {}) {
  if (!isRecord(body)) {
    throw new TypeError("protected body input must be an object.");
  }
  const center = normalizePoint(body.center, "body center");
  const radius = positiveFinite(body.radius, "body radius");
  const bodyPadding = optionNonNegative(
    options.bodyPadding,
    DEFAULT_STATUS_DOCK_OPTIONS.bodyPadding,
    "bodyPadding",
  );
  const selectionAllowance = optionNonNegative(
    options.selectionAllowance,
    DEFAULT_STATUS_DOCK_OPTIONS.selectionAllowance,
    "selectionAllowance",
  );
  const focusAllowance =
    body.controlled === true || body.selected === true ? selectionAllowance : 0;
  const extent = radius + bodyPadding + focusAllowance;
  return rectangle(
    center.x - extent,
    center.y - extent,
    center.x + extent,
    center.y + extent,
  );
}

/**
 * Whether two rectangles have positive-area overlap. Merely touching edges
 * is allowed.
 *
 * @param {Rectangle} first
 * @param {Rectangle} second
 * @returns {boolean}
 */
export function rectanglesIntersect(first, second) {
  const a = normalizeRectangle(first, "first rectangle");
  const b = normalizeRectangle(second, "second rectangle");
  return intersectionArea(a, b) > EPSILON;
}

/**
 * Sum the screen-pixel distance by which a rectangle escapes a viewport.
 *
 * @param {Rectangle} bounds
 * @param {Rectangle} viewport
 * @returns {number}
 */
export function viewportOverflow(bounds, viewport) {
  const candidate = normalizeRectangle(bounds, "bounds");
  const boundary = normalizeRectangle(viewport, "viewport");
  return (
    Math.max(0, boundary.left - candidate.left) +
    Math.max(0, candidate.right - boundary.right) +
    Math.max(0, boundary.top - candidate.top) +
    Math.max(0, candidate.bottom - boundary.bottom)
  );
}

/**
 * Lay out status docks from already-authorized screen-space agent facts.
 * The function never sorts or interprets statuses; their payload order is
 * retained verbatim.
 *
 * @param {StatusDockLayoutInput} input
 * @param {StatusDockOptions} [options]
 * @returns {StatusDockLayout}
 */
export function layoutStatusDocks(input, options = {}) {
  if (!isRecord(input) || !Array.isArray(input.agents)) {
    throw new TypeError("status dock input must contain an agents array.");
  }
  const viewport = normalizeRectangle(input.viewport, "viewport");
  const reservedRects = (input.reservedRects ?? []).map((bounds, index) =>
    normalizeRectangle(bounds, `reservedRects[${index}]`),
  );
  const resolvedOptions = resolveDockOptions(options);
  const agents = input.agents.map(normalizeAgent);
  assertUniqueSlots(agents);

  const protectedBodies = agents
    .map((agent) => ({
      globalSlot: agent.globalSlot,
      bounds: protectedBodyRect(agent, resolvedOptions),
    }))
    .sort((a, b) => a.globalSlot - b.globalSlot);
  const bodyRects = protectedBodies.map(({ bounds }) => bounds);
  const placementAgents = agents
    .filter((agent) => agent.statuses.length > 0)
    .sort(comparePlacementPriority);
  const placementInputs = placementAgents.map((agent, priorityIndex) => {
    const bodyBounds = protectedBodies.find(
      ({ globalSlot }) => globalSlot === agent.globalSlot,
    )?.bounds;
    if (!bodyBounds) {
      throw new Error(`missing protected body for slot ${agent.globalSlot}.`);
    }
    return Object.freeze({
      agent,
      priorityIndex,
      bodyBounds,
      viewport,
      bodyAndReservedRects: Object.freeze([...bodyRects, ...reservedRects]),
      options: resolvedOptions,
    });
  });
  const requiredInputs = placementInputs.filter(
    ({ agent }) => agent.required || agent.controlled || agent.selected,
  );
  const requiredPlacements = searchStatusDockPlacements(requiredInputs);
  /** @type {StatusDockPlacement[]} */
  const docks = requiredPlacements ? [...requiredPlacements] : [];
  /** @type {number[]} */
  const suppressedGlobalSlots = requiredPlacements
    ? []
    : requiredInputs.map(({ agent }) => agent.globalSlot);
  const priorDockBounds = docks.map(({ bounds }) => bounds);
  for (const input of placementInputs.filter(
    ({ agent }) => !agent.required && !agent.controlled && !agent.selected,
  )) {
    const placement = statusDockPlacementOptions({
      ...input,
      priorDockBounds,
    }).find((candidate) => candidate.collisionFree);
    if (!placement) {
      suppressedGlobalSlots.push(input.agent.globalSlot);
      continue;
    }
    docks.push(placement);
    priorDockBounds.push(placement.bounds);
  }
  return Object.freeze({
    docks: Object.freeze([...docks].sort((a, b) => a.globalSlot - b.globalSlot)),
    protectedBodies: Object.freeze(
      protectedBodies.map((entry) => Object.freeze(entry)),
    ),
    placementOrder: Object.freeze(placementAgents.map(({ globalSlot }) => globalSlot)),
    suppressedGlobalSlots: Object.freeze(
      [...suppressedGlobalSlots].sort((a, b) => a - b),
    ),
  });
}

/**
 * Jointly place heterogeneous required docks around one shared body field.
 *
 * Required status truth and required cooldown cues can have different cell
 * dimensions while still participating in the same deterministic search.
 * This avoids the priority inversion produced by laying out one category and
 * asking the other category to fit only after the first result is fixed.
 *
 * @param {RequiredDockLayoutInput} input
 * @param {StatusDockOptions} [options]
 * @returns {RequiredDockLayout}
 */
export function layoutRequiredDocks(input, options = {}) {
  if (
    !isRecord(input) ||
    !Array.isArray(input.agents) ||
    !Array.isArray(input.requests)
  ) {
    throw new TypeError("required dock input must contain agents and requests arrays.");
  }
  const viewport = normalizeRectangle(input.viewport, "viewport");
  const reservedRects = (input.reservedRects ?? []).map((bounds, index) =>
    normalizeRectangle(bounds, `reservedRects[${index}]`),
  );
  const sharedOptions = resolveDockOptions(options);
  const agents = input.agents.map((agent) =>
    normalizeAgent({ ...agent, statuses: [] }),
  );
  assertUniqueSlots(agents);
  const protectedBodies = agents
    .map((agent) => ({
      globalSlot: agent.globalSlot,
      bounds: protectedBodyRect(agent, sharedOptions),
    }))
    .sort((a, b) => a.globalSlot - b.globalSlot);
  const bodyRects = protectedBodies.map(({ bounds }) => bounds);
  const bodyBySlot = new Map(agents.map((agent) => [agent.globalSlot, agent]));
  const bodyBoundsBySlot = new Map(
    protectedBodies.map(({ globalSlot, bounds }) => [globalSlot, bounds]),
  );
  const seenLayoutKeys = new Set();
  const requests = input.requests
    .map((request) => {
      if (!isRecord(request)) {
        throw new TypeError("each required dock request must be an object.");
      }
      if (typeof request.layoutKey !== "string" || request.layoutKey.length === 0) {
        throw new TypeError("required dock layoutKey must be a non-empty string.");
      }
      if (seenLayoutKeys.has(request.layoutKey)) {
        throw new RangeError(`duplicate required dock layoutKey ${request.layoutKey}.`);
      }
      seenLayoutKeys.add(request.layoutKey);
      if (!Number.isInteger(request.globalSlot) || request.globalSlot < 0) {
        throw new RangeError(
          "required dock globalSlot must be a non-negative integer.",
        );
      }
      if (!Array.isArray(request.statuses) || request.statuses.length === 0) {
        throw new RangeError(
          "each required dock request must contain at least one status.",
        );
      }
      const body = bodyBySlot.get(request.globalSlot);
      if (!body) {
        throw new RangeError(
          `required dock ${request.layoutKey} references missing slot ${request.globalSlot}.`,
        );
      }
      const priority =
        request.priority === undefined
          ? 0
          : nonNegativeFinite(request.priority, "required dock priority");
      return Object.freeze({
        agent: normalizeAgent({
          ...body,
          statuses: request.statuses,
          required: true,
        }),
        dockOptions: resolveDockOptions({
          ...sharedOptions,
          ...(request.dockOptions ?? {}),
        }),
        fallbackDockOptions: resolveDockOptions({
          ...sharedOptions,
          ...REQUIRED_DOCK_FALLBACK_OPTIONS,
          ...(request.fallbackDockOptions ?? {}),
        }),
        layoutKey: request.layoutKey,
        priority,
      });
    })
    .sort(
      (first, second) =>
        first.priority - second.priority ||
        Number(second.agent.controlled) - Number(first.agent.controlled) ||
        Number(second.agent.selected) - Number(first.agent.selected) ||
        second.agent.statuses.length - first.agent.statuses.length ||
        first.agent.globalSlot - second.agent.globalSlot ||
        first.layoutKey.localeCompare(second.layoutKey),
    );
  const placementInputs = requests.map((request, priorityIndex) => {
    const bodyBounds = bodyBoundsBySlot.get(request.agent.globalSlot);
    if (!bodyBounds) {
      throw new Error(`missing protected body for slot ${request.agent.globalSlot}.`);
    }
    return Object.freeze({
      agent: request.agent,
      priorityIndex,
      bodyBounds,
      viewport,
      bodyAndReservedRects: Object.freeze([...bodyRects, ...reservedRects]),
      options: request.dockOptions,
    });
  });
  const { placements, suppressedPriorityIndexes } =
    searchPriorityPreservingDockPlacements(placementInputs);
  const fullPlacements = placements.map((placement) =>
    Object.freeze({
      ...placement,
      layoutKey: requests[placement.priorityIndex].layoutKey,
      compactFallback: false,
    }),
  );
  const occupiedDockBounds = fullPlacements.map(({ bounds }) => bounds);
  /** @type {RequiredDockPlacement[]} */
  const compactFallbacks = [];
  /** @type {number[]} */
  const unplacedPriorityIndexes = [];
  for (const priorityIndex of suppressedPriorityIndexes) {
    const request = requests[priorityIndex];
    const placementInput = placementInputs[priorityIndex];
    const compactPlacement = placeCompactRequiredDockFallback(
      {
        ...placementInput,
        options: request.fallbackDockOptions,
      },
      occupiedDockBounds,
    );
    if (compactPlacement === null) {
      unplacedPriorityIndexes.push(priorityIndex);
      continue;
    }
    compactFallbacks.push(
      Object.freeze({
        ...compactPlacement,
        layoutKey: request.layoutKey,
        compactFallback: true,
      }),
    );
    occupiedDockBounds.push(compactPlacement.bounds);
  }
  const docks = [...fullPlacements, ...compactFallbacks]
    .map((placement) =>
      Object.freeze({
        ...placement,
      }),
    )
    .sort(
      (first, second) =>
        first.globalSlot - second.globalSlot ||
        first.layoutKey.localeCompare(second.layoutKey),
    );
  return Object.freeze({
    docks: Object.freeze(docks),
    protectedBodies: Object.freeze(
      protectedBodies.map((entry) => Object.freeze(entry)),
    ),
    placementOrder: Object.freeze(requests.map(({ layoutKey }) => layoutKey)),
    compactedLayoutKeys: Object.freeze(
      compactFallbacks.map(({ layoutKey }) => layoutKey).sort(),
    ),
    suppressedLayoutKeys: Object.freeze(
      unplacedPriorityIndexes
        .map((priorityIndex) => requests[priorityIndex].layoutKey)
        .sort(),
    ),
  });
}

/**
 * Allocate one immutable presentation ledger across every enabled phase.
 * Rectangular semantic cues reserve space before routes. Their direct
 * connectors are a low-priority underlay and never participate in placement;
 * only the information-bearing cue rectangles avoid durable and peer bounds.
 * Routes occupy a dedicated layer behind those cues, so cue rectangles are not
 * route blockers. Routes still avoid every non-allowed durable protected region
 * and may cross one another; deterministic bridge-gap metadata marks route
 * crossings for the painter.
 *
 * Disabled requests are removed before their geometry fields are inspected.
 * An enabled request is either placed or causes a loud bounded-layout error;
 * this API never converts an information-bearing fact into suppression.
 *
 * @param {CrossPhaseLayoutInput} input
 * @param {Partial<typeof DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS>} [options]
 */
export function layoutCrossPhaseOccupancy(input, options = {}) {
  if (!isRecord(input) || !Array.isArray(input.requests)) {
    throw new TypeError("cross-phase layout input must contain a requests array.");
  }
  const viewport = normalizeRectangle(input.viewport, "viewport");
  const resolved = resolveCrossPhaseOptions(options);
  const protectedRegions = normalizeCrossPhaseProtectedRegions(
    input.protectedRects ?? [],
  );
  const { requests, filteredLayoutKeys } = normalizeCrossPhaseRequests(input.requests);
  const protectedKeys = new Set(protectedRegions.map(({ layoutKey }) => layoutKey));
  for (const { layoutKey } of requests) {
    if (protectedKeys.has(layoutKey)) {
      throw new RangeError(`duplicate cross-phase layoutKey ${layoutKey}.`);
    }
  }
  const ordered = [...requests].sort(compareCrossPhaseRequests);
  const cueRequests = ordered.filter(isCrossPhaseCueRequest);
  const routeRequests = ordered.filter(isCrossPhaseRouteRequest);
  const occupied = protectedRegions.map(({ bounds }) =>
    expandRectangle(bounds, resolved.clearance),
  );
  /** @type {Map<string, number>} */
  const recipientCounts = new Map();
  /** @type {Array<Record<string, any>>} */
  const cuePlacements = [];
  const cueAnchors = cueRequests.map(({ layoutKey, anchor }) => ({
    layoutKey,
    anchor,
  }));
  const cueRequestByKey = new Map(
    cueRequests.map((request) => [request.layoutKey, request]),
  );
  for (const request of cueRequests) {
    const recipientKey = request.recipientKey;
    const stackIndex = recipientCounts.get(recipientKey) ?? 0;
    recipientCounts.set(recipientKey, stackIndex + 1);
    const localCandidates =
      request.kind === "recipient_cue"
        ? crossPhaseRecipientCandidates(request, viewport, resolved)
        : [];
    const selected =
      firstFreeCueCandidate(
        localCandidates,
        occupied,
        request.anchor,
        cueAnchors,
        request.layoutKey,
      ) ??
      firstFreeCueCandidate(
        crossPhasePerimeterCandidates(request, viewport, occupied),
        occupied,
        request.anchor,
        cueAnchors,
        request.layoutKey,
      );
    if (selected === null) {
      throw new RangeError(
        `cross-phase cue ${request.layoutKey} has no bounded collision-free placement.`,
      );
    }
    const { candidate, leader } = selected;
    occupied.push(expandRectangle(candidate.bounds, resolved.clearance));
    cuePlacements.push(
      Object.freeze({
        layoutKey: request.layoutKey,
        kind: request.kind,
        priority: request.priority,
        stableOrder: request.stableOrder,
        recipientKey,
        stackIndex,
        disposition: localCandidates.includes(candidate)
          ? "recipient_stack"
          : "perimeter_callout",
        center: candidate.center,
        bounds: candidate.bounds,
        leader,
        collisionFree: true,
      }),
    );
  }

  // The greedy pass above is the total, proven-feasible allocation. Compact
  // only after every cue owns a valid placement, and treat every other final
  // cue as immutable so a visual improvement can never starve a later fact.
  // A second angular phase is reserved for cues still beyond their first outer
  // ring; nearby cues do not pay for refinement they cannot visibly need.
  for (const compactAngleOffsetDegrees of [
    CROSS_PHASE_RECIPIENT_COMPACTION_ANGLE_OFFSET_DEGREES,
    CROSS_PHASE_RECIPIENT_REFINEMENT_ANGLE_OFFSET_DEGREES,
  ]) {
    for (let index = cuePlacements.length - 1; index >= 0; index -= 1) {
      const current = cuePlacements[index];
      const request = cueRequestByKey.get(current.layoutKey);
      if (request?.kind !== "recipient_cue") {
        continue;
      }
      const currentDistance = Math.hypot(
        current.center.x - request.anchor.x,
        current.center.y - request.anchor.y,
      );
      const firstOuterRingDistance =
        request.anchorRadius +
        resolved.cueGap +
        Math.hypot(request.width, request.height) / 2 +
        Math.max(request.width, request.height) +
        resolved.stackGap;
      if (
        compactAngleOffsetDegrees ===
          CROSS_PHASE_RECIPIENT_REFINEMENT_ANGLE_OFFSET_DEGREES &&
        currentDistance <= firstOuterRingDistance + EPSILON
      ) {
        continue;
      }
      const compactCandidates = crossPhaseRecipientCandidates(
        request,
        viewport,
        resolved,
        currentDistance,
        compactAngleOffsetDegrees,
      );
      if (compactCandidates.length === 0) {
        continue;
      }
      const otherPlacements = cuePlacements.filter(
        (_placement, placementIndex) => placementIndex !== index,
      );
      const compactOccupied = [
        ...protectedRegions.map(({ bounds }) =>
          expandRectangle(bounds, resolved.clearance),
        ),
        ...otherPlacements.map(({ bounds }) =>
          expandRectangle(bounds, resolved.clearance),
        ),
      ];
      /** @type {{candidate: CrossPhaseCueCandidate, leader: Record<string, any>} | null} */
      let replacement = null;
      for (const candidate of compactCandidates) {
        const distance = Math.hypot(
          candidate.center.x - request.anchor.x,
          candidate.center.y - request.anchor.y,
        );
        if (distance >= currentDistance - EPSILON) {
          continue;
        }
        const selected = firstFreeCueCandidate(
          [candidate],
          compactOccupied,
          request.anchor,
          cueAnchors,
          request.layoutKey,
        );
        if (selected === null) {
          continue;
        }
        // Candidates are ordered nearest-first. The first valid replacement is
        // already a strict center-distance improvement, so scanning equivalent
        // angles would add latency without strengthening cue placement.
        replacement = {
          candidate: selected.candidate,
          leader: selected.leader,
        };
        break;
      }
      if (replacement !== null) {
        cuePlacements[index] = Object.freeze({
          ...current,
          disposition: "recipient_stack",
          center: replacement.candidate.center,
          bounds: replacement.candidate.bounds,
          leader: replacement.leader,
        });
      }
    }
  }

  const routeSeeds = crossPhaseRouteSeeds(routeRequests, resolved.routeLaneSpacing);
  /** @type {CrossPhaseRoutePlacement[]} */
  const routePlacements = [];
  for (const request of routeRequests) {
    const allowed = new Set(request.allowProtectedKeys);
    // Cue paint is composited above the dedicated route layer. Treating those
    // rectangles as route blockers can make a fully valid dense scene
    // unplaceable without protecting any information-bearing foreground fact.
    // Published durable bounds already include their owner-specific paint
    // allowance (notably body and focus-ring padding). Route centerlines avoid
    // those exact rectangles; adding cue clearance again would double-pad them
    // and can put a clipped endpoint inside every candidate blocker.
    const blockers = protectedRegions
      .filter(({ layoutKey }) => !allowed.has(layoutKey))
      .map(({ bounds }) => bounds);
    const markerBlockers = protectedRegions.map(({ bounds }) => bounds);
    const paddedBlockers = blockers.map((bounds) =>
      expandRectangle(bounds, request.pathPadding),
    );
    const preferredOffset = routeSeeds.get(request.layoutKey) ?? 0;
    const offsets = crossPhaseRouteOffsets(
      preferredOffset,
      resolved.routeLaneSpacing,
      resolved.routeLaneSearch,
    );
    const markerVariants = [
      Object.freeze({ markerVariant: "full", markerPadding: request.markerPadding }),
      ...(request.compactMarkerPadding === null
        ? []
        : [
            Object.freeze({
              markerVariant: "compact",
              markerPadding: request.compactMarkerPadding,
            }),
          ]),
    ];
    /** @type {{lane: number, geometry: ReturnType<typeof createRouteGeometry>, markerVariant: "full" | "compact", markerPadding: number} | null} */
    let selected = null;
    for (const { markerVariant, markerPadding } of markerVariants) {
      for (const [lane, offset] of offsets.entries()) {
        const preferredGeometry = createRouteGeometry(
          {
            eventId: request.layoutKey,
            source: request.source,
            target: request.target,
            sourceRadius: request.sourceRadius,
            targetRadius: request.targetRadius,
            sourceEndpointGap: request.sourceEndpointGap,
            targetEndpointGap: request.targetEndpointGap,
            offset,
          },
          {
            viewportBounds: viewport,
            routeMarkerPadding: markerPadding,
            markerProgress: request.markerProgress ?? undefined,
          },
        );
        for (const markerProgress of crossPhaseMarkerProgresses(
          preferredGeometry.markerProgress,
        )) {
          const geometry =
            markerProgress === preferredGeometry.markerProgress
              ? preferredGeometry
              : createRouteGeometry(
                  {
                    eventId: request.layoutKey,
                    source: request.source,
                    target: request.target,
                    sourceRadius: request.sourceRadius,
                    targetRadius: request.targetRadius,
                    sourceEndpointGap: request.sourceEndpointGap,
                    targetEndpointGap: request.targetEndpointGap,
                    offset,
                  },
                  {
                    viewportBounds: viewport,
                    routeMarkerPadding: markerPadding,
                    markerProgress,
                  },
                );
          if (
            routePaintIsClear(
              geometry,
              paddedBlockers,
              markerBlockers,
              viewport,
              request.pathPadding,
              markerPadding,
              geometry.markerProgress,
            )
          ) {
            selected = { lane, geometry, markerVariant, markerPadding };
            break;
          }
        }
        if (selected !== null) break;
      }
      if (selected === null) {
        const preferredGeometry = protectedPolylineRoute(
          request,
          protectedRegions,
          paddedBlockers,
          viewport,
        );
        if (preferredGeometry !== null) {
          if (preferredGeometry.kind !== "polyline") {
            throw new Error("Protected route fallback lost its polyline kind.");
          }
          for (const markerProgress of crossPhaseMarkerProgresses(
            preferredGeometry.markerProgress,
          )) {
            const geometry =
              markerProgress === preferredGeometry.markerProgress
                ? preferredGeometry
                : createPolylineRouteGeometry({
                    points: preferredGeometry.points,
                    offset: preferredGeometry.offset,
                    close: preferredGeometry.close,
                    markerProgress,
                  });
            if (
              routePaintIsClear(
                geometry,
                paddedBlockers,
                markerBlockers,
                viewport,
                request.pathPadding,
                markerPadding,
                geometry.markerProgress,
              )
            ) {
              selected = {
                lane: offsets.length,
                geometry,
                markerVariant,
                markerPadding,
              };
              break;
            }
          }
          if (selected === null) {
            const markerGeometry = markerSafePolylineRoute(
              preferredGeometry,
              paddedBlockers,
              markerBlockers,
              viewport,
              markerPadding,
              request.pathPadding,
            );
            if (
              markerGeometry !== null &&
              routePaintIsClear(
                markerGeometry,
                paddedBlockers,
                markerBlockers,
                viewport,
                request.pathPadding,
                markerPadding,
                markerGeometry.markerProgress,
              )
            ) {
              selected = {
                lane: offsets.length,
                geometry: markerGeometry,
                markerVariant,
                markerPadding,
              };
            }
          }
        }
      }
      if (selected !== null) break;
    }
    if (selected === null) {
      throw new RangeError(
        `cross-phase route ${request.layoutKey} has no bounded protected-region lane.`,
      );
    }
    routePlacements.push(
      Object.freeze({
        layoutKey: request.layoutKey,
        priority: request.priority,
        stableOrder: request.stableOrder,
        lane: selected.lane,
        bridgeGaps: Object.freeze([]),
        ...selected.geometry,
        markerVariant: selected.markerVariant,
        markerPadding: selected.markerPadding,
      }),
    );
  }
  const bridgedRoutes = addCrossPhaseRouteBridges(routePlacements, resolved.bridgeGap);
  const placementByKey = new Map(
    [...cuePlacements, ...bridgedRoutes].map((placement) => [
      placement.layoutKey,
      placement,
    ]),
  );
  const placements = ordered.map(({ layoutKey }) => placementByKey.get(layoutKey));
  if (placements.some((placement) => placement === undefined)) {
    throw new Error("cross-phase occupancy lost an enabled request.");
  }
  const occupancyLedger = Object.freeze([
    ...protectedRegions.map(({ layoutKey, bounds }) =>
      Object.freeze({ layoutKey, kind: "protected", bounds }),
    ),
    ...cuePlacements.map(({ layoutKey, bounds }) =>
      Object.freeze({ layoutKey, kind: "semantic_cue", bounds }),
    ),
    ...bridgedRoutes.map((route) =>
      Object.freeze({
        layoutKey: route.layoutKey,
        kind: "route",
        bounds: routeGeometryBounds(route),
      }),
    ),
  ]);
  return Object.freeze({
    viewport,
    placements: Object.freeze(placements),
    cuePlacements: Object.freeze(cuePlacements),
    routePlacements: Object.freeze(bridgedRoutes),
    occupancyLedger,
    protectedRegions,
    placementOrder: Object.freeze(ordered.map(({ layoutKey }) => layoutKey)),
    filteredLayoutKeys,
  });
}

/**
 * @param {Partial<typeof DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS>} options
 */
function resolveCrossPhaseOptions(options) {
  if (!isRecord(options)) {
    throw new TypeError("cross-phase layout options must be an object.");
  }
  const routeLaneSearch =
    options.routeLaneSearch === undefined
      ? DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS.routeLaneSearch
      : options.routeLaneSearch;
  if (
    !Number.isInteger(routeLaneSearch) ||
    routeLaneSearch < 0 ||
    routeLaneSearch > 32
  ) {
    throw new RangeError("routeLaneSearch must be an integer from 0 through 32.");
  }
  return Object.freeze({
    clearance: optionNonNegative(
      options.clearance,
      DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS.clearance,
      "clearance",
    ),
    cueGap: optionNonNegative(
      options.cueGap,
      DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS.cueGap,
      "cueGap",
    ),
    stackGap: optionNonNegative(
      options.stackGap,
      DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS.stackGap,
      "stackGap",
    ),
    routeLaneSpacing: optionPositive(
      options.routeLaneSpacing,
      DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS.routeLaneSpacing,
      "routeLaneSpacing",
    ),
    routeLaneSearch,
    bridgeGap: optionPositive(
      options.bridgeGap,
      DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS.bridgeGap,
      "bridgeGap",
    ),
  });
}

/**
 * @param {ReadonlyArray<CrossPhaseProtectedRegionInput>} regions
 */
function normalizeCrossPhaseProtectedRegions(regions) {
  if (!Array.isArray(regions)) {
    throw new TypeError("protectedRects must be an array.");
  }
  const seen = new Set();
  return Object.freeze(
    regions
      .map((region, index) => {
        if (!isRecord(region)) {
          throw new TypeError(`protectedRects[${index}] must be an object.`);
        }
        const layoutKey = nonEmptyString(
          region.layoutKey,
          `protectedRects[${index}].layoutKey`,
        );
        if (seen.has(layoutKey)) {
          throw new RangeError(`duplicate protected layoutKey ${layoutKey}.`);
        }
        seen.add(layoutKey);
        return Object.freeze({
          layoutKey,
          bounds: normalizeRectangle(
            region.bounds ?? region,
            `protectedRects[${index}]`,
          ),
        });
      })
      .sort((first, second) => first.layoutKey.localeCompare(second.layoutKey)),
  );
}

/**
 * Filter before inspecting any geometry-bearing request fields.
 *
 * @param {ReadonlyArray<CrossPhaseRequest>} rawRequests
 */
function normalizeCrossPhaseRequests(rawRequests) {
  const seen = new Set();
  /** @type {NormalizedCrossPhaseRequest[]} */
  const requests = [];
  /** @type {string[]} */
  const filteredLayoutKeys = [];
  for (const [index, raw] of rawRequests.entries()) {
    if (!isRecord(raw)) {
      throw new TypeError(`requests[${index}] must be an object.`);
    }
    const layoutKey = nonEmptyString(raw.layoutKey, `requests[${index}].layoutKey`);
    if (seen.has(layoutKey)) {
      throw new RangeError(`duplicate cross-phase layoutKey ${layoutKey}.`);
    }
    seen.add(layoutKey);
    if (raw.enabled !== undefined && typeof raw.enabled !== "boolean") {
      throw new TypeError(`requests[${index}].enabled must be boolean.`);
    }
    if (raw.enabled === false) {
      filteredLayoutKeys.push(layoutKey);
      continue;
    }
    const kind = raw.kind;
    if (kind !== "recipient_cue" && kind !== "perimeter_callout" && kind !== "route") {
      throw new RangeError(`requests[${index}].kind is unsupported.`);
    }
    if (!Number.isInteger(raw.stableOrder) || raw.stableOrder < 0) {
      throw new RangeError(
        `requests[${index}].stableOrder must be a non-negative integer.`,
      );
    }
    const priority = optionNonNegative(raw.priority, 0, `requests[${index}].priority`);
    if (kind === "route") {
      if (
        raw.allowProtectedKeys !== undefined &&
        !Array.isArray(raw.allowProtectedKeys)
      ) {
        throw new TypeError(`requests[${index}].allowProtectedKeys must be an array.`);
      }
      const allowProtectedKeys = (raw.allowProtectedKeys ?? []).map((key, keyIndex) =>
        nonEmptyString(key, `requests[${index}].allowProtectedKeys[${keyIndex}]`),
      );
      if (new Set(allowProtectedKeys).size !== allowProtectedKeys.length) {
        throw new RangeError(`requests[${index}].allowProtectedKeys must be unique.`);
      }
      const sourceProtectedKey =
        raw.sourceProtectedKey === undefined
          ? null
          : nonEmptyString(
              raw.sourceProtectedKey,
              `requests[${index}].sourceProtectedKey`,
            );
      const targetProtectedKey =
        raw.targetProtectedKey === undefined
          ? null
          : nonEmptyString(
              raw.targetProtectedKey,
              `requests[${index}].targetProtectedKey`,
            );
      for (const [role, key] of [
        ["source", sourceProtectedKey],
        ["target", targetProtectedKey],
      ]) {
        if (key !== null && !allowProtectedKeys.includes(key)) {
          throw new RangeError(
            `requests[${index}].${role}ProtectedKey must also be explicitly allowed.`,
          );
        }
      }
      const markerProgress =
        raw.markerProgress === undefined
          ? null
          : finite(raw.markerProgress, `requests[${index}].markerProgress`);
      if (markerProgress !== null && (markerProgress < 0 || markerProgress > 1)) {
        throw new RangeError(
          `requests[${index}].markerProgress must be between 0 and 1.`,
        );
      }
      const markerPadding = optionNonNegative(
        raw.markerPadding,
        0,
        `requests[${index}].markerPadding`,
      );
      const compactMarkerPadding =
        raw.compactMarkerPadding === undefined
          ? null
          : positiveFinite(
              raw.compactMarkerPadding,
              `requests[${index}].compactMarkerPadding`,
            );
      if (
        compactMarkerPadding !== null &&
        compactMarkerPadding >= markerPadding - EPSILON
      ) {
        throw new RangeError(
          `requests[${index}].compactMarkerPadding must be smaller than markerPadding.`,
        );
      }
      requests.push(
        Object.freeze({
          layoutKey,
          kind,
          priority,
          stableOrder: raw.stableOrder,
          source: normalizePoint(raw.source, `requests[${index}].source`),
          target: normalizePoint(raw.target, `requests[${index}].target`),
          sourceRadius: optionNonNegative(
            raw.sourceRadius,
            0,
            `requests[${index}].sourceRadius`,
          ),
          targetRadius: optionNonNegative(
            raw.targetRadius,
            0,
            `requests[${index}].targetRadius`,
          ),
          sourceEndpointGap: optionNonNegative(
            raw.sourceEndpointGap,
            3,
            `requests[${index}].sourceEndpointGap`,
          ),
          targetEndpointGap: optionNonNegative(
            raw.targetEndpointGap,
            3,
            `requests[${index}].targetEndpointGap`,
          ),
          pathPadding: optionNonNegative(
            raw.pathPadding,
            0,
            `requests[${index}].pathPadding`,
          ),
          markerPadding,
          compactMarkerPadding,
          markerProgress,
          allowProtectedKeys: Object.freeze([...allowProtectedKeys].sort()),
          sourceProtectedKey,
          targetProtectedKey,
        }),
      );
      continue;
    }
    const anchor = normalizePoint(raw.anchor, `requests[${index}].anchor`);
    if (
      raw.allowProtectedKeys !== undefined &&
      !Array.isArray(raw.allowProtectedKeys)
    ) {
      throw new TypeError(`requests[${index}].allowProtectedKeys must be an array.`);
    }
    const allowProtectedKeys = (raw.allowProtectedKeys ?? []).map((key, keyIndex) =>
      nonEmptyString(key, `requests[${index}].allowProtectedKeys[${keyIndex}]`),
    );
    if (new Set(allowProtectedKeys).size !== allowProtectedKeys.length) {
      throw new RangeError(`requests[${index}].allowProtectedKeys must be unique.`);
    }
    requests.push(
      Object.freeze({
        layoutKey,
        kind,
        priority,
        stableOrder: raw.stableOrder,
        anchor,
        anchorRadius: optionNonNegative(
          raw.anchorRadius,
          0,
          `requests[${index}].anchorRadius`,
        ),
        recipientKey:
          raw.recipientKey === undefined
            ? `point:${pointKey(anchor)}`
            : nonEmptyString(raw.recipientKey, `requests[${index}].recipientKey`),
        width: positiveFinite(raw.width, `requests[${index}].width`),
        height: positiveFinite(raw.height, `requests[${index}].height`),
        allowProtectedKeys: Object.freeze([...allowProtectedKeys].sort()),
      }),
    );
  }
  return Object.freeze({
    requests: Object.freeze(requests),
    filteredLayoutKeys: Object.freeze([...filteredLayoutKeys].sort()),
  });
}

/**
 * Lower numeric priority and earlier factual order place first. The key is the
 * total-order tie breaker that makes input permutations observationally equal.
 */
/**
 * @param {NormalizedCrossPhaseRequest} first
 * @param {NormalizedCrossPhaseRequest} second
 */
function compareCrossPhaseRequests(first, second) {
  return (
    first.priority - second.priority ||
    first.stableOrder - second.stableOrder ||
    first.layoutKey.localeCompare(second.layoutKey)
  );
}

/**
 * @param {NormalizedCrossPhaseRequest} request
 * @returns {request is NormalizedCrossPhaseCueRequest}
 */
function isCrossPhaseCueRequest(request) {
  return request.kind !== "route";
}

/**
 * @param {NormalizedCrossPhaseRequest} request
 * @returns {request is NormalizedCrossPhaseRouteRequest}
 */
function isCrossPhaseRouteRequest(request) {
  return request.kind === "route";
}

/**
 * @param {unknown} value
 * @param {string} name
 */
function nonEmptyString(value, name) {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`${name} must be a non-empty string.`);
  }
  return value;
}

/** @param {Point} point */
function pointKey(point) {
  return `${point.x},${point.y}`;
}

/**
 * @param {Rectangle} bounds
 * @param {number} amount
 */
function expandRectangle(bounds, amount) {
  return rectangle(
    bounds.left - amount,
    bounds.top - amount,
    bounds.right + amount,
    bounds.bottom + amount,
  );
}

/**
 * @param {Point} center
 * @param {number} width
 * @param {number} height
 * @returns {CrossPhaseCueCandidate}
 */
function cueCandidate(center, width, height) {
  return Object.freeze({
    center: frozenPoint(center.x, center.y),
    bounds: rectangle(
      center.x - width / 2,
      center.y - height / 2,
      center.x + width / 2,
      center.y + height / 2,
    ),
  });
}

/**
 * @param {NormalizedCrossPhaseCueRequest} request
 * @param {Rectangle} viewport
 * @param {ReturnType<typeof resolveCrossPhaseOptions>} options
 * @param {number | null} [compactBeforeDistance]
 * @param {number} [compactAngleOffsetDegrees]
 * @returns {ReadonlyArray<CrossPhaseCueCandidate>}
 */
function crossPhaseRecipientCandidates(
  request,
  viewport,
  options,
  compactBeforeDistance = null,
  compactAngleOffsetDegrees = CROSS_PHASE_RECIPIENT_COMPACTION_ANGLE_OFFSET_DEGREES,
) {
  const halfDiagonal = Math.hypot(request.width, request.height) / 2;
  const base = request.anchorRadius + options.cueGap + halfDiagonal;
  const stackStep = Math.max(request.width, request.height) + options.stackGap;
  const directions = [
    [0, -1],
    [1, 0],
    [-1, 0],
    [0, 1],
    [Math.SQRT1_2, -Math.SQRT1_2],
    [-Math.SQRT1_2, -Math.SQRT1_2],
    [Math.SQRT1_2, Math.SQRT1_2],
    [-Math.SQRT1_2, Math.SQRT1_2],
    ...(compactBeforeDistance === null
      ? []
      : Array.from({ length: 30 }, (_, index) => index).map((index) => {
          const degrees =
            compactAngleOffsetDegrees +
            index * CROSS_PHASE_RECIPIENT_COMPACTION_ANGLE_STEP_DEGREES;
          const radians = (degrees * Math.PI) / 180;
          return [Math.cos(radians), Math.sin(radians)];
        })),
  ];
  const distances =
    compactBeforeDistance === null
      ? Array.from({ length: 8 }, (_, ring) => base + ring * stackStep)
      : (() => {
          const maximumDistance = Math.min(
            compactBeforeDistance,
            base + 7 * stackStep + EPSILON,
          );
          const radialStep = Math.min(
            stackStep,
            CROSS_PHASE_RECIPIENT_COMPACTION_RADIAL_STEP,
          );
          const compactDistances = [base];
          for (
            let distance =
              base +
              Math.min(radialStep, CROSS_PHASE_RECIPIENT_COMPACTION_RADIAL_OFFSET);
            distance < maximumDistance - EPSILON;
            distance += radialStep
          ) {
            compactDistances.push(distance);
          }
          for (let ring = 0; ring < 8; ring += 1) {
            const distance = base + ring * stackStep;
            if (
              distance < maximumDistance - EPSILON &&
              !compactDistances.some(
                (candidate) => Math.abs(candidate - distance) <= EPSILON,
              )
            ) {
              compactDistances.push(distance);
            }
          }
          return compactDistances.sort((first, second) => first - second);
        })();
  /** @type {CrossPhaseCueCandidate[]} */
  const candidates = [];
  const seen = new Set();
  for (const distance of distances) {
    // Earlier cues already occupy their chosen rectangles. Starting every cue
    // at the nearest ring lets actual collisions—not ordinal position—decide
    // when a farther placement is necessary.
    for (const [dx, dy] of directions) {
      const candidate = cueCandidate(
        {
          x: request.anchor.x + dx * distance,
          y: request.anchor.y + dy * distance,
        },
        request.width,
        request.height,
      );
      const key = `${candidate.bounds.left},${candidate.bounds.top}`;
      if (!seen.has(key) && viewportOverflow(candidate.bounds, viewport) <= EPSILON) {
        seen.add(key);
        candidates.push(candidate);
      }
    }
  }
  return Object.freeze(candidates);
}

/**
 * @param {NormalizedCrossPhaseCueRequest} request
 * @param {Rectangle} viewport
 * @param {ReadonlyArray<Rectangle>} occupied
 * @returns {ReadonlyArray<CrossPhaseCueCandidate>}
 */
function crossPhasePerimeterCandidates(request, viewport, occupied) {
  const xPositions = candidateEdgePositions(
    viewport.left,
    viewport.right,
    request.width,
    occupied.map(({ left, right }) => ({ start: left, end: right })),
  );
  const yPositions = candidateEdgePositions(
    viewport.top,
    viewport.bottom,
    request.height,
    occupied.map(({ top, bottom }) => ({ start: top, end: bottom })),
  );
  return Object.freeze(
    yPositions
      .flatMap((top) =>
        xPositions.map((left) =>
          cueCandidate(
            { x: left + request.width / 2, y: top + request.height / 2 },
            request.width,
            request.height,
          ),
        ),
      )
      .sort((first, second) => {
        const firstCenter = first.center;
        const secondCenter = second.center;
        const firstPerimeter = Math.min(
          first.bounds.left - viewport.left,
          viewport.right - first.bounds.right,
          first.bounds.top - viewport.top,
          viewport.bottom - first.bounds.bottom,
        );
        const secondPerimeter = Math.min(
          second.bounds.left - viewport.left,
          viewport.right - second.bounds.right,
          second.bounds.top - viewport.top,
          viewport.bottom - second.bounds.bottom,
        );
        return (
          // The perimeter is a bounded fallback inventory, not a visual goal.
          // Prefer the valid fallback closest to the authorized event anchor.
          Math.hypot(
            firstCenter.x - request.anchor.x,
            firstCenter.y - request.anchor.y,
          ) -
            Math.hypot(
              secondCenter.x - request.anchor.x,
              secondCenter.y - request.anchor.y,
            ) ||
          firstPerimeter - secondPerimeter ||
          first.bounds.top - second.bounds.top ||
          first.bounds.left - second.bounds.left
        );
      }),
  );
}

/**
 * @param {ReadonlyArray<CrossPhaseCueCandidate>} candidates
 * @param {ReadonlyArray<Rectangle>} occupied
 * @param {Point} anchor
 * @param {ReadonlyArray<{layoutKey: string, anchor: Point}>} cueAnchors
 * @param {string} layoutKey
 * @returns {{candidate: CrossPhaseCueCandidate, leader: Record<string, any>} | null}
 */
function firstFreeCueCandidate(candidates, occupied, anchor, cueAnchors, layoutKey) {
  for (const candidate of candidates) {
    if (!occupied.every((bounds) => !rectanglesIntersect(candidate.bounds, bounds))) {
      continue;
    }
    const candidatePaintBounds = expandRectangle(
      candidate.bounds,
      CROSS_PHASE_CUE_PAINT_PADDING,
    );
    if (
      cueAnchors.some(
        ({ layoutKey: anchorLayoutKey, anchor: protectedAnchor }) =>
          anchorLayoutKey !== layoutKey &&
          pointTouchesRectangle(protectedAnchor, candidatePaintBounds),
      )
    ) {
      continue;
    }
    return Object.freeze({
      candidate,
      leader: cueLeader(anchor, candidate.center),
    });
  }
  return null;
}

/**
 * @param {Point} anchor
 * @param {Point} center
 */
function cueLeader(anchor, center) {
  const end = frozenPoint(center.x, center.y);
  return Object.freeze({
    kind: "line",
    start: anchor,
    end,
    points: Object.freeze([anchor, end]),
    path: `M ${numberKey(anchor.x)} ${numberKey(anchor.y)} L ${numberKey(end.x)} ${numberKey(end.y)}`,
  });
}

/**
 * @param {ReadonlyArray<Point>} points
 * @param {ReadonlyArray<Rectangle>} blockers
 * @param {Rectangle} viewport
 */
function polylineIsClear(points, blockers, viewport) {
  if (points.some((point) => !pointInOrOnRectangle(point, viewport))) {
    return false;
  }
  for (let index = 1; index < points.length; index += 1) {
    if (
      blockers.some((bounds) =>
        segmentIntersectsRectangle(points[index - 1], points[index], bounds),
      )
    ) {
      return false;
    }
  }
  return true;
}

/**
 * @param {Point} start
 * @param {Point} end
 * @param {ReadonlyArray<Rectangle>} blockers
 * @param {Rectangle} viewport
 * @returns {ReadonlyArray<Point> | null}
 */
function boundedVisibilityPolyline(start, end, blockers, viewport) {
  if (
    !pointInOrOnRectangle(start, viewport) ||
    !pointInOrOnRectangle(end, viewport) ||
    blockers.some(
      (bounds) =>
        pointTouchesRectangle(start, bounds) || pointTouchesRectangle(end, bounds),
    )
  ) {
    return null;
  }
  const nodeByPoint = new Map();
  for (const point of [
    start,
    end,
    ...blockers.flatMap((bounds) => visibilityCorners(bounds)),
  ]) {
    if (
      pointInOrOnRectangle(point, viewport) &&
      blockers.every((bounds) => !pointTouchesRectangle(point, bounds))
    ) {
      nodeByPoint.set(pointKey(point), point);
    }
  }
  const nodes = [...nodeByPoint.values()].sort(
    (left, right) => left.x - right.x || left.y - right.y,
  );
  if (nodes.length < 2 || nodes.length > CROSS_PHASE_ROUTE_GRAPH_NODE_LIMIT) {
    return null;
  }
  const startIndex = nodes.findIndex(
    (point) => point.x === start.x && point.y === start.y,
  );
  const endIndex = nodes.findIndex((point) => point.x === end.x && point.y === end.y);
  if (startIndex < 0 || endIndex < 0) {
    return null;
  }
  /** @type {Array<Array<{index: number, distance: number}>>} */
  const adjacency = Array.from({ length: nodes.length }, () => []);
  for (let first = 0; first < nodes.length; first += 1) {
    for (let second = first + 1; second < nodes.length; second += 1) {
      const distance = Math.hypot(
        nodes[second].x - nodes[first].x,
        nodes[second].y - nodes[first].y,
      );
      if (
        distance <= EPSILON ||
        blockers.some((bounds) =>
          segmentIntersectsRectangle(nodes[first], nodes[second], bounds),
        )
      ) {
        continue;
      }
      adjacency[first].push({ index: second, distance });
      adjacency[second].push({ index: first, distance });
    }
  }
  const distances = Array(nodes.length).fill(Number.POSITIVE_INFINITY);
  const signatures = Array(nodes.length).fill("");
  const previous = Array(nodes.length).fill(-1);
  const visited = Array(nodes.length).fill(false);
  distances[startIndex] = 0;
  signatures[startIndex] = graphPointSignature(start);
  for (let visit = 0; visit < nodes.length; visit += 1) {
    let current = -1;
    for (let index = 0; index < nodes.length; index += 1) {
      if (visited[index] || !Number.isFinite(distances[index])) continue;
      if (
        current < 0 ||
        distances[index] < distances[current] - EPSILON ||
        (Math.abs(distances[index] - distances[current]) <= EPSILON &&
          signatures[index].localeCompare(signatures[current]) < 0)
      ) {
        current = index;
      }
    }
    if (current < 0 || current === endIndex) break;
    visited[current] = true;
    for (const edge of adjacency[current]) {
      if (visited[edge.index]) continue;
      const distance = distances[current] + edge.distance;
      const signature = `${signatures[current]}>${graphPointSignature(nodes[edge.index])}`;
      if (
        distance < distances[edge.index] - EPSILON ||
        (Math.abs(distance - distances[edge.index]) <= EPSILON &&
          (signatures[edge.index] === "" ||
            signature.localeCompare(signatures[edge.index]) < 0))
      ) {
        distances[edge.index] = distance;
        signatures[edge.index] = signature;
        previous[edge.index] = current;
      }
    }
  }
  if (!Number.isFinite(distances[endIndex])) {
    return null;
  }
  const reversed = [];
  for (let index = endIndex; index >= 0; index = previous[index]) {
    reversed.push(nodes[index]);
    if (index === startIndex) break;
  }
  if (reversed.at(-1) !== nodes[startIndex]) {
    return null;
  }
  const points = simplifyPolyline(reversed.reverse());
  return polylineIsClear(points, blockers, viewport) ? points : null;
}

/**
 * @param {ReadonlyArray<NormalizedCrossPhaseRouteRequest>} requests
 * @param {number} spacing
 * @returns {Map<string, number>}
 */
function crossPhaseRouteSeeds(requests, spacing) {
  /** @type {Map<string, NormalizedCrossPhaseRouteRequest[]>} */
  const groups = new Map();
  for (const request of requests) {
    const sourceKey = pointKey(request.source);
    const targetKey = pointKey(request.target);
    const pair = [sourceKey, targetKey].sort().join("<->");
    const group = groups.get(pair) ?? [];
    group.push(request);
    groups.set(pair, group);
  }
  const seeds = new Map();
  for (const pair of [...groups.keys()].sort()) {
    const group = groups.get(pair) ?? [];
    /** @type {Map<string, NormalizedCrossPhaseRouteRequest[]>} */
    const directions = new Map();
    for (const request of group) {
      const direction = `${pointKey(request.source)}->${pointKey(request.target)}`;
      const entries = directions.get(direction) ?? [];
      entries.push(request);
      directions.set(direction, entries);
    }
    const reciprocal = directions.size > 1;
    for (const direction of [...directions.keys()].sort()) {
      const entries = [...(directions.get(direction) ?? [])].sort(
        compareCrossPhaseRequests,
      );
      for (const [index, request] of entries.entries()) {
        const centered = index - (entries.length - 1) / 2;
        seeds.set(
          request.layoutKey,
          reciprocal ? spacing * (index + 1) : spacing * centered,
        );
      }
    }
  }
  return seeds;
}

/**
 * @param {number} preferred
 * @param {number} spacing
 * @param {number} search
 */
function crossPhaseRouteOffsets(preferred, spacing, search) {
  const offsets = [preferred];
  for (let lane = 1; lane <= search; lane += 1) {
    offsets.push(preferred + lane * spacing, preferred - lane * spacing);
  }
  return Object.freeze([...new Set(offsets)]);
}

/**
 * Keep the authored marker position as the fast path, then search a small,
 * symmetric interior set. The chosen progress is stored on the immutable
 * route geometry, so allocation and paint consume exactly the same pose.
 *
 * @param {number} preferred
 */
function crossPhaseMarkerProgresses(preferred) {
  return Object.freeze([...new Set([preferred, 0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8])]);
}

/**
 * Find a bounded, deterministic multi-segment route. Named endpoint bodies use
 * only their exact planner-supplied keys; free endpoints remain exact unless a
 * durable foreground region already occludes them. Quadratic lanes stay the
 * fast path; this graph is the dense-scene completeness fallback.
 *
 * @param {NormalizedCrossPhaseRouteRequest} request
 * @param {ReadonlyArray<{layoutKey: string, bounds: Rectangle}>} protectedRegions
 * @param {ReadonlyArray<Rectangle>} blockers
 * @param {Rectangle} viewport
 * @returns {ReturnType<typeof createPolylineRouteGeometry> | null}
 */
function protectedPolylineRoute(request, protectedRegions, blockers, viewport) {
  const regionByKey = new Map(
    protectedRegions.map(({ layoutKey, bounds }) => [layoutKey, bounds]),
  );
  const sourceBounds =
    request.sourceProtectedKey === null
      ? null
      : (regionByKey.get(request.sourceProtectedKey) ?? null);
  const targetBounds =
    request.targetProtectedKey === null
      ? null
      : (regionByKey.get(request.targetProtectedKey) ?? null);
  if (
    (request.sourceProtectedKey !== null && !sourceBounds) ||
    (request.targetProtectedKey !== null && !targetBounds) ||
    (sourceBounds !== null && !pointInOrOnRectangle(request.source, sourceBounds)) ||
    (targetBounds !== null && !pointInOrOnRectangle(request.target, targetBounds))
  ) {
    return null;
  }

  const waypoints = blockers
    .flatMap((bounds) => visibilityCorners(bounds))
    .filter(
      (candidate) =>
        pointInOrOnRectangle(candidate, viewport) &&
        blockers.every((bounds) => !pointTouchesRectangle(candidate, bounds)),
    );
  const sourcePorts = sourceBounds
    ? bodyBoundaryPorts(sourceBounds, [request.target, ...waypoints])
    : freeEndpointPorts(request.source, blockers, viewport);
  const targetPorts = targetBounds
    ? bodyBoundaryPorts(targetBounds, [request.source, ...waypoints])
    : freeEndpointPorts(request.target, blockers, viewport);

  /** @type {Map<string, {point: Point, source: boolean, target: boolean}>} */
  const nodeByPoint = new Map();
  /** @param {Point} candidate @param {"source" | "target" | "waypoint"} role */
  const addNode = (candidate, role) => {
    if (
      !pointInOrOnRectangle(candidate, viewport) ||
      blockers.some((bounds) => pointTouchesRectangle(candidate, bounds))
    ) {
      return;
    }
    const key = pointKey(candidate);
    const prior = nodeByPoint.get(key);
    nodeByPoint.set(key, {
      point: prior?.point ?? candidate,
      source: prior?.source === true || role === "source",
      target: prior?.target === true || role === "target",
    });
  };
  for (const candidate of sourcePorts) addNode(candidate, "source");
  for (const candidate of targetPorts) addNode(candidate, "target");
  for (const candidate of waypoints) addNode(candidate, "waypoint");
  const nodes = [...nodeByPoint.values()].sort(
    (first, second) =>
      first.point.x - second.point.x ||
      first.point.y - second.point.y ||
      Number(second.source) - Number(first.source) ||
      Number(second.target) - Number(first.target),
  );
  if (
    nodes.length < 2 ||
    nodes.length > CROSS_PHASE_ROUTE_GRAPH_NODE_LIMIT ||
    !nodes.some(({ source }) => source) ||
    !nodes.some(({ target }) => target)
  ) {
    return null;
  }

  /** @type {Array<Array<{index: number, distance: number}>>} */
  const adjacency = Array.from({ length: nodes.length }, () => []);
  for (let first = 0; first < nodes.length; first += 1) {
    for (let second = first + 1; second < nodes.length; second += 1) {
      const start = nodes[first].point;
      const end = nodes[second].point;
      const distance = Math.hypot(end.x - start.x, end.y - start.y);
      if (
        distance <= EPSILON ||
        blockers.some((bounds) => segmentIntersectsRectangle(start, end, bounds))
      ) {
        continue;
      }
      adjacency[first].push({ index: second, distance });
      adjacency[second].push({ index: first, distance });
    }
  }

  const distances = Array(nodes.length).fill(Number.POSITIVE_INFINITY);
  const signatures = Array(nodes.length).fill("");
  const previous = Array(nodes.length).fill(-1);
  const visited = Array(nodes.length).fill(false);
  for (const [index, node] of nodes.entries()) {
    if (node.source) {
      distances[index] = 0;
      signatures[index] = graphPointSignature(node.point);
    }
  }
  let destination = -1;
  for (let visit = 0; visit < nodes.length; visit += 1) {
    let current = -1;
    for (let index = 0; index < nodes.length; index += 1) {
      if (visited[index] || !Number.isFinite(distances[index])) continue;
      if (
        current === -1 ||
        distances[index] < distances[current] - EPSILON ||
        (Math.abs(distances[index] - distances[current]) <= EPSILON &&
          signatures[index].localeCompare(signatures[current]) < 0)
      ) {
        current = index;
      }
    }
    if (current === -1) break;
    visited[current] = true;
    if (nodes[current].target && distances[current] > EPSILON) {
      destination = current;
      break;
    }
    for (const edge of adjacency[current]) {
      if (visited[edge.index]) continue;
      const distance = distances[current] + edge.distance;
      const signature = `${signatures[current]}>${graphPointSignature(
        nodes[edge.index].point,
      )}`;
      if (
        distance < distances[edge.index] - EPSILON ||
        (Math.abs(distance - distances[edge.index]) <= EPSILON &&
          (signatures[edge.index] === "" ||
            signature.localeCompare(signatures[edge.index]) < 0))
      ) {
        distances[edge.index] = distance;
        signatures[edge.index] = signature;
        previous[edge.index] = current;
      }
    }
  }
  if (destination === -1) {
    return null;
  }

  /** @type {Point[]} */
  const reversed = [];
  for (let index = destination; index !== -1; index = previous[index]) {
    reversed.push(nodes[index].point);
  }
  const points = simplifyPolyline(reversed.reverse());
  return points.length >= 2
    ? createPolylineRouteGeometry({
        points,
        offset: 0,
        close: false,
        markerProgress: request.markerProgress ?? undefined,
      })
    : null;
}

/**
 * Detour an otherwise valid protected polyline through one point where the
 * complete visible marker fits. Endpoint bodies remain path-only allowances:
 * marker candidates are checked against every durable region, including both
 * explicitly named endpoint owners.
 *
 * @param {ReturnType<typeof createPolylineRouteGeometry>} route
 * @param {ReadonlyArray<Rectangle>} pathBlockers
 * @param {ReadonlyArray<Rectangle>} markerBlockers
 * @param {Rectangle} viewport
 * @param {number} markerPadding
 * @param {number} pathPadding
 */
function markerSafePolylineRoute(
  route,
  pathBlockers,
  markerBlockers,
  viewport,
  markerPadding,
  pathPadding,
) {
  if (
    markerPadding <= 0 ||
    viewport.width < markerPadding * 2 ||
    viewport.height < markerPadding * 2
  ) {
    return null;
  }
  const markerViewport = rectangle(
    viewport.left + markerPadding,
    viewport.top + markerPadding,
    viewport.right - markerPadding,
    viewport.bottom - markerPadding,
  );
  const paddedMarkerBlockers = markerBlockers.map((bounds) =>
    expandRectangle(bounds, markerPadding),
  );
  const candidates = [
    frozenPoint(
      (markerViewport.left + markerViewport.right) / 2,
      (markerViewport.top + markerViewport.bottom) / 2,
    ),
    frozenPoint(markerViewport.left, markerViewport.top),
    frozenPoint(markerViewport.right, markerViewport.top),
    frozenPoint(markerViewport.right, markerViewport.bottom),
    frozenPoint(markerViewport.left, markerViewport.bottom),
    ...paddedMarkerBlockers.flatMap((bounds) => visibilityCorners(bounds)),
  ];
  const markerCandidates = [
    ...new Map(
      candidates.map((candidate) => [pointKey(candidate), candidate]),
    ).values(),
  ]
    .filter(
      (candidate) =>
        pointInOrOnRectangle(candidate, markerViewport) &&
        paddedMarkerBlockers.every(
          (bounds) => !pointTouchesRectangle(candidate, bounds),
        ),
    )
    .sort(
      (first, second) =>
        markerDetourLowerBound(route, first) - markerDetourLowerBound(route, second) ||
        first.x - second.x ||
        first.y - second.y,
    );
  for (const candidate of markerCandidates) {
    const leading = boundedVisibilityPolyline(
      route.start,
      candidate,
      pathBlockers,
      viewport,
    );
    const trailing = boundedVisibilityPolyline(
      candidate,
      route.end,
      pathBlockers,
      viewport,
    );
    if (leading === null || trailing === null) {
      continue;
    }
    const points = Object.freeze([...leading, ...trailing.slice(1)]);
    const length = polylineLength(points);
    const markerDistance = polylineLength(leading);
    if (length <= EPSILON || markerDistance <= EPSILON) {
      continue;
    }
    const geometry = createPolylineRouteGeometry({
      points,
      offset: route.offset,
      close: route.close,
      markerProgress: markerDistance / length,
    });
    if (
      !routePaintIsClear(
        geometry,
        pathBlockers,
        markerBlockers,
        viewport,
        pathPadding,
        markerPadding,
        geometry.markerProgress,
      )
    ) {
      continue;
    }
    return geometry;
  }
  return null;
}

/**
 * @param {ReturnType<typeof createPolylineRouteGeometry>} route
 * @param {Point} candidate
 */
function markerDetourLowerBound(route, candidate) {
  return (
    Math.hypot(candidate.x - route.start.x, candidate.y - route.start.y) +
    Math.hypot(route.end.x - candidate.x, route.end.y - candidate.y)
  );
}

/** @param {ReadonlyArray<Point>} points */
function polylineLength(points) {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) {
    length += Math.hypot(
      points[index].x - points[index - 1].x,
      points[index].y - points[index - 1].y,
    );
  }
  return length;
}

/** @param {Rectangle} bounds */
function visibilityCorners(bounds) {
  const margin = CROSS_PHASE_ROUTE_GRAPH_MARGIN;
  return Object.freeze([
    frozenPoint(bounds.left - margin, bounds.top - margin),
    frozenPoint(bounds.right + margin, bounds.top - margin),
    frozenPoint(bounds.right + margin, bounds.bottom + margin),
    frozenPoint(bounds.left - margin, bounds.bottom + margin),
  ]);
}

/**
 * A free route endpoint or cue anchor normally remains exact. When durable
 * foreground already covers that point, begin or end the visible geometry
 * immediately outside the complete set of containing blockers. The occluded
 * scientific point remains unchanged; the blocker is never inferred as an
 * owner or removed from later collision checks.
 *
 * @param {Point} endpoint
 * @param {ReadonlyArray<Rectangle>} blockers
 * @param {Rectangle} viewport
 */
function freeEndpointPorts(endpoint, blockers, viewport) {
  const containing = blockers.filter((bounds) =>
    pointInOrOnRectangle(endpoint, bounds),
  );
  if (containing.length === 0) {
    return Object.freeze([endpoint]);
  }
  const margin = CROSS_PHASE_ROUTE_GRAPH_MARGIN;
  const candidates = containing.flatMap((bounds) => [
    frozenPoint(
      bounds.left - margin,
      Math.min(bounds.bottom, Math.max(bounds.top, endpoint.y)),
    ),
    frozenPoint(
      bounds.right + margin,
      Math.min(bounds.bottom, Math.max(bounds.top, endpoint.y)),
    ),
    frozenPoint(
      Math.min(bounds.right, Math.max(bounds.left, endpoint.x)),
      bounds.top - margin,
    ),
    frozenPoint(
      Math.min(bounds.right, Math.max(bounds.left, endpoint.x)),
      bounds.bottom + margin,
    ),
    ...visibilityCorners(bounds),
  ]);
  return Object.freeze(
    [
      ...new Map(
        candidates.map((candidate) => [pointKey(candidate), candidate]),
      ).values(),
    ]
      .filter(
        (candidate) =>
          pointInOrOnRectangle(candidate, viewport) &&
          blockers.every((bounds) => !pointTouchesRectangle(candidate, bounds)),
      )
      .sort((first, second) => first.x - second.x || first.y - second.y),
  );
}

/**
 * @param {Rectangle} bounds
 * @param {ReadonlyArray<Point>} aims
 */
function bodyBoundaryPorts(bounds, aims) {
  const center = frozenPoint(
    (bounds.left + bounds.right) / 2,
    (bounds.top + bounds.bottom) / 2,
  );
  const candidates = [
    frozenPoint(center.x, bounds.top),
    frozenPoint(bounds.right, center.y),
    frozenPoint(center.x, bounds.bottom),
    frozenPoint(bounds.left, center.y),
    frozenPoint(bounds.left, bounds.top),
    frozenPoint(bounds.right, bounds.top),
    frozenPoint(bounds.right, bounds.bottom),
    frozenPoint(bounds.left, bounds.bottom),
  ];
  for (const aim of aims) {
    const deltaX = aim.x - center.x;
    const deltaY = aim.y - center.y;
    if (Math.abs(deltaX) <= EPSILON && Math.abs(deltaY) <= EPSILON) continue;
    const halfWidth = bounds.width / 2;
    const halfHeight = bounds.height / 2;
    const scale =
      1 / Math.max(Math.abs(deltaX) / halfWidth, Math.abs(deltaY) / halfHeight);
    candidates.push(frozenPoint(center.x + deltaX * scale, center.y + deltaY * scale));
  }
  return Object.freeze([
    ...new Map(
      candidates.map((candidate) => [pointKey(candidate), candidate]),
    ).values(),
  ]);
}

/** @param {Point} point @param {Rectangle} bounds */
function pointInOrOnRectangle(point, bounds) {
  return (
    point.x >= bounds.left - EPSILON &&
    point.x <= bounds.right + EPSILON &&
    point.y >= bounds.top - EPSILON &&
    point.y <= bounds.bottom + EPSILON
  );
}

/** @param {Point} point @param {Rectangle} bounds */
function pointTouchesRectangle(point, bounds) {
  return pointInOrOnRectangle(point, bounds);
}

/** @param {Point} point */
function graphPointSignature(point) {
  return `${numberKey(point.x)},${numberKey(point.y)}`;
}

/** @param {number} value */
function numberKey(value) {
  return value.toPrecision(15);
}

/** @param {ReadonlyArray<Point>} points */
function simplifyPolyline(points) {
  /** @type {Point[]} */
  const simplified = [];
  for (const point of points) {
    while (simplified.length >= 2) {
      const first = simplified[simplified.length - 2];
      const second = simplified[simplified.length - 1];
      const cross =
        (second.x - first.x) * (point.y - second.y) -
        (second.y - first.y) * (point.x - second.x);
      if (Math.abs(cross) > EPSILON) break;
      simplified.pop();
    }
    simplified.push(point);
  }
  return Object.freeze(simplified);
}

/**
 * @param {ReturnType<typeof createRouteGeometry>} route
 * @returns {ReadonlyArray<Point>}
 */
function routePoints(route) {
  if (route.kind === "polyline") {
    return route.points;
  }
  if (route.kind === "curve") {
    return Object.freeze(
      Array.from({ length: CROSS_PHASE_ROUTE_SAMPLES + 1 }, (_, index) => {
        const progress = index / CROSS_PHASE_ROUTE_SAMPLES;
        const remainder = 1 - progress;
        return frozenPoint(
          remainder * remainder * route.start.x +
            2 * remainder * progress * route.control.x +
            progress * progress * route.end.x,
          remainder * remainder * route.start.y +
            2 * remainder * progress * route.control.y +
            progress * progress * route.end.y,
        );
      }),
    );
  }
  const startAngle = Math.atan2(
    route.start.y - route.center.y,
    route.start.x - route.center.x,
  );
  const endAngle = Math.atan2(
    route.end.y - route.center.y,
    route.end.x - route.center.x,
  );
  const positiveSweep =
    (((endAngle - startAngle) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const sweep = route.sweep === 1 ? positiveSweep : positiveSweep - 2 * Math.PI;
  return Object.freeze(
    Array.from({ length: CROSS_PHASE_ROUTE_SAMPLES + 1 }, (_, index) => {
      const angle = startAngle + sweep * (index / CROSS_PHASE_ROUTE_SAMPLES);
      return frozenPoint(
        route.center.x + Math.cos(angle) * route.arcRadius,
        route.center.y + Math.sin(angle) * route.arcRadius,
      );
    }),
  );
}

/**
 * @param {Point} point
 * @param {Rectangle} bounds
 */
function pointInRectangle(point, bounds) {
  return (
    point.x > bounds.left + EPSILON &&
    point.x < bounds.right - EPSILON &&
    point.y > bounds.top + EPSILON &&
    point.y < bounds.bottom - EPSILON
  );
}

/**
 * @param {Point} firstStart
 * @param {Point} firstEnd
 * @param {Point} secondStart
 * @param {Point} secondEnd
 * @returns {Point | null}
 */
function segmentsIntersect(firstStart, firstEnd, secondStart, secondEnd) {
  const firstX = firstEnd.x - firstStart.x;
  const firstY = firstEnd.y - firstStart.y;
  const secondX = secondEnd.x - secondStart.x;
  const secondY = secondEnd.y - secondStart.y;
  const denominator = firstX * secondY - firstY * secondX;
  if (Math.abs(denominator) <= EPSILON) {
    return null;
  }
  const deltaX = secondStart.x - firstStart.x;
  const deltaY = secondStart.y - firstStart.y;
  const firstProgress = (deltaX * secondY - deltaY * secondX) / denominator;
  const secondProgress = (deltaX * firstY - deltaY * firstX) / denominator;
  if (
    firstProgress < -EPSILON ||
    firstProgress > 1 + EPSILON ||
    secondProgress < -EPSILON ||
    secondProgress > 1 + EPSILON
  ) {
    return null;
  }
  return frozenPoint(
    firstStart.x + firstProgress * firstX,
    firstStart.y + firstProgress * firstY,
  );
}

/**
 * @param {Point} start
 * @param {Point} end
 * @param {Rectangle} bounds
 */
function segmentIntersectsRectangle(start, end, bounds) {
  if (pointInRectangle(start, bounds) || pointInRectangle(end, bounds)) {
    return true;
  }
  const corners = [
    frozenPoint(bounds.left, bounds.top),
    frozenPoint(bounds.right, bounds.top),
    frozenPoint(bounds.right, bounds.bottom),
    frozenPoint(bounds.left, bounds.bottom),
  ];
  return corners.some((corner, index) =>
    segmentsIntersect(start, end, corner, corners[(index + 1) % corners.length]),
  );
}

/**
 * Validate the visible route corridor and its finite direction marker. The
 * transparent 10px interaction path and battlefield-coloured bridge
 * backplates are deliberately not foreground collision claims.
 *
 * @param {ReturnType<typeof createRouteGeometry>} route
 * @param {ReadonlyArray<Rectangle>} paddedBlockers
 * @param {ReadonlyArray<Rectangle>} blockers
 * @param {Rectangle} viewport
 * @param {number} pathPadding
 * @param {number} markerPadding
 * @param {number | null} markerProgress
 */
function routePaintIsClear(
  route,
  paddedBlockers,
  blockers,
  viewport,
  pathPadding,
  markerPadding,
  markerProgress,
) {
  if (routeIntersectsRectangles(route, paddedBlockers)) {
    return false;
  }
  if (
    viewportOverflow(
      expandRectangle(routeGeometryBounds(route), pathPadding),
      viewport,
    ) > EPSILON
  ) {
    return false;
  }
  if (markerPadding <= 0) {
    return true;
  }
  const marker = routeMarkerPose(route, markerProgress ?? undefined);
  const markerViewport = rectangle(
    viewport.left + markerPadding,
    viewport.top + markerPadding,
    viewport.right - markerPadding,
    viewport.bottom - markerPadding,
  );
  return (
    pointInOrOnRectangle(marker, markerViewport) &&
    blockers.every(
      (bounds) =>
        !pointTouchesRectangle(marker, expandRectangle(bounds, markerPadding)),
    )
  );
}

/**
 * @param {ReturnType<typeof createRouteGeometry>} route
 * @param {ReadonlyArray<Rectangle>} blockers
 */
function routeIntersectsRectangles(route, blockers) {
  const points = routePoints(route);
  for (let index = 1; index < points.length; index += 1) {
    if (
      blockers.some((bounds) =>
        segmentIntersectsRectangle(points[index - 1], points[index], bounds),
      )
    ) {
      return true;
    }
  }
  return false;
}

/**
 * @param {ReadonlyArray<CrossPhaseRoutePlacement>} routes
 * @param {number} bridgeGap
 * @returns {ReadonlyArray<CrossPhaseRoutePlacement>}
 */
function addCrossPhaseRouteBridges(routes, bridgeGap) {
  /** @type {CrossPhaseRoutePlacement[]} */
  const completed = [];
  for (const route of routes) {
    const points = routePoints(route);
    /** @type {{withLayoutKey: string, at: Point, gap: number}[]} */
    const bridgeGaps = [];
    for (const earlier of completed) {
      const earlierPoints = routePoints(earlier);
      for (let index = 1; index < points.length; index += 1) {
        for (
          let earlierIndex = 1;
          earlierIndex < earlierPoints.length;
          earlierIndex += 1
        ) {
          const at = segmentsIntersect(
            points[index - 1],
            points[index],
            earlierPoints[earlierIndex - 1],
            earlierPoints[earlierIndex],
          );
          if (at === null || nearRouteEndpoint(at, route, bridgeGap)) {
            continue;
          }
          if (
            bridgeGaps.some(
              (bridge) =>
                Math.hypot(bridge.at.x - at.x, bridge.at.y - at.y) < bridgeGap,
            )
          ) {
            continue;
          }
          bridgeGaps.push(
            Object.freeze({
              withLayoutKey: earlier.layoutKey,
              at,
              gap: bridgeGap,
            }),
          );
        }
      }
    }
    completed.push(Object.freeze({ ...route, bridgeGaps: Object.freeze(bridgeGaps) }));
  }
  return Object.freeze(completed);
}

/**
 * @param {Point} point
 * @param {ReturnType<typeof createRouteGeometry>} route
 * @param {number} distance
 */
function nearRouteEndpoint(point, route, distance) {
  return (
    Math.hypot(point.x - route.start.x, point.y - route.start.y) < distance ||
    Math.hypot(point.x - route.end.x, point.y - route.end.y) < distance
  );
}

/** @param {ReturnType<typeof createRouteGeometry>} route */
function routeGeometryBounds(route) {
  const points = routePoints(route);
  return rectangle(
    Math.min(...points.map(({ x }) => x)),
    Math.min(...points.map(({ y }) => y)),
    Math.max(...points.map(({ x }) => x)),
    Math.max(...points.map(({ y }) => y)),
  );
}

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * @param {unknown} value
 * @param {string} name
 * @returns {number}
 */
function finite(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be finite.`);
  }
  return value;
}

/**
 * @param {unknown} value
 * @param {string} name
 * @returns {number}
 */
function positiveFinite(value, name) {
  const number = finite(value, name);
  if (number <= 0) {
    throw new RangeError(`${name} must be positive.`);
  }
  return number;
}

/**
 * @param {unknown} value
 * @param {string} name
 * @returns {number}
 */
function nonNegativeFinite(value, name) {
  const number = finite(value, name);
  if (number < 0) {
    throw new RangeError(`${name} must be non-negative.`);
  }
  return number;
}

/**
 * @param {unknown} value
 * @param {number} fallback
 * @param {string} name
 * @returns {number}
 */
function optionNonNegative(value, fallback, name) {
  return value === undefined ? fallback : nonNegativeFinite(value, name);
}

/**
 * @param {unknown} value
 * @param {number} fallback
 * @param {string} name
 * @returns {number}
 */
function optionPositive(value, fallback, name) {
  return value === undefined ? fallback : positiveFinite(value, name);
}

/**
 * @param {number | Partial<Insets> | undefined} value
 * @returns {Insets}
 */
function normalizeInsets(value) {
  if (value === undefined) {
    return Object.freeze({ top: 0, right: 0, bottom: 0, left: 0 });
  }
  if (typeof value === "number") {
    const padding = nonNegativeFinite(value, "padding");
    return Object.freeze({
      top: padding,
      right: padding,
      bottom: padding,
      left: padding,
    });
  }
  if (!isRecord(value)) {
    throw new TypeError("padding must be a number or inset object.");
  }
  return Object.freeze({
    top: optionNonNegative(value.top, 0, "padding.top"),
    right: optionNonNegative(value.right, 0, "padding.right"),
    bottom: optionNonNegative(value.bottom, 0, "padding.bottom"),
    left: optionNonNegative(value.left, 0, "padding.left"),
  });
}

/**
 * @param {unknown} value
 * @param {string} name
 * @returns {Point}
 */
function normalizePoint(value, name) {
  if (Array.isArray(value) && value.length >= 2) {
    return frozenPoint(finite(value[0], `${name}.x`), finite(value[1], `${name}.y`));
  }
  if (isRecord(value)) {
    return frozenPoint(finite(value.x, `${name}.x`), finite(value.y, `${name}.y`));
  }
  throw new TypeError(`${name} must contain finite x and y coordinates.`);
}

/**
 * @param {number} x
 * @param {number} y
 * @returns {Point}
 */
function frozenPoint(x, y) {
  return Object.freeze({ x, y });
}

/**
 * @param {number} left
 * @param {number} top
 * @param {number} right
 * @param {number} bottom
 * @returns {Rectangle}
 */
function rectangle(left, top, right, bottom) {
  return Object.freeze({
    left,
    top,
    right,
    bottom,
    width: right - left,
    height: bottom - top,
  });
}

/**
 * @param {unknown} value
 * @param {string} name
 * @returns {Rectangle}
 */
function normalizeRectangle(value, name) {
  if (!isRecord(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
  const left = finite(value.left, `${name}.left`);
  const top = finite(value.top, `${name}.top`);
  const right = finite(value.right, `${name}.right`);
  const bottom = finite(value.bottom, `${name}.bottom`);
  if (right < left || bottom < top) {
    throw new RangeError(`${name} must have ordered edges.`);
  }
  return rectangle(left, top, right, bottom);
}

/**
 * @param {Rectangle} first
 * @param {Rectangle} second
 * @returns {number}
 */
function intersectionArea(first, second) {
  const width = Math.max(
    0,
    Math.min(first.right, second.right) - Math.max(first.left, second.left),
  );
  const height = Math.max(
    0,
    Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top),
  );
  return width * height;
}

/**
 * @param {StatusDockOptions} options
 */
function resolveDockOptions(options) {
  const maxTangentShift = optionNonNegative(
    options.maxTangentShift,
    DEFAULT_STATUS_DOCK_OPTIONS.maxTangentShift,
    "maxTangentShift",
  );
  if (maxTangentShift > 24) {
    throw new RangeError("maxTangentShift may not exceed 24 CSS pixels.");
  }
  const ordinaryVisibleLimit =
    options.ordinaryVisibleLimit === undefined
      ? DEFAULT_STATUS_DOCK_OPTIONS.ordinaryVisibleLimit
      : options.ordinaryVisibleLimit;
  if (
    !Number.isInteger(ordinaryVisibleLimit) ||
    ordinaryVisibleLimit < 0 ||
    ordinaryVisibleLimit > STATUS_DOCK_CAPACITY
  ) {
    throw new RangeError(
      `ordinaryVisibleLimit must be an integer from 0 through ${STATUS_DOCK_CAPACITY}.`,
    );
  }
  const requiredVisibleLimit =
    options.requiredVisibleLimit === undefined
      ? DEFAULT_STATUS_DOCK_OPTIONS.requiredVisibleLimit
      : options.requiredVisibleLimit;
  if (
    !Number.isInteger(requiredVisibleLimit) ||
    requiredVisibleLimit < 0 ||
    requiredVisibleLimit > STATUS_DOCK_CAPACITY
  ) {
    throw new RangeError(
      `requiredVisibleLimit must be an integer from 0 through ${STATUS_DOCK_CAPACITY}.`,
    );
  }
  const tangentStep = optionPositive(
    options.tangentStep,
    DEFAULT_STATUS_DOCK_OPTIONS.tangentStep,
    "tangentStep",
  );
  return Object.freeze({
    bodyPadding: optionNonNegative(
      options.bodyPadding,
      DEFAULT_STATUS_DOCK_OPTIONS.bodyPadding,
      "bodyPadding",
    ),
    selectionAllowance: optionNonNegative(
      options.selectionAllowance,
      DEFAULT_STATUS_DOCK_OPTIONS.selectionAllowance,
      "selectionAllowance",
    ),
    cellWidth: optionPositive(
      options.cellWidth,
      DEFAULT_STATUS_DOCK_OPTIONS.cellWidth,
      "cellWidth",
    ),
    cellHeight: optionPositive(
      options.cellHeight,
      DEFAULT_STATUS_DOCK_OPTIONS.cellHeight,
      "cellHeight",
    ),
    cellGap: optionNonNegative(
      options.cellGap,
      DEFAULT_STATUS_DOCK_OPTIONS.cellGap,
      "cellGap",
    ),
    dockGap: optionNonNegative(
      options.dockGap,
      DEFAULT_STATUS_DOCK_OPTIONS.dockGap,
      "dockGap",
    ),
    tangentStep,
    maxTangentShift,
    ordinaryVisibleLimit,
    requiredVisibleLimit,
  });
}

/**
 * @param {StatusDockAgent} agent
 */
function normalizeAgent(agent) {
  if (!isRecord(agent)) {
    throw new TypeError("each status dock agent must be an object.");
  }
  if (!Number.isInteger(agent.globalSlot) || agent.globalSlot < 0) {
    throw new RangeError("agent globalSlot must be a non-negative integer.");
  }
  if (!Array.isArray(agent.statuses)) {
    throw new TypeError("agent statuses must be an array.");
  }
  return Object.freeze({
    globalSlot: agent.globalSlot,
    center: normalizePoint(agent.center, "agent center"),
    radius: positiveFinite(agent.radius, "agent radius"),
    statuses: Object.freeze([...agent.statuses]),
    required: agent.required === true,
    controlled: agent.controlled === true,
    selected: agent.selected === true,
  });
}

/**
 * @param {ReadonlyArray<ReturnType<typeof normalizeAgent>>} agents
 */
function assertUniqueSlots(agents) {
  const slots = new Set();
  for (const { globalSlot } of agents) {
    if (slots.has(globalSlot)) {
      throw new RangeError(`duplicate agent globalSlot ${globalSlot}.`);
    }
    slots.add(globalSlot);
  }
}

/**
 * @param {ReturnType<typeof normalizeAgent>} first
 * @param {ReturnType<typeof normalizeAgent>} second
 * @returns {number}
 */
function comparePlacementPriority(first, second) {
  return (
    Number(second.required) - Number(first.required) ||
    Number(second.controlled) - Number(first.controlled) ||
    Number(second.selected) - Number(first.selected) ||
    second.statuses.length - first.statuses.length ||
    first.globalSlot - second.globalSlot
  );
}

/**
 * @typedef {{
 *   agent: ReturnType<typeof normalizeAgent>,
 *   priorityIndex: number,
 *   bodyBounds: Rectangle,
 *   viewport: Rectangle,
 *   bodyAndReservedRects: ReadonlyArray<Rectangle>,
 *   options: ReturnType<typeof resolveDockOptions>,
 * }} StatusDockPlacementInput
 * @typedef {StatusDockPlacementInput & {
 *   priorDockBounds: ReadonlyArray<Rectangle>,
 * }} PlaceStatusDockInput
 */

/**
 * Find the first complete collision-free arrangement in priority order.
 * Candidate order preserves expanded-before-collapsed disclosure and the
 * deterministic anchor/tangent score. The bounded fallback below retains a
 * usable result when the local geometry is physically unsatisfiable.
 *
 * @param {ReadonlyArray<StatusDockPlacementInput>} inputs
 * @param {number} [searchLimit]
 * @returns {ReadonlyArray<StatusDockPlacement> | null}
 */
function searchStatusDockPlacements(inputs, searchLimit = STATUS_DOCK_SEARCH_LIMIT) {
  let visited = 0;

  /**
   * @param {number} index
   * @param {ReadonlyArray<Rectangle>} priorDockBounds
   * @param {ReadonlyArray<StatusDockPlacement>} placements
   * @returns {ReadonlyArray<StatusDockPlacement> | null}
   */
  function visit(index, priorDockBounds, placements) {
    if (index === inputs.length) {
      return placements;
    }
    if (visited >= searchLimit) {
      return null;
    }
    const input = inputs[index];
    const candidates = statusDockPlacementOptions({
      ...input,
      priorDockBounds,
    });
    for (const candidate of candidates) {
      visited += 1;
      if (!candidate.collisionFree) {
        continue;
      }
      const result = visit(
        index + 1,
        [...priorDockBounds, candidate.bounds],
        [...placements, candidate],
      );
      if (result) {
        return result;
      }
      if (visited >= searchLimit) {
        return null;
      }
    }
    return null;
  }

  return visit(0, [], []);
}

/**
 * Preserve the complete joint result when it exists. If one lower-priority
 * request makes the complete set impossible, retain every higher-priority
 * request that remains jointly feasible and continue considering later
 * requests around that accepted set. A rejected request can never displace an
 * earlier accepted request.
 *
 * @param {ReadonlyArray<StatusDockPlacementInput>} inputs
 * @returns {{
 *   placements: ReadonlyArray<StatusDockPlacement>,
 *   suppressedPriorityIndexes: ReadonlyArray<number>,
 * }}
 */
function searchPriorityPreservingDockPlacements(inputs) {
  if (inputs.length <= REQUIRED_DOCK_JOINT_SEARCH_MAX_REQUESTS) {
    const completePlacements = searchStatusDockPlacements(
      inputs,
      REQUIRED_DOCK_SEARCH_LIMIT,
    );
    if (completePlacements !== null) {
      return Object.freeze({
        placements: completePlacements,
        suppressedPriorityIndexes: Object.freeze([]),
      });
    }
  }

  /** @type {StatusDockPlacement[]} */
  const placements = [];
  /** @type {Rectangle[]} */
  const priorDockBounds = [];
  /** @type {number[]} */
  const suppressedPriorityIndexes = [];
  for (const input of inputs) {
    const placement = statusDockPlacementOptions({
      ...input,
      priorDockBounds,
    }).find((candidate) => candidate.collisionFree);
    if (!placement) {
      suppressedPriorityIndexes.push(input.priorityIndex);
      continue;
    }
    placements.push(placement);
    priorDockBounds.push(placement.bounds);
  }
  return Object.freeze({
    placements: Object.freeze(placements),
    suppressedPriorityIndexes: Object.freeze(suppressedPriorityIndexes),
  });
}

/**
 * Reduce one unplaceable required request to a single associated marker while
 * retaining the complete authoritative payload behind that marker. Local
 * anchors are preferred. A deterministic viewport search is the final
 * collision-safe fallback for dense but supported layouts.
 *
 * @param {StatusDockPlacementInput} input
 * @param {ReadonlyArray<Rectangle>} priorDockBounds
 * @returns {StatusDockPlacement | null}
 */
function placeCompactRequiredDockFallback(input, priorDockBounds) {
  const markerInput = Object.freeze({
    ...input,
    agent: Object.freeze({
      ...input.agent,
      statuses: Object.freeze([null]),
      required: true,
    }),
  });
  const localPlacement = statusDockPlacementOptions({
    ...markerInput,
    priorDockBounds,
  }).find((candidate) => candidate.collisionFree);
  const markerPlacement =
    localPlacement ?? remoteCompactRequiredDockPlacement(markerInput, priorDockBounds);
  if (markerPlacement === null) {
    return null;
  }
  const hiddenStatuses = Object.freeze([...input.agent.statuses]);
  return Object.freeze({
    ...markerPlacement,
    required: true,
    expanded: false,
    visibleStatuses: Object.freeze([]),
    hiddenStatuses,
    visibleCount: 0,
    hiddenCount: hiddenStatuses.length,
    totalCount: hiddenStatuses.length,
    overflowLabel: `+${hiddenStatuses.length}`,
  });
}

/**
 * Find a remote one-cell marker when every normal anchor is occupied. Candidate
 * coordinates are derived from viewport and blocker edges, which bounds the
 * search quadratically in the number of rectangles rather than in viewport
 * pixels. Any axis-aligned free region has an equivalent placement touching a
 * viewport or blocker edge, so this remains complete for the compact marker.
 *
 * @param {StatusDockPlacementInput} input
 * @param {ReadonlyArray<Rectangle>} priorDockBounds
 * @returns {StatusDockPlacement | null}
 */
function remoteCompactRequiredDockPlacement(input, priorDockBounds) {
  const dimensions = dockDimensions(1, input.options);
  const blockers = [...input.bodyAndReservedRects, ...priorDockBounds];
  const xPositions = candidateEdgePositions(
    input.viewport.left,
    input.viewport.right,
    dimensions.width,
    blockers.map(({ left, right }) => ({ start: left, end: right })),
  );
  const yPositions = candidateEdgePositions(
    input.viewport.top,
    input.viewport.bottom,
    dimensions.height,
    blockers.map(({ top, bottom }) => ({ start: top, end: bottom })),
  );
  const bodyCenter = rectangleCenter(input.bodyBounds);
  /** @type {{
   *   anchor: "north" | "east" | "west" | "south",
   *   bounds: Rectangle,
   *   distance: number,
   *   score: StatusDockScore,
   * } | null} */
  let best = null;
  for (const top of yPositions) {
    for (const left of xPositions) {
      const bounds = rectangle(
        left,
        top,
        left + dimensions.width,
        top + dimensions.height,
      );
      const score = scoreCandidate({
        bounds,
        viewport: input.viewport,
        bodyAndReservedRects: input.bodyAndReservedRects,
        priorDockBounds,
        tangentShift: 0,
        anchorIndex: 0,
      });
      if (
        score.viewportOverflow > EPSILON ||
        score.bodyOrReservedIntersection > EPSILON ||
        score.priorDockIntersection > EPSILON
      ) {
        continue;
      }
      const markerCenter = rectangleCenter(bounds);
      const distance = Math.hypot(
        markerCenter.x - bodyCenter.x,
        markerCenter.y - bodyCenter.y,
      );
      const anchor = relativeAnchor(bodyCenter, markerCenter);
      if (
        best === null ||
        distance < best.distance - EPSILON ||
        (Math.abs(distance - best.distance) <= EPSILON &&
          (bounds.top < best.bounds.top - EPSILON ||
            (Math.abs(bounds.top - best.bounds.top) <= EPSILON &&
              bounds.left < best.bounds.left - EPSILON)))
      ) {
        best = { anchor, bounds, distance, score };
      }
    }
  }
  if (best === null) {
    return null;
  }
  return Object.freeze({
    globalSlot: input.agent.globalSlot,
    priorityIndex: input.priorityIndex,
    required: true,
    controlled: input.agent.controlled,
    selected: input.agent.selected,
    anchor: best.anchor,
    tangentShift: 0,
    bounds: best.bounds,
    leader: leaderLine(input.bodyBounds, best.bounds, best.anchor),
    columns: 1,
    rows: 1,
    expanded: true,
    collisionFree: true,
    visibleStatuses: Object.freeze([null]),
    hiddenStatuses: Object.freeze([]),
    visibleCount: 1,
    hiddenCount: 0,
    totalCount: 1,
    overflowLabel: null,
    score: best.score,
  });
}

/**
 * Candidate starts where a compact rectangle touches either the viewport or
 * one blocker edge. Filtering and sorting make the result deterministic.
 *
 * @param {number} viewportStart
 * @param {number} viewportEnd
 * @param {number} extent
 * @param {ReadonlyArray<{start: number, end: number}>} blockers
 * @returns {ReadonlyArray<number>}
 */
function candidateEdgePositions(viewportStart, viewportEnd, extent, blockers) {
  const latestStart = viewportEnd - extent;
  if (latestStart < viewportStart - EPSILON) {
    return Object.freeze([]);
  }
  const positions = new Set([viewportStart, latestStart]);
  for (const blocker of blockers) {
    positions.add(blocker.start - extent);
    positions.add(blocker.end);
  }
  return Object.freeze(
    [...positions]
      .filter(
        (position) =>
          position >= viewportStart - EPSILON && position <= latestStart + EPSILON,
      )
      .map((position) => Math.min(latestStart, Math.max(viewportStart, position)))
      .sort((first, second) => first - second)
      .filter(
        (position, index, ordered) =>
          index === 0 || Math.abs(position - ordered[index - 1]) > EPSILON,
      ),
  );
}

/**
 * @param {Rectangle} bounds
 * @returns {Point}
 */
function rectangleCenter(bounds) {
  return frozenPoint(
    (bounds.left + bounds.right) / 2,
    (bounds.top + bounds.bottom) / 2,
  );
}

/**
 * @param {Point} source
 * @param {Point} target
 * @returns {"north" | "east" | "west" | "south"}
 */
function relativeAnchor(source, target) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  if (Math.abs(dx) > Math.abs(dy)) {
    return dx >= 0 ? "east" : "west";
  }
  return dy >= 0 ? "south" : "north";
}

/**
 * Generate deterministic placement alternatives in disclosure-priority order:
 * every expanded anchor is considered before an ordinary dock collapses.
 *
 * @param {PlaceStatusDockInput} input
 * @returns {StatusDockPlacement[]}
 */
function statusDockPlacementOptions(input) {
  const totalCount = input.agent.statuses.length;
  const capacityExceeded = totalCount > STATUS_DOCK_CAPACITY;
  const expandedVisibleCount = capacityExceeded ? STATUS_DOCK_CAPACITY - 1 : totalCount;
  const required =
    input.agent.required || input.agent.controlled || input.agent.selected;
  const visibleLimit = required
    ? input.options.requiredVisibleLimit
    : input.options.ordinaryVisibleLimit;
  const initialVisibleCount = Math.min(expandedVisibleCount, visibleLimit);
  const forceExpanded =
    !capacityExceeded && required && initialVisibleCount === totalCount;
  const visibleCounts = [initialVisibleCount];
  if (!forceExpanded) {
    const maximumCollapsedVisible = Math.min(
      initialVisibleCount - 1,
      totalCount - 1,
      STATUS_DOCK_CAPACITY - 1,
    );
    for (
      let visibleCount = maximumCollapsedVisible;
      visibleCount >= 0;
      visibleCount -= 1
    ) {
      if (!visibleCounts.includes(visibleCount)) {
        visibleCounts.push(visibleCount);
      }
    }
  }
  return visibleCounts.flatMap((visibleCount) =>
    [
      ...buildCandidates({
        ...input,
        visibleCount,
        hiddenCount: totalCount - visibleCount,
      }),
    ]
      .sort(compareCandidates)
      .map((candidate) => placementFromCandidate(input, candidate)),
  );
}

/**
 * @param {PlaceStatusDockInput} input
 * @param {ReturnType<typeof buildCandidates>[number]} candidate
 * @returns {StatusDockPlacement}
 */
function placementFromCandidate(input, candidate) {
  const visibleStatuses = Object.freeze(
    input.agent.statuses.slice(0, candidate.visibleCount),
  );
  const hiddenStatuses = Object.freeze(
    input.agent.statuses.slice(candidate.visibleCount),
  );
  return Object.freeze({
    globalSlot: input.agent.globalSlot,
    priorityIndex: input.priorityIndex,
    required: input.agent.required,
    controlled: input.agent.controlled,
    selected: input.agent.selected,
    anchor: candidate.anchor,
    tangentShift: candidate.tangentShift,
    bounds: candidate.bounds,
    leader: leaderLine(input.bodyBounds, candidate.bounds, candidate.anchor),
    columns: candidate.columns,
    rows: candidate.rows,
    expanded: hiddenStatuses.length === 0,
    collisionFree: candidate.collisionFree,
    visibleStatuses,
    hiddenStatuses,
    visibleCount: visibleStatuses.length,
    hiddenCount: hiddenStatuses.length,
    totalCount: input.agent.statuses.length,
    overflowLabel: hiddenStatuses.length > 0 ? `+${hiddenStatuses.length}` : null,
    score: candidate.score,
  });
}

/**
 * @typedef {PlaceStatusDockInput & {
 *   visibleCount: number,
 *   hiddenCount: number,
 * }} BuildCandidatesInput
 */

/**
 * @param {BuildCandidatesInput} input
 */
function buildCandidates(input) {
  const cellCount = input.visibleCount + (input.hiddenCount > 0 ? 1 : 0);
  const dimensions = dockDimensions(cellCount, input.options);
  const tangentShifts = candidateTangentShifts(
    input.options.tangentStep,
    input.options.maxTangentShift,
  );
  return STATUS_DOCK_ANCHORS.flatMap((anchor, anchorIndex) =>
    tangentShifts.map((tangentShift, candidateIndex) => {
      const bounds = anchoredDockBounds(
        input.bodyBounds,
        dimensions,
        anchor,
        tangentShift,
        input.options.dockGap,
      );
      const score = scoreCandidate({
        bounds,
        viewport: input.viewport,
        bodyAndReservedRects: input.bodyAndReservedRects,
        priorDockBounds: input.priorDockBounds,
        tangentShift,
        anchorIndex,
      });
      return Object.freeze({
        anchor,
        anchorIndex,
        tangentShift,
        candidateIndex,
        bounds,
        columns: dimensions.columns,
        rows: dimensions.rows,
        visibleCount: input.visibleCount,
        score,
        collisionFree:
          score.viewportOverflow <= EPSILON &&
          score.bodyOrReservedIntersection <= EPSILON &&
          score.priorDockIntersection <= EPSILON,
      });
    }),
  );
}

/**
 * @param {number} cellCount
 * @param {ReturnType<typeof resolveDockOptions>} options
 */
function dockDimensions(cellCount, options) {
  if (!Number.isInteger(cellCount) || cellCount <= 0) {
    throw new RangeError("a status dock must contain at least one cell.");
  }
  const columns = Math.min(3, cellCount);
  const rows = Math.ceil(cellCount / columns);
  if (rows > 3) {
    throw new RangeError("a status dock may not exceed three rows.");
  }
  return Object.freeze({
    columns,
    rows,
    width: columns * options.cellWidth + (columns - 1) * options.cellGap,
    height: rows * options.cellHeight + (rows - 1) * options.cellGap,
  });
}

/**
 * @param {number} step
 * @param {number} maximum
 * @returns {ReadonlyArray<number>}
 */
function candidateTangentShifts(step, maximum) {
  /** @type {number[]} */
  const shifts = [0];
  for (
    let displacement = step;
    displacement <= maximum + EPSILON;
    displacement += step
  ) {
    const bounded = Math.min(displacement, maximum);
    shifts.push(-bounded, bounded);
    if (Math.abs(bounded - maximum) <= EPSILON) {
      break;
    }
  }
  if (
    maximum > 0 &&
    !shifts.some((shift) => Math.abs(Math.abs(shift) - maximum) <= EPSILON)
  ) {
    shifts.push(-maximum, maximum);
  }
  return Object.freeze(shifts);
}

/**
 * @param {Rectangle} body
 * @param {{width: number, height: number}} dimensions
 * @param {"north" | "east" | "west" | "south"} anchor
 * @param {number} tangentShift
 * @param {number} gap
 * @returns {Rectangle}
 */
function anchoredDockBounds(body, dimensions, anchor, tangentShift, gap) {
  const centerX = (body.left + body.right) / 2;
  const centerY = (body.top + body.bottom) / 2;
  if (anchor === "north") {
    const left = centerX - dimensions.width / 2 + tangentShift;
    const bottom = body.top - gap;
    return rectangle(left, bottom - dimensions.height, left + dimensions.width, bottom);
  }
  if (anchor === "south") {
    const left = centerX - dimensions.width / 2 + tangentShift;
    const top = body.bottom + gap;
    return rectangle(left, top, left + dimensions.width, top + dimensions.height);
  }
  if (anchor === "east") {
    const left = body.right + gap;
    const top = centerY - dimensions.height / 2 + tangentShift;
    return rectangle(left, top, left + dimensions.width, top + dimensions.height);
  }
  const right = body.left - gap;
  const top = centerY - dimensions.height / 2 + tangentShift;
  return rectangle(right - dimensions.width, top, right, top + dimensions.height);
}

/**
 * @typedef {{
 *   bounds: Rectangle,
 *   viewport: Rectangle,
 *   bodyAndReservedRects: ReadonlyArray<Rectangle>,
 *   priorDockBounds: ReadonlyArray<Rectangle>,
 *   tangentShift: number,
 *   anchorIndex: number,
 * }} ScoreCandidateInput
 */

/**
 * @param {ScoreCandidateInput} input
 * @returns {StatusDockScore}
 */
function scoreCandidate(input) {
  const bodyOrReservedIntersection = input.bodyAndReservedRects.reduce(
    (total, bounds) => total + intersectionArea(input.bounds, bounds),
    0,
  );
  const priorDockIntersection = input.priorDockBounds.reduce(
    (total, bounds) => total + intersectionArea(input.bounds, bounds),
    0,
  );
  return Object.freeze({
    viewportOverflow: viewportOverflow(input.bounds, input.viewport),
    bodyOrReservedIntersection,
    priorDockIntersection,
    displacement: Math.abs(input.tangentShift),
    anchorIndex: input.anchorIndex,
  });
}

/**
 * @param {ReturnType<typeof buildCandidates>[number]} first
 * @param {ReturnType<typeof buildCandidates>[number]} second
 * @returns {number}
 */
function compareCandidates(first, second) {
  return (
    first.score.viewportOverflow - second.score.viewportOverflow ||
    first.score.bodyOrReservedIntersection - second.score.bodyOrReservedIntersection ||
    first.score.priorDockIntersection - second.score.priorDockIntersection ||
    first.score.displacement - second.score.displacement ||
    first.score.anchorIndex - second.score.anchorIndex ||
    first.candidateIndex - second.candidateIndex
  );
}

/**
 * @param {Rectangle} body
 * @param {Rectangle} dock
 * @param {"north" | "east" | "west" | "south"} anchor
 * @returns {LeaderLine}
 */
function leaderLine(body, dock, anchor) {
  const bodyCenterX = (body.left + body.right) / 2;
  const bodyCenterY = (body.top + body.bottom) / 2;
  const dockCenterX = (dock.left + dock.right) / 2;
  const dockCenterY = (dock.top + dock.bottom) / 2;
  if (anchor === "north") {
    return Object.freeze({
      start: frozenPoint(bodyCenterX, body.top),
      end: frozenPoint(dockCenterX, dock.bottom),
    });
  }
  if (anchor === "south") {
    return Object.freeze({
      start: frozenPoint(bodyCenterX, body.bottom),
      end: frozenPoint(dockCenterX, dock.top),
    });
  }
  if (anchor === "east") {
    return Object.freeze({
      start: frozenPoint(body.right, bodyCenterY),
      end: frozenPoint(dock.left, dockCenterY),
    });
  }
  return Object.freeze({
    start: frozenPoint(body.left, bodyCenterY),
    end: frozenPoint(dock.right, dockCenterY),
  });
}
