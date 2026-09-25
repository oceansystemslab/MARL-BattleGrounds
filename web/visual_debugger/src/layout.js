/**
 * @file Place battlefield labels and transient cues in screen space without changing game facts.
 * createViewportTransform owns the fitted world/pixel conversion. layoutStatusDocks
 * and layoutRequiredDocks place disclosed status payloads around protected bodies.
 * layoutCrossPhaseOccupancy places transient cue boxes and directed routes across
 * animation phases. These functions use deterministic geometry only: no DOM reads,
 * network calls, random draws or simulator decisions. Distances are CSS pixels
 * except the world dimensions accepted by the transform and redZoneFloorIntervals,
 * which turns a recorded map.red_zone into at most two merged floor strips in world
 * x units. Returned layout records are frozen; opaque status payload objects remain
 * owned by their caller.
 */
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

/** Default cue clearances and route lanes in CSS pixels; routeLaneSearch is a count. */
export const DEFAULT_CROSS_PHASE_LAYOUT_OPTIONS = Object.freeze({
  clearance: 3,
  cueGap: 8,
  stackGap: 5,
  routeLaneSpacing: 18,
  routeLaneSearch: 8,
  bridgeGap: 10,
});

/**
 * Preferred body sides, in deterministic tie-breaking order.
 * @type {ReadonlyArray<"north" | "east" | "west" | "south">}
 */
export const STATUS_DOCK_ANCHORS = Object.freeze(["north", "east", "west", "south"]);

/** Maximum dock cells, including any overflow-count cell, in a three-by-three grid. */
export const STATUS_DOCK_CAPACITY = 9;

/** Default body/label distances in CSS pixels and visible payload limits in cells. */
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
 * Fit a world rectangle inside a padded viewport while preserving its aspect ratio.
 *
 * input gives positive finite worldWidth/worldHeight in world units and
 * viewportWidth/viewportHeight in CSS pixels. padding defaults to zero; a finite
 * nonnegative number applies to all sides, or an inset object supplies individual
 * top/right/bottom/left values with omitted sides set to zero. Center the fitted
 * map in the remaining space. World Y increases upward; screen Y increases downward.
 *
 * Return a frozen transform with pixels-per-world-unit scale, original world
 * dimensions, full viewportBounds, fitted mapBounds and three conversion methods.
 * Conversions do not clamp points to the map. No input is changed. Throw TypeError
 * for malformed objects or nonfinite/nonnumeric values, and RangeError for
 * nonpositive dimensions, negative padding or padding that leaves no map area.
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
    /**
     * Convert a world point to CSS pixels using this fitted map.
     *
     * point is a finite {x,y} record or coordinate array with at least two entries.
     * Return a frozen {x,y} point, reversing the Y direction. Points outside the world
     * remain outside the fitted map. Throw TypeError for invalid coordinates.
     */
    worldToScreen(point) {
      const world = normalizePoint(point, "world point");
      return frozenPoint(
        mapBounds.left + world.x * scale,
        mapBounds.top + (worldHeight - world.y) * scale,
      );
    },
    /**
     * Convert a CSS-pixel point back to world coordinates.
     *
     * point is a finite {x,y} record or coordinate array with at least two entries.
     * Return a frozen {x,y} point without clipping; viewport padding can therefore
     * map outside the world. Throw TypeError for invalid coordinates.
     */
    screenToWorld(point) {
      const screen = normalizePoint(point, "screen point");
      return frozenPoint(
        (screen.x - mapBounds.left) / scale,
        worldHeight - (screen.y - mapBounds.top) / scale,
      );
    },
    /**
     * Scale a nonnegative world distance into CSS pixels.
     *
     * length must be a finite number. Return length times the fitted scale; zero stays
     * zero. Throw TypeError for nonnumeric/nonfinite input and RangeError for a negative
     * length. This conversion does not change direction or position.
     */
    worldLengthToScreen(length) {
      return nonNegativeFinite(length, "world length") * scale;
    },
  });
}

/**
 * Reserve a square around a rendered circular body and its optional selection ring.
 *
 * body supplies a finite center and positive radius in CSS pixels. options defaults
 * to an empty object: bodyPadding adds 3 pixels, and selectionAllowance adds another
 * 5 only when controlled or selected is exactly true. Both options must be finite
 * and nonnegative. Return a frozen rectangle with edges, width and height. Throw
 * TypeError for malformed body/coordinates or nonfinite values, and RangeError for
 * invalid radius or negative allowances. Inputs are unchanged.
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
 * Check whether two rectangles overlap by more than the geometry tolerance.
 *
 * first and second supply finite ordered left/top/right/bottom edges. Return true
 * when their shared area exceeds 1e-9 square pixels; touching edges return false.
 * Throw TypeError for malformed edges and RangeError for reversed edges. Any supplied
 * width/height fields are ignored and recomputed during validation.
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
 * Measure how far a rectangle extends beyond the four viewport edges.
 *
 * bounds and viewport supply finite ordered edges in CSS pixels. Return the sum
 * of positive left, right, top and bottom excess distances; zero means contained.
 * Throw TypeError for malformed edges and RangeError for reversed edges. This is
 * an edge-distance score, not the area outside the viewport.
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
 * Place one status dock per body, keeping labels off bodies and reserved rectangles.
 *
 * input supplies agents, a viewport and optional reservedRects (default empty).
 * Each agent needs a unique nonnegative integer globalSlot, finite center, positive
 * radius and statuses array. Status order is kept; payload contents are not read.
 * required, controlled and selected count only when exactly true. Empty status
 * lists have no dock but still protect their body.
 *
 * options defaults to DEFAULT_STATUS_DOCK_OPTIONS: bodyPadding 3,
 * selectionAllowance 5, cellWidth 28, cellHeight 20, cellGap 3, dockGap 7,
 * tangentStep 8 and maxTangentShift 24 pixels; both visible limits default to 9.
 * Cell dimensions and tangentStep must be positive; other distances nonnegative;
 * maxTangentShift is at most 24 and visible limits are integers from 0 through 9.
 *
 * Required/controlled/selected docks are searched together with a bounded search.
 * If that set cannot be placed within the search, suppress that whole set. Place
 * remaining docks greedily in priority order. Return frozen docks sorted by slot,
 * protectedBodies, attempted placementOrder and sorted suppressedGlobalSlots.
 * Hidden statuses stay in each placed dock with an overflow count. Inputs are
 * unchanged; status objects are not deep-copied. Invalid structure/numbers throw
 * TypeError; duplicate slots, invalid sizes/limits or reversed edges throw RangeError.
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
 * Place named required docks, then compact requests that cannot keep their full size.
 *
 * input supplies agent body geometry, a viewport, optional reservedRects (empty by
 * default) and requests. Each request needs a unique nonempty layoutKey, an existing
 * agent globalSlot and nonempty statuses. Multiple requests may belong to one body.
 * priority is a nonnegative finite number, default zero; lower values go first,
 * then controlled/selected bodies, larger status lists, slot and key break ties.
 * Agent statuses are ignored here; each request owns its ordered opaque payload.
 *
 * options uses the same validated defaults as layoutStatusDocks. A request's
 * dockOptions overrides shared settings. Its fallbackDockOptions overrides shared
 * settings plus compact defaults: 32 by 16 cells, cellGap 0, dockGap 3,
 * tangentStep 6 and maxTangentShift 24 pixels. Try a bounded joint search for at
 * most six requests, then greedy priority placement. Rejected requests get a
 * one-cell local or remote fallback; a marker retains every hidden payload.
 *
 * Return frozen docks sorted by slot/key, protectedBodies, priority placementOrder,
 * sorted compactedLayoutKeys and sorted suppressedLayoutKeys. Suppression means no
 * collision-free marker was found. Inputs are unchanged; payload objects are shared.
 * Throw TypeError for malformed records/numeric fields and RangeError for duplicate
 * or unknown IDs, missing/empty payload arrays, invalid dimensions/limits or reversed
 * edges. An Error also guards against an unexpectedly missing protected body.
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
 * Allocate transient cue boxes and routes against durable screen-space regions.
 *
 * input supplies viewport, requests and optional protectedRects (default empty).
 * Every request needs a unique nonempty layoutKey. enabled defaults to true; false
 * requests are filtered before geometry validation. Enabled requests need a supported
 * kind, nonnegative integer stableOrder and finite nonnegative priority (default 0).
 * Lower priority/stableOrder/key sorts first. Protected keys must also be unique
 * and cannot equal an enabled request key.
 *
 * recipient_cue and perimeter_callout need anchor plus positive width/height.
 * anchorRadius defaults to 0; recipientKey defaults to the anchor's coordinate key.
 * route needs source/target. Its radii and pathPadding/markerPadding default to 0;
 * endpoint gaps default to 3 pixels. markerProgress is optional, or a number in
 * [0,1]. compactMarkerPadding is optional, positive and smaller than markerPadding.
 * Points need finite x/y coordinates. Radii, endpoint gaps and padding must be
 * finite and nonnegative; cue width/height must be finite and positive. Padding
 * reserves paint around a line or marker; endpoint gaps separate lines from bodies.
 * allowProtectedKeys defaults to empty and removes only named route-path blockers.
 * Optional sourceProtectedKey/targetProtectedKey must occur in that allowed list;
 * they identify endpoint ownership for fallback routing. Markers still avoid all
 * protected regions. Cue placement does not use protected-region exceptions.
 *
 * options defaults to clearance 3, cueGap 8, stackGap 5, routeLaneSpacing 18 and
 * bridgeGap 10 pixels, plus routeLaneSearch 8. The first three distances are
 * nonnegative; lane spacing and bridge gap are positive; lane search is an integer
 * from 0 through 32. clearance expands protected/cue boxes; cueGap and stackGap
 * set local candidate spacing. Lane spacing/search set route alternatives and
 * bridgeGap sets crossing backplate size and separation. Place all cue boxes
 * before compacting them toward recipients.
 * Place routes behind those cues, allowing route crossings with bridge metadata.
 * Leader lines are an underlay and do not reserve space. Curve clearance uses 32
 * line segments; fallback route graphs have a fixed node limit.
 *
 * Return frozen ordered placements, cuePlacements, routePlacements, occupancyLedger,
 * protectedRegions, placementOrder and sorted filteredLayoutKeys. The ledger records
 * bounds, not a claim that every layer blocks every other layer. Inputs are unchanged.
 * Throw TypeError for malformed fields, RangeError for invalid ranges/keys or when an
 * enabled cue/route has no supported placement, and Error if an allocated record is
 * unexpectedly missing. This function does not infer disclosure or game legality.
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
 * Validate transient-layout options and fill their omitted defaults.
 *
 * options must be a record. Return frozen nonnegative clearance/cueGap/stackGap
 * (defaults 3/8/5), positive routeLaneSpacing/bridgeGap (18/10) and integer
 * routeLaneSearch from 0 through 32 (8). Distances are CSS pixels. Throw TypeError
 * for malformed/nonfinite values and RangeError for values outside these bounds.
 *
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
 * Copy and sort the named durable rectangles used by transient layout.
 *
 * regions must be an array of records with unique nonempty layoutKey values. Each row
 * supplies bounds, or supplies its rectangle edges directly when bounds is nullish.
 * Return a frozen key-sorted array with validated frozen rectangles. Throw TypeError
 * for bad records/edges/keys and RangeError for duplicate keys or reversed edges.
 *
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
 * Validate enabled requests and separate explicitly disabled layout keys.
 *
 * rawRequests is the request array already checked by the public caller. Every row
 * must have a unique nonempty key and an optional Boolean enabled flag. Disabled rows
 * need no geometry. Enabled cue and route fields receive the defaults and checks
 * documented by layoutCrossPhaseOccupancy. Allowed protected keys are copied, sorted
 * and required to be unique. Return frozen requests and sorted filteredLayoutKeys;
 * do not mutate source rows. Throw TypeError for malformed fields and RangeError
 * for duplicate keys, unsupported kinds or invalid numeric bounds/ownership lists.
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
 * Order validated requests by priority, stable order and finally layout key.
 *
 * first and second are normalized requests. Return a sort comparator number;
 * lower numeric values come first. The key tie-breaker uses localeCompare.
 *
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
 * Narrow a validated request to a cue when its kind is not route.
 *
 * request must already belong to the normalized request union. Return a Boolean;
 * this helper does not validate unknown objects or recognize additional kinds.
 *
 * @param {NormalizedCrossPhaseRequest} request
 * @returns {request is NormalizedCrossPhaseCueRequest}
 */
function isCrossPhaseCueRequest(request) {
  return request.kind !== "route";
}

/**
 * Narrow a validated request to a route by its exact kind.
 *
 * request must already belong to the normalized request union. Return true only
 * for kind route; no object or geometry validation occurs here.
 *
 * @param {NormalizedCrossPhaseRequest} request
 * @returns {request is NormalizedCrossPhaseRouteRequest}
 */
function isCrossPhaseRouteRequest(request) {
  return request.kind === "route";
}

/**
 * Return value when it is a string with at least one character.
 *
 * name identifies the field in a TypeError for other values. Whitespace is accepted
 * and is not trimmed; callers own any vocabulary or identity rules.
 *
 * @param {unknown} value
 * @param {string} name
 */
function nonEmptyString(value, name) {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`${name} must be a non-empty string.`);
  }
  return value;
}

/**
 * Use the point's exact x/y number strings as a deterministic grouping key.
 *
 * point is already normalized. Return x,y joined by a comma without rounding;
 * this is a geometry key and does not identify an actor.
 *
 * @param {Point} point
 */
function pointKey(point) {
  return `${point.x},${point.y}`;
}

/**
 * Move all four rectangle edges outward by amount pixels.
 *
 * bounds is an existing rectangle and amount is supplied by validated callers.
 * Return a new frozen rectangle; do not validate amount or mutate bounds.
 *
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
 * Package one proposed cue center and its rectangular bounds.
 *
 * center is a pixel point; width and height are validated positive pixel dimensions.
 * Return a frozen candidate with a copied frozen center and frozen bounds.
 *
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
 * List nearby cue boxes in deterministic distance-and-direction order.
 *
 * request supplies normalized anchor/radius/box size; viewport bounds the candidates
 * and options supplies cue/stack gaps. With compactBeforeDistance null (default),
 * try eight radial rings and eight directions. A distance enables additional closer
 * rings and angles for the later compaction pass. compactAngleOffsetDegrees defaults
 * to 4 and rotates only those extra angles. Return frozen, deduplicated candidates
 * that fit the viewport. This helper does not check obstacles or mutate inputs.
 *
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
 * Build viewport/blocker-edge cue positions for the global placement fallback.
 *
 * request supplies cue size and anchor, viewport bounds the search, and occupied
 * contains blocker rectangles. Return candidates ordered by distance to the anchor,
 * then distance to a viewport edge, top and left. Positions come from edge-derived
 * coordinates and may lie inside the viewport rather than on its boundary. Obstacle
 * overlap is checked later by firstFreeCueCandidate.
 *
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
 * Choose the first cue box that clears occupied boxes and other cue anchors.
 *
 * candidates are already ordered; occupied contains padded rectangles. anchor is
 * this cue's attachment point, cueAnchors carries all keyed anchors, and layoutKey
 * identifies the one anchor that may lie under its own box. Also reserve one pixel
 * around candidate paint before checking other anchors. Return a frozen candidate
 * and direct leader, or null. Leader lines themselves do not block placement.
 *
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
 * Describe a straight screen-space leader from the source anchor to the cue center.
 *
 * anchor and center are normalized points. Return a frozen line record with start,
 * copied end, points and SVG path. The anchor reference is reused. No clipping or
 * obstacle avoidance is applied to this low-priority underlay.
 *
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
 * Check a polyline's points against the viewport and its segments against blockers.
 *
 * points and blockers contain normalized pixel geometry; viewport is the allowed
 * rectangle. Return false for any out-of-bounds point or intersecting segment.
 * An empty or single-point line has no segments to reject. Inputs are unchanged.
 *
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
 * Find a short obstacle-avoiding polyline through a bounded corner graph.
 *
 * start/end are pixel points, blockers are forbidden rectangles, and viewport bounds
 * all graph nodes. Nodes use endpoints and corners one pixel outside blockers.
 * Return a frozen simplified path, or null for blocked/outside endpoints, too many
 * nodes, no connection or a failed final clearance check. Deterministic shortest-path
 * ties use coordinate signatures. The 384-node cap and candidate graph bound the
 * search; null does not prove that no continuous path exists.
 *
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
 * Assign initial lane offsets to routes sharing the same endpoint coordinates.
 *
 * requests contains normalized routes and spacing is the validated lane distance.
 * Return a Map from layoutKey to pixel offset. Single-direction groups are centered;
 * opposite directions use separated positive lanes relative to their own direction.
 * Sort each group by request priority. Grouping uses points, not actor identity.
 *
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
 * Try a route's preferred offset before symmetric neighboring lanes.
 *
 * preferred is the initial pixel offset, spacing is the positive lane distance,
 * and search is the validated number of neighboring lanes. Return a frozen
 * deduplicated array beginning with preferred, then preferred plus/minus each
 * successive multiple of spacing.
 *
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
 * Try the preferred marker fraction before fixed interior alternatives.
 *
 * preferred is the normalized progress fraction selected by the caller. Return a
 * frozen deduplicated list with preferred first, then 0.5, 0.4, 0.6, 0.3, 0.7,
 * 0.2 and 0.8. This helper does not validate range or inspect obstacles.
 *
 * @param {number} preferred
 */
function crossPhaseMarkerProgresses(preferred) {
  return Object.freeze([...new Set([preferred, 0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8])]);
}

/**
 * Route between endpoint ports through a bounded visibility graph.
 *
 * request supplies normalized route fields and optional endpoint owner keys.
 * protectedRegions supplies the named durable rectangles; blockers already reflects
 * path clearance and allowed-region exclusions. viewport bounds all nodes. A named
 * endpoint must lie within its named region; unnamed blocked endpoints can use
 * nearby clear ports. Search ports and one-pixel-outside blocker corners with
 * deterministic shortest-path ties. Return polyline route geometry, or null for bad
 * ownership, an oversized graph, no connection or a zero-length route. Ports move
 * visible line endpoints without changing the request's underlying disclosed facts.
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
  /**
   * Add a clear in-viewport graph point and merge its endpoint roles.
   *
   * candidate is a pixel point and role is source, target or waypoint. Mutate the
   * enclosing node map: duplicate coordinate keys retain one node and combine source/
   * target flags. Blocked or outside candidates are ignored. Return undefined.
   *
   * @param {Point} candidate @param {"source" | "target" | "waypoint"} role
   */
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
 * Try polyline detours with enough room for the route's direction marker.
 *
 * route supplies endpoints; pathBlockers restrict the line and markerBlockers restrict
 * the marker, including endpoint bodies. viewport, markerPadding and pathPadding are
 * validated pixel geometry. Candidate marker sites come from the inset viewport and
 * padded blocker corners. Return the first clear polyline with marker progress set
 * at its detour join, or null when the marker has no positive size, the viewport is
 * too small or no supported detour passes. The search does not mutate the route.
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
 * Score a possible marker waypoint by straight-line travel through it.
 *
 * route supplies pixel start/end points and candidate is a possible marker point.
 * Return start-to-candidate plus candidate-to-end distance; obstacles can make the
 * actual route longer. No geometry is changed.
 *
 * @param {ReturnType<typeof createPolylineRouteGeometry>} route
 * @param {Point} candidate
 */
function markerDetourLowerBound(route, candidate) {
  return (
    Math.hypot(candidate.x - route.start.x, candidate.y - route.start.y) +
    Math.hypot(route.end.x - candidate.x, route.end.y - candidate.y)
  );
}

/**
 * Sum the Euclidean lengths between consecutive pixel points.
 *
 * points is an ordered, already-normalized list. Return zero for fewer than two
 * points. No validation, closing segment or input mutation is added.
 *
 * @param {ReadonlyArray<Point>} points
 */
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

/**
 * Return the four corners one pixel outside a blocking rectangle.
 *
 * bounds is normalized. The frozen result is ordered top-left, top-right,
 * bottom-right, bottom-left. The margin protects serialized SVG paths from rounding
 * back onto the blocker; viewport and other-blocker checks happen later.
 *
 * @param {Rectangle} bounds
 */
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
 * Find visible ports near an endpoint that lies under a protected rectangle.
 *
 * endpoint is the disclosed anchor; blockers and viewport contain pixel rectangles.
 * If no blocker contains/touches the endpoint, return it directly. Otherwise try
 * one-pixel-outside edges/corners of containing blockers, keeping only points inside
 * the viewport and clear of every blocker. Return frozen deduplicated ports sorted
 * by x/y, possibly empty. This does not grant ownership or remove any blocker.
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
 * Build possible line attachment points on a named body's rectangle.
 *
 * bounds is the owner's normalized rectangle and aims lists destination/waypoint
 * points. Return frozen deduplicated edge centers, corners and intersections of
 * center-to-aim rays with the boundary. Zero-length aim rays are skipped. The caller
 * checks viewport containment and other blockers; this helper does not.
 *
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

/**
 * Check inclusive rectangle membership with the 1e-9 coordinate tolerance.
 *
 * point and bounds are normalized pixel geometry. Return a Boolean; values within
 * the tolerance beyond an edge count as touching that rectangle.
 *
 * @param {Point} point @param {Rectangle} bounds
 */
function pointInOrOnRectangle(point, bounds) {
  return (
    point.x >= bounds.left - EPSILON &&
    point.x <= bounds.right + EPSILON &&
    point.y >= bounds.top - EPSILON &&
    point.y <= bounds.bottom + EPSILON
  );
}

/**
 * Use inclusive rectangle membership for blocker contact checks.
 *
 * point and bounds are normalized pixel geometry. Return the same Boolean as
 * pointInOrOnRectangle, including its 1e-9 edge tolerance.
 *
 * @param {Point} point @param {Rectangle} bounds
 */
function pointTouchesRectangle(point, bounds) {
  return pointInOrOnRectangle(point, bounds);
}

/**
 * Format a pixel point for deterministic shortest-path tie breaking.
 *
 * point is normalized. Return its x and y values at 15 significant digits, joined
 * by a comma. This presentation-graph signature is not a public entity identity.
 *
 * @param {Point} point
 */
function graphPointSignature(point) {
  return `${numberKey(point.x)},${numberKey(point.y)}`;
}

/**
 * Format a validated number with 15 significant digits.
 *
 * value must be a number. Return its string for stable path signatures and SVG
 * leader text; this helper does not clamp, validate or round the stored geometry.
 *
 * @param {number} value
 */
function numberKey(value) {
  return value.toPrecision(15);
}

/**
 * Remove interior points that are collinear with their retained neighbors.
 *
 * points is an ordered path. Return a frozen array reusing the point objects; the
 * first and last points are kept. Collinearity uses an absolute 1e-9 cross-product
 * tolerance. The caller owns route validation; direction reversals are not checked.
 *
 * @param {ReadonlyArray<Point>} points
 */
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
 * Expose route vertices or sample a curved route for geometry checks.
 *
 * route is valid route geometry. Return its existing polyline points unchanged, or
 * a frozen array of 33 points across a quadratic curve or directed circular arc.
 * The resulting 32 segments approximate curved paint; they are not an analytic
 * intersection test. Inputs are unchanged.
 *
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
 * Test strict interior membership, leaving a small margin beside each edge.
 *
 * point and bounds are normalized. Return true only when every coordinate is more
 * than 1e-9 inside the matching edge. Boundary contact is handled separately.
 *
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
 * Find the crossing point of two finite nonparallel line segments.
 *
 * The four inputs are normalized pixel endpoints. Return a frozen intersection
 * point when both segment fractions lie in [0,1] within the 1e-9 tolerance.
 * Return null for parallel/collinear segments or a crossing outside either segment.
 * This helper does not report the overlap interval of collinear segments.
 *
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
 * Check a segment's interior endpoints and crossings with rectangle edges.
 *
 * start/end and bounds are normalized pixel geometry. Return true when either
 * endpoint is strictly inside or a nonparallel segment-edge intersection exists.
 * Use the shared tolerance; collinear edge overlap alone is not a separate test.
 *
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
 * Check a visible route corridor and its finite direction marker.
 *
 * route is valid geometry. paddedBlockers already includes path clearance; blockers
 * is the full marker-blocker list. viewport bounds paint; pathPadding and
 * markerPadding are pixel half-extents. markerProgress may be null to use the route
 * default. Return false for sampled line intersections, padded sampled bounds outside
 * the viewport or a marker touching a padded blocker. Nonpositive markerPadding
 * skips marker checks. Transparent interaction paths and bridge backplates are not
 * foreground collision claims. No geometry is modified.
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
 * Check sampled route segments against each forbidden rectangle.
 *
 * route is valid route geometry and blockers is the rectangle list. Return true on
 * the first segment intersection, otherwise false. Polylines use their own vertices;
 * curves/arcs use the 32-segment approximation from routePoints.
 *
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
 * Annotate later routes where they cross earlier painted routes.
 *
 * routes is already in paint order and bridgeGap is a positive pixel distance.
 * Return a frozen array of copied frozen routes with bridgeGaps naming the earlier
 * layout key, intersection point and gap. Use sampled segments, skip intersections
 * near the current route's endpoints, and merge crossings closer than bridgeGap.
 * No route path is moved; the painter decides how to draw each bridge.
 *
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
 * Check whether a point is closer than distance to either route endpoint.
 *
 * point, route and distance use pixels and are already normalized. Return a Boolean;
 * equality at the distance threshold is false.
 *
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

/**
 * Bound the route's vertices or sampled curve points with a rectangle.
 *
 * route is valid geometry. Return frozen min/max bounds of routePoints, without
 * stroke or marker padding. Curved bounds are sampled rather than analytic extrema.
 *
 * @param {ReturnType<typeof createRouteGeometry>} route
 */
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
 * Return whether value is a non-null object other than an array.
 *
 * This structural check does not require a plain object or validate its fields.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return a finite numeric field unchanged or raise TypeError.
 *
 * value is unknown and name appears in the error message. Numeric strings, NaN and
 * infinities are rejected. No coercion occurs.
 *
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
 * Require a finite number greater than zero for a named field.
 *
 * value is unknown and name labels errors. Return the number unchanged. Throw
 * TypeError for nonnumeric/nonfinite input and RangeError for zero or negatives.
 *
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
 * Require a finite number greater than or equal to zero for a named field.
 *
 * value is unknown and name labels errors. Return the number unchanged. Throw
 * TypeError for nonnumeric/nonfinite input and RangeError for negatives.
 *
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
 * Use fallback for an omitted option, otherwise validate a nonnegative number.
 *
 * Only undefined selects fallback, which the caller has already validated. name
 * labels TypeError/RangeError from nonNegativeFinite. Return the selected number;
 * null is an invalid supplied value rather than an omission.
 *
 * @param {unknown} value
 * @param {number} fallback
 * @param {string} name
 * @returns {number}
 */
function optionNonNegative(value, fallback, name) {
  return value === undefined ? fallback : nonNegativeFinite(value, name);
}

/**
 * Use fallback for an omitted option, otherwise validate a positive number.
 *
 * Only undefined selects fallback, which the caller has already validated. name
 * labels TypeError/RangeError from positiveFinite. Return the selected number;
 * null is an invalid supplied value rather than an omission.
 *
 * @param {unknown} value
 * @param {number} fallback
 * @param {string} name
 * @returns {number}
 */
function optionPositive(value, fallback, name) {
  return value === undefined ? fallback : positiveFinite(value, name);
}

/**
 * Normalize optional viewport padding into four nonnegative pixel values.
 *
 * value may be undefined (all zero), one number (all sides), or an inset object
 * whose omitted sides default to zero. Return a frozen top/right/bottom/left record.
 * Throw TypeError for malformed/nonfinite values and RangeError for negatives.
 *
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
 * Copy finite coordinates from a point record or coordinate array.
 *
 * value may be {x,y} or an array with at least two entries; extra entries are ignored.
 * name labels TypeError for invalid shape or nonfinite/nonnumeric coordinates.
 * Return a new frozen point without changing the input.
 *
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
 * Package x and y in a new frozen point record.
 *
 * Both numbers are already checked by the caller. Return {x,y} without conversion
 * or further validation.
 *
 * @param {number} x
 * @param {number} y
 * @returns {Point}
 */
function frozenPoint(x, y) {
  return Object.freeze({ x, y });
}

/**
 * Package four edges and their derived width/height in a frozen rectangle.
 *
 * left/top/right/bottom use the same coordinate units. Return the new record;
 * ordering and finiteness are caller preconditions, not checks in this helper.
 *
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
 * Validate rectangle edges and recompute its dimensions.
 *
 * value must be a record with finite left/top/right/bottom edges; name labels
 * errors. Equal edges are accepted. Return a new frozen rectangle, ignoring any
 * input width/height. Throw TypeError for malformed edges and RangeError if reversed.
 *
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
 * Compute the shared area of two already-normalized rectangles.
 *
 * first and second use the same coordinate units. Return zero for disjoint or
 * edge-touching rectangles, otherwise their overlap area. Inputs are unchanged.
 *
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
 * Fill and validate the numeric settings shared by status-dock searches.
 *
 * options is the caller-supplied options object. Omitted fields use
 * DEFAULT_STATUS_DOCK_OPTIONS. Return a frozen record: positive cell dimensions and
 * tangentStep; nonnegative padding/gaps; maxTangentShift at most 24 pixels; and
 * integer visible limits from 0 through 9. Throw TypeError for nonfinite/nonnumeric
 * distance fields and RangeError for invalid bounds or visible limits.
 *
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
 * Copy the geometry and ordered payload list for one dock owner.
 *
 * agent needs a nonnegative integer globalSlot, finite center, positive radius and
 * statuses array. Optional required/controlled/selected flags mean true only when
 * exactly true. Return a frozen row with a frozen shallow status-array copy; payload
 * objects stay shared. Throw TypeError for malformed fields and RangeError for an
 * invalid slot or nonpositive radius.
 *
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
 * Reject a second normalized agent with the same global slot.
 *
 * agents supplies normalized rows. Return undefined when all slots differ; throw
 * RangeError naming the duplicate otherwise. The list is not changed.
 *
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
 * Order agents by required, controlled, selected, status count and slot.
 *
 * first and second are normalized rows. True flags and larger payload lists come
 * first; lower globalSlot breaks the final tie. Return a sort comparator number.
 *
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
 * Search for the first complete dock arrangement in supplied priority order.
 *
 * inputs contains normalized body/viewport/options records. searchLimit defaults
 * to 100000 candidate visits. Try disclosure levels and anchor/tangent scores in
 * their existing order, backtracking around earlier choices. Return the first full
 * placement array (empty for no inputs), or null when the bounded search fails.
 * Null can mean exhausted budget rather than physical impossibility. No partial
 * result is returned and inputs are unchanged.
 *
 * @param {ReadonlyArray<StatusDockPlacementInput>} inputs
 * @param {number} [searchLimit]
 * @returns {ReadonlyArray<StatusDockPlacement> | null}
 */
function searchStatusDockPlacements(inputs, searchLimit = STATUS_DOCK_SEARCH_LIMIT) {
  let visited = 0;

  /**
   * Extend one partial dock arrangement using the next request's candidates.
   *
   * index selects the next input; priorDockBounds and placements hold accepted choices
   * on this branch. Mutate only the enclosing visited counter. Return the complete
   * placement array, or null for a failed branch/budget. Candidate bounds are passed
   * in new arrays so backtracking does not alter an earlier branch.
   *
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
 * Try a small joint dock search, then keep greedy choices in priority order.
 *
 * inputs is already priority-sorted. For at most six requests, first try the complete
 * set with a 5000-visit budget. If that fails, or there are more requests, choose each
 * request's first clear candidate around already accepted docks. Rejected requests
 * cannot move earlier accepted docks. Return placements and suppressedPriorityIndexes
 * in a frozen record. The greedy fallback does not prove joint feasibility of every
 * accepted/rejected combination.
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
 * Reduce an unplaced required dock to one associated marker when space permits.
 *
 * input supplies a normalized request and compact sizing options; priorDockBounds
 * reserves already accepted docks. Try ordinary local candidates first, then the
 * edge-derived remote search. Return a frozen placement or null. A single payload
 * stays visible; multiple payloads remain together behind an overflow marker. The
 * original payload list and prior placements are unchanged.
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
  const singleton = input.agent.statuses.length === 1;
  const visibleStatuses = Object.freeze(singleton ? [...input.agent.statuses] : []);
  const hiddenStatuses = Object.freeze(singleton ? [] : [...input.agent.statuses]);
  return Object.freeze({
    ...markerPlacement,
    required: true,
    expanded: singleton,
    visibleStatuses,
    hiddenStatuses,
    visibleCount: visibleStatuses.length,
    hiddenCount: hiddenStatuses.length,
    totalCount: input.agent.statuses.length,
    overflowLabel: singleton ? null : `+${hiddenStatuses.length}`,
  });
}

/**
 * Find a nearby one-cell marker using viewport and blocker edges.
 *
 * input supplies the body, viewport, reserved/body rectangles and compact cell size;
 * priorDockBounds adds occupied dock rectangles. Search combinations of edge-derived
 * x/y coordinates, choosing the clear candidate nearest the body center, then top
 * and left. Return a frozen placeholder placement containing one null payload, or
 * null when none fits. The caller replaces that placeholder with the original facts.
 * The search depends on rectangle count rather than the viewport's pixel count.
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
 * List possible rectangle starts touching viewport or blocker edges on one axis.
 *
 * viewportStart/end bounds the axis, extent is the candidate's validated size, and
 * blockers supplies start/end intervals. Return frozen sorted unique starts that
 * keep the extent within the viewport, with 1e-9 edge tolerance. Return empty if
 * the extent cannot fit. This checks one axis only, not rectangle collisions.
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
 * Return a new frozen midpoint of the rectangle's edges.
 *
 * bounds is already normalized. Coordinates use the same units as its edges.
 *
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
 * Name the cardinal direction from source to target in screen coordinates.
 *
 * source and target are normalized points. Return east/west when horizontal distance
 * is larger, otherwise south/north by vertical sign. Equal-axis ties use vertical;
 * coincident points return south. No geometry changes.
 *
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
 * Generate candidate docks from greatest allowed disclosure to most compact.
 *
 * input supplies normalized agent/body/viewport/options plus previously occupied
 * docks. Respect the required or ordinary visible limit and the nine-cell capacity;
 * hidden payloads reserve one overflow cell. A fully visible required dock does not
 * collapse here. Return candidates sorted by geometric score within each disclosure
 * level, including candidates marked collisionFree false. Callers choose or search
 * that list; no payload is dropped.
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
 * Attach ordered status payloads and a leader to one geometric dock candidate.
 *
 * input owns the original payload list and candidate supplies bounds, grid and score.
 * If only one payload would be hidden, show it in the already reserved overflow cell
 * instead of a redundant +1 marker. Return a frozen placement with frozen visible/
 * hidden array copies, counts, expanded/collisionFree flags and optional +N label.
 * Payload objects remain shared; collisionFree reports the candidate's earlier score.
 *
 * @param {PlaceStatusDockInput} input
 * @param {ReturnType<typeof buildCandidates>[number]} candidate
 * @returns {StatusDockPlacement}
 */
function placementFromCandidate(input, candidate) {
  // An overflow cell occupies the same space as one actual fact. Keep the last
  // fact visible rather than spending that cell on a redundant +1 marker.
  const visibleCount =
    input.agent.statuses.length - candidate.visibleCount === 1
      ? input.agent.statuses.length
      : candidate.visibleCount;
  const visibleStatuses = Object.freeze(input.agent.statuses.slice(0, visibleCount));
  const hiddenStatuses = Object.freeze(input.agent.statuses.slice(visibleCount));
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
 * Enumerate cardinal anchors and tangent shifts for one disclosure level.
 *
 * input includes visibleCount and hiddenCount alongside the normalized placement
 * data. A nonzero hidden count adds one overflow cell. Return frozen candidate
 * records with bounds, grid size, scores and collisionFree flags; the containing
 * array is a new mutable array. Throw RangeError if the resulting grid is empty
 * or exceeds three rows. This helper does not select a winner.
 *
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
 * Size a grid of one through nine dock cells, using at most three columns.
 *
 * cellCount must be a positive integer; options supplies validated cell width,
 * height and gap in pixels. Return frozen columns, rows, width and height. Throw
 * RangeError for an empty/noninteger count or a grid needing more than three rows.
 *
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
 * Try zero shift, then symmetric offsets up to the permitted maximum.
 *
 * step must be positive and maximum nonnegative; callers validate both pixel values.
 * Return a frozen array ordered zero, negative step, positive step, and so on.
 * Include the exact maximum in both directions even when step does not divide it.
 *
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
 * Position a dock beside a protected body at a cardinal anchor.
 *
 * body is its protected rectangle; dimensions gives dock width/height. anchor is
 * north/east/west/south, tangentShift moves along that side, and gap separates the
 * normal edges. All distances are pixels and already validated. Return a frozen
 * rectangle; do not clamp it to the viewport or avoid other bodies here.
 *
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
 * Measure a dock candidate's conflicts and preferred displacement.
 *
 * input provides bounds, viewport, bodyAndReservedRects, priorDockBounds,
 * tangentShift and anchorIndex. Return a frozen score with summed edge overflow
 * in pixels, overlap areas in square pixels, absolute displacement and anchor index.
 * Overlapping blocker regions can contribute area more than once; the score is a
 * placement ordering aid, not a union-area measurement.
 *
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
 * Order dock candidates by conflicts before cosmetic preferences.
 *
 * first and second are scored candidates. Return a sort comparator number comparing
 * viewport overflow, body/reserved overlap, previous-dock overlap, displacement,
 * anchor index and finally candidate index, in that order.
 *
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
 * Join a body's facing edge center to a dock's facing edge center.
 *
 * body and dock are pixel rectangles; anchor is north/east/west/south relative to
 * the body. Return frozen start/end points in a frozen line record. No clipping,
 * obstacle avoidance or DOM drawing occurs here.
 *
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

/**
 * Turn a recorded Red Zone record into at most two floor strips to tint.
 *
 * displayWidth is the scene map's own width in world units (the floor drawn on
 * screen). redZone is the recorded `map.red_zone` record, or null when the rule
 * is off, the map record predates it, or the Red Zone Floors filter is off. Its
 * team_a_x_range and team_b_x_range are inclusive [x_min, x_max] world bounds
 * that the normalizer has already checked (float32 values; a range may be
 * collapsed, x_min == x_max, or reach just past a raw displayed width).
 *
 * Each range is clipped to [0, displayWidth]. A clipped interval with no
 * positive width is dropped, so a collapsed range paints nothing. The rest are
 * merged where they overlap or touch, so an overlap is never painted twice and
 * never looks darker. Return a frozen array of 0, 1 or 2 frozen
 * {start, end} intervals in rising order. Pure display arithmetic: it never
 * changes a record, a score or a metric.
 *
 * @param {number} displayWidth
 * @param {Readonly<{team_a_x_range: readonly number[], team_b_x_range: readonly number[]}> | null} redZone
 * @returns {ReadonlyArray<Readonly<{start: number, end: number}>>}
 */
export function redZoneFloorIntervals(displayWidth, redZone) {
  if (redZone === null || !(displayWidth > 0)) {
    return Object.freeze([]);
  }
  const clipped = [redZone.team_a_x_range, redZone.team_b_x_range]
    .map(([low, high]) => [Math.max(0, low), Math.min(displayWidth, high)])
    .filter(([low, high]) => high - low > 0)
    .sort((first, second) => first[0] - second[0]);
  /** @type {Array<{start: number, end: number}>} */
  const merged = [];
  for (const [low, high] of clipped) {
    const last = merged.at(-1);
    if (last !== undefined && low <= last.end) {
      last.end = Math.max(last.end, high);
    } else {
      merged.push({ start: low, end: high });
    }
  }
  return Object.freeze(merged.map((interval) => Object.freeze(interval)));
}
