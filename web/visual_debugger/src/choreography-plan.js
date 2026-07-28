import { layoutRouteSet, routeMarkerPose } from "./routes.js";
import {
  activationImpactSemantic,
  classTokenFromId,
  resolveVisualToken,
} from "./vocabulary.js";

export const CHOREOGRAPHY_PHASES = Object.freeze({
  activationStart: 0,
  travelStart: 80,
  impactStart: 360,
  outcomeStart: 420,
  submissionRelease: 450,
  settleStart: 760,
  total: 900,
  reducedTotal: 220,
});

const MAX_HEALTH_CUES_PER_RECIPIENT = 3;
const MAX_EVENT_RECORDS = 128;
const OUTCOME_CUE_CLEARANCE = 2;
const OUTCOME_CUE_GRID_COLUMNS = 48;
const OUTCOME_CUE_GRID_ROWS = 36;
const OUTCOME_CUE_TARGET_STEP = 8;
const TRANSIENT_ICON_CLEARANCE = Object.freeze({ width: 24, height: 24 });
const CHARGE_OWNERSHIP_CUE_DIMENSIONS = Object.freeze({ width: 72, height: 22 });
const CHARGE_OWNERSHIP_ROUTE_PROGRESS = Object.freeze([0.18, 0.32, 0.5, 0.68, 0.82]);
const CHARGE_OWNERSHIP_NORMAL_OFFSETS = Object.freeze([0, 16, -16, 32, -32, 48, -48]);
const BASIC_TARGET_ENDPOINT_GAP = 12;
const CHARGE_TARGET_ENDPOINT_GAP = 18;
const TRAP_TARGET_ENDPOINT_GAP = 26;
const NET_CUE_DIMENSIONS = Object.freeze({ width: 88, height: 36 });
const UNCHANGED_NET_CUE_DIMENSIONS = Object.freeze({ width: 102, height: 36 });
const LIFECYCLE_CUE_DIMENSIONS = Object.freeze({ width: 52, height: 52 });
const STATUS_LIFECYCLE_KINDS = new Set([
  "applied",
  "refreshed",
  "decremented",
  "expired",
  "trap_broken",
  "cleared_unclassified",
  "trap_broken_and_reapplied",
]);
const REJECTION_COMPONENTS = new Set(["movement", "combat", "complete_tuple_domain"]);

/**
 * @typedef {{
 *   worldToScreen: (point: readonly [number, number] | {x: number, y: number}) =>
 *     {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 *   viewportBounds?: {
 *     left: number,
 *     top: number,
 *     right: number,
 *     bottom: number,
 *     width: number,
 *     height: number,
 *   },
 *   protectedRects?: ReadonlyArray<{
 *     left: number,
 *     top: number,
 *     right: number,
 *     bottom: number,
 *     width: number,
 *     height: number,
 *   }>,
 * }} ProjectionSurface
 * @typedef {{
 *   eventId: string,
 *   sourceGlobalSlot: number,
 *   targetGlobalSlot: number,
 *   tokenId?: string,
 *   source: {x: number, y: number},
 *   target: {x: number, y: number},
 *   sourceRadius: number,
 *   targetRadius: number,
 *   targetEndpointGap?: number,
 *   prioritizeTargetClearance?: boolean,
 * }} RouteInput
 * @typedef {Record<string, any> & {
 *   events: ReadonlyArray<Record<string, any>>,
 * }} ChoreographyPlan
 */

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * @param {unknown} value
 * @returns {Record<string, any> | null}
 */
function record(value) {
  return isRecord(value) ? value : null;
}

/**
 * @param {unknown} value
 * @returns {any[]}
 */
function array(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * @param {unknown} value
 * @returns {number | null}
 */
function integer(value) {
  return Number.isInteger(value) ? Number(value) : null;
}

/**
 * @param {unknown} value
 * @returns {boolean | null}
 */
function boolean(value) {
  return typeof value === "boolean" ? value : null;
}

/**
 * @param {unknown} value
 * @returns {number | null}
 */
function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * @param {unknown} value
 * @returns {string | null}
 */
function identifier(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/**
 * @param {unknown} value
 * @returns {readonly [number, number] | null}
 */
function point(value) {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    !Number.isFinite(value[0]) ||
    !Number.isFinite(value[1])
  ) {
    return null;
  }
  return Object.freeze([Number(value[0]), Number(value[1])]);
}

/**
 * @param {readonly [number, number] | null} world
 * @param {ProjectionSurface | null} surface
 * @returns {{x: number, y: number} | null}
 */
function project(world, surface) {
  if (!world || !surface) {
    return null;
  }
  const projected = surface.worldToScreen(world);
  if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y)) {
    return null;
  }
  return Object.freeze({ x: Number(projected.x), y: Number(projected.y) });
}

/**
 * Place one priority class of recipient-level cue in deterministic screen-space
 * lanes outside durable protected rectangles and already placed higher-priority
 * semantic cues. This is presentation geometry only; it never changes or
 * derives an event fact.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {ProjectionSurface | null} surface
 * @param {"net_health" | "status_lifecycle"} eventKind
 */
function layoutOutcomeCues(events, surface, eventKind) {
  const viewport = normalizedRectangle(surface?.viewportBounds);
  if (!viewport) {
    return events;
  }
  const occupied = array(surface?.protectedRects)
    .map(normalizedRectangle)
    .filter((bounds) => bounds !== null)
    .map((bounds) => expandCueBounds(bounds, OUTCOME_CUE_CLEARANCE));
  occupied.push(
    ...activationIconBounds(events).map((bounds) =>
      expandCueBounds(bounds, OUTCOME_CUE_CLEARANCE),
    ),
    ...chargeOwnershipBounds(events).map((bounds) =>
      expandCueBounds(bounds, OUTCOME_CUE_CLEARANCE),
    ),
    ...outcomeCueBounds(events).map((bounds) =>
      expandCueBounds(bounds, OUTCOME_CUE_CLEARANCE),
    ),
  );
  /** @type {Map<Record<string, any>, Record<string, any>>} */
  const placed = new Map();

  const outcomes = events
    .filter((event) => event.spatial && event.recipient && event.kind === eventKind)
    .sort(outcomePlacementOrder);
  for (const event of outcomes) {
    const dimensions =
      event.kind === "net_health"
        ? event.outcome === "unchanged"
          ? UNCHANGED_NET_CUE_DIMENSIONS
          : NET_CUE_DIMENSIONS
        : LIFECYCLE_CUE_DIMENSIONS;
    const selected =
      selectCueCandidate(
        localCueCandidates(
          event,
          dimensions,
          viewport,
          eventKind === "net_health" && outcomes.length === 1,
        ),
        occupied,
      ) ??
      selectCueCandidate(
        criticalEdgeCueCandidates(event, dimensions, viewport, occupied),
        occupied,
      ) ??
      selectCueCandidate(viewportCueCandidates(event, dimensions, viewport), occupied);
    if (!selected) {
      placed.set(
        event,
        Object.freeze({
          ...event,
          cue: null,
          cueBounds: null,
          cueCollisionFree: false,
          cueSuppressionReason: "no_collision_free_position",
          spatialDisposition: "suppressed_collision",
        }),
      );
      continue;
    }
    occupied.push(expandCueBounds(selected.bounds, OUTCOME_CUE_CLEARANCE));
    placed.set(
      event,
      Object.freeze({
        ...event,
        cue: selected.center,
        cueBounds: selected.bounds,
        cueCollisionFree: true,
        spatialDisposition: "rendered",
      }),
    );
  }
  return events.map((event) => placed.get(event) ?? event);
}

/**
 * Place each public Charge ownership pill outside durable geometry and every
 * other ownership pill. Route-adjacent positions preserve immediate visual
 * association; the bounded whole-map fallback is retained for unusually dense
 * layouts and remains associated through the route anchor stored with the cue.
 *
 * Authoritative NET outcomes are planned first and reserve their screen-space
 * before these redundant ownership labels. Lifecycle decoration is planned
 * afterward and treats both cue classes as protected semantic geometry.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {ProjectionSurface | null} surface
 */
function layoutChargeOwnershipCues(events, surface) {
  const viewport = normalizedRectangle(surface?.viewportBounds);
  if (!viewport) {
    return events;
  }
  const occupied = array(surface?.protectedRects)
    .map(normalizedRectangle)
    .filter((bounds) => bounds !== null)
    .map((bounds) => expandCueBounds(bounds, OUTCOME_CUE_CLEARANCE));
  occupied.push(
    ...activationIconBounds(events).map((bounds) =>
      expandCueBounds(bounds, OUTCOME_CUE_CLEARANCE),
    ),
    ...outcomeCueBounds(events).map((bounds) =>
      expandCueBounds(bounds, OUTCOME_CUE_CLEARANCE),
    ),
  );
  /** @type {Map<Record<string, any>, Record<string, any>>} */
  const placed = new Map();
  const charges = events
    .filter(
      (event) =>
        event.spatial &&
        event.kind === "activation" &&
        event.tokenId === "warrior_charge" &&
        event.route,
    )
    .sort(
      (left, right) =>
        Number(left.sourceSlot) - Number(right.sourceSlot) ||
        Number(left.targetSlot) - Number(right.targetSlot) ||
        String(left.eventId).localeCompare(String(right.eventId)),
    );
  for (const event of charges) {
    const selected =
      selectCueCandidate(
        chargeOwnershipRouteCandidates(event.route, viewport),
        occupied,
      ) ??
      selectCueCandidate(
        chargeOwnershipViewportCandidates(event.route, viewport),
        occupied,
      );
    if (!selected) {
      placed.set(
        event,
        Object.freeze({
          ...event,
          ownershipAnchor: null,
          ownershipBounds: null,
          ownershipCue: null,
          ownershipCueCollisionFree: false,
          ownershipCueSuppressionReason: "no_collision_free_position",
          ownershipSpatialDisposition: "suppressed_collision",
        }),
      );
      continue;
    }
    occupied.push(expandCueBounds(selected.bounds, OUTCOME_CUE_CLEARANCE));
    placed.set(
      event,
      Object.freeze({
        ...event,
        ownershipAnchor: selected.anchor,
        ownershipBounds: selected.bounds,
        ownershipCue: selected.center,
        ownershipCueCollisionFree: true,
        ownershipSpatialDisposition: "rendered",
      }),
    );
  }
  return events.map((event) => placed.get(event) ?? event);
}

/**
 * @param {Parameters<typeof routeMarkerPose>[0]} route
 * @param {Record<string, number>} viewport
 */
function chargeOwnershipRouteCandidates(route, viewport) {
  const candidates = [];
  const candidateKeys = new Set();
  for (const offset of CHARGE_OWNERSHIP_NORMAL_OFFSETS) {
    for (const progress of CHARGE_OWNERSHIP_ROUTE_PROGRESS) {
      const anchor = routeMarkerPose(route, progress);
      const radians = (anchor.degrees * Math.PI) / 180;
      const raw = {
        x: anchor.x - Math.sin(radians) * offset,
        y: anchor.y + Math.cos(radians) * offset,
      };
      const center = clampCueCenter(raw, CHARGE_OWNERSHIP_CUE_DIMENSIONS, viewport);
      if (!center) {
        continue;
      }
      const key = `${center.x.toFixed(6)}:${center.y.toFixed(6)}`;
      if (candidateKeys.has(key)) {
        continue;
      }
      candidateKeys.add(key);
      candidates.push(
        Object.freeze({
          anchor: Object.freeze({ x: anchor.x, y: anchor.y }),
          center,
          bounds: cueBounds(center, CHARGE_OWNERSHIP_CUE_DIMENSIONS),
          displacement: Math.hypot(center.x - anchor.x, center.y - anchor.y),
        }),
      );
    }
  }
  return candidates;
}

/**
 * @param {Parameters<typeof routeMarkerPose>[0]} route
 * @param {Record<string, number>} viewport
 */
function chargeOwnershipViewportCandidates(route, viewport) {
  const anchor = routeMarkerPose(route, 0.5);
  const routeAnchor = Object.freeze({ x: anchor.x, y: anchor.y });
  return viewportCueCenters(CHARGE_OWNERSHIP_CUE_DIMENSIONS, viewport).map((center) =>
    Object.freeze({
      anchor: routeAnchor,
      center,
      bounds: cueBounds(center, CHARGE_OWNERSHIP_CUE_DIMENSIONS),
      displacement: Math.hypot(center.x - routeAnchor.x, center.y - routeAnchor.y),
    }),
  );
}

/**
 * @param {ReadonlyArray<Record<string, any>>} events
 */
function chargeOwnershipBounds(events) {
  return events.flatMap((event) =>
    event.ownershipCueCollisionFree === true && event.ownershipBounds
      ? [event.ownershipBounds]
      : [],
  );
}

/**
 * @param {ReadonlyArray<Record<string, any>>} events
 */
function outcomeCueBounds(events) {
  return events.flatMap((event) =>
    event.cueCollisionFree === true && event.cueBounds ? [event.cueBounds] : [],
  );
}

/**
 * Exact ending classifications outrank generic applications within the
 * lifecycle tier. NET cues are handled in an earlier layout pass. Returned
 * event order remains untouched.
 *
 * @param {Record<string, any>} left
 * @param {Record<string, any>} right
 */
function outcomePlacementOrder(left, right) {
  const kindDifference =
    (left.kind === "net_health" ? 0 : 1) - (right.kind === "net_health" ? 0 : 1);
  if (kindDifference !== 0) {
    return kindDifference;
  }
  return (
    lifecyclePlacementPriority(left) - lifecyclePlacementPriority(right) ||
    Number(left.recipientSlot) - Number(right.recipientSlot) ||
    Number(left.lane ?? 0) - Number(right.lane ?? 0) ||
    String(left.eventId).localeCompare(String(right.eventId))
  );
}

/**
 * @param {Record<string, any>} event
 */
function lifecyclePlacementPriority(event) {
  if (event.lifecycle === "trap_broken_and_reapplied") {
    return 0;
  }
  if (event.lifecycle === "trap_broken") {
    return 1;
  }
  if (event.lifecycle === "expired" || event.lifecycle === "cleared_unclassified") {
    return 2;
  }
  return 3;
}

/**
 * Reserve the fixed icon at every spatial activation source/impact. Routes
 * remain free to cross the map; only semantic iconography blocks numeric and
 * lifecycle cue placement.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 */
function activationIconBounds(events) {
  return events.flatMap((event) => {
    if (!event.spatial || event.kind !== "activation") {
      return [];
    }
    const center =
      event.presentationKind === "source_local"
        ? event.source
        : (event.route?.end ?? event.target);
    return center ? [cueBounds(center, TRANSIENT_ICON_CLEARANCE)] : [];
  });
}

/**
 * @param {Record<string, any>} event
 * @param {{width: number, height: number}} dimensions
 * @param {Record<string, number>} viewport
 */
function localCueCandidates(event, dimensions, viewport, isolatedNet = false) {
  const recipient = event.recipient;
  const baseAngle =
    event.kind === "status_lifecycle"
      ? -Math.PI / 2 +
        (Math.PI * 2 * Number(event.lane ?? 0)) /
          Math.max(1, Number(event.laneCount ?? 1))
      : -Math.PI / 2;
  const angleOffsets =
    event.kind === "net_health"
      ? isolatedNet
        ? [0, -Math.PI / 4, Math.PI / 4, -Math.PI / 2, Math.PI / 2]
        : [0, -Math.PI / 4, Math.PI / 4, Math.PI, -Math.PI / 2, Math.PI / 2]
      : [
          0,
          Math.PI / 6,
          -Math.PI / 6,
          Math.PI / 3,
          -Math.PI / 3,
          Math.PI / 2,
          -Math.PI / 2,
          (Math.PI * 2) / 3,
          (-Math.PI * 2) / 3,
          Math.PI,
        ];
  const radii =
    event.kind === "net_health"
      ? isolatedNet
        ? [44, 64, 84, 108, 132, 156]
        : [44, 64, 84, 108]
      : [42, 60, 78, 96, 118, 140];
  const candidates = [];
  const candidateKeys = new Set();
  for (const radius of radii) {
    for (const offset of angleOffsets) {
      const raw = {
        x: recipient.x + Math.cos(baseAngle + offset) * radius,
        y: recipient.y + Math.sin(baseAngle + offset) * radius,
      };
      const center = clampCueCenter(raw, dimensions, viewport);
      if (!center) {
        continue;
      }
      const key = `${center.x}:${center.y}`;
      if (candidateKeys.has(key)) {
        continue;
      }
      candidateKeys.add(key);
      candidates.push(
        Object.freeze({
          center,
          bounds: cueBounds(center, dimensions),
          displacement: Math.hypot(center.x - recipient.x, center.y - recipient.y),
        }),
      );
    }
  }
  return candidates;
}

/**
 * @param {Record<string, any>} event
 * @param {{width: number, height: number}} dimensions
 * @param {Record<string, number>} viewport
 */
function viewportCueCandidates(event, dimensions, viewport) {
  return viewportCueCenters(dimensions, viewport).map((center) =>
    Object.freeze({
      center,
      bounds: cueBounds(center, dimensions),
      displacement: Math.hypot(
        center.x - event.recipient.x,
        center.y - event.recipient.y,
      ),
    }),
  );
}

/**
 * Test the finite set of centers at which a cue sits exactly beside an
 * occupied edge. A regular lattice can step over a narrow valid corridor;
 * these critical coordinates find it without an unbounded pixel search.
 *
 * @param {Record<string, any>} event
 * @param {{width: number, height: number}} dimensions
 * @param {Record<string, number>} viewport
 * @param {ReadonlyArray<Record<string, number>>} occupied
 */
function criticalEdgeCueCandidates(event, dimensions, viewport, occupied) {
  const limits = cueCenterLimits(dimensions, viewport);
  if (!limits) {
    return [];
  }
  const horizontal = dimensions.width / 2;
  const vertical = dimensions.height / 2;
  const horizontalPositions = boundedCriticalAxisPositions(
    [
      limits.minimumX,
      limits.maximumX,
      event.recipient.x,
      ...occupied.flatMap((bounds) => [
        bounds.left - horizontal,
        bounds.right + horizontal,
      ]),
    ],
    limits.minimumX,
    limits.maximumX,
    event.recipient.x,
    OUTCOME_CUE_GRID_COLUMNS,
  );
  const verticalPositions = boundedCriticalAxisPositions(
    [
      limits.minimumY,
      limits.maximumY,
      event.recipient.y,
      ...occupied.flatMap((bounds) => [
        bounds.top - vertical,
        bounds.bottom + vertical,
      ]),
    ],
    limits.minimumY,
    limits.maximumY,
    event.recipient.y,
    OUTCOME_CUE_GRID_ROWS,
  );
  return verticalPositions.flatMap((y) =>
    horizontalPositions.map((x) =>
      Object.freeze({
        center: Object.freeze({ x, y }),
        bounds: cueBounds({ x, y }, dimensions),
        displacement: Math.hypot(x - event.recipient.x, y - event.recipient.y),
      }),
    ),
  );
}

/**
 * Keep edge-driven search bounded while retaining both viewport rails and the
 * positions nearest the recipient. Coordinates are presentation pixels.
 *
 * @param {ReadonlyArray<number>} candidates
 * @param {number} minimum
 * @param {number} maximum
 * @param {number} focus
 * @param {number} maximumCount
 */
function boundedCriticalAxisPositions(
  candidates,
  minimum,
  maximum,
  focus,
  maximumCount,
) {
  const unique = new Map();
  for (const candidate of candidates) {
    if (
      !Number.isFinite(candidate) ||
      candidate < minimum - Number.EPSILON ||
      candidate > maximum + Number.EPSILON
    ) {
      continue;
    }
    const bounded = Math.max(minimum, Math.min(maximum, candidate));
    unique.set(bounded.toFixed(6), bounded);
  }
  const rails = [minimum, maximum].filter(
    (value, index, values) => values.indexOf(value) === index,
  );
  const railKeys = new Set(rails.map((value) => value.toFixed(6)));
  const nearest = [...unique.entries()]
    .filter(([key]) => !railKeys.has(key))
    .map(([, value]) => value)
    .sort(
      (left, right) => Math.abs(left - focus) - Math.abs(right - focus) || left - right,
    )
    .slice(0, Math.max(0, maximumCount - rails.length));
  return [...rails, ...nearest];
}

/**
 * @param {ReadonlyArray<Record<string, any>>} candidates
 * @param {ReadonlyArray<Record<string, number>>} occupied
 */
function selectCueCandidate(candidates, occupied) {
  let selected = null;
  let selectedDisplacement = Number.POSITIVE_INFINITY;
  for (const candidate of candidates) {
    if (
      candidate.displacement >= selectedDisplacement ||
      occupied.some((bounds) => rectangleIntersectionArea(candidate.bounds, bounds) > 0)
    ) {
      continue;
    }
    selected = candidate;
    selectedDisplacement = candidate.displacement;
  }
  return selected;
}

/**
 * @param {{x: number, y: number}} center
 * @param {{width: number, height: number}} dimensions
 */
function cueBounds(center, dimensions) {
  return Object.freeze({
    left: center.x - dimensions.width / 2,
    top: center.y - dimensions.height / 2,
    right: center.x + dimensions.width / 2,
    bottom: center.y + dimensions.height / 2,
    width: dimensions.width,
    height: dimensions.height,
  });
}

/**
 * @param {Record<string, number>} bounds
 * @param {number} clearance
 */
function expandCueBounds(bounds, clearance) {
  return Object.freeze({
    left: bounds.left - clearance,
    top: bounds.top - clearance,
    right: bounds.right + clearance,
    bottom: bounds.bottom + clearance,
    width: bounds.width + clearance * 2,
    height: bounds.height + clearance * 2,
  });
}

/**
 * Deterministic whole-map fallback for dense scenes whose collision-free
 * location lies beyond the recipient-local radial candidates.
 *
 * @param {{width: number, height: number}} dimensions
 * @param {Record<string, number>} viewport
 */
function viewportCueCenters(dimensions, viewport) {
  const limits = cueCenterLimits(dimensions, viewport);
  if (!limits) {
    return [];
  }
  const horizontalPositions = cueAxisPositions(
    limits.minimumX,
    limits.maximumX,
    OUTCOME_CUE_GRID_COLUMNS,
  );
  const verticalPositions = cueAxisPositions(
    limits.minimumY,
    limits.maximumY,
    OUTCOME_CUE_GRID_ROWS,
  );
  return verticalPositions.flatMap((y) =>
    horizontalPositions.map((x) => Object.freeze({ x, y })),
  );
}

/**
 * @param {number} minimum
 * @param {number} maximum
 * @param {number} maximumCount
 */
function cueAxisPositions(minimum, maximum, maximumCount) {
  const span = maximum - minimum;
  if (span <= Number.EPSILON) {
    return [minimum];
  }
  const count = Math.min(
    maximumCount,
    Math.max(2, Math.floor(span / OUTCOME_CUE_TARGET_STEP) + 1),
  );
  return Array.from(
    { length: count },
    (_, index) => minimum + (span * index) / (count - 1),
  );
}

/**
 * @param {{x: number, y: number}} center
 * @param {{width: number, height: number}} dimensions
 * @param {Record<string, number>} viewport
 */
function clampCueCenter(center, dimensions, viewport) {
  const limits = cueCenterLimits(dimensions, viewport);
  if (!limits) {
    return null;
  }
  return Object.freeze({
    x: Math.max(limits.minimumX, Math.min(limits.maximumX, center.x)),
    y: Math.max(limits.minimumY, Math.min(limits.maximumY, center.y)),
  });
}

/**
 * @param {{width: number, height: number}} dimensions
 * @param {Record<string, number>} viewport
 */
function cueCenterLimits(dimensions, viewport) {
  const horizontal = dimensions.width / 2 + 4;
  const vertical = dimensions.height / 2 + 4;
  const minimumX = viewport.left + horizontal;
  const maximumX = viewport.right - horizontal;
  const minimumY = viewport.top + vertical;
  const maximumY = viewport.bottom - vertical;
  if (maximumX < minimumX || maximumY < minimumY) {
    return null;
  }
  return Object.freeze({
    minimumX,
    maximumX,
    minimumY,
    maximumY,
  });
}

/**
 * @param {unknown} value
 * @returns {Record<string, number> | null}
 */
function normalizedRectangle(value) {
  const bounds = record(value);
  if (!bounds) {
    return null;
  }
  const left = finiteNumber(bounds.left);
  const top = finiteNumber(bounds.top);
  const right = finiteNumber(bounds.right);
  const bottom = finiteNumber(bounds.bottom);
  if (
    left === null ||
    top === null ||
    right === null ||
    bottom === null ||
    right < left ||
    bottom < top
  ) {
    return null;
  }
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
 * @param {Record<string, number>} first
 * @param {Record<string, number>} second
 */
function rectangleIntersectionArea(first, second) {
  return (
    Math.max(
      0,
      Math.min(first.right, second.right) - Math.max(first.left, second.left),
    ) *
    Math.max(0, Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top))
  );
}

/**
 * @param {unknown} frame
 * @returns {Record<string, any> | null}
 */
function frameScene(frame) {
  const candidate = record(frame);
  if (!candidate) {
    return null;
  }
  return record(candidate.scene) ?? record(candidate.battlefield_scene);
}

/**
 * @param {unknown} frame
 * @returns {Record<string, any> | null}
 */
export function frameEventBatch(frame) {
  const candidate = record(frame);
  if (!candidate) {
    return null;
  }
  return record(candidate.event_batch) ?? record(candidate.visual_event_batch);
}

/**
 * @param {unknown} frame
 * @returns {string | null}
 */
export function transitionEpochKey(frame) {
  const candidate = record(frame);
  const batch = frameEventBatch(frame);
  const sessionId = identifier(candidate?.session_id);
  const runGeneration = integer(candidate?.run_generation);
  const transitionId = integer(batch?.transition_id ?? candidate?.transition_id);
  if (sessionId === null || runGeneration === null || transitionId === null) {
    return null;
  }
  return JSON.stringify([sessionId, runGeneration, transitionId]);
}

/**
 * @param {unknown} frame
 * @returns {string | null}
 */
export function authorizationContextKey(frame) {
  const candidate = record(frame);
  const scene = frameScene(frame);
  const sessionId = identifier(candidate?.session_id);
  const runGeneration = integer(candidate?.run_generation);
  const audience =
    scene?.audience === "researcher" || scene?.audience === "agent_pov"
      ? scene.audience
      : null;
  if (audience === null) {
    return null;
  }
  const controlled =
    audience === "agent_pov"
      ? integer(record(scene?.selection)?.controlled_global_slot)
      : null;
  if (
    sessionId === null ||
    runGeneration === null ||
    (audience === "agent_pov" && controlled === null)
  ) {
    return null;
  }
  return JSON.stringify([sessionId, runGeneration, audience, controlled]);
}

/**
 * @param {unknown} frame
 * @returns {string | null}
 */
export function eventFingerprint(frame) {
  const batch = frameEventBatch(frame);
  if (!batch) {
    return null;
  }
  const events = array(batch.events);
  const eventIds = events.map((event) => identifier(record(event)?.event_id));
  if (eventIds.some((eventId) => eventId === null)) {
    return null;
  }
  return `${events.length}:${hashText(JSON.stringify(events))}`;
}

/**
 * Deterministic non-cryptographic content identity for an already-authorized
 * event batch. The hash is presentation state, not a trust boundary.
 *
 * @param {string} value
 */
function hashText(value) {
  let hash = 2166136261;
  for (const character of value) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * Classify only the existing browser command envelope. This does not decide
 * simulator legality or acceptance.
 *
 * @param {unknown} command
 */
export function isSubmissionCommand(command) {
  const candidate = record(command);
  if (candidate?.command_type !== "keyboard") {
    return false;
  }
  if (typeof candidate.key !== "string") {
    return false;
  }
  const normalized =
    candidate.key === " " ? "space" : candidate.key.trim().toLowerCase();
  return normalized === "space" || normalized === "enter" || normalized === "n";
}

/**
 * @param {Record<string, any> | null} scene
 * @param {ProjectionSurface | null} surface
 */
function radiusBySlot(scene, surface) {
  /** @type {Map<number, number>} */
  const radii = new Map();
  if (!surface) {
    return radii;
  }
  for (const rawAgent of array(scene?.agents)) {
    const agent = record(rawAgent);
    const slot = integer(agent?.global_slot);
    const radius = finiteNumber(agent?.radius);
    if (slot === null || radius === null || radius < 0) {
      continue;
    }
    radii.set(slot, surface.worldLengthToScreen(radius));
  }
  return radii;
}

/**
 * Build one immutable presentation plan from the already-authorized latest
 * scene/event batch. No simulator fact is derived here.
 *
 * @param {unknown} frame
 * @param {ProjectionSurface | null} surface
 * @returns {Readonly<ChoreographyPlan> | null}
 */
export function buildChoreographyPlan(frame, surface = null) {
  const candidate = record(frame);
  const scene = frameScene(frame);
  const batch = frameEventBatch(frame);
  const epochKey = transitionEpochKey(frame);
  const authorizationKey = authorizationContextKey(frame);
  const fingerprint = eventFingerprint(frame);
  if (
    !candidate ||
    !scene ||
    !batch ||
    epochKey === null ||
    authorizationKey === null ||
    fingerprint === null
  ) {
    return null;
  }

  const transitionId = integer(batch.transition_id);
  const simulatorStep = integer(batch.simulator_step);
  if (transitionId === null || simulatorStep === null) {
    return null;
  }

  const radii = radiusBySlot(scene, surface);
  const rawEvents = array(batch.events);
  if (rawEvents.length > MAX_EVENT_RECORDS) {
    throw new RangeError(
      `visual event batch exceeds the ${MAX_EVENT_RECORDS}-event presentation limit.`,
    );
  }
  /** @type {Array<Record<string, any>>} */
  const planned = [];
  /** @type {RouteInput[]} */
  const acceptedRouteInputs = [];
  /** @type {RouteInput[]} */
  const rejectedRouteInputs = [];
  /** @type {Map<number, number>} */
  const healthLaneCounts = new Map();
  /** @type {Map<number, number>} */
  const lifecycleLaneCounts = new Map();
  const lifecycleTotals = new Map();
  const seenEventIds = new Set();

  for (const rawEvent of rawEvents) {
    const lifecycleEvent = record(rawEvent);
    const recipientSlot = integer(lifecycleEvent?.recipient_global_slot);
    if (
      lifecycleEvent?.event_type === "status_lifecycle" &&
      recipientSlot !== null &&
      lifecycleEvent.change !== "decremented"
    ) {
      lifecycleTotals.set(recipientSlot, (lifecycleTotals.get(recipientSlot) ?? 0) + 1);
    }
  }

  for (const rawEvent of rawEvents) {
    const event = record(rawEvent);
    const eventId = identifier(event?.event_id);
    const eventType = identifier(event?.event_type);
    if (!event || eventId === null || eventType === null || seenEventIds.has(eventId)) {
      continue;
    }
    seenEventIds.add(eventId);
    const common = {
      eventId,
      eventType,
      transitionId,
    };
    if (integer(event.transition_id) !== transitionId) {
      planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
      continue;
    }

    if (eventType === "accepted_activation") {
      const sourceSlot = integer(event.source_global_slot);
      const targetSlot = integer(event.target_global_slot);
      const disclosure = identifier(event.target_disclosure);
      const lane = integer(event.lane);
      const sourceClassId = integer(event.source_class_id);
      if (
        sourceSlot === null ||
        (lane !== 0 && lane !== 1) ||
        (disclosure !== "public" &&
          disclosure !== "target_none" &&
          disclosure !== "redacted")
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const token = resolveVisualToken("activation", event.token_id, event);
      const sourceClass = classTokenFromId(sourceClassId);
      const source = project(point(event.source_anchor), surface);
      const target =
        disclosure === "public" ? project(point(event.target_anchor), surface) : null;
      if (
        (disclosure === "public" && (targetSlot === null || target === null)) ||
        (disclosure !== "public" &&
          (targetSlot !== null || point(event.target_anchor) !== null)) ||
        (token.tokenId === "mage_burst" && disclosure !== "target_none") ||
        (token.tokenId !== "mage_burst" && disclosure === "target_none")
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const hasPublicRoute =
        disclosure === "public" &&
        source !== null &&
        target !== null &&
        targetSlot !== null;
      const presentationKind = hasPublicRoute
        ? "routed"
        : source !== null
          ? "source_local"
          : target !== null
            ? "target_only_impact"
            : "undisclosed";
      if (hasPublicRoute) {
        acceptedRouteInputs.push({
          eventId,
          sourceGlobalSlot: sourceSlot,
          targetGlobalSlot: targetSlot,
          tokenId: token.tokenId,
          source,
          target,
          sourceRadius: radii.get(sourceSlot) ?? 0,
          targetRadius: radii.get(targetSlot) ?? 0,
          targetEndpointGap:
            token.tokenId === "hunter_trap"
              ? TRAP_TARGET_ENDPOINT_GAP
              : token.tokenId === "warrior_charge"
                ? CHARGE_TARGET_ENDPOINT_GAP
                : undefined,
        });
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "activation",
          tokenId: token.tokenId,
          token,
          impactSemantic: activationImpactSemantic(token.tokenId),
          lane,
          sourceSlot,
          sourceClassId,
          sourceClass,
          targetSlot: disclosure === "public" ? targetSlot : null,
          targetDisclosure: disclosure,
          source,
          target,
          route: null,
          spatial: source !== null || target !== null,
          presentationKind,
          phaseStart:
            presentationKind === "target_only_impact"
              ? CHOREOGRAPHY_PHASES.impactStart
              : CHOREOGRAPHY_PHASES.activationStart,
          phaseImpact: CHOREOGRAPHY_PHASES.impactStart,
          phaseEnd: CHOREOGRAPHY_PHASES.settleStart,
        }),
      );
      continue;
    }

    if (eventType === "net_health") {
      const recipientSlot = integer(event.recipient_global_slot);
      const delta = finiteNumber(event.net_delta);
      const healthBefore = finiteNumber(event.health_before);
      const healthAfter = finiteNumber(event.health_after);
      const outcome =
        event.outcome === "damage" ||
        event.outcome === "healing" ||
        event.outcome === "unchanged"
          ? event.outcome
          : null;
      if (
        recipientSlot === null ||
        delta === null ||
        healthBefore === null ||
        healthAfter === null ||
        outcome === null ||
        Math.abs(healthAfter - healthBefore - delta) > 1e-6 ||
        (delta < 0 && outcome !== "damage") ||
        (delta > 0 && outcome !== "healing") ||
        (delta === 0 && outcome !== "unchanged")
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const lane = healthLaneCounts.get(recipientSlot) ?? 0;
      healthLaneCounts.set(recipientSlot, lane + 1);
      const recipient = project(point(event.recipient_anchor), surface);
      planned.push(
        Object.freeze({
          ...common,
          kind: "net_health",
          recipientSlot,
          recipient,
          netDelta: delta,
          healthBefore,
          healthAfter,
          outcome,
          lane,
          spatial: recipient !== null && lane < MAX_HEALTH_CUES_PER_RECIPIENT,
          phaseStart: CHOREOGRAPHY_PHASES.outcomeStart,
          phaseEnd: CHOREOGRAPHY_PHASES.total,
        }),
      );
      continue;
    }

    if (eventType === "charge_displacement") {
      const sourceSlot = integer(event.source_global_slot);
      const targetSlot = integer(event.target_global_slot);
      const start = project(point(event.start), surface);
      const end = project(point(event.end), surface);
      const pathKind =
        event.path_kind === "charge_only" ||
        event.path_kind === "combined_charge_and_movement"
          ? event.path_kind
          : null;
      if (
        sourceSlot === null ||
        targetSlot === null ||
        start === null ||
        end === null ||
        pathKind === null
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "charge_displacement",
          sourceSlot,
          targetSlot,
          start,
          end,
          pathKind,
          spatial: true,
          persistent: true,
          phaseStart: CHOREOGRAPHY_PHASES.impactStart,
          phaseEnd: CHOREOGRAPHY_PHASES.total,
        }),
      );
      continue;
    }

    if (eventType === "status_lifecycle") {
      const recipientSlot = integer(event.recipient_global_slot);
      const durationBefore = integer(event.duration_before);
      const durationAfter = integer(event.duration_after);
      const lifecycleKind = identifier(event.change);
      if (
        recipientSlot === null ||
        durationBefore === null ||
        durationBefore < 0 ||
        durationAfter === null ||
        durationAfter < 0 ||
        lifecycleKind === null ||
        !STATUS_LIFECYCLE_KINDS.has(lifecycleKind)
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const status = resolveVisualToken("status", event.token_id, event);
      const lifecycle = resolveVisualToken("lifecycle", lifecycleKind, event);
      const spatialLifecycle = lifecycleKind !== "decremented";
      const lane = spatialLifecycle ? (lifecycleLaneCounts.get(recipientSlot) ?? 0) : 0;
      if (spatialLifecycle) {
        lifecycleLaneCounts.set(recipientSlot, lane + 1);
      }
      const recipient = project(point(event.recipient_anchor), surface);
      planned.push(
        Object.freeze({
          ...common,
          kind: "status_lifecycle",
          tokenId: status.tokenId,
          token: status,
          lifecycle: lifecycle.tokenId,
          lifecycleToken: lifecycle,
          recipientSlot,
          recipient,
          durationBefore,
          durationAfter,
          lane,
          laneCount: lifecycleTotals.get(recipientSlot) ?? 1,
          applicationEventIds: Object.freeze(
            array(event.application_event_ids)
              .map(identifier)
              .filter((value) => value !== null),
          ),
          spatial: lifecycleKind !== "decremented" && recipient !== null,
          phaseStart: CHOREOGRAPHY_PHASES.outcomeStart,
          phaseEnd: CHOREOGRAPHY_PHASES.total,
        }),
      );
      continue;
    }

    if (eventType === "rejected_action") {
      const actorSlot = integer(event.actor_global_slot);
      const targetSlot = integer(event.target_global_slot);
      const actor = project(point(event.actor_anchor), surface);
      const component = identifier(event.component);
      const disclosure = identifier(event.target_disclosure);
      const movementMaskValue = boolean(event.movement_mask_value);
      const pairMaskValue = boolean(event.pair_mask_value);
      if (
        actorSlot === null ||
        component === null ||
        !REJECTION_COMPONENTS.has(component) ||
        (disclosure !== "public" &&
          disclosure !== "target_none" &&
          disclosure !== "redacted" &&
          disclosure !== "invalid") ||
        movementMaskValue === null ||
        pairMaskValue === null
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const target =
        disclosure === "public" ? project(point(event.target_anchor), surface) : null;
      if (
        (disclosure === "public" && (targetSlot === null || target === null)) ||
        (disclosure !== "public" &&
          (targetSlot !== null || point(event.target_anchor) !== null))
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const hasPublicRoute =
        component === "combat" &&
        actorSlot !== null &&
        targetSlot !== null &&
        actor !== null &&
        target !== null;
      if (hasPublicRoute) {
        rejectedRouteInputs.push({
          eventId,
          sourceGlobalSlot: actorSlot,
          targetGlobalSlot: targetSlot,
          source: actor,
          target,
          sourceRadius: radii.get(actorSlot) ?? 0,
          targetRadius: radii.get(targetSlot) ?? 0,
        });
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "rejected_action",
          actorSlot,
          targetSlot: disclosure === "public" ? targetSlot : null,
          actor,
          target,
          component,
          lane: integer(event.lane),
          movementMaskValue,
          pairMaskValue,
          route: null,
          spatial: actor !== null,
          phaseStart: CHOREOGRAPHY_PHASES.activationStart,
          phaseEnd: CHOREOGRAPHY_PHASES.settleStart,
        }),
      );
      continue;
    }

    planned.push(
      Object.freeze({
        ...common,
        kind: "unknown",
        spatial: false,
      }),
    );
  }

  const routeLayoutOptions = surface?.viewportBounds
    ? { viewportBounds: surface.viewportBounds }
    : {};
  const acceptedTargetCounts = new Map();
  for (const input of acceptedRouteInputs) {
    acceptedTargetCounts.set(
      input.targetGlobalSlot,
      (acceptedTargetCounts.get(input.targetGlobalSlot) ?? 0) + 1,
    );
  }
  const acceptedLayoutInputs = acceptedRouteInputs.map((input) => ({
    ...input,
    targetEndpointGap:
      (input.tokenId === "basic_damage" || input.tokenId === "basic_heal") &&
      (acceptedTargetCounts.get(input.targetGlobalSlot) ?? 0) >= 4
        ? BASIC_TARGET_ENDPOINT_GAP
        : input.targetEndpointGap,
    prioritizeTargetClearance:
      (input.tokenId === "basic_damage" || input.tokenId === "basic_heal") &&
      (acceptedTargetCounts.get(input.targetGlobalSlot) ?? 0) >= 4,
  }));
  const acceptedRoutes = new Map(
    layoutRouteSet(acceptedLayoutInputs, routeLayoutOptions).map((route) => [
      route.eventId,
      route,
    ]),
  );
  const rejectedRoutes = new Map(
    layoutRouteSet(rejectedRouteInputs, {
      ...routeLayoutOptions,
      spacing: 12,
    }).map((route) => [route.eventId, route]),
  );
  /** @type {Record<string, any>[]} */
  const routedEvents = planned.map((event) => {
    const route =
      event.kind === "activation"
        ? acceptedRoutes.get(event.eventId)
        : event.kind === "rejected_action"
          ? rejectedRoutes.get(event.eventId)
          : undefined;
    return route
      ? Object.freeze({
          ...event,
          route,
          routeMultiplicity:
            event.kind === "activation" && event.targetSlot !== null
              ? (acceptedTargetCounts.get(event.targetSlot) ?? 1)
              : 1,
        })
      : event;
  });
  const spatialEventCount = routedEvents.filter((event) => event.spatial).length;
  /** @type {ReadonlyArray<Record<string, any>>} */
  const netOutcomeEvents = layoutOutcomeCues(routedEvents, surface, "net_health");
  /** @type {ReadonlyArray<Record<string, any>>} */
  const ownershipEvents = layoutChargeOwnershipCues(netOutcomeEvents, surface);
  const events = Object.freeze(
    layoutOutcomeCues(ownershipEvents, surface, "status_lifecycle").map((event) =>
      Object.isFrozen(event) ? event : Object.freeze(event),
    ),
  );
  const persistentEventCount = events.filter((event) => event.persistent).length;

  return Object.freeze({
    epochKey,
    authorizationKey,
    fingerprint,
    transitionId,
    simulatorStep,
    phases: CHOREOGRAPHY_PHASES,
    events,
    bounds: Object.freeze({
      nodes: Math.min(events.length * 28 + 2, 512),
      animations: Math.min(spatialEventCount * 3 + 2, 512),
      persistentNodes: Math.min(persistentEventCount * 6, 64),
    }),
  });
}
