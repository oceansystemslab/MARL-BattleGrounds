import { layoutRouteSet, routeMarkerPose } from "./routes.js";
import {
  activationImpactSemantic,
  classTokenFromId,
  resolveVisualToken,
  ultimateTokenFromClassId,
} from "./vocabulary.js";

export const CHOREOGRAPHY_PHASES = Object.freeze({
  activationStart: 0,
  travelStart: 80,
  impactStart: 360,
  outcomeStart: 420,
  submissionRelease: 450,
  settleStart: 760,
  v2RejectionStart: 0,
  v2AbilityStart: 0,
  v2HealthResolutionStart: 420,
  v2CountdownAndRegenStart: 900,
  v2CooldownStart: 1220,
  v2ChargeStart: 1540,
  v2MovementStart: 2080,
  v2DeathStart: 2080,
  v2StatusStart: 2560,
  v2ShieldStart: 3040,
  v2RespawnWaveStart: 3520,
  v2RespawnStart: 4000,
  // Recipient POV deltas expose only adjacent observations, not privileged
  // transition phases. Spatial POV cues therefore share one explicitly
  // non-causal successor-observation phase.
  povSuccessorObservationStart: 0,
  total: 900,
  reducedTotal: 220,
});

const STATUS_EVENT_TYPES = new Set([
  "status_aged_to_zero",
  "status_broken_by_damage",
  "status_applied",
  "status_refreshed_or_extended",
  "status_cleared_by_new_death",
]);
/**
 * @typedef {"ability" | "rejection" | "health" | "recovery" | "cooldown" | "charge" | "movement" | "death" | "status" | "shield" | "respawn_wave" | "respawn"} ChoreographyFamily
 * @typedef {Exclude<ChoreographyFamily, "ability" | "rejection">} OutcomeFamily
 */
/** @type {ReadonlyArray<OutcomeFamily>} */
const OUTCOME_FAMILY_ORDER = Object.freeze([
  "health",
  "recovery",
  "cooldown",
  "charge",
  "movement",
  "death",
  "status",
  "shield",
  "respawn_wave",
  "respawn",
]);
const READABLE_OUTCOME_DWELL_MS =
  CHOREOGRAPHY_PHASES.total - CHOREOGRAPHY_PHASES.outcomeStart;
/** @type {Readonly<Record<OutcomeFamily, number>>} */
const FAMILY_DWELL_MS = Object.freeze({
  health: READABLE_OUTCOME_DWELL_MS,
  recovery: 320,
  cooldown: 320,
  charge: CHOREOGRAPHY_PHASES.total - CHOREOGRAPHY_PHASES.impactStart,
  movement: 0,
  death: READABLE_OUTCOME_DWELL_MS,
  status: READABLE_OUTCOME_DWELL_MS,
  shield: READABLE_OUTCOME_DWELL_MS,
  respawn_wave: READABLE_OUTCOME_DWELL_MS,
  respawn: 620,
});

const MAX_HEALTH_CUES_PER_RECIPIENT = 3;
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
const NET_CUE_DIMENSIONS = Object.freeze({ width: 88, height: 36 });
const UNCHANGED_NET_CUE_DIMENSIONS = Object.freeze({ width: 102, height: 36 });
const LIFECYCLE_CUE_DIMENSIONS = Object.freeze({ width: 52, height: 52 });

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
 * V2 scientific identities are canonical strings.
 *
 * @param {unknown} value
 * @returns {string | null}
 */
function scientificIdentity(value) {
  return identifier(value);
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
 * @param {unknown} value
 * @returns {readonly [number, number] | null}
 */
function anchorPoint(value) {
  return point(record(value)?.position) ?? point(value);
}

/**
 * Place a non-spatial team clock cue in a stable presentation corner. The
 * position is UI layout only; team identity comes directly from the event.
 *
 * @param {number | null} teamId
 * @param {ProjectionSurface | null} surface
 */
function teamClockPoint(teamId, surface) {
  const bounds = surface?.viewportBounds;
  if (!bounds || (teamId !== 1 && teamId !== 2)) {
    return null;
  }
  return Object.freeze({
    x: teamId === 1 ? bounds.left + 28 : bounds.right - 28,
    y: bounds.top + 28,
  });
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
 * lanes outside durable protected rectangles and already placed cues from the
 * same authored phase. Non-coexisting transient phases do not consume one
 * another's screen-space capacity. This is presentation geometry only; it
 * never changes or derives an event fact.
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
 * Only same-phase activation iconography reserves transient screen space here.
 * Earlier/later NET and lifecycle phases cannot collide with these labels and
 * therefore do not reduce their bounded placement capacity.
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
  if (
    event.lifecycle === "trap_broken_and_reapplied" ||
    event.lifecycle === "expired_then_reapplied"
  ) {
    return 0;
  }
  if (event.lifecycle === "trap_broken") {
    return 1;
  }
  if (
    event.lifecycle === "expired" ||
    event.lifecycle === "cleared_by_death" ||
    event.lifecycle === "cleared_unclassified"
  ) {
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
  return record(candidate.scene);
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
  return record(candidate.event_batch);
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
  const transitionId = scientificIdentity(
    candidate?.incoming_transition_id ??
      candidate?.incoming_pov_transition_id ??
      batch?.transition_id ??
      candidate?.transition_id,
  );
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
  const replayActor =
    candidate?.viewer_mode === "replay"
      ? candidate.replay_audience === "actor_pov"
        ? integer(candidate.pov_global_slot)
        : candidate.replay_audience === "shared_obs_source_material"
          ? integer(candidate.selected_global_slot)
          : null
      : null;
  const controlled =
    audience === "agent_pov"
      ? (replayActor ?? integer(record(scene?.selection)?.controlled_global_slot))
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
 * Compose simultaneous atomic status events without inventing status state.
 * Recipient slot, catalog channel, and status ID are serialized authority; the
 * browser only chooses one readable cue for each exact group. Every source
 * event keeps its own plan row and the primary cue carries all atomic IDs.
 *
 * @param {ReadonlyArray<unknown>} rawEvents
 * @param {string} transitionId
 */
function statusPresentationAssignments(rawEvents, transitionId) {
  /** @type {Map<string, {events: Record<string, any>[], firstIndex: number}>} */
  const groups = new Map();
  const seenEventIds = new Set();
  rawEvents.forEach((rawEvent, index) => {
    const event = record(rawEvent);
    const eventType = identifier(event?.event_type);
    const eventId = identifier(event?.event_id);
    const recipientSlot = integer(event?.recipient_global_slot);
    const statusChannel = integer(event?.status_channel);
    const statusId = identifier(event?.status_id);
    if (
      !event ||
      eventType === null ||
      !STATUS_EVENT_TYPES.has(eventType) ||
      eventId === null ||
      seenEventIds.has(eventId) ||
      scientificIdentity(event.transition_id) !== transitionId ||
      recipientSlot === null ||
      statusChannel === null ||
      statusId === null
    ) {
      return;
    }
    seenEventIds.add(eventId);
    const key = `${recipientSlot}:${statusChannel}:${statusId}`;
    const group = groups.get(key) ?? { events: [], firstIndex: index };
    group.events.push(event);
    groups.set(key, group);
  });

  /** @type {Map<string, {lane: number, laneCount: number}>} */
  const lanes = new Map();
  /** @type {Map<number, Array<[string, {events: Record<string, any>[], firstIndex: number}]>>} */
  const groupsByRecipient = new Map();
  for (const [key, group] of groups) {
    const recipientSlot = integer(group.events[0]?.recipient_global_slot);
    if (recipientSlot === null) continue;
    const recipientGroups = groupsByRecipient.get(recipientSlot) ?? [];
    recipientGroups.push([key, group]);
    groupsByRecipient.set(recipientSlot, recipientGroups);
  }
  for (const recipientGroups of groupsByRecipient.values()) {
    recipientGroups.sort((left, right) => left[1].firstIndex - right[1].firstIndex);
    recipientGroups.forEach(([key], lane) => {
      lanes.set(key, { lane, laneCount: recipientGroups.length });
    });
  }

  /** @type {Map<string, Record<string, any>>} */
  const assignments = new Map();
  for (const [key, group] of groups) {
    const appliedEvents = group.events.filter(
      (event) => event.event_type === "status_applied",
    );
    const applied = appliedEvents.at(-1) ?? null;
    const byType = new Map(
      group.events.map((event) => [identifier(event.event_type), event]),
    );
    const refreshed = byType.get("status_refreshed_or_extended") ?? null;
    const cleared = byType.get("status_cleared_by_new_death") ?? null;
    const broken = byType.get("status_broken_by_damage") ?? null;
    const expired = byType.get("status_aged_to_zero") ?? null;
    const lifecycle = cleared
      ? "cleared_by_death"
      : refreshed
        ? "refreshed"
        : broken && applied
          ? "trap_broken_and_reapplied"
          : expired && applied
            ? "expired_then_reapplied"
            : broken
              ? "trap_broken"
              : expired
                ? "expired"
                : "applied";
    const primary =
      cleared ??
      refreshed ??
      (applied && (broken || expired) ? applied : null) ??
      broken ??
      expired ??
      applied;
    if (!primary) continue;
    const atomicEventIds = Object.freeze(
      group.events.map((event) => identifier(event.event_id)).filter(Boolean),
    );
    const applicationEventIds = Object.freeze(
      appliedEvents.map((event) => identifier(event.event_id)).filter(Boolean),
    );
    const sourceEvents = Object.freeze([...appliedEvents]);
    const lane = lanes.get(key) ?? { lane: 0, laneCount: 1 };
    for (const event of group.events) {
      const eventId = identifier(event.event_id);
      if (eventId === null) continue;
      assignments.set(
        eventId,
        Object.freeze({
          primary: event === primary,
          lifecycle,
          atomicEventIds,
          applicationEventIds,
          sourceEvents,
          lane: lane.lane,
          laneCount: lane.laneCount,
        }),
      );
    }
  }
  return assignments;
}

/** @param {Record<string, any>} event @returns {ChoreographyFamily | null} */
function choreographyFamily(event) {
  if (event.kind === "activation") return "ability";
  if (event.kind === "rejected_action") return "rejection";
  if (event.kind === "net_health") return "health";
  if (event.kind === "charge_displacement") return "charge";
  if (event.kind === "movement_displacement") return "movement";
  if (event.kind === "status_lifecycle") return "status";
  if (event.kind !== "semantic_pulse") return null;
  if (
    event.cueSemantic === "combat_countdown_reset" ||
    event.cueSemantic === "health_regenerated"
  ) {
    return "recovery";
  }
  if (
    event.cueSemantic === "cooldown_started" ||
    event.cueSemantic === "cooldown_ready"
  ) {
    return "cooldown";
  }
  if (event.cueSemantic === "agent_died") return "death";
  if (event.cueSemantic === "spawn_shield_expired") return "shield";
  if (event.cueSemantic === "respawn_wave_occurred") return "respawn_wave";
  if (event.cueSemantic === "agent_respawned") return "respawn";
  return null;
}

/**
 * Allocate only families present in this authorized transition. M5 ability and
 * outcome anchors remain exact, while every readable outcome gets its own
 * complete dwell and later M6 lifecycle beats extend the clock in canonical
 * order. Ordinary movement deliberately owns a zero-duration presentation
 * slot because its scientific event remains feed-only spatially.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 */
function scheduleChoreography(events) {
  const families = new Set(
    events
      .filter((event) => event.spatial || event.kind === "movement_displacement")
      .map(choreographyFamily)
      .filter((family) => family !== null),
  );
  /** @type {Map<string, {start: number, end: number}>} */
  const windows = new Map();
  if (families.has("rejection")) {
    windows.set("rejection", {
      start: CHOREOGRAPHY_PHASES.activationStart,
      end: CHOREOGRAPHY_PHASES.settleStart,
    });
  }
  if (families.has("ability")) {
    windows.set("ability", {
      start: CHOREOGRAPHY_PHASES.activationStart,
      end: CHOREOGRAPHY_PHASES.settleStart,
    });
  }
  const activeOutcomes = OUTCOME_FAMILY_ORDER.filter((family) => families.has(family));
  let cursor =
    activeOutcomes.length > 0 && (families.has("ability") || families.has("rejection"))
      ? CHOREOGRAPHY_PHASES.outcomeStart
      : 0;
  for (const family of activeOutcomes) {
    const duration = FAMILY_DWELL_MS[family];
    windows.set(family, { start: cursor, end: cursor + duration });
    cursor += duration;
  }
  const total = Math.max(
    cursor,
    windows.get("ability")?.end ?? 0,
    windows.get("rejection")?.end ?? 0,
  );
  /** @param {ChoreographyFamily} family */
  const startFor = (family) => windows.get(family)?.start ?? total;
  const phases = Object.freeze({
    ...CHOREOGRAPHY_PHASES,
    outcomeStart: activeOutcomes.length > 0 ? startFor(activeOutcomes[0]) : total,
    v2RejectionStart: startFor("rejection"),
    v2AbilityStart: startFor("ability"),
    v2HealthResolutionStart: startFor("health"),
    v2CountdownAndRegenStart: startFor("recovery"),
    v2CooldownStart: startFor("cooldown"),
    v2ChargeStart: startFor("charge"),
    v2MovementStart: startFor("movement"),
    v2DeathStart: startFor("death"),
    v2StatusStart: startFor("status"),
    v2ShieldStart: startFor("shield"),
    v2RespawnWaveStart: startFor("respawn_wave"),
    v2RespawnStart: startFor("respawn"),
    povSuccessorObservationStart: startFor("health"),
    total,
    reducedTotal: Math.min(CHOREOGRAPHY_PHASES.reducedTotal, total),
  });
  /** @type {Record<string, any>[]} */
  const scheduledEvents = events.map((event) => {
    const family = choreographyFamily(event);
    const window = family === null ? null : windows.get(family);
    if (!window) return event;
    return /** @type {Record<string, any>} */ (
      Object.freeze({
        ...event,
        phaseStart: window.start,
        phaseImpact:
          family === "ability"
            ? Math.min(window.start + CHOREOGRAPHY_PHASES.impactStart, window.end)
            : event.phaseImpact,
        phaseEnd: window.end,
      })
    );
  });
  return Object.freeze({ phases, events: Object.freeze(scheduledEvents) });
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

  const transitionId = scientificIdentity(batch.transition_id);
  const simulatorStep = integer(batch.simulator_step);
  if (transitionId === null || simulatorStep === null) {
    return null;
  }

  const radii = radiusBySlot(scene, surface);
  const rawEvents = array(batch.events);
  const statusAssignments = statusPresentationAssignments(rawEvents, transitionId);
  /** @type {Array<Record<string, any>>} */
  const planned = [];
  /** @type {RouteInput[]} */
  const acceptedRouteInputs = [];
  /** @type {Map<number, number>} */
  const healthLaneCounts = new Map();
  const seenEventIds = new Set();
  const sceneAgentBySlot = new Map(
    array(scene.agents)
      .map(record)
      .filter((agent) => agent && integer(agent.global_slot) !== null)
      .map((agent) => {
        const row = /** @type {Record<string, any>} */ (agent);
        return [integer(row.global_slot), row];
      }),
  );
  const publicAgentIds = array(batch.public_agent_id_by_global_slot);
  /**
   * Event roles and their phase anchors are one semantic identity join. A slot
   * may remain internal, but it must never select a different public identity
   * than the exact authorized anchor carried by the same batch.
   *
   * @param {Record<string, any>} event
   * @param {string} slotField
   * @param {readonly string[]} anchorFields
   * @param {string | null} publicIdField
   */
  const eventRoleIdentityMatches = (
    event,
    slotField,
    anchorFields,
    publicIdField = null,
  ) => {
    const slot = integer(event[slotField]);
    const expectedPublicAgentId =
      slot === null ? null : identifier(publicAgentIds[slot]);
    if (
      publicIdField !== null &&
      Object.hasOwn(event, publicIdField) &&
      (slot === null || identifier(event[publicIdField]) !== expectedPublicAgentId)
    ) {
      return false;
    }
    for (const anchorField of anchorFields) {
      if (!Object.hasOwn(event, anchorField) || event[anchorField] === null) {
        continue;
      }
      const anchor = record(event[anchorField]);
      if (
        slot === null ||
        !anchor ||
        integer(anchor.global_slot) !== slot ||
        identifier(anchor.public_agent_id) !== expectedPublicAgentId
      ) {
        return false;
      }
    }
    return true;
  };
  /** @param {Record<string, any>} event */
  const researcherEventIdentityMatches = (event) =>
    eventRoleIdentityMatches(
      event,
      "actor_global_slot",
      ["actor_anchor"],
      "actor_public_agent_id",
    ) &&
    eventRoleIdentityMatches(event, "source_global_slot", ["source_anchor"]) &&
    eventRoleIdentityMatches(event, "recipient_global_slot", ["recipient_anchor"]) &&
    eventRoleIdentityMatches(event, "agent_global_slot", [
      "agent_anchor",
      "start_anchor",
      "end_anchor",
    ]);
  if (
    scene.audience === "researcher" &&
    (publicAgentIds.length === 0 ||
      [...sceneAgentBySlot].some(
        ([slot, agent]) =>
          slot === null ||
          identifier(publicAgentIds[slot]) === null ||
          identifier(publicAgentIds[slot]) !== identifier(agent.public_agent_id),
      ) ||
      rawEvents.some(
        (rawEvent) =>
          !record(rawEvent) ||
          !researcherEventIdentityMatches(
            /** @type {Record<string, any>} */ (record(rawEvent)),
          ),
      ) ||
      array(batch.agent_phase_trajectories).some((rawTrajectory) => {
        const trajectory = record(rawTrajectory);
        if (!trajectory) {
          return true;
        }
        const slot = integer(trajectory.global_slot);
        const expectedPublicAgentId =
          slot === null ? null : identifier(publicAgentIds[slot]);
        return ["transition_start", "post_charge", "successor"].some((phase) => {
          const anchor = record(trajectory[phase]);
          return (
            anchor !== null &&
            (slot === null ||
              integer(anchor.global_slot) !== slot ||
              identifier(anchor.public_agent_id) !== expectedPublicAgentId)
          );
        });
      }))
  ) {
    return null;
  }
  /**
   * Resolve display identity only from the same authorized scene/event roots.
   * Actor POV event batches intentionally omit the researcher roster, so their
   * self identity falls back to the exact recipient-sliced scene actor.
   *
   * @param {number | null} slot
   * @returns {string | null}
   */
  const publicAgentIdForSlot = (slot) => {
    const sceneAgent = slot === null ? null : sceneAgentBySlot.get(slot);
    if (slot === null || !sceneAgent) {
      return null;
    }
    return scene.audience === "researcher"
      ? identifier(publicAgentIds[slot])
      : identifier(sceneAgent.public_agent_id);
  };

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
    if (scientificIdentity(event.transition_id) !== transitionId) {
      planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
      continue;
    }

    if (eventType === "ability_activated") {
      const sourceSlot = integer(event.source_global_slot);
      const recipientSlot = integer(event.recipient_global_slot);
      const component = identifier(event.ability_component);
      const sourceClassId = integer(sceneAgentBySlot.get(sourceSlot)?.class_id);
      const source = project(anchorPoint(event.source_anchor), surface);
      const recipient = project(anchorPoint(event.recipient_anchor), surface);
      if (
        sourceSlot === null ||
        (component !== "basic" && component !== "ultimate") ||
        source === null ||
        (recipientSlot === null) !== (recipient === null)
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const token =
        component === "ultimate"
          ? ultimateTokenFromClassId(sourceClassId, event)
          : resolveVisualToken(
              "activation",
              sourceClassId === 5 ? "basic_heal" : "basic_damage",
              event,
            );
      const sourceClass = classTokenFromId(sourceClassId);
      const routed = recipientSlot !== null && recipient !== null;
      if (routed) {
        acceptedRouteInputs.push({
          eventId,
          sourceGlobalSlot: sourceSlot,
          targetGlobalSlot: recipientSlot,
          tokenId: token.tokenId,
          source,
          target: recipient,
          sourceRadius: radii.get(sourceSlot) ?? 0,
          targetRadius: radii.get(recipientSlot) ?? 0,
          targetEndpointGap:
            token.tokenId === "warrior_charge" ? CHARGE_TARGET_ENDPOINT_GAP : undefined,
        });
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "activation",
          tokenId: token.tokenId,
          token,
          impactSemantic: activationImpactSemantic(token.tokenId),
          lane: component === "basic" ? 0 : 1,
          sourceSlot,
          sourcePublicAgentId: publicAgentIdForSlot(sourceSlot),
          sourceClassId,
          sourceClass,
          targetSlot: recipientSlot,
          targetPublicAgentId: publicAgentIdForSlot(recipientSlot),
          targetDisclosure: recipientSlot === null ? "target_none" : "public",
          source,
          target: recipient,
          route: null,
          spatial: true,
          presentationKind: routed ? "routed" : "source_local",
          phaseStart: CHOREOGRAPHY_PHASES.v2AbilityStart,
          phaseImpact: CHOREOGRAPHY_PHASES.v2HealthResolutionStart - 20,
          phaseEnd: CHOREOGRAPHY_PHASES.v2HealthResolutionStart,
        }),
      );
      continue;
    }

    if (eventType === "recipient_health_resolution") {
      const recipientSlot = integer(event.recipient_global_slot);
      const delta = finiteNumber(event.realized_net_health_change);
      const healthBefore = finiteNumber(event.transition_start_health);
      const healthAfter = finiteNumber(event.health_after_combat_resolution);
      if (
        recipientSlot === null ||
        delta === null ||
        healthBefore === null ||
        healthAfter === null ||
        Math.abs(healthAfter - healthBefore - delta) > 1e-5
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const lane = healthLaneCounts.get(recipientSlot) ?? 0;
      healthLaneCounts.set(recipientSlot, lane + 1);
      const recipient = project(anchorPoint(event.recipient_anchor), surface);
      planned.push(
        Object.freeze({
          ...common,
          kind: "net_health",
          recipientSlot,
          recipientPublicAgentId: publicAgentIdForSlot(recipientSlot),
          recipient,
          netDelta: delta,
          healthBefore,
          healthAfter,
          outcome: delta < 0 ? "damage" : delta > 0 ? "healing" : "unchanged",
          lane,
          spatial: recipient !== null && lane < MAX_HEALTH_CUES_PER_RECIPIENT,
          phaseStart: CHOREOGRAPHY_PHASES.v2HealthResolutionStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2CountdownAndRegenStart,
        }),
      );
      continue;
    }

    if (
      eventType === "charge_phase_displacement" ||
      eventType === "ordinary_movement_phase_displacement"
    ) {
      const agentSlot = integer(event.agent_global_slot);
      const start = project(anchorPoint(event.start_anchor), surface);
      const end = project(anchorPoint(event.end_anchor), surface);
      if (agentSlot === null || start === null || end === null) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind:
            eventType === "charge_phase_displacement"
              ? "charge_displacement"
              : "movement_displacement",
          sourceSlot: agentSlot,
          sourcePublicAgentId: publicAgentIdForSlot(agentSlot),
          targetSlot: null,
          start,
          end,
          pathKind:
            eventType === "charge_phase_displacement"
              ? "charge_phase"
              : "ordinary_movement_phase",
          spatial: eventType === "charge_phase_displacement",
          presentationSuppressed: eventType === "ordinary_movement_phase_displacement",
          persistent: eventType === "charge_phase_displacement",
          phaseStart:
            eventType === "charge_phase_displacement"
              ? CHOREOGRAPHY_PHASES.v2ChargeStart
              : CHOREOGRAPHY_PHASES.v2MovementStart,
          phaseEnd:
            eventType === "charge_phase_displacement"
              ? CHOREOGRAPHY_PHASES.v2MovementStart
              : CHOREOGRAPHY_PHASES.v2DeathStart,
        }),
      );
      continue;
    }

    if (
      eventType === "status_aged_to_zero" ||
      eventType === "status_broken_by_damage" ||
      eventType === "status_applied" ||
      eventType === "status_refreshed_or_extended" ||
      eventType === "status_cleared_by_new_death"
    ) {
      const recipientSlot = integer(event.recipient_global_slot);
      const recipient = project(anchorPoint(event.recipient_anchor), surface);
      const assignment = statusAssignments.get(eventId) ?? null;
      const sourceEvents = array(assignment?.sourceEvents).map(record);
      const applicationSources = sourceEvents.map((sourceEvent) => {
        const applicationEventId = identifier(sourceEvent?.event_id);
        const sourceSlot = integer(sourceEvent?.source_global_slot);
        const source = project(anchorPoint(sourceEvent?.source_anchor), surface);
        return applicationEventId === null || sourceSlot === null || source === null
          ? null
          : Object.freeze({
              eventId: applicationEventId,
              sourceSlot,
              sourcePublicAgentId: publicAgentIdForSlot(sourceSlot),
              source,
            });
      });
      const directApplicationSource =
        eventType === "status_applied"
          ? (applicationSources.find((candidate) => candidate?.eventId === eventId) ??
            null)
          : null;
      const lifecycleKind = identifier(assignment?.lifecycle);
      if (
        recipientSlot === null ||
        recipient === null ||
        assignment === null ||
        lifecycleKind === null ||
        applicationSources.some((source) => source === null)
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const status = resolveVisualToken("status", event.status_id, event);
      const lifecycle = resolveVisualToken("lifecycle", lifecycleKind, event);
      planned.push(
        Object.freeze({
          ...common,
          kind: "status_lifecycle",
          tokenId: status.tokenId,
          token: status,
          lifecycle: lifecycle.tokenId,
          lifecycleToken: lifecycle,
          recipientSlot,
          recipientPublicAgentId: publicAgentIdForSlot(recipientSlot),
          recipient,
          sourceSlot: directApplicationSource?.sourceSlot ?? null,
          sourcePublicAgentId: directApplicationSource?.sourcePublicAgentId ?? null,
          source: directApplicationSource?.source ?? null,
          applicationSources: Object.freeze(applicationSources),
          durationBefore: null,
          durationAfter: null,
          lane: assignment.lane,
          laneCount: assignment.laneCount,
          atomicEventIds: assignment.atomicEventIds,
          applicationEventIds: assignment.applicationEventIds,
          presentationSuppressed: !assignment.primary,
          spatial: assignment.primary,
          phaseStart: CHOREOGRAPHY_PHASES.v2StatusStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2ShieldStart,
        }),
      );
      continue;
    }

    if (eventType === "action_rejected") {
      const actorSlot = integer(event.actor_global_slot);
      const component = identifier(event.rejection_component);
      const actor = project(anchorPoint(event.actor_anchor), surface);
      if (
        actorSlot === null ||
        (component !== "domain" &&
          component !== "movement" &&
          component !== "combat_pair")
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "rejected_action",
          actorSlot,
          actorPublicAgentId: publicAgentIdForSlot(actorSlot),
          targetSlot: null,
          actor,
          target: null,
          component,
          route: null,
          spatial: actor !== null,
          phaseStart: CHOREOGRAPHY_PHASES.v2RejectionStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2AbilityStart,
        }),
      );
      continue;
    }

    if (eventType === "own_position_changed") {
      const start = project(point(event.start_position), surface);
      const end = project(point(event.successor_position), surface);
      if (start === null || end === null) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "movement_displacement",
          sourceSlot: integer(scene.selection?.controlled_global_slot),
          sourcePublicAgentId: publicAgentIdForSlot(
            integer(scene.selection?.controlled_global_slot),
          ),
          targetSlot: null,
          start,
          end,
          pathKind: "observed_own_position_change",
          spatial: false,
          presentationSuppressed: true,
          persistent: false,
          phaseStart: CHOREOGRAPHY_PHASES.povSuccessorObservationStart,
          phaseEnd: CHOREOGRAPHY_PHASES.total,
        }),
      );
      continue;
    }

    if (eventType === "own_health_changed") {
      const recipientSlot = integer(scene.selection?.controlled_global_slot);
      const healthBefore = finiteNumber(event.start_health);
      const healthAfter = finiteNumber(event.successor_health);
      const recipientAgent = sceneAgentBySlot.get(recipientSlot);
      const recipient = project(point(recipientAgent?.position), surface);
      if (
        recipientSlot === null ||
        healthBefore === null ||
        healthAfter === null ||
        recipient === null
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const delta = healthAfter - healthBefore;
      planned.push(
        Object.freeze({
          ...common,
          kind: "net_health",
          recipientSlot,
          recipientPublicAgentId: publicAgentIdForSlot(recipientSlot),
          recipient,
          netDelta: delta,
          healthBefore,
          healthAfter,
          outcome: delta < 0 ? "damage" : delta > 0 ? "healing" : "unchanged",
          lane: 0,
          spatial: true,
          phaseStart: CHOREOGRAPHY_PHASES.povSuccessorObservationStart,
          phaseEnd: CHOREOGRAPHY_PHASES.total,
        }),
      );
      continue;
    }

    if (eventType === "own_lifecycle_changed") {
      const selfActor = record(scene.self_actor);
      const agentSlot = integer(selfActor?.global_slot);
      const sceneAgent = sceneAgentBySlot.get(agentSlot);
      const startActive = event.start_active;
      const successorActive = event.successor_active;
      const startAlive = event.start_alive;
      const successorAlive = event.successor_alive;
      const startShield = integer(event.start_spawn_shield_remaining_ticks);
      const successorShield = integer(event.successor_spawn_shield_remaining_ticks);
      if (
        scene.audience !== "agent_pov" ||
        batch.audience !== "agent_pov" ||
        !selfActor ||
        agentSlot === null ||
        !sceneAgent ||
        identifier(selfActor?.public_agent_id) !== publicAgentIdForSlot(agentSlot) ||
        typeof startActive !== "boolean" ||
        typeof successorActive !== "boolean" ||
        typeof startAlive !== "boolean" ||
        typeof successorAlive !== "boolean" ||
        startShield === null ||
        startShield < 0 ||
        successorShield === null ||
        successorShield < 0
      ) {
        planned.push(Object.freeze({ ...common, kind: "unknown", spatial: false }));
        continue;
      }
      const cueSemantic =
        startActive && successorActive && startAlive && !successorAlive
          ? "agent_died"
          : startActive &&
              successorActive &&
              !startAlive &&
              successorAlive &&
              successorShield > 0
            ? "agent_respawned"
            : startActive &&
                successorActive &&
                startAlive &&
                successorAlive &&
                startShield > 0 &&
                successorShield === 0
              ? "spawn_shield_expired"
              : null;
      if (cueSemantic === null) {
        planned.push(
          Object.freeze({
            ...common,
            kind: "feed_only",
            spatial: false,
          }),
        );
        continue;
      }
      const anchor = project(point(selfActor.position), surface);
      planned.push(
        Object.freeze({
          ...common,
          kind: "semantic_pulse",
          cueSemantic,
          anchor,
          agentSlot,
          agentPublicAgentId: publicAgentIdForSlot(agentSlot),
          startActive,
          successorActive,
          startAlive,
          successorAlive,
          startSpawnShieldRemaining: startShield,
          successorSpawnShieldRemaining: successorShield,
          spatial: anchor !== null,
          phaseStart: CHOREOGRAPHY_PHASES.povSuccessorObservationStart,
          phaseEnd: CHOREOGRAPHY_PHASES.total,
        }),
      );
      continue;
    }

    if (
      eventType === "combat_countdown_reset" ||
      eventType === "health_regenerated" ||
      eventType === "cooldown_started" ||
      eventType === "cooldown_ready" ||
      eventType === "agent_died" ||
      eventType === "spawn_shield_expired" ||
      eventType === "respawn_wave_occurred" ||
      eventType === "agent_respawned"
    ) {
      const agentSlot = integer(event.agent_global_slot ?? event.recipient_global_slot);
      const teamId = integer(event.team_id);
      const eventAnchor =
        eventType === "respawn_wave_occurred"
          ? teamClockPoint(teamId, surface)
          : project(anchorPoint(event.agent_anchor ?? event.recipient_anchor), surface);
      planned.push(
        Object.freeze({
          ...common,
          kind: "semantic_pulse",
          cueSemantic: eventType,
          anchor: eventAnchor,
          agentSlot,
          agentPublicAgentId: publicAgentIdForSlot(agentSlot),
          teamId,
          value:
            eventType === "health_regenerated"
              ? finiteNumber(event.actual_health_regenerated)
              : null,
          spatial: eventAnchor !== null,
          phaseStart:
            eventType === "combat_countdown_reset" || eventType === "health_regenerated"
              ? CHOREOGRAPHY_PHASES.v2CountdownAndRegenStart
              : eventType === "cooldown_started" || eventType === "cooldown_ready"
                ? CHOREOGRAPHY_PHASES.v2CooldownStart
                : eventType === "agent_died"
                  ? CHOREOGRAPHY_PHASES.v2DeathStart
                  : eventType === "spawn_shield_expired"
                    ? CHOREOGRAPHY_PHASES.v2ShieldStart
                    : eventType === "respawn_wave_occurred"
                      ? CHOREOGRAPHY_PHASES.v2RespawnWaveStart
                      : CHOREOGRAPHY_PHASES.v2RespawnStart,
          phaseEnd:
            eventType === "combat_countdown_reset" || eventType === "health_regenerated"
              ? CHOREOGRAPHY_PHASES.v2CooldownStart
              : eventType === "cooldown_started" || eventType === "cooldown_ready"
                ? CHOREOGRAPHY_PHASES.v2ChargeStart
                : eventType === "agent_died"
                  ? CHOREOGRAPHY_PHASES.v2StatusStart
                  : eventType === "spawn_shield_expired"
                    ? CHOREOGRAPHY_PHASES.v2RespawnWaveStart
                    : eventType === "respawn_wave_occurred"
                      ? CHOREOGRAPHY_PHASES.v2RespawnStart
                      : CHOREOGRAPHY_PHASES.total,
        }),
      );
      continue;
    }

    if (
      eventType === "source_damage_output" ||
      eventType === "source_healing_output" ||
      eventType === "lethal_damage_contribution" ||
      eventType === "own_action_outcome" ||
      eventType === "own_status_changed" ||
      eventType === "own_cooldown_changed" ||
      eventType === "visible_body_observation_changed" ||
      eventType === "episode_ended"
    ) {
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          spatial: false,
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
  /** @type {Record<string, any>[]} */
  const routedEvents = planned.map((event) => {
    const route =
      event.kind === "activation" ? acceptedRoutes.get(event.eventId) : undefined;
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
  const scheduled = scheduleChoreography(routedEvents);
  const spatialEventCount = scheduled.events.filter((event) => event.spatial).length;
  /** @type {ReadonlyArray<Record<string, any>>} */
  const netOutcomeEvents = layoutOutcomeCues(scheduled.events, surface, "net_health");
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
    phases: scheduled.phases,
    events,
    bounds: Object.freeze({
      nodes: Math.min(events.length * 28 + 2, 512),
      animations: Math.min(spatialEventCount * 3 + 2, 512),
      persistentNodes: Math.min(persistentEventCount * 6, 64),
    }),
  });
}
