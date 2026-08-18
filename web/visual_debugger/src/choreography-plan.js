import {
  authorizedPresentationAudience,
  authorizedPresentationIncomingRows,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { layoutCrossPhaseOccupancy } from "./layout.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualPaintPartEnabled,
  visualFilterPaintKey,
} from "./visual-filters.js";
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

/**
 * Shared allocator/painter geometry in presentation pixels. Stroke-inclusive
 * radii and flare extents deliberately fit inside their paired frozen
 * footprint; route padding excludes transparent hit targets and underpainting.
 */
export const CHOREOGRAPHY_PAINT_FOOTPRINTS = Object.freeze({
  activation: Object.freeze({
    basic_damage: Object.freeze({ width: 44, height: 44 }),
    basic_heal: Object.freeze({ width: 44, height: 44 }),
    holy_word: Object.freeze({ width: 56, height: 56, flareExtent: 27 }),
    hunter_trap: Object.freeze({ width: 27, height: 27, flareExtent: 25 }),
    rogue_poison: Object.freeze({ width: 48, height: 48 }),
    warrior_charge: Object.freeze({
      width: 32.16,
      height: 27.52,
      flareExtent: 26,
    }),
    mage_burst: Object.freeze({ width: 70, height: 70, flareExtent: 34 }),
    local: Object.freeze({ width: 48, height: 48 }),
  }),
  rejection: Object.freeze({ width: 52, height: 52, ringRadius: 24 }),
  lifecycleReapplication: Object.freeze({
    width: 45,
    height: 45,
    ringRadius: 21,
  }),
  chargeDisplacement: Object.freeze({
    start: Object.freeze({ width: 10, height: 10, radius: 4 }),
    end: Object.freeze({ width: 12, height: 12, radius: 5 }),
  }),
  chargeOwnership: Object.freeze({ width: 72, height: 22, labelLength: 60 }),
  respawnWave: Object.freeze({
    width: 152,
    height: 32,
    panelWidth: 150,
    panelHeight: 30,
  }),
  route: Object.freeze({
    // The standard activation envelope is the animated r3 particle plus its
    // 3px glow and 1px clearance. Holy Word alone needs 8px for its r4 particle
    // and 4px glow (and its 5px route stroke plus 5px glow spans 7.5px).
    activationPathPadding: Object.freeze({ default: 7, holy_word: 8 }),
    rejectionPathPadding: 1,
    chargePathPadding: 1.5,
    activationMarkerPadding: 17,
    activationCompactMarkerPadding: 8,
    chargeMarkerPadding: 14,
  }),
});

const NET_EFFECT_CUE_RECTANGLE = centeredCueRectangle(48, 36);
const NET_TEXT_CUE_RECTANGLE = centeredCueRectangle(88, 36);
const UNCHANGED_NET_TEXT_CUE_RECTANGLE = centeredCueRectangle(102, 36);
const LIFECYCLE_BREAK_CUE_RECTANGLE = centeredCueRectangle(52, 52);
const LIFECYCLE_REAPPLICATION_CUE_RECTANGLE = centeredCueRectangle(
  CHOREOGRAPHY_PAINT_FOOTPRINTS.lifecycleReapplication.width,
  CHOREOGRAPHY_PAINT_FOOTPRINTS.lifecycleReapplication.height,
);
const REGENERATION_EFFECT_CUE_RECTANGLE = centeredCueRectangle(48, 48);
const REGENERATION_TEXT_CUE_RECTANGLE = Object.freeze({
  left: -32,
  top: 16,
  right: 32,
  bottom: 34,
});

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
 *   protectedRects?: ReadonlyArray<Record<string, any>>,
 * }} ProjectionSurface
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
  const horizontalInset = Math.min(112, Math.max(76, Number(bounds.width) * 0.2));
  return Object.freeze({
    x: teamId === 1 ? bounds.left + horizontalInset : bounds.right - horizontalInset,
    y: bounds.top + 24,
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
 * Size collision reservations from the paint parts that survived the local
 * filter. A disabled sibling may not enlarge or displace the remaining cue.
 * Plans predating visual filters retain the CP5 all-on dimensions.
 *
 * @param {Record<string, any>} event
 */
function paintedOutcomeCueDimensions(event) {
  const parts = event.paintParts;
  /** @param {string} part */
  const enabled = (part) => !parts || parts[part] === true;
  /** @type {Readonly<Record<string, number>>[]} */
  const rectangles = [];
  if (event.kind === "net_health") {
    if (enabled("effect")) {
      rectangles.push(NET_EFFECT_CUE_RECTANGLE);
    }
    if (enabled("battleText") || enabled("recipientText")) {
      rectangles.push(
        event.outcome === "unchanged"
          ? UNCHANGED_NET_TEXT_CUE_RECTANGLE
          : NET_TEXT_CUE_RECTANGLE,
      );
    }
  } else if (event.kind === "regeneration") {
    if (enabled("effect")) {
      rectangles.push(REGENERATION_EFFECT_CUE_RECTANGLE);
    }
    if (enabled("battleText")) {
      rectangles.push(REGENERATION_TEXT_CUE_RECTANGLE);
    }
  } else {
    if (enabled("effect") || enabled("break")) {
      rectangles.push(LIFECYCLE_BREAK_CUE_RECTANGLE);
    }
    if (enabled("reapplication")) {
      rectangles.push(LIFECYCLE_REAPPLICATION_CUE_RECTANGLE);
    }
  }
  return unionCueRectangles(rectangles);
}

/** @param {number} width @param {number} height */
function centeredCueRectangle(width, height) {
  return Object.freeze({
    left: -width / 2,
    top: -height / 2,
    right: width / 2,
    bottom: height / 2,
  });
}

/** @param {ReadonlyArray<Readonly<Record<string, number>>>} rectangles */
function unionCueRectangles(rectangles) {
  if (rectangles.length === 0) {
    return Object.freeze({ width: 0, height: 0 });
  }
  const horizontalExtent = Math.max(
    ...rectangles.flatMap(({ left, right }) => [Math.abs(left), Math.abs(right)]),
  );
  const verticalExtent = Math.max(
    ...rectangles.flatMap(({ top, bottom }) => [Math.abs(top), Math.abs(bottom)]),
  );
  return Object.freeze({
    width: horizontalExtent * 2,
    height: verticalExtent * 2,
  });
}

/**
 * Give every layout-owned fragment a collision-safe identity without changing
 * the authorized event identity carried by the plan row.
 *
 * @param {unknown} eventId
 * @param {string} role
 */
function crossPhaseLayoutKey(eventId, role) {
  return JSON.stringify(["event", String(eventId), role]);
}

/**
 * @param {unknown} value
 * @returns {{left: number, top: number, right: number, bottom: number, width: number, height: number} | null}
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
 * Accept the named scene contract while keeping older test/adapter surfaces
 * deterministic until their callers migrate. The allocator itself still sees
 * only explicit, stable keys.
 *
 * @param {ProjectionSurface | null} surface
 */
function crossPhaseProtectedRects(surface) {
  return Object.freeze(
    array(surface?.protectedRects).map((candidate, index) => {
      const region = record(candidate);
      const bounds = normalizedRectangle(region?.bounds ?? region);
      if (!region || !bounds) {
        throw new TypeError(`protectedRects[${index}] must contain valid bounds.`);
      }
      return Object.freeze({
        layoutKey:
          identifier(region.layoutKey) ?? JSON.stringify(["legacy-protected", index]),
        bounds,
      });
    }),
  );
}

/**
 * Route endpoints may enter only the durable body regions that explicitly own
 * those endpoints. No containment or nearest-body inference is permitted.
 *
 * @param {ProjectionSurface | null} surface
 * @param {string | null | undefined} ownerPresentationKey
 */
function protectedBodyKeys(surface, ownerPresentationKey) {
  if (!ownerPresentationKey) {
    return [];
  }
  return array(surface?.protectedRects).flatMap((candidate) => {
    const region = record(candidate);
    const key = identifier(region?.layoutKey);
    return region?.protectedKind === "body" &&
      region.ownerPresentationKey === ownerPresentationKey &&
      key !== null
      ? [key]
      : [];
  });
}

/**
 * Resolve an endpoint role to one exact durable body identity. Multiple body
 * regions for one presentation owner are ambiguous and must never be guessed.
 *
 * @param {ProjectionSurface | null} surface
 * @param {string | null | undefined} ownerPresentationKey
 */
function protectedBodyKey(surface, ownerPresentationKey) {
  const keys = protectedBodyKeys(surface, ownerPresentationKey);
  if (keys.length > 1) {
    throw new RangeError(
      `presentation owner ${ownerPresentationKey} has multiple protected bodies.`,
    );
  }
  return keys[0] ?? null;
}

/** @param {Record<string, any>} event */
function activationCueDimensions(event) {
  if (!event.target) {
    return event.tokenId === "mage_burst"
      ? CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.mage_burst
      : CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.local;
  }
  if (event.paintParts?.ability !== true) {
    const semanticSize =
      event.tokenId === "hunter_trap"
        ? 16
        : event.tokenId === "warrior_charge"
          ? 18
          : 24;
    return Object.freeze({ width: semanticSize, height: semanticSize });
  }
  const tokenId = /** @type {keyof typeof CHOREOGRAPHY_PAINT_FOOTPRINTS.activation} */ (
    event.tokenId
  );
  return (
    CHOREOGRAPHY_PAINT_FOOTPRINTS.activation[tokenId] ??
    CHOREOGRAPHY_PAINT_FOOTPRINTS.activation.rogue_poison
  );
}

/** @param {Record<string, any>} event */
function semanticPulseCueDimensions(event) {
  if (event.cueSemantic === "respawn_wave_occurred") {
    return CHOREOGRAPHY_PAINT_FOOTPRINTS.respawnWave;
  }
  if (event.cueSemantic === "agent_died" || event.cueSemantic === "agent_respawned") {
    return Object.freeze({ width: 68, height: 68 });
  }
  return Object.freeze({ width: 62, height: 62 });
}

/**
 * @param {ProjectionSurface | null} surface
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @param {string | null | undefined} presentationKey
 */
function projectedAgentRadius(surface, sceneByKey, presentationKey) {
  const radius = finiteNumber(sceneByKey.get(presentationKey ?? "")?.radius);
  if (radius === null || !surface) {
    return 0;
  }
  const projected = surface.worldLengthToScreen(radius);
  return Number.isFinite(projected) ? Number(projected) : 0;
}

/**
 * Run one filter-first reservation pass for every surviving information-bearing
 * fragment, then attach allocator-owned geometry without changing event order
 * or scientific anchors.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {ProjectionSurface | null} surface
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @returns {ReadonlyArray<Record<string, any>> | null}
 */
function layoutCrossPhaseEvents(events, surface, sceneByKey) {
  /** @type {Record<string, any>[]} */
  const requests = [];
  /** @type {Map<string, Record<string, any>>} */
  const bindings = new Map();
  /** @type {Record<string, any>[]} */
  const patches = events.map(() => ({}));

  /**
   * @param {number} eventIndex
   * @param {string} role
   * @param {number} roleOrder
   * @param {Record<string, any>} request
   * @param {Record<string, any>} [binding]
   */
  const add = (eventIndex, role, roleOrder, request, binding = {}) => {
    const layoutKey = crossPhaseLayoutKey(events[eventIndex].eventId, role);
    requests.push(
      Object.freeze({
        layoutKey,
        priority: request.kind === "route" ? 1 : 0,
        stableOrder: eventIndex * 4 + roleOrder,
        ...request,
      }),
    );
    bindings.set(layoutKey, Object.freeze({ eventIndex, role, ...binding }));
  };

  for (const [eventIndex, event] of events.entries()) {
    if (!event.spatial) {
      continue;
    }
    if (event.kind === "activation") {
      const dimensions = activationCueDimensions(event);
      if (event.target) {
        add(eventIndex, "impact", 0, {
          kind: "recipient_cue",
          anchor: event.target,
          anchorRadius: 60,
          recipientKey: event.targetPresentationKey ?? event.eventId,
          allowProtectedKeys:
            event.endpointPhase === "successor"
              ? protectedBodyKeys(surface, event.targetPresentationKey)
              : [],
          ...dimensions,
        });
      } else if (event.source && event.paintParts?.ability === true) {
        add(eventIndex, "source", 0, {
          kind: "perimeter_callout",
          anchor: event.source,
          recipientKey: event.sourcePresentationKey ?? event.eventId,
          allowProtectedKeys: protectedBodyKeys(surface, event.sourcePresentationKey),
          ...dimensions,
        });
      }
      if (
        event.paintParts?.ability === true &&
        event.source &&
        event.target &&
        event.targetPresentationKey
      ) {
        const sourceRadius = projectedAgentRadius(
          surface,
          sceneByKey,
          event.sourcePresentationKey,
        );
        const targetRadius = projectedAgentRadius(
          surface,
          sceneByKey,
          event.targetPresentationKey,
        );
        // Scene body regions describe the current successor endpoint. They may
        // own successor-anchored routes, but never historical Charge activation
        // endpoints from transition_start.
        const successorAnchored = event.endpointPhase === "successor";
        const sourceProtectedKey = successorAnchored
          ? protectedBodyKey(surface, event.sourcePresentationKey)
          : null;
        const targetProtectedKey = successorAnchored
          ? protectedBodyKey(surface, event.targetPresentationKey)
          : null;
        const allowProtectedKeys = [sourceProtectedKey, targetProtectedKey].filter(
          (key) => key !== null,
        );
        const pairKey = JSON.stringify([
          event.sourcePresentationKey,
          event.targetPresentationKey,
        ]);
        add(
          eventIndex,
          "route",
          2,
          {
            kind: "route",
            source: event.source,
            target: event.target,
            sourceRadius: sourceProtectedKey === null ? 0 : sourceRadius,
            targetRadius: targetProtectedKey === null ? 0 : targetRadius,
            sourceEndpointGap: sourceProtectedKey === null ? 0 : 3,
            targetEndpointGap: targetProtectedKey === null ? 0 : 3,
            pathPadding:
              event.tokenId === "holy_word"
                ? CHOREOGRAPHY_PAINT_FOOTPRINTS.route.activationPathPadding.holy_word
                : CHOREOGRAPHY_PAINT_FOOTPRINTS.route.activationPathPadding.default,
            markerPadding: CHOREOGRAPHY_PAINT_FOOTPRINTS.route.activationMarkerPadding,
            compactMarkerPadding:
              CHOREOGRAPHY_PAINT_FOOTPRINTS.route.activationCompactMarkerPadding,
            markerProgress: event.tokenId === "warrior_charge" ? 0.42 : undefined,
            allowProtectedKeys: [...new Set(allowProtectedKeys)],
            sourceProtectedKey: sourceProtectedKey ?? undefined,
            targetProtectedKey: targetProtectedKey ?? undefined,
          },
          { pairKey },
        );
        if (event.tokenId === "warrior_charge") {
          const ownershipAnchor = Object.freeze({
            x: (event.source.x + event.target.x) / 2,
            y: (event.source.y + event.target.y) / 2,
          });
          patches[eventIndex].ownershipAnchor = ownershipAnchor;
          add(eventIndex, "ownership", 1, {
            kind: "perimeter_callout",
            anchor: ownershipAnchor,
            recipientKey: pairKey,
            width: CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeOwnership.width,
            height: CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeOwnership.height,
          });
        }
      }
      continue;
    }
    if (
      event.kind === "net_health" ||
      event.kind === "regeneration" ||
      event.kind === "status_lifecycle"
    ) {
      const dimensions = paintedOutcomeCueDimensions(event);
      const recipientPresentationKey =
        event.kind === "regeneration"
          ? event.agentPresentationKey
          : event.recipientPresentationKey;
      add(eventIndex, "cue", 0, {
        kind: "recipient_cue",
        anchor: event.recipient,
        anchorRadius: 60,
        recipientKey: recipientPresentationKey ?? event.eventId,
        allowProtectedKeys: protectedBodyKeys(surface, recipientPresentationKey),
        ...dimensions,
      });
      continue;
    }
    if (event.kind === "rejected_action") {
      add(eventIndex, "cue", 0, {
        kind: "perimeter_callout",
        anchor: event.actor,
        recipientKey: event.actorPresentationKey ?? event.eventId,
        allowProtectedKeys: protectedBodyKeys(surface, event.actorPresentationKey),
        width: CHOREOGRAPHY_PAINT_FOOTPRINTS.rejection.width,
        height: CHOREOGRAPHY_PAINT_FOOTPRINTS.rejection.height,
      });
      continue;
    }
    if (event.kind === "semantic_pulse") {
      add(eventIndex, "cue", 0, {
        kind: "perimeter_callout",
        anchor: event.anchor,
        recipientKey:
          event.agentPresentationKey ??
          (event.teamId ? `team:${event.teamId}` : event.eventId),
        allowProtectedKeys: protectedBodyKeys(surface, event.agentPresentationKey),
        ...semanticPulseCueDimensions(event),
      });
      continue;
    }
    if (event.kind === "charge_displacement") {
      const targetProtectedKey = protectedBodyKey(surface, event.sourcePresentationKey);
      const targetRadius = projectedAgentRadius(
        surface,
        sceneByKey,
        event.sourcePresentationKey,
      );
      add(eventIndex, "start", 0, {
        kind: "recipient_cue",
        anchor: event.start,
        anchorRadius: 20,
        recipientKey: `${event.sourcePresentationKey}:charge`,
        allowProtectedKeys: [],
        width: CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeDisplacement.start.width,
        height: CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeDisplacement.start.height,
      });
      add(eventIndex, "end", 1, {
        kind: "recipient_cue",
        anchor: event.end,
        anchorRadius: 20,
        recipientKey: `${event.sourcePresentationKey}:charge`,
        allowProtectedKeys: protectedBodyKeys(surface, event.sourcePresentationKey),
        width: CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeDisplacement.end.width,
        height: CHOREOGRAPHY_PAINT_FOOTPRINTS.chargeDisplacement.end.height,
      });
      const pairKey = JSON.stringify([
        event.sourcePresentationKey,
        event.start,
        event.end,
      ]);
      add(
        eventIndex,
        "route",
        2,
        {
          kind: "route",
          source: event.start,
          target: event.end,
          sourceRadius: 0,
          targetRadius: targetProtectedKey === null ? 0 : targetRadius,
          sourceEndpointGap: 0,
          targetEndpointGap: targetProtectedKey === null ? 0 : 3,
          pathPadding: CHOREOGRAPHY_PAINT_FOOTPRINTS.route.chargePathPadding,
          markerPadding: CHOREOGRAPHY_PAINT_FOOTPRINTS.route.chargeMarkerPadding,
          markerProgress: 0.76,
          allowProtectedKeys: targetProtectedKey === null ? [] : [targetProtectedKey],
          targetProtectedKey: targetProtectedKey ?? undefined,
        },
        { pairKey },
      );
    }
  }

  if (requests.length === 0) {
    return Object.freeze(events.map((event) => event));
  }
  if (!surface?.viewportBounds) {
    return null;
  }
  const layout = layoutCrossPhaseOccupancy(
    /** @type {any} */ ({
      viewport: surface.viewportBounds,
      protectedRects: crossPhaseProtectedRects(surface),
      requests,
    }),
  );
  const routeMultiplicity = new Map();
  for (const binding of bindings.values()) {
    if (binding.role === "route") {
      routeMultiplicity.set(
        binding.pairKey,
        (routeMultiplicity.get(binding.pairKey) ?? 0) + 1,
      );
    }
  }
  const placements = /** @type {ReadonlyArray<Record<string, any>>} */ (
    layout.placements
  );
  for (const placement of placements) {
    const binding = bindings.get(placement.layoutKey);
    if (!binding) {
      throw new Error(`missing cross-phase binding ${placement.layoutKey}.`);
    }
    const patch = patches[binding.eventIndex];
    if (binding.role === "route") {
      Object.assign(patch, {
        route: placement,
        routeLayoutKey: placement.layoutKey,
        routeLane: placement.lane,
        routeBridgeGaps: placement.bridgeGaps,
        routeMultiplicity: routeMultiplicity.get(binding.pairKey) ?? 1,
      });
      continue;
    }
    const cuePatch = {
      [`${binding.role}Cue`]: placement.center,
      [`${binding.role}Bounds`]: placement.bounds,
      [`${binding.role}Leader`]: placement.leader,
      [`${binding.role}Disposition`]: placement.disposition,
      [`${binding.role}CueCollisionFree`]: placement.collisionFree,
      [`${binding.role}LayoutKey`]: placement.layoutKey,
    };
    if (binding.role === "cue") {
      Object.assign(patch, {
        cue: placement.center,
        cueBounds: placement.bounds,
        cueLeader: placement.leader,
        cueDisposition: placement.disposition,
        cueCollisionFree: placement.collisionFree,
        cueLayoutKey: placement.layoutKey,
        spatialDisposition: "rendered",
      });
    } else if (binding.role === "start" || binding.role === "end") {
      Object.assign(patch, {
        [`${binding.role}Cue`]: placement.center,
        [`${binding.role}CueBounds`]: placement.bounds,
        [`${binding.role}CueLeader`]: placement.leader,
        [`${binding.role}CueDisposition`]: placement.disposition,
        [`${binding.role}CueCollisionFree`]: placement.collisionFree,
        [`${binding.role}CueLayoutKey`]: placement.layoutKey,
      });
    } else if (binding.role === "ownership") {
      Object.assign(patch, cuePatch, {
        ownershipSpatialDisposition: "rendered",
      });
    } else {
      Object.assign(patch, cuePatch);
    }
  }
  return Object.freeze(
    events.map((event, index) => Object.freeze({ ...event, ...patches[index] })),
  );
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

/** @param {Record<string, any>} event @returns {ChoreographyFamily | null} */
function choreographyFamily(event) {
  if (event.kind === "activation") return "ability";
  if (event.kind === "rejected_action") return "rejection";
  if (event.kind === "net_health") return "health";
  if (event.kind === "regeneration") return "recovery";
  if (event.kind === "charge_displacement") return "charge";
  if (event.kind === "movement_displacement") return "movement";
  if (event.kind === "status_lifecycle") return "status";
  if (event.kind !== "semantic_pulse") return null;
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
 * Resolve the independently owned visual parts of one authorized event. Rows
 * without a transient paint registration remain untouched: they still carry
 * feed/atomic identity, but never enter the paint registry by accident.
 *
 * @param {Record<string, any>} event
 * @param {Record<string, boolean>} visualFilters
 * @returns {Readonly<Record<string, boolean>> | null}
 */
function authorizedPaintParts(event, visualFilters) {
  /** @param {Record<string, string>} tag */
  const enabled = (tag) => isVisualPaintPartEnabled(visualFilters, tag);
  if (event.kind === "rejected_action") {
    return Object.freeze({
      effect: enabled({
        surface: "transient",
        kind: "rejected_action_feedback",
      }),
    });
  }
  if (event.kind === "activation") {
    const ability = enabled({
      surface: "transient",
      kind: "activation",
      component: event.component,
      part: "ability",
    });
    const semantic =
      event.impactSemantic === "damage" || event.impactSemantic === "healing"
        ? enabled({
            surface: "transient",
            kind: "activation",
            semantic: event.impactSemantic,
            part: "semantic",
          })
        : false;
    return Object.freeze({ ability, semantic });
  }
  if (event.kind === "net_health") {
    const effect =
      event.outcome === "damage" || event.outcome === "healing"
        ? enabled({
            surface: "transient",
            kind: "net_health",
            outcome: event.outcome,
            part: "effect",
          })
        : false;
    return Object.freeze({
      effect,
      battleText: enabled({
        surface: "transient",
        kind: "net_health",
        outcome: event.outcome,
        part: "battle_text",
      }),
      recipientText: enabled({
        surface: "transient",
        kind: "net_health",
        outcome: event.outcome,
        part: "recipient_text",
      }),
    });
  }
  if (event.kind === "regeneration") {
    return Object.freeze({
      effect: enabled({
        surface: "transient",
        kind: "regeneration",
        part: "effect",
      }),
      battleText: enabled({
        surface: "transient",
        kind: "regeneration",
        part: "battle_text",
      }),
    });
  }
  if (event.kind === "charge_displacement") {
    return Object.freeze({
      effect: enabled({ surface: "transient", kind: "charge_movement" }),
    });
  }
  if (event.kind === "status_lifecycle") {
    if (event.lifecycle === "trap_broken_and_reapplied") {
      return Object.freeze({
        break: enabled({
          surface: "transient",
          kind: "status_lifecycle",
          lifecycle: event.lifecycle,
          part: "break",
        }),
        reapplication: enabled({
          surface: "transient",
          kind: "status_lifecycle",
          lifecycle: event.lifecycle,
          part: "reapplication",
        }),
      });
    }
    return Object.freeze({
      effect: enabled({
        surface: "transient",
        kind: "status_lifecycle",
        lifecycle: event.lifecycle,
        part: "effect",
      }),
    });
  }
  if (event.kind !== "semantic_pulse") {
    return null;
  }
  /** @type {Record<string, string> | null} */
  const tag =
    event.cueSemantic === "cooldown_started"
      ? { surface: "transient", kind: "cooldown", semantic: "started" }
      : event.cueSemantic === "cooldown_ready"
        ? { surface: "transient", kind: "cooldown", semantic: "ready" }
        : event.cueSemantic === "agent_died"
          ? { surface: "transient", kind: "death_effect" }
          : event.cueSemantic === "respawn_wave_occurred"
            ? { surface: "transient", kind: "respawn_wave" }
            : event.cueSemantic === "agent_respawned"
              ? { surface: "transient", kind: "resurrection_effect" }
              : event.cueSemantic === "spawn_shield_expired"
                ? { surface: "transient", kind: "spawn_shield_expiry" }
                : null;
  return tag === null ? null : Object.freeze({ effect: enabled(tag) });
}

/**
 * Classify one registered transient before any presentation geometry exists.
 *
 * @param {Record<string, any>} event
 * @param {Record<string, boolean>} visualFilters
 */
function authorizedPaintDecision(event, visualFilters) {
  const paintParts = authorizedPaintParts(event, visualFilters);
  return paintParts === null
    ? null
    : Object.freeze({
        paintParts,
        enabled: Object.values(paintParts).some((part) => part === true),
      });
}

/**
 * Apply browser-local paint policy without deleting or rewriting any authorized
 * row identity. A fully disabled event remains inspectable in plan order but is
 * made non-spatial before phase allocation and collision layout.
 *
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {Record<string, boolean>} visualFilters
 */
function applyAuthorizedVisualFilters(events, visualFilters) {
  return events.map((event) => {
    const paintParts = event.paintParts ?? authorizedPaintParts(event, visualFilters);
    if (paintParts === null) {
      return event;
    }
    const enabled = Object.values(paintParts).some((part) => part === true);
    return Object.freeze({
      ...event,
      paintParts,
      presentationSuppressed: !enabled,
      spatial: Boolean(event.spatial && enabled),
    });
  });
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

/** @param {unknown} rawAnchor */
function authorizedWorldPoint(rawAnchor) {
  const anchor = record(rawAnchor);
  return anchor ? point(anchor.position) : null;
}

/** @param {unknown} rawAnchor @param {ProjectionSurface | null} surface */
function authorizedAnchor(rawAnchor, surface) {
  return project(authorizedWorldPoint(rawAnchor), surface);
}

/**
 * Validate the serialized team anchor without constructing viewport geometry.
 *
 * @param {unknown} rawAnchor
 */
function authorizedTeamWaveIdentity(rawAnchor) {
  const teamAnchor = record(rawAnchor);
  const teamIndex = integer(teamAnchor?.team_index);
  const teamId = integer(teamAnchor?.team_id);
  if (
    teamAnchor?.phase !== "successor" ||
    (teamIndex !== 0 && teamIndex !== 1) ||
    teamId !== teamIndex + 1
  ) {
    return null;
  }
  return Object.freeze({
    teamIndex,
    teamId,
    teamSide: teamIndex === 0 ? "left" : "right",
    label: `RESPAWNING · TEAM ${teamIndex === 0 ? "A" : "B"}`,
  });
}

/**
 * Require exact point equality without using a scene position as trajectory
 * authority. The scene comparison below is only a coherence check: the
 * serialized successor trajectory remains the endpoint source.
 *
 * @param {unknown} left
 * @param {unknown} right
 */
function sameAuthorizedPoint(left, right) {
  const leftPoint = point(left);
  const rightPoint = point(right);
  return (
    leftPoint !== null &&
    rightPoint !== null &&
    leftPoint[0] === rightPoint[0] &&
    leftPoint[1] === rightPoint[1]
  );
}

/**
 * Validate the Oracle trajectory identity graph once. Keys are opaque and
 * every public identity must be one-to-one across the trajectory and scene.
 *
 * @param {Record<string, any>} latest
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @returns {Map<string, Record<string, any>> | null}
 */
function authorizedTrajectoryMap(latest, sceneByKey) {
  if (!Array.isArray(latest.agent_phase_trajectories)) {
    return null;
  }
  const trajectories = latest.agent_phase_trajectories;
  if (trajectories.length !== sceneByKey.size) {
    return null;
  }
  /** @type {Map<string, Record<string, any>>} */
  const byKey = new Map();
  const publicIds = new Set();
  for (const candidate of trajectories) {
    const trajectory = record(candidate);
    const key = identifier(trajectory?.agent_presentation_key);
    const publicId = identifier(trajectory?.agent_public_agent_id);
    if (
      !trajectory ||
      key === null ||
      publicId === null ||
      byKey.has(key) ||
      publicIds.has(publicId)
    ) {
      return null;
    }
    const sceneAgent = sceneByKey.get(key);
    if (!sceneAgent || identifier(sceneAgent.public_agent_id) !== publicId) {
      return null;
    }
    for (const phase of ["transition_start", "post_charge", "successor"]) {
      const anchor = record(trajectory[phase]);
      if (
        !anchor ||
        anchor.phase !== phase ||
        identifier(anchor.presentation_key) !== key ||
        identifier(anchor.public_agent_id) !== publicId ||
        point(anchor.position) === null
      ) {
        return null;
      }
    }
    if (!sameAuthorizedPoint(trajectory.successor.position, sceneAgent.position)) {
      return null;
    }
    byKey.set(key, trajectory);
    publicIds.add(publicId);
  }
  return byKey;
}

/**
 * Exact-join one serialized event anchor to its phase trajectory. A key match
 * alone is insufficient: public identity, phase, and point must all agree.
 *
 * @param {Map<string, Record<string, any>>} trajectories
 * @param {unknown} rawAnchor
 * @param {"transition_start" | "post_charge" | "successor"} phase
 * @returns {Record<string, any> | null}
 */
function trajectoryForAuthorizedAnchor(trajectories, rawAnchor, phase) {
  const anchor = record(rawAnchor);
  const key = identifier(anchor?.presentation_key);
  const publicId = identifier(anchor?.public_agent_id);
  const trajectory = key === null ? null : (trajectories.get(key) ?? null);
  const trajectoryAnchor = record(trajectory?.[phase]);
  if (
    !anchor ||
    key === null ||
    publicId === null ||
    anchor.phase !== phase ||
    !trajectory ||
    identifier(trajectory.agent_public_agent_id) !== publicId ||
    !trajectoryAnchor ||
    !sameAuthorizedPoint(anchor.position, trajectoryAnchor.position)
  ) {
    return null;
  }
  return trajectory;
}

/**
 * @typedef {{
 *   row: Readonly<Record<string, any>>,
 *   event: Readonly<Record<string, any>>,
 *   applicationSource: Readonly<Record<string, any>> | null,
 * }} AuthorizedStatusAtom
 * @typedef {{
 *   atoms: AuthorizedStatusAtom[],
 *   firstIndex: number,
 *   recipientAnchor: Readonly<Record<string, any>>,
 *   recipientPresentationKey: string,
 *   recipientPublicAgentId: string,
 * }} AuthorizedStatusGroup
 */

/**
 * Resolve composition semantics from already validated status atomics without
 * constructing any screen-space geometry.
 *
 * @param {AuthorizedStatusGroup} group
 */
function authorizedStatusSelection(group) {
  /** @param {string} kind */
  const first = (kind) => group.atoms.find(({ row }) => row.kind === kind) ?? null;
  const cleared = first("status_cleared_by_new_death");
  const refreshed = first("status_refreshed_or_extended");
  const broken = first("status_broken_by_damage");
  const aged = first("status_aged_to_zero");
  const applications = group.atoms.filter(({ row }) => row.kind === "status_applied");
  const applied = applications.at(0) ?? null;
  const lifecycleId = cleared
    ? "cleared_by_death"
    : refreshed
      ? "refreshed"
      : broken && applied
        ? "trap_broken_and_reapplied"
        : aged && applied
          ? "reapplied"
          : broken
            ? "trap_broken"
            : aged
              ? "expired"
              : "applied";
  const primary =
    cleared ??
    refreshed ??
    (applied && (broken || aged) ? applied : null) ??
    broken ??
    aged ??
    applied;
  return primary === null
    ? null
    : Object.freeze({
        applications: Object.freeze(applications),
        lifecycleId,
        primary,
      });
}

/**
 * Compose exactly one display row per authorized status group while retaining
 * every serialized atomic and application identity as ordered metadata. The
 * normalized Latest Events rows remain separate and unchanged outside this
 * presentation-only plan.
 *
 * @param {ReadonlyArray<Readonly<{
 *   id: string,
 *   kind: string,
 *   vocabulary: "event" | "recipient_cue" | "observation_delta",
 *   payload: Readonly<Record<string, any>>,
 * }>>} rows
 * @param {string} transitionId
 * @param {Map<string, Record<string, any>>} sceneByKey
 * @param {Map<string, Record<string, any>> | null} trajectories
 * @param {ProjectionSurface | null} surface
 * @param {Record<string, boolean>} visualFilters
 * @returns {Map<string, Readonly<Record<string, any>>> | null}
 */
function authorizedStatusCompositions(
  rows,
  transitionId,
  sceneByKey,
  trajectories,
  surface,
  visualFilters,
) {
  /** @type {Map<string, AuthorizedStatusGroup>} */
  const groups = new Map();
  const seenEventIds = new Set();
  let statusRowCount = 0;
  for (const [index, row] of rows.entries()) {
    if (!STATUS_EVENT_TYPES.has(row.kind)) {
      continue;
    }
    statusRowCount += 1;
    const event = record(row.payload);
    const eventId = identifier(event?.event_id);
    const eventKind = identifier(event?.event_kind);
    const recipientAnchor = record(event?.recipient_anchor);
    const recipientKey = identifier(recipientAnchor?.presentation_key);
    const statusChannel = integer(event?.status_channel);
    const statusId = identifier(event?.status_id);
    if (
      trajectories === null ||
      row.vocabulary !== "event" ||
      !event ||
      eventId !== row.id ||
      eventKind !== row.kind ||
      seenEventIds.has(row.id) ||
      recipientKey === null ||
      statusChannel === null ||
      statusId === null
    ) {
      return null;
    }
    const recipientTrajectory = trajectoryForAuthorizedAnchor(
      trajectories,
      recipientAnchor,
      "successor",
    );
    const recipientAgent = sceneByKey.get(recipientKey);
    const recipientWorld = authorizedWorldPoint(recipientTrajectory?.successor);
    if (
      recipientTrajectory === null ||
      !recipientAgent ||
      identifier(recipientAgent.public_agent_id) !==
        identifier(recipientTrajectory.agent_public_agent_id) ||
      recipientWorld === null
    ) {
      return null;
    }
    /** @type {Readonly<Record<string, any>> | null} */
    let applicationSource = null;
    if (row.kind === "status_applied") {
      const sourceAnchor = record(event.source_anchor);
      const sourceKey = identifier(sourceAnchor?.presentation_key);
      const sourceTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        sourceAnchor,
        "successor",
      );
      const sourceAgent = sourceKey === null ? null : sceneByKey.get(sourceKey);
      if (
        sourceKey === null ||
        sourceTrajectory === null ||
        !sourceAgent ||
        identifier(sourceAgent.public_agent_id) !==
          identifier(sourceTrajectory.agent_public_agent_id)
      ) {
        return null;
      }
      applicationSource = Object.freeze({
        eventId: row.id,
        sourcePresentationKey: sourceKey,
        sourcePublicAgentId: sourceTrajectory.agent_public_agent_id,
      });
    }
    seenEventIds.add(row.id);
    const key = JSON.stringify([transitionId, recipientKey, statusChannel, statusId]);
    /** @type {AuthorizedStatusGroup} */
    const group = groups.get(key) ?? {
      atoms: [],
      firstIndex: index,
      recipientAnchor: recipientTrajectory.successor,
      recipientPresentationKey: recipientKey,
      recipientPublicAgentId: recipientTrajectory.agent_public_agent_id,
    };
    if (
      group.recipientPresentationKey !== recipientKey ||
      group.recipientPublicAgentId !== recipientTrajectory.agent_public_agent_id ||
      !sameAuthorizedPoint(
        group.recipientAnchor.position,
        recipientTrajectory.successor.position,
      )
    ) {
      return null;
    }
    group.atoms.push(
      Object.freeze({
        row,
        event,
        applicationSource,
      }),
    );
    groups.set(key, group);
  }
  if (statusRowCount === 0) {
    return new Map();
  }

  /** @type {Map<string, Readonly<Record<string, any>>>} */
  const presentationByKey = new Map();
  for (const [key, group] of groups) {
    const selection = authorizedStatusSelection(group);
    if (selection === null) {
      return null;
    }
    const paintParts = authorizedPaintParts(
      {
        kind: "status_lifecycle",
        lifecycle: selection.lifecycleId,
      },
      visualFilters,
    );
    if (paintParts === null) {
      return null;
    }
    presentationByKey.set(
      key,
      Object.freeze({
        ...selection,
        paintParts,
        enabled: Object.values(paintParts).some((part) => part === true),
      }),
    );
  }

  /** @type {Map<string, {lane: number, laneCount: number}>} */
  const lanes = new Map();
  /** @type {Map<string, Array<[string, any]>>} */
  const groupsByRecipient = new Map();
  for (const [key, group] of groups) {
    if (presentationByKey.get(key)?.enabled !== true) {
      continue;
    }
    const recipientGroups = groupsByRecipient.get(group.recipientPresentationKey) ?? [];
    recipientGroups.push([key, group]);
    groupsByRecipient.set(group.recipientPresentationKey, recipientGroups);
  }
  for (const recipientGroups of groupsByRecipient.values()) {
    recipientGroups.sort((left, right) => left[1].firstIndex - right[1].firstIndex);
    recipientGroups.forEach(([key], lane) => {
      lanes.set(key, { lane, laneCount: recipientGroups.length });
    });
  }

  /** @type {Map<string, Readonly<Record<string, any>>>} */
  const compositions = new Map();
  for (const [key, group] of groups) {
    const presentation = presentationByKey.get(key);
    if (!presentation) {
      return null;
    }
    const applications = /** @type {ReadonlyArray<AuthorizedStatusAtom>} */ (
      presentation.applications
    );
    const lifecycleId = /** @type {string} */ (presentation.lifecycleId);
    const primary = /** @type {AuthorizedStatusAtom} */ (presentation.primary);
    const paintParts = /** @type {Readonly<Record<string, boolean>>} */ (
      presentation.paintParts
    );
    const enabled = presentation.enabled === true;
    const atomicEventIds = Object.freeze(group.atoms.map(({ row }) => row.id));
    const applicationEventIds = Object.freeze(applications.map(({ row }) => row.id));
    const applicationSources = Object.freeze(
      applications.map(({ applicationSource }) => applicationSource),
    );
    if (applicationSources.some((source) => source === null)) {
      return null;
    }
    const lane = lanes.get(key) ?? { lane: 0, laneCount: 0 };
    const status = resolveVisualToken("status", primary.event.status_id, primary.event);
    const lifecycle = resolveVisualToken("lifecycle", lifecycleId, primary.event);
    const directApplicationSource = primary.applicationSource;
    const recipient = enabled ? authorizedAnchor(group.recipientAnchor, surface) : null;
    if (enabled && recipient === null) {
      return null;
    }
    const composition = Object.freeze({
      eventId: primary.row.id,
      eventType: primary.row.kind,
      transitionId,
      authorityVocabulary: primary.row.vocabulary,
      kind: "status_lifecycle",
      tokenId: status.tokenId,
      token: status,
      lifecycle: lifecycle.tokenId,
      lifecycleToken: lifecycle,
      recipientPresentationKey: group.recipientPresentationKey,
      recipientPublicAgentId: group.recipientPublicAgentId,
      recipient,
      sourcePresentationKey: directApplicationSource?.sourcePresentationKey ?? null,
      sourcePublicAgentId: directApplicationSource?.sourcePublicAgentId ?? null,
      applicationSources,
      durationBefore: null,
      durationAfter: null,
      lane: enabled ? lane.lane : null,
      laneCount: enabled ? lane.laneCount : 0,
      statusLayoutOrder: group.firstIndex,
      atomicEventIds,
      applicationEventIds,
      paintParts,
      presentationSuppressed: !enabled,
      spatial: enabled,
      phaseStart: CHOREOGRAPHY_PHASES.v2StatusStart,
      phaseEnd: CHOREOGRAPHY_PHASES.v2ShieldStart,
    });
    for (const atom of group.atoms) {
      compositions.set(atom.row.id, composition);
    }
  }
  return compositions;
}

/**
 * Build choreography strictly from the normalized presentation's Latest
 * Events branch. Outgoing inspection is intentionally outside this function.
 * Shared observation deltas retain noncausal feed-only vocabulary.
 *
 * @param {Record<string, any>} presentation
 * @param {ProjectionSurface | null} surface
 * @param {Record<string, boolean>} visualFilters
 * @returns {Readonly<ChoreographyPlan> | null}
 */
function buildAuthorizedPresentationChoreographyPlan(
  presentation,
  surface,
  visualFilters,
) {
  const latest = record(presentation.latest_events);
  const scene = authorizedPresentationSceneView(presentation);
  if (!latest || !scene) {
    return null;
  }
  const transitionId =
    scientificIdentity(latest.incoming_transition_id) ??
    scientificIdentity(latest.incoming_recipient_transition_id);
  const simulatorStep =
    integer(latest.incoming_successor_simulator_step_count) ??
    integer(presentation.simulator_step_count);
  if (transitionId === null || simulatorStep === null) {
    return null;
  }
  const rows = authorizedPresentationIncomingRows(presentation);
  const audience = authorizedPresentationAudience(presentation);
  /** @type {Map<string, Record<string, any>>} */
  const sceneByKey = new Map();
  const scenePublicIds = new Set();
  for (const candidate of array(scene.agents)) {
    const agent = record(candidate);
    const key = identifier(agent?.presentation_key);
    const publicId = identifier(agent?.public_agent_id);
    if (
      !agent ||
      key === null ||
      publicId === null ||
      sceneByKey.has(key) ||
      scenePublicIds.has(publicId)
    ) {
      return null;
    }
    sceneByKey.set(key, agent);
    scenePublicIds.add(publicId);
  }
  const trajectories =
    audience === "researcher" ? authorizedTrajectoryMap(latest, sceneByKey) : null;
  if (audience === "researcher" && trajectories === null) {
    return null;
  }
  const statusCompositions = authorizedStatusCompositions(
    rows,
    transitionId,
    sceneByKey,
    trajectories,
    surface,
    visualFilters,
  );
  if (statusCompositions === null) {
    return null;
  }
  /** @type {Record<string, any>[]} */
  const planned = [];
  const plannedStatusCompositions = new Set();
  const healthLaneByKey = new Map();
  for (const row of rows) {
    const event = row.payload;
    const common = {
      eventId: row.id,
      eventType: row.kind,
      transitionId,
      authorityVocabulary: row.vocabulary,
    };
    if (row.vocabulary === "observation_delta") {
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          spatial: false,
          noncausal: true,
        }),
      );
      continue;
    }
    if (row.vocabulary === "recipient_cue") {
      const recipientKey = presentation.recipient_presentation_key;
      const recipientAgent = sceneByKey.get(recipientKey);
      const recipientWorld = point(recipientAgent?.position);
      const healthBefore = finiteNumber(event.start_health);
      const healthAfter = finiteNumber(event.successor_health);
      if (
        row.kind === "own_health_changed" &&
        recipientWorld &&
        healthBefore !== null &&
        healthAfter !== null
      ) {
        const delta = healthAfter - healthBefore;
        const outcome = delta < 0 ? "damage" : delta > 0 ? "healing" : "unchanged";
        const paintDecision = authorizedPaintDecision(
          { kind: "net_health", outcome },
          visualFilters,
        );
        if (paintDecision === null) {
          return null;
        }
        const recipient = paintDecision.enabled
          ? project(recipientWorld, surface)
          : null;
        if (paintDecision.enabled && recipient === null) {
          planned.push(
            Object.freeze({
              ...common,
              kind: "feed_only",
              spatial: false,
              noncausal: true,
            }),
          );
          continue;
        }
        planned.push(
          Object.freeze({
            ...common,
            kind: "net_health",
            recipientPresentationKey: recipientKey,
            recipientPublicAgentId: presentation.recipient_public_agent_id,
            recipient,
            netDelta: delta,
            healthBefore,
            healthAfter,
            outcome,
            lane: paintDecision.enabled ? 0 : null,
            paintParts: paintDecision.paintParts,
            presentationSuppressed: !paintDecision.enabled,
            spatial: paintDecision.enabled,
            noncausal: true,
            presentationKind: "successor_observation",
            phaseStart: CHOREOGRAPHY_PHASES.povSuccessorObservationStart,
            phaseEnd: CHOREOGRAPHY_PHASES.total,
          }),
        );
      } else {
        planned.push(
          Object.freeze({
            ...common,
            kind: "feed_only",
            spatial: false,
            noncausal: true,
          }),
        );
      }
      continue;
    }

    if (STATUS_EVENT_TYPES.has(row.kind)) {
      const composition = statusCompositions.get(row.id);
      if (!composition) {
        return null;
      }
      if (!plannedStatusCompositions.has(composition)) {
        planned.push(composition);
        plannedStatusCompositions.add(composition);
      }
      continue;
    }

    const sourceAnchor = record(event.source_anchor);
    const recipientAnchor = record(event.recipient_anchor);
    const agentAnchor = record(event.agent_anchor);
    if (row.kind === "ability_activated") {
      if (trajectories === null) {
        return null;
      }
      const sourceTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        sourceAnchor,
        "transition_start",
      );
      const targetTrajectory =
        recipientAnchor === null
          ? null
          : trajectoryForAuthorizedAnchor(
              trajectories,
              recipientAnchor,
              "transition_start",
            );
      const sourceKey = identifier(sourceAnchor?.presentation_key);
      const targetKey = identifier(recipientAnchor?.presentation_key);
      if (
        sourceTrajectory === null ||
        (recipientAnchor !== null && targetTrajectory === null)
      ) {
        return null;
      }
      const sourceAgent = sourceKey === null ? null : sceneByKey.get(sourceKey);
      if (
        sourceKey === null ||
        !sourceAgent ||
        identifier(sourceAgent.public_agent_id) !==
          identifier(sourceTrajectory.agent_public_agent_id)
      ) {
        return null;
      }
      const component = event.ability_component;
      const token =
        component === "ultimate"
          ? ultimateTokenFromClassId(sourceAgent.class_id, event)
          : resolveVisualToken(
              "activation",
              sourceAgent.class_id === 5 ? "basic_heal" : "basic_damage",
              event,
            );
      const impactSemantic = activationImpactSemantic(token.tokenId);
      const paintParts = authorizedPaintParts(
        { kind: "activation", component, impactSemantic },
        visualFilters,
      );
      if (paintParts === null) {
        return null;
      }
      const abilityEnabled = paintParts.ability === true;
      const semanticEnabled = paintParts.semantic === true;
      const targetAgent = targetKey === null ? null : sceneByKey.get(targetKey);
      if (
        (targetKey === null) !== (targetTrajectory === null) ||
        (targetKey !== null &&
          (!targetAgent ||
            identifier(targetAgent.public_agent_id) !==
              identifier(targetTrajectory?.agent_public_agent_id)))
      ) {
        return null;
      }
      const endpointPhase =
        token.tokenId === "warrior_charge" ? "transition_start" : "successor";
      const authorizedSource = abilityEnabled
        ? authorizedAnchor(sourceTrajectory[endpointPhase], surface)
        : null;
      const authorizedTarget =
        abilityEnabled || semanticEnabled
          ? authorizedAnchor(targetTrajectory?.[endpointPhase], surface)
          : null;
      if (
        (abilityEnabled && authorizedSource === null) ||
        ((abilityEnabled || semanticEnabled) &&
          targetTrajectory !== null &&
          authorizedTarget === null)
      ) {
        return null;
      }
      const source = authorizedSource;
      const target = authorizedTarget;
      const routed = Boolean(abilityEnabled && source && target && targetAgent);
      planned.push(
        Object.freeze({
          ...common,
          kind: "activation",
          component,
          tokenId: token.tokenId,
          token,
          impactSemantic,
          lane: component === "ultimate" ? 1 : 0,
          sourcePresentationKey: sourceKey,
          sourcePublicAgentId: sourceTrajectory.agent_public_agent_id,
          sourceClassId: sourceAgent.class_id,
          sourceClass: classTokenFromId(sourceAgent.class_id),
          targetPresentationKey: targetKey,
          targetPublicAgentId: targetTrajectory?.agent_public_agent_id ?? null,
          targetDisclosure: targetKey === null ? "target_none" : "public",
          source,
          target,
          endpointPhase,
          route: null,
          paintParts,
          spatial:
            (abilityEnabled && source !== null) || (semanticEnabled && target !== null),
          presentationKind: routed
            ? "routed"
            : !abilityEnabled && semanticEnabled && target
              ? "target_only_impact"
              : "source_local",
          phaseStart: CHOREOGRAPHY_PHASES.v2AbilityStart,
          phaseImpact: CHOREOGRAPHY_PHASES.v2HealthResolutionStart - 20,
          phaseEnd: CHOREOGRAPHY_PHASES.v2HealthResolutionStart,
        }),
      );
      continue;
    }
    if (row.kind === "recipient_health_resolution") {
      if (trajectories === null) {
        return null;
      }
      const recipientTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        recipientAnchor,
        "transition_start",
      );
      const recipientKey = identifier(recipientAnchor?.presentation_key);
      const before = finiteNumber(event.transition_start_health);
      const after = finiteNumber(event.health_after_combat_resolution);
      const delta = finiteNumber(event.realized_net_health_change);
      const recipientWorld = authorizedWorldPoint(recipientTrajectory?.successor);
      if (
        recipientTrajectory === null ||
        recipientKey === null ||
        recipientWorld === null ||
        before === null ||
        after === null ||
        delta === null
      ) {
        return null;
      }
      const outcome = delta < 0 ? "damage" : delta > 0 ? "healing" : "unchanged";
      const paintDecision = authorizedPaintDecision(
        { kind: "net_health", outcome },
        visualFilters,
      );
      if (paintDecision === null) {
        return null;
      }
      const recipient = paintDecision.enabled ? project(recipientWorld, surface) : null;
      if (paintDecision.enabled && recipient === null) {
        return null;
      }
      const lane = paintDecision.enabled
        ? (healthLaneByKey.get(recipientKey) ?? 0)
        : null;
      if (lane !== null) {
        healthLaneByKey.set(recipientKey, lane + 1);
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "net_health",
          recipientPresentationKey: recipientKey,
          recipientPublicAgentId: recipientTrajectory.agent_public_agent_id,
          recipient,
          netDelta: delta,
          healthBefore: before,
          healthAfter: after,
          outcome,
          lane,
          paintParts: paintDecision.paintParts,
          presentationSuppressed: !paintDecision.enabled,
          spatial: lane !== null,
          phaseStart: CHOREOGRAPHY_PHASES.v2HealthResolutionStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2CountdownAndRegenStart,
        }),
      );
      continue;
    }
    if (row.kind === "charge_phase_displacement") {
      if (trajectories === null) {
        return null;
      }
      const startAnchor = record(event.start_anchor);
      const endAnchor = record(event.end_anchor);
      const trajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        startAnchor,
        "transition_start",
      );
      const endTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        endAnchor,
        "post_charge",
      );
      const key = identifier(startAnchor?.presentation_key);
      const startWorld = authorizedWorldPoint(trajectory?.transition_start);
      const endWorld = authorizedWorldPoint(trajectory?.successor);
      if (
        trajectory === null ||
        endTrajectory !== trajectory ||
        key === null ||
        startWorld === null ||
        endWorld === null
      ) {
        return null;
      }
      const paintDecision = authorizedPaintDecision(
        { kind: "charge_displacement" },
        visualFilters,
      );
      if (paintDecision === null) {
        return null;
      }
      const start = paintDecision.enabled ? project(startWorld, surface) : null;
      const end = paintDecision.enabled ? project(endWorld, surface) : null;
      if (paintDecision.enabled && (start === null || end === null)) {
        return null;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "charge_displacement",
          sourcePresentationKey: key,
          sourcePublicAgentId: trajectory.agent_public_agent_id,
          targetPresentationKey: null,
          start,
          end,
          pathKind: "charge_phase",
          paintParts: paintDecision.paintParts,
          presentationSuppressed: !paintDecision.enabled,
          spatial: paintDecision.enabled,
          persistent: paintDecision.enabled,
          phaseStart: CHOREOGRAPHY_PHASES.v2ChargeStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2MovementStart,
        }),
      );
      continue;
    }
    if (row.kind === "ordinary_movement_phase_displacement") {
      if (trajectories === null) {
        return null;
      }
      const trajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        event.start_anchor,
        "post_charge",
      );
      const endTrajectory = trajectoryForAuthorizedAnchor(
        trajectories,
        event.end_anchor,
        "successor",
      );
      if (trajectory === null || endTrajectory !== trajectory) {
        return null;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          spatial: false,
        }),
      );
      continue;
    }
    if (row.kind === "action_rejected") {
      const actorAnchor = record(event.actor_anchor);
      const actorTrajectory =
        trajectories === null
          ? null
          : trajectoryForAuthorizedAnchor(
              trajectories,
              actorAnchor,
              "transition_start",
            );
      const actorWorld = authorizedWorldPoint(actorAnchor);
      if (actorWorld === null || (trajectories !== null && actorTrajectory === null)) {
        return null;
      }
      const paintDecision = authorizedPaintDecision(
        { kind: "rejected_action" },
        visualFilters,
      );
      if (paintDecision === null) {
        return null;
      }
      const actor = paintDecision.enabled ? project(actorWorld, surface) : null;
      if (paintDecision.enabled && actor === null) {
        return null;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind: "rejected_action",
          actorPresentationKey: actorAnchor?.presentation_key ?? null,
          actorPublicAgentId:
            actorAnchor?.public_agent_id ??
            event.actor_identity?.public_agent_id ??
            null,
          actor,
          target: null,
          component: event.rejection_component,
          route: null,
          paintParts: paintDecision.paintParts,
          presentationSuppressed: !paintDecision.enabled,
          spatial: paintDecision.enabled,
          phaseStart: CHOREOGRAPHY_PHASES.v2RejectionStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2AbilityStart,
        }),
      );
      continue;
    }
    if (row.kind === "combat_countdown_reset") {
      planned.push(
        Object.freeze({
          ...common,
          kind: "feed_only",
          spatial: false,
        }),
      );
      continue;
    }
    if (row.kind === "respawn_wave_occurred") {
      const waveIdentity = authorizedTeamWaveIdentity(event.team_anchor);
      const paintDecision = authorizedPaintDecision(
        { kind: "semantic_pulse", cueSemantic: row.kind },
        visualFilters,
      );
      if (waveIdentity === null || paintDecision === null) {
        return null;
      }
      const anchor = paintDecision.enabled
        ? teamClockPoint(waveIdentity.teamId, surface)
        : null;
      planned.push(
        Object.freeze({
          ...common,
          kind: "semantic_pulse",
          cueSemantic: row.kind,
          anchor,
          teamIndex: waveIdentity.teamIndex,
          teamId: waveIdentity.teamId,
          teamSide: waveIdentity.teamSide,
          label: waveIdentity.label,
          paintParts: paintDecision.paintParts,
          presentationSuppressed: !paintDecision.enabled,
          spatial: paintDecision.enabled && anchor !== null,
          persistent: paintDecision.enabled,
          phaseStart: CHOREOGRAPHY_PHASES.v2RespawnWaveStart,
          phaseEnd: CHOREOGRAPHY_PHASES.v2RespawnStart,
        }),
      );
      continue;
    }
    if (
      [
        "health_regenerated",
        "cooldown_started",
        "cooldown_ready",
        "agent_died",
        "spawn_shield_expired",
        "agent_respawned",
      ].includes(row.kind)
    ) {
      const anchor = agentAnchor ?? recipientAnchor;
      let presentationAnchor = anchor;
      if (
        row.kind === "health_regenerated" ||
        row.kind === "cooldown_started" ||
        row.kind === "cooldown_ready"
      ) {
        if (trajectories === null) {
          return null;
        }
        const trajectory = trajectoryForAuthorizedAnchor(
          trajectories,
          agentAnchor,
          "transition_start",
        );
        if (trajectory === null) {
          return null;
        }
        presentationAnchor = trajectory.successor;
      } else if (trajectories !== null) {
        const trajectory = trajectoryForAuthorizedAnchor(
          trajectories,
          anchor,
          "successor",
        );
        if (trajectory === null) {
          return null;
        }
        presentationAnchor = trajectory.successor;
      }
      const kind =
        row.kind === "health_regenerated" ? "regeneration" : "semantic_pulse";
      const paintDecision = authorizedPaintDecision(
        { kind, cueSemantic: row.kind },
        visualFilters,
      );
      const presentationWorld = authorizedWorldPoint(presentationAnchor);
      if (paintDecision === null || presentationWorld === null) {
        return null;
      }
      const projectedAnchor = paintDecision.enabled
        ? project(presentationWorld, surface)
        : null;
      if (paintDecision.enabled && projectedAnchor === null) {
        return null;
      }
      planned.push(
        Object.freeze({
          ...common,
          kind,
          cueSemantic: row.kind,
          anchor: projectedAnchor,
          recipient: row.kind === "health_regenerated" ? projectedAnchor : null,
          agentPresentationKey: presentationAnchor?.presentation_key ?? null,
          agentPublicAgentId: presentationAnchor?.public_agent_id ?? null,
          value:
            row.kind === "health_regenerated"
              ? finiteNumber(event.actual_health_regenerated)
              : null,
          paintParts: paintDecision.paintParts,
          presentationSuppressed: !paintDecision.enabled,
          persistent:
            paintDecision.enabled &&
            (row.kind === "agent_died" || row.kind === "agent_respawned"),
          spatial: paintDecision.enabled,
          phaseStart: CHOREOGRAPHY_PHASES.v2CountdownAndRegenStart,
          phaseEnd: CHOREOGRAPHY_PHASES.total,
        }),
      );
      continue;
    }
    planned.push(
      Object.freeze({
        ...common,
        kind: "feed_only",
        spatial: false,
        noncausal: audience === "agent_pov",
      }),
    );
  }
  const paintFiltered = applyAuthorizedVisualFilters(planned, visualFilters);
  const laidOut = layoutCrossPhaseEvents(paintFiltered, surface, sceneByKey);
  if (laidOut === null) {
    return null;
  }
  const scheduled = scheduleChoreography(laidOut);
  const events = scheduled.events;
  const spatialEventCount = events.filter((event) => event.spatial).length;
  const persistentNodeUpperBound = events.reduce(
    (count, event) => count + persistentEventNodeUpperBound(event),
    0,
  );
  return Object.freeze({
    epochKey: JSON.stringify([presentation.session_id, transitionId]),
    authorizationKey: JSON.stringify([
      presentation.session_id,
      presentation.authority_epoch,
      presentation.presentation_kind,
      presentation.recipient_presentation_key,
    ]),
    fingerprint: `${events.length}:${hashText(JSON.stringify(rows.map(({ payload }) => payload)))}`,
    paintKey: visualFilterPaintKey(visualFilters),
    transitionId,
    simulatorStep,
    phases: scheduled.phases,
    events,
    bounds: Object.freeze({
      nodes: Math.min(events.length * 28 + 2, 512),
      animations: Math.min(spatialEventCount * 3 + 2, 512),
      persistentNodes: Math.min(persistentNodeUpperBound, 512),
    }),
  });
}

/**
 * Mirror the painter's maximum retained SVG shape for each planner-authored
 * persistent event kind. Charge owns two roots, five foreground descendants,
 * two route paths, and one backplate for every allocator-authored crossing.
 * Persistent lifecycle and team-wave pulses own at most five nodes each.
 *
 * @param {Record<string, any>} event
 */
function persistentEventNodeUpperBound(event) {
  if (!event.spatial || !event.persistent) {
    return 0;
  }
  if (event.kind === "charge_displacement") {
    const bridgeCount = Array.isArray(event.route?.bridgeGaps)
      ? event.route.bridgeGaps.length
      : 0;
    return 9 + bridgeCount;
  }
  if (event.kind === "semantic_pulse") {
    return 5;
  }
  throw new RangeError(
    `unsupported persistent choreography event kind ${String(event.kind)}.`,
  );
}

/**
 * Build one immutable presentation plan from the already-authorized latest
 * scene/event batch. No simulator fact is derived here.
 *
 * @param {unknown} frame
 * @param {ProjectionSurface | null} surface
 * @param {Record<string, boolean>} visualFilters
 * @returns {Readonly<ChoreographyPlan> | null}
 */
export function buildChoreographyPlan(
  frame,
  surface = null,
  visualFilters = DEFAULT_VISUAL_FILTER_STATE,
) {
  if (isAuthorizedPresentationFrame(frame)) {
    return buildAuthorizedPresentationChoreographyPlan(frame, surface, visualFilters);
  }
  return null;
}
