/**
 * @file Render durable, already-authorized battlefield facts into retained SVG layers.
 * BattlefieldRenderer projects disclosed maps/bodies, draws current selection,
 * status/cooldown/modifier/legality cues and exposes a narrow choreographySurface.
 * The transient controller owns animation children in its two designated layers.
 * Researcher reference data can enrich joined tooltip facts, but cannot add hidden
 * body positions to the scene. The browser renders supplied masks and events; it
 * does not decide simulator legality, advance games or fetch data.
 */
import { canonicalAgentIdentity } from "./agent-identity.js";
import {
  authorizedPresentationResearcherSceneView,
  authorizedPresentationSceneView,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { formatCompactDisplayNumber, formatDisplayNumber } from "./display.js";
import {
  createSpawnShieldView,
  explainAgent,
  explainAura,
  explainCooldown,
  explainLegality,
  explainModifier,
  explainObstacle,
  explainOverflow,
  explainPendingRoute,
  explainPovAgent,
  explainPovOverflow,
  explainPovStatus,
  explainRange,
  explainStatus,
} from "./explanations.js";
import { createSvgIcon } from "./icons.js";
import {
  createViewportTransform,
  layoutRequiredDocks,
  layoutStatusDocks,
  redZoneFloorIntervals,
} from "./layout.js";
import { createRouteGeometry, routeMarkerPose } from "./routes.js";
import { registerTooltipOwner } from "./tooltip.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualPaintPartEnabled,
} from "./visual-filters.js";
import {
  classTokenFromId,
  resolveVisualToken,
  teamTokenFromId,
  ultimateTokenFromClassId,
} from "./vocabulary.js";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
// Corner radius, in CSS pixels, shared by the floor boundary and the Red Zone tint.
const MAP_FLOOR_CORNER_RADIUS = 8;
const STATUS_DOCK_DIMENSIONS = Object.freeze({
  cellWidth: 28,
  cellHeight: 18,
  cellGap: 2,
});
const COOLDOWN_DOCK_DIMENSIONS = Object.freeze({
  cellWidth: 38,
  cellHeight: 18,
  cellGap: 2,
});
const MODIFIER_DOCK_DIMENSIONS = Object.freeze({
  cellWidth: 42,
  cellHeight: 16,
  cellGap: 2,
});
const LEGALITY_DOCK_DIMENSIONS = Object.freeze({
  cellWidth: 30,
  cellHeight: 18,
  cellGap: 3,
});
const DURABLE_VISUAL_PAINT_PARTS = Object.freeze({
  auraFields: Object.freeze({ surface: "durable", kind: "aura_field" }),
  auraModifierBadges: Object.freeze({
    surface: "durable",
    kind: "aura_modifier_badge",
  }),
  durationStatusBadges: Object.freeze({
    surface: "durable",
    kind: "duration_status_badge",
  }),
  povDurationStatusBadges: Object.freeze({
    surface: "durable",
    kind: "pov_duration_status_badge",
  }),
  spawnShield: Object.freeze({ surface: "durable", kind: "spawn_shield" }),
  cooldownBadges: Object.freeze({
    surface: "durable",
    kind: "cooldown_badge",
  }),
  selectionReticle: Object.freeze({
    surface: "durable",
    kind: "selection_reticle",
  }),
  selectedPairLegality: Object.freeze({
    surface: "durable",
    kind: "selected_pair_legality",
  }),
  redZoneFloors: Object.freeze({ surface: "durable", kind: "red_zone_floor" }),
});

export const BATTLEFIELD_LAYER_ORDER = Object.freeze([
  "map",
  "aura",
  "debug-range",
  "pending-route",
  "transient-route",
  "obstacle",
  "body",
  "selection-legality",
  "transient-events",
  "durable-status-modifier",
  "accessible-labels",
]);

/**
 * @typedef {Record<string, any>} JsonRecord
 */
/**
 * @typedef {{
 *   root: SVGElement,
 *   body: SVGElement,
 *   teamRing: SVGElement,
 *   teamMarker: SVGElement,
 *   healthTrack: SVGElement,
 *   health: SVGElement,
 *   classIcon: SVGSVGElement,
 *   classLetter: SVGElement,
 *   deadMark: SVGElement,
 *   shieldRoot: SVGElement | null,
 *   shieldChip: SVGElement | null,
 *   shieldIcon: SVGSVGElement | null,
 *   shieldText: SVGElement | null,
 *   selectionRoot: SVGElement,
 *   controlledHalo: SVGElement,
 *   selectedReticle: SVGElement,
 * }} AgentNodes
 */

/**
 * @typedef {ReturnType<typeof createViewportTransform>} ViewportTransform
 * @typedef {{
 *   layer: SVGElement,
 *   routeLayer: SVGElement,
 *   ownerDocument: Document,
 *   viewportKey: string,
 *   viewportBounds: Rectangle,
 *   protectedRects: ReadonlyArray<ProtectedRegion>,
 *   worldToScreen: (
 *     point: {x: number, y: number} | readonly [number, number],
 *   ) => {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 * }} ChoreographySurface
 * @typedef {{
 *   agent: JsonRecord,
 *   identityKey: number | string,
 *   layoutSlot: number,
 *   globalSlot: number | null,
 *   presentationKey: string | null,
 *   center: {x: number, y: number},
 *   radius: number,
 *   controlled: boolean,
 *   selected: boolean,
 *   statuses: any[],
 * }} ProjectedAgent
 * @typedef {{
 *   left: number,
 *   top: number,
 *   right: number,
 *   bottom: number,
 *   width: number,
 *   height: number,
 * }} Rectangle
 * @typedef {Rectangle & {
 *   layoutKey: string,
 *   bounds: Rectangle,
 *   protectedKind: "body" | "status" | "cooldown" | "modifier" | "legality",
 *   ownerPresentationKey: string | null,
 * }} ProtectedRegion
 * @typedef {{
 *   showAuraFields: boolean,
 *   showAuraModifierBadges: boolean,
 *   showDurationStatusBadges: boolean,
 *   showPovDurationStatusBadges: boolean,
 *   showSpawnShield: boolean,
 *   showCooldownBadges: boolean,
 *   showSelectionReticle: boolean,
 *   showSelectedPairLegality: boolean,
 *   showRedZoneFloors: boolean,
 * }} DurableVisualPolicy
 */

/**
 * Return whether value is a non-null object other than an array.
 *
 * This shape check does not validate the complete scene or information rights.
 *
 * @param {unknown} value
 * @returns {value is JsonRecord}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Package a durable drawing region with an explicit semantic owner.
 *
 * protectedKind names body/status/cooldown/modifier/legality; ownerIdentity gives
 * its stable local key; bounds is a pixel rectangle; ownerPresentationKey may be
 * null for legacy slot ownership. placementIdentity defaults to ownerIdentity and
 * separates multiple docks belonging to one body. Return a frozen region with a
 * copied frozen bounds record and compatibility top-level coordinates. No geometry
 * or ownership is inferred from proximity.
 *
 * @param {"body" | "status" | "cooldown" | "modifier" | "legality"} protectedKind
 * @param {string} ownerIdentity
 * @param {Rectangle} bounds
 * @param {string | null} ownerPresentationKey
 * @param {string} [placementIdentity]
 * @returns {ProtectedRegion}
 */
function choreographyProtectedRegion(
  protectedKind,
  ownerIdentity,
  bounds,
  ownerPresentationKey,
  placementIdentity = ownerIdentity,
) {
  const frozenBounds = Object.freeze({ ...bounds });
  return Object.freeze({
    layoutKey: JSON.stringify([
      "durable",
      protectedKind,
      ownerIdentity,
      placementIdentity,
    ]),
    bounds: frozenBounds,
    ...frozenBounds,
    protectedKind,
    ownerPresentationKey,
  });
}

/**
 * Return value unchanged when it is an array, otherwise a new empty array.
 *
 * No element validation, copying or mutation occurs.
 *
 * @param {unknown} value
 * @returns {any[]}
 */
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * Read token_id, falling back to status_id when nullish.
 *
 * value may be unknown. Return a nonempty string or null; whitespace is not trimmed
 * and token vocabulary is checked by its later authority.
 *
 * @param {unknown} value
 */
function statusTokenId(value) {
  const status = isRecord(value) ? value : {};
  const candidate = status.token_id ?? status.status_id;
  return typeof candidate === "string" && candidate.length > 0 ? candidate : null;
}

/**
 * Read remaining_duration, falling back to duration when nullish.
 *
 * value may be unknown. Return an integer number or null; this helper adds no
 * positivity or configured-duration bound.
 *
 * @param {unknown} value
 */
function statusRemainingDuration(value) {
  const status = isRecord(value) ? value : {};
  const candidate = status.remaining_duration ?? status.duration;
  return Number.isInteger(candidate) ? Number(candidate) : null;
}

/**
 * Compare exact facts while allowing one float32 representation step.
 *
 * local/researcher match through Object.is or when finite numbers equal one
 * another's float32 rounding. Return a Boolean without a general epsilon tolerance.
 *
 * @param {unknown} local @param {unknown} researcher
 */
function sameRecordedPublicValue(local, researcher) {
  if (Object.is(local, researcher)) return true;
  return (
    typeof local === "number" &&
    Number.isFinite(local) &&
    typeof researcher === "number" &&
    Number.isFinite(researcher) &&
    (local === Math.fround(researcher) || Math.fround(local) === researcher)
  );
}

/**
 * Check disclosed local status facts before joining researcher attribution.
 *
 * rawLocal/rawResearcher must be records with matching token and remaining duration.
 * Compare every optional field present locally against its researcher name, including
 * the mechanic_action_component alias. Return a Boolean. Missing local optional
 * fields impose no new requirement; this is a join check, not full schema validation.
 *
 * @param {unknown} rawLocal
 * @param {unknown} rawResearcher
 */
function samePublicStatusFacts(rawLocal, rawResearcher) {
  const local = isRecord(rawLocal) ? rawLocal : null;
  const researcher = isRecord(rawResearcher) ? rawResearcher : null;
  if (
    local === null ||
    researcher === null ||
    statusTokenId(local) !== statusTokenId(researcher) ||
    statusRemainingDuration(local) !== statusRemainingDuration(researcher)
  ) {
    return false;
  }
  const aliases = [
    ["status_channel", "status_channel"],
    ["configured_duration_steps", "configured_duration_steps"],
    ["family", "family"],
    ["source_class_id", "source_class_id"],
    ["source_class_name", "source_class_name"],
    ["source_action_component", "source_action_component"],
    ["mechanic_action_component", "source_action_component"],
    ["magnitude_kind", "magnitude_kind"],
    ["magnitude", "magnitude"],
    ["breaks_on_positive_damage", "breaks_on_positive_damage"],
  ];
  return aliases.every(([localKey, researcherKey]) => {
    if (local[localKey] === undefined) return true;
    return sameRecordedPublicValue(local[localKey], researcher[researcherKey]);
  });
}

/**
 * Resolve local filter state into the nine durable paint switches.
 *
 * state is validated/classified by the visual-filter authority. Return a frozen
 * policy for aura, badges, shield, selection, pair-legality and Red Zone floor
 * rendering. No accepted scene facts or input filter object are changed.
 *
 * @param {unknown} state
 * @returns {Readonly<DurableVisualPolicy>}
 */
function durableVisualPolicy(state) {
  return Object.freeze({
    showAuraFields: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.auraFields,
    ),
    showAuraModifierBadges: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.auraModifierBadges,
    ),
    showDurationStatusBadges: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.durationStatusBadges,
    ),
    showPovDurationStatusBadges: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.povDurationStatusBadges,
    ),
    showSpawnShield: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.spawnShield,
    ),
    showCooldownBadges: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.cooldownBadges,
    ),
    showSelectionReticle: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.selectionReticle,
    ),
    showSelectedPairLegality: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.selectedPairLegality,
    ),
    showRedZoneFloors: isVisualPaintPartEnabled(
      state,
      DURABLE_VISUAL_PAINT_PARTS.redZoneFloors,
    ),
  });
}

/**
 * Return finite numeric value, otherwise fallback (default zero).
 *
 * Numeric strings are not converted. The fallback is used as supplied.
 *
 * @param {unknown} value
 * @param {number} fallback
 */
function finiteNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Read an own enumerable data property without invoking an accessor.
 *
 * value may be unknown and key names the property. Return its stored value, or
 * undefined for missing/nonenumerable/accessor properties, nonobjects or a thrown
 * property-descriptor trap. Proxy traps may run while obtaining the descriptor;
 * errors are caught. This helper performs no mutation.
 *
 * @param {unknown} value
 * @param {string} key
 * @returns {unknown}
 */
function ownEnumerableDataValue(value, key) {
  try {
    if (typeof value !== "object" || value === null) return undefined;
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (
      descriptor === undefined ||
      !descriptor.enumerable ||
      !Object.hasOwn(descriptor, "value")
    ) {
      return undefined;
    }
    return descriptor.value;
  } catch {
    return undefined;
  }
}

/**
 * Format the public identity already carried by agent.
 *
 * Return canonicalAgentIdentity(agent).publicIdentity. Internal layout/global slots
 * are never promoted to public display IDs by this helper.
 *
 * @param {unknown} agent
 */
function agentIdentity(agent) {
  return canonicalAgentIdentity(agent).publicIdentity;
}

/**
 * Read the authorized display scene from a recognized presentation frame.
 *
 * frame may be unknown; localInspectedPresentationKey defaults to undefined and is
 * forwarded for local inspection selection. Return the adapter view or null for an
 * unrecognized frame. The adapter owns allowed inspection and disclosure rules.
 *
 * @param {unknown} frame
 * @param {string | null | undefined} [localInspectedPresentationKey]
 * @returns {JsonRecord | null}
 */
function frameScene(frame, localInspectedPresentationKey = undefined) {
  return isAuthorizedPresentationFrame(frame)
    ? authorizedPresentationSceneView(frame, localInspectedPresentationKey)
    : null;
}

/**
 * Choose a stable retained-node key from an accepted display row.
 *
 * agent.display_key takes precedence when nonempty; otherwise use an integer
 * global_slot, or null. This is a DOM join key, not a public label or new slot fact.
 *
 * @param {JsonRecord} agent
 * @returns {number | string | null}
 */
function agentDisplayIdentity(agent) {
  if (typeof agent.display_key === "string" && agent.display_key.length > 0) {
    return agent.display_key;
  }
  return Number.isInteger(agent.global_slot) ? Number(agent.global_slot) : null;
}

/**
 * Return the integer global_slot on agent, otherwise fallback.
 *
 * fallback is a renderer-local layout index. It may position a POV body but must
 * never be published as an undisclosed simulator global slot.
 *
 * @param {JsonRecord} agent @param {number} fallback
 */
function agentLayoutSlot(agent, fallback) {
  return Number.isInteger(agent.global_slot) ? Number(agent.global_slot) : fallback;
}

/**
 * Write only the identity actually present on record into element metadata.
 *
 * Prefer a nonempty presentation_key and remove data-slot; otherwise use integer
 * global_slot and remove data-presentation-key. With neither, do nothing. Return
 * undefined. The caller supplies an accepted display row.
 *
 * @param {SVGElement} element
 * @param {JsonRecord} record
 */
function setDisplayIdentityData(element, record) {
  if (
    typeof record.presentation_key === "string" &&
    record.presentation_key.length > 0
  ) {
    element.dataset.presentationKey = record.presentation_key;
    element.removeAttribute("data-slot");
  } else if (Number.isInteger(record.global_slot)) {
    element.dataset.slot = String(record.global_slot);
    element.removeAttribute("data-presentation-key");
  }
}

/**
 * Return identity attributes from record without inventing a slot.
 *
 * A string presentation_key takes precedence (including empty strings); otherwise
 * an integer global_slot gives data-slot, and neither gives an empty object. The
 * input is unchanged; upstream normalization owns identity validity.
 *
 * @param {JsonRecord} record
 */
function displayIdentityAttributes(record) {
  return typeof record.presentation_key === "string"
    ? { "data-presentation-key": record.presentation_key }
    : Number.isInteger(record.global_slot)
      ? { "data-slot": record.global_slot }
      : {};
}

/**
 * Copy only the accepted public ID and optional string presentation key.
 *
 * agent supplies those facts. Return a new tooltip input with no layout/global slot,
 * so a renderer-local index cannot become an Agent POV fact.
 *
 * @param {JsonRecord} agent
 */
function displayIdentityRecord(agent) {
  return {
    ...(typeof agent.presentation_key === "string"
      ? { presentation_key: agent.presentation_key }
      : {}),
    public_agent_id: agent.public_agent_id,
  };
}

/**
 * Project a coordinate array through the current viewport transform.
 *
 * point is expected to be a world [x,y] array; absent/nonarray coordinates or
 * nonfinite components fall back to zero. transform owns world-to-screen conversion.
 * Return its projected point; this tolerant renderer helper is not source validation.
 *
 * @param {unknown} point
 * @param {ViewportTransform} transform
 */
function screenPoint(point, transform) {
  const values = Array.isArray(point) ? point : [0, 0];
  return transform.worldToScreen([finiteNumber(values[0]), finiteNumber(values[1])]);
}

/**
 * Create a detached SVG node in the browser document.
 *
 * tagName selects its SVG type; attributes defaults to an empty map and is applied
 * by setAttributes. Return the node. Browser DOM errors propagate.
 *
 * @param {string} tagName
 * @param {Record<string, unknown>} attributes
 * @returns {SVGElement}
 */
function svgElement(tagName, attributes = {}) {
  const element = document.createElementNS(SVG_NAMESPACE, tagName);
  setAttributes(element, attributes);
  return element;
}

/**
 * Apply attributes to an existing SVG element.
 *
 * Remove null/undefined entries and stringify every other value, including false.
 * Return undefined. This mutates DOM; callers own safe attribute names and values.
 *
 * @param {SVGElement} element
 * @param {Record<string, unknown>} attributes
 */
function setAttributes(element, attributes) {
  for (const [name, value] of Object.entries(attributes)) {
    if (value === null || value === undefined) {
      element.removeAttribute(name);
    } else {
      element.setAttribute(name, String(value));
    }
  }
}

/**
 * Create a detached named SVG group for the battlefield layer stack.
 *
 * name sets data-layer; attributes defaults to an empty map and can override it.
 * Return the group; the constructor owns insertion order.
 *
 * @param {string} name
 * @param {Record<string, unknown>} attributes
 */
function createLayer(name, attributes = {}) {
  return svgElement("g", {
    "data-layer": name,
    ...attributes,
  });
}

/**
 * Return four corner strokes around a selected target body.
 *
 * centerX/centerY and outerRadius are caller-validated screen coordinates/lengths.
 * The open corners avoid covering the separate controlled-actor halo. No collision
 * or target legality is inferred.
 *
 * @param {number} centerX
 * @param {number} centerY
 * @param {number} outerRadius
 */
function targetReticlePath(centerX, centerY, outerRadius) {
  const arm = Math.max(outerRadius * 0.38, 0.12);
  const left = centerX - outerRadius;
  const right = centerX + outerRadius;
  const top = centerY - outerRadius;
  const bottom = centerY + outerRadius;
  return [
    `M ${left + arm} ${top} H ${left} V ${top + arm}`,
    `M ${right - arm} ${top} H ${right} V ${top + arm}`,
    `M ${left} ${bottom - arm} V ${bottom} H ${left + arm}`,
    `M ${right} ${bottom - arm} V ${bottom} H ${right - arm}`,
  ].join(" ");
}

/**
 * Replace one layer with authorized aura/range circles and optional hit regions.
 *
 * layer is the target SVG group; records contains accepted center/radius rows;
 * className and tokenAttribute identify their display grammar; transform maps world
 * to screen. classByIdentity defaults to an empty map. explain defaults to null;
 * when supplied it builds a tooltip or returns null to omit a circle. withStrokeHitRegion
 * defaults to false and adds a separate transparent circular hit surface when true.
 * Skip nonrecord/nonpositive-radius rows. Return undefined after replacing children;
 * no game range or visibility is computed.
 *
 * @param {SVGElement} layer
 * @param {any[]} records
 * @param {string} className
 * @param {"kind" | "token_id"} tokenAttribute
 * @param {ViewportTransform} transform
 * @param {ReadonlyMap<number | string, string>} [classByIdentity]
 * @param {((record: JsonRecord) => Record<string, any> | null) | null} [explain]
 * @param {boolean} [withStrokeHitRegion]
 */
function renderCircleLayer(
  layer,
  records,
  className,
  tokenAttribute,
  transform,
  classByIdentity = new Map(),
  explain = null,
  withStrokeHitRegion = false,
) {
  const circles = [];
  for (const record of records) {
    if (!isRecord(record)) {
      continue;
    }
    const center = screenPoint(record.center, transform);
    const radius = transform.worldLengthToScreen(finiteNumber(record.radius));
    if (radius <= 0) {
      continue;
    }
    const circle = svgElement("circle", {
      class: className,
      cx: center.x,
      cy: center.y,
      r: radius,
      role: explain === null ? null : "img",
      tabindex: explain === null ? null : "0",
    });
    if (typeof record[tokenAttribute] === "string") {
      if (tokenAttribute === "kind") {
        circle.dataset.kind = record[tokenAttribute];
      } else {
        circle.dataset.token = record[tokenAttribute];
      }
    }
    setDisplayIdentityData(circle, record);
    const identity =
      typeof record.presentation_key === "string" ? record.presentation_key : null;
    const classKey = identity === null ? undefined : classByIdentity.get(identity);
    if (classKey) {
      circle.dataset.class = classKey;
    }
    if (explain === null) {
      circles.push(circle);
      continue;
    }
    if (!withStrokeHitRegion) {
      const descriptor = explain(record);
      if (descriptor === null) {
        continue;
      }
      circle.setAttribute("aria-label", descriptor.title);
      registerTooltipOwner(circle, descriptor);
      circles.push(circle);
      continue;
    }
    const owner = svgElement("g", {
      class: `${className}-owner`,
    });
    const hitRegion = svgElement("circle", {
      class: `${className}-hit`,
      cx: center.x,
      cy: center.y,
      r: radius,
      "aria-hidden": "true",
    });
    if (typeof record[tokenAttribute] === "string") {
      hitRegion.dataset[tokenAttribute === "kind" ? "kind" : "token"] =
        record[tokenAttribute];
    }
    setDisplayIdentityData(hitRegion, record);
    owner.append(circle, hitRegion);
    const descriptor = explain(record);
    if (descriptor === null) {
      continue;
    }
    owner.setAttribute("role", "img");
    owner.setAttribute("tabindex", "0");
    owner.setAttribute("aria-label", descriptor.title);
    registerTooltipOwner(owner, descriptor);
    circles.push(owner);
  }
  layer.replaceChildren(...circles);
}

/**
 * Own durable SVG layers for an authoritative debugger or replay frame.
 *
 * The constructor replaces the supplied SVG's children with the fixed layer stack.
 * Later render calls update durable layers while retaining the SVG root and both
 * transient layers for CombatChoreographer. Bodies keep stable node identities where
 * possible. The class owns projection and drawing, not simulator rules or fetching.
 */
export class BattlefieldRenderer {
  /**
   * Install the battlefield layers into caller-supplied browser elements.
   *
   * The elements object requires battlefield (SVGSVGElement) and empty (HTMLElement).
   * Replace battlefield children, initialize retained-node/authority maps and schedule
   * numeric dock measurement after battlefield.ownerDocument.fonts.ready. The caller
   * owns the elements and coordinates transient cleanup. A browser DOM/font API is
   * required; unsupported element operations propagate their errors.
   *
   * @param {{
   *   battlefield: SVGSVGElement,
   *   empty: HTMLElement,
   * }} elements
   */
  constructor({ battlefield, empty }) {
    this.battlefield = battlefield;
    this.empty = empty;
    /** @type {ViewportTransform | null} */
    this.transform = null;
    /** @type {ReadonlyArray<ProtectedRegion>} */
    this.choreographyProtectedRects = Object.freeze([]);
    /** @type {Readonly<{
     *   base: ReadonlyArray<ProtectedRegion>,
     *   legality: ReadonlyArray<ProtectedRegion>,
     *   status: ReadonlyArray<ProtectedRegion>,
     * }>} */
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze([]),
      legality: Object.freeze([]),
      status: Object.freeze([]),
    });
    /** @type {ReadonlyMap<string, JsonRecord>} */
    this.agentByPresentationKey = new Map();
    /** @type {ReadonlyMap<number, JsonRecord>} */
    this.agentByLayoutSlot = new Map();
    /** @type {ReadonlyMap<string, JsonRecord>} */
    this.researcherAgentByPublicId = new Map();

    const map = createLayer("map", { "aria-hidden": "true" });
    const aura = createLayer("aura", { "aria-hidden": "true" });
    const debugRange = createLayer("debug-range", {
      "aria-label": "Authorized ability and effect ranges",
    });
    this.rangeCues = createLayer("range-cues", { "aria-hidden": "true" });
    debugRange.append(this.rangeCues);
    const pendingRoute = createLayer("pending-route", { "aria-hidden": "true" });
    const transientRoute = createLayer("transient-route", {
      "aria-label": "Authorized combat event routes",
    });
    const obstacle = createLayer("obstacle", { "aria-label": "Map obstacles" });
    const body = createLayer("body", { "aria-label": "Authorized agents" });
    const selectionLegality = createLayer("selection-legality", {
      "aria-label": "Selection and exact actor-owned legality",
    });
    this.selectionCues = createLayer("selection-cues", {
      "aria-hidden": "true",
    });
    this.legalityCues = createLayer("legality-cues");
    selectionLegality.append(this.selectionCues, this.legalityCues);
    const durableStatusModifier = createLayer("durable-status-modifier", {
      "aria-label": "Durable status, cooldown, and modifier cues",
    });
    const transientEvents = createLayer("transient-events", {
      "aria-label": "Authorized combat event summaries",
    });
    const accessibleLabels = createLayer("accessible-labels");

    this.layers = {
      map,
      aura,
      debugRange,
      pendingRoute,
      transientRoute,
      obstacle,
      body,
      selectionLegality,
      durableStatusModifier,
      transientEvents,
      accessibleLabels,
    };

    /** @type {Map<number | string, AgentNodes>} */
    this.agentNodes = new Map();
    /** @type {Map<string, SVGElement>} */
    this.observedBodyNodes = new Map();

    battlefield.replaceChildren(
      map,
      aura,
      debugRange,
      pendingRoute,
      transientRoute,
      obstacle,
      body,
      selectionLegality,
      transientEvents,
      durableStatusModifier,
      accessibleLabels,
    );
    void battlefield.ownerDocument.fonts.ready.then(() => {
      this.#resolveNumericDockCellContent();
    });
  }

  /**
   * Paint the durable facts of a normalized authorized frame.
   *
   * frame may be unknown. options.offline defaults to false and changes only empty
   * state text; showRanges defaults to true; localInspectedPresentationKey defaults to
   * undefined; visualFilterState defaults to DEFAULT_VISUAL_FILTER_STATE. Optional
   * renderPolicy is copied to SVG metadata without changing game time.
   *
   * Return true after rendering a recognized researcher/agent_pov scene with a positive
   * map size. Otherwise clear durable state, show an unavailable message and return
   * false; transient layers remain controller-owned. Use the element's measured viewport
   * or a map-based fallback, with 24 px padding. Project disclosed rows, then allocate
   * status/cooldown/modifier/legality docks and publish protected regions. Tooltips may
   * use joined researcher references; bodies use only the authorized display scene.
   * This mutates DOM and renderer caches. Layout, filter and browser errors propagate;
   * no network request or game command is sent.
   *
   * @param {unknown} frame
   * @param {{
   *   offline?: boolean,
   *   showRanges?: boolean,
   *   localInspectedPresentationKey?: string | null,
   *   visualFilterState?: Readonly<Record<string, boolean>>,
   *   renderPolicy?: "live_once" | "replay_animated" | "replay_static",
   * }} [options]
   * @returns {boolean} Whether the frame contained a paintable scene.
   */
  render(frame, options = {}) {
    const visualFilterState =
      options.visualFilterState === undefined
        ? DEFAULT_VISUAL_FILTER_STATE
        : options.visualFilterState;
    const visualPolicy = durableVisualPolicy(visualFilterState);
    const scene = frameScene(frame, options.localInspectedPresentationKey);
    const researcherScene = isAuthorizedPresentationFrame(frame)
      ? authorizedPresentationResearcherSceneView(frame)
      : scene;
    const map = isRecord(scene?.map) ? scene.map : null;
    const width = finiteNumber(map?.width);
    const height = finiteNumber(map?.height);

    if (
      !scene ||
      !map ||
      (scene.audience !== "researcher" && scene.audience !== "agent_pov") ||
      width <= 0 ||
      height <= 0
    ) {
      this.#clearDurableScene();
      this.battlefield.removeAttribute("viewBox");
      this.battlefield.removeAttribute("data-preset");
      this.battlefield.removeAttribute("data-audience");
      this.battlefield.removeAttribute("data-render-policy");
      this.battlefield.setAttribute(
        "aria-label",
        "Battlefield unavailable; no authorized scene was returned.",
      );
      this.empty.hidden = false;
      this.empty.textContent = options.offline
        ? "The local debugger service is unavailable. Commands are not being retried."
        : "No authorized battlefield scene was returned.";
      this.transform = null;
      this.researcherAgentByPublicId = new Map();
      return false;
    }

    const viewportWidth =
      this.battlefield.clientWidth > 0
        ? this.battlefield.clientWidth
        : Math.max(width * 40, 320);
    const viewportHeight =
      this.battlefield.clientHeight > 0
        ? this.battlefield.clientHeight
        : Math.max(height * 40, 240);
    const transform = createViewportTransform({
      worldWidth: width,
      worldHeight: height,
      viewportWidth,
      viewportHeight,
      padding: 24,
    });
    this.transform = transform;
    this.battlefield.setAttribute("viewBox", `0 0 ${viewportWidth} ${viewportHeight}`);
    this.battlefield.dataset.preset = "analysis";
    this.battlefield.dataset.audience = scene.audience;
    if (options.renderPolicy === undefined) {
      this.battlefield.removeAttribute("data-render-policy");
    } else {
      this.battlefield.dataset.renderPolicy = options.renderPolicy;
    }
    const audienceLabel = scene.audience === "researcher" ? "Oracle View" : "Agent POV";
    this.battlefield.setAttribute(
      "aria-label",
      `${audienceLabel} battlefield, ${width} by ${height}.`,
    );
    this.empty.hidden = true;

    this.#renderMap(
      transform,
      width,
      height,
      isRecord(frame) && frame.product_kind === "combat_debugger",
      visualPolicy.showRedZoneFloors && isRecord(map?.red_zone)
        ? /** @type {Readonly<{team_a_x_range: number[], team_b_x_range: number[]}>} */ (
            map.red_zone
          )
        : null,
    );
    const classByIdentity = new Map(
      asArray(scene.agents)
        .filter(
          (agent) => isRecord(agent) && typeof agent.presentation_key === "string",
        )
        .map((agent) => [
          String(agent.presentation_key),
          classTokenFromId(agent.class_id).cssKey,
        ]),
    );
    this.agentByPresentationKey = new Map(
      asArray(scene.agents)
        .filter(
          (agent) => isRecord(agent) && typeof agent.presentation_key === "string",
        )
        .map((agent) => [String(agent.presentation_key), agent]),
    );
    this.researcherAgentByPublicId = new Map(
      asArray(researcherScene?.agents)
        .filter((agent) => isRecord(agent) && typeof agent.public_agent_id === "string")
        .map((agent) => [String(agent.public_agent_id), agent]),
    );
    renderCircleLayer(
      this.layers.aura,
      visualPolicy.showAuraFields
        ? asArray(scene.aura_fields).filter(
            (field) => isRecord(field) && field.source_alive === true,
          )
        : [],
      "aura-field",
      "token_id",
      transform,
      new Map(),
      (record) => {
        const sourcePublicAgentId = ownEnumerableDataValue(
          record,
          "source_public_agent_id",
        );
        const sourceAgent =
          typeof sourcePublicAgentId === "string"
            ? (this.researcherAgentByPublicId.get(sourcePublicAgentId) ?? null)
            : null;
        const explanationRecord =
          sourceAgent === null
            ? record
            : {
                ...record,
                source_presentation_key: sourceAgent.presentation_key,
                source_public_agent_id: sourceAgent.public_agent_id,
              };
        return explainAura(explanationRecord, sourceAgent, scene.audience);
      },
    );
    renderCircleLayer(
      this.rangeCues,
      options.showRanges === false ? [] : asArray(scene.ranges),
      "range-ring",
      "kind",
      transform,
      classByIdentity,
      (record) => {
        const owner =
          (typeof record.presentation_key === "string"
            ? this.agentByPresentationKey.get(record.presentation_key)
            : null) ?? null;
        return explainRange(record, owner);
      },
      true,
    );
    this.#renderPendingRoute(scene, transform);
    this.#renderObstacles(map, transform);
    const projectedAgents = this.#renderAgents(scene, transform, visualPolicy);
    this.agentByLayoutSlot = new Map(
      projectedAgents.map((projected) => [projected.layoutSlot, projected.agent]),
    );
    this.#renderObservedBodies(
      scene,
      transform,
      visualPolicy.showPovDurationStatusBadges,
    );
    this.#renderStatusDocks(scene, projectedAgents, transform, {
      showLegality: visualPolicy.showSelectedPairLegality,
      showModifiers: visualPolicy.showAuraModifierBadges,
      showStatuses: visualPolicy.showDurationStatusBadges,
      showCooldowns: visualPolicy.showCooldownBadges,
      audience: scene.audience,
    });
    this.#refreshChoreographyProtectedRects();

    this.layers.accessibleLabels.replaceChildren();
    return true;
  }

  /**
   * Refresh protected regions for legacy callers and return false.
   *
   * _active is retained for compatibility and is ignored. Durable facts are not
   * suppressed by transient activity; the shared allocator reserves their regions.
   *
   * @param {boolean} _active
   * @returns {boolean} Whether the effective compact-active state changed.
   */
  setCompactActiveCombat(_active) {
    this.#refreshChoreographyProtectedRects();
    return false;
  }

  /**
   * Convert an SVG-local pointer point into debugger world_x/world_y.
   *
   * point must have finite x/y and a current transform must exist; otherwise return
   * null. Return a new coordinate record without clamping it to the map. This does
   * not issue a command or validate whether that position is actionable.
   *
   * @param {{x: number, y: number}} point
   * @returns {{world_x: number, world_y: number} | null}
   */
  toWorldPoint(point) {
    if (!this.transform || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return null;
    }
    const world = this.transform.screenToWorld(point);
    return {
      world_x: world.x,
      world_y: world.y,
    };
  }

  /**
   * Return the narrow layer/projection contract for transient choreography.
   *
   * Return null before a valid durable scene exists. Otherwise return a frozen surface
   * with foreground/route layers, ownerDocument, map bounds, immutable protected-region
   * snapshot and a viewportKey that includes those regions. Projection functions close
   * over this render's transform. Callers may append only their owned children to
   * layer/routeLayer and must project authorized event anchors instead of borrowing
   * current body positions. Other layers remain owned by this renderer.
   *
   * @returns {Readonly<ChoreographySurface> | null}
   */
  choreographySurface() {
    const transform = this.transform;
    if (!transform) {
      return null;
    }
    const protectedLayoutKey = this.choreographyProtectedRects
      .map((region) =>
        [
          region.bounds.left,
          region.bounds.top,
          region.bounds.right,
          region.bounds.bottom,
        ]
          .map((value) => Number(value).toFixed(3))
          .concat(region.layoutKey)
          .join(","),
      )
      .join(";");
    /** @type {ChoreographySurface} */
    const surface = {
      layer: this.layers.transientEvents,
      routeLayer: this.layers.transientRoute,
      ownerDocument: this.battlefield.ownerDocument,
      viewportKey: [
        transform.worldWidth,
        transform.worldHeight,
        transform.viewportBounds.width,
        transform.viewportBounds.height,
        transform.mapBounds.left,
        transform.mapBounds.top,
        transform.mapBounds.right,
        transform.mapBounds.bottom,
        protectedLayoutKey,
      ].join(":"),
      viewportBounds: Object.freeze({ ...transform.mapBounds }),
      protectedRects: this.choreographyProtectedRects,
      /**
       * Project a supplied authorized world point with this surface's captured transform.
       *
       * point is an x/y object or coordinate pair. Return screen coordinates; no body
       * lookup, authority expansion or DOM update occurs.
       */
      worldToScreen: (point) => transform.worldToScreen(point),
      /**
       * Convert a supplied world length with this surface's captured scale.
       *
       * length uses the transform's numeric contract. Return a screen-pixel length without
       * changing renderer state.
       */
      worldLengthToScreen: (length) => transform.worldLengthToScreen(length),
    };
    return Object.freeze(surface);
  }

  /**
   * Remove durable drawing children and reset retained scene/projection caches.
   *
   * Clear body/selection/status/range/map/obstacle/pending-route metadata and owned
   * node maps. Keep the SVG root, fixed layers and transient route/event children;
   * the animation controller must clear its own authority. Return undefined.
   */
  #clearDurableScene() {
    this.layers.map.replaceChildren();
    this.layers.aura.replaceChildren();
    this.rangeCues.replaceChildren();
    this.layers.pendingRoute.replaceChildren();
    this.layers.obstacle.replaceChildren();
    this.layers.body.replaceChildren();
    this.selectionCues.replaceChildren();
    this.legalityCues.replaceChildren();
    this.layers.durableStatusModifier.replaceChildren();
    this.layers.durableStatusModifier.removeAttribute("data-suppressed-status-slots");
    this.layers.durableStatusModifier.removeAttribute("data-suppressed-cooldown-slots");
    this.layers.durableStatusModifier.removeAttribute("data-suppressed-modifier-slots");
    this.layers.durableStatusModifier.removeAttribute("data-compacted-required-docks");
    this.layers.durableStatusModifier.removeAttribute(
      "data-suppressed-status-presentation-keys",
    );
    this.layers.durableStatusModifier.removeAttribute(
      "data-suppressed-cooldown-presentation-keys",
    );
    this.layers.durableStatusModifier.removeAttribute(
      "data-suppressed-modifier-presentation-keys",
    );
    this.layers.durableStatusModifier.removeAttribute(
      "data-compacted-required-presentations",
    );
    this.layers.accessibleLabels.replaceChildren();
    this.agentNodes.clear();
    this.observedBodyNodes.clear();
    this.agentByPresentationKey = new Map();
    this.agentByLayoutSlot = new Map();
    this.researcherAgentByPublicId = new Map();
    this.choreographyProtectedRects = Object.freeze([]);
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze([]),
      legality: Object.freeze([]),
      status: Object.freeze([]),
    });
    this.transform = null;
  }

  /**
   * Publish a frozen copy of current body/dock/legality protected regions.
   *
   * Read base, status and legality groups in that order and copy nested bounds.
   * Update choreographyProtectedRects; return undefined without recomputing layout.
   */
  #refreshChoreographyProtectedRects() {
    const groups = this.choreographyProtectedRectGroups;
    this.choreographyProtectedRects = Object.freeze(
      [...groups.base, ...groups.status, ...groups.legality].map((region) =>
        Object.freeze({
          ...region,
          bounds: Object.freeze({ ...region.bounds }),
        }),
      ),
    );
  }

  /**
   * Replace the map boundary, the optional Red Zone floor tint and the optional grid.
   *
   * transform supplies screen bounds; width/height are world dimensions. showUnitGrid
   * adds internal lines only when both dimensions are integers. redZone is the
   * recorded, already-validated `map.red_zone` record when the Red Zone Floors
   * filter is on, otherwise null (filter off, depth 0, or a map recorded before the
   * rule). Each strip from redZoneFloorIntervals becomes a nested `<svg
   * class="red-zone-floor-clip">` window holding one floor-sized `<rect
   * class="red-zone-floor">` with the floor's corner radius, so the tint follows the
   * rounded floor corners without `<clipPath>` ids (PNG export strips ids). Paint
   * order is boundary, tint, grid; the whole map layer stays below obstacles, bodies
   * and effects. Return undefined; this draws a display aid and does not generate
   * map geometry, obstacles, scores or Red Zone facts.
   *
   * @param {ViewportTransform} transform
   * @param {number} width
   * @param {number} height
   * @param {boolean} showUnitGrid
   * @param {Readonly<{team_a_x_range: readonly number[], team_b_x_range: readonly number[]}> | null} redZone
   */
  #renderMap(transform, width, height, showUnitGrid, redZone) {
    const bounds = transform.mapBounds;
    const redZoneFloors = redZoneFloorIntervals(width, redZone).map((interval) => {
      const left = screenPoint([interval.start, 0], transform).x;
      const right = screenPoint([interval.end, 0], transform).x;
      const clipLeft = Math.min(left, right);
      const clip = svgElement("svg", {
        class: "red-zone-floor-clip",
        x: clipLeft,
        y: bounds.top,
        width: Math.abs(right - left),
        height: bounds.height,
      });
      clip.append(
        svgElement("rect", {
          class: "red-zone-floor",
          x: bounds.left - clipLeft,
          y: 0,
          width: bounds.width,
          height: bounds.height,
          rx: MAP_FLOOR_CORNER_RADIUS,
        }),
      );
      return clip;
    });
    const gridLines = [];
    if (showUnitGrid && Number.isInteger(width) && Number.isInteger(height)) {
      for (let x = 1; x < width; x += 1) {
        const start = screenPoint([x, 0], transform);
        const end = screenPoint([x, height], transform);
        gridLines.push(
          svgElement("line", {
            class: "map-grid-line map-grid-line--vertical",
            x1: start.x,
            y1: start.y,
            x2: end.x,
            y2: end.y,
          }),
        );
      }
      for (let y = 1; y < height; y += 1) {
        const start = screenPoint([0, y], transform);
        const end = screenPoint([width, y], transform);
        gridLines.push(
          svgElement("line", {
            class: "map-grid-line map-grid-line--horizontal",
            x1: start.x,
            y1: start.y,
            x2: end.x,
            y2: end.y,
          }),
        );
      }
    }
    this.layers.map.replaceChildren(
      svgElement("rect", {
        class: "map-boundary",
        x: bounds.left,
        y: bounds.top,
        width: bounds.width,
        height: bounds.height,
        rx: MAP_FLOOR_CORNER_RADIUS,
      }),
      ...redZoneFloors,
      ...gridLines,
    );
  }

  /**
   * Draw the supplied inspected/draft route without treating it as an event.
   *
   * scene.pending_route owns disclosed source/target anchors, lane and legality.
   * transform converts them to screen space; absent route clears the layer. Create a
   * presentation curve with a directional marker and tooltip joined to matching scene
   * identities. Return undefined. No target position, legality or accepted transition
   * is inferred from drawing geometry.
   *
   * @param {JsonRecord} scene
   * @param {ViewportTransform} transform
   */
  #renderPendingRoute(scene, transform) {
    const route = isRecord(scene.pending_route) ? scene.pending_route : null;
    if (!route) {
      this.layers.pendingRoute.replaceChildren();
      return;
    }
    const source = screenPoint(route.source_anchor, transform);
    const target = screenPoint(route.target_anchor, transform);
    const routeIdentity = `${route.source_public_agent_id ?? "unknown"}:${route.target_public_agent_id ?? "unknown"}`;
    const geometry = createRouteGeometry(
      {
        eventId: `pending:${routeIdentity}:${route.lane ?? "unknown"}`,
        source,
        target,
        sourceRadius: transform.worldLengthToScreen(finiteNumber(route.source_radius)),
        targetRadius: transform.worldLengthToScreen(finiteNumber(route.target_radius)),
        offset: route.lane === 1 ? 12 : -12,
      },
      { viewportBounds: transform.mapBounds },
    );
    const path = svgElement("path", {
      class: "pending-route",
      d: geometry.path,
    });
    const hitPath = svgElement("path", {
      class: "pending-route-hit",
      d: geometry.path,
      role: "img",
      tabindex: "0",
    });
    const marker = routeMarkerPose(geometry, 1);
    const arrow = svgElement("path", {
      class: "pending-route-arrow",
      d: "M -13 -6 L 0 0 L -13 6 L -9 0 Z",
      transform: `translate(${marker.x} ${marker.y}) rotate(${marker.degrees})`,
      "aria-hidden": "true",
    });
    path.dataset.lane = String(route.lane ?? 0);
    path.dataset.legal = String(Boolean(route.legal));
    path.dataset.sourceAgentId = String(route.source_public_agent_id ?? "");
    path.dataset.targetAgentId = String(route.target_public_agent_id ?? "");
    if (Number.isInteger(route.source_global_slot)) {
      path.dataset.sourceSlot = String(route.source_global_slot);
    }
    if (Number.isInteger(route.target_global_slot)) {
      path.dataset.targetSlot = String(route.target_global_slot);
    }
    if (typeof route.source_presentation_key === "string") {
      path.dataset.sourcePresentationKey = route.source_presentation_key;
    }
    if (typeof route.target_presentation_key === "string") {
      path.dataset.targetPresentationKey = route.target_presentation_key;
    }
    path.dataset.routeKind = geometry.kind;
    arrow.dataset.lane = String(route.lane ?? 0);
    arrow.dataset.legal = String(Boolean(route.legal));
    arrow.dataset.sourceAgentId = String(route.source_public_agent_id ?? "");
    arrow.dataset.targetAgentId = String(route.target_public_agent_id ?? "");
    const sourceAgent = asArray(scene.agents).find(
      (agent) =>
        isRecord(agent) &&
        agent.presentation_key === route.source_presentation_key &&
        agent.public_agent_id === route.source_public_agent_id,
    );
    const recipientAgent = asArray(scene.agents).find(
      (agent) =>
        isRecord(agent) &&
        agent.presentation_key === route.target_presentation_key &&
        agent.public_agent_id === route.target_public_agent_id,
    );
    const routeDescriptor = explainPendingRoute(route, {
      sourceAgent,
      recipientAgent,
    });
    hitPath.setAttribute("aria-label", routeDescriptor.title);
    registerTooltipOwner(hitPath, routeDescriptor);
    this.layers.pendingRoute.replaceChildren(path, arrow, hitPath);
  }

  /**
   * Replace the obstacle layer using disclosed map geometry.
   *
   * map.obstacles supplies pillar circles or oriented wall rectangles; transform
   * projects coordinates/lengths and flips wall angle for screen y. Ignore other roots
   * or kinds. Register obstacle tooltips and return undefined; no collision is solved.
   *
   * @param {JsonRecord} map
   * @param {ViewportTransform} transform
   */
  #renderObstacles(map, transform) {
    const obstacles = [];
    for (const obstacle of asArray(map.obstacles)) {
      if (!isRecord(obstacle)) {
        continue;
      }
      const center = screenPoint(obstacle.center, transform);
      if (obstacle.kind === "pillar") {
        const node = svgElement("circle", {
          class: "obstacle",
          cx: center.x,
          cy: center.y,
          r: transform.worldLengthToScreen(finiteNumber(obstacle.radius)),
          role: "img",
          tabindex: "0",
          "aria-label": `Pillar ${obstacle.obstacle_id ?? ""}`,
        });
        registerTooltipOwner(node, explainObstacle(obstacle));
        obstacles.push(node);
      } else if (obstacle.kind === "wall") {
        const wallWidth = transform.worldLengthToScreen(finiteNumber(obstacle.width));
        const wallHeight = transform.worldLengthToScreen(finiteNumber(obstacle.height));
        const thetaDegrees = (-finiteNumber(obstacle.theta) * 180) / Math.PI;
        const node = svgElement("rect", {
          class: "obstacle",
          x: center.x - wallWidth / 2,
          y: center.y - wallHeight / 2,
          width: wallWidth,
          height: wallHeight,
          transform: `rotate(${thetaDegrees} ${center.x} ${center.y})`,
          role: "img",
          tabindex: "0",
          "aria-label": `Wall ${obstacle.obstacle_id ?? ""}`,
        });
        registerTooltipOwner(node, explainObstacle(obstacle));
        obstacles.push(node);
      }
    }
    this.layers.obstacle.replaceChildren(...obstacles);
  }

  /**
   * Update retained body nodes and return their projected layout records.
   *
   * scene supplies accepted agents/selection; transform projects positions/radii;
   * visualPolicy controls durable pieces. Remove missing identities, reuse/create
   * remaining nodes, register audience-appropriate tooltips and reorder DOM to scene
   * order. Return ProjectedAgent rows with screen center/radius and separate global,
   * presentation and layout identities. Layout fallback indices never become public
   * slot facts. Source records remain unchanged.
   *
   * @param {JsonRecord} scene
   * @param {ViewportTransform} transform
   * @param {Readonly<DurableVisualPolicy>} visualPolicy
   * @returns {ProjectedAgent[]}
   */
  #renderAgents(scene, transform, visualPolicy) {
    const agents = asArray(scene.agents).filter(
      (agent) => isRecord(agent) && agentDisplayIdentity(agent) !== null,
    );
    const nextIdentities = new Set(agents.map(agentDisplayIdentity));
    for (const [identityKey, nodes] of this.agentNodes) {
      if (!nextIdentities.has(identityKey)) {
        nodes.root.remove();
        nodes.shieldRoot?.remove();
        nodes.selectionRoot.remove();
        this.agentNodes.delete(identityKey);
      }
    }

    const selection = isRecord(scene.selection) ? scene.selection : {};
    /** @type {ProjectedAgent[]} */
    const projectedAgents = [];
    for (const [index, agent] of agents.entries()) {
      const identityKey = agentDisplayIdentity(agent);
      if (identityKey === null) {
        continue;
      }
      const globalSlot = Number.isInteger(agent.global_slot)
        ? Number(agent.global_slot)
        : null;
      const presentationKey =
        typeof agent.presentation_key === "string" ? agent.presentation_key : null;
      const layoutSlot = agentLayoutSlot(agent, index);
      const controlled =
        globalSlot !== null
          ? globalSlot === selection.controlled_global_slot
          : presentationKey === selection.controlled_presentation_key;
      const selected =
        globalSlot !== null
          ? globalSlot === selection.selected_global_slot
          : presentationKey === selection.selected_presentation_key;
      const center = screenPoint(agent.position, transform);
      const radius = transform.worldLengthToScreen(finiteNumber(agent.radius, 0.5));
      let nodes = this.agentNodes.get(identityKey);
      if (!nodes) {
        nodes = this.#createAgentNodes(agent, visualPolicy);
        this.agentNodes.set(identityKey, nodes);
      }
      this.#updateAgentNodes(
        nodes,
        agent,
        scene.spawn_shield_mechanics,
        center,
        radius,
        controlled,
        selected,
        visualPolicy,
      );
      registerTooltipOwner(
        nodes.root,
        scene.audience === "agent_pov"
          ? explainPovAgent(agent, { controlled, selected })
          : explainAgent(agent),
      );

      // Appending an existing child reorders it without replacing its identity.
      this.layers.body.append(nodes.root);
      if (visualPolicy.showSpawnShield && nodes.shieldRoot !== null) {
        this.layers.body.append(nodes.shieldRoot);
      } else {
        nodes.shieldRoot?.remove();
      }
      this.selectionCues.append(nodes.selectionRoot);
      projectedAgents.push({
        agent,
        identityKey,
        layoutSlot,
        globalSlot,
        presentationKey,
        center,
        radius,
        controlled,
        selected,
        statuses: asArray(agent.statuses),
      });
    }
    return projectedAgents;
  }

  /**
   * Draw recipient-observed bodies without assigning undisclosed global slots.
   *
   * scene.observed_bodies must carry an ally/enemy relation, integer observation_row
   * and exactly matching observation_key. transform projects their authorized values;
   * showDurationStatusBadges controls status paint and matching accessible text.
   * Retain nodes by relation-row key, remove absent bodies and update hover/keyboard
   * inspection. These hit regions do not become target/control regions. Return
   * undefined; no hidden body identity or position is invented.
   *
   * @param {JsonRecord} scene
   * @param {ViewportTransform} transform
   * @param {boolean} showDurationStatusBadges
   */
  #renderObservedBodies(scene, transform, showDurationStatusBadges) {
    const bodies = asArray(scene.observed_bodies).filter(
      (body) =>
        isRecord(body) &&
        (body.relation === "ally" || body.relation === "enemy") &&
        Number.isInteger(body.observation_row) &&
        body.observation_key === `${body.relation}:${body.observation_row}`,
    );
    const keys = new Set(bodies.map((body) => String(body.observation_key)));
    for (const [key, node] of this.observedBodyNodes) {
      if (!keys.has(key)) {
        node.remove();
        this.observedBodyNodes.delete(key);
      }
    }

    for (const body of bodies) {
      const key = String(body.observation_key);
      const center = screenPoint(body.position, transform);
      const radius = transform.worldLengthToScreen(finiteNumber(body.radius, 0.5));
      const classToken = classTokenFromId(body.class_id);
      const teamToken = teamTokenFromId(body.team_id);
      let root = this.observedBodyNodes.get(key);
      if (!root) {
        root = svgElement("g", {
          class: "agent pov-observed-body",
          tabindex: "0",
          role: "img",
          "data-observation-key": key,
        });
        root.append(
          svgElement("circle", {
            class: "agent-body",
            "data-zone": "observed-body",
          }),
          svgElement("circle", {
            class: "agent-team-ring",
            "data-zone": "observed-team",
          }),
          svgElement("circle", {
            class: "agent-health-track",
            "data-zone": "observed-health",
          }),
          svgElement("circle", {
            class: "agent-health",
            "data-zone": "observed-health",
            pathLength: 100,
          }),
          createSvgIcon(this.battlefield.ownerDocument, classToken.glyphKey, {
            className: "agent-class-icon",
          }),
          svgElement("text", { class: "pov-observed-body__label" }),
          svgElement("g", {
            class: "pov-observed-body__statuses",
            "data-zone": "observed-statuses",
          }),
        );
        this.observedBodyNodes.set(key, root);
      }

      root.dataset.relation = body.relation;
      root.dataset.observationRow = String(body.observation_row);
      root.dataset.publicAgentId = String(body.public_agent_id);
      root.dataset.team = teamToken.cssKey;
      root.dataset.class = classToken.cssKey;
      root.dataset.alive = String(Boolean(body.alive));
      const statuses = asArray(body.statuses);
      const paintedStatuses = showDurationStatusBadges ? statuses : [];
      const statusDescriptions = paintedStatuses.map((status) => {
        const token = resolveVisualToken("status", status.token_id);
        return `${token.accessibleName}, ${formatDisplayNumber(status.duration)} ticks`;
      });
      root.dataset.statusCount = String(statuses.length);
      root.setAttribute(
        "aria-label",
        [
          `${body.relation} observation row ${body.observation_row}`,
          agentIdentity(body),
          classToken.label,
          `life state ${body.alive ? "alive" : "corpse"}`,
          `health ${formatDisplayNumber(body.current_health)} of ${formatDisplayNumber(body.max_health)}`,
          `effective speed ${formatDisplayNumber(body.effective_movement_speed)}`,
          showDurationStatusBadges
            ? statusDescriptions.length === 0
              ? "no persistent statuses"
              : `statuses ${statusDescriptions.join(", ")}`
            : null,
        ]
          .filter((part) => part !== null)
          .join(", "),
      );

      const healthRadius = Math.max(radius - 4, radius * 0.7);
      const healthRatio = Math.max(
        0,
        Math.min(
          1,
          finiteNumber(body.current_health) /
            Math.max(finiteNumber(body.max_health, 1), Number.EPSILON),
        ),
      );
      const [bodyCircle, teamRing, healthTrack, health, icon, label, statusGroup] =
        root.children;
      setAttributes(/** @type {SVGElement} */ (bodyCircle), {
        cx: center.x,
        cy: center.y,
        r: radius,
      });
      setAttributes(/** @type {SVGElement} */ (teamRing), {
        cx: center.x,
        cy: center.y,
        r: radius,
      });
      setAttributes(/** @type {SVGElement} */ (healthTrack), {
        cx: center.x,
        cy: center.y,
        r: healthRadius,
      });
      setAttributes(/** @type {SVGElement} */ (health), {
        cx: center.x,
        cy: center.y,
        r: healthRadius,
        "stroke-dasharray": `${healthRatio * 100} ${100 - healthRatio * 100}`,
        transform: `rotate(-90 ${center.x} ${center.y})`,
      });
      const iconSize = Math.max(14, Math.min(radius * 0.95, 28));
      setAttributes(/** @type {SVGElement} */ (icon), {
        x: center.x - iconSize / 2,
        y: center.y - iconSize * 0.62,
        width: iconSize,
        height: iconSize,
      });
      setAttributes(/** @type {SVGElement} */ (label), {
        x: center.x,
        y: center.y + radius + 15,
      });
      label.textContent = `${body.relation === "ally" ? "ALLY" : "ENEMY"} ${body.observation_row}`;
      const statusCellWidth = 18;
      const statusCellHeight = 15;
      const statusColumns = Math.min(paintedStatuses.length, 5);
      const statusNodes = paintedStatuses.map((status, index) => {
        const token = resolveVisualToken("status", status.token_id);
        const rowIndex = Math.floor(index / 5);
        const columnIndex = index % 5;
        const rowCount = Math.min(5, paintedStatuses.length - rowIndex * 5);
        const x =
          center.x - (rowCount * statusCellWidth) / 2 + columnIndex * statusCellWidth;
        const y = center.y - radius - 19 - rowIndex * statusCellHeight;
        const effectClass = classTokenFromId(status.source_class_id);
        const cell = svgElement("g", {
          class: "pov-observed-status",
          transform: `translate(${x} ${y})`,
          role: "img",
          "aria-label": `${token.accessibleName}, duration ${formatDisplayNumber(status.duration)} ticks`,
          "data-token-id": token.tokenId,
          "data-duration": status.duration,
          "data-status-feature-index": status.status_feature_index,
          "data-effect-class-id": status.source_class_id,
          "data-effect-class": effectClass.cssKey,
        });
        const statusIcon = createSvgIcon(
          this.battlefield.ownerDocument,
          token.glyphKey,
          { className: "pov-observed-status__icon" },
        );
        setAttributes(statusIcon, {
          x: 2,
          y: 2,
          width: 8,
          height: 8,
        });
        cell.append(
          svgElement("rect", {
            class: "pov-observed-status__box",
            x: 0,
            y: 0,
            width: statusCellWidth - 2,
            height: statusCellHeight - 1,
            rx: 3,
          }),
          statusIcon,
          svgElement("text", {
            class: "pov-observed-status__duration",
            x: 12,
            y: 11,
          }),
        );
        const duration = cell.lastElementChild;
        if (duration) {
          duration.textContent = formatCompactDisplayNumber(status.duration);
        }
        registerTooltipOwner(
          cell,
          this.#explainStatus(
            status,
            {
              presentation_key: body.presentation_key,
              public_agent_id: body.public_agent_id,
              class_id: body.class_id,
              team_id: body.team_id,
            },
            "agent_pov",
          ),
        );
        return cell;
      });
      statusGroup.replaceChildren(...statusNodes);
      if (showDurationStatusBadges) {
        statusGroup.setAttribute("data-columns", String(statusColumns));
      } else {
        statusGroup.removeAttribute("data-columns");
      }
      registerTooltipOwner(
        root,
        explainPovAgent(body, { controlled: false, selected: false }),
      );
      this.layers.body.append(root);
    }
  }

  /**
   * Draw the selected pair's supplied Basic/Ultimate legality at its exact owner.
   *
   * scene.selected_legality must join one projectedAgents row by presentation key and
   * public ID. transform supplies viewport; reservedRects protects already placed
   * cues. Allocate two lane pills, register mask-based explanations and return their
   * ProtectedRegion array. Missing owner/legality, suppressed placement or unavailable
   * explanation gives an empty array. Geometry/cooldown/class facts never determine
   * legality. The legality layer is replaced; inputs stay unchanged.
   *
   * @param {JsonRecord} scene
   * @param {ProjectedAgent[]} projectedAgents
   * @param {ViewportTransform} transform
   * @param {Rectangle[]} reservedRects
   * @returns {ProtectedRegion[]}
   */
  #renderSelectedLegality(scene, projectedAgents, transform, reservedRects) {
    this.legalityCues.replaceChildren();
    const legality = isRecord(scene.selected_legality) ? scene.selected_legality : null;
    if (!legality) {
      return [];
    }
    const owner = projectedAgents.find(
      ({ presentationKey, agent }) =>
        typeof legality.owner_presentation_key === "string" &&
        typeof legality.owner_public_agent_id === "string" &&
        presentationKey === legality.owner_presentation_key &&
        agent.public_agent_id === legality.owner_public_agent_id,
    );
    if (!owner) {
      return [];
    }

    const layout = layoutStatusDocks(
      {
        agents: projectedAgents.map((agent) => ({
          globalSlot: agent.layoutSlot,
          center: agent.center,
          radius: agent.radius,
          statuses:
            agent.layoutSlot === owner.layoutSlot
              ? Object.freeze(["basic", "ultimate"])
              : Object.freeze([]),
          controlled: agent.controlled,
          selected: agent.selected || agent.layoutSlot === owner.layoutSlot,
        })),
        viewport: transform.viewportBounds,
        reservedRects,
      },
      {
        bodyPadding: 4,
        selectionAllowance: 12,
        ...LEGALITY_DOCK_DIMENSIONS,
        dockGap: 5,
      },
    );
    const placement = layout.docks.find(
      ({ globalSlot }) => globalSlot === owner.layoutSlot,
    );
    if (!placement) {
      if (owner.globalSlot !== null) {
        this.legalityCues.removeAttribute("data-suppressed-presentation-key");
        this.legalityCues.dataset.suppressedSlot = String(owner.globalSlot);
      } else {
        this.legalityCues.removeAttribute("data-suppressed-slot");
        this.legalityCues.dataset.suppressedPresentationKey = String(
          owner.presentationKey,
        );
      }
      return [];
    }
    this.legalityCues.removeAttribute("data-suppressed-slot");
    this.legalityCues.removeAttribute("data-suppressed-presentation-key");

    const group = svgElement("g", {
      class: "legality-dock",
      role: "group",
      "aria-label": `Exact actor-owned legality for ${agentIdentity(owner.agent)}`,
      "data-zone": "legality",
      "data-presentation-key": owner.presentationKey,
      "data-anchor": placement.anchor,
      "data-collision-free": placement.collisionFree,
    });
    if (placement.anchor !== "north" || placement.tangentShift !== 0) {
      group.append(
        svgElement("line", {
          class: "dock-leader legality-dock__leader",
          x1: placement.leader.start.x,
          y1: placement.leader.start.y,
          x2: placement.leader.end.x,
          y2: placement.leader.end.y,
          "aria-hidden": "true",
        }),
      );
    }
    const lanes = [
      {
        lane: 0,
        label: "0/B",
        name: "Basic",
        available: Boolean(legality.basic_available),
      },
      {
        lane: 1,
        label: "1/U",
        name: "Ultimate",
        available: Boolean(legality.ultimate_available),
      },
    ];
    for (const [index, lane] of lanes.entries()) {
      const x =
        placement.bounds.left +
        index * (LEGALITY_DOCK_DIMENSIONS.cellWidth + LEGALITY_DOCK_DIMENSIONS.cellGap);
      const y = placement.bounds.top;
      const armed = legality.armed_lane === lane.lane;
      const pill = svgElement("g", {
        class: "legality-pill",
        role: "img",
        "aria-label": `${lane.name} lane ${lane.available ? "available" : "unavailable"}${armed ? `, armed and ${legality.armed_pair_legal ? "legal" : "illegal"}` : ""} for ${agentIdentity(owner.agent)}`,
        "data-zone": "legality-pill",
        "data-lane": lane.lane,
        "data-available": lane.available,
        "data-armed": armed,
        "data-pair-legal": armed ? Boolean(legality.armed_pair_legal) : null,
      });
      const explanation = explainLegality(
        legality,
        lane.lane === 0 ? 0 : 1,
        owner.agent,
      );
      if (explanation === null) {
        this.legalityCues.replaceChildren();
        return [];
      }
      registerTooltipOwner(pill, explanation);
      pill.append(
        svgElement("rect", {
          class: "legality-pill__box",
          x,
          y,
          width: LEGALITY_DOCK_DIMENSIONS.cellWidth,
          height: LEGALITY_DOCK_DIMENSIONS.cellHeight,
          rx: 6,
        }),
        svgElement("text", {
          class: "legality-pill__label",
          x: x + LEGALITY_DOCK_DIMENSIONS.cellWidth / 2,
          y: y + LEGALITY_DOCK_DIMENSIONS.cellHeight / 2,
        }),
      );
      const label = pill.lastElementChild;
      if (label) {
        label.textContent = lane.label;
      }
      group.append(pill);
    }
    this.legalityCues.replaceChildren(group);
    const ownerPresentationKey =
      typeof owner.presentationKey === "string" ? owner.presentationKey : null;
    const ownerIdentity = ownerPresentationKey ?? `slot:${owner.layoutSlot}`;
    return [
      choreographyProtectedRegion(
        "legality",
        ownerIdentity,
        placement.bounds,
        ownerPresentationKey,
      ),
    ];
  }

  /**
   * Place separate current status, cooldown, modifier and legality docks.
   *
   * scene supplies exact legality; projectedAgents contains disclosed current bodies;
   * transform supplies pixel bounds. policy supplies four show switches and audience.
   * Required selected/controlled statuses and positive cooldowns are placed first,
   * then optional statuses, exact pair legality and nonneutral modifiers. Small
   * viewports use compact/overflow representations; the shared allocators own collision
   * choices. Replace durable dock DOM, publish suppression/compaction metadata and
   * protected regions, then measure numeric cells. Return undefined. Python-provided
   * status order, values and actor identity remain the semantic authority.
   *
   * @param {JsonRecord} scene
   * @param {ProjectedAgent[]} projectedAgents
   * @param {ViewportTransform} transform
   * @param {{
   *   showLegality: boolean,
   *   showModifiers: boolean,
   *   showStatuses: boolean,
   *   showCooldowns: boolean,
   *   audience: "researcher" | "agent_pov",
   * }} policy
   */
  #renderStatusDocks(scene, projectedAgents, transform, policy) {
    const compactMinimumViewport =
      transform.viewportBounds.width <= 600 && transform.viewportBounds.height <= 420;
    const requiredStatusSlots = new Set(
      policy.showStatuses
        ? projectedAgents
            .filter(
              (agent) =>
                agent.statuses.length > 0 && (agent.controlled || agent.selected),
            )
            .map(({ layoutSlot }) => layoutSlot)
        : [],
    );
    const requiredDockRequests = [
      ...projectedAgents
        .filter(({ layoutSlot }) => requiredStatusSlots.has(layoutSlot))
        .map((agent) => ({
          layoutKey: `status:${agent.layoutSlot}`,
          globalSlot: agent.layoutSlot,
          publicAgentId: agent.agent.public_agent_id,
          statuses: agent.statuses,
          dockOptions: {
            ...STATUS_DOCK_DIMENSIONS,
            dockGap: 5,
            requiredVisibleLimit: compactMinimumViewport ? 0 : 9,
          },
          fallbackDockOptions: {
            cellWidth: 32,
            cellHeight: 16,
            cellGap: 0,
            dockGap: 3,
          },
          priority: 0,
        })),
      ...projectedAgents.flatMap((agent) => {
        const ticks = agent.agent.ultimate_cooldown;
        if (!policy.showCooldowns || !Number.isInteger(ticks) || Number(ticks) <= 0) {
          return [];
        }
        return [
          {
            layoutKey: `cooldown:${agent.layoutSlot}`,
            globalSlot: agent.layoutSlot,
            publicAgentId: agent.agent.public_agent_id,
            statuses: Object.freeze([
              Object.freeze({
                classId: agent.agent.class_id,
                ticks: Number(ticks),
                publicAgentId: agent.agent.public_agent_id,
              }),
            ]),
            dockOptions: {
              ...COOLDOWN_DOCK_DIMENSIONS,
              dockGap: 5,
            },
            fallbackDockOptions: {
              ...COOLDOWN_DOCK_DIMENSIONS,
              dockGap: 5,
            },
            priority: 1,
          },
        ];
      }),
    ];
    const requiredDockLayout = layoutRequiredDocks(
      {
        agents: projectedAgents.map((agent) => ({
          globalSlot: agent.layoutSlot,
          center: agent.center,
          radius: agent.radius,
          statuses: Object.freeze([]),
          controlled: agent.controlled,
          selected: agent.selected,
        })),
        requests: requiredDockRequests,
        viewport: transform.viewportBounds,
      },
      {
        bodyPadding: 4,
        selectionAllowance: 12,
        dockGap: 5,
      },
    );
    const requiredStatusDocks = requiredDockLayout.docks.filter(
      ({ compactFallback, layoutKey }) =>
        !compactFallback && layoutKey.startsWith("status:"),
    );
    const cooldownDocks = requiredDockLayout.docks.filter(({ layoutKey }) =>
      layoutKey.startsWith("cooldown:"),
    );
    const compactRequiredDocks = requiredDockLayout.docks.filter(
      ({ compactFallback, layoutKey }) =>
        compactFallback && layoutKey.startsWith("status:"),
    );
    const requiredDockRects = requiredDockLayout.docks.map(({ bounds }) => bounds);
    const optionalStatusLayout = layoutStatusDocks(
      {
        agents: projectedAgents.map((agent) => ({
          globalSlot: agent.layoutSlot,
          center: agent.center,
          radius: agent.radius,
          statuses:
            policy.showStatuses && !requiredStatusSlots.has(agent.layoutSlot)
              ? agent.statuses
              : Object.freeze([]),
          controlled: agent.controlled,
          selected: agent.selected,
        })),
        viewport: transform.viewportBounds,
        reservedRects: requiredDockRects,
      },
      {
        bodyPadding: 4,
        selectionAllowance: 12,
        ...STATUS_DOCK_DIMENSIONS,
        dockGap: 5,
        ordinaryVisibleLimit: compactMinimumViewport ? 0 : 9,
      },
    );
    const suppressedStatusSlots = [
      ...requiredDockLayout.suppressedLayoutKeys
        .filter((layoutKey) => layoutKey.startsWith("status:"))
        .map((layoutKey) => Number(layoutKey.slice("status:".length))),
      ...optionalStatusLayout.suppressedGlobalSlots,
    ].sort((left, right) => left - right);
    const suppressedCooldownSlots = requiredDockLayout.suppressedLayoutKeys
      .filter((layoutKey) => layoutKey.startsWith("cooldown:"))
      .map((layoutKey) => Number(layoutKey.slice("cooldown:".length)))
      .sort((left, right) => left - right);
    const statusLayout = {
      docks: [...requiredStatusDocks, ...optionalStatusLayout.docks].sort(
        (left, right) => left.globalSlot - right.globalSlot,
      ),
      protectedBodies: requiredDockLayout.protectedBodies,
      placementOrder: [
        ...requiredStatusDocks.map(({ globalSlot }) => globalSlot),
        ...optionalStatusLayout.placementOrder,
      ],
      suppressedGlobalSlots: suppressedStatusSlots,
    };
    const cooldownLayout = {
      docks: cooldownDocks,
      suppressedGlobalSlots: suppressedCooldownSlots,
    };
    const statusRects = statusLayout.docks.map(({ bounds }) => bounds);
    const cooldownRects = cooldownDocks.map(({ bounds }) => bounds);
    const compactRequiredRects = compactRequiredDocks.map(({ bounds }) => bounds);
    const legalityRects = policy.showLegality
      ? this.#renderSelectedLegality(scene, projectedAgents, transform, [
          ...statusRects,
          ...cooldownRects,
          ...compactRequiredRects,
        ])
      : [];
    if (!policy.showLegality) {
      this.legalityCues.replaceChildren();
      this.legalityCues.removeAttribute("data-suppressed-slot");
      this.legalityCues.removeAttribute("data-suppressed-presentation-key");
    }
    const modifierLayout = policy.showModifiers
      ? layoutStatusDocks(
          {
            agents: projectedAgents.map((agent) => ({
              globalSlot: agent.layoutSlot,
              center: agent.center,
              radius: agent.radius,
              statuses: asArray(agent.agent.modifiers).filter(
                (modifier) =>
                  !isRecord(modifier) ||
                  typeof modifier.multiplier !== "number" ||
                  !Number.isFinite(modifier.multiplier) ||
                  modifier.multiplier !== 1,
              ),
              controlled: agent.controlled,
              selected: agent.selected,
            })),
            viewport: transform.viewportBounds,
            reservedRects: [
              ...statusLayout.protectedBodies.map(({ bounds }) => bounds),
              ...statusRects,
              ...cooldownRects,
              ...compactRequiredRects,
              ...legalityRects,
            ],
          },
          {
            bodyPadding: 4,
            selectionAllowance: 12,
            ...MODIFIER_DOCK_DIMENSIONS,
            dockGap: 5,
          },
        )
      : { docks: [], suppressedGlobalSlots: [] };

    const modifierNodes = modifierLayout.docks.map((placement) =>
      this.#renderFactDock(
        placement,
        "modifier",
        MODIFIER_DOCK_DIMENSIONS,
        policy.audience,
      ),
    );
    const statusNodes = statusLayout.docks.map((placement) =>
      this.#renderFactDock(
        placement,
        "status",
        STATUS_DOCK_DIMENSIONS,
        policy.audience,
      ),
    );
    const cooldownNodes = [
      ...cooldownLayout.docks.filter(({ compactFallback }) => compactFallback),
      ...cooldownLayout.docks.filter(({ compactFallback }) => !compactFallback),
    ].map((placement) => this.#renderCooldownDock(placement));
    const compactRequiredNodes = compactRequiredDocks.map((placement) =>
      this.#renderFactDock(
        placement,
        "status",
        {
          cellWidth: placement.bounds.width,
          cellHeight: placement.bounds.height,
          cellGap: 0,
        },
        policy.audience,
      ),
    );
    const usesPresentationKeys = projectedAgents.some(
      ({ presentationKey }) => presentationKey !== null,
    );
    /**
     * Map internal layout indices to already disclosed presentation keys.
     *
     * slots is an ordered index array; omit missing/nonstring keys. Return a new array
     * for suppression metadata, without exposing local indices as Agent POV slot facts.
     */
    const presentationKeysForLayoutSlots = (
      /** @type {ReadonlyArray<number>} */ slots,
    ) =>
      slots.flatMap((layoutSlot) => {
        const key = this.agentByLayoutSlot.get(layoutSlot)?.presentation_key;
        return typeof key === "string" ? [key] : [];
      });
    if (usesPresentationKeys) {
      this.layers.durableStatusModifier.removeAttribute("data-suppressed-status-slots");
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-cooldown-slots",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-modifier-slots",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-compacted-required-docks",
      );
      if (policy.showStatuses) {
        this.layers.durableStatusModifier.dataset.suppressedStatusPresentationKeys =
          presentationKeysForLayoutSlots(statusLayout.suppressedGlobalSlots).join(",");
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-status-presentation-keys",
        );
      }
      if (policy.showCooldowns) {
        this.layers.durableStatusModifier.dataset.suppressedCooldownPresentationKeys =
          presentationKeysForLayoutSlots(cooldownLayout.suppressedGlobalSlots).join(
            ",",
          );
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-cooldown-presentation-keys",
        );
      }
      if (policy.showModifiers) {
        this.layers.durableStatusModifier.dataset.suppressedModifierPresentationKeys =
          presentationKeysForLayoutSlots(modifierLayout.suppressedGlobalSlots).join(
            ",",
          );
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-modifier-presentation-keys",
        );
      }
      this.layers.durableStatusModifier.dataset.compactedRequiredPresentations =
        requiredDockLayout.compactedLayoutKeys
          .flatMap((layoutKey) => {
            const [kind, rawLayoutSlot] = layoutKey.split(":");
            const key = this.agentByLayoutSlot.get(
              Number(rawLayoutSlot),
            )?.presentation_key;
            return typeof key === "string" ? [`${kind}:${key}`] : [];
          })
          .join(",");
    } else {
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-status-presentation-keys",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-cooldown-presentation-keys",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-suppressed-modifier-presentation-keys",
      );
      this.layers.durableStatusModifier.removeAttribute(
        "data-compacted-required-presentations",
      );
      if (policy.showStatuses) {
        this.layers.durableStatusModifier.dataset.suppressedStatusSlots =
          statusLayout.suppressedGlobalSlots.join(",");
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-status-slots",
        );
      }
      if (policy.showCooldowns) {
        this.layers.durableStatusModifier.dataset.suppressedCooldownSlots =
          cooldownLayout.suppressedGlobalSlots.join(",");
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-cooldown-slots",
        );
      }
      this.layers.durableStatusModifier.dataset.compactedRequiredDocks =
        requiredDockLayout.compactedLayoutKeys.join(",");
      if (policy.showModifiers) {
        this.layers.durableStatusModifier.dataset.suppressedModifierSlots =
          modifierLayout.suppressedGlobalSlots.join(",");
      } else {
        this.layers.durableStatusModifier.removeAttribute(
          "data-suppressed-modifier-slots",
        );
      }
    }
    this.layers.durableStatusModifier.replaceChildren(
      ...modifierNodes,
      ...compactRequiredNodes,
      ...cooldownNodes,
      ...statusNodes,
    );
    this.#resolveNumericDockCellContent();
    /**
     * Resolve an internal layout slot to its protected-region identity.
     *
     * globalSlot is the allocator index. Return a frozen ownerPresentationKey and
     * ownerIdentity; a legacy slot key is used only when no presentation key exists.
     *
     * @param {number} globalSlot
     */
    const protectedOwner = (globalSlot) => {
      const key = this.agentByLayoutSlot.get(globalSlot)?.presentation_key;
      const ownerPresentationKey = typeof key === "string" ? key : null;
      return Object.freeze({
        ownerPresentationKey,
        ownerIdentity: ownerPresentationKey ?? `slot:${globalSlot}`,
      });
    };
    /**
     * Package one allocator-owned body or dock rectangle for choreography.
     *
     * kind names the region, globalSlot is its internal owner index, bounds is its
     * pixel rectangle, and placementIdentity distinguishes docks. Return a frozen
     * region using the disclosed owner key when available; no spatial owner is guessed.
     *
     * @param {"body" | "status" | "cooldown" | "modifier"} kind
     * @param {number} globalSlot
     * @param {Rectangle} bounds
     * @param {string} placementIdentity
     */
    const protectedDock = (kind, globalSlot, bounds, placementIdentity) => {
      const owner = protectedOwner(globalSlot);
      return choreographyProtectedRegion(
        kind,
        owner.ownerIdentity,
        bounds,
        owner.ownerPresentationKey,
        placementIdentity,
      );
    };
    const bodyProtectedRegions = statusLayout.protectedBodies.map(
      ({ globalSlot, bounds }) =>
        protectedDock("body", globalSlot, bounds, `body:${globalSlot}`),
    );
    const cooldownProtectedRegions = cooldownDocks.map(
      ({ globalSlot, bounds, layoutKey, compactFallback }) =>
        protectedDock(
          "cooldown",
          globalSlot,
          bounds,
          compactFallback ? `compact:${layoutKey}` : layoutKey,
        ),
    );
    const compactRequiredStatusRegions = compactRequiredDocks
      .filter(({ layoutKey }) => layoutKey.startsWith("status:"))
      .map(({ globalSlot, bounds, layoutKey }) =>
        protectedDock("status", globalSlot, bounds, `compact:${layoutKey}`),
      );
    const modifierProtectedRegions = modifierLayout.docks.map(
      ({ globalSlot, bounds }) =>
        protectedDock("modifier", globalSlot, bounds, `modifier:${globalSlot}`),
    );
    const statusProtectedRegions = statusLayout.docks.map((placement) =>
      protectedDock(
        "status",
        placement.globalSlot,
        placement.bounds,
        "layoutKey" in placement && typeof placement.layoutKey === "string"
          ? placement.layoutKey
          : `status:${placement.globalSlot}`,
      ),
    );
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze([
        ...bodyProtectedRegions,
        ...cooldownProtectedRegions,
        ...modifierProtectedRegions,
      ]),
      legality: Object.freeze(legalityRects),
      status: Object.freeze([
        ...statusProtectedRegions,
        ...compactRequiredStatusRegions,
      ]),
    });
    this.#refreshChoreographyProtectedRects();
  }

  /**
   * Keep dock values readable using measured browser text/glyph bounds.
   *
   * Inspect current status/cooldown/modifier cells. Ordinary supported status durations
   * 1..5 keep fixed compartments. Otherwise hide a colliding decorative icon and, if
   * needed, abbreviate the visible number. Exact values remain in data attributes and
   * semantic labels/tooltips. This reads browser layout and mutates current cells;
   * return undefined. Missing graphics nodes are skipped; DOM measurement errors
   * propagate. It also runs after fonts become ready.
   */
  #resolveNumericDockCellContent() {
    for (const cell of this.layers.durableStatusModifier.querySelectorAll(
      ".status-cell, .cooldown-cell, .modifier-cell",
    )) {
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
        continue;
      }
      const supportedStatusDuration =
        kind === "status" && /^[1-5]$/u.test(value.textContent ?? "");
      if (supportedStatusDuration) {
        cell.setAttribute("data-numeric-layout", "compartments");
        cell.removeAttribute("data-icon-suppressed");
        cell.removeAttribute("data-numeric-fallback");
        icon.removeAttribute("hidden");
        continue;
      }
      const boxBounds = box.getBBox();
      const iconScreenBounds = icon.getBoundingClientRect();
      const valueScreenBounds = value.getBoundingClientRect();
      let valueBounds = value.getBBox();
      if (iconScreenBounds.right + 2 > valueScreenBounds.left) {
        cell.setAttribute("data-icon-suppressed", "true");
        cell.setAttribute("data-numeric-fallback", "true");
        cell.setAttribute("data-numeric-layout", "measured-fallback");
        icon.setAttribute("hidden", "");
        value.setAttribute("x", String(boxBounds.x + boxBounds.width / 2));
        valueBounds = value.getBBox();
      }
      const availableWidth = Math.max(boxBounds.width - 6, 1);
      if (valueBounds.width > availableWidth) {
        const exactValue =
          kind === "status"
            ? Number(cell.getAttribute("data-duration"))
            : kind === "cooldown"
              ? Number(cell.getAttribute("data-ticks"))
              : Number(cell.getAttribute("data-multiplier"));
        const prefix = kind === "modifier" ? "×" : "";
        value.textContent = `${prefix}${formatCompactDisplayNumber(exactValue)}`;
        value.removeAttribute("textLength");
        value.removeAttribute("lengthAdjust");
        cell.setAttribute("data-numeric-layout", "compact-measured-fallback");
        cell.setAttribute("data-visible-value-abbreviated", "true");
      }
    }
  }

  /**
   * Create one class-specific Ultimate cooldown dock from an allocated placement.
   *
   * placement supplies a required-dock result and its one visible class/tick record.
   * Use the owner from the renderer's layout map. Positive integer ticks are shown
   * exactly before later measurement; invalid ticks display ?. A missing explanation
   * returns an aria-hidden empty group. Otherwise return detached SVG with separate
   * icon/value compartments and tooltip, including compact fallback metadata.
   *
   * @param {ReturnType<typeof layoutRequiredDocks>["docks"][number]} placement
   * @returns {SVGElement}
   */
  #renderCooldownDock(placement) {
    const fallbackPlacement = placement.compactFallback === true;
    const rawItem = placement.visibleStatuses[0];
    const item = isRecord(rawItem) ? rawItem : {};
    const ticks =
      Number.isInteger(item.ticks) && item.ticks > 0 ? Number(item.ticks) : "?";
    const classToken = classTokenFromId(item.classId);
    const token = ultimateTokenFromClassId(item.classId);
    const ownerAgent = this.agentByLayoutSlot.get(placement.globalSlot) ?? {};
    const group = svgElement("g", {
      class: "cooldown-dock",
      "data-zone": "cooldown-dock",
      ...displayIdentityAttributes(ownerAgent),
      "data-class": classToken.cssKey,
      "data-anchor": placement.anchor,
      "data-expanded": fallbackPlacement ? true : placement.expanded,
      "data-collision-free": placement.collisionFree,
      "data-visible-count": 1,
      "data-hidden-count": 0,
    });
    const explanation = explainCooldown(
      {
        ...displayIdentityRecord(ownerAgent),
        ultimate_cooldown: ticks,
      },
      ownerAgent,
    );
    if (explanation === null) {
      group.setAttribute("aria-hidden", "true");
      return group;
    }
    if (
      fallbackPlacement ||
      placement.anchor !== "north" ||
      placement.tangentShift !== 0
    ) {
      group.append(
        svgElement("line", {
          class: "dock-leader cooldown-dock__leader",
          x1: placement.leader.start.x,
          y1: placement.leader.start.y,
          x2: placement.leader.end.x,
          y2: placement.leader.end.y,
          "aria-hidden": "true",
        }),
      );
    }

    const x = placement.bounds.left;
    const y = placement.bounds.top;
    const accessibleTicks =
      typeof ticks === "number"
        ? `${ticks} ${ticks === 1 ? "tick" : "ticks"} remaining`
        : "remaining ticks unknown";
    const cell = svgElement("g", {
      class: "cooldown-cell",
      role: "img",
      tabindex: fallbackPlacement ? "0" : null,
      "aria-label": `${token.accessibleName} cooldown, ${accessibleTicks}, ${agentIdentity(ownerAgent)}`,
      "data-zone": "ultimate-cooldown",
      ...displayIdentityAttributes(ownerAgent),
      "data-layout-key": fallbackPlacement ? placement.layoutKey : null,
      "data-kind": fallbackPlacement ? "cooldown" : null,
      "data-compact-fallback": fallbackPlacement ? "true" : null,
      "data-owner-label": fallbackPlacement ? agentIdentity(ownerAgent) : null,
      "data-class": classToken.cssKey,
      "data-token": token.cssKey,
      "data-token-id": token.tokenId,
      "data-ticks": ticks,
      "data-numeric-layout": "compartments",
    });
    const box = svgElement("rect", {
      class: "cooldown-cell__box",
      x,
      y,
      width: COOLDOWN_DOCK_DIMENSIONS.cellWidth,
      height: COOLDOWN_DOCK_DIMENSIONS.cellHeight,
      rx: 5,
    });
    const iconCompartment = svgElement("rect", {
      class: "cooldown-cell__icon-compartment",
      x: x + 2,
      y: y + 2,
      width: 14,
      height: COOLDOWN_DOCK_DIMENSIONS.cellHeight - 4,
      "aria-hidden": "true",
    });
    const valueCompartment = svgElement("rect", {
      class: "cooldown-cell__value-compartment",
      x: x + 19,
      y: y + 2,
      width: 16,
      height: COOLDOWN_DOCK_DIMENSIONS.cellHeight - 4,
      "aria-hidden": "true",
    });
    const icon = createSvgIcon(this.battlefield.ownerDocument, token.glyphKey, {
      className: "cooldown-cell__icon",
    });
    setAttributes(icon, {
      x: x + 3,
      y: y + 3,
      width: 12,
      height: 12,
    });
    const value = svgElement("text", {
      class: "cooldown-cell__value",
      x: x + COOLDOWN_DOCK_DIMENSIONS.cellWidth - 3,
      y: y + COOLDOWN_DOCK_DIMENSIONS.cellHeight / 2,
    });
    value.textContent = String(ticks);
    registerTooltipOwner(cell, explanation);
    cell.append(box, iconCompartment, valueCompartment, icon, value);
    group.append(cell);
    return group;
  }

  /**
   * Create one status/modifier dock and its inspectable overflow cue.
   *
   * placement supplies ordered visible/hidden items and pixel geometry; kind is status
   * or modifier; dimensions gives pixel cell width/height/gap; audience selects the
   * researcher or Agent POV explanation path. Return a detached SVG group with exact
   * value metadata, icons and accessible labels. Unknown values show ? rather than
   * becoming measurements. Hidden items remain available through overflow tooltips;
   * local layout slots never become undisclosed public identities.
   *
   * @param {ReturnType<typeof layoutStatusDocks>["docks"][number]} placement
   * @param {"status" | "modifier"} kind
   * @param {{cellWidth: number, cellHeight: number, cellGap: number}} dimensions
   * @param {"researcher" | "agent_pov"} audience
   * @returns {SVGElement}
   */
  #renderFactDock(placement, kind, dimensions, audience) {
    const compact =
      "compactFallback" in placement && placement.compactFallback === true;
    const compactAttributes = compact
      ? { "data-compact-fallback": true, "data-kind": kind, tabindex: "0" }
      : {};
    const ownerAgent = this.agentByLayoutSlot.get(placement.globalSlot) ?? {};
    const group = svgElement("g", {
      class: `${kind}-dock`,
      "data-zone": `${kind}-dock`,
      ...displayIdentityAttributes(ownerAgent),
      "data-anchor": placement.anchor,
      "data-expanded": placement.expanded,
      "data-collision-free": placement.collisionFree,
      "data-visible-count": placement.visibleCount,
      "data-hidden-count": placement.hiddenCount,
    });
    if (
      placement.hiddenCount > 0 ||
      placement.anchor !== "north" ||
      placement.tangentShift !== 0
    ) {
      group.append(
        svgElement("line", {
          class: `dock-leader ${kind}-dock__leader`,
          x1: placement.leader.start.x,
          y1: placement.leader.start.y,
          x2: placement.leader.end.x,
          y2: placement.leader.end.y,
          "aria-hidden": "true",
        }),
      );
    }

    for (const [index, rawItem] of placement.visibleStatuses.entries()) {
      const item = isRecord(rawItem) ? rawItem : {};
      const token = resolveVisualToken(
        kind,
        item.token_id,
        audience === "agent_pov" && kind === "status" ? undefined : item,
      );
      const column = index % placement.columns;
      const row = Math.floor(index / placement.columns);
      const x =
        placement.bounds.left + column * (dimensions.cellWidth + dimensions.cellGap);
      const y =
        placement.bounds.top + row * (dimensions.cellHeight + dimensions.cellGap);
      const value =
        kind === "status"
          ? String(
              Number.isInteger(item.duration) && item.duration > 0
                ? item.duration
                : "?",
            )
          : Number.isFinite(item.multiplier)
            ? `×${formatDisplayNumber(item.multiplier)}`
            : "×?";
      const accessibleValue =
        kind === "status"
          ? `duration ${value}`
          : Number.isFinite(item.multiplier)
            ? `multiplier ${formatDisplayNumber(item.multiplier)}`
            : "multiplier unknown";
      const supportedStatusDuration =
        kind === "status" &&
        Number.isInteger(item.duration) &&
        item.duration >= 1 &&
        item.duration <= 5;
      const cell = svgElement("g", {
        class: `${kind}-cell${compact ? " required-dock-fallback" : ""}`,
        ...compactAttributes,
        role: "img",
        "aria-label": `${token.accessibleName}, ${accessibleValue}, ${agentIdentity(ownerAgent)}`,
        "data-zone": `${kind}-cell`,
        ...displayIdentityAttributes(ownerAgent),
        "data-token": token.cssKey,
        "data-token-id": token.tokenId,
        "data-index": index,
        "data-numeric-layout": supportedStatusDuration ? "compartments" : "measured",
        "data-supported-duration": kind === "status" ? supportedStatusDuration : null,
        "data-duration":
          kind === "status" && Number.isInteger(item.duration) ? item.duration : null,
        "data-multiplier":
          kind === "modifier" && Number.isFinite(item.multiplier)
            ? item.multiplier
            : null,
      });
      if (kind === "status") {
        cell.dataset.sourceClass = classTokenFromId(item.source_class_id).cssKey;
      }
      const box = svgElement("rect", {
        class: `${kind}-cell__box`,
        x,
        y,
        width: dimensions.cellWidth,
        height: dimensions.cellHeight,
        rx: 5,
      });
      const icon = createSvgIcon(this.battlefield.ownerDocument, token.glyphKey, {
        className: `${kind}-cell__icon`,
      });
      const iconSize = kind === "modifier" ? 9 : kind === "status" ? 12 : 10;
      setAttributes(icon, {
        x: x + (kind === "modifier" ? 3 : 2),
        y: y + (dimensions.cellHeight - iconSize) / 2,
        width: iconSize,
        height: iconSize,
      });
      const text = svgElement("text", {
        class: `${kind}-cell__value`,
        x: x + dimensions.cellWidth - 3,
        y: y + dimensions.cellHeight / 2,
      });
      text.textContent = value;
      const compartments =
        kind === "status"
          ? [
              svgElement("rect", {
                class: "status-cell__icon-compartment",
                x: x + 2,
                y: y + 2,
                width: 12,
                height: dimensions.cellHeight - 4,
                "aria-hidden": "true",
              }),
              svgElement("rect", {
                class: "status-cell__value-compartment",
                x: x + 16,
                y: y + 2,
                width: 9,
                height: dimensions.cellHeight - 4,
                "aria-hidden": "true",
              }),
            ]
          : [];
      registerTooltipOwner(
        cell,
        kind === "status"
          ? audience === "agent_pov"
            ? this.#explainStatus(item, ownerAgent, audience)
            : explainStatus(item, ownerAgent, [...this.agentByLayoutSlot.values()])
          : explainModifier(item, ownerAgent),
      );
      cell.append(box, ...compartments, icon, text);
      group.append(cell);
    }

    if (placement.hiddenCount > 0) {
      const index = placement.visibleCount;
      const column = index % placement.columns;
      const row = Math.floor(index / placement.columns);
      const x =
        placement.bounds.left + column * (dimensions.cellWidth + dimensions.cellGap);
      const y =
        placement.bounds.top + row * (dimensions.cellHeight + dimensions.cellGap);
      const hiddenLabels = placement.hiddenStatuses.map((rawItem) => {
        const item = isRecord(rawItem) ? rawItem : {};
        const token = resolveVisualToken(
          kind,
          item.token_id,
          audience === "agent_pov" && kind === "status" ? undefined : item,
        );
        return kind === "status"
          ? `${token.accessibleName}, duration ${item.duration ?? "unknown"}`
          : `${token.accessibleName}, multiplier ${formatDisplayNumber(item.multiplier)}`;
      });
      const overflow = svgElement("g", {
        class: `${kind}-overflow${compact ? " required-dock-fallback" : ""}`,
        ...compactAttributes,
        role: "img",
        "aria-label": `${placement.overflowLabel} hidden ${kind} cues for ${agentIdentity(ownerAgent)}: ${hiddenLabels.join("; ")}`,
        "data-zone": `${kind}-overflow`,
        ...displayIdentityAttributes(ownerAgent),
        "data-hidden-count": placement.hiddenCount,
        "data-owner-label": agentIdentity(ownerAgent),
      });
      registerTooltipOwner(
        overflow,
        audience === "agent_pov" && kind === "status"
          ? this.#explainStatusOverflow(placement.hiddenStatuses, ownerAgent, audience)
          : explainOverflow(placement.hiddenStatuses, kind, ownerAgent, [
              ...this.agentByLayoutSlot.values(),
            ]),
      );
      const overflowLabel = svgElement("text", {
        class: `${kind}-overflow__label`,
        x: x + dimensions.cellWidth / 2,
        y: y + dimensions.cellHeight / 2,
      });
      overflowLabel.textContent = placement.overflowLabel;
      overflow.append(
        svgElement("rect", {
          class: `${kind}-cell__box`,
          x,
          y,
          width: dimensions.cellWidth,
          height: dimensions.cellHeight,
          rx: 5,
        }),
        overflowLabel,
      );
      group.append(overflow);
    }
    return group;
  }

  /**
   * Explain an already displayed status with optional joined source attribution.
   *
   * rawStatus is the local status, localRecipient carries public ID/local presentation
   * key, and audience selects researcher or agent_pov. Researcher views use their
   * scene list directly. Agent views look up the same public recipient and require
   * exactly one matching public-status row before using researcher provenance, while
   * retaining the local key; otherwise use the limited POV explanation. Return the
   * semantic descriptor. This does not admit a status or add spatial geometry.
   *
   * @param {unknown} rawStatus
   * @param {JsonRecord} localRecipient
   * @param {"researcher" | "agent_pov"} audience
   */
  #explainStatus(rawStatus, localRecipient, audience) {
    if (audience !== "agent_pov") {
      return explainStatus(rawStatus, localRecipient, [
        ...this.agentByLayoutSlot.values(),
      ]);
    }
    const publicAgentId =
      typeof localRecipient.public_agent_id === "string"
        ? localRecipient.public_agent_id
        : null;
    const researcherRecipient =
      publicAgentId === null
        ? null
        : (this.researcherAgentByPublicId.get(publicAgentId) ?? null);
    const matches = asArray(researcherRecipient?.statuses).filter((candidate) =>
      samePublicStatusFacts(rawStatus, candidate),
    );
    const tooltipRecipient =
      researcherRecipient !== null &&
      typeof localRecipient.presentation_key === "string"
        ? {
            ...researcherRecipient,
            presentation_key: localRecipient.presentation_key,
          }
        : null;
    return tooltipRecipient !== null && matches.length === 1
      ? explainStatus(matches[0], tooltipRecipient, [
          ...this.researcherAgentByPublicId.values(),
        ])
      : explainPovStatus(rawStatus, localRecipient);
  }

  /**
   * Explain all hidden local statuses using one consistent disclosure path.
   *
   * rawStatuses is the ordered hidden list, localRecipient supplies identity, and
   * audience selects researcher/agent_pov. Agent views use researcher provenance only
   * when every status has exactly one public-fact match and the local key exists;
   * otherwise the whole list uses POV explanations. Return the descriptor without
   * changing statuses or admitting new visible facts.
   *
   * @param {ReadonlyArray<unknown>} rawStatuses
   * @param {JsonRecord} localRecipient
   * @param {"researcher" | "agent_pov"} audience
   */
  #explainStatusOverflow(rawStatuses, localRecipient, audience) {
    if (audience !== "agent_pov") {
      return explainOverflow(rawStatuses, "status", localRecipient, [
        ...this.agentByLayoutSlot.values(),
      ]);
    }
    const publicAgentId =
      typeof localRecipient.public_agent_id === "string"
        ? localRecipient.public_agent_id
        : null;
    const researcherRecipient =
      publicAgentId === null
        ? null
        : (this.researcherAgentByPublicId.get(publicAgentId) ?? null);
    const researcherStatuses = asArray(researcherRecipient?.statuses);
    const joined = rawStatuses.map((status) => {
      const matches = researcherStatuses.filter((candidate) =>
        samePublicStatusFacts(status, candidate),
      );
      return matches.length === 1 ? matches[0] : null;
    });
    const tooltipRecipient =
      researcherRecipient !== null &&
      typeof localRecipient.presentation_key === "string"
        ? {
            ...researcherRecipient,
            presentation_key: localRecipient.presentation_key,
          }
        : null;
    return tooltipRecipient !== null && joined.every((status) => status !== null)
      ? explainOverflow(joined, "status", tooltipRecipient, [
          ...this.researcherAgentByPublicId.values(),
        ])
      : explainPovOverflow(rawStatuses, localRecipient);
  }

  /**
   * Create detached shield-chip, icon and text nodes for an accepted agent.
   *
   * agent supplies only display identity at this stage. Return {root, chip, icon, text};
   * active state, position, counter and tooltip are set later by updateAgentNodes.
   *
   * @param {JsonRecord} agent
   * @returns {{
   *   root: SVGElement,
   *   chip: SVGElement,
   *   icon: SVGSVGElement,
   *   text: SVGElement,
   * }}
   */
  #createSpawnShieldNodes(agent) {
    const root = svgElement("g", {
      class: "agent-spawn-shield",
      role: "img",
      "data-zone": "spawn-shield",
      ...displayIdentityAttributes(agent),
    });
    const chip = svgElement("rect", {
      class: "agent-spawn-shield__chip",
      width: 27,
      height: 16,
      rx: 6,
      fill: "#000",
      stroke: "#fff",
      "stroke-width": "1.5",
      "vector-effect": "non-scaling-stroke",
    });
    const icon = createSvgIcon(this.battlefield.ownerDocument, "status-spawn-shield", {
      className: "agent-spawn-shield__icon",
    });
    const text = svgElement("text", {
      class: "agent-spawn-shield__ticks",
      fill: "#fff",
      "font-size": "9",
      "font-weight": "800",
      "text-anchor": "middle",
      "dominant-baseline": "central",
      "pointer-events": "none",
    });
    root.append(chip, icon, text);
    return { root, chip, icon, text };
  }

  /**
   * Create a retained body/selection node bundle for an accepted display agent.
   *
   * agent supplies display identity; visualPolicy.showSpawnShield decides whether to
   * allocate shield nodes immediately. Return AgentNodes with nullable shield members.
   * Nodes are detached and have no current geometry until updateAgentNodes runs.
   *
   * @param {JsonRecord} agent
   * @param {Readonly<DurableVisualPolicy>} visualPolicy
   * @returns {AgentNodes}
   */
  #createAgentNodes(agent, visualPolicy) {
    const root = svgElement("g", {
      class: "agent",
      tabindex: "-1",
      role: "img",
      ...displayIdentityAttributes(agent),
    });
    const body = svgElement("circle", {
      class: "agent-body",
      "data-zone": "body",
    });
    const teamRing = svgElement("circle", {
      class: "agent-team-ring",
      "data-zone": "team",
    });
    const teamMarker = svgElement("path", {
      class: "agent-team-marker",
      "data-zone": "team",
      "aria-hidden": "true",
    });
    const healthTrack = svgElement("circle", {
      class: "agent-health-track",
      "data-zone": "health",
    });
    const health = svgElement("circle", {
      class: "agent-health",
      "data-zone": "health",
      pathLength: 100,
    });
    const classIcon = createSvgIcon(this.battlefield.ownerDocument, "unknown", {
      className: "agent-class-icon",
    });
    const classLetter = svgElement("text", { class: "agent-class-letter" });
    const deadMark = svgElement("path", {
      class: "agent-dead-mark",
      "aria-hidden": "true",
    });
    const shieldNodes = visualPolicy.showSpawnShield
      ? this.#createSpawnShieldNodes(agent)
      : null;
    root.append(
      body,
      teamRing,
      teamMarker,
      healthTrack,
      health,
      classIcon,
      classLetter,
      deadMark,
    );

    const selectionRoot = svgElement("g", {
      class: "agent-selection",
      ...displayIdentityAttributes(agent),
      "data-zone": "selection",
      "aria-hidden": "true",
    });
    const controlledHalo = svgElement("circle", {
      class: "controlled-halo",
    });
    const selectedReticle = svgElement("path", {
      class: "selected-reticle",
    });
    selectionRoot.append(controlledHalo, selectedReticle);
    return {
      root,
      body,
      teamRing,
      teamMarker,
      healthTrack,
      health,
      classIcon,
      classLetter,
      deadMark,
      shieldRoot: shieldNodes?.root ?? null,
      shieldChip: shieldNodes?.chip ?? null,
      shieldIcon: shieldNodes?.icon ?? null,
      shieldText: shieldNodes?.text ?? null,
      selectionRoot,
      controlledHalo,
      selectedReticle,
    };
  }

  /**
   * Refresh current body, health, shield and selection paint on retained nodes.
   *
   * nodes is the mutable AgentNodes bundle; agent supplies accepted current facts;
   * spawnShieldMechanics supplies the public shield profile; center/radius are screen
   * geometry. controlled/selected are exact current selection flags and visualPolicy
   * controls shield/reticle paint. Update accessible labels, health arc, life marks,
   * class icon, shield counter/tooltip and selection outlines. Lazily allocate shield
   * nodes and blur a hidden focused shield. Return undefined. Health ratio is clipped
   * only for drawing; this neither changes game health nor derives simulator legality.
   *
   * @param {AgentNodes} nodes
   * @param {JsonRecord} agent
   * @param {unknown} spawnShieldMechanics
   * @param {{x: number, y: number}} center
   * @param {number} radius
   * @param {boolean} controlled
   * @param {boolean} selected
   * @param {Readonly<DurableVisualPolicy>} visualPolicy
   */
  #updateAgentNodes(
    nodes,
    agent,
    spawnShieldMechanics,
    center,
    radius,
    controlled,
    selected,
    visualPolicy,
  ) {
    const classToken = classTokenFromId(agent.class_id);
    const teamToken = teamTokenFromId(agent.team_id);
    if (visualPolicy.showSpawnShield && nodes.shieldRoot === null) {
      const shieldNodes = this.#createSpawnShieldNodes(agent);
      nodes.shieldRoot = shieldNodes.root;
      nodes.shieldChip = shieldNodes.chip;
      nodes.shieldIcon = shieldNodes.icon;
      nodes.shieldText = shieldNodes.text;
    }
    const healthRadius = Math.max(radius - 4, radius * 0.7);
    const healthRatio = Math.max(
      0,
      Math.min(
        1,
        finiteNumber(agent.current_health) /
          Math.max(finiteNumber(agent.max_health, 1), Number.EPSILON),
      ),
    );
    const stepsUntilOutOfCombat =
      Number.isInteger(agent.steps_until_out_of_combat) &&
      agent.steps_until_out_of_combat >= 0
        ? Number(agent.steps_until_out_of_combat)
        : 0;
    const inCombat = stepsUntilOutOfCombat > 0;

    nodes.root.dataset.team = teamToken.cssKey;
    nodes.root.dataset.class = classToken.cssKey;
    nodes.root.dataset.alive = String(Boolean(agent.alive));
    nodes.root.dataset.controlled = String(controlled);
    nodes.root.dataset.selected = String(selected);
    nodes.root.dataset.combatStatus = inCombat ? "IC" : "OOC";
    nodes.root.dataset.stepsUntilOutOfCombat = String(stepsUntilOutOfCombat);
    const spawnShieldView = visualPolicy.showSpawnShield
      ? createSpawnShieldView(agent, spawnShieldMechanics)
      : null;
    const spawnShieldRemaining = Number.isInteger(agent.spawn_shield_remaining)
      ? Math.max(0, Number(agent.spawn_shield_remaining))
      : 0;
    if (
      (!visualPolicy.showSpawnShield || spawnShieldView?.active !== true) &&
      nodes.shieldRoot !== null &&
      nodes.shieldRoot.ownerDocument.activeElement === nodes.shieldRoot
    ) {
      nodes.shieldRoot.blur();
    }
    nodes.root.dataset.spawnShieldRemaining = String(spawnShieldRemaining);
    nodes.root.dataset.respawnedOnIncomingTransition = String(
      agent.respawned_on_incoming_transition === true,
    );
    nodes.root.setAttribute(
      "aria-label",
      [
        agentIdentity(agent),
        classToken.label,
        teamToken.label,
        `health ${formatDisplayNumber(agent.current_health)} of ${formatDisplayNumber(agent.max_health)}`,
        agent.alive ? "alive" : "dead",
        spawnShieldView?.rootAriaLabel ?? null,
        controlled ? "controlled actor" : null,
        selected ? "selected target" : null,
      ]
        .filter((part) => part !== null)
        .join(", "),
    );
    setAttributes(nodes.body, {
      cx: center.x,
      cy: center.y,
      r: radius,
    });
    setAttributes(nodes.teamRing, {
      cx: center.x,
      cy: center.y,
      r: radius,
    });
    setAttributes(nodes.teamMarker, {
      d: [
        `M ${center.x + radius - 7} ${center.y - 4}`,
        `L ${center.x + radius - 3} ${center.y}`,
        `L ${center.x + radius - 7} ${center.y + 4}`,
      ].join(" "),
    });
    if (
      spawnShieldView !== null &&
      nodes.shieldRoot !== null &&
      nodes.shieldChip !== null &&
      nodes.shieldIcon !== null &&
      nodes.shieldText !== null
    ) {
      setAttributes(nodes.shieldRoot, {
        hidden: spawnShieldView.active ? null : "",
        tabindex: spawnShieldView.active ? "0" : "-1",
        "aria-label": spawnShieldView.shieldAriaLabel,
      });
      const shieldChipX = center.x + radius * 0.6;
      const shieldChipY = center.y - radius - 13;
      setAttributes(nodes.shieldChip, {
        x: shieldChipX,
        y: shieldChipY,
      });
      setAttributes(nodes.shieldIcon, {
        x: shieldChipX + 2.5,
        y: shieldChipY + 2.5,
        width: 11,
        height: 11,
      });
      setAttributes(nodes.shieldText, {
        x: shieldChipX + 20,
        y: shieldChipY + 8,
      });
      nodes.shieldText.textContent = spawnShieldView.badgeText;
      if (spawnShieldView.active) {
        nodes.shieldRoot.removeAttribute("aria-description");
        registerTooltipOwner(nodes.shieldRoot, spawnShieldView.descriptor);
      } else {
        nodes.shieldRoot.removeAttribute("data-tooltip-owner");
        nodes.shieldRoot.removeAttribute("aria-describedby");
        nodes.shieldRoot.removeAttribute("aria-description");
      }
    }
    setAttributes(nodes.healthTrack, {
      cx: center.x,
      cy: center.y,
      r: healthRadius,
    });
    setAttributes(nodes.health, {
      cx: center.x,
      cy: center.y,
      r: healthRadius,
      "stroke-dasharray": `${healthRatio * 100} ${100 - healthRatio * 100}`,
      transform: `rotate(-90 ${center.x} ${center.y})`,
    });

    if (nodes.classIcon.dataset.icon !== classToken.glyphKey) {
      const replacement = createSvgIcon(
        this.battlefield.ownerDocument,
        classToken.glyphKey,
        {
          className: "agent-class-icon",
        },
      );
      nodes.classIcon.replaceWith(replacement);
      nodes.classIcon = replacement;
    }
    const iconSize = Math.max(14, Math.min(radius * 0.95, 28));
    setAttributes(nodes.classIcon, {
      x: center.x - iconSize / 2,
      y: center.y - iconSize * 0.62,
      width: iconSize,
      height: iconSize,
    });
    setAttributes(nodes.classLetter, {
      x: center.x,
      y: center.y + Math.min(radius * 0.5, 11) + 2,
    });
    nodes.classLetter.textContent = classToken.fallback;

    const deadOffset = radius * 0.45;
    setAttributes(nodes.deadMark, {
      d: [
        `M ${center.x - deadOffset} ${center.y - deadOffset}`,
        `L ${center.x + deadOffset} ${center.y + deadOffset}`,
        `M ${center.x + deadOffset} ${center.y - deadOffset}`,
        `L ${center.x - deadOffset} ${center.y + deadOffset}`,
      ].join(" "),
      hidden: agent.alive ? "" : null,
    });

    setAttributes(nodes.controlledHalo, {
      cx: center.x,
      cy: center.y,
      r: radius + 8,
      hidden: controlled ? null : "",
    });
    setAttributes(nodes.selectedReticle, {
      d: targetReticlePath(center.x, center.y, radius + 12),
      hidden: selected && visualPolicy.showSelectionReticle ? null : "",
    });
  }
}
