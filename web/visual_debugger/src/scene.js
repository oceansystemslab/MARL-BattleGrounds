import {
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
 * }} DurableVisualPolicy
 */

/**
 * @param {unknown} value
 * @returns {value is JsonRecord}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Publish one durable obstacle with a stable semantic identity. Top-level
 * coordinates remain available to older surface consumers during migration;
 * the cross-phase allocator consumes the explicit nested bounds.
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
 * @param {unknown} value
 * @returns {any[]}
 */
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * Resolve browser-local paint policy without copying or changing the accepted
 * presentation. The visual-filter module owns validation and exact registry
 * classification.
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
  });
}

/**
 * @param {unknown} value
 * @param {number} fallback
 */
function finiteNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Read one normalized own data property without invoking an accessor or
 * surfacing a hostile Proxy trap.
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
 * Format only an already-authorized public identity. Global slots remain
 * internal join keys and are never promoted to a display fallback.
 *
 * @param {unknown} agent
 */
function agentIdentity(agent) {
  return isRecord(agent) &&
    typeof agent.public_agent_id === "string" &&
    agent.public_agent_id.trim()
    ? `Agent ID ${agent.public_agent_id}`
    : "Agent identity unavailable";
}

/**
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
 * @param {JsonRecord} agent
 * @returns {number | string | null}
 */
function agentDisplayIdentity(agent) {
  if (typeof agent.display_key === "string" && agent.display_key.length > 0) {
    return agent.display_key;
  }
  return Number.isInteger(agent.global_slot) ? Number(agent.global_slot) : null;
}

/** @param {JsonRecord} agent @param {number} fallback */
function agentLayoutSlot(agent, fallback) {
  return Number.isInteger(agent.global_slot) ? Number(agent.global_slot) : fallback;
}

/**
 * Add only the identity actually carried by the accepted display row.
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

/** @param {JsonRecord} record */
function displayIdentityAttributes(record) {
  return typeof record.presentation_key === "string"
    ? { "data-presentation-key": record.presentation_key }
    : Number.isInteger(record.global_slot)
      ? { "data-slot": record.global_slot }
      : {};
}

/**
 * Build tooltip input with the accepted display identity only. A layout slot
 * may position a body, but it never becomes an Agent POV global-slot fact.
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
 * @param {unknown} point
 * @param {ViewportTransform} transform
 */
function screenPoint(point, transform) {
  const values = Array.isArray(point) ? point : [0, 0];
  return transform.worldToScreen([finiteNumber(values[0]), finiteNumber(values[1])]);
}

/**
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
 * Build a four-corner target reticle around one body without covering the
 * controlled-actor halo.
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
 * Retained, presentation-only painter for an authoritative debugger frame.
 *
 * The renderer owns the SVG layer lifecycle but derives no simulator facts.
 * Durable rendering may replace children inside durable layers. It never
 * replaces the SVG root or clears the `transient-events` layer.
 */
export class BattlefieldRenderer {
  /**
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
   * Paint the durable facts in a debugger frame.
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
        let sourceAgent = null;
        if (scene.audience === "researcher") {
          const sourcePresentationKey = ownEnumerableDataValue(
            record,
            "source_presentation_key",
          );
          sourceAgent =
            (typeof sourcePresentationKey === "string"
              ? this.agentByPresentationKey.get(sourcePresentationKey)
              : null) ?? null;
        }
        return explainAura(record, sourceAgent, scene.audience);
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
   * Compatibility seam retained for callers predating cross-phase occupancy.
   * Durable scientific facts are never suppressed; the allocator now reserves
   * their named regions at every supported viewport.
   *
   * @param {boolean} _active
   * @returns {boolean} Whether the effective compact-active state changed.
   */
  setCompactActiveCombat(_active) {
    this.#refreshChoreographyProtectedRects();
    return false;
  }

  /**
   * Convert the SVG-local pointer coordinate supplied by `controls.js` back
   * into the Python debugger's world coordinate convention.
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
   * Return the narrow presentation surface used by transient choreography.
   *
   * The durable renderer retains ownership of every other layer. Callers may
   * append only an owned child beneath `layer` and must project authoritative
   * event anchors through these functions rather than reading successor body
   * coordinates.
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
      worldToScreen: (point) => transform.worldToScreen(point),
      worldLengthToScreen: (length) => transform.worldLengthToScreen(length),
    };
    return Object.freeze(surface);
  }

  /**
   * Clear only durable presentation state. The transient layer is owned by
   * the animation lifecycle and must survive every durable redraw.
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
    this.choreographyProtectedRects = Object.freeze([]);
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze([]),
      legality: Object.freeze([]),
      status: Object.freeze([]),
    });
    this.transform = null;
  }

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
   * @param {ViewportTransform} transform
   * @param {number} width
   * @param {number} height
   * @param {boolean} showUnitGrid
   */
  #renderMap(transform, width, height, showUnitGrid) {
    const bounds = transform.mapBounds;
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
        rx: 8,
      }),
      ...gridLines,
    );
  }

  /**
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
      d: "M -6 -3 L 0 0 L -6 3 L -4 0 Z",
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
   * Paint recipient-authorized POV body rows without assigning them simulator
   * global slots. Their stable browser identity is exactly the disclosed
   * `(relation, observation_row)` axis key; consequently they expose hover and
   * keyboard inspection but never become target/control hit regions.
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
          `Agent ID ${body.public_agent_id}`,
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
          "aria-label": `${token.accessibleName}, duration ${formatDisplayNumber(status.duration)} ticks; source agent identity is not disclosed`,
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
          explainPovStatus(status, {
            presentation_key: body.presentation_key,
            public_agent_id: body.public_agent_id,
            class_id: body.class_id,
            team_id: body.team_id,
          }),
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
   * Render the two exact selected-pair mask values without deriving legality
   * from geometry, cooldowns, class, or any other browser-visible fact.
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
   * Lay out Python-ordered status truth and exact modifier values as separate
   * screen-space docks. This method owns collision policy, not semantics.
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
              cellWidth: 32,
              cellHeight: 16,
              cellGap: 0,
              dockGap: 3,
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
    const cooldownDocks = requiredDockLayout.docks.filter(
      ({ compactFallback, layoutKey }) =>
        !compactFallback && layoutKey.startsWith("cooldown:"),
    );
    const compactRequiredDocks = requiredDockLayout.docks.filter(
      ({ compactFallback }) => compactFallback,
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
    const cooldownNodes = cooldownLayout.docks.map((placement) =>
      this.#renderCooldownDock(placement),
    );
    const compactRequiredNodes = compactRequiredDocks.map((placement) =>
      this.#renderRequiredDockFallback(placement, policy.audience),
    );
    const usesPresentationKeys = projectedAgents.some(
      ({ presentationKey }) => presentationKey !== null,
    );
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
    /** @param {number} globalSlot */
    const protectedOwner = (globalSlot) => {
      const key = this.agentByLayoutSlot.get(globalSlot)?.presentation_key;
      const ownerPresentationKey = typeof key === "string" ? key : null;
      return Object.freeze({
        ownerPresentationKey,
        ownerIdentity: ownerPresentationKey ?? `slot:${globalSlot}`,
      });
    };
    /**
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
      ({ globalSlot, bounds, layoutKey }) =>
        protectedDock("cooldown", globalSlot, bounds, layoutKey),
    );
    const compactRequiredStatusRegions = compactRequiredDocks
      .filter(({ layoutKey }) => layoutKey.startsWith("status:"))
      .map(({ globalSlot, bounds, layoutKey }) =>
        protectedDock("status", globalSlot, bounds, `compact:${layoutKey}`),
      );
    const compactRequiredCooldownRegions = compactRequiredDocks
      .filter(({ layoutKey }) => layoutKey.startsWith("cooldown:"))
      .map(({ globalSlot, bounds, layoutKey }) =>
        protectedDock("cooldown", globalSlot, bounds, `compact:${layoutKey}`),
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
        ...compactRequiredCooldownRegions,
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
   * Keep exact dock numbers and glyphs in disjoint measured regions.
   *
   * Ordinary values retain both. If future authoritative values exceed the
   * compact pill budget, the decorative glyph yields to a readable abbreviated
   * label. The exact value remains available in the cue's data, accessible
   * label, and singleton tooltip instead of being squeezed into illegibility.
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
   * Render one compact, explicitly associated marker when a complete required
   * dock cannot fit. The marker never discards the authoritative payload:
   * status fallbacks expose every item through the shared overflow explanation,
   * while cooldown fallbacks retain the exact tick count.
   *
   * @param {ReturnType<typeof layoutRequiredDocks>["docks"][number]} placement
   * @param {"researcher" | "agent_pov"} audience
   * @returns {SVGElement}
   */
  #renderRequiredDockFallback(placement, audience) {
    const isCooldown = placement.layoutKey.startsWith("cooldown:");
    const kind = isCooldown ? "cooldown" : "status";
    const rawItems = placement.hiddenStatuses;
    const firstItem = isRecord(rawItems[0]) ? rawItems[0] : {};
    const ticks =
      isCooldown && Number.isInteger(firstItem.ticks) && firstItem.ticks > 0
        ? Number(firstItem.ticks)
        : null;
    const ownerAgent = this.agentByLayoutSlot.get(placement.globalSlot) ?? {};
    const explanation = isCooldown
      ? explainCooldown(
          {
            ...displayIdentityRecord(ownerAgent),
            ultimate_cooldown: ticks,
          },
          ownerAgent,
        )
      : audience === "agent_pov"
        ? explainPovOverflow(rawItems, ownerAgent)
        : explainOverflow(rawItems, "status", ownerAgent, [
            ...this.agentByLayoutSlot.values(),
          ]);
    if (explanation === null) {
      return svgElement("g", { "aria-hidden": "true" });
    }
    const valueLabel = isCooldown ? `U${ticks ?? "?"}` : `+${placement.hiddenCount}`;
    const group = svgElement("g", {
      class: "required-dock-fallback-dock",
      "data-zone": "required-dock-fallback-dock",
      ...displayIdentityAttributes(ownerAgent),
      "data-layout-key":
        typeof ownerAgent.presentation_key === "string" ? null : placement.layoutKey,
      "data-kind": kind,
      "data-anchor": placement.anchor,
      "data-collision-free": placement.collisionFree,
    });
    group.append(
      svgElement("line", {
        class: "dock-leader required-dock-fallback__leader",
        x1: placement.leader.start.x,
        y1: placement.leader.start.y,
        x2: placement.leader.end.x,
        y2: placement.leader.end.y,
        "aria-hidden": "true",
      }),
    );
    const cell = svgElement("g", {
      class: "required-dock-fallback",
      role: "img",
      tabindex: "0",
      "aria-label": `${explanation.title}. ${explanation.summary}. ${explanation.rows
        .map((row) => `${row.label}: ${row.value}`)
        .join(". ")}`,
      "data-zone": isCooldown ? "ultimate-cooldown" : "status-overflow",
      ...displayIdentityAttributes(ownerAgent),
      "data-layout-key": placement.layoutKey,
      "data-kind": kind,
      "data-hidden-count": placement.hiddenCount,
      "data-ticks": ticks,
      "data-compact-fallback": "true",
      "data-owner-label": agentIdentity(ownerAgent),
    });
    if (isCooldown) {
      cell.dataset.class = classTokenFromId(firstItem.classId).cssKey;
    }
    registerTooltipOwner(cell, explanation);
    const labelText = svgElement("text", {
      class: "required-dock-fallback__label",
      x: placement.bounds.left + placement.bounds.width / 2,
      y: placement.bounds.top + placement.bounds.height / 2,
    });
    const ownerLine = svgElement("tspan", {
      class: "required-dock-fallback__owner",
      x: placement.bounds.left + placement.bounds.width / 2,
      dy: "-0.34em",
    });
    ownerLine.textContent = agentIdentity(ownerAgent);
    const valueLine = svgElement("tspan", {
      class: "required-dock-fallback__value",
      x: placement.bounds.left + placement.bounds.width / 2,
      dy: isCooldown ? "0.9em" : "0.35em",
    });
    valueLine.textContent = valueLabel;
    cell.append(
      svgElement("rect", {
        class: "required-dock-fallback__box",
        x: placement.bounds.left,
        y: placement.bounds.top,
        width: placement.bounds.width,
        height: placement.bounds.height,
        rx: 5,
      }),
      labelText,
    );
    if (isCooldown) {
      labelText.append(ownerLine, valueLine);
    } else {
      labelText.append(valueLine);
    }
    group.append(cell);
    return group;
  }

  /**
   * Render one mandatory class-specific Ultimate cooldown cue.
   *
   * The exact tick value and icon occupy separate fixed compartments. Layout
   * remains generic and collision-aware; this method owns only cooldown
   * presentation.
   *
   * @param {ReturnType<typeof layoutStatusDocks>["docks"][number]} placement
   * @returns {SVGElement}
   */
  #renderCooldownDock(placement) {
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
      "data-expanded": placement.expanded,
      "data-collision-free": placement.collisionFree,
      "data-visible-count": placement.visibleCount,
      "data-hidden-count": placement.hiddenCount,
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
    if (placement.anchor !== "north" || placement.tangentShift !== 0) {
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
      "aria-label": `${token.accessibleName} cooldown, ${accessibleTicks}, ${agentIdentity(ownerAgent)}`,
      "data-zone": "ultimate-cooldown",
      ...displayIdentityAttributes(ownerAgent),
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
   * @param {ReturnType<typeof layoutStatusDocks>["docks"][number]} placement
   * @param {"status" | "modifier"} kind
   * @param {{cellWidth: number, cellHeight: number, cellGap: number}} dimensions
   * @param {"researcher" | "agent_pov"} audience
   * @returns {SVGElement}
   */
  #renderFactDock(placement, kind, dimensions, audience) {
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
        class: `${kind}-cell`,
        role: "img",
        "aria-label": `${token.accessibleName}, ${accessibleValue}, ${agentIdentity(ownerAgent)}${audience === "agent_pov" && kind === "status" ? "; source agent identity is not disclosed" : ""}`,
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
            ? explainPovStatus(item, ownerAgent)
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
        class: `${kind}-overflow`,
        role: "img",
        "aria-label": `${placement.overflowLabel} hidden ${kind} cues for ${agentIdentity(ownerAgent)}: ${hiddenLabels.join("; ")}${audience === "agent_pov" && kind === "status" ? "; source agent identity is not disclosed" : ""}`,
        "data-zone": `${kind}-overflow`,
        ...displayIdentityAttributes(ownerAgent),
        "data-hidden-count": placement.hiddenCount,
        "data-owner-label": agentIdentity(ownerAgent),
      });
      registerTooltipOwner(
        overflow,
        audience === "agent_pov" && kind === "status"
          ? explainPovOverflow(placement.hiddenStatuses, ownerAgent)
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
        `Agent ID ${agent.public_agent_id}`,
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
