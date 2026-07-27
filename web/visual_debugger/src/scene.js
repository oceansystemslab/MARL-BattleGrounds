import { formatCompactDisplayNumber, formatDisplayNumber } from "./display.js";
import {
  explainAgent,
  explainAura,
  explainCooldown,
  explainLegality,
  explainModifier,
  explainObstacle,
  explainOverflow,
  explainPendingRoute,
  explainRange,
  explainStatus,
  explainVisibility,
} from "./explanations.js";
import { createSvgIcon } from "./icons.js";
import {
  createViewportTransform,
  layoutRequiredDocks,
  layoutStatusDocks,
  protectedBodyRect,
  rectanglesIntersect,
} from "./layout.js";
import { createRouteGeometry } from "./routes.js";
import { registerTooltipOwner } from "./tooltip.js";
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
 *   kind: string,
 *   id: string,
 *   title: string,
 *   details: string[],
 *   anchor: "element" | "pointer",
 * }} TooltipDescriptor
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
 *   idTag: SVGElement,
 *   idTagBox: SVGElement,
 *   idTagLabel: SVGElement,
 *   deadMark: SVGElement,
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
 *   protectedRects: ReadonlyArray<Rectangle>,
 *   worldToScreen: (
 *     point: {x: number, y: number} | readonly [number, number],
 *   ) => {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 * }} ChoreographySurface
 * @typedef {{
 *   agent: JsonRecord,
 *   globalSlot: number,
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
 */

/**
 * @param {unknown} value
 * @returns {value is JsonRecord}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * @param {unknown} value
 * @returns {any[]}
 */
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * @param {unknown} value
 * @param {number} fallback
 */
function finiteNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * @param {number} x
 * @param {number} y
 * @param {number} width
 * @param {number} height
 * @returns {Rectangle}
 */
function rectangle(x, y, width, height) {
  return {
    left: x,
    top: y,
    right: x + width,
    bottom: y + height,
    width,
    height,
  };
}

/**
 * @param {unknown} frame
 * @returns {JsonRecord | null}
 */
function frameScene(frame) {
  if (!isRecord(frame)) {
    return null;
  }
  if (isRecord(frame.scene)) {
    return frame.scene;
  }
  return isRecord(frame.battlefield_scene) ? frame.battlefield_scene : null;
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
 * @param {ReadonlyMap<number, string>} [classByGlobalSlot]
 * @param {((record: JsonRecord) => TooltipDescriptor) | null} [explain]
 * @param {boolean} [withStrokeHitRegion]
 */
function renderCircleLayer(
  layer,
  records,
  className,
  tokenAttribute,
  transform,
  classByGlobalSlot = new Map(),
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
    });
    if (typeof record[tokenAttribute] === "string") {
      if (tokenAttribute === "kind") {
        circle.dataset.kind = record[tokenAttribute];
      } else {
        circle.dataset.token = record[tokenAttribute];
      }
    }
    if (Number.isInteger(record.global_slot)) {
      circle.dataset.slot = String(record.global_slot);
      const classKey = classByGlobalSlot.get(record.global_slot);
      if (classKey) {
        circle.dataset.class = classKey;
      }
    }
    if (Number.isInteger(record.source_global_slot)) {
      circle.dataset.sourceSlot = String(record.source_global_slot);
    }
    if (explain === null) {
      circles.push(circle);
      continue;
    }
    if (!withStrokeHitRegion) {
      registerTooltipOwner(circle, explain(record));
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
    if (Number.isInteger(record.global_slot)) {
      hitRegion.dataset.slot = String(record.global_slot);
    }
    owner.append(circle, hitRegion);
    registerTooltipOwner(owner, explain(record));
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
    /** @type {ReadonlyArray<Rectangle>} */
    this.choreographyProtectedRects = Object.freeze([]);
    /** @type {Readonly<{
     *   base: ReadonlyArray<Rectangle>,
     *   legality: ReadonlyArray<Rectangle>,
     *   status: ReadonlyArray<Rectangle>,
     * }>} */
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze([]),
      legality: Object.freeze([]),
      status: Object.freeze([]),
    });
    this.compactActiveCombatRequested = false;
    this.compactActiveCombat = false;

    const map = createLayer("map", { "aria-hidden": "true" });
    const aura = createLayer("aura", { "aria-hidden": "true" });
    const debugRange = createLayer("debug-range", {
      "aria-label": "Exact analysis and presentation-layout debug overlays",
    });
    this.rangeCues = createLayer("range-cues", { "aria-hidden": "true" });
    this.visibilityCues = createLayer("visibility-cues");
    this.protectedZoneCues = createLayer("protected-zone-cues", {
      "aria-hidden": "true",
    });
    debugRange.append(this.rangeCues, this.visibilityCues, this.protectedZoneCues);
    const pendingRoute = createLayer("pending-route", { "aria-hidden": "true" });
    const transientRoute = createLayer("transient-route", {
      "aria-hidden": "true",
    });
    const obstacle = createLayer("obstacle", { "aria-label": "Map obstacles" });
    const body = createLayer("body", { "aria-label": "Authorized agents" });
    const selectionLegality = createLayer("selection-legality", {
      "aria-label": "Selection and exact selected-target legality",
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
      "aria-hidden": "true",
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

    /** @type {Map<number, AgentNodes>} */
    this.agentNodes = new Map();

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
   * @param {{offline?: boolean}} [options]
   * @returns {boolean} Whether the frame contained a paintable scene.
   */
  render(frame, options = {}) {
    const scene = frameScene(frame);
    const frameRecord = isRecord(frame) ? frame : {};
    const preset =
      frameRecord.preset === "presentation" ||
      frameRecord.preset === "analysis" ||
      frameRecord.preset === "debug"
        ? frameRecord.preset
        : "analysis";
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
    this.battlefield.dataset.preset = preset;
    this.battlefield.dataset.audience = scene.audience;
    this.battlefield.setAttribute(
      "aria-label",
      `${scene.audience_badge ?? "Debugger"} battlefield, ${width} by ${height}.`,
    );
    this.empty.hidden = true;

    this.#renderMap(transform.mapBounds);
    const classByGlobalSlot = new Map(
      asArray(scene.agents)
        .filter((agent) => isRecord(agent) && Number.isInteger(agent.global_slot))
        .map((agent) => [
          Number(agent.global_slot),
          classTokenFromId(agent.class_id).cssKey,
        ]),
    );
    renderCircleLayer(
      this.layers.aura,
      preset === "presentation" ? [] : asArray(scene.aura_fields),
      "aura-field",
      "token_id",
      transform,
      new Map(),
      explainAura,
    );
    renderCircleLayer(
      this.rangeCues,
      preset === "presentation" ? [] : asArray(scene.ranges),
      "range-ring",
      "kind",
      transform,
      classByGlobalSlot,
      explainRange,
      true,
    );
    this.#renderPendingRoute(scene, transform);
    this.#renderObstacles(map, transform);
    const projectedAgents = this.#renderAgents(scene, transform);
    this.#renderDebugOverlays(scene, projectedAgents, preset === "debug");
    this.#renderStatusDocks(scene, projectedAgents, transform, {
      showLegality: preset !== "presentation",
      showModifiers: preset !== "presentation",
    });
    this.#applyCompactActiveCombatPolicy();

    this.layers.accessibleLabels.replaceChildren();
    return true;
  }

  /**
   * Prioritize accepted combat truth while a compact battlefield is actively
   * presenting transient events. Suppressed SVG owners remain in the DOM with
   * their complete authorized metadata; only their battlefield paint and
   * choreography collision reservation change.
   *
   * @param {boolean} active
   * @returns {boolean} Whether the effective compact-active state changed.
   */
  setCompactActiveCombat(active) {
    this.compactActiveCombatRequested = Boolean(active);
    return this.#applyCompactActiveCombatPolicy();
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
      .map((bounds) =>
        [bounds.left, bounds.top, bounds.right, bounds.bottom]
          .map((value) => Number(value).toFixed(3))
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
    this.visibilityCues.replaceChildren();
    this.protectedZoneCues.replaceChildren();
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
    this.layers.accessibleLabels.replaceChildren();
    this.agentNodes.clear();
    this.choreographyProtectedRects = Object.freeze([]);
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze([]),
      legality: Object.freeze([]),
      status: Object.freeze([]),
    });
    this.compactActiveCombat = false;
    this.battlefield.dataset.compactActiveCombat = "false";
    this.battlefield.removeAttribute("data-compact-active-suppressed-facts");
    this.transform = null;
  }

  /**
   * @returns {boolean}
   */
  #applyCompactActiveCombatPolicy() {
    const transform = this.transform;
    const nextActive = Boolean(
      this.compactActiveCombatRequested &&
        transform &&
        transform.viewportBounds.width <= 600 &&
        transform.viewportBounds.height <= 420,
    );
    const changed = nextActive !== this.compactActiveCombat;
    this.compactActiveCombat = nextActive;
    this.battlefield.dataset.compactActiveCombat = String(nextActive);
    if (nextActive) {
      this.battlefield.dataset.compactActiveSuppressedFacts =
        "ranges,pending-route,selected-legality,status-summaries";
    } else {
      this.battlefield.removeAttribute("data-compact-active-suppressed-facts");
    }

    const suppressedOwners = [
      this.rangeCues,
      this.layers.pendingRoute,
      this.legalityCues,
      ...this.layers.durableStatusModifier.querySelectorAll(
        '.status-dock, .required-dock-fallback-dock[data-kind="status"]',
      ),
    ];
    for (const owner of suppressedOwners) {
      if (!(owner instanceof SVGElement)) {
        continue;
      }
      owner.dataset.compactActiveSuppressed = String(nextActive);
      if (nextActive) {
        owner.setAttribute("aria-hidden", "true");
      } else if (owner !== this.rangeCues && owner !== this.layers.pendingRoute) {
        owner.removeAttribute("aria-hidden");
      }
    }
    this.#refreshChoreographyProtectedRects();
    return changed;
  }

  #refreshChoreographyProtectedRects() {
    const groups = this.choreographyProtectedRectGroups;
    this.choreographyProtectedRects = Object.freeze(
      [
        ...groups.base,
        ...(this.compactActiveCombat ? [] : groups.status),
        ...(this.compactActiveCombat ? [] : groups.legality),
      ].map((bounds) => Object.freeze({ ...bounds })),
    );
  }

  /**
   * @param {Rectangle} bounds
   */
  #renderMap(bounds) {
    this.layers.map.replaceChildren(
      svgElement("rect", {
        class: "map-boundary",
        x: bounds.left,
        y: bounds.top,
        width: bounds.width,
        height: bounds.height,
        rx: 8,
      }),
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
    const agents = asArray(scene.agents).filter(isRecord);
    const sourceAgent = agents.find(
      (agent) => agent.global_slot === route.source_global_slot,
    );
    const targetAgent = agents.find(
      (agent) => agent.global_slot === route.target_global_slot,
    );
    const geometry = createRouteGeometry(
      {
        eventId: `pending:${route.source_global_slot ?? "unknown"}:${route.target_global_slot ?? "unknown"}:${route.lane ?? "unknown"}`,
        source,
        target,
        sourceRadius: transform.worldLengthToScreen(finiteNumber(sourceAgent?.radius)),
        targetRadius: transform.worldLengthToScreen(finiteNumber(targetAgent?.radius)),
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
      "aria-hidden": "true",
    });
    path.dataset.lane = String(route.lane ?? 0);
    path.dataset.legal = String(Boolean(route.legal));
    path.dataset.sourceSlot = String(route.source_global_slot ?? "");
    path.dataset.targetSlot = String(route.target_global_slot ?? "");
    path.dataset.routeKind = geometry.kind;
    registerTooltipOwner(hitPath, explainPendingRoute(route));
    this.layers.pendingRoute.replaceChildren(path, hitPath);
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
   * @returns {ProjectedAgent[]}
   */
  #renderAgents(scene, transform) {
    const agents = asArray(scene.agents).filter(
      (agent) => isRecord(agent) && Number.isInteger(agent.global_slot),
    );
    const nextSlots = new Set(agents.map((agent) => Number(agent.global_slot)));
    for (const [slot, nodes] of this.agentNodes) {
      if (!nextSlots.has(slot)) {
        nodes.root.remove();
        nodes.selectionRoot.remove();
        this.agentNodes.delete(slot);
      }
    }

    const selection = isRecord(scene.selection) ? scene.selection : {};
    /** @type {ProjectedAgent[]} */
    const projectedAgents = [];
    for (const agent of agents) {
      const slot = Number(agent.global_slot);
      const controlled = slot === selection.controlled_global_slot;
      const selected = slot === selection.selected_global_slot;
      const center = screenPoint(agent.position, transform);
      const radius = transform.worldLengthToScreen(finiteNumber(agent.radius, 0.5));
      let nodes = this.agentNodes.get(slot);
      if (!nodes) {
        nodes = this.#createAgentNodes(slot);
        this.agentNodes.set(slot, nodes);
      }
      this.#updateAgentNodes(nodes, agent, center, radius, controlled, selected);
      registerTooltipOwner(nodes.root, explainAgent(agent, { controlled, selected }));

      // Appending an existing child reorders it without replacing its identity.
      this.layers.body.append(nodes.root);
      this.selectionCues.append(nodes.selectionRoot);
      projectedAgents.push({
        agent,
        globalSlot: slot,
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
   * Render only researcher-authorized observer visibility and clearly marked
   * presentation-layout protected zones in Debug. Neither cue changes bodies
   * or claims to be simulator geometry.
   *
   * @param {JsonRecord} scene
   * @param {ProjectedAgent[]} projectedAgents
   * @param {boolean} showDebug
   */
  #renderDebugOverlays(scene, projectedAgents, showDebug) {
    this.visibilityCues.replaceChildren();
    this.protectedZoneCues.replaceChildren();
    if (!showDebug) {
      return;
    }

    const protectedZones = projectedAgents.map((agent) => {
      const bounds = protectedBodyRect(
        {
          center: agent.center,
          radius: agent.radius,
          controlled: agent.controlled,
          selected: agent.selected,
        },
        {
          bodyPadding: 4,
          selectionAllowance: 12,
        },
      );
      return svgElement("rect", {
        class: "debug-protected-zone",
        x: bounds.left,
        y: bounds.top,
        width: bounds.width,
        height: bounds.height,
        rx: 5,
        "data-zone": "debug-protected",
        "data-slot": agent.globalSlot,
      });
    });
    this.protectedZoneCues.replaceChildren(...protectedZones);

    if (scene.audience !== "researcher") {
      return;
    }
    const visibilityCues = [];
    for (const fact of asArray(scene.observer_visibility)) {
      if (!isRecord(fact) || !Number.isInteger(fact.candidate_global_slot)) {
        continue;
      }
      const agent = projectedAgents.find(
        ({ globalSlot }) => globalSlot === fact.candidate_global_slot,
      );
      if (!agent) {
        continue;
      }
      const visible = Boolean(fact.visible);
      const group = svgElement("g", {
        class: "debug-visibility-cue",
        role: "img",
        "aria-label": `Observer id_${fact.observer_global_slot} ${visible ? "can" : "cannot"} see id_${agent.globalSlot}`,
        "data-zone": "debug-visibility",
        "data-observer-slot": fact.observer_global_slot,
        "data-candidate-slot": agent.globalSlot,
        "data-visible": visible,
      });
      const radius = agent.radius + 16;
      registerTooltipOwner(group, explainVisibility(fact));
      group.append(
        svgElement("circle", {
          class: "debug-visibility-cue__ring",
          cx: agent.center.x,
          cy: agent.center.y,
          r: radius,
        }),
        svgElement("text", {
          class: "debug-visibility-cue__label",
          x: agent.center.x + radius * 0.72,
          y: agent.center.y - radius * 0.72,
        }),
      );
      const label = group.lastElementChild;
      if (label) {
        label.textContent = visible ? "V" : "H";
      }
      visibilityCues.push(group);
    }
    this.visibilityCues.replaceChildren(...visibilityCues);
  }

  /**
   * Render the two exact selected-pair mask values without deriving legality
   * from geometry, cooldowns, class, or any other browser-visible fact.
   *
   * @param {JsonRecord} scene
   * @param {ProjectedAgent[]} projectedAgents
   * @param {ViewportTransform} transform
   * @param {Rectangle[]} reservedRects
   * @returns {Rectangle[]}
   */
  #renderSelectedLegality(scene, projectedAgents, transform, reservedRects) {
    this.legalityCues.replaceChildren();
    const legality = isRecord(scene.selected_legality) ? scene.selected_legality : null;
    if (!legality) {
      return [];
    }
    const target = projectedAgents.find(
      ({ globalSlot }) => globalSlot === legality.target_global_slot,
    );
    if (!target) {
      return [];
    }

    const layout = layoutStatusDocks(
      {
        agents: projectedAgents.map((agent) => ({
          globalSlot: agent.globalSlot,
          center: agent.center,
          radius: agent.radius,
          statuses:
            agent.globalSlot === target.globalSlot
              ? Object.freeze(["basic", "ultimate"])
              : Object.freeze([]),
          controlled: agent.controlled,
          selected: agent.selected || agent.globalSlot === legality.target_global_slot,
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
      ({ globalSlot }) => globalSlot === target.globalSlot,
    );
    if (!placement) {
      this.legalityCues.dataset.suppressedSlot = String(target.globalSlot);
      return [];
    }
    this.legalityCues.removeAttribute("data-suppressed-slot");

    const group = svgElement("g", {
      class: "legality-dock",
      role: "group",
      "aria-label": `Exact selected-target legality for id_${target.globalSlot}`,
      "data-zone": "legality",
      "data-slot": target.globalSlot,
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
        available: Boolean(legality.lane_0_available),
      },
      {
        lane: 1,
        label: "1/U",
        name: "Ultimate",
        available: Boolean(legality.lane_1_available),
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
        "aria-label": `${lane.name} lane ${lane.available ? "available" : "unavailable"}${armed ? `, armed and ${legality.armed_pair_legal ? "legal" : "illegal"}` : ""} for id_${target.globalSlot}`,
        "data-zone": "legality-pill",
        "data-lane": lane.lane,
        "data-available": lane.available,
        "data-armed": armed,
        "data-pair-legal": armed ? Boolean(legality.armed_pair_legal) : null,
      });
      registerTooltipOwner(pill, explainLegality(legality, lane.lane === 0 ? 0 : 1));
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
    return [placement.bounds];
  }

  /**
   * Lay out Python-ordered status truth and exact modifier values as separate
   * screen-space docks. This method owns collision policy, not semantics.
   *
   * @param {JsonRecord} scene
   * @param {ProjectedAgent[]} projectedAgents
   * @param {ViewportTransform} transform
   * @param {{showLegality: boolean, showModifiers: boolean}} policy
   */
  #renderStatusDocks(scene, projectedAgents, transform, policy) {
    const compactMinimumViewport =
      transform.viewportBounds.width <= 600 && transform.viewportBounds.height <= 420;
    const requiredStatusSlots = new Set(
      projectedAgents
        .filter(
          (agent) => agent.statuses.length > 0 && (agent.controlled || agent.selected),
        )
        .map(({ globalSlot }) => globalSlot),
    );
    const requiredDockRequests = [
      ...projectedAgents
        .filter(({ globalSlot }) => requiredStatusSlots.has(globalSlot))
        .map((agent) => ({
          layoutKey: `status:${agent.globalSlot}`,
          globalSlot: agent.globalSlot,
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
        if (!Number.isInteger(ticks) || Number(ticks) <= 0) {
          return [];
        }
        return [
          {
            layoutKey: `cooldown:${agent.globalSlot}`,
            globalSlot: agent.globalSlot,
            statuses: Object.freeze([
              Object.freeze({
                classId: agent.agent.class_id,
                ticks: Number(ticks),
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
          globalSlot: agent.globalSlot,
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
          globalSlot: agent.globalSlot,
          center: agent.center,
          radius: agent.radius,
          statuses: requiredStatusSlots.has(agent.globalSlot)
            ? Object.freeze([])
            : agent.statuses,
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
    const identityRects = this.#updateSelectedIdentityLayout(
      projectedAgents,
      statusLayout,
      [...cooldownRects, ...compactRequiredRects],
    );
    const legalityRects = policy.showLegality
      ? this.#renderSelectedLegality(scene, projectedAgents, transform, [
          ...identityRects,
          ...statusRects,
          ...cooldownRects,
          ...compactRequiredRects,
        ])
      : [];
    if (!policy.showLegality) {
      this.legalityCues.replaceChildren();
      this.legalityCues.removeAttribute("data-suppressed-slot");
    }
    const modifierLayout = policy.showModifiers
      ? layoutStatusDocks(
          {
            agents: projectedAgents.map((agent) => ({
              globalSlot: agent.globalSlot,
              center: agent.center,
              radius: agent.radius,
              statuses: asArray(agent.agent.modifiers),
              controlled: false,
              selected: false,
            })),
            viewport: transform.viewportBounds,
            reservedRects: [
              ...statusLayout.protectedBodies.map(({ bounds }) => bounds),
              ...identityRects,
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
      this.#renderFactDock(placement, "modifier", MODIFIER_DOCK_DIMENSIONS),
    );
    const statusNodes = statusLayout.docks.map((placement) =>
      this.#renderFactDock(placement, "status", STATUS_DOCK_DIMENSIONS),
    );
    const cooldownNodes = cooldownLayout.docks.map((placement) =>
      this.#renderCooldownDock(placement),
    );
    const compactRequiredNodes = compactRequiredDocks.map((placement) =>
      this.#renderRequiredDockFallback(placement),
    );
    this.layers.durableStatusModifier.dataset.suppressedStatusSlots =
      statusLayout.suppressedGlobalSlots.join(",");
    this.layers.durableStatusModifier.dataset.suppressedCooldownSlots =
      cooldownLayout.suppressedGlobalSlots.join(",");
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
    this.layers.durableStatusModifier.replaceChildren(
      ...modifierNodes,
      ...compactRequiredNodes,
      ...cooldownNodes,
      ...statusNodes,
    );
    this.#resolveNumericDockCellContent();
    const compactRequiredStatusRects = compactRequiredDocks
      .filter(({ layoutKey }) => layoutKey.startsWith("status:"))
      .map(({ bounds }) => bounds);
    const compactRequiredCooldownRects = compactRequiredDocks
      .filter(({ layoutKey }) => layoutKey.startsWith("cooldown:"))
      .map(({ bounds }) => bounds);
    this.choreographyProtectedRectGroups = Object.freeze({
      base: Object.freeze(
        [
          ...statusLayout.protectedBodies.map(({ bounds }) => bounds),
          ...identityRects,
          ...cooldownRects,
          ...compactRequiredCooldownRects,
          ...modifierLayout.docks.map(({ bounds }) => bounds),
        ].map((bounds) => Object.freeze({ ...bounds })),
      ),
      legality: Object.freeze(
        legalityRects.map((bounds) => Object.freeze({ ...bounds })),
      ),
      status: Object.freeze(
        [...statusRects, ...compactRequiredStatusRects].map((bounds) =>
          Object.freeze({ ...bounds }),
        ),
      ),
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
   * Keep a selected battlefield ID tag only when its compact fixed cue does
   * not collide with higher-priority status truth or any protected body.
   * The keyed roster remains the always-visible exact identity source.
   *
   * @param {ProjectedAgent[]} projectedAgents
   * @param {ReturnType<typeof layoutStatusDocks>} statusLayout
   * @param {Rectangle[]} reservedRects
   * @returns {Rectangle[]}
   */
  #updateSelectedIdentityLayout(projectedAgents, statusLayout, reservedRects) {
    for (const nodes of this.agentNodes.values()) {
      nodes.idTag.dataset.layoutSuppressed = "false";
    }
    /** @type {Rectangle[]} */
    const visibleBounds = [];
    for (const agent of projectedAgents.filter(({ selected }) => selected)) {
      const bounds = rectangle(
        agent.center.x + agent.radius * 0.45,
        agent.center.y - agent.radius - 23,
        44,
        18,
      );
      const collision = [
        ...statusLayout.docks.map(({ bounds: dockBounds }) => dockBounds),
        ...reservedRects,
        ...statusLayout.protectedBodies
          .filter(({ globalSlot }) => globalSlot !== agent.globalSlot)
          .map(({ bounds: bodyBounds }) => bodyBounds),
      ].some((other) => rectanglesIntersect(bounds, other));
      const nodes = this.agentNodes.get(agent.globalSlot);
      if (nodes) {
        nodes.idTag.dataset.layoutSuppressed = String(collision);
      }
      if (!collision) {
        visibleBounds.push(bounds);
      }
    }
    return visibleBounds;
  }

  /**
   * Render one compact, explicitly associated marker when a complete required
   * dock cannot fit. The marker never discards the authoritative payload:
   * status fallbacks expose every item through the shared overflow explanation,
   * while cooldown fallbacks retain the exact tick count.
   *
   * @param {ReturnType<typeof layoutRequiredDocks>["docks"][number]} placement
   * @returns {SVGElement}
   */
  #renderRequiredDockFallback(placement) {
    const isCooldown = placement.layoutKey.startsWith("cooldown:");
    const kind = isCooldown ? "cooldown" : "status";
    const rawItems = placement.hiddenStatuses;
    const firstItem = isRecord(rawItems[0]) ? rawItems[0] : {};
    const ticks =
      isCooldown && Number.isInteger(firstItem.ticks) && firstItem.ticks > 0
        ? Number(firstItem.ticks)
        : null;
    const explanation = isCooldown
      ? explainCooldown({
          global_slot: placement.globalSlot,
          class_id: firstItem.classId,
          ultimate_cooldown: ticks,
        })
      : explainOverflow(rawItems, "status", placement.globalSlot);
    const valueLabel = isCooldown ? `U${ticks ?? "?"}` : `S${placement.hiddenCount}`;
    const group = svgElement("g", {
      class: "required-dock-fallback-dock",
      "data-zone": "required-dock-fallback-dock",
      "data-slot": placement.globalSlot,
      "data-layout-key": placement.layoutKey,
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
      "aria-label": `${explanation.title}. ${explanation.details.join(". ")}`,
      "data-zone": isCooldown ? "ultimate-cooldown" : "status-overflow",
      "data-slot": placement.globalSlot,
      "data-layout-key": placement.layoutKey,
      "data-kind": kind,
      "data-hidden-count": placement.hiddenCount,
      "data-ticks": ticks,
      "data-compact-fallback": "true",
      "data-owner-label": `id_${placement.globalSlot}`,
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
    ownerLine.textContent = `id_${placement.globalSlot}`;
    const valueLine = svgElement("tspan", {
      class: "required-dock-fallback__value",
      x: placement.bounds.left + placement.bounds.width / 2,
      dy: "0.9em",
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
    labelText.append(ownerLine, valueLine);
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
    const group = svgElement("g", {
      class: "cooldown-dock",
      "data-zone": "cooldown-dock",
      "data-slot": placement.globalSlot,
      "data-class": classToken.cssKey,
      "data-anchor": placement.anchor,
      "data-expanded": placement.expanded,
      "data-collision-free": placement.collisionFree,
      "data-visible-count": placement.visibleCount,
      "data-hidden-count": placement.hiddenCount,
    });
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
      "aria-label": `${token.accessibleName} cooldown, ${accessibleTicks}, id_${placement.globalSlot}`,
      "data-zone": "ultimate-cooldown",
      "data-slot": placement.globalSlot,
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
    registerTooltipOwner(
      cell,
      explainCooldown({
        global_slot: placement.globalSlot,
        class_id: item.classId,
        ultimate_cooldown: ticks,
      }),
    );
    cell.append(box, iconCompartment, valueCompartment, icon, value);
    group.append(cell);
    return group;
  }

  /**
   * @param {ReturnType<typeof layoutStatusDocks>["docks"][number]} placement
   * @param {"status" | "modifier"} kind
   * @param {{cellWidth: number, cellHeight: number, cellGap: number}} dimensions
   * @returns {SVGElement}
   */
  #renderFactDock(placement, kind, dimensions) {
    const group = svgElement("g", {
      class: `${kind}-dock`,
      "data-zone": `${kind}-dock`,
      "data-slot": placement.globalSlot,
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
      const token = resolveVisualToken(kind, item.token_id, item);
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
        "aria-label": `${token.accessibleName}, ${accessibleValue}, id_${placement.globalSlot}`,
        "data-zone": `${kind}-cell`,
        "data-slot": placement.globalSlot,
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
          ? explainStatus(item, placement.globalSlot)
          : explainModifier(item, placement.globalSlot),
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
        const token = resolveVisualToken(kind, item.token_id, item);
        return kind === "status"
          ? `${token.accessibleName}, duration ${item.duration ?? "unknown"}`
          : `${token.accessibleName}, multiplier ${formatDisplayNumber(item.multiplier)}`;
      });
      const overflow = svgElement("g", {
        class: `${kind}-overflow`,
        role: "img",
        "aria-label": `${placement.overflowLabel} hidden ${kind} cues for id_${placement.globalSlot}: ${hiddenLabels.join("; ")}`,
        "data-zone": `${kind}-overflow`,
        "data-slot": placement.globalSlot,
        "data-hidden-count": placement.hiddenCount,
        "data-owner-label": `id_${placement.globalSlot}`,
      });
      registerTooltipOwner(
        overflow,
        explainOverflow(placement.hiddenStatuses, kind, placement.globalSlot),
      );
      const overflowLabel = svgElement("text", {
        class: `${kind}-overflow__label`,
        x: x + dimensions.cellWidth / 2,
        y: y + dimensions.cellHeight / 2,
      });
      if (kind === "status") {
        const ownerLine = svgElement("tspan", {
          class: "status-overflow__owner",
          x: x + dimensions.cellWidth / 2,
          dy: "-0.34em",
        });
        ownerLine.textContent = `id_${placement.globalSlot}`;
        const countLine = svgElement("tspan", {
          class: "status-overflow__count",
          x: x + dimensions.cellWidth / 2,
          dy: "0.9em",
        });
        countLine.textContent = placement.overflowLabel;
        overflowLabel.append(ownerLine, countLine);
      } else {
        overflowLabel.textContent = placement.overflowLabel;
      }
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
   * @param {number} slot
   * @returns {AgentNodes}
   */
  #createAgentNodes(slot) {
    const root = svgElement("g", {
      class: "agent",
      tabindex: "-1",
      role: "img",
      "data-slot": slot,
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
    const idTag = svgElement("g", {
      class: "agent-id-tag",
      "data-zone": "identity",
      "aria-hidden": "true",
    });
    const idTagBox = svgElement("rect", {
      class: "agent-id-tag-box",
      rx: 5,
    });
    const idTagLabel = svgElement("text", { class: "agent-id-tag-label" });
    idTag.append(idTagBox, idTagLabel);
    root.append(
      body,
      teamRing,
      teamMarker,
      healthTrack,
      health,
      classIcon,
      classLetter,
      deadMark,
      idTag,
    );

    const selectionRoot = svgElement("g", {
      class: "agent-selection",
      "data-slot": slot,
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
      idTag,
      idTagBox,
      idTagLabel,
      deadMark,
      selectionRoot,
      controlledHalo,
      selectedReticle,
    };
  }

  /**
   * @param {AgentNodes} nodes
   * @param {JsonRecord} agent
   * @param {{x: number, y: number}} center
   * @param {number} radius
   * @param {boolean} controlled
   * @param {boolean} selected
   */
  #updateAgentNodes(nodes, agent, center, radius, controlled, selected) {
    const classToken = classTokenFromId(agent.class_id);
    const teamToken = teamTokenFromId(agent.team_id);
    const healthRadius = Math.max(radius - 4, radius * 0.7);
    const healthRatio = Math.max(
      0,
      Math.min(
        1,
        finiteNumber(agent.current_health) /
          Math.max(finiteNumber(agent.max_health, 1), Number.EPSILON),
      ),
    );

    nodes.root.dataset.team = teamToken.cssKey;
    nodes.root.dataset.class = classToken.cssKey;
    nodes.root.dataset.alive = String(Boolean(agent.alive));
    nodes.root.dataset.controlled = String(controlled);
    nodes.root.dataset.selected = String(selected);
    nodes.root.setAttribute(
      "aria-label",
      [
        `id_${agent.global_slot}`,
        classToken.label,
        teamToken.label,
        `health ${formatDisplayNumber(agent.current_health)} of ${formatDisplayNumber(agent.max_health)}`,
        agent.alive ? "alive" : "dead",
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

    const tagWidth = 44;
    const tagHeight = 18;
    const tagX = center.x + radius * 0.45;
    const tagY = center.y - radius - tagHeight - 5;
    setAttributes(nodes.idTagBox, {
      x: tagX,
      y: tagY,
      width: tagWidth,
      height: tagHeight,
      rx: 5,
    });
    setAttributes(nodes.idTagLabel, {
      x: tagX + tagWidth / 2,
      y: tagY + tagHeight / 2,
    });
    nodes.idTagLabel.textContent = `id_${agent.global_slot}`;

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
      hidden: selected ? null : "",
    });
  }
}
