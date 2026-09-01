import { authoringObjects, mapContent } from "./authoring-model.js";
import { createSvgIcon } from "./icons.js";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const DEFAULT_MAP_WIDTH = 20;
const DEFAULT_MAP_HEIGHT = 10;
const DEFAULT_AGENT_BODY_RADIUS = 0.45;
const DEFAULT_SPAWN_PAD_RADIUS = 0.5;
const AUTHORING_AGENT_CLASSES = Object.freeze(
  new Set(["mage", "warrior", "hunter", "rogue", "priest"]),
);

/** @param {string} name @param {Record<string, string | number>} attributes */
function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NAMESPACE, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

/** @param {unknown} mapWidth @param {unknown} mapHeight */
export function authoringMapDimensions(mapWidth, mapHeight) {
  const width = Number(mapWidth);
  const height = Number(mapHeight);
  if (![width, height].every(Number.isFinite) || width <= 0 || height <= 0) {
    return null;
  }
  return Object.freeze({ width, height });
}

/** @param {number} gridSpacing */
export function authoringGridPattern(gridSpacing) {
  const spacing = Number(gridSpacing);
  if (!Number.isFinite(spacing) || spacing <= 0) {
    throw new RangeError("Authoring grid spacing must be positive.");
  }
  return Object.freeze({
    width: spacing,
    height: spacing,
    path: `M ${spacing} 0 H 0 V ${spacing}`,
  });
}

/** @param {any} draft */
export function authoringPaintObjects(draft) {
  const objects = authoringObjects(draft);
  const pads = objects.filter((object) => object.kind === "spawn_pad");
  const obstacles = objects.filter(
    (object) => object.kind === "wall" || object.kind === "pillar",
  );
  const agents = objects.filter((object) => object.kind === "agent");
  return Object.freeze([...pads, ...obstacles, ...agents]);
}

/** @param {any} object @param {any} catalog */
export function authoringAgentBodyRadius(object, catalog) {
  const className = object?.roster?.class_name;
  const mechanics = Array.isArray(catalog?.class_mechanics)
    ? catalog.class_mechanics.find(
        (/** @type {any} */ candidate) =>
          typeof candidate?.class_name === "string" &&
          typeof className === "string" &&
          candidate.class_name.toLowerCase() === className.toLowerCase(),
      )
    : null;
  const radius = Number(mechanics?.body_radius);
  return Number.isFinite(radius) && radius > 0 ? radius : DEFAULT_AGENT_BODY_RADIUS;
}

/** @param {any} catalog */
export function authoringSpawnPadRadius(catalog) {
  const radii = Array.isArray(catalog?.class_mechanics)
    ? catalog.class_mechanics
        .map((/** @type {any} */ mechanics) => Number(mechanics?.body_radius))
        .filter((/** @type {number} */ radius) => Number.isFinite(radius) && radius > 0)
    : [];
  return radii.length > 0 ? Math.max(...radii) : DEFAULT_SPAWN_PAD_RADIUS;
}

/** @param {any} object */
export function authoringAgentVisual(object) {
  const requestedClass =
    typeof object?.roster?.class_name === "string"
      ? object.roster.class_name.trim().toLowerCase()
      : "unknown";
  const className = AUTHORING_AGENT_CLASSES.has(requestedClass)
    ? requestedClass
    : "unknown";
  return Object.freeze({
    className,
    glyphKey: className === "unknown" ? "unknown" : `class-${className}`,
    alive: object?.state?.alive === true,
  });
}

/** @param {unknown} camera @param {unknown} mapWidth @param {unknown} mapHeight */
export function normalizeAuthoringCamera(camera, mapWidth, mapHeight) {
  const dimensions =
    authoringMapDimensions(mapWidth, mapHeight) ??
    Object.freeze({ width: DEFAULT_MAP_WIDTH, height: DEFAULT_MAP_HEIGHT });
  const fallback = { x: 0, y: 0, ...dimensions };
  if (!camera || typeof camera !== "object") {
    return Object.freeze(fallback);
  }
  const candidate = /** @type {Record<string, unknown>} */ (camera);
  const width = Number(candidate.width);
  const height = Number(candidate.height);
  const x = Number(candidate.x);
  const y = Number(candidate.y);
  if (![width, height, x, y].every(Number.isFinite) || width <= 0 || height <= 0) {
    return Object.freeze(fallback);
  }
  return Object.freeze({ x, y, width, height });
}

/**
 * Convert one client coordinate through an SVG meet-fit and viewBox. Returned
 * Y uses the simulator's upward-positive world convention.
 *
 * @param {{left: number, top: number, width: number, height: number}} bounds
 * @param {{x: number, y: number, width: number, height: number}} camera
 * @param {number} mapHeight
 * @param {number} clientX
 * @param {number} clientY
 */
export function authoringClientPointToWorld(
  bounds,
  camera,
  mapHeight,
  clientX,
  clientY,
) {
  if (bounds.width <= 0 || bounds.height <= 0) {
    throw new RangeError("Authoring SVG bounds must be positive.");
  }
  const scale = Math.min(bounds.width / camera.width, bounds.height / camera.height);
  const fittedWidth = camera.width * scale;
  const fittedHeight = camera.height * scale;
  const left = bounds.left + (bounds.width - fittedWidth) / 2;
  const top = bounds.top + (bounds.height - fittedHeight) / 2;
  const visualX = camera.x + (clientX - left) / scale;
  const visualY = camera.y + (clientY - top) / scale;
  return Object.freeze({ x: visualX, y: mapHeight - visualY });
}

/**
 * Zoom around a client-resolved world anchor while keeping the map aspect
 * ratio and leaving authored content untouched.
 *
 * @param {{x: number, y: number, width: number, height: number}} camera
 * @param {number} mapWidth
 * @param {number} mapHeight
 * @param {{x: number, y: number}} worldAnchor
 * @param {number} factor
 */
export function zoomAuthoringCamera(camera, mapWidth, mapHeight, worldAnchor, factor) {
  const zoom = Number(factor);
  if (!Number.isFinite(zoom) || zoom <= 0) {
    throw new RangeError("Authoring zoom factor must be positive.");
  }
  const minimumWidth = mapWidth / 8;
  const maximumWidth = mapWidth * 2;
  const nextWidth = Math.min(maximumWidth, Math.max(minimumWidth, camera.width * zoom));
  const ratio = nextWidth / camera.width;
  const nextHeight = camera.height * ratio;
  const visualAnchorY = mapHeight - worldAnchor.y;
  const anchorFractionX = (worldAnchor.x - camera.x) / camera.width;
  const anchorFractionY = (visualAnchorY - camera.y) / camera.height;
  return Object.freeze({
    x: worldAnchor.x - anchorFractionX * nextWidth,
    y: visualAnchorY - anchorFractionY * nextHeight,
    width: nextWidth,
    height: nextHeight,
  });
}

/**
 * @param {{x: number, y: number, width: number, height: number}} camera
 * @param {number} deltaVisualX
 * @param {number} deltaVisualY
 */
export function panAuthoringCamera(camera, deltaVisualX, deltaVisualY) {
  return Object.freeze({
    ...camera,
    x: camera.x + deltaVisualX,
    y: camera.y + deltaVisualY,
  });
}

/** @param {SVGSVGElement} svg @param {any} draft @param {string | null} selectedId @param {any} camera @param {number} gridSpacing @param {any} catalog */
export function renderAuthoringSvg(
  svg,
  draft,
  selectedId,
  camera,
  gridSpacing,
  catalog = null,
) {
  const map = mapContent(draft);
  const dimensions = authoringMapDimensions(map.width, map.height);
  svg.replaceChildren();
  if (dimensions === null) {
    const fallbackCamera = normalizeAuthoringCamera(null, null, null);
    svg.dataset.renderState = "invalid-dimensions";
    svg.setAttribute(
      "viewBox",
      `${fallbackCamera.x} ${fallbackCamera.y} ${fallbackCamera.width} ${fallbackCamera.height}`,
    );
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.append(
      svgElement("rect", {
        class: "authoring-svg-invalid",
        x: 0,
        y: 0,
        width: fallbackCamera.width,
        height: fallbackCamera.height,
      }),
    );
    const message = svgElement("text", {
      class: "authoring-svg-invalid-message",
      x: fallbackCamera.width / 2,
      y: fallbackCamera.height / 2,
      "text-anchor": "middle",
    });
    message.textContent = "Map dimensions must be positive finite numbers.";
    svg.append(message);
    return fallbackCamera;
  }

  const resolvedCamera = normalizeAuthoringCamera(
    camera,
    dimensions.width,
    dimensions.height,
  );
  const gridPattern = authoringGridPattern(gridSpacing);
  svg.dataset.renderState = "ready";
  svg.setAttribute(
    "viewBox",
    `${resolvedCamera.x} ${resolvedCamera.y} ${resolvedCamera.width} ${resolvedCamera.height}`,
  );
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const defs = svgElement("defs");
  const clipPath = svgElement("clipPath", { id: "authoring-map-clip" });
  clipPath.append(
    svgElement("rect", {
      x: 0,
      y: 0,
      width: dimensions.width,
      height: dimensions.height,
    }),
  );
  const pattern = svgElement("pattern", {
    id: "authoring-grid-pattern",
    width: gridPattern.width,
    height: gridPattern.height,
    patternUnits: "userSpaceOnUse",
  });
  pattern.append(
    svgElement("path", {
      class: "authoring-svg-grid-line",
      d: gridPattern.path,
    }),
  );
  defs.append(clipPath, pattern);
  svg.append(defs);

  svg.append(
    svgElement("rect", {
      class: "authoring-svg-map",
      x: 0,
      y: 0,
      width: dimensions.width,
      height: dimensions.height,
    }),
  );
  svg.append(
    svgElement("rect", {
      class: "authoring-svg-grid",
      x: 0,
      y: 0,
      width: dimensions.width,
      height: dimensions.height,
      fill: "url(#authoring-grid-pattern)",
      "clip-path": "url(#authoring-map-clip)",
    }),
  );

  const objectLayer = svgElement("g");
  for (const object of authoringPaintObjects(draft)) {
    const visualY = dimensions.height - object.y;
    const agentVisual = object.kind === "agent" ? authoringAgentVisual(object) : null;
    let shape;
    if (object.kind === "wall") {
      shape = svgElement("rect", {
        x: object.x - object.obstacle.width / 2,
        y: visualY - object.obstacle.height / 2,
        width: object.obstacle.width,
        height: object.obstacle.height,
        transform: `rotate(${-object.obstacle.rotation_degrees} ${object.x} ${visualY})`,
      });
    } else if (object.kind === "pillar") {
      shape = svgElement("circle", {
        cx: object.x,
        cy: visualY,
        r: object.obstacle.radius,
      });
    } else if (object.kind === "spawn_pad") {
      shape = svgElement("circle", {
        cx: object.x,
        cy: visualY,
        r: authoringSpawnPadRadius(catalog),
      });
    } else {
      shape = svgElement("circle", {
        cx: object.x,
        cy: visualY,
        r: authoringAgentBodyRadius(object, catalog),
      });
    }
    shape.classList.add("authoring-svg-object");
    shape.dataset.objectId = object.object_id;
    shape.dataset.kind = object.kind;
    if (object.team) {
      shape.dataset.team = object.team === "A" ? "0" : "1";
    }
    if (agentVisual) {
      shape.dataset.class = agentVisual.className;
      shape.dataset.alive = String(agentVisual.alive);
    }
    shape.setAttribute("aria-selected", String(object.object_id === selectedId));
    shape.setAttribute("tabindex", "-1");
    objectLayer.append(shape);

    if (agentVisual) {
      const radius = authoringAgentBodyRadius(object, catalog);
      const iconSize = radius * 1.15;
      const icon = createSvgIcon(svg.ownerDocument, agentVisual.glyphKey, {
        className: "authoring-svg-agent-class",
      });
      icon.dataset.class = agentVisual.className;
      icon.setAttribute("x", String(object.x - iconSize / 2));
      icon.setAttribute("y", String(visualY - iconSize / 2));
      icon.setAttribute("width", String(iconSize));
      icon.setAttribute("height", String(iconSize));
      objectLayer.append(icon);
      if (!agentVisual.alive) {
        const markRadius = radius * 0.72;
        objectLayer.append(
          svgElement("path", {
            class: "authoring-svg-agent-dead-mark",
            d: `M ${object.x - markRadius} ${visualY - markRadius} L ${object.x + markRadius} ${visualY + markRadius} M ${object.x + markRadius} ${visualY - markRadius} L ${object.x - markRadius} ${visualY + markRadius}`,
          }),
        );
      }
    }

    if (object.kind === "spawn_pad" || object.kind === "agent") {
      const label = svgElement("text", {
        x: object.x,
        y:
          object.kind === "agent"
            ? visualY + authoringAgentBodyRadius(object, catalog) + 0.3
            : visualY + 0.13,
        "text-anchor": "middle",
        "font-size": 0.32,
        fill: "#f4f7fb",
        "pointer-events": "none",
      });
      label.textContent =
        object.kind === "agent"
          ? `${object.team}${object.roster.team_local_slot}`
          : `${object.team}${object.pad.team_local_slot}`;
      objectLayer.append(label);
    }
  }
  svg.append(objectLayer);
  return resolvedCamera;
}
