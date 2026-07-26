const EPSILON = 1e-9;
const STATUS_DOCK_SEARCH_LIMIT = 100_000;

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
 * }} StatusDockOptions
 * @typedef {{
 *   agents: ReadonlyArray<StatusDockAgent>,
 *   viewport: Rectangle,
 *   reservedRects?: ReadonlyArray<Rectangle>,
 * }} StatusDockLayoutInput
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
 * @returns {ReadonlyArray<StatusDockPlacement> | null}
 */
function searchStatusDockPlacements(inputs) {
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
    if (visited >= STATUS_DOCK_SEARCH_LIMIT) {
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
      if (visited >= STATUS_DOCK_SEARCH_LIMIT) {
        return null;
      }
    }
    return null;
  }

  return visit(0, [], []);
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
  const forceExpanded =
    !capacityExceeded &&
    (input.agent.required || input.agent.controlled || input.agent.selected);
  const visibleCounts = [expandedVisibleCount];
  if (!forceExpanded) {
    const maximumCollapsedVisible = Math.min(totalCount - 1, STATUS_DOCK_CAPACITY - 1);
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
