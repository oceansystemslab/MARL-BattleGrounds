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
