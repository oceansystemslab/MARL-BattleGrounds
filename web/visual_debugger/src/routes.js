const EPSILON = 1e-6;

/**
 * @typedef {{x: number, y: number}} RoutePoint
 * @typedef {{
 *   eventId: string,
 *   sourceGlobalSlot: number,
 *   targetGlobalSlot: number,
 *   source: RoutePoint | ReadonlyArray<number>,
 *   target: RoutePoint | ReadonlyArray<number>,
 *   sourceRadius?: number,
 *   targetRadius?: number,
 * }} RouteRecord
 * @typedef {{
 *   kind: "curve",
 *   start: RoutePoint,
 *   end: RoutePoint,
 *   control: RoutePoint,
 *   sourcePortAngle: number,
 *   targetPortAngle: number,
 *   offset: number,
 *   path: string,
 * } | {
 *   kind: "local_arc",
 *   start: RoutePoint,
 *   end: RoutePoint,
 *   center: RoutePoint,
 *   arcRadius: number,
 *   sourcePortAngle: number,
 *   targetPortAngle: number,
 *   offset: number,
 *   sweep: 0 | 1,
 *   path: string,
 * }} RouteGeometry
 */

/**
 * Clip a source-target segment to the outside of two circular body zones.
 *
 * @param {RoutePoint | ReadonlyArray<number>} source
 * @param {RoutePoint | ReadonlyArray<number>} target
 * @param {number} sourceRadius
 * @param {number} targetRadius
 * @param {number} [endpointGap]
 */
export function clipRouteEndpoints(
  source,
  target,
  sourceRadius,
  targetRadius,
  endpointGap = 0,
) {
  const startCenter = point(source, "source");
  const endCenter = point(target, "target");
  const sourceExtent =
    nonNegative(sourceRadius, "sourceRadius") + nonNegative(endpointGap, "endpointGap");
  const targetExtent =
    nonNegative(targetRadius, "targetRadius") + nonNegative(endpointGap, "endpointGap");
  const deltaX = endCenter.x - startCenter.x;
  const deltaY = endCenter.y - startCenter.y;
  const distance = Math.hypot(deltaX, deltaY);
  if (distance <= EPSILON) {
    return Object.freeze({
      start: startCenter,
      end: endCenter,
      distance,
      unit: frozenPoint(1, 0),
    });
  }
  const unit = frozenPoint(deltaX / distance, deltaY / distance);
  return Object.freeze({
    start: frozenPoint(
      startCenter.x + unit.x * Math.min(sourceExtent, distance / 2),
      startCenter.y + unit.y * Math.min(sourceExtent, distance / 2),
    ),
    end: frozenPoint(
      endCenter.x - unit.x * Math.min(targetExtent, distance / 2),
      endCenter.y - unit.y * Math.min(targetExtent, distance / 2),
    ),
    distance,
    unit,
  });
}

/**
 * Construct one presentation-only clipped curve or local arc.
 *
 * @param {{
 *   eventId: string,
 *   source: RoutePoint | ReadonlyArray<number>,
 *   target: RoutePoint | ReadonlyArray<number>,
 *   sourceRadius?: number,
 *   targetRadius?: number,
 *   offset?: number,
 * }} input
 * @param {{endpointGap?: number, localArcPadding?: number}} [options]
 * @returns {Readonly<RouteGeometry>}
 */
export function createRouteGeometry(input, options = {}) {
  if (!input || typeof input !== "object") {
    throw new TypeError("route geometry input must be an object.");
  }
  const eventId = identifier(input.eventId, "eventId");
  const source = point(input.source, "source");
  const target = point(input.target, "target");
  const sourceRadius = nonNegative(input.sourceRadius ?? 0, "sourceRadius");
  const targetRadius = nonNegative(input.targetRadius ?? 0, "targetRadius");
  const offset = finite(input.offset ?? 0, "offset");
  const endpointGap = nonNegative(options.endpointGap ?? 3, "endpointGap");
  const localArcPadding = nonNegative(options.localArcPadding ?? 8, "localArcPadding");
  const clipped = clipRouteEndpoints(
    source,
    target,
    sourceRadius,
    targetRadius,
    endpointGap,
  );

  const visibleSegmentLength = Math.hypot(
    clipped.end.x - clipped.start.x,
    clipped.end.y - clipped.start.y,
  );
  if (visibleSegmentLength <= EPSILON) {
    return localArc({
      eventId,
      center: frozenPoint((source.x + target.x) / 2, (source.y + target.y) / 2),
      radius: Math.max(sourceRadius, targetRadius, 1) + localArcPadding,
      offset,
    });
  }

  const normal = frozenPoint(-clipped.unit.y, clipped.unit.x);
  const control = frozenPoint(
    (clipped.start.x + clipped.end.x) / 2 + normal.x * offset,
    (clipped.start.y + clipped.end.y) / 2 + normal.y * offset,
  );
  const sourcePortAngle = Math.atan2(clipped.unit.y, clipped.unit.x);
  const targetPortAngle = normalizeAngle(sourcePortAngle + Math.PI);
  return Object.freeze({
    kind: "curve",
    start: clipped.start,
    end: clipped.end,
    control,
    sourcePortAngle,
    targetPortAngle,
    offset,
    path: `M ${number(clipped.start.x)} ${number(clipped.start.y)} Q ${number(control.x)} ${number(control.y)} ${number(clipped.end.x)} ${number(clipped.end.y)}`,
  });
}

/**
 * Assign stable curve offsets to a set of accepted/presentation routes.
 *
 * Grouping uses only public source/target identity. The returned curvature is
 * a collision-separation convention and never a simulator trajectory.
 *
 * @param {ReadonlyArray<RouteRecord>} records
 * @param {{
 *   spacing?: number,
 *   endpointGap?: number,
 *   localArcPadding?: number,
 * }} [options]
 */
export function layoutRouteSet(records, options = {}) {
  if (!Array.isArray(records)) {
    throw new TypeError("route records must be an array.");
  }
  const spacing = positive(options.spacing ?? 18, "spacing");
  const normalized = records.map(normalizeRouteRecord);
  const eventIds = new Set();
  for (const record of normalized) {
    if (eventIds.has(record.eventId)) {
      throw new RangeError(`duplicate route eventId ${record.eventId}.`);
    }
    eventIds.add(record.eventId);
  }

  /** @type {Map<string, ReturnType<typeof normalizeRouteRecord>[]>} */
  const groups = new Map();
  for (const record of normalized) {
    const groupKey = pairKey(record.sourceGlobalSlot, record.targetGlobalSlot);
    const group = groups.get(groupKey) ?? [];
    group.push(record);
    groups.set(groupKey, group);
  }

  const layouts = [];
  for (const groupKey of [...groups.keys()].sort()) {
    const group = groups.get(groupKey) ?? [];
    /** @type {Map<string, ReturnType<typeof normalizeRouteRecord>[]>} */
    const directions = new Map();
    for (const record of group) {
      const directionKey = `${record.sourceGlobalSlot}->${record.targetGlobalSlot}`;
      const direction = directions.get(directionKey) ?? [];
      direction.push(record);
      directions.set(directionKey, direction);
    }
    const reciprocal = directions.size > 1;
    for (const directionKey of [...directions.keys()].sort()) {
      const direction = [...(directions.get(directionKey) ?? [])].sort((a, b) =>
        a.eventId.localeCompare(b.eventId),
      );
      const offsets = reciprocal
        ? direction.map((_, index) => spacing * (index + 1))
        : centeredOffsets(direction.length, spacing);
      for (const [routeIndex, record] of direction.entries()) {
        const offset = offsets[routeIndex];
        const geometry = createRouteGeometry(
          {
            eventId: record.eventId,
            source: record.source,
            target: record.target,
            sourceRadius: record.sourceRadius,
            targetRadius: record.targetRadius,
            offset,
          },
          options,
        );
        layouts.push(
          Object.freeze({
            eventId: record.eventId,
            sourceGlobalSlot: record.sourceGlobalSlot,
            targetGlobalSlot: record.targetGlobalSlot,
            groupKey,
            directionKey,
            routeIndex,
            ...geometry,
          }),
        );
      }
    }
  }
  return Object.freeze(layouts.sort((a, b) => a.eventId.localeCompare(b.eventId)));
}

/**
 * @param {RouteRecord} record
 */
function normalizeRouteRecord(record) {
  if (!record || typeof record !== "object") {
    throw new TypeError("each route record must be an object.");
  }
  return Object.freeze({
    eventId: identifier(record.eventId, "eventId"),
    sourceGlobalSlot: slot(record.sourceGlobalSlot, "sourceGlobalSlot"),
    targetGlobalSlot: slot(record.targetGlobalSlot, "targetGlobalSlot"),
    source: point(record.source, "source"),
    target: point(record.target, "target"),
    sourceRadius: nonNegative(record.sourceRadius ?? 0, "sourceRadius"),
    targetRadius: nonNegative(record.targetRadius ?? 0, "targetRadius"),
  });
}

/**
 * @param {number} sourceSlot
 * @param {number} targetSlot
 */
function pairKey(sourceSlot, targetSlot) {
  return sourceSlot <= targetSlot
    ? `${sourceSlot}:${targetSlot}`
    : `${targetSlot}:${sourceSlot}`;
}

/**
 * @param {number} count
 * @param {number} spacing
 */
function centeredOffsets(count, spacing) {
  return Object.freeze(
    Array.from({ length: count }, (_, index) => (index - (count - 1) / 2) * spacing),
  );
}

/**
 * @param {{
 *   eventId: string,
 *   center: RoutePoint,
 *   radius: number,
 *   offset: number,
 * }} input
 * @returns {Readonly<RouteGeometry>}
 */
function localArc(input) {
  const orientation = hashFraction(input.eventId) * Math.PI * 2;
  const arcRadius = input.radius + Math.abs(input.offset) * 0.2;
  const halfSweep = 0.72;
  const startAngle = orientation - halfSweep;
  const endAngle = orientation + halfSweep;
  const start = frozenPoint(
    input.center.x + Math.cos(startAngle) * arcRadius,
    input.center.y + Math.sin(startAngle) * arcRadius,
  );
  const end = frozenPoint(
    input.center.x + Math.cos(endAngle) * arcRadius,
    input.center.y + Math.sin(endAngle) * arcRadius,
  );
  const sweep = input.offset < 0 ? 0 : 1;
  return Object.freeze({
    kind: "local_arc",
    start,
    end,
    center: input.center,
    arcRadius,
    sourcePortAngle: startAngle,
    targetPortAngle: endAngle,
    offset: input.offset,
    sweep,
    path: `M ${number(start.x)} ${number(start.y)} A ${number(arcRadius)} ${number(arcRadius)} 0 0 ${sweep} ${number(end.x)} ${number(end.y)}`,
  });
}

/**
 * @param {string} value
 */
function hashFraction(value) {
  let hash = 2166136261;
  for (const character of value) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 2 ** 32;
}

/**
 * @param {unknown} value
 * @param {string} name
 */
function identifier(value, name) {
  if (typeof value !== "string" || !value.trim()) {
    throw new TypeError(`${name} must be a non-empty string.`);
  }
  return value.trim();
}

/**
 * @param {unknown} value
 * @param {string} name
 */
function slot(value, name) {
  if (!Number.isInteger(value) || Number(value) < 0) {
    throw new RangeError(`${name} must be a non-negative integer.`);
  }
  return Number(value);
}

/**
 * @param {unknown} value
 * @param {string} name
 */
function point(value, name) {
  if (Array.isArray(value) && value.length >= 2) {
    return frozenPoint(finite(value[0], `${name}.x`), finite(value[1], `${name}.y`));
  }
  if (value && typeof value === "object") {
    const record = /** @type {{x?: unknown, y?: unknown}} */ (value);
    return frozenPoint(finite(record.x, `${name}.x`), finite(record.y, `${name}.y`));
  }
  throw new TypeError(`${name} must contain finite x and y coordinates.`);
}

/**
 * @param {unknown} value
 * @param {string} name
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
 */
function nonNegative(value, name) {
  const result = finite(value, name);
  if (result < 0) {
    throw new RangeError(`${name} must be non-negative.`);
  }
  return result;
}

/**
 * @param {unknown} value
 * @param {string} name
 */
function positive(value, name) {
  const result = finite(value, name);
  if (result <= 0) {
    throw new RangeError(`${name} must be positive.`);
  }
  return result;
}

/**
 * @param {number} x
 * @param {number} y
 * @returns {RoutePoint}
 */
function frozenPoint(x, y) {
  return Object.freeze({ x, y });
}

/**
 * @param {number} angle
 */
function normalizeAngle(angle) {
  const fullTurn = Math.PI * 2;
  return ((angle % fullTurn) + fullTurn) % fullTurn;
}

/**
 * Keep path strings stable while retaining sub-pixel precision.
 *
 * @param {number} value
 */
function number(value) {
  return Number(value.toFixed(4));
}
