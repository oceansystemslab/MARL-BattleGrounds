const MIN_VISIBLE_SEGMENT_FRACTION = 0.2;
const DEFAULT_ROUTE_MARKER_PADDING = 14;

/**
 * @typedef {{x: number, y: number}} RoutePoint
 * @typedef {{
 *   left: number,
 *   top: number,
 *   right: number,
 *   bottom: number,
 * }} RouteViewportBounds
 * @typedef {{
 *   spacing?: number,
 *   endpointGap?: number,
 *   localArcPadding?: number,
 *   viewportBounds?: RouteViewportBounds,
 *   routeMarkerPadding?: number,
 *   markerProgress?: number,
 * }} RouteLayoutOptions
 * @typedef {{
 *   eventId: string,
 *   sourceGlobalSlot: number,
 *   targetGlobalSlot: number,
 *   source: RoutePoint | ReadonlyArray<number>,
 *   target: RoutePoint | ReadonlyArray<number>,
 *   sourceRadius?: number,
 *   targetRadius?: number,
 *   sourceEndpointGap?: number,
 *   targetEndpointGap?: number,
 *   prioritizeTargetClearance?: boolean,
 * }} RouteRecord
 * @typedef {{
 *   kind: "curve",
 *   start: RoutePoint,
 *   end: RoutePoint,
 *   control: RoutePoint,
 *   sourcePortAngle: number,
 *   targetPortAngle: number,
 *   offset: number,
 *   close: boolean,
 *   path: string,
 *   markerProgress: number,
 * } | {
 *   kind: "local_arc",
 *   start: RoutePoint,
 *   end: RoutePoint,
 *   center: RoutePoint,
 *   arcRadius: number,
 *   sourcePortAngle: number,
 *   targetPortAngle: number,
 *   offset: number,
 *   close: boolean,
 *   sweep: 0 | 1,
 *   path: string,
 *   markerProgress: number,
 * } | {
 *   kind: "polyline",
 *   start: RoutePoint,
 *   end: RoutePoint,
 *   points: ReadonlyArray<RoutePoint>,
 *   sourcePortAngle: number,
 *   targetPortAngle: number,
 *   offset: number,
 *   close: boolean,
 *   path: string,
 *   markerProgress: number,
 * }} RouteGeometry
 */

/**
 * Clip a source-target segment toward two circular body boundaries.
 *
 * When the requested body extents overlap, compress both clips proportionally
 * by default. Dense convergence may instead preserve recipient clearance first;
 * either policy leaves a visible segment on the ordered source-target bearing.
 *
 * @param {RoutePoint | ReadonlyArray<number>} source
 * @param {RoutePoint | ReadonlyArray<number>} target
 * @param {number} sourceRadius
 * @param {number} targetRadius
 * @param {number} [endpointGap]
 * @param {number} [targetEndpointGap]
 * @param {boolean} [prioritizeTargetClearance]
 */
export function clipRouteEndpoints(
  source,
  target,
  sourceRadius,
  targetRadius,
  endpointGap = 0,
  targetEndpointGap = endpointGap,
  prioritizeTargetClearance = false,
) {
  const startCenter = point(source, "source");
  const endCenter = point(target, "target");
  const sourceExtent =
    nonNegative(sourceRadius, "sourceRadius") + nonNegative(endpointGap, "endpointGap");
  const targetExtent =
    nonNegative(targetRadius, "targetRadius") +
    nonNegative(targetEndpointGap, "targetEndpointGap");
  const deltaX = endCenter.x - startCenter.x;
  const deltaY = endCenter.y - startCenter.y;
  const distance = Math.hypot(deltaX, deltaY);
  if (distance === 0) {
    return Object.freeze({
      start: startCenter,
      end: endCenter,
      distance,
      unit: frozenPoint(1, 0),
    });
  }
  const unit = frozenPoint(deltaX / distance, deltaY / distance);
  const requestedExtent = sourceExtent + targetExtent;
  const maximumCombinedClip = distance * (1 - MIN_VISIBLE_SEGMENT_FRACTION);
  const clippedTargetExtent =
    prioritizeTargetClearance && requestedExtent > maximumCombinedClip
      ? Math.min(targetExtent, maximumCombinedClip)
      : targetExtent *
        (requestedExtent > maximumCombinedClip
          ? maximumCombinedClip / requestedExtent
          : 1);
  const clippedSourceExtent =
    prioritizeTargetClearance && requestedExtent > maximumCombinedClip
      ? Math.max(0, maximumCombinedClip - clippedTargetExtent)
      : sourceExtent *
        (requestedExtent > maximumCombinedClip
          ? maximumCombinedClip / requestedExtent
          : 1);
  return Object.freeze({
    start: frozenPoint(
      startCenter.x + unit.x * clippedSourceExtent,
      startCenter.y + unit.y * clippedSourceExtent,
    ),
    end: frozenPoint(
      endCenter.x - unit.x * clippedTargetExtent,
      endCenter.y - unit.y * clippedTargetExtent,
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
 *   sourceEndpointGap?: number,
 *   targetEndpointGap?: number,
 *   prioritizeTargetClearance?: boolean,
 *   offset?: number,
 * }} input
 * @param {RouteLayoutOptions} [options]
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
  const sourceEndpointGap = nonNegative(
    input.sourceEndpointGap ?? endpointGap,
    "sourceEndpointGap",
  );
  const targetEndpointGap = nonNegative(
    input.targetEndpointGap ?? endpointGap,
    "targetEndpointGap",
  );
  const prioritizeTargetClearance =
    input.prioritizeTargetClearance === undefined
      ? false
      : boolean(input.prioritizeTargetClearance, "prioritizeTargetClearance");
  const localArcPadding = nonNegative(options.localArcPadding ?? 8, "localArcPadding");
  const viewportBounds = optionalViewportBounds(options.viewportBounds);
  const routeMarkerPadding = nonNegative(
    options.routeMarkerPadding ?? DEFAULT_ROUTE_MARKER_PADDING,
    "routeMarkerPadding",
  );
  const requestedMarkerProgress =
    options.markerProgress === undefined
      ? null
      : unitInterval(options.markerProgress, "markerProgress");
  const clipped = clipRouteEndpoints(
    source,
    target,
    sourceRadius,
    targetRadius,
    sourceEndpointGap,
    targetEndpointGap,
    prioritizeTargetClearance,
  );

  if (clipped.distance === 0) {
    return localArc({
      eventId,
      center: frozenPoint((source.x + target.x) / 2, (source.y + target.y) / 2),
      radius: Math.max(sourceRadius, targetRadius, 1) + localArcPadding,
      offset,
      markerProgress: requestedMarkerProgress ?? 0.5,
    });
  }

  const requestedExtent =
    sourceRadius + sourceEndpointGap + targetRadius + targetEndpointGap;
  const close = clipped.distance <= requestedExtent;
  const markerProgress = requestedMarkerProgress ?? (close ? 0.5 : 0.76);
  const closeClearance = 2 * (Math.max(sourceRadius, targetRadius) + localArcPadding);
  const preferredOffset =
    close && Math.abs(offset) < closeClearance
      ? (offset < 0 ? -1 : 1) * (closeClearance + Math.abs(offset))
      : offset;
  const normal = frozenPoint(-clipped.unit.y, clipped.unit.x);
  const midpoint = frozenPoint(
    (clipped.start.x + clipped.end.x) / 2,
    (clipped.start.y + clipped.end.y) / 2,
  );
  const effectiveOffset = viewportBounds
    ? containedCurveOffset({
        start: clipped.start,
        end: clipped.end,
        midpoint,
        normal,
        requestedOffset: offset,
        preferredOffset,
        close,
        viewportBounds,
        routeMarkerPadding,
        markerProgress,
      })
    : preferredOffset;
  const control = frozenPoint(
    midpoint.x + normal.x * effectiveOffset,
    midpoint.y + normal.y * effectiveOffset,
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
    offset: effectiveOffset,
    close,
    markerProgress,
    path: `M ${number(clipped.start.x)} ${number(clipped.start.y)} Q ${number(control.x)} ${number(control.y)} ${number(clipped.end.x)} ${number(clipped.end.y)}`,
  });
}

/**
 * Construct one immutable directed polyline from allocator-owned waypoints.
 *
 * @param {{
 *   points: ReadonlyArray<RoutePoint | ReadonlyArray<number>>,
 *   offset?: number,
 *   close?: boolean,
 *   markerProgress?: number,
 * }} input
 * @returns {Readonly<RouteGeometry>}
 */
export function createPolylineRouteGeometry(input) {
  if (!input || typeof input !== "object" || !Array.isArray(input.points)) {
    throw new TypeError("polyline geometry input must contain a points array.");
  }
  const points = input.points.map((candidate, index) =>
    point(candidate, `points[${index}]`),
  );
  if (points.length < 2) {
    throw new RangeError("polyline geometry must contain at least two points.");
  }
  for (let index = 1; index < points.length; index += 1) {
    if (
      points[index - 1].x === points[index].x &&
      points[index - 1].y === points[index].y
    ) {
      throw new RangeError("polyline geometry may not contain repeated neighbors.");
    }
  }
  const offset = finite(input.offset ?? 0, "offset");
  const close = input.close === undefined ? false : boolean(input.close, "close");
  const markerProgress = unitInterval(
    input.markerProgress ?? (close ? 0.5 : 0.76),
    "markerProgress",
  );
  const start = points[0];
  const end = points.at(-1);
  if (!end) {
    throw new Error("polyline geometry lost its final point.");
  }
  const firstDirection = subtractPoints(points[1], start);
  const lastDirection = subtractPoints(end, points.at(-2) ?? start);
  return Object.freeze({
    kind: "polyline",
    start,
    end,
    points: Object.freeze(points),
    sourcePortAngle: Math.atan2(firstDirection.y, firstDirection.x),
    targetPortAngle: normalizeAngle(Math.atan2(-lastDirection.y, -lastDirection.x)),
    offset,
    close,
    markerProgress,
    path: points
      .map(
        (candidate, index) =>
          `${index === 0 ? "M" : "L"} ${number(candidate.x)} ${number(candidate.y)}`,
      )
      .join(" "),
  });
}

/**
 * Return a deterministic point and forward tangent on presentation geometry.
 *
 * Direction markers deliberately sit on the route instead of at its impact
 * endpoint, where bodies and consequence glyphs would obscure them.
 *
 * @param {RouteGeometry} route
 * @param {number} [progress]
 * @returns {Readonly<{x: number, y: number, degrees: number}>}
 */
export function routeMarkerPose(route, progress) {
  if (
    !route ||
    typeof route !== "object" ||
    (route.kind !== "curve" && route.kind !== "local_arc" && route.kind !== "polyline")
  ) {
    throw new TypeError("route must be curve, local_arc, or polyline geometry.");
  }
  const fraction =
    progress === undefined
      ? unitInterval(
          route.markerProgress ?? (route.close === true ? 0.5 : 0.76),
          "route.markerProgress",
        )
      : finite(progress, "progress");
  if (fraction < 0 || fraction > 1) {
    throw new RangeError("progress must be between 0 and 1.");
  }

  if (route.kind === "curve") {
    const start = point(route.start, "route.start");
    const control = point(route.control, "route.control");
    const end = point(route.end, "route.end");
    const remainder = 1 - fraction;
    const x =
      remainder * remainder * start.x +
      2 * remainder * fraction * control.x +
      fraction * fraction * end.x;
    const y =
      remainder * remainder * start.y +
      2 * remainder * fraction * control.y +
      fraction * fraction * end.y;
    const tangentX =
      2 * remainder * (control.x - start.x) + 2 * fraction * (end.x - control.x);
    const tangentY =
      2 * remainder * (control.y - start.y) + 2 * fraction * (end.y - control.y);
    return frozenPose(x, y, tangentX, tangentY);
  }

  if (route.kind === "polyline") {
    const points = route.points.map((candidate, index) =>
      point(candidate, `route.points[${index}]`),
    );
    if (points.length < 2) {
      throw new RangeError("polyline route must contain at least two points.");
    }
    const lengths = points
      .slice(1)
      .map((candidate, index) =>
        Math.hypot(candidate.x - points[index].x, candidate.y - points[index].y),
      );
    const totalLength = lengths.reduce((total, length) => total + length, 0);
    if (totalLength <= 0) {
      throw new RangeError("polyline route must have positive length.");
    }
    const requestedDistance = fraction * totalLength;
    let traversed = 0;
    for (const [index, length] of lengths.entries()) {
      if (length <= 0) {
        continue;
      }
      const isFinal = index === lengths.length - 1;
      if (requestedDistance <= traversed + length || isFinal) {
        const localProgress = Math.min(
          1,
          Math.max(0, (requestedDistance - traversed) / length),
        );
        const start = points[index];
        const end = points[index + 1];
        return frozenPose(
          start.x + (end.x - start.x) * localProgress,
          start.y + (end.y - start.y) * localProgress,
          end.x - start.x,
          end.y - start.y,
        );
      }
      traversed += length;
    }
    throw new Error("polyline route marker could not resolve a segment.");
  }

  const start = point(route.start, "route.start");
  const center = point(route.center, "route.center");
  const arcRadius = positive(route.arcRadius, "route.arcRadius");
  const startAngle = Math.atan2(start.y - center.y, start.x - center.x);
  const end = point(route.end, "route.end");
  const endAngle = Math.atan2(end.y - center.y, end.x - center.x);
  const sweep =
    route.sweep === 1
      ? positiveAngularDistance(startAngle, endAngle)
      : -positiveAngularDistance(endAngle, startAngle);
  const angle = startAngle + sweep * fraction;
  const tangentDirection = sweep < 0 ? -1 : 1;
  return frozenPose(
    center.x + Math.cos(angle) * arcRadius,
    center.y + Math.sin(angle) * arcRadius,
    -Math.sin(angle) * tangentDirection,
    Math.cos(angle) * tangentDirection,
  );
}

/**
 * Assign stable curve offsets to a set of accepted/presentation routes.
 *
 * Grouping uses only public source/target identity. The returned curvature is
 * a collision-separation convention and never a simulator trajectory.
 *
 * @param {ReadonlyArray<RouteRecord>} records
 * @param {RouteLayoutOptions} [options]
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
            sourceEndpointGap: record.sourceEndpointGap,
            targetEndpointGap: record.targetEndpointGap,
            prioritizeTargetClearance: record.prioritizeTargetClearance,
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
    sourceEndpointGap:
      record.sourceEndpointGap === undefined
        ? undefined
        : nonNegative(record.sourceEndpointGap, "sourceEndpointGap"),
    targetEndpointGap:
      record.targetEndpointGap === undefined
        ? undefined
        : nonNegative(record.targetEndpointGap, "targetEndpointGap"),
    prioritizeTargetClearance:
      record.prioritizeTargetClearance === undefined
        ? undefined
        : boolean(record.prioritizeTargetClearance, "prioritizeTargetClearance"),
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
 * Keep a directed quadratic curve and its presentation arrow inside the
 * battlefield rectangle without changing ordered source/recipient endpoints.
 *
 * A quadratic Bézier stays inside the convex hull of its two endpoints and
 * control point. Constraining the control point therefore contains the curve.
 * A second interval reserves an inset rectangle for the marker at its actual
 * presentation progress. Close routes with no assigned offset select the side
 * with more usable room; assigned signs are never flipped, preserving stable
 * same-direction and reciprocal separation.
 *
 * @param {{
 *   start: RoutePoint,
 *   end: RoutePoint,
 *   midpoint: RoutePoint,
 *   normal: RoutePoint,
 *   requestedOffset: number,
 *   preferredOffset: number,
 *   close: boolean,
 *   viewportBounds: RouteViewportBounds,
 *   routeMarkerPadding: number,
 *   markerProgress: number,
 * }} input
 */
function containedCurveOffset(input) {
  const controlRange = offsetRangeForPoint(
    input.midpoint,
    input.normal,
    input.viewportBounds,
  );
  const markerProgress = input.markerProgress;
  const markerBase = frozenPoint(
    input.start.x + (input.end.x - input.start.x) * markerProgress,
    input.start.y + (input.end.y - input.start.y) * markerProgress,
  );
  const markerInfluence = 2 * (1 - markerProgress) * markerProgress;
  const markerBounds = insetViewportBounds(
    input.viewportBounds,
    input.routeMarkerPadding,
  );
  const markerRange = offsetRangeForPoint(
    markerBase,
    frozenPoint(input.normal.x * markerInfluence, input.normal.y * markerInfluence),
    markerBounds,
  );
  const allowed = intersectOffsetRanges(controlRange, markerRange);
  if (!allowed) {
    return input.preferredOffset;
  }

  if (input.close && input.requestedOffset === 0) {
    const positiveRoom = Math.max(0, allowed.maximum);
    const negativeRoom = Math.max(0, -allowed.minimum);
    const sign = positiveRoom >= negativeRoom ? 1 : -1;
    const magnitude = Math.min(
      Math.abs(input.preferredOffset),
      Math.max(positiveRoom, negativeRoom),
    );
    return sign * magnitude;
  }
  if (input.preferredOffset > 0) {
    return Math.min(input.preferredOffset, Math.max(0, allowed.maximum));
  }
  if (input.preferredOffset < 0) {
    return Math.max(input.preferredOffset, Math.min(0, allowed.minimum));
  }
  return 0;
}

/**
 * @param {RoutePoint} origin
 * @param {RoutePoint} direction
 * @param {RouteViewportBounds} bounds
 */
function offsetRangeForPoint(origin, direction, bounds) {
  let minimum = Number.NEGATIVE_INFINITY;
  let maximum = Number.POSITIVE_INFINITY;
  for (const [coordinate, component, lower, upper] of [
    [origin.x, direction.x, bounds.left, bounds.right],
    [origin.y, direction.y, bounds.top, bounds.bottom],
  ]) {
    if (Math.abs(component) <= Number.EPSILON) {
      if (coordinate < lower || coordinate > upper) {
        return null;
      }
      continue;
    }
    const first = (lower - coordinate) / component;
    const second = (upper - coordinate) / component;
    minimum = Math.max(minimum, Math.min(first, second));
    maximum = Math.min(maximum, Math.max(first, second));
  }
  return minimum <= maximum ? Object.freeze({ minimum, maximum }) : null;
}

/**
 * @param {Readonly<{minimum: number, maximum: number}> | null} left
 * @param {Readonly<{minimum: number, maximum: number}> | null} right
 */
function intersectOffsetRanges(left, right) {
  if (!left || !right) {
    return null;
  }
  const minimum = Math.max(left.minimum, right.minimum);
  const maximum = Math.min(left.maximum, right.maximum);
  return minimum <= maximum ? Object.freeze({ minimum, maximum }) : null;
}

/**
 * @param {RouteViewportBounds} bounds
 * @param {number} requestedPadding
 */
function insetViewportBounds(bounds, requestedPadding) {
  const maximumHorizontal = Math.max(0, (bounds.right - bounds.left) / 2);
  const maximumVertical = Math.max(0, (bounds.bottom - bounds.top) / 2);
  const padding = Math.min(requestedPadding, maximumHorizontal, maximumVertical);
  return Object.freeze({
    left: bounds.left + padding,
    top: bounds.top + padding,
    right: bounds.right - padding,
    bottom: bounds.bottom - padding,
  });
}

/**
 * @param {{
 *   eventId: string,
 *   center: RoutePoint,
 *   radius: number,
 *   offset: number,
 *   markerProgress: number,
 * }} input
 * @returns {Readonly<RouteGeometry>}
 */
function localArc(input) {
  const orientation = hashFraction(input.eventId) * Math.PI * 2;
  const arcRadius = input.radius + Math.abs(input.offset) * 0.2;
  const halfSweep = 0.72;
  const sweep = input.offset < 0 ? 0 : 1;
  const startAngle = orientation + (sweep === 1 ? -halfSweep : halfSweep);
  const endAngle = orientation + (sweep === 1 ? halfSweep : -halfSweep);
  const start = frozenPoint(
    input.center.x + Math.cos(startAngle) * arcRadius,
    input.center.y + Math.sin(startAngle) * arcRadius,
  );
  const end = frozenPoint(
    input.center.x + Math.cos(endAngle) * arcRadius,
    input.center.y + Math.sin(endAngle) * arcRadius,
  );
  return Object.freeze({
    kind: "local_arc",
    start,
    end,
    center: input.center,
    arcRadius,
    sourcePortAngle: startAngle,
    targetPortAngle: endAngle,
    offset: input.offset,
    close: true,
    markerProgress: input.markerProgress,
    sweep,
    path: `M ${number(start.x)} ${number(start.y)} A ${number(arcRadius)} ${number(arcRadius)} 0 0 ${sweep} ${number(end.x)} ${number(end.y)}`,
  });
}

/**
 * @param {RouteViewportBounds | undefined} value
 * @returns {Readonly<RouteViewportBounds> | null}
 */
function optionalViewportBounds(value) {
  if (value === undefined) {
    return null;
  }
  if (!value || typeof value !== "object") {
    throw new TypeError("viewportBounds must be a rectangle.");
  }
  const left = finite(value.left, "viewportBounds.left");
  const top = finite(value.top, "viewportBounds.top");
  const right = finite(value.right, "viewportBounds.right");
  const bottom = finite(value.bottom, "viewportBounds.bottom");
  if (right <= left || bottom <= top) {
    throw new RangeError("viewportBounds must have positive width and height.");
  }
  return Object.freeze({ left, top, right, bottom });
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
function boolean(value, name) {
  if (typeof value !== "boolean") {
    throw new TypeError(`${name} must be boolean.`);
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
function unitInterval(value, name) {
  const result = finite(value, name);
  if (result < 0 || result > 1) {
    throw new RangeError(`${name} must be between 0 and 1.`);
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
 * @param {number} from
 * @param {number} to
 */
function positiveAngularDistance(from, to) {
  return normalizeAngle(to - from);
}

/**
 * @param {RoutePoint} end
 * @param {RoutePoint} start
 */
function subtractPoints(end, start) {
  return frozenPoint(end.x - start.x, end.y - start.y);
}

/**
 * @param {number} x
 * @param {number} y
 * @param {number} tangentX
 * @param {number} tangentY
 */
function frozenPose(x, y, tangentX, tangentY) {
  const tangentLength = Math.hypot(tangentX, tangentY);
  const safeTangentX = tangentLength > 0 ? tangentX : 1;
  const safeTangentY = tangentLength > 0 ? tangentY : 0;
  return Object.freeze({
    x,
    y,
    degrees: (Math.atan2(safeTangentY, safeTangentX) * 180) / Math.PI,
  });
}

/**
 * Keep path strings stable while retaining sub-pixel precision.
 *
 * @param {number} value
 */
function number(value) {
  return Number(value.toFixed(4));
}
