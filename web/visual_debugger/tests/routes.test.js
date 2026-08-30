import assert from "node:assert/strict";
import test from "node:test";

import {
  clipRouteEndpoints,
  createPolylineRouteGeometry,
  createRouteGeometry,
  layoutRouteSet,
  routeMarkerPose,
} from "../src/routes.js";

/**
 * @param {{x: number, y: number}} left
 * @param {{x: number, y: number}} right
 */
function dot(left, right) {
  return left.x * right.x + left.y * right.y;
}

/**
 * @param {{x: number, y: number}} left
 * @param {{x: number, y: number}} right
 */
function subtract(left, right) {
  return { x: left.x - right.x, y: left.y - right.y };
}

/**
 * @param {ReturnType<typeof createRouteGeometry>} route
 * @param {{left: number, top: number, right: number, bottom: number}} bounds
 */
function assertRouteContained(route, bounds) {
  if (route.kind !== "curve") {
    assert.fail("distinct centers must produce a directed curve.");
  }
  for (let step = 0; step <= 100; step += 1) {
    const point = routeMarkerPose(route, step / 100);
    assert.ok(point.x >= bounds.left - 1e-9);
    assert.ok(point.x <= bounds.right + 1e-9);
    assert.ok(point.y >= bounds.top - 1e-9);
    assert.ok(point.y <= bounds.bottom + 1e-9);
  }
}

/**
 * The live arrow is the polygon declared by choreography-painter.js. Checking
 * its transformed vertices proves containment of the visible marker, not only
 * its center point.
 *
 * @param {ReturnType<typeof routeMarkerPose>} marker
 * @param {{left: number, top: number, right: number, bottom: number}} bounds
 */
function assertVisibleMarkerContained(marker, bounds) {
  const radians = (marker.degrees * Math.PI) / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  for (const vertex of [
    { x: -11, y: -6 },
    { x: 2, y: 0 },
    { x: -11, y: 6 },
    { x: -7, y: 0 },
  ]) {
    const x = marker.x + vertex.x * cosine - vertex.y * sine;
    const y = marker.y + vertex.x * sine + vertex.y * cosine;
    assert.ok(x >= bounds.left - 1e-9);
    assert.ok(x <= bounds.right + 1e-9);
    assert.ok(y >= bounds.top - 1e-9);
    assert.ok(y <= bounds.bottom + 1e-9);
  }
}

test("route endpoints clip to body radii without changing direction", () => {
  const clipped = clipRouteEndpoints([0, 0], [10, 0], 1, 2);

  assert.deepEqual(clipped.start, { x: 1, y: 0 });
  assert.deepEqual(clipped.end, { x: 8, y: 0 });
  assert.deepEqual(clipped.unit, { x: 1, y: 0 });
  assert.equal(clipped.distance, 10);
});

test("asymmetric target clearance preserves the ordered route bearing", () => {
  const clipped = clipRouteEndpoints([0, 0], [100, 0], 10, 12, 3, 26);

  assert.deepEqual(clipped.start, { x: 13, y: 0 });
  assert.deepEqual(clipped.end, { x: 62, y: 0 });
  assert.deepEqual(clipped.unit, { x: 1, y: 0 });
  assert.ok(dot(subtract(clipped.end, clipped.start), clipped.unit) > 0);
});

test("dense convergence preserves recipient clearance when clips must compress", () => {
  const proportional = clipRouteEndpoints([0, 0], [40, 0], 14, 14, 3, 12);
  const targetPriority = clipRouteEndpoints([0, 0], [40, 0], 14, 14, 3, 12, true);

  assert.ok(Math.abs(40 - proportional.end.x) < 26);
  assert.deepEqual(targetPriority.start, { x: 6, y: 0 });
  assert.deepEqual(targetPriority.end, { x: 14, y: 0 });
  assert.deepEqual(targetPriority.unit, { x: 1, y: 0 });
  assert.equal(40 - targetPriority.end.x, 26);
  assert.ok(
    dot(subtract(targetPriority.end, targetPriority.start), targetPriority.unit) > 0,
  );
});

test("reciprocal routes bend to opposite sides with clipped endpoints", () => {
  const routes = layoutRouteSet([
    {
      eventId: "forward",
      sourceGlobalSlot: 1,
      targetGlobalSlot: 6,
      source: [20, 40],
      target: [180, 40],
      sourceRadius: 20,
      targetRadius: 20,
    },
    {
      eventId: "reverse",
      sourceGlobalSlot: 6,
      targetGlobalSlot: 1,
      source: [180, 40],
      target: [20, 40],
      sourceRadius: 20,
      targetRadius: 20,
    },
  ]);
  const forward = routes.find((route) => route.eventId === "forward");
  const reverse = routes.find((route) => route.eventId === "reverse");

  assert.ok(forward);
  assert.ok(reverse);
  assert.equal(forward.groupKey, reverse.groupKey);
  assert.equal(forward.offset, reverse.offset);
  assert.ok(forward.offset > 0);
  assert.equal(forward.kind, "curve");
  assert.equal(reverse.kind, "curve");
  assert.ok(forward.control.y > 40);
  assert.ok(reverse.control.y < 40);
  assert.ok(forward.start.x > 20);
  assert.ok(forward.end.x < 180);
  assert.ok(reverse.start.x < 180);
  assert.ok(reverse.end.x > 20);
  assert.ok(dot(subtract(forward.end, forward.control), { x: 180 - 20, y: 0 }) > 0);
  assert.ok(dot(subtract(reverse.end, reverse.control), { x: 20 - 180, y: 0 }) > 0);
});

test("same-direction duplicates use stable symmetric offsets under shuffling", () => {
  const records = ["event-c", "event-a", "event-b"].map((eventId) => ({
    eventId,
    sourceGlobalSlot: 2,
    targetGlobalSlot: 7,
    source: [10, 10],
    target: [110, 90],
    sourceRadius: 12,
    targetRadius: 12,
  }));

  const forward = layoutRouteSet(records);
  const shuffled = layoutRouteSet([records[1], records[2], records[0]]);

  assert.deepEqual(forward, shuffled);
  assert.deepEqual(
    forward.map(({ eventId, offset }) => [eventId, offset]),
    [
      ["event-a", -18],
      ["event-b", 0],
      ["event-c", 18],
    ],
  );
  assert.equal(new Set(forward.map(({ path }) => path)).size, 3);
  for (const route of forward) {
    if (route.kind !== "curve") {
      assert.fail("distinct route centers must produce directed curves.");
    }
    assert.ok(dot(subtract(route.end, route.start), { x: 110 - 10, y: 90 - 10 }) > 0);
    assert.ok(dot(subtract(route.end, route.control), { x: 110 - 10, y: 90 - 10 }) > 0);
  }
});

test("near-zero routes expose deterministic arc sweep for tangent arrows", () => {
  const route = createRouteGeometry({
    eventId: "local-impact",
    source: [10, 10],
    target: [10, 10],
    sourceRadius: 12,
    targetRadius: 12,
  });

  assert.equal(route.kind, "local_arc");
  assert.ok(route.sweep === 0 || route.sweep === 1);
});

test("crossing endpoint pairs retain independent deterministic groups", () => {
  const routes = layoutRouteSet([
    {
      eventId: "diagonal-a",
      sourceGlobalSlot: 0,
      targetGlobalSlot: 9,
      source: [0, 0],
      target: [100, 100],
    },
    {
      eventId: "diagonal-b",
      sourceGlobalSlot: 4,
      targetGlobalSlot: 5,
      source: [0, 100],
      target: [100, 0],
    },
  ]);

  assert.equal(routes.length, 2);
  assert.notEqual(routes[0].groupKey, routes[1].groupKey);
  assert.equal(routes[0].offset, 0);
  assert.equal(routes[1].offset, 0);
});

test("near-zero routes use a stable oriented local arc", () => {
  const first = createRouteGeometry({
    eventId: "local-impact",
    source: [50, 50],
    target: [50, 50],
    sourceRadius: 18,
    targetRadius: 18,
  });
  const repeated = createRouteGeometry({
    eventId: "local-impact",
    source: [50, 50],
    target: [50, 50],
    sourceRadius: 18,
    targetRadius: 18,
  });

  assert.deepEqual(first, repeated);
  assert.equal(first.kind, "local_arc");
  assert.ok(first.arcRadius > 18);
  assert.notDeepEqual(first.start, first.end);
  assert.match(first.path, /^M .* A /);
});

test("overlapping body extents retain ordered source-to-recipient geometry", () => {
  const route = createRouteGeometry({
    eventId: "overlapping-impact",
    source: [40, 50],
    target: [50, 50],
    sourceRadius: 12,
    targetRadius: 12,
    offset: 60,
  });
  const reverse = createRouteGeometry({
    eventId: "overlapping-impact-reverse",
    source: [50, 50],
    target: [40, 50],
    sourceRadius: 12,
    targetRadius: 12,
    offset: 60,
  });

  assert.equal(route.kind, "curve");
  assert.equal(reverse.kind, "curve");
  assert.ok(route.start.x < route.end.x);
  assert.ok(reverse.start.x > reverse.end.x);
  assert.equal(route.start.y, 50);
  assert.equal(route.end.y, 50);
  assert.ok(dot(subtract(route.end, route.start), { x: 10, y: 0 }) > 0);
  assert.ok(dot(subtract(route.end, route.control), { x: 10, y: 0 }) > 0);
  assert.ok(dot(subtract(reverse.end, reverse.start), { x: -10, y: 0 }) > 0);
  assert.ok(dot(subtract(reverse.end, reverse.control), { x: -10, y: 0 }) > 0);
  assert.match(route.path, /^M .* Q /);
});

test("only literally coincident centers use hash-oriented local arcs", () => {
  const first = createRouteGeometry({
    eventId: "close-a",
    source: [10, 10],
    target: [10.001, 10],
    sourceRadius: 20,
    targetRadius: 20,
  });
  const second = createRouteGeometry({
    eventId: "close-b",
    source: [10, 10],
    target: [10.001, 10],
    sourceRadius: 20,
    targetRadius: 20,
  });
  const coincident = createRouteGeometry({
    eventId: "coincident",
    source: [10, 10],
    target: [10, 10],
    sourceRadius: 20,
    targetRadius: 20,
  });

  assert.equal(first.kind, "curve");
  assert.equal(second.kind, "curve");
  assert.deepEqual(first, second);
  assert.ok(first.start.x < first.end.x);
  assert.ok(dot(subtract(first.end, first.control), { x: 0.001, y: 0 }) > 0);
  assert.equal(coincident.kind, "local_arc");
});

test("overlapping distinct bodies bow outward with a bearing-aligned marker", () => {
  const route = createRouteGeometry({
    eventId: "close-directed-marker",
    source: [40, 40],
    target: [41, 40.5],
    sourceRadius: 18,
    targetRadius: 18,
  });
  const marker = routeMarkerPose(route);
  const markerDirection = {
    x: Math.cos((marker.degrees * Math.PI) / 180),
    y: Math.sin((marker.degrees * Math.PI) / 180),
  };

  assert.equal(route.kind, "curve");
  assert.equal(route.close, true);
  assert.ok(Math.abs(route.offset) >= 52);
  assert.ok(
    Math.hypot(marker.x - route.end.x, marker.y - route.end.y) > Math.max(18, 18),
  );
  assert.ok(dot(markerDirection, { x: 1, y: 0.5 }) > 0);
});

test("close routes choose a contained bow and keep the visible marker inside bounds", () => {
  const viewportBounds = { left: 0, top: 0, right: 200, bottom: 100 };
  const input = {
    eventId: "close-bottom-edge",
    source: [70, 82],
    target: [71, 82.5],
    sourceRadius: 18,
    targetRadius: 18,
  };
  const route = createRouteGeometry(input, { viewportBounds });
  const repeated = createRouteGeometry(input, { viewportBounds });

  assert.equal(route.kind, "curve");
  assert.equal(route.close, true);
  assert.ok(route.offset < 0);
  assert.deepEqual(route, repeated);
  assertRouteContained(route, viewportBounds);
  assertVisibleMarkerContained(routeMarkerPose(route), viewportBounds);
  assert.ok(
    dot(subtract(route.end, route.control), {
      x: input.target[0] - input.source[0],
      y: input.target[1] - input.source[1],
    }) > 0,
  );
});

test("bounded reciprocal close routes retain opposite sides and directed markers", () => {
  const viewportBounds = { left: 0, top: 0, right: 200, bottom: 100 };
  const routes = layoutRouteSet(
    [
      {
        eventId: "edge-forward",
        sourceGlobalSlot: 1,
        targetGlobalSlot: 6,
        source: [70, 82],
        target: [71, 82],
        sourceRadius: 18,
        targetRadius: 18,
      },
      {
        eventId: "edge-reverse",
        sourceGlobalSlot: 6,
        targetGlobalSlot: 1,
        source: [71, 82],
        target: [70, 82],
        sourceRadius: 18,
        targetRadius: 18,
      },
    ],
    { viewportBounds },
  );
  const forward = routes.find((route) => route.eventId === "edge-forward");
  const reverse = routes.find((route) => route.eventId === "edge-reverse");
  assert.ok(forward);
  assert.ok(reverse);
  if (forward.kind !== "curve" || reverse.kind !== "curve") {
    assert.fail("distinct reciprocal centers must produce directed curves.");
  }

  assert.ok(forward.control.y > 82);
  assert.ok(reverse.control.y < 82);
  assert.notEqual(forward.path, reverse.path);
  for (const route of [forward, reverse]) {
    assertRouteContained(route, viewportBounds);
    assertVisibleMarkerContained(routeMarkerPose(route), viewportBounds);
  }
  assert.ok(dot(subtract(forward.end, forward.control), { x: 1, y: 0 }) > 0);
  assert.ok(dot(subtract(reverse.end, reverse.control), { x: -1, y: 0 }) > 0);
});

test("route marker poses stay visible before impact and retain forward direction", () => {
  const forward = createRouteGeometry({
    eventId: "forward-marker",
    source: [10, 20],
    target: [110, 60],
    sourceRadius: 12,
    targetRadius: 12,
    offset: 32,
  });
  const reverse = createRouteGeometry({
    eventId: "reverse-marker",
    source: [110, 60],
    target: [10, 20],
    sourceRadius: 12,
    targetRadius: 12,
    offset: 32,
  });
  const forwardPose = routeMarkerPose(forward);
  const reversePose = routeMarkerPose(reverse);

  assert.notDeepEqual({ x: forwardPose.x, y: forwardPose.y }, forward.end);
  assert.ok(
    dot(
      {
        x: Math.cos((forwardPose.degrees * Math.PI) / 180),
        y: Math.sin((forwardPose.degrees * Math.PI) / 180),
      },
      { x: 100, y: 40 },
    ) > 0,
  );
  assert.ok(
    dot(
      {
        x: Math.cos((reversePose.degrees * Math.PI) / 180),
        y: Math.sin((reversePose.degrees * Math.PI) / 180),
      },
      { x: -100, y: -40 },
    ) > 0,
  );
});

test("local-arc marker poses follow their declared sweep", () => {
  const clockwise = createRouteGeometry({
    eventId: "coincident-marker",
    source: [40, 40],
    target: [40, 40],
    sourceRadius: 14,
    targetRadius: 14,
  });
  const counterclockwise = createRouteGeometry({
    eventId: "coincident-marker-reverse",
    source: [40, 40],
    target: [40, 40],
    sourceRadius: 14,
    targetRadius: 14,
    offset: -20,
  });
  if (clockwise.kind !== "local_arc" || counterclockwise.kind !== "local_arc") {
    assert.fail("coincident route centers must produce local arcs.");
  }

  for (const arc of [clockwise, counterclockwise]) {
    const first = routeMarkerPose(arc, 0.25);
    const later = routeMarkerPose(arc, 0.75);
    const midpoint = routeMarkerPose(arc, 0.5);
    const midpointTangent = {
      x: Math.cos((midpoint.degrees * Math.PI) / 180),
      y: Math.sin((midpoint.degrees * Math.PI) / 180),
    };
    const midpointRadial = {
      x: (midpoint.x - arc.center.x) / arc.arcRadius,
      y: (midpoint.y - arc.center.y) / arc.arcRadius,
    };
    const declaredForward =
      arc.sweep === 1
        ? { x: -midpointRadial.y, y: midpointRadial.x }
        : { x: midpointRadial.y, y: -midpointRadial.x };

    assert.notDeepEqual(first, later);
    assert.ok(Number.isFinite(first.degrees));
    assert.ok(Number.isFinite(later.degrees));
    assert.ok(
      Math.abs(
        Math.hypot(midpoint.x - arc.center.x, midpoint.y - arc.center.y) -
          arc.arcRadius,
      ) < 1e-9,
    );
    assert.ok(dot(midpointTangent, declaredForward) > 1 - 1e-9);
    assert.ok(
      Math.hypot(
        routeMarkerPose(arc, 0).x - arc.start.x,
        routeMarkerPose(arc, 0).y - arc.start.y,
      ) < 1e-9,
    );
    assert.ok(
      Math.hypot(
        routeMarkerPose(arc, 1).x - arc.end.x,
        routeMarkerPose(arc, 1).y - arc.end.y,
      ) < 1e-9,
    );
  }
  assert.equal(clockwise.sweep, 1);
  assert.equal(counterclockwise.sweep, 0);
});

test("polyline marker pose follows finite arc-length progress through bends", () => {
  const route = createPolylineRouteGeometry({
    points: [
      [0, 0],
      [10, 0],
      [10, 30],
      [40, 30],
    ],
  });

  assert.equal(route.kind, "polyline");
  assert.match(route.path, /^M 0 0 L 10 0 L 10 30 L 40 30$/);
  assert.deepEqual(routeMarkerPose(route, 0), { x: 0, y: 0, degrees: 0 });
  assert.deepEqual(routeMarkerPose(route, 1), { x: 40, y: 30, degrees: 0 });
  const marker = routeMarkerPose(route);
  assert.ok(Number.isFinite(marker.x));
  assert.ok(Number.isFinite(marker.y));
  assert.ok(Number.isFinite(marker.degrees));
  assert.ok(marker.x > 10 && marker.x < 40);
  assert.equal(marker.y, 30);
  assert.equal(marker.degrees, 0);
});
