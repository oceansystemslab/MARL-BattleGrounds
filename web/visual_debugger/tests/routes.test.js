import assert from "node:assert/strict";
import test from "node:test";

import {
  clipRouteEndpoints,
  createRouteGeometry,
  layoutRouteSet,
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

test("route endpoints clip to body radii without changing direction", () => {
  const clipped = clipRouteEndpoints([0, 0], [10, 0], 1, 2);

  assert.deepEqual(clipped.start, { x: 1, y: 0 });
  assert.deepEqual(clipped.end, { x: 8, y: 0 });
  assert.deepEqual(clipped.unit, { x: 1, y: 0 });
  assert.equal(clipped.distance, 10);
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
  assert.ok(
    dot(
      subtract(forward.end, forward.control),
      { x: 180 - 20, y: 0 },
    ) > 0,
  );
  assert.ok(
    dot(
      subtract(reverse.end, reverse.control),
      { x: 20 - 180, y: 0 },
    ) > 0,
  );
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
    assert.ok(
      dot(
        subtract(route.end, route.start),
        { x: 110 - 10, y: 90 - 10 },
      ) > 0,
    );
    assert.ok(
      dot(
        subtract(route.end, route.control),
        { x: 110 - 10, y: 90 - 10 },
      ) > 0,
    );
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
