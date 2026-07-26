import assert from "node:assert/strict";
import test from "node:test";

import {
  createViewportTransform,
  layoutStatusDocks,
  protectedBodyRect,
  rectanglesIntersect,
  viewportOverflow,
} from "../src/layout.js";

const VIEWPORT = Object.freeze({
  left: 0,
  top: 0,
  right: 600,
  bottom: 400,
  width: 600,
  height: 400,
});

/**
 * @param {number} count
 * @param {string} prefix
 */
function statuses(count, prefix = "status") {
  return Array.from({ length: count }, (_, index) =>
    Object.freeze({ token_id: `${prefix}-${index}`, duration: index + 1 }),
  );
}

/**
 * @param {ReturnType<typeof layoutStatusDocks>} layout
 * @param {number} globalSlot
 */
function dockBySlot(layout, globalSlot) {
  const dock = layout.docks.find((candidate) => candidate.globalSlot === globalSlot);
  assert.ok(dock, `expected a status dock for slot ${globalSlot}`);
  return dock;
}

/**
 * @param {number} actual
 * @param {number} expected
 * @param {number} tolerance
 */
function assertClose(actual, expected, tolerance = 1e-9) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

test("viewport transform fits the map, inverts Y, and round-trips finite points", () => {
  const transform = createViewportTransform({
    worldWidth: 16,
    worldHeight: 12,
    viewportWidth: 900,
    viewportHeight: 600,
    padding: { top: 20, right: 30, bottom: 20, left: 30 },
  });

  assert.equal(transform.scale, 140 / 3);
  assert.deepEqual(transform.worldToScreen([0, 12]), {
    x: transform.mapBounds.left,
    y: transform.mapBounds.top,
  });
  assert.deepEqual(transform.worldToScreen([16, 0]), {
    x: transform.mapBounds.right,
    y: transform.mapBounds.bottom,
  });
  assert.equal(transform.worldLengthToScreen(0.5), 70 / 3);

  for (const world of [
    { x: 0, y: 0 },
    { x: 8.25, y: 5.75 },
    { x: 16, y: 12 },
    { x: -1.5, y: 13.5 },
  ]) {
    const roundTrip = transform.screenToWorld(transform.worldToScreen(world));
    assertClose(roundTrip.x, world.x);
    assertClose(roundTrip.y, world.y);
  }

  assert.throws(
    () =>
      createViewportTransform({
        worldWidth: Number.NaN,
        worldHeight: 12,
        viewportWidth: 900,
        viewportHeight: 600,
      }),
    /worldWidth must be finite/,
  );
  assert.throws(
    () => transform.worldToScreen({ x: 1, y: Number.POSITIVE_INFINITY }),
    /world point\.y must be finite/,
  );
});

test("isolated nine-status dock stays expanded, ordered, and bounded", () => {
  const orderedStatuses = statuses(9);
  const layout = layoutStatusDocks({
    viewport: VIEWPORT,
    agents: [
      {
        globalSlot: 4,
        center: { x: 300, y: 220 },
        radius: 24,
        statuses: orderedStatuses,
      },
    ],
  });
  const dock = dockBySlot(layout, 4);
  const body = layout.protectedBodies[0].bounds;

  assert.equal(dock.expanded, true);
  assert.equal(dock.anchor, "north");
  assert.equal(dock.tangentShift, 0);
  assert.equal(dock.columns, 3);
  assert.equal(dock.rows, 3);
  assert.equal(dock.visibleCount, 9);
  assert.equal(dock.hiddenCount, 0);
  assert.equal(dock.visibleCount + dock.hiddenCount, dock.totalCount);
  assert.deepEqual(dock.visibleStatuses, orderedStatuses);
  assert.deepEqual(dock.hiddenStatuses, []);
  assert.equal(dock.overflowLabel, null);
  assert.equal(dock.collisionFree, true);
  assert.equal(viewportOverflow(dock.bounds, VIEWPORT), 0);
  assert.equal(rectanglesIntersect(dock.bounds, body), false);
  assert.ok(Math.abs(dock.tangentShift) <= 24);
});

test("dense placement is input-order invariant and preserves priority truth", () => {
  const agents = [
    {
      globalSlot: 9,
      center: { x: 350, y: 190 },
      radius: 22,
      statuses: statuses(9, "ordinary-a"),
    },
    {
      globalSlot: 3,
      center: { x: 550, y: 200 },
      radius: 22,
      statuses: statuses(9, "selected"),
      selected: true,
    },
    {
      globalSlot: 7,
      center: { x: 350, y: 210 },
      radius: 22,
      statuses: statuses(8, "ordinary-b"),
    },
    {
      globalSlot: 5,
      center: { x: 150, y: 200 },
      radius: 22,
      statuses: statuses(9, "controlled"),
      controlled: true,
    },
  ];
  const input = {
    viewport: {
      left: 0,
      top: 0,
      right: 700,
      bottom: 400,
      width: 700,
      height: 400,
    },
    agents,
  };
  const forward = layoutStatusDocks(input);
  const shuffled = layoutStatusDocks({
    ...input,
    agents: [agents[2], agents[0], agents[3], agents[1]],
  });

  assert.deepEqual(forward, shuffled);
  assert.deepEqual(forward.placementOrder, [5, 3, 9, 7]);
  assert.equal(dockBySlot(forward, 5).expanded, true);
  assert.equal(dockBySlot(forward, 3).expanded, true);

  for (const dock of forward.docks) {
    assert.equal(dock.collisionFree, true);
    assert.equal(dock.visibleCount + dock.hiddenCount, dock.totalCount);
    assert.ok(Math.abs(dock.tangentShift) <= 24);
    const original = agents.find((agent) => agent.globalSlot === dock.globalSlot);
    assert.ok(original);
    assert.deepEqual(
      [...dock.visibleStatuses, ...dock.hiddenStatuses],
      original.statuses,
    );
  }
  assert.deepEqual(
    new Set([
      ...forward.docks.map(({ globalSlot }) => globalSlot),
      ...forward.suppressedGlobalSlots,
    ]),
    new Set(agents.map(({ globalSlot }) => globalSlot)),
  );
});

test("an unsatisfiable ordinary dock is suppressed instead of overlapping", () => {
  const reservedRects = [
    { left: 0, top: 0, right: 200, bottom: 45, width: 200, height: 45 },
    { left: 0, top: 75, right: 200, bottom: 120, width: 200, height: 45 },
    { left: 0, top: 0, right: 75, bottom: 120, width: 75, height: 120 },
    { left: 125, top: 0, right: 200, bottom: 120, width: 75, height: 120 },
  ];
  const layout = layoutStatusDocks({
    viewport: {
      left: 0,
      top: 0,
      right: 200,
      bottom: 120,
      width: 200,
      height: 120,
    },
    agents: [
      {
        globalSlot: 1,
        center: { x: 100, y: 60 },
        radius: 18,
        statuses: statuses(9, "suppressed"),
      },
    ],
    reservedRects,
  });

  assert.deepEqual(layout.docks, []);
  assert.deepEqual(layout.suppressedGlobalSlots, [1]);
});

test("viewport-edge and reserved-zone collisions move a dock deterministically", () => {
  const edgeAgent = {
    globalSlot: 1,
    center: { x: 28, y: 28 },
    radius: 18,
    statuses: statuses(4, "edge"),
  };
  const edgeLayout = layoutStatusDocks({
    viewport: {
      left: 0,
      top: 0,
      right: 300,
      bottom: 200,
      width: 300,
      height: 200,
    },
    agents: [edgeAgent],
  });
  const eastDock = dockBySlot(edgeLayout, 1);
  assert.equal(eastDock.anchor, "east");
  assert.equal(eastDock.collisionFree, true);
  assert.equal(
    viewportOverflow(eastDock.bounds, {
      left: 0,
      top: 0,
      right: 300,
      bottom: 200,
      width: 300,
      height: 200,
    }),
    0,
  );

  const reservedEast = {
    left: 56,
    top: 0,
    right: 170,
    bottom: 55,
    width: 114,
    height: 55,
  };
  const reservedLayout = layoutStatusDocks({
    viewport: {
      left: 0,
      top: 0,
      right: 300,
      bottom: 200,
      width: 300,
      height: 200,
    },
    agents: [edgeAgent],
    reservedRects: [reservedEast],
  });
  const southDock = dockBySlot(reservedLayout, 1);
  assert.equal(southDock.anchor, "south");
  assert.equal(southDock.collisionFree, true);
  assert.equal(rectanglesIntersect(southDock.bounds, reservedEast), false);
});

test("required one-cell docks remain expanded and collision-free in supported geometry", () => {
  const reservedStatus = {
    left: 275,
    top: 140,
    right: 325,
    bottom: 180,
    width: 50,
    height: 40,
  };
  const agents = [
    {
      globalSlot: 8,
      center: { x: 450, y: 210 },
      radius: 20,
      statuses: Object.freeze([{ ticks: 1 }]),
      required: true,
    },
    {
      globalSlot: 2,
      center: { x: 300, y: 210 },
      radius: 20,
      statuses: Object.freeze([{ ticks: 30 }]),
      required: true,
    },
  ];
  const options = {
    cellWidth: 38,
    cellHeight: 18,
    cellGap: 2,
    dockGap: 5,
    bodyPadding: 4,
    selectionAllowance: 12,
  };
  const input = {
    agents,
    viewport: VIEWPORT,
    reservedRects: [reservedStatus],
  };
  const forward = layoutStatusDocks(input, options);
  const reversed = layoutStatusDocks(
    { ...input, agents: [...agents].reverse() },
    options,
  );

  assert.deepEqual(forward, reversed);
  assert.deepEqual(forward.placementOrder, [2, 8]);
  assert.deepEqual(forward.suppressedGlobalSlots, []);
  for (const dock of forward.docks) {
    assert.equal(dock.required, true);
    assert.equal(dock.expanded, true);
    assert.equal(dock.visibleCount, 1);
    assert.equal(dock.hiddenCount, 0);
    assert.equal(dock.collisionFree, true);
    assert.equal(rectanglesIntersect(dock.bounds, reservedStatus), false);
    assert.equal(viewportOverflow(dock.bounds, VIEWPORT), 0);
  }
  assert.equal(
    rectanglesIntersect(forward.docks[0].bounds, forward.docks[1].bounds),
    false,
  );
});

test("protected body allowance grows only for selected or controlled agents", () => {
  const plain = protectedBodyRect({
    center: [50, 60],
    radius: 10,
  });
  const selected = protectedBodyRect({
    center: [50, 60],
    radius: 10,
    selected: true,
  });
  const controlled = protectedBodyRect({
    center: [50, 60],
    radius: 10,
    controlled: true,
  });

  assert.deepEqual(plain, {
    left: 37,
    top: 47,
    right: 63,
    bottom: 73,
    width: 26,
    height: 26,
  });
  assert.equal(selected.width, plain.width + 10);
  assert.deepEqual(controlled, selected);
  assert.equal(rectanglesIntersect(plain, selected), true);
});
