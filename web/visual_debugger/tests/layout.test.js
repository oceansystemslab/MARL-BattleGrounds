import assert from "node:assert/strict";
import test from "node:test";

import {
  createViewportTransform,
  layoutCrossPhaseOccupancy,
  layoutRequiredDocks,
  layoutStatusDocks,
  protectedBodyRect,
  rectanglesIntersect,
  viewportOverflow,
} from "../src/layout.js";
import { routeMarkerPose } from "../src/routes.js";

const VIEWPORT = Object.freeze({
  left: 0,
  top: 0,
  right: 600,
  bottom: 400,
  width: 600,
  height: 400,
});

/**
 * @param {number} left
 * @param {number} top
 * @param {number} right
 * @param {number} bottom
 */
function testRectangle(left, top, right, bottom) {
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

/**
 * @param {{x: number, y: number}} start
 * @param {{x: number, y: number}} end
 * @param {ReturnType<typeof testRectangle>} bounds
 */
function segmentTouchesRectangle(start, end, bounds) {
  let minimum = 0;
  let maximum = 1;
  for (const [origin, delta, lower, upper] of [
    [start.x, end.x - start.x, bounds.left, bounds.right],
    [start.y, end.y - start.y, bounds.top, bounds.bottom],
  ]) {
    if (Math.abs(delta) <= 1e-12) {
      if (origin < lower || origin > upper) return false;
      continue;
    }
    const first = (lower - origin) / delta;
    const second = (upper - origin) / delta;
    minimum = Math.max(minimum, Math.min(first, second));
    maximum = Math.min(maximum, Math.max(first, second));
    if (minimum > maximum) return false;
  }
  return true;
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

test("minimum-viewport status summaries retain controlled and ordinary truth", () => {
  const ordinaryStatuses = statuses(9, "ordinary");
  const controlledStatuses = statuses(9, "controlled");
  const layout = layoutStatusDocks(
    {
      viewport: {
        left: 0,
        top: 0,
        right: 557,
        bottom: 384,
        width: 557,
        height: 384,
      },
      agents: [
        {
          globalSlot: 0,
          center: { x: 120, y: 190 },
          radius: 22,
          statuses: controlledStatuses,
          controlled: true,
        },
        {
          globalSlot: 1,
          center: { x: 430, y: 190 },
          radius: 22,
          statuses: ordinaryStatuses,
        },
      ],
    },
    { ordinaryVisibleLimit: 0, requiredVisibleLimit: 0 },
  );

  const controlled = dockBySlot(layout, 0);
  const ordinary = dockBySlot(layout, 1);
  assert.equal(controlled.visibleCount, 0);
  assert.equal(controlled.hiddenCount, 9);
  assert.equal(controlled.expanded, false);
  assert.equal(controlled.overflowLabel, "+9");
  assert.deepEqual(controlled.hiddenStatuses, controlledStatuses);
  assert.equal(ordinary.visibleCount, 0);
  assert.equal(ordinary.hiddenCount, 9);
  assert.equal(ordinary.overflowLabel, "+9");
  assert.deepEqual(
    [...ordinary.visibleStatuses, ...ordinary.hiddenStatuses],
    ordinaryStatuses,
  );
  assert.equal(rectanglesIntersect(controlled.bounds, ordinary.bounds), false);
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

test("mixed required status and cooldown docks share one deterministic search", () => {
  const agents = [
    {
      globalSlot: 2,
      center: { x: 240, y: 200 },
      radius: 20,
      statuses: [],
      controlled: true,
    },
    {
      globalSlot: 7,
      center: { x: 360, y: 200 },
      radius: 20,
      statuses: [],
      selected: true,
    },
  ];
  const requests = [
    {
      layoutKey: "status:2",
      globalSlot: 2,
      statuses: statuses(9, "controlled"),
      dockOptions: {
        cellWidth: 28,
        cellHeight: 18,
        cellGap: 2,
        dockGap: 5,
      },
      priority: 0,
    },
    {
      layoutKey: "status:7",
      globalSlot: 7,
      statuses: statuses(9, "selected"),
      dockOptions: {
        cellWidth: 28,
        cellHeight: 18,
        cellGap: 2,
        dockGap: 5,
      },
      priority: 0,
    },
    {
      layoutKey: "cooldown:2",
      globalSlot: 2,
      statuses: [{ ticks: 30 }],
      dockOptions: {
        cellWidth: 38,
        cellHeight: 18,
        cellGap: 2,
        dockGap: 5,
      },
      priority: 1,
    },
    {
      layoutKey: "cooldown:7",
      globalSlot: 7,
      statuses: [{ ticks: 1 }],
      dockOptions: {
        cellWidth: 38,
        cellHeight: 18,
        cellGap: 2,
        dockGap: 5,
      },
      priority: 1,
    },
  ];
  const options = {
    bodyPadding: 4,
    selectionAllowance: 12,
    dockGap: 5,
  };
  const input = {
    agents,
    requests,
    viewport: VIEWPORT,
  };
  const forward = layoutRequiredDocks(input, options);
  const reversed = layoutRequiredDocks(
    {
      ...input,
      agents: [...agents].reverse(),
      requests: [...requests].reverse(),
    },
    options,
  );

  assert.deepEqual(forward, reversed);
  assert.deepEqual(forward.placementOrder, [
    "status:2",
    "status:7",
    "cooldown:2",
    "cooldown:7",
  ]);
  assert.deepEqual(forward.suppressedLayoutKeys, []);
  assert.deepEqual(
    forward.docks.map(({ layoutKey }) => layoutKey),
    ["cooldown:2", "status:2", "cooldown:7", "status:7"],
  );
  for (const dock of forward.docks) {
    assert.equal(dock.required, true);
    assert.equal(dock.expanded, true);
    assert.equal(dock.hiddenCount, 0);
    assert.equal(dock.collisionFree, true);
    assert.equal(viewportOverflow(dock.bounds, VIEWPORT), 0);
    for (const body of forward.protectedBodies) {
      assert.equal(rectanglesIntersect(dock.bounds, body.bounds), false);
    }
  }
  for (let index = 0; index < forward.docks.length; index += 1) {
    for (let other = index + 1; other < forward.docks.length; other += 1) {
      assert.equal(
        rectanglesIntersect(forward.docks[index].bounds, forward.docks[other].bounds),
        false,
      );
    }
  }
});

test("required dock fallback preserves feasible priority truth around one impossible cue", () => {
  const agents = [
    {
      globalSlot: 2,
      center: { x: 200, y: 200 },
      radius: 20,
      statuses: [],
      controlled: true,
    },
    {
      globalSlot: 7,
      center: { x: 400, y: 200 },
      radius: 20,
      statuses: [],
      selected: true,
    },
  ];
  const requests = [
    {
      layoutKey: "status:2",
      globalSlot: 2,
      statuses: statuses(9, "controlled"),
      dockOptions: {
        cellWidth: 28,
        cellHeight: 18,
        cellGap: 2,
        dockGap: 5,
      },
      priority: 0,
    },
    {
      layoutKey: "status:7",
      globalSlot: 7,
      statuses: statuses(9, "selected"),
      dockOptions: {
        cellWidth: 28,
        cellHeight: 18,
        cellGap: 2,
        dockGap: 5,
      },
      priority: 0,
    },
    {
      layoutKey: "cooldown:2",
      globalSlot: 2,
      statuses: [{ ticks: 30 }],
      dockOptions: {
        cellWidth: 1_000,
        cellHeight: 18,
        cellGap: 2,
        dockGap: 5,
      },
      fallbackDockOptions: {
        cellWidth: 38,
        cellHeight: 18,
        cellGap: 2,
        dockGap: 5,
      },
      priority: 1,
    },
    {
      layoutKey: "cooldown:7",
      globalSlot: 7,
      statuses: [{ ticks: 1 }],
      dockOptions: {
        cellWidth: 38,
        cellHeight: 18,
        cellGap: 2,
        dockGap: 5,
      },
      priority: 1,
    },
  ];
  const options = {
    bodyPadding: 4,
    selectionAllowance: 12,
    dockGap: 5,
  };
  const input = {
    agents,
    requests,
    viewport: VIEWPORT,
  };
  const forward = layoutRequiredDocks(input, options);
  const reversed = layoutRequiredDocks(
    {
      ...input,
      agents: [...agents].reverse(),
      requests: [...requests].reverse(),
    },
    options,
  );

  assert.deepEqual(forward, reversed);
  assert.deepEqual(forward.placementOrder, [
    "status:2",
    "status:7",
    "cooldown:2",
    "cooldown:7",
  ]);
  assert.deepEqual(forward.compactedLayoutKeys, ["cooldown:2"]);
  assert.deepEqual(forward.suppressedLayoutKeys, []);
  assert.deepEqual(
    forward.docks.map(({ layoutKey }) => layoutKey),
    ["cooldown:2", "status:2", "cooldown:7", "status:7"],
  );
  const cooldownFallback = forward.docks.find(
    ({ layoutKey }) => layoutKey === "cooldown:2",
  );
  assert.ok(cooldownFallback);
  assert.equal(cooldownFallback.compactFallback, true);
  assert.equal(cooldownFallback.visibleCount, 0);
  assert.equal(cooldownFallback.hiddenCount, 1);
  assert.deepEqual(cooldownFallback.hiddenStatuses, [{ ticks: 30 }]);
  assert.equal(cooldownFallback.collisionFree, true);
  assert.equal(viewportOverflow(cooldownFallback.bounds, VIEWPORT), 0);
  assert.equal(cooldownFallback.bounds.width, 38);
  assert.equal(cooldownFallback.bounds.height, 18);
  for (const statusKey of ["status:2", "status:7"]) {
    const dock = forward.docks.find(({ layoutKey }) => layoutKey === statusKey);
    assert.ok(dock);
    assert.equal(dock.compactFallback, false);
    assert.equal(dock.expanded, true);
    assert.equal(dock.visibleCount, 9);
    assert.equal(dock.hiddenCount, 0);
  }
  for (let index = 0; index < forward.docks.length; index += 1) {
    for (let other = index + 1; other < forward.docks.length; other += 1) {
      assert.equal(
        rectanglesIntersect(forward.docks[index].bounds, forward.docks[other].bounds),
        false,
      );
    }
  }
});

test("required status fallback retains every selected status behind one compact marker", () => {
  const selectedStatuses = statuses(9, "selected-required");
  const layout = layoutRequiredDocks(
    {
      agents: [
        {
          globalSlot: 4,
          center: { x: 300, y: 200 },
          radius: 20,
          statuses: [],
          selected: true,
        },
      ],
      requests: [
        {
          layoutKey: "status:4",
          globalSlot: 4,
          statuses: selectedStatuses,
          dockOptions: {
            cellWidth: 1_000,
            cellHeight: 18,
            cellGap: 2,
            dockGap: 5,
          },
          priority: 0,
        },
      ],
      viewport: VIEWPORT,
    },
    {
      bodyPadding: 4,
      selectionAllowance: 12,
      dockGap: 5,
    },
  );

  assert.deepEqual(layout.compactedLayoutKeys, ["status:4"]);
  assert.deepEqual(layout.suppressedLayoutKeys, []);
  assert.equal(layout.docks.length, 1);
  const fallback = layout.docks[0];
  assert.equal(fallback.layoutKey, "status:4");
  assert.equal(fallback.compactFallback, true);
  assert.equal(fallback.visibleCount, 0);
  assert.equal(fallback.hiddenCount, 9);
  assert.equal(fallback.totalCount, 9);
  assert.deepEqual(fallback.hiddenStatuses, selectedStatuses);
  assert.equal(fallback.collisionFree, true);
  assert.equal(viewportOverflow(fallback.bounds, VIEWPORT), 0);
  assert.equal(
    rectanglesIntersect(fallback.bounds, layout.protectedBodies[0].bounds),
    false,
  );
});

test("large required-dock sets take the bounded deterministic priority path", () => {
  const agents = Array.from({ length: 20 }, (_, globalSlot) => ({
    globalSlot,
    center: { x: 300, y: 200 },
    radius: 20,
    statuses: [],
    controlled: globalSlot === 0,
    selected: globalSlot === 1,
  }));
  const requests = agents.map(({ globalSlot }) => ({
    layoutKey: `cooldown:${globalSlot}`,
    globalSlot,
    statuses: [{ ticks: 30 }],
    dockOptions: {
      cellWidth: 38,
      cellHeight: 18,
      cellGap: 2,
      dockGap: 5,
    },
    priority: 1,
  }));
  const input = { agents, requests, viewport: VIEWPORT };

  const startedAt = performance.now();
  const forward = layoutRequiredDocks(input);
  const elapsedMilliseconds = performance.now() - startedAt;
  const reversed = layoutRequiredDocks({
    ...input,
    agents: [...agents].reverse(),
    requests: [...requests].reverse(),
  });

  assert.ok(
    elapsedMilliseconds < 1_000,
    `required dock layout exceeded its bounded budget: ${elapsedMilliseconds}ms`,
  );
  assert.deepEqual(forward, reversed);
  assert.equal(forward.suppressedLayoutKeys.length, 0);
  assert.equal(forward.docks.length, requests.length);
  assert.equal(
    forward.docks.filter(({ compactFallback }) => compactFallback).length > 0,
    true,
  );
  for (let index = 0; index < forward.docks.length; index += 1) {
    const dock = forward.docks[index];
    assert.equal(dock.collisionFree, true);
    assert.equal(viewportOverflow(dock.bounds, VIEWPORT), 0);
    for (let other = index + 1; other < forward.docks.length; other += 1) {
      assert.equal(
        rectanglesIntersect(dock.bounds, forward.docks[other].bounds),
        false,
      );
    }
  }
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

test("cross-phase occupancy filters first and is invariant to input order", () => {
  const filtered = /** @type {any} */ ({
    layoutKey: "filtered:private-geometry",
    enabled: false,
    stableOrder: 0,
    get kind() {
      throw new Error("filtered geometry was inspected");
    },
  });

  for (const viewport of [
    { left: 0, top: 0, right: 557, bottom: 384, width: 557, height: 384 },
    { left: 0, top: 0, right: 900, bottom: 600, width: 900, height: 600 },
  ]) {
    const anchor = { x: viewport.width / 2, y: viewport.height / 2 };
    const protectedRects = [
      {
        layoutKey: "body:recipient",
        bounds: testRectangle(
          anchor.x - 28,
          anchor.y - 28,
          anchor.x + 28,
          anchor.y + 28,
        ),
      },
      {
        layoutKey: "hud:northwest",
        bounds: testRectangle(8, 8, 96, 42),
      },
    ];
    const requests = [
      ...Array.from({ length: 8 }, (_, index) => ({
        layoutKey: `impact:${index}`,
        kind: "recipient_cue",
        priority: index % 2,
        stableOrder: index,
        anchor,
        anchorRadius: 28,
        recipientKey: "agent:recipient",
        allowProtectedKeys: ["body:recipient"],
        width: 40,
        height: 18,
      })),
      {
        layoutKey: "callout:team-wave",
        kind: "perimeter_callout",
        priority: 2,
        stableOrder: 9,
        anchor,
        anchorRadius: 28,
        recipientKey: "agent:recipient",
        allowProtectedKeys: ["body:recipient"],
        width: 72,
        height: 24,
      },
      filtered,
    ];
    const startedAt = performance.now();
    const forward = layoutCrossPhaseOccupancy({
      viewport,
      protectedRects,
      requests,
    });
    const elapsedMilliseconds = performance.now() - startedAt;
    const reversed = layoutCrossPhaseOccupancy({
      viewport,
      protectedRects: [...protectedRects].reverse(),
      requests: [...requests].reverse(),
    });

    assert.ok(elapsedMilliseconds < 1_000);
    assert.deepEqual(forward, reversed);
    assert.deepEqual(forward.filteredLayoutKeys, ["filtered:private-geometry"]);
    assert.equal(forward.placements.length, requests.length - 1);
    assert.equal(forward.cuePlacements.length, requests.length - 1);
    assert.equal(forward.cuePlacements.at(-1)?.disposition, "perimeter_callout");
    assert.deepEqual(forward.placementOrder.slice(0, 4), [
      "impact:0",
      "impact:2",
      "impact:4",
      "impact:6",
    ]);
    for (const [index, cue] of forward.cuePlacements.entries()) {
      assert.equal(viewportOverflow(cue.bounds, viewport), 0);
      assert.deepEqual(cue.leader.start, anchor);
      for (const region of forward.protectedRegions) {
        assert.equal(rectanglesIntersect(cue.bounds, region.bounds), false);
      }
      for (const other of forward.cuePlacements.slice(index + 1)) {
        assert.equal(rectanglesIntersect(cue.bounds, other.bounds), false);
      }
    }
  }
});

test("cross-phase recipient cues reuse every safe nearby direction without ordinal displacement", () => {
  const anchor = Object.freeze({ x: 300, y: 200 });
  const protectedRects = [
    {
      layoutKey: "body:recipient",
      bounds: testRectangle(277, 177, 323, 223),
    },
  ];
  const requests = Array.from({ length: 9 }, (_, index) => ({
    layoutKey: `cue:${String(index).padStart(2, "0")}`,
    kind: /** @type {const} */ ("recipient_cue"),
    stableOrder: index,
    anchor,
    anchorRadius: 60,
    recipientKey: "agent:recipient",
    allowProtectedKeys: ["body:recipient"],
    width: 42,
    height: 18,
  }));
  const forward = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects,
    requests,
  });
  const reversed = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects: [...protectedRects].reverse(),
    requests: [...requests].reverse(),
  });
  const centerDistance = (/** @type {Record<string, any>} */ cue) =>
    Math.hypot(cue.center.x - anchor.x, cue.center.y - anchor.y);
  const nearestDistance =
    requests[0].anchorRadius +
    8 +
    Math.hypot(requests[0].width, requests[0].height) / 2;

  assert.deepEqual(forward, reversed);
  assert.deepEqual(
    forward.cuePlacements.map(({ stackIndex }) => stackIndex),
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
  );
  for (const cue of forward.cuePlacements) {
    assertClose(centerDistance(cue), nearestDistance);
    assert.equal(cue.disposition, "recipient_stack");
    assert.equal(viewportOverflow(cue.bounds, VIEWPORT), 0);
  }
  for (const [index, cue] of forward.cuePlacements.entries()) {
    for (const other of forward.cuePlacements.slice(index + 1)) {
      assert.equal(rectanglesIntersect(cue.bounds, other.bounds), false);
    }
  }
});

test("cross-phase compaction uses a free interstitial angle without starving the sealed layout", () => {
  const anchor = Object.freeze({ x: 300, y: 200 });
  const width = 10;
  const height = 10;
  const anchorRadius = 20;
  const nearestDistance = anchorRadius + 8 + Math.hypot(width, height) / 2;
  const stackStep = Math.max(width, height) + 5;
  const coarseDirections = [
    [0, -1],
    [1, 0],
    [-1, 0],
    [0, 1],
    [Math.SQRT1_2, -Math.SQRT1_2],
    [-Math.SQRT1_2, -Math.SQRT1_2],
    [Math.SQRT1_2, Math.SQRT1_2],
    [-Math.SQRT1_2, Math.SQRT1_2],
  ];
  const protectedRects = [nearestDistance, nearestDistance + stackStep].flatMap(
    (distance, ring) =>
      coarseDirections.map(([dx, dy], index) => {
        const center = {
          x: anchor.x + dx * distance,
          y: anchor.y + dy * distance,
        };
        return {
          layoutKey: `blocker:coarse:${ring}:${index}`,
          bounds: testRectangle(center.x - 1, center.y - 1, center.x + 1, center.y + 1),
        };
      }),
  );
  const request = {
    layoutKey: "cue:interstitial",
    kind: /** @type {const} */ ("recipient_cue"),
    stableOrder: 0,
    anchor,
    anchorRadius,
    width,
    height,
  };
  const forward = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects,
    requests: [request],
  });
  const reversed = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects: [...protectedRects].reverse(),
    requests: [request],
  });
  const cue = forward.cuePlacements[0];

  assert.deepEqual(forward, reversed);
  assertClose(
    Math.hypot(cue.center.x - anchor.x, cue.center.y - anchor.y),
    nearestDistance,
  );
  assert.equal(forward.cuePlacements.length, 1);
  assert.equal(cue.stackIndex, 0);
  assert.equal(cue.disposition, "recipient_stack");
  assert.equal(cue.collisionFree, true);
  assert.equal(
    coarseDirections.some(
      ([dx, dy]) =>
        Math.hypot(
          cue.center.x - (anchor.x + dx * nearestDistance),
          cue.center.y - (anchor.y + dy * nearestDistance),
        ) <= 1e-6,
    ),
    false,
  );
  for (const region of forward.protectedRegions) {
    assert.equal(rectanglesIntersect(cue.bounds, region.bounds), false);
  }
});

test("cross-phase compaction uses a free radial interstice inside the first outer ring", () => {
  const anchor = Object.freeze({ x: 300, y: 200 });
  const width = 10;
  const height = 10;
  const anchorRadius = 20;
  const nearestDistance = anchorRadius + 8 + Math.hypot(width, height) / 2;
  const stackStep = Math.max(width, height) + 5;
  const directions = [
    [0, -1],
    [1, 0],
    [-1, 0],
    [0, 1],
    [Math.SQRT1_2, -Math.SQRT1_2],
    [-Math.SQRT1_2, -Math.SQRT1_2],
    [Math.SQRT1_2, Math.SQRT1_2],
    [-Math.SQRT1_2, Math.SQRT1_2],
    ...Array.from({ length: 30 }, (_, index) => {
      const radians = ((4 + index * 12) * Math.PI) / 180;
      return [Math.cos(radians), Math.sin(radians)];
    }),
  ];
  const protectedRects = directions.map(([dx, dy], index) => {
    const center = {
      x: anchor.x + dx * nearestDistance,
      y: anchor.y + dy * nearestDistance,
    };
    return {
      layoutKey: `blocker:nearest:${index}`,
      bounds: testRectangle(center.x - 1, center.y - 1, center.x + 1, center.y + 1),
    };
  });
  const result = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects,
    requests: [
      {
        layoutKey: "cue:radial-interstice",
        kind: "recipient_cue",
        stableOrder: 0,
        anchor,
        anchorRadius,
        width,
        height,
      },
    ],
  });
  const cue = result.cuePlacements[0];
  const distance = Math.hypot(cue.center.x - anchor.x, cue.center.y - anchor.y);

  assert.ok(distance > nearestDistance);
  assert.ok(distance < nearestDistance + stackStep);
  assert.equal(cue.disposition, "recipient_stack");
  assert.equal(cue.collisionFree, true);
  for (const region of result.protectedRegions) {
    assert.equal(rectanglesIntersect(cue.bounds, region.bounds), false);
  }
});

test("cross-phase compaction refines a still-far cue between primary angular spokes", () => {
  const anchor = Object.freeze({ x: 300, y: 200 });
  const width = 1;
  const height = 1;
  const anchorRadius = 60;
  const nearestDistance = anchorRadius + 8 + Math.hypot(width, height) / 2;
  const stackStep = Math.max(width, height) + 5;
  const primaryDirections = [
    [0, -1],
    [1, 0],
    [-1, 0],
    [0, 1],
    [Math.SQRT1_2, -Math.SQRT1_2],
    [-Math.SQRT1_2, -Math.SQRT1_2],
    [Math.SQRT1_2, Math.SQRT1_2],
    [-Math.SQRT1_2, Math.SQRT1_2],
    ...Array.from({ length: 30 }, (_, index) => {
      const radians = ((4 + index * 12) * Math.PI) / 180;
      return [Math.cos(radians), Math.sin(radians)];
    }),
  ];
  const protectedRects = [
    nearestDistance,
    nearestDistance + 4,
    nearestDistance + stackStep,
  ].flatMap((distance, ring) =>
    primaryDirections.map(([dx, dy], index) => {
      const center = {
        x: anchor.x + dx * distance,
        y: anchor.y + dy * distance,
      };
      return {
        layoutKey: `blocker:primary:${ring}:${index}`,
        bounds: testRectangle(
          center.x - 0.05,
          center.y - 0.05,
          center.x + 0.05,
          center.y + 0.05,
        ),
      };
    }),
  );
  const result = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects,
    requests: [
      {
        layoutKey: "cue:angular-refinement",
        kind: "recipient_cue",
        stableOrder: 0,
        anchor,
        anchorRadius,
        allowProtectedKeys: protectedRects.map(({ layoutKey }) => layoutKey),
        width,
        height,
      },
    ],
  });
  const cue = result.cuePlacements[0];

  assertClose(
    Math.hypot(cue.center.x - anchor.x, cue.center.y - anchor.y),
    nearestDistance,
  );
  assert.equal(cue.disposition, "recipient_stack");
  assert.equal(cue.collisionFree, true);
  for (const region of result.protectedRegions) {
    assert.equal(rectanglesIntersect(cue.bounds, region.bounds), false);
  }
});

test("cross-phase fallback prefers the nearest collision-free cue independent of its connector", () => {
  const anchor = Object.freeze({ x: 300, y: 200 });
  const protectedRects = [
    {
      layoutKey: "blocker:northwest",
      bounds: testRectangle(200, 10, 240, 50),
    },
    {
      layoutKey: "blocker:west",
      bounds: testRectangle(10, 140, 50, 180),
    },
  ];
  const request = {
    layoutKey: "callout:nearest",
    kind: /** @type {const} */ ("perimeter_callout"),
    stableOrder: 0,
    anchor,
    width: 40,
    height: 20,
  };
  const forward = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects,
    requests: [request],
  });
  const reversed = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects: [...protectedRects].reverse(),
    requests: [request],
  });
  const cue = forward.cuePlacements[0];

  assert.deepEqual(forward, reversed);
  assert.deepEqual(cue.center, { x: 263, y: 193 });
  assert.equal(cue.disposition, "perimeter_callout");
  assert.ok(Math.hypot(cue.center.x - anchor.x, cue.center.y - anchor.y) < 40);
  assert.equal(cue.leader.kind, "line");
  assert.deepEqual(cue.leader.points, [anchor, cue.center]);
  assert.deepEqual(cue.leader.start, anchor);
  assert.deepEqual(cue.leader.end, cue.center);
  assert.equal(viewportOverflow(cue.bounds, VIEWPORT), 0);
  for (const region of forward.protectedRegions) {
    assert.equal(rectanglesIntersect(cue.bounds, region.bounds), false);
  }
});

test("cross-phase routes use bounded lanes and deterministic bridge gaps", () => {
  const viewport = VIEWPORT;
  const protectedRects = [
    {
      layoutKey: "body:left",
      bounds: testRectangle(55, 175, 105, 225),
    },
    {
      layoutKey: "body:right",
      bounds: testRectangle(495, 175, 545, 225),
    },
    {
      layoutKey: "body:top",
      bounds: testRectangle(275, 25, 325, 75),
    },
    {
      layoutKey: "body:bottom",
      bounds: testRectangle(275, 325, 325, 375),
    },
  ];
  /**
   * @param {string} layoutKey
   * @param {number} stableOrder
   * @param {{x: number, y: number}} source
   * @param {{x: number, y: number}} target
   * @param {string[]} allowProtectedKeys
   */
  const route = (layoutKey, stableOrder, source, target, allowProtectedKeys) => ({
    layoutKey,
    kind: /** @type {const} */ ("route"),
    stableOrder,
    source,
    target,
    sourceRadius: 25,
    targetRadius: 25,
    allowProtectedKeys,
  });
  const requests = [
    route("route:left-right:a", 0, { x: 80, y: 200 }, { x: 520, y: 200 }, [
      "body:left",
      "body:right",
    ]),
    route("route:right-left", 1, { x: 520, y: 200 }, { x: 80, y: 200 }, [
      "body:right",
      "body:left",
    ]),
    route("route:top-bottom", 2, { x: 300, y: 50 }, { x: 300, y: 350 }, [
      "body:top",
      "body:bottom",
    ]),
    ...Array.from({ length: 5 }, (_, index) =>
      route(
        `route:left-right:dense:${index}`,
        index + 3,
        { x: 80, y: 200 },
        { x: 520, y: 200 },
        ["body:left", "body:right"],
      ),
    ),
  ];
  const forward = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects,
    requests,
  });
  const reversed = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects: [...protectedRects].reverse(),
    requests: [...requests].reverse(),
  });

  assert.deepEqual(forward, reversed);
  assert.equal(forward.routePlacements.length, requests.length);
  assert.equal(forward.placements.length, requests.length);
  assert.equal(
    new Set(forward.routePlacements.map(({ offset }) => offset)).size > 1,
    true,
  );
  assert.equal(
    forward.routePlacements.some(({ bridgeGaps }) => bridgeGaps.length > 0),
    true,
  );
  for (const routePlacement of forward.routePlacements) {
    const ledgerEntry = forward.occupancyLedger.find(
      ({ layoutKey }) => layoutKey === routePlacement.layoutKey,
    );
    assert.ok(ledgerEntry);
    assert.equal(viewportOverflow(ledgerEntry.bounds, viewport), 0);
  }
});

test("cross-phase routes pass behind cues while still avoiding durable regions", () => {
  const protectedRects = [
    {
      layoutKey: "status:durable-recipient",
      bounds: testRectangle(270, 202, 330, 242),
    },
  ];
  const routeRequest = {
    layoutKey: "route:layered-behind-cue",
    kind: /** @type {const} */ ("route"),
    stableOrder: 1,
    source: { x: 60, y: 200 },
    target: { x: 540, y: 200 },
  };
  const routeOnly = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects,
    requests: [routeRequest],
  }).routePlacements[0];
  assert.equal(routeOnly.kind, "curve");
  assert.equal(routeOnly.lane, 0);
  assert.equal(routeOnly.offset, 0);
  assert.equal(
    routeOnly.control.x > protectedRects[0].bounds.left &&
      routeOnly.control.x < protectedRects[0].bounds.right &&
      routeOnly.control.y < protectedRects[0].bounds.top &&
      routeOnly.control.y > protectedRects[0].bounds.top - 3,
    true,
  );

  const progress = 0.2;
  const remainder = 1 - progress;
  const routePoint = {
    x:
      remainder * remainder * routeOnly.start.x +
      2 * remainder * progress * routeOnly.control.x +
      progress * progress * routeOnly.end.x,
    y:
      remainder * remainder * routeOnly.start.y +
      2 * remainder * progress * routeOnly.control.y +
      progress * progress * routeOnly.end.y,
  };
  const cueWidth = 6;
  const cueHeight = 6;
  const firstCandidateDistance = 8 + Math.hypot(cueWidth, cueHeight) / 2;
  const layout = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects,
    requests: [
      {
        layoutKey: "cue:foreground",
        kind: "recipient_cue",
        stableOrder: 0,
        anchor: {
          x: routePoint.x,
          y: routePoint.y + firstCandidateDistance,
        },
        width: cueWidth,
        height: cueHeight,
      },
      routeRequest,
    ],
  });
  const cue = layout.cuePlacements[0];
  const route = layout.routePlacements[0];

  assert.deepEqual(route, routeOnly);
  assert.equal(
    routePoint.x > cue.bounds.left &&
      routePoint.x < cue.bounds.right &&
      routePoint.y > cue.bounds.top &&
      routePoint.y < cue.bounds.bottom,
    true,
  );
  for (const region of protectedRects) {
    for (let index = 0; index <= 128; index += 1) {
      const fraction = index / 128;
      const inverse = 1 - fraction;
      /** @type {{x: number, y: number}} */
      const point = {
        x:
          inverse * inverse * route.start.x +
          2 * inverse * fraction * route.control.x +
          fraction * fraction * route.end.x,
        y:
          inverse * inverse * route.start.y +
          2 * inverse * fraction * route.control.y +
          fraction * fraction * route.end.y,
      };
      assert.equal(
        point.x > region.bounds.left &&
          point.x < region.bounds.right &&
          point.y > region.bounds.top &&
          point.y < region.bounds.bottom,
        false,
        `${route.layoutKey} crossed ${region.layoutKey}`,
      );
    }
  }
});

test("cross-phase route markers honor their explicit painted padding", () => {
  const blocker = testRectangle(270, 175, 330, 225);
  const markerPadding = 17;
  const route = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects: [{ layoutKey: "durable:marker", bounds: blocker }],
    requests: [
      {
        layoutKey: "route:painted-marker",
        kind: "route",
        stableOrder: 0,
        source: { x: 50, y: 200 },
        target: { x: 550, y: 200 },
        pathPadding: 2,
        markerPadding,
        markerProgress: 0.5,
      },
    ],
  }).routePlacements[0];
  const marker = routeMarkerPose(route);
  assert.ok(marker.x >= markerPadding && marker.x <= VIEWPORT.right - markerPadding);
  assert.ok(marker.y >= markerPadding && marker.y <= VIEWPORT.bottom - markerPadding);
  assert.equal(
    marker.x >= blocker.left - markerPadding &&
      marker.x <= blocker.right + markerPadding &&
      marker.y >= blocker.top - markerPadding &&
      marker.y <= blocker.bottom + markerPadding,
    false,
  );
});

test("cross-phase routes persist the first collision-free marker progress", () => {
  const route = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects: [
      {
        layoutKey: "durable:preferred-marker",
        bounds: testRectangle(425, 180, 435, 190),
      },
    ],
    requests: [
      {
        layoutKey: "route:marker-progress-search",
        kind: "route",
        stableOrder: 0,
        source: { x: 50, y: 200 },
        target: { x: 550, y: 200 },
        sourceEndpointGap: 0,
        targetEndpointGap: 0,
        markerPadding: 12,
      },
    ],
  }).routePlacements[0];

  assert.equal(route.lane, 0);
  assert.equal(route.markerVariant, "full");
  assert.equal(route.markerPadding, 12);
  assert.equal(route.markerProgress, 0.5);
  assert.deepEqual(routeMarkerPose(route), {
    x: 300,
    y: 200,
    degrees: 0,
  });
});

test("cross-phase occupancy routes around protected regions and fails loudly when full", () => {
  const obstacle = {
    layoutKey: "obstacle:center",
    bounds: testRectangle(270, 165, 330, 235),
  };
  const layout = layoutCrossPhaseOccupancy({
    viewport: VIEWPORT,
    protectedRects: [obstacle],
    requests: [
      {
        layoutKey: "route:around-obstacle",
        kind: "route",
        stableOrder: 0,
        source: { x: 60, y: 200 },
        target: { x: 540, y: 200 },
      },
    ],
  });

  assert.ok(layout.routePlacements[0].lane > 0);
  assert.notEqual(layout.routePlacements[0].offset, 0);
  assert.throws(
    () =>
      layoutCrossPhaseOccupancy({
        viewport: VIEWPORT,
        protectedRects: [
          /** @type {any} */ ({
            bounds: testRectangle(10, 10, 20, 20),
          }),
        ],
        requests: [],
      }),
    /layoutKey must be a non-empty string/,
  );
  assert.throws(
    () =>
      layoutCrossPhaseOccupancy({
        viewport: testRectangle(0, 0, 40, 30),
        protectedRects: [
          {
            layoutKey: "protected:all",
            bounds: testRectangle(0, 0, 40, 30),
          },
        ],
        requests: [
          {
            layoutKey: "cue:no-space",
            kind: "recipient_cue",
            stableOrder: 0,
            anchor: { x: 20, y: 15 },
            width: 10,
            height: 10,
          },
        ],
      }),
    /no bounded collision-free placement/,
  );
});

test("cross-phase recovery route uses an exact-owner polyline through the full durable ledger", () => {
  const viewport = testRectangle(85, 24, 477, 360);
  const owners = [
    "oracle_5dd75cfa5732092528aea81a7cae9f028e3c4d5f88f5d065cb3bd78b7f8e74da",
    "oracle_664fb2cb113a27150fd817466a60195118673b5b0f14e5a7559363467a66acc3",
    "oracle_1f3573a7927eaf682411975883f3573add02ab40bc1ee5bd006e3e4b117cd65d",
    "oracle_786f10714c2c024c3efa8079315522d9c2825a3a1037a7a82496beeaadf2d6a4",
    "oracle_4e266edd9d60abf5a1f9fdbc03133daa3152a39697d7f9f755c115f812858f40",
    "oracle_937cef949c321a6118bc4c4965ae94b8abc26d99033e75e63a470581267e0e94",
    "oracle_e17d6ab5f10641e78ca7eddec47a68bed3bd95c86d71ad7f00b9e7b96506cf0e",
    "oracle_a1ca6eea50b21798284ccea4d05c77efcdfe284b8b68d067916a2b49050f8b43",
  ];
  /** @param {string} kind @param {number} slot @param {string} [placement] */
  const protectedKey = (kind, slot, placement = `${kind}:${slot}`) =>
    JSON.stringify(["durable", kind, owners[slot], placement]);
  const sourceProtectedKey = protectedKey("body", 2);
  const targetProtectedKey = protectedKey("body", 7);
  /** @type {Array<[string, ReturnType<typeof testRectangle>]>} */
  const protectedEntries = [
    [protectedKey("body", 0), testRectangle(235, 216, 271, 252)],
    [protectedKey("body", 1), testRectangle(235, 188, 271, 224)],
    [sourceProtectedKey, testRectangle(198.5999946594, 146, 234.5999946594, 182)],
    [protectedKey("body", 3), testRectangle(221, 118, 257, 154)],
    [protectedKey("body", 4), testRectangle(249, 146, 285, 182)],
    [protectedKey("body", 5), testRectangle(268.5999946594, 202, 304.5999946594, 238)],
    [protectedKey("body", 6), testRectangle(375, 90, 411, 126)],
    [targetProtectedKey, testRectangle(277, 132, 313, 168)],
    [protectedKey("modifier", 1), testRectangle(276, 182, 318, 198)],
    [
      protectedKey("modifier", 2),
      testRectangle(171.5999946594, 125, 213.5999946594, 141),
    ],
    [protectedKey("modifier", 3), testRectangle(174, 104, 216, 120)],
    [protectedKey("modifier", 7), testRectangle(318, 142, 360, 158)],
    [protectedKey("cooldown", 0), testRectangle(192, 225, 230, 243)],
    [protectedKey("cooldown", 1), testRectangle(192, 197, 230, 215)],
    [
      protectedKey("cooldown", 2),
      testRectangle(155.5999946594, 155, 193.5999946594, 173),
    ],
    [protectedKey("cooldown", 3), testRectangle(220, 95, 258, 113)],
    [
      protectedKey("status", 5),
      testRectangle(309.5999946594, 211, 337.5999946594, 229),
    ],
    [protectedKey("status", 7), testRectangle(281, 109, 309, 127)],
  ];
  const protectedRects = protectedEntries.map(([layoutKey, bounds]) => ({
    layoutKey,
    bounds,
  }));
  const request = {
    layoutKey: JSON.stringify(["event", "transition:0:event:0002", "route"]),
    kind: /** @type {const} */ ("route"),
    stableOrder: 2,
    source: { x: 216.5999946594, y: 164 },
    target: { x: 295, y: 150 },
    sourceRadius: 14,
    targetRadius: 14,
    sourceEndpointGap: 3,
    targetEndpointGap: 3,
    allowProtectedKeys: [sourceProtectedKey, targetProtectedKey],
    sourceProtectedKey,
    targetProtectedKey,
  };
  const forward = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects,
    requests: [request],
  });
  const reversed = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects: [...protectedRects].reverse(),
    requests: [request],
  });

  assert.deepEqual(forward, reversed);
  const route = forward.routePlacements[0];
  assert.equal(route.kind, "polyline");
  if (route.kind !== "polyline") assert.fail("expected dense polyline fallback");
  assert.ok(route.points.length >= 3);
  assert.ok(route.points.every(({ x, y }) => Number.isFinite(x) && Number.isFinite(y)));
  const sourceBounds = protectedRects.find(
    ({ layoutKey }) => layoutKey === sourceProtectedKey,
  )?.bounds;
  const targetBounds = protectedRects.find(
    ({ layoutKey }) => layoutKey === targetProtectedKey,
  )?.bounds;
  assert.ok(sourceBounds);
  assert.ok(targetBounds);
  /**
   * @param {{x: number, y: number}} point
   * @param {ReturnType<typeof testRectangle>} bounds
   */
  const onBoundary = (point, bounds) =>
    point.x >= bounds.left - 1e-9 &&
    point.x <= bounds.right + 1e-9 &&
    point.y >= bounds.top - 1e-9 &&
    point.y <= bounds.bottom + 1e-9 &&
    (Math.abs(point.x - bounds.left) <= 1e-9 ||
      Math.abs(point.x - bounds.right) <= 1e-9 ||
      Math.abs(point.y - bounds.top) <= 1e-9 ||
      Math.abs(point.y - bounds.bottom) <= 1e-9);
  assert.equal(onBoundary(route.start, sourceBounds), true);
  assert.equal(onBoundary(route.end, targetBounds), true);
  const blockers = protectedRects.filter(
    ({ layoutKey }) =>
      layoutKey !== sourceProtectedKey && layoutKey !== targetProtectedKey,
  );
  for (const point of route.points) {
    assert.equal(
      viewportOverflow(testRectangle(point.x, point.y, point.x, point.y), viewport),
      0,
    );
  }
  for (let index = 1; index < route.points.length; index += 1) {
    for (const blocker of blockers) {
      assert.equal(
        segmentTouchesRectangle(
          route.points[index - 1],
          route.points[index],
          blocker.bounds,
        ),
        false,
        `route segment ${index - 1} touched ${blocker.layoutKey}`,
      );
    }
  }

  const poisonSourceProtectedKey = protectedKey("body", 1);
  const poisonTargetProtectedKey = protectedKey("body", 5);
  const poisonRoute = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects,
    requests: [
      {
        layoutKey: JSON.stringify(["event", "transition:1:event:0000", "route"]),
        kind: "route",
        stableOrder: 2,
        source: { x: 253, y: 206 },
        target: { x: 286.5999946594, y: 220 },
        sourceRadius: 14,
        targetRadius: 14,
        sourceEndpointGap: 3,
        targetEndpointGap: 3,
        pathPadding: 7,
        markerPadding: 17,
        compactMarkerPadding: 8,
        allowProtectedKeys: [poisonSourceProtectedKey, poisonTargetProtectedKey],
        sourceProtectedKey: poisonSourceProtectedKey,
        targetProtectedKey: poisonTargetProtectedKey,
      },
    ],
  }).routePlacements[0];
  assert.equal(poisonRoute.markerVariant, "full");
  assert.equal(poisonRoute.markerPadding, 17);
  const poisonMarker = routeMarkerPose(poisonRoute);
  for (const blocker of protectedRects.filter(
    ({ layoutKey }) =>
      layoutKey !== poisonSourceProtectedKey && layoutKey !== poisonTargetProtectedKey,
  )) {
    assert.equal(
      poisonMarker.x >= blocker.bounds.left - poisonRoute.markerPadding &&
        poisonMarker.x <= blocker.bounds.right + poisonRoute.markerPadding &&
        poisonMarker.y >= blocker.bounds.top - poisonRoute.markerPadding &&
        poisonMarker.y <= blocker.bounds.bottom + poisonRoute.markerPadding,
      false,
      `compact activation marker touched ${blocker.layoutKey}`,
    );
  }

  const freeSourceProtectedRects = protectedRects.filter(
    ({ layoutKey }) => layoutKey !== sourceProtectedKey,
  );
  const freeSource = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects: freeSourceProtectedRects,
    requests: [
      {
        ...request,
        allowProtectedKeys: [targetProtectedKey],
        sourceProtectedKey: undefined,
      },
    ],
  }).routePlacements[0];
  assert.equal(freeSource.kind, "polyline");
  assert.deepEqual(freeSource.start, request.source);
  assert.equal(onBoundary(freeSource.end, targetBounds), true);

  const freeEndpoints = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects,
    requests: [
      {
        ...request,
        sourceProtectedKey: undefined,
        targetProtectedKey: undefined,
      },
    ],
  }).routePlacements[0];
  assert.equal(freeEndpoints.kind, "polyline");
  assert.deepEqual(freeEndpoints.start, request.source);
  assert.deepEqual(freeEndpoints.end, request.target);
});

test("cross-phase Rogue route uses its exact painted corridor through the full stationary ledger", () => {
  const viewport = testRectangle(166.16666666666663, 24, 786.8333333333334, 556);
  const owners = [
    "oracle_6056027dbfc142faeff1f285b4c6cdef77fac07e234da4751fec117b2564cae1",
    "oracle_400854500fbdbbac96f2d2a9f26889e5dfb087f0943e52d42ec611a48145dcc8",
    "oracle_7259b2c05c905e807fa6aa957193a4e0692818431c012d62c4800310f4038e8d",
    "oracle_4c023af28dd78dcc3ab0c40992897d8e0903a9ceea864257386395be3c6fe16a",
    "oracle_eb159f6835d32d6021522f681a812e161ebe9358df301b499573640879a26623",
    "oracle_bb38cf93d84b5caa9172ef73c3ff247dbd85eca5369cabe4558154ad69845d49",
    "oracle_6b804ae8c628cfd9b3c578834c26447f450c4fe52bd56c1582c12de84c51d25b",
    "oracle_6c9d5d03f9e9d2550eb7be8691a478e820f517970e9bdefc94428f0330b43dac",
  ];
  /** @param {string} kind @param {number} slot @param {string} [placement] */
  const protectedKey = (kind, slot, placement = `${kind}:${slot}`) =>
    JSON.stringify(["durable", kind, owners[slot], placement]);
  const sourceProtectedKey = protectedKey("body", 1);
  const targetProtectedKey = protectedKey("body", 5);
  const protectedRects = [
    [
      protectedKey("body", 0),
      testRectangle(394, 318.3333333333333, 470.3333333333333, 394.6666666666667),
    ],
    [
      sourceProtectedKey,
      testRectangle(406, 286, 458.3333333333333, 338.33333333333337),
    ],
    [
      protectedKey("body", 2),
      testRectangle(348.3666582107543, 219.5, 400.6999915440877, 271.83333333333337),
    ],
    [
      protectedKey("body", 3),
      testRectangle(383.8333333333333, 175.16666666666669, 436.1666666666667, 227.5),
    ],
    [
      protectedKey("body", 4),
      testRectangle(428.16666666666663, 219.5, 480.5, 271.83333333333337),
    ],
    [
      targetProtectedKey,
      testRectangle(459.1999915440877, 308.1666666666667, 511.5333248774211, 360.5),
    ],
    [
      protectedKey("body", 6),
      testRectangle(627.6666666666666, 130.83333333333331, 680, 183.16666666666669),
    ],
    [
      protectedKey("body", 7),
      testRectangle(472.5, 197.33333333333331, 524.8333333333333, 249.66666666666669),
    ],
    [
      protectedKey("modifier", 1),
      testRectangle(
        463.3333435058594,
        288.1666564941406,
        505.3333435058594,
        304.1666564941406,
      ),
    ],
    [
      protectedKey("modifier", 2),
      testRectangle(
        353.5333251953125,
        276.8333435058594,
        395.5333251953125,
        292.8333435058594,
      ),
    ],
    [
      protectedKey("modifier", 3),
      testRectangle(
        336.8333435058594,
        193.3333282470703,
        378.8333435058594,
        209.3333282470703,
      ),
    ],
    [
      protectedKey("modifier", 4),
      testRectangle(485.5, 253.6666717529297, 527.5, 269.6666717529297),
    ],
    [
      protectedKey("modifier", 7),
      testRectangle(529.8333129882812, 215.5, 571.8333129882812, 231.5),
    ],
    [protectedKey("cooldown", 0), testRectangle(351, 347.5, 389, 365.5)],
    [
      protectedKey("cooldown", 1),
      testRectangle(363, 295.1666564941406, 401, 313.1666564941406),
    ],
    [
      protectedKey("cooldown", 2),
      testRectangle(
        305.3666687011719,
        236.6666717529297,
        343.3666687011719,
        254.6666717529297,
      ),
    ],
    [
      protectedKey("cooldown", 3),
      testRectangle(391, 152.1666717529297, 429, 170.1666717529297),
    ],
    [
      protectedKey("status", 5),
      testRectangle(
        516.5333251953125,
        325.3333435058594,
        604.5333251953125,
        343.3333435058594,
      ),
    ],
    [
      protectedKey("status", 7),
      testRectangle(
        484.6666564941406,
        174.3333282470703,
        512.6666564941406,
        192.3333282470703,
      ),
    ],
    [
      protectedKey("legality", 0, owners[0]),
      testRectangle(
        400.6666564941406,
        399.6666564941406,
        463.6666564941406,
        417.6666564941406,
      ),
    ],
  ].map(([layoutKey, bounds]) => ({
    layoutKey: /** @type {string} */ (layoutKey),
    bounds: /** @type {ReturnType<typeof testRectangle>} */ (bounds),
  }));
  const request = {
    layoutKey: JSON.stringify([
      "event",
      "debugger-episode:82121c4fd8538462d081408245ae977189d2880b6626c054ba88792fb3c901b4:transition:1:event:0000",
      "route",
    ]),
    kind: /** @type {const} */ ("route"),
    stableOrder: 0,
    source: { x: 432.16666666666663, y: 312.1666666666667 },
    target: { x: 485.3666582107544, y: 334.33333333333337 },
    sourceRadius: 22.166666666666668,
    targetRadius: 22.166666666666668,
    sourceEndpointGap: 3,
    targetEndpointGap: 3,
    pathPadding: 7,
    markerPadding: 17,
    compactMarkerPadding: 8,
    allowProtectedKeys: [sourceProtectedKey, targetProtectedKey],
    sourceProtectedKey,
    targetProtectedKey,
  };

  const forward = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects,
    requests: [request],
  });
  const reversed = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects: [...protectedRects].reverse(),
    requests: [request],
  });
  assert.deepEqual(forward, reversed);
  assert.deepEqual(request.source, {
    x: 432.16666666666663,
    y: 312.1666666666667,
  });
  assert.deepEqual(request.target, {
    x: 485.3666582107544,
    y: 334.33333333333337,
  });

  const route = forward.routePlacements[0];
  assert.equal(route.kind, "polyline");
  if (route.kind !== "polyline") assert.fail("expected dense Rogue polyline");
  assert.equal(route.markerVariant, "full");
  assert.equal(route.markerPadding, 17);
  assert.ok(route.points.length >= 3);
  const sourceBounds = protectedRects.find(
    ({ layoutKey }) => layoutKey === sourceProtectedKey,
  )?.bounds;
  const targetBounds = protectedRects.find(
    ({ layoutKey }) => layoutKey === targetProtectedKey,
  )?.bounds;
  assert.ok(sourceBounds);
  assert.ok(targetBounds);
  /** @param {{x: number, y: number}} point @param {ReturnType<typeof testRectangle>} bounds */
  const onBoundary = (point, bounds) =>
    point.x >= bounds.left - 1e-9 &&
    point.x <= bounds.right + 1e-9 &&
    point.y >= bounds.top - 1e-9 &&
    point.y <= bounds.bottom + 1e-9 &&
    (Math.abs(point.x - bounds.left) <= 1e-9 ||
      Math.abs(point.x - bounds.right) <= 1e-9 ||
      Math.abs(point.y - bounds.top) <= 1e-9 ||
      Math.abs(point.y - bounds.bottom) <= 1e-9);
  assert.equal(onBoundary(route.start, sourceBounds), true);
  assert.equal(onBoundary(route.end, targetBounds), true);
  for (const point of route.points) {
    assert.equal(Number.isFinite(point.x) && Number.isFinite(point.y), true);
    assert.equal(
      viewportOverflow(testRectangle(point.x, point.y, point.x, point.y), viewport),
      0,
    );
  }
  const nonowners = protectedRects.filter(
    ({ layoutKey }) =>
      layoutKey !== sourceProtectedKey && layoutKey !== targetProtectedKey,
  );
  for (let index = 1; index < route.points.length; index += 1) {
    for (const blocker of nonowners) {
      assert.equal(
        segmentTouchesRectangle(
          route.points[index - 1],
          route.points[index],
          testRectangle(
            blocker.bounds.left - request.pathPadding,
            blocker.bounds.top - request.pathPadding,
            blocker.bounds.right + request.pathPadding,
            blocker.bounds.bottom + request.pathPadding,
          ),
        ),
        false,
        `Rogue route segment ${index - 1} touched ${blocker.layoutKey}`,
      );
    }
  }
  const marker = routeMarkerPose(route);
  assert.equal(Number.isFinite(marker.x) && Number.isFinite(marker.y), true);
  assert.ok(marker.x >= viewport.left + route.markerPadding);
  assert.ok(marker.x <= viewport.right - route.markerPadding);
  assert.ok(marker.y >= viewport.top + route.markerPadding);
  assert.ok(marker.y <= viewport.bottom - route.markerPadding);
  for (const blocker of protectedRects) {
    assert.equal(
      marker.x >= blocker.bounds.left - route.markerPadding &&
        marker.x <= blocker.bounds.right + route.markerPadding &&
        marker.y >= blocker.bounds.top - route.markerPadding &&
        marker.y <= blocker.bounds.bottom + route.markerPadding,
      false,
      `Rogue full marker touched ${blocker.layoutKey}`,
    );
  }

  assert.throws(
    () =>
      layoutCrossPhaseOccupancy({
        viewport,
        protectedRects,
        requests: [{ ...request, pathPadding: 8 }],
      }),
    /no bounded protected-region lane/,
  );
});

test("cross-phase Charge route escapes an occluding free-endpoint status without allowing it", () => {
  const viewport = testRectangle(134.5, 24, 818.5, 556);
  const body1 = JSON.stringify([
    "durable",
    "body",
    "oracle_e6c6570ae63ba9770fd6c25cb9ac4d24dafd54400f3522dcd56d18a7b716de76",
    "body:1",
  ]);
  /** @type {Array<[string, ReturnType<typeof testRectangle>]>} */
  const protectedEntries = [
    [
      '["durable","body","oracle_84a3dcb370cd4536c750f954679abc09d7f1d506714d08041f74ecba23f6359a","body:3"]',
      testRectangle(388.9000072479248, 77, 434.9000072479248, 123),
    ],
    [
      '["durable","body","oracle_89c850cbb8b13e3a3f81c79ba88f8e881c4ce85923954dec4f458699d0944cfe","body:0"]',
      testRectangle(213.5, 407, 283.5, 477),
    ],
    [
      '["durable","body","oracle_8ff5a0e97b8f3a26f62ddacf513b1de5db08ceb37131f50ee9c46450f55bb310","body:2"]',
      testRectangle(362.2999963760376, 191, 408.2999963760376, 237),
    ],
    [
      '["durable","body","oracle_a5d615518b4ba987dcf8a07041b87ec2f3f722b573ce874688d1c661f5457c73","body:5"]',
      testRectangle(681.5, 419, 727.5, 465),
    ],
    [
      '["durable","body","oracle_b410ab12e3e9a7a40e7ecd07b7cbf965e88c066ea0f4950bf7576601abd05fdd","body:6"]',
      testRectangle(377.5, 343, 423.5, 389),
    ],
    [
      '["durable","body","oracle_b7bb9105532a4edea57095b69b999548c15cb023b0e551579ff60d601870780b","body:8"]',
      testRectangle(442.0999927520752, 77, 488.0999927520752, 123),
    ],
    [
      '["durable","body","oracle_b838ccc1da1b0126157ceb6b66fb8121bbc2b4b7323a84673e074e14566b8122","body:4"]',
      testRectangle(263.5, 96, 309.5, 142),
    ],
    [body1, testRectangle(453.5, 343, 499.5, 389)],
    [
      '["durable","body","oracle_f5851956a3f2b95859ab25c642187a8f9381ea1f04d93a020b72ee65b5cbdd93","body:7"]',
      testRectangle(468.6999855041504, 191, 514.6999855041504, 237),
    ],
    [
      '["durable","body","oracle_fa9437269f9a029bd98c219947d904403710bd603acc453d8b52528e2758da5a","body:9"]',
      testRectangle(567.5, 96, 613.5, 142),
    ],
    [
      '["durable","cooldown","oracle_89c850cbb8b13e3a3f81c79ba88f8e881c4ce85923954dec4f458699d0944cfe","cooldown:0"]',
      testRectangle(288.5, 433, 326.5, 451),
    ],
    [
      '["durable","cooldown","oracle_a5d615518b4ba987dcf8a07041b87ec2f3f722b573ce874688d1c661f5457c73","cooldown:5"]',
      testRectangle(685.5, 396, 723.5, 414),
    ],
    [
      '["durable","cooldown","oracle_b410ab12e3e9a7a40e7ecd07b7cbf965e88c066ea0f4950bf7576601abd05fdd","cooldown:6"]',
      testRectangle(381.5, 320, 419.5, 338),
    ],
    [
      '["durable","cooldown","oracle_e6c6570ae63ba9770fd6c25cb9ac4d24dafd54400f3522dcd56d18a7b716de76","cooldown:1"]',
      testRectangle(457.5, 320, 495.5, 338),
    ],
    [
      '["durable","legality","oracle_89c850cbb8b13e3a3f81c79ba88f8e881c4ce85923954dec4f458699d0944cfe","oracle_89c850cbb8b13e3a3f81c79ba88f8e881c4ce85923954dec4f458699d0944cfe"]',
      testRectangle(145.5, 433, 208.5, 451),
    ],
    [
      '["durable","modifier","oracle_a5d615518b4ba987dcf8a07041b87ec2f3f722b573ce874688d1c661f5457c73","modifier:5"]',
      testRectangle(634.5, 434, 676.5, 450),
    ],
    [
      '["durable","modifier","oracle_b410ab12e3e9a7a40e7ecd07b7cbf965e88c066ea0f4950bf7576601abd05fdd","modifier:6"]',
      testRectangle(379.5, 394, 421.5, 410),
    ],
    [
      '["durable","modifier","oracle_e6c6570ae63ba9770fd6c25cb9ac4d24dafd54400f3522dcd56d18a7b716de76","modifier:1"]',
      testRectangle(455.5, 394, 497.5, 410),
    ],
    [
      '["durable","status","oracle_89c850cbb8b13e3a3f81c79ba88f8e881c4ce85923954dec4f458699d0944cfe","status:0"]',
      testRectangle(234.5, 384, 262.5, 402),
    ],
    [
      '["durable","status","oracle_a5d615518b4ba987dcf8a07041b87ec2f3f722b573ce874688d1c661f5457c73","status:5"]',
      testRectangle(732.5, 433, 760.5, 451),
    ],
    [
      '["durable","status","oracle_b410ab12e3e9a7a40e7ecd07b7cbf965e88c066ea0f4950bf7576601abd05fdd","status:6"]',
      testRectangle(314.5, 357, 372.5, 375),
    ],
    [
      '["durable","status","oracle_e6c6570ae63ba9770fd6c25cb9ac4d24dafd54400f3522dcd56d18a7b716de76","status:1"]',
      testRectangle(504.5, 357, 562.5, 375),
    ],
  ];
  const protectedRects = protectedEntries.map(([layoutKey, bounds]) => ({
    layoutKey,
    bounds,
  }));
  const request = {
    layoutKey:
      '["event","debugger-episode:41eaad23adc24f6f2a284baf66aa2702b6a9bb006bd5db43bc65c51bcaf721c2:transition:1:event:0010","route"]',
    kind: /** @type {const} */ ("route"),
    priority: 1,
    stableOrder: 42,
    source: { x: 362.5, y: 366 },
    target: { x: 476.5, y: 366 },
    sourceRadius: 0,
    targetRadius: 0,
    sourceEndpointGap: 3,
    targetEndpointGap: 3,
    pathPadding: 1.5,
    markerPadding: 14,
    markerProgress: 0.76,
    allowProtectedKeys: [body1],
  };
  const forward = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects,
    requests: [request],
  });
  const reversed = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects: [...protectedRects].reverse(),
    requests: [request],
  });

  assert.deepEqual(forward, reversed);
  const route = forward.routePlacements[0];
  assert.equal(route.kind, "polyline");
  if (route.kind !== "polyline") assert.fail("expected Charge polyline fallback");
  assert.notDeepEqual(route.start, request.source);
  assert.deepEqual(route.end, request.target);
  const blockers = protectedRects.filter(({ layoutKey }) => layoutKey !== body1);
  assert.equal(
    blockers.some(
      ({ bounds }) =>
        route.start.x >= bounds.left &&
        route.start.x <= bounds.right &&
        route.start.y >= bounds.top &&
        route.start.y <= bounds.bottom,
    ),
    false,
  );
  for (let index = 1; index < route.points.length; index += 1) {
    for (const blocker of blockers) {
      assert.equal(
        segmentTouchesRectangle(
          route.points[index - 1],
          route.points[index],
          blocker.bounds,
        ),
        false,
        `Charge segment ${index - 1} touched ${blocker.layoutKey}`,
      );
    }
  }
});

test("cross-phase Charge displacement clips only its successor target and keeps the full marker clear", () => {
  /**
   * @param {{
   *   name: string,
   *   viewport: ReturnType<typeof testRectangle>,
   *   protectedEntries: Array<[string, ReturnType<typeof testRectangle>]>,
   *   targetProtectedKey: string,
   *   source: {x: number, y: number},
   *   target: {x: number, y: number},
   *   targetRadius: number,
   *   stableOrder: number,
   * }} scenario
   */
  const proveScenario = (scenario) => {
    const protectedRects = scenario.protectedEntries.map(([layoutKey, bounds]) => ({
      layoutKey,
      bounds,
    }));
    const request = {
      layoutKey: `${scenario.name}:route`,
      kind: /** @type {const} */ ("route"),
      priority: 1,
      stableOrder: scenario.stableOrder,
      source: { ...scenario.source },
      target: { ...scenario.target },
      sourceRadius: 0,
      targetRadius: scenario.targetRadius,
      sourceEndpointGap: 0,
      targetEndpointGap: 3,
      pathPadding: 1.5,
      markerPadding: 14,
      markerProgress: 0.76,
      allowProtectedKeys: [scenario.targetProtectedKey],
      targetProtectedKey: scenario.targetProtectedKey,
    };
    const forward = layoutCrossPhaseOccupancy({
      viewport: scenario.viewport,
      protectedRects,
      requests: [request],
    });
    const reversed = layoutCrossPhaseOccupancy({
      viewport: scenario.viewport,
      protectedRects: [...protectedRects].reverse(),
      requests: [request],
    });

    assert.deepEqual(
      forward,
      reversed,
      `${scenario.name} changed under ledger reversal`,
    );
    assert.deepEqual(request.source, scenario.source);
    assert.deepEqual(request.target, scenario.target);
    const route = forward.routePlacements[0];
    assert.equal(route.kind, "polyline");
    if (route.kind !== "polyline") assert.fail("expected marker-safe Charge polyline");
    assert.deepEqual(route.start, scenario.source);
    assert.notDeepEqual(route.end, scenario.target);
    assert.equal(route.markerVariant, "full");
    assert.equal(route.markerPadding, 14);
    assert.ok(route.points.length >= 3);
    const targetBounds = protectedRects.find(
      ({ layoutKey }) => layoutKey === scenario.targetProtectedKey,
    )?.bounds;
    assert.ok(targetBounds);
    assert.equal(
      route.end.x >= targetBounds.left - 1e-9 &&
        route.end.x <= targetBounds.right + 1e-9 &&
        route.end.y >= targetBounds.top - 1e-9 &&
        route.end.y <= targetBounds.bottom + 1e-9 &&
        (Math.abs(route.end.x - targetBounds.left) <= 1e-9 ||
          Math.abs(route.end.x - targetBounds.right) <= 1e-9 ||
          Math.abs(route.end.y - targetBounds.top) <= 1e-9 ||
          Math.abs(route.end.y - targetBounds.bottom) <= 1e-9),
      true,
      `${scenario.name} did not terminate on its exact successor body boundary`,
    );
    for (const point of route.points) {
      assert.equal(Number.isFinite(point.x) && Number.isFinite(point.y), true);
      assert.equal(
        viewportOverflow(
          testRectangle(point.x, point.y, point.x, point.y),
          scenario.viewport,
        ),
        0,
      );
    }
    for (let index = 1; index < route.points.length; index += 1) {
      for (const blocker of protectedRects.filter(
        ({ layoutKey }) => layoutKey !== scenario.targetProtectedKey,
      )) {
        const padded = testRectangle(
          blocker.bounds.left - request.pathPadding,
          blocker.bounds.top - request.pathPadding,
          blocker.bounds.right + request.pathPadding,
          blocker.bounds.bottom + request.pathPadding,
        );
        assert.equal(
          segmentTouchesRectangle(route.points[index - 1], route.points[index], padded),
          false,
          `${scenario.name} segment ${index - 1} touched ${blocker.layoutKey}`,
        );
      }
    }
    const marker = routeMarkerPose(route);
    assert.equal(Number.isFinite(marker.x) && Number.isFinite(marker.y), true);
    assert.ok(marker.x >= scenario.viewport.left + 14);
    assert.ok(marker.x <= scenario.viewport.right - 14);
    assert.ok(marker.y >= scenario.viewport.top + 14);
    assert.ok(marker.y <= scenario.viewport.bottom - 14);
    for (const blocker of protectedRects) {
      assert.equal(
        marker.x >= blocker.bounds.left - 14 &&
          marker.x <= blocker.bounds.right + 14 &&
          marker.y >= blocker.bounds.top - 14 &&
          marker.y <= blocker.bounds.bottom + 14,
        false,
        `${scenario.name} full marker touched ${blocker.layoutKey}`,
      );
    }
  };

  const owners = [
    "oracle_32befe272aa8bce291b4694efa2e34c4f181484581bc0fcf2c04dd5b9d3b97eb",
    "oracle_d6570d7a87c1df84c2337066aad22d1a147f347b6e99ee6f60fe1c9b90f0ca4a",
    "oracle_8081e6334b800b4ebeaf765914a22d8defae742ffdeca7a8aec661fe9218dd14",
    "oracle_af18898cbf7bf861549d0c9578959e0bf3737951f3441b5972fa404518ceba6a",
    "oracle_bbb9d1fc83d3f667ea64f42d70f95c67df568c8cac023a3630aed9d26c31e265",
    "oracle_2f9df55071538beac28dadf7149037a785890108798582a2fc4113bb3317e208",
    "oracle_3238f8a9ba7f8da4821222b9b6108dd333ce589c945a8d62ca7a1d1d47f59fa3",
    "oracle_b1162c68ec03f4efd304d1fa8835630a7f746e41b829698c7a0d2799b2ce0868",
    "oracle_fd7c8a14326ffd977b19259c3794e531bd2ce8eb18aadfe79966d7f70dc956bc",
    "oracle_d581a4bab768a5f978b1ef7b3156464e1b75d69435fc4792e2dd3aa4dcdfb1a7",
  ];
  /** @param {string} kind @param {number} slot @param {string} [placement] */
  const protectedKey = (kind, slot, placement = `${kind}:${slot}`) =>
    JSON.stringify(["durable", kind, owners[slot], placement]);
  const targetProtectedKey = protectedKey("body", 1);
  proveScenario({
    name: "debugger-episode:2eb7a144f1c0acc7c6bdb5207bd69e24eec6799bc2c42243fde5d9781d351034:transition:1:event:0010",
    viewport: testRectangle(134.5, 24, 818.5, 556),
    targetProtectedKey,
    source: { x: 362.5, y: 366 },
    target: { x: 476.5, y: 366 },
    targetRadius: 19,
    stableOrder: 42,
    protectedEntries: [
      [protectedKey("body", 0), testRectangle(225.5, 419, 271.5, 465)],
      [targetProtectedKey, testRectangle(453.5, 343, 499.5, 389)],
      [
        protectedKey("body", 2),
        testRectangle(362.2999963760376, 191, 408.2999963760376, 237),
      ],
      [
        protectedKey("body", 3),
        testRectangle(388.9000072479248, 77, 434.9000072479248, 123),
      ],
      [protectedKey("body", 4), testRectangle(263.5, 96, 309.5, 142)],
      [protectedKey("body", 5), testRectangle(681.5, 419, 727.5, 465)],
      [protectedKey("body", 6), testRectangle(365.5, 331, 435.5, 401)],
      [
        protectedKey("body", 7),
        testRectangle(468.6999855041504, 191, 514.6999855041504, 237),
      ],
      [
        protectedKey("body", 8),
        testRectangle(442.0999927520752, 77, 488.0999927520752, 123),
      ],
      [protectedKey("body", 9), testRectangle(567.5, 96, 613.5, 142)],
      [protectedKey("cooldown", 0), testRectangle(229.5, 396, 267.5, 414)],
      [protectedKey("cooldown", 1), testRectangle(457.5, 320, 495.5, 338)],
      [protectedKey("cooldown", 5), testRectangle(685.5, 396, 723.5, 414)],
      [protectedKey("cooldown", 6), testRectangle(322.5, 357, 360.5, 375)],
      [protectedKey("modifier", 0), testRectangle(178.5, 434, 220.5, 450)],
      [protectedKey("modifier", 1), testRectangle(455.5, 394, 497.5, 410)],
      [protectedKey("modifier", 5), testRectangle(634.5, 434, 676.5, 450)],
      [protectedKey("status", 0), testRectangle(276.5, 433, 304.5, 451)],
      [protectedKey("status", 1), testRectangle(504.5, 357, 562.5, 375)],
      [protectedKey("status", 5), testRectangle(732.5, 433, 760.5, 451)],
      [protectedKey("status", 6), testRectangle(371.5, 308, 429.5, 326)],
      [protectedKey("legality", 6, owners[6]), testRectangle(369, 406, 432, 424)],
    ],
  });

  const convergenceOwners = {
    0: "oracle_d4b6cc3fdccc2cff09ef0ea7e8046f883dbf8a9aba5d94bb1d1ba6570a7e2de7",
    1: "oracle_13784267d8c210403cdf92d11eb65a2bd896bb9aac95e38dc451c592be7d3555",
    5: "oracle_234783f516e1e270668c7b9f19ccd125ca624fc3a6c030b86e7c19a607327347",
  };
  /** @param {string} kind @param {0 | 1 | 5} slot */
  const convergenceKey = (kind, slot) =>
    JSON.stringify(["durable", kind, convergenceOwners[slot], `${kind}:${slot}`]);
  const convergenceTargetKey = convergenceKey("body", 5);
  proveScenario({
    name: "debugger-episode:6abfbd6e6630da8d6301efb939233dfdec8467fad68b1a7b417ab973e095220c:transition:0:event:0016",
    viewport: testRectangle(166.16666666666663, 24, 786.8333333333334, 556),
    targetProtectedKey: convergenceTargetKey,
    source: { x: 520.8333333333333, y: 290 },
    target: { x: 340.3291385968526, y: 362.20167366663617 },
    targetRadius: 22.166666666666668,
    stableOrder: 66,
    protectedEntries: [
      [
        convergenceKey("body", 0),
        testRectangle(441.5041947364807, 274, 517.837528069814, 350.33333333333337),
      ],
      [
        convergenceKey("body", 1),
        testRectangle(
          453.5041947364807,
          241.66666666666669,
          505.837528069814,
          294.00000000000006,
        ),
      ],
      [
        convergenceTargetKey,
        testRectangle(
          314.1624719301859,
          336.0350069999695,
          366.4958052635193,
          388.36834033330285,
        ),
      ],
      [
        convergenceKey("cooldown", 0),
        testRectangle(
          398.5041809082031,
          303.1666564941406,
          436.5041809082031,
          321.1666564941406,
        ),
      ],
      [
        convergenceKey("cooldown", 1),
        testRectangle(
          460.6708679199219,
          218.6666717529297,
          498.6708679199219,
          236.6666717529297,
        ),
      ],
      [
        convergenceKey("cooldown", 5),
        testRectangle(
          321.3291320800781,
          313.0350036621094,
          359.3291320800781,
          331.0350036621094,
        ),
      ],
      [
        convergenceKey("modifier", 1),
        testRectangle(
          510.8375244140625,
          251.8333282470703,
          552.8375244140625,
          267.8333282470703,
        ),
      ],
      [
        convergenceKey("modifier", 5),
        testRectangle(
          267.1624755859375,
          354.20166015625,
          309.1624755859375,
          370.20166015625,
        ),
      ],
      [
        convergenceKey("status", 0),
        testRectangle(
          522.8375244140625,
          303.1666564941406,
          580.8375244140625,
          321.1666564941406,
        ),
      ],
      [
        convergenceKey("status", 5),
        testRectangle(
          371.4958190917969,
          353.20166015625,
          429.4958190917969,
          371.20166015625,
        ),
      ],
    ],
  });
});

test("cross-phase Charge impact keeps a straight historical-anchor connector through the durable ledger", () => {
  const viewport = testRectangle(134.5, 24, 818.5, 556);
  const owners = [
    "oracle_73842f605e7ae4d27877db75aebd26132dda09018d1f86fc5282e0b8ef938a3e",
    "oracle_ba7c1636712bbbd0254b4287ab099478d5b60b4b6b680b562158673535a3ea16",
    "oracle_31d992c85334c969a9215e134639b1034b908656ef842bc546142884e641b9cf",
    "oracle_07c03b166fd9c147409fb2b13ce7482b953b0a5ac18265ac528e32e0b475073e",
    "oracle_573437b3f34cdd7e6194923e7ba115ee574288f51a6423d683093ba38d3bb84d",
    "oracle_1c667fc8e428a18f9dcd1ed6cb77a372c1c8759774b52d41efe0f6f8dab4e4e5",
    "oracle_dcc916fa15ed5b9ae03e1a8d8387c358eb684b5c4d3007911ba8b95ae31da30c",
    "oracle_e20f4c24c10f6b7932cfc3b1ada5f6d6ce73d3b97ba99de3fe57db23690fe17b",
    "oracle_d13ec1116f23c27469dc1897a5c8577af1010ad4f458ad95b9eab30c112b5ce2",
    "oracle_8dfc92d0ca48b638bacb41a1e06ab808ac0619e9d92c80d850ba7df99aa3afa0",
  ];
  /** @param {string} kind @param {number} slot @param {string} [placement] */
  const protectedKey = (kind, slot, placement = `${kind}:${slot}`) =>
    JSON.stringify(["durable", kind, owners[slot], placement]);
  const protectedRects = [
    [protectedKey("body", 0), testRectangle(213.5, 407, 283.5, 477)],
    [protectedKey("body", 1), testRectangle(453.5, 343, 499.5, 389)],
    [
      protectedKey("body", 2),
      testRectangle(362.2999963760376, 191, 408.2999963760376, 237),
    ],
    [
      protectedKey("body", 3),
      testRectangle(388.9000072479248, 77, 434.9000072479248, 123),
    ],
    [protectedKey("body", 4), testRectangle(263.5, 96, 309.5, 142)],
    [protectedKey("body", 5), testRectangle(681.5, 419, 727.5, 465)],
    [protectedKey("body", 6), testRectangle(377.5, 343, 423.5, 389)],
    [
      protectedKey("body", 7),
      testRectangle(468.6999855041504, 191, 514.6999855041504, 237),
    ],
    [
      protectedKey("body", 8),
      testRectangle(442.0999927520752, 77, 488.0999927520752, 123),
    ],
    [protectedKey("body", 9), testRectangle(567.5, 96, 613.5, 142)],
    [protectedKey("cooldown", 0), testRectangle(288.5, 433, 326.5, 451)],
    [protectedKey("cooldown", 1), testRectangle(457.5, 320, 495.5, 338)],
    [protectedKey("cooldown", 5), testRectangle(685.5, 396, 723.5, 414)],
    [protectedKey("cooldown", 6), testRectangle(381.5, 320, 419.5, 338)],
    [protectedKey("modifier", 1), testRectangle(455.5, 394, 497.5, 410)],
    [protectedKey("modifier", 5), testRectangle(634.5, 434, 676.5, 450)],
    [protectedKey("modifier", 6), testRectangle(379.5, 394, 421.5, 410)],
    [protectedKey("status", 0), testRectangle(234.5, 384, 262.5, 402)],
    [protectedKey("status", 1), testRectangle(504.5, 357, 562.5, 375)],
    [protectedKey("status", 5), testRectangle(732.5, 433, 760.5, 451)],
    [protectedKey("status", 6), testRectangle(314.5, 357, 372.5, 375)],
    [protectedKey("legality", 0, owners[0]), testRectangle(145.5, 433, 208.5, 451)],
  ].map(([layoutKey, bounds]) => ({
    layoutKey: /** @type {string} */ (layoutKey),
    bounds: /** @type {ReturnType<typeof testRectangle>} */ (bounds),
  }));
  const anchor = Object.freeze({ x: 514.5, y: 366 });
  const request = {
    layoutKey: JSON.stringify([
      "event",
      "debugger-episode:3cb93fe519faa7ff1199438ca2e8eb873746e5d405267464e17ef032fe1ad487:transition:1:event:0000",
      "impact",
    ]),
    kind: /** @type {const} */ ("recipient_cue"),
    stableOrder: 0,
    anchor,
    anchorRadius: 60,
    recipientKey: owners[6],
    allowProtectedKeys: [],
    width: 32.16,
    height: 27.52,
  };
  const forward = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects,
    requests: [request],
  });
  const reversed = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects: [...protectedRects].reverse(),
    requests: [request],
  });

  assert.deepEqual(forward, reversed);
  assert.deepEqual(request.anchor, anchor);
  const cue = forward.cuePlacements[0];
  assert.equal(cue.leader.kind, "line");
  assert.deepEqual(cue.leader.start, anchor);
  assert.deepEqual(cue.leader.end, cue.center);
  assert.deepEqual(cue.leader.points, [anchor, cue.center]);
  assert.equal(cue.collisionFree, true);
  assert.equal(viewportOverflow(cue.bounds, viewport), 0);
  const occludingStatus = protectedRects.find(
    ({ layoutKey }) => layoutKey === protectedKey("status", 1),
  );
  assert.ok(occludingStatus);
  assert.equal(
    anchor.x >= occludingStatus.bounds.left &&
      anchor.x <= occludingStatus.bounds.right &&
      anchor.y >= occludingStatus.bounds.top &&
      anchor.y <= occludingStatus.bounds.bottom,
    true,
  );
  for (const point of cue.leader.points) {
    assert.equal(Number.isFinite(point.x) && Number.isFinite(point.y), true);
    assert.equal(
      viewportOverflow(testRectangle(point.x, point.y, point.x, point.y), viewport),
      0,
    );
  }
  assert.equal(
    segmentTouchesRectangle(cue.leader.start, cue.leader.end, occludingStatus.bounds),
    true,
  );
  const allowed = layoutCrossPhaseOccupancy({
    viewport,
    protectedRects,
    requests: [
      {
        ...request,
        allowProtectedKeys: [protectedKey("body", 1)],
      },
    ],
  });
  assert.deepEqual(allowed.cuePlacements[0].leader.points, [anchor, cue.center]);
});

test("cross-phase free-endpoint escape still fails when no exterior port exists", () => {
  assert.throws(
    () =>
      layoutCrossPhaseOccupancy({
        viewport: testRectangle(0, 0, 100, 100),
        protectedRects: [
          {
            layoutKey: "durable:sealed",
            bounds: testRectangle(0, 0, 100, 100),
          },
          {
            layoutKey: "body:allowed-target",
            bounds: testRectangle(75, 40, 95, 60),
          },
        ],
        requests: [
          {
            layoutKey: "route:occluded-and-sealed",
            kind: "route",
            stableOrder: 0,
            source: { x: 20, y: 50 },
            target: { x: 85, y: 50 },
            allowProtectedKeys: ["body:allowed-target"],
          },
        ],
      }),
    /no bounded protected-region lane/,
  );
});

test("cross-phase polyline fallback fails loudly when a durable wall seals the viewport", () => {
  const sourceProtectedKey = "body:source";
  const targetProtectedKey = "body:target";
  assert.throws(
    () =>
      layoutCrossPhaseOccupancy({
        viewport: testRectangle(0, 0, 100, 100),
        protectedRects: [
          {
            layoutKey: sourceProtectedKey,
            bounds: testRectangle(5, 40, 25, 60),
          },
          {
            layoutKey: "durable:wall",
            bounds: testRectangle(45, 0, 55, 100),
          },
          {
            layoutKey: targetProtectedKey,
            bounds: testRectangle(75, 40, 95, 60),
          },
        ],
        requests: [
          {
            layoutKey: "route:sealed",
            kind: "route",
            stableOrder: 0,
            source: { x: 15, y: 50 },
            target: { x: 85, y: 50 },
            allowProtectedKeys: [sourceProtectedKey, targetProtectedKey],
            sourceProtectedKey,
            targetProtectedKey,
          },
        ],
      }),
    /no bounded protected-region lane/,
  );
});
