import assert from "node:assert/strict";
import test from "node:test";

import {
  authorizationContextKey,
  buildChoreographyPlan,
  CHOREOGRAPHY_PHASES,
  eventFingerprint,
  isSubmissionCommand,
  transitionEpochKey,
} from "../src/choreography-plan.js";
import { routeMarkerPose } from "../src/routes.js";

/**
 * @typedef {{
 *   worldToScreen: (
 *     point: {x: number, y: number} | readonly [number, number],
 *   ) => {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 *   viewportBounds?: {
 *     left: number,
 *     top: number,
 *     right: number,
 *     bottom: number,
 *     width: number,
 *     height: number,
 *   },
 *   protectedRects?: ReadonlyArray<{
 *     left: number,
 *     top: number,
 *     right: number,
 *     bottom: number,
 *     width: number,
 *     height: number,
 *   }>,
 * }} ProjectionSurface
 */

/** @type {ProjectionSurface} */
const surface = {
  worldToScreen: (point) => {
    const x = "x" in point ? point.x : Number(point[0]);
    const y = "y" in point ? point.y : Number(point[1]);
    return { x: x * 10, y: 120 - y * 10 };
  },
  worldLengthToScreen: (length) => length * 10,
  viewportBounds: {
    left: 0,
    top: 0,
    right: 200,
    bottom: 120,
    width: 200,
    height: 120,
  },
  protectedRects: [],
};

/**
 * @param {string} eventId
 * @param {string} tokenId
 * @param {number} source
 * @param {number | null} target
 * @param {{
 *   disclosure?: string,
 *   sourceAnchor?: readonly [number, number] | null,
 *   targetAnchor?: readonly [number, number],
 * }} [options]
 */
function activation(
  eventId,
  tokenId,
  source,
  target,
  { disclosure = "public", sourceAnchor, targetAnchor } = {},
) {
  return {
    event_type: "accepted_activation",
    event_id: eventId,
    transition_id: 4,
    token_id: tokenId,
    source_global_slot: source,
    target_global_slot: disclosure === "public" && target !== null ? target : null,
    source_anchor: sourceAnchor === undefined ? [source + 1, 3] : sourceAnchor,
    target_anchor:
      disclosure === "public" && target !== null
        ? (targetAnchor ?? [target + 1, 3])
        : null,
    target_disclosure: disclosure,
    lane: tokenId === "basic_damage" || tokenId === "basic_heal" ? 0 : 1,
    source_class_id: 1,
  };
}

/**
 * @param {any[]} events
 * @param {{
 *   audience?: string,
 *   runGeneration?: number,
 *   revision?: number,
 *   controlledSlot?: number,
 * }} [options]
 */
function debuggerFrame(events, options = {}) {
  const audience = options.audience ?? "researcher";
  return {
    schema_version: 1,
    session_id: "session-a",
    run_generation: options.runGeneration ?? 2,
    revision: options.revision ?? 7,
    transition_id: 4,
    scene: {
      schema_version: 1,
      audience,
      selection: {
        controlled_global_slot: options.controlledSlot ?? 0,
        selected_global_slot: null,
      },
      agents: Array.from({ length: 10 }, (_, globalSlot) => ({
        global_slot: globalSlot,
        radius: 0.5,
      })),
    },
    event_batch: {
      schema_version: 1,
      transition_id: 4,
      simulator_step: 4,
      events,
    },
  };
}

/**
 * @param {ReadonlyArray<Record<string, any>>} events
 * @param {string} eventId
 * @returns {Record<string, any>}
 */
function eventById(events, eventId) {
  const event = events.find((candidate) => candidate.eventId === eventId);
  assert.ok(event, `missing planned event ${eventId}`);
  return event;
}

/**
 * @param {Record<string, number>} left
 * @param {Record<string, number>} right
 */
function rectangleIntersectionArea(left, right) {
  return (
    Math.max(0, Math.min(left.right, right.right) - Math.max(left.left, right.left)) *
    Math.max(0, Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top))
  );
}

test("transition identity excludes revision while authorization tracks POV actor", () => {
  const researcher = debuggerFrame([], { revision: 1 });
  const revised = debuggerFrame([], { revision: 99 });
  assert.equal(transitionEpochKey(researcher), transitionEpochKey(revised));
  assert.match(eventFingerprint(researcher) ?? "", /^0:[0-9a-f]{8}$/);
  assert.equal(eventFingerprint(researcher), eventFingerprint(revised));
  assert.equal(authorizationContextKey(researcher), authorizationContextKey(revised));

  const povZero = debuggerFrame([], { audience: "agent_pov", controlledSlot: 0 });
  const povOne = debuggerFrame([], { audience: "agent_pov", controlledSlot: 1 });
  assert.notEqual(
    authorizationContextKey(researcher),
    authorizationContextKey(povZero),
  );
  assert.notEqual(authorizationContextKey(povZero), authorizationContextKey(povOne));

  const publicEvent = debuggerFrame([activation("stable-id", "basic_damage", 0, 5)]);
  const changedContent = debuggerFrame([
    activation("stable-id", "basic_damage", 0, 5, {
      sourceAnchor: [3, 3],
    }),
  ]);
  assert.notEqual(eventFingerprint(publicEvent), eventFingerprint(changedContent));
});

test("submission classification is presentation gating, not legality", () => {
  for (const key of [" ", "Space", "Enter", "n", "N"]) {
    assert.equal(isSubmissionCommand({ command_type: "keyboard", key }), true, key);
  }
  assert.equal(isSubmissionCommand({ command_type: "keyboard", key: "g" }), false);
  assert.equal(isSubmissionCommand({ command_type: "reset" }), false);
});

test("accepted routes preserve multiplicity, direction, and simultaneous timing", () => {
  const events = [
    activation("a-0", "basic_damage", 0, 5),
    activation("a-1", "basic_damage", 5, 0),
    activation("a-2", "basic_damage", 0, 5),
    activation("a-3", "basic_damage", 2, 7, {
      sourceAnchor: [5, 5],
      targetAnchor: [5, 5],
    }),
    activation("a-4", "mage_burst", 1, null, {
      disclosure: "target_none",
    }),
    activation("a-5", "basic_damage", 0, 9, {
      disclosure: "redacted",
    }),
  ];
  const plan = buildChoreographyPlan(debuggerFrame(events), surface);
  assert.ok(plan);
  const activations = plan.events.filter(({ kind }) => kind === "activation");
  assert.equal(activations.length, events.length);
  assert.equal(activations.filter(({ route }) => route).length, 4);
  assert.equal(new Set(activations.map(({ eventId }) => eventId)).size, events.length);
  assert.deepEqual(
    new Set(activations.map(({ phaseStart }) => phaseStart)),
    new Set([CHOREOGRAPHY_PHASES.activationStart]),
  );
  assert.equal(
    activations.every((event) => !Object.hasOwn(event, "netDelta")),
    true,
  );
  assert.notEqual(
    eventById(activations, "a-0").route.offset,
    eventById(activations, "a-2").route.offset,
  );
  assert.notEqual(
    eventById(activations, "a-0").route.path,
    eventById(activations, "a-1").route.path,
  );
  assert.equal(eventById(activations, "a-3").route.kind, "local_arc");
  assert.equal(eventById(activations, "a-0").sourceClass.cssKey, "mage");
  assert.equal(eventById(activations, "a-4").target, null);
  assert.equal(eventById(activations, "a-5").route, null);
  assert.equal(eventById(activations, "a-5").targetSlot, null);
});

test("routed impacts reserve token-sized exterior recipient clearance", () => {
  /** @type {{
   *   sourceAnchor: readonly [number, number],
   *   targetAnchor: readonly [number, number],
   * }} */
  const common = {
    sourceAnchor: [1, 6],
    targetAnchor: [11, 6],
  };
  const plan = buildChoreographyPlan(
    debuggerFrame([
      activation("basic-port", "basic_damage", 0, 5, common),
      activation("charge-port", "warrior_charge", 1, 6, common),
      activation("trap-port", "hunter_trap", 2, 7, common),
    ]),
    surface,
  );
  assert.ok(plan);

  const target = surface.worldToScreen(common.targetAnchor);
  const basic = eventById(plan.events, "basic-port").route;
  const charge = eventById(plan.events, "charge-port").route;
  const trap = eventById(plan.events, "trap-port").route;
  assert.ok(basic && charge && trap);
  /** @param {Record<string, any>} route */
  const targetDistance = (route) =>
    Math.hypot(route.end.x - target.x, route.end.y - target.y);

  assert.equal(targetDistance(basic), 8);
  assert.equal(targetDistance(charge), 23);
  assert.equal(targetDistance(trap), 31);
  for (const route of [basic, charge, trap]) {
    const marker = routeMarkerPose(route);
    const radians = (marker.degrees * Math.PI) / 180;
    assert.ok(Math.cos(radians) > 0);
  }
});

test("four-source Basic convergence receives distinct exterior impact ports", () => {
  const events = [0, 1, 2, 3].map((source, index) =>
    activation(`focus-${source}`, "basic_damage", source, 5, {
      sourceAnchor: [1 + index * 2, 2 + index],
      targetAnchor: [12, 6],
    }),
  );
  const plan = buildChoreographyPlan(debuggerFrame(events), surface);
  assert.ok(plan);
  const target = surface.worldToScreen([12, 6]);
  const routes = events.map((event) => eventById(plan.events, event.event_id));
  for (const event of routes) {
    assert.equal(event.routeMultiplicity, 4);
    assert.ok(event.route);
    assert.ok(
      Math.abs(
        Math.hypot(event.route.end.x - target.x, event.route.end.y - target.y) - 17,
      ) < 1e-9,
    );
  }
  assert.equal(new Set(routes.map(({ route }) => route.path)).size, 4);
});

test("an isolated NET cue never falls below its lower-field recipient", () => {
  /** @type {ProjectionSurface} */
  const lowerFieldSurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 240,
      bottom: 200,
      width: 240,
      height: 200,
    },
    protectedRects: [
      { left: 60, top: 80, right: 180, bottom: 130, width: 120, height: 50 },
    ],
  };
  const plan = buildChoreographyPlan(
    debuggerFrame([
      {
        event_type: "net_health",
        event_id: "lower-field-net",
        transition_id: 4,
        recipient_global_slot: 5,
        recipient_anchor: [120, 150],
        health_before: 50,
        health_after: 42,
        net_delta: -8,
        outcome: "damage",
      },
    ]),
    lowerFieldSurface,
  );
  assert.ok(plan);
  const outcome = eventById(plan.events, "lower-field-net");
  assert.equal(outcome.cueCollisionFree, true);
  assert.ok(outcome.cue);
  assert.ok(outcome.cueBounds);
  assert.ok(outcome.cue.y <= outcome.recipient.y);
  assert.ok(outcome.cueBounds.bottom <= 200);
});

test("choreography passes viewport bounds into close-route containment", () => {
  const viewportBounds = {
    left: 0,
    top: 0,
    right: 200,
    bottom: 100,
    width: 200,
    height: 100,
  };
  /** @type {ProjectionSurface} */
  const boundedSurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length * 36,
    viewportBounds,
    protectedRects: [],
  };
  const plan = buildChoreographyPlan(
    debuggerFrame([
      activation("edge-route", "basic_damage", 0, 5, {
        sourceAnchor: [70, 82],
        targetAnchor: [71, 82.5],
      }),
    ]),
    boundedSurface,
  );
  assert.ok(plan);
  const route = eventById(plan.events, "edge-route").route;
  assert.equal(route.kind, "curve");
  assert.ok(route.offset < 0);
  const marker = routeMarkerPose(route);
  assert.ok(marker.x >= viewportBounds.left + 14);
  assert.ok(marker.x <= viewportBounds.right - 14);
  assert.ok(marker.y >= viewportBounds.top + 14);
  assert.ok(marker.y <= viewportBounds.bottom - 14);
});

test("NET outcomes reserve space before collision-free Charge ownership pills", () => {
  const protectedRects = [
    { left: 20, top: 60, right: 60, bottom: 100, width: 40, height: 40 },
    { left: 20, top: 100, right: 60, bottom: 140, width: 40, height: 40 },
    { left: 220, top: 60, right: 260, bottom: 100, width: 40, height: 40 },
    { left: 80, top: 60, right: 150, bottom: 86, width: 70, height: 26 },
  ];
  /** @type {ProjectionSurface} */
  const chargeSurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 320,
      bottom: 200,
      width: 320,
      height: 200,
    },
    protectedRects,
  };
  const events = [
    activation("charge-a", "warrior_charge", 0, 5, {
      sourceAnchor: [40, 80],
      targetAnchor: [240, 80],
    }),
    activation("charge-b", "warrior_charge", 1, 5, {
      sourceAnchor: [40, 120],
      targetAnchor: [240, 80],
    }),
    activation("charge-c", "warrior_charge", 5, 0, {
      sourceAnchor: [240, 80],
      targetAnchor: [40, 80],
    }),
    {
      event_type: "net_health",
      event_id: "charge-net",
      transition_id: 4,
      recipient_global_slot: 5,
      recipient_anchor: [240, 80],
      health_before: 50,
      health_after: 32,
      net_delta: -18,
      outcome: "damage",
    },
  ];
  const plan = buildChoreographyPlan(debuggerFrame(events), chargeSurface);
  assert.ok(plan);
  const ownershipEvents = ["charge-a", "charge-b", "charge-c"].map((eventId) =>
    eventById(plan.events, eventId),
  );
  for (const event of ownershipEvents) {
    assert.equal(event.ownershipCueCollisionFree, true);
    assert.equal(event.ownershipSpatialDisposition, "rendered");
    assert.ok(event.ownershipAnchor);
    assert.ok(event.ownershipCue);
    assert.ok(event.ownershipBounds);
    assert.ok(event.ownershipBounds.left >= 0);
    assert.ok(event.ownershipBounds.top >= 0);
    assert.ok(event.ownershipBounds.right <= 320);
    assert.ok(event.ownershipBounds.bottom <= 200);
    for (const protectedRect of protectedRects) {
      assert.equal(rectangleIntersectionArea(event.ownershipBounds, protectedRect), 0);
    }
  }
  for (const [index, event] of ownershipEvents.entries()) {
    for (const other of ownershipEvents.slice(index + 1)) {
      assert.equal(
        rectangleIntersectionArea(event.ownershipBounds, other.ownershipBounds),
        0,
      );
    }
  }
  const outcome = eventById(plan.events, "charge-net");
  assert.equal(outcome.cueCollisionFree, true);
  assert.ok(outcome.cueBounds);
  for (const event of ownershipEvents) {
    assert.equal(
      rectangleIntersectionArea(outcome.cueBounds, event.ownershipBounds),
      0,
    );
  }
  const repeated = buildChoreographyPlan(debuggerFrame(events), chargeSurface);
  assert.ok(repeated);
  assert.deepEqual(
    ownershipEvents.map(
      ({ eventId, ownershipAnchor, ownershipBounds, ownershipCue }) => ({
        eventId,
        ownershipAnchor,
        ownershipBounds,
        ownershipCue,
      }),
    ),
    ["charge-a", "charge-b", "charge-c"].map((eventId) => {
      const event = eventById(repeated.events, eventId);
      return {
        eventId,
        ownershipAnchor: event.ownershipAnchor,
        ownershipBounds: event.ownershipBounds,
        ownershipCue: event.ownershipCue,
      };
    }),
  );
});

test("authoritative NET truth suppresses redundant Charge ownership before lifecycle decoration", () => {
  /** @type {ProjectionSurface} */
  const prioritySurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 180,
      bottom: 60,
      width: 180,
      height: 60,
    },
    protectedRects: [],
  };
  const events = [
    activation("charge-priority", "warrior_charge", 0, 5, {
      sourceAnchor: [10, 30],
      targetAnchor: [170, 30],
    }),
    {
      event_type: "status_lifecycle",
      event_id: "lifecycle-after-charge",
      transition_id: 4,
      recipient_global_slot: 5,
      recipient_anchor: [90, 30],
      token_id: "stun_warrior_charge",
      change: "applied",
      duration_before: 0,
      duration_after: 2,
      source_class_id: 2,
      application_event_ids: ["charge-priority"],
    },
    {
      event_type: "net_health",
      event_id: "net-before-charge",
      transition_id: 4,
      recipient_global_slot: 5,
      recipient_anchor: [90, 30],
      health_before: 50,
      health_after: 40,
      net_delta: -10,
      outcome: "damage",
    },
  ];

  const plan = buildChoreographyPlan(debuggerFrame(events), prioritySurface);
  assert.ok(plan);
  const net = eventById(plan.events, "net-before-charge");
  const charge = eventById(plan.events, "charge-priority");
  const lifecycle = eventById(plan.events, "lifecycle-after-charge");

  assert.equal(net.spatialDisposition, "rendered");
  assert.equal(net.cueCollisionFree, true);
  assert.ok(net.cueBounds);
  assert.equal(charge.ownershipSpatialDisposition, "suppressed_collision");
  assert.equal(charge.ownershipCueCollisionFree, false);
  assert.equal(charge.ownershipCue, null);
  assert.equal(charge.ownershipCueSuppressionReason, "no_collision_free_position");
  assert.equal(lifecycle.spatialDisposition, "suppressed_collision");
  assert.equal(lifecycle.cueCollisionFree, false);
  assert.equal(lifecycle.cue, null);
  assert.equal(lifecycle.cueSuppressionReason, "no_collision_free_position");
  assert.deepEqual(
    buildChoreographyPlan(debuggerFrame(events), prioritySurface)?.events,
    plan.events,
  );
});

test("crowded minimum-view geometry retains all eight authoritative NET outcomes", () => {
  /** @type {Map<number, readonly [number, number]>} */
  const centers = new Map([
    [0, [155.3, 248]],
    [1, [205.7, 248]],
    [2, [256.1, 248]],
    [3, [306.5, 248]],
    [4, [356.9, 248]],
    [5, [172.1, 178]],
    [6, [222.5, 178]],
    [7, [272.9, 178]],
    [8, [323.3, 178]],
    [9, [373.7, 178]],
  ]);
  const protectedRects = [
    [125.3, 218, 185.3, 278],
    [187.7, 230, 223.7, 266],
    [238.1, 230, 274.1, 266],
    [288.5, 230, 324.5, 266],
    [338.9, 230, 374.9, 266],
    [154.1, 160, 190.1, 196],
    [204.5, 160, 240.5, 196],
    [242.9, 148, 302.9, 208],
    [305.3, 160, 341.3, 196],
    [355.7, 160, 391.7, 196],
    [32.3, 219, 120.3, 277],
    [185.7, 271, 273.7, 329],
    [278.5, 271, 366.5, 329],
    [379.9, 219, 467.9, 277],
    [128.1, 97, 216.1, 155],
    [208.5, 201, 236.5, 219],
    [228.9, 85, 316.9, 143],
    [303.3, 201, 391.3, 219],
    [329.7, 97, 417.7, 155],
    [243.1, 209, 285.1, 225],
    [63.1, 170, 149.1, 186],
    [396.7, 170, 482.7, 186],
  ].map(([left, top, right, bottom]) => ({
    left,
    top,
    right,
    bottom,
    width: right - left,
    height: bottom - top,
  }));
  const viewportBounds = {
    left: 54.5,
    top: 24,
    right: 502.5,
    bottom: 360,
    width: 448,
    height: 336,
  };
  /** @type {ProjectionSurface} */
  const crowdedSurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length * 28,
    viewportBounds,
    protectedRects,
  };
  const activations = [
    ["crowded-basic-a", "basic_damage", 0, 5],
    ["crowded-basic-b", "basic_damage", 5, 0],
    ["crowded-charge-a", "warrior_charge", 1, 6],
    ["crowded-charge-b", "warrior_charge", 6, 1],
    ["crowded-trap-a", "hunter_trap", 2, 7],
    ["crowded-trap-b", "hunter_trap", 7, 2],
    ["crowded-poison-a", "rogue_poison", 3, 8],
    ["crowded-poison-b", "rogue_poison", 8, 3],
    ["crowded-holy-a", "holy_word", 4, 4],
    ["crowded-holy-b", "holy_word", 9, 9],
  ].map(([eventId, tokenId, source, target]) =>
    activation(String(eventId), String(tokenId), Number(source), Number(target), {
      sourceAnchor: centers.get(Number(source)),
      targetAnchor: centers.get(Number(target)),
    }),
  );
  const netRecipients = [0, 1, 3, 4, 5, 6, 8, 9];
  const netEvents = netRecipients.map((recipient, index) => ({
    event_type: "net_health",
    event_id: `crowded-net-${recipient}`,
    transition_id: 4,
    recipient_global_slot: recipient,
    recipient_anchor: centers.get(recipient),
    health_before: 50,
    health_after: index === 3 || index === 7 ? 60 : 42,
    net_delta: index === 3 || index === 7 ? 10 : -8,
    outcome: index === 3 || index === 7 ? "healing" : "damage",
  }));

  const plan = buildChoreographyPlan(
    debuggerFrame([...activations, ...netEvents]),
    crowdedSurface,
  );
  assert.ok(plan);
  const outcomes = netRecipients.map((recipient) =>
    eventById(plan.events, `crowded-net-${recipient}`),
  );
  assert.equal(
    outcomes.every(
      (event) =>
        event.cueCollisionFree === true &&
        event.spatialDisposition === "rendered" &&
        event.cueBounds,
    ),
    true,
  );
  for (const [index, event] of outcomes.entries()) {
    assert.ok(event.cueBounds.left >= viewportBounds.left);
    assert.ok(event.cueBounds.top >= viewportBounds.top);
    assert.ok(event.cueBounds.right <= viewportBounds.right);
    assert.ok(event.cueBounds.bottom <= viewportBounds.bottom);
    for (const protectedRect of protectedRects) {
      assert.equal(rectangleIntersectionArea(event.cueBounds, protectedRect), 0);
    }
    for (const other of outcomes.slice(index + 1)) {
      assert.equal(rectangleIntersectionArea(event.cueBounds, other.cueBounds), 0);
    }
  }
  for (const chargeId of ["crowded-charge-a", "crowded-charge-b"]) {
    const charge = eventById(plan.events, chargeId);
    if (charge.ownershipCueCollisionFree) {
      for (const outcome of outcomes) {
        assert.equal(
          rectangleIntersectionArea(charge.ownershipBounds, outcome.cueBounds),
          0,
        );
      }
    } else {
      assert.equal(charge.ownershipCue, null);
      assert.equal(charge.ownershipCueSuppressionReason, "no_collision_free_position");
    }
  }
  assert.deepEqual(
    buildChoreographyPlan(debuggerFrame([...activations, ...netEvents]), crowdedSurface)
      ?.events,
    plan.events,
  );
});

test("accepted activations carry recipient impact grammar without health amounts", () => {
  const events = [
    activation("damage-basic", "basic_damage", 0, 5),
    activation("healing-basic", "basic_heal", 9, 5),
    activation("damage-charge", "warrior_charge", 1, 6),
    activation("neutral-trap", "hunter_trap", 2, 7),
    activation("damage-poison", "rogue_poison", 3, 8),
    activation("healing-holy", "holy_word", 4, 9),
    activation("local-burst", "mage_burst", 0, null, {
      disclosure: "target_none",
    }),
  ];
  const plan = buildChoreographyPlan(debuggerFrame(events), surface);
  assert.ok(plan);
  const activations = plan.events.filter(({ kind }) => kind === "activation");
  assert.deepEqual(
    activations.map(({ impactSemantic }) => impactSemantic),
    ["damage", "healing", "damage", "neutral", "damage", "healing", "local"],
  );
  assert.equal(
    activations.every(
      (event) =>
        !Object.hasOwn(event, "netDelta") &&
        !Object.hasOwn(event, "healthBefore") &&
        !Object.hasOwn(event, "healthAfter"),
    ),
    true,
  );
});

test("a public target with a hidden source becomes an impact-phase target cue", () => {
  const targetOnly = activation("target-only", "holy_word", 4, 5, {
    sourceAnchor: null,
    targetAnchor: [6, 3],
  });
  const plan = buildChoreographyPlan(
    debuggerFrame([targetOnly], {
      audience: "agent_pov",
      controlledSlot: 5,
    }),
    surface,
  );
  assert.ok(plan);
  const event = eventById(plan.events, "target-only");
  assert.equal(event.presentationKind, "target_only_impact");
  assert.equal(event.source, null);
  assert.ok(event.target);
  assert.equal(event.route, null);
  assert.equal(event.phaseStart, CHOREOGRAPHY_PHASES.impactStart);
});

test("recipient net outcomes remain separate and health lanes are bounded", () => {
  const events = [
    activation("a-damage", "basic_damage", 0, 5),
    activation("a-heal", "basic_heal", 9, 5),
    ...Array.from({ length: 4 }, (_, index) => ({
      event_type: "net_health",
      event_id: `net-${index}`,
      transition_id: 4,
      recipient_global_slot: 5,
      recipient_anchor: [6, 4],
      health_before: 50,
      health_after: 50,
      net_delta: 0,
      outcome: "unchanged",
    })),
  ];
  const plan = buildChoreographyPlan(debuggerFrame(events), {
    ...surface,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 300,
      bottom: 240,
      width: 300,
      height: 240,
    },
  });
  assert.ok(plan);
  const health = plan.events.filter(({ kind }) => kind === "net_health");
  assert.equal(health.length, 4);
  assert.deepEqual(
    health.map(({ lane }) => lane),
    [0, 1, 2, 3],
  );
  assert.deepEqual(
    health.map(({ spatial }) => spatial),
    [true, true, true, false],
  );
  assert.equal(
    new Set(
      health.filter(({ spatial }) => spatial).map(({ cue }) => `${cue.x}:${cue.y}`),
    ).size,
    3,
  );
  assert.equal(
    health.every(({ netDelta }) => netDelta === 0),
    true,
  );
  assert.equal(
    plan.events
      .filter(({ kind }) => kind === "activation")
      .every((event) => !Object.hasOwn(event, "healthBefore")),
    true,
  );
});

test("dense outcome layout searches the whole viewport before suppressing a cue", () => {
  /** @type {ProjectionSurface} */
  const denseSurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 600,
      bottom: 200,
      width: 600,
      height: 200,
    },
    protectedRects: [
      {
        left: 0,
        top: 0,
        right: 260,
        bottom: 200,
        width: 260,
        height: 200,
      },
    ],
  };
  const frame = debuggerFrame([
    {
      event_type: "net_health",
      event_id: "net-global-fallback",
      transition_id: 4,
      recipient_global_slot: 5,
      recipient_anchor: [100, 100],
      health_before: 50,
      health_after: 40,
      net_delta: -10,
      outcome: "damage",
    },
  ]);
  const localPlan = buildChoreographyPlan(frame, {
    ...denseSurface,
    protectedRects: [],
  });
  assert.ok(localPlan);
  assert.deepEqual(eventById(localPlan.events, "net-global-fallback").cue, {
    x: 100,
    y: 56,
  });

  const plan = buildChoreographyPlan(frame, denseSurface);
  assert.ok(plan);
  const outcome = eventById(plan.events, "net-global-fallback");
  assert.equal(outcome.spatialDisposition, "rendered");
  assert.equal(outcome.cueCollisionFree, true);
  assert.ok(outcome.cue.x >= 304);
  assert.deepEqual(buildChoreographyPlan(frame, denseSurface)?.events, plan.events);
});

test("dense outcome layout checks protected edges that a fixed grid can skip", () => {
  /** @type {ProjectionSurface} */
  const narrowCorridorSurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 320,
      bottom: 100,
      width: 320,
      height: 100,
    },
    protectedRects: [
      {
        left: 0,
        top: 0,
        right: 110,
        bottom: 100,
        width: 110,
        height: 100,
      },
      {
        left: 202,
        top: 0,
        right: 320,
        bottom: 100,
        width: 118,
        height: 100,
      },
    ],
  };
  const frame = debuggerFrame([
    {
      event_type: "net_health",
      event_id: "net-critical-edge",
      transition_id: 4,
      recipient_global_slot: 5,
      recipient_anchor: [60, 50],
      health_before: 50,
      health_after: 40,
      net_delta: -10,
      outcome: "damage",
    },
  ]);
  const plan = buildChoreographyPlan(frame, narrowCorridorSurface);
  assert.ok(plan);
  const outcome = eventById(plan.events, "net-critical-edge");
  assert.equal(outcome.spatialDisposition, "rendered");
  assert.equal(outcome.cueCollisionFree, true);
  assert.deepEqual(outcome.cue, { x: 156, y: 50 });
  assert.deepEqual(
    buildChoreographyPlan(frame, narrowCorridorSurface)?.events,
    plan.events,
  );
});

test("NET placement priority is order-independent and suppression stays retained", () => {
  /** @type {ProjectionSurface} */
  const prioritySurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 96,
      bottom: 60,
      width: 96,
      height: 60,
    },
    protectedRects: [],
  };
  const net = {
    event_type: "net_health",
    event_id: "net-priority",
    transition_id: 4,
    recipient_global_slot: 5,
    recipient_anchor: [48, 24],
    health_before: 50,
    health_after: 40,
    net_delta: -10,
    outcome: "damage",
  };
  const lifecycle = {
    event_type: "status_lifecycle",
    event_id: "lifecycle-priority",
    transition_id: 4,
    recipient_global_slot: 5,
    recipient_anchor: [48, 24],
    token_id: "stun_hunter_trap",
    change: "applied",
    duration_before: 0,
    duration_after: 4,
    source_class_id: 3,
    application_event_ids: [],
  };
  for (const events of [
    [lifecycle, net],
    [net, lifecycle],
  ]) {
    const plan = buildChoreographyPlan(debuggerFrame(events), prioritySurface);
    assert.ok(plan);
    const placedNet = eventById(plan.events, "net-priority");
    const suppressedLifecycle = eventById(plan.events, "lifecycle-priority");
    assert.equal(placedNet.spatialDisposition, "rendered");
    assert.equal(placedNet.cueCollisionFree, true);
    assert.equal(suppressedLifecycle.spatial, true);
    assert.equal(suppressedLifecycle.spatialDisposition, "suppressed_collision");
    assert.equal(suppressedLifecycle.cue, null);
    assert.equal(
      suppressedLifecycle.cueSuppressionReason,
      "no_collision_free_position",
    );
    assert.deepEqual(
      plan.events.map(({ eventId }) => eventId),
      events.map(({ event_id: eventId }) => eventId),
    );
  }
});

test("an impossible NET cue remains an explicit retained suppression", () => {
  /** @type {ProjectionSurface} */
  const impossibleSurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 80,
      bottom: 24,
      width: 80,
      height: 24,
    },
    protectedRects: [],
  };
  const plan = buildChoreographyPlan(
    debuggerFrame([
      {
        event_type: "net_health",
        event_id: "net-impossible",
        transition_id: 4,
        recipient_global_slot: 5,
        recipient_anchor: [40, 12],
        health_before: 50,
        health_after: 40,
        net_delta: -10,
        outcome: "damage",
      },
    ]),
    impossibleSurface,
  );
  assert.ok(plan);
  const outcome = eventById(plan.events, "net-impossible");
  assert.equal(outcome.spatial, true);
  assert.equal(outcome.spatialDisposition, "suppressed_collision");
  assert.equal(outcome.cue, null);
  assert.equal(outcome.netDelta, -10);
});

test("exact Trap endings outrank generic lifecycle decoration under pressure", () => {
  /** @type {ProjectionSurface} */
  const singleCueSurface = {
    worldToScreen: (value) => ({
      x: "x" in value ? value.x : Number(value[0]),
      y: "y" in value ? value.y : Number(value[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 60,
      bottom: 60,
      width: 60,
      height: 60,
    },
    protectedRects: [],
  };
  const lifecycleBase = {
    event_type: "status_lifecycle",
    transition_id: 4,
    recipient_global_slot: 5,
    recipient_anchor: [30, 30],
    token_id: "stun_hunter_trap",
    source_class_id: 3,
    application_event_ids: [],
  };
  const plan = buildChoreographyPlan(
    debuggerFrame([
      {
        ...lifecycleBase,
        event_id: "generic-application",
        change: "applied",
        duration_before: 0,
        duration_after: 4,
      },
      {
        ...lifecycleBase,
        event_id: "exact-break",
        change: "trap_broken",
        duration_before: 2,
        duration_after: 0,
      },
    ]),
    singleCueSurface,
  );
  assert.ok(plan);
  assert.equal(eventById(plan.events, "exact-break").spatialDisposition, "rendered");
  assert.equal(
    eventById(plan.events, "generic-application").spatialDisposition,
    "suppressed_collision",
  );
  assert.deepEqual(
    plan.events.map(({ eventId }) => eventId),
    ["generic-application", "exact-break"],
  );
});

test("lifecycle, Charge, rejection, and unknown facts fail closed", () => {
  const events = [
    {
      event_type: "charge_displacement",
      event_id: "charge-0",
      transition_id: 4,
      source_global_slot: 1,
      target_global_slot: 6,
      start: [2, 2],
      end: [5, 5],
      path_kind: "combined_charge_and_movement",
    },
    {
      event_type: "status_lifecycle",
      event_id: "status-break",
      transition_id: 4,
      recipient_global_slot: 6,
      recipient_anchor: [5, 5],
      token_id: "stun_hunter_trap",
      change: "trap_broken",
      duration_before: 2,
      duration_after: 0,
      source_class_id: 3,
      application_event_ids: [],
    },
    {
      event_type: "status_lifecycle",
      event_id: "status-ambiguous",
      transition_id: 4,
      recipient_global_slot: 7,
      recipient_anchor: [6, 5],
      token_id: "stun_hunter_trap",
      change: "cleared_unclassified",
      duration_before: 1,
      duration_after: 0,
      source_class_id: 3,
      application_event_ids: [],
    },
    {
      event_type: "status_lifecycle",
      event_id: "status-age",
      transition_id: 4,
      recipient_global_slot: 8,
      recipient_anchor: [7, 5],
      token_id: "stun_hunter_trap",
      change: "decremented",
      duration_before: 3,
      duration_after: 2,
      source_class_id: 3,
      application_event_ids: [],
    },
    {
      event_type: "rejected_action",
      event_id: "reject-0",
      transition_id: 4,
      actor_global_slot: 0,
      component: "combat",
      actor_anchor: [2, 2],
      target_global_slot: 5,
      target_anchor: [6, 2],
      target_disclosure: "public",
      lane: 1,
      movement_mask_value: true,
      pair_mask_value: false,
    },
    {
      event_type: "future_private_event",
      event_id: "unknown-0",
      transition_id: 4,
      secret_anchor: [99, 99],
    },
  ];
  const plan = buildChoreographyPlan(debuggerFrame(events), surface);
  assert.ok(plan);
  const charge = eventById(plan.events, "charge-0");
  assert.equal(charge.persistent, true);
  assert.equal(charge.pathKind, "combined_charge_and_movement");
  assert.equal(eventById(plan.events, "status-break").lifecycle, "trap_broken");
  assert.equal(
    eventById(plan.events, "status-ambiguous").lifecycle,
    "cleared_unclassified",
  );
  assert.equal(eventById(plan.events, "status-age").spatial, false);
  assert.ok(eventById(plan.events, "reject-0").route);
  assert.deepEqual(eventById(plan.events, "unknown-0"), {
    eventId: "unknown-0",
    eventType: "future_private_event",
    transitionId: 4,
    kind: "unknown",
    spatial: false,
  });
});

test("malformed and oversized payloads fail closed before SVG planning", () => {
  const invalidAudience = debuggerFrame([], { audience: "unexpected" });
  assert.equal(authorizationContextKey(invalidAudience), null);
  assert.equal(buildChoreographyPlan(invalidAudience, surface), null);

  const malformed = debuggerFrame([
    {
      event_type: "net_health",
      event_id: "bad-net",
      transition_id: 4,
      recipient_global_slot: 5,
      recipient_anchor: [6, 4],
      health_before: 50,
      health_after: 50,
      net_delta: "0",
      outcome: "unchanged",
    },
    {
      event_type: "rejected_action",
      event_id: "bad-mask",
      transition_id: 4,
      actor_global_slot: 0,
      component: "combat",
      actor_anchor: [2, 2],
      target_global_slot: 5,
      target_anchor: [6, 2],
      target_disclosure: "public",
      lane: 1,
      movement_mask_value: "true",
      pair_mask_value: false,
    },
  ]);
  const malformedPlan = buildChoreographyPlan(malformed, surface);
  assert.ok(malformedPlan);
  assert.equal(
    malformedPlan.events.every(({ kind }) => kind === "unknown"),
    true,
  );

  const oversizedEvents = Array.from({ length: 129 }, (_, index) => ({
    event_type: "future_event",
    event_id: `oversized-${index}`,
    transition_id: 4,
  }));
  assert.throws(
    () => buildChoreographyPlan(debuggerFrame(oversizedEvents), surface),
    /128-event presentation limit/,
  );

  const tooManySpatialEvents = Array.from({ length: 97 }, (_, index) =>
    activation(`spatial-${index}`, "mage_burst", 0, null, {
      disclosure: "target_none",
    }),
  );
  assert.equal(
    buildChoreographyPlan(debuggerFrame(tooManySpatialEvents), surface)?.events.length,
    97,
  );
});
