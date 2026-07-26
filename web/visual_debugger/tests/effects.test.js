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
  assert.equal(eventById(activations, "a-4").target, null);
  assert.equal(eventById(activations, "a-5").route, null);
  assert.equal(eventById(activations, "a-5").targetSlot, null);
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
