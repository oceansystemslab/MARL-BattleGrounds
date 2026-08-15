import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  loadRendererFixture,
  syntheticDebuggerPresentationFrame,
} from "../e2e/support/renderer-fixture.js";
import {
  explainChoreographyEvent,
  statusApplicationRoutes,
} from "../src/choreography-painter.js";
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
 *   worldToScreen: (point: readonly [number, number] | {x: number, y: number}) =>
 *     {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 *   viewportBounds: {
 *     left: number,
 *     top: number,
 *     right: number,
 *     bottom: number,
 *     width: number,
 *     height: number,
 *   },
 *   protectedRects: ReadonlyArray<{
 *     left: number,
 *     top: number,
 *     right: number,
 *     bottom: number,
 *     width: number,
 *     height: number,
 *   }>,
 * }} ProjectionSurface
 * @typedef {{
 *   grammar: Record<string, any>,
 *   routes: Record<string, any>,
 *   crowded: Record<string, any>,
 *   pov: Record<string, any>,
 * }} CanonicalFrames
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

/** @type {Promise<CanonicalFrames> | undefined} */
let canonicalFramesPromise;

/** @returns {Promise<CanonicalFrames>} */
async function canonicalFrames() {
  if (!canonicalFramesPromise) {
    canonicalFramesPromise = (async () => {
      const grammar = syntheticDebuggerPresentationFrame(
        await loadRendererFixture("canonical_event_vocabulary"),
      );
      const routes = syntheticDebuggerPresentationFrame(
        await loadRendererFixture("route_collision"),
      );
      const crowded = syntheticDebuggerPresentationFrame(
        await loadRendererFixture("crowded_teamfight"),
      );
      const pov = syntheticDebuggerPresentationFrame(
        await loadRendererFixture("pov_redaction"),
      );
      return { grammar, routes, crowded, pov };
    })();
  }
  return canonicalFramesPromise;
}

test("transition identity excludes revision while authorization tracks POV actor", async () => {
  const { grammar, pov } = await canonicalFrames();
  const revised = { ...grammar, revision: grammar.revision + 1 };
  assert.equal(transitionEpochKey(revised), transitionEpochKey(grammar));
  assert.equal(eventFingerprint(revised), eventFingerprint(grammar));
  assert.deepEqual(JSON.parse(authorizationContextKey(grammar) ?? "null"), [
    grammar.session_id,
    grammar.run_generation,
    "researcher",
    null,
  ]);
  assert.deepEqual(JSON.parse(authorizationContextKey(pov) ?? "null"), [
    pov.session_id,
    pov.run_generation,
    "agent_pov",
    pov.scene.self_actor.global_slot,
  ]);
  const replayPov = {
    ...pov,
    viewer_mode: "replay",
    replay_audience: "actor_pov",
    pov_global_slot: pov.scene.self_actor.global_slot,
    scene: {
      ...pov.scene,
      selection: {
        controlled_global_slot: null,
        selected_global_slot: null,
      },
    },
  };
  assert.deepEqual(JSON.parse(authorizationContextKey(replayPov) ?? "null"), [
    pov.session_id,
    pov.run_generation,
    "agent_pov",
    pov.scene.self_actor.global_slot,
  ]);
  const rejectedDiagnosticReplay = {
    ...replayPov,
    replay_audience: "shared_obs_source_material",
    selected_global_slot: pov.scene.self_actor.global_slot,
  };
  assert.equal(authorizationContextKey(rejectedDiagnosticReplay), null);
  assert.equal(buildChoreographyPlan(rejectedDiagnosticReplay, surface), null);
  assert.notEqual(authorizationContextKey(grammar), authorizationContextKey(pov));
});

test("submission classification is presentation gating, not legality", () => {
  assert.equal(isSubmissionCommand({ command_type: "keyboard", key: "Enter" }), true);
  assert.equal(isSubmissionCommand({ command_type: "keyboard", key: "N" }), true);
  assert.equal(
    isSubmissionCommand({ command_type: "keyboard", key: "ArrowLeft" }),
    false,
  );
  assert.equal(
    isSubmissionCommand({ command_type: "actor_pov_target_action", target_action: 1 }),
    false,
  );
});

test("canonical V2 displacement and rejection selectors own transient color", async () => {
  const css = await readFile(new URL("../styles.css", import.meta.url), "utf8");
  assert.match(css, /data-event-type="charge_phase_displacement"/u);
  assert.match(css, /data-event-type="action_rejected"/u);
});

test("scene has one Analysis battlefield branch and owns durable shield hooks", async () => {
  const source = await readFile(new URL("../src/scene.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /debug-protected-zone/u);
  assert.doesNotMatch(source, /debug-visibility-cue/u);
  assert.doesNotMatch(source, /preset === "debug"/u);
  assert.doesNotMatch(source, /preset !== "presentation"/u);
  assert.doesNotMatch(source, /scene\.audience_badge/u);
  assert.match(source, /dataset\.preset = "analysis"/u);
  assert.match(
    source,
    /scene\.audience === "researcher" \? "Oracle View" : "Agent POV"/u,
  );
  assert.match(source, /showLegality: true/u);
  assert.match(source, /agent-spawn-shield__shell/u);
  assert.match(source, /agent-spawn-shield__ticks/u);
  assert.match(source, /registerTooltipOwner\(nodes\.shieldRoot, explainSpawnShield/u);
  assert.match(source, /Spawn Shield, invulnerable/u);
});

test("canonical fixture gives every V2 event one exact ordered disposition", async () => {
  const { grammar } = await canonicalFrames();
  const plan = buildChoreographyPlan(grammar, surface);
  assert.ok(plan);
  assert.equal(plan.transitionId, grammar.incoming_transition_id);
  const sourceEvents = /** @type {Record<string, any>[]} */ (
    grammar.event_batch.events
  );
  assert.equal(plan.events.length, sourceEvents.length);
  assert.deepEqual(
    plan.events.map((event) => event.eventId),
    sourceEvents.map((event) => event.event_id),
  );
  assert.deepEqual(
    plan.events.map((event) => event.eventType),
    sourceEvents.map((event) => event.event_type),
  );
  assert.equal(
    plan.events.some((event) => event.kind === "unknown"),
    false,
  );
});

test("canonical V2 phases retain readable M5 windows and extend for M6 beats", async () => {
  const { grammar } = await canonicalFrames();
  const plan = buildChoreographyPlan(grammar, surface);
  assert.ok(plan);
  const byType = new Map(plan.events.map((event) => [event.eventType, event]));
  const rejection = byType.get("action_rejected");
  const ability = byType.get("ability_activated");
  assert.ok(rejection);
  assert.ok(ability);
  assert.deepEqual(
    [rejection.phaseStart, rejection.phaseEnd],
    [CHOREOGRAPHY_PHASES.activationStart, CHOREOGRAPHY_PHASES.settleStart],
  );
  assert.deepEqual(
    [ability.phaseStart, ability.phaseImpact, ability.phaseEnd],
    [
      CHOREOGRAPHY_PHASES.activationStart,
      CHOREOGRAPHY_PHASES.impactStart,
      CHOREOGRAPHY_PHASES.settleStart,
    ],
  );

  const orderedOutcomeTypes = [
    "recipient_health_resolution",
    "combat_countdown_reset",
    "cooldown_started",
    "charge_phase_displacement",
    "ordinary_movement_phase_displacement",
    "agent_died",
    "status_applied",
    "spawn_shield_expired",
    "respawn_wave_occurred",
    "agent_respawned",
  ];
  for (let index = 1; index < orderedOutcomeTypes.length; index += 1) {
    const previous = byType.get(orderedOutcomeTypes[index - 1]);
    const current = byType.get(orderedOutcomeTypes[index]);
    assert.ok(previous);
    assert.ok(current);
    assert.ok(previous.phaseEnd <= current.phaseStart);
  }
  for (const type of [
    "recipient_health_resolution",
    "charge_phase_displacement",
    "agent_died",
    "status_applied",
    "spawn_shield_expired",
    "respawn_wave_occurred",
    "agent_respawned",
  ]) {
    const event = byType.get(type);
    assert.ok(event);
    assert.ok(event.phaseEnd - event.phaseStart >= 480, `${type} needs readable dwell`);
  }
  const charge = byType.get("charge_phase_displacement");
  assert.equal(charge?.phaseEnd - charge?.phaseStart, 540);
  assert.equal(charge?.persistent, true);
  const movement = byType.get("ordinary_movement_phase_displacement");
  assert.ok(movement);
  assert.equal(movement.phaseStart, movement.phaseEnd);
  assert.equal(movement.spatial, false);
  assert.equal(movement.presentationSuppressed, true);
  assert.equal(movement.persistent, false);
  assert.equal(byType.get("agent_respawned")?.phaseEnd, plan.phases.total);
  assert.ok(plan.phases.total > CHOREOGRAPHY_PHASES.total);
});

test("absent choreography families reserve no presentation time", async () => {
  const { grammar } = await canonicalFrames();
  const sourceEvents = /** @type {Record<string, any>[]} */ (
    grammar.event_batch.events
  );
  /** @param {string[]} eventTypes */
  const planFor = (eventTypes) =>
    buildChoreographyPlan(
      {
        ...grammar,
        event_batch: {
          ...grammar.event_batch,
          events: sourceEvents.filter((event) => eventTypes.includes(event.event_type)),
        },
      },
      surface,
    );

  const abilityOnly = planFor(["ability_activated"]);
  assert.ok(abilityOnly);
  assert.equal(abilityOnly.phases.total, CHOREOGRAPHY_PHASES.settleStart);
  assert.ok(
    abilityOnly.events.every(
      (event) =>
        event.phaseStart === CHOREOGRAPHY_PHASES.activationStart &&
        event.phaseEnd === CHOREOGRAPHY_PHASES.settleStart,
    ),
  );

  const healthOnly = planFor(["recipient_health_resolution"]);
  assert.ok(healthOnly);
  assert.equal(healthOnly.phases.total, 480);
  assert.ok(
    healthOnly.events.every(
      (event) => event.phaseStart === 0 && event.phaseEnd === 480,
    ),
  );

  const movementOnly = planFor(["ordinary_movement_phase_displacement"]);
  assert.ok(movementOnly);
  assert.equal(movementOnly.phases.total, 0);
  assert.equal(movementOnly.events[0]?.spatial, false);
  assert.equal(movementOnly.events[0]?.presentationSuppressed, true);

  const feedOnly = planFor(["source_damage_output"]);
  assert.ok(feedOnly);
  assert.equal(feedOnly.phases.total, 0);
  assert.equal(feedOnly.events[0]?.kind, "feed_only");
});

test("status presentation composes serialized atomic events without dropping IDs", async () => {
  const { grammar } = await canonicalFrames();
  const sourceEvents = /** @type {Record<string, any>[]} */ (
    grammar.event_batch.events
  );
  const expired = structuredClone(
    sourceEvents.find((event) => event.event_type === "status_aged_to_zero"),
  );
  const broken = structuredClone(
    sourceEvents.find((event) => event.event_type === "status_broken_by_damage"),
  );
  const applied = structuredClone(
    sourceEvents.find((event) => event.event_type === "status_applied"),
  );
  assert.ok(expired);
  assert.ok(broken);
  assert.ok(applied);

  /** @param {Record<string, any>} predecessor */
  const composeWithApplied = (predecessor) => {
    const alignedPredecessor = /** @type {Record<string, any>} */ ({
      ...predecessor,
      recipient_global_slot: applied.recipient_global_slot,
      recipient_anchor: structuredClone(applied.recipient_anchor),
      status_channel: applied.status_channel,
      status_id: applied.status_id,
    });
    const events = /** @type {Record<string, any>[]} */ ([alignedPredecessor, applied]);
    const plan = buildChoreographyPlan(
      {
        ...grammar,
        event_batch: { ...grammar.event_batch, events },
      },
      surface,
    );
    assert.ok(plan);
    assert.deepEqual(
      plan.events.map((event) => event.eventId),
      events.map((event) => event.event_id),
    );
    const first = plan.events[0];
    const current = plan.events[1];
    assert.ok(first);
    assert.ok(current);
    assert.deepEqual(
      current.atomicEventIds,
      events.map((event) => event.event_id),
    );
    assert.deepEqual(current.applicationEventIds, [applied.event_id]);
    assert.deepEqual(first.atomicEventIds, current.atomicEventIds);
    assert.equal(first.presentationSuppressed, true);
    assert.equal(first.spatial, false);
    assert.equal(current.presentationSuppressed, false);
    assert.equal(current.spatial, true);
    assert.equal(current.durationBefore, null);
    assert.equal(current.durationAfter, null);
    return current;
  };

  assert.equal(composeWithApplied(expired).lifecycle, "expired_then_reapplied");
  assert.equal(composeWithApplied(broken).lifecycle, "trap_broken_and_reapplied");
});

test("status presentation preserves the valid five-source application maximum", async () => {
  const { grammar } = await canonicalFrames();
  const maximumFrame = structuredClone(grammar);
  const sourceEvents = /** @type {Record<string, any>[]} */ (
    maximumFrame.event_batch.events
  );
  const appliedTemplate = structuredClone(
    sourceEvents.find((event) => event.event_type === "status_applied"),
  );
  assert.ok(appliedTemplate);
  const sourceSlots = [5, 6, 7, 8, 9];
  const successorAnchors = new Map(
    maximumFrame.event_batch.agent_phase_trajectories.map(
      /** @param {Record<string, any>} trajectory */ (trajectory) => [
        trajectory.global_slot,
        trajectory.successor,
      ],
    ),
  );
  const applications = sourceSlots.map((sourceSlot, index) => ({
    ...structuredClone(appliedTemplate),
    event_id: `${appliedTemplate.transition_id}:event:maximum-application:${sourceSlot}`,
    ordinal: appliedTemplate.ordinal + index,
    source_global_slot: sourceSlot,
    source_anchor: structuredClone(successorAnchors.get(sourceSlot)),
    status_channel: 4,
    status_id: "hunter_trap_stun",
  }));
  for (const sourceSlot of sourceSlots) {
    const sourceAgent = maximumFrame.scene.agents.find(
      /** @param {Record<string, any>} agent */ (agent) =>
        agent.global_slot === sourceSlot,
    );
    assert.ok(sourceAgent);
    sourceAgent.class_id = 3;
  }

  const plan = buildChoreographyPlan(
    {
      ...maximumFrame,
      event_batch: { ...maximumFrame.event_batch, events: applications },
    },
    surface,
  );
  assert.ok(plan);
  assert.equal(plan.events.length, sourceSlots.length);
  const applicationEventIds = applications.map((event) => event.event_id);
  assert.deepEqual(
    plan.events.map((event) => event.eventId),
    applicationEventIds,
  );
  assert.equal(
    plan.events.filter((event) => event.presentationSuppressed !== true).length,
    1,
  );
  for (const [index, event] of plan.events.entries()) {
    assert.deepEqual(event.atomicEventIds, applicationEventIds);
    assert.deepEqual(event.applicationEventIds, applicationEventIds);
    assert.deepEqual(
      event.applicationSources.map(
        /** @param {Record<string, any>} source */ (source) => source.eventId,
      ),
      applicationEventIds,
    );
    assert.deepEqual(
      event.applicationSources.map(
        /** @param {Record<string, any>} source */ (source) => source.sourceSlot,
      ),
      sourceSlots,
    );
    assert.equal(event.sourceSlot, sourceSlots[index]);
  }

  const primary = plan.events.at(-1);
  assert.ok(primary);
  const routes = statusApplicationRoutes(primary);
  assert.deepEqual(
    routes.map((route) => route.applicationEventId),
    applicationEventIds,
  );
  assert.deepEqual(
    routes.map((route) => route.sourceSlot),
    sourceSlots,
  );
  assert.ok(routes.every((route) => route.path.endsWith("L 20 90")));
  const explanation = explainChoreographyEvent(primary);
  assert.deepEqual(
    explanation.rows.map(({ label, value }) => [label, value]),
    [
      [
        "Application Sources",
        "Agent ID 5; Agent ID 6; Agent ID 7; Agent ID 8; Agent ID 9",
      ],
      ["Recipient", "Agent ID 0"],
    ],
  );
});

test("invalid-transition status rows cannot influence a valid atomic cue", async () => {
  const { grammar } = await canonicalFrames();
  const sourceEvents = /** @type {Record<string, any>[]} */ (
    grammar.event_batch.events
  );
  const expired = structuredClone(
    sourceEvents.find((event) => event.event_type === "status_aged_to_zero"),
  );
  const applied = structuredClone(
    sourceEvents.find((event) => event.event_type === "status_applied"),
  );
  assert.ok(expired);
  assert.ok(applied);
  applied.recipient_global_slot = expired.recipient_global_slot;
  applied.recipient_anchor = structuredClone(expired.recipient_anchor);
  applied.status_channel = expired.status_channel;
  applied.status_id = expired.status_id;
  applied.transition_id = "different-transition";
  const plan = buildChoreographyPlan(
    {
      ...grammar,
      event_batch: { ...grammar.event_batch, events: [expired, applied] },
    },
    surface,
  );
  assert.ok(plan);
  assert.equal(plan.events[0]?.lifecycle, "expired");
  assert.deepEqual(plan.events[0]?.atomicEventIds, [expired.event_id]);
  assert.equal(plan.events[1]?.kind, "unknown");
});

test("standalone serialized status events keep exact presentation meanings", async () => {
  const { grammar } = await canonicalFrames();
  const statusEvents = /** @type {Record<string, any>[]} */ (
    grammar.event_batch.events.filter(
      /** @param {Record<string, any>} event */ (event) =>
        event.event_type.startsWith("status_"),
    )
  );
  const plan = buildChoreographyPlan(
    {
      ...grammar,
      event_batch: { ...grammar.event_batch, events: statusEvents },
    },
    surface,
  );
  assert.ok(plan);
  assert.deepEqual(
    plan.events.map((event) => [event.eventType, event.lifecycle]),
    [
      ["status_aged_to_zero", "expired"],
      ["status_broken_by_damage", "trap_broken"],
      ["status_applied", "applied"],
      ["status_refreshed_or_extended", "refreshed"],
      ["status_cleared_by_new_death", "cleared_by_death"],
    ],
  );
  assert.ok(
    plan.events.every(
      (event) =>
        event.atomicEventIds?.length === 1 &&
        event.atomicEventIds[0] === event.eventId &&
        event.phaseStart === 0 &&
        event.phaseEnd === 480,
    ),
  );
});

test("durable status countdown changes never synthesize browser lifecycle events", async () => {
  const { grammar } = await canonicalFrames();
  const countdownOnly = structuredClone(grammar);
  const statusOwner = countdownOnly.scene.agents.find(
    /** @param {Record<string, any>} agent */ (agent) =>
      Array.isArray(agent.statuses) && agent.statuses.length > 0,
  );
  assert.ok(statusOwner);
  statusOwner.statuses[0].remaining_duration = Math.max(
    0,
    Number(statusOwner.statuses[0].remaining_duration ?? 1) - 1,
  );
  countdownOnly.event_batch.events = [];
  const plan = buildChoreographyPlan(countdownOnly, surface);
  assert.ok(plan);
  assert.deepEqual(plan.events, []);
  assert.equal(plan.phases.total, 0);
});

test("route fixture preserves multiplicity and exact V2 event identities", async () => {
  const { routes } = await canonicalFrames();
  const plan = buildChoreographyPlan(routes, surface);
  assert.ok(plan);
  assert.equal(plan.events.length, 9);
  assert.ok(plan.events.every((event) => event.kind === "activation"));
  const sourceEvents = /** @type {Record<string, any>[]} */ (routes.event_batch.events);
  assert.deepEqual(
    plan.events.map((event) => event.eventId),
    sourceEvents.map((event) => event.event_id),
  );
  assert.equal(new Set(plan.events.map((event) => event.eventId)).size, 9);
  assert.ok(plan.events.every((event) => event.route));
  assert.equal(
    plan.events.filter((event) => event.sourceSlot === 1 && event.targetSlot === 6)
      .length,
    2,
  );
});

test("crowded fixture retains every health and status cue under pressure", async () => {
  const { crowded } = await canonicalFrames();
  const plan = buildChoreographyPlan(crowded, surface);
  assert.ok(plan);
  assert.equal(plan.events.length, 32);
  assert.equal(plan.events.filter((event) => event.kind === "net_health").length, 8);
  assert.equal(
    plan.events.filter((event) => event.kind === "status_lifecycle").length,
    12,
  );
  assert.equal(plan.events.filter((event) => event.kind === "activation").length, 10);
  assert.equal(
    plan.events.filter((event) => event.kind === "charge_displacement").length,
    2,
  );
});

test("Freezing Trap uses ordinary target-body route clearance while Charge stays exceptional", async () => {
  const { crowded } = await canonicalFrames();
  const plan = buildChoreographyPlan(crowded, surface);
  assert.ok(plan);
  const agentsBySlot = new Map(
    crowded.scene.agents.map(
      /** @param {Record<string, any>} agent */ (agent) => [agent.global_slot, agent],
    ),
  );
  const routed = plan.events.filter(
    (event) =>
      event.kind === "activation" &&
      ["hunter_trap", "rogue_poison", "warrior_charge"].includes(event.tokenId),
  );
  assert.equal(routed.length, 6);

  /** @param {Record<string, any>} event */
  const targetClearance = (event) => {
    assert.ok(event.route);
    assert.ok(event.target);
    return Math.hypot(
      event.route.end.x - event.target.x,
      event.route.end.y - event.target.y,
    );
  };
  for (const event of routed) {
    const target = agentsBySlot.get(event.targetSlot);
    assert.ok(target);
    const targetRadius = surface.worldLengthToScreen(target.radius);
    const expectedGap = event.tokenId === "warrior_charge" ? 18 : 3;
    assert.ok(
      Math.abs(targetClearance(event) - (targetRadius + expectedGap)) < 1e-9,
      `${event.tokenId} must stop at its declared target-body clearance`,
    );
  }
  const trapClearances = routed
    .filter((event) => event.tokenId === "hunter_trap")
    .map(targetClearance);
  const ordinaryClearances = routed
    .filter((event) => event.tokenId === "rogue_poison")
    .map(targetClearance);
  assert.deepEqual(trapClearances, ordinaryClearances);
  assert.ok(
    routed
      .filter((event) => event.tokenId === "warrior_charge")
      .every((event) => targetClearance(event) > Math.max(...trapClearances)),
  );
});

test("choreography preserves nonnumeric public identities independently of slots", async () => {
  const { grammar } = await canonicalFrames();
  const publicIds = Object.freeze([
    "zulu",
    "alpha",
    "kestrel",
    "bravo",
    "ember",
    "quartz",
    "delta",
    "sierra",
    "nova",
    "cobalt",
  ]);
  const remapped = structuredClone(grammar);
  /** @param {unknown} value */
  const remapExplicitPublicIds = (value) => {
    if (Array.isArray(value)) {
      value.forEach(remapExplicitPublicIds);
      return;
    }
    if (!value || typeof value !== "object") {
      return;
    }
    const candidate = /** @type {Record<string, any>} */ (value);
    for (const [key, child] of Object.entries(candidate)) {
      if (
        key.includes("public_agent_id") &&
        typeof child === "string" &&
        Number.isInteger(Number(child)) &&
        publicIds[Number(child)] !== undefined
      ) {
        candidate[key] = publicIds[Number(child)];
      } else if (key === "public_agent_id_by_global_slot" && Array.isArray(child)) {
        candidate[key] = [...publicIds];
      } else {
        remapExplicitPublicIds(child);
      }
    }
  };
  remapExplicitPublicIds(remapped);

  const plan = buildChoreographyPlan(remapped, surface);
  assert.ok(plan);
  const roles = ["source", "target", "recipient", "actor", "agent"];
  let identityJoinCount = 0;
  for (const event of plan.events) {
    for (const role of roles) {
      const slot = event[`${role}Slot`];
      if (!Number.isInteger(slot)) {
        continue;
      }
      identityJoinCount += 1;
      assert.equal(event[`${role}PublicAgentId`], publicIds[slot]);
      assert.notEqual(event[`${role}PublicAgentId`], String(slot));
    }
  }
  assert.ok(identityJoinCount > 0);

  const splitIdentityRoot = structuredClone(remapped);
  splitIdentityRoot.event_batch.public_agent_id_by_global_slot[0] = "impostor";
  assert.equal(buildChoreographyPlan(splitIdentityRoot, surface), null);

  const truncatedIdentityRoot = structuredClone(remapped);
  truncatedIdentityRoot.event_batch.public_agent_id_by_global_slot.pop();
  assert.equal(buildChoreographyPlan(truncatedIdentityRoot, surface), null);

  const permutedIdentityRoot = structuredClone(remapped);
  [
    permutedIdentityRoot.event_batch.public_agent_id_by_global_slot[0],
    permutedIdentityRoot.event_batch.public_agent_id_by_global_slot[1],
  ] = [
    permutedIdentityRoot.event_batch.public_agent_id_by_global_slot[1],
    permutedIdentityRoot.event_batch.public_agent_id_by_global_slot[0],
  ];
  assert.equal(buildChoreographyPlan(permutedIdentityRoot, surface), null);

  const mismatchedEventRoot = structuredClone(remapped);
  const mismatchedEvent = mismatchedEventRoot.event_batch.events.find(
    /** @param {Record<string, any>} event */ (event) =>
      Number.isInteger(event.source_global_slot) &&
      Number.isInteger(event.recipient_global_slot) &&
      event.source_global_slot !== event.recipient_global_slot &&
      event.source_anchor,
  );
  assert.ok(mismatchedEvent);
  mismatchedEvent.source_global_slot = mismatchedEvent.recipient_global_slot;
  assert.equal(buildChoreographyPlan(mismatchedEventRoot, surface), null);

  const absentPaddedSlot = structuredClone(remapped);
  absentPaddedSlot.scene.agents = absentPaddedSlot.scene.agents.filter(
    /** @param {Record<string, any>} agent */ (agent) => agent.global_slot !== 9,
  );
  const paddedEvent = absentPaddedSlot.event_batch.events.find(
    /** @param {Record<string, any>} event */ (event) =>
      event.event_type === "recipient_health_resolution",
  );
  assert.ok(paddedEvent);
  paddedEvent.recipient_global_slot = 9;
  paddedEvent.recipient_anchor = {
    ...paddedEvent.recipient_anchor,
    global_slot: 9,
    public_agent_id: publicIds[9],
  };
  const paddedPlan = buildChoreographyPlan(absentPaddedSlot, surface);
  assert.ok(paddedPlan);
  assert.equal(
    paddedPlan.events.find((event) => event.eventId === paddedEvent.event_id)
      ?.recipientPublicAgentId,
    null,
  );

  const semanticEvent = plan.events.find(
    (event) => event.kind === "status_lifecycle" && event.sourceSlot !== null,
  );
  assert.ok(semanticEvent);
  const serializedDescriptor = JSON.stringify(explainChoreographyEvent(semanticEvent));
  assert.equal(serializedDescriptor.includes(semanticEvent.eventId), false);
  assert.equal(serializedDescriptor.includes(semanticEvent.transitionId), false);
  assert.ok(
    serializedDescriptor.includes(`Agent ID ${publicIds[semanticEvent.sourceSlot]}`),
  );
  assert.ok(
    serializedDescriptor.includes(`Agent ID ${publicIds[semanticEvent.recipientSlot]}`),
  );
});

test("non-coexisting activation and health phases do not reserve each other's geometry", async () => {
  const { grammar } = await canonicalFrames();
  const events = /** @type {Record<string, any>[]} */ (grammar.event_batch.events);
  const ability = {
    ...events.find((event) => event.event_type === "ability_activated"),
    source_anchor: {
      ...events.find((event) => event.event_type === "ability_activated")
        ?.source_anchor,
      position: [150, 100],
    },
  };
  const health = {
    ...events.find((event) => event.event_type === "recipient_health_resolution"),
    recipient_anchor: {
      ...events.find((event) => event.event_type === "recipient_health_resolution")
        ?.recipient_anchor,
      position: [100, 100],
    },
  };
  const phaseSurface = /** @type {ProjectionSurface} */ ({
    worldToScreen: (point) => ({
      x: "x" in point ? point.x : Number(point[0]),
      y: "y" in point ? point.y : Number(point[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 200,
      bottom: 200,
      width: 200,
      height: 200,
    },
    protectedRects: [],
  });
  /** @param {Record<string, any>[]} selectedEvents */
  const planFor = (selectedEvents) =>
    buildChoreographyPlan(
      {
        ...grammar,
        event_batch: { ...grammar.event_batch, events: selectedEvents },
      },
      phaseSurface,
    );
  const healthOnly = planFor([health]);
  const withActivation = planFor([ability, health]);
  assert.ok(healthOnly);
  assert.ok(withActivation);
  assert.deepEqual(
    withActivation.events.find((event) => event.kind === "net_health")?.cue,
    healthOnly.events.find((event) => event.kind === "net_health")?.cue,
  );
});

test("POV planning consumes only recipient-local cue identities", async () => {
  const { pov } = await canonicalFrames();
  const plan = buildChoreographyPlan(pov, surface);
  assert.ok(plan);
  assert.equal(plan.transitionId, pov.incoming_pov_transition_id);
  const sourceCues = /** @type {Record<string, any>[]} */ (
    pov.projection.incoming_cues
  );
  assert.deepEqual(
    plan.events.map((event) => event.eventId),
    sourceCues.map((cue) => cue.cue_id),
  );
  assert.ok(plan.events.every((event) => event.eventId.includes(":actor-pov:")));
  assert.ok(
    plan.events.every(
      (event) => !Object.hasOwn(event, "sourceGlobalSlot") && !event.targetSlot,
    ),
  );
  const successorDeltas = plan.events.filter(
    (event) =>
      event.eventType === "own_position_changed" ||
      event.eventType === "own_health_changed",
  );
  assert.deepEqual(
    successorDeltas.map((event) => event.eventType),
    ["own_position_changed", "own_health_changed"],
  );
  const position = successorDeltas.find(
    (event) => event.eventType === "own_position_changed",
  );
  const health = successorDeltas.find(
    (event) => event.eventType === "own_health_changed",
  );
  assert.ok(position);
  assert.ok(health);
  assert.equal(position.spatial, false);
  assert.equal(position.presentationSuppressed, true);
  assert.equal(position.phaseStart, position.phaseEnd);
  assert.equal(health.spatial, true);
  assert.equal(health.phaseStart, 0);
  assert.equal(health.phaseEnd, plan.phases.total);
  assert.equal(plan.phases.total, 480);
});

test("POV self lifecycle edges receive only exact authorized transient meanings", async () => {
  const { pov } = await canonicalFrames();
  const transitionId = pov.event_batch.transition_id;
  /**
   * @param {string} label
   * @param {{
   *   startActive: boolean,
   *   successorActive: boolean,
   *   startAlive: boolean,
   *   successorAlive: boolean,
   *   startShield: number,
   *   successorShield: number,
   * }} lifecycle
   */
  const planFor = (label, lifecycle) => {
    const eventId = `${transitionId}:cue:self-lifecycle:${label}`;
    const event = {
      event_id: eventId,
      event_type: "own_lifecycle_changed",
      transition_id: transitionId,
      start_active: lifecycle.startActive,
      successor_active: lifecycle.successorActive,
      start_alive: lifecycle.startAlive,
      successor_alive: lifecycle.successorAlive,
      start_spawn_shield_remaining_ticks: lifecycle.startShield,
      successor_spawn_shield_remaining_ticks: lifecycle.successorShield,
    };
    const selfActor = {
      ...pov.scene.self_actor,
      alive: lifecycle.successorAlive,
      spawn_shield_remaining: lifecycle.successorShield,
    };
    const frame = {
      ...pov,
      scene: {
        ...pov.scene,
        self_actor: selfActor,
        agents: [
          {
            ...pov.scene.agents[0],
            alive: lifecycle.successorAlive,
            spawn_shield_remaining: lifecycle.successorShield,
          },
        ],
      },
      event_batch: { ...pov.event_batch, events: [event] },
    };
    const plan = buildChoreographyPlan(frame, surface);
    assert.ok(plan);
    assert.equal(plan.events.length, 1);
    assert.equal(plan.events[0]?.eventId, eventId);
    return { event: plan.events[0], plan };
  };

  const death = planFor("death", {
    startActive: true,
    successorActive: true,
    startAlive: true,
    successorAlive: false,
    startShield: 0,
    successorShield: 0,
  });
  assert.equal(death.event.kind, "semantic_pulse");
  assert.equal(death.event.cueSemantic, "agent_died");
  assert.deepEqual(death.event.anchor, { x: 30, y: 80 });
  assert.equal(death.event.agentSlot, pov.scene.self_actor.global_slot);
  assert.equal(death.event.agentPublicAgentId, pov.scene.self_actor.public_agent_id);
  assert.equal(death.event.phaseStart, 0);
  assert.equal(death.event.phaseEnd, 480);

  const respawn = planFor("respawn", {
    startActive: true,
    successorActive: true,
    startAlive: false,
    successorAlive: true,
    startShield: 0,
    successorShield: 3,
  });
  assert.equal(respawn.event.kind, "semantic_pulse");
  assert.equal(respawn.event.cueSemantic, "agent_respawned");
  assert.deepEqual(respawn.event.anchor, death.event.anchor);
  assert.equal(respawn.event.successorSpawnShieldRemaining, 3);
  assert.equal(respawn.event.phaseStart, 0);
  assert.equal(respawn.event.phaseEnd, 620);

  const shieldExpiry = planFor("shield-expiry", {
    startActive: true,
    successorActive: true,
    startAlive: true,
    successorAlive: true,
    startShield: 1,
    successorShield: 0,
  });
  assert.equal(shieldExpiry.event.kind, "semantic_pulse");
  assert.equal(shieldExpiry.event.cueSemantic, "spawn_shield_expired");
  assert.deepEqual(shieldExpiry.event.anchor, death.event.anchor);
  assert.equal(shieldExpiry.event.phaseStart, 0);
  assert.equal(shieldExpiry.event.phaseEnd, 480);

  const countdown = planFor("shield-countdown", {
    startActive: true,
    successorActive: true,
    startAlive: true,
    successorAlive: true,
    startShield: 3,
    successorShield: 2,
  });
  assert.equal(countdown.event.kind, "feed_only");
  assert.equal(countdown.event.spatial, false);
  assert.equal(Object.hasOwn(countdown.event, "cueSemantic"), false);
  assert.equal(countdown.plan.phases.total, 0);

  const missingSelf = {
    ...pov,
    scene: {
      ...pov.scene,
      self_actor: null,
    },
    event_batch: {
      ...pov.event_batch,
      events: [
        {
          event_id: `${transitionId}:cue:self-lifecycle:missing-self`,
          event_type: "own_lifecycle_changed",
          transition_id: transitionId,
          start_active: true,
          successor_active: true,
          start_alive: true,
          successor_alive: false,
          start_spawn_shield_remaining_ticks: 0,
          successor_spawn_shield_remaining_ticks: 0,
        },
      ],
    },
  };
  const missingSelfPlan = buildChoreographyPlan(missingSelf, surface);
  assert.ok(missingSelfPlan);
  assert.equal(missingSelfPlan.events[0]?.kind, "unknown");
  assert.equal(missingSelfPlan.events[0]?.spatial, false);
});

test("numeric transition identities fail closed", async () => {
  const { grammar } = await canonicalFrames();
  assert.equal(
    buildChoreographyPlan(
      {
        ...grammar,
        incoming_transition_id: 1,
        transition_id: 1,
        event_batch: { ...grammar.event_batch, transition_id: 1 },
      },
      surface,
    ),
    null,
  );
});
