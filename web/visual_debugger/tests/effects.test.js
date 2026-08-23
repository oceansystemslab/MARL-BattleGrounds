import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { explainChoreographyEvent } from "../src/choreography-painter.js";
import {
  buildChoreographyPlan,
  CHOREOGRAPHY_PAINT_FOOTPRINTS,
  isSubmissionCommand,
} from "../src/choreography-plan.js";
import { BATTLEFIELD_LAYER_ORDER } from "../src/scene.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  setVisualFilterEnabled,
  VISUAL_FILTER_IDS,
  visualFilterPaintKey,
} from "../src/visual-filters.js";

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
    right: 800,
    bottom: 600,
    width: 800,
    height: 600,
  },
  protectedRects: [],
};

/** @type {Promise<Record<string, any>> | undefined} */
let authorizedFixturePromise;

/** @returns {Promise<Record<string, any>>} */
async function authorizedFixture() {
  if (!authorizedFixturePromise) {
    authorizedFixturePromise = readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ).then(JSON.parse);
  }
  return authorizedFixturePromise;
}

/** @param {...string} filterIds */
function filtersDisabled(...filterIds) {
  return filterIds.reduce(
    (state, filterId) => setVisualFilterEnabled(state, filterId, false),
    DEFAULT_VISUAL_FILTER_STATE,
  );
}

test("strict authorized fixture preserves exact choreography identity", async () => {
  const fixture = await authorizedFixture();
  const rawPresentation = fixture.presentations.replay_oracle;
  const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
  const serializedBefore = JSON.stringify(presentation);
  const plan = buildChoreographyPlan(presentation, surface);
  assert.ok(plan);
  assert.deepEqual(buildChoreographyPlan(presentation, surface), plan);
  assert.equal(plan.transitionId, rawPresentation.latest_events.incoming_transition_id);
  assert.deepEqual(
    plan.events.flatMap((event) => event.atomicEventIds ?? [event.eventId]),
    rawPresentation.latest_events.ordered_event_ids,
  );
  assert.equal(JSON.stringify(presentation), serializedBefore);
  assert.equal(
    plan.events.some(
      (event) =>
        event.cueSuppressionReason ||
        event.spatialDisposition === "suppressed_collision",
    ),
    false,
  );
  for (const event of plan.events.filter((candidate) => candidate.spatial)) {
    if (event.kind === "activation") {
      assert.ok(event.route?.end ?? event.target ?? event.sourceCue, event.eventId);
      if (event.target) {
        assert.equal(event.impactCue ?? null, null, event.eventId);
        assert.equal(event.impactLeader ?? null, null, event.eventId);
        assert.equal(event.impactBounds ?? null, null, event.eventId);
        assert.equal(event.impactLayoutKey ?? null, null, event.eventId);
        assert.equal(event.impactDisposition ?? null, null, event.eventId);
        assert.equal(event.impactCueCollisionFree ?? null, null, event.eventId);
      }
      if (event.paintParts.ability && event.target) {
        assert.ok(event.route, event.eventId);
        assert.ok(Array.isArray(event.route.bridgeGaps), event.eventId);
      }
    } else {
      assert.ok(event.cue, event.eventId);
      assert.equal(event.cueCollisionFree, true, event.eventId);
      assert.ok(event.cueBounds && event.cueLeader, event.eventId);
    }
  }
  assert.equal(buildChoreographyPlan(rawPresentation, surface), null);
  assert.equal(buildChoreographyPlan({ ...presentation }, surface), null);
  assert.equal(
    buildChoreographyPlan(fixture.pairs.live_oracle.transport, surface),
    null,
  );
  assert.equal(buildChoreographyPlan(null, surface), null);
  assert.equal(buildChoreographyPlan({}, surface), null);
});

test("visual filters preserve scientific identity and retain suppressed atomics before scheduling", async () => {
  const fixture = await authorizedFixture();
  const raw = fixture.presentations.replay_oracle;
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const allOn = buildChoreographyPlan(frame, surface);
  const allOffState = VISUAL_FILTER_IDS.reduce(
    (state, filterId) => setVisualFilterEnabled(state, filterId, false),
    DEFAULT_VISUAL_FILTER_STATE,
  );
  const allOff = buildChoreographyPlan(frame, surface, allOffState);
  assert.ok(allOn);
  assert.ok(allOff);
  assert.equal(allOn.paintKey, visualFilterPaintKey(DEFAULT_VISUAL_FILTER_STATE));
  assert.equal(allOff.paintKey, visualFilterPaintKey(allOffState));
  assert.notEqual(allOff.paintKey, allOn.paintKey);
  assert.deepEqual(
    [allOff.epochKey, allOff.authorizationKey, allOff.fingerprint],
    [allOn.epochKey, allOn.authorizationKey, allOn.fingerprint],
  );
  assert.deepEqual(
    allOff.events.flatMap((event) => event.atomicEventIds ?? [event.eventId]),
    raw.latest_events.ordered_event_ids,
  );
  assert.equal(
    allOff.events
      .filter((event) => event.paintParts)
      .every(
        (event) => event.presentationSuppressed === true && event.spatial === false,
      ),
    true,
  );
  assert.equal(allOff.phases.total, 0);
  assert.equal(JSON.stringify(frame), serializedBefore);
});

test("all registered transient families validate without constructing disabled geometry", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  /** @type {Readonly<Record<string, number>>} */
  const phaseRank = {
    action_rejected: 10,
    ability_activated: 20,
    recipient_health_resolution: 40,
    health_regenerated: 50,
    cooldown_started: 60,
    cooldown_ready: 60,
    charge_phase_displacement: 70,
    agent_died: 90,
    status_aged_to_zero: 100,
    status_broken_by_damage: 100,
    status_applied: 100,
    status_refreshed_or_extended: 100,
    status_cleared_by_new_death: 100,
    spawn_shield_expired: 110,
    respawn_wave_occurred: 120,
    agent_respawned: 120,
  };
  const statusIds = [
    "warrior_charge_slow",
    "hunter_basic_slow",
    "rogue_poison_slow",
    "warrior_charge_stun",
    "hunter_trap_stun",
    "rogue_poison_stun",
    "rogue_poison_anti_heal",
  ];
  /** @param {number} classId */
  const trajectoryForClass = (classId) => {
    const agent = raw.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ candidate) => candidate.class_id === classId,
    );
    assert.ok(agent);
    const trajectory = raw.latest_events.agent_phase_trajectories.find(
      (/** @type {Record<string, any>} */ candidate) =>
        candidate.agent_presentation_key === agent.presentation_key,
    );
    assert.ok(trajectory);
    return trajectory;
  };
  /** @param {number} classId @param {string} phase */
  const anchor = (classId, phase) =>
    structuredClone(trajectoryForClass(classId)[phase]);
  /**
   * @param {string} eventKind
   * @param {number} channel
   * @param {number} [recipientClass]
   */
  const statusEvent = (eventKind, channel, recipientClass = 2) => ({
    event_kind: eventKind,
    recipient_anchor: anchor(recipientClass, "successor"),
    status_channel: channel,
    status_id: statusIds[channel],
    ...(eventKind === "status_applied"
      ? { source_anchor: anchor(3, "successor") }
      : {}),
  });
  const rejectedActor = anchor(1, "transition_start");
  const rejectedAction = {
    move_action: 0,
    target_action: 99,
    use_ultimate_action: 0,
  };
  const rejectedActionRow = raw.latest_transition.action_rows.find(
    (/** @type {Record<string, any>} */ row) =>
      row.actor_public_agent_id === rejectedActor.public_agent_id,
  );
  assert.ok(rejectedActionRow);
  rejectedActionRow.submitted_action = structuredClone(rejectedAction);
  const chargeTrajectory = trajectoryForClass(2);
  const respawnAnchor = anchor(3, "successor");
  const events = [
    {
      event_kind: "action_rejected",
      actor_anchor: rejectedActor,
      actor_configured_active: true,
      actor_identity: {
        identity_kind: "authorized_agent",
        presentation_key: rejectedActor.presentation_key,
        public_agent_id: rejectedActor.public_agent_id,
      },
      rejection_component: "domain",
      submitted_action: structuredClone(rejectedAction),
    },
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: anchor(5, "transition_start"),
      recipient_anchor: anchor(2, "transition_start"),
    },
    {
      event_kind: "ability_activated",
      ability_component: "ultimate",
      source_anchor: anchor(3, "transition_start"),
      recipient_anchor: anchor(1, "transition_start"),
    },
    {
      event_kind: "recipient_health_resolution",
      recipient_anchor: anchor(1, "transition_start"),
      transition_start_health: 100,
      total_effective_damage: 7,
      total_effective_healing: 0,
      health_after_combat_resolution: 93,
      realized_net_health_change: -7,
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: anchor(4, "transition_start"),
      actual_health_regenerated: 3,
    },
    {
      event_kind: "cooldown_started",
      agent_anchor: anchor(1, "transition_start"),
    },
    {
      event_kind: "cooldown_ready",
      agent_anchor: anchor(5, "transition_start"),
    },
    {
      event_kind: "charge_phase_displacement",
      realized_displacement: [
        chargeTrajectory.post_charge.position[0] -
          chargeTrajectory.transition_start.position[0],
        chargeTrajectory.post_charge.position[1] -
          chargeTrajectory.transition_start.position[1],
      ],
      start_anchor: anchor(2, "transition_start"),
      end_anchor: anchor(2, "post_charge"),
    },
    {
      event_kind: "agent_died",
      recipient_anchor: anchor(4, "successor"),
    },
    statusEvent("status_applied", 0),
    statusEvent("status_refreshed_or_extended", 1),
    statusEvent("status_aged_to_zero", 2),
    statusEvent("status_broken_by_damage", 3),
    statusEvent("status_broken_by_damage", 4),
    statusEvent("status_applied", 4),
    statusEvent("status_cleared_by_new_death", 5),
    {
      event_kind: "spawn_shield_expired",
      agent_anchor: anchor(1, "successor"),
    },
    {
      event_kind: "respawn_wave_occurred",
      team_anchor: { phase: "successor", team_index: 0, team_id: 1 },
    },
    {
      event_kind: "agent_respawned",
      agent_anchor: respawnAnchor,
      team_id: 2,
      realized_successor_position: structuredClone(respawnAnchor.position),
    },
  ];
  const transitionId = raw.latest_events.incoming_transition_id;
  raw.latest_events.events = events.map((event, ordinal) => ({
    ...event,
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
    phase_rank: phaseRank[event.event_kind],
  }));
  raw.latest_events.event_count = raw.latest_events.events.length;
  raw.latest_events.ordered_event_ids = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  raw.latest_events.ordered_event_kinds = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const allOffState = VISUAL_FILTER_IDS.reduce(
    (state, filterId) => setVisualFilterEnabled(state, filterId, false),
    DEFAULT_VISUAL_FILTER_STATE,
  );
  const forbiddenSurface = new Proxy(
    {},
    {
      get(_target, property) {
        throw new Error(`disabled choreography read surface.${String(property)}`);
      },
    },
  );
  const plan = buildChoreographyPlan(
    frame,
    /** @type {any} */ (forbiddenSurface),
    allOffState,
  );
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(plan.paintKey, visualFilterPaintKey(allOffState));
  assert.deepEqual(
    plan.events.flatMap((event) => event.atomicEventIds ?? [event.eventId]),
    raw.latest_events.ordered_event_ids,
  );
  assert.equal(
    plan.events.every(
      (event) =>
        event.paintParts &&
        Object.values(event.paintParts).every((enabled) => enabled === false) &&
        event.presentationSuppressed === true &&
        event.spatial === false,
    ),
    true,
  );
  for (const event of plan.events) {
    for (const geometryKey of [
      "source",
      "target",
      "recipient",
      "anchor",
      "actor",
      "start",
      "end",
      "cue",
      "cueBounds",
      "route",
    ]) {
      assert.equal(
        event[geometryKey] ?? null,
        null,
        `${event.eventType}:${geometryKey}`,
      );
    }
  }
  const activationParts = plan.events
    .filter((event) => event.kind === "activation")
    .map(({ component, impactSemantic, paintParts }) => ({
      component,
      impactSemantic,
      paintParts,
    }));
  assert.deepEqual(activationParts, [
    {
      component: "basic",
      impactSemantic: "healing",
      paintParts: { ability: false, semantic: false },
    },
    {
      component: "ultimate",
      impactSemantic: "damage",
      paintParts: { ability: false, semantic: false },
    },
  ]);
  assert.deepEqual(
    plan.events
      .filter((event) => event.kind === "status_lifecycle")
      .map(({ lifecycle }) => lifecycle),
    [
      "applied",
      "refreshed",
      "expired",
      "trap_broken",
      "trap_broken_and_reapplied",
      "cleared_by_death",
    ],
  );
  assert.deepEqual(
    plan.events
      .filter((event) => event.kind === "semantic_pulse")
      .map(({ cueSemantic }) => cueSemantic),
    [
      "cooldown_started",
      "cooldown_ready",
      "agent_died",
      "spawn_shield_expired",
      "respawn_wave_occurred",
      "agent_respawned",
    ],
  );
  const wave = plan.events.find(
    (event) => event.cueSemantic === "respawn_wave_occurred",
  );
  assert.deepEqual(
    wave && {
      teamIndex: wave.teamIndex,
      teamId: wave.teamId,
      teamSide: wave.teamSide,
      label: wave.label,
    },
    {
      teamIndex: 0,
      teamId: 1,
      teamSide: "left",
      label: "EVENT: Team A Respawn",
    },
  );
  assert.equal(
    plan.events.some((event) => event.kind === "rejected_action"),
    true,
  );
  assert.equal(
    plan.events.some((event) => event.kind === "net_health"),
    true,
  );
  assert.equal(
    plan.events.some((event) => event.kind === "regeneration"),
    true,
  );
  assert.equal(
    plan.events.some((event) => event.kind === "charge_displacement"),
    true,
  );
  assert.equal(plan.phases.total, 0);
  assert.equal(plan.bounds.persistentNodes, 0);
});

test("activation ability and semantic paint parts are independently filterable", async () => {
  const fixture = await authorizedFixture();
  const frame = await normalizeAuthorizedPresentationFrameV1(
    fixture.presentations.replay_oracle,
  );
  const cases = [
    {
      disabled: [],
      expected: { ability: true, semantic: true },
      suppressed: false,
    },
    {
      disabled: ["basic_ability_effects"],
      expected: { ability: false, semantic: true },
      suppressed: false,
    },
    {
      disabled: ["healing_effects"],
      expected: { ability: true, semantic: false },
      suppressed: false,
    },
    {
      disabled: ["basic_ability_effects", "healing_effects"],
      expected: { ability: false, semantic: false },
      suppressed: true,
    },
  ];
  for (const { disabled, expected, suppressed } of cases) {
    const plan = buildChoreographyPlan(frame, surface, filtersDisabled(...disabled));
    assert.ok(plan);
    const activation = plan.events.find((event) => event.kind === "activation");
    assert.ok(activation);
    assert.equal(activation.component, "basic");
    assert.equal(activation.impactSemantic, "healing");
    assert.deepEqual(activation.paintParts, expected);
    assert.equal(activation.presentationSuppressed, suppressed);
    assert.equal(activation.spatial, !suppressed);
    if (disabled.includes("basic_ability_effects")) {
      assert.equal(activation.route, null);
      assert.equal(activation.source, null);
      if (suppressed) {
        assert.equal(activation.target, null);
      } else {
        assert.ok(activation.target);
      }
      assert.equal(
        activation.presentationKind,
        suppressed ? "source_local" : "target_only_impact",
      );
    } else {
      assert.ok(activation.route);
      assert.ok(activation.source);
      assert.equal(activation.presentationKind, "routed");
    }
  }

  const rawDamage = structuredClone(fixture.presentations.replay_oracle);
  const activationEvent = rawDamage.latest_events.events.find(
    (/** @type {Record<string, any>} */ event) =>
      event.event_kind === "ability_activated",
  );
  assert.ok(activationEvent);
  /** @param {number} classId */
  const anchorForClass = (classId) => {
    const agent = rawDamage.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ candidate) => candidate.class_id === classId,
    );
    assert.ok(agent);
    const trajectory = rawDamage.latest_events.agent_phase_trajectories.find(
      (/** @type {Record<string, any>} */ candidate) =>
        candidate.agent_presentation_key === agent.presentation_key,
    );
    assert.ok(trajectory);
    return structuredClone(trajectory.transition_start);
  };
  activationEvent.source_anchor = anchorForClass(1);
  activationEvent.recipient_anchor = anchorForClass(3);
  const damageFrame = await normalizeAuthorizedPresentationFrameV1(rawDamage);
  const damagePlan = buildChoreographyPlan(
    damageFrame,
    surface,
    filtersDisabled("damage_effects"),
  );
  assert.ok(damagePlan);
  const damageActivation = damagePlan.events.find(
    (event) => event.kind === "activation",
  );
  assert.ok(damageActivation);
  assert.equal(damageActivation.impactSemantic, "damage");
  assert.deepEqual(damageActivation.paintParts, {
    ability: true,
    semantic: false,
  });

  const activationOnlyRaw = structuredClone(fixture.presentations.replay_oracle);
  const activationOnly = activationOnlyRaw.latest_events.events.filter(
    (/** @type {Record<string, any>} */ event) =>
      event.event_kind === "ability_activated",
  );
  activationOnlyRaw.latest_events.events = activationOnly;
  activationOnlyRaw.latest_events.event_count = activationOnly.length;
  activationOnlyRaw.latest_events.ordered_event_ids = activationOnly.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  activationOnlyRaw.latest_events.ordered_event_kinds = activationOnly.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );
  const activationOnlyFrame =
    await normalizeAuthorizedPresentationFrameV1(activationOnlyRaw);
  /** @param {...string} disabled */
  const projectedActivation = (...disabled) => {
    let pointProjections = 0;
    let lengthProjections = 0;
    const countingSurface = {
      ...surface,
      worldToScreen(
        /** @type {readonly [number, number] | {x: number, y: number}} */ point,
      ) {
        pointProjections += 1;
        return surface.worldToScreen(point);
      },
      worldLengthToScreen(/** @type {number} */ length) {
        lengthProjections += 1;
        return surface.worldLengthToScreen(length);
      },
    };
    const plan = buildChoreographyPlan(
      activationOnlyFrame,
      countingSurface,
      filtersDisabled(...disabled),
    );
    assert.ok(plan);
    return { event: plan.events[0], pointProjections, lengthProjections };
  };
  const semanticOnly = projectedActivation("basic_ability_effects");
  assert.deepEqual(
    [semanticOnly.pointProjections, semanticOnly.lengthProjections],
    [1, 0],
  );
  assert.equal(semanticOnly.event.presentationKind, "target_only_impact");
  assert.equal(semanticOnly.event.source, null);
  assert.equal(semanticOnly.event.route, null);
  assert.ok(semanticOnly.event.target);
  assert.equal(semanticOnly.event.impactCue ?? null, null);
  assert.equal(semanticOnly.event.impactBounds ?? null, null);
  assert.equal(semanticOnly.event.impactLeader ?? null, null);
  assert.equal(semanticOnly.event.impactLayoutKey ?? null, null);
  assert.equal(semanticOnly.event.impactDisposition ?? null, null);
  assert.equal(semanticOnly.event.impactCueCollisionFree ?? null, null);

  const routeOnly = projectedActivation("healing_effects");
  assert.deepEqual([routeOnly.pointProjections, routeOnly.lengthProjections], [2, 2]);
  assert.equal(routeOnly.event.presentationKind, "routed");
  assert.ok(routeOnly.event.source);
  assert.ok(routeOnly.event.target);
  assert.ok(routeOnly.event.route);
  assert.equal(routeOnly.event.impactCue ?? null, null);
  assert.equal(routeOnly.event.impactBounds ?? null, null);
  assert.equal(routeOnly.event.impactLeader ?? null, null);
  assert.equal(routeOnly.event.impactLayoutKey ?? null, null);
  assert.equal(routeOnly.event.impactDisposition ?? null, null);
  assert.equal(routeOnly.event.impactCueCollisionFree ?? null, null);
  assert.ok(Array.isArray(routeOnly.event.route.bridgeGaps));

  const fullySuppressed = projectedActivation(
    "basic_ability_effects",
    "healing_effects",
  );
  assert.deepEqual(
    [fullySuppressed.pointProjections, fullySuppressed.lengthProjections],
    [0, 0],
  );
  assert.equal(fullySuppressed.event.source, null);
  assert.equal(fullySuppressed.event.target, null);
  assert.equal(fullySuppressed.event.route, null);
  assert.equal(fullySuppressed.event.impactCue ?? null, null);
});

test("transition identity excludes revision while authorization tracks POV actor", async () => {
  const fixture = await authorizedFixture();
  const rawOracle = fixture.presentations.replay_oracle;
  const rawPov = fixture.presentations.replay_no_shared_obs_agent_pov;
  const revisedRawOracle = structuredClone(rawOracle);
  revisedRawOracle.source.source_revision += 1;
  revisedRawOracle.source.source_authority_epoch += 1;
  const oracle = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(rawOracle),
    surface,
  );
  const revised = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(revisedRawOracle),
    surface,
  );
  const pov = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(rawPov),
    surface,
  );
  assert.ok(oracle);
  assert.ok(revised);
  assert.ok(pov);
  assert.equal(revised.epochKey, oracle.epochKey);
  assert.equal(revised.fingerprint, oracle.fingerprint);
  assert.notEqual(revised.authorizationKey, oracle.authorizationKey);
  assert.deepEqual(JSON.parse(oracle.authorizationKey), [
    rawOracle.source.source_session_id,
    rawOracle.source.source_authority_epoch,
    rawOracle.presentation_kind,
    null,
  ]);
  assert.deepEqual(JSON.parse(pov.authorizationKey), [
    rawPov.source.source_session_id,
    rawPov.source.source_authority_epoch,
    rawPov.presentation_kind,
    rawPov.authority.recipient_presentation_key,
  ]);
  const rejectedDiagnosticReplay = structuredClone(rawPov);
  rejectedDiagnosticReplay.authority.projection_basis = "shared_obs_source_material";
  assert.equal(buildChoreographyPlan(rejectedDiagnosticReplay, surface), null);
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(rejectedDiagnosticReplay),
  );
  assert.notEqual(oracle.authorizationKey, pov.authorizationKey);
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

test("transient activation routes remain beneath every information-bearing layer", () => {
  const routeIndex = BATTLEFIELD_LAYER_ORDER.indexOf("transient-route");
  assert.ok(routeIndex >= 0);
  for (const foregroundLayer of [
    "obstacle",
    "body",
    "selection-legality",
    "transient-events",
    "durable-status-modifier",
    "accessible-labels",
  ]) {
    assert.ok(
      BATTLEFIELD_LAYER_ORDER.indexOf(foregroundLayer) > routeIndex,
      `${foregroundLayer} must paint over transient activation routes`,
    );
  }
});

test("activation arrows prefer full paint before an explicit compact fallback", async () => {
  const [painter, css] = await Promise.all([
    readFile(new URL("../src/choreography-painter.js", import.meta.url), "utf8"),
    readFile(new URL("../styles.css", import.meta.url), "utf8"),
  ]);
  const activationArrowRule = css.match(/\.combat-route__arrow\s*\{[^}]*\}/u)?.[0];
  const compactArrowRule = css.match(
    /\.combat-route__arrow\[data-marker-variant="compact"\]\s*\{[^}]*\}/u,
  )?.[0];
  const chargeArrowRule = css.match(/\.combat-charge__direction\s*\{[^}]*\}/u)?.[0];
  const activationParticleRule = css.match(
    /\.combat-route__particle\s*\{[^}]*\}/u,
  )?.[0];
  const holyParticleRule = css.match(
    /\.combat-route-effect\[data-token-id="holy_word"\] \.combat-route__particle\s*\{[^}]*\}/u,
  )?.[0];

  assert.equal(CHOREOGRAPHY_PAINT_FOOTPRINTS.route.activationMarkerPadding, 17);
  assert.equal(CHOREOGRAPHY_PAINT_FOOTPRINTS.route.activationCompactMarkerPadding, 8);
  assert.equal(CHOREOGRAPHY_PAINT_FOOTPRINTS.route.chargeMarkerPadding, 14);
  assert.equal(CHOREOGRAPHY_PAINT_FOOTPRINTS.route.activationPathPadding.default, 7);
  assert.equal(CHOREOGRAPHY_PAINT_FOOTPRINTS.route.activationPathPadding.holy_word, 8);
  assert.match(painter, /d: "M -11 -6 L 2 0 L -11 6 L -7 0 Z"/u);
  assert.match(painter, /M -6 -3 L 2 0 L -6 3 L -4 0 Z/u);
  assert.ok(activationArrowRule);
  assert.match(activationArrowRule, /drop-shadow\(0 0 3px currentColor\)/u);
  assert.ok(compactArrowRule);
  assert.match(compactArrowRule, /filter: none/u);
  assert.ok(chargeArrowRule);
  assert.match(chargeArrowRule, /drop-shadow\(0 0 3px currentColor\)/u);
  assert.ok(activationParticleRule);
  assert.match(activationParticleRule, /drop-shadow\(0 0 3px currentColor\)/u);
  assert.ok(holyParticleRule);
  assert.match(holyParticleRule, /drop-shadow\(0 0 4px currentColor\)/u);
});

test("scene has one Analysis battlefield branch and owns durable shield and status hooks", async () => {
  const [source, css] = await Promise.all([
    readFile(new URL("../src/scene.js", import.meta.url), "utf8"),
    readFile(new URL("../styles.css", import.meta.url), "utf8"),
  ]);
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
  assert.match(source, /duration_status_badge/u);
  assert.doesNotMatch(source, /agent-combat-state-icon/u);
  assert.doesNotMatch(source, /showCombatStatusIcon/u);
  assert.doesNotMatch(source, /combat status in combat/u);
  assert.doesNotMatch(source, /combat status out of combat/u);
  assert.match(css, /\.status-cell\[data-token-id="in_combat"\]/u);
  assert.match(css, /\.pov-observed-status\[data-token-id="in_combat"\]/u);
  assert.match(css, /\.combat-effect\[data-token-id="in_combat"\]/u);
  assert.match(source, /createSpawnShieldView\(agent, spawnShieldMechanics\)/u);
  assert.match(source, /layoutKey: JSON\.stringify\(/u);
  assert.match(source, /bounds: frozenBounds/u);
  assert.match(source, /protectedKind/u);
  assert.match(source, /ownerPresentationKey/u);
  assert.match(
    source,
    /registerTooltipOwner\(nodes\.shieldRoot, spawnShieldView\.descriptor\)/u,
  );
});

test("authorized regeneration owns a packed plus cue instead of the generic onion", async () => {
  const [planSource, painterSource, css] = await Promise.all([
    readFile(new URL("../src/choreography-plan.js", import.meta.url), "utf8"),
    readFile(new URL("../src/choreography-painter.js", import.meta.url), "utf8"),
    readFile(new URL("../styles.css", import.meta.url), "utf8"),
  ]);
  assert.match(
    planSource,
    /row\.kind === "health_regenerated" \? "regeneration" : "semantic_pulse"/u,
  );
  assert.match(
    planSource,
    /layoutCrossPhaseEvents\(paintFiltered, surface, sceneByKey\)/u,
  );
  assert.match(painterSource, /combat-regeneration__plus/u);
  assert.match(painterSource, /combat-regeneration__value/u);
  assert.doesNotMatch(painterSource, /event\.cueSemantic === "health_regenerated"/u);
  assert.match(css, /\.combat-regeneration__pulse/u);
  assert.match(css, /\.combat-regeneration__recipient-anchor/u);
  assert.doesNotMatch(css, /\.combat-semantic-pulse--health-regenerated/u);
});

test("NET and regeneration effects remain independent from scrolling battle text", async () => {
  const fixture = await authorizedFixture();
  const rawNet = structuredClone(fixture.presentations.replay_oracle);
  const netEvent = rawNet.latest_events.events.find(
    (/** @type {Record<string, any>} */ event) =>
      event.event_kind === "recipient_health_resolution",
  );
  assert.ok(netEvent);
  netEvent.transition_start_health = 200;
  netEvent.total_effective_damage = 10;
  netEvent.total_effective_healing = 0;
  netEvent.health_after_combat_resolution = 190;
  netEvent.realized_net_health_change = -10;
  const netFrame = await normalizeAuthorizedPresentationFrameV1(rawNet);
  /**
   * @type {ReadonlyArray<readonly [
   *   string[],
   *   {effect: boolean, battleText: boolean, recipientText: boolean},
   *   boolean,
   * ]>}
   */
  const netCases = [
    [[], { effect: true, battleText: true, recipientText: true }, false],
    [
      ["scrolling_battle_text"],
      { effect: true, battleText: false, recipientText: false },
      false,
    ],
    [
      ["damage_effects"],
      { effect: false, battleText: true, recipientText: true },
      false,
    ],
    [
      ["damage_effects", "scrolling_battle_text"],
      { effect: false, battleText: false, recipientText: false },
      true,
    ],
  ];
  for (const [disabled, expected, suppressed] of netCases) {
    const plan = buildChoreographyPlan(netFrame, surface, filtersDisabled(...disabled));
    assert.ok(plan);
    const net = plan.events.find((event) => event.kind === "net_health");
    assert.ok(net);
    assert.deepEqual(net.paintParts, expected);
    assert.equal(net.presentationSuppressed, suppressed);
    if (disabled.length === 0) {
      assert.deepEqual(
        { width: net.cueBounds?.width, height: net.cueBounds?.height },
        { width: 88, height: 36 },
      );
    }
    if (disabled.length === 1 && disabled[0] === "scrolling_battle_text") {
      assert.deepEqual(
        { width: net.cueBounds?.width, height: net.cueBounds?.height },
        { width: 48, height: 36 },
      );
    }
    if (disabled.length === 1 && disabled[0] === "damage_effects") {
      assert.deepEqual(
        { width: net.cueBounds?.width, height: net.cueBounds?.height },
        { width: 88, height: 36 },
      );
    }
  }

  const rawRegeneration = structuredClone(fixture.presentations.replay_oracle);
  const [trajectory] = rawRegeneration.latest_events.agent_phase_trajectories;
  assert.ok(trajectory);
  const transitionId = rawRegeneration.latest_events.incoming_transition_id;
  const regenerationEvent = {
    event_kind: "health_regenerated",
    event_id: `${transitionId}:event:0000`,
    ordinal: 0,
    phase_rank: 50,
    agent_anchor: structuredClone(trajectory.transition_start),
    actual_health_regenerated: 3,
  };
  rawRegeneration.latest_events.events = [regenerationEvent];
  rawRegeneration.latest_events.event_count = 1;
  rawRegeneration.latest_events.ordered_event_ids = [regenerationEvent.event_id];
  rawRegeneration.latest_events.ordered_event_kinds = [regenerationEvent.event_kind];
  const regenerationFrame =
    await normalizeAuthorizedPresentationFrameV1(rawRegeneration);
  /**
   * @type {ReadonlyArray<readonly [
   *   string[],
   *   {effect: boolean, battleText: boolean},
   *   boolean,
   * ]>}
   */
  const regenerationCases = [
    [[], { effect: true, battleText: true }, false],
    [["scrolling_battle_text"], { effect: true, battleText: false }, false],
    [["regeneration_effects"], { effect: false, battleText: true }, false],
    [
      ["regeneration_effects", "scrolling_battle_text"],
      { effect: false, battleText: false },
      true,
    ],
  ];
  for (const [disabled, expected, suppressed] of regenerationCases) {
    const plan = buildChoreographyPlan(
      regenerationFrame,
      surface,
      filtersDisabled(...disabled),
    );
    assert.ok(plan);
    const regeneration = plan.events.find((event) => event.kind === "regeneration");
    assert.ok(regeneration);
    assert.deepEqual(regeneration.paintParts, expected);
    assert.equal(regeneration.presentationSuppressed, suppressed);
    if (disabled.length === 0) {
      assert.deepEqual(
        {
          width: Math.round(regeneration.cueBounds?.width ?? 0),
          height: Math.round(regeneration.cueBounds?.height ?? 0),
        },
        { width: 64, height: 68 },
      );
    }
    if (disabled.length === 1 && disabled[0] === "scrolling_battle_text") {
      assert.deepEqual(
        {
          width: Math.round(regeneration.cueBounds?.width ?? 0),
          height: Math.round(regeneration.cueBounds?.height ?? 0),
        },
        { width: 48, height: 48 },
      );
    }
    if (disabled.length === 1 && disabled[0] === "regeneration_effects") {
      assert.deepEqual(
        {
          width: Math.round(regeneration.cueBounds?.width ?? 0),
          height: Math.round(regeneration.cueBounds?.height ?? 0),
        },
        { width: 64, height: 68 },
      );
    }
    if (suppressed) {
      assert.equal(plan.phases.total, 0);
    }
  }
});

test("authorized age plus application is one Reapplied plan row without changing atomics", async () => {
  const fixture = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const applied = structuredClone(
    raw.latest_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "status_applied",
    ),
  );
  assert.ok(applied);
  const aged = structuredClone(applied);
  aged.event_kind = "status_aged_to_zero";
  delete aged.source_anchor;
  const events = [aged, applied].map((event, ordinal) => ({
    ...event,
    event_id: `${raw.latest_events.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
    phase_rank: 100,
  }));
  raw.latest_events.events = events;
  raw.latest_events.event_count = events.length;
  raw.latest_events.ordered_event_ids = events.map(({ event_id }) => event_id);
  raw.latest_events.ordered_event_kinds = events.map(({ event_kind }) => event_kind);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const incomingEventsBefore = JSON.stringify(frame.latest_events.events);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(JSON.stringify(frame.latest_events.events), incomingEventsBefore);
  assert.equal(plan.events.length, 1);
  assert.equal(plan.bounds.nodes, 30);
  assert.deepEqual(
    plan.events.map(({ eventId, eventType }) => [eventId, eventType]),
    [[events[1].event_id, events[1].event_kind]],
  );
  const [primaryApplication] = plan.events;
  assert.deepEqual(
    primaryApplication.atomicEventIds,
    events.map(({ event_id }) => event_id),
  );
  assert.deepEqual(primaryApplication.applicationEventIds, [events[1].event_id]);
  assert.equal(primaryApplication.presentationSuppressed, false);
  assert.equal(primaryApplication.spatial, true);
  assert.equal(primaryApplication.lifecycle, "reapplied");
  assert.equal(primaryApplication.lifecycleToken.label, "Reapplied");
  assert.doesNotMatch(
    `${primaryApplication.lifecycleToken.accessibleName} ${primaryApplication.lifecycleToken.glyphKey}`,
    /expir/iu,
  );
});

test("authorized break plus application is one Reapplied plan row without changing atomics", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const applied = structuredClone(
    raw.latest_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "status_applied",
    ),
  );
  assert.ok(applied);
  const broken = structuredClone(applied);
  broken.event_kind = "status_broken_by_damage";
  delete broken.source_anchor;
  const events = [broken, applied].map((event, ordinal) => ({
    ...event,
    event_id: `${raw.latest_events.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
    phase_rank: 100,
  }));
  raw.latest_events.events = events;
  raw.latest_events.event_count = events.length;
  raw.latest_events.ordered_event_ids = events.map(({ event_id }) => event_id);
  raw.latest_events.ordered_event_kinds = events.map(({ event_kind }) => event_kind);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(plan.events.length, 1);
  assert.deepEqual(
    plan.events[0]?.atomicEventIds,
    events.map(({ event_id }) => event_id),
  );
  assert.deepEqual(plan.events[0]?.applicationEventIds, [events[1].event_id]);
  assert.equal(plan.events[0]?.lifecycle, "trap_broken_and_reapplied");
  assert.equal(plan.events[0]?.lifecycleToken.label, "Broken, then reapplied");
  assert.deepEqual(
    {
      width: Math.round(plan.events[0]?.cueBounds?.width ?? 0),
      height: Math.round(plan.events[0]?.cueBounds?.height ?? 0),
    },
    { width: 52, height: 52 },
  );

  /**
   * @type {ReadonlyArray<readonly [
   *   string[],
   *   {break: boolean, reapplication: boolean},
   *   boolean,
   * ]>}
   */
  const cases = [
    [[], { break: true, reapplication: true }, false],
    [["freezing_trap_break"], { break: false, reapplication: true }, false],
    [["status_reapplication"], { break: true, reapplication: false }, false],
    [
      ["freezing_trap_break", "status_reapplication"],
      { break: false, reapplication: false },
      true,
    ],
  ];
  for (const [disabled, expected, suppressed] of cases) {
    const filtered = buildChoreographyPlan(
      frame,
      surface,
      filtersDisabled(...disabled),
    );
    assert.ok(filtered);
    assert.equal(filtered.events.length, 1);
    assert.deepEqual(filtered.events[0].paintParts, expected);
    assert.equal(filtered.events[0].presentationSuppressed, suppressed);
    assert.deepEqual(
      filtered.events[0].atomicEventIds,
      events.map(({ event_id }) => event_id),
    );
    if (suppressed) {
      assert.equal(filtered.events[0].spatial, false);
      assert.equal(filtered.phases.total, 0);
    }
  }

  const breakOnly = buildChoreographyPlan(
    frame,
    surface,
    filtersDisabled("status_reapplication"),
  );
  const reapplicationOnly = buildChoreographyPlan(
    frame,
    surface,
    filtersDisabled("freezing_trap_break"),
  );
  assert.ok(breakOnly);
  assert.ok(reapplicationOnly);
  const breakExplanation = explainChoreographyEvent(breakOnly.events[0]);
  const reapplicationExplanation = explainChoreographyEvent(
    reapplicationOnly.events[0],
  );
  assert.equal(breakExplanation.title, "Broken");
  assert.deepEqual(
    breakExplanation.rows.map(({ label }) => label),
    ["Recipient"],
  );
  assert.doesNotMatch(JSON.stringify(breakExplanation), /reappl|status_applied/iu);
  assert.deepEqual(
    {
      width: Math.round(breakOnly.events[0].cueBounds?.width ?? 0),
      height: Math.round(breakOnly.events[0].cueBounds?.height ?? 0),
    },
    { width: 52, height: 52 },
  );
  assert.equal(reapplicationExplanation.title, "Reapplied");
  assert.deepEqual(
    reapplicationExplanation.rows.map(({ label }) => label),
    ["Source", "Recipient"],
  );
  assert.doesNotMatch(JSON.stringify(reapplicationExplanation), /broken/iu);
  assert.deepEqual(
    {
      width: Math.round(reapplicationOnly.events[0].cueBounds?.width ?? 0),
      height: Math.round(reapplicationOnly.events[0].cueBounds?.height ?? 0),
    },
    { width: 45, height: 45 },
  );
});

test("authorized Trap and Charge routes retain exact public endpoint owners", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const agentsByKey = new Map(
    raw.current_endpoint.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => [agent.presentation_key, agent],
    ),
  );
  /** @param {number} classId @returns {Record<string, any>} */
  const trajectoryForClass = (classId) => {
    const trajectory = raw.latest_events.agent_phase_trajectories.find(
      (/** @type {Record<string, any>} */ trajectory) =>
        agentsByKey.get(trajectory.agent_presentation_key)?.class_id === classId,
    );
    assert.ok(trajectory);
    return trajectory;
  };
  /** @param {number} sourceClass @param {number} targetClass @param {number} ordinal */
  const ability = (sourceClass, targetClass, ordinal) => ({
    event_kind: "ability_activated",
    ability_component: "ultimate",
    event_id: `${raw.latest_events.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
    phase_rank: 20,
    source_anchor: structuredClone(trajectoryForClass(sourceClass).transition_start),
    recipient_anchor: structuredClone(trajectoryForClass(targetClass).transition_start),
  });
  const events = [ability(3, 1, 0), ability(2, 3, 1)];
  raw.latest_events.events = events;
  raw.latest_events.event_count = events.length;
  raw.latest_events.ordered_event_ids = events.map(({ event_id }) => event_id);
  raw.latest_events.ordered_event_kinds = events.map(({ event_kind }) => event_kind);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.deepEqual(
    plan.events.map((event) => event.tokenId),
    ["hunter_trap", "warrior_charge"],
  );
  /** @type {ReadonlyArray<readonly [Record<string, any>, number]>} */
  const endpointCases = [
    [plan.events[0], 1],
    [plan.events[1], 3],
  ];
  for (const [event, targetClass] of endpointCases) {
    const targetTrajectory = trajectoryForClass(targetClass);
    assert.equal(event.targetPresentationKey, targetTrajectory.agent_presentation_key);
    assert.equal(event.targetPublicAgentId, targetTrajectory.agent_public_agent_id);
    assert.deepEqual(
      event.target,
      surface.worldToScreen(targetTrajectory.successor.position),
    );
    assert.ok(event.route);
    assert.deepEqual(event.route.end, event.target);
  }
});

test("status presentation preserves the valid five-source application maximum", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const appliedTemplate = structuredClone(
    raw.latest_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "status_applied",
    ),
  );
  assert.ok(appliedTemplate);
  const sourceTrajectories = /** @type {Record<string, any>[]} */ (
    raw.latest_events.agent_phase_trajectories
  );
  const applications = sourceTrajectories.map(
    (/** @type {Record<string, any>} */ trajectory, index) => ({
      ...structuredClone(appliedTemplate),
      event_id: `${raw.latest_events.incoming_transition_id}:event:${String(index).padStart(4, "0")}`,
      ordinal: index,
      source_anchor: structuredClone(trajectory.successor),
    }),
  );
  raw.latest_events.events = applications;
  raw.latest_events.event_count = applications.length;
  raw.latest_events.ordered_event_ids = applications.map(
    (/** @type {Record<string, any>} */ { event_id }) => event_id,
  );
  raw.latest_events.ordered_event_kinds = applications.map(
    (/** @type {Record<string, any>} */ { event_kind }) => event_kind,
  );
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(plan.events.length, 1);
  const applicationEventIds = applications.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  const [primary] = plan.events;
  assert.ok(primary);
  assert.deepEqual(primary.atomicEventIds, applicationEventIds);
  assert.deepEqual(primary.applicationEventIds, applicationEventIds);
  assert.deepEqual(
    primary.applicationSources.map(
      (/** @type {Record<string, any>} */ { eventId }) => eventId,
    ),
    applicationEventIds,
  );
  assert.deepEqual(
    primary.applicationSources.map(
      (/** @type {Record<string, any>} */ { sourcePublicAgentId }) =>
        sourcePublicAgentId,
    ),
    sourceTrajectories.map(
      (/** @type {Record<string, any>} */ { agent_public_agent_id }) =>
        agent_public_agent_id,
    ),
  );
  const explanation = explainChoreographyEvent(primary);
  assert.deepEqual(
    explanation.rows.map(({ label, value }) => [label, value]),
    [
      [
        "Application Sources",
        sourceTrajectories
          .map(
            (/** @type {Record<string, any>} */ { agent_public_agent_id }) =>
              `Agent ID ${agent_public_agent_id}`,
          )
          .join("; "),
      ],
      ["Recipient", `Agent ID ${applications[0].recipient_anchor.public_agent_id}`],
    ],
  );
});

test("authorized normalization rejects cross-transition status rows", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const applied = raw.latest_events.events.find(
    (/** @type {Record<string, any>} */ { event_kind }) =>
      event_kind === "status_applied",
  );
  assert.ok(applied);
  applied.event_id = "different-transition:event:0003";
  raw.latest_events.ordered_event_ids[3] = applied.event_id;
  await assert.rejects(normalizeAuthorizedPresentationFrameV1(raw), TypeError);
});

test("standalone serialized status events keep exact presentation meanings", async () => {
  const fixture = await authorizedFixture();
  const expectedCases = /** @type {ReadonlyArray<readonly [string, string]>} */ ([
    ["status_aged_to_zero", "expired"],
    ["status_broken_by_damage", "trap_broken"],
    ["status_applied", "applied"],
    ["status_refreshed_or_extended", "refreshed"],
    ["status_cleared_by_new_death", "cleared_by_death"],
  ]);
  for (const [eventKind, lifecycle] of expectedCases) {
    const raw = structuredClone(fixture.presentations.replay_oracle);
    const event = structuredClone(
      raw.latest_events.events.find(
        (/** @type {Record<string, any>} */ { event_kind }) =>
          event_kind === "status_applied",
      ),
    );
    event.event_kind = eventKind;
    event.event_id = `${raw.latest_events.incoming_transition_id}:event:0000`;
    event.ordinal = 0;
    if (eventKind !== "status_applied") {
      delete event.source_anchor;
    }
    raw.latest_events.events = [event];
    raw.latest_events.event_count = 1;
    raw.latest_events.ordered_event_ids = [event.event_id];
    raw.latest_events.ordered_event_kinds = [event.event_kind];
    const frame = await normalizeAuthorizedPresentationFrameV1(raw);
    const plan = buildChoreographyPlan(frame, surface);
    assert.ok(plan);
    assert.equal(plan.events.length, 1);
    assert.deepEqual(
      [plan.events[0].eventType, plan.events[0].lifecycle],
      [eventKind, lifecycle],
    );
    assert.deepEqual(plan.events[0].atomicEventIds, [event.event_id]);
    assert.equal(
      plan.events[0].presentationSuppressed,
      eventKind === "status_refreshed_or_extended",
    );
    assert.equal(plan.events[0].spatial, eventKind !== "status_refreshed_or_extended");
    assert.equal(
      plan.events[0].phaseStart,
      eventKind === "status_refreshed_or_extended" ? 2560 : 0,
    );
    assert.equal(
      plan.events[0].phaseEnd,
      eventKind === "status_refreshed_or_extended" ? 3040 : 480,
    );
  }
});

test("durable status countdown changes never synthesize browser lifecycle events", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  raw.latest_events.events = [];
  raw.latest_events.event_count = 0;
  raw.latest_events.ordered_event_ids = [];
  raw.latest_events.ordered_event_kinds = [];
  const countdownOnly = await normalizeAuthorizedPresentationFrameV1(raw);
  const plan = buildChoreographyPlan(countdownOnly, surface);
  assert.ok(plan);
  assert.deepEqual(plan.events, []);
  assert.equal(plan.phases.total, 0);
});
