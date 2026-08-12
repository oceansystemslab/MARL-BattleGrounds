import assert from "node:assert/strict";
import test from "node:test";

import { extractFrame } from "../src/api.js";
import { buildChoreographyPlan, transitionEpochKey } from "../src/choreography-plan.js";
import {
  liveDebuggerFrameIsScripted,
  liveDebuggerScenarioControlsAvailable,
  normalizeLiveDebuggerFrameV2,
  normalizeRecordingStatusV1,
  researcherEventTypesV2,
} from "../src/frame-normalizer.js";

const episodeId = "evaluation-episode";
const transitionId = `${episodeId}:transition:0`;

/** @param {any} event */
function eventId(event) {
  return event.event_id;
}

/** @param {any} event */
function eventType(event) {
  return event.event_type;
}

/**
 * @param {number} globalSlot
 * @param {string} phase
 * @param {readonly [number, number]} position
 */
function anchor(globalSlot, phase, position) {
  return {
    phase,
    global_slot: globalSlot,
    public_agent_id: String(globalSlot),
    position,
  };
}

const startZero = anchor(0, "transition_start", [1, 1]);
const startOne = anchor(1, "transition_start", [3, 1]);
const chargeZero = anchor(0, "post_charge", [1.5, 1]);
const successorZero = anchor(0, "successor", [2, 1]);
const successorOne = anchor(1, "successor", [3, 1]);

/** @returns {any[]} */
function v2Events() {
  const payloads = [
    {
      event_type: "action_rejected",
      actor_global_slot: 0,
      actor_public_agent_id: "0",
      actor_configured_active: true,
      rejection_component: "movement",
      actor_anchor: startZero,
    },
    {
      event_type: "ability_activated",
      source_global_slot: 0,
      recipient_global_slot: 1,
      ability_component: "basic",
      source_anchor: startZero,
      recipient_anchor: startOne,
    },
    {
      event_type: "source_damage_output",
      source_global_slot: 0,
      recipient_global_slot: 1,
      raw_damage_output: 4,
      source_modified_damage_output: 4,
      recipient_damage_modifier: 1,
      source_anchor: startZero,
      recipient_anchor: startOne,
    },
    {
      event_type: "source_healing_output",
      source_global_slot: 0,
      recipient_global_slot: 1,
      raw_healing_output: 1,
      source_modified_healing_output: 1,
      recipient_healing_modifier: 1,
      source_anchor: startZero,
      recipient_anchor: startOne,
    },
    {
      event_type: "recipient_health_resolution",
      recipient_global_slot: 1,
      transition_start_health: 10,
      total_effective_damage: 4,
      total_effective_healing: 1,
      health_after_combat_resolution: 7,
      realized_net_health_change: -3,
      recipient_anchor: startOne,
    },
    {
      event_type: "combat_countdown_reset",
      agent_global_slot: 0,
      agent_anchor: startZero,
    },
    {
      event_type: "health_regenerated",
      agent_global_slot: 0,
      actual_health_regenerated: 0.5,
      agent_anchor: startZero,
    },
    {
      event_type: "cooldown_started",
      agent_global_slot: 0,
      agent_anchor: startZero,
    },
    {
      event_type: "cooldown_ready",
      agent_global_slot: 0,
      agent_anchor: startZero,
    },
    {
      event_type: "charge_phase_displacement",
      agent_global_slot: 0,
      start_anchor: startZero,
      end_anchor: chargeZero,
    },
    {
      event_type: "ordinary_movement_phase_displacement",
      agent_global_slot: 0,
      start_anchor: chargeZero,
      end_anchor: successorZero,
    },
    {
      event_type: "agent_died",
      recipient_global_slot: 1,
      recipient_anchor: successorOne,
    },
    {
      event_type: "lethal_damage_contribution",
      source_global_slot: 0,
      recipient_global_slot: 1,
      attributed_death_damage: 3,
      source_anchor: successorZero,
      recipient_anchor: successorOne,
    },
    {
      event_type: "status_aged_to_zero",
      recipient_global_slot: 1,
      status_id: "hunter_basic_slow",
      recipient_anchor: successorOne,
    },
    {
      event_type: "status_broken_by_damage",
      recipient_global_slot: 1,
      status_id: "hunter_trap_stun",
      recipient_anchor: successorOne,
    },
    {
      event_type: "status_applied",
      source_global_slot: 0,
      recipient_global_slot: 1,
      status_id: "hunter_basic_slow",
      source_anchor: successorZero,
      recipient_anchor: successorOne,
    },
    {
      event_type: "status_refreshed_or_extended",
      recipient_global_slot: 1,
      status_id: "hunter_basic_slow",
      recipient_anchor: successorOne,
    },
    {
      event_type: "status_cleared_by_new_death",
      recipient_global_slot: 1,
      status_id: "hunter_basic_slow",
      recipient_anchor: successorOne,
    },
    {
      event_type: "spawn_shield_expired",
      agent_global_slot: 0,
      agent_anchor: successorZero,
    },
    {
      event_type: "respawn_wave_occurred",
      team_index: 0,
      team_id: 1,
      team_anchor: { phase: "successor", team_index: 0, team_id: 1 },
    },
    {
      event_type: "agent_respawned",
      agent_global_slot: 0,
      team_id: 1,
      realized_successor_position: [2, 1],
      agent_anchor: successorZero,
    },
  ];
  return payloads.map((payload, ordinal) => ({
    ...payload,
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    transition_id: transitionId,
    ordinal,
  }));
}

/** @returns {any} */
function researcherFrame() {
  const events = v2Events();
  return {
    schema_version: 2,
    frame_kind: "researcher_live_debugger",
    session_id: "live-session",
    run_generation: 3,
    revision: 9,
    episode_id: episodeId,
    frame_index: 1,
    recording: null,
    frame_id: `${episodeId}:frame:1`,
    simulator_step_count: 1,
    incoming_transition_index: 0,
    incoming_transition_id: transitionId,
    view_mode: "researcher",
    preset: "analysis",
    projection: {
      schema_version: 2,
      scene: {
        schema_version: 2,
        audience: "researcher",
        audience_badge: "PRIVILEGED RESEARCHER",
        episode_id: episodeId,
        frame_index: 1,
        frame_id: `${episodeId}:frame:1`,
        simulator_step_count: 1,
        incoming_transition_id: transitionId,
        incoming_event_ids: events.map(({ event_id }) => event_id),
        map: { width: 10, height: 8, obstacles: [] },
        agents: [
          {
            global_slot: 0,
            public_agent_id: "0",
            team_id: 1,
            class_id: 1,
            position: [2, 1],
            radius: 0.5,
            life_state: "alive",
            current_health: 9,
            max_health: 10,
            effective_movement_speed: 1,
            ultimate_cooldown_remaining: 2,
            statuses: [],
            aura_modifiers: [{ aura_id: "mage_damage_amplification", multiplier: 1.1 }],
          },
          {
            global_slot: 1,
            public_agent_id: "1",
            team_id: 1,
            class_id: 2,
            position: [3, 1],
            radius: 0.5,
            life_state: "corpse",
            current_health: 0,
            max_health: 10,
            effective_movement_speed: 1,
            ultimate_cooldown_remaining: 0,
            statuses: [
              {
                status_id: "hunter_basic_slow",
                remaining_duration: 2,
                source_class_id: 3,
              },
            ],
            aura_modifiers: [],
          },
        ],
        aura_fields: [
          {
            aura_id: "warrior_damage_mitigation",
            source_global_slot: 1,
            center: [3, 1],
            radius: 2,
          },
        ],
        ranges: [],
        selection: { controlled_global_slot: 0, selected_global_slot: 1 },
        next_decision_selected_legality: null,
        observer_visibility: [
          {
            observer_global_slot: 0,
            candidate_global_slot: 0,
            visible: true,
          },
          {
            observer_global_slot: 0,
            candidate_global_slot: 1,
            visible: false,
          },
        ],
      },
      incoming_events: {
        schema_version: 2,
        episode_id: episodeId,
        transition_index: 0,
        transition_id: transitionId,
        start_frame_id: `${episodeId}:frame:0`,
        successor_frame_id: `${episodeId}:frame:1`,
        start_simulator_step_count: 0,
        successor_simulator_step_count: 1,
        events,
      },
      status_source_evidence: {},
    },
    terminal: {
      is_sealed: false,
      terminated: false,
      truncated: false,
      reached_declared_horizon: false,
      reason: null,
    },
    scenario: {},
    available_scenarios: [],
    hud: {
      pending_action: {
        actor_global_slot: 0,
        target_action: 2,
        armed_lane: 1,
        target: { disclosure: "public", global_slot: 1 },
        pair_mask_value: false,
      },
    },
  };
}

test("API boundary normalizes researcher V2 once and preserves canonical identities", () => {
  const raw = researcherFrame();
  const normalized = extractFrame({ schema_version: 2, frame: raw });
  assert.ok(normalized);
  assert.notEqual(normalized, raw);
  assert.equal(normalized.transition_id, transitionId);
  assert.equal(normalized.event_batch.transition_id, transitionId);
  assert.equal(
    transitionEpochKey(normalized),
    '["live-session",3,"evaluation-episode:transition:0"]',
  );
  assert.deepEqual(
    normalized.event_batch.events.map(eventId),
    raw.projection.incoming_events.events.map(eventId),
  );
  assert.deepEqual(
    normalized.event_batch.events.map(eventType),
    researcherEventTypesV2,
  );
  assert.equal(normalized.scene.agents[0].effective_speed, 1);
  assert.equal(normalized.scene.agents[0].ultimate_cooldown, 2);
  assert.equal(normalized.scene.agents[0].modifiers[0].token_id, "mage_amplification");
  assert.equal(normalized.scene.agents[1].statuses[0].status_id, "hunter_basic_slow");
  assert.equal(normalized.scene.agents[1].statuses[0].token_id, "slow_hunter_basic");
  assert.equal(normalized.scene.aura_fields[0].token_id, "warrior_mitigation");
  assert.deepEqual(normalized.scene.observer_visibility, [
    { observer_global_slot: 0, candidate_global_slot: 0, visible: true },
    { observer_global_slot: 0, candidate_global_slot: 1, visible: false },
  ]);
  assert.deepEqual(normalized.scene.pending_route, {
    audience: "researcher",
    source_global_slot: 0,
    target_global_slot: 1,
    source_public_agent_id: "0",
    target_public_agent_id: "1",
    target_action: 2,
    source_anchor: [2, 1],
    target_anchor: [3, 1],
    source_radius: 0.5,
    target_radius: 0.5,
    lane: 1,
    legal: false,
  });
});

test("choreography gives every canonical V2 event one explicit disposition", () => {
  const frame = normalizeLiveDebuggerFrameV2(researcherFrame());
  const plan = buildChoreographyPlan(frame, {
    worldToScreen: (point) => ({
      x: ("x" in point ? point.x : point[0]) * 10,
      y: ("y" in point ? point.y : point[1]) * 10,
    }),
    worldLengthToScreen: (length) => length * 10,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 100,
      bottom: 80,
      width: 100,
      height: 80,
    },
  });
  assert.ok(plan);
  assert.equal(plan.transitionId, transitionId);
  assert.equal(plan.events.length, 21);
  assert.deepEqual(
    plan.events.map(({ eventId }) => eventId),
    frame.event_batch.events.map(eventId),
  );
  assert.deepEqual(
    plan.events.map(({ eventType }) => eventType),
    researcherEventTypesV2,
  );
  assert.equal(
    plan.events.some(({ kind }) => kind === "unknown"),
    false,
  );
  assert.equal(
    plan.events.find(({ eventType }) => eventType === "source_damage_output")?.kind,
    "feed_only",
  );
  assert.equal(
    plan.events.find(({ eventType }) => eventType === "ability_activated")?.kind,
    "activation",
  );
  assert.equal(
    plan.events.find(
      ({ eventType }) => eventType === "ordinary_movement_phase_displacement",
    )?.kind,
    "movement_displacement",
  );

  const byType = new Map(plan.events.map((event) => [event.eventType, event]));
  for (const eventType of [
    "combat_countdown_reset",
    "health_regenerated",
    "cooldown_started",
    "cooldown_ready",
    "agent_died",
    "spawn_shield_expired",
    "respawn_wave_occurred",
    "agent_respawned",
  ]) {
    assert.equal(byType.get(eventType)?.kind, "semantic_pulse", eventType);
    assert.equal(byType.get(eventType)?.spatial, true, eventType);
  }

  const statusApplied = byType.get("status_applied");
  assert.equal(statusApplied?.tokenId, "slow_hunter_basic");
  assert.equal(statusApplied?.sourceSlot, 0);
  assert.deepEqual(statusApplied?.source, { x: 20, y: 10 });
  assert.equal(statusApplied?.recipientSlot, 1);
  assert.deepEqual(statusApplied?.recipient, { x: 30, y: 10 });
  const statusRefresh = byType.get("status_refreshed_or_extended");
  assert.equal(statusRefresh?.sourceSlot, null);
  assert.equal(statusRefresh?.source, null);
  const statusDeathClear = byType.get("status_cleared_by_new_death");
  assert.equal(statusDeathClear?.lifecycle, "cleared_by_death");
  assert.match(
    statusDeathClear?.lifecycleToken?.accessibleName ?? "",
    /cleared.*recorded new death/u,
  );
  assert.notEqual(
    statusDeathClear?.lifecycle,
    byType.get("status_aged_to_zero")?.lifecycle,
  );
  assert.notEqual(
    statusDeathClear?.lifecycle,
    byType.get("status_broken_by_damage")?.lifecycle,
  );

  const ability = byType.get("ability_activated");
  const health = byType.get("recipient_health_resolution");
  const countdown = byType.get("combat_countdown_reset");
  const cooldown = byType.get("cooldown_started");
  const charge = byType.get("charge_phase_displacement");
  const movement = byType.get("ordinary_movement_phase_displacement");
  const death = byType.get("agent_died");
  const status = byType.get("status_applied");
  const shield = byType.get("spawn_shield_expired");
  const wave = byType.get("respawn_wave_occurred");
  const respawn = byType.get("agent_respawned");
  for (const [earlier, later] of [
    [ability, health],
    [health, countdown],
    [countdown, cooldown],
    [cooldown, charge],
    [charge, movement],
    [movement, death],
    [death, status],
    [status, shield],
    [shield, wave],
    [wave, respawn],
  ]) {
    assert.ok(earlier);
    assert.ok(later);
    assert.ok(
      earlier.phaseEnd <= later.phaseStart,
      `${earlier.eventType} must finish before ${later.eventType}`,
    );
  }
});

test("V2 planning preserves canonical identity and order beyond 128 events", () => {
  const raw = researcherFrame();
  const events = Array.from({ length: 160 }, (_, ordinal) => ({
    event_type: "status_aged_to_zero",
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    transition_id: transitionId,
    ordinal,
    recipient_global_slot: 1,
    status_id: "hunter_basic_slow",
    recipient_anchor: successorOne,
  }));
  raw.projection.incoming_events.events = events;
  raw.projection.scene.incoming_event_ids = events.map(eventId);
  const frame = normalizeLiveDebuggerFrameV2(raw);
  const plan = buildChoreographyPlan(frame, {
    worldToScreen: (point) => ({
      x: ("x" in point ? point.x : point[0]) * 10,
      y: ("y" in point ? point.y : point[1]) * 10,
    }),
    worldLengthToScreen: (length) => length * 10,
  });
  assert.ok(plan);
  assert.equal(plan.events.length, events.length);
  assert.deepEqual(
    plan.events.map(({ eventId }) => eventId),
    events.map(eventId),
  );
  assert.equal(
    plan.events.every(({ eventType }) => eventType === "status_aged_to_zero"),
    true,
  );
});

/** @returns {any} */
function povFrame() {
  const publicAgentId = "agent-zero";
  const povTransitionId = `${episodeId}:actor-pov:${publicAgentId}:transition:0`;
  return {
    schema_version: 2,
    frame_kind: "actor_pov_live_debugger",
    session_id: "live-session",
    run_generation: 3,
    revision: 10,
    episode_id: episodeId,
    frame_index: 1,
    recording: null,
    frame_id: `${episodeId}:frame:1`,
    simulator_step_count: 1,
    incoming_pov_transition_id: povTransitionId,
    view_mode: "pov",
    preset: "analysis",
    projection: {
      scene: {
        schema_version: 1,
        audience_badge: "AGENT POV · EXACT",
        observation_materialization: "exact_no_shared_obs_actor_input",
        episode_id: episodeId,
        frame_index: 1,
        pov_frame_id: `${episodeId}:actor-pov:${publicAgentId}:frame:1`,
        source_frame_id: `${episodeId}:frame:1`,
        simulator_step_count: 1,
        map: { width: 10, height: 8, obstacles: [] },
        self_actor: {
          global_slot: 0,
          public_agent_id: publicAgentId,
          team_id: 1,
          class_id: 1,
          position: [2, 1],
          radius: 0.5,
          alive: true,
          current_health: 9,
          max_health: 10,
          effective_movement_speed: 1,
          ultimate_cooldown_remaining: 2,
          status_feature_values: [1, 2, 3, 0.5, 0.85, 0.5, 4, 5, 6, 7, 0.5, 8, 9, 0.85],
        },
        visible_bodies: [
          {
            relation: "enemy",
            observation_row: 2,
            public_agent_id: "visible-enemy-row-2",
            team_id: 2,
            class_id: 3,
            position: [4, 2],
            radius: 0.5,
            alive: true,
            current_health: 5,
            max_health: 10,
            status_feature_values: [0, 0, 0, 1, 1, 1, 0, 2, 0, 0, 1, 0, 0, 0],
          },
        ],
        spawn_pads: [],
        respawn_waves: [],
      },
      next_decision_action_mask: {},
      incoming_transition_id: povTransitionId,
      incoming_cues: [
        {
          cue_type: "own_position_changed",
          cue_id: `${povTransitionId}:cue:0`,
          pov_transition_id: povTransitionId,
          ordinal: 0,
          start_position: [1, 1],
          successor_position: [2, 1],
        },
      ],
    },
    terminal: {},
    hud: {
      pending_action: {
        actor_public_agent_id: publicAgentId,
        target: {
          target_action: 2,
          public_agent_id: "visible-enemy-row-2",
        },
        armed_lane: 0,
        pair_mask_value: true,
      },
    },
  };
}

test("POV normalization retains visible rows without manufacturing global slots", () => {
  const normalized = extractFrame({ schema_version: 2, frame: povFrame() });
  assert.ok(normalized);
  assert.equal(normalized.scene.audience, "agent_pov");
  assert.equal(normalized.scene.agents.length, 1);
  const normalizedSelfStatuses = /** @type {Array<{
   * token_id: string,
   * duration: number,
   * status_feature_index: number,
   * source_class_id: number,
   * source_evidence: string,
   * }>} */ (normalized.scene.agents[0].statuses);
  assert.deepEqual(
    normalizedSelfStatuses.map(
      ({ token_id, duration, status_feature_index, source_class_id }) => ({
        token_id,
        duration,
        status_feature_index,
        source_class_id,
      }),
    ),
    [
      {
        token_id: "stun_warrior_charge",
        duration: 4,
        status_feature_index: 21,
        source_class_id: 2,
      },
      {
        token_id: "stun_hunter_trap",
        duration: 5,
        status_feature_index: 22,
        source_class_id: 3,
      },
      {
        token_id: "stun_rogue_poison",
        duration: 6,
        status_feature_index: 23,
        source_class_id: 4,
      },
      {
        token_id: "slow_warrior_charge",
        duration: 1,
        status_feature_index: 15,
        source_class_id: 2,
      },
      {
        token_id: "slow_hunter_basic",
        duration: 2,
        status_feature_index: 16,
        source_class_id: 3,
      },
      {
        token_id: "slow_rogue_poison",
        duration: 3,
        status_feature_index: 17,
        source_class_id: 4,
      },
      {
        token_id: "anti_heal_rogue_poison",
        duration: 7,
        status_feature_index: 24,
        source_class_id: 4,
      },
      {
        token_id: "priest_freedom",
        duration: 9,
        status_feature_index: 27,
        source_class_id: 5,
      },
      {
        token_id: "mage_burst",
        duration: 8,
        status_feature_index: 26,
        source_class_id: 1,
      },
    ],
  );
  assert.equal(
    normalizedSelfStatuses.every(
      ({ source_evidence }) => source_evidence === "effect_channel_only",
    ),
    true,
  );
  assert.equal(normalized.scene.observed_bodies.length, 1);
  assert.deepEqual(normalized.scene.observed_bodies[0].statuses, [
    {
      token_id: "stun_hunter_trap",
      duration: 2,
      status_feature_index: 22,
      source_class_id: 3,
      source_evidence: "effect_channel_only",
    },
  ]);
  assert.equal(
    Object.hasOwn(
      normalized.scene.observed_bodies[0].statuses[0],
      "source_global_slot",
    ),
    false,
  );
  assert.equal(normalized.scene.observed_bodies[0].observation_key, "enemy:2");
  assert.equal(
    Object.hasOwn(normalized.scene.observed_bodies[0], "global_slot"),
    false,
  );
  assert.equal(normalized.event_batch.events[0].event_type, "own_position_changed");
  assert.equal(
    normalized.event_batch.events[0].event_id,
    `${episodeId}:actor-pov:agent-zero:transition:0:cue:0`,
  );
  assert.equal(
    transitionEpochKey(normalized),
    '["live-session",3,"evaluation-episode:actor-pov:agent-zero:transition:0"]',
  );
  assert.deepEqual(normalized.scene.pending_route, {
    audience: "agent_pov",
    source_public_agent_id: "agent-zero",
    target_public_agent_id: "visible-enemy-row-2",
    target_action: 2,
    source_anchor: [2, 1],
    target_anchor: [4, 2],
    source_radius: 0.5,
    target_radius: 0.5,
    lane: 0,
    legal: true,
  });
  assert.equal(
    Object.hasOwn(normalized.scene.pending_route, "source_global_slot"),
    false,
  );
  assert.equal(
    Object.hasOwn(normalized.scene.pending_route, "target_global_slot"),
    false,
  );
});

test("POV status normalization rejects malformed or multiplier-shaped duration evidence", () => {
  const wrongLength = povFrame();
  wrongLength.projection.scene.self_actor.status_feature_values.pop();
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(wrongLength),
    /retain 14 finite nonnegative values/u,
  );

  const fractionalDuration = povFrame();
  fractionalDuration.projection.scene.self_actor.status_feature_values[6] = 0.5;
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(fractionalDuration),
    /status durations must be integer-valued/u,
  );
});

test("pending route normalization fails closed without a selected lane or visible endpoint", () => {
  const researcher = researcherFrame();
  researcher.hud.pending_action.armed_lane = null;
  assert.equal(normalizeLiveDebuggerFrameV2(researcher).scene.pending_route, null);

  const pov = povFrame();
  pov.hud.pending_action.target.public_agent_id = "hidden-agent";
  assert.equal(normalizeLiveDebuggerFrameV2(pov).scene.pending_route, null);
});

test("researcher visibility rows must join the controlled ordered roster", () => {
  const wrongObserver = researcherFrame();
  wrongObserver.projection.scene.observer_visibility[0].observer_global_slot = 1;
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(wrongObserver),
    /controlled ordered roster/u,
  );

  const missingCandidate = researcherFrame();
  missingCandidate.projection.scene.observer_visibility.pop();
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(missingCandidate),
    /match scene selection authority/u,
  );
});

test("audience scenes must retain the exact live simulator epoch", () => {
  const researcher = researcherFrame();
  researcher.projection.scene.simulator_step_count = 2;
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(researcher),
    /does not join its live frame/u,
  );

  const pov = povFrame();
  pov.projection.scene.simulator_step_count = 2;
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(pov),
    /does not join its live frame/u,
  );
});

test("POV normalization rejects noncanonical local cue identity", () => {
  const frame = povFrame();
  frame.projection.incoming_cues[0].cue_id = "forged-cue";
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(frame),
    /cue order or local identity/u,
  );
});

function activeRecordingStatus() {
  return {
    schema_version: 1,
    lifecycle: "recording",
    captured_transition_count: 1,
    expected_transition_count: 5,
    completion_state: null,
    completion_reason: null,
    restart_fenced: true,
    finish_available: true,
    review_available: false,
    retry_available: false,
    save_as_available: false,
    discard_available: true,
    persistence_error_code: null,
  };
}

test("recording status is exact, path-free, frozen, and joined to both live audiences", () => {
  assert.equal(normalizeRecordingStatusV1(null, 1), null);
  const status = activeRecordingStatus();
  const normalized = normalizeRecordingStatusV1(status, 1);
  assert.deepEqual(normalized, status);
  assert.notEqual(normalized, status);
  assert.equal(Object.isFrozen(normalized), true);

  const researcher = researcherFrame();
  researcher.recording = status;
  const pov = povFrame();
  pov.recording = structuredClone(status);
  assert.deepEqual(normalizeLiveDebuggerFrameV2(researcher).recording, status);
  assert.deepEqual(normalizeLiveDebuggerFrameV2(pov).recording, status);
  assert.equal(Object.hasOwn(normalized, "path"), false);
  assert.equal(Object.hasOwn(normalized, "detail"), false);
});

test("recording status rejects missing, extra, inconsistent, and over-disclosing fields", () => {
  const { completion_reason: _omittedReason, ...missing } = activeRecordingStatus();
  assert.throws(
    () => normalizeRecordingStatusV1(missing, 1),
    /unknown or missing fields/u,
  );

  for (const extra of [
    { path: "/private/replay.marlbg-replay.json" },
    { details: "raw host failure" },
  ]) {
    assert.throws(
      () => normalizeRecordingStatusV1({ ...activeRecordingStatus(), ...extra }, 1),
      /unknown or missing fields/u,
    );
  }

  for (const mutation of [
    { captured_transition_count: 0 },
    { expected_transition_count: 0 },
    { lifecycle: "unknown" },
    { restart_fenced: false },
    { finish_available: false },
    { retry_available: true },
    { persistence_error_code: "publication_failed" },
    { completion_state: "partial", completion_reason: null },
  ]) {
    assert.throws(() =>
      normalizeRecordingStatusV1({ ...activeRecordingStatus(), ...mutation }, 1),
    );
  }

  const missingAuthority = researcherFrame();
  delete missingAuthority.recording;
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(missingAuthority),
    /missing recording authority/u,
  );
});

test("both live audience roots reject path-shaped and arbitrary top-level siblings", () => {
  for (const frame of [researcherFrame(), povFrame()]) {
    for (const mutation of [
      { replay_path: "/private/episode.marlbg-replay.json" },
      { persistence_details: "host exception detail" },
      { unknown_extension: true },
    ]) {
      assert.throws(
        () => normalizeLiveDebuggerFrameV2({ ...frame, ...mutation }),
        /unknown or missing top-level fields/u,
      );
    }
  }
});

test("recording status accepts only canonical saved and persistence-failed availability", () => {
  const completed = {
    ...activeRecordingStatus(),
    captured_transition_count: 5,
    lifecycle: "saved",
    completion_state: "complete",
    completion_reason: null,
    finish_available: false,
    review_available: true,
    discard_available: false,
  };
  assert.deepEqual(normalizeRecordingStatusV1(completed, 5), completed);

  const failed = {
    ...completed,
    captured_transition_count: 3,
    lifecycle: "persistence_failed",
    completion_state: "partial",
    completion_reason: "user_finish_and_review",
    review_available: false,
    retry_available: true,
    save_as_available: true,
    persistence_error_code: "verification_failed",
  };
  assert.deepEqual(normalizeRecordingStatusV1(failed, 3), failed);
});

test("recording status accepts every declared lifecycle only with its canonical joins", () => {
  const cases = [
    {
      lifecycle: "recording",
      captured: 0,
      completionState: null,
      completionReason: null,
    },
    {
      lifecycle: "sealed",
      captured: 1,
      completionState: "complete",
      completionReason: null,
    },
    {
      lifecycle: "finalized_unsaved",
      captured: 1,
      completionState: "partial",
      completionReason: "user_finish_and_review",
    },
    {
      lifecycle: "persistence_failed",
      captured: 1,
      completionState: "partial",
      completionReason: "user_finish_and_review",
    },
    {
      lifecycle: "saved",
      captured: 1,
      completionState: "partial",
      completionReason: "user_finish_and_review",
    },
    {
      lifecycle: "reviewing",
      captured: 1,
      completionState: "partial",
      completionReason: "user_finish_and_review",
    },
    {
      lifecycle: "discarded",
      captured: 0,
      completionState: null,
      completionReason: null,
    },
  ];
  for (const { captured, completionReason, completionState, lifecycle } of cases) {
    const persistenceFailed = lifecycle === "persistence_failed";
    const status = {
      ...activeRecordingStatus(),
      lifecycle,
      captured_transition_count: captured,
      completion_state: completionState,
      completion_reason: completionReason,
      restart_fenced: captured > 0 || lifecycle !== "recording",
      finish_available: lifecycle === "recording",
      review_available: lifecycle === "saved",
      retry_available: persistenceFailed,
      save_as_available: persistenceFailed,
      discard_available: lifecycle === "recording" && captured > 0,
      persistence_error_code: persistenceFailed ? "publication_failed" : null,
    };
    assert.deepEqual(normalizeRecordingStatusV1(status, captured), status, lifecycle);
  }
});

test("scripted authority comes from the audience-owned live envelope", () => {
  assert.equal(
    liveDebuggerFrameIsScripted({
      frame_kind: "researcher_live_debugger",
      scenario: { mode: "scripted" },
    }),
    true,
  );
  assert.equal(
    liveDebuggerFrameIsScripted({
      frame_kind: "actor_pov_live_debugger",
      hud: { pending_submission_scope: "scripted_playback" },
    }),
    true,
  );
  assert.equal(
    liveDebuggerFrameIsScripted({
      frame_kind: "actor_pov_live_debugger",
      hud: { pending_submission_scope: "controlled_actor" },
      scenario: { mode: "scripted" },
    }),
    false,
  );
  assert.equal(
    liveDebuggerScenarioControlsAvailable({
      frame_kind: "researcher_live_debugger",
    }),
    true,
  );
  assert.equal(
    liveDebuggerScenarioControlsAvailable({
      frame_kind: "actor_pov_live_debugger",
      hud: { pending_submission_scope: "controlled_actor" },
    }),
    false,
  );
});
