import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { extractFrame } from "../src/api.js";
import {
  liveDebuggerFrameIsScripted,
  liveDebuggerScenarioControlsAvailable,
  normalizeDebuggerAudienceProjectionV2,
  normalizeLiveDebuggerFrameV2,
  normalizeRecordingStatusV1,
  researcherEventTypesV2,
} from "../src/frame-normalizer.js";

const episodeId = "evaluation-episode";
const transitionId = `${episodeId}:transition:0`;
const phaseRankByEventType = Object.freeze({
  action_rejected: 10,
  ability_activated: 20,
  source_damage_output: 30,
  source_healing_output: 30,
  recipient_health_resolution: 40,
  combat_countdown_reset: 50,
  health_regenerated: 50,
  cooldown_started: 60,
  cooldown_ready: 60,
  charge_phase_displacement: 70,
  ordinary_movement_phase_displacement: 80,
  agent_died: 90,
  lethal_damage_contribution: 90,
  status_aged_to_zero: 100,
  status_broken_by_damage: 100,
  status_applied: 100,
  status_refreshed_or_extended: 100,
  status_cleared_by_new_death: 100,
  spawn_shield_expired: 110,
  respawn_wave_occurred: 120,
  agent_respawned: 120,
});

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

test("production-captured live projections replace the synthetic registry seam", async () => {
  const fixture = await authorizedFixture();
  for (const [kind, audience] of [
    ["live_oracle", "researcher"],
    ["live_no_shared_obs_agent_pov", "agent_pov"],
  ]) {
    const raw = structuredClone(fixture.pairs[kind].transport);
    const serializedBefore = JSON.stringify(raw);
    const normalized = normalizeLiveDebuggerFrameV2(raw);
    assert.equal(normalized.scene.audience, audience);
    assert.equal(JSON.stringify(raw), serializedBefore);
    assert.equal(Object.isFrozen(normalized), true);
    assert.equal(Object.isFrozen(normalized.scene), true);
  }
});

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

const classNames = Object.freeze([
  null,
  "Mage",
  "Warrior",
  "Hunter",
  "Rogue",
  "Priest",
]);
const statusMechanics = Object.freeze([
  [0, "warrior_charge_slow", "slow", 2, "ultimate", "movement_multiplier", 0.5, false],
  [1, "hunter_basic_slow", "slow", 3, "basic", "movement_multiplier", 0.8, false],
  [2, "rogue_poison_slow", "slow", 4, "ultimate", "movement_multiplier", 0.6, false],
  [3, "warrior_charge_stun", "stun", 2, "ultimate", "none", null, false],
  [4, "hunter_trap_stun", "stun", 3, "ultimate", "none", null, true],
  [5, "rogue_poison_stun", "stun", 4, "ultimate", "none", null, false],
  [
    6,
    "rogue_poison_anti_heal",
    "anti_heal",
    4,
    "ultimate",
    "healing_multiplier",
    0,
    false,
  ],
  [
    7,
    "mage_burst_damage_amplification",
    "damage_amplification",
    1,
    "ultimate",
    "damage_multiplier",
    1.5,
    false,
  ],
  [
    8,
    "priest_blessing_of_freedom_movement_floor",
    "movement_floor",
    5,
    "basic",
    "movement_floor",
    1,
    false,
  ],
]);

/** @returns {any[]} */
function v2ClassMechanics() {
  return [1, 2, 3, 4, 5].map((classId) => ({
    class_id: classId,
    class_name: classNames[classId],
    maximum_health: 100,
    body_radius: 0.5,
    base_movement_speed: 1,
    observation_radius: 6,
    basic_target_mode: classId === 5 ? "ally" : "enemy",
    basic_interaction_radius: 3,
    basic_raw_damage: classId === 5 ? 0 : 10,
    basic_raw_healing: classId === 5 ? 10 : 0,
    ultimate_target_mode:
      classId === 1 ? "target_none" : classId === 5 ? "ally" : "enemy",
    ultimate_interaction_radius: 4,
    ultimate_cooldown_steps: 5,
    ultimate_raw_damage: classId === 5 ? 0 : 15,
    ultimate_raw_healing: classId === 5 ? 20 : 0,
    out_of_combat_delay_steps: 3,
    out_of_combat_health_regeneration_fraction_per_step: 0.05,
    status_mechanics: statusMechanics
      .filter((row) => row[3] === classId)
      .map(
        ([
          statusChannel,
          statusId,
          family,
          _sourceClassId,
          sourceActionComponent,
          magnitudeKind,
          magnitude,
          breaks,
        ]) => ({
          status_channel: statusChannel,
          status_id: statusId,
          family,
          source_action_component: sourceActionComponent,
          duration_steps: 2,
          magnitude_kind: magnitudeKind,
          magnitude,
          breaks_on_positive_damage: breaks,
        }),
      ),
    aura_mechanics:
      classId === 1
        ? [
            {
              aura_id: "mage_damage_amplification",
              radius: 4,
              per_emitter_multiplier: 1.1,
              stacking_rule: "multiply_then_clamp",
              clamp_kind: "ceiling",
              clamp_value: 1.5,
            },
          ]
        : classId === 2
          ? [
              {
                aura_id: "warrior_damage_mitigation",
                radius: 4,
                per_emitter_multiplier: 0.9,
                stacking_rule: "multiply_then_clamp",
                clamp_kind: "floor",
                clamp_value: 0.5,
              },
            ]
          : [],
  }));
}

/** @returns {any[]} */
function v2Events() {
  const payloads = [
    {
      event_type: "action_rejected",
      actor_global_slot: 0,
      actor_public_agent_id: "0",
      actor_configured_active: true,
      rejection_component: "movement",
      submitted_move_action: 1,
      submitted_select_target_action: 1,
      submitted_use_ultimate_action: 0,
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
      raw_damage_output: 10,
      source_modified_damage_output: 10,
      recipient_damage_modifier: 1,
      mage_damage_aura_covering_emitter_global_slots: [],
      warrior_mitigation_aura_covering_emitter_global_slots: [],
      source_anchor: startZero,
      recipient_anchor: startOne,
    },
    {
      event_type: "source_healing_output",
      source_global_slot: 1,
      recipient_global_slot: 0,
      raw_healing_output: 0,
      source_modified_healing_output: 0,
      recipient_healing_modifier: 1,
      source_anchor: startOne,
      recipient_anchor: startZero,
    },
    {
      event_type: "recipient_health_resolution",
      recipient_global_slot: 1,
      transition_start_health: 100,
      total_effective_damage: 10,
      total_effective_healing: 1,
      health_after_combat_resolution: 91,
      realized_net_health_change: -9,
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
      agent_global_slot: 1,
      agent_anchor: startOne,
    },
    {
      event_type: "charge_phase_displacement",
      agent_global_slot: 0,
      realized_displacement: [0.5, 0],
      start_anchor: startZero,
      end_anchor: chargeZero,
    },
    {
      event_type: "ordinary_movement_phase_displacement",
      agent_global_slot: 0,
      realized_displacement: [0.5, 0],
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
      status_channel: 1,
      status_id: "hunter_basic_slow",
      recipient_anchor: successorOne,
    },
    {
      event_type: "status_broken_by_damage",
      recipient_global_slot: 1,
      status_channel: 4,
      status_id: "hunter_trap_stun",
      recipient_anchor: successorOne,
    },
    {
      event_type: "status_applied",
      source_global_slot: 0,
      recipient_global_slot: 1,
      status_channel: 1,
      status_id: "hunter_basic_slow",
      source_anchor: successorZero,
      recipient_anchor: successorOne,
    },
    {
      event_type: "status_refreshed_or_extended",
      recipient_global_slot: 1,
      status_channel: 1,
      status_id: "hunter_basic_slow",
      recipient_anchor: successorOne,
    },
    {
      event_type: "status_cleared_by_new_death",
      recipient_global_slot: 1,
      status_channel: 1,
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
    ...structuredClone(payload),
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    transition_id: transitionId,
    ordinal,
    phase_rank:
      phaseRankByEventType[
        /** @type {keyof typeof phaseRankByEventType} */ (payload.event_type)
      ],
  }));
}

/** @returns {any} */
function researcherFrame() {
  const events = [v2Events()[0]];
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
    verbose: false,
    show_ranges: true,
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
            team_local_slot: 0,
            class_id: 3,
            position: [2, 1],
            radius: 0.5,
            life_state: "alive",
            current_health: 90,
            max_health: 100,
            effective_movement_speed: 1,
            ultimate_cooldown_remaining: 2,
            spawn_shield_remaining: 0,
            steps_until_out_of_combat: 0,
            respawned_on_incoming_transition: false,
            respawn_event_id: null,
            statuses: [],
            aura_modifiers: [
              { aura_id: "mage_damage_amplification", multiplier: 1.1 },
              { aura_id: "warrior_damage_mitigation", multiplier: 1 },
            ],
          },
          {
            global_slot: 1,
            public_agent_id: "1",
            team_id: 1,
            team_local_slot: 1,
            class_id: 2,
            position: [3, 1],
            radius: 0.5,
            life_state: "alive",
            current_health: 100,
            max_health: 100,
            effective_movement_speed: 1,
            ultimate_cooldown_remaining: 0,
            spawn_shield_remaining: 0,
            steps_until_out_of_combat: 0,
            respawned_on_incoming_transition: false,
            respawn_event_id: null,
            statuses: [
              {
                status_channel: 1,
                status_id: "hunter_basic_slow",
                family: "slow",
                remaining_duration: 2,
                source_class_id: 3,
                source_class_name: "Hunter",
                source_action_component: "basic",
                magnitude_kind: "movement_multiplier",
                magnitude: 0.8,
                breaks_on_positive_damage: false,
                direct_source_evidence: [],
              },
            ],
            aura_modifiers: [
              { aura_id: "mage_damage_amplification", multiplier: 1 },
              { aura_id: "warrior_damage_mitigation", multiplier: 1 },
            ],
          },
        ],
        class_mechanics: v2ClassMechanics(),
        aura_fields: [
          {
            aura_id: "warrior_damage_mitigation",
            source_global_slot: 1,
            source_public_agent_id: "1",
            source_class_id: 2,
            source_class_name: "Warrior",
            source_alive: true,
            center: [3, 1],
            radius: 4,
            beneficiary_relation: "same_team",
            per_emitter_multiplier: 0.9,
            stacking_rule: "multiply_then_clamp",
            clamp_kind: "floor",
            clamp_value: 0.5,
          },
        ],
        spawn_pads: [
          {
            team_id: 1,
            team_local_slot: 0,
            assigned_global_slot: 0,
            assigned_public_agent_id: "0",
            position: [2, 1],
          },
          {
            team_id: 1,
            team_local_slot: 1,
            assigned_global_slot: 1,
            assigned_public_agent_id: "1",
            position: [3, 1],
          },
        ],
        respawn_waves: [
          { team_index: 0, team_id: 1, period_steps: 10, countdown_steps: 4 },
          { team_index: 1, team_id: 2, period_steps: 10, countdown_steps: 7 },
        ],
        ranges: [{ global_slot: 0, center: [2, 1], radius: 6, kind: "observation" }],
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
        public_agent_id_by_global_slot: Array.from({ length: 10 }, (_, slot) =>
          String(slot),
        ),
        configured_active_by_global_slot: [
          true,
          true,
          false,
          false,
          false,
          false,
          false,
          false,
          false,
          false,
        ],
        agent_phase_trajectories: [
          {
            global_slot: 0,
            public_agent_id: "0",
            transition_start: structuredClone(startZero),
            post_charge: structuredClone(chargeZero),
            successor: structuredClone(successorZero),
          },
          {
            global_slot: 1,
            public_agent_id: "1",
            transition_start: structuredClone(startOne),
            post_charge: anchor(1, "post_charge", [3, 1]),
            successor: structuredClone(successorOne),
          },
        ],
        events,
      },
      status_source_evidence: {
        schema_version: 2,
        episode_id: episodeId,
        frame_index: 1,
        frame_id: `${episodeId}:frame:1`,
        active_statuses: [
          {
            recipient_global_slot: 1,
            recipient_public_agent_id: "1",
            status_channel: 1,
            status_id: "hunter_basic_slow",
            direct_source_evidence: [],
          },
        ],
      },
    },
    terminal: {
      is_sealed: false,
      terminated: false,
      truncated: false,
      reached_declared_horizon: false,
      reason: null,
    },
    scenario: {
      name: "arena_5v5",
      title: "Arena 5v5",
      description: "Interactive arena.",
      mode: "interactive",
      audience: "researcher",
      ordinary_movement_distance_scale: 1,
      completed_frame_count: 0,
      frame_count: 0,
      next_frame_index: null,
      next_frame_label: null,
      next_frame_description: null,
      script_complete: false,
    },
    available_scenarios: [],
    hud: {
      roster_global_slots: [0, 1],
      controlled_global_slot: 0,
      selected_global_slot: 1,
      pending_submission_scope: "controlled_actor",
      pending_actions: [
        {
          label: "PENDING / WILL SUBMIT",
          actor_global_slot: 0,
          move_action: 0,
          target_action: 2,
          armed_lane: 1,
          arm_origin: "explicit",
          target: { disclosure: "public", global_slot: 1 },
          movement_mask_value: true,
          pair_mask_value: true,
          summary: "STAY + BASIC → Agent ID 1",
        },
      ],
      pending_action: {
        label: "PENDING / WILL SUBMIT",
        actor_global_slot: 0,
        move_action: 0,
        target_action: 2,
        armed_lane: 1,
        arm_origin: "explicit",
        target: { disclosure: "public", global_slot: 1 },
        movement_mask_value: true,
        pair_mask_value: true,
        summary: "STAY + BASIC → Agent ID 1",
      },
      latest_transition: null,
      movement_legalities: Array.from({ length: 9 }, (_, moveAction) => ({
        move_action: moveAction,
        available: true,
      })),
      candidate_legalities: [],
      diagnostics: [],
    },
  };
}

/** @param {any[]} events */
function researcherFrameWithEvents(events) {
  const frame = researcherFrame();
  const normalizedEvents = events.map((event, ordinal) => ({
    ...structuredClone(event),
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
  }));
  frame.projection.incoming_events.events = normalizedEvents;
  frame.projection.scene.incoming_event_ids = normalizedEvents.map(eventId);
  return frame;
}

test("API boundary normalizes researcher V2 once and preserves canonical identities", () => {
  const raw = researcherFrame();
  const normalized = extractFrame({ schema_version: 2, frame: raw });
  assert.ok(normalized);
  assert.notEqual(normalized, raw);
  assert.equal(normalized.transition_id, transitionId);
  assert.equal(normalized.event_batch.transition_id, transitionId);
  assert.deepEqual(
    [normalized.session_id, normalized.run_generation, normalized.transition_id],
    ["live-session", 3, "evaluation-episode:transition:0"],
  );
  assert.deepEqual(
    normalized.event_batch.events.map(eventId),
    raw.projection.incoming_events.events.map(eventId),
  );
  assert.deepEqual(normalized.event_batch.events.map(eventType), ["action_rejected"]);
  assert.equal(normalized.scene.agents[0].effective_speed, 1);
  assert.equal(normalized.scene.agents[0].ultimate_cooldown, 2);
  assert.equal(normalized.scene.agents[0].modifiers[0].token_id, "mage_amplification");
  assert.equal(normalized.scene.agents[1].statuses[0].status_id, "hunter_basic_slow");
  assert.equal(normalized.scene.agents[1].statuses[0].token_id, "slow_hunter_basic");
  assert.equal(normalized.scene.aura_fields[0].token_id, "warrior_mitigation");
  assert.equal(normalized.scene.class_mechanics[0].class_id, 1);
  assert.equal(
    normalized.scene.class_mechanics[0].status_mechanics[0].token_id,
    "mage_burst",
  );
  assert.equal(
    normalized.scene.class_mechanics[0].aura_mechanics[0].token_id,
    "mage_amplification",
  );
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
    legal: true,
  });
});

test("researcher class mechanics are exact, ordered, and class-owned", () => {
  for (const mutate of [
    /** @param {any} frame */ (frame) => {
      delete frame.projection.scene.class_mechanics[0].status_mechanics[0]
        .duration_steps;
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.class_mechanics[0].status_mechanics[0].extra = true;
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.class_mechanics[0].status_mechanics[0].status_id =
        "hunter_basic_slow";
    },
    /** @param {any} frame */ (frame) => {
      const mage = frame.projection.scene.class_mechanics[0];
      const warrior = frame.projection.scene.class_mechanics[1];
      [mage.aura_mechanics, warrior.aura_mechanics] = [
        warrior.aura_mechanics,
        mage.aura_mechanics,
      ];
    },
    /** @param {any} frame */ (frame) => {
      const mage = frame.projection.scene.class_mechanics[0];
      const warrior = frame.projection.scene.class_mechanics[1];
      mage.aura_mechanics.push(structuredClone(mage.aura_mechanics[0]));
      warrior.aura_mechanics = [];
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.class_mechanics[3].status_mechanics.reverse();
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.class_mechanics[0].maximum_health = 0;
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.class_mechanics[0].body_radius = 0;
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.class_mechanics[0].out_of_combat_health_regeneration_fraction_per_step = 1.01;
    },
  ]) {
    const frame = researcherFrame();
    mutate(frame);
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(frame),
      /class (?:status|aura|mechanics)|catalog axes/iu,
    );
  }
});

test("researcher status and aura semantics are exact and catalog-joined", () => {
  const mutations = [
    /** @param {any} frame */ (frame) => {
      delete frame.projection.scene.agents[1].statuses[0].source_class_name;
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.agents[1].statuses[0].extra = "secret";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.agents[0].aura_modifiers[0].extra = true;
    },
    /** @param {any} frame */ (frame) => {
      delete frame.projection.scene.agents[0].aura_modifiers[0].multiplier;
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.agents[0].aura_modifiers[0].aura_id = "unknown";
    },
    /** @param {any} frame */ (frame) => {
      delete frame.projection.scene.aura_fields[0].source_alive;
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.aura_fields[0].extra = "secret";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.aura_fields[0].source_public_agent_id = "0";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.aura_fields[0].center = [4, 1];
    },
  ];
  for (const mutate of mutations) {
    const frame = researcherFrame();
    mutate(frame);
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(frame),
      /researcher (?:status|aura)/iu,
    );
  }
});

test("researcher status and aura joins preserve serialized numeric tuning", () => {
  const frame = researcherFrame();
  const hunterSlow = frame.projection.scene.class_mechanics[2].status_mechanics.find(
    /** @param {any} status */ (status) => status.status_channel === 1,
  );
  hunterSlow.magnitude = 0.731;
  frame.projection.scene.agents[1].statuses[0].magnitude = 0.731;

  const warriorAura = frame.projection.scene.class_mechanics[1].aura_mechanics[0];
  warriorAura.radius = 7.25;
  warriorAura.per_emitter_multiplier = 0.843;
  warriorAura.clamp_value = 0.417;
  Object.assign(frame.projection.scene.aura_fields[0], {
    radius: 7.25,
    per_emitter_multiplier: 0.843,
    clamp_value: 0.417,
  });
  frame.projection.scene.agents[0].aura_modifiers[0].multiplier = 1.237;

  const normalized = normalizeLiveDebuggerFrameV2(frame);
  assert.equal(normalized.scene.agents[1].statuses[0].magnitude, 0.731);
  assert.equal(normalized.scene.aura_fields[0].radius, 7.25);
  assert.equal(normalized.scene.aura_fields[0].per_emitter_multiplier, 0.843);
  assert.equal(normalized.scene.aura_fields[0].clamp_value, 0.417);
  assert.equal(normalized.scene.agents[0].modifiers[0].multiplier, 1.237);
});

test("researcher normalization preserves locally valid scientific values", () => {
  const eventFrame = researcherFrameWithEvents(v2Events());
  /** @param {string} type */
  const eventOfType = (type) =>
    eventFrame.projection.incoming_events.events.find(
      (/** @type {any} */ row) => row.event_type === type,
    );
  Object.assign(eventOfType("source_damage_output"), {
    raw_damage_output: 17.125,
    source_modified_damage_output: 23.75,
    recipient_damage_modifier: 0.4375,
    mage_damage_aura_covering_emitter_global_slots: [0, 5],
    warrior_mitigation_aura_covering_emitter_global_slots: [1, 6],
  });
  Object.assign(eventOfType("source_healing_output"), {
    raw_healing_output: 6.25,
    source_modified_healing_output: 4.75,
    recipient_healing_modifier: 0.375,
  });
  Object.assign(eventOfType("recipient_health_resolution"), {
    transition_start_health: 41.25,
    total_effective_damage: 7.5,
    total_effective_healing: 12.25,
    health_after_combat_resolution: 99.5,
    realized_net_health_change: -123.75,
  });
  eventOfType("health_regenerated").actual_health_regenerated = 0.125;
  eventOfType("lethal_damage_contribution").attributed_death_damage = 0.75;
  const eventSceneAgents = eventFrame.projection.scene.agents;
  const cooldownReady = eventOfType("cooldown_ready");
  const died = eventOfType("agent_died");
  const shieldExpired = eventOfType("spawn_shield_expired");
  const respawned = eventOfType("agent_respawned");
  eventSceneAgents.find(
    (/** @type {any} */ agent) => agent.global_slot === cooldownReady.agent_global_slot,
  ).ultimate_cooldown_remaining = 13;
  Object.assign(
    eventSceneAgents.find(
      (/** @type {any} */ agent) => agent.global_slot === died.recipient_global_slot,
    ),
    {
      current_health: 1,
    },
  );
  eventSceneAgents.find(
    (/** @type {any} */ agent) => agent.global_slot === shieldExpired.agent_global_slot,
  ).spawn_shield_remaining = 13;
  Object.assign(
    eventSceneAgents.find(
      (/** @type {any} */ agent) => agent.global_slot === respawned.agent_global_slot,
    ),
    {
      life_state: "corpse",
      current_health: 0,
      respawned_on_incoming_transition: false,
      respawn_event_id: null,
    },
  );

  const normalizedEventFrame = normalizeLiveDebuggerFrameV2(eventFrame);
  const normalizedEvents = normalizedEventFrame.event_batch.events;
  /** @param {string} type */
  const normalizedEventOfType = (type) =>
    normalizedEvents.find((/** @type {any} */ row) => row.event_type === type);
  assert.deepEqual(
    normalizedEventOfType("source_damage_output"),
    eventOfType("source_damage_output"),
  );
  assert.deepEqual(
    normalizedEventOfType("source_healing_output"),
    eventOfType("source_healing_output"),
  );
  assert.deepEqual(
    normalizedEventOfType("recipient_health_resolution"),
    eventOfType("recipient_health_resolution"),
  );
  assert.equal(
    normalizedEventOfType("health_regenerated").actual_health_regenerated,
    0.125,
  );
  assert.equal(
    normalizedEventOfType("lethal_damage_contribution").attributed_death_damage,
    0.75,
  );
  const normalizedEventSceneAgents = normalizedEventFrame.scene.agents;
  assert.equal(
    normalizedEventSceneAgents.find(
      (/** @type {any} */ agent) =>
        agent.global_slot === cooldownReady.agent_global_slot,
    ).ultimate_cooldown_remaining,
    13,
  );
  assert.equal(
    normalizedEventSceneAgents.find(
      (/** @type {any} */ agent) => agent.global_slot === died.recipient_global_slot,
    ).current_health,
    1,
  );
  assert.equal(
    normalizedEventSceneAgents.find(
      (/** @type {any} */ agent) => agent.global_slot === died.recipient_global_slot,
    ).respawn_event_id,
    null,
  );
  assert.equal(
    normalizedEventSceneAgents.find(
      (/** @type {any} */ agent) =>
        agent.global_slot === shieldExpired.agent_global_slot,
    ).spawn_shield_remaining,
    13,
  );
  assert.equal(
    normalizedEventSceneAgents.find(
      (/** @type {any} */ agent) => agent.global_slot === respawned.agent_global_slot,
    ).life_state,
    "corpse",
  );

  const sceneFrame = researcherFrame();
  Object.assign(sceneFrame.projection.scene.agents[0], {
    current_health: 0,
    max_health: 137.5,
    radius: 0.375,
    steps_until_out_of_combat: 777,
    ultimate_cooldown_remaining: 999,
  });
  Object.assign(sceneFrame.projection.scene.agents[1].statuses[0], {
    family: "movement_floor",
    remaining_duration: 987,
    source_action_component: "ultimate",
    magnitude_kind: "movement_floor",
    magnitude: 0.731,
    breaks_on_positive_damage: true,
  });
  Object.assign(sceneFrame.projection.scene.aura_fields[0], {
    radius: 7.25,
    per_emitter_multiplier: 0.843,
    clamp_value: 0.417,
  });
  sceneFrame.projection.scene.agents[0].aura_modifiers[0].multiplier = 1.237;
  sceneFrame.projection.scene.ranges[0].radius = 123.5;
  sceneFrame.hud.pending_action.target_action = 1;
  sceneFrame.hud.pending_actions[0].target_action = 1;

  const normalizedScene = normalizeLiveDebuggerFrameV2(sceneFrame);
  assert.deepEqual(
    {
      current_health: normalizedScene.scene.agents[0].current_health,
      max_health: normalizedScene.scene.agents[0].max_health,
      radius: normalizedScene.scene.agents[0].radius,
      steps_until_out_of_combat:
        normalizedScene.scene.agents[0].steps_until_out_of_combat,
      ultimate_cooldown_remaining:
        normalizedScene.scene.agents[0].ultimate_cooldown_remaining,
    },
    {
      current_health: 0,
      max_health: 137.5,
      radius: 0.375,
      steps_until_out_of_combat: 777,
      ultimate_cooldown_remaining: 999,
    },
  );
  assert.equal(normalizedScene.scene.agents[1].statuses[0].remaining_duration, 987);
  assert.equal(normalizedScene.scene.agents[1].statuses[0].family, "movement_floor");
  assert.equal(
    normalizedScene.scene.agents[1].statuses[0].source_action_component,
    "ultimate",
  );
  assert.equal(
    normalizedScene.scene.agents[1].statuses[0].magnitude_kind,
    "movement_floor",
  );
  assert.equal(normalizedScene.scene.agents[1].statuses[0].magnitude, 0.731);
  assert.equal(
    normalizedScene.scene.agents[1].statuses[0].breaks_on_positive_damage,
    true,
  );
  assert.equal(normalizedScene.scene.aura_fields[0].radius, 7.25);
  assert.equal(normalizedScene.scene.aura_fields[0].per_emitter_multiplier, 0.843);
  assert.equal(normalizedScene.scene.aura_fields[0].clamp_value, 0.417);
  assert.equal(normalizedScene.scene.agents[0].modifiers[0].multiplier, 1.237);
  assert.equal(normalizedScene.scene.ranges[0].radius, 123.5);
  assert.equal(normalizedScene.hud.pending_action.target_action, 1);
  assert.equal(normalizedScene.hud.pending_action.target.global_slot, 1);
});

test("researcher normalizer contains no geometry or combat reconstruction helpers", async () => {
  const source = await readFile(
    new URL("../src/frame-normalizer.js", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(
    source,
    /researcherTargetSlot|researcherPointDistance|researcherAuraCoverageAtTransitionStart|researcherStatusMechanic|activationForEffect|expectedHealthRecipients/u,
  );
  assert.doesNotMatch(
    source,
    /Math\.hypot|source_modified_damage_output\s*\*|source_modified_healing_output\s*\*|cooldown-ready event must join|death event must agree|shield-expiry event must join|respawn event must join successor/u,
  );
});

test("researcher direct status source evidence is exact and roster-joined", () => {
  const sourceJoined = researcherFrame();
  const evidence = {
    source_global_slot: 2,
    source_public_agent_id: "2",
    event_id: `${transitionId}:event:0015`,
  };
  sourceJoined.projection.scene.agents.push({
    ...structuredClone(sourceJoined.projection.scene.agents[0]),
    global_slot: 2,
    public_agent_id: "2",
    team_local_slot: 2,
    class_id: 3,
    position: [4, 1],
    statuses: [],
  });
  sourceJoined.projection.scene.spawn_pads.push({
    team_id: 1,
    team_local_slot: 2,
    assigned_global_slot: 2,
    assigned_public_agent_id: "2",
    position: [4, 1],
  });
  sourceJoined.hud.roster_global_slots.push(2);
  sourceJoined.projection.scene.observer_visibility.push({
    observer_global_slot: 0,
    candidate_global_slot: 2,
    visible: true,
  });
  sourceJoined.projection.incoming_events.configured_active_by_global_slot[2] = true;
  sourceJoined.projection.incoming_events.agent_phase_trajectories.push({
    global_slot: 2,
    public_agent_id: "2",
    transition_start: anchor(2, "transition_start", [4, 1]),
    post_charge: anchor(2, "post_charge", [4, 1]),
    successor: anchor(2, "successor", [4, 1]),
  });
  sourceJoined.projection.scene.agents[1].statuses[0].direct_source_evidence = [
    evidence,
  ];
  sourceJoined.projection.status_source_evidence.active_statuses[0].direct_source_evidence =
    [structuredClone(evidence)];
  const normalized = normalizeLiveDebuggerFrameV2(sourceJoined);
  assert.deepEqual(
    normalized.scene.agents[1].statuses[0].direct_source_evidence,
    sourceJoined.projection.scene.agents[1].statuses[0].direct_source_evidence,
  );
  assert.ok(
    Object.isFrozen(normalized.scene.agents[1].statuses[0].direct_source_evidence[0]),
  );

  const reordered = structuredClone(sourceJoined);
  const sourceEvidence =
    reordered.projection.scene.agents[1].statuses[0].direct_source_evidence[0];
  reordered.projection.scene.agents[1].statuses[0].direct_source_evidence[0] = {
    event_id: sourceEvidence.event_id,
    source_public_agent_id: sourceEvidence.source_public_agent_id,
    source_global_slot: sourceEvidence.source_global_slot,
  };
  assert.deepEqual(
    normalizeLiveDebuggerFrameV2(reordered).scene.agents[1].statuses[0]
      .direct_source_evidence,
    normalized.scene.agents[1].statuses[0].direct_source_evidence,
  );

  const leadingZeroTransition = structuredClone(sourceJoined);
  leadingZeroTransition.projection.scene.agents[1].statuses[0].direct_source_evidence[0].event_id = `${episodeId}:transition:00:event:0015`;
  leadingZeroTransition.projection.status_source_evidence.active_statuses[0].direct_source_evidence[0].event_id = `${episodeId}:transition:00:event:0015`;
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(leadingZeroTransition),
    /status source evidence/iu,
  );

  for (const mutate of [
    /** @param {any} row */ (row) => {
      delete row.event_id;
    },
    /** @param {any} row */ (row) => {
      row.extra = "secret";
    },
    /** @param {any} row */ (row) => {
      row.source_public_agent_id = "1";
    },
  ]) {
    const frame = structuredClone(sourceJoined);
    mutate(frame.projection.scene.agents[1].statuses[0].direct_source_evidence[0]);
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(frame),
      /status source evidence/iu,
    );
  }
});

test("researcher wire shells are exact rebuilt roots for Technical JSON", () => {
  for (const mutate of [
    /** @param {any} frame */ (frame) => {
      frame.projection.extra = "projection-secret";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.extra = "scene-secret";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.map.extra = "map-secret";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.agents[0].extra = "agent-secret";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.agents[0].life_state = "SECRET";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.agents[0].current_health = "9";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.ranges[0].extra = "range-secret";
    },
    /** @param {any} frame */ (frame) => {
      frame.projection.scene.ranges[0].radius = "6";
    },
  ]) {
    const frame = researcherFrame();
    mutate(frame);
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(frame),
      /researcher (?:projection|scene|map|agent|range)/iu,
    );
  }

  const obstacle = researcherFrame();
  obstacle.projection.scene.map.obstacles = [
    {
      obstacle_id: "pillar-1",
      kind: "pillar",
      center: [5, 4],
      radius: 1,
      width: null,
      height: null,
      theta: 0,
      extra: "obstacle-secret",
    },
  ];
  assert.throws(() => normalizeLiveDebuggerFrameV2(obstacle), /researcher obstacle/iu);

  const raw = researcherFrame();
  const normalized = normalizeLiveDebuggerFrameV2(raw);
  assert.notEqual(normalized.projection, raw.projection);
  assert.notEqual(normalized.projection.scene, raw.projection.scene);
  assert.notEqual(
    normalized.projection.incoming_events,
    raw.projection.incoming_events,
  );
  assert.ok(Object.isFrozen(normalized.projection));
  assert.ok(Object.isFrozen(normalized.projection.scene));
  assert.ok(Object.isFrozen(normalized.projection.scene.agents[0]));
  assert.ok(Object.isFrozen(normalized.projection.incoming_events.events[0]));
  assert.equal("alive" in normalized.projection.scene.agents[0], false);
  assert.equal("token_id" in normalized.projection.scene.aura_fields[0], false);
  assert.equal(normalized.scene.agents[0].alive, true);
  assert.equal(normalized.scene.aura_fields[0].token_id, "warrior_mitigation");
});

test("researcher event and scene normalization rejects semantic drift", () => {
  const eventFrame = researcherFrameWithEvents(v2Events());
  /** @param {any} frame @param {string} type */
  const event = (frame, type) =>
    frame.projection.incoming_events.events.find(
      (/** @type {any} */ row) => row.event_type === type,
    );
  const normalizedEventFrame = normalizeLiveDebuggerFrameV2(
    structuredClone(eventFrame),
  );
  assert.deepEqual(
    new Set(
      normalizedEventFrame.event_batch.events.map(
        (/** @type {any} */ row) => row.event_type,
      ),
    ),
    new Set(researcherEventTypesV2),
  );
  const normalizedDamage = normalizedEventFrame.event_batch.events.find(
    (/** @type {any} */ row) => row.event_type === "source_damage_output",
  );
  assert.deepEqual(normalizedDamage.mage_damage_aura_covering_emitter_global_slots, []);
  assert.deepEqual(
    normalizedDamage.warrior_mitigation_aura_covering_emitter_global_slots,
    [],
  );
  assert.equal(normalizedEventFrame.projection.scene.agents[0].life_state, "alive");
  assert.equal(
    normalizedEventFrame.projection.scene.agents[0].spawn_shield_remaining,
    0,
  );
  /** @type {Array<[string, Record<string, any>, (frame: any) => void]>} */
  const mutations = [
    [
      "negative source damage",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "source_damage_output").raw_damage_output = -1;
      },
    ],
    [
      "ability source slot-anchor mismatch",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "ability_activated").source_global_slot = 1;
      },
    ],
    [
      "wrong event anchor phase",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "ability_activated").source_anchor.phase = "successor";
      },
    ],
    [
      "status channel-ID mismatch",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "status_aged_to_zero").status_id = "hunter_trap_stun";
      },
    ],
    [
      "Charge displacement-endpoint mismatch",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "charge_phase_displacement").realized_displacement = [9, 9];
      },
    ],
    [
      "trajectory successor-scene mismatch",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        frame.projection.incoming_events.agent_phase_trajectories[0].successor.position[0] += 1;
      },
    ],
    [
      "nullable recipient-anchor mismatch",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "source_damage_output").recipient_anchor = null;
      },
    ],
    [
      "duplicate aura emitter slots",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(
          frame,
          "source_damage_output",
        ).mage_damage_aura_covering_emitter_global_slots = [0, 0];
      },
    ],
    [
      "ordinary movement displacement-endpoint mismatch",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "ordinary_movement_phase_displacement").realized_displacement = [
          2, 2,
        ];
      },
    ],
    [
      "inactive rejected actor with anchor",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "action_rejected").actor_configured_active = false;
      },
    ],
    [
      "respawn team mismatch",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "agent_respawned").team_id = 2;
      },
    ],
    [
      "event anchor drift from trajectory",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        event(frame, "cooldown_ready").agent_anchor.position[0] += 1;
      },
    ],
    [
      "respawn countdown reaches serialized period",
      eventFrame,
      /** @param {any} frame */ (frame) => {
        frame.projection.scene.respawn_waves[0].countdown_steps =
          frame.projection.scene.respawn_waves[0].period_steps;
      },
    ],
  ];
  for (const [name, source, mutate] of mutations) {
    const frame = structuredClone(source);
    mutate(frame);
    assert.throws(() => normalizeLiveDebuggerFrameV2(frame), /researcher/iu, name);
  }
});

test("researcher scene collections retain canonical cross-root joins", () => {
  /** @type {Array<[string, (frame: any) => void]>} */
  const mutations = [
    [
      "duplicate spawn-pad identity",
      /** @param {any} frame */ (frame) => {
        frame.projection.scene.spawn_pads[1] = structuredClone(
          frame.projection.scene.spawn_pads[0],
        );
      },
    ],
    [
      "opaque respawn evidence outside the incoming event axis",
      /** @param {any} frame */ (frame) => {
        frame.projection.scene.agents[0].respawned_on_incoming_transition = true;
        frame.projection.scene.agents[0].respawn_event_id = "opaque-respawn-evidence";
      },
    ],
    [
      "canonical but absent respawn evidence",
      /** @param {any} frame */ (frame) => {
        frame.projection.scene.agents[0].respawned_on_incoming_transition = true;
        frame.projection.scene.agents[0].respawn_event_id = `${transitionId}:event:9999`;
      },
    ],
  ];
  for (const [name, mutate] of mutations) {
    const frame = researcherFrame();
    mutate(frame);
    assert.throws(() => normalizeLiveDebuggerFrameV2(frame), /researcher/iu, name);
  }

  const statusOrder = researcherFrame();
  const slow = structuredClone(statusOrder.projection.scene.agents[1].statuses[0]);
  const stun = {
    ...structuredClone(slow),
    status_channel: 4,
    status_id: "hunter_trap_stun",
    family: "stun",
    source_action_component: "ultimate",
    magnitude_kind: "none",
    magnitude: null,
    breaks_on_positive_damage: true,
  };
  statusOrder.projection.scene.agents[1].statuses = [slow, stun];
  statusOrder.projection.status_source_evidence.active_statuses.push({
    recipient_global_slot: 1,
    recipient_public_agent_id: "1",
    status_channel: 4,
    status_id: "hunter_trap_stun",
    direct_source_evidence: [],
  });
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(statusOrder),
    /canonical presentation order/iu,
  );

  const auraOrder = researcherFrame();
  const secondAura = structuredClone(auraOrder.projection.scene.aura_fields[0]);
  auraOrder.projection.scene.aura_fields.push(secondAura);
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(auraOrder),
    /canonical source\/aura order/iu,
  );
});

test("researcher respawn evidence names the same agent's incoming respawn event", () => {
  const canonical = researcherFrame();
  const events = v2Events();
  const abilityTemplate = events.find(
    (/** @type {any} */ event) => event.event_type === "ability_activated",
  );
  const respawnTemplate = events.find(
    (/** @type {any} */ event) => event.event_type === "agent_respawned",
  );
  assert.ok(abilityTemplate);
  assert.ok(respawnTemplate);
  const abilityEvent = {
    ...abilityTemplate,
    event_id: `${transitionId}:event:0000`,
    ordinal: 0,
  };
  const respawnEvent = {
    ...respawnTemplate,
    event_id: `${transitionId}:event:0001`,
    ordinal: 1,
    agent_global_slot: 1,
    realized_successor_position: [3, 1],
    agent_anchor: structuredClone(successorOne),
  };
  canonical.projection.incoming_events.events = [abilityEvent, respawnEvent];
  canonical.projection.scene.incoming_event_ids = [
    abilityEvent.event_id,
    respawnEvent.event_id,
  ];
  canonical.projection.scene.agents[1].respawned_on_incoming_transition = true;
  canonical.projection.scene.agents[1].respawn_event_id = respawnEvent.event_id;
  normalizeLiveDebuggerFrameV2(structuredClone(canonical));
  const respawned = canonical.projection.scene.agents.find(
    (/** @type {any} */ agent) => agent.respawn_event_id !== null,
  );
  assert.ok(respawned);

  for (const eventId of [
    canonical.projection.incoming_events.events[0].event_id,
    respawned.respawn_event_id,
  ]) {
    const frame = structuredClone(canonical);
    const claimant =
      eventId === respawned.respawn_event_id
        ? frame.projection.scene.agents.find(
            (/** @type {any} */ agent) => agent.global_slot === 0,
          )
        : frame.projection.scene.agents.find(
            (/** @type {any} */ agent) => agent.global_slot === respawned.global_slot,
          );
    claimant.respawned_on_incoming_transition = true;
    claimant.respawn_event_id = eventId;
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(frame),
      /same agent's incoming respawn event/u,
    );
  }
});

test("researcher frame zero cannot claim incoming or respawn evidence", () => {
  const frame = researcherFrame();
  frame.frame_index = 0;
  frame.frame_id = `${episodeId}:frame:0`;
  frame.simulator_step_count = 0;
  frame.incoming_transition_index = null;
  frame.incoming_transition_id = null;
  frame.projection.scene.frame_index = 0;
  frame.projection.scene.frame_id = frame.frame_id;
  frame.projection.scene.simulator_step_count = 0;
  frame.projection.scene.incoming_transition_id = null;
  frame.projection.scene.incoming_event_ids = [];
  frame.projection.incoming_events = null;
  frame.projection.status_source_evidence.frame_index = 0;
  frame.projection.status_source_evidence.frame_id = frame.frame_id;
  frame.hud.latest_transition = null;
  assert.equal(frame.frame_index, 0);
  assert.equal(frame.projection.incoming_events, null);
  assert.deepEqual(frame.projection.scene.incoming_event_ids, []);
  normalizeLiveDebuggerFrameV2(structuredClone(frame));

  frame.projection.scene.incoming_event_ids = ["null:event:0000"];
  frame.projection.scene.agents[0].respawned_on_incoming_transition = true;
  frame.projection.scene.agents[0].respawn_event_id = "null:event:0000";
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(frame),
    /Only researcher frame zero may omit incoming events/u,
  );
});

test("researcher HUD is an exact rebuilt scene-joined root", () => {
  for (const mutate of [
    /** @param {any} frame */ (frame) => {
      frame.hud.extra = "hud-secret";
    },
    /** @param {any} frame */ (frame) => {
      frame.hud.pending_action.extra = "pending-secret";
    },
    /** @param {any} frame */ (frame) => {
      frame.hud.pending_action.summary = { secret: true };
    },
    /** @param {any} frame */ (frame) => {
      frame.hud.pending_action.target.global_slot = 9;
      frame.hud.pending_actions[0].target.global_slot = 9;
    },
    /** @param {any} frame */ (frame) => {
      frame.hud.movement_legalities[0].available = "true";
    },
    /** @param {any} frame */ (frame) => {
      frame.hud.roster_global_slots.reverse();
    },
  ]) {
    const frame = researcherFrame();
    mutate(frame);
    assert.throws(() => normalizeLiveDebuggerFrameV2(frame), /researcher/iu);
  }

  const raw = researcherFrame();
  raw.hud.candidate_legalities = [
    {
      target_action: 0,
      target: { disclosure: "target_none", global_slot: null },
      lane_0_available: true,
      lane_1_available: false,
      basic_available: false,
      ultimate_available: false,
    },
    {
      target_action: 1,
      target: { disclosure: "public", global_slot: 0 },
      lane_0_available: true,
      lane_1_available: true,
      basic_available: true,
      ultimate_available: true,
    },
    {
      target_action: 2,
      target: { disclosure: "public", global_slot: 1 },
      lane_0_available: false,
      lane_1_available: true,
      basic_available: false,
      ultimate_available: true,
    },
  ];
  const canonicalNoop = {
    move_action: 0,
    target_action: 0,
    use_ultimate_action: 0,
    target: { disclosure: "target_none", global_slot: null },
    summary: "STAY + NO COMBAT",
  };
  raw.hud.latest_transition = {
    label: "LATEST ACCEPTED RESULT",
    transition_index: 0,
    transition_id: transitionId,
    submission_kind: "interactive",
    actors: [
      {
        actor_global_slot: 0,
        submitted: structuredClone(canonicalNoop),
        accepted: structuredClone(canonicalNoop),
        movement_mask_value: true,
        pair_mask_value: true,
        movement_accepted: true,
        combat_result: "canonical_noop",
      },
    ],
  };
  raw.hud.diagnostics = [
    {
      fact_id: "simulator_step",
      label: "Simulator step",
      value: "1",
      technical: true,
    },
  ];
  const normalized = normalizeLiveDebuggerFrameV2(raw);
  assert.notEqual(normalized.hud, raw.hud);
  assert.notEqual(normalized.hud.pending_action, raw.hud.pending_action);
  assert.equal(normalized.hud.pending_action, normalized.hud.pending_actions[0]);
  assert.ok(Object.isFrozen(normalized.hud));
  assert.ok(Object.isFrozen(normalized.hud.pending_actions));
  assert.ok(Object.isFrozen(normalized.hud.pending_action.target));
  assert.ok(Object.isFrozen(normalized.hud.movement_legalities[0]));
  assert.ok(Object.isFrozen(normalized.hud.candidate_legalities[1].target));
  assert.ok(Object.isFrozen(normalized.hud.latest_transition.actors[0].submitted));
  assert.ok(Object.isFrozen(normalized.hud.diagnostics[0]));

  raw.hud.pending_action.summary = "MUTATED";
  raw.hud.pending_actions[0].summary = "MUTATED";
  raw.hud.movement_legalities[0].available = false;
  raw.hud.candidate_legalities[1].target.global_slot = 1;
  raw.hud.latest_transition.actors[0].submitted.summary = "MUTATED";
  raw.hud.diagnostics[0].value = "SECRET";
  assert.equal(normalized.hud.pending_action.summary, "STAY + BASIC → Agent ID 1");
  assert.equal(normalized.hud.movement_legalities[0].available, true);
  assert.equal(normalized.hud.candidate_legalities[1].target.global_slot, 0);
  assert.equal(
    normalized.hud.latest_transition.actors[0].submitted.summary,
    "STAY + NO COMBAT",
  );
  assert.equal(normalized.hud.diagnostics[0].value, "1");
});

test("V2 normalization preserves canonical identity and order beyond 128 events", () => {
  const raw = researcherFrame();
  const events = Array.from({ length: 160 }, (_, ordinal) => ({
    event_type: "status_aged_to_zero",
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    transition_id: transitionId,
    ordinal,
    phase_rank: 100,
    recipient_global_slot: 1,
    status_channel: 1,
    status_id: "hunter_basic_slow",
    recipient_anchor: successorOne,
  }));
  raw.projection.incoming_events.events = events;
  raw.projection.scene.incoming_event_ids = events.map(eventId);
  raw.projection.scene.agents[1].statuses = [];
  raw.projection.status_source_evidence.active_statuses = [];
  raw.projection.scene.agents[0].respawned_on_incoming_transition = false;
  raw.projection.scene.agents[0].respawn_event_id = null;
  const frame = normalizeLiveDebuggerFrameV2(raw);
  assert.equal(frame.event_batch.events.length, events.length);
  assert.deepEqual(frame.event_batch.events.map(eventId), events.map(eventId));
  assert.equal(
    frame.event_batch.events.every(
      (/** @type {Record<string, any>} */ { event_type }) =>
        event_type === "status_aged_to_zero",
    ),
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
    verbose: false,
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
          team_local_slot: 0,
          team_id: 1,
          class_id: 1,
          position: [2, 1],
          radius: 0.5,
          alive: true,
          current_health: 9,
          max_health: 10,
          effective_movement_speed: 1,
          ultimate_cooldown_remaining: 2,
          steps_until_out_of_combat: 3,
          spawn_shield_remaining: 0,
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
            effective_movement_speed: 0.75,
            ultimate_cooldown_remaining: 0,
            steps_until_out_of_combat: 2,
            status_feature_values: [0, 0, 0, 1, 1, 1, 0, 2, 0, 0, 1, 0, 0, 0],
          },
        ],
        spawn_pads: [],
        respawn_waves: [
          {
            actor_relative_team_index: 0,
            team_relation: "own",
            team_label: "Own Team",
            period_steps: 10,
            countdown_steps: 4,
          },
          {
            actor_relative_team_index: 1,
            team_relation: "opponent",
            team_label: "Opponent Team",
            period_steps: 10,
            countdown_steps: 7,
          },
        ],
      },
      next_decision_action_mask: {
        schema_id: "marl_battlegrounds.evaluation.actor_pov_action_mask",
        schema_version: 1,
        move: Array(9).fill(true),
        select_target: Array.from({ length: 11 }, (_, index) => index === 8),
        use_ultimate: [true, false],
        select_target_use_ultimate_joint: Array.from({ length: 11 }, (_, index) => [
          index === 8,
          false,
        ]),
      },
      incoming_transition_id: povTransitionId,
      incoming_cues: [
        {
          schema_id: "marl_battlegrounds.evaluation.actor_pov_cue",
          schema_version: 1,
          cue_type: "own_position_changed",
          cue_id: `${povTransitionId}:cue:0`,
          pov_transition_id: povTransitionId,
          ordinal: 0,
          start_position: [1, 1],
          successor_position: [2, 1],
        },
      ],
    },
    terminal: {
      is_sealed: false,
      terminated: false,
      truncated: false,
      reached_declared_horizon: false,
      reason: null,
    },
    hud: {
      controlled_public_agent_id: publicAgentId,
      pending_submission_scope: "controlled_actor",
      pending_action: {
        label: "PENDING / WILL SUBMIT",
        actor_public_agent_id: publicAgentId,
        move_action: 0,
        target: {
          target_action: 8,
          public_agent_id: "visible-enemy-row-2",
        },
        armed_lane: 0,
        arm_origin: "explicit",
        movement_mask_value: true,
        pair_mask_value: true,
        summary: "STAY + BASIC",
      },
      latest_transition: null,
      movement_legalities: Array.from({ length: 9 }, (_, move_action) => ({
        move_action,
        available: true,
      })),
      candidate_legalities: [
        null,
        publicAgentId,
        "ally-row-1",
        "ally-row-2",
        "ally-row-3",
        "ally-row-4",
        "enemy-row-0",
        "enemy-row-1",
        "visible-enemy-row-2",
        "enemy-row-3",
        "enemy-row-4",
      ].map((public_agent_id, target_action) => ({
        target: { target_action, public_agent_id },
        lane_0_available: target_action === 8,
        lane_1_available: false,
        basic_available: target_action === 8,
        ultimate_available: false,
      })),
      diagnostics: [],
    },
  };
}

/** @returns {any} */
function povFrameZero() {
  const frame = povFrame();
  frame.frame_index = 0;
  frame.frame_id = `${episodeId}:frame:0`;
  frame.simulator_step_count = 0;
  frame.incoming_pov_transition_id = null;
  frame.projection.scene.frame_index = 0;
  frame.projection.scene.pov_frame_id = `${episodeId}:actor-pov:agent-zero:frame:0`;
  frame.projection.scene.source_frame_id = frame.frame_id;
  frame.projection.scene.simulator_step_count = 0;
  frame.projection.incoming_transition_id = null;
  frame.projection.incoming_cues = [];
  frame.hud.latest_transition = null;
  return frame;
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
  assert.deepEqual(
    [normalized.session_id, normalized.run_generation, normalized.transition_id],
    ["live-session", 3, "evaluation-episode:actor-pov:agent-zero:transition:0"],
  );
  assert.deepEqual(normalized.scene.pending_route, {
    audience: "agent_pov",
    source_public_agent_id: "agent-zero",
    target_public_agent_id: "visible-enemy-row-2",
    target_action: 8,
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

test("POV latest-transition evidence is absent at frame zero and canonical thereafter", () => {
  const initial = povFrameZero();
  assert.equal(normalizeLiveDebuggerFrameV2(initial).hud.latest_transition, null);

  initial.hud.latest_transition = {
    label: "LATEST ACCEPTED RESULT",
    transition_index: -1,
    pov_transition_id: null,
    submission_kind: "interactive",
    actor: {},
  };
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(initial),
    /frame zero cannot carry a latest transition/u,
  );

  const emptyTransitionId = povFrame();
  emptyTransitionId.hud.latest_transition = {
    label: "LATEST ACCEPTED RESULT",
    transition_index: 0,
    pov_transition_id: "",
    submission_kind: "interactive",
    actor: {},
  };
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(emptyTransitionId),
    /latest transition ID must be a nonempty string/u,
  );
});

test("pending route normalization fails closed without a selected lane or visible endpoint", () => {
  const researcher = researcherFrame();
  researcher.hud.pending_action.armed_lane = null;
  researcher.hud.pending_actions[0].armed_lane = null;
  assert.equal(normalizeLiveDebuggerFrameV2(researcher).scene.pending_route, null);

  const pov = povFrame();
  pov.projection.scene.visible_bodies = [];
  assert.equal(normalizeLiveDebuggerFrameV2(pov).scene.pending_route, null);
});

test("pending routes require exact true staged-pair legality for both audiences", () => {
  for (const pairMaskValue of [false, null]) {
    const researcher = researcherFrame();
    researcher.hud.pending_action.pair_mask_value = pairMaskValue;
    researcher.hud.pending_actions[0].pair_mask_value = pairMaskValue;
    assert.equal(normalizeLiveDebuggerFrameV2(researcher).scene.pending_route, null);

    if (pairMaskValue === false) {
      const pov = povFrame();
      pov.projection.next_decision_action_mask.select_target_use_ultimate_joint[8][0] = false;
      pov.projection.next_decision_action_mask.select_target[8] = false;
      pov.projection.next_decision_action_mask.use_ultimate[0] = false;
      pov.hud.candidate_legalities[8].lane_0_available = false;
      pov.hud.candidate_legalities[8].basic_available = false;
      pov.hud.pending_action.pair_mask_value = false;
      assert.equal(normalizeLiveDebuggerFrameV2(pov).scene.pending_route, null);
    }
  }
});

test("live presentation authority is exact and audience-scoped", () => {
  const researcher = normalizeLiveDebuggerFrameV2(researcherFrame());
  const pov = normalizeLiveDebuggerFrameV2(povFrame());
  assert.equal(researcher.preset, "analysis");
  assert.equal(pov.preset, "analysis");
  assert.equal(researcher.show_ranges, true);
  assert.equal(researcher.verbose, false);
  assert.equal(pov.verbose, false);
  assert.equal(Object.hasOwn(pov, "show_ranges"), false);

  for (const field of ["show_ranges", "verbose"]) {
    const missing = researcherFrame();
    delete missing[field];
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(missing),
      /unknown or missing top-level fields/u,
    );
  }
  const missingPovVerbose = povFrame();
  delete missingPovVerbose.verbose;
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(missingPovVerbose),
    /unknown or missing top-level fields/u,
  );
  assert.throws(
    () => normalizeLiveDebuggerFrameV2({ ...povFrame(), show_ranges: true }),
    /unknown or missing top-level fields/u,
  );
  for (const value of [0, 1, "true", null, [], {}]) {
    for (const field of ["show_ranges", "verbose"]) {
      const malformed = researcherFrame();
      malformed[field] = value;
      assert.throws(
        () => normalizeLiveDebuggerFrameV2(malformed),
        /envelope authority is invalid/u,
      );
    }
    const malformedPov = povFrame();
    malformedPov.verbose = value;
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(malformedPov),
      /envelope authority is invalid/u,
    );
  }
  for (const malformed of [researcherFrame(), povFrame()]) {
    malformed.verbose = true;
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(malformed),
      /envelope authority is invalid/u,
    );
  }
  for (const malformed of [researcherFrame(), povFrame()]) {
    malformed.preset = "debug";
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(malformed),
      /envelope authority is invalid/u,
    );
  }
  for (const legacy of [researcherFrame(), povFrame()]) {
    legacy.preset = "presentation";
    assert.equal(normalizeLiveDebuggerFrameV2(legacy).preset, "analysis");
  }
});

test("raw researcher ScenarioMetadata is exact, typed, and never coerced", () => {
  const integerScale = researcherFrame();
  integerScale.scenario.ordinary_movement_distance_scale = 1;
  assert.equal(
    normalizeLiveDebuggerFrameV2(integerScale).scenario
      .ordinary_movement_distance_scale,
    1,
  );

  const scenarioKeys = Object.keys(researcherFrame().scenario);
  for (const key of scenarioKeys) {
    const malformed = researcherFrame();
    delete malformed.scenario[key];
    assert.throws(
      () => normalizeLiveDebuggerFrameV2(malformed),
      /unknown or missing fields/u,
    );
  }

  const wrongTypeByKey = {
    audience: 1,
    completed_frame_count: false,
    description: null,
    frame_count: 0.5,
    mode: [],
    name: true,
    next_frame_description: [],
    next_frame_index: "0",
    next_frame_label: {},
    ordinary_movement_distance_scale: [1],
    script_complete: 0,
    title: false,
  };
  assert.deepEqual(Object.keys(wrongTypeByKey).sort(), scenarioKeys.sort());
  for (const [key, value] of Object.entries(wrongTypeByKey)) {
    const malformed = researcherFrame();
    malformed.scenario[key] = value;
    assert.throws(() => normalizeLiveDebuggerFrameV2(malformed));
  }
  const extra = researcherFrame();
  extra.scenario.private_extension = true;
  assert.throws(
    () => normalizeLiveDebuggerFrameV2(extra),
    /unknown or missing fields/u,
  );

  for (const mutation of [
    { completed_frame_count: 1 },
    { ordinary_movement_distance_scale: 0.5 },
    { next_frame_index: 0 },
    { next_frame_label: "Next" },
    { next_frame_description: "Next frame" },
    { script_complete: true },
  ]) {
    const incoherent = researcherFrame();
    Object.assign(incoherent.scenario, mutation);
    assert.throws(() => normalizeLiveDebuggerFrameV2(incoherent));
  }

  const scripted = researcherFrame();
  Object.assign(scripted.scenario, {
    mode: "scripted",
    frame_count: 2,
    completed_frame_count: 0,
    next_frame_index: 0,
    next_frame_label: "Frame 1",
    next_frame_description: "Advance once.",
    script_complete: false,
  });
  assert.equal(normalizeLiveDebuggerFrameV2(scripted).scenario.mode, "scripted");
  for (const mutation of [
    { next_frame_index: 1 },
    { next_frame_label: null },
    { next_frame_description: null },
    { script_complete: true },
  ]) {
    const incoherent = structuredClone(scripted);
    Object.assign(incoherent.scenario, mutation);
    assert.throws(() => normalizeLiveDebuggerFrameV2(incoherent));
  }
  const completeScript = structuredClone(scripted);
  Object.assign(completeScript.scenario, {
    completed_frame_count: 2,
    next_frame_index: null,
    next_frame_label: null,
    next_frame_description: null,
    script_complete: true,
  });
  assert.equal(
    normalizeLiveDebuggerFrameV2(completeScript).scenario.script_complete,
    true,
  );
});

test("available scenario rows use the exact ScenarioOption boundary", () => {
  const valid = {
    name: "basic_support",
    title: "Basic support",
    description: "Scripted support sequence.",
    mode: "scripted",
    audience: "researcher",
  };
  const frame = researcherFrame();
  frame.available_scenarios = [valid];
  assert.deepEqual(normalizeLiveDebuggerFrameV2(frame).available_scenarios, [valid]);

  for (const key of Object.keys(valid)) {
    const missing = researcherFrame();
    const row = /** @type {Record<string, any>} */ ({ ...valid });
    delete row[key];
    missing.available_scenarios = [row];
    assert.throws(() => normalizeLiveDebuggerFrameV2(missing));
  }
  for (const row of [
    { ...valid, private_extension: true },
    { ...valid, name: "Basic Support" },
    { ...valid, mode: "unknown" },
    { ...valid, audience: "actor_pov" },
    { ...valid, title: null },
    { ...valid, description: [] },
  ]) {
    const malformed = researcherFrame();
    malformed.available_scenarios = [row];
    assert.throws(() => normalizeLiveDebuggerFrameV2(malformed));
  }
});

test("projection compatibility adapter accepts only exact join roots", () => {
  const live = researcherFrame();
  const adapter = {
    frame_kind: live.frame_kind,
    episode_id: live.episode_id,
    frame_index: live.frame_index,
    frame_id: live.frame_id,
    simulator_step_count: live.simulator_step_count,
    incoming_transition_id: live.incoming_transition_id,
    projection: live.projection,
    hud: live.hud,
  };
  const normalized = normalizeDebuggerAudienceProjectionV2(adapter);
  assert.equal(normalized.scene.frame_id, live.frame_id);
  assert.equal(normalized.eventBatch?.transition_id, live.incoming_transition_id);
  const forgedIdentity = structuredClone(adapter);
  forgedIdentity.frame_id = "forged-frame";
  forgedIdentity.projection.scene.frame_id = "forged-frame";
  forgedIdentity.projection.status_source_evidence.frame_id = "forged-frame";
  forgedIdentity.projection.incoming_events.successor_frame_id = "forged-frame";
  assert.throws(
    () => normalizeDebuggerAudienceProjectionV2(forgedIdentity),
    /frame identity is not canonical/u,
  );
  for (const leaked of [
    { scenario: live.scenario },
    { available_scenarios: live.available_scenarios },
    { recording: null },
    { show_ranges: true },
    { verbose: false },
  ]) {
    assert.throws(
      () => normalizeDebuggerAudienceProjectionV2({ ...adapter, ...leaked }),
      /unknown or missing fields/u,
    );
  }
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
