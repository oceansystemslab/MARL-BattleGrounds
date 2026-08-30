import assert from "node:assert/strict";
import test from "node:test";

import {
  isReplayViewerFrame,
  joinReplayFrameAndTimeline,
  normalizeReplayApiErrorV1,
  normalizeReplayArtifactFactsV1,
  normalizeReplayCommandResponseV1,
  normalizeReplayTimelineV1,
  normalizeReplayViewerFrameV1,
  validateReplayFrameContinuity,
} from "../src/replay-frame-normalizer.js";

const episodeId = "replay-episode";
const artifactId = `${episodeId}:replay`;
const digest = "a".repeat(64);

function replayReference() {
  return {
    schema_id: "marl_battlegrounds.evaluation.replay_artifact_reference",
    schema_version: 1,
    artifact_id: artifactId,
    episode_id: episodeId,
    replay_schema_version: 1,
    context_digest_sha256: digest,
    trajectory_content_digest_sha256: digest,
    canonical_digest_sha256: digest,
    canonical_byte_length: 256,
  };
}

function summary(metricReportAvailability = "missing") {
  return {
    schema_version: 1,
    replay_reference: replayReference(),
    expected_transition_count: 2,
    recorded_transition_count: 1,
    recorded_frame_count: 2,
    metric_report_availability: metricReportAvailability,
  };
}

function cursor(frameIndex = 0, choreographyGeneration = 0) {
  return {
    schema_version: 1,
    frame_index: frameIndex,
    final_frame_index: 1,
    cursor_generation: frameIndex + 4,
    choreography_generation: choreographyGeneration,
  };
}

function researcherCompletion() {
  return {
    schema_version: 1,
    episode_id: episodeId,
    completion_state: "partial",
    expected_transition_count: 2,
    validated_transition_count: 1,
    last_valid_frame_index: 1,
    last_valid_frame_id: `${episodeId}:frame:1`,
    terminated: false,
    truncated: false,
    completion_bases: [],
    end_or_failure_reason: "captured prefix",
    failure_origin: null,
  };
}

function povCompletion() {
  return {
    schema_version: 1,
    episode_id: episodeId,
    completion_state: "partial",
    expected_transition_count: 2,
    captured_transition_count: 1,
    terminated: false,
    truncated: false,
    completion_bases: [],
    public_end_or_failure_reason: "captured prefix",
  };
}

function processing() {
  return {
    schema_version: 1,
    status: "succeeded",
    processed_transition_count: 1,
    failure_stage: null,
    failure_code: null,
    attempted_transition_index: null,
  };
}

function artifactFacts(metricReportAvailability = "missing") {
  return {
    schema_version: 1,
    artifact_summary: summary(metricReportAvailability),
    completion: researcherCompletion(),
    processing: processing(),
  };
}

function selfActor() {
  return {
    global_slot: 0,
    public_agent_id: "agent-0",
    team_local_slot: 0,
    team_id: 1,
    class_id: 1,
    position: [1, 1],
    radius: 0.5,
    alive: true,
    current_health: 10,
    max_health: 10,
    effective_movement_speed: 1,
    ultimate_cooldown_remaining: 0,
    steps_until_out_of_combat: 0,
    spawn_shield_remaining: 0,
    status_feature_values: Array(14).fill(0),
  };
}

const researcherClassNames = Object.freeze([
  null,
  "Mage",
  "Warrior",
  "Hunter",
  "Rogue",
  "Priest",
]);
const researcherStatusMechanics = Object.freeze([
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

/** Return the exact five-row BattlefieldSceneV2 mechanics catalog. */
function researcherClassMechanics() {
  return [1, 2, 3, 4, 5].map((classId) => ({
    class_id: classId,
    class_name: researcherClassNames[classId],
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
    status_mechanics: researcherStatusMechanics
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

/**
 * @param {number} frameIndex
 * @returns {any}
 */
function researcherProjection(frameIndex) {
  const frameId = `${episodeId}:frame:${frameIndex}`;
  const transitionId = frameIndex === 0 ? null : `${episodeId}:transition:0`;
  const events =
    frameIndex === 0
      ? []
      : [
          {
            event_type: "action_rejected",
            event_id: `${transitionId}:event:0000`,
            transition_id: transitionId,
            ordinal: 0,
            phase_rank: 10,
            actor_global_slot: 0,
            actor_public_agent_id: "0",
            actor_configured_active: true,
            rejection_component: "movement",
            submitted_move_action: 1,
            submitted_select_target_action: 1,
            submitted_use_ultimate_action: 0,
            actor_anchor: {
              phase: "transition_start",
              global_slot: 0,
              public_agent_id: "0",
              position: [1, 1],
            },
          },
        ];
  const agents = [
    {
      global_slot: 0,
      public_agent_id: "0",
      team_id: 1,
      team_local_slot: 0,
      class_id: 1,
      position: [1, 1],
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
      statuses: [],
      aura_modifiers: [
        { aura_id: "mage_damage_amplification", multiplier: 1 },
        { aura_id: "warrior_damage_mitigation", multiplier: 1 },
      ],
    },
  ];
  return {
    schema_version: 2,
    scene: {
      schema_version: 2,
      audience: "researcher",
      audience_badge: "PRIVILEGED RESEARCHER",
      episode_id: episodeId,
      frame_index: frameIndex,
      frame_id: frameId,
      simulator_step_count: frameIndex,
      incoming_transition_id: transitionId,
      incoming_event_ids: events.map((event) => event.event_id),
      map: { width: 10, height: 8, obstacles: [] },
      agents,
      class_mechanics: researcherClassMechanics(),
      aura_fields: [],
      spawn_pads: [
        {
          team_id: 1,
          team_local_slot: 0,
          assigned_global_slot: 0,
          assigned_public_agent_id: "0",
          position: [1, 1],
        },
      ],
      respawn_waves: [
        { team_index: 0, team_id: 1, period_steps: 10, countdown_steps: 4 },
        { team_index: 1, team_id: 2, period_steps: 10, countdown_steps: 7 },
      ],
      ranges: [],
      selection: null,
      next_decision_selected_legality: null,
      observer_visibility: [],
    },
    incoming_events:
      frameIndex === 0
        ? null
        : {
            schema_version: 2,
            episode_id: episodeId,
            transition_index: 0,
            transition_id: transitionId,
            start_frame_id: `${episodeId}:frame:0`,
            successor_frame_id: frameId,
            start_simulator_step_count: 0,
            successor_simulator_step_count: frameIndex,
            public_agent_id_by_global_slot: Array.from({ length: 10 }, (_, slot) =>
              String(slot),
            ),
            configured_active_by_global_slot: [
              true,
              false,
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
                transition_start: {
                  phase: "transition_start",
                  global_slot: 0,
                  public_agent_id: "0",
                  position: [1, 1],
                },
                post_charge: {
                  phase: "post_charge",
                  global_slot: 0,
                  public_agent_id: "0",
                  position: [1, 1],
                },
                successor: {
                  phase: "successor",
                  global_slot: 0,
                  public_agent_id: "0",
                  position: [1, 1],
                },
              },
            ],
            events,
          },
    status_source_evidence: {
      schema_version: 2,
      episode_id: episodeId,
      frame_index: frameIndex,
      frame_id: frameId,
      active_statuses: [],
    },
  };
}

function researcherFrame(frameIndex = 0, choreographyGeneration = 0) {
  return {
    schema_version: 1,
    frame_kind: "researcher_replay_viewer",
    viewer_session_id: "viewer-session",
    revision: frameIndex + 3,
    artifact_summary: summary(),
    timeline_id: `${artifactId}:timeline:researcher`,
    cursor: cursor(frameIndex, choreographyGeneration),
    preset: "analysis",
    verbose: false,
    view_mode: "researcher",
    frame_id: `${episodeId}:frame:${frameIndex}`,
    simulator_step_count: frameIndex,
    incoming_transition_index: frameIndex === 0 ? null : 0,
    incoming_transition_id: frameIndex === 0 ? null : `${episodeId}:transition:0`,
    completion: researcherCompletion(),
    processing: processing(),
    show_ranges: true,
    recorded_ordinary_movement_distance_scale: 1.0,
    projection: researcherProjection(frameIndex),
  };
}

function povProjection() {
  const jointMask = Array.from({ length: 11 }, (_, target) => [target === 0, false]);
  return {
    scene: {
      schema_version: 1,
      observation_materialization: "exact_no_shared_obs_actor_input",
      audience_badge: "AGENT POV · EXACT",
      episode_id: episodeId,
      frame_index: 0,
      pov_frame_id: `${episodeId}:actor-pov:agent-0:frame:0`,
      source_frame_id: `${episodeId}:frame:0`,
      simulator_step_count: 0,
      self_actor: selfActor(),
      map: { width: 10, height: 8, obstacles: [] },
      visible_bodies: [],
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
    incoming_transition_id: null,
    incoming_cues: [],
    next_decision_action_mask: {
      schema_id: "marl_battlegrounds.evaluation.actor_pov_action_mask",
      schema_version: 1,
      move: Array(9).fill(true),
      select_target: [true, ...Array(10).fill(false)],
      use_ultimate: [true, false],
      select_target_use_ultimate_joint: jointMask,
    },
  };
}

function povFrame() {
  return {
    schema_version: 1,
    frame_kind: "actor_pov_replay_viewer",
    viewer_session_id: "viewer-session",
    revision: 3,
    artifact_facts: artifactFacts(),
    artifact_summary: summary("not_available_in_actor_pov"),
    timeline_id: `${artifactId}:timeline:actor-pov:agent-0`,
    cursor: cursor(),
    preset: "analysis",
    verbose: false,
    view_mode: "pov",
    pov_global_slot: 0,
    public_agent_id: "agent-0",
    pov_frame_id: `${episodeId}:actor-pov:agent-0:frame:0`,
    simulator_step_count: 0,
    incoming_pov_transition_id: null,
    completion: povCompletion(),
    processing_disclosure: {
      schema_version: 1,
      disclosure: "not_available_in_actor_pov",
    },
    projection: povProjection(),
  };
}

/** @param {{ended?: boolean, terminal?: boolean}} [options] @returns {any} */
function finalPovFrame({ ended = false, terminal = false } = {}) {
  /** @type {any} */
  const frame = povFrame();
  const transitionId = `${episodeId}:actor-pov:agent-0:transition:0`;
  frame.cursor = cursor(1);
  frame.pov_frame_id = `${episodeId}:actor-pov:agent-0:frame:1`;
  frame.simulator_step_count = 1;
  frame.incoming_pov_transition_id = transitionId;
  frame.projection.scene.frame_index = 1;
  frame.projection.scene.pov_frame_id = frame.pov_frame_id;
  frame.projection.scene.source_frame_id = `${episodeId}:frame:1`;
  frame.projection.scene.simulator_step_count = 1;
  frame.projection.incoming_transition_id = transitionId;
  frame.projection.incoming_cues = ended
    ? [
        {
          schema_id: "marl_battlegrounds.evaluation.actor_pov_cue",
          schema_version: 1,
          cue_id: `${transitionId}:cue:0`,
          pov_transition_id: transitionId,
          ordinal: 0,
          cue_type: "episode_ended",
          terminated: true,
          truncated: false,
          public_end_reason: "task complete",
        },
      ]
    : [];
  if (terminal) {
    frame.completion.completion_state = "complete";
    frame.completion.terminated = true;
    frame.completion.completion_bases = ["task_terminal"];
    frame.completion.public_end_or_failure_reason = "task complete";
    frame.artifact_facts.completion.completion_state = "complete";
    frame.artifact_facts.completion.terminated = true;
    frame.artifact_facts.completion.completion_bases = ["task_terminal"];
    frame.artifact_facts.completion.end_or_failure_reason = "task complete";
  }
  return frame;
}

/** @param {number} rows @param {number} columns @param {number} [value] */
function numberMatrix(rows, columns, value = 0) {
  return Array.from({ length: rows }, () => Array(columns).fill(value));
}

/** @param {number} teamId @param {number} classId @param {number} x */
function sourceUnitFeatures(teamId, classId, x) {
  const features = Array(58).fill(0);
  features[0] = x;
  features[1] = teamId;
  features[2] = 0.5;
  features[3] = teamId;
  features[4] = 1;
  features[5] = 1;
  features[6] = classId;
  features[8] = 1;
  features[12] = 10;
  features[13] = 10;
  return features;
}

function sourceProjection() {
  const sourceMaterialFrameId = `${episodeId}:shared-obs-source-material:agent-0:frame:0`;
  const disclosure = "SOURCE MATERIAL ONLY · NOT MATERIALIZED SHAREDOBS ACTOR INPUT";
  const allyIds = ["agent-0", "agent-1", "agent-2", "agent-3", "agent-4"];
  const enemyIds = ["agent-5", "agent-6", "agent-7", "agent-8", "agent-9"];
  const selfFeatures = sourceUnitFeatures(1, 1, 1);
  selfFeatures[1] = 1;
  const allyFeatures = numberMatrix(5, 58);
  allyFeatures[1] = sourceUnitFeatures(1, 2, 2);
  const enemyFeatures = numberMatrix(5, 58);
  enemyFeatures[0] = sourceUnitFeatures(2, 3, 7);
  const obstacleFeatures = numberMatrix(16, 8);
  obstacleFeatures[0] = [1, 4, 3, 1, 0, 0, 0, 1];
  obstacleFeatures[1] = [2, 6, 3, 0, 1, 2, 0.5, 1];
  const contextFeatures = Array(19).fill(0);
  contextFeatures[2] = 10;
  contextFeatures[3] = 8;
  const spawnPositions = Array.from({ length: 2 }, (_, team) =>
    Array.from({ length: 5 }, (_, slot) => [team * 6 + slot + 0.5, team + 0.5]),
  );
  const activeMasks = [
    [true, true, false, false, false],
    [true, false, false, false, false],
  ];
  const lifecycle = {
    schema_id: "marl_battlegrounds.evaluation.actor_pov_spawn_lifecycle",
    schema_version: 1,
    spawn_pad_positions_by_team: spawnPositions,
    spawn_shield_actual_durations_by_team: numberMatrix(2, 5),
    spawn_shield_configured_duration: 0,
    spawn_shield_speed: 0,
    respawn_wave_period_step_count_by_team: [5, 5],
    respawn_wave_countdowns_by_team: [0, 0],
    active_mask_by_team: activeMasks,
    alive_mask_by_team: activeMasks,
  };
  const jointMask = Array.from({ length: 11 }, (_, target) => [target === 0, false]);
  const baseSensorFrame = {
    schema_version: 1,
    observation_materialization: "source_material_only",
    episode_id: episodeId,
    public_agent_id: "agent-0",
    frame_index: 0,
    source_material_frame_id: sourceMaterialFrameId,
    source_frame_id: `${episodeId}:frame:0`,
    simulator_step_count: 0,
    self_features: selfFeatures,
    ally_unit_features: allyFeatures,
    enemy_unit_features: enemyFeatures,
    map_obstacle_features: obstacleFeatures,
    objective_features: numberMatrix(8, 12),
    context_features: contextFeatures,
    ally_visibility_mask: [false, true, false, false, false],
    enemy_visibility_mask: [true, false, false, false, false],
    previous_timestep_actions: {
      schema_id: "marl_battlegrounds.evaluation.actor_pov_previous_actions",
      schema_version: 1,
      ally_move_actions_one_hot: numberMatrix(5, 9),
      enemy_move_actions_one_hot: numberMatrix(5, 9),
      ally_select_target_actions_one_hot: numberMatrix(5, 11),
      enemy_select_target_actions_one_hot: numberMatrix(5, 11),
      ally_use_ultimate_actions_one_hot: numberMatrix(5, 2),
      enemy_use_ultimate_actions_one_hot: numberMatrix(5, 2),
    },
    spawn_lifecycle: lifecycle,
    action_mask: {
      schema_id: "marl_battlegrounds.evaluation.actor_pov_action_mask",
      schema_version: 1,
      move: [true, false, false, false, false, false, false, false, false],
      select_target: [
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
      ],
      use_ultimate: [true, false],
      select_target_use_ultimate_joint: jointMask,
    },
  };
  const spawnPads = spawnPositions.flatMap((positions, team) =>
    positions.map((position, slot) => ({
      actor_relative_team_index: team,
      team_relation: team === 0 ? "own" : "opponent",
      team_label: team === 0 ? "Own Team" : "Opponent Team",
      team_local_slot: slot,
      position,
      configured_active: activeMasks[team][slot],
      currently_alive: activeMasks[team][slot],
      spawn_shield_remaining: 0,
    })),
  );
  const availability = [...allyIds, ...enemyIds].map((publicAgentId, slot) => {
    const configuredActive = slot === 0 || slot === 1 || slot === 5;
    return {
      sensor_source_global_slot: slot,
      sensor_source_public_agent_id: publicAgentId,
      sensor_source_team_local_slot: slot % 5,
      sensor_source_configured_team_id: configuredActive ? (slot < 5 ? 1 : 2) : 0,
      sensor_source_configured_active: configuredActive,
      relation_to_recipient: !configuredActive
        ? "inactive"
        : slot === 0
          ? "self"
          : slot < 5
            ? "ally"
            : "opponent",
      base_sensor_relation_axis: slot < 5 ? "ally" : "enemy",
      base_sensor_observation_row: slot % 5,
      recorded_available: slot === 1,
    };
  });
  return {
    schema_version: 1,
    disclosure_label: disclosure,
    observation_materialization: "source_material_only",
    exact_actor_input_export_available: false,
    axis_mapping: {
      schema_id: "marl_battlegrounds.evaluation.actor_pov_axis_mapping",
      schema_version: 1,
      actor_projection_identifier: "shared_obs_source_material",
      actor_projection_version: 1,
      source_context_schema_id: "marl_battlegrounds.evaluation.episode_context",
      source_context_schema_version: 1,
      source_frame_schema_id: "marl_battlegrounds.evaluation.frame",
      source_frame_schema_version: 1,
      source_transition_schema_id: "marl_battlegrounds.evaluation.transition",
      source_transition_schema_version: 1,
      target_action_recipient_public_agent_id_by_id: [null, ...allyIds, ...enemyIds],
      ally_observation_row_public_agent_id_by_id: allyIds,
      enemy_observation_row_public_agent_id_by_id: enemyIds,
      movement_action_name_by_id: Array.from(
        { length: 9 },
        (_, index) => `move-${index}`,
      ),
      unit_direction_vector_by_movement_action: Array.from(
        { length: 9 },
        (_, index) => [index === 0 ? 0 : 1, 0],
      ),
      target_action_name_by_id: Array.from(
        { length: 11 },
        (_, index) => `target-${index}`,
      ),
      use_ultimate_action_name_by_id: ["hold", "use"],
      spawn_lifecycle_team_axis_name_by_id: ["Own Team", "Opponent Team"],
    },
    ally_observation_row_global_slot_by_id: [0, 1, 2, 3, 4],
    enemy_observation_row_global_slot_by_id: [5, 6, 7, 8, 9],
    base_sensor_frame: baseSensorFrame,
    base_sensor_scene: {
      schema_version: 1,
      audience_badge: disclosure,
      observation_materialization: "source_material_only",
      episode_id: episodeId,
      frame_index: 0,
      source_frame_id: `${episodeId}:frame:0`,
      simulator_step_count: 0,
      map: {
        width: 10,
        height: 8,
        obstacles: [
          {
            obstacle_id: "base-sensor-obstacle-0",
            kind: "pillar",
            center: [4, 3],
            radius: 1,
            width: null,
            height: null,
            theta: 0,
          },
          {
            obstacle_id: "base-sensor-obstacle-1",
            kind: "wall",
            center: [6, 3],
            radius: null,
            width: 1,
            height: 2,
            theta: 0.5,
          },
        ],
      },
      self_actor: selfActor(),
      visible_bodies: [
        {
          relation: "ally",
          observation_row: 1,
          public_agent_id: "agent-1",
          position: [2, 1],
          radius: 0.5,
          team_id: 1,
          class_id: 2,
          alive: true,
          current_health: 10,
          max_health: 10,
          effective_movement_speed: 1,
          ultimate_cooldown_remaining: 0,
          steps_until_out_of_combat: 0,
          status_feature_values: Array(14).fill(0),
        },
        {
          relation: "enemy",
          observation_row: 0,
          public_agent_id: "agent-5",
          position: [7, 2],
          radius: 0.5,
          team_id: 2,
          class_id: 3,
          alive: true,
          current_health: 10,
          max_health: 10,
          effective_movement_speed: 1,
          ultimate_cooldown_remaining: 0,
          steps_until_out_of_combat: 0,
          status_feature_values: Array(14).fill(0),
        },
      ],
      spawn_pads: spawnPads,
      respawn_waves: [0, 1].map((team) => ({
        actor_relative_team_index: team,
        team_relation: team === 0 ? "own" : "opponent",
        team_label: team === 0 ? "Own Team" : "Opponent Team",
        period_steps: 5,
        countdown_steps: 0,
      })),
    },
    incoming_transition_id: null,
    sensor_source_availability: availability,
  };
}

function sourceFrame() {
  return {
    schema_version: 1,
    frame_kind: "shared_obs_source_material_replay_viewer",
    viewer_session_id: "viewer-session",
    revision: 3,
    artifact_summary: summary(),
    timeline_id: `${artifactId}:timeline:shared-obs-source-material:agent-0`,
    cursor: cursor(),
    preset: "analysis",
    verbose: false,
    view_mode: "pov",
    selected_global_slot: 0,
    public_agent_id: "agent-0",
    observation_materialization: "source_material_only",
    source_material_frame_id: `${episodeId}:shared-obs-source-material:agent-0:frame:0`,
    source_frame_id: `${episodeId}:frame:0`,
    simulator_step_count: 0,
    incoming_transition_id: null,
    completion: researcherCompletion(),
    processing: processing(),
    projection: sourceProjection(),
  };
}

/**
 * Enumerate every object-record path in one JSON wire root. Arrays are
 * traversed because their record elements are independently strict schema
 * families, but only records receive an injected field.
 *
 * @param {unknown} value
 * @param {(string | number)[]} [path]
 * @param {Array<(string | number)[]>} [paths]
 * @returns {Array<(string | number)[]>}
 */
function recordPaths(value, path = [], paths = []) {
  if (Array.isArray(value)) {
    value.forEach((child, index) => {
      recordPaths(child, [...path, index], paths);
    });
    return paths;
  }
  if (!value || typeof value !== "object") {
    return paths;
  }
  paths.push(path);
  for (const [key, child] of Object.entries(value)) {
    recordPaths(child, [...path, key], paths);
  }
  return paths;
}

/** @param {Record<string, any>} root @param {(string | number)[]} path */
function recordAtPath(root, path) {
  return path.reduce((value, segment) => value[segment], root);
}

/** @param {(string | number)[]} path */
function displayPath(path) {
  return path.length === 0
    ? "$"
    : path.reduce(
        (label, segment) =>
          typeof segment === "number" ? `${label}[${segment}]` : `${label}.${segment}`,
        "$",
      );
}

/**
 * Assert that every array and record in a normalized graph is frozen and
 * shares no mutable container identity with its raw wire input.
 *
 * @param {unknown} raw
 * @param {unknown} normalized
 * @param {string} label
 */
function assertRecursivelyFrozenAndUnaliased(raw, normalized, label) {
  const rawContainers = new Set();
  /** @param {unknown} value */
  const collectRaw = (value) => {
    if (!value || typeof value !== "object" || rawContainers.has(value)) {
      return;
    }
    rawContainers.add(value);
    for (const child of Array.isArray(value) ? value : Object.values(value)) {
      collectRaw(child);
    }
  };
  collectRaw(raw);

  const visited = new Set();
  /** @param {unknown} value @param {string} path */
  const inspectNormalized = (value, path) => {
    if (!value || typeof value !== "object" || visited.has(value)) {
      return;
    }
    visited.add(value);
    assert.equal(Object.isFrozen(value), true, `${label} ${path} is not frozen`);
    assert.equal(
      rawContainers.has(value),
      false,
      `${label} ${path} aliases a raw wire container`,
    );
    const children = Array.isArray(value)
      ? value.map((child, index) => [index, child])
      : Object.entries(value);
    for (const [key, child] of children) {
      inspectNormalized(
        child,
        typeof key === "number" ? `${path}[${key}]` : `${path}.${key}`,
      );
    }
  };
  inspectNormalized(normalized, "$normalized");
}

/**
 * Destructively perturb every mutable raw container after normalization. This
 * deliberately changes existing scalar values as well as container shapes;
 * installed normalized bytes must remain unaffected.
 *
 * @param {unknown} value
 * @param {Set<object>} [visited]
 */
function mutateRawGraph(value, visited = new Set()) {
  if (!value || typeof value !== "object" || visited.has(value)) {
    return;
  }
  visited.add(value);
  if (Array.isArray(value)) {
    for (const child of value) {
      mutateRawGraph(child, visited);
    }
    value.push("__cp75_raw_array_mutation__");
    return;
  }
  const rawRecord = /** @type {Record<string, any>} */ (value);
  for (const [key, child] of Object.entries(rawRecord)) {
    mutateRawGraph(child, visited);
    if (child === null) {
      rawRecord[key] = "__cp75_raw_null_mutation__";
    } else if (typeof child === "string") {
      rawRecord[key] = "__cp75_raw_string_mutation__";
    } else if (typeof child === "number") {
      rawRecord[key] = child + 10_000;
    } else if (typeof child === "boolean") {
      rawRecord[key] = !child;
    }
  }
  rawRecord.__cp75_raw_record_mutation__ = true;
}

test("every replay frame and timeline record family rejects an unknown key", () => {
  /** @type {Array<readonly [string, () => Record<string, any>, (value: unknown) => Readonly<Record<string, any>>]>} */
  const cases = [
    ["researcher frame", () => researcherFrame(), normalizeReplayViewerFrameV1],
    ["Actor POV frame", () => povFrame(), normalizeReplayViewerFrameV1],
    ["researcher timeline", () => timeline("researcher"), normalizeReplayTimelineV1],
    ["Actor POV timeline", () => timeline("actor_pov"), normalizeReplayTimelineV1],
  ];

  for (const [label, factory, normalize] of cases) {
    const paths = recordPaths(factory());
    assert.ok(paths.length > 1, `${label} did not expose nested record families`);
    for (const path of paths) {
      const candidate = structuredClone(factory());
      recordAtPath(candidate, path).__cp75_unknown_field__ = "researcher-only";
      assert.throws(
        () => normalize(candidate),
        /unknown(?: or missing)? fields?/u,
        `${label} accepted an unknown key at ${displayPath(path)}`,
      );
    }
  }
});

test("supported replay audiences and timelines are recursively frozen, unaliased, and raw-mutation independent", () => {
  /** @type {Array<readonly [string, () => Record<string, any>, (value: unknown) => Readonly<Record<string, any>>]>} */
  const cases = [
    ["researcher frame", () => researcherFrame(), normalizeReplayViewerFrameV1],
    ["Actor POV frame", () => povFrame(), normalizeReplayViewerFrameV1],
    ["researcher timeline", () => timeline("researcher"), normalizeReplayTimelineV1],
    ["Actor POV timeline", () => timeline("actor_pov"), normalizeReplayTimelineV1],
  ];

  for (const [label, factory, normalize] of cases) {
    const raw = factory();
    const normalized = normalize(raw);
    const installedBytes = JSON.stringify(normalized);
    assertRecursivelyFrozenAndUnaliased(raw, normalized, label);
    mutateRawGraph(raw);
    assert.equal(
      JSON.stringify(normalized),
      installedBytes,
      `${label} changed after its raw wire graph was mutated`,
    );
  }
});

test("researcher and Actor POV replay roots normalize through separate audience boundaries", () => {
  const researcher = normalizeReplayViewerFrameV1(researcherFrame());
  const pov = normalizeReplayViewerFrameV1(povFrame());

  assert.equal(researcher.replay_audience, "researcher");
  assert.equal(researcher.scene.audience, "researcher");
  assert.equal(researcher.session_id, "viewer-session");
  assert.equal(researcher.timeline_id, `${artifactId}:timeline:researcher`);
  assert.equal(researcher.animate_incoming, false);
  assert.equal(researcher.preset, "analysis");
  assert.equal(researcher.show_ranges, true);
  assert.equal(researcher.verbose, false);
  assert.equal(researcher.recorded_ordinary_movement_distance_scale, 1);
  assert.equal(Object.hasOwn(researcher, "scenario"), false);
  assert.equal(Object.hasOwn(researcher, "available_scenarios"), false);
  assert.equal(Object.hasOwn(researcher, "recording"), false);
  assert.equal(pov.replay_audience, "actor_pov");
  assert.equal(pov.scene.audience, "agent_pov");
  assert.equal(pov.verbose, false);
  assert.equal(pov.preset, "analysis");
  assert.equal(Object.hasOwn(pov, "show_ranges"), false);
  assert.equal(Object.hasOwn(pov, "scenario"), false);
  assert.equal(Object.hasOwn(pov, "recorded_ordinary_movement_distance_scale"), false);
  assert.deepEqual(pov.processing, {
    schema_version: 1,
    disclosure: "not_available_in_actor_pov",
  });
  assert.deepEqual(pov.artifact_facts, {
    schema_version: 1,
    artifact_summary: researcher.artifact_summary,
    completion: researcher.completion,
    processing: researcher.processing,
  });
  assert.equal(
    pov.artifact_facts.artifact_summary.metric_report_availability,
    "missing",
  );
  assert.equal(Object.isFrozen(pov.artifact_facts), true);
  assert.equal(Object.hasOwn(pov, "processing"), true);
  assert.equal(Object.hasOwn(povFrame(), "processing"), false);

  for (const raw of [researcherFrame(), povFrame()]) {
    raw.verbose = true;
    assert.throws(() => normalizeReplayViewerFrameV1(raw), /scalar contract/u);
  }
  for (const raw of [researcherFrame(), povFrame()]) {
    raw.preset = "debug";
    assert.throws(() => normalizeReplayViewerFrameV1(raw), /scalar contract/u);
  }
  for (const legacy of [researcherFrame(), povFrame()]) {
    legacy.preset = "presentation";
    assert.equal(normalizeReplayViewerFrameV1(legacy).preset, "analysis");
  }
});

test("researcher replay preserves one recorded experimental movement scale", () => {
  const raw = researcherFrame();
  raw.recorded_ordinary_movement_distance_scale = 0.375;

  const normalized = normalizeReplayViewerFrameV1(raw);

  assert.equal(normalized.recorded_ordinary_movement_distance_scale, 0.375);

  for (const invalid of [0, -0.1, 1.01, Number.POSITIVE_INFINITY, Number.NaN]) {
    const candidate = researcherFrame();
    candidate.recorded_ordinary_movement_distance_scale = invalid;
    assert.throws(
      () => normalizeReplayViewerFrameV1(candidate),
      /recorded_ordinary_movement_distance_scale/u,
    );
  }

  const missing = /** @type {any} */ (researcherFrame());
  delete missing.recorded_ordinary_movement_distance_scale;
  assert.throws(
    () => normalizeReplayViewerFrameV1(missing),
    /unknown or missing fields/u,
  );

  const hiddenAudience = /** @type {Record<string, any>} */ (povFrame());
  hiddenAudience.recorded_ordinary_movement_distance_scale = 0.375;
  assert.throws(
    () => normalizeReplayViewerFrameV1(hiddenAudience),
    /unknown or missing fields/u,
  );
});

test("researcher replay installs the rebuilt frozen projection boundary", () => {
  const raw = researcherFrame(1);
  const normalized = normalizeReplayViewerFrameV1(raw);
  assert.notEqual(normalized.projection, raw.projection);
  assert.notEqual(normalized.projection.scene, raw.projection.scene);
  assert.ok(Object.isFrozen(normalized.projection));
  assert.ok(Object.isFrozen(normalized.projection.scene));
  assert.ok(Object.isFrozen(normalized.projection.incoming_events.events[0]));
  assert.equal("alive" in normalized.projection.scene.agents[0], false);

  raw.projection.scene.agents[0].current_health = 1;
  assert.equal(normalized.projection.scene.agents[0].current_health, 100);
});

test("actor POV replay processing disclosure is sanitized, frozen, and unaliased", () => {
  const wire = povFrame();
  const rawDisclosure = wire.processing_disclosure;
  const normalized = normalizeReplayViewerFrameV1(wire);

  assert.notEqual(normalized.processing_disclosure, rawDisclosure);
  assert.equal(normalized.processing_disclosure, normalized.processing);
  assert.equal(Object.isFrozen(normalized.processing_disclosure), true);

  rawDisclosure.disclosure = "HOST SECRET";
  assert.equal(JSON.stringify(normalized).includes("HOST SECRET"), false);
});

test("artifact-wide replay facts are canonical, exact, frozen, and unaliased", () => {
  const raw = artifactFacts("available");
  const normalized = normalizeReplayArtifactFactsV1(raw);

  assert.notEqual(normalized, raw);
  assert.notEqual(normalized.artifact_summary, raw.artifact_summary);
  assert.notEqual(normalized.completion, raw.completion);
  assert.notEqual(normalized.processing, raw.processing);
  assert.equal(normalized.artifact_summary.metric_report_availability, "available");
  assertRecursivelyFrozenAndUnaliased(raw, normalized, "artifact facts");

  raw.artifact_summary.metric_report_availability = "missing";
  raw.completion.end_or_failure_reason = "mutated";
  assert.equal(normalized.artifact_summary.metric_report_availability, "available");
  assert.equal(normalized.completion.end_or_failure_reason, "captured prefix");

  assert.throws(
    () =>
      normalizeReplayArtifactFactsV1({
        ...artifactFacts(),
        unknown: true,
      }),
    /unknown or missing fields/u,
  );
  assert.throws(
    () => normalizeReplayArtifactFactsV1(artifactFacts("not_available_in_actor_pov")),
    /canonical metric-report availability/u,
  );
  const mismatchedProcessing = artifactFacts();
  mismatchedProcessing.processing.processed_transition_count = 0;
  assert.throws(
    () => normalizeReplayArtifactFactsV1(mismatchedProcessing),
    /do not join/u,
  );
});

test("Shared diagnostic replay roots reject at the discriminator without reading nested source material", () => {
  const diagnosticFrame = sourceFrame();
  let projectionReads = 0;
  Object.defineProperty(diagnosticFrame, "projection", {
    configurable: true,
    enumerable: true,
    get() {
      projectionReads += 1;
      throw new Error("diagnostic projection must remain unread");
    },
  });

  assert.equal(isReplayViewerFrame(diagnosticFrame), false);
  assert.throws(
    () => normalizeReplayViewerFrameV1(diagnosticFrame),
    /Unknown replay viewer frame kind/u,
  );
  assert.throws(
    () =>
      normalizeReplayCommandResponseV1({
        schema_version: 1,
        result: "no_op",
        frame: diagnosticFrame,
        notice: null,
        animate_incoming: false,
      }),
    /Unknown replay viewer frame kind/u,
  );
  assert.throws(
    () =>
      normalizeReplayApiErrorV1({
        schema_version: 1,
        error_code: "stale_revision",
        message: "stale replay cursor",
        latest_frame: diagnosticFrame,
      }),
    /Unknown replay viewer frame kind/u,
  );
  assert.equal(projectionReads, 0);

  const diagnosticTimeline = timeline("shared_obs_source_material");
  let rowReads = 0;
  Object.defineProperty(diagnosticTimeline, "rows", {
    configurable: true,
    enumerable: true,
    get() {
      rowReads += 1;
      throw new Error("diagnostic timeline rows must remain unread");
    },
  });
  assert.throws(
    () => normalizeReplayTimelineV1(diagnosticTimeline),
    /Unknown replay timeline kind/u,
  );
  assert.equal(rowReads, 0);

  const privateTimeline = {
    schema_version: 1,
    timeline_kind: "shared_obs_agent_pov",
  };
  let privateRowReads = 0;
  Object.defineProperty(privateTimeline, "rows", {
    configurable: true,
    enumerable: true,
    get() {
      privateRowReads += 1;
      throw new Error("private Shared timeline rows must remain unread");
    },
  });
  assert.throws(
    () => normalizeReplayTimelineV1(privateTimeline),
    /Unknown replay timeline kind/u,
  );
  assert.equal(privateRowReads, 0);

  const privateSharedRoot = {
    schema_version: 1,
    frame_kind: "shared_obs_agent_pov_replay_viewer",
  };
  assert.equal(isReplayViewerFrame(privateSharedRoot), false);
  assert.throws(
    () => normalizeReplayViewerFrameV1(privateSharedRoot),
    /Unknown replay viewer frame kind/u,
  );
});

test("completion boundary requires nonempty reasons and exact failure origins", () => {
  const emptyReason = /** @type {any} */ (structuredClone(researcherFrame()));
  emptyReason.completion.end_or_failure_reason = "";
  assert.throws(
    () => normalizeReplayViewerFrameV1(emptyReason),
    /completion evidence/u,
  );

  const nonfailedOrigin = /** @type {any} */ (structuredClone(researcherFrame()));
  nonfailedOrigin.completion.failure_origin = "unrecognized-origin";
  assert.throws(() => normalizeReplayViewerFrameV1(nonfailedOrigin), /failure origin/u);

  const invalidFailedOrigin = /** @type {any} */ (structuredClone(researcherFrame()));
  invalidFailedOrigin.completion.completion_state = "failed";
  invalidFailedOrigin.completion.failure_origin = "unrecognized-origin";
  assert.throws(
    () => normalizeReplayViewerFrameV1(invalidFailedOrigin),
    /failure origin/u,
  );

  const emptyPovReason = /** @type {any} */ (structuredClone(povFrame()));
  emptyPovReason.completion.public_end_or_failure_reason = "";
  assert.throws(
    () => normalizeReplayViewerFrameV1(emptyPovReason),
    /completion evidence/u,
  );
});

test("actor POV replay ending cues join physical completion at the final cursor", () => {
  const terminal = finalPovFrame({ ended: true, terminal: true });
  const normalized = normalizeReplayViewerFrameV1(terminal);
  assert.equal(normalized.completion.terminated, true);
  assert.equal(normalized.projection.incoming_cues.at(-1).cue_type, "episode_ended");

  const endedWithoutCompletion = finalPovFrame({ ended: true });
  assert.throws(
    () => normalizeReplayViewerFrameV1(endedWithoutCompletion),
    /episode-ended cue does not join completion/u,
  );

  const completionWithoutEnded = finalPovFrame({ terminal: true });
  assert.throws(
    () => normalizeReplayViewerFrameV1(completionWithoutEnded),
    /episode-ended cue does not join completion/u,
  );

  const endedBeforeAnotherCue = finalPovFrame({ ended: true, terminal: true });
  const transitionId = endedBeforeAnotherCue.incoming_pov_transition_id;
  endedBeforeAnotherCue.projection.incoming_cues.push({
    schema_id: "marl_battlegrounds.evaluation.actor_pov_cue",
    schema_version: 1,
    cue_id: `${transitionId}:cue:1`,
    pov_transition_id: transitionId,
    ordinal: 1,
    cue_type: "own_health_changed",
    start_health: 10,
    successor_health: 9,
  });
  assert.throws(
    () => normalizeReplayViewerFrameV1(endedBeforeAnotherCue),
    /episode-ended cue does not join completion/u,
  );

  const declaredHorizonOnly = finalPovFrame();
  declaredHorizonOnly.artifact_summary.expected_transition_count = 1;
  declaredHorizonOnly.completion.expected_transition_count = 1;
  declaredHorizonOnly.completion.completion_state = "complete";
  declaredHorizonOnly.completion.completion_bases = ["declared_horizon"];
  declaredHorizonOnly.completion.public_end_or_failure_reason = null;
  declaredHorizonOnly.artifact_facts.artifact_summary.expected_transition_count = 1;
  declaredHorizonOnly.artifact_facts.completion.expected_transition_count = 1;
  declaredHorizonOnly.artifact_facts.completion.completion_state = "complete";
  declaredHorizonOnly.artifact_facts.completion.completion_bases = ["declared_horizon"];
  declaredHorizonOnly.artifact_facts.completion.end_or_failure_reason = null;
  assert.equal(
    normalizeReplayViewerFrameV1(declaredHorizonOnly).projection.incoming_cues.length,
    0,
  );
});

test("viewer launch identity isolates equal replay epochs across sessions", () => {
  const firstWireFrame = researcherFrame(1, 1);
  const secondWireFrame = {
    ...researcherFrame(1, 1),
    viewer_session_id: "viewer-session-restarted",
  };
  const first = normalizeReplayViewerFrameV1(firstWireFrame);
  const second = normalizeReplayViewerFrameV1(secondWireFrame);

  assert.equal(first.timeline_id, second.timeline_id);
  assert.equal(first.transition_id, second.transition_id);
  assert.equal(first.run_generation, second.run_generation);
  assert.notEqual(first.viewer_session_id, second.viewer_session_id);
  assert.equal(first.viewer_session_id, "viewer-session");
  assert.equal(second.viewer_session_id, "viewer-session-restarted");
});

test("replay frames reject cross-audience and response-only fields", () => {
  assert.throws(
    () =>
      normalizeReplayViewerFrameV1({
        ...researcherFrame(),
        pov_global_slot: 0,
      }),
    /unknown or missing/u,
  );
  assert.throws(
    () => normalizeReplayViewerFrameV1({ ...povFrame(), processing: processing() }),
    /unknown or missing/u,
  );
  assert.throws(
    () =>
      normalizeReplayViewerFrameV1({ ...researcherFrame(), animate_incoming: true }),
    /unknown or missing/u,
  );
  assert.throws(
    () =>
      normalizeReplayViewerFrameV1({
        ...povFrame(),
        public_agent_id: "hidden-agent",
      }),
    /does not join/u,
  );
  assert.throws(
    () =>
      normalizeReplayViewerFrameV1({
        ...povFrame(),
        artifact_summary: summary("available"),
      }),
    /hide metric-report/u,
  );
  assert.throws(
    () =>
      normalizeReplayViewerFrameV1({
        ...researcherFrame(),
        artifact_summary: summary("not_available_in_actor_pov"),
      }),
    /metric disclosure/u,
  );
  const missingArtifactFacts = povFrame();
  Reflect.deleteProperty(missingArtifactFacts, "artifact_facts");
  assert.throws(
    () => normalizeReplayViewerFrameV1(missingArtifactFacts),
    /unknown or missing/u,
  );
  const mismatchedArtifactFacts = povFrame();
  mismatchedArtifactFacts.artifact_facts.artifact_summary.replay_reference.canonical_digest_sha256 =
    "b".repeat(64);
  assert.throws(
    () => normalizeReplayViewerFrameV1(mismatchedArtifactFacts),
    /do not join/u,
  );
  const mismatchedArtifactCompletion = povFrame();
  mismatchedArtifactCompletion.artifact_facts.completion.completion_state =
    "interrupted";
  assert.throws(
    () => normalizeReplayViewerFrameV1(mismatchedArtifactCompletion),
    /do not join/u,
  );
  const staleCursor = researcherFrame();
  Reflect.deleteProperty(staleCursor.cursor, "schema_version");
  assert.throws(() => normalizeReplayViewerFrameV1(staleCursor), /unknown or missing/u);
});

test("animate_incoming exists only on a coherent command response", () => {
  const response = normalizeReplayCommandResponseV1({
    schema_version: 1,
    result: "applied",
    frame: researcherFrame(1, 1),
    notice: null,
    animate_incoming: true,
  });
  assert.equal(response.animate_incoming, true);
  assert.equal(response.frame.animate_incoming, true);
  assert.equal(response.frame.run_generation, 1);
  assert.throws(
    () =>
      normalizeReplayCommandResponseV1({
        schema_version: 1,
        result: "duplicate",
        frame: researcherFrame(1, 1),
        notice: null,
        animate_incoming: true,
      }),
    /incoherent/u,
  );
});

test("replay errors are exact, settled, and never gain animation authority", () => {
  const normalized = normalizeReplayApiErrorV1({
    schema_version: 1,
    error_code: "stale_revision",
    message: "stale",
    latest_frame: researcherFrame(1, 1),
  });
  assert.equal(normalized.latest_frame.animate_incoming, false);
  assert.throws(
    () =>
      normalizeReplayApiErrorV1({
        schema_version: 1,
        error_code: "stale_revision",
        message: "stale",
        latest_frame: researcherFrame(1, 1),
        animate_incoming: true,
      }),
    /unknown or missing/u,
  );
  assert.throws(
    () =>
      normalizeReplayApiErrorV1({
        schema_version: 1,
        error_code: "not_a_replay_error",
        message: "invalid",
        latest_frame: null,
      }),
    /scalar contract/u,
  );
});

/**
 * @param {string} kind
 * @returns {any}
 */
function timeline(kind) {
  const timelineId =
    kind === "researcher"
      ? `${artifactId}:timeline:researcher`
      : kind === "actor_pov"
        ? `${artifactId}:timeline:actor-pov:agent-0`
        : `${artifactId}:timeline:shared-obs-source-material:agent-0`;
  const common = {
    schema_version: 1,
    timeline_kind: kind,
    timeline_id: timelineId,
    artifact_summary:
      kind === "actor_pov" ? summary("not_available_in_actor_pov") : summary(),
    final_frame_index: 1,
    completion: kind === "actor_pov" ? povCompletion() : researcherCompletion(),
  };
  if (kind === "researcher") {
    return {
      ...common,
      rows: [
        {
          frame_index: 0,
          frame_id: `${episodeId}:frame:0`,
          simulator_step_count: 0,
          incoming_transition_id: null,
          incoming_event_count: 0,
          endpoint_kind: "none",
        },
        {
          frame_index: 1,
          frame_id: `${episodeId}:frame:1`,
          simulator_step_count: 1,
          incoming_transition_id: `${episodeId}:transition:0`,
          incoming_event_count: 1,
          endpoint_kind: "captured_prefix",
        },
      ],
    };
  }
  if (kind === "actor_pov") {
    return {
      ...common,
      pov_global_slot: 0,
      public_agent_id: "agent-0",
      rows: [0, 1].map((frameIndex) => ({
        frame_index: frameIndex,
        pov_frame_id: `${episodeId}:actor-pov:agent-0:frame:${frameIndex}`,
        simulator_step_count: frameIndex,
        incoming_pov_transition_id:
          frameIndex === 0 ? null : `${episodeId}:actor-pov:agent-0:transition:0`,
        incoming_cue_count: frameIndex,
        endpoint_kind: frameIndex === 0 ? "none" : "captured_prefix",
      })),
    };
  }
  return {
    ...common,
    selected_global_slot: 0,
    public_agent_id: "agent-0",
    observation_materialization: "source_material_only",
    rows: [0, 1].map((frameIndex) => ({
      frame_index: frameIndex,
      source_material_frame_id: `${episodeId}:shared-obs-source-material:agent-0:frame:${frameIndex}`,
      simulator_step_count: frameIndex,
      incoming_transition_id: frameIndex === 0 ? null : `${episodeId}:transition:0`,
      endpoint_kind: frameIndex === 0 ? "none" : "captured_prefix",
    })),
  };
}

test("timeline roots preserve their audience-specific identities and counts", () => {
  for (const kind of ["researcher", "actor_pov"]) {
    const normalized = normalizeReplayTimelineV1(timeline(kind));
    assert.equal(normalized.timeline_kind, kind);
    assert.equal(normalized.rows.length, 2);
    assert.equal(normalized.rows[0].endpoint_kind, "none");
    assert.equal(normalized.rows[1].endpoint_kind, "captured_prefix");
  }
});

test("independently fetched timelines reject audience races and artifact drift", () => {
  const frame = normalizeReplayViewerFrameV1(researcherFrame());
  const researcherTimeline = normalizeReplayTimelineV1(timeline("researcher"));
  assert.equal(
    joinReplayFrameAndTimeline(frame, researcherTimeline),
    researcherTimeline,
  );
  assert.throws(
    () =>
      joinReplayFrameAndTimeline(
        frame,
        normalizeReplayTimelineV1(timeline("actor_pov")),
      ),
    /current audience frame/u,
  );
  const drifted = timeline("researcher");
  drifted.artifact_summary.replay_reference.canonical_digest_sha256 = "b".repeat(64);
  assert.throws(
    () => joinReplayFrameAndTimeline(frame, normalizeReplayTimelineV1(drifted)),
    /current audience frame/u,
  );

  const sidecarDrift = timeline("researcher");
  sidecarDrift.artifact_summary.metric_report_availability = "available";
  assert.throws(
    () => joinReplayFrameAndTimeline(frame, normalizeReplayTimelineV1(sidecarDrift)),
    /current audience frame/u,
  );

  const epochDrift = timeline("researcher");
  epochDrift.rows[0].simulator_step_count = 7;
  epochDrift.rows[1].simulator_step_count = 8;
  assert.throws(
    () => joinReplayFrameAndTimeline(frame, normalizeReplayTimelineV1(epochDrift)),
    /current audience frame/u,
  );

  const completionDrift = timeline("researcher");
  completionDrift.completion.end_or_failure_reason = "different captured prefix";
  assert.throws(
    () => joinReplayFrameAndTimeline(frame, normalizeReplayTimelineV1(completionDrift)),
    /current audience frame/u,
  );

  const povSlotDrift = timeline("actor_pov");
  povSlotDrift.pov_global_slot = 1;
  assert.throws(
    () =>
      joinReplayFrameAndTimeline(
        normalizeReplayViewerFrameV1(povFrame()),
        normalizeReplayTimelineV1(povSlotDrift),
      ),
    /current audience frame/u,
  );
});

test("command candidates pin launch, artifact, final bound, and revision before install", () => {
  const previous = normalizeReplayViewerFrameV1(researcherFrame());
  const applied = normalizeReplayCommandResponseV1({
    schema_version: 1,
    result: "applied",
    frame: researcherFrame(1, 1),
    notice: null,
    animate_incoming: true,
  }).frame;
  assert.equal(validateReplayFrameContinuity(previous, applied, "applied"), applied);

  const povAudienceCandidate = povFrame();
  povAudienceCandidate.revision = previous.revision + 1;
  const normalizedPovAudienceCandidate = normalizeReplayCommandResponseV1({
    schema_version: 1,
    result: "applied",
    frame: povAudienceCandidate,
    notice: null,
    animate_incoming: false,
  }).frame;
  assert.equal(
    validateReplayFrameContinuity(previous, normalizedPovAudienceCandidate, "applied"),
    normalizedPovAudienceCandidate,
  );

  const artifactFactsDrift = povFrame();
  artifactFactsDrift.revision = previous.revision + 1;
  artifactFactsDrift.artifact_facts.artifact_summary.metric_report_availability =
    "available";
  const normalizedArtifactFactsDrift = normalizeReplayCommandResponseV1({
    schema_version: 1,
    result: "applied",
    frame: artifactFactsDrift,
    notice: null,
    animate_incoming: false,
  }).frame;
  assert.throws(
    () =>
      validateReplayFrameContinuity(previous, normalizedArtifactFactsDrift, "applied"),
    /continuity/u,
  );

  const interleavedDuplicateCandidate = povFrame();
  interleavedDuplicateCandidate.revision = previous.revision + 3;
  const normalizedInterleavedDuplicate = normalizeReplayCommandResponseV1({
    schema_version: 1,
    result: "duplicate",
    frame: interleavedDuplicateCandidate,
    notice: "Command already processed; current frame returned.",
    animate_incoming: false,
  }).frame;
  assert.equal(
    validateReplayFrameContinuity(
      previous,
      normalizedInterleavedDuplicate,
      "duplicate",
    ),
    normalizedInterleavedDuplicate,
  );

  const regressedStaleGeneration = researcherFrame();
  regressedStaleGeneration.revision = previous.revision + 1;
  regressedStaleGeneration.cursor.cursor_generation =
    previous.cursor.cursor_generation - 1;
  assert.throws(
    () =>
      validateReplayFrameContinuity(
        previous,
        normalizeReplayViewerFrameV1(regressedStaleGeneration),
        "stale_resync",
      ),
    /continuity/u,
  );

  const completionDrift = researcherFrame(1, 1);
  completionDrift.completion.end_or_failure_reason = "different captured prefix";
  const normalizedCompletionDrift = normalizeReplayCommandResponseV1({
    schema_version: 1,
    result: "applied",
    frame: completionDrift,
    notice: null,
    animate_incoming: true,
  }).frame;
  const matchingCompletionTimeline = timeline("researcher");
  matchingCompletionTimeline.completion.end_or_failure_reason =
    "different captured prefix";
  joinReplayFrameAndTimeline(
    normalizedCompletionDrift,
    normalizeReplayTimelineV1(matchingCompletionTimeline),
  );
  assert.throws(
    () => validateReplayFrameContinuity(previous, normalizedCompletionDrift, "applied"),
    /continuity/u,
  );

  const processingDrift = /** @type {any} */ (researcherFrame(1, 1));
  processingDrift.processing = {
    schema_version: 1,
    status: "failed",
    processed_transition_count: 0,
    failure_stage: "initial_validation",
    failure_code: "different-processing-result",
    attempted_transition_index: null,
  };
  const normalizedProcessingDrift = normalizeReplayCommandResponseV1({
    schema_version: 1,
    result: "applied",
    frame: processingDrift,
    notice: null,
    animate_incoming: true,
  }).frame;
  joinReplayFrameAndTimeline(
    normalizedProcessingDrift,
    normalizeReplayTimelineV1(timeline("researcher")),
  );
  assert.throws(
    () => validateReplayFrameContinuity(previous, normalizedProcessingDrift, "applied"),
    /continuity/u,
  );

  const sessionSwap = researcherFrame(1, 1);
  sessionSwap.viewer_session_id = "different-viewer-launch";
  assert.throws(
    () =>
      validateReplayFrameContinuity(
        previous,
        normalizeReplayCommandResponseV1({
          schema_version: 1,
          result: "applied",
          frame: sessionSwap,
          notice: null,
          animate_incoming: true,
        }).frame,
        "applied",
      ),
    /continuity/u,
  );

  const digestSwap = researcherFrame(1, 1);
  digestSwap.artifact_summary.replay_reference.canonical_digest_sha256 = "b".repeat(64);
  assert.throws(
    () =>
      validateReplayFrameContinuity(
        previous,
        normalizeReplayCommandResponseV1({
          schema_version: 1,
          result: "applied",
          frame: digestSwap,
          notice: null,
          animate_incoming: true,
        }).frame,
        "applied",
      ),
    /continuity/u,
  );

  const sidecarSwap = researcherFrame(1, 1);
  sidecarSwap.artifact_summary.metric_report_availability = "available";
  assert.throws(
    () =>
      validateReplayFrameContinuity(
        previous,
        normalizeReplayCommandResponseV1({
          schema_version: 1,
          result: "applied",
          frame: sidecarSwap,
          notice: null,
          animate_incoming: true,
        }).frame,
        "applied",
      ),
    /continuity/u,
  );

  const revisionJump = researcherFrame(1, 1);
  revisionJump.revision = 99;
  assert.throws(
    () =>
      validateReplayFrameContinuity(
        previous,
        normalizeReplayCommandResponseV1({
          schema_version: 1,
          result: "applied",
          frame: revisionJump,
          notice: null,
          animate_incoming: true,
        }).frame,
        "applied",
      ),
    /continuity/u,
  );
});

test("failed initial and audience timeline joins retain the prior atomic candidate", () => {
  /** @type {{frame: Readonly<Record<string, any>> | null, timeline: Readonly<Record<string, any>> | null}} */
  const initial = Object.freeze({ frame: null, timeline: null });
  /** @type {{frame: Readonly<Record<string, any>> | null, timeline: Readonly<Record<string, any>> | null}} */
  let installed = initial;
  assert.throws(() => {
    const frame = normalizeReplayViewerFrameV1(researcherFrame());
    const candidateTimeline = normalizeReplayTimelineV1(timeline("actor_pov"));
    const joined = joinReplayFrameAndTimeline(frame, candidateTimeline);
    installed = Object.freeze({ frame, timeline: joined });
  }, /current audience frame/u);
  assert.equal(installed, initial);

  const previous = Object.freeze({
    frame: normalizeReplayViewerFrameV1(researcherFrame()),
    timeline: normalizeReplayTimelineV1(timeline("researcher")),
  });
  installed = previous;
  assert.throws(() => {
    const frame = normalizeReplayViewerFrameV1(povFrame());
    const wrongTimeline = normalizeReplayTimelineV1(timeline("researcher"));
    const joined = joinReplayFrameAndTimeline(frame, wrongTimeline);
    installed = Object.freeze({ frame, timeline: joined });
  }, /current audience frame/u);
  assert.equal(installed, previous);
});

test("timeline normalization rejects privilege mixing, gaps, and early endpoints", () => {
  assert.throws(
    () =>
      normalizeReplayTimelineV1({ ...timeline("actor_pov"), processing: processing() }),
    /unknown or missing/u,
  );
  const missing = timeline("researcher");
  missing.rows = [missing.rows[0]];
  assert.throws(() => normalizeReplayTimelineV1(missing), /every captured frame/u);
  const earlyEndpoint = timeline("researcher");
  earlyEndpoint.rows[0].endpoint_kind = "captured_prefix";
  assert.throws(() => normalizeReplayTimelineV1(earlyEndpoint), /order or endpoint/u);
});

test("replay-kind detection never aliases live frames", () => {
  assert.equal(isReplayViewerFrame(researcherFrame()), true);
  assert.equal(
    isReplayViewerFrame({ schema_version: 2, frame_kind: "researcher_live_debugger" }),
    false,
  );
  assert.equal(isReplayViewerFrame(null), false);
});
