import { CANONICAL_STATUS_ORDER, statusTokenIdFromCatalogId } from "./vocabulary.js";

const RESEARCHER_EVENT_TYPES_V2 = new Set([
  "action_rejected",
  "ability_activated",
  "source_damage_output",
  "source_healing_output",
  "recipient_health_resolution",
  "combat_countdown_reset",
  "agent_left_combat",
  "health_regenerated",
  "cooldown_started",
  "cooldown_ready",
  "charge_phase_displacement",
  "ordinary_movement_phase_displacement",
  "agent_died",
  "lethal_damage_contribution",
  "status_aged_to_zero",
  "status_broken_by_damage",
  "status_applied",
  "status_refreshed_or_extended",
  "status_cleared_by_new_death",
  "spawn_shield_expired",
  "respawn_wave_occurred",
  "agent_respawned",
  "team_deathmatch_score_changed",
  "team_deathmatch_completed",
]);

const POV_CUE_TYPES_V1 = new Set([
  "own_action_outcome",
  "own_position_changed",
  "own_health_changed",
  "own_status_changed",
  "own_cooldown_changed",
  "own_lifecycle_changed",
  "visible_body_observation_changed",
  "episode_ended",
]);

const RECORDING_STATUS_KEYS_V1 = Object.freeze([
  "captured_transition_count",
  "completion_reason",
  "completion_state",
  "discard_available",
  "expected_transition_count",
  "finish_available",
  "lifecycle",
  "persistence_error_code",
  "restart_fenced",
  "retry_available",
  "review_available",
  "save_as_available",
  "schema_version",
]);

const RECORDING_LIFECYCLES_V1 = new Set([
  "recording",
  "sealed",
  "finalized_unsaved",
  "persistence_failed",
  "saved",
  "reviewing",
  "discarded",
]);

const RECORDING_COMPLETION_STATES_V1 = new Set([
  "complete",
  "partial",
  "interrupted",
  "failed",
]);

const RECORDING_PERSISTENCE_ERRORS_V1 = new Set([
  "target_unavailable",
  "publication_failed",
  "verification_failed",
]);

const SCENARIO_OPTION_KEYS_V1 = Object.freeze([
  "audience",
  "description",
  "mode",
  "name",
  "title",
]);

const SCENARIO_METADATA_KEYS_V1 = Object.freeze([
  "audience",
  "completed_frame_count",
  "description",
  "frame_count",
  "mode",
  "name",
  "next_frame_description",
  "next_frame_index",
  "next_frame_label",
  "ordinary_movement_distance_scale",
  "script_complete",
  "title",
]);

const COMBAT_CONFIGURATION_KEYS_V1 = Object.freeze([
  "execution_information_mode",
  "team_a_controller",
  "team_b_controller",
]);

const RESEARCHER_LIVE_FRAME_KEYS_V2 = Object.freeze([
  "available_scenarios",
  "combat_configuration",
  "episode_id",
  "frame_id",
  "frame_index",
  "frame_kind",
  "hud",
  "incoming_transition_id",
  "incoming_transition_index",
  "preset",
  "projection",
  "recording",
  "revision",
  "run_generation",
  "scenario",
  "schema_version",
  "session_id",
  "show_ranges",
  "simulator_step_count",
  "terminal",
  "verbose",
  "view_mode",
]);

const ACTOR_POV_LIVE_FRAME_KEYS_V2 = Object.freeze([
  "combat_configuration",
  "episode_id",
  "frame_id",
  "frame_index",
  "frame_kind",
  "hud",
  "incoming_pov_transition_id",
  "preset",
  "projection",
  "recording",
  "revision",
  "run_generation",
  "schema_version",
  "session_id",
  "simulator_step_count",
  "terminal",
  "verbose",
  "view_mode",
]);

const SHARED_OBS_AGENT_POV_LIVE_FRAME_KEYS_V2 = Object.freeze([
  "combat_configuration",
  "episode_id",
  "frame_id",
  "frame_index",
  "frame_kind",
  "incoming_recipient_transition_id",
  "pending_submission_scope",
  "preset",
  "recipient_frame_id",
  "recipient_public_agent_id",
  "recording",
  "revision",
  "run_generation",
  "schema_version",
  "session_id",
  "simulator_step_count",
  "terminal",
  "verbose",
  "view_mode",
]);

const RESEARCHER_PROJECTION_ADAPTER_KEYS = Object.freeze([
  "episode_id",
  "frame_id",
  "frame_index",
  "frame_kind",
  "hud",
  "incoming_transition_id",
  "projection",
  "simulator_step_count",
]);

const ACTOR_POV_PROJECTION_ADAPTER_KEYS = Object.freeze([
  "episode_id",
  "frame_id",
  "frame_index",
  "frame_kind",
  "hud",
  "incoming_pov_transition_id",
  "projection",
  "simulator_step_count",
]);

// ActorPovSelfSceneV1 retains observation columns 15..28 exactly.  Only the
// nine duration columns denote durable status presence; multiplier/fraction
// columns remain policy input but must not be misread as additional statuses.
// Keep this table in the renderer's canonical presentation order while
// preserving each absolute V1 feature index as recipient-visible evidence.
const POV_STATUS_DURATION_FEATURES_V1 = Object.freeze([
  Object.freeze({
    offset: 6,
    featureIndex: 21,
    tokenId: "stun_warrior_charge",
    effectClassId: 2,
  }),
  Object.freeze({
    offset: 7,
    featureIndex: 22,
    tokenId: "stun_hunter_trap",
    effectClassId: 3,
  }),
  Object.freeze({
    offset: 8,
    featureIndex: 23,
    tokenId: "stun_rogue_poison",
    effectClassId: 4,
  }),
  Object.freeze({
    offset: 0,
    featureIndex: 15,
    tokenId: "slow_warrior_charge",
    effectClassId: 2,
  }),
  Object.freeze({
    offset: 1,
    featureIndex: 16,
    tokenId: "slow_hunter_basic",
    effectClassId: 3,
  }),
  Object.freeze({
    offset: 2,
    featureIndex: 17,
    tokenId: "slow_rogue_poison",
    effectClassId: 4,
  }),
  Object.freeze({
    offset: 9,
    featureIndex: 24,
    tokenId: "anti_heal_rogue_poison",
    effectClassId: 4,
  }),
  Object.freeze({
    offset: 12,
    featureIndex: 27,
    tokenId: "priest_freedom",
    effectClassId: 5,
  }),
  Object.freeze({
    offset: 11,
    featureIndex: 26,
    tokenId: "mage_burst",
    effectClassId: 1,
  }),
]);

/** @type {Readonly<Record<string, string>>} */
const AURA_TOKEN_BY_ID = Object.freeze({
  mage_damage_amplification: "mage_amplification",
  warrior_damage_mitigation: "warrior_mitigation",
});
const CATALOG_STATUS_ID_BY_CHANNEL_V1 = Object.freeze([
  "warrior_charge_slow",
  "hunter_basic_slow",
  "rogue_poison_slow",
  "warrior_charge_stun",
  "hunter_trap_stun",
  "rogue_poison_stun",
  "rogue_poison_anti_heal",
  "mage_burst_damage_amplification",
  "priest_blessing_of_freedom_movement_floor",
]);
const STATUS_SOURCE_CLASS_BY_CHANNEL_V1 = Object.freeze([2, 3, 4, 2, 3, 4, 4, 1, 5]);
const AURA_SOURCE_CLASS_BY_ID_V1 = Object.freeze({
  mage_damage_amplification: 1,
  warrior_damage_mitigation: 2,
});
const CLASS_NAME_BY_ID_V1 = Object.freeze([
  null,
  "Mage",
  "Warrior",
  "Hunter",
  "Rogue",
  "Priest",
]);

const ACTOR_POV_PROJECTION_KEYS_V1 = Object.freeze([
  "incoming_cues",
  "incoming_transition_id",
  "next_decision_action_mask",
  "scene",
]);
const ACTOR_POV_SCENE_KEYS_V1 = Object.freeze([
  "audience_badge",
  "episode_id",
  "frame_index",
  "map",
  "observation_materialization",
  "pov_frame_id",
  "respawn_waves",
  "schema_version",
  "self_actor",
  "simulator_step_count",
  "source_frame_id",
  "spawn_pads",
  "visible_bodies",
]);
const ACTOR_POV_SELF_KEYS_V1 = Object.freeze([
  "alive",
  "class_id",
  "current_health",
  "effective_movement_speed",
  "global_slot",
  "max_health",
  "position",
  "public_agent_id",
  "radius",
  "spawn_shield_remaining",
  "status_feature_values",
  "steps_until_out_of_combat",
  "team_id",
  "team_local_slot",
  "ultimate_cooldown_remaining",
]);
const ACTOR_POV_BODY_KEYS_V1 = Object.freeze([
  "alive",
  "class_id",
  "current_health",
  "effective_movement_speed",
  "max_health",
  "observation_row",
  "position",
  "public_agent_id",
  "radius",
  "relation",
  "status_feature_values",
  "steps_until_out_of_combat",
  "team_id",
  "ultimate_cooldown_remaining",
]);
const ACTOR_POV_SPAWN_PAD_KEYS_V1 = Object.freeze([
  "actor_relative_team_index",
  "configured_active",
  "currently_alive",
  "position",
  "spawn_shield_remaining",
  "team_label",
  "team_local_slot",
  "team_relation",
]);
const ACTOR_POV_RESPAWN_WAVE_KEYS_V1 = Object.freeze([
  "actor_relative_team_index",
  "countdown_steps",
  "period_steps",
  "team_label",
  "team_relation",
]);
const ACTOR_POV_MAP_KEYS_V1 = Object.freeze(["height", "obstacles", "width"]);
const ACTOR_POV_OBSTACLE_KEYS_V1 = Object.freeze([
  "center",
  "height",
  "kind",
  "obstacle_id",
  "radius",
  "theta",
  "width",
]);
const ACTOR_POV_ACTION_MASK_KEYS_V1 = Object.freeze([
  "move",
  "schema_id",
  "schema_version",
  "select_target",
  "select_target_use_ultimate_joint",
  "use_ultimate",
]);
const ACTOR_POV_CUE_BASE_KEYS_V1 = Object.freeze([
  "cue_id",
  "cue_type",
  "ordinal",
  "pov_transition_id",
  "schema_id",
  "schema_version",
]);
const ACTOR_POV_CUE_SUFFIX_KEYS_V1 = Object.freeze({
  own_action_outcome: Object.freeze(["outcome"]),
  own_position_changed: Object.freeze(["start_position", "successor_position"]),
  own_health_changed: Object.freeze(["start_health", "successor_health"]),
  own_status_changed: Object.freeze([
    "changed_feature_indices",
    "start_values",
    "successor_values",
  ]),
  own_cooldown_changed: Object.freeze([
    "start_remaining_ticks",
    "successor_remaining_ticks",
  ]),
  own_lifecycle_changed: Object.freeze([
    "start_active",
    "start_alive",
    "start_spawn_shield_remaining_ticks",
    "successor_active",
    "successor_alive",
    "successor_spawn_shield_remaining_ticks",
  ]),
  visible_body_observation_changed: Object.freeze([
    "observation_row",
    "observed_payload_changed",
    "relation",
    "start_visible",
    "successor_visible",
  ]),
  episode_ended: Object.freeze(["public_end_reason", "terminated", "truncated"]),
});
const TERMINAL_STATE_KEYS_V2 = Object.freeze([
  "is_sealed",
  "reached_declared_horizon",
  "reason",
  "terminated",
  "truncated",
]);
const ACTOR_POV_HUD_KEYS_V1 = Object.freeze([
  "candidate_legalities",
  "controlled_public_agent_id",
  "diagnostics",
  "latest_transition",
  "movement_legalities",
  "pending_action",
  "pending_submission_scope",
]);
const ACTOR_POV_TARGET_KEYS_V1 = Object.freeze(["public_agent_id", "target_action"]);
const ACTOR_POV_PENDING_KEYS_V1 = Object.freeze([
  "actor_public_agent_id",
  "arm_origin",
  "armed_lane",
  "label",
  "move_action",
  "movement_mask_value",
  "pair_mask_value",
  "summary",
  "target",
]);
const ACTOR_POV_MOVEMENT_LEGALITY_KEYS_V1 = Object.freeze(["available", "move_action"]);
const ACTOR_POV_CANDIDATE_KEYS_V1 = Object.freeze([
  "basic_available",
  "lane_0_available",
  "lane_1_available",
  "target",
  "ultimate_available",
]);
const ACTOR_POV_LATEST_KEYS_V1 = Object.freeze([
  "actor",
  "label",
  "pov_transition_id",
  "submission_kind",
  "transition_index",
]);
const ACTOR_POV_RESULT_KEYS_V1 = Object.freeze([
  "accepted",
  "actor_public_agent_id",
  "combat_pair_rejected",
  "combat_result",
  "movement_accepted",
  "movement_rejected",
  "submitted",
  "submitted_tuple_is_out_of_domain",
]);
const ACTOR_POV_ACTION_KEYS_V1 = Object.freeze([
  "move_action",
  "summary",
  "target",
  "use_ultimate_action",
]);
const DIAGNOSTIC_FACT_KEYS_V1 = Object.freeze([
  "fact_id",
  "label",
  "technical",
  "value",
]);

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * @param {unknown} value
 * @param {string} message
 * @returns {Record<string, any>}
 */
function requireRecord(value, message) {
  if (!isRecord(value)) {
    throw new TypeError(message);
  }
  return value;
}

/**
 * @param {Record<string, any>} value
 * @param {readonly string[]} expected
 * @param {string} message
 */
function requireExactKeys(value, expected, message) {
  const actual = Object.keys(value).sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new TypeError(message);
  }
}

/** @param {unknown} value */
function normalizeCombatConfigurationV1(value) {
  const configuration = requireRecord(
    value,
    "Live combat configuration must be an object.",
  );
  requireExactKeys(
    configuration,
    COMBAT_CONFIGURATION_KEYS_V1,
    "Live combat configuration has unknown or missing fields.",
  );
  if (
    !["manual", "scripted_tdm"].includes(configuration.team_a_controller) ||
    !["manual", "scripted_tdm"].includes(configuration.team_b_controller) ||
    !["shared_obs", "no_shared_obs"].includes(configuration.execution_information_mode)
  ) {
    throw new TypeError("Live combat configuration is invalid.");
  }
  return Object.freeze({
    team_a_controller: configuration.team_a_controller,
    team_b_controller: configuration.team_b_controller,
    execution_information_mode: configuration.execution_information_mode,
  });
}

/**
 * Validate one raw scenario menu row before any presentation code consumes it.
 *
 * @param {unknown} value
 * @param {string} label
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeScenarioOptionV1(value, label) {
  const option = requireRecord(value, `${label} must be an object.`);
  requireExactKeys(
    option,
    SCENARIO_OPTION_KEYS_V1,
    `${label} has unknown or missing fields.`,
  );
  if (
    typeof option.name !== "string" ||
    option.name.length < 1 ||
    option.name.length > 64 ||
    !/^[a-z0-9_]+$/u.test(option.name) ||
    typeof option.title !== "string" ||
    typeof option.description !== "string" ||
    !["interactive", "scripted"].includes(option.mode) ||
    !["researcher", "stress"].includes(option.audience)
  ) {
    throw new TypeError(`${label} has invalid field types or values.`);
  }
  return Object.freeze({
    name: option.name,
    title: option.title,
    description: option.description,
    mode: option.mode,
    audience: option.audience,
  });
}

/**
 * Validate exact raw ScenarioMetadataV1, including cursor coherence, before
 * exposing the same names as presentation aliases. JSON integer-valued
 * numbers remain numbers in JavaScript and are valid; coercion is forbidden.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeScenarioMetadataV1(value) {
  const scenario = requireRecord(
    value,
    "Live researcher scenario metadata must be an object.",
  );
  requireExactKeys(
    scenario,
    SCENARIO_METADATA_KEYS_V1,
    "Live researcher scenario metadata has unknown or missing fields.",
  );
  const option = normalizeScenarioOptionV1(
    {
      name: scenario.name,
      title: scenario.title,
      description: scenario.description,
      mode: scenario.mode,
      audience: scenario.audience,
    },
    "Live researcher scenario option",
  );
  if (
    typeof scenario.ordinary_movement_distance_scale !== "number" ||
    !Number.isFinite(scenario.ordinary_movement_distance_scale) ||
    scenario.ordinary_movement_distance_scale !== 1 ||
    !Number.isInteger(scenario.completed_frame_count) ||
    scenario.completed_frame_count < 0 ||
    !Number.isInteger(scenario.frame_count) ||
    scenario.frame_count < 0 ||
    scenario.completed_frame_count > scenario.frame_count ||
    typeof scenario.script_complete !== "boolean"
  ) {
    throw new TypeError("Live researcher scenario metadata is invalid.");
  }
  const hasNext = scenario.next_frame_index !== null;
  if (
    (hasNext &&
      (!Number.isInteger(scenario.next_frame_index) ||
        scenario.next_frame_index < 0)) ||
    (!hasNext && scenario.next_frame_index !== null) ||
    (scenario.next_frame_label !== null &&
      typeof scenario.next_frame_label !== "string") ||
    (scenario.next_frame_description !== null &&
      typeof scenario.next_frame_description !== "string") ||
    hasNext !== (scenario.next_frame_label !== null) ||
    hasNext !== (scenario.next_frame_description !== null)
  ) {
    throw new TypeError("Live researcher scenario cursor fields are invalid.");
  }
  if (
    (option.mode === "interactive" &&
      (scenario.frame_count !== 0 ||
        scenario.completed_frame_count !== 0 ||
        hasNext ||
        scenario.script_complete)) ||
    (option.mode === "scripted" &&
      (scenario.script_complete === hasNext ||
        (hasNext && scenario.next_frame_index !== scenario.completed_frame_count)))
  ) {
    throw new TypeError("Live researcher scenario cursor is incoherent.");
  }
  return Object.freeze({ ...scenario });
}

/**
 * Strictly normalize the path-free recording lifecycle shared by both live
 * audiences. Availability is joined here rather than inferred later by the
 * UI, so malformed or over-disclosing state never reaches a control surface.
 *
 * @param {unknown} value
 * @param {number} frameIndex
 * @returns {Readonly<Record<string, any>> | null}
 */
export function normalizeRecordingStatusV1(value, frameIndex) {
  if (value === null) {
    return null;
  }
  const status = requireRecord(
    value,
    "Live debugger recording status must be an object or null.",
  );
  requireExactKeys(
    status,
    RECORDING_STATUS_KEYS_V1,
    "Live debugger recording status has unknown or missing fields.",
  );
  if (status.schema_version !== 1) {
    throw new TypeError("Live debugger recording status must use schema version 1.");
  }
  if (!RECORDING_LIFECYCLES_V1.has(status.lifecycle)) {
    throw new TypeError("Live debugger recording lifecycle is unknown.");
  }
  if (
    !Number.isInteger(status.captured_transition_count) ||
    status.captured_transition_count < 0 ||
    !Number.isInteger(status.expected_transition_count) ||
    status.expected_transition_count <= 0 ||
    status.captured_transition_count > status.expected_transition_count ||
    status.captured_transition_count !== frameIndex
  ) {
    throw new TypeError(
      "Live debugger recording progress does not join its frame and horizon.",
    );
  }
  if (
    status.completion_state !== null &&
    !RECORDING_COMPLETION_STATES_V1.has(status.completion_state)
  ) {
    throw new TypeError("Live debugger recording completion state is unknown.");
  }
  const reasonLength =
    typeof status.completion_reason === "string"
      ? [...status.completion_reason].length
      : 0;
  if (
    status.completion_reason !== null &&
    (typeof status.completion_reason !== "string" ||
      reasonLength < 1 ||
      reasonLength > 256)
  ) {
    throw new TypeError("Live debugger recording completion reason is invalid.");
  }
  const booleanFields = [
    "restart_fenced",
    "finish_available",
    "review_available",
    "retry_available",
    "save_as_available",
    "discard_available",
  ];
  if (booleanFields.some((field) => typeof status[field] !== "boolean")) {
    throw new TypeError("Live debugger recording availability must be boolean.");
  }
  if (
    status.persistence_error_code !== null &&
    !RECORDING_PERSISTENCE_ERRORS_V1.has(status.persistence_error_code)
  ) {
    throw new TypeError("Live debugger recording persistence error is unknown.");
  }

  const finalized = [
    "sealed",
    "finalized_unsaved",
    "persistence_failed",
    "saved",
    "reviewing",
  ].includes(status.lifecycle);
  const reasonRequired = ["partial", "interrupted", "failed"].includes(
    status.completion_state,
  );
  const restartFenced =
    status.captured_transition_count > 0 || status.lifecycle !== "recording";
  const finishAvailable = status.lifecycle === "recording";
  const reviewAvailable = status.lifecycle === "saved";
  const retryAvailable = status.lifecycle === "persistence_failed";
  const discardAvailable =
    status.lifecycle === "recording" && status.captured_transition_count > 0;
  if (
    finalized !== (status.completion_state !== null) ||
    reasonRequired !== (status.completion_reason !== null) ||
    status.restart_fenced !== restartFenced ||
    status.finish_available !== finishAvailable ||
    status.review_available !== reviewAvailable ||
    status.retry_available !== retryAvailable ||
    status.save_as_available !== retryAvailable ||
    status.discard_available !== discardAvailable ||
    (status.persistence_error_code !== null) !== retryAvailable
  ) {
    throw new TypeError(
      "Live debugger recording lifecycle and availability are not canonical.",
    );
  }

  return Object.freeze({
    schema_version: status.schema_version,
    lifecycle: status.lifecycle,
    captured_transition_count: status.captured_transition_count,
    expected_transition_count: status.expected_transition_count,
    completion_state: status.completion_state,
    completion_reason: status.completion_reason,
    restart_fenced: status.restart_fenced,
    finish_available: status.finish_available,
    review_available: status.review_available,
    retry_available: status.retry_available,
    save_as_available: status.save_as_available,
    discard_available: status.discard_available,
    persistence_error_code: status.persistence_error_code,
  });
}

/**
 * @param {unknown} value
 * @param {string} message
 * @returns {any[]}
 */
function requireArray(value, message) {
  if (!Array.isArray(value)) {
    throw new TypeError(message);
  }
  return value;
}

const STATUS_SOURCE_EVIDENCE_KEYS_V2 = Object.freeze([
  "event_id",
  "source_global_slot",
  "source_public_agent_id",
]);
const STATUS_SCENE_KEYS_V2 = Object.freeze([
  "breaks_on_positive_damage",
  "direct_source_evidence",
  "family",
  "magnitude",
  "magnitude_kind",
  "remaining_duration",
  "source_action_component",
  "source_class_id",
  "source_class_name",
  "status_channel",
  "status_id",
]);
const AURA_RECIPIENT_MODIFIER_KEYS_V2 = Object.freeze(["aura_id", "multiplier"]);
const AURA_FIELD_KEYS_V2 = Object.freeze([
  "aura_id",
  "beneficiary_relation",
  "center",
  "clamp_kind",
  "clamp_value",
  "per_emitter_multiplier",
  "radius",
  "source_alive",
  "source_class_id",
  "source_class_name",
  "source_global_slot",
  "source_public_agent_id",
  "stacking_rule",
]);
const RESEARCHER_PROJECTION_KEYS_V2 = Object.freeze([
  "incoming_events",
  "scene",
  "schema_version",
  "status_source_evidence",
]);
const RESEARCHER_SCENE_KEYS_V2 = Object.freeze([
  "agents",
  "audience",
  "audience_badge",
  "aura_fields",
  "class_mechanics",
  "episode_id",
  "frame_id",
  "frame_index",
  "incoming_event_ids",
  "incoming_transition_id",
  "map",
  "next_decision_selected_legality",
  "observer_visibility",
  "ranges",
  "respawn_waves",
  "schema_version",
  "selection",
  "simulator_step_count",
  "spawn_pads",
]);
const RESEARCHER_MAP_KEYS_V1 = Object.freeze(["height", "obstacles", "width"]);
const RESEARCHER_OBSTACLE_KEYS_V1 = Object.freeze([
  "center",
  "height",
  "kind",
  "obstacle_id",
  "radius",
  "theta",
  "width",
]);
const RESEARCHER_AGENT_KEYS_V2 = Object.freeze([
  "aura_modifiers",
  "class_id",
  "current_health",
  "effective_movement_speed",
  "global_slot",
  "life_state",
  "max_health",
  "position",
  "public_agent_id",
  "radius",
  "respawn_event_id",
  "respawned_on_incoming_transition",
  "spawn_shield_remaining",
  "statuses",
  "steps_until_out_of_combat",
  "team_id",
  "team_local_slot",
  "ultimate_cooldown_remaining",
]);
const RESEARCHER_RANGE_KEYS_V1 = Object.freeze([
  "center",
  "global_slot",
  "kind",
  "radius",
]);
const RESEARCHER_SPAWN_PAD_KEYS_V2 = Object.freeze([
  "assigned_global_slot",
  "assigned_public_agent_id",
  "position",
  "team_id",
  "team_local_slot",
]);
const RESEARCHER_RESPAWN_WAVE_KEYS_V2 = Object.freeze([
  "countdown_steps",
  "period_steps",
  "team_id",
  "team_index",
]);
const RESEARCHER_SELECTION_KEYS_V1 = Object.freeze([
  "controlled_global_slot",
  "selected_global_slot",
]);
const RESEARCHER_SELECTED_LEGALITY_KEYS_V1 = Object.freeze([
  "armed_lane",
  "armed_pair_legal",
  "controlled_global_slot",
  "lane_0_available",
  "lane_1_available",
  "target_action",
  "target_global_slot",
]);
const RESEARCHER_VISIBILITY_KEYS_V1 = Object.freeze([
  "candidate_global_slot",
  "observer_global_slot",
  "visible",
]);
const RESEARCHER_HUD_KEYS_V2 = Object.freeze([
  "candidate_legalities",
  "controlled_global_slot",
  "diagnostics",
  "latest_transition",
  "movement_legalities",
  "pending_action",
  "pending_actions",
  "pending_submission_scope",
  "roster_global_slots",
  "selected_global_slot",
]);
const RESEARCHER_TARGET_REFERENCE_KEYS_V1 = Object.freeze([
  "disclosure",
  "global_slot",
]);
const RESEARCHER_PENDING_ACTION_KEYS_V1 = Object.freeze([
  "actor_global_slot",
  "arm_origin",
  "armed_lane",
  "label",
  "move_action",
  "movement_mask_value",
  "pair_mask_value",
  "summary",
  "target",
  "target_action",
]);
const RESEARCHER_MOVEMENT_LEGALITY_KEYS_V1 = Object.freeze([
  "available",
  "move_action",
]);
const RESEARCHER_CANDIDATE_LEGALITY_KEYS_V1 = Object.freeze([
  "basic_available",
  "lane_0_available",
  "lane_1_available",
  "target",
  "target_action",
  "ultimate_available",
]);
const RESEARCHER_LATEST_TRANSITION_KEYS_V2 = Object.freeze([
  "actors",
  "label",
  "submission_kind",
  "transition_id",
  "transition_index",
]);
const RESEARCHER_ACTION_RESULT_KEYS_V1 = Object.freeze([
  "accepted",
  "actor_global_slot",
  "combat_result",
  "movement_accepted",
  "movement_mask_value",
  "pair_mask_value",
  "submitted",
]);
const RESEARCHER_ACTION_TUPLE_KEYS_V1 = Object.freeze([
  "move_action",
  "summary",
  "target",
  "target_action",
  "use_ultimate_action",
]);
const STATUS_SOURCE_STATE_KEYS_V2 = Object.freeze([
  "active_statuses",
  "episode_id",
  "frame_id",
  "frame_index",
  "schema_version",
]);
const STATUS_SOURCE_CHANNEL_KEYS_V2 = Object.freeze([
  "direct_source_evidence",
  "recipient_global_slot",
  "recipient_public_agent_id",
  "status_channel",
  "status_id",
]);
const RESEARCHER_EVENT_BATCH_KEYS_V2 = Object.freeze([
  "agent_phase_trajectories",
  "configured_active_by_global_slot",
  "episode_id",
  "events",
  "public_agent_id_by_global_slot",
  "schema_version",
  "start_frame_id",
  "start_simulator_step_count",
  "successor_frame_id",
  "successor_simulator_step_count",
  "transition_id",
  "transition_index",
]);
const RESEARCHER_AGENT_ANCHOR_KEYS_V2 = Object.freeze([
  "global_slot",
  "phase",
  "position",
  "public_agent_id",
]);
const RESEARCHER_TRAJECTORY_KEYS_V2 = Object.freeze([
  "global_slot",
  "post_charge",
  "public_agent_id",
  "successor",
  "transition_start",
]);
const RESEARCHER_TEAM_ANCHOR_KEYS_V2 = Object.freeze([
  "phase",
  "team_id",
  "team_index",
]);
const RESEARCHER_EVENT_BASE_KEYS_V2 = Object.freeze([
  "event_id",
  "event_type",
  "ordinal",
  "phase_rank",
  "transition_id",
]);
const RESEARCHER_EVENT_SUFFIX_KEYS_V2 = Object.freeze({
  action_rejected: Object.freeze([
    "actor_anchor",
    "actor_configured_active",
    "actor_global_slot",
    "actor_public_agent_id",
    "rejection_component",
    "submitted_move_action",
    "submitted_select_target_action",
    "submitted_use_ultimate_action",
  ]),
  ability_activated: Object.freeze([
    "ability_component",
    "recipient_anchor",
    "recipient_global_slot",
    "source_anchor",
    "source_global_slot",
  ]),
  source_damage_output: Object.freeze([
    "mage_damage_aura_covering_emitter_global_slots",
    "raw_damage_output",
    "recipient_anchor",
    "recipient_damage_modifier",
    "recipient_global_slot",
    "source_anchor",
    "source_global_slot",
    "source_modified_damage_output",
    "warrior_mitigation_aura_covering_emitter_global_slots",
  ]),
  source_healing_output: Object.freeze([
    "raw_healing_output",
    "recipient_anchor",
    "recipient_global_slot",
    "recipient_healing_modifier",
    "source_anchor",
    "source_global_slot",
    "source_modified_healing_output",
  ]),
  recipient_health_resolution: Object.freeze([
    "health_after_combat_resolution",
    "realized_net_health_change",
    "recipient_anchor",
    "recipient_global_slot",
    "total_effective_damage",
    "total_effective_healing",
    "transition_start_health",
  ]),
  combat_countdown_reset: Object.freeze(["agent_anchor", "agent_global_slot"]),
  agent_left_combat: Object.freeze(["agent_anchor", "agent_global_slot"]),
  health_regenerated: Object.freeze([
    "actual_health_regenerated",
    "agent_anchor",
    "agent_global_slot",
  ]),
  cooldown_started: Object.freeze(["agent_anchor", "agent_global_slot"]),
  cooldown_ready: Object.freeze(["agent_anchor", "agent_global_slot"]),
  charge_phase_displacement: Object.freeze([
    "agent_global_slot",
    "end_anchor",
    "realized_displacement",
    "start_anchor",
  ]),
  ordinary_movement_phase_displacement: Object.freeze([
    "agent_global_slot",
    "end_anchor",
    "realized_displacement",
    "start_anchor",
  ]),
  agent_died: Object.freeze(["recipient_anchor", "recipient_global_slot"]),
  lethal_damage_contribution: Object.freeze([
    "attributed_death_damage",
    "recipient_anchor",
    "recipient_global_slot",
    "source_anchor",
    "source_global_slot",
  ]),
  status_aged_to_zero: Object.freeze([
    "recipient_anchor",
    "recipient_global_slot",
    "status_channel",
    "status_id",
  ]),
  status_broken_by_damage: Object.freeze([
    "recipient_anchor",
    "recipient_global_slot",
    "status_channel",
    "status_id",
  ]),
  status_applied: Object.freeze([
    "recipient_anchor",
    "recipient_global_slot",
    "source_anchor",
    "source_global_slot",
    "status_channel",
    "status_id",
  ]),
  status_refreshed_or_extended: Object.freeze([
    "recipient_anchor",
    "recipient_global_slot",
    "status_channel",
    "status_id",
  ]),
  status_cleared_by_new_death: Object.freeze([
    "recipient_anchor",
    "recipient_global_slot",
    "status_channel",
    "status_id",
  ]),
  spawn_shield_expired: Object.freeze(["agent_anchor", "agent_global_slot"]),
  respawn_wave_occurred: Object.freeze(["team_anchor", "team_id", "team_index"]),
  agent_respawned: Object.freeze([
    "agent_anchor",
    "agent_global_slot",
    "realized_successor_position",
    "team_id",
  ]),
  team_deathmatch_score_changed: Object.freeze([
    "previous_score",
    "score_increment",
    "successor_score",
    "team_anchor",
    "team_id",
    "team_index",
  ]),
  team_deathmatch_completed: Object.freeze(["completion_basis", "outcome"]),
});
const RESEARCHER_EVENT_PHASE_RANK_V2 = Object.freeze({
  action_rejected: 10,
  ability_activated: 20,
  source_damage_output: 30,
  source_healing_output: 30,
  recipient_health_resolution: 40,
  combat_countdown_reset: 50,
  agent_left_combat: 50,
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
  team_deathmatch_score_changed: 130,
  team_deathmatch_completed: 140,
});

/**
 * @param {unknown} value
 * @returns {value is readonly [number, number]}
 */
function isFinitePoint(value) {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    value.every(
      (coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate),
    )
  );
}

/** @param {unknown} value @param {string} label @returns {string} */
function requireNonemptyString(value, label) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError(`${label} must be a nonempty string.`);
  }
  return value;
}

/** @param {unknown} value @param {string} label @param {number} [minimum] @returns {number} */
function requireInteger(value, label, minimum = 0) {
  if (typeof value !== "number" || !Number.isInteger(value) || value < minimum) {
    throw new TypeError(`${label} must be an integer at least ${minimum}.`);
  }
  return value;
}

/** @param {unknown} value @param {string} label @param {number} [minimum] @returns {number} */
function requireFinite(value, label, minimum = -Infinity) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw new TypeError(`${label} must be a finite number at least ${minimum}.`);
  }
  return value;
}

/** @param {unknown} value @param {string} label @returns {boolean} */
function requireBoolean(value, label) {
  if (typeof value !== "boolean") {
    throw new TypeError(`${label} must be a boolean.`);
  }
  return value;
}

/** @param {unknown} value @param {string} label @returns {readonly [number, number]} */
function requirePoint(value, label) {
  if (!isFinitePoint(value)) {
    throw new TypeError(`${label} must be a finite two-coordinate point.`);
  }
  return Object.freeze(/** @type {[number, number]} */ ([...value]));
}

/**
 * @param {unknown} value
 * @param {number} length
 * @param {string} label
 */
function requireBooleanVector(value, length, label) {
  const vector = requireArray(value, `${label} must be an array.`);
  if (vector.length !== length || vector.some((item) => typeof item !== "boolean")) {
    throw new TypeError(`${label} must contain ${length} booleans.`);
  }
  return Object.freeze(/** @type {boolean[]} */ ([...vector]));
}

/** @param {unknown} value */
function normalizeTerminalStateV2(value) {
  const terminal = requireRecord(value, "Terminal state must be an object.");
  requireExactKeys(
    terminal,
    TERMINAL_STATE_KEYS_V2,
    "Terminal state has unknown or missing fields.",
  );
  const terminated = requireBoolean(terminal.terminated, "terminal.terminated");
  const truncated = requireBoolean(terminal.truncated, "terminal.truncated");
  const reachedHorizon = requireBoolean(
    terminal.reached_declared_horizon,
    "terminal.reached_declared_horizon",
  );
  const expectedReason = terminated
    ? "terminated"
    : truncated
      ? "truncated"
      : reachedHorizon
        ? "declared_horizon"
        : null;
  if (
    requireBoolean(terminal.is_sealed, "terminal.is_sealed") !==
      (expectedReason !== null) ||
    terminal.reason !== expectedReason
  ) {
    throw new TypeError("Terminal state flags and reason are incoherent.");
  }
  return Object.freeze({
    is_sealed: terminal.is_sealed,
    terminated,
    truncated,
    reached_declared_horizon: reachedHorizon,
    reason: expectedReason,
  });
}

/** @param {unknown} value @param {string} label */
function normalizeResearcherTargetReference(value, label) {
  const target = requireRecord(value, `${label} must be an object.`);
  requireExactKeys(
    target,
    RESEARCHER_TARGET_REFERENCE_KEYS_V1,
    `${label} has unknown or missing fields.`,
  );
  if (!["public", "target_none", "redacted", "invalid"].includes(target.disclosure)) {
    throw new TypeError(`${label} disclosure is invalid.`);
  }
  const globalSlot =
    target.global_slot === null
      ? null
      : requireInteger(target.global_slot, `${label} global slot`);
  if (
    (target.disclosure === "public" && (globalSlot === null || globalSlot >= 10)) ||
    (target.disclosure !== "public" && globalSlot !== null)
  ) {
    throw new TypeError(`${label} disclosure and global slot are incoherent.`);
  }
  return Object.freeze({ disclosure: target.disclosure, global_slot: globalSlot });
}

/**
 * @param {Readonly<Record<string, any>>} target
 * @param {number | null} targetAction
 * @param {ReadonlyMap<number, Readonly<Record<string, any>>>} roster
 * @param {string} label
 * @param {boolean} allowInvalid
 */
function requireResearcherTargetJoin(
  target,
  targetAction,
  roster,
  label,
  allowInvalid,
) {
  if (
    (target.disclosure === "public" &&
      (targetAction === null ||
        targetAction <= 0 ||
        !roster.has(target.global_slot))) ||
    (target.disclosure === "target_none" && targetAction !== 0) ||
    (target.disclosure === "redacted" && targetAction !== null) ||
    (target.disclosure === "invalid" && (!allowInvalid || targetAction === null))
  ) {
    throw new TypeError(`${label} does not join the researcher target axis.`);
  }
}

/**
 * @param {Record<string, any>} raw
 * @param {string} scope
 * @param {ReadonlyMap<number, Readonly<Record<string, any>>>} roster
 */
function normalizeResearcherPendingAction(raw, scope, roster) {
  requireExactKeys(
    raw,
    RESEARCHER_PENDING_ACTION_KEYS_V1,
    "Researcher pending action has unknown or missing fields.",
  );
  const actorGlobalSlot = requireInteger(
    raw.actor_global_slot,
    "Researcher pending actor slot",
  );
  if (!roster.has(actorGlobalSlot)) {
    throw new TypeError("Researcher pending actor must occur in the scene roster.");
  }
  const moveAction = requireInteger(raw.move_action, "Researcher pending movement");
  const targetAction =
    raw.target_action === null
      ? null
      : requireInteger(raw.target_action, "Researcher pending target action");
  const target = normalizeResearcherTargetReference(
    raw.target,
    "Researcher pending target",
  );
  requireResearcherTargetJoin(
    target,
    targetAction,
    roster,
    "Researcher pending target",
    false,
  );
  const expectedLabel =
    scope === "scripted_playback"
      ? "PLAYBACK / INSPECTION ONLY"
      : "PENDING / WILL SUBMIT";
  const armedLane = raw.armed_lane;
  const armOrigin = raw.arm_origin;
  const pairMaskValue =
    raw.pair_mask_value === null
      ? null
      : requireBoolean(raw.pair_mask_value, "Researcher pending pair-mask value");
  if (
    raw.label !== expectedLabel ||
    ![null, 0, 1].includes(armedLane) ||
    ![null, "automatic", "explicit"].includes(armOrigin) ||
    (target.disclosure === "redacted" && pairMaskValue !== null) ||
    typeof raw.summary !== "string"
  ) {
    throw new TypeError("Researcher pending action semantics are invalid.");
  }
  return Object.freeze({
    label: expectedLabel,
    actor_global_slot: actorGlobalSlot,
    move_action: moveAction,
    target_action: targetAction,
    armed_lane: armedLane,
    arm_origin: armOrigin,
    target,
    movement_mask_value: requireBoolean(
      raw.movement_mask_value,
      "Researcher pending movement-mask value",
    ),
    pair_mask_value: pairMaskValue,
    summary: raw.summary,
  });
}

/**
 * @param {Record<string, any>} raw
 * @param {ReadonlyMap<number, Readonly<Record<string, any>>>} roster
 * @param {string} label
 */
function normalizeResearcherActionTuple(raw, roster, label) {
  requireExactKeys(
    raw,
    RESEARCHER_ACTION_TUPLE_KEYS_V1,
    `${label} has unknown or missing fields.`,
  );
  const targetAction =
    raw.target_action === null
      ? null
      : requireInteger(raw.target_action, `${label} target action`, -Infinity);
  const target = normalizeResearcherTargetReference(raw.target, `${label} target`);
  requireResearcherTargetJoin(target, targetAction, roster, `${label} target`, true);
  if (typeof raw.summary !== "string") {
    throw new TypeError(`${label} summary must be a string.`);
  }
  return Object.freeze({
    move_action: requireInteger(raw.move_action, `${label} movement`, -Infinity),
    target_action: targetAction,
    use_ultimate_action: requireInteger(
      raw.use_ultimate_action,
      `${label} Ultimate lane`,
      -Infinity,
    ),
    target,
    summary: raw.summary,
  });
}

/**
 * @param {Record<string, any>} raw
 * @param {ReadonlyMap<number, Readonly<Record<string, any>>>} roster
 */
function normalizeResearcherActionResult(raw, roster) {
  requireExactKeys(
    raw,
    RESEARCHER_ACTION_RESULT_KEYS_V1,
    "Researcher action result has unknown or missing fields.",
  );
  const actorGlobalSlot = requireInteger(
    raw.actor_global_slot,
    "Researcher result actor slot",
  );
  if (!roster.has(actorGlobalSlot)) {
    throw new TypeError("Researcher result actor must occur in the scene roster.");
  }
  const submitted = normalizeResearcherActionTuple(
    requireRecord(raw.submitted, "Researcher submitted action must be an object."),
    roster,
    "Researcher submitted action",
  );
  const accepted = normalizeResearcherActionTuple(
    requireRecord(raw.accepted, "Researcher accepted action must be an object."),
    roster,
    "Researcher accepted action",
  );
  const movementMaskValue = requireBoolean(
    raw.movement_mask_value,
    "Researcher result movement-mask value",
  );
  const pairMaskValue =
    raw.pair_mask_value === null
      ? null
      : requireBoolean(raw.pair_mask_value, "Researcher result pair-mask value");
  const movementAccepted = requireBoolean(
    raw.movement_accepted,
    "Researcher result movement acceptance",
  );
  const acceptedInDomain =
    accepted.move_action >= 0 &&
    accepted.move_action < 9 &&
    accepted.target_action !== null &&
    accepted.target_action >= 0 &&
    accepted.target_action < 11 &&
    accepted.use_ultimate_action >= 0 &&
    accepted.use_ultimate_action < 2;
  if (
    !acceptedInDomain ||
    !["accepted", "canonical_noop", "rejected"].includes(raw.combat_result)
  ) {
    throw new TypeError("Researcher action result semantics are incoherent.");
  }
  return Object.freeze({
    actor_global_slot: actorGlobalSlot,
    submitted,
    accepted,
    movement_mask_value: movementMaskValue,
    pair_mask_value: pairMaskValue,
    movement_accepted: movementAccepted,
    combat_result: raw.combat_result,
  });
}

/**
 * @param {Record<string, any>} rawHud
 * @param {Readonly<Record<string, any>>} scene
 * @param {Record<string, any>} frame
 */
function normalizeResearcherHud(rawHud, scene, frame) {
  requireExactKeys(
    rawHud,
    RESEARCHER_HUD_KEYS_V2,
    "Researcher HUD has unknown or missing fields.",
  );
  const agents = requireArray(scene.agents, "Researcher HUD scene roster is invalid.");
  const rosterSlots = Object.freeze(
    requireArray(
      rawHud.roster_global_slots,
      "Researcher HUD roster slots must be an array.",
    ).map((slot) => requireInteger(slot, "Researcher HUD roster slot")),
  );
  const expectedRosterSlots = agents.map((agent) => agent.global_slot);
  if (
    rosterSlots.length !== expectedRosterSlots.length ||
    rosterSlots.some((slot, index) => slot !== expectedRosterSlots[index])
  ) {
    throw new TypeError("Researcher HUD roster must join the ordered scene roster.");
  }
  const roster = new Map(agents.map((agent) => [agent.global_slot, agent]));
  const controlledGlobalSlot = requireInteger(
    rawHud.controlled_global_slot,
    "Researcher HUD controlled slot",
  );
  const selectedGlobalSlot =
    rawHud.selected_global_slot === null
      ? null
      : requireInteger(rawHud.selected_global_slot, "Researcher HUD selected slot");
  const selection = scene.selection;
  if (
    !isRecord(selection) ||
    controlledGlobalSlot !== selection.controlled_global_slot ||
    selectedGlobalSlot !== selection.selected_global_slot
  ) {
    throw new TypeError(
      "Researcher HUD selection must join scene selection authority.",
    );
  }
  const scope = rawHud.pending_submission_scope;
  if (!["joint_turn", "controlled_actor", "scripted_playback"].includes(scope)) {
    throw new TypeError("Researcher HUD pending submission scope is invalid.");
  }
  const pendingActions = Object.freeze(
    requireArray(
      rawHud.pending_actions,
      "Researcher HUD pending actions must be an array.",
    ).map((rawPending) =>
      normalizeResearcherPendingAction(
        requireRecord(rawPending, "Invalid researcher pending-action row."),
        scope,
        roster,
      ),
    ),
  );
  const expectedPendingSlots =
    scope === "joint_turn" ? rosterSlots : [controlledGlobalSlot];
  if (
    pendingActions.length !== expectedPendingSlots.length ||
    pendingActions.some(
      (pending, index) => pending.actor_global_slot !== expectedPendingSlots[index],
    )
  ) {
    throw new TypeError(
      "Researcher pending actions must exactly join their submission scope.",
    );
  }
  const pendingAction = normalizeResearcherPendingAction(
    requireRecord(
      rawHud.pending_action,
      "Researcher pending action must be an object.",
    ),
    scope,
    roster,
  );
  const controlledPending = pendingActions.find(
    (pending) => pending.actor_global_slot === controlledGlobalSlot,
  );
  if (
    controlledPending === undefined ||
    JSON.stringify(pendingAction) !== JSON.stringify(controlledPending)
  ) {
    throw new TypeError(
      "Researcher pending action must equal the controlled pending-actions row.",
    );
  }
  const movementLegalities = Object.freeze(
    requireArray(
      rawHud.movement_legalities,
      "Researcher movement legalities must be an array.",
    ).map((rawRow, index) => {
      const row = requireRecord(rawRow, "Invalid researcher movement-legality row.");
      requireExactKeys(
        row,
        RESEARCHER_MOVEMENT_LEGALITY_KEYS_V1,
        "Researcher movement-legality row has unknown or missing fields.",
      );
      if (requireInteger(row.move_action, "Researcher movement action") !== index) {
        throw new TypeError(
          "Researcher movement legalities must use canonical action order.",
        );
      }
      return Object.freeze({
        move_action: index,
        available: requireBoolean(row.available, "Researcher movement availability"),
      });
    }),
  );
  if (movementLegalities.length !== 9) {
    throw new TypeError("Researcher movement legalities must cover nine actions.");
  }
  const candidateLegalities = Object.freeze(
    requireArray(
      rawHud.candidate_legalities,
      "Researcher candidate legalities must be an array.",
    ).map((rawRow, index, rows) => {
      const row = requireRecord(rawRow, "Invalid researcher candidate-legality row.");
      requireExactKeys(
        row,
        RESEARCHER_CANDIDATE_LEGALITY_KEYS_V1,
        "Researcher candidate-legality row has unknown or missing fields.",
      );
      const targetAction = requireInteger(
        row.target_action,
        "Researcher candidate target action",
      );
      if (
        targetAction >= 11 ||
        (index > 0 && targetAction <= rows[index - 1].target_action)
      ) {
        throw new TypeError("Researcher candidate target actions are not canonical.");
      }
      const target = normalizeResearcherTargetReference(
        row.target,
        "Researcher candidate target",
      );
      if (!["public", "target_none"].includes(target.disclosure)) {
        throw new TypeError("Researcher candidate target disclosure is invalid.");
      }
      requireResearcherTargetJoin(
        target,
        targetAction,
        roster,
        "Researcher candidate target",
        false,
      );
      const lane0 = requireBoolean(
        row.lane_0_available,
        "Researcher candidate lane-zero availability",
      );
      const lane1 = requireBoolean(
        row.lane_1_available,
        "Researcher candidate lane-one availability",
      );
      const basic = requireBoolean(
        row.basic_available,
        "Researcher candidate Basic availability",
      );
      const ultimate = requireBoolean(
        row.ultimate_available,
        "Researcher candidate Ultimate availability",
      );
      return Object.freeze({
        target_action: targetAction,
        target,
        lane_0_available: lane0,
        lane_1_available: lane1,
        basic_available: basic,
        ultimate_available: ultimate,
      });
    }),
  );
  if (candidateLegalities.length > 0) {
    const targetNoneRows = candidateLegalities.filter(
      (candidate) => candidate.target.disclosure === "target_none",
    );
    const publicSlots = candidateLegalities
      .filter((candidate) => candidate.target.disclosure === "public")
      .map((candidate) => candidate.target.global_slot);
    if (
      targetNoneRows.length !== 1 ||
      publicSlots.length !== rosterSlots.length ||
      new Set(publicSlots).size !== publicSlots.length ||
      publicSlots.some((slot) => !roster.has(slot))
    ) {
      throw new TypeError(
        "Researcher candidate legalities must exactly cover target-none and roster identities.",
      );
    }
  }
  let latestTransition = null;
  if (rawHud.latest_transition !== null) {
    const rawLatest = requireRecord(
      rawHud.latest_transition,
      "Researcher latest transition must be an object or null.",
    );
    requireExactKeys(
      rawLatest,
      RESEARCHER_LATEST_TRANSITION_KEYS_V2,
      "Researcher latest transition has unknown or missing fields.",
    );
    const transitionIndex = requireInteger(
      rawLatest.transition_index,
      "Researcher latest transition index",
    );
    const transitionId = requireNonemptyString(
      rawLatest.transition_id,
      "Researcher latest transition ID",
    );
    if (
      frame.frame_index === 0 ||
      transitionIndex !== frame.frame_index - 1 ||
      transitionId !== frame.incoming_transition_id ||
      rawLatest.label !== "LATEST ACCEPTED RESULT" ||
      !["interactive", "scripted"].includes(rawLatest.submission_kind)
    ) {
      throw new TypeError(
        "Researcher latest transition does not join the live envelope.",
      );
    }
    const actors = Object.freeze(
      requireArray(
        rawLatest.actors,
        "Researcher latest transition actors must be an array.",
      ).map((rawActor) =>
        normalizeResearcherActionResult(
          requireRecord(rawActor, "Invalid researcher action-result row."),
          roster,
        ),
      ),
    );
    if (
      new Set(actors.map((actor) => actor.actor_global_slot)).size !== actors.length
    ) {
      throw new TypeError("Researcher latest transition actor slots must be unique.");
    }
    latestTransition = Object.freeze({
      label: "LATEST ACCEPTED RESULT",
      transition_index: transitionIndex,
      transition_id: transitionId,
      submission_kind: rawLatest.submission_kind,
      actors,
    });
  }
  const diagnostics = Object.freeze(
    requireArray(rawHud.diagnostics, "Researcher diagnostics must be an array.").map(
      (rawFact) => {
        const fact = requireRecord(rawFact, "Invalid researcher diagnostic fact.");
        requireExactKeys(
          fact,
          DIAGNOSTIC_FACT_KEYS_V1,
          "Researcher diagnostic fact has unknown or missing fields.",
        );
        return Object.freeze({
          fact_id: requireNonemptyString(fact.fact_id, "Researcher diagnostic fact ID"),
          label: requireNonemptyString(fact.label, "Researcher diagnostic label"),
          value: requireNonemptyString(fact.value, "Researcher diagnostic value"),
          technical: requireBoolean(
            fact.technical,
            "Researcher diagnostic technical flag",
          ),
        });
      },
    ),
  );
  return Object.freeze({
    roster_global_slots: rosterSlots,
    controlled_global_slot: controlledGlobalSlot,
    selected_global_slot: selectedGlobalSlot,
    pending_submission_scope: scope,
    pending_actions: pendingActions,
    pending_action: controlledPending,
    latest_transition: latestTransition,
    movement_legalities: movementLegalities,
    candidate_legalities: candidateLegalities,
    diagnostics,
  });
}

/**
 * Compose a live-only pending route from exact researcher HUD identity and the
 * already-authorized settled-scene body anchors. This projection never enters
 * the renderer-neutral SceneV2 wire contract.
 *
 * @param {Record<string, any>} scene
 * @param {Record<string, any>} frame
 */
function researcherPendingRoute(scene, frame) {
  const hud = isRecord(frame.hud) ? frame.hud : null;
  const pending = hud && isRecord(hud.pending_action) ? hud.pending_action : null;
  const target = pending && isRecord(pending.target) ? pending.target : null;
  const lane = pending?.armed_lane;
  if (
    !pending ||
    !target ||
    (lane !== 0 && lane !== 1) ||
    target.disclosure !== "public" ||
    !Number.isInteger(pending.actor_global_slot) ||
    !Number.isInteger(target.global_slot) ||
    pending.pair_mask_value !== true
  ) {
    return null;
  }
  const agents = requireArray(
    scene.agents,
    "Researcher scene agents must be an array.",
  );
  const source = agents.find(
    (agent) => isRecord(agent) && agent.global_slot === pending.actor_global_slot,
  );
  const recipient = agents.find(
    (agent) => isRecord(agent) && agent.global_slot === target.global_slot,
  );
  if (
    !isRecord(source) ||
    !isRecord(recipient) ||
    typeof source.public_agent_id !== "string" ||
    typeof recipient.public_agent_id !== "string" ||
    !isFinitePoint(source.position) ||
    !isFinitePoint(recipient.position) ||
    typeof source.radius !== "number" ||
    !Number.isFinite(source.radius) ||
    source.radius <= 0 ||
    typeof recipient.radius !== "number" ||
    !Number.isFinite(recipient.radius) ||
    recipient.radius <= 0
  ) {
    return null;
  }
  return Object.freeze({
    audience: "researcher",
    source_global_slot: pending.actor_global_slot,
    target_global_slot: target.global_slot,
    source_public_agent_id: source.public_agent_id,
    target_public_agent_id: recipient.public_agent_id,
    target_action: pending.target_action,
    source_anchor: source.position,
    target_anchor: recipient.position,
    source_radius: source.radius,
    target_radius: recipient.radius,
    lane,
    legal: true,
  });
}

/**
 * Compose the same live-only affordance for a recipient POV without assigning
 * global identities to visible observation rows.
 *
 * @param {Record<string, any>} scene
 * @param {Record<string, any>} frame
 */
function povPendingRoute(scene, frame) {
  const hud = isRecord(frame.hud) ? frame.hud : null;
  const pending = hud && isRecord(hud.pending_action) ? hud.pending_action : null;
  const target = pending && isRecord(pending.target) ? pending.target : null;
  const lane = pending?.armed_lane;
  const targetAction = target?.target_action;
  if (
    !pending ||
    !target ||
    (lane !== 0 && lane !== 1) ||
    !Number.isInteger(targetAction) ||
    targetAction <= 0 ||
    typeof pending.actor_public_agent_id !== "string" ||
    typeof target.public_agent_id !== "string" ||
    pending.pair_mask_value !== true
  ) {
    return null;
  }
  const source = requireArray(scene.agents, "POV self agents must be an array.").find(
    (agent) =>
      isRecord(agent) && agent.public_agent_id === pending.actor_public_agent_id,
  );
  const bodies = [
    ...requireArray(scene.agents, "POV self agents must be an array."),
    ...requireArray(scene.observed_bodies, "POV observed bodies must be an array."),
  ];
  const recipient = bodies.find(
    (body) => isRecord(body) && body.public_agent_id === target.public_agent_id,
  );
  if (
    !isRecord(source) ||
    !isRecord(recipient) ||
    !isFinitePoint(source.position) ||
    !isFinitePoint(recipient.position) ||
    typeof source.radius !== "number" ||
    !Number.isFinite(source.radius) ||
    source.radius <= 0 ||
    typeof recipient.radius !== "number" ||
    !Number.isFinite(recipient.radius) ||
    recipient.radius <= 0
  ) {
    return null;
  }
  return Object.freeze({
    audience: "agent_pov",
    source_public_agent_id: source.public_agent_id,
    target_public_agent_id: recipient.public_agent_id,
    target_action: targetAction,
    source_anchor: source.position,
    target_anchor: recipient.position,
    source_radius: source.radius,
    target_radius: recipient.radius,
    lane,
    legal: true,
  });
}

/**
 * Preserve the complete V2 agent row while publishing the stable names used
 * by the SVG/HUD presentation layer. These aliases are display vocabulary;
 * they do not recompute health, cooldown, status, or lifecycle truth.
 *
 * @param {Record<string, any>} agent
 * @param {readonly Record<string, any>[]} roster
 * @returns {{authorized: Readonly<Record<string, any>>, presentation: Readonly<Record<string, any>>}}
 */
function normalizeResearcherAgent(agent, roster) {
  requireExactKeys(
    agent,
    RESEARCHER_AGENT_KEYS_V2,
    "Researcher agent has unknown or missing fields.",
  );
  const globalSlot = requireInteger(agent.global_slot, "Researcher agent global slot");
  const teamLocalSlot = requireInteger(
    agent.team_local_slot,
    "Researcher agent team-local slot",
  );
  const teamId = requireInteger(agent.team_id, "Researcher agent team ID", 1);
  const classId = requireInteger(agent.class_id, "Researcher agent class ID", 1);
  const currentHealth = requireFinite(
    agent.current_health,
    "Researcher agent current health",
    0,
  );
  const maxHealth = requireFinite(
    agent.max_health,
    "Researcher agent maximum health",
    Number.EPSILON,
  );
  const radius = requireFinite(agent.radius, "Researcher agent radius", Number.EPSILON);
  if (
    globalSlot >= 10 ||
    teamLocalSlot >= 5 ||
    teamId > 2 ||
    classId > 5 ||
    teamId !== (globalSlot < 5 ? 1 : 2) ||
    teamLocalSlot !== globalSlot % 5 ||
    !["alive", "corpse"].includes(agent.life_state) ||
    currentHealth > maxHealth ||
    radius <= 0 ||
    maxHealth <= 0
  ) {
    throw new TypeError("Researcher agent identity, life state, or body is invalid.");
  }
  const respawned = requireBoolean(
    agent.respawned_on_incoming_transition,
    "Researcher agent respawn flag",
  );
  if (
    (agent.respawn_event_id === null) !== !respawned ||
    (agent.respawn_event_id !== null &&
      (typeof agent.respawn_event_id !== "string" ||
        agent.respawn_event_id.length === 0))
  ) {
    throw new TypeError("Researcher agent respawn evidence is invalid.");
  }
  const statuses = requireArray(
    agent.statuses,
    "Researcher V2 agent statuses must be an array.",
  ).map((status) =>
    normalizeResearcherStatus(requireRecord(status, "Invalid V2 status row."), roster),
  );
  if (
    new Set(
      statuses.map(
        (/** @type {Record<string, any>} */ status) => status.status_channel,
      ),
    ).size !== statuses.length ||
    statuses.some(
      (status, index) =>
        index > 0 &&
        CANONICAL_STATUS_ORDER.indexOf(status.token_id) <=
          CANONICAL_STATUS_ORDER.indexOf(statuses[index - 1].token_id),
    )
  ) {
    throw new TypeError(
      "Researcher V2 statuses require unique canonical presentation order.",
    );
  }
  const modifiers = requireArray(
    agent.aura_modifiers,
    "Researcher V2 agent aura modifiers must be an array.",
  ).map((modifier) =>
    normalizeResearcherAuraModifier(
      requireRecord(modifier, "Invalid V2 aura modifier row."),
    ),
  );
  if (
    new Set(
      modifiers.map((/** @type {Record<string, any>} */ modifier) => modifier.aura_id),
    ).size !== modifiers.length
  ) {
    throw new TypeError("Researcher V2 aura modifiers require unique aura identities.");
  }
  if (
    modifiers
      .map((/** @type {Record<string, any>} */ modifier) => modifier.aura_id)
      .join(",") !== "mage_damage_amplification,warrior_damage_mitigation"
  ) {
    throw new TypeError("Researcher agents require both ordered aura modifiers.");
  }
  const authorizedStatuses = Object.freeze(
    statuses.map((/** @type {Record<string, any>} */ status) =>
      Object.freeze({
        breaks_on_positive_damage: status.breaks_on_positive_damage,
        direct_source_evidence: status.direct_source_evidence,
        family: status.family,
        magnitude: status.magnitude,
        magnitude_kind: status.magnitude_kind,
        remaining_duration: status.remaining_duration,
        source_action_component: status.source_action_component,
        source_class_id: status.source_class_id,
        source_class_name: status.source_class_name,
        status_channel: status.status_channel,
        status_id: status.status_id,
      }),
    ),
  );
  const authorizedModifiers = Object.freeze(
    modifiers.map((/** @type {Record<string, any>} */ modifier) =>
      Object.freeze({
        aura_id: modifier.aura_id,
        multiplier: modifier.multiplier,
      }),
    ),
  );
  const effectiveMovementSpeed = requireFinite(
    agent.effective_movement_speed,
    "Researcher agent effective movement speed",
    0,
  );
  const ultimateCooldown = requireInteger(
    agent.ultimate_cooldown_remaining,
    "Researcher agent Ultimate cooldown",
  );
  const spawnShield = requireInteger(
    agent.spawn_shield_remaining,
    "Researcher agent spawn shield",
  );
  const combatCountdown = requireInteger(
    agent.steps_until_out_of_combat,
    "Researcher agent combat countdown",
  );
  const authorized = Object.freeze({
    aura_modifiers: authorizedModifiers,
    class_id: classId,
    current_health: currentHealth,
    effective_movement_speed: effectiveMovementSpeed,
    global_slot: globalSlot,
    life_state: agent.life_state,
    max_health: maxHealth,
    position: requirePoint(agent.position, "Researcher agent position"),
    public_agent_id: requireNonemptyString(
      agent.public_agent_id,
      "Researcher agent public ID",
    ),
    radius,
    respawn_event_id: agent.respawn_event_id,
    respawned_on_incoming_transition: respawned,
    spawn_shield_remaining: spawnShield,
    statuses: authorizedStatuses,
    steps_until_out_of_combat: combatCountdown,
    team_id: teamId,
    team_local_slot: teamLocalSlot,
    ultimate_cooldown_remaining: ultimateCooldown,
  });
  return {
    authorized,
    presentation: Object.freeze({
      ...authorized,
      alive: authorized.life_state === "alive",
      effective_speed: authorized.effective_movement_speed,
      ultimate_cooldown: authorized.ultimate_cooldown_remaining,
      statuses: Object.freeze(statuses),
      modifiers: Object.freeze(modifiers),
    }),
  };
}

const CLASS_MECHANICS_KEYS_V2 = Object.freeze([
  "aura_mechanics",
  "base_movement_speed",
  "basic_interaction_radius",
  "basic_raw_damage",
  "basic_raw_healing",
  "basic_target_mode",
  "body_radius",
  "class_id",
  "class_name",
  "maximum_health",
  "observation_radius",
  "out_of_combat_delay_steps",
  "out_of_combat_health_regeneration_fraction_per_step",
  "status_mechanics",
  "ultimate_cooldown_steps",
  "ultimate_interaction_radius",
  "ultimate_raw_damage",
  "ultimate_raw_healing",
  "ultimate_target_mode",
]);
const CLASS_STATUS_MECHANIC_KEYS_V2 = Object.freeze([
  "breaks_on_positive_damage",
  "duration_steps",
  "family",
  "magnitude",
  "magnitude_kind",
  "source_action_component",
  "status_channel",
  "status_id",
]);
const CLASS_AURA_MECHANIC_KEYS_V2 = Object.freeze([
  "aura_id",
  "clamp_kind",
  "clamp_value",
  "per_emitter_multiplier",
  "radius",
  "stacking_rule",
]);

/** @param {unknown} value @param {string} message */
function requireFiniteNumber(value, message) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(message);
  }
  return value;
}

/** @param {unknown} rawMechanics */
function normalizeClassMechanicsV2(rawMechanics) {
  const mechanics = requireRecord(rawMechanics, "Invalid class mechanics row.");
  requireExactKeys(
    mechanics,
    CLASS_MECHANICS_KEYS_V2,
    "Class mechanics have unknown or missing fields.",
  );
  if (
    !Number.isInteger(mechanics.class_id) ||
    mechanics.class_id < 1 ||
    mechanics.class_id > 5 ||
    mechanics.class_name !== CLASS_NAME_BY_ID_V1[mechanics.class_id]
  ) {
    throw new TypeError("Class mechanics require an exact real class identity.");
  }
  for (const key of [
    "maximum_health",
    "body_radius",
    "base_movement_speed",
    "observation_radius",
    "basic_interaction_radius",
    "basic_raw_damage",
    "basic_raw_healing",
    "ultimate_interaction_radius",
    "ultimate_raw_damage",
    "ultimate_raw_healing",
    "out_of_combat_health_regeneration_fraction_per_step",
  ]) {
    if (requireFiniteNumber(mechanics[key], `Class mechanics ${key} is invalid.`) < 0) {
      throw new TypeError(`Class mechanics ${key} must be nonnegative.`);
    }
  }
  if (
    mechanics.maximum_health <= 0 ||
    mechanics.body_radius <= 0 ||
    mechanics.out_of_combat_health_regeneration_fraction_per_step > 1
  ) {
    throw new TypeError(
      "Class mechanics health/radius must be positive and regeneration must be a fraction.",
    );
  }
  if (
    !Number.isInteger(mechanics.ultimate_cooldown_steps) ||
    mechanics.ultimate_cooldown_steps < 0 ||
    !Number.isInteger(mechanics.out_of_combat_delay_steps) ||
    mechanics.out_of_combat_delay_steps < 0 ||
    !["unavailable", "ally", "enemy"].includes(mechanics.basic_target_mode) ||
    !["unavailable", "target_none", "ally", "enemy"].includes(
      mechanics.ultimate_target_mode,
    )
  ) {
    throw new TypeError("Class mechanics scalar vocabulary is invalid.");
  }
  const statusMechanics = requireArray(
    mechanics.status_mechanics,
    "Class status mechanics must be an array.",
  ).map((rawStatus) => {
    const status = requireRecord(rawStatus, "Invalid class status mechanic.");
    requireExactKeys(
      status,
      CLASS_STATUS_MECHANIC_KEYS_V2,
      "Class status mechanic has unknown or missing fields.",
    );
    if (
      !Number.isInteger(status.status_channel) ||
      status.status_channel < 0 ||
      status.status_channel >= 9 ||
      typeof status.status_id !== "string" ||
      !["slow", "stun", "anti_heal", "damage_amplification", "movement_floor"].includes(
        status.family,
      ) ||
      !["basic", "ultimate"].includes(status.source_action_component) ||
      !Number.isInteger(status.duration_steps) ||
      status.duration_steps < 1 ||
      ![
        "movement_multiplier",
        "none",
        "healing_multiplier",
        "damage_multiplier",
        "movement_floor",
      ].includes(status.magnitude_kind) ||
      typeof status.breaks_on_positive_damage !== "boolean" ||
      (status.magnitude_kind === "none"
        ? status.magnitude !== null
        : typeof status.magnitude !== "number" || !Number.isFinite(status.magnitude)) ||
      CATALOG_STATUS_ID_BY_CHANNEL_V1[status.status_channel] !== status.status_id ||
      STATUS_SOURCE_CLASS_BY_CHANNEL_V1[status.status_channel] !== mechanics.class_id
    ) {
      throw new TypeError("Class status mechanic values are invalid.");
    }
    return /** @type {Readonly<Record<string, any>>} */ (
      Object.freeze({
        ...status,
        token_id: statusTokenIdFromCatalogId(status.status_id),
      })
    );
  });
  const auraMechanics = requireArray(
    mechanics.aura_mechanics,
    "Class aura mechanics must be an array.",
  ).map((rawAura) => {
    const aura = requireRecord(rawAura, "Invalid class aura mechanic.");
    requireExactKeys(
      aura,
      CLASS_AURA_MECHANIC_KEYS_V2,
      "Class aura mechanic has unknown or missing fields.",
    );
    if (
      typeof aura.aura_id !== "string" ||
      typeof AURA_TOKEN_BY_ID[aura.aura_id] !== "string" ||
      AURA_SOURCE_CLASS_BY_ID_V1[
        /** @type {keyof typeof AURA_SOURCE_CLASS_BY_ID_V1} */ (aura.aura_id)
      ] !== mechanics.class_id ||
      aura.stacking_rule !== "multiply_then_clamp" ||
      !["ceiling", "floor"].includes(aura.clamp_kind) ||
      [aura.radius, aura.per_emitter_multiplier, aura.clamp_value].some(
        (value) => typeof value !== "number" || !Number.isFinite(value) || value < 0,
      )
    ) {
      throw new TypeError("Class aura mechanic values are invalid.");
    }
    return /** @type {Readonly<Record<string, any>>} */ (
      Object.freeze({
        ...aura,
        token_id: AURA_TOKEN_BY_ID[aura.aura_id],
      })
    );
  });
  return /** @type {Readonly<Record<string, any>>} */ (
    Object.freeze({
      ...mechanics,
      status_mechanics: Object.freeze(statusMechanics),
      aura_mechanics: Object.freeze(auraMechanics),
    })
  );
}

/** @param {Record<string, any>} raw @param {readonly Record<string, any>[]} roster */
function normalizeResearcherStatusSourceEvidence(
  raw,
  roster,
  episodeId = null,
  frameIndex = null,
) {
  requireExactKeys(
    raw,
    STATUS_SOURCE_EVIDENCE_KEYS_V2,
    "Researcher status source evidence has unknown or missing fields.",
  );
  const source = roster.find((agent) => agent.global_slot === raw.source_global_slot);
  if (
    !Number.isInteger(raw.source_global_slot) ||
    raw.source_global_slot < 0 ||
    typeof raw.source_public_agent_id !== "string" ||
    raw.source_public_agent_id.length === 0 ||
    typeof raw.event_id !== "string" ||
    raw.event_id.length === 0 ||
    !source ||
    typeof source.public_agent_id !== "string" ||
    source.public_agent_id.length === 0 ||
    source.public_agent_id !== raw.source_public_agent_id ||
    (episodeId !== null &&
      frameIndex !== null &&
      (() => {
        const prefix = `${episodeId}:transition:`;
        if (!raw.event_id.startsWith(prefix)) {
          return true;
        }
        const suffix = raw.event_id.slice(prefix.length);
        const [transitionText, eventText, extra] = suffix.split(":event:");
        const transitionIndex = Number(transitionText);
        return (
          extra !== undefined ||
          !/^(?:0|[1-9]\d*)$/u.test(transitionText) ||
          !Number.isSafeInteger(transitionIndex) ||
          !/^\d{4}$/u.test(eventText) ||
          transitionIndex >= frameIndex
        );
      })())
  ) {
    throw new TypeError(
      "Researcher status source evidence must join its exact source roster identity.",
    );
  }
  return Object.freeze({
    event_id: raw.event_id,
    source_global_slot: raw.source_global_slot,
    source_public_agent_id: raw.source_public_agent_id,
  });
}

/** @param {Record<string, any>} status @param {readonly Record<string, any>[]} roster */
function normalizeResearcherStatus(status, roster) {
  requireExactKeys(
    status,
    STATUS_SCENE_KEYS_V2,
    "Researcher status has unknown or missing fields.",
  );
  if (
    !Number.isInteger(status.status_channel) ||
    status.status_channel < 0 ||
    status.status_channel >= CATALOG_STATUS_ID_BY_CHANNEL_V1.length ||
    !Number.isInteger(status.remaining_duration) ||
    status.remaining_duration < 1
  ) {
    throw new TypeError("Researcher status channel and duration are invalid.");
  }
  const magnitudeIsValid =
    status.magnitude_kind === "none"
      ? status.magnitude === null
      : typeof status.magnitude === "number" && Number.isFinite(status.magnitude);
  if (
    status.status_id !== CATALOG_STATUS_ID_BY_CHANNEL_V1[status.status_channel] ||
    !["slow", "stun", "anti_heal", "damage_amplification", "movement_floor"].includes(
      status.family,
    ) ||
    !Number.isInteger(status.source_class_id) ||
    status.source_class_id < 1 ||
    status.source_class_id > 5 ||
    status.source_class_name !== CLASS_NAME_BY_ID_V1[status.source_class_id] ||
    !["basic", "ultimate"].includes(status.source_action_component) ||
    ![
      "movement_multiplier",
      "none",
      "healing_multiplier",
      "damage_multiplier",
      "movement_floor",
    ].includes(status.magnitude_kind) ||
    !magnitudeIsValid ||
    typeof status.breaks_on_positive_damage !== "boolean"
  ) {
    throw new TypeError("Researcher status values are invalid.");
  }
  const evidence = requireArray(
    status.direct_source_evidence,
    "Researcher status direct source evidence must be an array.",
  ).map((row) =>
    normalizeResearcherStatusSourceEvidence(
      requireRecord(row, "Invalid researcher status source evidence row."),
      roster,
    ),
  );
  const evidenceEventIds = evidence.map((row) => row.event_id);
  const evidenceKeys = evidence.map(
    (row) => `${String(row.source_global_slot).padStart(2, "0")}:${row.event_id}`,
  );
  if (
    evidenceKeys.some((key, index) => index > 0 && key <= evidenceKeys[index - 1]) ||
    new Set(evidenceEventIds).size !== evidenceEventIds.length
  ) {
    throw new TypeError(
      "Researcher status source evidence must be canonical and unique.",
    );
  }
  return Object.freeze({
    ...status,
    direct_source_evidence: Object.freeze(evidence),
    token_id: statusTokenIdFromCatalogId(status.status_id),
    duration: status.remaining_duration,
  });
}

/** @param {Record<string, any>} modifier */
function normalizeResearcherAuraModifier(modifier) {
  requireExactKeys(
    modifier,
    AURA_RECIPIENT_MODIFIER_KEYS_V2,
    "Researcher aura modifier has unknown or missing fields.",
  );
  const tokenId =
    AURA_TOKEN_BY_ID[/** @type {keyof typeof AURA_TOKEN_BY_ID} */ (modifier.aura_id)];
  if (
    typeof tokenId !== "string" ||
    typeof modifier.multiplier !== "number" ||
    !Number.isFinite(modifier.multiplier) ||
    modifier.multiplier < 0
  ) {
    throw new TypeError("Researcher aura modifier values are invalid.");
  }
  return Object.freeze({ ...modifier, token_id: tokenId });
}

/** @param {Record<string, any>} field @param {readonly Record<string, any>[]} roster */
function normalizeResearcherAuraField(field, roster) {
  requireExactKeys(
    field,
    AURA_FIELD_KEYS_V2,
    "Researcher aura field has unknown or missing fields.",
  );
  const source = roster.find((agent) => agent.global_slot === field.source_global_slot);
  const tokenId =
    AURA_TOKEN_BY_ID[/** @type {keyof typeof AURA_TOKEN_BY_ID} */ (field.aura_id)];
  if (
    !source ||
    typeof tokenId !== "string" ||
    source.public_agent_id !== field.source_public_agent_id ||
    source.class_id !== field.source_class_id ||
    field.source_class_name !== CLASS_NAME_BY_ID_V1[field.source_class_id] ||
    typeof field.source_alive !== "boolean" ||
    (source.life_state === "alive") !== field.source_alive ||
    !isFinitePoint(field.center) ||
    !isFinitePoint(source.position) ||
    !["alive", "corpse"].includes(source.life_state) ||
    !field.center.every((value, index) => Object.is(value, source.position[index])) ||
    field.beneficiary_relation !== "same_team" ||
    requireFinite(field.radius, "Researcher aura radius", Number.EPSILON) <= 0 ||
    requireFinite(
      field.per_emitter_multiplier,
      "Researcher aura per-emitter multiplier",
      0,
    ) < 0 ||
    field.stacking_rule !== "multiply_then_clamp" ||
    !["ceiling", "floor"].includes(field.clamp_kind) ||
    requireFinite(field.clamp_value, "Researcher aura clamp value", 0) < 0
  ) {
    throw new TypeError("Researcher aura field must join its roster source.");
  }
  return Object.freeze({
    ...field,
    center: Object.freeze([...field.center]),
    token_id: tokenId,
  });
}

/** @param {Record<string, any>} rawMap */
function normalizeResearcherMap(rawMap) {
  requireExactKeys(
    rawMap,
    RESEARCHER_MAP_KEYS_V1,
    "Researcher map has unknown or missing fields.",
  );
  const seenIds = new Set();
  const obstacles = requireArray(
    rawMap.obstacles,
    "Researcher map obstacles must be an array.",
  ).map((rawObstacle) => {
    const obstacle = requireRecord(rawObstacle, "Invalid researcher obstacle row.");
    requireExactKeys(
      obstacle,
      RESEARCHER_OBSTACLE_KEYS_V1,
      "Researcher obstacle has unknown or missing fields.",
    );
    const obstacleId = requireNonemptyString(
      obstacle.obstacle_id,
      "Researcher obstacle ID",
    );
    if (seenIds.has(obstacleId)) {
      throw new TypeError("Researcher obstacle IDs must be unique.");
    }
    seenIds.add(obstacleId);
    const center = requirePoint(obstacle.center, "Researcher obstacle center");
    const theta = requireFinite(obstacle.theta, "Researcher obstacle theta");
    if (obstacle.kind === "pillar") {
      if (
        requireFinite(obstacle.radius, "Researcher pillar radius", Number.EPSILON) <=
          0 ||
        obstacle.width !== null ||
        obstacle.height !== null
      ) {
        throw new TypeError("Researcher pillar geometry is invalid.");
      }
    } else if (obstacle.kind === "wall") {
      if (
        requireFinite(obstacle.width, "Researcher wall width", Number.EPSILON) <= 0 ||
        requireFinite(obstacle.height, "Researcher wall height", Number.EPSILON) <= 0 ||
        obstacle.radius !== null
      ) {
        throw new TypeError("Researcher wall geometry is invalid.");
      }
    } else {
      throw new TypeError("Researcher obstacle kind is invalid.");
    }
    return Object.freeze({
      center,
      height: obstacle.height,
      kind: obstacle.kind,
      obstacle_id: obstacleId,
      radius: obstacle.radius,
      theta,
      width: obstacle.width,
    });
  });
  return Object.freeze({
    height: requireFinite(rawMap.height, "Researcher map height", Number.EPSILON),
    obstacles: Object.freeze(obstacles),
    width: requireFinite(rawMap.width, "Researcher map width", Number.EPSILON),
  });
}

/**
 * @param {Record<string, any>} raw
 * @param {readonly Record<string, any>[]} agents
 */
function normalizeResearcherRange(raw, agents) {
  requireExactKeys(
    raw,
    RESEARCHER_RANGE_KEYS_V1,
    "Researcher range has unknown or missing fields.",
  );
  const globalSlot = requireInteger(raw.global_slot, "Researcher range global slot");
  const owner = agents.find(
    (/** @type {Record<string, any>} */ agent) => agent.global_slot === globalSlot,
  );
  const center = requirePoint(raw.center, "Researcher range center");
  const radius = requireFinite(raw.radius, "Researcher range radius", 0);
  if (
    !owner ||
    !["observation", "basic", "ultimate"].includes(raw.kind) ||
    !researcherPointsEqual(center, owner.position)
  ) {
    throw new TypeError("Researcher range must join its owner position.");
  }
  return Object.freeze({
    center,
    global_slot: globalSlot,
    kind: raw.kind,
    radius,
  });
}

/** @param {unknown} raw @param {readonly Record<string, any>[]} agents */
function normalizeResearcherSelection(raw, agents) {
  if (raw === null) {
    return null;
  }
  const selection = requireRecord(raw, "Invalid researcher selection.");
  requireExactKeys(
    selection,
    RESEARCHER_SELECTION_KEYS_V1,
    "Researcher selection has unknown or missing fields.",
  );
  const controlled = requireInteger(
    selection.controlled_global_slot,
    "Researcher controlled global slot",
  );
  const selected = selection.selected_global_slot;
  if (
    !agents.some((agent) => agent.global_slot === controlled) ||
    (selected !== null &&
      (!Number.isInteger(selected) ||
        selected < 0 ||
        !agents.some((agent) => agent.global_slot === selected)))
  ) {
    throw new TypeError("Researcher selection must join the active roster.");
  }
  return Object.freeze({
    controlled_global_slot: controlled,
    selected_global_slot: selected,
  });
}

/** @param {unknown} raw @param {Readonly<Record<string, any>> | null} selection */
function normalizeResearcherSelectedLegality(raw, selection) {
  if (raw === null) {
    return null;
  }
  const legality = requireRecord(raw, "Invalid researcher selected legality.");
  requireExactKeys(
    legality,
    RESEARCHER_SELECTED_LEGALITY_KEYS_V1,
    "Researcher selected legality has unknown or missing fields.",
  );
  const controlled = requireInteger(
    legality.controlled_global_slot,
    "Researcher legality controlled slot",
  );
  const target = requireInteger(
    legality.target_global_slot,
    "Researcher legality target slot",
  );
  const targetAction = requireInteger(
    legality.target_action,
    "Researcher legality target action",
    1,
  );
  const lane0 = requireBoolean(
    legality.lane_0_available,
    "Researcher legality lane zero",
  );
  const lane1 = requireBoolean(
    legality.lane_1_available,
    "Researcher legality lane one",
  );
  const armedLane = legality.armed_lane;
  if (
    !selection ||
    controlled !== selection.controlled_global_slot ||
    target !== selection.selected_global_slot ||
    targetAction > 10 ||
    ![null, 0, 1].includes(armedLane)
  ) {
    throw new TypeError("Researcher selected legality is incoherent.");
  }
  return Object.freeze({
    armed_lane: armedLane,
    armed_pair_legal: requireBoolean(
      legality.armed_pair_legal,
      "Researcher legality armed pair",
    ),
    controlled_global_slot: controlled,
    lane_0_available: lane0,
    lane_1_available: lane1,
    target_action: targetAction,
    target_global_slot: target,
  });
}

/** @param {Record<string, any>} raw @param {readonly Record<string, any>[]} agents */
function normalizeResearcherSpawnPad(raw, agents) {
  requireExactKeys(
    raw,
    RESEARCHER_SPAWN_PAD_KEYS_V2,
    "Researcher spawn pad has unknown or missing fields.",
  );
  const globalSlot = requireInteger(
    raw.assigned_global_slot,
    "Researcher spawn-pad assigned slot",
  );
  const assigned = agents.find((agent) => agent.global_slot === globalSlot);
  const teamId = requireInteger(raw.team_id, "Researcher spawn-pad team ID", 1);
  const teamLocalSlot = requireInteger(
    raw.team_local_slot,
    "Researcher spawn-pad team-local slot",
  );
  if (
    !assigned ||
    teamId > 2 ||
    teamLocalSlot >= 5 ||
    assigned.team_id !== teamId ||
    assigned.team_local_slot !== teamLocalSlot ||
    assigned.public_agent_id !== raw.assigned_public_agent_id
  ) {
    throw new TypeError("Researcher spawn pad must join its assigned roster identity.");
  }
  return Object.freeze({
    assigned_global_slot: globalSlot,
    assigned_public_agent_id: requireNonemptyString(
      raw.assigned_public_agent_id,
      "Researcher spawn-pad public ID",
    ),
    position: requirePoint(raw.position, "Researcher spawn-pad position"),
    team_id: teamId,
    team_local_slot: teamLocalSlot,
  });
}

/** @param {Record<string, any>} raw */
function normalizeResearcherRespawnWave(raw) {
  requireExactKeys(
    raw,
    RESEARCHER_RESPAWN_WAVE_KEYS_V2,
    "Researcher respawn wave has unknown or missing fields.",
  );
  const teamIndex = requireInteger(raw.team_index, "Researcher wave team index");
  const teamId = requireInteger(raw.team_id, "Researcher wave team ID", 1);
  const countdown = requireInteger(raw.countdown_steps, "Researcher wave countdown");
  const period = requireInteger(raw.period_steps, "Researcher wave period", 1);
  if (teamIndex > 1 || teamId !== teamIndex + 1 || countdown >= period) {
    throw new TypeError("Researcher respawn-wave team identity is invalid.");
  }
  return Object.freeze({
    countdown_steps: countdown,
    period_steps: period,
    team_id: teamId,
    team_index: teamIndex,
  });
}

/** @param {Record<string, any>} raw @param {Readonly<Record<string, any>> | null} selection @param {readonly Record<string, any>[]} agents @param {number} index */
function normalizeResearcherVisibility(raw, selection, agents, index) {
  requireExactKeys(
    raw,
    RESEARCHER_VISIBILITY_KEYS_V1,
    "Researcher visibility has unknown or missing fields.",
  );
  const candidate = agents[index];
  if (
    !selection ||
    raw.observer_global_slot !== selection.controlled_global_slot ||
    raw.candidate_global_slot !== candidate?.global_slot ||
    typeof raw.visible !== "boolean"
  ) {
    throw new TypeError(
      "Researcher V2 observer visibility must join the controlled ordered roster.",
    );
  }
  return Object.freeze({
    candidate_global_slot: raw.candidate_global_slot,
    observer_global_slot: raw.observer_global_slot,
    visible: raw.visible,
  });
}

/** @param {Record<string, any>} raw @param {Record<string, any>} frame @param {readonly Record<string, any>[]} roster */
function normalizeResearcherStatusSourceState(raw, frame, roster) {
  requireExactKeys(
    raw,
    STATUS_SOURCE_STATE_KEYS_V2,
    "Researcher status-source state has unknown or missing fields.",
  );
  if (
    raw.schema_version !== 2 ||
    raw.episode_id !== frame.episode_id ||
    raw.frame_index !== frame.frame_index ||
    raw.frame_id !== frame.frame_id
  ) {
    throw new TypeError("Researcher status-source state does not join its frame.");
  }
  const rows = requireArray(
    raw.active_statuses,
    "Researcher active status-source rows must be an array.",
  ).map((rawRow) => {
    const row = requireRecord(rawRow, "Invalid researcher status-source channel.");
    requireExactKeys(
      row,
      STATUS_SOURCE_CHANNEL_KEYS_V2,
      "Researcher status-source channel has unknown or missing fields.",
    );
    const recipientSlot = requireInteger(
      row.recipient_global_slot,
      "Researcher status-source recipient slot",
    );
    const recipient = roster.find((agent) => agent.global_slot === recipientSlot);
    const channel = requireInteger(
      row.status_channel,
      "Researcher status-source channel",
    );
    if (
      !recipient ||
      recipient.public_agent_id !== row.recipient_public_agent_id ||
      CATALOG_STATUS_ID_BY_CHANNEL_V1[channel] !== row.status_id
    ) {
      throw new TypeError(
        "Researcher status-source channel must join its recipient and catalog identity.",
      );
    }
    const evidence = requireArray(
      row.direct_source_evidence,
      "Researcher status-source evidence must be an array.",
    ).map((rawEvidence) =>
      normalizeResearcherStatusSourceEvidence(
        requireRecord(rawEvidence, "Invalid researcher status-source evidence."),
        roster,
        frame.episode_id,
        frame.frame_index,
      ),
    );
    const evidenceKeys = evidence.map(
      (row) => `${String(row.source_global_slot).padStart(2, "0")}:${row.event_id}`,
    );
    if (
      evidenceKeys.some((key, index) => index > 0 && key <= evidenceKeys[index - 1])
    ) {
      throw new TypeError(
        "Researcher status-source evidence requires canonical unique keys.",
      );
    }
    return Object.freeze({
      direct_source_evidence: Object.freeze(evidence),
      recipient_global_slot: recipientSlot,
      recipient_public_agent_id: row.recipient_public_agent_id,
      status_channel: channel,
      status_id: row.status_id,
    });
  });
  const keys = rows.map((row) => `${row.recipient_global_slot}:${row.status_channel}`);
  const directEventIds = rows.flatMap((row) =>
    row.direct_source_evidence.map((evidence) => evidence.event_id),
  );
  if (
    keys.join("|") !== [...keys].sort().join("|") ||
    new Set(keys).size !== keys.length ||
    new Set(directEventIds).size !== directEventIds.length
  ) {
    throw new TypeError("Researcher status-source rows must be canonical and unique.");
  }
  return Object.freeze({
    active_statuses: Object.freeze(rows),
    episode_id: raw.episode_id,
    frame_id: raw.frame_id,
    frame_index: raw.frame_index,
    schema_version: 2,
  });
}

/** @param {Record<string, any>} raw @param {readonly Record<string, any>[]} roster */
function normalizeResearcherAgentAnchor(raw, roster) {
  requireExactKeys(
    raw,
    RESEARCHER_AGENT_ANCHOR_KEYS_V2,
    "Researcher event anchor has unknown or missing fields.",
  );
  const globalSlot = requireInteger(raw.global_slot, "Researcher anchor global slot");
  const agent = roster.find((row) => row.global_slot === globalSlot);
  if (
    !agent ||
    agent.public_agent_id !== raw.public_agent_id ||
    !["transition_start", "post_charge", "successor"].includes(raw.phase)
  ) {
    throw new TypeError("Researcher event anchor identity or phase is invalid.");
  }
  return Object.freeze({
    global_slot: globalSlot,
    phase: raw.phase,
    position: requirePoint(raw.position, "Researcher event anchor position"),
    public_agent_id: raw.public_agent_id,
  });
}

/** @param {Record<string, any>} raw @param {readonly Record<string, any>[]} roster */
function normalizeResearcherTrajectory(raw, roster) {
  requireExactKeys(
    raw,
    RESEARCHER_TRAJECTORY_KEYS_V2,
    "Researcher phase trajectory has unknown or missing fields.",
  );
  const globalSlot = requireInteger(
    raw.global_slot,
    "Researcher trajectory global slot",
  );
  const agent = roster.find((row) => row.global_slot === globalSlot);
  if (!agent || agent.public_agent_id !== raw.public_agent_id) {
    throw new TypeError("Researcher phase trajectory does not join its agent.");
  }
  const transitionStart = normalizeResearcherAgentAnchor(
    requireRecord(raw.transition_start, "Invalid transition-start anchor."),
    roster,
  );
  const postCharge = normalizeResearcherAgentAnchor(
    requireRecord(raw.post_charge, "Invalid post-charge anchor."),
    roster,
  );
  const successor = normalizeResearcherAgentAnchor(
    requireRecord(raw.successor, "Invalid successor anchor."),
    roster,
  );
  if (
    transitionStart.phase !== "transition_start" ||
    postCharge.phase !== "post_charge" ||
    successor.phase !== "successor" ||
    [transitionStart, postCharge, successor].some(
      (anchor) =>
        anchor.global_slot !== globalSlot ||
        anchor.public_agent_id !== raw.public_agent_id,
    )
  ) {
    throw new TypeError("Researcher trajectory anchors are incoherent.");
  }
  return Object.freeze({
    global_slot: globalSlot,
    post_charge: postCharge,
    public_agent_id: raw.public_agent_id,
    successor,
    transition_start: transitionStart,
  });
}

/** @param {unknown} value @param {string} label @returns {unknown} */
function normalizeResearcherEventValue(value, label) {
  if (value === null || ["string", "number", "boolean"].includes(typeof value)) {
    if (typeof value === "number" && !Number.isFinite(value)) {
      throw new TypeError(`${label} must be finite.`);
    }
    return value;
  }
  if (Array.isArray(value)) {
    return Object.freeze(
      value.map((item, index) =>
        normalizeResearcherEventValue(item, `${label}[${index}]`),
      ),
    );
  }
  throw new TypeError(`${label} has an invalid value shape.`);
}

/** @param {number} left @param {number} right */
function researcherNumbersClose(left, right) {
  return (
    Math.abs(left - right) <=
    Math.max(1e-5, 1e-6 * Math.max(Math.abs(left), Math.abs(right)))
  );
}

/** @param {readonly number[]} left @param {readonly number[]} right */
function researcherPointsEqual(left, right) {
  return left[0] === right[0] && left[1] === right[1];
}

/** @param {unknown} value @param {string} label */
function requireResearcherEventSlot(value, label) {
  const slot = requireInteger(value, label);
  if (slot >= 10) {
    throw new TypeError(`${label} must identify one fixed global slot.`);
  }
  return slot;
}

/** @param {unknown} value @param {string} label */
function requireResearcherSignedInt32(value, label) {
  const decoded = requireInteger(value, label, -(2 ** 31));
  if (decoded >= 2 ** 31) {
    throw new TypeError(`${label} must fit signed int32.`);
  }
  return decoded;
}

/**
 * @param {unknown} value
 * @param {string} label
 * @returns {readonly number[]}
 */
function normalizeResearcherEventSlotTuple(value, label) {
  const slots = requireArray(value, `${label} must be an array.`).map((slot) =>
    requireResearcherEventSlot(slot, `${label} item`),
  );
  if (slots.some((slot, index) => index > 0 && slot <= slots[index - 1])) {
    throw new TypeError(`${label} must be sorted and unique.`);
  }
  return Object.freeze(slots);
}

/**
 * @param {unknown} value
 * @param {number} globalSlot
 * @param {"transition_start" | "post_charge" | "successor"} phase
 * @param {ReadonlyMap<number, Readonly<Record<string, any>>>} trajectories
 * @param {string} label
 * @returns {Readonly<Record<string, any>>}
 */
function requireResearcherTrajectoryAnchor(
  value,
  globalSlot,
  phase,
  trajectories,
  label,
) {
  const anchor = requireRecord(value, `${label} must be an event anchor.`);
  const trajectory = trajectories.get(globalSlot);
  const expected = trajectory?.[phase];
  if (
    !isRecord(expected) ||
    anchor.global_slot !== globalSlot ||
    anchor.phase !== phase ||
    anchor.public_agent_id !== expected.public_agent_id ||
    !Array.isArray(anchor.position) ||
    !Array.isArray(expected.position) ||
    !researcherPointsEqual(anchor.position, expected.position)
  ) {
    throw new TypeError(`${label} must join its exact phase trajectory.`);
  }
  return anchor;
}

/**
 * @param {unknown} value
 * @param {number | null} globalSlot
 * @param {"transition_start" | "post_charge" | "successor"} phase
 * @param {ReadonlyMap<number, Readonly<Record<string, any>>>} trajectories
 * @param {string} label
 */
function requireResearcherOptionalTrajectoryAnchor(
  value,
  globalSlot,
  phase,
  trajectories,
  label,
) {
  if ((value === null) !== (globalSlot === null)) {
    throw new TypeError(`${label} presence must match its nullable slot.`);
  }
  return value === null || globalSlot === null
    ? null
    : requireResearcherTrajectoryAnchor(value, globalSlot, phase, trajectories, label);
}

/**
 * @param {Record<string, any>} raw
 * @param {number} ordinal
 * @param {string} transitionId
 * @param {readonly Record<string, any>[]} roster
 * @param {ReadonlyMap<number, Readonly<Record<string, any>>>} trajectories
 * @param {readonly string[]} publicIds
 * @param {readonly boolean[]} active
 */
function normalizeResearcherEvent(
  raw,
  ordinal,
  transitionId,
  roster,
  trajectories,
  publicIds,
  active,
) {
  if (!RESEARCHER_EVENT_TYPES_V2.has(raw.event_type)) {
    throw new TypeError(`Unknown V2 researcher event type: ${raw.event_type}.`);
  }
  const eventType = /** @type {keyof typeof RESEARCHER_EVENT_SUFFIX_KEYS_V2} */ (
    raw.event_type
  );
  const suffix = RESEARCHER_EVENT_SUFFIX_KEYS_V2[eventType];
  const expectedKeys = [...RESEARCHER_EVENT_BASE_KEYS_V2, ...suffix].sort();
  const actualKeys = Object.keys(raw).sort();
  if (
    actualKeys.length !== expectedKeys.length ||
    actualKeys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new TypeError(
      `Researcher ${raw.event_type} event has unknown or missing fields (${actualKeys.join(",")} versus ${expectedKeys.join(",")}).`,
    );
  }
  /* istanbul ignore next -- retained as the shared exact-key assertion idiom. */
  requireExactKeys(
    raw,
    Object.freeze(expectedKeys),
    "Researcher event has unknown or missing fields.",
  );
  const expectedEventId = `${transitionId}:event:${String(ordinal).padStart(4, "0")}`;
  if (
    raw.ordinal !== ordinal ||
    raw.transition_id !== transitionId ||
    raw.event_id !== expectedEventId ||
    raw.phase_rank !== RESEARCHER_EVENT_PHASE_RANK_V2[eventType]
  ) {
    throw new TypeError("V2 event order, rank, or canonical identity is invalid.");
  }
  /** @type {Record<string, any>} */
  const normalized = {};
  for (const key of Object.keys(raw)) {
    const value = raw[key];
    if (
      [
        "actor_anchor",
        "agent_anchor",
        "end_anchor",
        "recipient_anchor",
        "source_anchor",
        "start_anchor",
      ].includes(key)
    ) {
      normalized[key] =
        value === null
          ? null
          : normalizeResearcherAgentAnchor(
              requireRecord(value, `Invalid researcher event ${key}.`),
              roster,
            );
    } else if (key === "team_anchor") {
      const anchor = requireRecord(value, "Invalid researcher team anchor.");
      requireExactKeys(
        anchor,
        RESEARCHER_TEAM_ANCHOR_KEYS_V2,
        "Researcher team anchor has unknown or missing fields.",
      );
      const teamIndex = requireInteger(
        anchor.team_index,
        "Researcher team anchor index",
      );
      const teamId = requireInteger(anchor.team_id, "Researcher team anchor ID", 1);
      if (anchor.phase !== "successor" || teamIndex > 1 || teamId !== teamIndex + 1) {
        throw new TypeError("Researcher team anchor is invalid.");
      }
      normalized[key] = Object.freeze({
        phase: "successor",
        team_id: teamId,
        team_index: teamIndex,
      });
    } else {
      normalized[key] = normalizeResearcherEventValue(value, `Researcher event ${key}`);
    }
  }
  /** @param {string} key */
  const eventSlot = (key) =>
    requireResearcherEventSlot(normalized[key], `Researcher event ${key}`);
  /** @param {string} key @param {number} slot @param {"transition_start" | "post_charge" | "successor"} phase */
  const eventAnchor = (key, slot, phase) =>
    requireResearcherTrajectoryAnchor(
      normalized[key],
      slot,
      phase,
      trajectories,
      `Researcher event ${key}`,
    );
  /** @param {string} key */
  const optionalSlot = (key) => (normalized[key] === null ? null : eventSlot(key));
  /** @param {string} key */
  const nonnegative = (key) =>
    requireFinite(normalized[key], `Researcher event ${key}`, 0);
  const rosterBySlot = new Map(roster.map((agent) => [agent.global_slot, agent]));
  switch (eventType) {
    case "action_rejected": {
      const actorSlot = eventSlot("actor_global_slot");
      const configuredActive = requireBoolean(
        normalized.actor_configured_active,
        "Researcher rejected actor configured-active",
      );
      if (
        normalized.actor_public_agent_id !== publicIds[actorSlot] ||
        configuredActive !== active[actorSlot] ||
        !["domain", "movement", "combat_pair"].includes(normalized.rejection_component)
      ) {
        throw new TypeError(
          "Researcher rejected action does not join batch authority.",
        );
      }
      for (const key of [
        "submitted_move_action",
        "submitted_select_target_action",
        "submitted_use_ultimate_action",
      ]) {
        normalized[key] = requireResearcherSignedInt32(
          normalized[key],
          `Researcher event ${key}`,
        );
      }
      if (configuredActive) {
        eventAnchor("actor_anchor", actorSlot, "transition_start");
      } else if (normalized.actor_anchor !== null) {
        throw new TypeError("Inactive rejected actors must remain feed-only.");
      }
      break;
    }
    case "ability_activated": {
      const sourceSlot = eventSlot("source_global_slot");
      const recipientSlot = optionalSlot("recipient_global_slot");
      if (!["basic", "ultimate"].includes(normalized.ability_component)) {
        throw new TypeError("Researcher ability component is invalid.");
      }
      eventAnchor("source_anchor", sourceSlot, "transition_start");
      requireResearcherOptionalTrajectoryAnchor(
        normalized.recipient_anchor,
        recipientSlot,
        "transition_start",
        trajectories,
        "Researcher event recipient_anchor",
      );
      break;
    }
    case "source_damage_output": {
      const sourceSlot = eventSlot("source_global_slot");
      const recipientSlot = optionalSlot("recipient_global_slot");
      nonnegative("raw_damage_output");
      nonnegative("source_modified_damage_output");
      nonnegative("recipient_damage_modifier");
      normalized.mage_damage_aura_covering_emitter_global_slots =
        normalizeResearcherEventSlotTuple(
          normalized.mage_damage_aura_covering_emitter_global_slots,
          "Researcher Mage aura emitter slots",
        );
      normalized.warrior_mitigation_aura_covering_emitter_global_slots =
        normalizeResearcherEventSlotTuple(
          normalized.warrior_mitigation_aura_covering_emitter_global_slots,
          "Researcher Warrior aura emitter slots",
        );
      eventAnchor("source_anchor", sourceSlot, "transition_start");
      requireResearcherOptionalTrajectoryAnchor(
        normalized.recipient_anchor,
        recipientSlot,
        "transition_start",
        trajectories,
        "Researcher event recipient_anchor",
      );
      break;
    }
    case "source_healing_output": {
      const sourceSlot = eventSlot("source_global_slot");
      const recipientSlot = optionalSlot("recipient_global_slot");
      nonnegative("raw_healing_output");
      nonnegative("source_modified_healing_output");
      nonnegative("recipient_healing_modifier");
      eventAnchor("source_anchor", sourceSlot, "transition_start");
      requireResearcherOptionalTrajectoryAnchor(
        normalized.recipient_anchor,
        recipientSlot,
        "transition_start",
        trajectories,
        "Researcher event recipient_anchor",
      );
      break;
    }
    case "recipient_health_resolution": {
      const recipientSlot = eventSlot("recipient_global_slot");
      nonnegative("transition_start_health");
      nonnegative("total_effective_damage");
      nonnegative("total_effective_healing");
      nonnegative("health_after_combat_resolution");
      requireFinite(
        normalized.realized_net_health_change,
        "Researcher event realized_net_health_change",
      );
      eventAnchor("recipient_anchor", recipientSlot, "transition_start");
      break;
    }
    case "combat_countdown_reset":
    case "cooldown_started":
    case "cooldown_ready": {
      const agentSlot = eventSlot("agent_global_slot");
      eventAnchor("agent_anchor", agentSlot, "transition_start");
      break;
    }
    case "agent_left_combat": {
      const agentSlot = eventSlot("agent_global_slot");
      eventAnchor("agent_anchor", agentSlot, "successor");
      break;
    }
    case "health_regenerated": {
      const agentSlot = eventSlot("agent_global_slot");
      nonnegative("actual_health_regenerated");
      eventAnchor("agent_anchor", agentSlot, "transition_start");
      break;
    }
    case "charge_phase_displacement":
    case "ordinary_movement_phase_displacement": {
      const agentSlot = eventSlot("agent_global_slot");
      const startPhase =
        eventType === "charge_phase_displacement" ? "transition_start" : "post_charge";
      const endPhase =
        eventType === "charge_phase_displacement" ? "post_charge" : "successor";
      const displacement = requirePoint(
        normalized.realized_displacement,
        "Researcher realized displacement",
      );
      normalized.realized_displacement = displacement;
      const start = eventAnchor("start_anchor", agentSlot, startPhase);
      const end = eventAnchor("end_anchor", agentSlot, endPhase);
      if (
        !researcherNumbersClose(end.position[0], start.position[0] + displacement[0]) ||
        !researcherNumbersClose(end.position[1], start.position[1] + displacement[1])
      ) {
        throw new TypeError(
          "Researcher displacement must join its recorded phase endpoints.",
        );
      }
      break;
    }
    case "agent_died": {
      const recipientSlot = eventSlot("recipient_global_slot");
      eventAnchor("recipient_anchor", recipientSlot, "successor");
      break;
    }
    case "lethal_damage_contribution": {
      const sourceSlot = eventSlot("source_global_slot");
      const recipientSlot = eventSlot("recipient_global_slot");
      nonnegative("attributed_death_damage");
      eventAnchor("source_anchor", sourceSlot, "successor");
      eventAnchor("recipient_anchor", recipientSlot, "successor");
      break;
    }
    case "status_aged_to_zero":
    case "status_broken_by_damage":
    case "status_applied":
    case "status_refreshed_or_extended":
    case "status_cleared_by_new_death": {
      const recipientSlot = eventSlot("recipient_global_slot");
      const channel = requireInteger(
        normalized.status_channel,
        "Researcher event status channel",
      );
      if (CATALOG_STATUS_ID_BY_CHANNEL_V1[channel] !== normalized.status_id) {
        throw new TypeError(
          "Researcher status event must retain its catalog identity.",
        );
      }
      eventAnchor("recipient_anchor", recipientSlot, "successor");
      if (eventType === "status_applied") {
        const sourceSlot = eventSlot("source_global_slot");
        eventAnchor("source_anchor", sourceSlot, "successor");
      }
      break;
    }
    case "spawn_shield_expired": {
      const agentSlot = eventSlot("agent_global_slot");
      eventAnchor("agent_anchor", agentSlot, "successor");
      break;
    }
    case "respawn_wave_occurred": {
      const teamIndex = requireInteger(
        normalized.team_index,
        "Researcher respawn-wave team index",
      );
      const teamId = requireInteger(
        normalized.team_id,
        "Researcher respawn-wave team ID",
        1,
      );
      const teamAnchor = requireRecord(
        normalized.team_anchor,
        "Researcher respawn-wave team anchor is invalid.",
      );
      if (
        teamIndex > 1 ||
        teamId !== teamIndex + 1 ||
        teamAnchor.phase !== "successor" ||
        teamAnchor.team_index !== teamIndex ||
        teamAnchor.team_id !== teamId
      ) {
        throw new TypeError("Researcher respawn-wave team identity is incoherent.");
      }
      break;
    }
    case "agent_respawned": {
      const agentSlot = eventSlot("agent_global_slot");
      const teamId = requireInteger(
        normalized.team_id,
        "Researcher respawned-agent team ID",
        1,
      );
      const position = requirePoint(
        normalized.realized_successor_position,
        "Researcher respawned-agent position",
      );
      normalized.realized_successor_position = position;
      const agent = rosterBySlot.get(agentSlot);
      const anchor = eventAnchor("agent_anchor", agentSlot, "successor");
      if (
        !agent ||
        teamId > 2 ||
        agent.team_id !== teamId ||
        !researcherPointsEqual(anchor.position, position)
      ) {
        throw new TypeError("Researcher respawn event identity is invalid.");
      }
      break;
    }
    case "team_deathmatch_score_changed": {
      const teamIndex = requireInteger(
        normalized.team_index,
        "Researcher Team Deathmatch scoring-team index",
      );
      const teamId = requireInteger(
        normalized.team_id,
        "Researcher Team Deathmatch scoring-team ID",
        1,
      );
      const scoreIncrement = requireResearcherSignedInt32(
        normalized.score_increment,
        "Researcher Team Deathmatch score increment",
      );
      const previousScore = requireResearcherSignedInt32(
        normalized.previous_score,
        "Researcher Team Deathmatch previous score",
      );
      const successorScore = requireResearcherSignedInt32(
        normalized.successor_score,
        "Researcher Team Deathmatch successor score",
      );
      const teamAnchor = requireRecord(
        normalized.team_anchor,
        "Researcher Team Deathmatch team anchor is invalid.",
      );
      if (
        teamIndex > 1 ||
        teamId !== teamIndex + 1 ||
        scoreIncrement <= 0 ||
        previousScore < 0 ||
        successorScore !== previousScore + scoreIncrement ||
        teamAnchor.phase !== "successor" ||
        teamAnchor.team_index !== teamIndex ||
        teamAnchor.team_id !== teamId
      ) {
        throw new TypeError("Researcher Team Deathmatch score event is incoherent.");
      }
      break;
    }
    case "team_deathmatch_completed": {
      if (
        !["team_a_win", "team_b_win", "draw"].includes(normalized.outcome) ||
        !["score_threshold", "horizon", "score_threshold_at_horizon"].includes(
          normalized.completion_basis,
        )
      ) {
        throw new TypeError("Researcher Team Deathmatch completion event is invalid.");
      }
      break;
    }
    default:
      throw new TypeError("Unknown researcher event semantic variant.");
  }
  return /** @type {Readonly<Record<string, any>>} */ (Object.freeze(normalized));
}

/** @param {Record<string, any>} batch @param {Record<string, any>} frame @param {readonly Record<string, any>[]} roster */
function normalizeResearcherEventBatch(batch, frame, roster) {
  requireExactKeys(
    batch,
    RESEARCHER_EVENT_BATCH_KEYS_V2,
    "Researcher event batch has unknown or missing fields.",
  );
  const expectedTransitionIndex = Number.isInteger(frame.incoming_transition_index)
    ? frame.incoming_transition_index
    : frame.frame_index - 1;
  if (
    batch.schema_version !== 2 ||
    batch.episode_id !== frame.episode_id ||
    batch.transition_index !== expectedTransitionIndex ||
    batch.transition_id !== frame.incoming_transition_id ||
    batch.start_frame_id !== `${frame.episode_id}:frame:${frame.frame_index - 1}` ||
    batch.successor_frame_id !== frame.frame_id ||
    !Number.isInteger(batch.start_simulator_step_count) ||
    batch.start_simulator_step_count < 0 ||
    batch.successor_simulator_step_count !== frame.simulator_step_count
  ) {
    throw new TypeError("Researcher V2 event batch does not join its live frame.");
  }
  const publicIds = requireArray(
    batch.public_agent_id_by_global_slot,
    "Researcher event-batch public IDs must be an array.",
  ).map((value) => requireNonemptyString(value, "Researcher batch public ID"));
  const active = requireBooleanVector(
    batch.configured_active_by_global_slot,
    publicIds.length,
    "Researcher batch configured-active axis",
  );
  const rosterBySlot = new Map(roster.map((agent) => [agent.global_slot, agent]));
  if (
    publicIds.length !== 10 ||
    new Set(publicIds).size !== 10 ||
    publicIds.some((publicId, globalSlot) => {
      const agent = rosterBySlot.get(globalSlot);
      return (
        active[globalSlot] !== Boolean(agent) ||
        (agent !== undefined && agent.public_agent_id !== publicId)
      );
    }) ||
    batch.successor_simulator_step_count !== batch.start_simulator_step_count + 1
  ) {
    throw new TypeError("Researcher event-batch roster axes do not join the scene.");
  }
  const trajectories = requireArray(
    batch.agent_phase_trajectories,
    "Researcher phase trajectories must be an array.",
  ).map((raw) =>
    normalizeResearcherTrajectory(
      requireRecord(raw, "Invalid researcher phase trajectory."),
      roster,
    ),
  );
  if (
    trajectories.map((row) => row.global_slot).join(",") !==
      roster.map((row) => row.global_slot).join(",") ||
    trajectories.some((trajectory) => {
      const agent = rosterBySlot.get(trajectory.global_slot);
      return (
        agent === undefined ||
        !researcherPointsEqual(trajectory.successor.position, agent.position)
      );
    })
  ) {
    throw new TypeError(
      "Researcher phase trajectories must cover and join the successor scene roster.",
    );
  }
  const trajectoryBySlot = new Map(
    trajectories.map((trajectory) => [trajectory.global_slot, trajectory]),
  );
  const events = requireArray(
    batch.events,
    "V2 event batch events must be an array.",
  ).map((raw, ordinal) =>
    normalizeResearcherEvent(
      requireRecord(raw, "Invalid V2 event row."),
      ordinal,
      batch.transition_id,
      roster,
      trajectoryBySlot,
      publicIds,
      active,
    ),
  );
  if (
    events.some(
      (event, index) => index > 0 && event.phase_rank < events[index - 1].phase_rank,
    )
  ) {
    throw new TypeError("Researcher event phase ranks must be nondecreasing.");
  }
  const authorized = Object.freeze({
    agent_phase_trajectories: Object.freeze(trajectories),
    configured_active_by_global_slot: active,
    episode_id: batch.episode_id,
    events: Object.freeze(events),
    public_agent_id_by_global_slot: Object.freeze(publicIds),
    schema_version: 2,
    start_frame_id: batch.start_frame_id,
    start_simulator_step_count: batch.start_simulator_step_count,
    successor_frame_id: batch.successor_frame_id,
    successor_simulator_step_count: batch.successor_simulator_step_count,
    transition_id: batch.transition_id,
    transition_index: batch.transition_index,
  });
  return {
    authorized,
    presentation: Object.freeze({
      ...authorized,
      simulator_step: authorized.successor_simulator_step_count,
    }),
  };
}

/**
 * @param {Record<string, any>} projection
 * @param {Record<string, any>} frame
 */
function normalizeResearcherProjection(projection, frame) {
  requireExactKeys(
    projection,
    RESEARCHER_PROJECTION_KEYS_V2,
    "Researcher projection has unknown or missing fields.",
  );
  if (projection.schema_version !== 2) {
    throw new TypeError("Researcher projection must use schema version 2.");
  }
  const sourceScene = requireRecord(
    projection.scene,
    "Researcher projection is missing its V2 scene.",
  );
  requireExactKeys(
    sourceScene,
    RESEARCHER_SCENE_KEYS_V2,
    "Researcher scene has unknown or missing fields.",
  );
  if (sourceScene.schema_version !== 2 || sourceScene.audience !== "researcher") {
    throw new TypeError("Researcher projection must contain BattlefieldSceneV2.");
  }
  if (
    sourceScene.episode_id !== frame.episode_id ||
    sourceScene.frame_id !== frame.frame_id ||
    sourceScene.frame_index !== frame.frame_index ||
    sourceScene.simulator_step_count !== frame.simulator_step_count ||
    sourceScene.incoming_transition_id !== frame.incoming_transition_id
  ) {
    throw new TypeError("Researcher projection does not join its live frame.");
  }

  const classMechanics = requireArray(
    sourceScene.class_mechanics,
    "Researcher V2 class mechanics must be an array.",
  ).map((mechanics) => normalizeClassMechanicsV2(mechanics));
  if (
    classMechanics.map((row) => row.class_id).join(",") !== "1,2,3,4,5" ||
    classMechanics.some(
      (row) =>
        row.status_mechanics
          .map((/** @type {Record<string, any>} */ status) => status.status_channel)
          .join(",") !==
        [...row.status_mechanics]
          .sort(
            (
              /** @type {Record<string, any>} */ left,
              /** @type {Record<string, any>} */ right,
            ) => left.status_channel - right.status_channel,
          )
          .map((/** @type {Record<string, any>} */ status) => status.status_channel)
          .join(","),
    ) ||
    classMechanics
      .flatMap((row) =>
        row.status_mechanics.map(
          (/** @type {Record<string, any>} */ status) => status.status_channel,
        ),
      )
      .sort((left, right) => left - right)
      .join(",") !== "0,1,2,3,4,5,6,7,8" ||
    classMechanics
      .flatMap((row) =>
        row.aura_mechanics.map(
          (/** @type {Record<string, any>} */ aura) => aura.aura_id,
        ),
      )
      .join(",") !== "mage_damage_amplification,warrior_damage_mitigation"
  ) {
    throw new TypeError("Researcher V2 class mechanics must cover exact catalog axes.");
  }
  const rawAgents = requireArray(
    sourceScene.agents,
    "Researcher V2 scene agents must be an array.",
  ).map((agent) => requireRecord(agent, "Invalid V2 agent row."));
  const normalizedAgents = rawAgents.map((agent) =>
    normalizeResearcherAgent(agent, rawAgents),
  );
  const authorizedAgents = normalizedAgents.map((agent) => agent.authorized);
  const agents = normalizedAgents.map((agent) => agent.presentation);
  const incomingEventIds = Object.freeze(
    requireArray(
      sourceScene.incoming_event_ids,
      "Researcher scene incoming event IDs must be an array.",
    ).map((eventId) => requireNonemptyString(eventId, "Researcher event ID")),
  );
  const expectedIncomingEventIds = incomingEventIds.map(
    (_eventId, ordinal) =>
      `${sourceScene.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
  );
  if (
    incomingEventIds.some(
      (eventId, ordinal) => eventId !== expectedIncomingEventIds[ordinal],
    ) ||
    authorizedAgents.some(
      (agent, index) =>
        index > 0 && agent.global_slot <= authorizedAgents[index - 1].global_slot,
    ) ||
    new Set(authorizedAgents.map((agent) => agent.public_agent_id)).size !==
      authorizedAgents.length ||
    authorizedAgents.some(
      (agent) =>
        agent.respawn_event_id !== null &&
        !incomingEventIds.includes(agent.respawn_event_id),
    )
  ) {
    throw new TypeError(
      "Researcher agents must have ordered unique identities and joined respawn evidence.",
    );
  }
  const map = normalizeResearcherMap(
    requireRecord(sourceScene.map, "Researcher scene map must be an object."),
  );
  const selection = normalizeResearcherSelection(
    sourceScene.selection,
    authorizedAgents,
  );
  const selectedLegality = normalizeResearcherSelectedLegality(
    sourceScene.next_decision_selected_legality,
    selection,
  );
  const ranges = requireArray(
    sourceScene.ranges,
    "Researcher scene ranges must be an array.",
  ).map((rawRange) =>
    normalizeResearcherRange(
      requireRecord(rawRange, "Invalid researcher range row."),
      authorizedAgents,
    ),
  );
  const spawnPads = requireArray(
    sourceScene.spawn_pads,
    "Researcher scene spawn pads must be an array.",
  ).map((rawPad) =>
    normalizeResearcherSpawnPad(
      requireRecord(rawPad, "Invalid researcher spawn-pad row."),
      authorizedAgents,
    ),
  );
  const spawnPadKeys = spawnPads.map((pad) => `${pad.team_id}:${pad.team_local_slot}`);
  if (
    spawnPads.length !== authorizedAgents.length ||
    spawnPadKeys.some((key, index) => index > 0 && key <= spawnPadKeys[index - 1]) ||
    new Set(spawnPads.map((pad) => pad.assigned_global_slot)).size !==
      authorizedAgents.length
  ) {
    throw new TypeError(
      "Researcher scene requires unique canonical spawn-pad coverage.",
    );
  }
  const respawnWaves = requireArray(
    sourceScene.respawn_waves,
    "Researcher scene respawn waves must be an array.",
  ).map((rawWave) =>
    normalizeResearcherRespawnWave(
      requireRecord(rawWave, "Invalid researcher respawn-wave row."),
    ),
  );
  if (respawnWaves.map((wave) => wave.team_index).join(",") !== "0,1") {
    throw new TypeError("Researcher scene requires exact ordered team waves.");
  }
  /** @type {Readonly<Record<string, any>>[]} */
  const auraFields = requireArray(
    sourceScene.aura_fields,
    "Researcher V2 aura fields must be an array.",
  ).map((field) =>
    normalizeResearcherAuraField(
      requireRecord(field, "Invalid V2 aura field row."),
      rawAgents,
    ),
  );
  const auraKeys = auraFields.map(
    (field) => `${String(field.source_global_slot).padStart(2, "0")}:${field.aura_id}`,
  );
  if (auraKeys.some((key, index) => index > 0 && key <= auraKeys[index - 1])) {
    throw new TypeError(
      "Researcher aura fields require unique canonical source/aura order.",
    );
  }
  const observerVisibility = requireArray(
    sourceScene.observer_visibility,
    "Researcher V2 observer visibility must be an array.",
  ).map((rawFact, index) =>
    normalizeResearcherVisibility(
      requireRecord(rawFact, "Invalid V2 observer-visibility row."),
      selection,
      authorizedAgents,
      index,
    ),
  );
  if (
    (selection !== null && observerVisibility.length !== agents.length) ||
    (selection === null && observerVisibility.length !== 0)
  ) {
    throw new TypeError(
      "Researcher V2 observer visibility must match scene selection authority.",
    );
  }
  const authorizedScene = Object.freeze({
    agents: Object.freeze(authorizedAgents),
    audience: "researcher",
    audience_badge: (() => {
      const badge = requireNonemptyString(
        sourceScene.audience_badge,
        "Researcher audience badge",
      );
      if (!badge.includes("PRIVILEGED")) {
        throw new TypeError("Researcher scenes require an explicit PRIVILEGED badge.");
      }
      return badge;
    })(),
    aura_fields: Object.freeze(
      auraFields.map(({ token_id: _tokenId, ...field }) => Object.freeze(field)),
    ),
    class_mechanics: Object.freeze(
      classMechanics.map((mechanics) =>
        Object.freeze({
          ...mechanics,
          aura_mechanics: Object.freeze(
            mechanics.aura_mechanics.map(
              (/** @type {Record<string, any>} */ mechanicsRow) => {
                const { token_id: _tokenId, ...row } = mechanicsRow;
                return Object.freeze(row);
              },
            ),
          ),
          status_mechanics: Object.freeze(
            mechanics.status_mechanics.map(
              (/** @type {Record<string, any>} */ mechanicsRow) => {
                const { token_id: _tokenId, ...row } = mechanicsRow;
                return Object.freeze(row);
              },
            ),
          ),
        }),
      ),
    ),
    episode_id: sourceScene.episode_id,
    frame_id: sourceScene.frame_id,
    frame_index: sourceScene.frame_index,
    incoming_event_ids: incomingEventIds,
    incoming_transition_id: sourceScene.incoming_transition_id,
    map,
    next_decision_selected_legality: selectedLegality,
    observer_visibility: Object.freeze(observerVisibility),
    ranges: Object.freeze(ranges),
    respawn_waves: Object.freeze(respawnWaves),
    schema_version: 2,
    selection,
    simulator_step_count: sourceScene.simulator_step_count,
    spawn_pads: Object.freeze(spawnPads),
  });
  const sceneShell = {
    ...authorizedScene,
    agents: Object.freeze(agents),
    class_mechanics: Object.freeze(classMechanics),
    aura_fields: Object.freeze(auraFields),
    selected_legality: selectedLegality,
  };
  const scene = Object.freeze({
    ...sceneShell,
    pending_route: null,
  });

  const sourceBatch = projection.incoming_events;
  const statusSourceEvidence = normalizeResearcherStatusSourceState(
    requireRecord(
      projection.status_source_evidence,
      "Researcher projection is missing status-source evidence.",
    ),
    frame,
    authorizedAgents,
  );
  const evidenceByRecipientAndChannel = new Map(
    statusSourceEvidence.active_statuses.map((row) => [
      `${row.recipient_global_slot}:${row.status_channel}`,
      row.direct_source_evidence,
    ]),
  );
  for (const agent of authorizedAgents) {
    for (const status of agent.statuses) {
      const evidence =
        evidenceByRecipientAndChannel.get(
          `${agent.global_slot}:${status.status_channel}`,
        ) ?? null;
      if (
        evidence === null ||
        evidence.length !== status.direct_source_evidence.length ||
        evidence.some((row, index) => {
          const statusRow = status.direct_source_evidence[index];
          return (
            row.event_id !== statusRow.event_id ||
            row.source_global_slot !== statusRow.source_global_slot ||
            row.source_public_agent_id !== statusRow.source_public_agent_id
          );
        })
      ) {
        throw new TypeError(
          "Researcher status rows must join authoritative status-source evidence.",
        );
      }
    }
  }
  if (
    statusSourceEvidence.active_statuses.length !==
    authorizedAgents.reduce((count, agent) => count + agent.statuses.length, 0)
  ) {
    throw new TypeError(
      "Researcher status-source evidence must cover exactly the active status rows.",
    );
  }
  const authorizedProjectionShell = {
    schema_version: 2,
    scene: authorizedScene,
    status_source_evidence: statusSourceEvidence,
  };
  if (sourceBatch === null || sourceBatch === undefined) {
    if (
      frame.frame_index !== 0 ||
      frame.incoming_transition_id !== null ||
      incomingEventIds.length !== 0
    ) {
      throw new TypeError("Only researcher frame zero may omit incoming events.");
    }
    return {
      projection: Object.freeze({
        ...authorizedProjectionShell,
        incoming_events: null,
      }),
      scene,
      eventBatch: null,
    };
  }
  const batch = requireRecord(sourceBatch, "Invalid researcher V2 event batch.");
  const normalizedBatch = normalizeResearcherEventBatch(batch, frame, authorizedAgents);
  const normalizedEvents = normalizedBatch.authorized.events;
  const sceneEventIds = requireArray(
    sourceScene.incoming_event_ids,
    "Researcher V2 scene incoming event IDs must be an array.",
  );
  if (
    sceneEventIds.length !== normalizedEvents.length ||
    sceneEventIds.some(
      (eventId, ordinal) => eventId !== normalizedEvents[ordinal].event_id,
    )
  ) {
    throw new TypeError("Researcher scene and event batch IDs do not join.");
  }
  const eventById = new Map(normalizedEvents.map((event) => [event.event_id, event]));
  if (
    authorizedAgents.some((agent) => {
      if (agent.respawn_event_id === null) {
        return false;
      }
      const event = eventById.get(agent.respawn_event_id);
      return (
        event?.event_type !== "agent_respawned" ||
        event.agent_global_slot !== agent.global_slot
      );
    })
  ) {
    throw new TypeError(
      "Researcher respawn evidence must identify the same agent's incoming respawn event.",
    );
  }
  return {
    projection: Object.freeze({
      ...authorizedProjectionShell,
      incoming_events: normalizedBatch.authorized,
    }),
    scene,
    eventBatch: normalizedBatch.presentation,
  };
}

/**
 * Decode only recipient-visible V1 status-duration columns. The fixed effect
 * class follows from the published feature channel; no source actor identity
 * or researcher-only attribution enters this boundary.
 *
 * @param {unknown} rawValues
 */
function normalizePovStatuses(rawValues) {
  const statusFeatureValues = requireArray(
    rawValues,
    "POV status features must be the exact V1 vector.",
  );
  if (
    statusFeatureValues.length !== 14 ||
    statusFeatureValues.some(
      (value) => typeof value !== "number" || !Number.isFinite(value) || value < 0,
    )
  ) {
    throw new TypeError(
      "POV status features must retain 14 finite nonnegative values.",
    );
  }
  return Object.freeze(
    POV_STATUS_DURATION_FEATURES_V1.flatMap(
      ({ offset, featureIndex, tokenId, effectClassId }) => {
        const duration = statusFeatureValues[offset];
        if (!Number.isInteger(duration)) {
          throw new TypeError(
            "POV status durations must be integer-valued V1 features.",
          );
        }
        return duration === 0
          ? []
          : [
              Object.freeze({
                token_id: tokenId,
                duration,
                status_feature_index: featureIndex,
                source_class_id: effectClassId,
                source_evidence: "effect_channel_only",
              }),
            ];
      },
    ),
  );
}

/** @param {Record<string, any>} rawMap */
function normalizePovMap(rawMap) {
  requireExactKeys(rawMap, ACTOR_POV_MAP_KEYS_V1, "POV map has unknown fields.");
  const width = requireFinite(rawMap.width, "POV map width", Number.EPSILON);
  const height = requireFinite(rawMap.height, "POV map height", Number.EPSILON);
  const seenIds = new Set();
  const obstacles = requireArray(
    rawMap.obstacles,
    "POV map obstacles must be an array.",
  ).map((rawObstacle) => {
    const obstacle = requireRecord(rawObstacle, "Invalid POV obstacle row.");
    requireExactKeys(
      obstacle,
      ACTOR_POV_OBSTACLE_KEYS_V1,
      "POV obstacle has unknown or missing fields.",
    );
    const obstacleId = requireNonemptyString(obstacle.obstacle_id, "POV obstacle ID");
    if (seenIds.has(obstacleId)) {
      throw new TypeError("POV obstacle IDs must be unique.");
    }
    seenIds.add(obstacleId);
    const center = requirePoint(obstacle.center, "POV obstacle center");
    const theta = requireFinite(obstacle.theta, "POV obstacle theta");
    if (obstacle.kind === "pillar") {
      if (
        requireFinite(obstacle.radius, "POV pillar radius", Number.EPSILON) <= 0 ||
        obstacle.width !== null ||
        obstacle.height !== null
      ) {
        throw new TypeError("POV pillar geometry is invalid.");
      }
    } else if (obstacle.kind === "wall") {
      if (
        requireFinite(obstacle.width, "POV wall width", Number.EPSILON) <= 0 ||
        requireFinite(obstacle.height, "POV wall height", Number.EPSILON) <= 0 ||
        obstacle.radius !== null
      ) {
        throw new TypeError("POV wall geometry is invalid.");
      }
    } else {
      throw new TypeError("POV obstacle kind is invalid.");
    }
    return Object.freeze({
      obstacle_id: obstacleId,
      kind: obstacle.kind,
      center,
      radius: obstacle.radius,
      width: obstacle.width,
      height: obstacle.height,
      theta,
    });
  });
  return Object.freeze({ width, height, obstacles: Object.freeze(obstacles) });
}

/** @param {Record<string, any>} actor */
function normalizePovSelf(actor) {
  requireExactKeys(
    actor,
    ACTOR_POV_SELF_KEYS_V1,
    "POV self actor has unknown or missing fields.",
  );
  const globalSlot = requireInteger(actor.global_slot, "POV self global slot");
  const teamLocalSlot = requireInteger(
    actor.team_local_slot,
    "POV self team-local slot",
  );
  const teamId = requireInteger(actor.team_id, "POV self team ID", 1);
  const classId = requireInteger(actor.class_id, "POV self class ID", 1);
  if (
    globalSlot >= 10 ||
    teamLocalSlot >= 5 ||
    teamId > 2 ||
    classId > 5 ||
    teamLocalSlot !== globalSlot % 5 ||
    teamId !== (globalSlot < 5 ? 1 : 2)
  ) {
    throw new TypeError("POV self identity axes are invalid.");
  }
  const currentHealth = requireFinite(
    actor.current_health,
    "POV self current health",
    0,
  );
  const maxHealth = requireFinite(actor.max_health, "POV self maximum health", 0);
  const radius = requireFinite(actor.radius, "POV self radius", Number.EPSILON);
  if (currentHealth > maxHealth || maxHealth <= 0 || radius <= 0) {
    throw new TypeError("POV self body values are invalid.");
  }
  const statusFeatureValues = Object.freeze(
    requireArray(
      actor.status_feature_values,
      "POV self status features must be an array.",
    ).map((value) => requireFinite(value, "POV self status feature", 0)),
  );
  normalizePovStatuses(statusFeatureValues);
  return Object.freeze({
    global_slot: globalSlot,
    public_agent_id: requireNonemptyString(
      actor.public_agent_id,
      "POV self public agent ID",
    ),
    team_local_slot: teamLocalSlot,
    team_id: teamId,
    class_id: classId,
    position: requirePoint(actor.position, "POV self position"),
    radius,
    alive: requireBoolean(actor.alive, "POV self alive"),
    current_health: currentHealth,
    max_health: maxHealth,
    effective_movement_speed: requireFinite(
      actor.effective_movement_speed,
      "POV self effective movement speed",
      0,
    ),
    ultimate_cooldown_remaining: requireInteger(
      actor.ultimate_cooldown_remaining,
      "POV self Ultimate cooldown",
    ),
    steps_until_out_of_combat: requireInteger(
      actor.steps_until_out_of_combat,
      "POV self combat countdown",
    ),
    spawn_shield_remaining: requireInteger(
      actor.spawn_shield_remaining,
      "POV self spawn shield",
    ),
    status_feature_values: statusFeatureValues,
  });
}

/**
 * @param {Record<string, any>} body
 * @param {number} index
 * @param {number} selfTeamId
 */
function normalizePovBody(body, index, selfTeamId) {
  requireExactKeys(
    body,
    ACTOR_POV_BODY_KEYS_V1,
    "POV visible body has unknown or missing fields.",
  );
  const relation = body.relation;
  const observationRow = requireInteger(
    body.observation_row,
    `POV body ${index} observation row`,
  );
  const teamId = requireInteger(body.team_id, `POV body ${index} team ID`, 1);
  const classId = requireInteger(body.class_id, `POV body ${index} class ID`, 1);
  if (
    !["ally", "enemy"].includes(relation) ||
    observationRow >= 5 ||
    teamId > 2 ||
    classId > 5 ||
    teamId !== (relation === "ally" ? selfTeamId : 3 - selfTeamId)
  ) {
    throw new TypeError("POV visible body identity axes are invalid.");
  }
  const currentHealth = requireFinite(
    body.current_health,
    `POV body ${index} current health`,
    0,
  );
  const maxHealth = requireFinite(
    body.max_health,
    `POV body ${index} maximum health`,
    Number.EPSILON,
  );
  const radius = requireFinite(body.radius, `POV body ${index} radius`, Number.EPSILON);
  if (currentHealth > maxHealth || maxHealth <= 0 || radius <= 0) {
    throw new TypeError("POV visible body values are invalid.");
  }
  const statusFeatureValues = Object.freeze(
    requireArray(
      body.status_feature_values,
      `POV body ${index} status features must be an array.`,
    ).map((value) => requireFinite(value, `POV body ${index} status feature`, 0)),
  );
  normalizePovStatuses(statusFeatureValues);
  return Object.freeze({
    relation,
    observation_row: observationRow,
    public_agent_id: requireNonemptyString(
      body.public_agent_id,
      `POV body ${index} public agent ID`,
    ),
    position: requirePoint(body.position, `POV body ${index} position`),
    radius,
    team_id: teamId,
    class_id: classId,
    alive: requireBoolean(body.alive, `POV body ${index} alive`),
    current_health: currentHealth,
    max_health: maxHealth,
    effective_movement_speed: requireFinite(
      body.effective_movement_speed,
      `POV body ${index} effective movement speed`,
      0,
    ),
    ultimate_cooldown_remaining: requireInteger(
      body.ultimate_cooldown_remaining,
      `POV body ${index} Ultimate cooldown`,
    ),
    steps_until_out_of_combat: requireInteger(
      body.steps_until_out_of_combat,
      `POV body ${index} combat countdown`,
    ),
    status_feature_values: statusFeatureValues,
  });
}

/** @param {Record<string, any>} pad @param {number} index */
function normalizePovSpawnPad(pad, index) {
  requireExactKeys(
    pad,
    ACTOR_POV_SPAWN_PAD_KEYS_V1,
    "POV spawn pad has unknown or missing fields.",
  );
  const teamIndex = requireInteger(
    pad.actor_relative_team_index,
    `POV spawn pad ${index} team index`,
  );
  const teamLocalSlot = requireInteger(
    pad.team_local_slot,
    `POV spawn pad ${index} team-local slot`,
  );
  const expectedRelation = teamIndex === 0 ? "own" : "opponent";
  const expectedLabel = teamIndex === 0 ? "Own Team" : "Opponent Team";
  if (
    teamIndex > 1 ||
    teamLocalSlot >= 5 ||
    pad.team_relation !== expectedRelation ||
    pad.team_label !== expectedLabel
  ) {
    throw new TypeError("POV spawn pad axes are invalid.");
  }
  return Object.freeze({
    actor_relative_team_index: teamIndex,
    team_relation: expectedRelation,
    team_label: expectedLabel,
    team_local_slot: teamLocalSlot,
    position: requirePoint(pad.position, `POV spawn pad ${index} position`),
    configured_active: requireBoolean(
      pad.configured_active,
      `POV spawn pad ${index} configured-active`,
    ),
    currently_alive: requireBoolean(
      pad.currently_alive,
      `POV spawn pad ${index} alive`,
    ),
    spawn_shield_remaining: requireInteger(
      pad.spawn_shield_remaining,
      `POV spawn pad ${index} spawn shield`,
    ),
  });
}

/** @param {Record<string, any>} wave @param {number} index */
function normalizePovRespawnWave(wave, index) {
  requireExactKeys(
    wave,
    ACTOR_POV_RESPAWN_WAVE_KEYS_V1,
    "POV respawn wave has unknown or missing fields.",
  );
  const expectedRelation = index === 0 ? "own" : "opponent";
  const expectedLabel = index === 0 ? "Own Team" : "Opponent Team";
  if (
    requireInteger(
      wave.actor_relative_team_index,
      `POV respawn wave ${index} team index`,
    ) !== index ||
    wave.team_relation !== expectedRelation ||
    wave.team_label !== expectedLabel
  ) {
    throw new TypeError("POV respawn wave axes are invalid.");
  }
  return Object.freeze({
    actor_relative_team_index: index,
    team_relation: expectedRelation,
    team_label: expectedLabel,
    period_steps: requireInteger(
      wave.period_steps,
      `POV respawn wave ${index} period`,
      1,
    ),
    countdown_steps: requireInteger(
      wave.countdown_steps,
      `POV respawn wave ${index} countdown`,
    ),
  });
}

/** @param {Record<string, any>} mask */
function normalizePovActionMask(mask) {
  requireExactKeys(
    mask,
    ACTOR_POV_ACTION_MASK_KEYS_V1,
    "POV action mask has unknown or missing fields.",
  );
  if (
    mask.schema_id !== "marl_battlegrounds.evaluation.actor_pov_action_mask" ||
    mask.schema_version !== 1
  ) {
    throw new TypeError("POV action mask identity is invalid.");
  }
  const move = requireBooleanVector(mask.move, 9, "POV movement mask");
  const selectTarget = requireBooleanVector(mask.select_target, 11, "POV target mask");
  const useUltimate = requireBooleanVector(mask.use_ultimate, 2, "POV Ultimate mask");
  const joint = Object.freeze(
    requireArray(
      mask.select_target_use_ultimate_joint,
      "POV joint mask must be an array.",
    ).map((row, index) => requireBooleanVector(row, 2, `POV joint mask row ${index}`)),
  );
  if (joint.length !== 11) {
    throw new TypeError("POV joint mask must contain 11 target rows.");
  }
  const targetMarginal = joint.map((row) => row.some(Boolean));
  const ultimateMarginal = [0, 1].map((lane) => joint.some((row) => row[lane]));
  if (
    targetMarginal.some((value, index) => value !== selectTarget[index]) ||
    ultimateMarginal.some((value, index) => value !== useUltimate[index])
  ) {
    throw new TypeError("POV action-mask marginals are incoherent.");
  }
  return Object.freeze({
    schema_id: mask.schema_id,
    schema_version: 1,
    move,
    select_target: selectTarget,
    use_ultimate: useUltimate,
    select_target_use_ultimate_joint: joint,
  });
}

/**
 * @param {Record<string, any>} cue
 * @param {number} ordinal
 * @param {string} transitionId
 */
function normalizePovCue(cue, ordinal, transitionId) {
  if (!POV_CUE_TYPES_V1.has(cue.cue_type)) {
    throw new TypeError(`Unknown actor-POV cue type: ${cue.cue_type}.`);
  }
  const suffix =
    ACTOR_POV_CUE_SUFFIX_KEYS_V1[
      /** @type {keyof typeof ACTOR_POV_CUE_SUFFIX_KEYS_V1} */ (cue.cue_type)
    ];
  requireExactKeys(
    cue,
    [...ACTOR_POV_CUE_BASE_KEYS_V1, ...suffix].sort(),
    "POV cue has unknown or missing fields.",
  );
  const expectedCueId = `${transitionId}:cue:${ordinal}`;
  if (
    cue.schema_id !== "marl_battlegrounds.evaluation.actor_pov_cue" ||
    cue.schema_version !== 1 ||
    cue.ordinal !== ordinal ||
    cue.pov_transition_id !== transitionId ||
    cue.cue_id !== expectedCueId
  ) {
    throw new TypeError("POV cue order or local identity is invalid.");
  }
  /** @type {Record<string, any>} */
  let payload;
  switch (cue.cue_type) {
    case "own_action_outcome": {
      if (!["accepted", "rejected"].includes(cue.outcome)) {
        throw new TypeError("POV action-outcome cue is invalid.");
      }
      payload = { outcome: cue.outcome };
      break;
    }
    case "own_position_changed": {
      const start = requirePoint(cue.start_position, "POV cue start position");
      const successor = requirePoint(
        cue.successor_position,
        "POV cue successor position",
      );
      if (start[0] === successor[0] && start[1] === successor[1]) {
        throw new TypeError("POV position cue must record a change.");
      }
      payload = { start_position: start, successor_position: successor };
      break;
    }
    case "own_health_changed": {
      const start = requireFinite(cue.start_health, "POV cue start health");
      const successor = requireFinite(cue.successor_health, "POV cue successor health");
      if (start === successor) {
        throw new TypeError("POV health cue must record a change.");
      }
      payload = { start_health: start, successor_health: successor };
      break;
    }
    case "own_status_changed": {
      const changed = Object.freeze(
        requireArray(
          cue.changed_feature_indices,
          "POV status-cue indices must be an array.",
        ).map((value) => requireInteger(value, "POV status-cue feature index", 15)),
      );
      const start = Object.freeze(
        requireArray(
          cue.start_values,
          "POV status-cue start values must be an array.",
        ).map((value) => requireFinite(value, "POV status-cue start value")),
      );
      const successor = Object.freeze(
        requireArray(
          cue.successor_values,
          "POV status-cue successor values must be an array.",
        ).map((value) => requireFinite(value, "POV status-cue successor value")),
      );
      if (
        changed.length === 0 ||
        changed.length !== start.length ||
        changed.length !== successor.length ||
        changed.some((value) => value >= 29) ||
        changed.some((value, index) => index > 0 && value <= changed[index - 1]) ||
        start.some((value, index) => value === successor[index])
      ) {
        throw new TypeError("POV status cue is incoherent.");
      }
      payload = {
        changed_feature_indices: changed,
        start_values: start,
        successor_values: successor,
      };
      break;
    }
    case "own_cooldown_changed": {
      const start = requireFinite(
        cue.start_remaining_ticks,
        "POV cue start cooldown",
        0,
      );
      const successor = requireFinite(
        cue.successor_remaining_ticks,
        "POV cue successor cooldown",
        0,
      );
      if (start === successor) {
        throw new TypeError("POV cooldown cue must record a change.");
      }
      payload = {
        start_remaining_ticks: start,
        successor_remaining_ticks: successor,
      };
      break;
    }
    case "own_lifecycle_changed": {
      const startActive = requireBoolean(cue.start_active, "POV cue start active");
      const startAlive = requireBoolean(cue.start_alive, "POV cue start alive");
      const startShield = requireInteger(
        cue.start_spawn_shield_remaining_ticks,
        "POV cue start spawn shield",
      );
      const successorActive = requireBoolean(
        cue.successor_active,
        "POV cue successor active",
      );
      const successorAlive = requireBoolean(
        cue.successor_alive,
        "POV cue successor alive",
      );
      const successorShield = requireInteger(
        cue.successor_spawn_shield_remaining_ticks,
        "POV cue successor spawn shield",
      );
      if (
        startActive === successorActive &&
        startAlive === successorAlive &&
        startShield === successorShield
      ) {
        throw new TypeError("POV lifecycle cue must record a change.");
      }
      payload = {
        start_active: startActive,
        successor_active: successorActive,
        start_alive: startAlive,
        successor_alive: successorAlive,
        start_spawn_shield_remaining_ticks: startShield,
        successor_spawn_shield_remaining_ticks: successorShield,
      };
      break;
    }
    case "visible_body_observation_changed": {
      if (!["ally", "enemy"].includes(cue.relation)) {
        throw new TypeError("POV visible-body cue relation is invalid.");
      }
      const row = requireInteger(cue.observation_row, "POV visible-body cue row");
      const startVisible = requireBoolean(
        cue.start_visible,
        "POV visible-body cue start visibility",
      );
      const successorVisible = requireBoolean(
        cue.successor_visible,
        "POV visible-body cue successor visibility",
      );
      const changed = requireBoolean(
        cue.observed_payload_changed,
        "POV visible-body cue payload flag",
      );
      if (
        row >= 5 ||
        (!startVisible && !successorVisible) ||
        (startVisible === successorVisible && !changed)
      ) {
        throw new TypeError("POV visible-body cue is incoherent.");
      }
      payload = {
        relation: cue.relation,
        observation_row: row,
        start_visible: startVisible,
        successor_visible: successorVisible,
        observed_payload_changed: changed,
      };
      break;
    }
    case "episode_ended": {
      const terminated = requireBoolean(cue.terminated, "POV ended cue terminated");
      const truncated = requireBoolean(cue.truncated, "POV ended cue truncated");
      if (
        (!terminated && !truncated) ||
        (cue.public_end_reason !== null && typeof cue.public_end_reason !== "string")
      ) {
        throw new TypeError("POV episode-ended cue is invalid.");
      }
      payload = {
        terminated,
        truncated,
        public_end_reason: cue.public_end_reason,
      };
      break;
    }
    default:
      throw new TypeError("Unknown actor-POV cue type.");
  }
  return Object.freeze({
    schema_id: cue.schema_id,
    schema_version: 1,
    cue_id: cue.cue_id,
    pov_transition_id: transitionId,
    ordinal,
    cue_type: cue.cue_type,
    ...payload,
  });
}

/** @param {Record<string, any>} value @param {string} label */
function normalizePovTarget(value, label) {
  requireExactKeys(
    value,
    ACTOR_POV_TARGET_KEYS_V1,
    `${label} has unknown or missing fields.`,
  );
  const targetAction = requireInteger(
    value.target_action,
    `${label} action`,
    -1_000_000,
  );
  const inDomain = targetAction >= 0 && targetAction < 11;
  if (
    (!inDomain && value.public_agent_id !== null) ||
    (targetAction === 0 && value.public_agent_id !== null) ||
    (targetAction > 0 && targetAction < 11 && typeof value.public_agent_id !== "string")
  ) {
    throw new TypeError(`${label} identity is invalid.`);
  }
  return Object.freeze({
    target_action: targetAction,
    public_agent_id:
      value.public_agent_id === null
        ? null
        : requireNonemptyString(value.public_agent_id, `${label} public ID`),
  });
}

/**
 * @param {Record<string, any>} value
 * @param {string} label
 * @param {readonly (string | null)[]} targetIds
 */
function normalizePovActionCard(value, label, targetIds) {
  requireExactKeys(
    value,
    ACTOR_POV_ACTION_KEYS_V1,
    `${label} has unknown or missing fields.`,
  );
  const target = normalizePovTarget(
    requireRecord(value.target, `${label} target must be an object.`),
    `${label} target`,
  );
  if (
    (target.target_action >= 0 &&
      target.target_action < targetIds.length &&
      target.public_agent_id !== targetIds[target.target_action]) ||
    typeof value.summary !== "string"
  ) {
    throw new TypeError(`${label} does not join the POV target axis.`);
  }
  return Object.freeze({
    move_action: requireInteger(value.move_action, `${label} movement`, -1_000_000),
    target,
    use_ultimate_action: requireInteger(
      value.use_ultimate_action,
      `${label} Ultimate lane`,
      -1_000_000,
    ),
    summary: value.summary,
  });
}

/**
 * @param {Record<string, any>} value
 * @param {readonly (string | null)[]} targetIds
 * @param {string} selfId
 */
function normalizePovActionResult(value, targetIds, selfId) {
  requireExactKeys(
    value,
    ACTOR_POV_RESULT_KEYS_V1,
    "POV result has unknown or missing fields.",
  );
  const submitted = normalizePovActionCard(
    requireRecord(value.submitted, "POV submitted action must be an object."),
    "POV submitted action",
    targetIds,
  );
  const accepted = normalizePovActionCard(
    requireRecord(value.accepted, "POV accepted action must be an object."),
    "POV accepted action",
    targetIds,
  );
  const outOfDomain = requireBoolean(
    value.submitted_tuple_is_out_of_domain,
    "POV submitted out-of-domain flag",
  );
  const movementRejected = requireBoolean(
    value.movement_rejected,
    "POV movement rejection",
  );
  const combatRejected = requireBoolean(
    value.combat_pair_rejected,
    "POV combat rejection",
  );
  const movementAccepted = requireBoolean(
    value.movement_accepted,
    "POV movement acceptance",
  );
  const acceptedInDomain =
    accepted.move_action >= 0 &&
    accepted.move_action < 9 &&
    accepted.target.target_action >= 0 &&
    accepted.target.target_action < 11 &&
    accepted.use_ultimate_action >= 0 &&
    accepted.use_ultimate_action < 2;
  const submittedInDomain =
    submitted.move_action >= 0 &&
    submitted.move_action < 9 &&
    submitted.target.target_action >= 0 &&
    submitted.target.target_action < 11 &&
    submitted.use_ultimate_action >= 0 &&
    submitted.use_ultimate_action < 2;
  const expectedCombatResult =
    outOfDomain || combatRejected
      ? "rejected"
      : accepted.target.target_action === 0 && accepted.use_ultimate_action === 0
        ? "canonical_noop"
        : "accepted";
  if (
    value.actor_public_agent_id !== selfId ||
    !acceptedInDomain ||
    outOfDomain === submittedInDomain ||
    movementAccepted !== !(outOfDomain || movementRejected) ||
    value.combat_result !== expectedCombatResult
  ) {
    throw new TypeError("POV result truth is incoherent.");
  }
  return Object.freeze({
    actor_public_agent_id: selfId,
    submitted,
    accepted,
    submitted_tuple_is_out_of_domain: outOfDomain,
    movement_rejected: movementRejected,
    combat_pair_rejected: combatRejected,
    movement_accepted: movementAccepted,
    combat_result: expectedCombatResult,
  });
}

/**
 * @param {Record<string, any>} rawHud
 * @param {Readonly<Record<string, any>>} selfActor
 * @param {Readonly<Record<string, any>>} mask
 * @param {readonly Readonly<Record<string, any>>[]} visibleBodies
 * @param {Record<string, any>} frame
 */
function normalizePovHud(rawHud, selfActor, mask, visibleBodies, frame) {
  requireExactKeys(
    rawHud,
    ACTOR_POV_HUD_KEYS_V1,
    "POV HUD has unknown or missing fields.",
  );
  const selfId = selfActor.public_agent_id;
  if (
    rawHud.controlled_public_agent_id !== selfId ||
    !["joint_turn", "scripted_playback"].includes(rawHud.pending_submission_scope)
  ) {
    throw new TypeError("POV HUD actor or submission scope is invalid.");
  }
  const movementLegalities = Object.freeze(
    requireArray(
      rawHud.movement_legalities,
      "POV movement legalities must be an array.",
    ).map((rawRow, index) => {
      const row = requireRecord(rawRow, "Invalid POV movement-legality row.");
      requireExactKeys(
        row,
        ACTOR_POV_MOVEMENT_LEGALITY_KEYS_V1,
        "POV movement-legality row has unknown or missing fields.",
      );
      if (
        row.move_action !== index ||
        requireBoolean(row.available, "POV movement availability") !== mask.move[index]
      ) {
        throw new TypeError("POV movement legalities do not join the action mask.");
      }
      return Object.freeze({ move_action: index, available: row.available });
    }),
  );
  if (movementLegalities.length !== 9) {
    throw new TypeError("POV movement legalities must cover the exact action axis.");
  }
  /** @type {(string | null)[]} */
  const targetIds = [];
  const candidateLegalities = Object.freeze(
    requireArray(
      rawHud.candidate_legalities,
      "POV candidate legalities must be an array.",
    ).map((rawRow, index) => {
      const row = requireRecord(rawRow, "Invalid POV candidate-legality row.");
      requireExactKeys(
        row,
        ACTOR_POV_CANDIDATE_KEYS_V1,
        "POV candidate-legality row has unknown or missing fields.",
      );
      const target = normalizePovTarget(
        requireRecord(row.target, "POV candidate target must be an object."),
        "POV candidate target",
      );
      const lane0 = requireBoolean(row.lane_0_available, "POV lane-zero availability");
      const lane1 = requireBoolean(row.lane_1_available, "POV lane-one availability");
      const basic = requireBoolean(row.basic_available, "POV Basic availability");
      const ultimate = requireBoolean(
        row.ultimate_available,
        "POV Ultimate availability",
      );
      if (
        target.target_action !== index ||
        lane0 !== mask.select_target_use_ultimate_joint[index]?.[0] ||
        lane1 !== mask.select_target_use_ultimate_joint[index]?.[1] ||
        basic !== (index > 0 && lane0) ||
        ultimate !== lane1
      ) {
        throw new TypeError("POV candidate legalities do not join the action mask.");
      }
      targetIds.push(target.public_agent_id);
      return Object.freeze({
        target,
        lane_0_available: lane0,
        lane_1_available: lane1,
        basic_available: basic,
        ultimate_available: ultimate,
      });
    }),
  );
  if (
    candidateLegalities.length !== 11 ||
    targetIds[0] !== null ||
    new Set(targetIds.slice(1)).size !== 10 ||
    targetIds[selfActor.team_local_slot + 1] !== selfId ||
    visibleBodies.some((body) => {
      const targetAction =
        body.relation === "ally" ? body.observation_row + 1 : body.observation_row + 6;
      return targetIds[targetAction] !== body.public_agent_id;
    })
  ) {
    throw new TypeError("POV candidate identities do not join recipient axes.");
  }
  const pending = requireRecord(
    rawHud.pending_action,
    "POV pending action must be an object.",
  );
  requireExactKeys(
    pending,
    ACTOR_POV_PENDING_KEYS_V1,
    "POV pending action has unknown or missing fields.",
  );
  const pendingTarget = normalizePovTarget(
    requireRecord(pending.target, "POV pending target must be an object."),
    "POV pending target",
  );
  const moveAction = requireInteger(pending.move_action, "POV pending movement");
  const lane = pending.armed_lane;
  const origin = pending.arm_origin;
  const pairValue = pending.pair_mask_value;
  const expectedLabel =
    rawHud.pending_submission_scope === "scripted_playback"
      ? "PLAYBACK / INSPECTION ONLY"
      : "PENDING / WILL SUBMIT";
  if (
    pending.actor_public_agent_id !== selfId ||
    pending.label !== expectedLabel ||
    moveAction >= 9 ||
    pendingTarget.target_action < 0 ||
    pendingTarget.target_action >= targetIds.length ||
    pendingTarget.public_agent_id !== targetIds[pendingTarget.target_action] ||
    ![null, 0, 1].includes(lane) ||
    ![null, "automatic", "explicit"].includes(origin) ||
    (lane === null) !== (origin === null) ||
    requireBoolean(pending.movement_mask_value, "POV pending movement-mask value") !==
      mask.move[moveAction] ||
    (lane === null
      ? pairValue !== null
      : requireBoolean(pairValue, "POV pending pair-mask value") !==
        mask.select_target_use_ultimate_joint[pendingTarget.target_action][lane]) ||
    typeof pending.summary !== "string"
  ) {
    throw new TypeError("POV pending action is incoherent.");
  }
  const normalizedPending = Object.freeze({
    label: expectedLabel,
    actor_public_agent_id: selfId,
    move_action: moveAction,
    target: pendingTarget,
    armed_lane: lane,
    arm_origin: origin,
    movement_mask_value: pending.movement_mask_value,
    pair_mask_value: pairValue,
    summary: pending.summary,
  });
  let latest = null;
  if (rawHud.latest_transition !== null) {
    const rawLatest = requireRecord(
      rawHud.latest_transition,
      "POV latest transition must be an object or null.",
    );
    requireExactKeys(
      rawLatest,
      ACTOR_POV_LATEST_KEYS_V1,
      "POV latest transition has unknown or missing fields.",
    );
    if (frame.frame_index === 0) {
      throw new TypeError("POV frame zero cannot carry a latest transition.");
    }
    const latestTransitionIndex = requireInteger(
      rawLatest.transition_index,
      "POV latest transition index",
    );
    const latestTransitionId = requireNonemptyString(
      rawLatest.pov_transition_id,
      "POV latest transition ID",
    );
    if (
      rawLatest.label !== "LATEST ACCEPTED RESULT" ||
      latestTransitionIndex !== frame.frame_index - 1 ||
      latestTransitionId !== frame.incoming_pov_transition_id ||
      !["interactive", "scripted"].includes(rawLatest.submission_kind)
    ) {
      throw new TypeError("POV latest transition does not join the envelope.");
    }
    latest = Object.freeze({
      label: rawLatest.label,
      transition_index: latestTransitionIndex,
      pov_transition_id: latestTransitionId,
      submission_kind: rawLatest.submission_kind,
      actor: normalizePovActionResult(
        requireRecord(rawLatest.actor, "POV result must be an object."),
        targetIds,
        selfId,
      ),
    });
  }
  const diagnostics = Object.freeze(
    requireArray(rawHud.diagnostics, "POV diagnostics must be an array.").map(
      (rawFact) => {
        const fact = requireRecord(rawFact, "Invalid POV diagnostic fact.");
        requireExactKeys(
          fact,
          DIAGNOSTIC_FACT_KEYS_V1,
          "POV diagnostic fact has unknown or missing fields.",
        );
        return Object.freeze({
          fact_id: requireNonemptyString(fact.fact_id, "POV diagnostic fact ID"),
          label: requireNonemptyString(fact.label, "POV diagnostic label"),
          value: requireNonemptyString(fact.value, "POV diagnostic value"),
          technical: requireBoolean(fact.technical, "POV diagnostic technical flag"),
        });
      },
    ),
  );
  return Object.freeze({
    controlled_public_agent_id: selfId,
    pending_submission_scope: rawHud.pending_submission_scope,
    pending_action: normalizedPending,
    latest_transition: latest,
    movement_legalities: movementLegalities,
    candidate_legalities: candidateLegalities,
    diagnostics,
  });
}

/**
 * Convert the already recipient-sliced POV projection into the common visual
 * shell without assigning global identities to visible observation rows.
 *
 * @param {Record<string, any>} projection
 * @param {Record<string, any>} frame
 */
function normalizePovProjection(projection, frame) {
  requireExactKeys(
    projection,
    ACTOR_POV_PROJECTION_KEYS_V1,
    "POV projection has unknown or missing fields.",
  );
  const sourceScene = requireRecord(
    projection.scene,
    "POV projection is missing its authorized scene.",
  );
  requireExactKeys(
    sourceScene,
    ACTOR_POV_SCENE_KEYS_V1,
    "POV scene has unknown or missing fields.",
  );
  if (
    sourceScene.schema_version !== 1 ||
    sourceScene.observation_materialization !== "exact_no_shared_obs_actor_input" ||
    typeof sourceScene.audience_badge !== "string" ||
    !sourceScene.audience_badge.includes("AGENT POV")
  ) {
    throw new TypeError("POV projection must contain the exact V1 actor-input roots.");
  }
  const selfActor = normalizePovSelf(
    requireRecord(sourceScene.self_actor, "POV scene is missing its self actor."),
  );
  const expectedPovFrameId = `${frame.episode_id}:actor-pov:${selfActor.public_agent_id}:frame:${frame.frame_index}`;
  if (
    sourceScene.episode_id !== frame.episode_id ||
    sourceScene.source_frame_id !== frame.frame_id ||
    sourceScene.frame_index !== frame.frame_index ||
    sourceScene.simulator_step_count !== frame.simulator_step_count ||
    sourceScene.pov_frame_id !== expectedPovFrameId
  ) {
    throw new TypeError("POV projection does not join its live frame.");
  }
  const map = normalizePovMap(
    requireRecord(sourceScene.map, "POV scene map must be an object."),
  );
  const visibleBodies = requireArray(
    sourceScene.visible_bodies,
    "POV visible bodies must be an array.",
  ).map((body, index) =>
    normalizePovBody(
      requireRecord(body, "Invalid POV body row."),
      index,
      selfActor.team_id,
    ),
  );
  const bodyKeys = visibleBodies.map(
    (body) => `${body.relation}:${body.observation_row}`,
  );
  if (
    bodyKeys.length !== new Set(bodyKeys).size ||
    bodyKeys.join("|") !== [...bodyKeys].sort().join("|")
  ) {
    throw new TypeError("POV visible bodies require unique canonical keys.");
  }
  const spawnPads = requireArray(
    sourceScene.spawn_pads,
    "POV spawn pads must be an array.",
  ).map((pad, index) =>
    normalizePovSpawnPad(requireRecord(pad, "Invalid POV spawn-pad row."), index),
  );
  const padKeys = spawnPads.map(
    (pad) => `${pad.actor_relative_team_index}:${pad.team_local_slot}`,
  );
  if (
    padKeys.length !== new Set(padKeys).size ||
    padKeys.join("|") !== [...padKeys].sort().join("|")
  ) {
    throw new TypeError("POV spawn pads require unique canonical keys.");
  }
  const respawnWaves = requireArray(
    sourceScene.respawn_waves,
    "POV respawn waves must be an array.",
  ).map((wave, index) =>
    normalizePovRespawnWave(
      requireRecord(wave, "Invalid POV respawn-wave row."),
      index,
    ),
  );
  if (respawnWaves.length !== 2) {
    throw new TypeError("POV respawn waves must contain the exact team axis.");
  }
  const actionMask = normalizePovActionMask(
    requireRecord(
      projection.next_decision_action_mask,
      "POV action mask must be an object.",
    ),
  );
  const selfStatuses = normalizePovStatuses(selfActor.status_feature_values);
  const selfAgent = Object.freeze({
    ...selfActor,
    effective_speed: selfActor.effective_movement_speed,
    ultimate_cooldown: selfActor.ultimate_cooldown_remaining,
    statuses: Object.freeze(selfStatuses),
  });
  const observedBodies = visibleBodies.map((row) => {
    return Object.freeze({
      ...row,
      observation_key: `${row.relation}:${row.observation_row}`,
      effective_speed: row.effective_movement_speed,
      ultimate_cooldown: row.ultimate_cooldown_remaining,
      statuses: normalizePovStatuses(row.status_feature_values),
    });
  });
  const sanitizedScene = Object.freeze({
    schema_version: 1,
    audience_badge: sourceScene.audience_badge,
    observation_materialization: sourceScene.observation_materialization,
    episode_id: sourceScene.episode_id,
    frame_index: sourceScene.frame_index,
    pov_frame_id: sourceScene.pov_frame_id,
    source_frame_id: sourceScene.source_frame_id,
    simulator_step_count: sourceScene.simulator_step_count,
    map,
    self_actor: selfActor,
    visible_bodies: Object.freeze(visibleBodies),
    spawn_pads: Object.freeze(spawnPads),
    respawn_waves: Object.freeze(respawnWaves),
  });
  const scene = Object.freeze({
    ...sanitizedScene,
    audience: "agent_pov",
    frame_id: sanitizedScene.source_frame_id,
    incoming_transition_id: frame.incoming_pov_transition_id,
    agents: Object.freeze([selfAgent]),
    observed_bodies: Object.freeze(observedBodies),
    aura_fields: Object.freeze([]),
    ranges: Object.freeze([]),
    observer_visibility: Object.freeze([]),
    selection: Object.freeze({
      controlled_global_slot: selfActor.global_slot,
      selected_global_slot: null,
    }),
    selected_legality: null,
    pending_route: null,
  });

  const incomingTransitionId = projection.incoming_transition_id;
  const cues = requireArray(
    projection.incoming_cues,
    "POV incoming cues must be an array.",
  );
  if (incomingTransitionId === null) {
    if (
      frame.frame_index !== 0 ||
      frame.incoming_pov_transition_id !== null ||
      cues.length !== 0
    ) {
      throw new TypeError("Only POV frame zero may omit its incoming transition.");
    }
    return {
      projection: Object.freeze({
        scene: sanitizedScene,
        next_decision_action_mask: actionMask,
        incoming_transition_id: null,
        incoming_cues: Object.freeze([]),
      }),
      scene,
      eventBatch: null,
    };
  }
  const expectedPovTransitionId = `${frame.episode_id}:actor-pov:${selfActor.public_agent_id}:transition:${frame.frame_index - 1}`;
  if (incomingTransitionId !== expectedPovTransitionId) {
    throw new TypeError("POV incoming transition identity is not canonical.");
  }
  if (frame.incoming_pov_transition_id !== expectedPovTransitionId) {
    throw new TypeError("POV live envelope does not join its local transition.");
  }
  const sanitizedCues = cues.map((rawCue, ordinal) =>
    normalizePovCue(
      requireRecord(rawCue, "Invalid POV cue row."),
      ordinal,
      incomingTransitionId,
    ),
  );
  const normalizedCues = sanitizedCues.map((cue) => {
    return Object.freeze({
      ...cue,
      event_id: cue.cue_id,
      event_type: cue.cue_type,
      transition_id: cue.pov_transition_id,
    });
  });
  return {
    projection: Object.freeze({
      scene: sanitizedScene,
      next_decision_action_mask: actionMask,
      incoming_transition_id: incomingTransitionId,
      incoming_cues: Object.freeze(sanitizedCues),
    }),
    scene,
    eventBatch: Object.freeze({
      schema_version: 1,
      audience: "agent_pov",
      transition_id: incomingTransitionId,
      simulator_step: frame.simulator_step_count,
      events: Object.freeze(normalizedCues),
    }),
  };
}

/**
 * Normalize an audience-owned projection for a non-live presentation adapter.
 * This intentionally accepts only the identity/join fields consumed by the
 * strict projection normalizers; replay adapters never fabricate live
 * scenario, recording, or presentation-control authority.
 *
 * @param {unknown} value
 * @returns {{projection: Readonly<Record<string, any>>, scene: Readonly<Record<string, any>>, eventBatch: Readonly<Record<string, any>> | null}}
 */
export function normalizeDebuggerAudienceProjectionV2(value) {
  const adapter = requireRecord(
    value,
    "Debugger audience projection adapter must be an object.",
  );
  if (
    adapter.frame_kind !== "researcher_live_debugger" &&
    adapter.frame_kind !== "actor_pov_live_debugger"
  ) {
    throw new TypeError("Debugger audience projection adapter kind is invalid.");
  }
  requireExactKeys(
    adapter,
    adapter.frame_kind === "researcher_live_debugger"
      ? RESEARCHER_PROJECTION_ADAPTER_KEYS
      : ACTOR_POV_PROJECTION_ADAPTER_KEYS,
    "Debugger audience projection adapter has unknown or missing fields.",
  );
  const episodeId = requireNonemptyString(
    adapter.episode_id,
    "Debugger projection episode ID",
  );
  const frameIndex = requireInteger(
    adapter.frame_index,
    "Debugger projection frame index",
  );
  requireInteger(
    adapter.simulator_step_count,
    "Debugger projection simulator-step count",
  );
  if (adapter.frame_id !== `${episodeId}:frame:${frameIndex}`) {
    throw new TypeError("Debugger projection frame identity is not canonical.");
  }
  if (adapter.frame_kind === "researcher_live_debugger") {
    const expectedTransitionId =
      frameIndex === 0 ? null : `${episodeId}:transition:${frameIndex - 1}`;
    if (adapter.incoming_transition_id !== expectedTransitionId) {
      throw new TypeError(
        "Debugger researcher projection transition identity is not canonical.",
      );
    }
  }
  const projection = requireRecord(
    adapter.projection,
    "Debugger audience projection adapter is missing its projection.",
  );
  const normalized =
    adapter.frame_kind === "researcher_live_debugger"
      ? normalizeResearcherProjection(projection, adapter)
      : normalizePovProjection(projection, adapter);
  return Object.freeze({
    projection: normalized.projection ?? Object.freeze({ ...projection }),
    scene: normalized.scene,
    eventBatch: normalized.eventBatch,
  });
}

/**
 * Normalize the projection-free SharedObs live recipient transport. Battlefield
 * and decision authority arrive only through the separately joined authorized
 * presentation.
 *
 * @param {Record<string, any>} frame
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeSharedObsAgentPovLiveFrameV2(frame) {
  requireExactKeys(
    frame,
    SHARED_OBS_AGENT_POV_LIVE_FRAME_KEYS_V2,
    "SharedObs live frame has unknown or missing top-level fields.",
  );
  if (
    frame.schema_version !== 2 ||
    frame.frame_kind !== "shared_obs_agent_pov_live_debugger" ||
    frame.view_mode !== "pov" ||
    !["presentation", "analysis"].includes(frame.preset) ||
    frame.verbose !== false
  ) {
    throw new TypeError("SharedObs live frame identity is invalid.");
  }
  const sessionId = requireNonemptyString(
    frame.session_id,
    "SharedObs live session ID",
  );
  const runGeneration = requireInteger(
    frame.run_generation,
    "SharedObs live run generation",
  );
  const revision = requireInteger(frame.revision, "SharedObs live revision");
  const episodeId = requireNonemptyString(
    frame.episode_id,
    "SharedObs live episode ID",
  );
  const frameIndex = requireInteger(frame.frame_index, "SharedObs live frame index");
  const simulatorStepCount = requireInteger(
    frame.simulator_step_count,
    "SharedObs live simulator-step count",
  );
  const frameId = requireNonemptyString(frame.frame_id, "SharedObs live frame ID");
  const recipient = requireNonemptyString(
    frame.recipient_public_agent_id,
    "SharedObs live recipient",
  );
  const recipientFrameId = requireNonemptyString(
    frame.recipient_frame_id,
    "SharedObs live recipient frame ID",
  );
  if (frameId !== `${episodeId}:frame:${frameIndex}`) {
    throw new TypeError("SharedObs live global frame ID is not canonical.");
  }
  const prefix = `${episodeId}:shared-obs-visual-union:${recipient}`;
  if (recipientFrameId !== `${prefix}:frame:${frameIndex}`) {
    throw new TypeError("SharedObs live recipient frame ID is not canonical.");
  }
  const expectedTransitionId =
    frameIndex === 0 ? null : `${prefix}:transition:${frameIndex - 1}`;
  if (frame.incoming_recipient_transition_id !== expectedTransitionId) {
    throw new TypeError("SharedObs live transition identity is not canonical.");
  }
  if (!["joint_turn", "scripted_playback"].includes(frame.pending_submission_scope)) {
    throw new TypeError("SharedObs live submission scope is invalid.");
  }
  const terminal = normalizeTerminalStateV2(frame.terminal);
  const recording = normalizeRecordingStatusV1(frame.recording, frameIndex);
  const combatConfiguration = normalizeCombatConfigurationV1(
    frame.combat_configuration,
  );
  if (combatConfiguration.execution_information_mode !== "shared_obs") {
    throw new TypeError("SharedObs live frame requires SharedObs configuration.");
  }
  return Object.freeze({
    session_id: sessionId,
    run_generation: runGeneration,
    revision,
    episode_id: episodeId,
    frame_index: frameIndex,
    frame_id: frameId,
    simulator_step_count: simulatorStepCount,
    preset: "analysis",
    verbose: false,
    terminal,
    recording,
    combat_configuration: combatConfiguration,
    frame_kind: "shared_obs_agent_pov_live_debugger",
    schema_version: 2,
    view_mode: "pov",
    recipient_public_agent_id: recipient,
    recipient_frame_id: recipientFrameId,
    incoming_recipient_transition_id: expectedTransitionId,
    pending_submission_scope: frame.pending_submission_scope,
    simulator_step: simulatorStepCount,
    transition_id: expectedTransitionId,
  });
}

/**
 * Normalize one validated LiveDebuggerFrameV2 at the API boundary.
 *
 * All downstream browser components consume only `scene` and `event_batch`.
 * The rebuilt audience-specific `projection` remains available for the
 * Technical panel, so no downstream component needs to guess projection
 * aliases, consume raw wire roots, or filter a privileged researcher frame
 * into a POV frame.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
export function normalizeLiveDebuggerFrameV2(value) {
  const frame = requireRecord(value, "Live debugger frame must be an object.");
  if (frame.frame_kind === "shared_obs_agent_pov_live_debugger") {
    return normalizeSharedObsAgentPovLiveFrameV2(frame);
  }
  if (frame.schema_version !== 2) {
    throw new TypeError("Live debugger frame must use schema version 2.");
  }
  if (
    (frame.frame_kind !== "researcher_live_debugger" ||
      frame.view_mode !== "researcher") &&
    (frame.frame_kind !== "actor_pov_live_debugger" || frame.view_mode !== "pov")
  ) {
    throw new TypeError("Live debugger frame has an unknown audience mode.");
  }
  if (!Object.hasOwn(frame, "recording")) {
    throw new TypeError("Live debugger frame is missing recording authority.");
  }
  requireExactKeys(
    frame,
    frame.frame_kind === "researcher_live_debugger"
      ? RESEARCHER_LIVE_FRAME_KEYS_V2
      : ACTOR_POV_LIVE_FRAME_KEYS_V2,
    "Live debugger frame has unknown or missing top-level fields.",
  );
  if (
    requireNonemptyString(frame.session_id, "Live debugger session ID") !==
      frame.session_id ||
    requireInteger(frame.run_generation, "Live debugger run generation") !==
      frame.run_generation ||
    requireInteger(frame.revision, "Live debugger revision") !== frame.revision ||
    requireNonemptyString(frame.episode_id, "Live debugger episode ID") !==
      frame.episode_id ||
    requireInteger(frame.frame_index, "Live debugger frame index") !==
      frame.frame_index ||
    requireInteger(frame.simulator_step_count, "Live debugger simulator-step count") !==
      frame.simulator_step_count ||
    !["presentation", "analysis"].includes(frame.preset) ||
    frame.verbose !== false ||
    (frame.frame_kind === "researcher_live_debugger" &&
      typeof frame.show_ranges !== "boolean")
  ) {
    throw new TypeError("Live debugger envelope authority is invalid.");
  }
  const terminal = normalizeTerminalStateV2(frame.terminal);
  if (
    !Number.isInteger(frame.frame_index) ||
    typeof frame.episode_id !== "string" ||
    frame.frame_id !== `${frame.episode_id}:frame:${frame.frame_index}`
  ) {
    throw new TypeError("Live debugger frame identity is not canonical.");
  }
  const recording = normalizeRecordingStatusV1(frame.recording, frame.frame_index);
  const combatConfiguration = normalizeCombatConfigurationV1(
    frame.combat_configuration,
  );
  if (
    frame.frame_kind === "actor_pov_live_debugger" &&
    combatConfiguration.execution_information_mode !== "no_shared_obs"
  ) {
    throw new TypeError("NoSharedObs live frame requires NoSharedObs configuration.");
  }
  if (frame.frame_kind === "researcher_live_debugger") {
    const expectedTransitionIndex =
      frame.frame_index === 0 ? null : frame.frame_index - 1;
    const expectedTransitionId =
      expectedTransitionIndex === null
        ? null
        : `${frame.episode_id}:transition:${expectedTransitionIndex}`;
    if (
      frame.incoming_transition_index !== expectedTransitionIndex ||
      frame.incoming_transition_id !== expectedTransitionId
    ) {
      throw new TypeError(
        "Live debugger incoming transition identity is not canonical.",
      );
    }
  }
  const scenario =
    frame.frame_kind === "researcher_live_debugger"
      ? normalizeScenarioMetadataV1(frame.scenario)
      : null;
  const availableScenarios =
    frame.frame_kind === "researcher_live_debugger"
      ? Object.freeze(
          requireArray(
            frame.available_scenarios,
            "Live researcher scenario menu must be an array.",
          ).map((option, index) =>
            normalizeScenarioOptionV1(
              option,
              `Live researcher scenario menu row ${index}`,
            ),
          ),
        )
      : null;
  const projection = requireRecord(
    frame.projection,
    "Live debugger frame is missing its audience projection.",
  );
  /** @type {Record<string, any>} */
  let normalized =
    frame.frame_kind === "researcher_live_debugger"
      ? normalizeResearcherProjection(projection, frame)
      : normalizePovProjection(projection, frame);
  let hud = frame.hud;
  if (frame.frame_kind === "researcher_live_debugger") {
    const authorizedScene = normalized.projection.scene;
    hud = normalizeResearcherHud(
      requireRecord(frame.hud, "Researcher HUD must be an object."),
      authorizedScene,
      frame,
    );
    const scene = Object.freeze({
      ...normalized.scene,
      pending_route: researcherPendingRoute(normalized.scene, { ...frame, hud }),
    });
    normalized = { ...normalized, scene };
  } else {
    const authorizedProjection = normalized.projection;
    const authorizedScene = authorizedProjection.scene;
    hud = normalizePovHud(
      requireRecord(frame.hud, "POV HUD must be an object."),
      authorizedScene.self_actor,
      authorizedProjection.next_decision_action_mask,
      authorizedScene.visible_bodies,
      frame,
    );
    const scene = Object.freeze({
      ...normalized.scene,
      pending_route: povPendingRoute(normalized.scene, { ...frame, hud }),
    });
    normalized = { ...normalized, scene };
    const endedCues = authorizedProjection.incoming_cues.filter(
      (/** @type {Record<string, any>} */ cue) => cue.cue_type === "episode_ended",
    );
    if (
      endedCues.length > 1 ||
      (endedCues.length === 1 &&
        (endedCues[0] !== authorizedProjection.incoming_cues.at(-1) ||
          endedCues[0].terminated !== terminal.terminated ||
          endedCues[0].truncated !== terminal.truncated)) ||
      ((terminal.terminated || terminal.truncated) && endedCues.length !== 1)
    ) {
      throw new TypeError("POV episode-ended cue does not join terminal authority.");
    }
  }
  const presentationTransitionId =
    frame.frame_kind === "researcher_live_debugger"
      ? frame.incoming_transition_id
      : frame.incoming_pov_transition_id;
  const normalizedFrame = {
    session_id: frame.session_id,
    run_generation: frame.run_generation,
    revision: frame.revision,
    episode_id: frame.episode_id,
    frame_index: frame.frame_index,
    frame_id: frame.frame_id,
    simulator_step_count: frame.simulator_step_count,
    preset: "analysis",
    verbose: frame.verbose,
    terminal,
    recording,
    combat_configuration: combatConfiguration,
    frame_kind: frame.frame_kind,
    schema_version: 2,
    view_mode: frame.view_mode,
    ...(frame.frame_kind === "researcher_live_debugger"
      ? {
          incoming_transition_index: frame.incoming_transition_index,
          incoming_transition_id: frame.incoming_transition_id,
          show_ranges: frame.show_ranges,
          scenario,
          available_scenarios: availableScenarios,
        }
      : { incoming_pov_transition_id: frame.incoming_pov_transition_id }),
    projection: normalized.projection ?? projection,
    hud,
    simulator_step: frame.simulator_step_count,
    transition_id: presentationTransitionId,
    scene: normalized.scene,
    event_batch: normalized.eventBatch,
  };
  return Object.freeze(normalizedFrame);
}

export const researcherEventTypesV2 = Object.freeze([...RESEARCHER_EVENT_TYPES_V2]);

/**
 * Resolve live scripted authority from the audience-owned envelope. POV
 * deliberately omits researcher scenario metadata and exposes this fact only
 * through its recipient-safe HUD submission scope.
 *
 * @param {unknown} value
 */
export function liveDebuggerFrameIsScripted(value) {
  if (!isRecord(value)) {
    return false;
  }
  if (value.frame_kind === "actor_pov_live_debugger") {
    return (
      isRecord(value.hud) && value.hud.pending_submission_scope === "scripted_playback"
    );
  }
  if (value.frame_kind === "shared_obs_agent_pov_live_debugger") {
    return value.pending_submission_scope === "scripted_playback";
  }
  return isRecord(value.scenario) && value.scenario.mode === "scripted";
}

/**
 * Researcher scenario controls are intentionally absent from recipient POV.
 * This answers only transport authority; it never infers scenario metadata.
 *
 * @param {unknown} value
 */
export function liveDebuggerScenarioControlsAvailable(value) {
  return isRecord(value) && value.frame_kind === "researcher_live_debugger";
}
