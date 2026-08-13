import { normalizeDebuggerAudienceProjectionV2 } from "./frame-normalizer.js";

const REPLAY_FRAME_KINDS = new Set([
  "researcher_replay_viewer",
  "actor_pov_replay_viewer",
  "shared_obs_source_material_replay_viewer",
]);
const TIMELINE_KINDS = new Set([
  "researcher",
  "actor_pov",
  "shared_obs_source_material",
]);
const PRESETS = new Set(["presentation", "analysis", "debug"]);
const COMPLETION_STATES = new Set(["complete", "partial", "interrupted", "failed"]);
const FAILURE_ORIGINS = new Set(["simulation", "policy", "validation", "capture"]);
const PROCESSING_STATES = new Set(["succeeded", "failed"]);
const PROCESSING_FAILURE_STAGES = new Set([
  "initial_validation",
  "reducer_initialize",
  "transition_validation",
  "reducer_advance",
  "completion_validation",
  "reducer_finalize",
  "statistic_materialization",
  "report_validation",
  "lifecycle",
]);
const REPLAY_ERROR_CODES = new Set([
  "invalid_request",
  "invalid_cursor",
  "audience_unavailable",
  "unauthorized",
  "forbidden_origin",
  "not_found",
  "method_not_allowed",
  "payload_too_large",
  "unsupported_media_type",
  "stale_revision",
  "command_id_conflict",
  "server_shutting_down",
  "internal_error",
]);
const ENDPOINT_KINDS = new Set([
  "none",
  "task_terminal",
  "declared_horizon",
  "task_terminal_and_declared_horizon",
  "captured_prefix",
]);
const COMMON_FRAME_KEYS = [
  "schema_version",
  "frame_kind",
  "viewer_session_id",
  "revision",
  "artifact_summary",
  "timeline_id",
  "cursor",
  "preset",
  "verbose",
];
const ARTIFACT_SUMMARY_KEYS = [
  "schema_version",
  "replay_reference",
  "expected_transition_count",
  "recorded_transition_count",
  "recorded_frame_count",
  "metric_report_availability",
];
const ARTIFACT_REFERENCE_KEYS = [
  "schema_id",
  "schema_version",
  "artifact_id",
  "episode_id",
  "replay_schema_version",
  "context_digest_sha256",
  "trajectory_content_digest_sha256",
  "canonical_digest_sha256",
  "canonical_byte_length",
];
const REPLAY_ARTIFACT_REFERENCE_SCHEMA_ID =
  "marl_battlegrounds.evaluation.replay_artifact_reference";
const RESEARCHER_COMPLETION_KEYS = [
  "schema_version",
  "episode_id",
  "completion_state",
  "expected_transition_count",
  "validated_transition_count",
  "last_valid_frame_index",
  "last_valid_frame_id",
  "terminated",
  "truncated",
  "completion_bases",
  "end_or_failure_reason",
  "failure_origin",
];
const POV_COMPLETION_KEYS = [
  "schema_version",
  "episode_id",
  "completion_state",
  "expected_transition_count",
  "captured_transition_count",
  "terminated",
  "truncated",
  "completion_bases",
  "public_end_or_failure_reason",
];
const PROCESSING_KEYS = [
  "schema_version",
  "status",
  "processed_transition_count",
  "failure_stage",
  "failure_code",
  "attempted_transition_index",
];
const POV_PROCESSING_KEYS = ["schema_version", "disclosure"];
const SOURCE_MATERIAL_DISCLOSURE =
  "SOURCE MATERIAL ONLY · NOT MATERIALIZED SHAREDOBS ACTOR INPUT";
const SOURCE_PROJECTION_KEYS = [
  "schema_version",
  "disclosure_label",
  "observation_materialization",
  "exact_actor_input_export_available",
  "axis_mapping",
  "ally_observation_row_global_slot_by_id",
  "enemy_observation_row_global_slot_by_id",
  "base_sensor_frame",
  "base_sensor_scene",
  "incoming_transition_id",
  "sensor_source_availability",
];
const SOURCE_AXIS_KEYS = [
  "schema_id",
  "schema_version",
  "actor_projection_identifier",
  "actor_projection_version",
  "source_context_schema_id",
  "source_context_schema_version",
  "source_frame_schema_id",
  "source_frame_schema_version",
  "source_transition_schema_id",
  "source_transition_schema_version",
  "target_action_recipient_public_agent_id_by_id",
  "ally_observation_row_public_agent_id_by_id",
  "enemy_observation_row_public_agent_id_by_id",
  "movement_action_name_by_id",
  "unit_direction_vector_by_movement_action",
  "target_action_name_by_id",
  "use_ultimate_action_name_by_id",
  "spawn_lifecycle_team_axis_name_by_id",
];
const SOURCE_BASE_FRAME_KEYS = [
  "schema_version",
  "observation_materialization",
  "episode_id",
  "public_agent_id",
  "frame_index",
  "source_material_frame_id",
  "source_frame_id",
  "simulator_step_count",
  "self_features",
  "ally_unit_features",
  "enemy_unit_features",
  "map_obstacle_features",
  "objective_features",
  "context_features",
  "ally_visibility_mask",
  "enemy_visibility_mask",
  "previous_timestep_actions",
  "spawn_lifecycle",
  "action_mask",
];
const SOURCE_BASE_SCENE_KEYS = [
  "schema_version",
  "audience_badge",
  "observation_materialization",
  "episode_id",
  "frame_index",
  "source_frame_id",
  "simulator_step_count",
  "map",
  "self_actor",
  "visible_bodies",
  "spawn_pads",
  "respawn_waves",
];
const SOURCE_SELF_KEYS = [
  "global_slot",
  "public_agent_id",
  "team_local_slot",
  "team_id",
  "class_id",
  "position",
  "radius",
  "alive",
  "current_health",
  "max_health",
  "effective_movement_speed",
  "ultimate_cooldown_remaining",
  "steps_until_out_of_combat",
  "spawn_shield_remaining",
  "status_feature_values",
];
const SOURCE_BODY_KEYS = [
  "relation",
  "observation_row",
  "public_agent_id",
  "position",
  "radius",
  "team_id",
  "class_id",
  "alive",
  "current_health",
  "max_health",
  "effective_movement_speed",
  "ultimate_cooldown_remaining",
  "steps_until_out_of_combat",
  "status_feature_values",
];
const SOURCE_SPAWN_PAD_KEYS = [
  "actor_relative_team_index",
  "team_relation",
  "team_label",
  "team_local_slot",
  "position",
  "configured_active",
  "currently_alive",
  "spawn_shield_remaining",
];
const SOURCE_RESPAWN_WAVE_KEYS = [
  "actor_relative_team_index",
  "team_relation",
  "team_label",
  "period_steps",
  "countdown_steps",
];
const SOURCE_AVAILABILITY_KEYS = [
  "sensor_source_global_slot",
  "sensor_source_public_agent_id",
  "sensor_source_team_local_slot",
  "sensor_source_configured_team_id",
  "sensor_source_configured_active",
  "relation_to_recipient",
  "base_sensor_relation_axis",
  "base_sensor_observation_row",
  "recorded_available",
];
/** @type {ReadonlyArray<readonly [number, number, string, number]>} */
const STATUS_FEATURES = Object.freeze([
  [6, 21, "stun_warrior_charge", 2],
  [7, 22, "stun_hunter_trap", 3],
  [8, 23, "stun_rogue_poison", 4],
  [0, 15, "slow_warrior_charge", 2],
  [1, 16, "slow_hunter_basic", 3],
  [2, 17, "slow_rogue_poison", 4],
  [9, 24, "anti_heal_rogue_poison", 4],
  [12, 27, "priest_freedom", 5],
  [11, 26, "mage_burst", 1],
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
 * @param {string} label
 * @returns {Record<string, any>}
 */
function record(value, label) {
  if (!isRecord(value)) {
    throw new TypeError(`${label} must be an object.`);
  }
  return value;
}

/**
 * @param {Record<string, any>} value
 * @param {readonly string[]} expected
 * @param {string} label
 */
function exactKeys(value, expected, label) {
  const actual = Object.keys(value).sort();
  const canonical = [...expected].sort();
  if (
    actual.length !== canonical.length ||
    actual.some((key, index) => key !== canonical[index])
  ) {
    throw new TypeError(`${label} has unknown or missing fields.`);
  }
}

/** @param {unknown} left @param {unknown} right @returns {boolean} */
function structurallyEqual(left, right) {
  if (Object.is(left, right)) {
    return true;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((item, index) => structurallyEqual(item, right[index]))
    );
  }
  if (!isRecord(left) || !isRecord(right)) {
    return false;
  }
  const leftKeys = Object.keys(left).sort();
  const rightKeys = Object.keys(right).sort();
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every(
      (key, index) =>
        key === rightKeys[index] && structurallyEqual(left[key], right[key]),
    )
  );
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function nonNegativeInteger(value, label) {
  if (!Number.isInteger(value) || Number(value) < 0) {
    throw new TypeError(`${label} must be a non-negative integer.`);
  }
  return Number(value);
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function positiveInteger(value, label) {
  const normalized = nonNegativeInteger(value, label);
  if (normalized === 0) {
    throw new TypeError(`${label} must be a positive integer.`);
  }
  return normalized;
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function globalSlot(value, label) {
  const normalized = nonNegativeInteger(value, label);
  if (normalized >= 10) {
    throw new TypeError(`${label} must be on the ten-slot V1 axis.`);
  }
  return normalized;
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function nonEmptyString(value, label) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${label} must be a non-empty string.`);
  }
  return value;
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function nullableString(value, label) {
  if (value !== null && typeof value !== "string") {
    throw new TypeError(`${label} must be a string or null.`);
  }
  return value;
}

/**
 * @param {Record<string, any>} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeArtifactReference(value) {
  exactKeys(value, ARTIFACT_REFERENCE_KEYS, "Replay artifact reference");
  if (
    value.schema_id !== REPLAY_ARTIFACT_REFERENCE_SCHEMA_ID ||
    value.schema_version !== 1 ||
    value.replay_schema_version !== 1
  ) {
    throw new TypeError("Replay artifact reference root is invalid.");
  }
  const artifactId = nonEmptyString(
    value.artifact_id,
    "artifact_summary.replay_reference.artifact_id",
  );
  const episodeId = nonEmptyString(
    value.episode_id,
    "artifact_summary.replay_reference.episode_id",
  );
  if (artifactId !== `${episodeId}:replay`) {
    throw new TypeError("Replay artifact identity is not canonical.");
  }
  for (const key of [
    "context_digest_sha256",
    "trajectory_content_digest_sha256",
    "canonical_digest_sha256",
  ]) {
    if (!/^[0-9a-f]{64}$/u.test(String(value[key]))) {
      throw new TypeError(`Replay artifact reference ${key} is invalid.`);
    }
  }
  positiveInteger(
    value.canonical_byte_length,
    "artifact_summary.replay_reference.canonical_byte_length",
  );
  return Object.freeze({ ...value });
}

/**
 * @param {Record<string, any>} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeArtifactSummary(value) {
  exactKeys(value, ARTIFACT_SUMMARY_KEYS, "Replay artifact summary");
  if (value.schema_version !== 1) {
    throw new TypeError("Replay artifact summary must use schema version 1.");
  }
  const replayReference = normalizeArtifactReference(
    record(value.replay_reference, "Replay artifact reference"),
  );
  const expected = positiveInteger(
    value.expected_transition_count,
    "artifact_summary.expected_transition_count",
  );
  const recorded = nonNegativeInteger(
    value.recorded_transition_count,
    "artifact_summary.recorded_transition_count",
  );
  const frameCount = positiveInteger(
    value.recorded_frame_count,
    "artifact_summary.recorded_frame_count",
  );
  if (recorded > expected || frameCount !== recorded + 1) {
    throw new TypeError("Replay artifact summary counts are incoherent.");
  }
  if (
    value.metric_report_availability !== "available" &&
    value.metric_report_availability !== "missing" &&
    value.metric_report_availability !== "not_available_in_actor_pov"
  ) {
    throw new TypeError("Replay metric-report availability is invalid.");
  }
  return Object.freeze({ ...value, replay_reference: replayReference });
}

/**
 * @param {Record<string, any>} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeCursor(value) {
  exactKeys(
    value,
    [
      "frame_index",
      "final_frame_index",
      "cursor_generation",
      "choreography_generation",
      "schema_version",
    ],
    "Replay cursor",
  );
  const normalized = Object.freeze({
    schema_version: value.schema_version,
    frame_index: nonNegativeInteger(value.frame_index, "cursor.frame_index"),
    final_frame_index: nonNegativeInteger(
      value.final_frame_index,
      "cursor.final_frame_index",
    ),
    cursor_generation: nonNegativeInteger(
      value.cursor_generation,
      "cursor.cursor_generation",
    ),
    choreography_generation: nonNegativeInteger(
      value.choreography_generation,
      "cursor.choreography_generation",
    ),
  });
  if (normalized.schema_version !== 1) {
    throw new TypeError("Replay cursor must use schema version 1.");
  }
  if (
    normalized.frame_index > normalized.final_frame_index ||
    normalized.choreography_generation > normalized.cursor_generation
  ) {
    throw new TypeError("Replay cursor exceeds the final frame.");
  }
  return normalized;
}

/**
 * @param {unknown} value
 * @param {string} label
 */
function completionBases(value, label) {
  if (
    !Array.isArray(value) ||
    value.some((item) => item !== "task_terminal" && item !== "declared_horizon")
  ) {
    throw new TypeError(`${label} must contain only canonical completion bases.`);
  }
  return Object.freeze([...value]);
}

/** @param {Record<string, any>} completion */
function expectedEndpointKind(completion) {
  if (completion.completion_state !== "complete") {
    return "captured_prefix";
  }
  const bases = completion.completion_bases;
  if (
    bases.length === 2 &&
    bases[0] === "task_terminal" &&
    bases[1] === "declared_horizon"
  ) {
    return "task_terminal_and_declared_horizon";
  }
  if (bases.length === 1 && bases[0] === "task_terminal") {
    return "task_terminal";
  }
  if (bases.length === 1 && bases[0] === "declared_horizon") {
    return "declared_horizon";
  }
  throw new TypeError("Complete replay has invalid completion bases.");
}

/**
 * @param {Record<string, any>} value
 * @param {number} captured
 * @param {string | null} reason
 * @param {string | null | undefined} failureOrigin
 */
function validateCompletionSemantics(value, captured, reason, failureOrigin) {
  if (captured === 0 && (value.terminated || value.truncated)) {
    throw new TypeError("A zero-transition replay cannot carry terminal flags.");
  }
  /** @type {string[]} */
  const expectedBases = [];
  if (value.terminated) {
    expectedBases.push("task_terminal");
  }
  if (captured === value.expected_transition_count) {
    expectedBases.push("declared_horizon");
  }
  const bases = /** @type {string[]} */ (value.completion_bases);
  if (value.completion_state === "complete") {
    if (
      expectedBases.length === 0 ||
      bases.length !== expectedBases.length ||
      bases.some((basis, index) => basis !== expectedBases[index]) ||
      (failureOrigin !== undefined && failureOrigin !== null)
    ) {
      throw new TypeError("Complete replay completion evidence is incoherent.");
    }
    return;
  }
  if (
    expectedBases.length > 0 ||
    bases.length > 0 ||
    typeof reason !== "string" ||
    reason.trim() === ""
  ) {
    throw new TypeError("Incomplete replay completion evidence is incoherent.");
  }
  if (failureOrigin !== undefined) {
    if (
      value.completion_state === "failed"
        ? typeof failureOrigin !== "string" || !FAILURE_ORIGINS.has(failureOrigin)
        : failureOrigin !== null
    ) {
      throw new TypeError("Replay completion failure origin is incoherent.");
    }
  }
}

/**
 * @param {Record<string, any>} value
 * @param {ReturnType<typeof normalizeArtifactSummary>} summary
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeResearcherCompletion(value, summary) {
  exactKeys(value, RESEARCHER_COMPLETION_KEYS, "Researcher completion");
  if (value.schema_version !== 1 || !COMPLETION_STATES.has(value.completion_state)) {
    throw new TypeError("Researcher completion root or state is invalid.");
  }
  nonEmptyString(value.episode_id, "completion.episode_id");
  const expected = nonNegativeInteger(
    value.expected_transition_count,
    "completion.expected_transition_count",
  );
  const validated = nonNegativeInteger(
    value.validated_transition_count,
    "completion.validated_transition_count",
  );
  const finalIndex = nonNegativeInteger(
    value.last_valid_frame_index,
    "completion.last_valid_frame_index",
  );
  nonEmptyString(value.last_valid_frame_id, "completion.last_valid_frame_id");
  if (
    expected !== summary.expected_transition_count ||
    validated !== summary.recorded_transition_count ||
    finalIndex !== summary.recorded_transition_count
  ) {
    throw new TypeError("Researcher completion counts do not join the replay.");
  }
  if (typeof value.terminated !== "boolean" || typeof value.truncated !== "boolean") {
    throw new TypeError("Researcher completion terminal flags must be booleans.");
  }
  nullableString(value.end_or_failure_reason, "completion.end_or_failure_reason");
  nullableString(value.failure_origin, "completion.failure_origin");
  if (value.last_valid_frame_id !== `${value.episode_id}:frame:${validated}`) {
    throw new TypeError("Researcher completion frame identity is not canonical.");
  }
  validateCompletionSemantics(
    value,
    validated,
    value.end_or_failure_reason,
    value.failure_origin,
  );
  return Object.freeze({
    ...value,
    completion_bases: completionBases(
      value.completion_bases,
      "completion.completion_bases",
    ),
  });
}

/**
 * @param {Record<string, any>} value
 * @param {ReturnType<typeof normalizeArtifactSummary>} summary
 * @returns {Readonly<Record<string, any>>}
 */
function normalizePovCompletion(value, summary) {
  exactKeys(value, POV_COMPLETION_KEYS, "Actor POV completion");
  if (value.schema_version !== 1 || !COMPLETION_STATES.has(value.completion_state)) {
    throw new TypeError("Actor POV completion root or state is invalid.");
  }
  nonEmptyString(value.episode_id, "completion.episode_id");
  const expected = nonNegativeInteger(
    value.expected_transition_count,
    "completion.expected_transition_count",
  );
  const captured = nonNegativeInteger(
    value.captured_transition_count,
    "completion.captured_transition_count",
  );
  if (
    expected !== summary.expected_transition_count ||
    captured !== summary.recorded_transition_count
  ) {
    throw new TypeError("Actor POV completion counts do not join the replay.");
  }
  if (typeof value.terminated !== "boolean" || typeof value.truncated !== "boolean") {
    throw new TypeError("Actor POV completion terminal flags must be booleans.");
  }
  nullableString(
    value.public_end_or_failure_reason,
    "completion.public_end_or_failure_reason",
  );
  validateCompletionSemantics(
    value,
    captured,
    value.public_end_or_failure_reason,
    undefined,
  );
  return Object.freeze({
    ...value,
    completion_bases: completionBases(
      value.completion_bases,
      "completion.completion_bases",
    ),
  });
}

/** @param {Record<string, any>} value */
function normalizeProcessing(value) {
  exactKeys(value, PROCESSING_KEYS, "Replay processing");
  if (value.schema_version !== 1 || !PROCESSING_STATES.has(value.status)) {
    throw new TypeError("Replay processing must use schema version 1.");
  }
  nonNegativeInteger(
    value.processed_transition_count,
    "processing.processed_transition_count",
  );
  nullableString(value.failure_stage, "processing.failure_stage");
  nullableString(value.failure_code, "processing.failure_code");
  if (
    value.attempted_transition_index !== null &&
    (!Number.isInteger(value.attempted_transition_index) ||
      value.attempted_transition_index < 0)
  ) {
    throw new TypeError("processing.attempted_transition_index is invalid.");
  }
  if (value.status === "succeeded") {
    if (
      value.failure_stage !== null ||
      value.failure_code !== null ||
      value.attempted_transition_index !== null
    ) {
      throw new TypeError("Successful replay processing forbids failure fields.");
    }
  } else if (
    !PROCESSING_FAILURE_STAGES.has(value.failure_stage) ||
    typeof value.failure_code !== "string" ||
    value.failure_code.length === 0
  ) {
    throw new TypeError("Failed replay processing requires stable failure fields.");
  }
  const permitsAttemptedIndex =
    value.failure_stage === "transition_validation" ||
    value.failure_stage === "reducer_advance";
  if (
    (!permitsAttemptedIndex && value.attempted_transition_index !== null) ||
    (value.failure_stage === "reducer_advance" &&
      value.attempted_transition_index === null)
  ) {
    throw new TypeError("Replay processing attempted index is incoherent.");
  }
  return Object.freeze({ ...value });
}

/** @param {Record<string, any>} value */
function normalizePovProcessing(value) {
  exactKeys(value, POV_PROCESSING_KEYS, "Actor POV processing disclosure");
  if (value.schema_version !== 1 || value.disclosure !== "not_available_in_actor_pov") {
    throw new TypeError("Actor POV processing disclosure is invalid.");
  }
  return Object.freeze({ ...value });
}

/** @param {unknown} value @param {string} label */
function finiteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`);
  }
  return value;
}

/** @param {unknown} value @param {string} label */
function nonNegativeFiniteNumber(value, label) {
  const normalized = finiteNumber(value, label);
  if (normalized < 0) {
    throw new TypeError(`${label} must be non-negative.`);
  }
  return normalized;
}

/** @param {unknown} value @param {string} label */
function booleanValue(value, label) {
  if (typeof value !== "boolean") {
    throw new TypeError(`${label} must be a boolean.`);
  }
  return value;
}

/**
 * @param {unknown} value
 * @param {readonly number[]} shape
 * @param {string} label
 * @param {(leaf: unknown, label: string) => unknown} leafNormalizer
 * @returns {ReadonlyArray<any>}
 */
function shapedArray(value, shape, label, leafNormalizer) {
  if (!Array.isArray(value) || value.length !== shape[0]) {
    throw new TypeError(`${label} must have exact shape ${shape.join("×")}.`);
  }
  if (shape.length === 1) {
    return Object.freeze(
      value.map((leaf, index) => leafNormalizer(leaf, `${label}[${index}]`)),
    );
  }
  return Object.freeze(
    value.map((item, index) =>
      shapedArray(item, shape.slice(1), `${label}[${index}]`, leafNormalizer),
    ),
  );
}

/** @param {unknown} value @param {string} label */
function finitePoint(value, label) {
  return shapedArray(value, [2], label, finiteNumber);
}

/** @param {unknown} value @param {string} label */
function nonEmptySourceString(value, label) {
  return nonEmptyString(value, label);
}

/** @param {unknown} value @param {string} label */
function nullableSourceString(value, label) {
  return value === null ? null : nonEmptyString(value, label);
}

/** @param {Record<string, any>} value */
function normalizeSourceAxis(value) {
  exactKeys(value, SOURCE_AXIS_KEYS, "Source-material axis mapping");
  if (
    value.schema_id !== "marl_battlegrounds.evaluation.actor_pov_axis_mapping" ||
    value.schema_version !== 1 ||
    value.source_context_schema_id !==
      "marl_battlegrounds.evaluation.episode_context" ||
    value.source_context_schema_version !== 1 ||
    value.source_frame_schema_id !== "marl_battlegrounds.evaluation.frame" ||
    value.source_frame_schema_version !== 1 ||
    value.source_transition_schema_id !== "marl_battlegrounds.evaluation.transition" ||
    value.source_transition_schema_version !== 1
  ) {
    throw new TypeError("Source-material axis mapping schema identity is invalid.");
  }
  nonEmptyString(
    value.actor_projection_identifier,
    "axis_mapping.actor_projection_identifier",
  );
  positiveInteger(
    value.actor_projection_version,
    "axis_mapping.actor_projection_version",
  );
  const targetIds = shapedArray(
    value.target_action_recipient_public_agent_id_by_id,
    [11],
    "axis_mapping.target_action_recipient_public_agent_id_by_id",
    nullableSourceString,
  );
  const allyIds = shapedArray(
    value.ally_observation_row_public_agent_id_by_id,
    [5],
    "axis_mapping.ally_observation_row_public_agent_id_by_id",
    nonEmptySourceString,
  );
  const enemyIds = shapedArray(
    value.enemy_observation_row_public_agent_id_by_id,
    [5],
    "axis_mapping.enemy_observation_row_public_agent_id_by_id",
    nonEmptySourceString,
  );
  const movementNames = shapedArray(
    value.movement_action_name_by_id,
    [9],
    "axis_mapping.movement_action_name_by_id",
    nonEmptySourceString,
  );
  const directions = shapedArray(
    value.unit_direction_vector_by_movement_action,
    [9, 2],
    "axis_mapping.unit_direction_vector_by_movement_action",
    finiteNumber,
  );
  const targetNames = shapedArray(
    value.target_action_name_by_id,
    [11],
    "axis_mapping.target_action_name_by_id",
    nonEmptySourceString,
  );
  const ultimateNames = shapedArray(
    value.use_ultimate_action_name_by_id,
    [2],
    "axis_mapping.use_ultimate_action_name_by_id",
    nonEmptySourceString,
  );
  const teamNames = shapedArray(
    value.spawn_lifecycle_team_axis_name_by_id,
    [2],
    "axis_mapping.spawn_lifecycle_team_axis_name_by_id",
    nonEmptySourceString,
  );
  if (teamNames[0] !== "Own Team" || teamNames[1] !== "Opponent Team") {
    throw new TypeError("Source-material lifecycle team names are not canonical.");
  }
  const relationIds = [...allyIds, ...enemyIds];
  if (
    !structurallyEqual(targetIds, [null, ...relationIds]) ||
    new Set(relationIds).size !== 10 ||
    new Set(movementNames).size !== 9 ||
    new Set(targetNames).size !== 11 ||
    new Set(ultimateNames).size !== 2
  ) {
    throw new TypeError("Source-material categorical axes are not canonical.");
  }
  return Object.freeze({
    ...value,
    target_action_recipient_public_agent_id_by_id: targetIds,
    ally_observation_row_public_agent_id_by_id: allyIds,
    enemy_observation_row_public_agent_id_by_id: enemyIds,
    movement_action_name_by_id: movementNames,
    unit_direction_vector_by_movement_action: directions,
    target_action_name_by_id: targetNames,
    use_ultimate_action_name_by_id: ultimateNames,
    spawn_lifecycle_team_axis_name_by_id: teamNames,
  });
}

/**
 * @param {Record<string, any>} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizePreviousActions(value) {
  const keys = [
    "schema_id",
    "schema_version",
    "ally_move_actions_one_hot",
    "enemy_move_actions_one_hot",
    "ally_select_target_actions_one_hot",
    "enemy_select_target_actions_one_hot",
    "ally_use_ultimate_actions_one_hot",
    "enemy_use_ultimate_actions_one_hot",
  ];
  exactKeys(value, keys, "Source-material previous actions");
  if (
    value.schema_id !== "marl_battlegrounds.evaluation.actor_pov_previous_actions" ||
    value.schema_version !== 1
  ) {
    throw new TypeError("Source-material previous actions schema is invalid.");
  }
  const normalized = { ...value };
  /** @type {ReadonlyArray<readonly [string, number]>} */
  const actionShapes = [
    ["ally_move_actions_one_hot", 9],
    ["enemy_move_actions_one_hot", 9],
    ["ally_select_target_actions_one_hot", 11],
    ["enemy_select_target_actions_one_hot", 11],
    ["ally_use_ultimate_actions_one_hot", 2],
    ["enemy_use_ultimate_actions_one_hot", 2],
  ];
  for (const [key, categories] of actionShapes) {
    normalized[key] = shapedArray(
      value[key],
      [5, categories],
      `previous_timestep_actions.${key}`,
      finiteNumber,
    );
  }
  return Object.freeze(normalized);
}

/**
 * @param {Record<string, any>} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeSpawnLifecycle(value) {
  exactKeys(
    value,
    [
      "schema_id",
      "schema_version",
      "spawn_pad_positions_by_team",
      "spawn_shield_actual_durations_by_team",
      "spawn_shield_configured_duration",
      "spawn_shield_speed",
      "respawn_wave_period_step_count_by_team",
      "respawn_wave_countdowns_by_team",
      "active_mask_by_team",
      "alive_mask_by_team",
    ],
    "Source-material spawn lifecycle",
  );
  if (
    value.schema_id !== "marl_battlegrounds.evaluation.actor_pov_spawn_lifecycle" ||
    value.schema_version !== 1
  ) {
    throw new TypeError("Source-material spawn lifecycle schema is invalid.");
  }
  return Object.freeze({
    ...value,
    spawn_pad_positions_by_team: shapedArray(
      value.spawn_pad_positions_by_team,
      [2, 5, 2],
      "spawn_lifecycle.spawn_pad_positions_by_team",
      finiteNumber,
    ),
    spawn_shield_actual_durations_by_team: shapedArray(
      value.spawn_shield_actual_durations_by_team,
      [2, 5],
      "spawn_lifecycle.spawn_shield_actual_durations_by_team",
      nonNegativeInteger,
    ),
    spawn_shield_configured_duration: nonNegativeInteger(
      value.spawn_shield_configured_duration,
      "spawn_lifecycle.spawn_shield_configured_duration",
    ),
    spawn_shield_speed: nonNegativeFiniteNumber(
      value.spawn_shield_speed,
      "spawn_lifecycle.spawn_shield_speed",
    ),
    respawn_wave_period_step_count_by_team: shapedArray(
      value.respawn_wave_period_step_count_by_team,
      [2],
      "spawn_lifecycle.respawn_wave_period_step_count_by_team",
      nonNegativeInteger,
    ),
    respawn_wave_countdowns_by_team: shapedArray(
      value.respawn_wave_countdowns_by_team,
      [2],
      "spawn_lifecycle.respawn_wave_countdowns_by_team",
      nonNegativeInteger,
    ),
    active_mask_by_team: shapedArray(
      value.active_mask_by_team,
      [2, 5],
      "spawn_lifecycle.active_mask_by_team",
      booleanValue,
    ),
    alive_mask_by_team: shapedArray(
      value.alive_mask_by_team,
      [2, 5],
      "spawn_lifecycle.alive_mask_by_team",
      booleanValue,
    ),
  });
}

/**
 * @param {Record<string, any>} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeSourceActionMask(value) {
  exactKeys(
    value,
    [
      "schema_id",
      "schema_version",
      "move",
      "select_target",
      "use_ultimate",
      "select_target_use_ultimate_joint",
    ],
    "Source-material action mask",
  );
  if (
    value.schema_id !== "marl_battlegrounds.evaluation.actor_pov_action_mask" ||
    value.schema_version !== 1
  ) {
    throw new TypeError("Source-material action mask schema is invalid.");
  }
  const move = shapedArray(value.move, [9], "action_mask.move", booleanValue);
  const selectTarget = shapedArray(
    value.select_target,
    [11],
    "action_mask.select_target",
    booleanValue,
  );
  const useUltimate = shapedArray(
    value.use_ultimate,
    [2],
    "action_mask.use_ultimate",
    booleanValue,
  );
  const joint = shapedArray(
    value.select_target_use_ultimate_joint,
    [11, 2],
    "action_mask.select_target_use_ultimate_joint",
    booleanValue,
  );
  for (let target = 0; target < 11; target += 1) {
    if (selectTarget[target] !== joint[target].some(Boolean)) {
      throw new TypeError("Source-material target mask disagrees with its joint mask.");
    }
  }
  for (let ultimate = 0; ultimate < 2; ultimate += 1) {
    const marginal = joint.some((row) => row[ultimate]);
    if (useUltimate[ultimate] !== marginal) {
      throw new TypeError(
        "Source-material ultimate mask disagrees with its joint mask.",
      );
    }
  }
  return Object.freeze({
    ...value,
    move,
    select_target: selectTarget,
    use_ultimate: useUltimate,
    select_target_use_ultimate_joint: joint,
  });
}

/**
 * @param {Record<string, any>} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeSourceBaseFrame(value) {
  exactKeys(value, SOURCE_BASE_FRAME_KEYS, "Source-material base sensor frame");
  if (
    value.schema_version !== 1 ||
    value.observation_materialization !== "source_material_only"
  ) {
    throw new TypeError("Source-material base sensor frame root is invalid.");
  }
  nonEmptyString(value.episode_id, "base_sensor_frame.episode_id");
  nonEmptyString(value.public_agent_id, "base_sensor_frame.public_agent_id");
  nonNegativeInteger(value.frame_index, "base_sensor_frame.frame_index");
  nonEmptyString(
    value.source_material_frame_id,
    "base_sensor_frame.source_material_frame_id",
  );
  nonEmptyString(value.source_frame_id, "base_sensor_frame.source_frame_id");
  nonNegativeInteger(
    value.simulator_step_count,
    "base_sensor_frame.simulator_step_count",
  );
  const expectedMaterialId = `${value.episode_id}:shared-obs-source-material:${value.public_agent_id}:frame:${value.frame_index}`;
  if (
    value.source_material_frame_id !== expectedMaterialId ||
    value.source_frame_id !== `${value.episode_id}:frame:${value.frame_index}`
  ) {
    throw new TypeError("Source-material base sensor frame identity is invalid.");
  }
  return Object.freeze({
    ...value,
    self_features: shapedArray(
      value.self_features,
      [58],
      "base_sensor_frame.self_features",
      finiteNumber,
    ),
    ally_unit_features: shapedArray(
      value.ally_unit_features,
      [5, 58],
      "base_sensor_frame.ally_unit_features",
      finiteNumber,
    ),
    enemy_unit_features: shapedArray(
      value.enemy_unit_features,
      [5, 58],
      "base_sensor_frame.enemy_unit_features",
      finiteNumber,
    ),
    map_obstacle_features: shapedArray(
      value.map_obstacle_features,
      [16, 8],
      "base_sensor_frame.map_obstacle_features",
      finiteNumber,
    ),
    objective_features: shapedArray(
      value.objective_features,
      [8, 12],
      "base_sensor_frame.objective_features",
      finiteNumber,
    ),
    context_features: shapedArray(
      value.context_features,
      [19],
      "base_sensor_frame.context_features",
      finiteNumber,
    ),
    ally_visibility_mask: shapedArray(
      value.ally_visibility_mask,
      [5],
      "base_sensor_frame.ally_visibility_mask",
      booleanValue,
    ),
    enemy_visibility_mask: shapedArray(
      value.enemy_visibility_mask,
      [5],
      "base_sensor_frame.enemy_visibility_mask",
      booleanValue,
    ),
    previous_timestep_actions: normalizePreviousActions(
      record(value.previous_timestep_actions, "Source-material previous actions"),
    ),
    spawn_lifecycle: normalizeSpawnLifecycle(
      record(value.spawn_lifecycle, "Source-material spawn lifecycle"),
    ),
    action_mask: normalizeSourceActionMask(
      record(value.action_mask, "Source-material action mask"),
    ),
  });
}

/**
 * @param {Record<string, any>} value
 * @param {string} label
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeSourceMap(value, label) {
  exactKeys(value, ["width", "height", "obstacles"], label);
  const width = finiteNumber(value.width, `${label}.width`);
  const height = finiteNumber(value.height, `${label}.height`);
  if (width <= 0 || height <= 0 || !Array.isArray(value.obstacles)) {
    throw new TypeError(`${label} bounds or obstacles are invalid.`);
  }
  const obstacleIds = new Set();
  const obstacles = value.obstacles.map((rawObstacle, index) => {
    const obstacle = record(rawObstacle, `${label}.obstacles[${index}]`);
    exactKeys(
      obstacle,
      ["obstacle_id", "kind", "center", "radius", "width", "height", "theta"],
      `${label} obstacle`,
    );
    const obstacleId = nonEmptyString(
      obstacle.obstacle_id,
      `${label}.obstacles[${index}].obstacle_id`,
    );
    if (obstacleIds.has(obstacleId)) {
      throw new TypeError(`${label} obstacle IDs must be unique.`);
    }
    obstacleIds.add(obstacleId);
    const center = finitePoint(obstacle.center, `${label}.obstacles[${index}].center`);
    const theta = finiteNumber(obstacle.theta, `${label}.obstacles[${index}].theta`);
    if (obstacle.kind === "pillar") {
      if (
        finiteNumber(obstacle.radius, `${label}.obstacles[${index}].radius`) <= 0 ||
        obstacle.width !== null ||
        obstacle.height !== null
      ) {
        throw new TypeError(`${label} pillar shape is invalid.`);
      }
    } else if (obstacle.kind === "wall") {
      if (
        obstacle.radius !== null ||
        finiteNumber(obstacle.width, `${label}.obstacles[${index}].width`) <= 0 ||
        finiteNumber(obstacle.height, `${label}.obstacles[${index}].height`) <= 0
      ) {
        throw new TypeError(`${label} wall shape is invalid.`);
      }
    } else {
      throw new TypeError(`${label} obstacle kind is invalid.`);
    }
    return Object.freeze({ ...obstacle, center, theta });
  });
  return Object.freeze({ width, height, obstacles: Object.freeze(obstacles) });
}

/**
 * @param {Record<string, any>} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeSourceSelf(value) {
  exactKeys(value, SOURCE_SELF_KEYS, "Source-material self actor");
  globalSlot(value.global_slot, "base_sensor_scene.self_actor.global_slot");
  nonEmptyString(value.public_agent_id, "base_sensor_scene.self_actor.public_agent_id");
  const teamLocalSlot = nonNegativeInteger(
    value.team_local_slot,
    "base_sensor_scene.self_actor.team_local_slot",
  );
  if (teamLocalSlot >= 5 || (value.team_id !== 1 && value.team_id !== 2)) {
    throw new TypeError("Source-material self actor team identity is invalid.");
  }
  if (!Number.isInteger(value.class_id) || value.class_id < 1 || value.class_id > 5) {
    throw new TypeError("Source-material self actor class is invalid.");
  }
  const position = finitePoint(value.position, "base_sensor_scene.self_actor.position");
  const radius = nonNegativeFiniteNumber(
    value.radius,
    "base_sensor_scene.self_actor.radius",
  );
  const currentHealth = nonNegativeFiniteNumber(
    value.current_health,
    "base_sensor_scene.self_actor.current_health",
  );
  const maxHealth = nonNegativeFiniteNumber(
    value.max_health,
    "base_sensor_scene.self_actor.max_health",
  );
  if (radius <= 0 || maxHealth <= 0 || currentHealth > maxHealth) {
    throw new TypeError("Source-material self actor body values are invalid.");
  }
  booleanValue(value.alive, "base_sensor_scene.self_actor.alive");
  nonNegativeFiniteNumber(
    value.effective_movement_speed,
    "base_sensor_scene.self_actor.effective_movement_speed",
  );
  for (const key of [
    "ultimate_cooldown_remaining",
    "steps_until_out_of_combat",
    "spawn_shield_remaining",
  ]) {
    nonNegativeInteger(value[key], `base_sensor_scene.self_actor.${key}`);
  }
  const statusValues = shapedArray(
    value.status_feature_values,
    [14],
    "base_sensor_scene.self_actor.status_feature_values",
    nonNegativeFiniteNumber,
  );
  normalizeStatusFeatures(statusValues);
  return Object.freeze({ ...value, position, status_feature_values: statusValues });
}

/**
 * @param {Record<string, any>} value
 * @param {number} index
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeSourceBody(value, index) {
  const label = `base_sensor_scene.visible_bodies[${index}]`;
  exactKeys(value, SOURCE_BODY_KEYS, "Source-material visible body");
  if (value.relation !== "ally" && value.relation !== "enemy") {
    throw new TypeError(`${label}.relation is invalid.`);
  }
  const row = nonNegativeInteger(value.observation_row, `${label}.observation_row`);
  if (row >= 5) {
    throw new TypeError(`${label}.observation_row is invalid.`);
  }
  nonEmptyString(value.public_agent_id, `${label}.public_agent_id`);
  const position = finitePoint(value.position, `${label}.position`);
  const radius = nonNegativeFiniteNumber(value.radius, `${label}.radius`);
  const currentHealth = nonNegativeFiniteNumber(
    value.current_health,
    `${label}.current_health`,
  );
  const maxHealth = nonNegativeFiniteNumber(value.max_health, `${label}.max_health`);
  if (
    radius <= 0 ||
    maxHealth <= 0 ||
    currentHealth > maxHealth ||
    (value.team_id !== 1 && value.team_id !== 2) ||
    !Number.isInteger(value.class_id) ||
    value.class_id < 1 ||
    value.class_id > 5
  ) {
    throw new TypeError(`${label} body values are invalid.`);
  }
  booleanValue(value.alive, `${label}.alive`);
  nonNegativeFiniteNumber(
    value.effective_movement_speed,
    `${label}.effective_movement_speed`,
  );
  nonNegativeInteger(
    value.ultimate_cooldown_remaining,
    `${label}.ultimate_cooldown_remaining`,
  );
  nonNegativeInteger(
    value.steps_until_out_of_combat,
    `${label}.steps_until_out_of_combat`,
  );
  const statusValues = shapedArray(
    value.status_feature_values,
    [14],
    `${label}.status_feature_values`,
    nonNegativeFiniteNumber,
  );
  normalizeStatusFeatures(statusValues);
  return Object.freeze({ ...value, position, status_feature_values: statusValues });
}

/** @param {unknown} value @param {string} label */
function sourceWireBoolean(value, label) {
  const normalized = finiteNumber(value, label);
  if (normalized !== 0 && normalized !== 1) {
    throw new TypeError(`${label} must be the exact V1 wire boolean 0 or 1.`);
  }
  return normalized === 1;
}

/**
 * @param {unknown} value
 * @param {string} label
 * @param {number} [minimum]
 * @param {number} [maximum]
 */
function sourceWireInteger(value, label, minimum = 0, maximum = Infinity) {
  const normalized = finiteNumber(value, label);
  if (!Number.isInteger(normalized) || normalized < minimum || normalized > maximum) {
    throw new TypeError(`${label} is outside its V1 integer wire domain.`);
  }
  return normalized;
}

/**
 * @param {Record<string, any>} value
 * @param {number} index
 * @param {ReturnType<typeof normalizeSpawnLifecycle>} lifecycle
 * @param {ReadonlyArray<any>} teamNames
 */
function normalizeSourceSpawnPad(value, index, lifecycle, teamNames) {
  const label = `base_sensor_scene.spawn_pads[${index}]`;
  exactKeys(value, SOURCE_SPAWN_PAD_KEYS, "Source-material spawn pad");
  const teamIndex = Math.floor(index / 5);
  const teamLocalSlot = index % 5;
  const expectedRelation = teamIndex === 0 ? "own" : "opponent";
  const normalized = Object.freeze({
    actor_relative_team_index: nonNegativeInteger(
      value.actor_relative_team_index,
      `${label}.actor_relative_team_index`,
    ),
    team_relation: value.team_relation,
    team_label: nonEmptyString(value.team_label, `${label}.team_label`),
    team_local_slot: nonNegativeInteger(
      value.team_local_slot,
      `${label}.team_local_slot`,
    ),
    position: finitePoint(value.position, `${label}.position`),
    configured_active: booleanValue(
      value.configured_active,
      `${label}.configured_active`,
    ),
    currently_alive: booleanValue(value.currently_alive, `${label}.currently_alive`),
    spawn_shield_remaining: nonNegativeInteger(
      value.spawn_shield_remaining,
      `${label}.spawn_shield_remaining`,
    ),
  });
  const expected = {
    actor_relative_team_index: teamIndex,
    team_relation: expectedRelation,
    team_label: teamNames[teamIndex],
    team_local_slot: teamLocalSlot,
    position: lifecycle.spawn_pad_positions_by_team[teamIndex][teamLocalSlot],
    configured_active: lifecycle.active_mask_by_team[teamIndex][teamLocalSlot],
    currently_alive: lifecycle.alive_mask_by_team[teamIndex][teamLocalSlot],
    spawn_shield_remaining:
      lifecycle.spawn_shield_actual_durations_by_team[teamIndex][teamLocalSlot],
  };
  if (!structurallyEqual(normalized, expected)) {
    throw new TypeError(`${label} does not join the base-sensor lifecycle.`);
  }
  return normalized;
}

/**
 * @param {Record<string, any>} value
 * @param {number} index
 * @param {ReturnType<typeof normalizeSpawnLifecycle>} lifecycle
 * @param {ReadonlyArray<any>} teamNames
 */
function normalizeSourceRespawnWave(value, index, lifecycle, teamNames) {
  const label = `base_sensor_scene.respawn_waves[${index}]`;
  exactKeys(value, SOURCE_RESPAWN_WAVE_KEYS, "Source-material respawn wave");
  const normalized = Object.freeze({
    actor_relative_team_index: nonNegativeInteger(
      value.actor_relative_team_index,
      `${label}.actor_relative_team_index`,
    ),
    team_relation: value.team_relation,
    team_label: nonEmptyString(value.team_label, `${label}.team_label`),
    period_steps: positiveInteger(value.period_steps, `${label}.period_steps`),
    countdown_steps: nonNegativeInteger(
      value.countdown_steps,
      `${label}.countdown_steps`,
    ),
  });
  const expected = {
    actor_relative_team_index: index,
    team_relation: index === 0 ? "own" : "opponent",
    team_label: teamNames[index],
    period_steps: lifecycle.respawn_wave_period_step_count_by_team[index],
    countdown_steps: lifecycle.respawn_wave_countdowns_by_team[index],
  };
  if (!structurallyEqual(normalized, expected)) {
    throw new TypeError(`${label} does not join the base-sensor lifecycle.`);
  }
  return normalized;
}

/**
 * @param {Record<string, any>} value
 * @param {number} index
 * @param {number} selectedGlobalSlot
 * @param {number} recipientTeam
 * @param {ReadonlyArray<any>} allySlots
 * @param {ReadonlyArray<any>} enemySlots
 * @param {ReturnType<typeof normalizeSourceAxis>} axis
 * @param {string} recipientPublicAgentId
 */
function normalizeSourceAvailability(
  value,
  index,
  selectedGlobalSlot,
  recipientTeam,
  allySlots,
  enemySlots,
  axis,
  recipientPublicAgentId,
) {
  const label = `sensor_source_availability[${index}]`;
  exactKeys(value, SOURCE_AVAILABILITY_KEYS, "Source-material availability row");
  const normalized = Object.freeze({
    sensor_source_global_slot: globalSlot(
      value.sensor_source_global_slot,
      `${label}.sensor_source_global_slot`,
    ),
    sensor_source_public_agent_id: nonEmptyString(
      value.sensor_source_public_agent_id,
      `${label}.sensor_source_public_agent_id`,
    ),
    sensor_source_team_local_slot: nonNegativeInteger(
      value.sensor_source_team_local_slot,
      `${label}.sensor_source_team_local_slot`,
    ),
    sensor_source_configured_team_id: nonNegativeInteger(
      value.sensor_source_configured_team_id,
      `${label}.sensor_source_configured_team_id`,
    ),
    sensor_source_configured_active: booleanValue(
      value.sensor_source_configured_active,
      `${label}.sensor_source_configured_active`,
    ),
    relation_to_recipient: value.relation_to_recipient,
    base_sensor_relation_axis: value.base_sensor_relation_axis,
    base_sensor_observation_row: nonNegativeInteger(
      value.base_sensor_observation_row,
      `${label}.base_sensor_observation_row`,
    ),
    recorded_available: booleanValue(
      value.recorded_available,
      `${label}.recorded_available`,
    ),
  });
  if (
    normalized.sensor_source_global_slot !== index ||
    normalized.sensor_source_team_local_slot !== index % 5 ||
    normalized.sensor_source_team_local_slot >= 5 ||
    normalized.sensor_source_configured_team_id > 2 ||
    normalized.base_sensor_observation_row >= 5 ||
    !["self", "ally", "opponent", "inactive"].includes(
      normalized.relation_to_recipient,
    ) ||
    !["ally", "enemy"].includes(normalized.base_sensor_relation_axis)
  ) {
    throw new TypeError(`${label} is outside the exact V1 source axis.`);
  }
  const sourceBlockTeam = index < 5 ? 1 : 2;
  const expectedConfiguredTeam = normalized.sensor_source_configured_active
    ? sourceBlockTeam
    : 0;
  const expectedRelation = !normalized.sensor_source_configured_active
    ? "inactive"
    : index === selectedGlobalSlot
      ? "self"
      : sourceBlockTeam === recipientTeam
        ? "ally"
        : "opponent";
  const expectedAxis = sourceBlockTeam === recipientTeam ? "ally" : "enemy";
  const axisSlots = expectedAxis === "ally" ? allySlots : enemySlots;
  const axisIds =
    expectedAxis === "ally"
      ? axis.ally_observation_row_public_agent_id_by_id
      : axis.enemy_observation_row_public_agent_id_by_id;
  const row = normalized.base_sensor_observation_row;
  if (
    normalized.sensor_source_configured_team_id !== expectedConfiguredTeam ||
    normalized.relation_to_recipient !== expectedRelation ||
    (index === selectedGlobalSlot) !== (normalized.relation_to_recipient === "self") ||
    normalized.base_sensor_relation_axis !== expectedAxis ||
    axisSlots[row] !== index ||
    axisIds[row] !== normalized.sensor_source_public_agent_id ||
    (normalized.recorded_available && expectedRelation !== "ally") ||
    (expectedRelation === "self" &&
      normalized.sensor_source_public_agent_id !== recipientPublicAgentId)
  ) {
    throw new TypeError(`${label} does not join the recipient-relative axes.`);
  }
  return normalized;
}

/**
 * Derive the exact source-only scalar scene that the Python adapter publishes.
 * Comparing against this allowlisted reconstruction prevents nested wire
 * fields from crossing the browser privacy boundary through object spreads.
 *
 * @param {ReturnType<typeof normalizeSourceBaseFrame>} base
 * @param {number} selectedGlobalSlot
 * @param {ReturnType<typeof normalizeSourceAxis>} axis
 */
function deriveSourceScene(base, selectedGlobalSlot, axis) {
  const mapObstacles = [];
  for (let slot = 0; slot < base.map_obstacle_features.length; slot += 1) {
    const row = base.map_obstacle_features[slot];
    if (!sourceWireBoolean(row[7], `map_obstacle_features[${slot}][7]`)) {
      continue;
    }
    const kind = sourceWireInteger(row[0], `map_obstacle_features[${slot}][0]`, 0, 2);
    if (kind !== 1 && kind !== 2) {
      throw new TypeError("Visible source-material obstacle kind is invalid.");
    }
    mapObstacles.push(
      Object.freeze({
        obstacle_id: `base-sensor-obstacle-${slot}`,
        kind: kind === 1 ? "pillar" : "wall",
        center: Object.freeze([row[1], row[2]]),
        radius: kind === 1 ? row[3] : null,
        width: kind === 2 ? row[4] : null,
        height: kind === 2 ? row[5] : null,
        theta: row[6],
      }),
    );
  }
  const selfFeatures = base.self_features;
  if (!sourceWireBoolean(selfFeatures[4], "self_features[4]")) {
    throw new TypeError("Selected source-material actor must be configured active.");
  }
  const recipientTeam = sourceWireInteger(selfFeatures[3], "self_features[3]", 1, 2);
  const expectedBlockTeam = selectedGlobalSlot < 5 ? 1 : 2;
  const teamLocalSlot = selectedGlobalSlot % 5;
  if (recipientTeam !== expectedBlockTeam) {
    throw new TypeError("Source-material self team does not join its global slot.");
  }
  const selfActor = Object.freeze({
    global_slot: selectedGlobalSlot,
    public_agent_id: base.public_agent_id,
    team_local_slot: teamLocalSlot,
    team_id: recipientTeam,
    class_id: sourceWireInteger(selfFeatures[6], "self_features[6]", 1, 5),
    position: Object.freeze([selfFeatures[0], selfFeatures[1]]),
    radius: selfFeatures[2],
    alive: sourceWireBoolean(selfFeatures[5], "self_features[5]"),
    current_health: selfFeatures[12],
    max_health: selfFeatures[13],
    effective_movement_speed: selfFeatures[8],
    ultimate_cooldown_remaining: sourceWireInteger(
      selfFeatures[14],
      "self_features[14]",
    ),
    steps_until_out_of_combat: sourceWireInteger(selfFeatures[29], "self_features[29]"),
    spawn_shield_remaining:
      base.spawn_lifecycle.spawn_shield_actual_durations_by_team[0][teamLocalSlot],
    status_feature_values: Object.freeze(selfFeatures.slice(15, 29)),
  });
  const visibleBodies = [];
  for (const [relation, rows, visibility, publicIds] of [
    [
      "ally",
      base.ally_unit_features,
      base.ally_visibility_mask,
      axis.ally_observation_row_public_agent_id_by_id,
    ],
    [
      "enemy",
      base.enemy_unit_features,
      base.enemy_visibility_mask,
      axis.enemy_observation_row_public_agent_id_by_id,
    ],
  ]) {
    for (let observationRow = 0; observationRow < 5; observationRow += 1) {
      if (!visibility[observationRow]) {
        continue;
      }
      const row = rows[observationRow];
      if (
        !sourceWireBoolean(row[4], `${relation}_unit_features[${observationRow}][4]`)
      ) {
        throw new TypeError("Visible source-material body must be recorded active.");
      }
      visibleBodies.push(
        Object.freeze({
          relation,
          observation_row: observationRow,
          public_agent_id: publicIds[observationRow],
          position: Object.freeze([row[0], row[1]]),
          radius: row[2],
          team_id: sourceWireInteger(
            row[3],
            `${relation}_unit_features[${observationRow}][3]`,
            1,
            2,
          ),
          class_id: sourceWireInteger(
            row[6],
            `${relation}_unit_features[${observationRow}][6]`,
            1,
            5,
          ),
          alive: sourceWireBoolean(
            row[5],
            `${relation}_unit_features[${observationRow}][5]`,
          ),
          current_health: row[12],
          max_health: row[13],
          effective_movement_speed: row[8],
          ultimate_cooldown_remaining: sourceWireInteger(
            row[14],
            `${relation}_unit_features[${observationRow}][14]`,
          ),
          steps_until_out_of_combat: sourceWireInteger(
            row[29],
            `${relation}_unit_features[${observationRow}][29]`,
          ),
          status_feature_values: Object.freeze(row.slice(15, 29)),
        }),
      );
    }
  }
  return Object.freeze({
    map: Object.freeze({
      width: base.context_features[2],
      height: base.context_features[3],
      obstacles: Object.freeze(mapObstacles),
    }),
    self_actor: selfActor,
    visible_bodies: Object.freeze(visibleBodies),
  });
}

/** @param {unknown} rawValues */
function normalizeStatusFeatures(rawValues) {
  if (
    !Array.isArray(rawValues) ||
    rawValues.length !== 14 ||
    rawValues.some(
      (value) => typeof value !== "number" || !Number.isFinite(value) || value < 0,
    )
  ) {
    throw new TypeError("Source-material status features must retain V1 shape.");
  }
  return Object.freeze(
    STATUS_FEATURES.flatMap(([offset, featureIndex, tokenId, effectClassId]) => {
      const duration = rawValues[offset];
      if (!Number.isInteger(duration)) {
        throw new TypeError("Source-material status durations must be integers.");
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
    }),
  );
}

/**
 * @param {Record<string, any>} frame
 * @param {ReturnType<typeof normalizeCursor>} cursor
 */
function normalizeResearcherProjection(frame, cursor) {
  const projection = record(frame.projection, "Researcher replay projection");
  const scene = record(projection.scene, "Researcher replay scene");
  const episodeId = nonEmptyString(scene.episode_id, "scene.episode_id");
  const normalized = normalizeDebuggerAudienceProjectionV2({
    frame_kind: "researcher_live_debugger",
    episode_id: episodeId,
    frame_index: cursor.frame_index,
    frame_id: frame.frame_id,
    simulator_step_count: frame.simulator_step_count,
    incoming_transition_id: frame.incoming_transition_id,
    projection,
    hud: {},
  });
  return { normalized, episodeId };
}

/**
 * @param {Record<string, any>} frame
 * @param {ReturnType<typeof normalizeCursor>} cursor
 */
function normalizePovProjection(frame, cursor) {
  const projection = record(frame.projection, "Actor POV replay projection");
  const scene = record(projection.scene, "Actor POV replay scene");
  const sourceFrameId = nonEmptyString(scene.source_frame_id, "scene.source_frame_id");
  const episodeId = nonEmptyString(scene.episode_id, "scene.episode_id");
  const selfActor = record(scene.self_actor, "Actor POV replay self actor");
  if (
    frame.public_agent_id !== selfActor.public_agent_id ||
    frame.pov_global_slot !== selfActor.global_slot ||
    frame.pov_frame_id !== scene.pov_frame_id
  ) {
    throw new TypeError("Actor POV replay identity does not join its projection.");
  }
  const normalized = normalizeDebuggerAudienceProjectionV2({
    frame_kind: "actor_pov_live_debugger",
    episode_id: episodeId,
    frame_index: cursor.frame_index,
    frame_id: sourceFrameId,
    simulator_step_count: frame.simulator_step_count,
    incoming_pov_transition_id: frame.incoming_pov_transition_id,
    projection,
    hud: {},
  });
  return { normalized, episodeId };
}

/**
 * @param {Record<string, any>} frame
 * @param {ReturnType<typeof normalizeCursor>} cursor
 */
function normalizeSourceMaterialProjection(frame, cursor) {
  const projection = record(frame.projection, "Source-material replay projection");
  exactKeys(projection, SOURCE_PROJECTION_KEYS, "Source-material replay projection");
  if (
    projection.schema_version !== 1 ||
    projection.disclosure_label !== SOURCE_MATERIAL_DISCLOSURE ||
    projection.observation_materialization !== "source_material_only" ||
    projection.exact_actor_input_export_available !== false
  ) {
    throw new TypeError("Source-material replay disclosure root is invalid.");
  }
  const axis = normalizeSourceAxis(
    record(projection.axis_mapping, "Source-material axis mapping"),
  );
  const allySlots = shapedArray(
    projection.ally_observation_row_global_slot_by_id,
    [5],
    "ally_observation_row_global_slot_by_id",
    globalSlot,
  );
  const enemySlots = shapedArray(
    projection.enemy_observation_row_global_slot_by_id,
    [5],
    "enemy_observation_row_global_slot_by_id",
    globalSlot,
  );
  if (new Set([...allySlots, ...enemySlots]).size !== 10) {
    throw new TypeError("Source-material relation axes must partition all ten slots.");
  }
  const base = normalizeSourceBaseFrame(
    record(projection.base_sensor_frame, "Source-material base sensor frame"),
  );
  const sourceScene = record(
    projection.base_sensor_scene,
    "Source-material replay scene",
  );
  exactKeys(sourceScene, SOURCE_BASE_SCENE_KEYS, "Source-material replay scene");
  const episodeId = nonEmptyString(
    sourceScene.episode_id,
    "base_sensor_scene.episode_id",
  );
  const frameIndex = nonNegativeInteger(
    sourceScene.frame_index,
    "base_sensor_scene.frame_index",
  );
  const simulatorStep = nonNegativeInteger(
    sourceScene.simulator_step_count,
    "base_sensor_scene.simulator_step_count",
  );
  const sourceFrameId = nonEmptyString(
    sourceScene.source_frame_id,
    "base_sensor_scene.source_frame_id",
  );
  const normalizedMap = normalizeSourceMap(
    record(sourceScene.map, "Source-material map"),
    "base_sensor_scene.map",
  );
  const selfActor = normalizeSourceSelf(
    record(sourceScene.self_actor, "Source-material self actor"),
  );
  if (!Array.isArray(sourceScene.visible_bodies)) {
    throw new TypeError("Source-material visible bodies must be an array.");
  }
  const normalizedBodies = Object.freeze(
    sourceScene.visible_bodies.map((body, index) =>
      normalizeSourceBody(record(body, `Source-material visible body ${index}`), index),
    ),
  );
  if (!Array.isArray(sourceScene.spawn_pads) || sourceScene.spawn_pads.length !== 10) {
    throw new TypeError("Source-material spawn pads must retain the ten-slot axis.");
  }
  const teamNames = axis.spawn_lifecycle_team_axis_name_by_id;
  const normalizedPads = Object.freeze(
    sourceScene.spawn_pads.map((pad, index) =>
      normalizeSourceSpawnPad(
        record(pad, `Source-material spawn pad ${index}`),
        index,
        base.spawn_lifecycle,
        teamNames,
      ),
    ),
  );
  if (
    !Array.isArray(sourceScene.respawn_waves) ||
    sourceScene.respawn_waves.length !== 2
  ) {
    throw new TypeError("Source-material respawn waves must retain both team rows.");
  }
  const normalizedWaves = Object.freeze(
    sourceScene.respawn_waves.map((wave, index) =>
      normalizeSourceRespawnWave(
        record(wave, `Source-material respawn wave ${index}`),
        index,
        base.spawn_lifecycle,
        teamNames,
      ),
    ),
  );
  const derivedScene = deriveSourceScene(base, frame.selected_global_slot, axis);
  if (
    !structurallyEqual(normalizedMap, derivedScene.map) ||
    !structurallyEqual(selfActor, derivedScene.self_actor) ||
    !structurallyEqual(normalizedBodies, derivedScene.visible_bodies)
  ) {
    throw new TypeError(
      "Source-material scalar scene does not derive from its base-sensor frame.",
    );
  }
  if (
    !Array.isArray(projection.sensor_source_availability) ||
    projection.sensor_source_availability.length !== 10
  ) {
    throw new TypeError("Source-material availability must retain all ten rows.");
  }
  const availability = Object.freeze(
    projection.sensor_source_availability.map((row, index) =>
      normalizeSourceAvailability(
        record(row, `Source-material availability row ${index}`),
        index,
        frame.selected_global_slot,
        selfActor.team_id,
        allySlots,
        enemySlots,
        axis,
        base.public_agent_id,
      ),
    ),
  );
  const expectedTransitionId =
    cursor.frame_index === 0
      ? null
      : `${episodeId}:transition:${cursor.frame_index - 1}`;
  if (
    sourceScene.schema_version !== 1 ||
    sourceScene.audience_badge !== SOURCE_MATERIAL_DISCLOSURE ||
    sourceScene.observation_materialization !== "source_material_only" ||
    frame.public_agent_id !== selfActor.public_agent_id ||
    frame.selected_global_slot !== selfActor.global_slot ||
    selfActor.team_local_slot !== frame.selected_global_slot % 5 ||
    base.episode_id !== episodeId ||
    base.public_agent_id !== frame.public_agent_id ||
    base.frame_index !== cursor.frame_index ||
    base.simulator_step_count !== frame.simulator_step_count ||
    base.source_material_frame_id !== frame.source_material_frame_id ||
    base.source_frame_id !== frame.source_frame_id ||
    frame.source_material_frame_id !==
      `${episodeId}:shared-obs-source-material:${frame.public_agent_id}:frame:${cursor.frame_index}` ||
    frame.source_frame_id !== `${episodeId}:frame:${cursor.frame_index}` ||
    sourceScene.source_frame_id !== frame.source_frame_id ||
    sourceFrameId !== frame.source_frame_id ||
    frameIndex !== cursor.frame_index ||
    simulatorStep !== frame.simulator_step_count ||
    projection.incoming_transition_id !== frame.incoming_transition_id ||
    frame.incoming_transition_id !== expectedTransitionId
  ) {
    throw new TypeError("Source-material replay identity or disclosure is invalid.");
  }
  const normalizedSourceScene = Object.freeze({
    schema_version: 1,
    audience_badge: SOURCE_MATERIAL_DISCLOSURE,
    observation_materialization: "source_material_only",
    episode_id: episodeId,
    frame_index: frameIndex,
    source_frame_id: sourceFrameId,
    simulator_step_count: simulatorStep,
    map: normalizedMap,
    self_actor: selfActor,
    visible_bodies: normalizedBodies,
    spawn_pads: normalizedPads,
    respawn_waves: normalizedWaves,
  });
  const normalizedProjection = Object.freeze({
    schema_version: 1,
    disclosure_label: SOURCE_MATERIAL_DISCLOSURE,
    observation_materialization: "source_material_only",
    exact_actor_input_export_available: false,
    axis_mapping: axis,
    ally_observation_row_global_slot_by_id: allySlots,
    enemy_observation_row_global_slot_by_id: enemySlots,
    base_sensor_frame: base,
    base_sensor_scene: normalizedSourceScene,
    incoming_transition_id: projection.incoming_transition_id,
    sensor_source_availability: availability,
  });
  const visibleBodies = normalizedBodies.map((body) => {
    return Object.freeze({
      ...body,
      observation_key: `${body.relation}:${body.observation_row}`,
      effective_speed: body.effective_movement_speed,
      ultimate_cooldown: body.ultimate_cooldown_remaining,
      statuses: normalizeStatusFeatures(body.status_feature_values),
    });
  });
  const normalizedSelf = Object.freeze({
    ...selfActor,
    effective_speed: selfActor.effective_movement_speed,
    ultimate_cooldown: selfActor.ultimate_cooldown_remaining,
    statuses: normalizeStatusFeatures(selfActor.status_feature_values),
  });
  const scene = Object.freeze({
    ...normalizedSourceScene,
    audience: "agent_pov",
    audience_badge: normalizedProjection.disclosure_label,
    frame_id: frame.source_frame_id,
    incoming_transition_id: frame.incoming_transition_id,
    agents: Object.freeze([normalizedSelf]),
    observed_bodies: Object.freeze(visibleBodies),
    aura_fields: Object.freeze([]),
    ranges: Object.freeze([]),
    observer_visibility: Object.freeze([]),
    selection: Object.freeze({
      controlled_global_slot: null,
      selected_global_slot: null,
    }),
    selected_legality: null,
    pending_route: null,
  });
  return { scene, episodeId, projection: normalizedProjection };
}

/** @param {unknown} value */
export function isReplayViewerFrame(value) {
  return isRecord(value) && REPLAY_FRAME_KINDS.has(value.frame_kind);
}

/**
 * Strictly normalize one audience-owned replay wire frame for shared visual
 * components. `animateIncoming` is response transport state and is never read
 * from the durable frame itself.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeReplayViewerFrame(value, animateIncoming = false) {
  const frame = record(value, "Replay viewer frame");
  if (!REPLAY_FRAME_KINDS.has(frame.frame_kind)) {
    throw new TypeError("Unknown replay viewer frame kind.");
  }
  const kindKeys =
    frame.frame_kind === "researcher_replay_viewer"
      ? [
          "view_mode",
          "frame_id",
          "simulator_step_count",
          "incoming_transition_index",
          "incoming_transition_id",
          "completion",
          "processing",
          "show_ranges",
          "projection",
        ]
      : frame.frame_kind === "actor_pov_replay_viewer"
        ? [
            "view_mode",
            "pov_global_slot",
            "public_agent_id",
            "pov_frame_id",
            "simulator_step_count",
            "incoming_pov_transition_id",
            "completion",
            "processing_disclosure",
            "projection",
          ]
        : [
            "view_mode",
            "selected_global_slot",
            "public_agent_id",
            "observation_materialization",
            "source_material_frame_id",
            "source_frame_id",
            "simulator_step_count",
            "incoming_transition_id",
            "completion",
            "processing",
            "projection",
          ];
  exactKeys(frame, [...COMMON_FRAME_KEYS, ...kindKeys], "Replay viewer frame");
  if (
    frame.schema_version !== 1 ||
    !Number.isInteger(frame.revision) ||
    frame.revision < 0 ||
    !PRESETS.has(frame.preset) ||
    typeof frame.verbose !== "boolean"
  ) {
    throw new TypeError("Replay viewer frame scalar contract is invalid.");
  }
  nonEmptyString(frame.viewer_session_id, "viewer_session_id");
  nonEmptyString(frame.timeline_id, "timeline_id");
  nonNegativeInteger(frame.simulator_step_count, "simulator_step_count");
  const summary = normalizeArtifactSummary(
    record(frame.artifact_summary, "Replay artifact summary"),
  );
  const cursor = normalizeCursor(record(frame.cursor, "Replay cursor"));
  if (cursor.final_frame_index !== summary.recorded_transition_count) {
    throw new TypeError("Replay cursor final index does not join the artifact.");
  }
  if (typeof animateIncoming !== "boolean") {
    throw new TypeError("Replay animation intent must be a boolean.");
  }

  /** @type {Record<string, any>} */
  let scene;
  /** @type {Record<string, any> | null} */
  let eventBatch = null;
  /** @type {string} */
  let episodeId;
  /** @type {Record<string, any>} */
  let completion;
  /** @type {Record<string, any>} */
  let processing;
  /** @type {string} */
  let replayAudience;
  /** @type {string | null} */
  let transitionId;
  /** @type {Record<string, any>} */
  let normalizedProjection = frame.projection;
  const replayReference = summary.replay_reference;
  const expectedEpisodeId = replayReference.episode_id;
  let expectedTimelineId;
  if (frame.frame_kind === "researcher_replay_viewer") {
    if (summary.metric_report_availability === "not_available_in_actor_pov") {
      throw new TypeError("Researcher replay cannot use actor-POV metric disclosure.");
    }
    if (frame.view_mode !== "researcher" || typeof frame.show_ranges !== "boolean") {
      throw new TypeError("Researcher replay mode fields are invalid.");
    }
    const projectionResult = normalizeResearcherProjection(frame, cursor);
    normalizedProjection = projectionResult.normalized.projection;
    const researcherScene = projectionResult.normalized.scene;
    const researcherSelection = isRecord(researcherScene.selection)
      ? researcherScene.selection
      : {};
    scene = Object.freeze({
      ...researcherScene,
      selection: Object.freeze({
        ...researcherSelection,
        controlled_global_slot: null,
      }),
      selected_legality: null,
    });
    eventBatch = projectionResult.normalized.eventBatch;
    episodeId = projectionResult.episodeId;
    completion = normalizeResearcherCompletion(
      record(frame.completion, "Researcher completion"),
      summary,
    );
    processing = normalizeProcessing(record(frame.processing, "Replay processing"));
    replayAudience = "researcher";
    transitionId = frame.incoming_transition_id;
    expectedTimelineId = `${replayReference.artifact_id}:timeline:researcher`;
    const expectedTransitionIndex =
      cursor.frame_index === 0 ? null : cursor.frame_index - 1;
    const expectedTransitionId =
      expectedTransitionIndex === null
        ? null
        : `${expectedEpisodeId}:transition:${expectedTransitionIndex}`;
    if (
      frame.frame_id !== `${expectedEpisodeId}:frame:${cursor.frame_index}` ||
      frame.incoming_transition_index !== expectedTransitionIndex ||
      frame.incoming_transition_id !== expectedTransitionId
    ) {
      throw new TypeError("Researcher replay frame identity is not canonical.");
    }
  } else if (frame.frame_kind === "actor_pov_replay_viewer") {
    if (summary.metric_report_availability !== "not_available_in_actor_pov") {
      throw new TypeError("Actor POV replay must hide metric-report availability.");
    }
    if (frame.view_mode !== "pov") {
      throw new TypeError("Actor POV replay must use POV view mode.");
    }
    globalSlot(frame.pov_global_slot, "pov_global_slot");
    nonEmptyString(frame.public_agent_id, "public_agent_id");
    const projectionResult = normalizePovProjection(frame, cursor);
    normalizedProjection = projectionResult.normalized.projection;
    scene = Object.freeze({
      ...projectionResult.normalized.scene,
      selection: Object.freeze({
        controlled_global_slot: null,
        selected_global_slot: null,
      }),
      selected_legality: null,
    });
    eventBatch = projectionResult.normalized.eventBatch;
    episodeId = projectionResult.episodeId;
    completion = normalizePovCompletion(
      record(frame.completion, "Actor POV completion"),
      summary,
    );
    processing = normalizePovProcessing(
      record(frame.processing_disclosure, "Actor POV processing disclosure"),
    );
    replayAudience = "actor_pov";
    transitionId = frame.incoming_pov_transition_id;
    expectedTimelineId = `${replayReference.artifact_id}:timeline:actor-pov:${frame.public_agent_id}`;
    const expectedPovFrameId = `${expectedEpisodeId}:actor-pov:${frame.public_agent_id}:frame:${cursor.frame_index}`;
    const expectedPovTransitionId =
      cursor.frame_index === 0
        ? null
        : `${expectedEpisodeId}:actor-pov:${frame.public_agent_id}:transition:${cursor.frame_index - 1}`;
    if (
      frame.pov_frame_id !== expectedPovFrameId ||
      frame.incoming_pov_transition_id !== expectedPovTransitionId
    ) {
      throw new TypeError("Actor POV replay frame identity is not canonical.");
    }
    const incomingCues = normalizedProjection.incoming_cues;
    const endedCues = incomingCues.filter(
      (/** @type {Record<string, any>} */ cue) => cue.cue_type === "episode_ended",
    );
    const isFinalCursor = cursor.frame_index === cursor.final_frame_index;
    const physicalEpisodeEnded = completion.terminated || completion.truncated;
    if (
      endedCues.length > 1 ||
      (!isFinalCursor && endedCues.length !== 0) ||
      (isFinalCursor && physicalEpisodeEnded && endedCues.length !== 1) ||
      (endedCues.length === 1 &&
        (endedCues[0] !== incomingCues.at(-1) ||
          endedCues[0].terminated !== completion.terminated ||
          endedCues[0].truncated !== completion.truncated ||
          (endedCues[0].public_end_reason !== null &&
            endedCues[0].public_end_reason !==
              completion.public_end_or_failure_reason)))
    ) {
      throw new TypeError(
        "Actor POV replay episode-ended cue does not join completion and cursor authority.",
      );
    }
  } else {
    if (summary.metric_report_availability === "not_available_in_actor_pov") {
      throw new TypeError(
        "Source-material replay cannot use actor-POV metric disclosure.",
      );
    }
    if (
      frame.view_mode !== "pov" ||
      frame.observation_materialization !== "source_material_only"
    ) {
      throw new TypeError("Source-material replay must use POV view mode.");
    }
    globalSlot(frame.selected_global_slot, "selected_global_slot");
    nonEmptyString(frame.public_agent_id, "public_agent_id");
    const projectionResult = normalizeSourceMaterialProjection(frame, cursor);
    scene = projectionResult.scene;
    episodeId = projectionResult.episodeId;
    normalizedProjection = projectionResult.projection;
    completion = normalizeResearcherCompletion(
      record(frame.completion, "Source-material completion"),
      summary,
    );
    processing = normalizeProcessing(record(frame.processing, "Replay processing"));
    replayAudience = "shared_obs_source_material";
    transitionId = frame.incoming_transition_id;
    expectedTimelineId = `${replayReference.artifact_id}:timeline:shared-obs-source-material:${frame.public_agent_id}`;
  }
  if (
    completion.episode_id !== episodeId ||
    episodeId !== expectedEpisodeId ||
    frame.timeline_id !== expectedTimelineId
  ) {
    throw new TypeError("Replay completion episode does not join the projection.");
  }
  if (
    processing.disclosure !== "not_available_in_actor_pov" &&
    (processing.processed_transition_count > completion.validated_transition_count ||
      (processing.status === "succeeded" &&
        processing.processed_transition_count !==
          completion.validated_transition_count))
  ) {
    throw new TypeError("Replay processing progress does not join completion.");
  }
  return Object.freeze({
    ...frame,
    artifact_summary: summary,
    cursor,
    completion,
    processing,
    ...(frame.frame_kind === "actor_pov_replay_viewer"
      ? { processing_disclosure: processing }
      : {}),
    projection: normalizedProjection,
    viewer_mode: "replay",
    replay_audience: replayAudience,
    session_id: frame.viewer_session_id,
    run_generation: cursor.choreography_generation,
    episode_id: episodeId,
    frame_index: cursor.frame_index,
    simulator_step: frame.simulator_step_count,
    transition_id: transitionId,
    scene,
    event_batch: eventBatch,
    hud: Object.freeze({}),
    animate_incoming: animateIncoming,
  });
}

/**
 * Normalize a durable replay frame. Direct frames are always settled;
 * animation authority is intentionally unavailable through this API.
 *
 * @param {unknown} value
 */
export function normalizeReplayViewerFrameV1(value) {
  return normalizeReplayViewerFrame(value, false);
}

/**
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
export function normalizeReplayCommandResponseV1(value) {
  const response = record(value, "Replay command response");
  exactKeys(
    response,
    ["schema_version", "result", "frame", "notice", "animate_incoming"],
    "Replay command response",
  );
  if (
    response.schema_version !== 1 ||
    !["applied", "duplicate", "no_op", "shutdown_scheduled"].includes(
      response.result,
    ) ||
    (response.notice !== null && typeof response.notice !== "string") ||
    typeof response.animate_incoming !== "boolean"
  ) {
    throw new TypeError("Replay command response scalar contract is invalid.");
  }
  const frame = normalizeReplayViewerFrame(response.frame, response.animate_incoming);
  if (
    response.animate_incoming &&
    (response.result !== "applied" ||
      frame.cursor.frame_index === 0 ||
      frame.cursor.choreography_generation === 0)
  ) {
    throw new TypeError("Replay response animation intent is incoherent.");
  }
  return Object.freeze({ ...response, frame });
}

/**
 * Strictly normalize the replay-only failure envelope. Error responses never
 * carry animation authority; an optional latest frame is therefore always
 * installed settled.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
export function normalizeReplayApiErrorV1(value) {
  const error = record(value, "Replay API error");
  exactKeys(
    error,
    ["schema_version", "error_code", "message", "latest_frame"],
    "Replay API error",
  );
  if (
    error.schema_version !== 1 ||
    !REPLAY_ERROR_CODES.has(error.error_code) ||
    typeof error.message !== "string" ||
    error.message.trim() === ""
  ) {
    throw new TypeError("Replay API error scalar contract is invalid.");
  }
  const latestFrame =
    error.latest_frame === null
      ? null
      : normalizeReplayViewerFrameV1(error.latest_frame);
  return Object.freeze({ ...error, latest_frame: latestFrame });
}

/**
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
export function normalizeReplayTimelineV1(value) {
  const timeline = record(value, "Replay timeline");
  if (!TIMELINE_KINDS.has(timeline.timeline_kind)) {
    throw new TypeError("Unknown replay timeline kind.");
  }
  const audienceKeys =
    timeline.timeline_kind === "researcher"
      ? []
      : timeline.timeline_kind === "actor_pov"
        ? ["pov_global_slot", "public_agent_id"]
        : ["selected_global_slot", "public_agent_id", "observation_materialization"];
  exactKeys(
    timeline,
    [
      "schema_version",
      "timeline_kind",
      "timeline_id",
      "artifact_summary",
      "final_frame_index",
      "completion",
      "rows",
      ...audienceKeys,
    ],
    "Replay timeline",
  );
  if (timeline.schema_version !== 1) {
    throw new TypeError("Replay timeline must use schema version 1.");
  }
  nonEmptyString(timeline.timeline_id, "timeline_id");
  const summary = normalizeArtifactSummary(
    record(timeline.artifact_summary, "Replay artifact summary"),
  );
  const finalFrameIndex = nonNegativeInteger(
    timeline.final_frame_index,
    "timeline.final_frame_index",
  );
  if (finalFrameIndex !== summary.recorded_transition_count) {
    throw new TypeError("Replay timeline final index does not join the artifact.");
  }
  const completion =
    timeline.timeline_kind === "actor_pov"
      ? normalizePovCompletion(
          record(timeline.completion, "Actor POV completion"),
          summary,
        )
      : normalizeResearcherCompletion(
          record(timeline.completion, "Replay completion"),
          summary,
        );
  if (
    (timeline.timeline_kind === "actor_pov") !==
    (summary.metric_report_availability === "not_available_in_actor_pov")
  ) {
    throw new TypeError("Replay timeline metric disclosure is audience-incoherent.");
  }
  const reference = summary.replay_reference;
  const publicAgentId =
    timeline.timeline_kind === "researcher"
      ? null
      : nonEmptyString(timeline.public_agent_id, "timeline.public_agent_id");
  const expectedTimelineId =
    timeline.timeline_kind === "researcher"
      ? `${reference.artifact_id}:timeline:researcher`
      : timeline.timeline_kind === "actor_pov"
        ? `${reference.artifact_id}:timeline:actor-pov:${publicAgentId}`
        : `${reference.artifact_id}:timeline:shared-obs-source-material:${publicAgentId}`;
  if (
    timeline.timeline_id !== expectedTimelineId ||
    completion.episode_id !== reference.episode_id
  ) {
    throw new TypeError("Replay timeline identity does not join its artifact.");
  }
  if (!Array.isArray(timeline.rows) || timeline.rows.length !== finalFrameIndex + 1) {
    throw new TypeError("Replay timeline rows must cover every captured frame.");
  }
  const rowKeys =
    timeline.timeline_kind === "researcher"
      ? [
          "frame_index",
          "frame_id",
          "simulator_step_count",
          "incoming_transition_id",
          "incoming_event_count",
          "endpoint_kind",
        ]
      : timeline.timeline_kind === "actor_pov"
        ? [
            "frame_index",
            "pov_frame_id",
            "simulator_step_count",
            "incoming_pov_transition_id",
            "incoming_cue_count",
            "endpoint_kind",
          ]
        : [
            "frame_index",
            "source_material_frame_id",
            "simulator_step_count",
            "incoming_transition_id",
            "endpoint_kind",
          ];
  const endpointKind = expectedEndpointKind(completion);
  /** @type {number | null} */
  let previousSimulatorStep = null;
  const rows = timeline.rows.map((rawRow, index) => {
    const row = record(rawRow, "Replay timeline row");
    exactKeys(row, rowKeys, "Replay timeline row");
    if (
      row.frame_index !== index ||
      !Number.isInteger(row.simulator_step_count) ||
      row.simulator_step_count < 0 ||
      !ENDPOINT_KINDS.has(row.endpoint_kind) ||
      row.endpoint_kind !== (index === finalFrameIndex ? endpointKind : "none") ||
      (previousSimulatorStep !== null &&
        row.simulator_step_count !== previousSimulatorStep + 1)
    ) {
      throw new TypeError("Replay timeline row order or endpoint is invalid.");
    }
    previousSimulatorStep = row.simulator_step_count;
    if (timeline.timeline_kind === "researcher") {
      if (row.frame_id !== `${reference.episode_id}:frame:${index}`) {
        throw new TypeError("Researcher timeline frame identity is invalid.");
      }
      nonNegativeInteger(row.incoming_event_count, "incoming_event_count");
      if (index === 0 && row.incoming_event_count !== 0) {
        throw new TypeError("Researcher frame zero cannot carry incoming events.");
      }
    } else if (timeline.timeline_kind === "actor_pov") {
      if (
        row.pov_frame_id !==
        `${reference.episode_id}:actor-pov:${publicAgentId}:frame:${index}`
      ) {
        throw new TypeError("Actor POV timeline frame identity is invalid.");
      }
      nonNegativeInteger(row.incoming_cue_count, "incoming_cue_count");
      if (index === 0 && row.incoming_cue_count !== 0) {
        throw new TypeError("Actor POV frame zero cannot carry incoming cues.");
      }
    } else {
      if (
        row.source_material_frame_id !==
        `${reference.episode_id}:shared-obs-source-material:${publicAgentId}:frame:${index}`
      ) {
        throw new TypeError("Source-material timeline frame identity is invalid.");
      }
    }
    const incoming =
      timeline.timeline_kind === "actor_pov"
        ? row.incoming_pov_transition_id
        : row.incoming_transition_id;
    const expectedIncoming =
      index === 0
        ? null
        : timeline.timeline_kind === "actor_pov"
          ? `${reference.episode_id}:actor-pov:${publicAgentId}:transition:${index - 1}`
          : `${reference.episode_id}:transition:${index - 1}`;
    if (incoming !== expectedIncoming) {
      throw new TypeError("Replay timeline incoming identity is invalid.");
    }
    return Object.freeze({ ...row });
  });
  if (timeline.timeline_kind === "actor_pov") {
    globalSlot(timeline.pov_global_slot, "timeline.pov_global_slot");
    nonEmptyString(timeline.public_agent_id, "timeline.public_agent_id");
  } else if (timeline.timeline_kind === "shared_obs_source_material") {
    globalSlot(timeline.selected_global_slot, "timeline.selected_global_slot");
    nonEmptyString(timeline.public_agent_id, "timeline.public_agent_id");
    if (timeline.observation_materialization !== "source_material_only") {
      throw new TypeError("Source-material timeline disclosure is invalid.");
    }
  }
  return Object.freeze({
    ...timeline,
    artifact_summary: summary,
    completion,
    rows: Object.freeze(rows),
  });
}

/**
 * Join independently fetched replay frame/timeline roots after each has
 * crossed its strict audience normalizer. This rejects a late timeline
 * response from an audience that was superseded while the request was in
 * flight.
 *
 * @param {Readonly<Record<string, any>>} frame
 * @param {Readonly<Record<string, any>>} timeline
 */
export function joinReplayFrameAndTimeline(frame, timeline) {
  const expectedTimelineKind =
    frame.replay_audience === "researcher"
      ? "researcher"
      : frame.replay_audience === "actor_pov"
        ? "actor_pov"
        : frame.replay_audience === "shared_obs_source_material"
          ? "shared_obs_source_material"
          : null;
  const frameReference = record(
    frame.artifact_summary?.replay_reference,
    "Replay frame artifact reference",
  );
  const timelineReference = record(
    timeline.artifact_summary?.replay_reference,
    "Replay timeline artifact reference",
  );
  const currentRow = Array.isArray(timeline.rows)
    ? timeline.rows[frame.cursor?.frame_index]
    : null;
  const expectedIncoming =
    frame.replay_audience === "actor_pov"
      ? frame.incoming_pov_transition_id
      : frame.incoming_transition_id;
  const rowIncoming =
    frame.replay_audience === "actor_pov"
      ? currentRow?.incoming_pov_transition_id
      : currentRow?.incoming_transition_id;
  const audienceIdentityMatches =
    frame.replay_audience === "researcher" ||
    (frame.replay_audience === "actor_pov"
      ? timeline.pov_global_slot === frame.pov_global_slot &&
        timeline.public_agent_id === frame.public_agent_id
      : timeline.selected_global_slot === frame.selected_global_slot &&
        timeline.public_agent_id === frame.public_agent_id);
  if (
    frame.viewer_mode !== "replay" ||
    expectedTimelineKind === null ||
    timeline.timeline_kind !== expectedTimelineKind ||
    timeline.timeline_id !== frame.timeline_id ||
    timeline.final_frame_index !== frame.cursor?.final_frame_index ||
    !audienceIdentityMatches ||
    !structurallyEqual(timeline.artifact_summary, frame.artifact_summary) ||
    !structurallyEqual(timeline.completion, frame.completion) ||
    !isRecord(currentRow) ||
    currentRow.frame_index !== frame.cursor?.frame_index ||
    currentRow.simulator_step_count !== frame.simulator_step_count ||
    rowIncoming !== expectedIncoming ||
    timelineReference.artifact_id !== frameReference.artifact_id
  ) {
    throw new TypeError("Replay timeline does not join the current audience frame.");
  }
  return timeline;
}

/**
 * Pin immutable launch/artifact identity and exact revision movement before a
 * command response is eligible for frame/timeline installation.
 *
 * @param {Readonly<Record<string, any>>} previous
 * @param {Readonly<Record<string, any>>} next
 * @param {unknown} result
 */
export function validateReplayFrameContinuity(previous, next, result) {
  const previousReference = record(
    previous.artifact_summary?.replay_reference,
    "Previous replay artifact reference",
  );
  const nextReference = record(
    next.artifact_summary?.replay_reference,
    "Next replay artifact reference",
  );
  const revisionValid =
    result === "stale_resync" || result === "duplicate"
      ? next.revision >= previous.revision
      : next.revision ===
        (result === "applied" ? previous.revision + 1 : previous.revision);
  const settledGenerationValid =
    result !== "stale_resync" && result !== "duplicate"
      ? true
      : next.cursor?.cursor_generation >= previous.cursor?.cursor_generation &&
        next.cursor?.choreography_generation >=
          previous.cursor?.choreography_generation &&
        next.cursor?.choreography_generation -
          previous.cursor?.choreography_generation <=
          next.cursor?.cursor_generation - previous.cursor?.cursor_generation;
  const crossesActorPovDisclosureBoundary =
    previous.replay_audience !== next.replay_audience &&
    (previous.replay_audience === "actor_pov" || next.replay_audience === "actor_pov");
  const sharesCompletionAuthority =
    (previous.replay_audience === "actor_pov") ===
    (next.replay_audience === "actor_pov");
  const completionContinuityValid = sharesCompletionAuthority
    ? structurallyEqual(next.completion, previous.completion)
    : next.completion?.schema_version === previous.completion?.schema_version &&
      next.completion?.episode_id === previous.completion?.episode_id &&
      next.completion?.completion_state === previous.completion?.completion_state &&
      next.completion?.expected_transition_count ===
        previous.completion?.expected_transition_count &&
      next.completion?.terminated === previous.completion?.terminated &&
      next.completion?.truncated === previous.completion?.truncated &&
      structurallyEqual(
        next.completion?.completion_bases,
        previous.completion?.completion_bases,
      );
  const processingContinuityValid =
    !sharesCompletionAuthority ||
    structurallyEqual(next.processing, previous.processing);
  if (
    previous.viewer_mode !== "replay" ||
    next.viewer_mode !== "replay" ||
    next.viewer_session_id !== previous.viewer_session_id ||
    !structurallyEqual(nextReference, previousReference) ||
    next.artifact_summary?.expected_transition_count !==
      previous.artifact_summary?.expected_transition_count ||
    next.artifact_summary?.recorded_transition_count !==
      previous.artifact_summary?.recorded_transition_count ||
    next.artifact_summary?.recorded_frame_count !==
      previous.artifact_summary?.recorded_frame_count ||
    (!crossesActorPovDisclosureBoundary &&
      next.artifact_summary?.metric_report_availability !==
        previous.artifact_summary?.metric_report_availability) ||
    !completionContinuityValid ||
    !processingContinuityValid ||
    next.cursor?.final_frame_index !== previous.cursor?.final_frame_index ||
    !revisionValid ||
    !settledGenerationValid
  ) {
    throw new TypeError(
      "Replay command response breaks launch or artifact continuity.",
    );
  }
  return next;
}
