/**
 * @file Check replay transport frames, timelines, command responses, and artifact facts.
 * The Replay Viewer and authorized-presentation join layer use these helpers before
 * installing fetched data. Validation keeps artifact identity, selected audience,
 * completion evidence, and cursor generations coherent. The shared live projection
 * normalizer owns scene geometry and event/cue checks.
 * All work is synchronous and in memory. Returned records are frozen at the owned
 * record/array boundaries; this is not a general recursive copier. No file or network
 * request is made. Normalized transport does not by itself carry the separate
 * six-branch authorized-presentation approval marker.
 */
import { normalizeDebuggerAudienceProjectionV2 } from "./frame-normalizer.js";

const REPLAY_FRAME_KINDS = new Set([
  "researcher_replay_viewer",
  "actor_pov_replay_viewer",
]);
const TIMELINE_KINDS = new Set(["researcher", "actor_pov"]);
const PRESETS = new Set(["presentation", "analysis"]);
const COMPLETION_STATES = new Set(["complete", "partial", "interrupted", "failed"]);
const FAILURE_ORIGINS = new Set(["simulation", "policy", "validation", "capture"]);
const PROCESSING_STATES = new Set(["succeeded", "failed", "not_requested"]);
const PROCESSING_FAILURE_STAGES = new Set([
  "initial_validation",
  "reducer_initialize",
  "transition_validation",
  "reducer_advance",
  "offline_reducer_initialize",
  "offline_reducer_advance",
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
const ARTIFACT_FACTS_KEYS = [
  "schema_version",
  "artifact_summary",
  "completion",
  "processing",
];

/**
 * Return whether value is a non-null object that is not an array.
 * This checks only its container shape, not its replay schema or authority.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return value unchanged when it is a non-array object. Otherwise throw TypeError
 * using label to name the invalid field. Nested schema validation belongs to callers.
 *
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
 * Check that value has exactly the enumerable string keys in expected, ignoring
 * key order. Return undefined on success; throw TypeError naming label for missing
 * or extra keys. The caller supplies a record; symbol/non-enumerable keys are not
 * included by Object.keys.
 *
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

/**
 * Compare left and right recursively by Object.is for scalars, ordered array
 * entries, and equal sorted enumerable record keys. Return a boolean without changing
 * inputs. Inputs are finite JSON-like trees; cyclic objects are unsupported.
 *
 * @param {unknown} left @param {unknown} right @returns {boolean}
 */
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
 * Return numeric value when it is an integer at least zero, otherwise throw
 * TypeError naming label. Strings and booleans are not converted into valid inputs;
 * this helper does not impose Number.MAX_SAFE_INTEGER as an additional upper bound.
 *
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
 * Return integer value when it is greater than zero; otherwise throw TypeError
 * naming label. Uses the same exact-number input rules as nonNegativeInteger.
 *
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
 * Return integer value on the fixed simulator-global axis 0 through 9.
 * Throw TypeError naming label for an invalid type or range. Slots 0 through 4 are
 * Team A and 5 through 9 are Team B; this does not prove roster activity.
 *
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
 * Return value unchanged when it is a string with non-whitespace content.
 * Throw TypeError naming label otherwise. Accepted strings are not trimmed.
 *
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
 * Return value unchanged when it is a string or null; otherwise throw TypeError
 * naming label. Empty strings are allowed here; callers apply any stronger rule.
 *
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
 * Validate and shallow-copy the record value as a frozen replay reference.
 * Require its exact fields, reference/replay version 1, 2, 3 or 4 (4 records the Team
 * Deathmatch Red Zone rule) with matching versions,
 * canonical episode:replay ID, lowercase 64-character digest text, and positive byte
 * length. Throw TypeError for an invalid field or join. Digests are checked for syntax;
 * this helper does not read source bytes or recompute their hashes.
 *
 * @param {Record<string, any>} value
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeArtifactReference(value) {
  exactKeys(value, ARTIFACT_REFERENCE_KEYS, "Replay artifact reference");
  if (
    value.schema_id !== REPLAY_ARTIFACT_REFERENCE_SCHEMA_ID ||
    ![1, 2, 3, 4].includes(value.schema_version) ||
    value.replay_schema_version !== value.schema_version
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
 * Validate record value and return a frozen artifact summary with its normalized
 * replay reference. Require schema 1, a positive horizon, a captured count within it,
 * exactly T+1 frames, and a recognized metric-report availability state. Throw TypeError
 * for invalid structure or counts. Missing, not-recorded, and POV-hidden metrics remain
 * distinct; audience-specific helpers enforce which disclosure is allowed.
 *
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
    value.metric_report_availability !== "not_recorded" &&
    value.metric_report_availability !== "not_available_in_actor_pov"
  ) {
    throw new TypeError("Replay metric-report availability is invalid.");
  }
  return Object.freeze({ ...value, replay_reference: replayReference });
}

/**
 * Validate record value as a replay cursor and return a frozen copy.
 * Require exact schema-1 fields and nonnegative integer frame/final/generation values.
 * The current frame cannot exceed the final frame, and choreography generation cannot
 * exceed cursor generation. Throw TypeError otherwise; artifact count joins are checked
 * by the containing frame normalizer.
 *
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
 * Return a frozen copy of array value containing only task_terminal or
 * declared_horizon labels. Throw TypeError naming label for another shape or label.
 * This helper does not check ordering or duplicates; completion semantics does.
 *
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

/**
 * Return captured_prefix for an incomplete completion record. For a complete record,
 * map the exact ordered bases to task_terminal, declared_horizon, or their combined
 * endpoint kind. Throw TypeError if complete evidence has another ordering/content.
 * The caller supplies a normalized completion record.
 *
 * @param {Record<string, any>} completion
 */
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
 * Check completion evidence without copying value; return undefined on success.
 * captured is the saved transition count and reason its public failure/end reason.
 * failureOrigin is a researcher failure label or null; undefined means the POV record
 * does not expose that field. A zero-step prefix cannot be done. Complete data needs
 * exact ordered terminal/horizon evidence and no failure origin. Incomplete data needs
 * no completion basis and a nonblank reason; failed researcher data also needs a known
 * failure origin. Throw TypeError for contradictions. Other scalar checks occur first.
 *
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
 * Validate researcher completion record value against normalized artifact summary.
 * Counts must agree with the saved horizon/prefix, the final frame ID must be canonical,
 * and flags/reason/failure origin must match completion semantics. Return a frozen copy
 * with a frozen bases array, or throw TypeError. The enclosing frame additionally joins
 * the episode ID to its artifact; this helper does not infer a completion from playback.
 *
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
 * Validate actor-visible completion record value against normalized summary.
 * Require schema 1, matching horizon/captured counts, boolean done flags, and valid
 * public completion evidence/reason. Return a frozen copy with frozen bases, or throw
 * TypeError. No researcher failure-origin or processing details are added.
 *
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

/**
 * Validate researcher processing record value and return a frozen copy.
 * Keep succeeded, failed, and not_requested distinct. Successful/unrequested records
 * have null failure fields. Failure requires a known stage and nonempty code; only
 * transition-validation or reducer-advance stages admit an attempted index, and reducer
 * advance requires one. Counts/indices are nonnegative integers. Throw TypeError for
 * invalid fields or combinations; the enclosing artifact checks count totals.
 *
 * @param {Record<string, any>} value
 */
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
  if (value.status === "succeeded" || value.status === "not_requested") {
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
    value.failure_stage === "reducer_advance" ||
    value.failure_stage === "offline_reducer_advance";
  if (
    (!permitsAttemptedIndex && value.attempted_transition_index !== null) ||
    (["reducer_advance", "offline_reducer_advance"].includes(value.failure_stage) &&
      value.attempted_transition_index === null)
  ) {
    throw new TypeError("Replay processing attempted index is incoherent.");
  }
  return Object.freeze({ ...value });
}

/**
 * Require record value to contain only schema version 1 and the explicit
 * not_available_in_actor_pov disclosure. Return a frozen copy or throw TypeError.
 * This does not reveal whether researcher metric processing succeeded or failed.
 *
 * @param {Record<string, any>} value
 */
function normalizePovProcessing(value) {
  exactKeys(value, POV_PROCESSING_KEYS, "Actor POV processing disclosure");
  if (value.schema_version !== 1 || value.disclosure !== "not_available_in_actor_pov") {
    throw new TypeError("Actor POV processing disclosure is invalid.");
  }
  return Object.freeze({ ...value });
}

/**
 * Validate artifact-wide replay facts independently of the current fogged scene.
 * value must contain exact schema-1 artifact_summary, researcher completion, and
 * processing fields. Require consistent episode identity and processing counts; the
 * summary keeps actual metric availability rather than the POV-hidden placeholder.
 * Return a frozen object containing the three normalized records, or throw TypeError.
 * These facts support researcher panels; they do not authorize hidden battlefield data.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
export function normalizeReplayArtifactFactsV1(value) {
  const facts = record(value, "Replay artifact facts");
  exactKeys(facts, ARTIFACT_FACTS_KEYS, "Replay artifact facts");
  if (facts.schema_version !== 1) {
    throw new TypeError("Replay artifact facts must use schema version 1.");
  }
  const artifactSummary = normalizeArtifactSummary(
    record(facts.artifact_summary, "Replay artifact facts summary"),
  );
  if (artifactSummary.metric_report_availability === "not_available_in_actor_pov") {
    throw new TypeError(
      "Replay artifact facts must preserve canonical metric-report availability.",
    );
  }
  const completion = normalizeResearcherCompletion(
    record(facts.completion, "Replay artifact facts completion"),
    artifactSummary,
  );
  const processing = normalizeProcessing(
    record(facts.processing, "Replay artifact facts processing"),
  );
  if (
    completion.episode_id !== artifactSummary.replay_reference.episode_id ||
    processing.processed_transition_count > completion.validated_transition_count ||
    (processing.status === "succeeded" &&
      processing.processed_transition_count !== completion.validated_transition_count)
  ) {
    throw new TypeError("Replay artifact facts do not join one canonical artifact.");
  }
  return Object.freeze({
    schema_version: 1,
    artifact_summary: artifactSummary,
    completion,
    processing,
  });
}

/**
 * Check that normalized summary and recipient completion agree with normalized facts.
 * Compare the source reference, horizon/frame/transition counts, episode, completion
 * state, done flags, and ordered bases. Return undefined when they join, otherwise
 * throw TypeError. Researcher-only failure details and metric disclosure are not copied
 * into the actor-local completion record.
 *
 * @param {Readonly<Record<string, any>>} summary
 * @param {Readonly<Record<string, any>>} completion
 * @param {Readonly<Record<string, any>>} facts
 */
function validatePovArtifactFacts(summary, completion, facts) {
  const factSummary = facts.artifact_summary;
  const factCompletion = facts.completion;
  if (
    !structurallyEqual(factSummary.replay_reference, summary.replay_reference) ||
    factSummary.expected_transition_count !== summary.expected_transition_count ||
    factSummary.recorded_transition_count !== summary.recorded_transition_count ||
    factSummary.recorded_frame_count !== summary.recorded_frame_count ||
    factCompletion.episode_id !== completion.episode_id ||
    factCompletion.completion_state !== completion.completion_state ||
    factCompletion.expected_transition_count !== completion.expected_transition_count ||
    factCompletion.validated_transition_count !==
      completion.captured_transition_count ||
    factCompletion.terminated !== completion.terminated ||
    factCompletion.truncated !== completion.truncated ||
    !structurallyEqual(factCompletion.completion_bases, completion.completion_bases)
  ) {
    throw new TypeError(
      "Actor POV replay artifact facts do not join its recipient-local roots.",
    );
  }
}

/**
 * Return value when it is a finite JavaScript number; otherwise throw TypeError
 * naming label. Numeric strings, NaN, and infinities are rejected.
 *
 * @param {unknown} value @param {string} label
 */
function finiteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number.`);
  }
  return value;
}

/**
 * Normalize frame's researcher replay projection using the shared live scene authority.
 * cursor supplies the already checked artifact frame index. Return {normalized,
 * episodeId}; normalized contains scene, event batch, and normalized projection.
 * Throw TypeError for invalid scene/identity data. The temporary live-shaped carrier
 * is only an adapter input; it does not grant live command or animation authority.
 *
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
 * Normalize frame's actor replay projection using the shared POV scene authority.
 * cursor supplies the checked frame index. First require public actor ID, global slot,
 * and POV frame ID to agree with the source scene, then return {normalized, episodeId}.
 * Throw TypeError for failed joins or source validation. The adapter passes an empty
 * HUD and never constructs privileged researcher events.
 *
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
 * Return whether value is a record with a recognized researcher or actor-POV replay
 * frame_kind. This is a dispatch hint only: it does not validate the complete frame.
 *
 * @param {unknown} value
 */
export function isReplayViewerFrame(value) {
  return isRecord(value) && REPLAY_FRAME_KINDS.has(value.frame_kind);
}

/**
 * Validate an audience-specific replay wire value and build the frozen display frame.
 * animateIncoming defaults to false and must be a boolean supplied by a checked command
 * response, never by a durable frame field. Validate exact fields, schema, artifact,
 * cursor, source projection, completion, processing, canonical IDs, and any final POV
 * end cue. Return the normalized source fields plus scene/event_batch, replay audience,
 * session/episode/frame aliases, empty HUD, and animation intent. The output preset is
 * analysis; controlled selection and selected legality are cleared for replay.
 * Throw TypeError for unsupported fields, audience disclosures, values, or joins.
 * This transport normalizer does not attach authorized-presentation approval.
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
          "recorded_ordinary_movement_distance_scale",
          "projection",
        ]
      : [
          "artifact_facts",
          "view_mode",
          "pov_global_slot",
          "public_agent_id",
          "pov_frame_id",
          "simulator_step_count",
          "incoming_pov_transition_id",
          "completion",
          "processing_disclosure",
          "projection",
        ];
  exactKeys(frame, [...COMMON_FRAME_KEYS, ...kindKeys], "Replay viewer frame");
  if (
    frame.schema_version !== 1 ||
    !Number.isInteger(frame.revision) ||
    frame.revision < 0 ||
    !PRESETS.has(frame.preset) ||
    frame.verbose !== false
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
  /** @type {Readonly<Record<string, any>> | null} */
  let artifactFacts = null;
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
    const recordedMovementScale = finiteNumber(
      frame.recorded_ordinary_movement_distance_scale,
      "recorded_ordinary_movement_distance_scale",
    );
    if (recordedMovementScale <= 0 || recordedMovementScale > 1) {
      throw new TypeError(
        "recorded_ordinary_movement_distance_scale must be in (0, 1].",
      );
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
  } else {
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
    artifactFacts = normalizeReplayArtifactFactsV1(frame.artifact_facts);
    validatePovArtifactFacts(summary, completion, artifactFacts);
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
    preset: "analysis",
    artifact_summary: summary,
    cursor,
    completion,
    processing,
    ...(frame.frame_kind === "actor_pov_replay_viewer"
      ? { artifact_facts: artifactFacts, processing_disclosure: processing }
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
 * Validate durable replay wire value and return its frozen normalized display frame.
 * Accept researcher_replay_viewer or actor_pov_replay_viewer under schema 1. Strict
 * artifact, scene, cursor, identity, and completion checks run through the shared
 * normalizer. Throw TypeError on invalid input. animate_incoming is always false;
 * loading a durable frame never grants animation intent. No I/O is performed.
 *
 * @param {unknown} value
 */
export function normalizeReplayViewerFrameV1(value) {
  return normalizeReplayViewerFrame(value, false);
}

/**
 * Validate command-response value and return a frozen envelope with normalized frame.
 * Require exact schema-1 fields: result, frame, nullable notice, and boolean
 * animate_incoming. Result is applied, duplicate, no_op, or shutdown_scheduled.
 * Animation may be true only for applied responses after frame zero with positive
 * choreography generation. Throw TypeError for invalid fields, frame, or animation
 * intent. Continuity with the previously installed frame is checked separately.
 *
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
 * Validate replay API error value and return a frozen failure envelope.
 * Require schema 1, a recognized replay error_code, nonblank message, and latest_frame
 * that is null or a valid durable replay frame. A supplied latest frame is normalized
 * with animation disabled. Throw TypeError on invalid input; this helper does not throw
 * merely because the envelope describes a server error, and it performs no recovery I/O.
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
 * Validate a complete researcher or actor-POV timeline value and freeze its rows.
 * Require schema 1, exact audience fields, matching artifact/completion identity,
 * exactly T+1 ordered rows, adjacent simulator ticks, canonical incoming references,
 * and the correct final endpoint kind. Frame zero has no incoming reference or events/
 * cues. Actor POV includes its public ID and global slot but hides metric availability.
 * Return a frozen timeline with normalized summary/completion and frozen row copies.
 * Throw TypeError for a malformed record, audience disclosure, identity, count, or join.
 * Join it to the current frame separately before installing an asynchronously fetched result.
 *
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
      : ["pov_global_slot", "public_agent_id"];
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
      : `${reference.artifact_id}:timeline:actor-pov:${publicAgentId}`;
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
      : [
          "frame_index",
          "pov_frame_id",
          "simulator_step_count",
          "incoming_pov_transition_id",
          "incoming_cue_count",
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
    } else {
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
  }
  return Object.freeze({
    ...timeline,
    artifact_summary: summary,
    completion,
    rows: Object.freeze(rows),
  });
}

/**
 * Return timeline unchanged only when it joins the already normalized frame.
 * Both inputs must have crossed their audience-specific normalizers. Check audience,
 * selected actor where present, timeline/artifact/completion identity, final index,
 * and the current row's index, simulator tick, and incoming transition. Throw TypeError
 * for a mismatch, including a late response for a superseded audience. No copying,
 * fetching, or installation occurs here.
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
      : false);
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
 * Return normalized next only when it continues normalized previous for result.
 * Require the same launch session, source artifact, immutable counts, completion, and
 * artifact-wide facts. Audience changes may change the permitted local disclosure.
 * For applied, revision increases by one; duplicate or stale_resync permits a newer or
 * equal revision with nondecreasing, coherent cursor/choreography generations. Other
 * results require the same revision. The caller supplies an already validated result
 * label; this helper does not reject unknown labels separately.
 * Throw TypeError when continuity fails. This checks installation eligibility, not
 * that a requested command took effect or that playback advanced by one frame.
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
  const previousArtifactFacts =
    previous.replay_audience === "researcher"
      ? {
          schema_version: 1,
          artifact_summary: previous.artifact_summary,
          completion: previous.completion,
          processing: previous.processing,
        }
      : previous.artifact_facts;
  const nextArtifactFacts =
    next.replay_audience === "researcher"
      ? {
          schema_version: 1,
          artifact_summary: next.artifact_summary,
          completion: next.completion,
          processing: next.processing,
        }
      : next.artifact_facts;
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
    !structurallyEqual(nextArtifactFacts, previousArtifactFacts) ||
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
