/**
 * @file Validate and join the browser's six authorized presentation variants.
 * normalizeAuthorizedPresentationFrameV1 checks the generated schema, semantic joins,
 * privacy limits, endpoint hashes, and opaque identity keys before recursively freezing
 * and privately marking the result. Oracle and Agent POV, live and replay, retain their
 * separate information rights. The same-origin Python producer is the authority root;
 * hashes detect inconsistent content but do not authenticate an arbitrary producer.
 * Transport joins reject stale mixed responses before installation. Adapter modules
 * must test this module's private marker, not accept look-alike objects. Work is in
 * memory, with asynchronous Web Crypto SHA-256 checks and no network or file I/O.
 */

import { AUTHORIZED_PRESENTATION_SCHEMA_V1 } from "./authorized-presentation-schema.js";
import { normalizeLiveDebuggerFrameV2 } from "./frame-normalizer.js";
import {
  joinReplayFrameAndTimeline,
  normalizeReplayArtifactFactsV1,
  normalizeReplayCommandResponseV1,
  normalizeReplayTimelineV1,
  normalizeReplayViewerFrameV1,
  validateReplayFrameContinuity,
} from "./replay-frame-normalizer.js";
import {
  isTeamController,
  requiresSharedObs,
  systemTransportKeys,
} from "./system-controls.js";
import { CANONICAL_STATUS_ORDER, statusTokenIdFromCatalogId } from "./vocabulary.js";

const PRESENTATION_KINDS = new Set([
  "live_oracle",
  "live_no_shared_obs_agent_pov",
  "live_shared_obs_agent_pov",
  "replay_oracle",
  "replay_no_shared_obs_agent_pov",
  "replay_shared_obs_agent_pov",
]);
/** @type {Readonly<Record<string, readonly string[]>>} */
const RAW_FRAME_KEYS = Object.freeze({
  researcher_live_debugger: [
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
  ],
  actor_pov_live_debugger: [
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
  ],
  shared_obs_agent_pov_live_debugger: [
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
  ],
  researcher_replay_viewer: [
    "artifact_summary",
    "completion",
    "cursor",
    "frame_id",
    "frame_kind",
    "incoming_transition_id",
    "incoming_transition_index",
    "preset",
    "processing",
    "projection",
    "recorded_ordinary_movement_distance_scale",
    "revision",
    "schema_version",
    "show_ranges",
    "simulator_step_count",
    "timeline_id",
    "verbose",
    "view_mode",
    "viewer_session_id",
  ],
  actor_pov_replay_viewer: [
    "artifact_facts",
    "artifact_summary",
    "completion",
    "cursor",
    "frame_kind",
    "incoming_pov_transition_id",
    "pov_frame_id",
    "pov_global_slot",
    "preset",
    "processing_disclosure",
    "projection",
    "public_agent_id",
    "revision",
    "schema_version",
    "simulator_step_count",
    "timeline_id",
    "verbose",
    "view_mode",
    "viewer_session_id",
  ],
  shared_obs_agent_pov_replay_viewer: [
    "artifact_facts",
    "artifact_summary",
    "completion",
    "cursor",
    "frame_kind",
    "incoming_recipient_transition_id",
    "preset",
    "public_agent_id",
    "recipient_frame_id",
    "revision",
    "schema_version",
    "simulator_step_count",
    "timeline_id",
    "verbose",
    "view_mode",
    "viewer_session_id",
  ],
});

const SUPPORTED_SCHEMA_KEYWORDS = new Set([
  "$defs",
  "$ref",
  "additionalProperties",
  "anyOf",
  "const",
  "discriminator",
  "enum",
  "exclusiveMaximum",
  "exclusiveMinimum",
  "items",
  "maxItems",
  "maxLength",
  "maximum",
  "minItems",
  "minLength",
  "minimum",
  "oneOf",
  "pattern",
  "prefixItems",
  "properties",
  "required",
  "type",
]);

/** @type {Readonly<Record<string, string>>} */
const PRESENTATION_KEY_PUBLIC_FIELDS = Object.freeze({
  presentation_key: "public_agent_id",
  source_presentation_key: "source_public_agent_id",
  agent_presentation_key: "agent_public_agent_id",
  recipient_presentation_key: "recipient_public_agent_id",
  owner_presentation_key: "owner_public_agent_id",
  target_presentation_key: "target_public_agent_id",
  assigned_presentation_key: "assigned_public_agent_id",
  actor_presentation_key: "actor_public_agent_id",
});

const AGENT_FORBIDDEN_KEYS = new Set([
  "global_slot",
  "pov_global_slot",
  "selected_global_slot",
  "controlled_global_slot",
  "artifact_id",
  "timeline_id",
  "context_digest_sha256",
  "trajectory_content_digest_sha256",
  "canonical_digest_sha256",
  "metric_report",
  "processing",
  "source_material_frame_id",
  "source_frame_id",
]);
const AGENT_PAIRED_FORBIDDEN_VALUE_FIELDS = new Set([
  "artifact_id",
  "timeline_id",
  "recipient_replay_id",
  "context_digest_sha256",
  "trajectory_content_digest_sha256",
  "canonical_digest_sha256",
  "artifact_digest_sha256",
  "source_material_frame_id",
]);
const RESEARCHER_SPACE_RECORDED_ID_FIELDS = new Set([
  "incoming_transition_id",
  "incoming_start_frame_id",
  "incoming_successor_frame_id",
  "outgoing_transition_id",
  "outgoing_start_frame_id",
  "outgoing_successor_frame_id",
]);
/** @type {Readonly<Record<string, number>>} */
const ORACLE_EVENT_PHASE_RANK = Object.freeze({
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
const AGENT_VISUAL_EVENT_KINDS = new Set([
  "action_rejected",
  "ability_activated",
  "recipient_health_resolution",
  "agent_left_combat",
  "health_regenerated",
  "cooldown_started",
  "cooldown_ready",
  "agent_died",
  "status_aged_to_zero",
  "status_broken_by_damage",
  "status_applied",
  "status_refreshed_or_extended",
  "status_cleared_by_new_death",
  "spawn_shield_expired",
  "agent_respawned",
]);
const TEAM_DEATHMATCH_OUTCOMES = new Set(["team_a_win", "team_b_win", "draw"]);
const TEAM_DEATHMATCH_COMPLETION_BASES = new Set([
  "score_threshold",
  "horizon",
  "score_threshold_at_horizon",
]);
const STATUS_ID_BY_CHANNEL = Object.freeze([
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
const STATUS_SOURCE_CLASS_BY_CHANNEL = Object.freeze([2, 3, 4, 2, 3, 4, 4, 1, 5]);
/** @type {Readonly<Record<string, number>>} */
const AURA_SOURCE_CLASS_BY_ID = Object.freeze({
  mage_damage_amplification: 1,
  warrior_damage_mitigation: 2,
});
/** @type {Readonly<Record<number, string>>} */
const CLASS_NAME_BY_ID = Object.freeze({
  1: "Mage",
  2: "Warrior",
  3: "Hunter",
  4: "Rogue",
  5: "Priest",
});
const INCOMING_OBSERVATION_STATIC_FIELDS = Object.freeze([
  "presentation_key",
  "public_agent_id",
  "relation",
  "team_id",
  "class_id",
  "class_name",
  "radius",
  "maximum_health",
  "base_movement_speed",
  "observation_radius",
  "basic_interaction_radius",
  "ultimate_interaction_radius",
  "out_of_combat_delay_steps",
  "out_of_combat_health_regeneration_fraction_per_step",
]);
const INCOMING_STATUS_STATIC_FIELDS = Object.freeze([
  "status_channel",
  "status_id",
  "family",
  "configured_duration_steps",
  "mechanic_action_component",
  "magnitude_kind",
  "magnitude",
  "breaks_on_positive_damage",
]);
const SHARED_DYNAMIC_FIELD_ORDER = Object.freeze([
  "position",
  "life_state",
  "current_health",
  "effective_movement_speed",
  "ultimate_cooldown_remaining",
  "spawn_shield_remaining",
  "steps_until_out_of_combat",
  "statuses",
  "aura_modifiers",
]);
const ORACLE_EVENT_NONNEGATIVE_FIELDS = Object.freeze([
  "raw_damage_output",
  "source_modified_damage_output",
  "recipient_damage_modifier",
  "raw_healing_output",
  "source_modified_healing_output",
  "recipient_healing_modifier",
  "transition_start_health",
  "total_effective_damage",
  "total_effective_healing",
  "health_after_combat_resolution",
  "actual_health_regenerated",
  "attributed_death_damage",
  "score_increment",
  "previous_score",
  "successor_score",
]);
const NORMALIZED_PRESENTATION_ROOTS = new WeakSet();
const JOINED_PRESENTATION_ROOTS = new WeakSet();
// The smallest positive normal float32. A smaller positive Red Zone depth would
// underflow in Core, so no recorded strip may use one.
const FLOAT32_SMALLEST_NORMAL = 2 ** -126;

/**
 * Identify a coherent transport/presentation pair fetched from different refreshes.
 * Callers may recognize this error with isPresentationJoinRace and refetch through their
 * normal transport workflow. It is distinct from TypeError for malformed content.
 * The constructor keeps the supplied message and sets name to PresentationJoinMismatchError.
 */
export class PresentationJoinMismatchError extends Error {
  /**
   * Create a join-mismatch error with the caller's message. This does not retry a
   * request or change browser state.
   *
   * @param {string} message
   */
  constructor(message) {
    super(message);
    this.name = "PresentationJoinMismatchError";
  }
}

/**
 * Throw TypeError with message for malformed or unauthorized presentation content.
 * This helper never returns.
 *
 * @param {string} message @returns {never}
 */
function invalid(message) {
  throw new TypeError(message);
}

/**
 * Throw PresentationJoinMismatchError with message for mismatched refresh identities.
 * This helper never returns and does not perform recovery.
 *
 * @param {string} message @returns {never}
 */
function joinMismatch(message) {
  throw new PresentationJoinMismatchError(message);
}

/**
 * Return whether error is a PresentationJoinMismatchError from this module.
 * Other errors, including malformed-payload TypeError, return false.
 *
 * @param {unknown} error
 */
export function isPresentationJoinRace(error) {
  return error instanceof PresentationJoinMismatchError;
}

/**
 * Check value for JSON Schema keywords implemented by this visitor.
 * context defaults to schema; names treats property/definition keys as field names,
 * and mapping leaves discriminator mappings untouched. Recursively inspect child
 * schemas and reject unsupported keywords/discriminator structure with TypeError.
 * Return undefined on success. This guards the generated schema, not a wire payload.
 *
 * @param {unknown} value
 * @param {"schema" | "names" | "mapping"} context
 */
function assertSupportedSchema(value, context = "schema") {
  if (Array.isArray(value)) {
    for (const item of value) assertSupportedSchema(item, "schema");
    return;
  }
  if (!value || typeof value !== "object") return;
  const record = /** @type {Record<string, any>} */ (value);
  if (context === "names") {
    for (const item of Object.values(record)) assertSupportedSchema(item, "schema");
    return;
  }
  if (context === "mapping") return;
  for (const [key, item] of Object.entries(record)) {
    if (!SUPPORTED_SCHEMA_KEYWORDS.has(key)) {
      invalid(`Authorized presentation schema uses unsupported keyword ${key}.`);
    }
    if (key === "$defs" || key === "properties") {
      assertSupportedSchema(item, "names");
    } else if (key === "discriminator") {
      const discriminator = /** @type {Record<string, any>} */ (item);
      const keys = Object.keys(discriminator).sort();
      if (
        keys.length !== 2 ||
        keys[0] !== "mapping" ||
        keys[1] !== "propertyName" ||
        typeof discriminator.propertyName !== "string"
      ) {
        invalid("Authorized presentation schema discriminator is unsupported.");
      }
      assertSupportedSchema(discriminator.mapping, "mapping");
    } else if (key !== "required" && key !== "enum") {
      assertSupportedSchema(item, "schema");
    }
  }
}

if (!Object.isFrozen(AUTHORIZED_PRESENTATION_SCHEMA_V1)) {
  invalid("Authorized presentation schema must be recursively frozen.");
}
assertSupportedSchema(AUTHORIZED_PRESENTATION_SCHEMA_V1);

/**
 * Copy value's own enumerable data properties into a new null-prototype record.
 * Require a plain Object or null prototype, no symbols, and no accessors/non-enumerable
 * fields. Throw TypeError naming label otherwise. Child values remain references until
 * recursive schema validation copies them. This avoids invoking property getters; it
 * is intended for decoded JSON, not arbitrary Proxy behavior.
 *
 * @param {unknown} value
 * @param {string} label
 * @returns {Record<string, any>}
 */
function snapshotRecord(value, label) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(`${label} must be an object.`);
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(`${label} must use a plain JSON object prototype.`);
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    invalid(`${label} must not contain symbol fields.`);
  }
  /** @type {Record<string, any>} */
  const snapshot = Object.create(null);
  for (const [key, descriptor] of Object.entries(descriptors)) {
    if (!("value" in descriptor) || !descriptor.enumerable) {
      invalid(`${label}.${key} must be an enumerable JSON data field.`);
    }
    snapshot[key] = descriptor.value;
  }
  return snapshot;
}

/**
 * Copy dense plain-array value without invoking element getters.
 * Require exactly numeric element keys followed by length, Array.prototype, no extra
 * fields, and enumerable data elements. Return a fresh array of existing child values
 * or throw TypeError naming label. Recursive validation copies children separately.
 *
 * @param {unknown} value
 * @param {string} label
 * @returns {any[]}
 */
function snapshotArray(value, label) {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) {
    invalid(`${label} must be a plain JSON array.`);
  }
  const keys = Reflect.ownKeys(value);
  const expectedKeys = Array.from({ length: value.length }, (_, index) =>
    String(index),
  );
  expectedKeys.push("length");
  if (
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index])
  ) {
    invalid(`${label} must be dense and contain no extra array fields.`);
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  return Array.from({ length: value.length }, (_, index) => {
    const descriptor = descriptors[String(index)];
    if (!descriptor || !("value" in descriptor) || !descriptor.enumerable) {
      invalid(`${label}[${index}] must be an enumerable JSON data element.`);
    }
    return descriptor.value;
  });
}

/**
 * Return schema unchanged unless it has a local #/$defs reference.
 * Resolve that reference against the generated presentation schema and return its
 * definition. Throw TypeError for external or unknown references. No network is used.
 *
 * @param {Record<string, any>} schema @returns {Record<string, any>}
 */
function resolveSchema(schema) {
  if (typeof schema.$ref !== "string") {
    return schema;
  }
  const prefix = "#/$defs/";
  if (!schema.$ref.startsWith(prefix)) {
    invalid("Authorized presentation schema contains an external reference.");
  }
  const definition =
    AUTHORIZED_PRESENTATION_SCHEMA_V1.$defs[schema.$ref.slice(prefix.length)];
  if (!definition) {
    invalid("Authorized presentation schema reference is unknown.");
  }
  return definition;
}

/**
 * Validate value against inputSchema and return its normalized in-memory tree.
 * label identifies errors. Handle exact literals/enums, discriminator variants, oneOf/
 * anyOf alternatives, finite numbers, safe integers, strings, arrays, and record fields.
 * Containers are copied through strict JSON snapshots; allowed scalar values remain
 * unchanged. Throw TypeError when the schema contract fails. This is structural
 * validation only; semantic/privacy/hash checks and final freezing happen afterward.
 *
 * @param {unknown} value
 * @param {Record<string, any>} inputSchema
 * @param {string} label
 * @returns {any}
 */
function validateSchema(value, inputSchema, label) {
  const schema = resolveSchema(inputSchema);
  if (schema.discriminator && Array.isArray(schema.oneOf)) {
    const snapshot = snapshotRecord(value, label);
    const propertyName = schema.discriminator.propertyName;
    const discriminator = snapshot[propertyName];
    const reference = schema.discriminator.mapping?.[discriminator];
    if (typeof discriminator !== "string" || typeof reference !== "string") {
      invalid(`${label} has an unknown ${propertyName} discriminator.`);
    }
    return validateSchema(value, { $ref: reference }, label);
  }
  if (Array.isArray(schema.oneOf)) {
    const matches = [];
    for (const branch of schema.oneOf) {
      try {
        matches.push(validateSchema(value, branch, label));
      } catch {
        // One-of alternatives are intentionally isolated.
      }
    }
    if (matches.length !== 1) {
      invalid(`${label} must match exactly one strict variant.`);
    }
    return matches[0];
  }
  if (Array.isArray(schema.anyOf)) {
    for (const branch of schema.anyOf) {
      try {
        return validateSchema(value, branch, label);
      } catch {
        // Continue to the next closed alternative.
      }
    }
    invalid(`${label} does not match any allowed strict variant.`);
  }
  if (Object.hasOwn(schema, "const") && !Object.is(value, schema.const)) {
    invalid(`${label} must equal its exact literal.`);
  }
  if (
    Array.isArray(schema.enum) &&
    !(/** @type {any[]} */ (schema.enum).some((item) => Object.is(item, value)))
  ) {
    invalid(`${label} is outside its closed enum.`);
  }
  if (schema.type === "null") {
    if (value !== null) invalid(`${label} must be null.`);
    return null;
  }
  if (schema.type === "boolean") {
    if (typeof value !== "boolean") invalid(`${label} must be a boolean.`);
    return value;
  }
  if (schema.type === "integer") {
    if (!Number.isSafeInteger(value)) invalid(`${label} must be a safe integer.`);
    validateNumericBounds(/** @type {number} */ (value), schema, label);
    return value;
  }
  if (schema.type === "number") {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      invalid(`${label} must be a finite number.`);
    }
    validateNumericBounds(value, schema, label);
    return value;
  }
  if (schema.type === "string") {
    if (typeof value !== "string") invalid(`${label} must be a string.`);
    const codePointLength = [...value].length;
    if (schema.minLength !== undefined && codePointLength < schema.minLength) {
      invalid(`${label} is shorter than its contract.`);
    }
    if (schema.maxLength !== undefined && codePointLength > schema.maxLength) {
      invalid(`${label} is longer than its contract.`);
    }
    if (schema.pattern !== undefined && !new RegExp(schema.pattern, "u").test(value)) {
      invalid(`${label} does not match its contract pattern.`);
    }
    return value;
  }
  if (schema.type === "array") {
    const values = snapshotArray(value, label);
    if (schema.minItems !== undefined && values.length < schema.minItems) {
      invalid(`${label} has too few items.`);
    }
    if (schema.maxItems !== undefined && values.length > schema.maxItems) {
      invalid(`${label} has too many items.`);
    }
    if (Array.isArray(schema.prefixItems)) {
      if (!schema.items && values.length > schema.prefixItems.length) {
        invalid(`${label} contains undeclared tuple items.`);
      }
      return values.map((item, index) =>
        validateSchema(
          item,
          schema.prefixItems[index] ?? schema.items,
          `${label}[${index}]`,
        ),
      );
    }
    if (schema.items) {
      return values.map((item, index) =>
        validateSchema(item, schema.items, `${label}[${index}]`),
      );
    }
    return values;
  }
  if (schema.type === "object") {
    const snapshot = snapshotRecord(value, label);
    const properties = schema.properties ?? {};
    const required = schema.required ?? [];
    const actualKeys = Object.keys(snapshot);
    for (const key of required) {
      if (!Object.hasOwn(snapshot, key)) invalid(`${label}.${key} is required.`);
    }
    if (
      schema.additionalProperties === false &&
      actualKeys.some((key) => !Object.hasOwn(properties, key))
    ) {
      invalid(`${label} contains an unknown field.`);
    }
    /** @type {Record<string, any>} */
    const normalized = {};
    for (const key of actualKeys) {
      const propertySchema = properties[key];
      normalized[key] = propertySchema
        ? validateSchema(snapshot[key], propertySchema, `${label}.${key}`)
        : snapshot[key];
    }
    return normalized;
  }
  return value;
}

/**
 * Check already numeric value against schema's inclusive and exclusive bounds.
 * Return undefined on success or throw TypeError naming label. Type/finite checks are
 * owned by validateSchema before this helper runs.
 *
 * @param {number} value
 * @param {Record<string, any>} schema
 * @param {string} label
 */
function validateNumericBounds(value, schema, label) {
  if (schema.minimum !== undefined && value < schema.minimum) {
    invalid(`${label} is below its minimum.`);
  }
  if (schema.maximum !== undefined && value > schema.maximum) {
    invalid(`${label} is above its maximum.`);
  }
  if (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum) {
    invalid(`${label} is below its exclusive minimum.`);
  }
  if (schema.exclusiveMaximum !== undefined && value >= schema.exclusiveMaximum) {
    invalid(`${label} is above its exclusive maximum.`);
  }
}

/**
 * Select the unique schema branch for already validated value under inputSchema.
 * label names errors. Resolve local references/discriminators, or recheck alternatives
 * and require exactly one match. Return that concrete schema or throw TypeError.
 * The selected number/integer type preserves Python JSON spelling after JSON.parse
 * has erased the distinction between tokens such as 20 and 20.0.
 *
 * @param {unknown} value
 * @param {Record<string, any>} inputSchema
 * @param {string} label
 * @returns {Record<string, any>}
 */
function selectCanonicalSchema(value, inputSchema, label) {
  const schema = resolveSchema(inputSchema);
  if (schema.discriminator && Array.isArray(schema.oneOf)) {
    const propertyName = schema.discriminator.propertyName;
    const discriminator = /** @type {Record<string, any>} */ (value)[propertyName];
    const reference = schema.discriminator.mapping?.[discriminator];
    if (typeof reference !== "string") {
      invalid(`${label} has no canonical discriminator branch.`);
    }
    return selectCanonicalSchema(value, { $ref: reference }, label);
  }
  for (const keyword of ["oneOf", "anyOf"]) {
    if (!Array.isArray(schema[keyword])) continue;
    const matches = schema[keyword].filter((branch) => {
      try {
        validateSchema(value, branch, label);
        return true;
      } catch {
        return false;
      }
    });
    if (matches.length !== 1) {
      invalid(`${label} has no unique canonical schema branch.`);
    }
    return selectCanonicalSchema(value, matches[0], label);
  }
  return schema;
}

/**
 * Encode finite numeric value using the Python float notation expected by the wire hash.
 * Return a string preserving negative zero, a decimal point for whole floats, and Python's
 * fixed/exponent thresholds and exponent padding. Throw TypeError for a non-finite value.
 * This is canonical serialization for validated float fields, not display formatting.
 *
 * @param {number} value
 */
function canonicalPythonFloat(value) {
  if (!Number.isFinite(value)) invalid("Canonical endpoint float must be finite.");
  if (Object.is(value, -0)) return "-0.0";
  if (value === 0) return "0.0";
  const negative = value < 0;
  const text = Math.abs(value).toString().toLowerCase();
  let digits;
  let scientificExponent;
  if (text.includes("e")) {
    const [coefficient, exponentText] = text.split("e");
    const decimalIndex = coefficient.indexOf(".");
    const integerDigits = decimalIndex < 0 ? coefficient.length : decimalIndex;
    digits = coefficient.replace(".", "").replace(/^0+/u, "");
    scientificExponent = Number.parseInt(exponentText, 10) + integerDigits - 1;
  } else {
    const [integerPart, fractionPart = ""] = text.split(".");
    const combined = integerPart + fractionPart;
    const firstSignificant = combined.search(/[1-9]/u);
    if (firstSignificant < integerPart.length) {
      scientificExponent = integerPart.length - firstSignificant - 1;
    } else {
      scientificExponent = -(firstSignificant - integerPart.length + 1);
    }
    digits = combined.slice(firstSignificant);
  }
  digits = digits.replace(/0+$/u, "") || "0";
  let encoded;
  if (scientificExponent < -4 || scientificExponent >= 16) {
    const coefficient =
      digits.length === 1 ? digits : `${digits[0]}.${digits.slice(1)}`;
    const exponentSign = scientificExponent >= 0 ? "+" : "-";
    const exponent = String(Math.abs(scientificExponent)).padStart(2, "0");
    encoded = `${coefficient}e${exponentSign}${exponent}`;
  } else {
    const point = scientificExponent + 1;
    if (point <= 0) {
      encoded = `0.${"0".repeat(-point)}${digits}`;
    } else if (point >= digits.length) {
      encoded = `${digits}${"0".repeat(point - digits.length)}.0`;
    } else {
      encoded = `${digits.slice(0, point)}.${digits.slice(point)}`;
    }
  }
  return negative ? `-${encoded}` : encoded;
}

/**
 * Encode already validated value according to inputSchema into compact canonical JSON.
 * label identifies branch/type errors. Object keys are sorted; schema number fields use
 * Python float spelling while integer fields keep integer spelling. omitObjectKeys is
 * an optional set applied only to the current object, typically to omit its own digest.
 * Return a string or throw TypeError for an ambiguous/unsupported canonical type.
 * No file is written and no input is changed.
 *
 * @param {unknown} value
 * @param {Record<string, any>} inputSchema
 * @param {string} label
 * @param {ReadonlySet<string>} [omitObjectKeys]
 * @returns {string}
 */
function canonicalPythonJson(value, inputSchema, label, omitObjectKeys) {
  const schema = selectCanonicalSchema(value, inputSchema, label);
  if (value === null) return "null";
  if (schema.type === "boolean")
    return /** @type {boolean} */ (value) ? "true" : "false";
  if (schema.type === "integer") return String(value);
  if (schema.type === "number")
    return canonicalPythonFloat(/** @type {number} */ (value));
  if (schema.type === "string") return JSON.stringify(/** @type {string} */ (value));
  if (schema.type === "array") {
    const arrayValue = /** @type {unknown[]} */ (value);
    const itemSchemas = Array.isArray(schema.prefixItems)
      ? arrayValue.map((_, index) => schema.prefixItems[index] ?? schema.items)
      : arrayValue.map(() => schema.items);
    return `[${arrayValue
      .map((item, index) =>
        canonicalPythonJson(
          item,
          /** @type {Record<string, any>} */ (itemSchemas[index]),
          `${label}[${index}]`,
        ),
      )
      .join(",")}]`;
  }
  if (schema.type === "object") {
    const objectValue = /** @type {Record<string, unknown>} */ (value);
    const keys = Object.keys(objectValue)
      .filter((key) => !omitObjectKeys?.has(key))
      .sort();
    return `{${keys
      .map(
        (key) =>
          `${JSON.stringify(key)}:${canonicalPythonJson(
            objectValue[key],
            /** @type {Record<string, any>} */ (schema.properties[key]),
            `${label}.${key}`,
          )}`,
      )
      .join(",")}}`;
  }
  invalid(`${label} has no canonical JSON wire type.`);
}

/**
 * Verify frame.current_endpoint's SHA-256 against its declared canonical digest.
 * frame must already pass schema checks. Serialize with its exact generated schema,
 * omitting only the endpoint digest field, then await Web Crypto. Resolve undefined
 * on agreement; reject with TypeError for missing Web Crypto or a digest mismatch.
 * This checks content consistency, not the authenticity of the sender.
 *
 * @param {Record<string, any>} frame
 */
async function verifyAuthorizedEndpointDigest(frame) {
  const rootSchema = selectCanonicalSchema(
    frame,
    AUTHORIZED_PRESENTATION_SCHEMA_V1,
    "Authorized presentation frame",
  );
  const endpointSchema = rootSchema.properties.current_endpoint;
  const endpoint = frame.current_endpoint;
  const encoded = canonicalPythonJson(
    endpoint,
    endpointSchema,
    "Authorized current endpoint",
    new Set(["authorized_endpoint_digest_sha256"]),
  );
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) invalid("Web Crypto is required for endpoint-digest validation.");
  const digest = await subtle.digest("SHA-256", new TextEncoder().encode(encoded));
  const hex = [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  if (hex !== endpoint.authorized_endpoint_digest_sha256) {
    invalid("Authorized endpoint digest does not match its canonical content.");
  }
}

/**
 * Verify the required local corpse-overlay digest on an Agent POV frame.
 * For Oracle frames, resolve undefined without hashing. Otherwise encode the already
 * validated overlay with its exact schema and omit only its digest field; await Web
 * Crypto SHA-256 and reject with TypeError on mismatch or unavailable crypto.
 * The same-origin Python producer remains the authority root. The digest and redundant
 * public facts detect stale/spliced content; they do not grant independent authenticity.
 * Older Agent payloads missing this required field are not accepted.
 *
 * @param {Record<string, any>} frame
 */
async function verifyLocalOracleCorpseOverlayDigest(frame) {
  if (frame.authority.authority_kind !== "agent_pov") return;
  const rootSchema = selectCanonicalSchema(
    frame,
    AUTHORIZED_PRESENTATION_SCHEMA_V1,
    "Authorized presentation frame",
  );
  const overlaySchema = rootSchema.properties.local_oracle_corpse_overlay;
  const overlay = frame.local_oracle_corpse_overlay;
  const encoded = canonicalPythonJson(
    overlay,
    overlaySchema,
    "Local-Oracle corpse overlay",
    new Set(["authorized_overlay_digest_sha256"]),
  );
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) invalid("Web Crypto is required for overlay-digest validation.");
  const digest = await subtle.digest("SHA-256", new TextEncoder().encode(encoded));
  const hex = [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  if (hex !== overlay.authorized_overlay_digest_sha256) {
    invalid("Local-Oracle corpse overlay digest does not match its content.");
  }
}

/**
 * Recursively freeze value's object/array children and return value itself.
 * The caller supplies an acyclic, already copied JSON-like tree. This mutates freeze
 * state of that tree, not the original wire object retained by the caller.
 *
 * @param {unknown} value @returns {any}
 */
function deepFreeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

/**
 * Require values to contain no duplicate entries under Set equality.
 * Return undefined on success; otherwise throw TypeError naming label.
 *
 * @param {unknown[]} values @param {string} label
 */
function requireUnique(values, label) {
  if (new Set(values).size !== values.length) invalid(`${label} must be unique.`);
}

/**
 * Require selector(value, index) to equal each zero-based array index by Object.is.
 * Return undefined for canonical values order, otherwise throw TypeError naming label.
 * The selector reads the already validated category/index field.
 *
 * @param {any[]} values @param {(value: any, index: number) => unknown} selector @param {string} label
 */
function requireExactOrder(values, selector, label) {
  if (values.some((value, index) => !Object.is(selector(value, index), index))) {
    invalid(`${label} must retain canonical order.`);
  }
}

/**
 * Return recursive equality of left and right JSON-like trees.
 * Use Object.is for scalars, ordered equality for arrays, and sorted enumerable keys
 * for records. Inputs must be acyclic; nothing is changed.
 *
 * @param {unknown} left @param {unknown} right @returns {boolean}
 */
function structurallyEqual(left, right) {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((item, index) => structurallyEqual(item, right[index]))
    );
  }
  if (!left || !right || typeof left !== "object" || typeof right !== "object") {
    return false;
  }
  const leftRecord = /** @type {Record<string, any>} */ (left);
  const rightRecord = /** @type {Record<string, any>} */ (right);
  const leftKeys = Object.keys(leftRecord).sort();
  const rightKeys = Object.keys(rightRecord).sort();
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every(
      (key, index) =>
        key === rightKeys[index] &&
        structurallyEqual(leftRecord[key], rightRecord[key]),
    )
  );
}

/**
 * Return whether finite numbers left and right differ by at most absolute 1e-5 or
 * relative 1e-6, whichever is larger. Non-finite input returns false. This matches the
 * chosen Python semantic-comparison tolerances, not math.isclose's default settings.
 *
 * @param {number} left
 * @param {number} right
 */
function pythonIsClose(left, right) {
  return (
    Number.isFinite(left) &&
    Number.isFinite(right) &&
    Math.abs(left - right) <=
      Math.max(1e-5, 1e-6 * Math.max(Math.abs(left), Math.abs(right)))
  );
}

/**
 * Check already schema-valid axis for exact 9/11/2 movement/target/Ultimate categories.
 * Require canonical category indices, target-none then five allies then five opponents,
 * unique target public IDs, and unique labels within each head. Return undefined or
 * throw TypeError. No legality mask is computed here.
 *
 * @param {Record<string, any>} axis
 */
function validateActionAxis(axis) {
  const movementActions = /** @type {any[]} */ (axis.movement_actions);
  const targetActions = /** @type {any[]} */ (axis.target_actions);
  const ultimateChoices = /** @type {any[]} */ (axis.ultimate_choices);
  if (
    movementActions.length !== 9 ||
    targetActions.length !== 11 ||
    ultimateChoices.length !== 2
  ) {
    invalid("Authorized action axes must retain exact 9/11/2 shapes.");
  }
  requireExactOrder(movementActions, (row) => row.move_action, "Movement axis");
  requireExactOrder(targetActions, (row) => row.target_action, "Target axis");
  requireExactOrder(ultimateChoices, (row) => row.use_ultimate_action, "Ultimate axis");
  if (
    targetActions[0].target_kind !== "target_none" ||
    targetActions
      .slice(1, 6)
      .some(
        (row) =>
          row.target_kind !== "public_agent" || row.target_relation !== "same_team",
      ) ||
    targetActions
      .slice(6)
      .some(
        (row) =>
          row.target_kind !== "public_agent" || row.target_relation !== "opponent",
      )
  ) {
    invalid("Target axis must retain none/five allies/five opponents.");
  }
  requireUnique(
    targetActions.slice(1).map((row) => row.target_public_agent_id),
    "Target-axis public identities",
  );
  for (const rows of [movementActions, targetActions, ultimateChoices]) {
    requireUnique(
      rows.map((row) => row.display_name),
      "Action-axis display names",
    );
  }
}

/**
 * Check already schema-valid mask using either supported field-name spelling.
 * Require shapes 9, 11, 2, and (11, 2), with target/Ultimate marginals equal to any-true
 * reductions of the joint mask. Return undefined or throw TypeError naming label.
 * The schema owns boolean entry types; this helper checks shape and agreement.
 *
 * @param {Record<string, any>} mask @param {string} label
 */
function validateDecisionMask(mask, label) {
  const movement = /** @type {any[]} */ (mask.move ?? mask.movement_action_mask);
  const target = /** @type {any[]} */ (mask.select_target ?? mask.target_action_mask);
  const ultimate = /** @type {any[]} */ (
    mask.use_ultimate ?? mask.use_ultimate_action_mask
  );
  const joint = /** @type {any[][]} */ (
    mask.select_target_use_ultimate_joint ?? mask.target_use_ultimate_joint_mask
  );
  if (
    movement.length !== 9 ||
    target.length !== 11 ||
    ultimate.length !== 2 ||
    joint.length !== 11 ||
    joint.some((row) => row.length !== 2)
  ) {
    invalid(`${label} must retain exact 9/11/2/11x2 shapes.`);
  }
  const targetMarginal = joint.map((row) => row.some(Boolean));
  const ultimateMarginal = [0, 1].map((column) => joint.some((row) => row[column]));
  if (
    target.some((value, index) => value !== targetMarginal[index]) ||
    ultimate.some((value, index) => value !== ultimateMarginal[index])
  ) {
    invalid(`${label} marginals must equal its joint mask.`);
  }
}

/**
 * Check root's opaque-key/public-ID pairs and return their unique pair list.
 * options defaults to {}; authorityKind defaults to root.authority.authority_kind,
 * and excludedRootFields can skip separately owned branches only at the root level.
 * Require oracle_/pov_ SHA-256-shaped keys and a one-to-one key/public-ID relationship.
 * Throw TypeError for mismatches. Cryptographic derivation is checked separately.
 *
 * @param {Record<string, any>} root
 * @param {{authorityKind?: "oracle" | "agent_pov", excludedRootFields?: ReadonlySet<string>}} [options]
 */
function validatePresentationKeyGraph(root, options = {}) {
  const authorityKind = options.authorityKind ?? root.authority.authority_kind;
  const expectedPrefix = authorityKind === "oracle" ? "oracle_" : "pov_";
  const publicByKey = new Map();
  const keyByPublic = new Map();
  /**
   * Walk value and collect matching key/public-ID pairs in the enclosing maps.
   * rootLevel defaults to false; only the initial root visit honors excludedRootFields.
   * Throw TypeError for bad prefix/shape, nullable-pair mismatch, or a nonunique mapping.
   *
   * @param {unknown} value
   */
  function visit(value, rootLevel = false) {
    if (Array.isArray(value)) {
      for (const child of value) visit(child);
      return;
    }
    if (!value || typeof value !== "object") return;
    const record = /** @type {Record<string, any>} */ (value);
    for (const [keyField, publicField] of Object.entries(
      PRESENTATION_KEY_PUBLIC_FIELDS,
    )) {
      if (!Object.hasOwn(record, keyField)) continue;
      const key = record[keyField];
      const publicId = record[publicField];
      if ((key === null) !== (publicId === null)) {
        invalid(`Authorized ${keyField} must pair with ${publicField}.`);
      }
      if (key === null) continue;
      if (
        !key.startsWith(expectedPrefix) ||
        !/^(?:oracle|pov)_[0-9a-f]{64}$/u.test(key) ||
        typeof publicId !== "string"
      ) {
        invalid(`Authorized ${keyField} is not an opaque V1 presentation key.`);
      }
      if (
        (publicByKey.has(key) && publicByKey.get(key) !== publicId) ||
        (keyByPublic.has(publicId) && keyByPublic.get(publicId) !== key)
      ) {
        invalid("Authorized presentation key/public identity graph is inconsistent.");
      }
      publicByKey.set(key, publicId);
      keyByPublic.set(publicId, key);
    }
    for (const [name, child] of Object.entries(record)) {
      if (rootLevel && options.excludedRootFields?.has(name)) continue;
      visit(child);
    }
  }
  visit(root, true);
  return [...keyByPublic.entries()].map(([publicId, key]) => ({ key, publicId }));
}

/**
 * Await SHA-256 derivation checks for all pairs under session and authorityKind.
 * Oracle keys use the public ID; Agent POV keys also include recipient. pairs contains
 * validated {key, publicId} rows. Resolve undefined when every opaque key matches its
 * exact authority identity. Reject with TypeError for missing Web Crypto or mismatch.
 * These deterministic names are not secret credentials and do not authenticate the producer.
 *
 * @param {string} session
 * @param {"oracle" | "agent_pov"} authorityKind
 * @param {string | null} recipient
 * @param {{key: string, publicId: string}[]} pairs
 */
async function verifyPresentationKeyPairs(session, authorityKind, recipient, pairs) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) invalid("Web Crypto is required for presentation-key validation.");
  const encoder = new TextEncoder();
  const oracle = authorityKind === "oracle";
  await Promise.all(
    pairs.map(async ({ key, publicId }) => {
      const payload = oracle
        ? `oracle\0${session}\0${publicId}`
        : `agent_pov\0${session}\0${recipient}\0${publicId}`;
      const digest = await subtle.digest("SHA-256", encoder.encode(payload));
      const hex = [...new Uint8Array(digest)]
        .map((value) => value.toString(16).padStart(2, "0"))
        .join("");
      const expected = `${oracle ? "oracle" : "pov"}_${hex}`;
      if (key !== expected) {
        invalid("Presentation key does not derive from its exact authority identity.");
      }
    }),
  );
}

/**
 * Verify pairs using frame's source session and authority/recipient identity.
 * frame and pairs have already passed structural/key-graph checks. Return the promise
 * from verifyPresentationKeyPairs; rejection means unavailable crypto or a wrong key.
 *
 * @param {Record<string, any>} frame
 * @param {{key: string, publicId: string}[]} pairs
 */
async function verifyPresentationKeyDerivation(frame, pairs) {
  const oracle = frame.authority.authority_kind === "oracle";
  return verifyPresentationKeyPairs(
    frame.source.source_session_id,
    oracle ? "oracle" : "agent_pov",
    oracle ? null : frame.authority.recipient_public_agent_id,
    pairs,
  );
}

/**
 * Check already schema-valid Agent root for prohibited keys and Oracle identity values.
 * Traverse the entire tree while allowing only the explicitly named recorded-ID fields
 * inside researcher_space and the source-frame field in the local corpse overlay.
 * Return undefined or throw TypeError. This enforces declared disclosure boundaries;
 * it does not prove the producer's underlying sensor authorization independently.
 *
 * @param {Record<string, any>} root
 */
function validateAgentPrivacy(root) {
  const episodeId = root.source.episode_id;
  const recipientId = root.authority.recipient_public_agent_id;
  const escapedEpisodeId = episodeId.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
  const escapedRecipientId = recipientId.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
  const oracleIdentity = new RegExp(
    `^${escapedEpisodeId}:(?:frame:[0-9]+|transition:[0-9]+(?::event:[0-9]{4})?|replay(?:|:timeline(?::[^\\s]+)?)|(?:actor-pov|shared-obs-visual-union):${escapedRecipientId}:(?:replay|timeline))$`,
    "u",
  );
  /**
   * Walk value with field context, initially null. researcherSpace and localCorpseOverlay
   * default false and track the two separately allowed branches. Reject prohibited keys
   * or exact Oracle identity strings outside their admitted fields; return undefined.
   *
   * @param {unknown} value
   * @param {string | null} field
   * @param {boolean} researcherSpace
   * @param {boolean} localCorpseOverlay
   */
  function visit(
    value,
    field = null,
    researcherSpace = false,
    localCorpseOverlay = false,
  ) {
    if (typeof value === "string") {
      const allowedOverlaySourceFrame =
        localCorpseOverlay && field === "source_frame_id";
      if (
        oracleIdentity.test(value) &&
        !allowedOverlaySourceFrame &&
        !(researcherSpace && RESEARCHER_SPACE_RECORDED_ID_FIELDS.has(field ?? ""))
      ) {
        invalid("Agent presentation contains a forbidden Oracle/diagnostic value.");
      }
      return;
    }
    if (Array.isArray(value)) {
      for (const child of value) {
        visit(child, field, researcherSpace, localCorpseOverlay);
      }
      return;
    }
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      const childIsOverlay =
        localCorpseOverlay || key === "local_oracle_corpse_overlay";
      if (
        AGENT_FORBIDDEN_KEYS.has(key) &&
        !(childIsOverlay && key === "source_frame_id")
      ) {
        invalid(`Agent presentation contains forbidden key ${key}.`);
      }
      visit(child, key, researcherSpace || key === "researcher_space", childIsOverlay);
    }
  }
  visit(root);
}

/**
 * Reject exact private transport strings reflected into an Agent presentation.
 * transport and presentation are already structurally checked. Collect values from the
 * fixed forbidden transport fields, then compare presentation strings with explicit
 * exceptions for permitted endpoint/overlay digests and researcher recorded IDs.
 * Return undefined for Oracle or a clean Agent pair; throw TypeError for a reflection.
 * Ordinary prose containing words such as metric or processing is not rejected.
 *
 * @param {Record<string, any>} transport
 * @param {Record<string, any>} presentation
 */
function validatePairedAgentPrivacy(transport, presentation) {
  if (presentation.authority.authority_kind !== "agent_pov") return;
  const forbiddenValues = new Set();
  /**
   * Collect nonempty string values from the enclosing fixed forbidden-field set while
   * walking transport value. Arrays recurse; scalar values outside those fields are ignored.
   *
   * @param {unknown} value
   */
  function collect(value) {
    if (Array.isArray(value)) {
      for (const child of value) collect(child);
      return;
    }
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      if (
        AGENT_PAIRED_FORBIDDEN_VALUE_FIELDS.has(key) &&
        typeof child === "string" &&
        child.length > 0
      ) {
        forbiddenValues.add(child);
      }
      collect(child);
    }
  }
  collect(transport);
  /**
   * Walk presentation value and reject a string equal to a collected private transport
   * value unless its exact field is allowed. field defaults null; researcherSpace and
   * localCorpseOverlay default false and track the approved exceptions. Throw TypeError
   * on a forbidden reflection; otherwise return undefined.
   *
   * @param {unknown} value
   * @param {string | null} [field]
   * @param {boolean} [researcherSpace]
   * @param {boolean} [localCorpseOverlay]
   */
  function rejectReflections(
    value,
    field = null,
    researcherSpace = false,
    localCorpseOverlay = false,
  ) {
    if (typeof value === "string") {
      const allowedEndpointDigest =
        field === "authorized_endpoint_digest_sha256" ||
        field === "source_authorized_endpoint_digest_sha256";
      const allowedOverlayValue =
        localCorpseOverlay &&
        (field === "source_frame_id" || field === "authorized_overlay_digest_sha256");
      const allowedRecordedIdentity =
        researcherSpace && RESEARCHER_SPACE_RECORDED_ID_FIELDS.has(field ?? "");
      if (
        !allowedEndpointDigest &&
        !allowedOverlayValue &&
        !allowedRecordedIdentity &&
        forbiddenValues.has(value)
      ) {
        invalid("Agent presentation reflects a forbidden paired transport value.");
      }
      return;
    }
    if (Array.isArray(value)) {
      for (const child of value) {
        rejectReflections(child, field, researcherSpace, localCorpseOverlay);
      }
      return;
    }
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      rejectReflections(
        child,
        key,
        researcherSpace || key === "researcher_space",
        localCorpseOverlay || key === "local_oracle_corpse_overlay",
      );
    }
  }
  rejectReflections(presentation);
}

/**
 * Check action's three submitted heads as signed int32-range integers.
 * Return undefined or throw TypeError naming label and field. Out-of-domain action
 * categories remain valid evidence here; accepted actions use the stricter helper.
 *
 * @param {Record<string, any>} action @param {string} label
 */
function validateSubmittedActionTuple(action, label) {
  for (const name of ["move_action", "target_action", "use_ultimate_action"]) {
    if (
      !Number.isInteger(action[name]) ||
      action[name] < -(2 ** 31) ||
      action[name] > 2 ** 31 - 1
    ) {
      invalid(`${label}.${name} must be a signed 32-bit integer.`);
    }
  }
}

/**
 * Check action's movement/target/Ultimate heads against category counts 9/11/2.
 * Return undefined or throw TypeError naming label and field. This checks category
 * bounds, not whether the action was legal under a particular decision mask.
 *
 * @param {Record<string, any>} action @param {string} label
 */
function validateAcceptedActionTuple(action, label) {
  const domains = { move_action: 9, target_action: 11, use_ultimate_action: 2 };
  for (const [name, count] of Object.entries(domains)) {
    if (!Number.isInteger(action[name]) || action[name] < 0 || action[name] >= count) {
      invalid(`${label}.${name} lies outside its accepted action domain.`);
    }
  }
}

/**
 * Check pending against the full ordered active roster at simulatorStep.
 * draft names the selected editable actor and its armed action. Require one bounded
 * pending action per roster row and exact identity order; the selected row must equal
 * the effective draft, with an unarmed combat lane represented as target-none/no-Ultimate.
 * Return undefined or throw TypeError naming label. Does not submit or simulate actions.
 *
 * @param {Record<string, any> | null} pending
 * @param {any[]} roster
 * @param {number} simulatorStep
 * @param {Record<string, any>} draft
 * @param {string} label
 */
function validateLivePendingJointAction(pending, roster, simulatorStep, draft, label) {
  if (
    pending === null ||
    pending.current_simulator_step_count !== simulatorStep ||
    !Array.isArray(pending.action_rows) ||
    pending.action_rows.length !== roster.length
  ) {
    invalid(`${label} misses its active roster or decision epoch.`);
  }
  const rows = /** @type {any[]} */ (pending.action_rows);
  if (
    rows.some(
      (row, index) =>
        row.actor_presentation_key !== roster[index]?.presentation_key ||
        row.actor_public_agent_id !== roster[index]?.public_agent_id,
    )
  ) {
    invalid(`${label} changed active actor identity or order.`);
  }
  for (const [index, row] of rows.entries()) {
    validateAcceptedActionTuple(row.pending_action, `${label}[${index}].pending`);
  }
  const selected = rows.find(
    (row) => row.actor_public_agent_id === draft.actor_public_agent_id,
  );
  const expected = {
    move_action: draft.draft_action.move_action,
    target_action:
      draft.draft_action.armed_lane === "none" ? 0 : draft.draft_action.target_action,
    use_ultimate_action: draft.draft_action.armed_lane === "ultimate" ? 1 : 0,
  };
  if (!selected || !structurallyEqual(selected.pending_action, expected)) {
    invalid(`${label} changed the selected editable draft.`);
  }
}

/**
 * Check latest as one adjacent incoming transition under prefix and episodeId.
 * Require canonical start/successor/transition IDs, adjacent simulator ticks, nonempty
 * unique actor rows, signed submitted heads, bounded accepted heads, and eleven-entry
 * target axes with null at zero and unique public IDs thereafter. Return undefined or
 * throw TypeError. The surrounding state check joins it to the current endpoint.
 *
 * @param {Record<string, any>} latest
 * @param {string} prefix
 * @param {string} episodeId
 */
function validateLatestTransition(latest, prefix, episodeId) {
  const actionRows = /** @type {any[]} */ (latest.action_rows);
  const index = latest.incoming_transition_index;
  if (
    latest.episode_id !== episodeId ||
    latest.incoming_transition_id !== `${prefix}:transition:${index}` ||
    latest.incoming_start_frame_id !== `${prefix}:frame:${index}` ||
    latest.incoming_successor_frame_id !== `${prefix}:frame:${index + 1}` ||
    latest.incoming_successor_simulator_step_count !==
      latest.incoming_start_simulator_step_count + 1 ||
    actionRows.length === 0
  ) {
    invalid("Latest Transition does not retain one adjacent canonical epoch.");
  }
  requireUnique(
    actionRows.map((row) => row.actor_public_agent_id),
    "Latest Transition actors",
  );
  for (const row of actionRows) {
    validateSubmittedActionTuple(
      row.submitted_action,
      "Latest Transition submitted action",
    );
    validateAcceptedActionTuple(
      row.accepted_action,
      "Latest Transition accepted action",
    );
    if (
      row.target_action_recipient_public_agent_id_by_id.length !== 11 ||
      row.target_action_recipient_public_agent_id_by_id[0] !== null
    ) {
      invalid("Latest Transition target axis must retain eleven rows.");
    }
    requireUnique(
      row.target_action_recipient_public_agent_id_by_id.slice(1),
      "Latest Transition target identities",
    );
  }
}

/**
 * Check upcoming as one adjacent recorded outgoing transition under prefix and episodeId.
 * Require canonical endpoint/transition IDs, adjacent ticks, unique nonempty actor rows,
 * valid submitted/accepted head domains, and eleven-entry target identity axes. Return
 * undefined or throw TypeError. This validates stored future playback, not a prediction.
 *
 * @param {Record<string, any>} upcoming
 * @param {string} prefix
 * @param {string} episodeId
 */
function validateUpcomingTransition(upcoming, prefix, episodeId) {
  const actionRows = /** @type {any[]} */ (upcoming.action_rows);
  const index = upcoming.outgoing_transition_index;
  if (
    upcoming.episode_id !== episodeId ||
    upcoming.outgoing_transition_id !== `${prefix}:transition:${index}` ||
    upcoming.outgoing_start_frame_id !== `${prefix}:frame:${index}` ||
    upcoming.outgoing_successor_frame_id !== `${prefix}:frame:${index + 1}` ||
    upcoming.outgoing_successor_simulator_step_count !==
      upcoming.outgoing_start_simulator_step_count + 1 ||
    actionRows.length === 0
  ) {
    invalid("Upcoming Transition does not retain one adjacent canonical epoch.");
  }
  requireUnique(
    actionRows.map((row) => row.actor_public_agent_id),
    "Upcoming Transition actors",
  );
  for (const row of actionRows) {
    validateSubmittedActionTuple(
      row.submitted_action,
      "Upcoming Transition submitted action",
    );
    validateAcceptedActionTuple(
      row.accepted_action,
      "Upcoming Transition accepted action",
    );
    if (
      row.target_action_recipient_public_agent_id_by_id.length !== 11 ||
      row.target_action_recipient_public_agent_id_by_id[0] !== null
    ) {
      invalid("Upcoming Transition target axis must retain eleven rows.");
    }
    requireUnique(
      row.target_action_recipient_public_agent_id_by_id.slice(1),
      "Upcoming Transition target identities",
    );
  }
}

/**
 * Return event's present spatial anchors paired with their required transition phases.
 * Use the event_kind to distinguish transition_start, post_charge, and successor; include
 * any aura emitter arrays where that event carries them. Missing/null anchors are omitted.
 * The caller supplies a schema-valid event and validates each returned anchor later.
 *
 * @param {Record<string, any>} event
 * @returns {{anchor: Record<string, any>, phase: string}[]}
 */
function oracleEventAnchors(event) {
  /** @type {{anchor: Record<string, any>, phase: string}[]} */
  const anchors = [];
  /**
   * Append event[field] with the supplied phase when that anchor is neither null nor
   * undefined. The enclosing event is already schema-valid; return undefined.
   *
   * @param {string} field @param {string} phase
   */
  const add = (field, phase) => {
    if (event[field] !== null && event[field] !== undefined) {
      anchors.push({ anchor: event[field], phase });
    }
  };
  /**
   * Append every anchor from event[field] with phase, treating an absent/null list as
   * empty. The caller's schema check owns array shape; return undefined.
   *
   * @param {string} field @param {string} phase
   */
  const addMany = (field, phase) => {
    for (const anchor of event[field] ?? []) anchors.push({ anchor, phase });
  };
  switch (event.event_kind) {
    case "action_rejected":
      add("actor_anchor", "transition_start");
      break;
    case "ability_activated":
    case "source_healing_output":
      add("source_anchor", "transition_start");
      add("recipient_anchor", "transition_start");
      break;
    case "source_damage_output":
      add("source_anchor", "transition_start");
      add("recipient_anchor", "transition_start");
      addMany("mage_damage_aura_covering_emitters", "transition_start");
      addMany("warrior_mitigation_aura_covering_emitters", "transition_start");
      break;
    case "recipient_health_resolution":
      add("recipient_anchor", "transition_start");
      break;
    case "combat_countdown_reset":
    case "health_regenerated":
    case "cooldown_started":
    case "cooldown_ready":
      add("agent_anchor", "transition_start");
      break;
    case "agent_left_combat":
      add("agent_anchor", "successor");
      break;
    case "charge_phase_displacement":
      add("start_anchor", "transition_start");
      add("end_anchor", "post_charge");
      break;
    case "ordinary_movement_phase_displacement":
      add("start_anchor", "post_charge");
      add("end_anchor", "successor");
      break;
    case "agent_died":
      add("recipient_anchor", "successor");
      break;
    case "lethal_damage_contribution":
      add("source_anchor", "successor");
      add("recipient_anchor", "successor");
      break;
    case "status_applied":
      add("source_anchor", "successor");
      add("recipient_anchor", "successor");
      break;
    case "status_aged_to_zero":
    case "status_broken_by_damage":
    case "status_refreshed_or_extended":
    case "status_cleared_by_new_death":
      add("recipient_anchor", "successor");
      break;
    case "spawn_shield_expired":
    case "agent_respawned":
      add("agent_anchor", "successor");
      break;
  }
  return anchors;
}

/**
 * Check statuses as a unique canonical status-axis snapshot.
 * Require matching channel/ID, presentation order, positive remaining/configured ticks
 * with remaining no greater than configured, and null magnitude only for kind none.
 * Return undefined or throw TypeError naming label. Schema checks own primitive types.
 *
 * @param {Record<string, any>[]} statuses @param {string} label
 */
function validateIncomingStatuses(statuses, label) {
  const statusChannels = new Set();
  let previousPresentationRank = -1;
  for (const status of statuses) {
    const presentationRank = CANONICAL_STATUS_ORDER.indexOf(
      statusTokenIdFromCatalogId(status.status_id),
    );
    if (
      statusChannels.has(status.status_channel) ||
      STATUS_ID_BY_CHANNEL[status.status_channel] !== status.status_id ||
      presentationRank <= previousPresentationRank ||
      status.configured_duration_steps < 1 ||
      status.remaining_duration < 1 ||
      status.remaining_duration > status.configured_duration_steps ||
      (status.magnitude === null) !== (status.magnitude_kind === "none")
    ) {
      invalid(`${label} is not a canonical status-axis snapshot.`);
    }
    statusChannels.add(status.status_channel);
    previousPresentationRank = presentationRank;
  }
}

/**
 * Require modifiers to have nonnegative, non-neutral multipliers and strictly
 * increasing aura IDs. Return undefined or throw TypeError naming label. A multiplier
 * of one is omitted from this compact incoming inventory rather than stored.
 *
 * @param {Record<string, any>[]} modifiers @param {string} label
 */
function validateIncomingAuraModifiers(modifiers, label) {
  /** @type {string | null} */
  let previousId = null;
  for (const modifier of modifiers) {
    if (
      modifier.multiplier < 0 ||
      modifier.multiplier === 1 ||
      (previousId !== null && modifier.aura_id <= previousId)
    ) {
      invalid(`${label} is not a canonical unique non-neutral aura inventory.`);
    }
    previousId = modifier.aura_id;
  }
}

/**
 * Check already schema-valid observation's public class/body/health/speed/range values,
 * countdown bounds, regeneration fraction, statuses, and non-neutral aura inventory.
 * Return undefined or throw TypeError naming label. Coordinates and profile values are
 * checked locally; source/recipient identity joins belong to the containing summary.
 *
 * @param {Record<string, any>} observation @param {string} label
 */
function validateIncomingObservationLocal(observation, label) {
  if (
    CLASS_NAME_BY_ID[observation.class_id] !== observation.class_name ||
    [
      "radius",
      "current_health",
      "maximum_health",
      "base_movement_speed",
      "effective_movement_speed",
      "observation_radius",
      "basic_interaction_radius",
      "ultimate_interaction_radius",
      "out_of_combat_health_regeneration_fraction_per_step",
    ].some((field) => observation[field] < 0) ||
    observation.radius <= 0 ||
    observation.maximum_health <= 0 ||
    observation.ultimate_cooldown_remaining < 0 ||
    observation.spawn_shield_remaining < 0 ||
    observation.steps_until_out_of_combat < 0 ||
    observation.out_of_combat_delay_steps < 0 ||
    observation.current_health > observation.maximum_health ||
    observation.steps_until_out_of_combat > observation.out_of_combat_delay_steps ||
    observation.out_of_combat_health_regeneration_fraction_per_step > 1
  ) {
    invalid(`${label} contains a non-canonical Agent observation.`);
  }
  validateIncomingStatuses(observation.statuses, `${label}.statuses`);
  validateIncomingAuraModifiers(observation.aura_modifiers, `${label}.aura_modifiers`);
}

/**
 * Require the declared static observation fields to agree between start and successor.
 * For status channels present at both endpoints, also require their static mechanic
 * fields to agree. Return undefined or throw TypeError naming label. Dynamic fields
 * and appearance/disappearance are handled by the containing change variant.
 *
 * @param {Record<string, any>} start
 * @param {Record<string, any>} successor
 * @param {string} label
 */
function validateRetainedIncomingStaticProfile(start, successor, label) {
  if (
    INCOMING_OBSERVATION_STATIC_FIELDS.some(
      (field) => !structurallyEqual(start[field], successor[field]),
    )
  ) {
    invalid(`${label} changed a retained observation static profile.`);
  }
  const successorStatuses = new Map(
    /** @type {any[]} */ (successor.statuses).map((status) => [
      status.status_channel,
      status,
    ]),
  );
  for (const startStatus of start.statuses) {
    const successorStatus = successorStatuses.get(startStatus.status_channel);
    if (
      successorStatus &&
      INCOMING_STATUS_STATIC_FIELDS.some(
        (field) => !Object.is(startStatus[field], successorStatus[field]),
      )
    ) {
      invalid(`${label} changed a retained status static profile.`);
    }
  }
}

/**
 * Check sources as a nonempty, unique, ordered sensor-source list.
 * Recipient base comes before teammate sources; entries then sort by public ID. Public
 * IDs and presentation keys cannot repeat. Return undefined or throw TypeError naming
 * label. Recipient identity and source-kind joins are checked by the containing summary.
 *
 * @param {Record<string, any>[]} sources @param {string} label
 */
function validateSharedObservationSources(sources, label) {
  if (sources.length === 0) invalid(`${label} must retain an observation source.`);
  const publicIds = new Set();
  const presentationKeys = new Set();
  let previous = null;
  for (const source of sources) {
    const rank = source.source_kind === "recipient_base" ? 0 : 1;
    const sortKey = [rank, source.source_public_agent_id];
    if (
      publicIds.has(source.source_public_agent_id) ||
      presentationKeys.has(source.source_presentation_key) ||
      (previous !== null &&
        (sortKey[0] < previous[0] ||
          (sortKey[0] === previous[0] && sortKey[1] <= previous[1])))
    ) {
      invalid(`${label} is not a canonical unique observation-source inventory.`);
    }
    publicIds.add(source.source_public_agent_id);
    presentationKeys.add(source.source_presentation_key);
    previous = sortKey;
  }
}

/**
 * Check latest's recipient-local incoming cue inventory after schema validation.
 * Require one initial action outcome, canonical IDs/ordinals and family order, singleton
 * own-change cues, and unique body changes with correct endpoint presence/identity.
 * Retained observations must actually change without changing their static profiles.
 * Recipient body stays self; an episode-ended cue must be final and have a done flag.
 * Return undefined or throw TypeError. Does not read privileged events.
 *
 * @param {Record<string, any>} latest
 */
function validateNoSharedIncomingSummary(latest) {
  /** @type {Readonly<Record<string, number>>} */
  const familyRanks = Object.freeze({
    own_action_outcome: 0,
    own_position_changed: 1,
    own_health_changed: 2,
    own_status_changed: 3,
    own_cooldown_changed: 4,
    own_lifecycle_changed: 5,
    visible_body_observation_changed: 6,
    episode_ended: 7,
  });
  const singletonTypes = new Set([
    "own_action_outcome",
    "own_position_changed",
    "own_health_changed",
    "own_status_changed",
    "own_cooldown_changed",
    "own_lifecycle_changed",
    "episode_ended",
  ]);
  if (latest.cues.length === 0 || latest.cues[0].cue_type !== "own_action_outcome") {
    invalid("NoSharedObs incoming inventory must begin with one action outcome.");
  }
  const singletonCounts = new Map();
  const bodyPublicIds = new Set();
  const bodyKeys = new Set();
  let previousRank = -1;
  /** @type {any[]} */ (latest.cues).forEach((cue, index) => {
    const rank = familyRanks[cue.cue_type];
    if (
      cue.ordinal !== index ||
      cue.pov_transition_id !== latest.incoming_recipient_transition_id ||
      cue.cue_id !== `${latest.incoming_recipient_transition_id}:cue:${index}` ||
      rank < previousRank
    ) {
      invalid("NoSharedObs cue inventory is not exact and canonically ordered.");
    }
    previousRank = rank;
    if (singletonTypes.has(cue.cue_type)) {
      const count = (singletonCounts.get(cue.cue_type) ?? 0) + 1;
      singletonCounts.set(cue.cue_type, count);
      if (count > 1) invalid("NoSharedObs cue kind multiplicity is invalid.");
    }
    switch (cue.cue_type) {
      case "own_position_changed":
        if (structurallyEqual(cue.start_position, cue.successor_position)) {
          invalid("NoSharedObs position cue did not change position.");
        }
        break;
      case "own_health_changed":
        if (
          cue.start_health < 0 ||
          cue.successor_health < 0 ||
          Object.is(cue.start_health, cue.successor_health)
        ) {
          invalid("NoSharedObs health cue did not change health.");
        }
        break;
      case "own_status_changed": {
        validateIncomingStatuses(cue.start_statuses, "NoSharedObs start statuses");
        validateIncomingStatuses(
          cue.successor_statuses,
          "NoSharedObs successor statuses",
        );
        if (structurallyEqual(cue.start_statuses, cue.successor_statuses)) {
          invalid("NoSharedObs status cue did not change statuses.");
        }
        const successorByChannel = new Map(
          /** @type {any[]} */ (cue.successor_statuses).map((status) => [
            status.status_channel,
            status,
          ]),
        );
        for (const status of cue.start_statuses) {
          const successor = successorByChannel.get(status.status_channel);
          if (
            successor &&
            INCOMING_STATUS_STATIC_FIELDS.some(
              (field) => !Object.is(status[field], successor[field]),
            )
          ) {
            invalid("NoSharedObs status cue changed a retained mechanic profile.");
          }
        }
        break;
      }
      case "own_cooldown_changed":
        if (
          cue.start_remaining_ticks < 0 ||
          cue.successor_remaining_ticks < 0 ||
          cue.start_remaining_ticks === cue.successor_remaining_ticks
        ) {
          invalid("NoSharedObs cooldown cue did not change cooldown.");
        }
        break;
      case "own_lifecycle_changed":
        if (
          !cue.start_active ||
          !cue.successor_active ||
          cue.start_spawn_shield_remaining_ticks < 0 ||
          cue.successor_spawn_shield_remaining_ticks < 0 ||
          (cue.start_life_state === cue.successor_life_state &&
            cue.start_spawn_shield_remaining_ticks ===
              cue.successor_spawn_shield_remaining_ticks)
        ) {
          invalid("NoSharedObs lifecycle cue is not a changed configured lifecycle.");
        }
        break;
      case "visible_body_observation_changed": {
        if (
          bodyPublicIds.has(cue.agent_public_agent_id) ||
          bodyKeys.has(cue.agent_presentation_key)
        ) {
          invalid("NoSharedObs body cue identity is repeated.");
        }
        bodyPublicIds.add(cue.agent_public_agent_id);
        bodyKeys.add(cue.agent_presentation_key);
        const expectedPresence = /** @type {Record<string, boolean[]>} */ ({
          appearance: [false, true],
          disappearance: [true, false],
          observed_values_change: [true, true],
        })[cue.observation_change_kind];
        const observations = [cue.start_observation, cue.successor_observation];
        const actualPresence = observations.map((row) => row !== null);
        if (!structurallyEqual(actualPresence, expectedPresence)) {
          invalid("NoSharedObs body change kind contradicts endpoint presence.");
        }
        for (const observation of observations) {
          if (observation === null) continue;
          validateIncomingObservationLocal(observation, "NoSharedObs body observation");
          if (
            observation.presentation_key !== cue.agent_presentation_key ||
            observation.public_agent_id !== cue.agent_public_agent_id
          ) {
            invalid("NoSharedObs body observation does not join cue identity.");
          }
        }
        if (cue.observation_change_kind === "observed_values_change") {
          if (
            !cue.observed_payload_changed ||
            structurallyEqual(cue.start_observation, cue.successor_observation)
          ) {
            invalid("NoSharedObs retained-body cue did not change its payload.");
          }
          validateRetainedIncomingStaticProfile(
            cue.start_observation,
            cue.successor_observation,
            "NoSharedObs retained body",
          );
        }
        const isRecipientPublic =
          cue.agent_public_agent_id === latest.recipient_public_agent_id;
        const isRecipientKey =
          cue.agent_presentation_key === latest.recipient_presentation_key;
        if (isRecipientPublic !== isRecipientKey) {
          invalid("NoSharedObs recipient body identity is not bijective.");
        }
        if (
          isRecipientPublic &&
          (cue.observation_change_kind !== "observed_values_change" ||
            observations.some((row) => row?.relation !== "self"))
        ) {
          invalid(
            "NoSharedObs recipient body must remain a retained self observation.",
          );
        }
        if (
          !isRecipientPublic &&
          observations.some((row) => row !== null && row.relation === "self")
        ) {
          invalid("NoSharedObs nonrecipient body cannot claim self relation.");
        }
        break;
      }
      case "episode_ended":
        if (!cue.terminated && !cue.truncated) {
          invalid("NoSharedObs episode-ended cue requires a done flag.");
        }
        break;
    }
  });
  if ((singletonCounts.get("own_action_outcome") ?? 0) !== 1) {
    invalid("NoSharedObs incoming inventory requires one action outcome.");
  }
  if (
    (singletonCounts.get("episode_ended") ?? 0) === 1 &&
    latest.cues.at(-1).cue_type !== "episode_ended"
  ) {
    invalid("NoSharedObs episode-ended cue must be final.");
  }
}

/**
 * Check latest's shared-observation delta inventory after schema validation.
 * Require canonical IDs/ordinals, one-to-one public/key identity, consistent recipient
 * and source roles, valid observation/source snapshots, and exact changed-field lists.
 * Per-agent deltas must be contiguous and follow the permitted appearance, disappearance,
 * value-change, and provenance-change combinations. Return undefined or throw TypeError.
 * A change in sensor sources stays distinct from a change in observed body values.
 *
 * @param {Record<string, any>} latest
 */
function validateSharedIncomingSummary(latest) {
  const publicToKey = new Map();
  const keyToPublic = new Map();
  const sourceKindByPublic = new Map();
  const relationByPublic = new Map();
  /** @type {{publicId: string, kinds: string[]}[]} */
  const groups = [];
  const seenGroupPublicIds = new Set();
  /** @type {{publicId: string, kinds: string[]} | null} */
  let currentGroup = null;
  /** @type {any[]} */ (latest.deltas).forEach((delta, index) => {
    if (
      delta.ordinal !== index ||
      delta.recipient_transition_id !== latest.incoming_recipient_transition_id ||
      delta.cue_id !== `${latest.incoming_recipient_transition_id}:cue:${index}`
    ) {
      invalid("SharedObs delta inventory is not exact and ordered.");
    }
    const priorKey = publicToKey.get(delta.agent_public_agent_id);
    const priorPublic = keyToPublic.get(delta.agent_presentation_key);
    if (
      (priorKey !== undefined && priorKey !== delta.agent_presentation_key) ||
      (priorPublic !== undefined && priorPublic !== delta.agent_public_agent_id)
    ) {
      invalid("SharedObs delta identity mapping is not one-to-one.");
    }
    publicToKey.set(delta.agent_public_agent_id, delta.agent_presentation_key);
    keyToPublic.set(delta.agent_presentation_key, delta.agent_public_agent_id);
    const isRecipientPublic =
      delta.agent_public_agent_id === latest.recipient_public_agent_id;
    const isRecipientKey =
      delta.agent_presentation_key === latest.recipient_presentation_key;
    if (isRecipientPublic !== isRecipientKey) {
      invalid("SharedObs recipient delta identity is not bijective.");
    }
    /** @type {any[]} */
    let observations = [];
    /** @type {any[][]} */
    let sourceLists = [];
    switch (delta.delta_kind) {
      case "appearance":
        if (isRecipientPublic) invalid("SharedObs recipient cannot appear.");
        observations = [delta.successor_observation];
        sourceLists = [delta.successor_observation_sources];
        break;
      case "disappearance":
        if (isRecipientPublic) invalid("SharedObs recipient cannot disappear.");
        observations = [delta.start_observation];
        sourceLists = [delta.start_observation_sources];
        break;
      case "observed_values_change": {
        observations = [delta.start_observation, delta.successor_observation];
        validateRetainedIncomingStaticProfile(
          delta.start_observation,
          delta.successor_observation,
          "SharedObs retained body",
        );
        const expectedFields = SHARED_DYNAMIC_FIELD_ORDER.filter(
          (field) =>
            !structurallyEqual(
              delta.start_observation[field],
              delta.successor_observation[field],
            ),
        );
        if (
          expectedFields.length === 0 ||
          !structurallyEqual(delta.changed_dynamic_fields, expectedFields)
        ) {
          invalid("SharedObs changed dynamic field inventory is not exact.");
        }
        break;
      }
      case "observation_provenance_change":
        sourceLists = [
          delta.start_observation_sources,
          delta.successor_observation_sources,
        ];
        if (structurallyEqual(sourceLists[0], sourceLists[1])) {
          invalid("SharedObs provenance delta did not change sources.");
        }
        break;
    }
    for (const observation of observations) {
      validateIncomingObservationLocal(observation, "SharedObs incoming observation");
      if (
        observation.presentation_key !== delta.agent_presentation_key ||
        observation.public_agent_id !== delta.agent_public_agent_id ||
        (isRecipientPublic
          ? observation.relation !== "self"
          : observation.relation === "self")
      ) {
        invalid("SharedObs incoming observation does not join its delta identity.");
      }
      const priorRelation = relationByPublic.get(observation.public_agent_id);
      if (priorRelation !== undefined && priorRelation !== observation.relation) {
        invalid("SharedObs observation relation changed within one summary.");
      }
      relationByPublic.set(observation.public_agent_id, observation.relation);
    }
    for (const sources of sourceLists) {
      validateSharedObservationSources(sources, "SharedObs observation sources");
      for (const source of sources) {
        const sourcePriorKey = publicToKey.get(source.source_public_agent_id);
        const sourcePriorPublic = keyToPublic.get(source.source_presentation_key);
        const sourcePriorKind = sourceKindByPublic.get(source.source_public_agent_id);
        if (
          (sourcePriorKey !== undefined &&
            sourcePriorKey !== source.source_presentation_key) ||
          (sourcePriorPublic !== undefined &&
            sourcePriorPublic !== source.source_public_agent_id) ||
          (sourcePriorKind !== undefined && sourcePriorKind !== source.source_kind)
        ) {
          invalid("SharedObs source identity is not stable and bijective.");
        }
        publicToKey.set(source.source_public_agent_id, source.source_presentation_key);
        keyToPublic.set(source.source_presentation_key, source.source_public_agent_id);
        sourceKindByPublic.set(source.source_public_agent_id, source.source_kind);
        const sourceIsRecipientPublic =
          source.source_public_agent_id === latest.recipient_public_agent_id;
        const sourceIsRecipientKey =
          source.source_presentation_key === latest.recipient_presentation_key;
        if (
          (source.source_kind === "recipient_base" &&
            !(sourceIsRecipientPublic && sourceIsRecipientKey)) ||
          (source.source_kind !== "recipient_base" &&
            (sourceIsRecipientPublic || sourceIsRecipientKey))
        ) {
          invalid("SharedObs source kind does not join recipient identity.");
        }
      }
    }
    if (currentGroup?.publicId !== delta.agent_public_agent_id) {
      if (seenGroupPublicIds.has(delta.agent_public_agent_id)) {
        invalid("SharedObs delta identity groups are not contiguous.");
      }
      seenGroupPublicIds.add(delta.agent_public_agent_id);
      currentGroup = { publicId: delta.agent_public_agent_id, kinds: [] };
      groups.push(currentGroup);
    }
    /** @type {{publicId: string, kinds: string[]}} */ (currentGroup).kinds.push(
      delta.delta_kind,
    );
  });
  for (const [publicId, sourceKind] of sourceKindByPublic) {
    const relation = relationByPublic.get(publicId);
    if (
      relation !== undefined &&
      relation !== (sourceKind === "recipient_base" ? "self" : "ally")
    ) {
      invalid("SharedObs source kind conflicts with observed relation.");
    }
  }
  const allowedGroups = new Set([
    "appearance",
    "disappearance",
    "observed_values_change",
    "observation_provenance_change",
    "observed_values_change,observation_provenance_change",
  ]);
  if (groups.some((group) => !allowedGroups.has(group.kinds.join(",")))) {
    invalid("SharedObs per-agent delta group is not canonical.");
  }
}

/**
 * Check one recorded incoming event summary against its own ordering and anchors.
 *
 * `latest` is a schema-checked Oracle, NoSharedObs, or SharedObs summary. The
 * successor tick must be the start tick plus one. Oracle events must have canonical
 * IDs, phase order, valid status channels, coherent score/completion fields, and
 * anchors that exactly match their declared trajectories. Agent summaries are
 * checked by the matching observation-summary validator.
 *
 * Returns undefined. Throws TypeError on a broken join, count, order, or field.
 * This checks the recorded explanation; it does not run the simulator again.
 *
 * @param {Record<string, any>} latest
 */
function validateLatestEvents(latest) {
  if (
    latest.incoming_successor_simulator_step_count !==
    latest.incoming_start_simulator_step_count + 1
  ) {
    invalid("Latest Events simulator epochs must be adjacent.");
  }
  if (latest.summary_kind === "replay_incoming_inventory") {
    const trajectories = /** @type {any[]} */ (latest.agent_phase_trajectories);
    const oracleEvents = /** @type {any[]} */ (latest.events);
    if (
      latest.event_count !== latest.events.length ||
      latest.event_count !== latest.ordered_event_ids.length ||
      latest.event_count !== latest.ordered_event_kinds.length
    ) {
      invalid("Oracle Latest Events inventories must have equal lengths.");
    }
    const trajectoryByKey = new Map();
    for (const trajectory of trajectories) {
      if (
        trajectoryByKey.has(trajectory.agent_presentation_key) ||
        [...trajectoryByKey.values()].some(
          (row) => row.agent_public_agent_id === trajectory.agent_public_agent_id,
        )
      ) {
        invalid("Oracle incoming trajectories repeat an authorized identity.");
      }
      for (const phase of ["transition_start", "post_charge", "successor"]) {
        const anchor = trajectory[phase];
        if (
          anchor.phase !== phase ||
          anchor.presentation_key !== trajectory.agent_presentation_key ||
          anchor.public_agent_id !== trajectory.agent_public_agent_id
        ) {
          invalid("Oracle trajectory anchors changed identity or phase.");
        }
      }
      trajectoryByKey.set(trajectory.agent_presentation_key, trajectory);
    }
    oracleEvents.forEach((event, index) => {
      const expectedId = `${latest.incoming_transition_id}:event:${String(index).padStart(4, "0")}`;
      if (
        event.ordinal !== index ||
        event.event_id !== expectedId ||
        latest.ordered_event_ids[index] !== expectedId ||
        latest.ordered_event_kinds[index] !== event.event_kind ||
        event.phase_rank !== ORACLE_EVENT_PHASE_RANK[event.event_kind] ||
        (index > 0 && event.phase_rank < oracleEvents[index - 1].phase_rank)
      ) {
        invalid("Oracle Latest Events inventory is not exact and ordered.");
      }
      if (event.event_kind === "action_rejected") {
        validateSubmittedActionTuple(
          event.submitted_action,
          "Oracle rejection submitted action",
        );
      }
      if (
        Object.hasOwn(event, "status_channel") &&
        STATUS_ID_BY_CHANNEL[event.status_channel] !== event.status_id
      ) {
        invalid("Oracle event status channel and identity are not canonical.");
      }
      if (
        ORACLE_EVENT_NONNEGATIVE_FIELDS.some(
          (field) => Object.hasOwn(event, field) && event[field] < 0,
        )
      ) {
        invalid("Oracle event contains a negative nonnegative-domain fact.");
      }
      if (event.event_kind === "team_deathmatch_score_changed") {
        const teamAnchor = event.team_anchor;
        if (
          !Number.isInteger(event.team_index) ||
          (event.team_index !== 0 && event.team_index !== 1) ||
          !Number.isInteger(event.team_id) ||
          event.team_id !== event.team_index + 1 ||
          !Number.isInteger(event.score_increment) ||
          event.score_increment <= 0 ||
          !Number.isInteger(event.previous_score) ||
          event.previous_score < 0 ||
          !Number.isInteger(event.successor_score) ||
          event.successor_score !== event.previous_score + event.score_increment ||
          !teamAnchor ||
          typeof teamAnchor !== "object" ||
          teamAnchor.phase !== "successor" ||
          teamAnchor.team_index !== event.team_index ||
          teamAnchor.team_id !== event.team_id
        ) {
          invalid("Oracle Team Deathmatch score evidence is not an exact transition.");
        }
      }
      if (
        event.event_kind === "team_deathmatch_completed" &&
        (!TEAM_DEATHMATCH_OUTCOMES.has(event.outcome) ||
          !TEAM_DEATHMATCH_COMPLETION_BASES.has(event.completion_basis))
      ) {
        invalid("Oracle Team Deathmatch completion evidence is not canonical.");
      }
      for (const { anchor, phase } of oracleEventAnchors(event)) {
        const trajectory = trajectoryByKey.get(anchor.presentation_key);
        if (
          anchor.phase !== phase ||
          !trajectory ||
          trajectory.agent_public_agent_id !== anchor.public_agent_id ||
          !structurallyEqual(anchor, trajectory[phase])
        ) {
          invalid("Oracle event anchor does not equal its trajectory anchor.");
        }
      }
    });
    return;
  }
  const rows = /** @type {any[]} */ (latest.cues ?? latest.deltas);
  const count = latest.cue_count ?? latest.delta_count;
  if (!rows || rows.length !== count) {
    invalid("Agent Latest Events count is not exact.");
  }
  rows.forEach((row, index) => {
    if (
      row.ordinal !== index ||
      row.cue_id !== `${latest.incoming_recipient_transition_id}:cue:${index}`
    ) {
      invalid("Agent Latest Events inventory is not exact and ordered.");
    }
  });
  if (latest.summary_kind === "no_shared_obs_recipient_cues") {
    validateNoSharedIncomingSummary(latest);
  } else if (latest.summary_kind === "shared_obs_recipient_observation_deltas") {
    validateSharedIncomingSummary(latest);
  } else {
    invalid("Agent Latest Events has an unknown summary kind.");
  }
}

/**
 * Check the drawing-only events visible to one actor across an incoming step.
 *
 * `visual` is a schema-checked fog-filtered summary. Its event counts, IDs, phase
 * order, recipient identity, and start/successor trajectories must agree. Every
 * spatial event anchor must match an allowed trajectory anchor. After-step facts
 * need a visible successor. The payload must not contain hidden intermediate
 * movement/respawn events or forbidden aggregate damage/healing fields.
 *
 * Returns undefined or throws TypeError. This validates the producer's visible
 * inventory and consistency; it does not recompute visibility or combat.
 *
 * @param {Record<string, any>} visual
 */
function validateAgentVisualEvents(visual) {
  if (
    visual.summary_kind !== "agent_pov_fog_filtered_visual_events" ||
    visual.incoming_successor_simulator_step_count !==
      visual.incoming_start_simulator_step_count + 1
  ) {
    invalid("Agent visual events do not retain one adjacent local epoch.");
  }
  const trajectories = /** @type {any[]} */ (visual.agent_phase_trajectories);
  const events = /** @type {any[]} */ (visual.events);
  if (
    visual.event_count !== events.length ||
    visual.event_count !== visual.ordered_event_ids.length ||
    visual.event_count !== visual.ordered_event_kinds.length
  ) {
    invalid("Agent visual-event inventories must have equal lengths.");
  }
  const trajectoryByKey = new Map();
  for (const trajectory of trajectories) {
    if (
      trajectoryByKey.has(trajectory.agent_presentation_key) ||
      [...trajectoryByKey.values()].some(
        (row) => row.agent_public_agent_id === trajectory.agent_public_agent_id,
      ) ||
      !Number.isInteger(trajectory.agent_class_id) ||
      trajectory.agent_class_id < 1 ||
      trajectory.agent_class_id > 5 ||
      (trajectory.transition_start == null && trajectory.successor == null) ||
      Object.hasOwn(trajectory, "post_charge")
    ) {
      invalid("Agent visual trajectories repeat or over-disclose an identity.");
    }
    for (const phase of ["transition_start", "successor"]) {
      const anchor = trajectory[phase];
      if (anchor == null) continue;
      if (
        anchor.phase !== phase ||
        anchor.presentation_key !== trajectory.agent_presentation_key ||
        anchor.public_agent_id !== trajectory.agent_public_agent_id
      ) {
        invalid("Agent visual trajectory anchors changed identity or phase.");
      }
    }
    trajectoryByKey.set(trajectory.agent_presentation_key, trajectory);
  }
  const recipientTrajectory = trajectoryByKey.get(visual.recipient_presentation_key);
  if (
    !recipientTrajectory ||
    recipientTrajectory.agent_public_agent_id !== visual.recipient_public_agent_id ||
    recipientTrajectory.transition_start === null ||
    recipientTrajectory.successor === null
  ) {
    invalid("Agent visual recipient must remain authorized at both endpoints.");
  }
  const trajectoryOrderByKey = new Map(
    trajectories.map((trajectory, index) => [trajectory.agent_presentation_key, index]),
  );
  events.forEach((event, index) => {
    const expectedId =
      `${visual.incoming_recipient_transition_id}:visual-event:` +
      String(index).padStart(4, "0");
    if (
      event.ordinal !== index ||
      event.event_id !== expectedId ||
      visual.ordered_event_ids[index] !== expectedId ||
      visual.ordered_event_kinds[index] !== event.event_kind ||
      !AGENT_VISUAL_EVENT_KINDS.has(event.event_kind) ||
      event.phase_rank !== ORACLE_EVENT_PHASE_RANK[event.event_kind] ||
      (index > 0 && event.phase_rank < events[index - 1].phase_rank) ||
      [
        "charge_phase_displacement",
        "ordinary_movement_phase_displacement",
        "respawn_wave_occurred",
      ].includes(event.event_kind)
    ) {
      invalid("Agent visual-event inventory is not exact and ordered.");
    }
    if (event.event_kind === "action_rejected") {
      validateSubmittedActionTuple(
        event.submitted_action,
        "Agent visual rejection submitted action",
      );
      if (
        !event.actor_configured_active ||
        event.actor_identity?.identity_kind !== "authorized_agent" ||
        !event.actor_anchor
      ) {
        invalid("Agent visual rejection does not retain an active actor anchor.");
      }
      if (
        event.actor_identity.presentation_key !== event.actor_anchor.presentation_key ||
        event.actor_identity.public_agent_id !== event.actor_anchor.public_agent_id
      ) {
        invalid("Agent visual rejection actor does not join its start anchor.");
      }
    }
    if (
      event.event_kind === "recipient_health_resolution" &&
      ["total_effective_damage", "total_effective_healing"].some((field) =>
        Object.hasOwn(event, field),
      )
    ) {
      invalid("Agent visual health resolution contains forbidden aggregate totals.");
    }
    if (
      event.event_kind === "recipient_health_resolution" &&
      !pythonIsClose(
        event.health_after_combat_resolution - event.transition_start_health,
        event.realized_net_health_change,
      )
    ) {
      invalid("Agent visual health result does not equal its visible net change.");
    }
    if (
      Object.hasOwn(event, "status_channel") &&
      STATUS_ID_BY_CHANNEL[event.status_channel] !== event.status_id
    ) {
      invalid("Agent visual status channel and identity are not canonical.");
    }
    if (
      ORACLE_EVENT_NONNEGATIVE_FIELDS.some(
        (field) => Object.hasOwn(event, field) && event[field] < 0,
      )
    ) {
      invalid("Agent visual event contains a negative nonnegative-domain fact.");
    }
    for (const { anchor, phase } of oracleEventAnchors(event)) {
      const trajectory = trajectoryByKey.get(anchor.presentation_key);
      const trajectoryAnchor = trajectory?.[phase];
      if (
        anchor.phase !== phase ||
        !trajectory ||
        !trajectoryAnchor ||
        trajectory.agent_public_agent_id !== anchor.public_agent_id ||
        !structurallyEqual(anchor, trajectoryAnchor)
      ) {
        invalid("Agent visual event anchor does not equal its trajectory anchor.");
      }
    }
    const successorDerivedAnchor =
      event.event_kind === "recipient_health_resolution"
        ? event.recipient_anchor
        : ["health_regenerated", "cooldown_started", "cooldown_ready"].includes(
              event.event_kind,
            )
          ? event.agent_anchor
          : null;
    if (
      successorDerivedAnchor !== null &&
      trajectoryByKey.get(successorDerivedAnchor.presentation_key)?.successor == null
    ) {
      invalid("Agent visual successor-derived event has no authorized successor.");
    }
    if (event.event_kind === "source_damage_output") {
      for (const emitters of /** @type {any[][]} */ ([
        event.mage_damage_aura_covering_emitters,
        event.warrior_mitigation_aura_covering_emitters,
      ])) {
        const order = /** @type {number[]} */ (
          emitters.map((anchor) => trajectoryOrderByKey.get(anchor.presentation_key))
        );
        if (
          order.some(
            (value, itemIndex) => itemIndex > 0 && value <= order[itemIndex - 1],
          )
        ) {
          invalid("Agent visual aura emitters do not use authorized scene order.");
        }
      }
    }
  });
}

/**
 * Select the actor-body fields used to compare an incoming observation.
 *
 * `agent` is a validated scene body. Returns a new mutable record containing its
 * identity, position, health, speed, status, and aura observation fields. Status
 * records are copied and their mechanic-action field is renamed for the incoming
 * schema. Position and aura arrays remain borrowed references.
 *
 * The caller compares this value; this helper does not authorize, freeze, or
 * validate it and does not change `agent`.
 *
 * @param {Record<string, any>} agent
 */
function projectAgentIncomingObservation(agent) {
  return {
    presentation_key: agent.presentation_key,
    public_agent_id: agent.public_agent_id,
    relation: agent.relation,
    team_id: agent.team_id,
    class_id: agent.class_id,
    class_name: agent.class_name,
    position: agent.position,
    radius: agent.radius,
    life_state: agent.life_state,
    current_health: agent.current_health,
    maximum_health: agent.maximum_health,
    base_movement_speed: agent.base_movement_speed,
    effective_movement_speed: agent.effective_movement_speed,
    observation_radius: agent.observation_radius,
    basic_interaction_radius: agent.basic_interaction_radius,
    ultimate_interaction_radius: agent.ultimate_interaction_radius,
    ultimate_cooldown_remaining: agent.ultimate_cooldown_remaining,
    spawn_shield_remaining: agent.spawn_shield_remaining,
    steps_until_out_of_combat: agent.steps_until_out_of_combat,
    out_of_combat_delay_steps: agent.out_of_combat_delay_steps,
    out_of_combat_health_regeneration_fraction_per_step:
      agent.out_of_combat_health_regeneration_fraction_per_step,
    statuses: /** @type {any[]} */ (agent.statuses).map((status) => ({
      status_channel: status.status_channel,
      status_id: status.status_id,
      family: status.family,
      configured_duration_steps: status.configured_duration_steps,
      remaining_duration: status.remaining_duration,
      mechanic_action_component: status.source_action_component,
      magnitude_kind: status.magnitude_kind,
      magnitude: status.magnitude,
      breaks_on_positive_damage: status.breaks_on_positive_damage,
    })),
    aura_modifiers: agent.aura_modifiers,
  };
}

/**
 * Compare a recorded number with the catalog number or its float32 rounding.
 *
 * `recorded` and `catalog` are already checked numeric fields. Returns true for
 * exact equality or equality after rounding `catalog` with Math.fround. This
 * allows Python/JAX float32 storage without accepting an arbitrary tolerance.
 * It does not check finiteness or convert strings.
 *
 * @param {number} recorded @param {number} catalog
 */
function joinsCatalogFloat(recorded, catalog) {
  return recorded === catalog || recorded === Math.fround(catalog);
}

/**
 * Check that a map's recorded Red Zone strips are exactly the strips Core scores with.
 *
 * `width` is the schema-checked map width in world units; only its float32 value
 * `w = Math.fround(width)` is used. `redZone` is the schema-checked
 * `{depth, team_a_x_range, team_b_x_range}` record of an AuthorizedMapV2. The stored
 * depth must be finite, exactly a float32 value (`Math.fround(depth) === depth`), at
 * least the smallest normal float32 (2 ** -126) and at most `w`. The only permitted
 * inclusive `[x_min, x_max]` strips are the left strip `[0, depth]` and the right
 * strip `[Math.fround(w - depth), w]`; each team's range must equal one of them
 * element for element. This is the same rule as Python's AuthorizedMapV2:
 * `Math.fround(w - depth)` rounds one subtraction of two float32 values through a
 * double, which equals Core's float32 `width - depth`. Both teams on one side is
 * legal, and a collapsed range (`x_min === x_max`) is legal. Clipping for display
 * belongs to the renderer and never changes this record.
 *
 * Returns undefined or throws TypeError naming the broken rule. It does not decide
 * which side a team spawns on; the trusted Python producer records that.
 *
 * @param {number} width @param {Record<string, any>} redZone
 */
function validateAuthorizedRedZone(width, redZone) {
  const w = Math.fround(width);
  const depth = redZone.depth;
  if (
    typeof depth !== "number" ||
    !Number.isFinite(depth) ||
    Math.fround(depth) !== depth ||
    depth < FLOAT32_SMALLEST_NORMAL ||
    depth > w
  ) {
    invalid(
      "Scene Red Zone depth must be a normal float32 value within the map width.",
    );
  }
  const permitted = [
    [0, depth],
    [Math.fround(w - depth), w],
  ];
  for (const [label, range] of [
    ["Team A", redZone.team_a_x_range],
    ["Team B", redZone.team_b_x_range],
  ]) {
    if (
      !Array.isArray(range) ||
      range.length !== 2 ||
      !permitted.some(([low, high]) => range[0] === low && range[1] === high)
    ) {
      invalid(`Scene ${label} Red Zone range must be one exact scoring strip.`);
    }
  }
}

/**
 * Check the class mechanics needed by a scene and index their declarations.
 *
 * `classMechanics` contains schema-checked profiles in the exact order given by
 * `representedClassIds`. `scope` names this part of the payload in errors. The
 * profiles must use one supported mechanics version, agree on version-2
 * documentation, contain valid bounds, and declare the exact status/aura inventory
 * for those classes. A missing mechanics version means version 1.
 *
 * Returns a mutable record with `mechanicsById` and `statusMechanicsByChannel`
 * Maps whose values borrow input records. Throws TypeError on invalid profiles.
 * This checks the supplied mechanics contract; it does not import simulator rules
 * or independently prove every numeric value against the current simulator.
 *
 * @param {any[]} classMechanics
 * @param {number[]} representedClassIds
 * @param {string} scope
 */
function validateAuthorizedClassMechanics(classMechanics, representedClassIds, scope) {
  const mechanicsIds = classMechanics.map((mechanics) => mechanics.class_id);
  if (!structurallyEqual(mechanicsIds, representedClassIds)) {
    invalid(`${scope} class mechanics must exactly equal represented class order.`);
  }
  const mechanicsVersions = new Set(
    classMechanics.map((mechanics) => (mechanics.mechanics_version === 2 ? 2 : 1)),
  );
  if (mechanicsVersions.size > 1) {
    invalid(`${scope} class mechanics must be entirely V1 or entirely V2.`);
  }
  if (
    mechanicsVersions.has(2) &&
    classMechanics.some(
      (mechanics) =>
        !structurallyEqual(
          mechanics.documentation_profile,
          classMechanics[0].documentation_profile,
        ),
    )
  ) {
    invalid(`${scope} V2 class mechanics must share one documentation profile.`);
  }

  const mechanicsById = new Map();
  const statusMechanicsByChannel = new Map();
  const projectedStatusChannels = [];
  const projectedAuraIds = [];
  for (const mechanics of classMechanics) {
    if (
      CLASS_NAME_BY_ID[mechanics.class_id] !== mechanics.class_name ||
      [
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
      ].some((field) => mechanics[field] < 0) ||
      mechanics.maximum_health <= 0 ||
      mechanics.body_radius <= 0 ||
      mechanics.ultimate_cooldown_steps < 0 ||
      mechanics.out_of_combat_delay_steps < 0 ||
      mechanics.out_of_combat_health_regeneration_fraction_per_step > 1
    ) {
      invalid(`${scope} class mechanics changed canonical identity or bounds.`);
    }
    mechanicsById.set(mechanics.class_id, mechanics);
    let previousStatusChannel = -1;
    for (const status of mechanics.status_mechanics) {
      if (
        status.status_channel <= previousStatusChannel ||
        STATUS_ID_BY_CHANNEL[status.status_channel] !== status.status_id ||
        STATUS_SOURCE_CLASS_BY_CHANNEL[status.status_channel] !== mechanics.class_id ||
        status.duration_steps < 1 ||
        (status.magnitude === null) !== (status.magnitude_kind === "none")
      ) {
        invalid(
          `${scope} class status mechanics changed the canonical V1 status axis.`,
        );
      }
      previousStatusChannel = status.status_channel;
      projectedStatusChannels.push(status.status_channel);
      statusMechanicsByChannel.set(status.status_channel, status);
    }
    const auraIds = new Set();
    for (const aura of mechanics.aura_mechanics) {
      if (
        auraIds.has(aura.aura_id) ||
        AURA_SOURCE_CLASS_BY_ID[aura.aura_id] !== mechanics.class_id ||
        aura.radius < 0 ||
        aura.per_emitter_multiplier < 0 ||
        aura.clamp_value < 0
      ) {
        invalid(`${scope} class aura mechanics changed the canonical V1 aura axis.`);
      }
      auraIds.add(aura.aura_id);
      projectedAuraIds.push(aura.aura_id);
    }
  }
  /** @type {number[]} */
  const expectedStatusChannels = [];
  for (const classId of representedClassIds) {
    STATUS_SOURCE_CLASS_BY_CHANNEL.forEach((sourceClassId, channel) => {
      if (sourceClassId === classId) expectedStatusChannels.push(channel);
    });
  }
  const expectedAuraIds = Object.entries(AURA_SOURCE_CLASS_BY_ID)
    .filter(([, sourceClassId]) => representedClassIds.includes(sourceClassId))
    .map(([auraId]) => auraId);
  if (
    !structurallyEqual(projectedStatusChannels, expectedStatusChannels) ||
    !structurallyEqual(projectedAuraIds, expectedAuraIds)
  ) {
    invalid(`${scope} mechanics inventories do not equal represented catalog axes.`);
  }
  return { mechanicsById, statusMechanicsByChannel };
}

/**
 * Check one scene body's status and aura references against available facts.
 *
 * `agent` is a schema-checked body, `agentsByKey` maps visible keys to bodies, and
 * `statusMechanicsByChannel` maps declared channels to class mechanics. `scope`
 * is included in TypeError messages. Status IDs, duration bounds, source class,
 * and any available mechanics declaration must agree; direct sources must refer
 * to matching bodies. Aura IDs must be known, unique, and have valid nonneutral
 * multipliers.
 *
 * Returns undefined. It reads the records without changing them and does not
 * infer an omitted source or recompute an effect.
 *
 * @param {Record<string, any>} agent
 * @param {Map<string, Record<string, any>>} agentsByKey
 * @param {Map<number, Record<string, any>>} statusMechanicsByChannel
 * @param {string} scope
 */
function validateAuthorizedAgentEffects(
  agent,
  agentsByKey,
  statusMechanicsByChannel,
  scope,
) {
  const statusChannels = new Set();
  for (const status of agent.statuses) {
    if (
      statusChannels.has(status.status_channel) ||
      STATUS_ID_BY_CHANNEL[status.status_channel] !== status.status_id ||
      status.configured_duration_steps < 1 ||
      status.remaining_duration < 1 ||
      status.remaining_duration > status.configured_duration_steps ||
      STATUS_SOURCE_CLASS_BY_CHANNEL[status.status_channel] !==
        status.source_class_id ||
      CLASS_NAME_BY_ID[status.source_class_id] !== status.source_class_name ||
      (status.magnitude === null) !== (status.magnitude_kind === "none")
    ) {
      invalid(`${scope} durable status changed its canonical identity or bounds.`);
    }
    statusChannels.add(status.status_channel);
    const statusMechanic = statusMechanicsByChannel.get(status.status_channel);
    if (
      statusMechanic &&
      (statusMechanic.status_id !== status.status_id ||
        statusMechanic.duration_steps !== status.configured_duration_steps ||
        statusMechanic.family !== status.family ||
        statusMechanic.source_action_component !== status.source_action_component ||
        statusMechanic.magnitude_kind !== status.magnitude_kind ||
        (status.magnitude === null || statusMechanic.magnitude === null
          ? status.magnitude !== statusMechanic.magnitude
          : !joinsCatalogFloat(status.magnitude, statusMechanic.magnitude)) ||
        statusMechanic.breaks_on_positive_damage !== status.breaks_on_positive_damage)
    ) {
      invalid(`${scope} durable status does not join its catalog mechanic.`);
    }
    const directSourceKeys = new Set();
    for (const source of status.direct_sources) {
      const sourceAgent = agentsByKey.get(source.source_presentation_key);
      if (
        directSourceKeys.has(source.source_presentation_key) ||
        !sourceAgent ||
        sourceAgent.public_agent_id !== source.source_public_agent_id ||
        sourceAgent.class_id !== status.source_class_id ||
        sourceAgent.class_name !== status.source_class_name
      ) {
        invalid(
          `${scope} status source does not join an authorized source-class agent.`,
        );
      }
      directSourceKeys.add(source.source_presentation_key);
    }
  }
  const auraIds = new Set();
  for (const modifier of agent.aura_modifiers) {
    if (
      auraIds.has(modifier.aura_id) ||
      !Object.hasOwn(AURA_SOURCE_CLASS_BY_ID, modifier.aura_id) ||
      modifier.multiplier < 0 ||
      modifier.multiplier === 1
    ) {
      invalid(`${scope} aura modifiers are not canonical and unique.`);
    }
    auraIds.add(modifier.aura_id);
  }
}

/**
 * Check the public body facts of a researcher-wide roster.
 *
 * `roster` contains schema-checked active-agent records. `classMechanics` supplies
 * exactly the represented classes; `scope` labels TypeError messages. This checks
 * class names, health/speed bounds, counters, cooldown limits, float32-compatible
 * static values, and status/aura references against the supplied declarations.
 *
 * Returns undefined. It does not validate world positions, compute effects, or
 * turn these researcher-wide facts into an actor's policy input.
 *
 * @param {any[]} roster
 * @param {any[]} classMechanics
 * @param {string} scope
 */
function validateResearcherRosterFacts(roster, classMechanics, scope) {
  const representedClassIds = [...new Set(roster.map((agent) => agent.class_id))].sort(
    (left, right) => left - right,
  );
  const { mechanicsById, statusMechanicsByChannel } = validateAuthorizedClassMechanics(
    classMechanics,
    representedClassIds,
    scope,
  );
  const agentsByKey = new Map(roster.map((agent) => [agent.presentation_key, agent]));
  for (const agent of roster) {
    const mechanics = mechanicsById.get(agent.class_id);
    if (
      CLASS_NAME_BY_ID[agent.class_id] !== agent.class_name ||
      !mechanics ||
      mechanics.class_name !== agent.class_name ||
      ["current_health", "maximum_health", "effective_movement_speed"].some(
        (field) => agent[field] < 0,
      ) ||
      agent.maximum_health <= 0 ||
      agent.ultimate_cooldown_remaining < 0 ||
      agent.spawn_shield_remaining < 0 ||
      agent.steps_until_out_of_combat < 0 ||
      agent.out_of_combat_delay_steps < 0 ||
      agent.current_health > agent.maximum_health ||
      agent.steps_until_out_of_combat > agent.out_of_combat_delay_steps ||
      agent.ultimate_cooldown_remaining > mechanics.ultimate_cooldown_steps ||
      !joinsCatalogFloat(agent.maximum_health, mechanics.maximum_health) ||
      agent.out_of_combat_delay_steps !== mechanics.out_of_combat_delay_steps
    ) {
      invalid(`${scope} roster actor changed canonical identity or bounds.`);
    }
    validateAuthorizedAgentEffects(
      agent,
      agentsByKey,
      statusMechanicsByChannel,
      `${scope} roster actor`,
    );
  }
}

/**
 * Check the internal joins of a schema-checked scene.
 *
 * `scene` supplies positive map dimensions, represented class profiles, bodies,
 * auras, spawn pads, shield settings, and two ordered team wave clocks. This
 * checks body identities and bounds, class/effect joins, aura source anchors,
 * spawn assignments, and timer limits. Oracle-only static facts must also agree
 * with their declared profiles. A version-2 map with a recorded Red Zone must hold
 * the exact scoring strips (validateAuthorizedRedZone); `red_zone: null` means the
 * rule was recorded with depth 0.
 *
 * Returns undefined or throws TypeError. The trusted producer decides which
 * bodies may appear; this helper does not recompute geometry, visibility, or the
 * simulation that produced the scene.
 *
 * @param {Record<string, any>} scene
 */
function validateAuthorizedScene(scene) {
  const agents = /** @type {any[]} */ (scene.agents);
  const classMechanics = /** @type {any[]} */ (scene.class_mechanics);
  const auraFields = /** @type {any[]} */ (scene.aura_fields);
  const spawnPads = /** @type {any[]} */ (scene.spawn_pads);
  const respawnWaves = /** @type {any[]} */ (scene.respawn_waves);
  if (scene.map.width <= 0 || scene.map.height <= 0) {
    invalid("Scene map dimensions must be positive.");
  }
  if (scene.map.map_version === 2 && scene.map.red_zone !== null) {
    validateAuthorizedRedZone(scene.map.width, scene.map.red_zone);
  }
  const agentsByKey = new Map(agents.map((agent) => [agent.presentation_key, agent]));
  const representedClassIds = [...new Set(agents.map((agent) => agent.class_id))].sort(
    (left, right) => left - right,
  );
  const { mechanicsById, statusMechanicsByChannel } = validateAuthorizedClassMechanics(
    classMechanics,
    representedClassIds,
    "Scene",
  );

  for (const agent of agents) {
    const mechanics = mechanicsById.get(agent.class_id);
    if (
      CLASS_NAME_BY_ID[agent.class_id] !== agent.class_name ||
      !mechanics ||
      mechanics.class_name !== agent.class_name ||
      [
        "radius",
        "current_health",
        "maximum_health",
        "base_movement_speed",
        "effective_movement_speed",
        "observation_radius",
        "basic_interaction_radius",
        "ultimate_interaction_radius",
        "out_of_combat_health_regeneration_fraction_per_step",
      ].some((field) => agent[field] < 0) ||
      agent.radius <= 0 ||
      agent.maximum_health <= 0 ||
      agent.ultimate_cooldown_remaining < 0 ||
      agent.spawn_shield_remaining < 0 ||
      agent.steps_until_out_of_combat < 0 ||
      agent.out_of_combat_delay_steps < 0 ||
      agent.current_health > agent.maximum_health ||
      agent.steps_until_out_of_combat > agent.out_of_combat_delay_steps ||
      agent.out_of_combat_health_regeneration_fraction_per_step > 1 ||
      agent.ultimate_cooldown_remaining > mechanics.ultimate_cooldown_steps
    ) {
      invalid("Scene agent does not join its class mechanics and local bounds.");
    }
    if (
      agent.relation === "oracle" &&
      (!joinsCatalogFloat(agent.maximum_health, mechanics.maximum_health) ||
        !joinsCatalogFloat(agent.radius, mechanics.body_radius) ||
        !joinsCatalogFloat(agent.base_movement_speed, mechanics.base_movement_speed) ||
        !joinsCatalogFloat(agent.observation_radius, mechanics.observation_radius) ||
        !joinsCatalogFloat(
          agent.basic_interaction_radius,
          mechanics.basic_interaction_radius,
        ) ||
        !joinsCatalogFloat(
          agent.ultimate_interaction_radius,
          mechanics.ultimate_interaction_radius,
        ) ||
        agent.out_of_combat_delay_steps !== mechanics.out_of_combat_delay_steps ||
        !joinsCatalogFloat(
          agent.out_of_combat_health_regeneration_fraction_per_step,
          mechanics.out_of_combat_health_regeneration_fraction_per_step,
        ))
    ) {
      invalid("Oracle agent static facts do not join public class mechanics.");
    }
    validateAuthorizedAgentEffects(
      agent,
      agentsByKey,
      statusMechanicsByChannel,
      "Scene agent",
    );
  }

  for (const field of auraFields) {
    const source = agentsByKey.get(field.source_presentation_key);
    if (
      !source ||
      source.public_agent_id !== field.source_public_agent_id ||
      source.class_id !== field.source_class_id ||
      source.class_name !== field.source_class_name ||
      !structurallyEqual(source.position, field.center) ||
      (source.life_state === "alive") !== field.source_alive
    ) {
      invalid("Scene aura field does not join its authorized source agent.");
    }
    const sourceMechanics = mechanicsById.get(field.source_class_id);
    const matchingMechanics = sourceMechanics
      ? /** @type {any[]} */ (sourceMechanics.aura_mechanics).filter(
          (row) => row.aura_id === field.aura_id,
        )
      : undefined;
    if (
      matchingMechanics?.length !== 1 ||
      field.radius <= 0 ||
      field.per_emitter_multiplier < 0 ||
      field.clamp_value < 0 ||
      !joinsCatalogFloat(field.radius, matchingMechanics[0].radius) ||
      !joinsCatalogFloat(
        field.per_emitter_multiplier,
        matchingMechanics[0].per_emitter_multiplier,
      ) ||
      field.stacking_rule !== matchingMechanics[0].stacking_rule ||
      field.clamp_kind !== matchingMechanics[0].clamp_kind ||
      field.clamp_value !== matchingMechanics[0].clamp_value
    ) {
      invalid("Scene aura field does not equal its one catalog mechanic.");
    }
  }

  let previousPadKey = null;
  for (const pad of spawnPads) {
    const padKey = [pad.team_id, pad.team_local_slot];
    if (
      ![1, 2].includes(pad.team_id) ||
      pad.team_local_slot < 0 ||
      pad.team_local_slot >= 5 ||
      pad.spawn_shield_remaining < 0 ||
      (previousPadKey !== null &&
        (padKey[0] < previousPadKey[0] ||
          (padKey[0] === previousPadKey[0] && padKey[1] <= previousPadKey[1]))) ||
      (pad.assigned_presentation_key === null) !==
        (pad.assigned_public_agent_id === null) ||
      (pad.currently_alive && !pad.configured_active) ||
      (!pad.configured_active &&
        (pad.assigned_presentation_key !== null || pad.spawn_shield_remaining !== 0))
    ) {
      invalid("Scene spawn pads are not canonical ordered lifecycle rows.");
    }
    previousPadKey = padKey;
    if (pad.assigned_presentation_key !== null) {
      const assigned = agentsByKey.get(pad.assigned_presentation_key);
      if (
        !assigned ||
        assigned.public_agent_id !== pad.assigned_public_agent_id ||
        assigned.team_id !== pad.team_id ||
        pad.currently_alive !== (assigned.life_state === "alive") ||
        pad.spawn_shield_remaining !== assigned.spawn_shield_remaining
      ) {
        invalid("Scene spawn pad does not join its authorized assignee.");
      }
    }
  }
  if (
    scene.spawn_shield_mechanics.availability_kind === "available" ||
    scene.spawn_shield_mechanics.availability_kind === "available_v2"
  ) {
    const duration = scene.spawn_shield_mechanics.configured_duration_steps;
    if (
      duration < 0 ||
      scene.spawn_shield_mechanics.movement_speed <= 0 ||
      agents.some((agent) => agent.spawn_shield_remaining > duration) ||
      spawnPads.some((pad) => pad.spawn_shield_remaining > duration)
    ) {
      invalid("Scene spawn-shield remaining duration exceeds configuration.");
    }
  }
  if (
    respawnWaves.length !== 2 ||
    respawnWaves.some(
      (wave, index) =>
        wave.team_index !== index ||
        wave.team_id !== index + 1 ||
        wave.period_steps < 1 ||
        wave.countdown_steps < 0 ||
        wave.countdown_steps >= wave.period_steps,
    )
  ) {
    invalid("Scene respawn waves do not retain the ordered two-team lifecycle.");
  }
}

/**
 * Check that observed bodies use the recipient's declared actor-relative axis.
 *
 * `observation` is a schema-checked incoming body observation. `endpoint` is the
 * matching authorized actor endpoint, including its recipient and action-target
 * axis. Each body ID must be on that axis with the correct self/ally/opponent
 * relation and team. Returns undefined or throws TypeError.
 *
 * This checks identity and order semantics without making unseen axis entries
 * visible or changing the observation.
 *
 * @param {Record<string, any>} observation
 * @param {Record<string, any>} endpoint
 */
function validateAgentIncomingObservationAxis(observation, endpoint) {
  const parts = endpoint.parts;
  const recipient = parts.recipient_public_agent_id;
  const self = /** @type {any[]} */ (parts.scene.agents).find(
    (row) => row.public_agent_id === recipient,
  );
  const targetIds = /** @type {any[]} */ (endpoint.action_axis.target_actions)
    .slice(1)
    .map((row) => row.target_public_agent_id);
  const index = targetIds.indexOf(observation.public_agent_id);
  if (!self || index < 0) {
    invalid("Agent incoming observation identity lies outside its action axis.");
  }
  const expectedRelation =
    observation.public_agent_id === recipient
      ? "self"
      : index < 5
        ? "ally"
        : "opponent";
  const expectedTeam = index < 5 ? self.team_id : self.team_id === 1 ? 2 : 1;
  if (
    observation.relation !== expectedRelation ||
    observation.team_id !== expectedTeam
  ) {
    invalid("Agent incoming observation relation/team does not join its axis.");
  }
}

/**
 * Join the incoming explanation to the current endpoint and action records.
 *
 * `frame`, `source`, and `endpoint` are already schema-checked members of one
 * presentation. `oracle` and `shared` select its authority branch. Frame zero has
 * no incoming step, so this helper returns immediately; the caller checks its
 * required absence. Later frames must use the preceding transition's canonical
 * IDs and adjacent ticks.
 *
 * Oracle action rows, trajectories, target axes, and rejection events must agree.
 * Actor summaries must join their own action row, observations, visible successor
 * bodies, and SharedObs provenance. Returns undefined or throws TypeError. It
 * checks cross-field consistency without replaying the transition.
 *
 * @param {Record<string, any>} frame
 * @param {Record<string, any>} source
 * @param {Record<string, any>} endpoint
 * @param {boolean} oracle
 * @param {boolean} shared
 */
function validateIncomingStateMatrix(frame, source, endpoint, oracle, shared) {
  const index = source.source_frame_index;
  if (index === 0) return;
  const events = frame.latest_events;
  const transition = frame.latest_transition;
  if (oracle) {
    const transitionId = `${source.episode_id}:transition:${index - 1}`;
    const startId = `${source.episode_id}:frame:${index - 1}`;
    if (
      events.incoming_transition_index !== index - 1 ||
      events.incoming_transition_id !== transitionId ||
      events.incoming_start_frame_id !== startId ||
      events.incoming_successor_frame_id !== source.source_frame_id ||
      events.incoming_successor_simulator_step_count !==
        source.source_simulator_step_count ||
      transition.incoming_transition_index !== index - 1 ||
      transition.incoming_transition_id !== transitionId ||
      transition.incoming_start_frame_id !== startId ||
      transition.incoming_successor_frame_id !== source.source_frame_id ||
      transition.incoming_start_simulator_step_count !==
        events.incoming_start_simulator_step_count ||
      transition.incoming_successor_simulator_step_count !==
        events.incoming_successor_simulator_step_count
    ) {
      invalid("Oracle incoming branches do not enter the current endpoint.");
    }
    const directory = /** @type {any[]} */ (endpoint.identity_directory.identities);
    const actionRows = /** @type {any[]} */ (transition.action_rows);
    const trajectories = /** @type {any[]} */ (events.agent_phase_trajectories);
    const sceneAgents = /** @type {any[]} */ (endpoint.scene.agents);
    const oracleEvents = /** @type {any[]} */ (events.events);
    const activeIds = directory
      .filter((row) => row.configured_active)
      .map((row) => row.public_agent_id);
    if (
      actionRows.length !== activeIds.length ||
      actionRows.some(
        (row, rowIndex) => row.actor_public_agent_id !== activeIds[rowIndex],
      )
    ) {
      invalid("Oracle Latest Transition actors do not equal active directory order.");
    }
    const successors = trajectories.map((row) => [
      row.agent_presentation_key,
      row.agent_public_agent_id,
      row.successor.position,
    ]);
    const current = sceneAgents.map((row) => [
      row.presentation_key,
      row.public_agent_id,
      row.position,
    ]);
    if (!structurallyEqual(successors, current)) {
      invalid("Oracle incoming successors do not join its current scene.");
    }
    for (const row of actionRows) {
      const actor = directory.find(
        (candidate) => candidate.public_agent_id === row.actor_public_agent_id,
      );
      const expectedTargets = [
        ...directory.filter((candidate) => candidate.team_id === actor.team_id),
        ...directory.filter((candidate) => candidate.team_id !== actor.team_id),
      ].map((candidate) => candidate.public_agent_id);
      if (
        !structurallyEqual(row.target_action_recipient_public_agent_id_by_id, [
          null,
          ...expectedTargets,
        ])
      ) {
        invalid("Oracle Latest Transition target axis changed actor-relative order.");
      }
    }
    const rejectedRows = new Set(
      actionRows
        .filter((row) => !structurallyEqual(row.submitted_action, row.accepted_action))
        .map((row) => row.actor_public_agent_id),
    );
    const rejectedEvents = new Set(
      oracleEvents
        .filter(
          (event) =>
            event.event_kind === "action_rejected" &&
            event.actor_identity.identity_kind === "authorized_agent",
        )
        .map((event) => event.actor_identity.public_agent_id),
    );
    if (
      rejectedRows.size !== rejectedEvents.size ||
      [...rejectedRows].some((publicId) => !rejectedEvents.has(publicId))
    ) {
      invalid("Oracle rejected action rows do not equal active rejection events.");
    }
    return;
  }
  const parts = endpoint.parts;
  const recipient = source.source_recipient_public_agent_id;
  const mode = shared ? "shared-obs-visual-union" : "actor-pov";
  const prefix = `${source.episode_id}:${mode}:${recipient}`;
  const transitionId = `${prefix}:transition:${index - 1}`;
  const startId = `${prefix}:frame:${index - 1}`;
  if (
    events.source_episode_id !== source.episode_id ||
    events.recipient_public_agent_id !== recipient ||
    events.recipient_presentation_key !== parts.recipient_presentation_key ||
    events.incoming_transition_index !== index - 1 ||
    events.incoming_recipient_transition_id !== transitionId ||
    events.incoming_start_recipient_frame_id !== startId ||
    events.incoming_successor_recipient_frame_id !== source.source_recipient_frame_id ||
    events.incoming_successor_simulator_step_count !==
      source.source_simulator_step_count ||
    transition.incoming_transition_index !== index - 1 ||
    transition.incoming_transition_id !== transitionId ||
    transition.incoming_start_frame_id !== startId ||
    transition.incoming_successor_frame_id !== source.source_recipient_frame_id ||
    transition.incoming_start_simulator_step_count !==
      events.incoming_start_simulator_step_count ||
    transition.incoming_successor_simulator_step_count !==
      events.incoming_successor_simulator_step_count ||
    transition.recipient_public_agent_id !== recipient ||
    transition.recipient_presentation_key !== parts.recipient_presentation_key ||
    transition.action_rows.length !== 1 ||
    transition.action_rows[0].actor_public_agent_id !== recipient ||
    transition.action_rows[0].actor_presentation_key !==
      parts.recipient_presentation_key
  ) {
    invalid("Agent incoming branches do not enter the current endpoint.");
  }
  const targetIds = /** @type {any[]} */ (endpoint.action_axis.target_actions).map(
    (row) => (row.target_action === 0 ? null : row.target_public_agent_id),
  );
  if (
    !structurallyEqual(
      transition.action_rows[0].target_action_recipient_public_agent_id_by_id,
      targetIds,
    )
  ) {
    invalid("Agent Latest Transition target axis does not equal its current axis.");
  }
  const actionRow = transition.action_rows[0];
  if (!shared) {
    const outcome = events.cues[0];
    const expectedOutcome = structurallyEqual(
      actionRow.submitted_action,
      actionRow.accepted_action,
    )
      ? "accepted"
      : "rejected";
    if (
      outcome.cue_type !== "own_action_outcome" ||
      outcome.outcome !== expectedOutcome
    ) {
      invalid("NoSharedObs action outcome does not join Latest Transition.");
    }
  }
  const sceneById = new Map(
    /** @type {any[]} */ (parts.scene.agents).map((row) => [row.public_agent_id, row]),
  );
  const provenanceById = new Map(
    /** @type {any[]} */ (parts.agent_observation_provenance ?? []).map((row) => [
      row.agent_public_agent_id,
      row,
    ]),
  );
  const selfAgent = sceneById.get(recipient);
  const expectedSelfObservation = selfAgent
    ? projectAgentIncomingObservation(selfAgent)
    : null;
  for (const row of events.cues ?? events.deltas) {
    for (const observation of [row.start_observation, row.successor_observation]) {
      if (observation) validateAgentIncomingObservationAxis(observation, endpoint);
    }
    if (!shared) {
      if (!selfAgent || !expectedSelfObservation) {
        invalid("NoSharedObs incoming summary requires its current self row.");
      }
      if (
        (row.cue_type === "own_position_changed" &&
          !structurallyEqual(row.successor_position, selfAgent.position)) ||
        (row.cue_type === "own_health_changed" &&
          !Object.is(row.successor_health, selfAgent.current_health)) ||
        (row.cue_type === "own_status_changed" &&
          !structurallyEqual(
            row.successor_statuses,
            expectedSelfObservation.statuses,
          )) ||
        (row.cue_type === "own_cooldown_changed" &&
          row.successor_remaining_ticks !== selfAgent.ultimate_cooldown_remaining) ||
        (row.cue_type === "own_lifecycle_changed" &&
          (!row.successor_active ||
            row.successor_life_state !== selfAgent.life_state ||
            row.successor_spawn_shield_remaining_ticks !==
              selfAgent.spawn_shield_remaining))
      ) {
        invalid("NoSharedObs own-cue successor does not join the current self row.");
      }
    }
    const publicId = row.agent_public_agent_id;
    if (!publicId) continue;
    const sceneRow = sceneById.get(publicId);
    if (
      row.cue_type === "visible_body_observation_changed" ||
      row.delta_kind === "appearance" ||
      row.delta_kind === "observed_values_change"
    ) {
      if (row.successor_observation === null) {
        if (sceneRow) invalid("Disappeared Agent identity remains in current scene.");
      } else if (
        !sceneRow ||
        !structurallyEqual(
          row.successor_observation,
          projectAgentIncomingObservation(sceneRow),
        )
      ) {
        invalid("Agent incoming successor observation does not equal current scene.");
      }
    }
    if (row.delta_kind === "disappearance") {
      if (sceneRow || provenanceById.has(publicId)) {
        invalid("SharedObs disappeared identity remains in current endpoint.");
      }
    }
    if (
      row.delta_kind === "appearance" ||
      row.delta_kind === "observation_provenance_change"
    ) {
      const provenance = provenanceById.get(publicId);
      if (
        !provenance ||
        !structurallyEqual(
          row.successor_observation_sources,
          provenance.observation_sources,
        )
      ) {
        invalid("SharedObs successor provenance does not equal current endpoint.");
      }
    }
  }
}

/**
 * Join actor-only drawing events to the frame without widening policy input.
 *
 * `frame`, `source`, and `endpoint` are schema-checked. `oracle` and `shared`
 * select the audience; `renderedScene` may also include checked corpse overlays.
 * Oracle frames must omit visual events. Actor frame zero requires null; later
 * actor frames need a matching incoming summary, identities, and action rejection
 * facts. A corpse overlay may supply a successor anchor only for its own death
 * choreography, not for other after-step facts.
 *
 * Returns undefined or throws TypeError. The base endpoint remains the source of
 * observations, legal masks, and targets; drawing additions do not change it.
 *
 * @param {Record<string, any>} frame
 * @param {Record<string, any>} source
 * @param {Record<string, any>} endpoint
 * @param {boolean} oracle
 * @param {boolean} shared
 * @param {Record<string, any>} renderedScene
 */
function validateAgentVisualStateMatrix(
  frame,
  source,
  endpoint,
  oracle,
  shared,
  renderedScene,
) {
  const index = source.source_frame_index;
  if (oracle) {
    if (Object.hasOwn(frame, "visual_events")) {
      invalid("Oracle presentations must not contain an Agent visual channel.");
    }
    return;
  }
  const visual = frame.visual_events;
  if (index === 0) {
    if (visual !== null) {
      invalid("Initial Agent presentations must not contain incoming visual events.");
    }
    return;
  }
  if (!visual) {
    invalid("Noninitial Agent presentations require incoming visual events.");
  }
  validateAgentVisualEvents(visual);
  const recipient = source.source_recipient_public_agent_id;
  const prefix = `${source.episode_id}:${shared ? "shared-obs-visual-union" : "actor-pov"}:${recipient}`;
  const transitionId = `${prefix}:transition:${index - 1}`;
  const startId = `${prefix}:frame:${index - 1}`;
  const latest = frame.latest_events;
  if (
    visual.source_episode_id !== source.episode_id ||
    visual.recipient_public_agent_id !== recipient ||
    visual.recipient_presentation_key !== endpoint.parts.recipient_presentation_key ||
    visual.incoming_transition_index !== index - 1 ||
    visual.incoming_recipient_transition_id !== transitionId ||
    visual.incoming_start_recipient_frame_id !== startId ||
    visual.incoming_successor_recipient_frame_id !== source.source_recipient_frame_id ||
    visual.incoming_successor_simulator_step_count !==
      source.source_simulator_step_count ||
    visual.incoming_recipient_transition_id !==
      latest.incoming_recipient_transition_id ||
    visual.incoming_start_recipient_frame_id !==
      latest.incoming_start_recipient_frame_id ||
    visual.incoming_successor_recipient_frame_id !==
      latest.incoming_successor_recipient_frame_id ||
    visual.incoming_start_simulator_step_count !==
      latest.incoming_start_simulator_step_count ||
    visual.incoming_successor_simulator_step_count !==
      latest.incoming_successor_simulator_step_count
  ) {
    invalid("Agent visual events do not join their local incoming evidence.");
  }
  const trajectories = /** @type {any[]} */ (visual.agent_phase_trajectories);
  const baseAgents = /** @type {any[]} */ (endpoint.parts.scene.agents);
  const overlayAgents = /** @type {any[]} */ (
    frame.local_oracle_corpse_overlay.corpse_observations
  ).map((row) => row.corpse);
  if (!structurallyEqual(renderedScene.agents, [...baseAgents, ...overlayAgents])) {
    invalid("Agent rendered scene does not equal its base plus corpse overlay.");
  }
  const overlayByKey = new Map(overlayAgents.map((row) => [row.presentation_key, row]));
  const deathOverlayKeys = new Set(
    /** @type {any[]} */ (visual.events)
      .filter(
        (event) =>
          event.event_kind === "agent_died" &&
          overlayByKey.has(event.recipient_anchor.presentation_key),
      )
      .map((event) => event.recipient_anchor.presentation_key),
  );
  const currentAgents = [
    ...baseAgents,
    ...[...deathOverlayKeys].map((key) => overlayByKey.get(key)),
  ];
  const currentByKey = new Map(currentAgents.map((row) => [row.presentation_key, row]));
  const currentByPublicId = new Map(
    currentAgents.map((row) => [row.public_agent_id, row]),
  );
  const successorTrajectories = trajectories.filter((row) => row.successor !== null);
  if (
    successorTrajectories.length !== currentAgents.length ||
    trajectories.some((row) => {
      const currentByTrajectoryKey = currentByKey.get(row.agent_presentation_key);
      const currentByTrajectoryPublicId = currentByPublicId.get(
        row.agent_public_agent_id,
      );
      if (row.successor === null) {
        return (
          currentByTrajectoryKey !== undefined ||
          currentByTrajectoryPublicId !== undefined
        );
      }
      return (
        !currentByTrajectoryKey ||
        currentByTrajectoryKey !== currentByTrajectoryPublicId ||
        currentByTrajectoryKey.public_agent_id !== row.agent_public_agent_id ||
        currentByTrajectoryKey.class_id !== row.agent_class_id ||
        !structurallyEqual(currentByTrajectoryKey.position, row.successor.position)
      );
    })
  ) {
    invalid(
      "Agent visual successors do not join the actor-input scene plus death-owned corpse endpoints.",
    );
  }
  const overlayOnlyKeys = new Set(overlayByKey.keys());
  const afterStateEventKinds = new Set([
    "recipient_health_resolution",
    "health_regenerated",
    "cooldown_started",
    "cooldown_ready",
  ]);
  for (const event of /** @type {any[]} */ (visual.events)) {
    if (event.event_kind === "agent_died") continue;
    const anchors = oracleEventAnchors(event).map(({ anchor }) => anchor);
    if (
      anchors.some(
        (anchor) =>
          anchor.phase === "successor" && overlayOnlyKeys.has(anchor.presentation_key),
      ) ||
      (afterStateEventKinds.has(event.event_kind) &&
        anchors.some((anchor) => overlayOnlyKeys.has(anchor.presentation_key)))
    ) {
      invalid("Only death choreography may consume a corpse-overlay endpoint.");
    }
  }
  const actionRow = frame.latest_transition.action_rows[0];
  const rejectionEvents = /** @type {any[]} */ (visual.events).filter(
    (event) => event.event_kind === "action_rejected",
  );
  if (
    rejectionEvents.some(
      (event) =>
        event.actor_identity?.identity_kind !== "authorized_agent" ||
        event.actor_identity.public_agent_id !== recipient ||
        event.actor_identity.presentation_key !==
          endpoint.parts.recipient_presentation_key ||
        event.actor_anchor?.public_agent_id !== recipient ||
        event.actor_anchor?.presentation_key !==
          endpoint.parts.recipient_presentation_key,
    )
  ) {
    invalid("Agent visual rejection actor is not the fixed recipient.");
  }
  const submitted = actionRow.submitted_action;
  const accepted = actionRow.accepted_action;
  const submittedIsOutOfDomain =
    submitted.move_action < 0 ||
    submitted.move_action >= 9 ||
    submitted.target_action < 0 ||
    submitted.target_action >= 11 ||
    submitted.use_ultimate_action < 0 ||
    submitted.use_ultimate_action >= 2;
  const expectedRejectionComponents = submittedIsOutOfDomain
    ? ["domain"]
    : [
        ...(submitted.move_action !== accepted.move_action ? ["movement"] : []),
        ...(submitted.target_action !== accepted.target_action ||
        submitted.use_ultimate_action !== accepted.use_ultimate_action
          ? ["combat_pair"]
          : []),
      ];
  if (
    !structurallyEqual(
      rejectionEvents.map((event) => event.rejection_component),
      expectedRejectionComponents,
    ) ||
    rejectionEvents.some(
      (event) => !structurallyEqual(event.submitted_action, submitted),
    )
  ) {
    invalid("Agent own visual rejection does not join Latest Transition.");
  }
}

/**
 * Check the replay's outgoing action records against the current frame.
 *
 * `frame`, `source`, and `endpoint` are schema-checked. `oracle` and `shared`
 * choose global or actor-local transition IDs and rows. The final frame must have
 * no upcoming transition; other frames must have the canonical next transition.
 * Oracle rows cover the active roster in order. Actor rows contain only the
 * recipient, with its exact target axis and any matching inspection action.
 *
 * Returns undefined or throws TypeError. It does not execute the recorded action
 * or check the successor frame here.
 *
 * @param {Record<string, any>} frame
 * @param {Record<string, any>} source
 * @param {Record<string, any>} endpoint
 * @param {boolean} oracle
 * @param {boolean} shared
 */
function validateUpcomingStateMatrix(frame, source, endpoint, oracle, shared) {
  const final = source.source_frame_index === source.source_final_frame_index;
  const upcoming = frame.upcoming_transition;
  if (final) {
    if (upcoming !== null) {
      invalid("Final replay frame cannot carry Upcoming Transition.");
    }
    return;
  }
  if (upcoming === null) {
    invalid("Non-final replay frame requires Upcoming Transition.");
  }
  const recipient = oracle ? null : source.source_recipient_public_agent_id;
  const prefix = oracle
    ? source.episode_id
    : `${source.episode_id}:${shared ? "shared-obs-visual-union" : "actor-pov"}:${recipient}`;
  validateUpcomingTransition(upcoming, prefix, source.episode_id);
  const currentFrameId = oracle
    ? source.source_frame_id
    : source.source_recipient_frame_id;
  if (
    upcoming.outgoing_transition_index !== source.source_frame_index ||
    upcoming.outgoing_start_frame_id !== currentFrameId ||
    upcoming.outgoing_start_simulator_step_count !== source.source_simulator_step_count
  ) {
    invalid("Upcoming Transition does not leave the current source frame.");
  }
  const actionRows = /** @type {any[]} */ (upcoming.action_rows);
  if (oracle) {
    const directory = /** @type {any[]} */ (endpoint.identity_directory.identities);
    const sceneById = new Map(
      /** @type {any[]} */ (endpoint.scene.agents).map((row) => [
        row.public_agent_id,
        row,
      ]),
    );
    const activeIds = directory
      .filter((row) => row.configured_active)
      .map((row) => row.public_agent_id);
    if (
      actionRows.length !== activeIds.length ||
      actionRows.some(
        (row, index) =>
          row.actor_public_agent_id !== activeIds[index] ||
          row.actor_presentation_key !==
            sceneById.get(row.actor_public_agent_id)?.presentation_key,
      )
    ) {
      invalid("Oracle Upcoming Transition actors do not equal active scene order.");
    }
    for (const row of actionRows) {
      const actor = directory.find(
        (candidate) => candidate.public_agent_id === row.actor_public_agent_id,
      );
      const expectedTargets = [
        ...directory.filter((candidate) => candidate.team_id === actor.team_id),
        ...directory.filter((candidate) => candidate.team_id !== actor.team_id),
      ].map((candidate) => candidate.public_agent_id);
      if (
        !structurallyEqual(row.target_action_recipient_public_agent_id_by_id, [
          null,
          ...expectedTargets,
        ])
      ) {
        invalid("Oracle Upcoming Transition target axis changed actor-relative order.");
      }
    }
  } else {
    const parts = endpoint.parts;
    const targetIds = /** @type {any[]} */ (endpoint.action_axis.target_actions).map(
      (row) => (row.target_action === 0 ? null : row.target_public_agent_id),
    );
    if (
      upcoming.recipient_public_agent_id !== recipient ||
      upcoming.recipient_presentation_key !== parts.recipient_presentation_key ||
      actionRows.length !== 1 ||
      actionRows[0].actor_public_agent_id !== recipient ||
      actionRows[0].actor_presentation_key !== parts.recipient_presentation_key ||
      !structurallyEqual(
        actionRows[0].target_action_recipient_public_agent_id_by_id,
        targetIds,
      )
    ) {
      invalid("Agent Upcoming Transition must contain only its fixed recipient.");
    }
  }
  const inspection = frame.replay_inspection;
  if (inspection === null) return;
  const inspectedRow = actionRows.find(
    (row) => row.actor_public_agent_id === inspection.actor_public_agent_id,
  );
  if (
    !inspectedRow ||
    inspectedRow.actor_presentation_key !== inspection.actor_presentation_key ||
    !structurallyEqual(inspectedRow.submitted_action, inspection.submitted_action) ||
    !structurallyEqual(inspectedRow.accepted_action, inspection.accepted_action)
  ) {
    invalid("Replay inspection does not equal its Upcoming Transition row.");
  }
}

/**
 * Validate the researcher-wide facts attached to an actor replay frame.
 *
 * `frame` is schema-checked and contains a replay researcher-space record. This
 * joins its episode, cursor, ticks, selected recipient, ten-slot identity
 * directory, active roster, class mechanics, and incoming/outgoing action rows.
 * Any overlapping actor-local body facts must match, allowing the defined float
 * tolerance where stored precision differs.
 *
 * Returns the mutable list of presentation-key/public-ID pairs collected under
 * Oracle rules, for later digest-derived key checks. Throws TypeError on invalid
 * joins. These facts support researcher inspection; they are not actor input.
 *
 * @param {Record<string, any>} frame
 * @returns {{key: string, publicId: string}[]}
 */
function validateReplayResearcherSpace(frame) {
  const source = frame.source;
  const researcher = frame.researcher_space;
  if (
    researcher.researcher_space_kind !== "global_replay_researcher_space" ||
    researcher.episode_id !== source.episode_id ||
    researcher.frame_index !== source.source_frame_index ||
    researcher.final_frame_index !== source.source_final_frame_index ||
    researcher.simulator_step_count !== source.source_simulator_step_count ||
    researcher.selected_public_agent_id !== source.source_recipient_public_agent_id
  ) {
    invalid("Replay researcher space does not join its Agent source epoch.");
  }

  const directory = /** @type {any[]} */ (researcher.identity_directory.identities);
  if (directory.length !== 10) {
    invalid("Replay researcher directory requires ten rows.");
  }
  directory.forEach((row, index) => {
    if (
      row.team_id !== Math.floor(index / 5) + 1 ||
      row.team_local_slot !== index % 5 ||
      row.configured_active !== (row.class_id !== null) ||
      (row.class_id === null) !== (row.class_name === null)
    ) {
      invalid("Replay researcher directory lost fixed team topology.");
    }
  });
  requireUnique(
    directory.map((row) => row.public_agent_id),
    "Replay researcher directory identities",
  );

  const activeDirectory = directory.filter((row) => row.configured_active);
  const roster = /** @type {any[]} */ (researcher.roster_agents);
  if (
    roster.length !== activeDirectory.length ||
    roster.some((row, index) => {
      const identity = activeDirectory[index];
      return (
        row.public_agent_id !== identity.public_agent_id ||
        row.team_id !== identity.team_id ||
        row.team_local_slot !== identity.team_local_slot ||
        row.class_id !== identity.class_id ||
        row.class_name !== identity.class_name
      );
    }) ||
    !roster.some((row) => row.public_agent_id === researcher.selected_public_agent_id)
  ) {
    invalid("Replay researcher roster does not exactly join active identities.");
  }
  validateResearcherRosterFacts(
    roster,
    /** @type {any[]} */ (researcher.class_mechanics),
    "Replay researcher",
  );
  const researcherClasses = new Map(
    /** @type {any[]} */ (researcher.class_mechanics).map((row) => [row.class_id, row]),
  );
  for (const localClass of /** @type {any[]} */ (
    frame.current_endpoint.parts.scene.class_mechanics
  )) {
    if (!structurallyEqual(researcherClasses.get(localClass.class_id), localClass)) {
      invalid("Replay researcher class mechanics changed a fog-authorized class.");
    }
  }
  const researcherRoster = new Map(roster.map((row) => [row.public_agent_id, row]));
  for (const localActor of /** @type {any[]} */ (
    frame.current_endpoint.parts.scene.agents
  )) {
    const globalActor = researcherRoster.get(localActor.public_agent_id);
    const globalStatuses = /** @type {any[]} */ (globalActor?.statuses ?? []);
    const localStatuses = /** @type {any[]} */ (localActor.statuses);
    const globalAuras = /** @type {any[]} */ (globalActor?.aura_modifiers ?? []);
    const localAuras = /** @type {any[]} */ (localActor.aura_modifiers);
    if (
      !globalActor ||
      typeof globalActor !== "object" ||
      globalActor.team_id !== localActor.team_id ||
      globalActor.class_id !== localActor.class_id ||
      globalActor.class_name !== localActor.class_name ||
      globalActor.life_state !== localActor.life_state ||
      !liveResearcherFloatMatches(
        globalActor.current_health,
        localActor.current_health,
      ) ||
      !liveResearcherFloatMatches(
        globalActor.maximum_health,
        localActor.maximum_health,
      ) ||
      !liveResearcherFloatMatches(
        globalActor.effective_movement_speed,
        localActor.effective_movement_speed,
      ) ||
      globalActor.ultimate_cooldown_remaining !==
        localActor.ultimate_cooldown_remaining ||
      globalActor.spawn_shield_remaining !== localActor.spawn_shield_remaining ||
      globalActor.steps_until_out_of_combat !== localActor.steps_until_out_of_combat ||
      globalActor.out_of_combat_delay_steps !== localActor.out_of_combat_delay_steps ||
      globalStatuses.length !== localStatuses.length ||
      globalStatuses.some(
        (status, index) => !liveResearcherStatusMatches(status, localStatuses[index]),
      ) ||
      globalAuras.length !== localAuras.length ||
      globalAuras.some(
        (aura, index) => !liveResearcherAuraMatches(aura, localAuras[index]),
      )
    ) {
      invalid("Replay researcher roster changed a fog-authorized actor fact.");
    }
  }

  /**
   * Check one researcher-wide replay transition in the enclosing frame.
   *
   * `transition` is the nullable incoming or outgoing record; `incoming` selects
   * which side of the current frame is checked. Presence, adjacent ticks, canonical
   * IDs, complete roster order, and each actor-relative target axis must match the
   * enclosing validated researcher facts. Returns undefined or throws TypeError.
   * It neither changes the record nor executes the action.
   *
   * @param {Record<string, any> | null} transition
   * @param {boolean} incoming
   */
  const validateTransition = (transition, incoming) => {
    const expected = incoming
      ? source.source_frame_index > 0
      : source.source_frame_index < source.source_final_frame_index;
    if (!expected) {
      if (transition !== null) {
        invalid("Replay researcher transition presence changed at an edge.");
      }
      return;
    }
    if (transition === null) {
      invalid("Replay researcher transition is missing from a recorded epoch.");
    }
    if (incoming) {
      validateLatestTransition(transition, source.episode_id, source.episode_id);
      if (
        transition.incoming_transition_index !== source.source_frame_index - 1 ||
        transition.incoming_successor_frame_id !==
          `${source.episode_id}:frame:${source.source_frame_index}` ||
        transition.incoming_successor_simulator_step_count !==
          source.source_simulator_step_count
      ) {
        invalid("Replay researcher Latest Transition misses current s_n.");
      }
    } else {
      validateUpcomingTransition(transition, source.episode_id, source.episode_id);
      if (
        transition.outgoing_transition_index !== source.source_frame_index ||
        transition.outgoing_start_frame_id !==
          `${source.episode_id}:frame:${source.source_frame_index}` ||
        transition.outgoing_start_simulator_step_count !==
          source.source_simulator_step_count
      ) {
        invalid("Replay researcher Upcoming Transition does not leave current s_n.");
      }
    }
    const rows = /** @type {any[]} */ (transition.action_rows);
    if (
      rows.length !== roster.length ||
      rows.some(
        (row, index) =>
          row.actor_public_agent_id !== roster[index].public_agent_id ||
          row.actor_presentation_key !== roster[index].presentation_key,
      )
    ) {
      invalid("Replay researcher transition actors changed roster order.");
    }
    for (const row of rows) {
      const actor = directory.find(
        (identity) => identity.public_agent_id === row.actor_public_agent_id,
      );
      const expectedTargets = [
        ...directory.filter((identity) => identity.team_id === actor.team_id),
        ...directory.filter((identity) => identity.team_id !== actor.team_id),
      ].map((identity) => identity.public_agent_id);
      if (
        !structurallyEqual(row.target_action_recipient_public_agent_id_by_id, [
          null,
          ...expectedTargets,
        ])
      ) {
        invalid("Replay researcher transition target axis changed team order.");
      }
    }
  };

  validateTransition(researcher.latest_transition, true);
  validateTransition(researcher.upcoming_transition, false);
  return validatePresentationKeyGraph(researcher, {
    authorityKind: "oracle",
  });
}

/**
 * Compare two finite body facts using the live/replay precision allowance.
 *
 * `left` and `right` must be numbers. Returns true when both are finite and their
 * absolute difference is at most the larger of 1e-8 and 1e-6 times the larger
 * magnitude. Returns false for wrong types or nonfinite values. No coercion or
 * mutation occurs.
 *
 * @param {number} left @param {number} right
 */
function liveResearcherFloatMatches(left, right) {
  return (
    Number.isFinite(left) &&
    Number.isFinite(right) &&
    Math.abs(left - right) <=
      Math.max(1e-8, 1e-6 * Math.max(Math.abs(left), Math.abs(right)))
  );
}

/**
 * Compare the shared public fields of two status records.
 *
 * `globalStatus` and `localStatus` are schema-checked researcher and actor records.
 * Channel identity, names, source class, duration, and other static fields must
 * match exactly; a numeric magnitude uses the declared float tolerance, and null
 * must match null. Returns a boolean.
 *
 * Direct-source lists are deliberately outside this comparison: the actor record
 * may not disclose sources that the researcher record contains.
 *
 * @param {Record<string, any>} globalStatus @param {Record<string, any>} localStatus
 */
function liveResearcherStatusMatches(globalStatus, localStatus) {
  for (const field of [
    "status_channel",
    "status_id",
    "family",
    "configured_duration_steps",
    "remaining_duration",
    "source_class_id",
    "source_class_name",
    "source_action_component",
    "magnitude_kind",
    "breaks_on_positive_damage",
  ]) {
    if (!Object.is(globalStatus[field], localStatus[field])) {
      return false;
    }
  }
  if (globalStatus.magnitude === null || localStatus.magnitude === null) {
    return globalStatus.magnitude === localStatus.magnitude;
  }
  return liveResearcherFloatMatches(globalStatus.magnitude, localStatus.magnitude);
}

/**
 * Compare the public fields of two already checked aura modifiers.
 *
 * `globalAura` and `localAura` must have the same aura ID and multipliers that
 * match within the live/replay float tolerance. Returns a boolean without changing
 * either record or recomputing which emitters caused the modifier.
 *
 * @param {Record<string, any>} globalAura @param {Record<string, any>} localAura
 */
function liveResearcherAuraMatches(globalAura, localAura) {
  return (
    globalAura.aura_id === localAura.aura_id &&
    liveResearcherFloatMatches(globalAura.multiplier, localAura.multiplier)
  );
}

/**
 * Compare one drawable corpse with its supplied public-facts record.
 *
 * `corpse` and `facts` are schema-checked records. Identity, life state, counters,
 * and static labels must match exactly. Position, body values, and effect
 * magnitudes use the declared float allowance. Status direct-source lists must be
 * empty on both records. Returns a boolean without changing either record.
 *
 * This compares two supplied records; it does not independently prove visibility
 * or the corpse's position from global simulator state.
 *
 * @param {Record<string, any>} corpse
 * @param {Record<string, any>} facts
 */
function localOracleCorpseMatchesPublicFacts(corpse, facts) {
  for (const field of [
    "public_agent_id",
    "team_id",
    "class_id",
    "class_name",
    "life_state",
    "ultimate_cooldown_remaining",
    "spawn_shield_remaining",
    "steps_until_out_of_combat",
    "out_of_combat_delay_steps",
  ]) {
    if (!Object.is(corpse[field], facts[field])) return false;
  }
  if (
    corpse.position.length !== facts.position.length ||
    corpse.position.some(
      (/** @type {number} */ value, /** @type {number} */ index) =>
        !liveResearcherFloatMatches(value, facts.position[index]),
    )
  ) {
    return false;
  }
  for (const field of [
    "radius",
    "current_health",
    "maximum_health",
    "base_movement_speed",
    "effective_movement_speed",
    "observation_radius",
    "basic_interaction_radius",
    "ultimate_interaction_radius",
    "out_of_combat_health_regeneration_fraction_per_step",
  ]) {
    if (!liveResearcherFloatMatches(corpse[field], facts[field])) return false;
  }
  const corpseStatuses = /** @type {any[]} */ (corpse.statuses);
  const factStatuses = /** @type {any[]} */ (facts.statuses);
  if (
    corpseStatuses.length !== factStatuses.length ||
    corpseStatuses.some(
      (status, index) =>
        status.direct_sources.length !== 0 ||
        factStatuses[index].direct_sources.length !== 0 ||
        !liveResearcherStatusMatches(factStatuses[index], status),
    )
  ) {
    return false;
  }
  const corpseAuras = /** @type {any[]} */ (corpse.aura_modifiers);
  const factAuras = /** @type {any[]} */ (facts.aura_modifiers);
  return (
    corpseAuras.length === factAuras.length &&
    corpseAuras.every((aura, index) =>
      liveResearcherAuraMatches(factAuras[index], aura),
    )
  );
}

/**
 * Add checked drawing-only corpses to an actor scene.
 *
 * `frame` supplies the schema-checked corpse overlay and researcher facts;
 * `baseScene` contains the actor's authorized bodies; `shared` selects which
 * living sensor IDs may explain an overlay. The overlay must join the same epoch
 * and recipient, use ordered unique target IDs, name only allowed living sensors,
 * and agree with the declared corpse/public/roster facts.
 *
 * Returns a new mutable scene record with a new combined agent array and the
 * needed class profiles. Body/profile records are borrowed. Throws TypeError on
 * invalid joins. It does not recompute radius or line of sight: the trusted
 * producer owns that projection. The original endpoint still owns observations,
 * masks, and targets; these corpses add only drawing and inspection information.
 *
 * @param {Record<string, any>} frame
 * @param {Record<string, any>} baseScene
 * @param {boolean} shared
 * @returns {Record<string, any>}
 */
function composeLocalOracleCorpseOverlay(frame, baseScene, shared) {
  const overlay = frame.local_oracle_corpse_overlay;
  const source = frame.source;
  const parts = frame.current_endpoint.parts;
  const recipientId = parts.recipient_public_agent_id;
  if (
    overlay.overlay_kind !== "local_oracle_corpse_overlay" ||
    overlay.projection_basis !==
      "same_epoch_living_sensor_radius_and_static_line_of_sight" ||
    overlay.source_episode_id !== source.episode_id ||
    overlay.source_frame_index !== source.source_frame_index ||
    overlay.source_frame_id !==
      `${source.episode_id}:frame:${source.source_frame_index}` ||
    overlay.source_simulator_step_count !== source.source_simulator_step_count ||
    overlay.source_authority_epoch !== source.source_authority_epoch ||
    overlay.recipient_public_agent_id !== recipientId ||
    overlay.recipient_presentation_key !== parts.recipient_presentation_key
  ) {
    invalid("Local-Oracle corpse overlay does not join its Agent epoch.");
  }

  const baseAgents = /** @type {any[]} */ (baseScene.agents);
  const baseById = new Map(baseAgents.map((row) => [row.public_agent_id, row]));
  const livingBaseIds = new Set(
    baseAgents
      .filter((row) => row.life_state === "alive")
      .map((row) => row.public_agent_id),
  );
  const expectedSensors = /** @type {string[]} */ (
    shared
      ? /** @type {any[]} */ (parts.authorized_sensor_sources)
          .map((row) => row.source_public_agent_id)
          .filter((publicId) => livingBaseIds.has(publicId))
      : livingBaseIds.has(recipientId)
        ? [recipientId]
        : []
  );
  if (!structurallyEqual(overlay.living_sensor_public_agent_ids, expectedSensors)) {
    invalid("Local-Oracle corpse sensors changed their living authority set.");
  }

  const targetIds = /** @type {any[]} */ (
    frame.current_endpoint.action_axis.target_actions
  )
    .slice(1)
    .map((row) => row.target_public_agent_id);
  const recipient = baseById.get(recipientId);
  const researcherRoster = new Map(
    /** @type {any[]} */ (frame.researcher_space.roster_agents).map((row) => [
      row.public_agent_id,
      row,
    ]),
  );
  const mechanicsByClassId = new Map(
    /** @type {any[]} */ (frame.researcher_space.class_mechanics).map((row) => [
      row.class_id,
      row,
    ]),
  );
  const corpseRows = /** @type {any[]} */ (overlay.corpse_observations);
  requireUnique(
    corpseRows.map((row) => row.corpse.public_agent_id),
    "Local-Oracle corpse identities",
  );
  requireUnique(
    corpseRows.map((row) => row.corpse.presentation_key),
    "Local-Oracle corpse presentation keys",
  );
  let previousTargetIndex = -1;
  for (const observation of corpseRows) {
    const corpse = observation.corpse;
    const oraclePublicFacts = observation.oracle_public_facts;
    const targetIndex = targetIds.indexOf(corpse.public_agent_id);
    const globalActor = researcherRoster.get(corpse.public_agent_id);
    const classMechanics = mechanicsByClassId.get(corpse.class_id);
    const globalStatuses = /** @type {any[]} */ (globalActor?.statuses ?? []);
    const localStatuses = /** @type {any[]} */ (corpse.statuses);
    const globalAuras = /** @type {any[]} */ (globalActor?.aura_modifiers ?? []);
    const localAuras = /** @type {any[]} */ (corpse.aura_modifiers);
    const expectedRelation = targetIndex < 5 ? "ally" : "opponent";
    const expectedTeam =
      targetIndex < 5 ? recipient.team_id : recipient.team_id === 1 ? 2 : 1;
    if (
      targetIndex < 0 ||
      targetIndex <= previousTargetIndex ||
      corpse.public_agent_id === recipientId ||
      baseById.has(corpse.public_agent_id) ||
      corpse.life_state !== "corpse" ||
      corpse.relation !== expectedRelation ||
      corpse.team_id !== expectedTeam ||
      observation.observing_sensor_public_agent_ids.length === 0 ||
      !structurallyEqual(
        observation.observing_sensor_public_agent_ids,
        expectedSensors.filter((publicId) =>
          observation.observing_sensor_public_agent_ids.includes(publicId),
        ),
      ) ||
      observation.observing_sensor_public_agent_ids.some(
        (/** @type {string} */ publicId) => !expectedSensors.includes(publicId),
      ) ||
      localStatuses.some((status) => status.direct_sources.length !== 0) ||
      !localOracleCorpseMatchesPublicFacts(corpse, oraclePublicFacts) ||
      !globalActor ||
      !classMechanics ||
      globalActor.life_state !== "corpse" ||
      globalActor.team_id !== corpse.team_id ||
      globalActor.class_id !== corpse.class_id ||
      globalActor.class_name !== corpse.class_name ||
      !liveResearcherFloatMatches(globalActor.current_health, corpse.current_health) ||
      !liveResearcherFloatMatches(globalActor.maximum_health, corpse.maximum_health) ||
      !liveResearcherFloatMatches(
        globalActor.effective_movement_speed,
        corpse.effective_movement_speed,
      ) ||
      globalActor.ultimate_cooldown_remaining !== corpse.ultimate_cooldown_remaining ||
      globalActor.spawn_shield_remaining !== corpse.spawn_shield_remaining ||
      globalActor.steps_until_out_of_combat !== corpse.steps_until_out_of_combat ||
      globalActor.out_of_combat_delay_steps !== corpse.out_of_combat_delay_steps ||
      classMechanics.class_name !== corpse.class_name ||
      !liveResearcherFloatMatches(classMechanics.body_radius, corpse.radius) ||
      !liveResearcherFloatMatches(
        classMechanics.maximum_health,
        corpse.maximum_health,
      ) ||
      !liveResearcherFloatMatches(
        classMechanics.base_movement_speed,
        corpse.base_movement_speed,
      ) ||
      !liveResearcherFloatMatches(
        classMechanics.observation_radius,
        corpse.observation_radius,
      ) ||
      !liveResearcherFloatMatches(
        classMechanics.basic_interaction_radius,
        corpse.basic_interaction_radius,
      ) ||
      !liveResearcherFloatMatches(
        classMechanics.ultimate_interaction_radius,
        corpse.ultimate_interaction_radius,
      ) ||
      classMechanics.out_of_combat_delay_steps !== corpse.out_of_combat_delay_steps ||
      !liveResearcherFloatMatches(
        classMechanics.out_of_combat_health_regeneration_fraction_per_step,
        corpse.out_of_combat_health_regeneration_fraction_per_step,
      ) ||
      globalStatuses.length !== localStatuses.length ||
      globalStatuses.some(
        (status, index) => !liveResearcherStatusMatches(status, localStatuses[index]),
      ) ||
      globalAuras.length !== localAuras.length ||
      globalAuras.some(
        (aura, index) => !liveResearcherAuraMatches(aura, localAuras[index]),
      )
    ) {
      invalid("Local-Oracle corpse facts changed from authorized Oracle truth.");
    }
    // Position is intentionally absent from researcher space. The trusted
    // producer binds it to the same-epoch global snapshot before sealing this
    // overlay; the browser never receives the hidden global geometry root.
    previousTargetIndex = targetIndex;
  }

  const agents = [...baseAgents, ...corpseRows.map((row) => row.corpse)];
  const representedClassIds = [...new Set(agents.map((row) => row.class_id))].sort(
    (left, right) => left - right,
  );
  const researcherMechanics = new Map(
    /** @type {any[]} */ (frame.researcher_space.class_mechanics).map((row) => [
      row.class_id,
      row,
    ]),
  );
  const classMechanics = representedClassIds.map((classId) =>
    researcherMechanics.get(classId),
  );
  if (classMechanics.some((row) => !row)) {
    invalid("Local-Oracle corpse scene lacks authorized class mechanics.");
  }
  return {
    ...baseScene,
    agents,
    class_mechanics: classMechanics,
  };
}

/**
 * Validate researcher-wide facts attached to a live actor presentation.
 *
 * `frame` is schema-checked. This joins the session, generation, revision, epoch,
 * selected recipient, ten-slot directory, active roster, incoming actions, and
 * technical-frame identity. Editable joint turns must also agree on pending joint
 * actions and the selected actor's draft, axis, and masks. Scripted playback has
 * no pending joint-action draft. Shared public body facts must agree between the
 * researcher and actor views.
 *
 * Returns a mutable list of Oracle presentation-key/public-ID pairs for later key
 * checks. Throws TypeError on invalid joins. Researcher data remains inspection
 * information; it does not replace the actor's authorized endpoint.
 *
 * @param {Record<string, any>} frame
 * @returns {{key: string, publicId: string}[]}
 */
function validateLiveResearcherSpace(frame) {
  const source = frame.source;
  const researcher = frame.researcher_space;
  if (
    researcher.researcher_space_kind !== "global_live_researcher_space" ||
    researcher.source_session_id !== source.source_session_id ||
    researcher.source_run_generation !== source.source_run_generation ||
    researcher.source_revision !== source.source_revision ||
    researcher.source_authority_epoch !== source.source_authority_epoch ||
    researcher.episode_id !== source.episode_id ||
    researcher.frame_index !== source.source_frame_index ||
    researcher.simulator_step_count !== source.source_simulator_step_count ||
    researcher.selected_public_agent_id !== source.source_recipient_public_agent_id
  ) {
    invalid("Live researcher space does not join its Agent source epoch.");
  }

  const directory = /** @type {any[]} */ (researcher.identity_directory.identities);
  if (directory.length !== 10) {
    invalid("Live researcher directory requires ten rows.");
  }
  directory.forEach((row, index) => {
    if (
      row.team_id !== Math.floor(index / 5) + 1 ||
      row.team_local_slot !== index % 5 ||
      row.configured_active !== (row.class_id !== null) ||
      (row.class_id === null) !== (row.class_name === null)
    ) {
      invalid("Live researcher directory lost fixed team topology.");
    }
  });
  requireUnique(
    directory.map((row) => row.public_agent_id),
    "Live researcher directory identities",
  );

  const activeDirectory = directory.filter((row) => row.configured_active);
  const roster = /** @type {any[]} */ (researcher.roster_agents);
  if (
    roster.length !== activeDirectory.length ||
    roster.some((row, index) => {
      const identity = activeDirectory[index];
      return (
        row.public_agent_id !== identity.public_agent_id ||
        row.team_id !== identity.team_id ||
        row.team_local_slot !== identity.team_local_slot ||
        row.class_id !== identity.class_id ||
        row.class_name !== identity.class_name
      );
    }) ||
    !roster.some((row) => row.public_agent_id === researcher.selected_public_agent_id)
  ) {
    invalid("Live researcher roster does not exactly join active identities.");
  }
  validateResearcherRosterFacts(
    roster,
    /** @type {any[]} */ (researcher.class_mechanics),
    "Live researcher",
  );

  const latest = researcher.latest_transition;
  if (source.source_frame_index === 0) {
    if (latest !== null) {
      invalid("Live researcher frame zero cannot carry Latest Transition.");
    }
  } else {
    if (latest === null) {
      invalid("Live researcher Latest Transition is missing.");
    }
    validateLatestTransition(latest, source.episode_id, source.episode_id);
    if (
      latest.incoming_transition_index !== source.source_frame_index - 1 ||
      latest.incoming_successor_frame_id !==
        `${source.episode_id}:frame:${source.source_frame_index}` ||
      latest.incoming_successor_simulator_step_count !==
        source.source_simulator_step_count ||
      latest.action_rows.length !== roster.length ||
      /** @type {any[]} */ (latest.action_rows).some(
        (row, index) =>
          row.actor_public_agent_id !== roster[index]?.public_agent_id ||
          row.actor_presentation_key !== roster[index]?.presentation_key,
      )
    ) {
      invalid("Live researcher Latest Transition misses current joint s_n.");
    }
    for (const row of /** @type {any[]} */ (latest.action_rows)) {
      const actor = directory.find(
        (identity) => identity.public_agent_id === row.actor_public_agent_id,
      );
      const expectedTargets = [
        ...directory.filter((identity) => identity.team_id === actor.team_id),
        ...directory.filter((identity) => identity.team_id !== actor.team_id),
      ].map((identity) => identity.public_agent_id);
      if (
        !structurallyEqual(row.target_action_recipient_public_agent_id_by_id, [
          null,
          ...expectedTargets,
        ])
      ) {
        invalid("Live researcher Latest target axis changed team order.");
      }
    }
  }

  const technical = researcher.technical_frame;
  if (
    technical.technical_kind !== "live_oracle_technical_frame" ||
    technical.episode_id !== source.episode_id ||
    technical.evaluation_frame_index !== source.source_frame_index ||
    technical.simulator_step_count !== source.source_simulator_step_count ||
    technical.incoming_transition_id !==
      (latest === null ? null : latest.incoming_transition_id)
  ) {
    invalid("Live researcher Technical Frame does not join its epoch.");
  }

  const pending = researcher.pending_inspection;
  if (pending.submission_scope !== source.source_submission_scope) {
    invalid("Live researcher pending scope does not join its source.");
  }
  if (pending.inspection_kind === "editable_live_draft") {
    if (pending.submission_scope !== "joint_turn") {
      invalid("Live researcher editable draft must submit one joint turn.");
    }
    const draft = pending.draft;
    const owner = roster.find(
      (row) => row.public_agent_id === researcher.selected_public_agent_id,
    );
    const decision = draft.decision_mask;
    validateDecisionMask(decision, "Live researcher decision mask");
    if (
      draft.current_simulator_step_count !== source.source_simulator_step_count ||
      draft.actor_public_agent_id !== researcher.selected_public_agent_id ||
      draft.actor_public_agent_id !== owner.public_agent_id ||
      draft.actor_presentation_key !== owner.presentation_key ||
      decision.owner_public_agent_id !== draft.actor_public_agent_id ||
      decision.owner_presentation_key !== draft.actor_presentation_key ||
      decision.target_actions[0].target_kind !== "no_target" ||
      /** @type {any[]} */ (decision.target_actions)
        .slice(1)
        .some((row) => row.target_kind !== "axis_only_authorized_agent")
    ) {
      invalid("Live researcher draft is not geometry-free selected-actor truth.");
    }
    const ownerDirectory = directory.find(
      (row) => row.public_agent_id === draft.actor_public_agent_id,
    );
    const expectedTargets = [
      ...directory.filter((row) => row.team_id === ownerDirectory.team_id),
      ...directory.filter((row) => row.team_id !== ownerDirectory.team_id),
    ].map((row) => row.public_agent_id);
    if (
      !structurallyEqual(
        /** @type {any[]} */ (decision.target_actions)
          .slice(1)
          .map((row) => row.target_public_agent_id),
        expectedTargets,
      )
    ) {
      invalid("Live researcher draft target axis changed team order.");
    }
    const action = draft.draft_action;
    const legality = draft.draft_legality;
    validateLivePendingJointAction(
      researcher.pending_joint_action,
      roster,
      source.source_simulator_step_count,
      draft,
      "Live researcher Pending Joint Action",
    );
    if (
      !structurallyEqual(
        draft.draft_target,
        decision.target_actions[action.target_action],
      ) ||
      legality.move_action_is_legal !==
        decision.movement_action_mask[action.move_action] ||
      legality.target_action_is_legal !==
        decision.target_action_mask[action.target_action]
    ) {
      invalid("Live researcher draft does not join its exact decision row.");
    }
    if (action.armed_lane === "none") {
      if (
        legality.armed_lane_is_legal !== null ||
        legality.combat_pair_is_legal !== null
      ) {
        invalid("Unarmed live researcher draft cannot carry combat legality.");
      }
    } else {
      const lane = action.armed_lane === "basic" ? 0 : 1;
      if (
        legality.armed_lane_is_legal !== decision.use_ultimate_action_mask[lane] ||
        legality.combat_pair_is_legal !==
          decision.target_use_ultimate_joint_mask[action.target_action][lane]
      ) {
        invalid("Live researcher draft joint legality changed.");
      }
    }

    const localInspection = frame.live_inspection.inspection;
    if (localInspection.inspection_kind !== "editable_live_draft") {
      invalid("Live researcher and Agent draft modes diverged.");
    }
    const localDraft = localInspection.draft;
    const localDecision = localDraft.decision_mask;
    const researcherTargetIds = /** @type {any[]} */ (decision.target_actions).map(
      (row) => (row.target_kind === "no_target" ? null : row.target_public_agent_id),
    );
    const localTargetIds = /** @type {any[]} */ (localDecision.target_actions).map(
      (row) => (row.target_kind === "no_target" ? null : row.target_public_agent_id),
    );
    if (
      !structurallyEqual(
        decision.movement_action_display_names,
        localDecision.movement_action_display_names,
      ) ||
      !structurallyEqual(
        decision.movement_action_mask,
        localDecision.movement_action_mask,
      ) ||
      !structurallyEqual(
        /** @type {any[]} */ (decision.target_actions).map((row) => row.display_name),
        /** @type {any[]} */ (localDecision.target_actions).map(
          (row) => row.display_name,
        ),
      ) ||
      !structurallyEqual(researcherTargetIds, localTargetIds) ||
      !structurallyEqual(
        decision.target_action_mask,
        localDecision.target_action_mask,
      ) ||
      !structurallyEqual(
        decision.use_ultimate_action_display_names,
        localDecision.use_ultimate_action_display_names,
      ) ||
      !structurallyEqual(
        decision.use_ultimate_action_mask,
        localDecision.use_ultimate_action_mask,
      ) ||
      !structurallyEqual(
        decision.target_use_ultimate_joint_mask,
        localDecision.target_use_ultimate_joint_mask,
      ) ||
      !structurallyEqual(draft.draft_action, localDraft.draft_action) ||
      !structurallyEqual(draft.draft_legality, localDraft.draft_legality)
    ) {
      invalid("Live researcher draft changed Agent action semantics.");
    }
  } else if (pending.inspection_kind === "scripted_playback_inspection") {
    if (researcher.pending_joint_action !== null) {
      invalid("Scripted researcher presentation cannot carry pending joint intent.");
    }
  } else {
    invalid("Live researcher inspection uses an unknown variant.");
  }

  const localScene = frame.current_endpoint.parts.scene;
  const researcherClasses = new Map(
    /** @type {any[]} */ (researcher.class_mechanics).map((row) => [row.class_id, row]),
  );
  for (const localClass of /** @type {any[]} */ (localScene.class_mechanics)) {
    if (!structurallyEqual(researcherClasses.get(localClass.class_id), localClass)) {
      invalid("Live researcher class mechanics changed a fog-authorized class.");
    }
  }

  const researcherRoster = new Map(roster.map((row) => [row.public_agent_id, row]));
  for (const localActor of /** @type {any[]} */ (localScene.agents)) {
    const globalActor = researcherRoster.get(localActor.public_agent_id);
    const globalStatuses = /** @type {any[]} */ (globalActor?.statuses ?? []);
    const localStatuses = /** @type {any[]} */ (localActor.statuses);
    const globalAuras = /** @type {any[]} */ (globalActor?.aura_modifiers ?? []);
    const localAuras = /** @type {any[]} */ (localActor.aura_modifiers);
    if (
      !globalActor ||
      typeof globalActor !== "object" ||
      globalActor.team_id !== localActor.team_id ||
      globalActor.class_id !== localActor.class_id ||
      globalActor.class_name !== localActor.class_name ||
      globalActor.life_state !== localActor.life_state ||
      !liveResearcherFloatMatches(
        globalActor.current_health,
        localActor.current_health,
      ) ||
      !liveResearcherFloatMatches(
        globalActor.maximum_health,
        localActor.maximum_health,
      ) ||
      !liveResearcherFloatMatches(
        globalActor.effective_movement_speed,
        localActor.effective_movement_speed,
      ) ||
      globalActor.ultimate_cooldown_remaining !==
        localActor.ultimate_cooldown_remaining ||
      globalActor.spawn_shield_remaining !== localActor.spawn_shield_remaining ||
      globalActor.steps_until_out_of_combat !== localActor.steps_until_out_of_combat ||
      globalActor.out_of_combat_delay_steps !== localActor.out_of_combat_delay_steps ||
      globalStatuses.length !== localStatuses.length ||
      globalStatuses.some(
        (status, index) => !liveResearcherStatusMatches(status, localStatuses[index]),
      ) ||
      globalAuras.length !== localAuras.length ||
      globalAuras.some(
        (aura, index) => !liveResearcherAuraMatches(aura, localAuras[index]),
      )
    ) {
      invalid("Live researcher roster changed a fog-authorized actor fact.");
    }
  }

  if (latest !== null) {
    const localLatest = frame.latest_transition;
    if (localLatest === null) {
      invalid("Live researcher Latest lacks its Agent transition.");
    }
    const globalRow = /** @type {any[]} */ (latest.action_rows).find(
      (row) => row.actor_public_agent_id === source.source_recipient_public_agent_id,
    );
    const localRow = localLatest.action_rows[0];
    if (
      !globalRow ||
      typeof globalRow !== "object" ||
      !structurallyEqual(
        globalRow.target_action_recipient_public_agent_id_by_id,
        localRow.target_action_recipient_public_agent_id_by_id,
      ) ||
      !structurallyEqual(globalRow.submitted_action, localRow.submitted_action) ||
      !structurallyEqual(globalRow.accepted_action, localRow.accepted_action)
    ) {
      invalid("Live researcher Latest changed Agent action semantics.");
    }
  }
  return validatePresentationKeyGraph(researcher, {
    authorityKind: "oracle",
  });
}

/**
 * Join action inspection to its exact decision epoch and authorized axis.
 *
 * `frame`, `source`, and `endpoint` are schema-checked. `actionAxis` may be null
 * only where that frame permits it; `scene` is the base actor-input scene, without
 * extra drawing-only corpses. `live`, `oracle`, and `shared` select the contract.
 * Live editable drafts and replay outgoing actions must join the owner, tick,
 * target IDs, labels, and legal mask. Final replay frames have no action
 * inspection; scripted live playback returns after checking its scope.
 *
 * Returns undefined or throws TypeError. Actor masks must equal the endpoint's
 * mask. Replay accepted actions must be legal; submitted int32 values can record
 * invalid intent. This validates records without selecting or applying actions.
 *
 * @param {Record<string, any>} frame
 * @param {Record<string, any>} source
 * @param {Record<string, any>} endpoint
 * @param {Record<string, any> | null} actionAxis
 * @param {Record<string, any>} scene
 * @param {boolean} live
 * @param {boolean} oracle
 * @param {boolean} shared
 */
function validateInspectionStateMatrix(
  frame,
  source,
  endpoint,
  actionAxis,
  scene,
  live,
  oracle,
  shared,
) {
  let inspection;
  if (live) {
    const envelope = frame.live_inspection;
    for (const key of [
      "source_session_id",
      "source_run_generation",
      "source_revision",
      "source_authority_epoch",
      "episode_id",
      "source_frame_index",
      "source_simulator_step_count",
    ]) {
      if (!Object.is(envelope[key], source[key])) {
        invalid(`Live inspection envelope does not join source field ${key}.`);
      }
    }
    if (oracle) {
      if (envelope.source_frame_id !== source.source_frame_id) {
        invalid("Live Oracle inspection frame does not join its source.");
      }
    } else if (
      envelope.source_recipient_public_agent_id !==
        source.source_recipient_public_agent_id ||
      envelope.source_recipient_frame_id !== source.source_recipient_frame_id
    ) {
      invalid("Live Agent inspection recipient does not join its source.");
    }
    const wrapper = envelope.inspection;
    if (wrapper.submission_scope !== source.source_submission_scope) {
      invalid("Live inspection submission scope does not join its source.");
    }
    if (wrapper.inspection_kind === "scripted_playback_inspection") {
      if (source.source_submission_scope !== "scripted_playback") {
        invalid("Scripted inspection requires scripted source authority.");
      }
      return;
    }
    inspection = wrapper.draft;
  } else {
    if (source.source_frame_index > source.source_final_frame_index) {
      invalid("Replay source frame exceeds its retained prefix.");
    }
    const final = source.source_frame_index === source.source_final_frame_index;
    inspection = frame.replay_inspection;
    if (final && inspection !== null) {
      invalid("Final replay frame cannot carry outgoing inspection.");
    }
    if (!final && !oracle && inspection === null) {
      invalid("Non-final Agent replay requires outgoing inspection.");
    }
    if (inspection === null) {
      if (!final && actionAxis !== null) {
        invalid("Uninspected Oracle replay must omit its action axis.");
      }
      return;
    }
    if (
      inspection.episode_id !== source.episode_id ||
      inspection.outgoing_transition_index !== source.source_frame_index
    ) {
      invalid("Replay inspection is not outgoing T_n.");
    }
    const reference = inspection.transition_reference;
    const recipient = oracle ? null : source.source_recipient_public_agent_id;
    const prefix = oracle
      ? source.episode_id
      : `${source.episode_id}:${shared ? "shared-obs-visual-union" : "actor-pov"}:${recipient}`;
    const expectedKind = oracle
      ? "oracle_recorded_transition"
      : shared
        ? "shared_obs_visual_union_transition"
        : "no_shared_obs_actor_pov_transition";
    const index = source.source_frame_index;
    if (
      reference.reference_kind !== expectedKind ||
      (!oracle && reference.recipient_public_agent_id !== recipient) ||
      reference.transition_id !== `${prefix}:transition:${index}` ||
      reference.start_frame_id !== `${prefix}:frame:${index}` ||
      reference.successor_frame_id !== `${prefix}:frame:${index + 1}`
    ) {
      invalid("Replay inspection transition reference is not canonical outgoing T_n.");
    }
  }
  if (actionAxis === null) invalid("Inspected presentation requires an action axis.");
  const movementActions = /** @type {any[]} */ (actionAxis.movement_actions);
  const targetActions = /** @type {any[]} */ (actionAxis.target_actions);
  const ultimateChoices = /** @type {any[]} */ (actionAxis.ultimate_choices);
  const sceneAgents = /** @type {any[]} */ (scene.agents);
  if (
    inspection.current_simulator_step_count !== source.source_simulator_step_count ||
    inspection.actor_presentation_key !== actionAxis.owner_presentation_key ||
    inspection.actor_public_agent_id !== actionAxis.owner_public_agent_id
  ) {
    invalid("Inspection owner and epoch do not join the current endpoint.");
  }
  const actor = sceneAgents.find(
    (row) => row.public_agent_id === inspection.actor_public_agent_id,
  );
  if (
    !actor ||
    actor.presentation_key !== inspection.actor_presentation_key ||
    !structurallyEqual(actor.position, inspection.actor_anchor)
  ) {
    invalid("Inspection actor anchor does not join the current scene.");
  }
  const decision = inspection.decision_mask;
  validateDecisionMask(decision, "Inspection decision mask");
  if (
    decision.target_actions.length !== targetActions.length ||
    decision.owner_presentation_key !== actionAxis.owner_presentation_key ||
    decision.owner_public_agent_id !== actionAxis.owner_public_agent_id ||
    !structurallyEqual(
      decision.movement_action_display_names,
      movementActions.map((row) => row.display_name),
    ) ||
    !structurallyEqual(
      decision.use_ultimate_action_display_names,
      ultimateChoices.map((row) => row.display_name),
    ) ||
    /** @type {any[]} */ (decision.target_actions).some(
      (row, index) =>
        row.target_action !== targetActions[index].target_action ||
        row.display_name !== targetActions[index].display_name ||
        (index > 0 &&
          row.target_public_agent_id !== targetActions[index].target_public_agent_id),
    )
  ) {
    invalid("Inspection decision surface does not join its action axis.");
  }
  const sceneById = new Map(sceneAgents.map((row) => [row.public_agent_id, row]));
  for (let index = 1; index < decision.target_actions.length; index += 1) {
    const target = decision.target_actions[index];
    const visible = sceneById.get(target.target_public_agent_id);
    if (visible) {
      if (
        target.target_kind !== "visible_authorized_agent" ||
        target.target_presentation_key !== visible.presentation_key ||
        !structurallyEqual(target.target_anchor, visible.position)
      ) {
        invalid("Visible inspection target does not join the current scene.");
      }
    } else if (target.target_kind !== "axis_only_authorized_agent") {
      invalid("An inspection target absent from scene must remain axis-only.");
    }
  }
  if (!oracle) {
    const sourceMask = endpoint.parts.next_decision_action_mask;
    if (
      !structurallyEqual(decision.movement_action_mask, sourceMask.move) ||
      !structurallyEqual(decision.target_action_mask, sourceMask.select_target) ||
      !structurallyEqual(decision.use_ultimate_action_mask, sourceMask.use_ultimate) ||
      !structurallyEqual(
        decision.target_use_ultimate_joint_mask,
        sourceMask.select_target_use_ultimate_joint,
      )
    ) {
      invalid("Agent inspection legality does not equal its endpoint mask.");
    }
  }
  if (!live) {
    validateSubmittedActionTuple(
      inspection.submitted_action,
      "Replay inspection submitted action",
    );
    const accepted = inspection.accepted_action;
    const expectedLane =
      accepted.use_ultimate_action === 1
        ? "ultimate"
        : accepted.target_action === 0
          ? "none"
          : "basic";
    if (
      accepted.move_action < 0 ||
      accepted.move_action >= 9 ||
      accepted.target_action < 0 ||
      accepted.target_action >= 11 ||
      accepted.use_ultimate_action < 0 ||
      accepted.use_ultimate_action >= 2 ||
      !decision.movement_action_mask[accepted.move_action] ||
      !decision.target_use_ultimate_joint_mask[accepted.target_action][
        accepted.use_ultimate_action
      ] ||
      !structurallyEqual(
        inspection.accepted_target,
        decision.target_actions[accepted.target_action],
      ) ||
      inspection.combat_lane !== expectedLane
    ) {
      invalid("Replay accepted action is not its exact legal decision row.");
    }
  } else {
    const action = inspection.draft_action;
    const legality = inspection.draft_legality;
    if (
      !structurallyEqual(
        inspection.draft_target,
        decision.target_actions[action.target_action],
      ) ||
      legality.move_action_is_legal !==
        decision.movement_action_mask[action.move_action] ||
      legality.target_action_is_legal !==
        decision.target_action_mask[action.target_action]
    ) {
      invalid("Live draft target and marginal legality do not join its decision mask.");
    }
    if (action.armed_lane === "none") {
      if (
        legality.armed_lane_is_legal !== null ||
        legality.combat_pair_is_legal !== null
      ) {
        invalid("Unarmed live draft cannot carry combat legality.");
      }
    } else {
      const lane = action.armed_lane === "basic" ? 0 : 1;
      if (
        legality.armed_lane_is_legal !== decision.use_ultimate_action_mask[lane] ||
        legality.combat_pair_is_legal !==
          decision.target_use_ultimate_joint_mask[action.target_action][lane]
      ) {
        invalid("Armed live draft legality does not equal its exact joint mask.");
      }
    }
  }
}

/**
 * Check the cross-field meaning of one schema-checked presentation frame.
 *
 * `frame` is the cloned wire record. This joins product/authority/source kinds,
 * epochs, identity directories, actor-relative axes, scene facts, masks,
 * provenance, researcher inspection data, and incoming/outgoing action and event
 * records. It checks the base actor scene separately from drawing-only corpse
 * additions. The producer remains responsible for simulation and visibility.
 *
 * Returns a mutable derived record containing `actionAxis`, `decisionMask`,
 * `frameId`, `live`, `oracle`, both presentation-key pair lists, and the composed
 * `scene`. Values may borrow records from `frame`. Throws TypeError for a broken
 * contract. It does not yet verify hashes, freeze data, or brand the result;
 * normalizeAuthorizedPresentationFrameV1 owns those final steps.
 *
 * @param {Record<string, any>} frame
 */
function validateSemanticFrame(frame) {
  const source = frame.source;
  const authority = frame.authority;
  const endpoint = frame.current_endpoint;
  if (source.source_authority_epoch !== source.source_revision) {
    invalid("Presentation source authority epoch must equal its revision.");
  }
  if (
    source.source_authorized_endpoint_digest_sha256 !==
    endpoint.authorized_endpoint_digest_sha256
  ) {
    invalid("Presentation source endpoint digest does not join its endpoint.");
  }
  const live = frame.product_kind === "combat_debugger";
  const oracle = authority.authority_kind === "oracle";
  const shared = authority.observation_mode === "shared_obs_visual_union";
  if (live !== frame.presentation_kind.startsWith("live_")) {
    invalid("Presentation product and leaf kinds disagree.");
  }
  const expectedSourceKind = /** @type {Readonly<Record<string, string>>} */ ({
    live_oracle: "live_oracle_frame",
    live_no_shared_obs_agent_pov: "live_no_shared_obs_frame",
    live_shared_obs_agent_pov: "live_shared_obs_visual_union_frame",
    replay_oracle: "replay_oracle_frame",
    replay_no_shared_obs_agent_pov: "replay_no_shared_obs_frame",
    replay_shared_obs_agent_pov: "replay_shared_obs_visual_union_frame",
  })[frame.presentation_kind];
  if (source.source_kind !== expectedSourceKind) {
    invalid("Presentation leaf and source discriminators disagree.");
  }
  if (
    frame.presentation_kind === "replay_oracle" &&
    (source.source_artifact_id !== `${source.episode_id}:replay` ||
      source.source_timeline_id !==
        `${source.source_artifact_id}:timeline:researcher` ||
      source.source_choreography_generation > source.source_cursor_generation)
  ) {
    invalid("Replay Oracle source replay identity/generation is not canonical.");
  }
  /** @type {Record<string, any>} */
  let scene;
  /** @type {Record<string, any> | null} */
  let actionAxis;
  let decisionMask = null;
  let frameId;
  /** @type {{key: string, publicId: string}[]} */
  let researcherPresentationKeyPairs = [];
  if (oracle) {
    if (!frame.presentation_kind.endsWith("oracle")) {
      invalid("Oracle authority is attached to an Agent presentation leaf.");
    }
    scene = endpoint.scene;
    actionAxis = endpoint.action_axis;
    frameId = source.source_frame_id;
    if (
      endpoint.episode_id !== source.episode_id ||
      endpoint.frame_index !== source.source_frame_index ||
      endpoint.frame_id !== source.source_frame_id ||
      endpoint.simulator_step_count !== source.source_simulator_step_count ||
      endpoint.frame_id !== `${source.episode_id}:frame:${source.source_frame_index}`
    ) {
      invalid("Oracle endpoint does not join its source epoch.");
    }
    const directory = /** @type {any[]} */ (endpoint.identity_directory.identities);
    if (directory.length !== 10)
      invalid("Oracle identity directory requires ten rows.");
    directory.forEach((row, index) => {
      if (
        row.team_id !== Math.floor(index / 5) + 1 ||
        row.team_local_slot !== index % 5
      ) {
        invalid("Oracle identity directory lost fixed team topology.");
      }
    });
    requireUnique(
      directory.map((row) => row.public_agent_id),
      "Oracle directory identities",
    );
    const activeIds = directory
      .filter((row) => row.configured_active)
      .map((row) => row.public_agent_id);
    const sceneAgents = /** @type {any[]} */ (scene.agents);
    if (
      sceneAgents.length !== activeIds.length ||
      sceneAgents.some(
        (row, index) =>
          row.public_agent_id !== activeIds[index] || row.relation !== "oracle",
      )
    ) {
      invalid("Oracle scene identities do not equal its active directory.");
    }
    const sceneById = new Map(sceneAgents.map((row) => [row.public_agent_id, row]));
    const classNameById = new Map(
      /** @type {any[]} */ (scene.class_mechanics).map((row) => [
        row.class_id,
        row.class_name,
      ]),
    );
    for (const row of directory) {
      if (
        row.configured_active !== (row.class_id !== null) ||
        (row.class_id === null) !== (row.class_name === null)
      ) {
        invalid("Oracle directory active/class identity is inconsistent.");
      }
      if (!row.configured_active) continue;
      const agent = sceneById.get(row.public_agent_id);
      if (
        !agent ||
        agent.team_id !== row.team_id ||
        agent.class_id !== row.class_id ||
        agent.class_name !== row.class_name ||
        classNameById.get(row.class_id) !== row.class_name
      ) {
        invalid("Oracle directory identity facts do not join its scene.");
      }
    }
    for (const pad of /** @type {any[]} */ (scene.spawn_pads)) {
      if (pad.assigned_public_agent_id === null) continue;
      const row = directory.find(
        (candidate) => candidate.public_agent_id === pad.assigned_public_agent_id,
      );
      if (
        !row?.configured_active ||
        row.team_id !== pad.team_id ||
        row.team_local_slot !== pad.team_local_slot
      ) {
        invalid("Oracle spawn-pad assignment does not join directory topology.");
      }
    }
    if (actionAxis !== null) {
      const targetActions = /** @type {any[]} */ (actionAxis.target_actions);
      const ownerPublicId = actionAxis.owner_public_agent_id;
      const owner = directory.find((row) => row.public_agent_id === ownerPublicId);
      if (!owner?.configured_active) {
        invalid("Oracle action-axis owner must be an active directory identity.");
      }
      const expectedTargets = [
        ...directory.filter((row) => row.team_id === owner.team_id),
        ...directory.filter((row) => row.team_id !== owner.team_id),
      ].map((row) => row.public_agent_id);
      if (
        targetActions
          .slice(1)
          .some((row, index) => row.target_public_agent_id !== expectedTargets[index])
      ) {
        invalid("Oracle action-axis order changed team-local target semantics.");
      }
    }
  } else {
    if (!frame.presentation_kind.endsWith("agent_pov")) {
      invalid("Agent authority is attached to an Oracle presentation leaf.");
    }
    if (
      authority.recipient_public_agent_id !== source.source_recipient_public_agent_id ||
      (authority.observation_mode !== source.source_observation_mode && !live)
    ) {
      invalid("Agent authority does not join its source recipient.");
    }
    const parts = endpoint.parts;
    scene = parts.scene;
    actionAxis = /** @type {Record<string, any>} */ (endpoint.action_axis);
    decisionMask = parts.next_decision_action_mask;
    frameId = source.source_recipient_frame_id;
    if (
      parts.source_episode_id !== source.episode_id ||
      parts.source_frame_index !== source.source_frame_index ||
      parts.source_recipient_frame_id !== source.source_recipient_frame_id ||
      parts.source_simulator_step_count !== source.source_simulator_step_count ||
      parts.recipient_public_agent_id !== authority.recipient_public_agent_id ||
      parts.recipient_presentation_key !== authority.recipient_presentation_key ||
      actionAxis.owner_public_agent_id !== authority.recipient_public_agent_id ||
      actionAxis.owner_presentation_key !== authority.recipient_presentation_key
    ) {
      invalid("Agent endpoint does not join its source and recipient authority.");
    }
    const mode = shared ? "shared-obs-visual-union" : "actor-pov";
    if (
      frameId !==
      `${source.episode_id}:${mode}:${authority.recipient_public_agent_id}:frame:${source.source_frame_index}`
    ) {
      invalid("Agent recipient frame identity is not canonical.");
    }
    const sceneAgents = /** @type {any[]} */ (scene.agents);
    const targetActions = /** @type {any[]} */ (actionAxis.target_actions);
    const selfRows = sceneAgents.filter((row) => row.relation === "self");
    if (
      selfRows.length !== 1 ||
      selfRows[0].public_agent_id !== authority.recipient_public_agent_id ||
      selfRows[0].presentation_key !== authority.recipient_presentation_key
    ) {
      invalid("Agent scene must contain exactly its fixed recipient self row.");
    }
    const targetIds = targetActions.slice(1).map((row) => row.target_public_agent_id);
    const recipientTeam = selfRows[0].team_id;
    const recipientTargetIndex = targetIds.indexOf(authority.recipient_public_agent_id);
    if (recipientTargetIndex < 0 || recipientTargetIndex >= 5) {
      invalid("Agent recipient must remain in its same-team target block.");
    }
    for (const agent of sceneAgents) {
      const targetIndex = targetIds.indexOf(agent.public_agent_id);
      if (targetIndex < 0) {
        invalid("Agent scene identity lies outside its action axis.");
      }
      const expectedRelation =
        agent.public_agent_id === authority.recipient_public_agent_id
          ? "self"
          : targetIndex < 5
            ? "ally"
            : "opponent";
      const expectedTeam =
        targetIndex < 5 ? recipientTeam : recipientTeam === 1 ? 2 : 1;
      if (agent.relation !== expectedRelation || agent.team_id !== expectedTeam) {
        invalid("Agent scene relation/team does not join its target-axis block.");
      }
      if (
        /** @type {any[]} */ (agent.statuses).some(
          (status) => status.direct_sources.length !== 0,
        )
      ) {
        invalid("Agent status cannot disclose direct source identities.");
      }
    }
    if (shared) {
      const sources = /** @type {any[]} */ (parts.authorized_sensor_sources);
      const provenanceRows = /** @type {any[]} */ (parts.agent_observation_provenance);
      /**
       * Compare two checked SharedObs source records in canonical order.
       *
       * `left` and `right` are source records from the enclosing actor endpoint. Returns
       * true when `left` sorts earlier: the recipient's source comes first, followed by
       * other sources in public-agent-ID string order. It does not sort or mutate the
       * input list.
       *
       * @param {any} left @param {any} right
       */
      const sourceSortsBefore = (left, right) => {
        const leftRank = left.source_kind === "recipient_base" ? 0 : 1;
        const rightRank = right.source_kind === "recipient_base" ? 0 : 1;
        return (
          leftRank < rightRank ||
          (leftRank === rightRank &&
            left.source_public_agent_id < right.source_public_agent_id)
        );
      };
      if (
        sources.length === 0 ||
        sources.some(
          (source, index) =>
            index > 0 && !sourceSortsBefore(sources[index - 1], source),
        ) ||
        sources.filter((source) => source.source_kind === "recipient_base").length !== 1
      ) {
        invalid("SharedObs authorized sensor sources are not canonical.");
      }
      requireUnique(
        sources.map((source) => source.source_public_agent_id),
        "SharedObs authorized sensor public identities",
      );
      requireUnique(
        sources.map((source) => source.source_presentation_key),
        "SharedObs authorized sensor presentation keys",
      );
      const sourceById = new Map(
        sources.map((source) => [source.source_public_agent_id, source]),
      );
      const sceneById = new Map(
        sceneAgents.map((agent) => [agent.public_agent_id, agent]),
      );
      if (
        provenanceRows.length !== sceneAgents.length ||
        provenanceRows.some(
          (row, index) =>
            row.agent_public_agent_id !== sceneAgents[index].public_agent_id,
        )
      ) {
        invalid("SharedObs provenance must exactly cover scene order.");
      }
      for (const sensor of sources) {
        const targetIndex = targetIds.indexOf(sensor.source_public_agent_id);
        const sourceAgent = sceneById.get(sensor.source_public_agent_id);
        const sourceProvenance = provenanceRows.find(
          (row) => row.agent_public_agent_id === sensor.source_public_agent_id,
        );
        if (
          targetIndex < 0 ||
          !sourceAgent ||
          sourceAgent.presentation_key !== sensor.source_presentation_key ||
          !(
            /** @type {any[]} */ (sourceProvenance?.observation_sources ?? []).some(
              (row) => structurallyEqual(row, sensor),
            )
          ) ||
          (sensor.source_kind === "recipient_base" &&
            (sensor.source_public_agent_id !== authority.recipient_public_agent_id ||
              sensor.source_presentation_key !==
                authority.recipient_presentation_key)) ||
          (sensor.source_kind === "shared_sensor_source" &&
            (sensor.source_public_agent_id === authority.recipient_public_agent_id ||
              targetIndex >= 5 ||
              sourceAgent.relation !== "ally"))
        ) {
          invalid("SharedObs sensor source lies outside recipient/teammate authority.");
        }
      }
      for (const provenance of provenanceRows) {
        const observationSources = /** @type {any[]} */ (
          provenance.observation_sources
        );
        if (
          observationSources.length === 0 ||
          observationSources.some(
            (source, index) =>
              index > 0 && !sourceSortsBefore(observationSources[index - 1], source),
          ) ||
          new Set(observationSources.map((source) => source.source_public_agent_id))
            .size !== observationSources.length ||
          observationSources.some(
            (source) =>
              !structurallyEqual(sourceById.get(source.source_public_agent_id), source),
          ) ||
          !targetIds.includes(provenance.agent_public_agent_id) ||
          !sceneAgents.some(
            (row) =>
              row.public_agent_id === provenance.agent_public_agent_id &&
              row.presentation_key === provenance.agent_presentation_key,
          )
        ) {
          invalid("SharedObs observation provenance does not join its scene and axis.");
        }
      }
    }
    validateDecisionMask(decisionMask, "Agent next-decision mask");
    validateAgentPrivacy(frame);
    researcherPresentationKeyPairs = live
      ? validateLiveResearcherSpace(frame)
      : validateReplayResearcherSpace(frame);
  }
  // The endpoint scene is immutable actor-input authority. The additive corpse
  // projection may participate in painting and ordinary corpse inspection, but
  // it must never turn an axis-only target into a visible decision target.
  const actorInputScene = scene;
  const renderedScene = oracle
    ? actorInputScene
    : composeLocalOracleCorpseOverlay(frame, actorInputScene, shared);
  validateAuthorizedScene(actorInputScene);
  if (renderedScene !== actorInputScene) validateAuthorizedScene(renderedScene);
  const sceneAgents = /** @type {any[]} */ (renderedScene.agents);
  requireUnique(
    sceneAgents.map((row) => row.presentation_key),
    "Scene presentation keys",
  );
  requireUnique(
    sceneAgents.map((row) => row.public_agent_id),
    "Scene public identities",
  );
  if (actionAxis !== null) validateActionAxis(actionAxis);
  if (live && oracle) {
    const wrapper = frame.live_inspection.inspection;
    if (wrapper.inspection_kind === "editable_live_draft") {
      validateLivePendingJointAction(
        frame.pending_joint_action,
        sceneAgents,
        source.source_simulator_step_count,
        wrapper.draft,
        "Live Oracle Pending Joint Action",
      );
    } else if (frame.pending_joint_action !== null) {
      invalid("Scripted Oracle presentation cannot carry pending joint intent.");
    }
  }
  const presentationKeyPairs = validatePresentationKeyGraph(frame, {
    excludedRootFields: !oracle ? new Set(["researcher_space"]) : new Set(),
  });

  const incomingIndex =
    source.source_frame_index === 0 ? null : source.source_frame_index - 1;
  if ((frame.latest_events === null) !== (incomingIndex === null)) {
    invalid("Latest Events presence does not match the incoming epoch.");
  }
  if ((frame.latest_transition === null) !== (incomingIndex === null)) {
    invalid("Latest Transition presence does not match the incoming epoch.");
  }
  if (frame.latest_events !== null) validateLatestEvents(frame.latest_events);
  if (frame.latest_transition !== null) {
    const prefix = oracle
      ? source.episode_id
      : `${source.episode_id}:${shared ? "shared-obs-visual-union" : "actor-pov"}:${authority.recipient_public_agent_id}`;
    validateLatestTransition(frame.latest_transition, prefix, source.episode_id);
    if (frame.latest_transition.incoming_transition_index !== incomingIndex) {
      invalid("Latest Transition does not enter the current source frame.");
    }
  }
  validateIncomingStateMatrix(frame, source, endpoint, oracle, shared);
  validateAgentVisualStateMatrix(
    frame,
    source,
    endpoint,
    oracle,
    shared,
    renderedScene,
  );
  const technical = frame.technical_frame;
  const expectedIncomingId =
    source.source_frame_index === 0
      ? null
      : oracle
        ? `${source.episode_id}:transition:${source.source_frame_index - 1}`
        : `${source.episode_id}:${shared ? "shared-obs-visual-union" : "actor-pov"}:${authority.recipient_public_agent_id}:transition:${source.source_frame_index - 1}`;
  if (
    (technical.frame_index ??
      technical.evaluation_frame_index ??
      technical.recipient_frame_index) !== source.source_frame_index ||
    technical.simulator_step_count !== source.source_simulator_step_count ||
    (technical.incoming_transition_id ??
      technical.incoming_recipient_transition_id ??
      null) !== expectedIncomingId ||
    (live && technical.episode_id !== source.episode_id)
  ) {
    invalid("Technical Frame does not join the source epoch.");
  }
  if (
    frame.presentation_kind === "replay_oracle" &&
    (technical.artifact_digest_prefix !==
      source.source_artifact_digest_sha256.slice(0, 12) ||
      technical.recorded_ordinary_movement_distance_scale !==
        source.source_recorded_ordinary_movement_distance_scale)
  ) {
    invalid("Replay Oracle Technical Frame does not join artifact provenance.");
  }
  if (!live) {
    validateUpcomingStateMatrix(frame, source, endpoint, oracle, shared);
  }
  validateInspectionStateMatrix(
    frame,
    source,
    endpoint,
    actionAxis,
    actorInputScene,
    live,
    oracle,
    shared,
  );
  return {
    actionAxis,
    decisionMask,
    frameId,
    live,
    oracle,
    presentationKeyPairs,
    researcherPresentationKeyPairs,
    scene: renderedScene,
  };
}

/**
 * Validate, copy, and freeze one authorized presentation wire frame.
 *
 * `value` is an untrusted version-1 record for one of the six supported live/replay
 * and Oracle/actor leaves. The record must use plain data properties, the exact
 * schema fields, coherent identities/epochs/scenes/actions/events, and matching
 * endpoint, overlay, and presentation-key hashes. Match-summary facts are checked
 * when present. Hashes establish consistency with supplied identities and data;
 * the trusted Python producer remains the authority for permitted information.
 *
 * Resolves to a new deeply frozen record containing the validated wire fields and
 * renderer aliases such as `scene`, `frame_id`, `action_axis`, and `decision_mask`.
 * The root is remembered by identity for later guards. The input is unchanged;
 * cloning the result loses that remembered identity. This performs local Web
 * Crypto work, not network or filesystem I/O. Rejects with TypeError on malformed
 * or inconsistent data; missing/failing Web Crypto also prevents completion.
 *
 * @param {unknown} value
 * @returns {Promise<Readonly<Record<string, any>>>}
 */
export async function normalizeAuthorizedPresentationFrameV1(value) {
  const rootSnapshot = snapshotRecord(value, "Authorized presentation frame");
  if (!PRESENTATION_KINDS.has(rootSnapshot.presentation_kind)) {
    invalid("Authorized presentation has an unknown leaf discriminator.");
  }
  const frame = validateSchema(
    value,
    AUTHORIZED_PRESENTATION_SCHEMA_V1,
    "Authorized presentation frame",
  );
  const semantic = validateSemanticFrame(frame);
  const match = frame.match_summary;
  if (match != null) {
    const sides = match.teams.map(
      (/** @type {Record<string, any>} */ team) => team.display_side ?? null,
    );
    if (
      !(
        sides.every((/** @type {unknown} */ side) => side === null) ||
        (sides[0] === "left" && sides[1] === "right") ||
        (sides[0] === "right" && sides[1] === "left")
      )
    ) {
      invalid("Match display sides must be opposite or both unavailable.");
    }
    const roster =
      frame.researcher_space?.roster_agents ??
      frame.current_endpoint?.scene?.agents ??
      [];
    const deaths = match.deaths ?? [];
    if (
      new Set(
        deaths.map((/** @type {Record<string, any>} */ death) => death.public_agent_id),
      ).size !== deaths.length ||
      deaths.some(
        (/** @type {Record<string, any>} */ death) =>
          !roster.some(
            (/** @type {Record<string, any>} */ agent) =>
              agent.public_agent_id === death.public_agent_id &&
              agent.class_id === death.class_id &&
              agent.team_id === death.team_id,
          ),
      )
    ) {
      invalid("Match deaths must join unique identities in the researcher roster.");
    }
    if (deaths.length > 0) {
      const rosterById = new Map(
        roster.map((/** @type {Record<string, any>} */ agent) => [
          agent.public_agent_id,
          agent,
        ]),
      );
      const directory =
        frame.researcher_space?.identity_directory ??
        frame.current_endpoint.identity_directory;
      const slotById = new Map(
        directory.identities.map((/** @type {Record<string, any>} */ identity) => [
          identity.public_agent_id,
          (identity.team_id - 1) * 5 + identity.team_local_slot,
        ]),
      );
      for (const death of deaths) {
        const killingTeam = death.killing_team_id ?? null;
        const contributors = death.contributors ?? null;
        if ((killingTeam === null) !== (contributors === null)) {
          invalid(
            "Death attribution must provide both killing team and contributors, or neither.",
          );
        }
        if (contributors === null) continue;
        let previousSlot = -1;
        if (killingTeam === death.team_id || contributors.length === 0) {
          invalid("Death contributors must identify a nonempty opposing team.");
        }
        for (const contributor of contributors) {
          const agent = rosterById.get(contributor.public_agent_id);
          const slot = slotById.get(contributor.public_agent_id);
          if (
            !agent ||
            agent.class_id !== contributor.class_id ||
            agent.team_id !== contributor.team_id ||
            contributor.team_id !== killingTeam ||
            slot === undefined ||
            slot <= previousSlot
          ) {
            invalid(
              "Death contributors must join unique researcher identities in numeric agent order.",
            );
          }
          previousSlot = slot;
        }
      }
    }
    if (
      match.episode_id !== frame.source.episode_id ||
      match.source_frame_index !== frame.source.source_frame_index ||
      match.simulator_step_count !== frame.source.source_simulator_step_count ||
      match.teams[0].team_id !== 1 ||
      match.teams[1].team_id !== 2 ||
      (match.task_mode === 0 &&
        (match.score_threshold !== 0 ||
          match.scores.some((/** @type {number} */ score) => score !== 0) ||
          match.outcome !== "not_applicable")) ||
      (match.task_mode === 1 &&
        (match.score_threshold === 0 || match.outcome === "not_applicable"))
    ) {
      invalid("Match summary does not join the current task and source epoch.");
    }
  }
  await Promise.all([
    verifyPresentationKeyDerivation(frame, semantic.presentationKeyPairs),
    verifyPresentationKeyPairs(
      frame.source.source_session_id,
      "oracle",
      null,
      semantic.researcherPresentationKeyPairs,
    ),
    verifyAuthorizedEndpointDigest(frame),
    verifyLocalOracleCorpseOverlayDigest(frame),
  ]);
  const inspection = semantic.live
    ? frame.live_inspection.inspection
    : frame.replay_inspection;
  const normalized = deepFreeze({
    ...frame,
    viewer_mode: semantic.live ? "live" : "replay",
    session_id: frame.source.source_session_id,
    revision: frame.source.source_revision,
    authority_epoch: frame.source.source_authority_epoch,
    episode_id: frame.source.episode_id,
    frame_index: frame.source.source_frame_index,
    frame_id: semantic.frameId,
    simulator_step_count: frame.source.source_simulator_step_count,
    scene: semantic.scene,
    action_axis: semantic.actionAxis,
    decision_mask: semantic.decisionMask,
    inspection,
    recipient_public_agent_id: semantic.oracle
      ? null
      : frame.authority.recipient_public_agent_id,
    recipient_presentation_key: semantic.oracle
      ? null
      : frame.authority.recipient_presentation_key,
  });
  NORMALIZED_PRESENTATION_ROOTS.add(normalized);
  return normalized;
}

/**
 * Test whether this exact object completed presentation normalization here.
 *
 * `value` may be anything. Returns true only for a root object recorded by this
 * module after successful normalization. A look-alike or cloned record returns
 * false. This checks object identity; it does not rerun schema or hash checks.
 *
 * @param {unknown} value
 * @returns {value is Readonly<Record<string, any>>}
 */
export function isNormalizedAuthorizedPresentationFrameV1(value) {
  return (
    typeof value === "object" &&
    value !== null &&
    NORMALIZED_PRESENTATION_ROOTS.has(value)
  );
}

/**
 * Validate the one supported unavailable-audience API error payload.
 *
 * `value` must be a plain version-1 record with exactly the schema version,
 * `audience_unavailable` error code, and the canonical unavailable-audience
 * message. Returns a new deeply frozen copy. Throws TypeError for unknown fields,
 * wrong values, or malformed properties. It performs no request and does not
 * change UI state.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
export function normalizePresentationApiErrorV1(value) {
  const error = snapshotRecord(value, "Presentation API error");
  const expected = ["error_code", "message", "schema_version"];
  const keys = Object.keys(error).sort();
  if (
    keys.length !== expected.length ||
    keys.some((key, index) => key !== expected[index]) ||
    error.schema_version !== 1 ||
    error.error_code !== "audience_unavailable" ||
    error.message !== "Authorized presentation is unavailable for the active audience."
  ) {
    invalid("Presentation API error is not the exact V1 unavailable root.");
  }
  return deepFreeze({ ...error });
}

/**
 * Snapshot a plain record and require exactly the named own fields.
 *
 * `value` is the candidate, `expected` lists every permitted field name, and
 * `label` identifies the record in TypeError messages. Returns a new mutable
 * null-prototype record with borrowed field values. Accessors and malformed
 * records are rejected before field reads; unknown or missing keys also fail.
 * This is a shallow snapshot, not recursive field validation.
 *
 * @param {unknown} value
 * @param {readonly string[]} expected
 * @param {string} label
 */
function exactRecord(value, expected, label) {
  const record = snapshotRecord(value, label);
  const keys = Object.keys(record).sort();
  const canonical = [...expected].sort();
  if (
    keys.length !== canonical.length ||
    keys.some((key, index) => key !== canonical[index])
  ) {
    invalid(`${label} has unknown or missing fields.`);
  }
  return record;
}

/**
 * Require a safe nonnegative integer identity/count field.
 *
 * `value` is returned unchanged when it is a number accepted by
 * Number.isSafeInteger and is at least zero. Otherwise throws TypeError naming
 * `label`. Strings are not converted and no default is supplied.
 *
 * @param {unknown} value @param {string} label
 */
function exactNonnegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || Number(value) < 0) {
    invalid(`${label} must be a non-negative safe integer.`);
  }
  return Number(value);
}

/**
 * Require a nonempty string without trimming or converting it.
 *
 * Returns `value` unchanged if it is a string with positive length; whitespace
 * alone is accepted. Otherwise throws TypeError naming `label`. This checks
 * presence, not scientific-ID syntax.
 *
 * @param {unknown} value @param {string} label
 */
function exactNonemptyString(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    invalid(`${label} must be a non-empty string.`);
  }
  return value;
}

/**
 * Require the bounded ASCII syntax used by scientific artifact IDs.
 *
 * `value` must be a string of 1 through 512 letters, digits, underscores, periods,
 * colons, or hyphens. Returns it unchanged or throws TypeError naming `label`.
 * No trimming, normalization, or uniqueness check occurs.
 *
 * @param {unknown} value @param {string} label
 */
function exactScientificId(value, label) {
  const text = exactNonemptyString(value, label);
  if (text.length > 512 || !/^[A-Za-z0-9_.:-]+$/u.test(text)) {
    invalid(`${label} must be an exact ScientificId.`);
  }
  return text;
}

/**
 * Require the bounded ASCII syntax used by public agent IDs.
 *
 * `value` must be a string of 1 through 128 letters, digits, underscores, periods,
 * colons, or hyphens. Returns it unchanged or throws TypeError naming `label`.
 * This does not prove that an agent exists in a particular roster.
 *
 * @param {unknown} value @param {string} label
 */
function exactPublicAgentId(value, label) {
  const text = exactNonemptyString(value, label);
  if (text.length > 128 || !/^[A-Za-z0-9_.:-]+$/u.test(text)) {
    invalid(`${label} must be an exact PublicAgentId.`);
  }
  return text;
}

/**
 * Validate and freeze a private SharedObs replay transport frame.
 *
 * `value` is the plain version-1 SharedObs actor replay wire record, with exact
 * fields and an analysis/nonverbose/POV audience. `animateIncoming` defaults to
 * false and must be a boolean; it records caller intent. This checks the session,
 * revision, canonical replay/frame/transition IDs, cursor bounds, captured counts,
 * completion facts, and their researcher-wide artifact-facts joins.
 *
 * Returns a new deeply frozen transport with replay/session/cursor aliases and
 * `animate_incoming`. It adds no scene, HUD, or policy-input authority and does
 * not acquire the joined-pair identity mark. Throws TypeError on malformed data.
 * The outer join checks whether requested incoming animation fits this cursor.
 * No I/O occurs and the input is unchanged.
 *
 * @param {unknown} value
 * @param {boolean} animateIncoming
 * @returns {Readonly<Record<string, any>>}
 */
export function normalizeSharedObsAgentPovReplayTransportV1(
  value,
  animateIncoming = false,
) {
  if (typeof animateIncoming !== "boolean") {
    invalid("Shared replay animation intent must be an exact boolean.");
  }
  const frame = exactRecord(
    value,
    [
      "schema_version",
      "frame_kind",
      "viewer_session_id",
      "revision",
      "artifact_facts",
      "artifact_summary",
      "timeline_id",
      "cursor",
      "preset",
      "verbose",
      "view_mode",
      "public_agent_id",
      "recipient_frame_id",
      "simulator_step_count",
      "incoming_recipient_transition_id",
      "completion",
    ],
    "Private Shared replay transport",
  );
  if (
    frame.schema_version !== 1 ||
    frame.frame_kind !== "shared_obs_agent_pov_replay_viewer" ||
    frame.preset !== "analysis" ||
    frame.verbose !== false ||
    frame.view_mode !== "pov"
  ) {
    invalid("Private Shared replay transport literals are invalid.");
  }
  const sessionId = exactNonemptyString(frame.viewer_session_id, "viewer_session_id");
  if (!/^[A-Za-z0-9_-]{1,128}$/u.test(sessionId)) {
    invalid("Private Shared replay viewer session ID is invalid.");
  }
  exactNonnegativeInteger(frame.revision, "revision");
  const simulatorStep = exactNonnegativeInteger(
    frame.simulator_step_count,
    "simulator_step_count",
  );
  const summary = exactRecord(
    frame.artifact_summary,
    [
      "schema_version",
      "recipient_replay_id",
      "episode_id",
      "public_agent_id",
      "expected_transition_count",
      "captured_transition_count",
      "captured_frame_count",
    ],
    "Private Shared replay summary",
  );
  const expectedCount = exactNonnegativeInteger(
    summary.expected_transition_count,
    "artifact_summary.expected_transition_count",
  );
  const capturedCount = exactNonnegativeInteger(
    summary.captured_transition_count,
    "artifact_summary.captured_transition_count",
  );
  const capturedFrames = exactNonnegativeInteger(
    summary.captured_frame_count,
    "artifact_summary.captured_frame_count",
  );
  const episodeId = exactScientificId(
    summary.episode_id,
    "artifact_summary.episode_id",
  );
  const publicId = exactPublicAgentId(
    summary.public_agent_id,
    "artifact_summary.public_agent_id",
  );
  exactScientificId(
    summary.recipient_replay_id,
    "artifact_summary.recipient_replay_id",
  );
  if (
    summary.schema_version !== 1 ||
    expectedCount === 0 ||
    capturedCount > expectedCount ||
    capturedFrames !== capturedCount + 1 ||
    summary.recipient_replay_id !==
      `${episodeId}:shared-obs-visual-union:${publicId}:replay`
  ) {
    invalid("Private Shared replay summary identity/counts are invalid.");
  }
  const cursor = exactRecord(
    frame.cursor,
    [
      "schema_version",
      "frame_index",
      "final_frame_index",
      "cursor_generation",
      "choreography_generation",
    ],
    "Private Shared replay cursor",
  );
  const frameIndex = exactNonnegativeInteger(cursor.frame_index, "cursor.frame_index");
  const finalIndex = exactNonnegativeInteger(
    cursor.final_frame_index,
    "cursor.final_frame_index",
  );
  const cursorGeneration = exactNonnegativeInteger(
    cursor.cursor_generation,
    "cursor.cursor_generation",
  );
  const choreographyGeneration = exactNonnegativeInteger(
    cursor.choreography_generation,
    "cursor.choreography_generation",
  );
  if (
    cursor.schema_version !== 1 ||
    frameIndex > finalIndex ||
    finalIndex !== capturedCount ||
    choreographyGeneration > cursorGeneration
  ) {
    invalid("Private Shared replay cursor is incoherent.");
  }
  const completion = exactRecord(
    frame.completion,
    [
      "schema_version",
      "episode_id",
      "completion_state",
      "expected_transition_count",
      "captured_transition_count",
      "terminated",
      "truncated",
      "completion_bases",
      "public_end_or_failure_reason",
    ],
    "Private Shared replay completion",
  );
  const completionBases = snapshotArray(
    completion.completion_bases,
    "completion.completion_bases",
  );
  exactScientificId(completion.episode_id, "completion.episode_id");
  const expectedCompletionBases = [];
  if (completion.terminated) expectedCompletionBases.push("task_terminal");
  if (capturedCount === expectedCount) {
    expectedCompletionBases.push("declared_horizon");
  }
  const complete = completion.completion_state === "complete";
  if (
    completion.schema_version !== 1 ||
    completion.episode_id !== episodeId ||
    completion.expected_transition_count !== expectedCount ||
    completion.captured_transition_count !== capturedCount ||
    typeof completion.terminated !== "boolean" ||
    typeof completion.truncated !== "boolean" ||
    (capturedCount === 0 && (completion.terminated || completion.truncated)) ||
    !["complete", "partial", "interrupted", "failed"].includes(
      completion.completion_state,
    ) ||
    completionBases.some(
      (basis) => !["task_terminal", "declared_horizon"].includes(basis),
    ) ||
    new Set(completionBases).size !== completionBases.length ||
    (complete
      ? expectedCompletionBases.length === 0 ||
        !structurallyEqual(completionBases, expectedCompletionBases) ||
        completion.public_end_or_failure_reason !== null
      : expectedCompletionBases.length !== 0 ||
        completionBases.length !== 0 ||
        completion.public_end_or_failure_reason !== "captured_prefix")
  ) {
    invalid("Private Shared replay completion disclosure is invalid.");
  }
  const artifactFacts = normalizeReplayArtifactFactsV1(frame.artifact_facts);
  const canonicalSummary = artifactFacts.artifact_summary;
  const canonicalCompletion = artifactFacts.completion;
  if (
    canonicalSummary.replay_reference.episode_id !== episodeId ||
    canonicalSummary.expected_transition_count !== expectedCount ||
    canonicalSummary.recorded_transition_count !== capturedCount ||
    canonicalSummary.recorded_frame_count !== capturedFrames ||
    canonicalCompletion.episode_id !== completion.episode_id ||
    canonicalCompletion.completion_state !== completion.completion_state ||
    canonicalCompletion.expected_transition_count !==
      completion.expected_transition_count ||
    canonicalCompletion.validated_transition_count !==
      completion.captured_transition_count ||
    canonicalCompletion.terminated !== completion.terminated ||
    canonicalCompletion.truncated !== completion.truncated ||
    !structurallyEqual(canonicalCompletion.completion_bases, completionBases)
  ) {
    invalid(
      "Private Shared replay artifact facts do not join its recipient-local roots.",
    );
  }
  const timelineId = `${episodeId}:shared-obs-visual-union:${publicId}:timeline`;
  const recipientFrameId = `${episodeId}:shared-obs-visual-union:${publicId}:frame:${frameIndex}`;
  const incomingId =
    frameIndex === 0
      ? null
      : `${episodeId}:shared-obs-visual-union:${publicId}:transition:${frameIndex - 1}`;
  exactPublicAgentId(frame.public_agent_id, "public_agent_id");
  exactScientificId(frame.timeline_id, "timeline_id");
  exactScientificId(frame.recipient_frame_id, "recipient_frame_id");
  if (frame.incoming_recipient_transition_id !== null) {
    exactScientificId(
      frame.incoming_recipient_transition_id,
      "incoming_recipient_transition_id",
    );
  }
  if (
    frame.public_agent_id !== publicId ||
    frame.timeline_id !== timelineId ||
    frame.recipient_frame_id !== recipientFrameId ||
    frame.incoming_recipient_transition_id !== incomingId
  ) {
    invalid("Private Shared replay transport identity is not canonical.");
  }
  return deepFreeze({
    ...frame,
    artifact_facts: artifactFacts,
    artifact_summary: { ...summary },
    cursor: { ...cursor },
    completion: { ...completion, completion_bases: completionBases },
    viewer_mode: "replay",
    replay_audience: "actor_pov",
    session_id: sessionId,
    run_generation: choreographyGeneration,
    episode_id: episodeId,
    frame_index: frameIndex,
    simulator_step: simulatorStep,
    transition_id: incomingId,
    animate_incoming: animateIncoming,
  });
}

/**
 * Validate and freeze a private SharedObs replay's complete cursor timeline.
 *
 * `value` is a plain version-1 timeline record with exact fields. The summary,
 * completion, and timeline ID must agree. There must be one row per captured
 * frame, in index order, with canonical recipient-frame and incoming-transition
 * IDs and adjacent simulator ticks. Only the final row carries the endpoint kind;
 * its kind follows the checked completion evidence or captured-prefix status.
 *
 * Returns a new deeply frozen record with copied rows and replay/audience/episode/
 * recipient aliases. Throws TypeError on malformed fields or inconsistent joins.
 * The first tick need not be zero. This does not load frames, verify their content,
 * or join a current frame; joinReplayTransportAndTimelineV1 owns that step.
 *
 * @param {unknown} value
 */
export function normalizeSharedObsAgentPovReplayTimelineTransportV1(value) {
  const timeline = exactRecord(
    value,
    [
      "schema_version",
      "timeline_kind",
      "timeline_id",
      "artifact_summary",
      "final_frame_index",
      "completion",
      "rows",
    ],
    "Private Shared replay timeline",
  );
  if (
    timeline.schema_version !== 1 ||
    timeline.timeline_kind !== "shared_obs_agent_pov"
  ) {
    invalid("Private Shared replay timeline literals are invalid.");
  }
  const summary = exactRecord(
    timeline.artifact_summary,
    [
      "schema_version",
      "recipient_replay_id",
      "episode_id",
      "public_agent_id",
      "expected_transition_count",
      "captured_transition_count",
      "captured_frame_count",
    ],
    "Private Shared replay timeline summary",
  );
  const episodeId = exactScientificId(
    summary.episode_id,
    "artifact_summary.episode_id",
  );
  const publicId = exactPublicAgentId(
    summary.public_agent_id,
    "artifact_summary.public_agent_id",
  );
  const expectedCount = exactNonnegativeInteger(
    summary.expected_transition_count,
    "artifact_summary.expected_transition_count",
  );
  const capturedCount = exactNonnegativeInteger(
    summary.captured_transition_count,
    "artifact_summary.captured_transition_count",
  );
  const capturedFrames = exactNonnegativeInteger(
    summary.captured_frame_count,
    "artifact_summary.captured_frame_count",
  );
  exactScientificId(
    summary.recipient_replay_id,
    "artifact_summary.recipient_replay_id",
  );
  if (
    summary.schema_version !== 1 ||
    expectedCount === 0 ||
    capturedCount > expectedCount ||
    capturedFrames !== capturedCount + 1 ||
    summary.recipient_replay_id !==
      `${episodeId}:shared-obs-visual-union:${publicId}:replay`
  ) {
    invalid("Private Shared replay timeline summary is invalid.");
  }
  const completion = exactRecord(
    timeline.completion,
    [
      "schema_version",
      "episode_id",
      "completion_state",
      "expected_transition_count",
      "captured_transition_count",
      "terminated",
      "truncated",
      "completion_bases",
      "public_end_or_failure_reason",
    ],
    "Private Shared replay timeline completion",
  );
  const completionBases = snapshotArray(
    completion.completion_bases,
    "completion.completion_bases",
  );
  exactScientificId(completion.episode_id, "completion.episode_id");
  const expectedBases = [];
  if (completion.terminated) expectedBases.push("task_terminal");
  if (capturedCount === expectedCount) expectedBases.push("declared_horizon");
  const complete = completion.completion_state === "complete";
  if (
    completion.schema_version !== 1 ||
    completion.episode_id !== episodeId ||
    completion.expected_transition_count !== expectedCount ||
    completion.captured_transition_count !== capturedCount ||
    typeof completion.terminated !== "boolean" ||
    typeof completion.truncated !== "boolean" ||
    (capturedCount === 0 && (completion.terminated || completion.truncated)) ||
    !["complete", "partial", "interrupted", "failed"].includes(
      completion.completion_state,
    ) ||
    completionBases.some(
      (basis) => !["task_terminal", "declared_horizon"].includes(basis),
    ) ||
    new Set(completionBases).size !== completionBases.length ||
    (complete
      ? expectedBases.length === 0 ||
        !structurallyEqual(completionBases, expectedBases) ||
        completion.public_end_or_failure_reason !== null
      : expectedBases.length !== 0 ||
        completionBases.length !== 0 ||
        completion.public_end_or_failure_reason !== "captured_prefix")
  ) {
    invalid("Private Shared replay timeline completion is invalid.");
  }
  const finalFrameIndex = exactNonnegativeInteger(
    timeline.final_frame_index,
    "final_frame_index",
  );
  const timelineId = exactScientificId(timeline.timeline_id, "timeline_id");
  const expectedTimelineId = `${episodeId}:shared-obs-visual-union:${publicId}:timeline`;
  const rows = snapshotArray(timeline.rows, "Private Shared replay timeline rows");
  const endpointKind = complete
    ? completionBases.length === 2
      ? "task_terminal_and_declared_horizon"
      : completionBases[0]
    : "captured_prefix";
  if (
    finalFrameIndex !== capturedCount ||
    timelineId !== expectedTimelineId ||
    rows.length !== capturedFrames
  ) {
    invalid("Private Shared replay timeline identity/counts are invalid.");
  }
  /** @type {number | null} */
  let previousStep = null;
  const normalizedRows = rows.map((value, index) => {
    const row = exactRecord(
      value,
      [
        "frame_index",
        "recipient_frame_id",
        "simulator_step_count",
        "incoming_recipient_transition_id",
        "endpoint_kind",
      ],
      `Private Shared replay timeline row ${index}`,
    );
    const frameIndex = exactNonnegativeInteger(
      row.frame_index,
      `rows[${index}].frame_index`,
    );
    const simulatorStep = exactNonnegativeInteger(
      row.simulator_step_count,
      `rows[${index}].simulator_step_count`,
    );
    exactScientificId(row.recipient_frame_id, `rows[${index}].recipient_frame_id`);
    if (row.incoming_recipient_transition_id !== null) {
      exactScientificId(
        row.incoming_recipient_transition_id,
        `rows[${index}].incoming_recipient_transition_id`,
      );
    }
    const expectedFrameId = `${episodeId}:shared-obs-visual-union:${publicId}:frame:${index}`;
    const expectedIncoming =
      index === 0
        ? null
        : `${episodeId}:shared-obs-visual-union:${publicId}:transition:${index - 1}`;
    const expectedEndpoint = index === finalFrameIndex ? endpointKind : "none";
    if (
      frameIndex !== index ||
      row.recipient_frame_id !== expectedFrameId ||
      row.incoming_recipient_transition_id !== expectedIncoming ||
      row.endpoint_kind !== expectedEndpoint ||
      (previousStep !== null && simulatorStep !== previousStep + 1)
    ) {
      invalid("Private Shared replay timeline row is not canonical and adjacent.");
    }
    previousStep = simulatorStep;
    return { ...row };
  });
  return deepFreeze({
    ...timeline,
    artifact_summary: { ...summary },
    completion: { ...completion, completion_bases: completionBases },
    rows: normalizedRows,
    viewer_mode: "replay",
    replay_audience: "actor_pov",
    episode_id: episodeId,
    public_agent_id: publicId,
  });
}

/**
 * Require a previously snapshotted record to have exactly the expected keys.
 *
 * `value` is the safe shallow snapshot; `expected` lists the permitted own names;
 * `label` is included in TypeError messages. Returns undefined if the sorted name
 * sets agree and throws otherwise. Input order is irrelevant and neither input
 * is changed. This does not inspect values or create the snapshot.
 *
 * @param {Record<string, any>} value @param {readonly string[]} expected @param {string} label
 */
function requireExactSnapshotKeys(value, expected, label) {
  const actual = Object.keys(value).sort();
  const canonical = [...expected].sort();
  if (
    actual.length !== canonical.length ||
    actual.some((key, index) => key !== canonical[index])
  ) {
    invalid(`${label} has unknown or missing fields.`);
  }
}

/**
 * Compare two well-formed scalar identities and classify a response race.
 *
 * `left` and `right` must both be nonempty strings, finite numbers, or null, and
 * must have the same JavaScript type. Invalid scalars throw TypeError naming
 * `label`. Otherwise unequal values throw PresentationJoinMismatchError, which
 * the caller may handle as a bounded GET retry. Equal values return undefined.
 *
 * Object.is defines equality, including the distinction between positive and
 * negative zero. This does not issue a request or retry it.
 *
 * @param {unknown} left
 * @param {unknown} right
 * @param {string} label
 */
function requireJoinEqual(left, right, label) {
  /**
   * Recognize the scalar forms allowed by the enclosing identity comparison.
   *
   * `value` may be any value. Returns true for a nonempty string, finite number, or
   * null. It does not check nonnegative integer bounds or scientific-ID syntax;
   * the appropriate field validators own those restrictions.
   *
   * @param {unknown} value
   */
  const valid = (value) =>
    (typeof value === "string" && value.length > 0) ||
    (typeof value === "number" && Number.isFinite(value)) ||
    value === null;
  if (!valid(left) || !valid(right) || typeof left !== typeof right) {
    invalid(`${label} contains a malformed identity scalar.`);
  }
  if (!Object.is(left, right)) joinMismatch(`${label} raced between GET responses.`);
}

/**
 * Compare response identities before validating their nested presentation data.
 *
 * `rawValue` and `presentationValue` are untrusted wire candidates. This takes
 * safe shallow snapshots, requires the appropriate exact root fields, checks
 * source/authority literals and scalar syntax, then joins session, epoch, frame,
 * audience, and relevant replay artifact identities. Returns the raw frame-kind
 * string when the pair agrees.
 *
 * Malformed data throws TypeError. A mismatch between otherwise valid compared
 * identities throws PresentationJoinMismatchError for the caller's bounded GET
 * retry. The identity phase does not read nested endpoint, event, technical-frame,
 * or inspection payloads; those are validated only after the pair agrees. It
 * does not prove either full payload is valid or perform any request.
 *
 * @param {unknown} rawValue
 * @param {unknown} presentationValue
 */
function preflightTransportPresentationIdentity(rawValue, presentationValue) {
  const raw = snapshotRecord(rawValue, "Raw transport candidate");
  const presentation = snapshotRecord(
    presentationValue,
    "Authorized presentation candidate",
  );
  const source = snapshotRecord(presentation.source, "Presentation source identity");
  const authority = snapshotRecord(
    presentation.authority,
    "Presentation authority identity",
  );
  const expectedFrameKind = /** @type {Readonly<Record<string, string>>} */ ({
    live_oracle: "researcher_live_debugger",
    live_no_shared_obs_agent_pov: "actor_pov_live_debugger",
    live_shared_obs_agent_pov: "shared_obs_agent_pov_live_debugger",
    replay_oracle: "researcher_replay_viewer",
    replay_no_shared_obs_agent_pov: "actor_pov_replay_viewer",
    replay_shared_obs_agent_pov: "shared_obs_agent_pov_replay_viewer",
  })[presentation.presentation_kind];
  if (!PRESENTATION_KINDS.has(presentation.presentation_kind)) {
    invalid("Authorized presentation has an unknown leaf discriminator.");
  }
  if (!Object.hasOwn(RAW_FRAME_KEYS, raw.frame_kind)) {
    invalid("Raw transport has an unknown frame discriminator.");
  }
  const rootSchema = selectCanonicalSchema(
    presentation,
    AUTHORIZED_PRESENTATION_SCHEMA_V1,
    "Authorized presentation candidate",
  );
  requireExactSnapshotKeys(
    presentation,
    Object.keys(rootSchema.properties),
    "Authorized presentation candidate",
  );
  validateSchema(source, rootSchema.properties.source, "Presentation source identity");
  validateSchema(
    authority,
    rootSchema.properties.authority,
    "Presentation authority identity",
  );
  const live = presentation.presentation_kind.startsWith("live_");
  const oracle = presentation.presentation_kind.endsWith("oracle");
  const shared =
    presentation.presentation_kind === "live_shared_obs_agent_pov" ||
    presentation.presentation_kind === "replay_shared_obs_agent_pov";
  const expectedSourceKind = /** @type {Readonly<Record<string, string>>} */ ({
    live_oracle: "live_oracle_frame",
    live_no_shared_obs_agent_pov: "live_no_shared_obs_frame",
    live_shared_obs_agent_pov: "live_shared_obs_visual_union_frame",
    replay_oracle: "replay_oracle_frame",
    replay_no_shared_obs_agent_pov: "replay_no_shared_obs_frame",
    replay_shared_obs_agent_pov: "replay_shared_obs_visual_union_frame",
  })[presentation.presentation_kind];
  if (
    presentation.schema_version !== 1 ||
    presentation.product_kind !== (live ? "combat_debugger" : "replay_viewer") ||
    source.source_kind !== expectedSourceKind ||
    source.source_authority_epoch !== source.source_revision ||
    authority.authority_kind !== (oracle ? "oracle" : "agent_pov") ||
    (!oracle &&
      authority.observation_mode !==
        (shared ? "shared_obs_visual_union" : "no_shared_obs"))
  ) {
    invalid("Presentation identity literals are internally inconsistent.");
  }
  requireExactSnapshotKeys(
    raw,
    live
      ? systemTransportKeys(RAW_FRAME_KEYS[raw.frame_kind], raw)
      : RAW_FRAME_KEYS[raw.frame_kind],
    "Raw transport candidate",
  );
  if (raw.frame_kind !== expectedFrameKind) {
    joinMismatch("Raw transport and presentation kinds raced between GET responses.");
  }
  if (
    raw.schema_version !== (live ? 2 : 1) ||
    raw.view_mode !== (oracle ? "researcher" : "pov") ||
    raw.preset !== "analysis" ||
    typeof raw.verbose !== "boolean"
  ) {
    invalid("Raw transport identity literals are malformed.");
  }
  exactNonnegativeInteger(raw.revision, "Raw transport revision");
  const rawSessionId = live ? raw.session_id : raw.viewer_session_id;
  if (
    typeof rawSessionId !== "string" ||
    !/^[A-Za-z0-9_-]{1,128}$/u.test(rawSessionId)
  ) {
    invalid("Raw transport session ID is malformed.");
  }
  requireJoinEqual(
    source.source_session_id,
    rawSessionId,
    "Raw/presentation session identity",
  );
  requireJoinEqual(
    source.source_revision,
    raw.revision,
    "Raw/presentation revision identity",
  );
  requireJoinEqual(
    source.source_authority_epoch,
    raw.revision,
    "Raw/presentation authority epoch",
  );
  if (live) {
    const configuration = snapshotRecord(
      raw.combat_configuration,
      "Live combat configuration identity",
    );
    requireExactSnapshotKeys(
      configuration,
      ["execution_information_mode", "team_a_controller", "team_b_controller"],
      "Live combat configuration identity",
    );
    // Both teams accept the same five controllers; the three reactive ones
    // (ALPHA, BETA and GAMMA) need SharedObs whichever team uses them.
    if (
      !isTeamController(configuration.team_a_controller) ||
      !isTeamController(configuration.team_b_controller) ||
      !["shared_obs", "no_shared_obs"].includes(
        configuration.execution_information_mode,
      ) ||
      ((requiresSharedObs(configuration.team_a_controller) ||
        requiresSharedObs(configuration.team_b_controller)) &&
        configuration.execution_information_mode !== "shared_obs") ||
      (!oracle &&
        configuration.execution_information_mode !==
          (shared ? "shared_obs" : "no_shared_obs"))
    ) {
      invalid("Live combat configuration identity is inconsistent.");
    }
    const hud = shared ? null : snapshotRecord(raw.hud, "Live transport HUD identity");
    const submissionScope = shared
      ? raw.pending_submission_scope
      : /** @type {Record<string, any>} */ (hud).pending_submission_scope;
    if (!["joint_turn", "scripted_playback"].includes(submissionScope)) {
      invalid("Live raw pending submission scope is malformed.");
    }
    exactNonnegativeInteger(raw.run_generation, "run_generation");
    exactNonnegativeInteger(raw.revision, "revision");
    exactNonnegativeInteger(raw.frame_index, "frame_index");
    exactNonnegativeInteger(raw.simulator_step_count, "simulator_step_count");
    exactScientificId(raw.episode_id, "episode_id");
    exactScientificId(raw.frame_id, "frame_id");
    if (raw.frame_id !== `${raw.episode_id}:frame:${raw.frame_index}`) {
      invalid("Live raw frame identity is not canonical.");
    }
    requireJoinEqual(
      source.source_run_generation,
      raw.run_generation,
      "Live run generation",
    );
    requireJoinEqual(source.episode_id, raw.episode_id, "Live episode identity");
    requireJoinEqual(source.source_frame_index, raw.frame_index, "Live frame index");
    requireJoinEqual(
      source.source_simulator_step_count,
      raw.simulator_step_count,
      "Live simulator step",
    );
    requireJoinEqual(
      source.source_submission_scope,
      submissionScope,
      "Live pending submission scope",
    );
    if (presentation.presentation_kind === "live_oracle") {
      if (
        source.source_frame_id !==
        `${source.episode_id}:frame:${source.source_frame_index}`
      ) {
        invalid("Live Oracle presentation source frame ID is not canonical.");
      }
      requireJoinEqual(source.source_frame_id, raw.frame_id, "Live Oracle frame ID");
    } else {
      const recipient = shared
        ? raw.recipient_public_agent_id
        : /** @type {Record<string, any>} */ (hud).controlled_public_agent_id;
      const localMode = shared ? "shared-obs-visual-union" : "actor-pov";
      if (
        source.source_recipient_frame_id !==
        `${source.episode_id}:${localMode}:${source.source_recipient_public_agent_id}:frame:${source.source_frame_index}`
      ) {
        invalid("Live Agent presentation source frame ID is not canonical.");
      }
      exactPublicAgentId(recipient, "hud.controlled_public_agent_id");
      requireJoinEqual(
        source.source_recipient_public_agent_id,
        recipient,
        "Live Agent source recipient",
      );
      requireJoinEqual(
        authority.recipient_public_agent_id,
        recipient,
        "Live Agent authority recipient",
      );
      if (shared) {
        requireJoinEqual(
          source.source_recipient_frame_id,
          raw.recipient_frame_id,
          "Live SharedObs recipient frame ID",
        );
      }
    }
    return raw.frame_kind;
  }
  const cursor = snapshotRecord(raw.cursor, "Replay raw cursor identity");
  const summary = snapshotRecord(raw.artifact_summary, "Replay raw summary identity");
  for (const [name, value] of Object.entries({
    revision: raw.revision,
    simulator_step_count: raw.simulator_step_count,
    frame_index: cursor.frame_index,
    final_frame_index: cursor.final_frame_index,
    cursor_generation: cursor.cursor_generation,
    choreography_generation: cursor.choreography_generation,
  })) {
    exactNonnegativeInteger(value, name);
  }
  if (
    cursor.frame_index > cursor.final_frame_index ||
    cursor.choreography_generation > cursor.cursor_generation ||
    source.source_frame_index > source.source_final_frame_index
  ) {
    invalid("Replay cursor/source identity is internally incoherent.");
  }
  requireJoinEqual(source.source_frame_index, cursor.frame_index, "Replay frame index");
  requireJoinEqual(
    source.source_final_frame_index,
    cursor.final_frame_index,
    "Replay final frame index",
  );
  requireJoinEqual(
    source.source_simulator_step_count,
    raw.simulator_step_count,
    "Replay simulator step",
  );
  if (presentation.presentation_kind === "replay_oracle") {
    const reference = snapshotRecord(
      summary.replay_reference,
      "Replay Oracle artifact identity",
    );
    exactScientificId(reference.episode_id, "replay_reference.episode_id");
    exactScientificId(reference.artifact_id, "replay_reference.artifact_id");
    exactScientificId(raw.timeline_id, "timeline_id");
    exactScientificId(raw.frame_id, "frame_id");
    exactNonnegativeInteger(
      reference.replay_schema_version,
      "replay_reference.replay_schema_version",
    );
    if (
      ![1, 2, 3, 4].includes(reference.replay_schema_version) ||
      !/^[0-9a-f]{64}$/u.test(reference.context_digest_sha256) ||
      !/^[0-9a-f]{64}$/u.test(reference.trajectory_content_digest_sha256) ||
      !/^[0-9a-f]{64}$/u.test(reference.canonical_digest_sha256) ||
      typeof raw.recorded_ordinary_movement_distance_scale !== "number" ||
      !Number.isFinite(raw.recorded_ordinary_movement_distance_scale) ||
      raw.recorded_ordinary_movement_distance_scale <= 0 ||
      raw.recorded_ordinary_movement_distance_scale > 1
    ) {
      invalid("Replay Oracle compared artifact identity is malformed.");
    }
    if (
      source.source_artifact_id !== `${source.episode_id}:replay` ||
      source.source_timeline_id !==
        `${source.source_artifact_id}:timeline:researcher` ||
      source.source_frame_id !==
        `${source.episode_id}:frame:${source.source_frame_index}` ||
      reference.artifact_id !== `${reference.episode_id}:replay` ||
      raw.frame_id !== `${reference.episode_id}:frame:${cursor.frame_index}`
    ) {
      invalid("Replay Oracle source/raw identity is not canonical.");
    }
    for (const [left, right, label] of [
      [source.episode_id, reference.episode_id, "Replay Oracle episode"],
      [source.source_artifact_id, reference.artifact_id, "Replay Oracle artifact"],
      [source.source_timeline_id, raw.timeline_id, "Replay Oracle timeline"],
      [
        source.source_replay_schema_version,
        reference.replay_schema_version,
        "Replay schema",
      ],
      [
        source.source_context_digest_sha256,
        reference.context_digest_sha256,
        "Replay context digest",
      ],
      [
        source.source_trajectory_content_digest_sha256,
        reference.trajectory_content_digest_sha256,
        "Replay trajectory digest",
      ],
      [
        source.source_artifact_digest_sha256,
        reference.canonical_digest_sha256,
        "Replay artifact digest",
      ],
      [source.source_frame_id, raw.frame_id, "Replay Oracle frame ID"],
      [
        source.source_cursor_generation,
        cursor.cursor_generation,
        "Replay cursor generation",
      ],
      [
        source.source_choreography_generation,
        cursor.choreography_generation,
        "Replay choreography generation",
      ],
      [
        source.source_recorded_ordinary_movement_distance_scale,
        raw.recorded_ordinary_movement_distance_scale,
        "Replay movement scale",
      ],
    ]) {
      requireJoinEqual(left, right, label);
    }
    return raw.frame_kind;
  }
  const privateShared =
    presentation.presentation_kind === "replay_shared_obs_agent_pov";
  const reference = privateShared
    ? null
    : snapshotRecord(summary.replay_reference, "Replay Agent artifact identity");
  const episodeId = privateShared
    ? summary.episode_id
    : /** @type {Record<string, any>} */ (reference).episode_id;
  const recipient = raw.public_agent_id;
  const rawFrameId = privateShared ? raw.recipient_frame_id : raw.pov_frame_id;
  const expectedMode = privateShared ? "shared_obs_visual_union" : "no_shared_obs";
  exactScientificId(episodeId, "Replay Agent episode_id");
  exactPublicAgentId(recipient, "Replay Agent public_agent_id");
  exactScientificId(rawFrameId, "Replay Agent frame ID");
  const mode = privateShared ? "shared-obs-visual-union" : "actor-pov";
  if (
    authority.observation_mode !== expectedMode ||
    source.source_recipient_frame_id !==
      `${source.episode_id}:${mode}:${source.source_recipient_public_agent_id}:frame:${source.source_frame_index}` ||
    rawFrameId !== `${episodeId}:${mode}:${recipient}:frame:${cursor.frame_index}`
  ) {
    invalid("Replay Agent source/raw recipient identity is not canonical.");
  }
  requireJoinEqual(source.episode_id, episodeId, "Replay Agent episode");
  requireJoinEqual(
    source.source_recipient_public_agent_id,
    recipient,
    "Replay Agent source recipient",
  );
  requireJoinEqual(
    authority.recipient_public_agent_id,
    recipient,
    "Replay Agent authority recipient",
  );
  requireJoinEqual(
    source.source_recipient_frame_id,
    rawFrameId,
    "Replay Agent frame ID",
  );
  if (privateShared) {
    requireJoinEqual(
      summary.public_agent_id,
      recipient,
      "Private Shared replay recipient",
    );
  }
  return raw.frame_kind;
}

/**
 * Join matching raw transport and authorized presentation response records.
 *
 * `rawValue` and `presentationValue` are untrusted wire records from the same
 * session, audience, and frame epoch. `animateIncoming` defaults to false and
 * must be a boolean. For replay it requests incoming choreography and must fit
 * the replay cursor; live normalization does not use the animation flag.
 *
 * Checks identity first, normalizes each record using its own contract, verifies
 * presentation hashes, then checks cross-record actor privacy. Resolves to a new
 * deeply frozen `{transport, presentation}` pair remembered by object identity.
 * Both roots remain separate and neither input is changed. A valid identity race
 * rejects with PresentationJoinMismatchError; malformed/inconsistent payloads
 * reject with TypeError. Web Crypto failures also propagate. No fetch, retry,
 * rendering, or persistent write occurs here.
 *
 * @param {unknown} rawValue
 * @param {unknown} presentationValue
 * @returns {Promise<Readonly<{transport: Record<string, any>, presentation: Record<string, any>}>>}
 */
export async function joinTransportAndAuthorizedPresentationV1(
  rawValue,
  presentationValue,
  animateIncoming = false,
) {
  if (typeof animateIncoming !== "boolean") {
    invalid("Joined replay animation intent must be an exact boolean.");
  }
  const frameKind = preflightTransportPresentationIdentity(rawValue, presentationValue);
  const transport =
    frameKind === "shared_obs_agent_pov_replay_viewer"
      ? normalizeSharedObsAgentPovReplayTransportV1(rawValue, animateIncoming)
      : frameKind.endsWith("_live_debugger")
        ? normalizeLiveDebuggerFrameV2(rawValue)
        : animateIncoming
          ? normalizeReplayCommandResponseV1({
              schema_version: 1,
              result: "applied",
              frame: rawValue,
              notice: null,
              animate_incoming: true,
            }).frame
          : normalizeReplayViewerFrameV1(rawValue);
  if (
    animateIncoming &&
    frameKind === "shared_obs_agent_pov_replay_viewer" &&
    (transport.cursor.frame_index === 0 ||
      transport.cursor.choreography_generation === 0)
  ) {
    invalid("Private Shared replay animation intent is incoherent.");
  }
  const presentation = await normalizeAuthorizedPresentationFrameV1(presentationValue);
  validatePairedAgentPrivacy(transport, presentation);
  const joined = deepFreeze({ transport, presentation });
  JOINED_PRESENTATION_ROOTS.add(joined);
  return joined;
}

/**
 * Test whether this exact object was installed as an authorized pair here.
 *
 * `value` may be anything. Returns true for a root recorded by either frame-pair
 * or replay-timeline joining in this module. A clone or look-alike returns false.
 * This is an identity check, not fresh payload validation or a source signature.
 *
 * @param {unknown} value @returns {value is Readonly<Record<string, any>>}
 */
export function isJoinedTransportAndAuthorizedPresentationV1(value) {
  return (
    typeof value === "object" && value !== null && JOINED_PRESENTATION_ROOTS.has(value)
  );
}

/**
 * Attach a matching replay timeline to an already normalized frame pair.
 *
 * `joinedValue` must be the exact remembered result of this module's pair join;
 * `timelineValue` is an untrusted replay timeline wire record. The timeline is
 * normalized for the transport audience, then its artifact, completion, cursor,
 * and current-row facts are joined to the transport.
 *
 * Returns a new deeply frozen remembered root with borrowed `transport` and
 * `presentation` roots plus the normalized `timeline`. Inputs are unchanged.
 * Malformed timelines or an unrecognized pair throw TypeError; a mismatch after
 * timeline normalization throws PresentationJoinMismatchError for a possible
 * bounded GET retry. This does not fetch, retry, or move the replay cursor.
 *
 * @param {unknown} joinedValue
 * @param {unknown} timelineValue
 */
export function joinReplayTransportAndTimelineV1(joinedValue, timelineValue) {
  if (!isJoinedTransportAndAuthorizedPresentationV1(joinedValue)) {
    invalid("Replay timeline join requires an unforgeable authorized pair.");
  }
  const joined = /** @type {Record<string, any>} */ (joinedValue);
  const transport = joined.transport;
  let timeline;
  if (transport.frame_kind === "shared_obs_agent_pov_replay_viewer") {
    timeline = normalizeSharedObsAgentPovReplayTimelineTransportV1(timelineValue);
    const currentRow = timeline.rows[transport.cursor.frame_index];
    if (
      timeline.timeline_id !== transport.timeline_id ||
      timeline.final_frame_index !== transport.cursor.final_frame_index ||
      !structurallyEqual(timeline.artifact_summary, transport.artifact_summary) ||
      !structurallyEqual(timeline.completion, transport.completion) ||
      !currentRow ||
      currentRow.frame_index !== transport.cursor.frame_index ||
      currentRow.recipient_frame_id !== transport.recipient_frame_id ||
      currentRow.simulator_step_count !== transport.simulator_step_count ||
      currentRow.incoming_recipient_transition_id !==
        transport.incoming_recipient_transition_id
    ) {
      joinMismatch("Private Shared replay timeline raced its authorized frame pair.");
    }
  } else {
    timeline = normalizeReplayTimelineV1(timelineValue);
    try {
      timeline = joinReplayFrameAndTimeline(transport, timeline);
    } catch (error) {
      if (error instanceof TypeError) {
        joinMismatch("Replay timeline raced its authorized frame pair.");
      }
      throw error;
    }
  }
  const installed = deepFreeze({
    transport,
    presentation: joined.presentation,
    timeline,
  });
  JOINED_PRESENTATION_ROOTS.add(installed);
  return installed;
}

/**
 * Check that two remembered replay pairs can belong to one command sequence.
 *
 * `previousValue` and `nextValue` must be exact pair roots installed by this
 * module. `result` is the caller's already checked command-result string, such as
 * `applied`, `duplicate`, or `stale_resync`. Legacy-only pairs use the legacy
 * continuity check. If either pair uses private SharedObs transport, this checks
 * session/artifact/count/completion stability and the result-dependent revision
 * and resync-generation bounds across audience changes.
 *
 * Returns `nextValue` itself on success; throws TypeError for unknown pairs or
 * broken continuity. It does not clear old audience state, check every command's
 * cursor delta, validate `result` against an enum, or issue commands. The caller
 * owns command-result validation and presentation-authority state clearing.
 *
 * @param {unknown} previousValue
 * @param {unknown} nextValue
 * @param {unknown} result
 */
export function validateReplayTransportContinuityV1(previousValue, nextValue, result) {
  if (
    !isJoinedTransportAndAuthorizedPresentationV1(previousValue) ||
    !isJoinedTransportAndAuthorizedPresentationV1(nextValue)
  ) {
    invalid("Replay continuity requires two unforgeable authorized pairs.");
  }
  const previous = /** @type {Record<string, any>} */ (previousValue).transport;
  const next = /** @type {Record<string, any>} */ (nextValue).transport;
  const privateBoundary =
    previous.frame_kind === "shared_obs_agent_pov_replay_viewer" ||
    next.frame_kind === "shared_obs_agent_pov_replay_viewer";
  if (!privateBoundary) {
    validateReplayFrameContinuity(previous, next, result);
    return nextValue;
  }
  const revisionValid =
    result === "stale_resync" || result === "duplicate"
      ? next.revision >= previous.revision
      : next.revision ===
        (result === "applied" ? previous.revision + 1 : previous.revision);
  const generationValid =
    result !== "stale_resync" && result !== "duplicate"
      ? true
      : next.cursor.cursor_generation >= previous.cursor.cursor_generation &&
        next.cursor.choreography_generation >=
          previous.cursor.choreography_generation &&
        next.cursor.choreography_generation - previous.cursor.choreography_generation <=
          next.cursor.cursor_generation - previous.cursor.cursor_generation;
  const previousSummary = previous.artifact_summary;
  const nextSummary = next.artifact_summary;
  const previousEpisode =
    previousSummary.episode_id ?? previousSummary.replay_reference?.episode_id;
  const nextEpisode =
    nextSummary.episode_id ?? nextSummary.replay_reference?.episode_id;
  const previousCaptured =
    previousSummary.captured_transition_count ??
    previousSummary.recorded_transition_count;
  const nextCaptured =
    nextSummary.captured_transition_count ?? nextSummary.recorded_transition_count;
  const previousFrames =
    previousSummary.captured_frame_count ?? previousSummary.recorded_frame_count;
  const nextFrames =
    nextSummary.captured_frame_count ?? nextSummary.recorded_frame_count;
  const bothPrivateShared =
    previous.frame_kind === "shared_obs_agent_pov_replay_viewer" &&
    next.frame_kind === "shared_obs_agent_pov_replay_viewer";
  const previousCompletion = previous.completion;
  const nextCompletion = next.completion;
  const previousArtifactFacts =
    previous.frame_kind === "researcher_replay_viewer"
      ? {
          schema_version: 1,
          artifact_summary: previous.artifact_summary,
          completion: previous.completion,
          processing: previous.processing,
        }
      : previous.artifact_facts;
  const nextArtifactFacts =
    next.frame_kind === "researcher_replay_viewer"
      ? {
          schema_version: 1,
          artifact_summary: next.artifact_summary,
          completion: next.completion,
          processing: next.processing,
        }
      : next.artifact_facts;
  const completionValid = bothPrivateShared
    ? structurallyEqual(nextCompletion, previousCompletion)
    : // Audience-local reasons intentionally differ. The exact canonical reason
      // is still pinned by the artifact_facts comparison below.
      structurallyEqual(
        {
          completion_state: nextCompletion.completion_state,
          episode_id: nextCompletion.episode_id,
          expected_transition_count: nextCompletion.expected_transition_count,
          captured_transition_count:
            nextCompletion.captured_transition_count ??
            nextCompletion.validated_transition_count,
          terminated: nextCompletion.terminated,
          truncated: nextCompletion.truncated,
          completion_bases: nextCompletion.completion_bases,
        },
        {
          completion_state: previousCompletion.completion_state,
          episode_id: previousCompletion.episode_id,
          expected_transition_count: previousCompletion.expected_transition_count,
          captured_transition_count:
            previousCompletion.captured_transition_count ??
            previousCompletion.validated_transition_count,
          terminated: previousCompletion.terminated,
          truncated: previousCompletion.truncated,
          completion_bases: previousCompletion.completion_bases,
        },
      );
  if (
    previous.viewer_mode !== "replay" ||
    next.viewer_mode !== "replay" ||
    next.viewer_session_id !== previous.viewer_session_id ||
    nextEpisode !== previousEpisode ||
    nextSummary.expected_transition_count !==
      previousSummary.expected_transition_count ||
    nextCaptured !== previousCaptured ||
    nextFrames !== previousFrames ||
    next.cursor.final_frame_index !== previous.cursor.final_frame_index ||
    !structurallyEqual(nextArtifactFacts, previousArtifactFacts) ||
    !completionValid ||
    !revisionValid ||
    !generationValid
  ) {
    invalid("Replay command response breaks private Shared continuity.");
  }
  return nextValue;
}
