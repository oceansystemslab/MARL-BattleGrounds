import { AUTHORIZED_PRESENTATION_SCHEMA_V1 } from "./authorized-presentation-schema.js";
import { normalizeLiveDebuggerFrameV2 } from "./frame-normalizer.js";
import {
  joinReplayFrameAndTimeline,
  normalizeReplayCommandResponseV1,
  normalizeReplayTimelineV1,
  normalizeReplayViewerFrameV1,
  validateReplayFrameContinuity,
} from "./replay-frame-normalizer.js";

const PRESENTATION_KINDS = new Set([
  "live_oracle",
  "live_no_shared_obs_agent_pov",
  "replay_oracle",
  "replay_no_shared_obs_agent_pov",
  "replay_shared_obs_agent_pov",
]);
/** @type {Readonly<Record<string, readonly string[]>>} */
const RAW_FRAME_KEYS = Object.freeze({
  researcher_live_debugger: [
    "available_scenarios",
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
});
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
]);
const NORMALIZED_PRESENTATION_ROOTS = new WeakSet();
const JOINED_PRESENTATION_ROOTS = new WeakSet();

export class PresentationJoinMismatchError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = "PresentationJoinMismatchError";
  }
}

/** @param {string} message @returns {never} */
function invalid(message) {
  throw new TypeError(message);
}

/** @param {string} message @returns {never} */
function joinMismatch(message) {
  throw new PresentationJoinMismatchError(message);
}

/** @param {unknown} error */
export function isPresentationJoinRace(error) {
  return error instanceof PresentationJoinMismatchError;
}

/**
 * Fail closed if Python ever emits a JSON Schema keyword the browser visitor
 * does not implement. Property and definition names are deliberately handled
 * as names, so a future wire field called `title` cannot be stripped or
 * mistaken for schema metadata.
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
 * Snapshot a JSON object without invoking accessors. JSON-decoded values have
 * only enumerable own data properties; accepting anything broader would make
 * validation order observable to a hostile caller.
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

/** @param {Record<string, any>} schema @returns {Record<string, any>} */
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
 * Select the exact already-validated schema branch for canonical encoding.
 * This is separate from validation so number-vs-integer wire types remain
 * available after JSON.parse has erased a token such as `20.0`.
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
 * Match CPython's finite-float representation used by json.dumps. ECMAScript
 * and CPython share shortest-roundtrip significant digits but select fixed vs
 * exponent notation at different thresholds, so the notation is rebuilt from
 * the shortest digits and the schema-owned float type.
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

/** @param {unknown} value @returns {any} */
function deepFreeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

/** @param {unknown[]} values @param {string} label */
function requireUnique(values, label) {
  if (new Set(values).size !== values.length) invalid(`${label} must be unique.`);
}

/** @param {any[]} values @param {(value: any, index: number) => unknown} selector @param {string} label */
function requireExactOrder(values, selector, label) {
  if (values.some((value, index) => !Object.is(selector(value, index), index))) {
    invalid(`${label} must retain canonical order.`);
  }
}

/** @param {unknown} left @param {unknown} right @returns {boolean} */
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

/** @param {Record<string, any>} axis */
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

/** @param {Record<string, any>} mask @param {string} label */
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

/** @param {Record<string, any>} root */
function validatePresentationKeyGraph(root) {
  const expectedPrefix =
    root.authority.authority_kind === "oracle" ? "oracle_" : "pov_";
  const publicByKey = new Map();
  const keyByPublic = new Map();
  /** @param {unknown} value */
  function visit(value) {
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
    for (const child of Object.values(record)) visit(child);
  }
  visit(root);
  return [...keyByPublic.entries()].map(([publicId, key]) => ({ key, publicId }));
}

/**
 * @param {Record<string, any>} frame
 * @param {{key: string, publicId: string}[]} pairs
 */
async function verifyPresentationKeyDerivation(frame, pairs) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) invalid("Web Crypto is required for presentation-key validation.");
  const encoder = new TextEncoder();
  const oracle = frame.authority.authority_kind === "oracle";
  const session = frame.source.source_session_id;
  const recipient = oracle ? null : frame.authority.recipient_public_agent_id;
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

/** @param {Record<string, any>} root */
function validateAgentPrivacy(root) {
  const episodeId = root.source.episode_id;
  const recipientId = root.authority.recipient_public_agent_id;
  const escapedEpisodeId = episodeId.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
  const escapedRecipientId = recipientId.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
  const oracleIdentity = new RegExp(
    `^${escapedEpisodeId}:(?:frame:[0-9]+|transition:[0-9]+(?::event:[0-9]{4})?|replay(?:|:timeline(?::[^\\s]+)?)|(?:actor-pov|shared-obs-visual-union):${escapedRecipientId}:(?:replay|timeline))$`,
    "u",
  );
  /** @param {unknown} value */
  function visit(value) {
    if (typeof value === "string") {
      if (oracleIdentity.test(value)) {
        invalid("Agent presentation contains a forbidden Oracle/diagnostic value.");
      }
      return;
    }
    if (Array.isArray(value)) {
      for (const child of value) visit(child);
      return;
    }
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      if (AGENT_FORBIDDEN_KEYS.has(key)) {
        invalid(`Agent presentation contains forbidden key ${key}.`);
      }
      visit(child);
    }
  }
  visit(root);
}

/**
 * A joined transport gives the browser exact private values that must never be
 * reflected through an Agent presentation. This is intentionally equality-
 * based: arbitrary public IDs and display prose may contain words such as
 * "source-material", "metric", or "processing".
 *
 * @param {Record<string, any>} transport
 * @param {Record<string, any>} presentation
 */
function validatePairedAgentPrivacy(transport, presentation) {
  if (presentation.authority.authority_kind !== "agent_pov") return;
  const forbiddenValues = new Set();
  /** @param {unknown} value */
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
  /** @param {unknown} value @param {string | null} [field] */
  function rejectReflections(value, field = null) {
    if (typeof value === "string") {
      const allowedEndpointDigest =
        field === "authorized_endpoint_digest_sha256" ||
        field === "source_authorized_endpoint_digest_sha256";
      if (!allowedEndpointDigest && forbiddenValues.has(value)) {
        invalid("Agent presentation reflects a forbidden paired transport value.");
      }
      return;
    }
    if (Array.isArray(value)) {
      for (const child of value) rejectReflections(child, field);
      return;
    }
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      rejectReflections(child, key);
    }
  }
  rejectReflections(presentation);
}

/** @param {Record<string, any>} action @param {string} label */
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

/** @param {Record<string, any>} action @param {string} label */
function validateAcceptedActionTuple(action, label) {
  const domains = { move_action: 9, target_action: 11, use_ultimate_action: 2 };
  for (const [name, count] of Object.entries(domains)) {
    if (!Number.isInteger(action[name]) || action[name] < 0 || action[name] >= count) {
      invalid(`${label}.${name} lies outside its accepted action domain.`);
    }
  }
}

/**
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
 * @param {Record<string, any>} event
 * @returns {{anchor: Record<string, any>, phase: string}[]}
 */
function oracleEventAnchors(event) {
  /** @type {{anchor: Record<string, any>, phase: string}[]} */
  const anchors = [];
  /** @param {string} field @param {string} phase */
  const add = (field, phase) => {
    if (event[field] !== null && event[field] !== undefined) {
      anchors.push({ anchor: event[field], phase });
    }
  };
  /** @param {string} field @param {string} phase */
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

/** @param {Record<string, any>[]} statuses @param {string} label */
function validateIncomingStatuses(statuses, label) {
  let previousChannel = -1;
  for (const status of statuses) {
    if (
      status.status_channel <= previousChannel ||
      STATUS_ID_BY_CHANNEL[status.status_channel] !== status.status_id ||
      status.configured_duration_steps < 1 ||
      status.remaining_duration < 1 ||
      status.remaining_duration > status.configured_duration_steps ||
      (status.magnitude === null) !== (status.magnitude_kind === "none")
    ) {
      invalid(`${label} is not a canonical status-axis snapshot.`);
    }
    previousChannel = status.status_channel;
  }
}

/** @param {Record<string, any>[]} modifiers @param {string} label */
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

/** @param {Record<string, any>} observation @param {string} label */
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

/** @param {Record<string, any>[]} sources @param {string} label */
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

/** @param {Record<string, any>} latest */
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

/** @param {Record<string, any>} latest */
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

/** @param {Record<string, any>} latest */
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

/** @param {Record<string, any>} agent */
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

/** @param {number} recorded @param {number} catalog */
function joinsCatalogFloat(recorded, catalog) {
  return recorded === catalog || recorded === Math.fround(catalog);
}

/**
 * Port the closed semantic joins of Python's AuthorizedBattlefieldSceneV1.
 * JSON Schema owns local wire shapes; this visitor owns catalog identity,
 * cross-row cardinality/order, and source/mechanic/lifecycle coherence.
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
  const agentsByKey = new Map(agents.map((agent) => [agent.presentation_key, agent]));
  const representedClassIds = [...new Set(agents.map((agent) => agent.class_id))].sort(
    (left, right) => left - right,
  );
  const mechanicsIds = classMechanics.map((mechanics) => mechanics.class_id);
  if (!structurallyEqual(mechanicsIds, representedClassIds)) {
    invalid("Scene class mechanics must exactly equal represented class order.");
  }
  const mechanicsVersions = new Set(
    classMechanics.map((mechanics) => (mechanics.mechanics_version === 2 ? 2 : 1)),
  );
  if (mechanicsVersions.size > 1) {
    invalid("Scene class mechanics must be entirely V1 or entirely V2.");
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
    invalid("Scene V2 class mechanics must share one documentation profile.");
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
      invalid("Scene class mechanics changed canonical class identity or bounds.");
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
        invalid("Scene class status mechanics changed the canonical V1 status axis.");
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
        invalid("Scene class aura mechanics changed the canonical V1 aura axis.");
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
    invalid("Scene mechanics inventories do not equal represented catalog axes.");
  }

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
        invalid("Scene durable status changed its canonical identity or bounds.");
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
        invalid("Scene durable status does not join its catalog mechanic.");
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
            "Scene status source does not join an authorized source-class agent.",
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
        invalid("Scene agent aura modifiers are not canonical and unique.");
      }
      auraIds.add(modifier.aura_id);
    }
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

/** @param {Record<string, any>} frame */
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
      /** @param {any} left @param {any} right */
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
  }
  validateAuthorizedScene(scene);
  const sceneAgents = /** @type {any[]} */ (scene.agents);
  requireUnique(
    sceneAgents.map((row) => row.presentation_key),
    "Scene presentation keys",
  );
  requireUnique(
    sceneAgents.map((row) => row.public_agent_id),
    "Scene public identities",
  );
  if (actionAxis !== null) validateActionAxis(actionAxis);
  const presentationKeyPairs = validatePresentationKeyGraph(frame);

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
  validateInspectionStateMatrix(
    frame,
    source,
    endpoint,
    actionAxis,
    scene,
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
    scene,
  };
}

/**
 * Strictly normalize one of the five Python-owned authorized presentation
 * leaves. The result contains the exact wire branches plus authority-neutral
 * aliases used by browser consumers.
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
  await Promise.all([
    verifyPresentationKeyDerivation(frame, semantic.presentationKeyPairs),
    verifyAuthorizedEndpointDigest(frame),
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
 * Return true only for an exact root produced by this module after complete
 * structural and semantic validation. A lookalike frozen object cannot enter
 * presentation-owned rendering through this guard.
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
 * Validate the exact audience-unavailable response served with HTTP 422.
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

/** @param {unknown} value @param {string} label */
function exactNonnegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || Number(value) < 0) {
    invalid(`${label} must be a non-negative safe integer.`);
  }
  return Number(value);
}

/** @param {unknown} value @param {string} label */
function exactNonemptyString(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    invalid(`${label} must be a non-empty string.`);
  }
  return value;
}

/** @param {unknown} value @param {string} label */
function exactScientificId(value, label) {
  const text = exactNonemptyString(value, label);
  if (text.length > 512 || !/^[A-Za-z0-9_.:-]+$/u.test(text)) {
    invalid(`${label} must be an exact ScientificId.`);
  }
  return text;
}

/** @param {unknown} value @param {string} label */
function exactPublicAgentId(value, label) {
  const text = exactNonemptyString(value, label);
  if (text.length > 128 || !/^[A-Za-z0-9_.:-]+$/u.test(text)) {
    invalid(`${label} must be an exact PublicAgentId.`);
  }
  return text;
}

/**
 * Strictly normalize the private identity-only Shared replay transport. It
 * deliberately creates no scene, event, legality, projection, or HUD aliases.
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
 * Normalize the existing Python-owned private Shared timeline without adding
 * scene, projection, source-material, HUD, or command aliases.
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

/** @param {Record<string, any>} value @param {readonly string[]} expected @param {string} label */
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
 * Classify only two individually scalar-well-formed unequal identity values as
 * a GET race. Wrong types/nonfinite values are protocol poison and never retry.
 *
 * @param {unknown} left
 * @param {unknown} right
 * @param {string} label
 */
function requireJoinEqual(left, right, label) {
  /** @param {unknown} value */
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
 * Compare only raw/source/authority identity fields before presentation nested
 * payload validation. This ordering is a security property: a mismatched pair
 * cannot trigger an endpoint, event, Technical Frame, or inspection getter.
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
  const shared = presentation.presentation_kind === "replay_shared_obs_agent_pov";
  const expectedSourceKind = /** @type {Readonly<Record<string, string>>} */ ({
    live_oracle: "live_oracle_frame",
    live_no_shared_obs_agent_pov: "live_no_shared_obs_frame",
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
    RAW_FRAME_KEYS[raw.frame_kind],
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
    const hud = snapshotRecord(raw.hud, "Live transport HUD identity");
    if (
      !["joint_turn", "controlled_actor", "scripted_playback"].includes(
        hud.pending_submission_scope,
      )
    ) {
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
      hud.pending_submission_scope,
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
      const recipient = hud.controlled_public_agent_id;
      if (
        source.source_recipient_frame_id !==
        `${source.episode_id}:actor-pov:${source.source_recipient_public_agent_id}:frame:${source.source_frame_index}`
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
      reference.replay_schema_version !== 1 ||
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
 * Join one exact raw transport frame to one separately authorized presentation.
 * The two validated roots remain separate and neither input is mutated.
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

/** @param {unknown} value @returns {value is Readonly<Record<string, any>>} */
export function isJoinedTransportAndAuthorizedPresentationV1(value) {
  return (
    typeof value === "object" && value !== null && JOINED_PRESENTATION_ROOTS.has(value)
  );
}

/**
 * Normalize and join the independently fetched replay timeline to one already
 * authorized pair. Structurally valid epoch/audience mismatch is a bounded-GET
 * race; malformed timeline bytes remain an ordinary TypeError.
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
 * Validate previous→next replay command continuity across legacy and private
 * Shared transports. Inputs must be unforgeable authorized pairs; presentation
 * authority switching remains a separate main-state clearing concern.
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
  const completionValid = bothPrivateShared
    ? structurallyEqual(nextCompletion, previousCompletion)
    : structurallyEqual(
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
          public_end_or_failure_reason: Object.hasOwn(
            nextCompletion,
            "public_end_or_failure_reason",
          )
            ? nextCompletion.public_end_or_failure_reason
            : nextCompletion.end_or_failure_reason,
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
          public_end_or_failure_reason: Object.hasOwn(
            previousCompletion,
            "public_end_or_failure_reason",
          )
            ? previousCompletion.public_end_or_failure_reason
            : previousCompletion.end_or_failure_reason,
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
    !completionValid ||
    !revisionValid ||
    !generationValid
  ) {
    invalid("Replay command response breaks private Shared continuity.");
  }
  return nextValue;
}
