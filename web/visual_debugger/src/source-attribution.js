/**
 * @file Build source labels by joining recorded public source references to exact
 * authorized scene identities. Preserve serialized source order and reject
 * conflicting identities. Never search by global slot, class, position or
 * proximity, and never infer an emitter from an aggregate aura multiplier.
 */
import { exactAuthorizedAgentIdentityV1 } from "./agent-identity.js";

const OPTION_KEYS = Object.freeze([
  "attribution_kind",
  "audience",
  "direct_sources",
  "authorized_agents",
]);
const SOURCE_KEYS = Object.freeze([
  "source_presentation_key",
  "source_public_agent_id",
]);

/** @typedef {"researcher" | "agent_pov"} SourceAudience */
/** @typedef {"direct" | "aggregate_aura"} SourceAttributionKind */
/** @typedef {"single" | "multiple" | "unavailable"} SourceAttributionState */
/**
 * @typedef {{
 *   state: SourceAttributionState,
 *   label: "Source" | "Sources",
 *   value: string,
 *   text: string,
 * }} SourceAttribution
 */
/**
 * @typedef {{
 *   attribution_kind: SourceAttributionKind,
 *   audience: SourceAudience,
 *   direct_sources: unknown,
 *   authorized_agents: unknown,
 * }} SourceAttributionOptions
 */

const UNAVAILABLE = attribution(
  "unavailable",
  "Source",
  "Unavailable in this artifact",
  "Source unavailable in this artifact",
);
/**
 * Return a direct-source label, unavailable record or intentionally omitted row.
 *
 * rawOptions must be an exact plain record with attribution_kind, audience,
 * direct_sources and authorized_agents. Reject malformed options or unknown
 * kind/audience with TypeError. aggregate_aura always returns null; agent_pov
 * also returns null, because battlefield geometry alone does not authorize
 * source identities. For researcher/direct input, require dense ordinary
 * arrays and valid public identities. Join each unique source by both key and
 * ID to exactly one agent, preserving first occurrence. Bad, absent or
 * conflicting joins return the shared frozen unavailable record. Success is a
 * frozen single/multiple label. Inputs and scene authority are not changed.
 *
 * @param {unknown} rawOptions
 * @returns {Readonly<SourceAttribution> | null}
 */
export function authorizedSourceAttributionV1(rawOptions) {
  const options = snapshotRecord(rawOptions, OPTION_KEYS, true);
  if (options === null) {
    throw new TypeError("source attribution options must be one exact plain record.");
  }
  const kind = options.attribution_kind;
  const audience = options.audience;
  if (kind !== "direct" && kind !== "aggregate_aura") {
    throw new TypeError("source attribution kind must be direct or aggregate_aura.");
  }
  if (audience !== "researcher" && audience !== "agent_pov") {
    throw new TypeError("source attribution audience must be researcher or agent_pov.");
  }
  if (kind === "aggregate_aura") {
    return null;
  }
  // Agent battlefield geometry never authorizes a hidden source identity on
  // its own. Callers either supply the separately validated researcher-space
  // status context or omit the unavailable row entirely.
  if (audience === "agent_pov") return null;
  const directSources = snapshotDensePlainArray(options.direct_sources);
  const authorizedAgents = snapshotDensePlainArray(options.authorized_agents);
  if (directSources === null || authorizedAgents === null) {
    return UNAVAILABLE;
  }

  const references = exactFirstOccurrenceReferences(directSources);
  if (references === null || references.length === 0) {
    return UNAVAILABLE;
  }

  const candidates = [];
  for (let index = 0; index < authorizedAgents.length; index += 1) {
    const candidate = exactAuthorizedAgentIdentityV1(authorizedAgents[index]);
    if (candidate === null) return UNAVAILABLE;
    candidates.push(candidate);
  }
  const identities = [];
  for (const reference of references) {
    const joined = [];
    for (const candidate of candidates) {
      if (
        candidate.presentationKey === reference.presentationKey ||
        candidate.publicAgentId === reference.publicAgentId
      ) {
        joined.push(candidate);
      }
    }
    const sourceAgent = joined[0];
    if (
      joined.length !== 1 ||
      sourceAgent === undefined ||
      sourceAgent.presentationKey !== reference.presentationKey ||
      sourceAgent.publicAgentId !== reference.publicAgentId ||
      sourceAgent.title.length === 0
    ) {
      return UNAVAILABLE;
    }
    identities.push(sourceAgent.title);
  }

  if (identities.length === 1) {
    const value = identities[0];
    return attribution("single", "Source", value, `Source: ${value}`);
  }
  const value = identities.join("; ");
  return attribution("multiple", "Sources", value, `Sources: ${value}`);
}

/**
 * Return a frozen source-label record containing the supplied state, label,
 * value and complete text. The owning formatter validates meaning; this small
 * constructor neither checks input nor modifies it.
 *
 * @param {SourceAttributionState} state
 * @param {"Source" | "Sources"} label
 * @param {string} value
 * @param {string} text
 * @returns {Readonly<SourceAttribution>}
 */
function attribution(state, label, value, text) {
  return Object.freeze({ state, label, value, text });
}

/**
 * Validate and deduplicate rawSources in their serialized order.
 *
 * Each source must have exactly source_presentation_key and
 * source_public_agent_id as valid data strings. Remove exact duplicate pairs;
 * a key mapped to several IDs, or an ID mapped to several keys, returns null.
 * Success returns a frozen array of frozen reference records, including an
 * empty array for no sources. The caller supplies an already snapshotted array.
 *
 * @param {readonly unknown[]} rawSources
 * @returns {readonly Readonly<{presentationKey: string, publicAgentId: string}>[] | null}
 */
function exactFirstOccurrenceReferences(rawSources) {
  const seenPairs = new Set();
  const publicIdByKey = new Map();
  const keyByPublicId = new Map();
  const references = [];
  for (let index = 0; index < rawSources.length; index += 1) {
    const rawSource = rawSources[index];
    const source = snapshotRecord(rawSource, SOURCE_KEYS, true);
    if (source === null) return null;
    const presentationKey = exactIdentifier(source.source_presentation_key);
    const publicAgentId = exactIdentifier(source.source_public_agent_id);
    if (presentationKey === null || publicAgentId === null) return null;

    const priorPublicId = publicIdByKey.get(presentationKey);
    const priorKey = keyByPublicId.get(publicAgentId);
    if (
      (priorPublicId !== undefined && priorPublicId !== publicAgentId) ||
      (priorKey !== undefined && priorKey !== presentationKey)
    ) {
      return null;
    }
    publicIdByKey.set(presentationKey, publicAgentId);
    keyByPublicId.set(publicAgentId, presentationKey);

    const pair = `${presentationKey}\u0000${publicAgentId}`;
    if (seenPairs.has(pair)) continue;
    seenPairs.add(pair);
    references.push(Object.freeze({ presentationKey, publicAgentId }));
  }
  return Object.freeze(references);
}

/**
 * Copy a dense ordinary array without invoking its iterator or indexed getters.
 *
 * value must use Array.prototype and have exactly its own length plus one
 * enumerable data property per index. Holes, extra/symbol keys, accessors and
 * inspection failures return null. Success returns a frozen shallow copy.
 * Element objects are reused and validated separately; inputs are unchanged.
 *
 * @param {unknown} value
 * @returns {readonly unknown[] | null}
 */
function snapshotDensePlainArray(value) {
  try {
    if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) {
      return null;
    }
    const descriptors = /** @type {Record<PropertyKey, PropertyDescriptor>} */ (
      /** @type {unknown} */ (Object.getOwnPropertyDescriptors(value))
    );
    const lengthDescriptor = descriptors.length;
    if (!lengthDescriptor || !Object.hasOwn(lengthDescriptor, "value")) {
      return null;
    }
    const length = lengthDescriptor.value;
    if (typeof length !== "number" || !Number.isSafeInteger(length) || length < 0) {
      return null;
    }
    const keys = Reflect.ownKeys(descriptors);
    if (keys.length !== length + 1 || keys.some((key) => typeof key !== "string")) {
      return null;
    }
    const snapshot = [];
    for (let index = 0; index < length; index += 1) {
      const key = String(index);
      const descriptor = descriptors[key];
      if (
        !descriptor ||
        !Object.hasOwn(descriptor, "value") ||
        !descriptor.enumerable
      ) {
        return null;
      }
      snapshot.push(descriptor.value);
    }
    return Object.freeze(snapshot);
  } catch {
    return null;
  }
}

/**
 * Copy required own data fields from a plain record, or return null.
 *
 * value must use Object.prototype or a null prototype and have no symbol keys.
 * requiredKeys must exist as enumerable data properties. If exact is true,
 * reject extra fields too; if false, ignore other string-named fields. Return
 * a frozen null-prototype shallow copy. Getters are not called; inspection
 * errors return null. Nested values retain their original references.
 *
 * @param {unknown} value
 * @param {readonly string[]} requiredKeys
 * @param {boolean} exact
 * @returns {Readonly<Record<string, unknown>> | null}
 */
function snapshotRecord(value, requiredKeys, exact) {
  try {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return null;
    const keys = Reflect.ownKeys(value);
    if (
      keys.some((key) => typeof key !== "string") ||
      requiredKeys.some((key) => !keys.includes(key)) ||
      (exact &&
        (keys.length !== requiredKeys.length ||
          keys.some((key) => !requiredKeys.includes(/** @type {string} */ (key)))))
    ) {
      return null;
    }
    const descriptors = Object.getOwnPropertyDescriptors(value);
    /** @type {Record<string, unknown>} */
    const snapshot = Object.create(null);
    for (const key of requiredKeys) {
      const descriptor = descriptors[key];
      if (!descriptor || !("value" in descriptor) || !descriptor.enumerable) {
        return null;
      }
      snapshot[key] = descriptor.value;
    }
    return Object.freeze(snapshot);
  } catch {
    return null;
  }
}

/**
 * Return value unchanged when it is a nonempty string of at most 512
 * characters with no outer whitespace or CR/LF. Return null otherwise. This
 * checks string shape only; the caller checks the source-to-agent identity join.
 *
 * @param {unknown} value @returns {string | null}
 */
function exactIdentifier(value) {
  return typeof value === "string" &&
    value.length > 0 &&
    value.length <= 512 &&
    value.trim() === value &&
    !/[\r\n]/u.test(value)
    ? value
    : null;
}
