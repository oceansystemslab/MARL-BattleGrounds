/**
 * Pure authority-safe source attribution for scientific cards.
 *
 * The formatter consumes only presentation keys and public IDs already present
 * in an authorized source reference plus the corresponding authorized scene
 * identity. It never searches by slot, class, event, position, proximity, or
 * any hidden/raw identity.
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
/** @typedef {"single" | "multiple" | "unavailable" | "redacted"} SourceAttributionState */
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
const REDACTED = attribution(
  "redacted",
  "Source",
  "Not disclosed in Agent POV",
  "Source not disclosed in Agent POV",
);

/**
 * Format direct-source attribution or deliberately omit aggregate-aura source.
 *
 * `aggregate_aura` always returns `null`: an aggregate multiplier has no
 * serialized emitter attribution. Spawn Shield is intentionally not an
 * accepted attribution kind; its separate card owns only its authorized Owner
 * identity and never manufactures a Source row.
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
  if (audience === "agent_pov") {
    return REDACTED;
  }
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
 * Keep serialized first occurrence, deduplicate only an exact repeated pair,
 * and reject conflicting key/ID aliases.
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
 * Snapshot a caller-owned array without invoking its iterator, indexed
 * accessors, or overridden array methods. Only a dense ordinary Array with the
 * own `length` property and exactly one enumerable data property per index is
 * accepted.
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
 * Snapshot plain enumerable data properties without invoking accessors.
 * Source/option records are exact; authorized agents may carry additional
 * already-normalized scientific fields, but their identity fields must all be
 * own enumerable data properties.
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

/** @param {unknown} value @returns {string | null} */
function exactIdentifier(value) {
  return typeof value === "string" &&
    value.length > 0 &&
    value.length <= 512 &&
    value.trim() === value &&
    !/[\r\n]/u.test(value)
    ? value
    : null;
}
