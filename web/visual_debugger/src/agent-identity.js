/**
 * Canonical public agent identity shared by scientific presentation builders.
 * This leaf owns only stable class/team vocabulary and identity validation; it
 * has no scene, authority, tooltip, or simulator dependencies.
 */

const CLASS_BY_ID = Object.freeze({
  1: Object.freeze({ label: "Mage", accent: "mage" }),
  2: Object.freeze({ label: "Warrior", accent: "warrior" }),
  3: Object.freeze({ label: "Hunter", accent: "hunter" }),
  4: Object.freeze({ label: "Rogue", accent: "rogue" }),
  5: Object.freeze({ label: "Priest", accent: "priest" }),
});

const TEAM_BY_ID = Object.freeze({
  1: "Team A",
  2: "Team B",
});

const AUTHORIZED_IDENTITY_KEYS = Object.freeze([
  "presentation_key",
  "public_agent_id",
  "class_id",
  "team_id",
]);

/**
 * Return a neutral, display-safe identity for arbitrary scientific-card input.
 * Invalid fields degrade independently and never fall back to slots, keys, or
 * other private identity channels.
 *
 * @param {unknown} rawAgent
 * @returns {Readonly<{
 *   title: string,
 *   publicIdentity: string,
 *   classLabel: string,
 *   teamLabel: string,
 *   accent: "none" | "mage" | "warrior" | "hunter" | "rogue" | "priest",
 * }>}
 */
export function canonicalAgentIdentity(rawAgent) {
  const fields = identityDataFields(rawAgent);
  return formatIdentity(fields?.public_agent_id, fields?.class_id, fields?.team_id);
}

/**
 * Validate one complete authorized scene identity without invoking accessors.
 * Additional normalized scientific fields are allowed, but the four identity
 * fields must be own enumerable data properties and agree exactly.
 *
 * @param {unknown} rawAgent
 * @returns {Readonly<{
 *   presentationKey: string,
 *   publicAgentId: string,
 *   title: string,
 *   publicIdentity: string,
 *   classLabel: string,
 *   teamLabel: string,
 *   accent: "mage" | "warrior" | "hunter" | "rogue" | "priest",
 * }> | null}
 */
export function exactAuthorizedAgentIdentityV1(rawAgent) {
  const fields = exactAuthorizedIdentityFields(rawAgent);
  if (fields === null) return null;
  const presentationKey = exactIdentifier(fields.presentation_key);
  const publicAgentId = exactIdentifier(fields.public_agent_id);
  const classId = fields.class_id;
  const teamId = fields.team_id;
  const classIdentity =
    Number.isSafeInteger(classId) &&
    Object.hasOwn(CLASS_BY_ID, /** @type {number} */ (classId))
      ? CLASS_BY_ID[/** @type {keyof typeof CLASS_BY_ID} */ (classId)]
      : null;
  const teamLabel =
    Number.isSafeInteger(teamId) &&
    Object.hasOwn(TEAM_BY_ID, /** @type {number} */ (teamId))
      ? TEAM_BY_ID[/** @type {keyof typeof TEAM_BY_ID} */ (teamId)]
      : null;
  if (
    presentationKey === null ||
    publicAgentId === null ||
    classIdentity === null ||
    teamLabel === null
  ) {
    return null;
  }
  const identity = formatIdentity(publicAgentId, classId, teamId);
  return Object.freeze({
    presentationKey,
    publicAgentId,
    title: identity.title,
    publicIdentity: identity.publicIdentity,
    classLabel: identity.classLabel,
    teamLabel: identity.teamLabel,
    accent: /** @type {"mage" | "warrior" | "hunter" | "rogue" | "priest"} */ (
      identity.accent
    ),
  });
}

/** @param {unknown} publicAgentId @param {unknown} classId @param {unknown} teamId */
function formatIdentity(publicAgentId, classId, teamId) {
  const normalizedPublicId = displayIdentifier(publicAgentId);
  const classIdentity =
    Number.isInteger(classId) &&
    Object.hasOwn(CLASS_BY_ID, /** @type {number} */ (classId))
      ? CLASS_BY_ID[/** @type {keyof typeof CLASS_BY_ID} */ (classId)]
      : null;
  const teamLabel =
    Number.isInteger(teamId) &&
    Object.hasOwn(TEAM_BY_ID, /** @type {number} */ (teamId))
      ? TEAM_BY_ID[/** @type {keyof typeof TEAM_BY_ID} */ (teamId)]
      : "Unknown";
  const publicIdentity =
    normalizedPublicId === null
      ? "Agent ID unavailable"
      : `Agent ID ${normalizedPublicId}`;
  const classLabel = classIdentity?.label ?? "Unknown";
  return Object.freeze({
    title: `${publicIdentity} · ${classLabel} · ${teamLabel}`,
    publicIdentity,
    classLabel,
    teamLabel,
    accent: classIdentity?.accent ?? /** @type {const} */ ("none"),
  });
}

/** @param {unknown} value */
function displayIdentifier(value) {
  return exactIdentifier(value);
}

/** @param {unknown} value */
function exactIdentifier(value) {
  return typeof value === "string" &&
    value.length > 0 &&
    value.length <= 512 &&
    value.trim() === value &&
    !/[\r\n]/u.test(value)
    ? value
    : null;
}

/**
 * @param {unknown} value
 * @returns {Readonly<Record<string, unknown>> | null}
 */
function identityDataFields(value) {
  try {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const descriptors = Object.getOwnPropertyDescriptors(value);
    /** @type {Record<string, unknown>} */
    const fields = Object.create(null);
    for (const key of ["public_agent_id", "class_id", "team_id"]) {
      const descriptor = descriptors[key];
      fields[key] =
        descriptor && Object.hasOwn(descriptor, "value") ? descriptor.value : undefined;
    }
    return fields;
  } catch {
    return null;
  }
}

/**
 * @param {unknown} value
 * @returns {Readonly<Record<string, unknown>> | null}
 */
function exactAuthorizedIdentityFields(value) {
  try {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return null;
    const keys = Reflect.ownKeys(value);
    if (
      keys.some((key) => typeof key !== "string") ||
      AUTHORIZED_IDENTITY_KEYS.some((key) => !keys.includes(key))
    ) {
      return null;
    }
    const descriptors = Object.getOwnPropertyDescriptors(value);
    /** @type {Record<string, unknown>} */
    const fields = Object.create(null);
    for (const key of AUTHORIZED_IDENTITY_KEYS) {
      const descriptor = descriptors[key];
      if (
        !descriptor ||
        !Object.hasOwn(descriptor, "value") ||
        !descriptor.enumerable
      ) {
        return null;
      }
      fields[key] = descriptor.value;
    }
    return fields;
  } catch {
    return null;
  }
}
