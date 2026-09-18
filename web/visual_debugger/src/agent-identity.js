/**
 * @file Format public agent identities for scientific browser cards. This module
 * owns class/team labels and safe identity-field reads. It never derives an ID
 * from a global slot, position or private simulator record. Source attribution
 * checks belong to the caller that joins identities to recorded references.
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
 * Return a display label from rawAgent's own data fields.
 *
 * rawAgent may be any value. Read public_agent_id, class_id, team_id and
 * optional display_agent_id without calling property getters. Invalid fields
 * independently become Unknown, Agent ID unavailable or the neutral accent.
 * A one-digit display ID takes precedence over the public ID. Return a frozen
 * record with title, publicIdentity, classLabel, teamLabel and accent. This
 * formats a label; it does not certify information rights or mutate the input.
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
  return formatIdentity(
    fields?.public_agent_id,
    fields?.class_id,
    fields?.team_id,
    fields?.display_agent_id,
  );
}

/**
 * Check a plain identity record and return its frozen display projection.
 *
 * rawAgent must have own enumerable data fields presentation_key,
 * public_agent_id, class_id and team_id. Class IDs are integers 1–5 and team IDs
 * are 1–2. IDs are nonempty strings of at most 512 characters, with no outer
 * whitespace or line breaks. Extra string-named fields are allowed; symbol
 * keys and getter-backed required fields are rejected. A valid optional
 * display_agent_id may override the shown ID. Invalid input returns null.
 * Matching these fields alone does not establish a source-attribution join.
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
  const identity = formatIdentity(
    publicAgentId,
    classId,
    teamId,
    fields.display_agent_id,
  );
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

/**
 * Create a frozen title and class/team labels from four unchecked values.
 *
 * publicAgentId supplies the shown identifier unless displayId is one decimal
 * digit. classId 1–5 selects a class/accent; teamId 1–2 selects Team A/B. Invalid
 * parts receive neutral unavailable labels independently. No coercion, input
 * mutation, source lookup or authority check occurs.
 *
 * @param {unknown} publicAgentId @param {unknown} classId @param {unknown} teamId @param {unknown} displayId
 */
function formatIdentity(publicAgentId, classId, teamId, displayId) {
  const normalizedPublicId =
    typeof displayId === "string" && /^[0-9]$/u.test(displayId)
      ? displayId
      : displayIdentifier(publicAgentId);
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

/**
 * Return value when it passes exactIdentifier, otherwise null.
 * The function preserves the string exactly; it does not trim or derive an ID.
 *
 * @param {unknown} value
 */
function displayIdentifier(value) {
  return exactIdentifier(value);
}

/**
 * Accept a nonempty string of at most 512 characters with no outer whitespace
 * or CR/LF. value is returned unchanged when valid; all other inputs return
 * null. The function performs no registry lookup or identity comparison.
 *
 * @param {unknown} value
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

/**
 * Read four own data fields from value without calling their getters.
 *
 * Reject null, arrays and nonobjects. Copy public_agent_id, class_id, team_id
 * and display_agent_id into a new null-prototype record; absent/getter-backed
 * fields become undefined. The returned record is not frozen. Descriptor
 * inspection errors return null. Field validity and authority are not checked.
 *
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
    for (const key of ["public_agent_id", "class_id", "team_id", "display_agent_id"]) {
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
 * Read required identity data from a plain object or return null.
 *
 * value must have Object.prototype or a null prototype, no symbol keys and
 * four own enumerable data fields named in AUTHORIZED_IDENTITY_KEYS. Extra
 * string fields are allowed. Copy those fields and an eligible display_agent_id
 * into a new null-prototype record, without freezing it or changing value.
 * Inspection errors return null; field values are checked by the next helper.
 *
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
    const display = descriptors.display_agent_id;
    if (display && Object.hasOwn(display, "value") && display.enumerable) {
      fields.display_agent_id = display.value;
    }
    return fields;
  } catch {
    return null;
  }
}
