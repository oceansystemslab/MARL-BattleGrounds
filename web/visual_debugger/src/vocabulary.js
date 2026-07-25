/**
 * Display-only vocabulary for renderer-neutral debugger scene records.
 *
 * This module maps stable semantic IDs to labels and visual keys. It does not
 * define durations, legality, ranges, acceptance, combat values, or any other
 * simulator behavior.
 */

/**
 * @typedef {"class" | "team" | "status" | "activation" | "modifier" | "lifecycle"} VisualTokenKind
 */

/**
 * @typedef {{
 *   tokenId: string,
 *   label: string,
 *   shortLabel: string,
 *   accessibleName: string,
 *   glyphKey: string,
 *   cssKey: string,
 *   fallback: string,
 * }} VisualToken
 */

/**
 * @typedef {Readonly<Record<string, Readonly<VisualToken>>>} VisualTokenRegistry
 */

/**
 * @typedef {{
 *   label?: unknown,
 *   short_label?: unknown,
 *   shortLabel?: unknown,
 *   accessible_name?: unknown,
 *   accessibleName?: unknown,
 * }} TokenPayload
 */

/**
 * @param {VisualToken} definition
 * @returns {Readonly<VisualToken>}
 */
function token(definition) {
  return Object.freeze(definition);
}

const CLASS_TOKENS = Object.freeze({
  mage: token({
    tokenId: "mage",
    label: "Mage",
    shortLabel: "Mage",
    accessibleName: "Mage class",
    glyphKey: "class-mage",
    cssKey: "mage",
    fallback: "M",
  }),
  warrior: token({
    tokenId: "warrior",
    label: "Warrior",
    shortLabel: "Warrior",
    accessibleName: "Warrior class",
    glyphKey: "class-warrior",
    cssKey: "warrior",
    fallback: "W",
  }),
  hunter: token({
    tokenId: "hunter",
    label: "Hunter",
    shortLabel: "Hunter",
    accessibleName: "Hunter class",
    glyphKey: "class-hunter",
    cssKey: "hunter",
    fallback: "H",
  }),
  rogue: token({
    tokenId: "rogue",
    label: "Rogue",
    shortLabel: "Rogue",
    accessibleName: "Rogue class",
    glyphKey: "class-rogue",
    cssKey: "rogue",
    fallback: "R",
  }),
  priest: token({
    tokenId: "priest",
    label: "Priest",
    shortLabel: "Priest",
    accessibleName: "Priest class",
    glyphKey: "class-priest",
    cssKey: "priest",
    fallback: "P",
  }),
});

const TEAM_TOKENS = Object.freeze({
  team_a: token({
    tokenId: "team_a",
    label: "Team A",
    shortLabel: "A",
    accessibleName: "Team A, solid outline",
    glyphKey: "team-a",
    cssKey: "team-a",
    fallback: "A",
  }),
  team_b: token({
    tokenId: "team_b",
    label: "Team B",
    shortLabel: "B",
    accessibleName: "Team B, solid outline",
    glyphKey: "team-b",
    cssKey: "team-b",
    fallback: "B",
  }),
});

const STATUS_TOKENS = Object.freeze({
  stun_warrior_charge: token({
    tokenId: "stun_warrior_charge",
    label: "Charge stun",
    shortLabel: "C-STN",
    accessibleName: "Warrior Charge stun",
    glyphKey: "status-charge-stun",
    cssKey: "stun-warrior-charge",
    fallback: "CS",
  }),
  stun_hunter_trap: token({
    tokenId: "stun_hunter_trap",
    label: "Trap",
    shortLabel: "TRAP",
    accessibleName: "Hunter Trap stun",
    glyphKey: "status-trap",
    cssKey: "stun-hunter-trap",
    fallback: "T",
  }),
  stun_rogue_poison: token({
    tokenId: "stun_rogue_poison",
    label: "Poison stun",
    shortLabel: "P-STN",
    accessibleName: "Rogue Poison stun",
    glyphKey: "status-poison-stun",
    cssKey: "stun-rogue-poison",
    fallback: "PS",
  }),
  slow_warrior_charge: token({
    tokenId: "slow_warrior_charge",
    label: "Charge slow",
    shortLabel: "C-SLW",
    accessibleName: "Warrior Charge slow",
    glyphKey: "status-slow",
    cssKey: "slow-warrior-charge",
    fallback: "CS",
  }),
  slow_hunter_basic: token({
    tokenId: "slow_hunter_basic",
    label: "Hunter slow",
    shortLabel: "H-SLW",
    accessibleName: "Hunter Basic slow",
    glyphKey: "status-slow",
    cssKey: "slow-hunter-basic",
    fallback: "HS",
  }),
  slow_rogue_poison: token({
    tokenId: "slow_rogue_poison",
    label: "Poison slow",
    shortLabel: "P-SLW",
    accessibleName: "Rogue Poison slow",
    glyphKey: "status-slow",
    cssKey: "slow-rogue-poison",
    fallback: "PS",
  }),
  anti_heal_rogue_poison: token({
    tokenId: "anti_heal_rogue_poison",
    label: "Anti-heal",
    shortLabel: "ANTI",
    accessibleName: "Rogue Poison anti-heal",
    glyphKey: "status-anti-heal",
    cssKey: "anti-heal-rogue-poison",
    fallback: "AH",
  }),
  priest_freedom: token({
    tokenId: "priest_freedom",
    label: "Freedom",
    shortLabel: "FREE",
    accessibleName: "Priest Blessing of Freedom",
    glyphKey: "status-freedom",
    cssKey: "priest-freedom",
    fallback: "F",
  }),
  mage_burst: token({
    tokenId: "mage_burst",
    label: "Burst",
    shortLabel: "BURST",
    accessibleName: "Mage Burst damage amplification",
    glyphKey: "status-burst",
    cssKey: "mage-burst",
    fallback: "B",
  }),
});

const ACTIVATION_TOKENS = Object.freeze({
  basic_damage: token({
    tokenId: "basic_damage",
    label: "Basic damage",
    shortLabel: "Basic",
    accessibleName: "Accepted Basic damage activation",
    glyphKey: "activation-basic-damage",
    cssKey: "basic-damage",
    fallback: "B",
  }),
  basic_heal: token({
    tokenId: "basic_heal",
    label: "Basic healing",
    shortLabel: "Basic",
    accessibleName: "Accepted Priest Basic healing activation",
    glyphKey: "activation-basic-heal",
    cssKey: "basic-heal",
    fallback: "B",
  }),
  holy_word: token({
    tokenId: "holy_word",
    label: "Holy Word",
    shortLabel: "Holy",
    accessibleName: "Accepted Priest Holy Word activation",
    glyphKey: "activation-holy-word",
    cssKey: "holy-word",
    fallback: "U",
  }),
  mage_burst: token({
    tokenId: "mage_burst",
    label: "Burst activation",
    shortLabel: "Burst",
    accessibleName: "Accepted Mage Burst activation",
    glyphKey: "activation-burst",
    cssKey: "mage-burst",
    fallback: "U",
  }),
  warrior_charge: token({
    tokenId: "warrior_charge",
    label: "Charge",
    shortLabel: "Charge",
    accessibleName: "Accepted Warrior Charge activation",
    glyphKey: "activation-charge",
    cssKey: "warrior-charge",
    fallback: "U",
  }),
  hunter_trap: token({
    tokenId: "hunter_trap",
    label: "Trap activation",
    shortLabel: "Trap",
    accessibleName: "Accepted Hunter Trap activation",
    glyphKey: "activation-trap",
    cssKey: "hunter-trap",
    fallback: "U",
  }),
  rogue_poison: token({
    tokenId: "rogue_poison",
    label: "Poison",
    shortLabel: "Poison",
    accessibleName: "Accepted Rogue Poison activation",
    glyphKey: "activation-poison",
    cssKey: "rogue-poison",
    fallback: "U",
  }),
});

const MODIFIER_TOKENS = Object.freeze({
  mage_amplification: token({
    tokenId: "mage_amplification",
    label: "Mage aura amplification",
    shortLabel: "AMP",
    accessibleName: "Effective Mage damage amplification aura modifier",
    glyphKey: "modifier-amplification",
    cssKey: "mage-amplification",
    fallback: "AMP",
  }),
  warrior_mitigation: token({
    tokenId: "warrior_mitigation",
    label: "Warrior aura mitigation",
    shortLabel: "MIT",
    accessibleName: "Effective Warrior damage mitigation aura modifier",
    glyphKey: "modifier-mitigation",
    cssKey: "warrior-mitigation",
    fallback: "MIT",
  }),
  rogue_anti_heal: token({
    tokenId: "rogue_anti_heal",
    label: "Anti-heal modifier",
    shortLabel: "ANTI",
    accessibleName: "Effective Rogue Poison anti-heal modifier",
    glyphKey: "status-anti-heal",
    cssKey: "rogue-anti-heal",
    fallback: "AH",
  }),
  priest_freedom: token({
    tokenId: "priest_freedom",
    label: "Freedom speed floor",
    shortLabel: "FREE",
    accessibleName: "Effective Priest Freedom movement-speed floor",
    glyphKey: "status-freedom",
    cssKey: "priest-freedom",
    fallback: "F",
  }),
  mage_burst: token({
    tokenId: "mage_burst",
    label: "Burst amplification",
    shortLabel: "BURST",
    accessibleName: "Effective Mage Burst damage amplification modifier",
    glyphKey: "status-burst",
    cssKey: "mage-burst",
    fallback: "B",
  }),
});

const LIFECYCLE_TOKENS = Object.freeze({
  applied: token({
    tokenId: "applied",
    label: "Applied",
    shortLabel: "Apply",
    accessibleName: "Status applied",
    glyphKey: "lifecycle-applied",
    cssKey: "applied",
    fallback: "+",
  }),
  refreshed: token({
    tokenId: "refreshed",
    label: "Refreshed",
    shortLabel: "Refresh",
    accessibleName: "Status refreshed or reapplied",
    glyphKey: "lifecycle-refreshed",
    cssKey: "refreshed",
    fallback: "R",
  }),
  decremented: token({
    tokenId: "decremented",
    label: "Aged",
    shortLabel: "Age",
    accessibleName: "Status duration decremented",
    glyphKey: "lifecycle-decremented",
    cssKey: "decremented",
    fallback: "-",
  }),
  expired: token({
    tokenId: "expired",
    label: "Expired",
    shortLabel: "Expire",
    accessibleName: "Status expired naturally",
    glyphKey: "lifecycle-expired",
    cssKey: "expired",
    fallback: "E",
  }),
  trap_broken: token({
    tokenId: "trap_broken",
    label: "Trap broken",
    shortLabel: "Break",
    accessibleName: "Hunter Trap ended by accepted damage",
    glyphKey: "lifecycle-trap-broken",
    cssKey: "trap-broken",
    fallback: "X",
  }),
  cleared_unclassified: token({
    tokenId: "cleared_unclassified",
    label: "Status ended",
    shortLabel: "End",
    accessibleName: "Status ended for an unclassified or ambiguous reason",
    glyphKey: "lifecycle-ended",
    cssKey: "cleared-unclassified",
    fallback: "?",
  }),
  trap_broken_and_reapplied: token({
    tokenId: "trap_broken_and_reapplied",
    label: "Trap broken and reapplied",
    shortLabel: "Break+",
    accessibleName: "Hunter Trap was broken and exactly reapplied",
    glyphKey: "lifecycle-trap-broken-reapplied",
    cssKey: "trap-broken-and-reapplied",
    fallback: "X+",
  }),
});

export const CLASS_TOKEN_IDS = Object.freeze([
  "mage",
  "warrior",
  "hunter",
  "rogue",
  "priest",
]);

export const TEAM_TOKEN_IDS = Object.freeze(["team_a", "team_b"]);

export const CANONICAL_STATUS_ORDER = Object.freeze([
  "stun_warrior_charge",
  "stun_hunter_trap",
  "stun_rogue_poison",
  "slow_warrior_charge",
  "slow_hunter_basic",
  "slow_rogue_poison",
  "anti_heal_rogue_poison",
  "priest_freedom",
  "mage_burst",
]);

export const ACTIVATION_TOKEN_IDS = Object.freeze([
  "basic_damage",
  "basic_heal",
  "holy_word",
  "mage_burst",
  "warrior_charge",
  "hunter_trap",
  "rogue_poison",
]);

export const MODIFIER_TOKEN_IDS = Object.freeze([
  "mage_amplification",
  "warrior_mitigation",
  "rogue_anti_heal",
  "priest_freedom",
  "mage_burst",
]);

export const LIFECYCLE_TOKEN_IDS = Object.freeze([
  "applied",
  "refreshed",
  "decremented",
  "expired",
  "trap_broken",
  "cleared_unclassified",
  "trap_broken_and_reapplied",
]);

/** @type {Readonly<Record<VisualTokenKind, VisualTokenRegistry>>} */
const TOKEN_REGISTRIES = Object.freeze({
  class: CLASS_TOKENS,
  team: TEAM_TOKENS,
  status: STATUS_TOKENS,
  activation: ACTIVATION_TOKENS,
  modifier: MODIFIER_TOKENS,
  lifecycle: LIFECYCLE_TOKENS,
});

/** @type {Readonly<Record<number, string>>} */
const CLASS_ID_TO_TOKEN = Object.freeze({
  1: "mage",
  2: "warrior",
  3: "hunter",
  4: "rogue",
  5: "priest",
});

/** @type {Readonly<Record<number, string>>} */
const TEAM_ID_TO_TOKEN = Object.freeze({
  1: "team_a",
  2: "team_b",
});

/**
 * @param {unknown} value
 * @returns {string | null}
 */
function displayString(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/**
 * @param {unknown} payload
 * @returns {TokenPayload}
 */
function tokenPayload(payload) {
  return typeof payload === "object" && payload !== null
    ? /** @type {TokenPayload} */ (payload)
    : {};
}

/**
 * @param {unknown} tokenId
 * @returns {string}
 */
function normalizedTokenId(tokenId) {
  return displayString(tokenId) ?? "unknown";
}

/**
 * @param {VisualTokenKind} kind
 * @param {string} tokenId
 * @returns {Readonly<VisualToken>}
 */
function unknownToken(kind, tokenId) {
  return token({
    tokenId,
    label: "Unknown",
    shortLabel: "?",
    accessibleName: `Unknown ${kind} visual token ${tokenId}`,
    glyphKey: "unknown",
    cssKey: "unknown",
    fallback: "?",
  });
}

/**
 * Apply Python-authored display prose without allowing payloads to select
 * glyphs or CSS hooks.
 *
 * @param {Readonly<VisualToken>} definition
 * @param {unknown} payload
 * @returns {Readonly<VisualToken>}
 */
function withPayloadProse(definition, payload) {
  const record = tokenPayload(payload);
  const label = displayString(record.label) ?? definition.label;
  const shortLabel =
    displayString(record.short_label) ??
    displayString(record.shortLabel) ??
    definition.shortLabel;
  const accessibleName =
    displayString(record.accessible_name) ??
    displayString(record.accessibleName) ??
    definition.accessibleName;

  if (
    label === definition.label &&
    shortLabel === definition.shortLabel &&
    accessibleName === definition.accessibleName
  ) {
    return definition;
  }
  return token({
    ...definition,
    label,
    shortLabel,
    accessibleName,
  });
}

/**
 * Resolve a stable semantic ID to display-only metadata.
 *
 * Python-provided labels and accessible names override registry prose. Glyph
 * and CSS keys always come from this allowlisted registry so future/unknown
 * payload IDs cannot inject selectors or markup.
 *
 * @param {VisualTokenKind} kind
 * @param {unknown} tokenId
 * @param {unknown} [payload]
 * @returns {Readonly<VisualToken>}
 */
export function resolveVisualToken(kind, tokenId, payload) {
  const normalized = normalizedTokenId(tokenId);
  const registry = TOKEN_REGISTRIES[kind];
  const definition = Object.hasOwn(registry, normalized)
    ? registry[normalized]
    : unknownToken(kind, normalized);
  return withPayloadProse(definition, payload);
}

/**
 * Resolve a simulator class identity without copying class mechanics.
 *
 * @param {unknown} classId
 * @param {unknown} [payload]
 * @returns {Readonly<VisualToken>}
 */
export function classTokenFromId(classId, payload) {
  const numericId =
    typeof classId === "number" && Number.isInteger(classId) ? classId : null;
  const tokenId = numericId === null ? undefined : CLASS_ID_TO_TOKEN[numericId];
  return resolveVisualToken(
    "class",
    tokenId ?? (numericId === null ? "unknown" : `class_id_${numericId}`),
    payload,
  );
}

/**
 * Resolve a simulator team identity without copying team mechanics.
 *
 * @param {unknown} teamId
 * @param {unknown} [payload]
 * @returns {Readonly<VisualToken>}
 */
export function teamTokenFromId(teamId, payload) {
  const numericId =
    typeof teamId === "number" && Number.isInteger(teamId) ? teamId : null;
  const tokenId = numericId === null ? undefined : TEAM_ID_TO_TOKEN[numericId];
  return resolveVisualToken(
    "team",
    tokenId ?? (numericId === null ? "unknown" : `team_id_${numericId}`),
    payload,
  );
}
