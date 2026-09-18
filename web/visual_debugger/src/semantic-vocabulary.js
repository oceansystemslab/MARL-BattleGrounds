/**
 * @file Supply qualitative status/aura labels for browser explanations.
 * These frozen definitions contain no tuning quantities. Callers obtain exact
 * durations, radii and effect values from their authorized presentation.
 * Unknown tokens receive neutral labels or no lifecycle result.
 */
const STATUS_PRESENTATION = Object.freeze({
  spawn_shield: statusProfile(
    "Spawn Shield",
    "While the spawn shield is active, this agent is protected, concealed from opponents, untargetable, excluded from aura effects, and limited to movement. It can move through other agents while shielded; collision resumes at the end of the shield's final transition.",
    "none",
    "none",
  ),
  in_combat: statusProfile(
    "In Combat",
    "Shows how many transitions remain before this agent leaves combat. Participating in combat restarts the duration.",
    "none",
    "none",
  ),
  stun_warrior_charge: statusProfile(
    "Warrior (Ultimate: Charge): Stun",
    "A Warrior's concussive Charge prevents this agent's voluntary movement and combat for its duration. Physics may still displace the body.",
    "warrior",
    "none",
  ),
  stun_hunter_trap: statusProfile(
    "Hunter (Ultimate: Freezing Trap): Stun",
    "A Hunter's Freezing Trap prevents this agent's voluntary movement and combat for its duration. Physics may still displace the body.",
    "hunter",
    "none",
    true,
  ),
  stun_rogue_poison: statusProfile(
    "Rogue (Ultimate: Crippling Poison): Stun",
    "A Rogue's Crippling Poison prevents this agent's voluntary movement and combat for its duration. Physics may still displace the body.",
    "rogue",
    "none",
  ),
  slow_warrior_charge: statusProfile(
    "Warrior (Ultimate: Charge): Slow",
    "A Warrior's concussive Charge slows this agent's movement for its duration.",
    "warrior",
    "movement_multiplier",
  ),
  slow_hunter_basic: statusProfile(
    "Hunter (Basic: Attack): Slow",
    "A Hunter's Serrated Arrows slow this agent's movement for their duration.",
    "hunter",
    "movement_multiplier",
  ),
  slow_rogue_poison: statusProfile(
    "Rogue (Ultimate: Crippling Poison): Slow",
    "A Rogue's Crippling Poison slows this agent's movement for its duration.",
    "rogue",
    "movement_multiplier",
  ),
  anti_heal_rogue_poison: statusProfile(
    "Rogue (Ultimate: Crippling Poison): Anti-Heal",
    "A Rogue's Crippling Poison reduces incoming healing and out-of-combat regeneration for its duration.",
    "rogue",
    "healing_multiplier",
  ),
  priest_freedom: statusProfile(
    "Priest (Basic: Heal): Blessing of Freedom",
    "Freedom is applied when a Priest heals a same-team target, including itself where same-team targeting permits it. It prevents slow effects from reducing this agent's ordinary movement below the authorized floor; it does not override stun.",
    "priest",
    "movement_floor",
  ),
  mage_burst: statusProfile(
    "Mage (Ultimate: Burst): Damage Amplification",
    "This Mage's Burst increases its outgoing damage for the authorized duration.",
    "mage",
    "damage_multiplier",
  ),
});

const AURA_PRESENTATION = Object.freeze({
  mage_damage_amplification: auraProfile(
    "Sorcerer's Empowerment · Mage Damage Amplification Aura",
    "This Mage radiates arcane magic, amplifying outgoing damage for eligible unshielded same-team agents in its radius, including itself.",
    "This agent benefits from authorized Mage aura coverage.",
    "Damage Amplification Effect",
    "Aggregated Damage Amplification Effect",
    "mage",
    "damage_dealt",
  ),
  warrior_damage_mitigation: auraProfile(
    "Guardian's Barrier · Warrior Damage Mitigation Aura",
    "This Warrior emanates a defensive aura, mitigating incoming damage for eligible unshielded same-team agents in its radius, including itself.",
    "This agent benefits from authorized Warrior aura coverage.",
    "Damage Mitigation Effect",
    "Aggregated Damage Mitigation Effect",
    "warrior",
    "damage_received",
  ),
});

const AURA_ID_ALIASES = Object.freeze({
  mage_amplification: "mage_damage_amplification",
  warrior_mitigation: "warrior_damage_mitigation",
});

const STATUS_LIFECYCLE_PREFIX = Object.freeze({
  applied: "Applied",
  refreshed: "Refreshed",
  decremented: "Aged",
  expired: "Expired",
  trap_broken: "Broken",
  cleared_by_death: "Cleared On Death",
  cleared_unclassified: "Ended",
  trap_broken_and_reapplied: "Broken, Then Reapplied",
  reapplied: "Reapplied",
});

/**
 * Freeze a status display definition without validating its arguments.
 *
 * title and effect are authored text. accent selects the class color and
 * magnitudeKind names the kind of numerical field a caller may display.
 * positiveDamageBreak defaults to false and marks effects broken by damage.
 * Return a frozen record of those values; no quantities are computed.
 *
 * @param {string} title
 * @param {string} effect
 * @param {"mage" | "warrior" | "hunter" | "rogue" | "priest" | "none"} accent
 * @param {"none" | "movement_multiplier" | "healing_multiplier" | "damage_multiplier" | "movement_floor"} magnitudeKind
 * @param {boolean} [positiveDamageBreak]
 */
function statusProfile(
  title,
  effect,
  accent,
  magnitudeKind,
  positiveDamageBreak = false,
) {
  return Object.freeze({
    title,
    effect,
    accent,
    magnitudeKind,
    positiveDamageBreak,
  });
}

/**
 * Freeze the qualitative labels for one aura definition.
 *
 * title serves both the field and recipient cards. fieldEffect and
 * aggregateEffect describe those views; fieldEffectLabel and
 * aggregateEffectLabel name their numerical rows. accent selects Mage/Warrior
 * and effectKind distinguishes damage dealt/received. Return a frozen record
 * without validation, numerical calculation or input changes.
 *
 * @param {string} title
 * @param {string} fieldEffect
 * @param {string} aggregateEffect
 * @param {string} fieldEffectLabel
 * @param {string} aggregateEffectLabel
 * @param {"mage" | "warrior"} accent
 * @param {"damage_dealt" | "damage_received"} effectKind
 */
function auraProfile(
  title,
  fieldEffect,
  aggregateEffect,
  fieldEffectLabel,
  aggregateEffectLabel,
  accent,
  effectKind,
) {
  return Object.freeze({
    fieldTitle: title,
    recipientTitle: title,
    fieldEffect,
    aggregateEffect,
    fieldEffectLabel,
    aggregateEffectLabel,
    accent,
    effectKind,
  });
}

/**
 * Return a frozen status definition for tokenId.
 *
 * String input is trimmed before lookup. Unknown/nonstring input receives a
 * neutral Recorded Status record with no magnitude kind or damage-break flag.
 * Known definitions are shared. This lookup does not decide whether an actor
 * is allowed to see the status or supply its numerical duration.
 *
 * @param {unknown} tokenId
 */
export function statusPresentation(tokenId) {
  const key = typeof tokenId === "string" ? tokenId.trim() : "";
  return Object.hasOwn(STATUS_PRESENTATION, key)
    ? STATUS_PRESENTATION[/** @type {keyof typeof STATUS_PRESENTATION} */ (key)]
    : Object.freeze({
        title: "Recorded Status",
        effect: "Represents an authorized status effect channel.",
        accent: "none",
        magnitudeKind: "none",
        positiveDamageBreak: false,
      });
}

/**
 * Build a frozen lifecycle title/summary, or return null for an unknown event.
 *
 * lifecycleTokenId is trimmed and must name a known lifecycle. statusTokenId
 * selects a status definition, with neutral fallback. Expiry and death clearing
 * have a null summary. Exact in_combat expiry and exact stun_hunter_trap
 * breakage receive special text; those special comparisons use the untrimmed
 * status input. Other summaries use the status effect description. Inputs are
 * unchanged; visibility and duration checks remain the caller's responsibility.
 *
 * @param {unknown} statusTokenId
 * @param {unknown} lifecycleTokenId
 * @returns {Readonly<{title: string, summary: string | null}> | null}
 */
export function statusLifecyclePresentation(statusTokenId, lifecycleTokenId) {
  const lifecycleKey =
    typeof lifecycleTokenId === "string" ? lifecycleTokenId.trim() : "";
  if (!Object.hasOwn(STATUS_LIFECYCLE_PREFIX, lifecycleKey)) {
    return null;
  }
  const status = statusPresentation(statusTokenId);
  const subject = status.title.replace("): ", ") ");
  const title =
    lifecycleKey === "cleared_by_death"
      ? `Cleared ${subject} On Death`
      : lifecycleKey === "expired" && statusTokenId === "in_combat"
        ? "Out of Combat"
        : `${STATUS_LIFECYCLE_PREFIX[/** @type {keyof typeof STATUS_LIFECYCLE_PREFIX} */ (lifecycleKey)]} ${subject}`;
  const summary =
    lifecycleKey === "expired" || lifecycleKey === "cleared_by_death"
      ? null
      : lifecycleKey === "trap_broken" && statusTokenId === "stun_hunter_trap"
        ? "The Freezing Trap stun ended early because the recipient received damage."
        : status.effect;
  return Object.freeze({
    title,
    summary,
  });
}

/**
 * Return a frozen qualitative aura definition, with a neutral fallback.
 *
 * auraId is trimmed when it is a string. Support the two declared legacy
 * aliases, then look up Mage damage amplification or Warrior damage mitigation.
 * Unknown values return generic aura labels. No radius, multiplier, emitter
 * identity or permission is inferred. Known definitions are shared.
 *
 * @param {unknown} auraId
 */
export function auraPresentation(auraId) {
  const rawKey = typeof auraId === "string" ? auraId.trim() : "";
  const key = Object.hasOwn(AURA_ID_ALIASES, rawKey)
    ? AURA_ID_ALIASES[/** @type {keyof typeof AURA_ID_ALIASES} */ (rawKey)]
    : rawKey;
  return Object.hasOwn(AURA_PRESENTATION, key)
    ? AURA_PRESENTATION[/** @type {keyof typeof AURA_PRESENTATION} */ (key)]
    : Object.freeze({
        fieldTitle: "Recorded Aura Field",
        recipientTitle: "Recorded Aura",
        fieldEffect: "Represents an authorized aura field.",
        aggregateEffect: "This agent has an authorized aggregate aura effect.",
        fieldEffectLabel: "Aura Effect",
        aggregateEffectLabel: "Aggregated Aura Effect",
        accent: "none",
        effectKind: "generic",
      });
}
