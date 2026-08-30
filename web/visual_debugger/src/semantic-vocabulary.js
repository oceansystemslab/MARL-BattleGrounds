/**
 * Finite, qualitative presentation copy for stable status and aura identities.
 *
 * This registry deliberately contains no tuning values, rankings, matchup
 * claims, or outcome guarantees. Exact quantities always come from the joined
 * normalized scene record at render time.
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
 * Resolve one status lifecycle into status-specific explanatory copy. The
 * Applications retain the exact durable-badge explanation. Expiry and death
 * clearing are self-explanatory and therefore intentionally carry no summary.
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
 * Resolve only the two stable aura identities into qualitative display names.
 * Exact radii and multipliers remain outside this numeric-free registry.
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
