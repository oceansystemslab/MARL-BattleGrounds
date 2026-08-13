/**
 * Finite, qualitative presentation copy for stable class and status identities.
 *
 * This registry deliberately contains no tuning values, rankings, matchup
 * claims, or outcome guarantees. Exact quantities always come from the joined
 * normalized scene record at render time.
 */

const CLASS_PRESENTATION = Object.freeze({
  mage: profile({
    ultimateName: "Burst",
    role: "Area pressure and allied damage support.",
    strengths: "Can support allied damage and apply a timed amplification effect.",
    limitations: "Positioning and recorded availability constrain its actions.",
    teamwork: "Coordinates damage windows with nearby allies.",
    counterplay: "Track its visible range, statuses, and cooldown before committing.",
  }),
  warrior: profile({
    ultimateName: "Charge",
    role: "Front-line control and allied damage mitigation.",
    strengths: "Can displace itself and apply recorded control effects.",
    limitations: "Must work within its recorded target and range constraints.",
    teamwork: "Supports nearby allies through its mitigation field.",
    counterplay: "Respect its visible control range and cooldown state.",
  }),
  hunter: profile({
    ultimateName: "Freezing Trap",
    role: "Ranged control and target pressure.",
    strengths: "Can apply recorded slow and stun effects at range.",
    limitations: "Target access and cooldown state constrain its control windows.",
    teamwork: "Creates control windows teammates can coordinate around.",
    counterplay: "Watch its visible ranges and current availability.",
  }),
  rogue: profile({
    ultimateName: "Crippling Poison",
    role: "Target disruption and persistent control.",
    strengths: "Can apply recorded slow, stun, and healing-reduction effects.",
    limitations: "Its effects are timed and depend on an authorized target.",
    teamwork: "Disrupts a target while allies apply their own pressure or support.",
    counterplay: "Track current status durations and the Rogue's cooldown.",
  }),
  priest: profile({
    ultimateName: "Holy Word: Salvation",
    role: "Allied healing and control support.",
    strengths: "Can restore allied health and apply a recorded freedom effect.",
    limitations:
      "Support actions remain subject to recorded target and range constraints.",
    teamwork: "Sustains allies and helps them act through recorded control effects.",
    counterplay: "Track its visible support range and cooldown state.",
  }),
});

const STATUS_PRESENTATION = Object.freeze({
  stun_warrior_charge: statusProfile(
    "Warrior (Ultimate: Charge) Stun",
    "A Warrior's concussive Charge incapacitates this agent, preventing action while the status remains.",
    "warrior",
  ),
  stun_hunter_trap: statusProfile(
    "Hunter (Ultimate: Freezing Trap) Stun",
    "A Hunter's Freezing Trap incapacitates this agent, preventing action while the status remains.",
    "hunter",
  ),
  stun_rogue_poison: statusProfile(
    "Rogue (Ultimate: Crippling Poison) Stun",
    "A Rogue's Crippling Poison incapacitates this agent, preventing action while the status remains.",
    "rogue",
  ),
  slow_warrior_charge: statusProfile(
    "Warrior (Ultimate: Charge) Slow",
    "A Warrior's concussive Charge slows this agent while the status remains.",
    "warrior",
  ),
  slow_hunter_basic: statusProfile(
    "Hunter (Basic: Attack) Slow",
    "A Hunter's serrated arrows slow this agent while the status remains.",
    "hunter",
  ),
  slow_rogue_poison: statusProfile(
    "Rogue (Ultimate: Crippling Poison) Slow",
    "A Rogue's Crippling Poison slows this agent while the status remains.",
    "rogue",
  ),
  anti_heal_rogue_poison: statusProfile(
    "Rogue (Ultimate: Crippling Poison) Anti-Heal",
    "A Rogue's Crippling Poison reduces the healing this agent receives while the status remains.",
    "rogue",
  ),
  priest_freedom: statusProfile(
    "Priest (Basic: Heal) Freedom",
    "A Priest's healing uplifts this agent, enforcing the recorded movement floor while the status remains.",
    "priest",
  ),
  mage_burst: statusProfile(
    "Mage (Ultimate: Burst) Damage Amplification",
    "Burst fills this Mage with magical energy, increasing damage dealt while the status remains.",
    "mage",
  ),
});

const AURA_PRESENTATION = Object.freeze({
  mage_damage_amplification: auraProfile(
    "Sorcerer’s Empowerment",
    "Sorcerer’s Empowerment",
    "mage",
    "damage_dealt",
  ),
  warrior_damage_mitigation: auraProfile(
    "Guardian’s Barrier",
    "Guardian’s Barrier",
    "warrior",
    "damage_received",
  ),
});

const AURA_ID_ALIASES = Object.freeze({
  mage_amplification: "mage_damage_amplification",
  warrior_mitigation: "warrior_damage_mitigation",
});

const UNKNOWN_CLASS = profile({
  ultimateName: "Ultimate",
  role: "Class role not recorded.",
  strengths: "Class strengths not recorded.",
  limitations: "Class limitations not recorded.",
  teamwork: "Class teamwork guidance not recorded.",
  counterplay: "Class counterplay guidance not recorded.",
});

/** @param {Record<string, string>} value */
function profile(value) {
  return Object.freeze(value);
}

/**
 * @param {string} title
 * @param {string} effect
 * @param {"mage" | "warrior" | "hunter" | "rogue" | "priest"} accent
 */
function statusProfile(title, effect, accent) {
  return Object.freeze({ title, effect, accent });
}

/**
 * @param {string} fieldTitle
 * @param {string} recipientTitle
 * @param {"mage" | "warrior"} accent
 * @param {"damage_dealt" | "damage_received"} effectKind
 */
function auraProfile(fieldTitle, recipientTitle, accent, effectKind) {
  return Object.freeze({ fieldTitle, recipientTitle, accent, effectKind });
}

/**
 * @param {unknown} className
 */
export function classPresentation(className) {
  const key =
    typeof className === "string" ? className.trim().toLowerCase() : "unknown";
  return Object.hasOwn(CLASS_PRESENTATION, key)
    ? CLASS_PRESENTATION[/** @type {keyof typeof CLASS_PRESENTATION} */ (key)]
    : UNKNOWN_CLASS;
}

/**
 * @param {unknown} tokenId
 */
export function statusPresentation(tokenId) {
  const key = typeof tokenId === "string" ? tokenId.trim() : "";
  return Object.hasOwn(STATUS_PRESENTATION, key)
    ? STATUS_PRESENTATION[/** @type {keyof typeof STATUS_PRESENTATION} */ (key)]
    : Object.freeze({
        title: "Recorded status",
        effect: "Represents an authorized status effect channel.",
        accent: "none",
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
        accent: "none",
        effectKind: "generic",
      });
}
