const CANONICAL_PROFILE_ID = "marl_battlegrounds.class_documentation.canonical_v1";
const TEMPLATE_TOKEN = /\{\{([A-Za-z][A-Za-z0-9]*)\}\}/gu;

/** @param {string} label @param {string} value */
function tacticalRow(label, value) {
  return Object.freeze({ label, value });
}

/**
 * @param {{
 *   overview: string,
 *   tacticalGuideRows: readonly Readonly<{label: string, value: string}>[],
 *   ultimateName: string,
 *   ultimateTemplate: string,
 *   passiveName: string,
 *   passiveTemplate: string,
 * }} value
 */
function definition(value) {
  const requiredValueNames = [
    ...new Set(
      [value.ultimateTemplate, value.passiveTemplate].flatMap((template) =>
        [...template.matchAll(TEMPLATE_TOKEN)].map((match) => match[1]),
      ),
    ),
  ];
  return Object.freeze({
    ...value,
    tacticalGuideRows: Object.freeze([...value.tacticalGuideRows]),
    requiredValueNames: Object.freeze(requiredValueNames),
  });
}

const CLASS_DOCUMENTATION = Object.freeze({
  1: definition({
    overview:
      "The Mage has the highest Basic damage per tick in the game, which is further enhanced by its own passive and Ultimate (Burst) damage amplification effects. This skill set makes the Mage the game’s ranged “Artillery” class. Despite its high damage output, it is also the most vulnerable: the Mage has the lowest health in the game, making it a “Glass Cannon” that is reliant on strong team formations and allies for protection.",
    tacticalGuideRows: [
      tacticalRow(
        "Role",
        "Primary Damage Dealer · Burst Damage · Damage Amplification",
      ),
      tacticalRow(
        "Primary Strength",
        "Highest Basic damage among damage-dealing classes.",
      ),
      tacticalRow("Primary Weakness", "Lowest maximum health; glass cannon."),
    ],
    ultimateName: "Burst",
    ultimateTemplate:
      "For {{burstDuration}}, Burst applies {{burstDamageEffect}} to this Mage's damage, starting on the next tick.",
    passiveName: "Sorcerer's Empowerment (Mage Damage Amplification Aura)",
    passiveTemplate:
      "While alive and without a spawn shield, the Mage emits Sorcerer's Empowerment. It strengthens the attacks of living teammates within {{auraRadius}} who also have no spawn shield, including the Mage itself. Each aura provides {{perEmitterDamageAmplificationEffect}}. Overlapping auras multiply their damage bonuses, up to {{damageAmplificationCeiling}} times normal damage.",
  }),
  2: definition({
    overview:
      "The Warrior has the highest maximum health in the game—twice that of the next-highest classes—so it is the game’s melee “Tank”/“Guardian” class. The Warrior thrives on the front line where it can initiate engagements, provide passive damage mitigation to allies, and body-block or peel aggressors. The Warrior’s Ultimate (Charge) propels it toward an enemy, instantaneously applying considerable damage, a stun, and a slow. With relatively low follow-up damage, the Warrior is reliant on its allies to secure kills.",
    tacticalGuideRows: [
      tacticalRow("Role", "Tank · Frontline Damage · Damage Mitigation"),
      tacticalRow("Primary Strength", "Highest maximum health."),
      tacticalRow(
        "Primary Weakness",
        "Second-lowest Basic damage among damage-dealing classes.",
      ),
    ],
    ultimateName: "Charge",
    ultimateTemplate:
      "Charge moves the Warrior toward an enemy before normal movement. It deals {{ultimateRawDamage}} base damage, stuns for {{chargeStunDuration}}, and applies {{chargeSlowEffect}} for {{chargeSlowDuration}}. Damage bonuses and reductions affect the damage dealt.",
    passiveName: "Guardian's Barrier (Warrior Damage Mitigation Aura)",
    passiveTemplate:
      "While alive and without a spawn shield, the Warrior emits Guardian's Barrier. It protects living teammates within {{auraRadius}} who also have no spawn shield, including the Warrior itself. Each barrier reduces incoming damage by {{perEmitterDamageMitigationEffect}}. When barriers overlap, each reduces the damage left after the others, down to {{damageMitigationFloor}} of the original incoming damage.",
  }),
  3: definition({
    overview:
      "The Hunter is the game’s ranged “Crowd Controller” class. Its job is to use its superior Basic attack and observation range to scout out and slow or disable enemies to help its allies win engagements. The Hunter’s low damage makes it reliant on allies to secure kills, but its Ultimate (Freezing Trap) is the longest stun in the game and, if timed well, can change the tide of battle—but be careful not to break it!",
    tacticalGuideRows: [
      tacticalRow("Role", "Disabler · Crowd Controller · Scout"),
      tacticalRow("Primary Strength", "Longest stun duration."),
      tacticalRow(
        "Primary Weakness",
        "Lowest Basic damage among damage-dealing classes.",
      ),
    ],
    ultimateName: "Freezing Trap",
    ultimateTemplate:
      "Freezing Trap deals {{ultimateRawDamage}} base damage to one enemy and stuns it for {{trapStunDuration}}. Damage bonuses and reductions affect the damage dealt. Damage breaks an existing trap.",
    passiveName: "Serrated Arrows",
    passiveTemplate:
      "Each successful Hunter Basic attack applies Serrated Arrows, causing {{hunterBasicMovementEffect}} for {{hunterBasicSlowDuration}}. Further Basic attacks refresh the slow.",
  }),
  4: definition({
    overview:
      "Having the highest base movement speed in the game—alongside a high-damage Ultimate (Crippling Poison) that also stuns, slows, and applies an anti-heal effect—the Rogue is the melee “Assassin” class. Because it is a melee class and lacks a gap closer like the Warrior’s “Charge” Ultimate, it is heavily reliant on surrounding obstacles and advancing allies to execute flanking maneuvers without being counterattacked.",
    tacticalGuideRows: [
      tacticalRow("Role", "Ambusher · Flanker · Assassin · Anti-Heal"),
      tacticalRow("Primary Strength", "Highest base movement speed."),
      tacticalRow("Primary Weakness", "Short attack range with no gap closer."),
    ],
    ultimateName: "Crippling Poison",
    ultimateTemplate:
      "Crippling Poison deals {{ultimateRawDamage}} base damage to one enemy, stuns it for {{poisonStunDuration}}, and applies {{poisonSlowEffect}} for {{poisonSlowDuration}}. It also applies {{poisonAntiHealEffect}} to healing and out-of-combat regeneration for {{poisonAntiHealDuration}}. Damage bonuses and reductions affect the damage dealt.",
    passiveName: "Phantom's Quickness",
    passiveTemplate:
      "Phantom's Quickness gives the Rogue a base movement speed of {{baseMovementSpeed}}, the highest in the game. After {{outOfCombatDelay}} out of combat, it regenerates health each tick at the rate shown under Out-of-Combat Regeneration.",
  }),
  5: definition({
    overview:
      "The Priest is the game’s ranged “Healer” class. It cannot deal damage, so it relies on strong team formations and allies for protection. Its Ultimate (Holy Word: Salvation) delivers a massive amount of healing in a single tick, allowing it to save allies who would otherwise die in emergency situations.",
    tacticalGuideRows: [
      tacticalRow("Role", "Healer · Medic"),
      tacticalRow("Primary Strength", "Only class with healing abilities."),
      tacticalRow("Primary Weakness", "Cannot deal damage."),
    ],
    ultimateName: "Holy Word: Salvation",
    ultimateTemplate:
      "Holy Word: Salvation heals one ally for {{ultimateRawHealing}}, before healing reductions and the maximum-health limit.",
    passiveName: "Blessing of Freedom",
    passiveTemplate:
      "Each successful use of the Priest's Basic healing ability applies Blessing of Freedom to its target for {{freedomDuration}}. The Priest can target itself. Freedom prevents slows from reducing movement below {{freedomMovementFloor}}, but it does not prevent stuns.",
  }),
});

/**
 * Snapshot an exact plain record without invoking accessors or coercions.
 *
 * @param {unknown} value
 * @param {readonly string[]} expectedKeys
 * @returns {Readonly<Record<string, string>> | null}
 */
function exactStringRecord(value, expectedKeys) {
  try {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return null;
    const keys = Reflect.ownKeys(value);
    if (
      keys.length !== expectedKeys.length ||
      keys.some((key) => typeof key !== "string") ||
      expectedKeys.some((key) => !keys.includes(key))
    ) {
      return null;
    }
    const descriptors = Object.getOwnPropertyDescriptors(value);
    /** @type {Record<string, string>} */
    const snapshot = Object.create(null);
    for (const key of expectedKeys) {
      const descriptor = descriptors[key];
      if (!descriptor || !("value" in descriptor) || !descriptor.enumerable) {
        return null;
      }
      const item = descriptor.value;
      if (
        typeof item !== "string" ||
        item.length === 0 ||
        item.trim() !== item ||
        /[\r\n]/u.test(item) ||
        /\bunavailable\b/iu.test(item)
      ) {
        return null;
      }
      snapshot[key] = item;
    }
    return snapshot;
  } catch {
    return null;
  }
}

/** @param {string} template @param {Readonly<Record<string, string>>} values */
function interpolate(template, values) {
  return template.replace(TEMPLATE_TOKEN, (_token, name) => values[name]);
}

/**
 * @param {unknown} documentationProfile
 * @param {unknown} classId
 */
function certifiedDefinition(documentationProfile, classId) {
  const profile = exactStringRecord(documentationProfile, [
    "availability_kind",
    "profile_id",
  ]);
  if (
    profile?.availability_kind !== "available" ||
    profile.profile_id !== CANONICAL_PROFILE_ID ||
    !Number.isSafeInteger(classId) ||
    /** @type {number} */ (classId) < 1 ||
    /** @type {number} */ (classId) > 5
  ) {
    return null;
  }
  return CLASS_DOCUMENTATION[/** @type {1 | 2 | 3 | 4 | 5} */ (classId)];
}

/**
 * Return the exact ordered interpolation keys for one certified class guide.
 *
 * @param {unknown} documentationProfile
 * @param {unknown} classId
 * @returns {readonly string[] | null}
 */
export function requiredClassDocumentationValueNamesV1(documentationProfile, classId) {
  return certifiedDefinition(documentationProfile, classId)?.requiredValueNames ?? null;
}

/**
 * Resolve exact authored documentation only for Python's certified profile.
 * All quantities must arrive as named, already-formatted authorized strings.
 *
 * @param {unknown} documentationProfile
 * @param {unknown} classId
 * @param {unknown} authorizedValues
 * @returns {Readonly<{
 *   overview: string,
 *   tacticalGuideRows: readonly Readonly<{label: string, value: string}>[],
 *   ultimate: Readonly<{name: string, description: string}>,
 *   passive: Readonly<{name: string, description: string}>,
 * }> | null}
 */
export function resolveClassDocumentationV1(
  documentationProfile,
  classId,
  authorizedValues,
) {
  const selected = certifiedDefinition(documentationProfile, classId);
  if (selected === null) return null;
  const values = exactStringRecord(authorizedValues, selected.requiredValueNames);
  if (values === null) return null;
  return Object.freeze({
    overview: selected.overview,
    tacticalGuideRows: selected.tacticalGuideRows,
    ultimate: Object.freeze({
      name: selected.ultimateName,
      description: interpolate(selected.ultimateTemplate, values),
    }),
    passive: Object.freeze({
      name: selected.passiveName,
      description: interpolate(selected.passiveTemplate, values),
    }),
  });
}
