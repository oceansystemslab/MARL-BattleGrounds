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
      "The Mage is a ranged damage dealer with the lowest canonical maximum health and highest canonical basic raw damage. Burst temporarily amplifies the Mage's outgoing damage.",
    tacticalGuideRows: [
      tacticalRow("Role", "Ranged Damage · Damage Amplification"),
      tacticalRow(
        "Primary Strength",
        "Highest basic raw damage among canonical damage-dealing classes.",
      ),
      tacticalRow("Primary Weakness", "Lowest canonical maximum health; glass cannon."),
    ],
    ultimateName: "Burst",
    ultimateTemplate:
      "For {{burstDuration}}, Burst multiplies this Mage's outgoing damage by {{burstDamageEffect}}, beginning with the successor decision.",
    passiveName: "Sorcerer's Empowerment (Mage Damage Amplification Aura)",
    passiveTemplate:
      "An eligible unshielded Mage emits Sorcerer's Empowerment. Eligible unshielded same-team agents within {{auraRadius}}, including the Mage, receive {{perEmitterDamageAmplificationEffect}}; overlapping emitters multiply up to {{damageAmplificationCeiling}}.",
  }),
  2: definition({
    overview:
      "The Warrior has the highest canonical maximum health. Charge moves the Warrior toward an enemy target and applies raw damage, stun, and slow.",
    tacticalGuideRows: [
      tacticalRow("Role", "Frontline Damage · Stun · Slow · Damage Mitigation"),
      tacticalRow("Primary Strength", "Highest canonical maximum health."),
      tacticalRow(
        "Primary Weakness",
        "Second-lowest positive basic raw damage among canonical damage-dealing classes.",
      ),
    ],
    ultimateName: "Charge",
    ultimateTemplate:
      "Charge moves the Warrior toward an enemy target during the Charge phase before ordinary movement. The accepted ultimate also applies {{ultimateRawDamage}} raw damage before source and recipient damage modifiers, {{chargeStunDuration}} of stun, and {{chargeSlowEffect}} for {{chargeSlowDuration}}.",
    passiveName: "Guardian's Barrier (Warrior Damage Mitigation Aura)",
    passiveTemplate:
      "An eligible unshielded Warrior emits Guardian's Barrier. Eligible unshielded same-team agents within {{auraRadius}}, including the Warrior, receive {{perEmitterDamageMitigationEffect}}; overlapping emitters multiply down to {{damageMitigationFloor}}.",
  }),
  3: definition({
    overview:
      "The Hunter is a ranged damage dealer and disabler. Its basic applies a slow, and Freezing Trap applies raw damage and the longest canonical configured stun.",
    tacticalGuideRows: [
      tacticalRow("Role", "Disabler · Crowd Controller"),
      tacticalRow("Primary Strength", "Longest canonical configured stun duration."),
      tacticalRow(
        "Primary Weakness",
        "Lowest positive basic raw damage among canonical damage-dealing classes.",
      ),
    ],
    ultimateName: "Freezing Trap",
    ultimateTemplate:
      "Freezing Trap applies {{ultimateRawDamage}} raw damage to an enemy target before source and recipient damage modifiers and applies a stun for {{trapStunDuration}}. Accepted positive raw damage ends an existing trap before any same-transition reapplication.",
    passiveName: "Serrated Arrows",
    passiveTemplate:
      "Every accepted Hunter basic applies Serrated Arrows for {{hunterBasicSlowDuration}}, imposing {{hunterBasicMovementEffect}}. Later accepted Hunter basics refresh the remaining duration.",
  }),
  4: definition({
    overview:
      "The Rogue has the highest canonical base movement speed. Crippling Poison applies raw damage, stun, slow, and anti-heal.",
    tacticalGuideRows: [
      tacticalRow("Role", "Fast Damage · Stun · Slow · Anti-Heal"),
      tacticalRow("Primary Strength", "Highest canonical base movement speed."),
      tacticalRow(
        "Primary Weakness",
        "No raw healing and lower maximum health than the Warrior.",
      ),
    ],
    ultimateName: "Crippling Poison",
    ultimateTemplate:
      "Crippling Poison applies {{ultimateRawDamage}} raw damage to an enemy target before source and recipient damage modifiers, a stun for {{poisonStunDuration}}, {{poisonSlowEffect}} for {{poisonSlowDuration}}, and {{poisonAntiHealEffect}} to incoming healing and out-of-combat regeneration for {{poisonAntiHealDuration}}.",
    passiveName: "Phantom's Quickness",
    passiveTemplate:
      "This Rogue's {{baseMovementSpeed}} is the highest in the certified profile. After {{outOfCombatDelay}} without combat participation, it becomes eligible for the displayed Out-of-Combat Regeneration on each transition tick.",
  }),
  5: definition({
    overview:
      "The Priest is the only canonical class with positive raw healing and cannot deal raw damage. Its basic and Ultimate target same-team agents.",
    tacticalGuideRows: [
      tacticalRow("Role", "Healer · Medic"),
      tacticalRow(
        "Primary Strength",
        "Only canonical class with positive raw healing.",
      ),
      tacticalRow("Primary Weakness", "Cannot deal raw damage."),
    ],
    ultimateName: "Holy Word: Salvation",
    ultimateTemplate:
      "Holy Word: Salvation applies {{ultimateRawHealing}} raw healing to a same-team target before recipient healing modifiers and maximum-health clamping.",
    passiveName: "Blessing of Freedom",
    passiveTemplate:
      "Every accepted Priest basic applies Blessing of Freedom to its same-team target, including the Priest where same-team targeting permits it, for {{freedomDuration}}. Freedom limits how far slow effects can reduce ordinary movement, using {{freedomMovementFloor}}; it does not override stun.",
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
