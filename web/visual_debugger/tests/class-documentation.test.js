import assert from "node:assert/strict";
import test from "node:test";

import {
  requiredClassDocumentationValueNamesV1,
  resolveClassDocumentationV1,
} from "../src/class-documentation.js";

const PROFILE = Object.freeze({
  availability_kind: "available",
  profile_id: "marl_battlegrounds.class_documentation.canonical_v1",
});

/** @param {unknown} value */
function assertRecursivelyFrozen(value) {
  if (value === null || typeof value !== "object") return;
  assert.equal(Object.isFrozen(value), true);
  for (const child of Object.values(value)) assertRecursivelyFrozen(child);
}

/** @param {unknown} value @returns {string[]} */
function allStrings(value) {
  if (typeof value === "string") return [value];
  if (value === null || typeof value !== "object") return [];
  return Object.values(value).flatMap(allStrings);
}

const cases = [
  {
    classId: 1,
    values: {
      burstDuration: "<Burst Duration>",
      burstDamageEffect: "<Burst Damage Effect>",
      auraRadius: "<Aura Radius>",
      perEmitterDamageAmplificationEffect: "<Per-Emitter Damage Amplification Effect>",
      damageAmplificationCeiling: "<Damage Amplification Ceiling>",
    },
    expected: {
      overview:
        "The Mage is a fragile backline damage dealer that creates explosive ranged-damage windows with Burst and relies on allied protection to operate.",
      tacticalGuideRows: [
        { label: "Role", value: "Explosive Ranged Damage · Team-fight MVP" },
        {
          label: "Primary Strength",
          value: "Highest basic raw damage among canonical damage-dealing classes.",
        },
        {
          label: "Primary Weakness",
          value: "Lowest canonical maximum health; glass cannon.",
        },
        { label: "Counters", value: "Priest." },
        { label: "Countered By", value: "Rogue." },
      ],
      ultimate: {
        name: "Burst",
        description:
          "For <Burst Duration>, Burst multiplies this Mage's outgoing damage by <Burst Damage Effect>, beginning with the successor decision.",
      },
      passive: {
        name: "Sorcerer's Empowerment (Mage Damage Amplification Aura)",
        description:
          "An eligible unshielded Mage emits Sorcerer's Empowerment. Eligible unshielded same-team agents within <Aura Radius>, including the Mage, receive <Per-Emitter Damage Amplification Effect>; overlapping emitters multiply up to <Damage Amplification Ceiling>.",
      },
    },
  },
  {
    classId: 2,
    values: {
      ultimateRawDamage: "<Ultimate Raw Damage>",
      chargeStunDuration: "<Charge Stun Duration>",
      chargeSlowEffect: "<Charge Slow Effect>",
      chargeSlowDuration: "<Charge Slow Duration>",
      auraRadius: "<Aura Radius>",
      perEmitterDamageMitigationEffect: "<Per-Emitter Damage Mitigation Effect>",
      damageMitigationFloor: "<Damage Mitigation Floor>",
    },
    expected: {
      overview:
        "The Warrior is a durable frontline unit whose Charge can initiate a team fight or come to the aid of an ally, but it relies on allied follow-up to secure kills.",
      tacticalGuideRows: [
        { label: "Role", value: "Guardian · Team-fight Initiator · Tank" },
        { label: "Primary Strength", value: "Highest canonical maximum health." },
        {
          label: "Primary Weakness",
          value:
            "Second-lowest positive basic raw damage among canonical damage-dealing classes.",
        },
        { label: "Counters", value: "Rogue." },
        { label: "Countered By", value: "Hunter." },
      ],
      ultimate: {
        name: "Charge",
        description:
          "Charge moves the Warrior toward an enemy target during the Charge phase before ordinary movement. The accepted ultimate also applies <Ultimate Raw Damage> raw damage before source and recipient damage modifiers, <Charge Stun Duration> of stun, and <Charge Slow Effect> for <Charge Slow Duration>.",
      },
      passive: {
        name: "Guardian's Barrier (Warrior Damage Mitigation Aura)",
        description:
          "An eligible unshielded Warrior emits Guardian's Barrier. Eligible unshielded same-team agents within <Aura Radius>, including the Warrior, receive <Per-Emitter Damage Mitigation Effect>; overlapping emitters multiply down to <Damage Mitigation Floor>.",
      },
    },
  },
  {
    classId: 3,
    values: {
      ultimateRawDamage: "<Ultimate Raw Damage>",
      trapStunDuration: "<Trap Stun Duration>",
      hunterBasicSlowDuration: "<Hunter Basic Slow Duration>",
      hunterBasicMovementEffect: "<Hunter Basic Movement Effect>",
    },
    expected: {
      overview:
        "The Hunter is a backline ranged disabler whose basic range, observation radius, and control effects create plays for allied follow-up.",
      tacticalGuideRows: [
        { label: "Role", value: "Disabler · Crowd Controller" },
        {
          label: "Primary Strength",
          value: "Longest canonical configured stun duration.",
        },
        {
          label: "Primary Weakness",
          value:
            "Lowest positive basic raw damage among canonical damage-dealing classes.",
        },
        { label: "Counters", value: "Warrior." },
        { label: "Countered By", value: "Priest." },
      ],
      ultimate: {
        name: "Freezing Trap",
        description:
          "Freezing Trap applies <Ultimate Raw Damage> raw damage to an enemy target before source and recipient damage modifiers and applies a stun for <Trap Stun Duration>. Accepted positive raw damage ends an existing trap before any same-transition reapplication.",
      },
      passive: {
        name: "Serrated Arrows",
        description:
          "Every accepted Hunter basic applies Serrated Arrows for <Hunter Basic Slow Duration>, imposing <Hunter Basic Movement Effect>. Later accepted Hunter basics refresh the remaining duration.",
      },
    },
  },
  {
    classId: 4,
    values: {
      ultimateRawDamage: "<Ultimate Raw Damage>",
      poisonStunDuration: "<Poison Stun Duration>",
      poisonSlowEffect: "<Poison Slow Effect>",
      poisonSlowDuration: "<Poison Slow Duration>",
      poisonAntiHealEffect: "<Poison Anti-Heal Effect>",
      poisonAntiHealDuration: "<Poison Anti-Heal Duration>",
      baseMovementSpeed: "<Base Movement Speed>",
      outOfCombatDelay: "<Out-of-combat Delay>",
    },
    expected: {
      overview:
        "The Rogue is a fast ambusher whose superior movement speed supports flanks against high-value backline targets, but poor team-relative positioning leaves it vulnerable.",
      tacticalGuideRows: [
        { label: "Role", value: "Ambusher · Flanker · Assassin" },
        {
          label: "Primary Strength",
          value: "Highest canonical base movement speed.",
        },
        {
          label: "Primary Weakness",
          value: "Relies on favorable team positioning for ambushes and flanks.",
        },
        { label: "Counters", value: "Mage, Priest." },
        { label: "Countered By", value: "Warrior." },
      ],
      ultimate: {
        name: "Crippling Poison",
        description:
          "Crippling Poison applies <Ultimate Raw Damage> raw damage to an enemy target before source and recipient damage modifiers, a stun for <Poison Stun Duration>, <Poison Slow Effect> for <Poison Slow Duration>, and <Poison Anti-Heal Effect> to incoming healing and out-of-combat regeneration for <Poison Anti-Heal Duration>.",
      },
      passive: {
        name: "Phantom's Quickness",
        description:
          "This Rogue's <Base Movement Speed> is the highest in the certified profile. After <Out-of-combat Delay> without combat participation, it becomes eligible for the displayed Out-of-combat Regeneration on each transition tick.",
      },
    },
  },
  {
    classId: 5,
    values: {
      ultimateRawHealing: "<Ultimate Raw Healing>",
      freedomDuration: "<Freedom Duration>",
      freedomMovementFloor: "<Freedom Movement Floor>",
    },
    expected: {
      overview:
        "The Priest is a backline support unit focused on keeping allies alive and therefore depends on teammates for protection and damage.",
      tacticalGuideRows: [
        { label: "Role", value: "Healer · Medic" },
        {
          label: "Primary Strength",
          value: "Only canonical class with positive raw healing.",
        },
        { label: "Primary Weakness", value: "Cannot deal raw damage." },
        { label: "Counters", value: "Hunter." },
        { label: "Countered By", value: "Rogue." },
      ],
      ultimate: {
        name: "Holy Word: Salvation",
        description:
          "Holy Word: Salvation applies <Ultimate Raw Healing> raw healing to a same-team target before recipient healing modifiers and maximum-health clamping.",
      },
      passive: {
        name: "Blessing of Freedom",
        description:
          "Every accepted Priest basic applies Blessing of Freedom to its same-team target, including the Priest where same-team targeting permits it, for <Freedom Duration>. Freedom limits how far slow effects can reduce ordinary movement, using <Freedom Movement Floor>; it does not override stun.",
      },
    },
  },
];

test("the certified profile resolves the exact five numeric-free class guides", () => {
  for (const { classId, values, expected } of cases) {
    const requiredNames = requiredClassDocumentationValueNamesV1(PROFILE, classId);
    assert.deepEqual(requiredNames, Object.keys(values));
    assertRecursivelyFrozen(requiredNames);
    const before = structuredClone(values);
    const result = resolveClassDocumentationV1(PROFILE, classId, values);
    assert.deepEqual(result, expected);
    assert.deepEqual(values, before);
    assertRecursivelyFrozen(result);
    assert.equal(
      allStrings(result).some((value) => /\d/u.test(value)),
      false,
    );
    assert.equal(
      allStrings(result).some((value) => /\{\{/u.test(value)),
      false,
    );
  }
});

test("profile and class selection fail closed without name, row, or slot inference", () => {
  const values = cases[0].values;
  const invalidProfiles = [
    null,
    undefined,
    PROFILE.profile_id,
    { availability_kind: "unavailable" },
    { availability_kind: "available", profile_id: "future_profile" },
    { ...PROFILE, extra: true },
    { profile_id: PROFILE.profile_id },
  ];
  for (const profile of invalidProfiles) {
    assert.equal(requiredClassDocumentationValueNamesV1(profile, 1), null);
    assert.equal(resolveClassDocumentationV1(profile, 1, values), null);
  }
  for (const classId of [0, 6, "1", "Mage", true, 1.5, { class_id: 1 }, [1]]) {
    assert.equal(requiredClassDocumentationValueNamesV1(PROFILE, classId), null);
    assert.equal(resolveClassDocumentationV1(PROFILE, classId, values), null);
  }

  let reads = 0;
  const accessorProfile = { availability_kind: "available" };
  Object.defineProperty(accessorProfile, "profile_id", {
    enumerable: true,
    get() {
      reads += 1;
      return PROFILE.profile_id;
    },
  });
  assert.equal(requiredClassDocumentationValueNamesV1(accessorProfile, 1), null);
  assert.equal(resolveClassDocumentationV1(accessorProfile, 1, values), null);
  assert.equal(reads, 0);
});

test("interpolation requires the exact named preformatted value record", () => {
  const valid = cases[4].values;
  const missing = { ...valid };
  delete missing.freedomDuration;
  const accessor = { ...valid };
  let reads = 0;
  Object.defineProperty(accessor, "freedomDuration", {
    enumerable: true,
    get() {
      reads += 1;
      return valid.freedomDuration;
    },
  });
  const withSymbol = { ...valid };
  Object.defineProperty(withSymbol, Symbol("future"), {
    enumerable: true,
    value: "future",
  });
  for (const values of [
    null,
    [],
    missing,
    { ...valid, futureValue: "future" },
    { ...valid, freedomDuration: 3 },
    { ...valid, freedomDuration: " <Freedom Duration>" },
    { ...valid, freedomDuration: "<Freedom Duration>\n" },
    { ...valid, freedomDuration: "Unavailable" },
    { ...valid, freedomDuration: "Effect unavailable" },
    accessor,
    withSymbol,
  ]) {
    assert.equal(resolveClassDocumentationV1(PROFILE, 5, values), null);
  }
  assert.equal(reads, 0);
});
