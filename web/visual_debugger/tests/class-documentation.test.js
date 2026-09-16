/**
 * @file Check the five class guides, required profile identity and exact named value
 * substitution.
 */
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
        "The Mage has the highest Basic damage per tick in the game, which is further enhanced by its own passive and Ultimate (Burst) damage amplification effects. This skill set makes the Mage the game’s ranged “Artillery” class. Despite its high damage output, it is also the most vulnerable: the Mage has the lowest health in the game, making it a “Glass Cannon” that is reliant on strong team formations and allies for protection.",
      tacticalGuideRows: [
        {
          label: "Role",
          value: "Primary Damage Dealer · Burst Damage · Damage Amplification",
        },
        {
          label: "Primary Strength",
          value: "Highest Basic damage among damage-dealing classes.",
        },
        {
          label: "Primary Weakness",
          value: "Lowest maximum health; glass cannon.",
        },
      ],
      ultimate: {
        name: "Burst",
        description:
          "For <Burst Duration>, Burst applies <Burst Damage Effect> to this Mage's damage, starting on the next tick.",
      },
      passive: {
        name: "Sorcerer's Empowerment (Mage Damage Amplification Aura)",
        description:
          "While alive and without a spawn shield, the Mage emits Sorcerer's Empowerment. It strengthens the attacks of living teammates within <Aura Radius> who also have no spawn shield, including the Mage itself. Each aura provides <Per-Emitter Damage Amplification Effect>. Overlapping auras multiply their damage bonuses, up to <Damage Amplification Ceiling> times normal damage.",
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
        "The Warrior has the highest maximum health in the game—twice that of the next-highest classes—so it is the game’s melee “Tank”/“Guardian” class. The Warrior thrives on the front line where it can initiate engagements, provide passive damage mitigation to allies, and body-block or peel aggressors. The Warrior’s Ultimate (Charge) propels it toward an enemy, instantaneously applying considerable damage, a stun, and a slow. With relatively low follow-up damage, the Warrior is reliant on its allies to secure kills.",
      tacticalGuideRows: [
        {
          label: "Role",
          value: "Tank · Frontline Damage · Damage Mitigation",
        },
        { label: "Primary Strength", value: "Highest maximum health." },
        {
          label: "Primary Weakness",
          value: "Second-lowest Basic damage among damage-dealing classes.",
        },
      ],
      ultimate: {
        name: "Charge",
        description:
          "Charge moves the Warrior toward an enemy before normal movement. It deals <Ultimate Raw Damage> base damage, stuns for <Charge Stun Duration>, and applies <Charge Slow Effect> for <Charge Slow Duration>. Damage bonuses and reductions affect the damage dealt.",
      },
      passive: {
        name: "Guardian's Barrier (Warrior Damage Mitigation Aura)",
        description:
          "While alive and without a spawn shield, the Warrior emits Guardian's Barrier. It protects living teammates within <Aura Radius> who also have no spawn shield, including the Warrior itself. Each barrier reduces incoming damage by <Per-Emitter Damage Mitigation Effect>. When barriers overlap, each reduces the damage left after the others, down to <Damage Mitigation Floor> of the original incoming damage.",
      },
    },
  },
  {
    classId: 3,
    values: {
      ultimateRawDamage: "<Ultimate Raw Damage>",
      trapStunDuration: "<Trap Stun Duration>",
      hunterBasicMovementEffect: "<Hunter Basic Movement Effect>",
      hunterBasicSlowDuration: "<Hunter Basic Slow Duration>",
    },
    expected: {
      overview:
        "The Hunter is the game’s ranged “Crowd Controller” class. Its job is to use its superior Basic attack and observation range to scout out and slow or disable enemies to help its allies win engagements. The Hunter’s low damage makes it reliant on allies to secure kills, but its Ultimate (Freezing Trap) is the longest stun in the game and, if timed well, can change the tide of battle—but be careful not to break it!",
      tacticalGuideRows: [
        { label: "Role", value: "Disabler · Crowd Controller · Scout" },
        {
          label: "Primary Strength",
          value: "Longest stun duration.",
        },
        {
          label: "Primary Weakness",
          value: "Lowest Basic damage among damage-dealing classes.",
        },
      ],
      ultimate: {
        name: "Freezing Trap",
        description:
          "Freezing Trap deals <Ultimate Raw Damage> base damage to one enemy and stuns it for <Trap Stun Duration>. Damage bonuses and reductions affect the damage dealt. Damage breaks an existing trap.",
      },
      passive: {
        name: "Serrated Arrows",
        description:
          "Each successful Hunter Basic attack applies Serrated Arrows, causing <Hunter Basic Movement Effect> for <Hunter Basic Slow Duration>. Further Basic attacks refresh the slow.",
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
      outOfCombatDelay: "<Out-of-Combat Delay>",
    },
    expected: {
      overview:
        "Having the highest base movement speed in the game—alongside a high-damage Ultimate (Crippling Poison) that also stuns, slows, and applies an anti-heal effect—the Rogue is the melee “Assassin” class. Because it is a melee class and lacks a gap closer like the Warrior’s “Charge” Ultimate, it is heavily reliant on surrounding obstacles and advancing allies to execute flanking maneuvers without being counterattacked.",
      tacticalGuideRows: [
        { label: "Role", value: "Ambusher · Flanker · Assassin · Anti-Heal" },
        {
          label: "Primary Strength",
          value: "Highest base movement speed.",
        },
        {
          label: "Primary Weakness",
          value: "Short attack range with no gap closer.",
        },
      ],
      ultimate: {
        name: "Crippling Poison",
        description:
          "Crippling Poison deals <Ultimate Raw Damage> base damage to one enemy, stuns it for <Poison Stun Duration>, and applies <Poison Slow Effect> for <Poison Slow Duration>. It also applies <Poison Anti-Heal Effect> to healing and out-of-combat regeneration for <Poison Anti-Heal Duration>. Damage bonuses and reductions affect the damage dealt.",
      },
      passive: {
        name: "Phantom's Quickness",
        description:
          "Phantom's Quickness gives the Rogue a base movement speed of <Base Movement Speed>, the highest in the game. After <Out-of-Combat Delay> out of combat, it regenerates health each tick at the rate shown under Out-of-Combat Regeneration.",
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
        "The Priest is the game’s ranged “Healer” class. It cannot deal damage, so it relies on strong team formations and allies for protection. Its Ultimate (Holy Word: Salvation) delivers a massive amount of healing in a single tick, allowing it to save allies who would otherwise die in emergency situations.",
      tacticalGuideRows: [
        { label: "Role", value: "Healer · Medic" },
        {
          label: "Primary Strength",
          value: "Only class with healing abilities.",
        },
        { label: "Primary Weakness", value: "Cannot deal damage." },
      ],
      ultimate: {
        name: "Holy Word: Salvation",
        description:
          "Holy Word: Salvation heals one ally for <Ultimate Raw Healing>, before healing reductions and the maximum-health limit.",
      },
      passive: {
        name: "Blessing of Freedom",
        description:
          "Each successful use of the Priest's Basic healing ability applies Blessing of Freedom to its target for <Freedom Duration>. The Priest can target itself. Freedom prevents slows from reducing movement below <Freedom Movement Floor>, but it does not prevent stuns.",
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
