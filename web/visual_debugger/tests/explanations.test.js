import assert from "node:assert/strict";
import test from "node:test";

import {
  explainActivation,
  explainAgent,
  explainAura,
  explainCooldown,
  explainLegality,
  explainModifier,
  explainNetHealth,
  explainObstacle,
  explainOverflow,
  explainPendingRoute,
  explainPovAgent,
  explainPovOverflow,
  explainPovStatus,
  explainRange,
  explainSpawnShield,
  explainStatus,
  explainVisibility,
} from "../src/explanations.js";
import { createSemanticDescriptor, projectSemanticDescriptor } from "../src/tooltip.js";

const RECIPIENT = Object.freeze({
  global_slot: 8,
  public_agent_id: "recipient:<unsafe>&42",
  team_id: 2,
  class_id: 5,
});
const SOURCE_A = Object.freeze({
  global_slot: 1,
  public_agent_id: "alpha/9001",
  team_id: 1,
  class_id: 3,
});
const SOURCE_B = Object.freeze({
  global_slot: 7,
  public_agent_id: "beta.17",
  team_id: 2,
  class_id: 4,
});

/** @param {ReturnType<typeof createSemanticDescriptor>} descriptor @param {string} label */
function rowValue(descriptor, label) {
  const found = descriptor.rows.find((row) => row.label === label);
  assert.ok(found, `missing row ${label}`);
  return found.value;
}

/** @param {ReturnType<typeof createSemanticDescriptor>} descriptor @param {string} title */
function sectionRows(descriptor, title) {
  const found = descriptor.sections.find((section) => section.title === title);
  assert.ok(found, `missing section ${title}`);
  return found.rows;
}

/** @param {ReturnType<typeof createSemanticDescriptor>} descriptor */
function fullText(descriptor) {
  return JSON.stringify(projectSemanticDescriptor(descriptor, "full"));
}

test("all semantic descriptors and nested projections are recursively immutable", () => {
  const descriptor = explainAgent(
    {
      global_slot: 8,
      public_agent_id: "arbitrary-public-id",
      class_id: 1,
      team_id: 2,
      current_health: 87.654,
      max_health: 100,
      effective_movement_speed: 0.333,
      ultimate_cooldown_remaining: 3,
      steps_until_out_of_combat: 2,
      statuses: [],
      aura_modifiers: [],
    },
    { controlled: true },
    classMechanics(1, "Mage"),
  );
  assert.equal(Object.isFrozen(descriptor), true);
  assert.equal(Object.isFrozen(descriptor.rows), true);
  assert.equal(Object.isFrozen(descriptor.rows[0]), true);
  assert.equal(Object.isFrozen(descriptor.rows[0].metadata), true);
  assert.equal(Object.isFrozen(descriptor.sections), true);
  assert.equal(Object.isFrozen(descriptor.sections[0].rows), true);
  assert.equal(rowValue(descriptor, "Identity"), "Agent ID arbitrary-public-id");
  assert.equal(rowValue(descriptor, "Effective Speed"), "0.33");
  assert.doesNotMatch(fullText(descriptor), /id_8/u);
});

const STATUS_CASES = Object.freeze([
  [
    "warrior_charge_stun",
    "stun_warrior_charge",
    "Warrior (Ultimate: Charge) Stun",
    "A Warrior's concussive Charge incapacitates",
  ],
  [
    "hunter_trap_stun",
    "stun_hunter_trap",
    "Hunter (Ultimate: Freezing Trap) Stun",
    "A Hunter's Freezing Trap incapacitates",
  ],
  [
    "rogue_poison_stun",
    "stun_rogue_poison",
    "Rogue (Ultimate: Crippling Poison) Stun",
    "A Rogue's Crippling Poison incapacitates",
  ],
  [
    "warrior_charge_slow",
    "slow_warrior_charge",
    "Warrior (Ultimate: Charge) Slow",
    "A Warrior's concussive Charge slows",
  ],
  [
    "hunter_basic_slow",
    "slow_hunter_basic",
    "Hunter (Basic: Attack) Slow",
    "A Hunter's serrated arrows slow",
  ],
  [
    "rogue_poison_slow",
    "slow_rogue_poison",
    "Rogue (Ultimate: Crippling Poison) Slow",
    "A Rogue's Crippling Poison slows",
  ],
  [
    "rogue_poison_anti_heal",
    "anti_heal_rogue_poison",
    "Rogue (Ultimate: Crippling Poison) Anti-Heal",
    "A Rogue's Crippling Poison reduces",
  ],
  [
    "priest_freedom",
    "priest_freedom",
    "Priest (Basic: Heal) Freedom",
    "A Priest's healing uplifts",
  ],
  [
    "mage_burst",
    "mage_burst",
    "Mage (Ultimate: Burst) Damage Amplification",
    "Burst fills this Mage",
  ],
]);

test("all nine status channels use stable prose and exact normalized quantities", () => {
  for (const [statusId, tokenId, expectedTitle, expectedEffect] of STATUS_CASES) {
    const descriptor = explainStatus(
      {
        status_channel: STATUS_CASES.findIndex((row) => row[0] === statusId),
        status_id: statusId,
        token_id: tokenId,
        source_class_id: 3,
        source_class_name: "Hunter",
        source_action_component: "basic",
        remaining_duration: 123456789,
        magnitude_kind: tokenId.includes("stun") ? "none" : "movement_multiplier",
        magnitude: tokenId.includes("stun") ? null : 0.876543,
        breaks_on_positive_damage: tokenId.includes("stun"),
        direct_source_evidence: [],
      },
      RECIPIENT,
      [SOURCE_A, SOURCE_B],
    );
    assert.equal(descriptor.title, expectedTitle);
    assert.equal(descriptor.summary.includes(expectedEffect), true);
    assert.match(descriptor.summary, /while the status remains/u);
    assert.match(
      descriptor.summary,
      /positive-damage break rule|recorded positive damage/u,
    );
    assert.equal(rowValue(descriptor, "Duration"), "123456789 Ticks");
    assert.equal(
      descriptor.rows.some((row) => row.label === "Recipient"),
      false,
    );
    assert.equal(
      sectionRows(descriptor, "Direct Source")[0].value,
      "Source agent not recorded.",
    );
    assert.deepEqual(
      projectSemanticDescriptor(descriptor, "compact").sections[0].rows.map(
        (row) => row.value,
      ),
      ["Source agent not recorded."],
    );
  }
});

test("status sources require exact same-scene slot and public-ID joins", () => {
  const descriptor = explainStatus(
    researcherStatus({
      direct_source_evidence: [
        {
          source_global_slot: 1,
          source_public_agent_id: "alpha/9001",
          event_id: "event:first",
        },
        {
          source_global_slot: 1,
          source_public_agent_id: "alpha/9001",
          event_id: "event:repeat",
        },
        {
          source_global_slot: 7,
          source_public_agent_id: "beta.17",
          event_id: "event:second-source",
        },
        {
          source_global_slot: 1,
          source_public_agent_id: "wrong-id",
          event_id: "event:mismatch",
        },
        {
          source_global_slot: 4,
          source_public_agent_id: "not-in-roster",
          event_id: "event:unjoined",
        },
      ],
    }),
    RECIPIENT,
    [SOURCE_A, SOURCE_B],
  );
  assert.deepEqual(
    sectionRows(descriptor, "Direct Source").map((row) => row.value),
    ["Agent ID alpha/9001 · Team A · Hunter", "Agent ID beta.17 · Team B · Rogue"],
  );
  assert.deepEqual(
    projectSemanticDescriptor(descriptor, "compact").sections[0].rows.map(
      (row) => row.value,
    ),
    ["Agent ID alpha/9001 · Team A · Hunter", "Agent ID beta.17 · Team B · Rogue"],
  );
  const serialized = fullText(descriptor);
  assert.doesNotMatch(serialized, /event:first|event:repeat|event:second-source/u);
  assert.doesNotMatch(serialized, /wrong-id|not-in-roster/u);
});

test("status source display preserves upstream first-occurrence order", () => {
  const descriptor = explainStatus(
    researcherStatus({
      direct_source_evidence: [
        {
          source_global_slot: 7,
          source_public_agent_id: "beta.17",
          event_id: "canonical-a",
        },
        {
          source_global_slot: 1,
          source_public_agent_id: "alpha/9001",
          event_id: "canonical-b",
        },
      ],
    }),
    RECIPIENT,
    [SOURCE_A, SOURCE_B],
  );
  assert.deepEqual(
    sectionRows(descriptor, "Direct Source").map((row) => row.value),
    ["Agent ID beta.17 · Team B · Rogue", "Agent ID alpha/9001 · Team A · Hunter"],
  );
});

test("POV status builder accepts only reduced fields and never widens researcher data", () => {
  const malicious = {
    token_id: "slow_hunter_basic",
    duration: 3,
    status_feature_index: 16,
    source_class_id: 3,
    source_evidence: "effect_channel_only",
    magnitude: 0.001,
    magnitude_kind: "movement_multiplier",
    breaks_on_positive_damage: true,
    direct_source_evidence: [
      {
        source_global_slot: 1,
        source_public_agent_id: "secret-source",
        event_id: "secret-event",
      },
    ],
    source_action_component: "basic",
    label: "SECRET STATUS LABEL",
    short_label: "SECRET SHORT LABEL",
    accessible_name: "SECRET ACCESSIBLE LABEL",
  };
  const direct = explainPovStatus(malicious, RECIPIENT);
  const serialized = fullText(direct);
  assert.equal(rowValue(direct, "Source"), "Source agent identity is not disclosed.");
  assert.match(direct.summary, /Source agent identity is not disclosed\./u);
  assert.doesNotMatch(
    serialized,
    /0\.001|secret-source|secret-event|secret status|secret short|secret accessible|break|magnitude|source action|source class/iu,
  );
  assert.equal(
    projectSemanticDescriptor(direct, "compact").rows.some(
      (row) => row.label === "Recipient",
    ),
    false,
  );
});

test("researcher status magnitudes render exact wire-authored percentages by kind", () => {
  /** @type {Array<[string, number, string, string]>} */
  const cases = [
    ["movement_multiplier", 0.731, "Movement Effect", "26.9% slower (×0.73)"],
    [
      "healing_multiplier",
      0.623,
      "Healing Effect",
      "37.7% less healing received (×0.62)",
    ],
    ["damage_multiplier", 1.417, "Damage Effect", "41.7% more damage dealt (×1.42)"],
    ["movement_floor", 0.843, "Movement Floor", "84.3% of base movement speed (×0.84)"],
  ];
  for (const [magnitudeKind, magnitude, expectedLabel, expectedValue] of cases) {
    const descriptor = explainStatus(
      researcherStatus({ magnitude_kind: magnitudeKind, magnitude }),
      RECIPIENT,
      [],
    );
    assert.equal(rowValue(descriptor, expectedLabel), expectedValue);
    assert.equal(descriptor.summary.includes(expectedValue), true);
    assert.match(
      projectSemanticDescriptor(descriptor, "compact")
        .rows.map((row) => row.value)
        .join(" "),
      /%/u,
    );
  }

  const stun = explainStatus(
    researcherStatus({
      token_id: "stun_hunter_trap",
      magnitude_kind: "none",
      magnitude: null,
    }),
    RECIPIENT,
    [],
  );
  assert.equal(
    stun.rows.some((row) => row.label.includes("Magnitude")),
    false,
  );
  assert.equal(
    stun.rows.some((row) => row.value.includes("%")),
    false,
  );
});

test("aggregate aura modifier is noninterfering with emitter-shaped extras", () => {
  const first = explainModifier(
    {
      aura_id: "mage_damage_amplification",
      token_id: "mage_amplification",
      multiplier: 1.23456,
      emitter_global_slots: [1, 7],
      nearest_emitter: { public_agent_id: "alpha/9001", position: [0, 0] },
    },
    RECIPIENT,
  );
  const second = explainModifier(
    {
      aura_id: "mage_damage_amplification",
      token_id: "mage_amplification",
      multiplier: 1.23456,
      emitter_global_slots: [99],
      nearest_emitter: { public_agent_id: "different", position: [999, -42] },
    },
    RECIPIENT,
  );
  assert.equal(JSON.stringify(first), JSON.stringify(second));
  assert.equal(first.title, "Sorcerer’s Empowerment");
  assert.equal(first.accent, "mage");
  assert.equal(rowValue(first, "Aggregate Multiplier"), "×1.23");
  assert.equal(rowValue(first, "Recipient Effect"), "23.46% more damage dealt");
  assert.equal(
    first.summary,
    "This recipient has 23.46% more damage dealt from the exact aggregate aura multiplier.",
  );
  assert.match(fullText(first), /emitter identity is not recorded/u);
});

test("the two aggregate recipient aura cards use their locked qualitative titles", () => {
  assert.equal(
    explainModifier(
      {
        aura_id: "mage_damage_amplification",
        token_id: "mage_amplification",
        multiplier: 1.2,
      },
      RECIPIENT,
    ).title,
    "Sorcerer’s Empowerment",
  );
  assert.equal(
    explainModifier(
      {
        aura_id: "warrior_damage_mitigation",
        token_id: "warrior_mitigation",
        multiplier: 0.8,
      },
      RECIPIENT,
    ).title,
    "Guardian’s Barrier",
  );
  assert.equal(
    rowValue(
      explainModifier(
        {
          aura_id: "warrior_damage_mitigation",
          token_id: "warrior_mitigation",
          multiplier: 0.763,
        },
        RECIPIENT,
      ),
      "Recipient Effect",
    ),
    "23.7% less damage received",
  );
});

test("agent presentation suppresses only exact neutral aura modifiers", () => {
  const descriptor = explainAgent(
    {
      ...RECIPIENT,
      current_health: 100,
      max_health: 100,
      effective_movement_speed: 1,
      ultimate_cooldown_remaining: 0,
      steps_until_out_of_combat: 0,
      statuses: [],
      aura_modifiers: [
        {
          aura_id: "mage_damage_amplification",
          token_id: "mage_amplification",
          multiplier: 1,
        },
        {
          aura_id: "warrior_damage_mitigation",
          token_id: "warrior_mitigation",
          multiplier: 0.999999,
        },
      ],
    },
    {},
    classMechanics(5, "Priest"),
  );
  assert.equal(
    sectionRows(descriptor, "Current State").find(
      (row) => row.label === "Aggregate Aura Modifiers",
    )?.value,
    "1",
  );
  const effectRows = sectionRows(descriptor, "Current Aura Modifier Details");
  assert.equal(effectRows.length, 1);
  assert.match(effectRows[0].label, /Guardian’s Barrier/u);
  assert.doesNotMatch(fullText(descriptor), /Sorcerer’s Empowerment/u);
});

test("spawn shield explanation states invulnerability and exact remaining ticks", () => {
  const descriptor = explainSpawnShield({
    public_agent_id: "opaque-shielded",
    spawn_shield_remaining: 3,
  });
  assert.equal(descriptor.kind, "status");
  assert.equal(descriptor.title, "Spawn Shield");
  assert.equal(rowValue(descriptor, "Protection"), "Invulnerable");
  assert.equal(rowValue(descriptor, "Remaining"), "3 Ticks");
  assert.match(descriptor.summary, /invulnerable/u);
  assert.match(descriptor.summary, /3 Ticks/u);
});

test("five cooldown cards use exact class names, ultimate names, ticks, and public IDs", () => {
  /** @type {Array<[number, string, string]>} */
  const cases = [
    [1, "Mage", "Burst"],
    [2, "Warrior", "Charge"],
    [3, "Hunter", "Freezing Trap"],
    [4, "Rogue", "Crippling Poison"],
    [5, "Priest", "Holy Word: Salvation"],
  ];
  for (const [classId, className, ultimateName] of cases) {
    const owner = {
      global_slot: classId,
      public_agent_id: `opaque-${classId * 17}`,
      class_id: classId,
      ultimate_cooldown_remaining: classId - 1,
    };
    const descriptor = explainCooldown(
      owner,
      owner,
      classMechanics(classId, className),
    );
    assert.equal(descriptor.title, `${className} Ultimate: ${ultimateName} Cooldown`);
    assert.equal(rowValue(descriptor, "Source"), `Agent ID opaque-${classId * 17}`);
    assert.equal(
      rowValue(descriptor, "Cooldown Remaining"),
      `${classId - 1} ${classId === 2 ? "Tick" : "Ticks"}`,
    );
  }
});

test("three range cards explain purpose, exact radius, and joined owner identity", () => {
  for (const [kind, summaryFragment] of [
    ["observation", "observation range"],
    ["basic", "Basic interaction"],
    ["ultimate", "Ultimate interaction"],
  ]) {
    const descriptor = explainRange(
      { global_slot: 8, kind, radius: Math.PI },
      RECIPIENT,
      classMechanics(5, "Priest"),
    );
    assert.equal(rowValue(descriptor, "Radius"), "3.14");
    assert.equal(rowValue(descriptor, "Owner ID"), "Agent ID recipient:<unsafe>&42");
    assert.match(descriptor.summary, new RegExp(summaryFragment, "u"));
  }

  const mismatchedOwner = explainRange(
    { global_slot: 999, kind: "basic", radius: 2.5 },
    RECIPIENT,
    classMechanics(5, "Priest"),
  );
  assert.equal(rowValue(mismatchedOwner, "Owner ID"), "Agent ID unavailable");
  assert.equal(rowValue(mismatchedOwner, "Class"), "Unknown");
  assert.doesNotMatch(fullText(mismatchedOwner), /recipient:<unsafe>&42|Priest/u);
});

test("wall and pillar cards retain exact IDs, centers, and shape dimensions", () => {
  const wall = explainObstacle({
    obstacle_id: "wall:<north>&1",
    kind: "wall",
    center: [1.234, 9.876],
    width: 4.567,
    height: 0.25,
    theta: 1.5,
  });
  assert.deepEqual(
    wall.rows.map(({ label, value }) => [label, value]),
    [
      ["Obstacle ID", "wall:<north>&1"],
      ["Width", "4.57"],
      ["Height", "0.25"],
      ["Rotation", "1.5"],
      ["Center", "(1.23, 9.88)"],
    ],
  );
  assert.equal(wall.title, "Wall");
  const pillar = explainObstacle({
    obstacle_id: "pillar-900",
    kind: "pillar",
    center: [3, 4],
    radius: 0.75,
  });
  assert.equal(rowValue(pillar, "Radius"), "0.75");
  assert.equal(pillar.title, "Pillar");
  assert.equal(rowValue(pillar, "Center"), "(3, 4)");
  assert.equal(
    pillar.rows.some((row) => row.label === "Width"),
    false,
  );
});

test("aura fields expose exact catalog capability without claiming realized effect", () => {
  /** @type {Array<[Record<string, any>, string]>} */
  const cases = [
    [
      {
        aura_id: "mage_damage_amplification",
        token_id: "mage_amplification",
        source_global_slot: 1,
        source_public_agent_id: "alpha/9001",
        source_class_id: 1,
        source_class_name: "Mage",
        radius: 4.567,
        per_emitter_multiplier: 1.2,
        beneficiary_relation: "same_team",
        stacking_rule: "multiply_then_clamp",
        clamp_kind: "ceiling",
        clamp_value: 2,
        center: [1, 2],
      },
      "Sorcerer’s Empowerment",
    ],
    [
      {
        aura_id: "warrior_damage_mitigation",
        token_id: "warrior_mitigation",
        source_global_slot: 7,
        source_public_agent_id: "beta.17",
        source_class_id: 2,
        source_class_name: "Warrior",
        radius: 3,
        per_emitter_multiplier: 0.8,
        beneficiary_relation: "same_team",
        stacking_rule: "multiply_then_clamp",
        clamp_kind: "floor",
        clamp_value: 0.2,
        center: [4, 5],
      },
      "Guardian’s Barrier",
    ],
  ];
  for (const [field, expectedTitle] of cases) {
    const source = field.source_global_slot === 1 ? SOURCE_A : SOURCE_B;
    const descriptor = explainAura(field, {
      ...source,
      class_id: field.source_class_id,
    });
    assert.equal(descriptor.title, expectedTitle);
    assert.equal(
      descriptor.accent,
      field.aura_id === "mage_damage_amplification" ? "mage" : "warrior",
    );
    assert.equal(
      rowValue(descriptor, "Source ID"),
      `Agent ID ${field.source_public_agent_id}`,
    );
    assert.equal(rowValue(descriptor, "Radius"), format(field.radius));
    assert.equal(
      rowValue(descriptor, "Catalog Multiplier"),
      `×${format(field.per_emitter_multiplier)}`,
    );
    assert.equal(
      rowValue(descriptor, "Catalog Effect"),
      field.aura_id === "mage_damage_amplification"
        ? "20% more damage dealt per recorded emitter"
        : "20% less damage received per recorded emitter",
    );
    assert.match(descriptor.summary, /catalog-declared.*may receive/iu);
    assert.match(descriptor.summary, /exact aggregate modifier.*realized effect/iu);
    assert.doesNotMatch(descriptor.summary, /allies inside this field have/iu);
  }
  const mismatched = explainAura(
    {
      aura_id: "mage_damage_amplification",
      token_id: "mage_amplification",
      source_global_slot: 1,
      source_public_agent_id: "alpha/9001",
      radius: 3,
      per_emitter_multiplier: 1.2,
    },
    { ...SOURCE_A, public_agent_id: "wrong-agent", class_id: 1 },
  );
  assert.equal(rowValue(mismatched, "Source ID"), "Agent ID unavailable");
  assert.equal(rowValue(mismatched, "Source Class"), "Unavailable");
  assert.doesNotMatch(fullText(mismatched), /wrong-agent/u);

  for (const malformedField of [
    {
      token_id: "mage_amplification",
      source_public_agent_id: "alpha/9001",
      radius: 3,
      per_emitter_multiplier: 1.2,
    },
    {
      token_id: "mage_amplification",
      source_global_slot: "1",
      source_public_agent_id: "alpha/9001",
      radius: 3,
      per_emitter_multiplier: 1.2,
    },
    {
      token_id: "mage_amplification",
      source_global_slot: 1,
      radius: 3,
      per_emitter_multiplier: 1.2,
    },
  ]) {
    const malformed = explainAura(malformedField, { ...SOURCE_A, class_id: 1 });
    assert.equal(rowValue(malformed, "Source ID"), "Agent ID unavailable");
    assert.equal(rowValue(malformed, "Source Class"), "Unavailable");
    assert.doesNotMatch(fullText(malformed), /alpha\/9001/u);
  }
});

test("agent full cards show class copy and only positive raw catalog outputs", () => {
  const agent = {
    ...RECIPIENT,
    current_health: 55,
    max_health: 101,
    effective_movement_speed: 1.25,
    ultimate_cooldown_remaining: 0,
    steps_until_out_of_combat: 0,
    life_state: "alive",
    statuses: [{}],
    aura_modifiers: [{}, {}],
  };
  const mechanics = {
    ...classMechanics(5, "Priest"),
    basic_raw_damage: 0,
    basic_raw_healing: 11,
    ultimate_raw_damage: 0,
    ultimate_raw_healing: 29,
  };
  const descriptor = explainAgent(agent, {}, mechanics);
  const mechanicsRows = sectionRows(descriptor, "Exact Class Mechanics");
  assert.equal(rowValue(descriptor, "Ultimate Name"), "Holy Word: Salvation");
  assert.equal(rowValue(descriptor, "Ultimate Status"), "Ready");
  assert.equal(
    mechanicsRows.find((row) => row.label === "Ultimate Name")?.value,
    "Holy Word: Salvation",
  );
  assert.equal(
    mechanicsRows.some((row) => row.label === "Basic Raw Damage"),
    false,
  );
  assert.equal(
    mechanicsRows.some((row) => row.label === "Ultimate Raw Damage"),
    false,
  );
  assert.equal(
    mechanicsRows.find((row) => row.label === "Basic Raw Healing")?.value,
    "11",
  );
  assert.match(
    fullText(descriptor),
    /Role|Strengths|Limitations|Teamwork|Counterplay/u,
  );
  assert.match(fullText(descriptor), /Out of combat/u);

  const coolingDown = explainAgent(
    { ...agent, ultimate_cooldown_remaining: 3 },
    {},
    mechanics,
  );
  assert.equal(rowValue(coolingDown, "Ultimate Name"), "Holy Word: Salvation");
  assert.equal(rowValue(coolingDown, "Ultimate Status"), "On cooldown (3 Ticks)");
});

test("agent full cards retain exact authored and realized status and aura facts", () => {
  const mechanics = {
    ...classMechanics(1, "Mage"),
    status_mechanics: [
      {
        status_channel: 7,
        status_id: "mage_burst_damage_amplification",
        source_action_component: "ultimate",
        duration_steps: 7,
        magnitude_kind: "damage_multiplier",
        magnitude: 1.73,
        breaks_on_positive_damage: false,
      },
    ],
    aura_mechanics: [
      {
        aura_id: "mage_damage_amplification",
        radius: 4.75,
        per_emitter_multiplier: 1.17,
        stacking_rule: "multiply_then_clamp",
        clamp_kind: "ceiling",
        clamp_value: 1.5,
      },
    ],
  };
  const realized = researcherStatus({
    status_channel: 7,
    status_id: "mage_burst_damage_amplification",
    token_id: "mage_burst",
    source_class_id: 1,
    source_class_name: "Mage",
    source_action_component: "ultimate",
    remaining_duration: 3,
    magnitude_kind: "damage_multiplier",
    magnitude: 1.5,
    direct_source_evidence: [
      {
        source_global_slot: SOURCE_A.global_slot,
        source_public_agent_id: SOURCE_A.public_agent_id,
        event_id: "technical-event",
      },
    ],
  });
  const descriptor = explainAgent(
    {
      ...RECIPIENT,
      current_health: 50,
      max_health: 100,
      effective_movement_speed: 1,
      ultimate_cooldown_remaining: 0,
      statuses: [realized],
      aura_modifiers: [{ aura_id: "mage_damage_amplification", multiplier: 1.17 }],
    },
    {},
    mechanics,
    [SOURCE_A],
  );
  assert.match(
    sectionRows(descriptor, "Authored Status Mechanics")[0].value,
    /7 Ticks.*73% more damage dealt/u,
  );
  assert.match(
    sectionRows(descriptor, "Authored Passive Mechanics")[0].value,
    /Radius 4\.75.*17% more damage dealt/u,
  );
  assert.match(
    sectionRows(descriptor, "Current Status Details")[0].value,
    /3 Ticks.*Agent ID alpha\/9001/u,
  );
  assert.match(
    sectionRows(descriptor, "Current Aura Modifier Details")[0].value,
    /17% more damage dealt/u,
  );
  assert.doesNotMatch(fullText(descriptor), /technical-event/u);
});

test("POV agent builder is byte-noninterfering with researcher-only extras", () => {
  const authorized = {
    ...RECIPIENT,
    current_health: 50,
    max_health: 100,
    effective_movement_speed: 1.25,
    ultimate_cooldown_remaining: 2,
    steps_until_out_of_combat: 2,
    alive: false,
    statuses: [
      {
        token_id: "slow_hunter_basic",
        duration: 3,
        status_feature_index: 16,
        source_class_id: 3,
        source_evidence: "effect_channel_only",
      },
    ],
  };
  const direct = explainPovAgent(authorized, { controlled: true });
  const descriptor = explainAgent(
    {
      ...authorized,
      aura_modifiers: [{ multiplier: 9 }],
      effective_speed: 99,
      ultimate_cooldown: 99,
      direct_source_evidence: [{ source_public_agent_id: "secret" }],
      life_state: "SECRET_RESEARCHER_STATE",
    },
    { audience: "agent_pov", controlled: true },
    { ...classMechanics(5, "Priest"), secret_formula: "do-not-display" },
  );
  assert.equal(JSON.stringify(descriptor), JSON.stringify(direct));
  assert.equal(
    descriptor.sections.some((section) => section.title === "Exact Class Mechanics"),
    false,
  );
  assert.doesNotMatch(fullText(descriptor), /secret_formula|Raw Damage|Raw Healing/u);
  assert.equal(rowValue(descriptor, "Effective Speed"), "1.25");
  assert.equal(rowValue(descriptor, "Ultimate Status"), "On cooldown (2 Ticks)");
  assert.equal(
    sectionRows(descriptor, "Current State").find((row) => row.label === "Life State")
      ?.value,
    "Corpse",
  );
  assert.match(
    sectionRows(descriptor, "Current State").find((row) => row.label === "Combat State")
      ?.value ?? "",
    /2 Ticks/u,
  );
  assert.equal(
    sectionRows(descriptor, "Current State").find(
      (row) => row.label === "Aggregate Aura Modifiers",
    )?.value,
    "Unavailable",
  );
  assert.match(
    sectionRows(descriptor, "Current Status Details")[0].value,
    /Duration: 3 Ticks.*Source agent identity is not disclosed\./u,
  );
  assert.doesNotMatch(fullText(descriptor), /secret|×9|99/u);
});

test("POV status overflow is byte-noninterfering and discloses no source identity", () => {
  const authorized = [
    {
      token_id: "slow_hunter_basic",
      duration: 3,
      status_feature_index: 16,
      source_class_id: 3,
      source_evidence: "effect_channel_only",
    },
    {
      token_id: "stun_rogue_poison",
      duration: 5,
      status_feature_index: 23,
      source_class_id: 4,
      source_evidence: "effect_channel_only",
    },
  ];
  const baseline = explainPovOverflow(authorized, RECIPIENT);
  const injected = explainPovOverflow(
    authorized.map((status) => ({
      ...status,
      magnitude: 0.001,
      breaks_on_positive_damage: true,
      source_public_agent_id: "secret-overflow-source",
      direct_source_evidence: [{ event_id: "secret-overflow-event" }],
      accessible_name: "secret-overflow-accessible-name",
    })),
    { ...RECIPIENT, global_slot: 999, life_state: "secret" },
  );
  assert.equal(JSON.stringify(injected), JSON.stringify(baseline));
  assert.equal(baseline.title, "2 Hidden Statuses");
  assert.equal(baseline.rows.length, 2);
  assert.match(fullText(baseline), /Source agent identity is not disclosed\./u);
  assert.doesNotMatch(
    fullText(injected),
    /secret-overflow-source|secret-overflow-event|secret-overflow-accessible-name|0\.001|break/iu,
  );
});

test("researcher agent status composition stays full when modifiers are unavailable", () => {
  const descriptor = explainAgent(
    {
      ...RECIPIENT,
      current_health: 75,
      max_health: 100,
      effective_movement_speed: 1,
      ultimate_cooldown_remaining: 0,
      life_state: "alive",
      steps_until_out_of_combat: 1,
      statuses: [
        researcherStatus({
          direct_source_evidence: [
            {
              source_global_slot: SOURCE_A.global_slot,
              source_public_agent_id: SOURCE_A.public_agent_id,
              event_id: "private-event-id",
            },
          ],
        }),
      ],
    },
    {},
    null,
    [SOURCE_A],
  );
  assert.equal(
    sectionRows(descriptor, "Current State").find(
      (row) => row.label === "Aggregate Aura Modifiers",
    )?.value,
    "Unavailable",
  );
  assert.match(
    sectionRows(descriptor, "Current Status Details")[0].value,
    /Agent ID alpha\/9001 · Team A · Hunter/u,
  );
  assert.doesNotMatch(fullText(descriptor), /private-event-id/u);
});

test("researcher status explanation ignores an injected POV discriminator", () => {
  const status = researcherStatus({
    direct_source_evidence: [
      {
        source_global_slot: SOURCE_A.global_slot,
        source_public_agent_id: SOURCE_A.public_agent_id,
        event_id: "technical-only",
      },
    ],
  });
  const baseline = explainStatus(status, RECIPIENT, [SOURCE_A]);
  const injected = explainStatus(
    { ...status, source_evidence: "effect_channel_only" },
    RECIPIENT,
    [SOURCE_A],
  );
  assert.equal(JSON.stringify(injected), JSON.stringify(baseline));
  assert.match(fullText(injected), /Source.*alpha\/9001|Damage|Duration/u);
});

test("legality is locked to exact True/False and one exact sentence", () => {
  /** @type {Array<[0 | 1, boolean, string]>} */
  const cases = [
    [0, true, "Basic"],
    [1, false, "Ultimate"],
  ];
  for (const [lane, available, laneName] of cases) {
    const descriptor = explainLegality(
      {
        target_global_slot: 99,
        lane_0_available: lane === 0 ? available : false,
        lane_1_available: lane === 1 ? available : true,
        armed_lane: lane,
        armed_pair_legal: false,
        python_mask: "secret",
      },
      lane,
    );
    assert.equal(descriptor.title, `${laneName} Legality`);
    assert.equal(rowValue(descriptor, "Status"), available ? "True" : "False");
    assert.equal(
      descriptor.summary,
      `${laneName} ability is ${available ? "" : "not "}available this tick.`,
    );
    assert.doesNotMatch(fullText(descriptor), /mask|armed|pair|target|Python|99/iu);
  }
  assert.throws(() => explainLegality({ lane_0_available: 1 }, 0), /exact boolean/u);
});

test("pending route exposes only exact Source and Selected Target public IDs", () => {
  const descriptor = explainPendingRoute({
    source_global_slot: 1,
    target_global_slot: 7,
    source_public_agent_id: "source::<x>",
    target_public_agent_id: "target&y",
    lane: 1,
    legal: false,
    source_anchor: [1, 2],
    target_anchor: [8, 9],
  });
  assert.equal(descriptor.title, "Ultimate Action Route");
  assert.equal(
    descriptor.summary,
    "Currently selected action intent; no physical path is implied.",
  );
  assert.deepEqual(
    descriptor.rows.map(({ label, value }) => [label, value]),
    [
      ["Source", "Agent ID source::<x>"],
      ["Selected Target", "Agent ID target&y"],
    ],
  );
  assert.doesNotMatch(
    fullText(descriptor),
    /source_global_slot|target_global_slot|legal|source_anchor|target_anchor|\[1|\[8/u,
  );
});

test("visibility and attribution builders never manufacture slot identities", () => {
  const visibility = explainVisibility(
    { observer_global_slot: 1, candidate_global_slot: 7, visible: false },
    { observerAgent: SOURCE_A, candidateAgent: SOURCE_B },
  );
  assert.equal(rowValue(visibility, "Observer"), "Agent ID alpha/9001");
  assert.equal(rowValue(visibility, "Candidate"), "Agent ID beta.17");
  assert.equal(rowValue(visibility, "Visible"), "False");

  const activation = explainActivation({
    eventId: "secret-pov-cue-id",
    tokenId: "rogue_poison",
    sourceSlot: 3,
    targetSlot: 8,
    targetDisclosure: "redacted",
    source: { x: 10, y: 10 },
  });
  assert.equal(rowValue(activation, "Source"), "Agent ID unavailable");
  assert.equal(
    rowValue(activation, "Target"),
    "Target endpoint not disclosed in this view",
  );
  assert.doesNotMatch(
    JSON.stringify(activation),
    /secret-pov-cue-id|id_3|id_8|"x":10/u,
  );

  const net = explainNetHealth({
    eventId: "secret-researcher-event-id",
    netDelta: -0.004,
  });
  assert.equal(rowValue(net, "NET"), "−<0.01");
  assert.equal(rowValue(net, "Recipient"), "Agent ID unavailable");
  assert.doesNotMatch(JSON.stringify(net), /secret-researcher-event-id/u);
});

test("overflow projects every hidden semantic item without slot-derived identity", () => {
  const descriptor = explainOverflow(
    [
      researcherStatus({
        remaining_duration: 7,
        direct_source_evidence: [
          {
            source_global_slot: 1,
            source_public_agent_id: "alpha/9001",
            event_id: "must-not-enter-overflow",
          },
        ],
      }),
      researcherStatus({ token_id: "stun_hunter_trap", remaining_duration: 4 }),
    ],
    "status",
    RECIPIENT,
    [SOURCE_A],
  );
  assert.equal(descriptor.rows.length, 2);
  assert.equal(descriptor.title, "2 Hidden Statuses");
  assert.match(descriptor.rows[0].value, /Duration: 7 Ticks/u);
  assert.match(
    descriptor.rows[0].value,
    /Source: Agent ID alpha\/9001 · Team A · Hunter/u,
  );
  assert.match(descriptor.rows[1].value, /Duration: 4 Ticks/u);
  assert.match(descriptor.rows[1].value, /Source agent not recorded\./u);
  assert.doesNotMatch(fullText(descriptor), /id_8|must-not-enter-overflow/u);
});

function researcherStatus(overrides = {}) {
  return {
    status_channel: 4,
    status_id: "hunter_basic_slow",
    token_id: "slow_hunter_basic",
    source_class_id: 3,
    source_class_name: "Hunter",
    source_action_component: "basic",
    remaining_duration: 2,
    magnitude_kind: "movement_multiplier",
    magnitude: 0.8,
    breaks_on_positive_damage: false,
    direct_source_evidence: [],
    ...overrides,
  };
}

/** @param {number} classId @param {string} className */
function classMechanics(classId, className) {
  return {
    class_id: classId,
    class_name: className,
    maximum_health: 100,
    body_radius: 0.5,
    base_movement_speed: 1,
    observation_radius: 6,
    basic_target_mode: classId === 5 ? "ally" : "enemy",
    basic_interaction_radius: 3,
    basic_raw_damage: classId === 5 ? 0 : 10,
    basic_raw_healing: classId === 5 ? 10 : 0,
    ultimate_target_mode:
      classId === 1 ? "target_none" : classId === 5 ? "ally" : "enemy",
    ultimate_interaction_radius: 4,
    ultimate_cooldown_steps: 5,
    ultimate_raw_damage: classId === 5 ? 0 : 15,
    ultimate_raw_healing: classId === 5 ? 20 : 0,
    out_of_combat_delay_steps: 3,
    out_of_combat_health_regeneration_fraction_per_step: 0.05,
    status_mechanics: [],
    aura_mechanics: [],
  };
}

/** @param {number} value */
function format(value) {
  return Number(value.toFixed(2)).toString();
}
