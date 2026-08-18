import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalAgentIdentity,
  createSpawnShieldView,
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
  explainTechnicalFact,
  explainVisibility,
} from "../src/explanations.js";
import { createSemanticDescriptor, projectSemanticDescriptor } from "../src/tooltip.js";

const RECIPIENT = Object.freeze({
  global_slot: 8,
  public_agent_id: "recipient:<unsafe>&42",
  team_id: 2,
  class_id: 5,
});
const AUTHORIZED_RECIPIENT = Object.freeze({
  ...RECIPIENT,
  presentation_key: "recipient:p5",
});
const SOURCE_A = Object.freeze({
  global_slot: 1,
  presentation_key: "source:h3",
  public_agent_id: "alpha/9001",
  team_id: 1,
  class_id: 3,
  class_name: "Hunter",
});
const SOURCE_B = Object.freeze({
  global_slot: 7,
  presentation_key: "source:r4",
  public_agent_id: "beta.17",
  team_id: 2,
  class_id: 4,
  class_name: "Rogue",
});
const SOURCE_REFERENCE_A = Object.freeze({
  source_presentation_key: "source:h3",
  source_public_agent_id: "alpha/9001",
});

/** @param {ReturnType<typeof createSemanticDescriptor>} descriptor @param {string} label */
function rowValue(descriptor, label) {
  const found = descriptor.rows.find((row) => row.label === label);
  assert.ok(found, `missing row ${label}`);
  return found.value;
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
      statuses: [{ secret: "not a compact fact" }],
      aura_modifiers: [{ secret: "not a compact fact" }],
    },
    { audience: "researcher" },
  );
  assert.equal(Object.isFrozen(descriptor), true);
  assert.equal(Object.isFrozen(descriptor.rows), true);
  assert.equal(Object.isFrozen(descriptor.rows[0]), true);
  assert.equal(Object.isFrozen(descriptor.rows[0].metadata), true);
  assert.equal(Object.isFrozen(descriptor.sections), true);
  assert.equal(descriptor.summary, null);
  assert.equal(descriptor.title, "Agent ID arbitrary-public-id · Mage · Team B");
  assert.deepEqual(
    descriptor.rows.map((row) => row.label),
    [
      "Health",
      "Effective Speed",
      "Ultimate Status",
      "Combat Status",
      "Steps until OOC",
    ],
  );
  assert.equal(rowValue(descriptor, "Effective Speed"), "0.33");
  assert.equal(rowValue(descriptor, "Combat Status"), "IC");
  assert.doesNotMatch(fullText(descriptor), /id_8|not a compact fact/u);
});

test("Technical Frame and replay operational help use the exact finite vocabulary", () => {
  const expected = {
    episode: [
      "Episode",
      "Identifies the authorized live episode represented by this frame.",
    ],
    artifact_digest_prefix: [
      "Artifact digest prefix",
      "These 12 hexadecimal characters locate the canonical Oracle replay without displaying its full hash.",
    ],
    incoming_transition: [
      "Incoming transition",
      "Identifies the authorized transition that produced this displayed frame. The initial frame has no incoming transition.",
    ],
    completion: [
      "Completion",
      "How the captured rollout ended. Rollout completion is independent of host-side processing success.",
    ],
    processing: [
      "Processing",
      "Whether host-side evaluation output was produced successfully. Processing does not change how the rollout ended.",
    ],
    frame: [
      "Frame",
      "The zero-based authorized frame index represented by this presentation.",
    ],
    simulator_step: [
      "Simulator step",
      "The simulator decision step represented by this authorized frame.",
    ],
    ordinary_movement_distance_scale: [
      "Ordinary movement distance scale",
      "The recorded multiplier applied to ordinary voluntary movement distance. Spawn Shield uses its separately authorized absolute movement speed.",
    ],
  };
  for (const [factId, [title, summary]] of Object.entries(expected)) {
    const descriptor = explainTechnicalFact(factId);
    const compact = projectSemanticDescriptor(descriptor, "compact");
    assert.equal(compact.title, title);
    assert.equal(compact.summary, summary);
    assert.deepEqual(compact.rows, []);
    assert.deepEqual(compact.sections, []);
    assert.equal(Object.isFrozen(descriptor), true);
  }
  assert.throws(() => explainTechnicalFact("revision"), /Unknown Technical Frame/u);
});

test("canonical agent identity covers every class and team without slot fallback", () => {
  const classes = [
    [1, "Mage", "mage"],
    [2, "Warrior", "warrior"],
    [3, "Hunter", "hunter"],
    [4, "Rogue", "rogue"],
    [5, "Priest", "priest"],
  ];
  const teams = [
    [1, "Team A"],
    [2, "Team B"],
  ];
  for (const [classId, classLabel, accent] of classes) {
    for (const [teamId, teamLabel] of teams) {
      const publicId = `opaque-${classId}-${teamId}`;
      const identity = canonicalAgentIdentity({
        public_agent_id: publicId,
        class_id: classId,
        team_id: teamId,
        global_slot: 700 + Number(classId),
        presentation_key: `private-slot-${classId}`,
      });
      assert.deepEqual(identity, {
        title: `Agent ID ${publicId} · ${classLabel} · ${teamLabel}`,
        publicIdentity: `Agent ID ${publicId}`,
        classLabel,
        teamLabel,
        accent,
      });
      assert.equal(Object.isFrozen(identity), true);
      assert.doesNotMatch(identity.title, /private-slot|70[1-5]/u);
    }
  }

  for (const malformed of [
    null,
    {},
    {
      public_agent_id: null,
      class_id: 99,
      team_id: 99,
      global_slot: 42,
      presentation_key: "secret-slot-42",
    },
    {
      public_agent_id: null,
      class_id: "1",
      team_id: "2",
      global_slot: 42,
      presentation_key: "secret-slot-42",
    },
  ]) {
    const identity = canonicalAgentIdentity(malformed);
    assert.equal(identity.title, "Agent ID unavailable · Unknown · Unknown");
    assert.equal(identity.accent, "none");
    assert.doesNotMatch(JSON.stringify(identity), /42|secret|slot/u);
  }

  for (const malformedPublicId of [" padded-id ", "line\nbreak", "x".repeat(513)]) {
    assert.deepEqual(
      canonicalAgentIdentity({
        public_agent_id: malformedPublicId,
        class_id: 1,
        team_id: 1,
      }),
      {
        title: "Agent ID unavailable · Mage · Team A",
        publicIdentity: "Agent ID unavailable",
        classLabel: "Mage",
        teamLabel: "Team A",
        accent: "mage",
      },
    );
  }
});

test("compact agent facts have one exact in-combat and out-of-combat allowlist", () => {
  const forbiddenLabels = new Set([
    "Identity",
    "Class",
    "Team",
    "Ultimate Name",
    "Spawn Shield",
    "Selection",
    "Reference",
    "Controlled",
    "Inspected",
    "Now",
    "Life State",
    "Persistent Statuses",
    "Aggregate Aura Modifiers",
  ]);
  for (const classId of [1, 2, 3, 4, 5]) {
    for (const inCombat of [false, true]) {
      const descriptor = explainAgent(
        {
          public_agent_id: `agent-${classId}`,
          class_id: classId,
          team_id: classId % 2 === 0 ? 2 : 1,
          current_health: 55,
          max_health: 100,
          effective_movement_speed: 1.25,
          ultimate_cooldown_remaining: classId - 1,
          steps_until_out_of_combat: inCombat ? 2 : 0,
          spawn_shield_remaining: 9,
          statuses: [{ secret_status: "forbidden-status" }],
          aura_modifiers: [{ secret_aura: "forbidden-aura" }],
          life_state: "forbidden-life-state",
          selected: true,
        },
        {
          audience: "researcher",
          controlled: true,
          selected: true,
          reference: true,
          inspected: true,
        },
      );
      assert.equal(descriptor.summary, null);
      assert.deepEqual(
        descriptor.rows.map((row) => row.label),
        inCombat
          ? [
              "Health",
              "Effective Speed",
              "Ultimate Status",
              "Combat Status",
              "Steps until OOC",
            ]
          : ["Health", "Effective Speed", "Ultimate Status", "Combat Status"],
      );
      assert.equal(rowValue(descriptor, "Health"), "55 / 100");
      assert.equal(rowValue(descriptor, "Effective Speed"), "1.25");
      assert.equal(
        rowValue(descriptor, "Ultimate Status"),
        classId === 1
          ? "Ready"
          : `On cooldown (${classId - 1} ${classId === 2 ? "Tick" : "Ticks"})`,
      );
      assert.equal(rowValue(descriptor, "Combat Status"), inCombat ? "IC" : "OOC");
      if (inCombat) {
        assert.equal(rowValue(descriptor, "Steps until OOC"), "2 Ticks");
      }
      assert.equal(descriptor.sections.length, 0);
      assert.equal(
        descriptor.rows.some((row) => forbiddenLabels.has(row.label)),
        false,
      );
      assert.doesNotMatch(
        JSON.stringify(descriptor),
        /forbidden-|spawn.shield|ultimate name|selection|reference|controlled|inspected|now/iu,
      );
    }
  }
});

const STATUS_CASES = Object.freeze([
  {
    statusId: "priest_blessing_of_freedom_movement_floor",
    title: "Priest (Basic: Heal): Blessing of Freedom",
    effect:
      "Freedom is applied when a Priest heals a same-team target, including itself where same-team targeting permits it. It prevents slow effects from reducing this agent's ordinary movement below the authorized floor; it does not override stun.",
    magnitudeKind: "movement_floor",
    magnitude: 0.85,
    magnitudeLabel: "Movement Floor",
    magnitudeValue: "85% of base movement speed (×0.85)",
    duration: 1,
  },
  {
    statusId: "rogue_poison_stun",
    title: "Rogue (Ultimate: Crippling Poison): Stun",
    effect:
      "A Rogue's Crippling Poison prevents this agent's voluntary movement and combat for its duration. Physics may still displace the body.",
    magnitudeKind: "none",
    magnitude: null,
    magnitudeLabel: null,
    magnitudeValue: null,
    duration: 1,
  },
  {
    statusId: "rogue_poison_slow",
    title: "Rogue (Ultimate: Crippling Poison): Slow",
    effect: "A Rogue's Crippling Poison slows this agent's movement for its duration.",
    magnitudeKind: "movement_multiplier",
    magnitude: 0.5,
    magnitudeLabel: "Movement Effect",
    magnitudeValue: "50% slower (×0.5)",
    duration: 5,
  },
  {
    statusId: "rogue_poison_anti_heal",
    title: "Rogue (Ultimate: Crippling Poison): Anti-Heal",
    effect:
      "A Rogue's Crippling Poison reduces incoming healing and out-of-combat regeneration for its duration.",
    magnitudeKind: "healing_multiplier",
    magnitude: 0.5,
    magnitudeLabel: "Healing Effect",
    magnitudeValue: "50% less healing received (×0.5)",
    duration: 4,
  },
  {
    statusId: "hunter_basic_slow",
    title: "Hunter (Basic: Attack): Slow",
    effect: "A Hunter's Serrated Arrows slow this agent's movement for their duration.",
    magnitudeKind: "movement_multiplier",
    magnitude: 0.85,
    magnitudeLabel: "Movement Effect",
    magnitudeValue: "15% slower (×0.85)",
    duration: 1,
  },
  {
    statusId: "hunter_trap_stun",
    title: "Hunter (Ultimate: Freezing Trap): Stun",
    effect:
      "A Hunter's Freezing Trap prevents this agent's voluntary movement and combat for its duration. Physics may still displace the body.",
    magnitudeKind: "none",
    magnitude: null,
    magnitudeLabel: null,
    magnitudeValue: null,
    duration: 4,
  },
  {
    statusId: "mage_burst_damage_amplification",
    title: "Mage (Ultimate: Burst): Damage Amplification",
    effect:
      "This Mage's Burst increases its outgoing damage for the authorized duration.",
    magnitudeKind: "damage_multiplier",
    magnitude: 1.5,
    magnitudeLabel: "Damage Amplification Effect",
    magnitudeValue: "50% more damage dealt (×1.5)",
    duration: 5,
  },
  {
    statusId: "warrior_charge_slow",
    title: "Warrior (Ultimate: Charge): Slow",
    effect:
      "A Warrior's concussive Charge slows this agent's movement for its duration.",
    magnitudeKind: "movement_multiplier",
    magnitude: 0.5,
    magnitudeLabel: "Movement Effect",
    magnitudeValue: "50% slower (×0.5)",
    duration: 5,
  },
  {
    statusId: "warrior_charge_stun",
    title: "Warrior (Ultimate: Charge): Stun",
    effect:
      "A Warrior's concussive Charge prevents this agent's voluntary movement and combat for its duration. Physics may still displace the body.",
    magnitudeKind: "none",
    magnitude: null,
    magnitudeLabel: null,
    magnitudeValue: null,
    duration: 1,
  },
]);

/**
 * @param {(typeof STATUS_CASES)[number]} statusCase
 * @param {Record<string, unknown>} [overrides]
 */
function durableStatus(statusCase, overrides = {}) {
  return {
    status_channel: STATUS_CASES.indexOf(statusCase),
    status_id: statusCase.statusId,
    configured_duration_steps: statusCase.duration,
    remaining_duration: Math.max(1, statusCase.duration - 1),
    magnitude_kind: statusCase.magnitudeKind,
    magnitude: statusCase.magnitude,
    breaks_on_positive_damage: statusCase.statusId === "hunter_trap_stun",
    direct_sources: [SOURCE_REFERENCE_A],
    ...overrides,
  };
}

test("all nine durable statuses preserve exact facts across audience and source states", () => {
  for (const statusCase of STATUS_CASES) {
    const status = durableStatus(statusCase);
    const researcher = explainStatus(status, RECIPIENT, [SOURCE_A, SOURCE_B]);
    const absent = explainStatus({ ...status, direct_sources: [] }, RECIPIENT, [
      SOURCE_A,
      SOURCE_B,
    ]);
    const pov = explainPovStatus(
      {
        ...status,
        direct_sources: [
          {
            source_global_slot: 999,
            source_public_agent_id: "must-not-be-read",
          },
        ],
      },
      RECIPIENT,
    );

    assert.equal(researcher.title, statusCase.title);
    assert.equal(researcher.summary, statusCase.effect);
    assert.equal(absent.title, statusCase.title);
    assert.equal(absent.summary, statusCase.effect);
    assert.equal(pov.title, statusCase.title);
    assert.equal(pov.summary, statusCase.effect);
    assert.equal(
      rowValue(researcher, "Effect Duration"),
      `${statusCase.duration} ${statusCase.duration === 1 ? "Tick" : "Ticks"}`,
    );
    assert.equal(
      rowValue(researcher, "Duration Remaining"),
      `${Math.max(1, statusCase.duration - 1)} ${Math.max(1, statusCase.duration - 1) === 1 ? "Tick" : "Ticks"}`,
    );
    if (statusCase.magnitudeLabel === null) {
      assert.equal(
        researcher.rows.some((row) =>
          [
            "Movement Effect",
            "Healing Effect",
            "Damage Amplification Effect",
            "Movement Floor",
          ].includes(row.label),
        ),
        false,
      );
    } else {
      assert.equal(
        rowValue(researcher, statusCase.magnitudeLabel),
        statusCase.magnitudeValue,
      );
    }
    assert.equal(
      rowValue(researcher, "Source"),
      "Agent ID alpha/9001 · Hunter · Team A",
    );
    assert.equal(rowValue(absent, "Source"), "Unavailable in this artifact");
    assert.equal(rowValue(pov, "Source"), "Not disclosed in Agent POV");
    assert.deepEqual(
      researcher.rows.filter((row) => row.label !== "Source"),
      pov.rows.filter((row) => row.label !== "Source"),
    );
    assert.deepEqual(researcher.sections, []);
    assert.deepEqual(pov.sections, []);

    const hasBreak = statusCase.statusId === "hunter_trap_stun";
    assert.equal(
      researcher.rows.some((row) => row.label === "Break Rule"),
      hasBreak,
    );
    if (hasBreak) {
      assert.equal(
        rowValue(researcher, "Break Rule"),
        "Ends when this agent receives positive raw damage",
      );
    } else {
      const forged = explainStatus(
        durableStatus(statusCase, { breaks_on_positive_damage: true }),
        RECIPIENT,
        [SOURCE_A],
      );
      assert.equal(
        forged.rows.some((row) => row.label === "Break Rule"),
        false,
      );
    }
  }
});

test("status source integration authorizes only exact presentation-key and public-ID joins", () => {
  const statusCase = STATUS_CASES.find(
    (candidate) => candidate.statusId === "hunter_basic_slow",
  );
  assert.ok(statusCase);
  const exact = explainStatus(durableStatus(statusCase), RECIPIENT, [SOURCE_A]);
  const wrongKey = explainStatus(
    durableStatus(statusCase, {
      direct_sources: [
        {
          source_presentation_key: "wrong-key",
          source_public_agent_id: SOURCE_A.public_agent_id,
        },
      ],
    }),
    RECIPIENT,
    [SOURCE_A],
  );
  const slotFallback = explainStatus(
    durableStatus(statusCase, {
      direct_sources: [
        {
          source_global_slot: SOURCE_A.global_slot,
          source_public_agent_id: SOURCE_A.public_agent_id,
        },
      ],
    }),
    RECIPIENT,
    [SOURCE_A],
  );

  assert.equal(rowValue(exact, "Source"), "Agent ID alpha/9001 · Hunter · Team A");
  assert.equal(rowValue(wrongKey, "Source"), "Unavailable in this artifact");
  assert.equal(rowValue(slotFallback, "Source"), "Unavailable in this artifact");
  assert.doesNotMatch(fullText(slotFallback), /global.slot|id_1/u);
});

test("Freezing Trap refresh and reapplication preserve configured duration and exact break copy", () => {
  const trap = STATUS_CASES.find(
    (candidate) => candidate.statusId === "hunter_trap_stun",
  );
  assert.ok(trap);
  for (const remaining of [1, 2, 4]) {
    const descriptor = explainStatus(
      durableStatus(trap, { remaining_duration: remaining }),
      RECIPIENT,
      [SOURCE_A],
    );
    assert.equal(rowValue(descriptor, "Effect Duration"), "4 Ticks");
    assert.equal(
      rowValue(descriptor, "Duration Remaining"),
      `${remaining} ${remaining === 1 ? "Tick" : "Ticks"}`,
    );
    assert.equal(
      rowValue(descriptor, "Break Rule"),
      "Ends when this agent receives positive raw damage",
    );
  }
  const noBreak = explainStatus(
    durableStatus(trap, { breaks_on_positive_damage: false }),
    RECIPIENT,
    [SOURCE_A],
  );
  assert.equal(
    noBreak.rows.some((row) => row.label === "Break Rule"),
    false,
  );
});

test("POV status preserves effect facts while leaving source payloads unread", () => {
  const statusCase = STATUS_CASES.find(
    (candidate) => candidate.statusId === "hunter_basic_slow",
  );
  assert.ok(statusCase);
  let sourceReads = 0;
  /** @type {unknown[]} */
  const hiddenSources = [];
  Object.defineProperty(hiddenSources, "0", {
    enumerable: true,
    get() {
      sourceReads += 1;
      return SOURCE_REFERENCE_A;
    },
  });
  hiddenSources.length = 1;
  const descriptor = explainPovStatus(
    durableStatus(statusCase, { direct_sources: hiddenSources }),
    RECIPIENT,
  );

  assert.equal(rowValue(descriptor, "Movement Effect"), "15% slower (×0.85)");
  assert.equal(rowValue(descriptor, "Effect Duration"), "1 Tick");
  assert.equal(rowValue(descriptor, "Duration Remaining"), "1 Tick");
  assert.equal(rowValue(descriptor, "Source"), "Not disclosed in Agent POV");
  assert.equal(sourceReads, 0);
});

test("durable status cards exclude superseded, inferred, and forbidden copy", () => {
  const serialized = STATUS_CASES.map((statusCase) =>
    fullText(explainStatus(durableStatus(statusCase), RECIPIENT, [SOURCE_A])),
  ).join("\n");
  assert.doesNotMatch(
    serialized,
    /incapacitat|preventing action|while the status remains|recipient takes recorded|No positive-damage|Source agent not recorded|Direct Source|source.global.slot|event.id|source.action.component/iu,
  );
  assert.equal(
    [...serialized.matchAll(/Ends when this agent receives positive raw damage/gu)]
      .length,
    1,
  );
});

test("aggregate aura cards use exact copy, one effect row, and no source", () => {
  const cases = [
    {
      aura_id: "mage_damage_amplification",
      token_id: "mage_amplification",
      multiplier: 1.23456,
      title: "Sorcerer's Empowerment · Mage Damage Amplification Aura",
      summary: "This agent benefits from authorized Mage aura coverage.",
      label: "Aggregated Damage Amplification Effect",
      value: "23.46% more damage dealt",
      accent: "mage",
    },
    {
      aura_id: "warrior_damage_mitigation",
      token_id: "warrior_mitigation",
      multiplier: 0.763,
      title: "Guardian's Barrier · Warrior Damage Mitigation Aura",
      summary: "This agent benefits from authorized Warrior aura coverage.",
      label: "Aggregated Damage Mitigation Effect",
      value: "23.7% less damage received",
      accent: "warrior",
    },
  ];
  for (const auraCase of cases) {
    const baseline = explainModifier(auraCase, AUTHORIZED_RECIPIENT);
    const injected = explainModifier(
      {
        ...auraCase,
        emitter_global_slots: [1, 7, 999],
        source_presentation_key: "secret-source",
        source_public_agent_id: "secret-public-id",
        nearest_emitter: { position: [999, -42] },
      },
      AUTHORIZED_RECIPIENT,
    );
    assert.ok(baseline);
    assert.ok(injected);
    assert.equal(JSON.stringify(injected), JSON.stringify(baseline));
    assert.equal(baseline.title, auraCase.title);
    assert.equal(baseline.summary, auraCase.summary);
    assert.equal(baseline.accent, auraCase.accent);
    assert.deepEqual(
      baseline.rows.map(({ label, value }) => [label, value]),
      [[auraCase.label, auraCase.value]],
    );
    assert.deepEqual(baseline.sections, []);
    assert.doesNotMatch(fullText(baseline), /Source|emitter|secret/iu);
  }
});

test("aggregate aura cards require one exact authorized recipient identity", () => {
  const modifier = {
    aura_id: "mage_damage_amplification",
    token_id: "mage_amplification",
    multiplier: 1.15,
  };
  assert.equal(explainModifier(modifier, {}), null);
  assert.equal(
    explainModifier(modifier, {
      global_slot: 8,
      public_agent_id: "recipient:<unsafe>&42",
      class_id: 5,
      team_id: 2,
    }),
    null,
  );

  let classReads = 0;
  const accessorRecipient = {
    presentation_key: AUTHORIZED_RECIPIENT.presentation_key,
    public_agent_id: AUTHORIZED_RECIPIENT.public_agent_id,
    team_id: AUTHORIZED_RECIPIENT.team_id,
  };
  Object.defineProperty(accessorRecipient, "class_id", {
    enumerable: true,
    get() {
      classReads += 1;
      return AUTHORIZED_RECIPIENT.class_id;
    },
  });
  assert.equal(explainModifier(modifier, accessorRecipient), null);
  assert.equal(classReads, 0);

  const hostileRecipient = new Proxy(AUTHORIZED_RECIPIENT, {
    getOwnPropertyDescriptor() {
      throw new Error("recipient identity must fail closed");
    },
  });
  assert.equal(explainModifier(modifier, hostileRecipient), null);
});

test("spawn shield view is exact across V1/V2/unavailable and every audience", () => {
  const v2Summary =
    "While the spawn shield is active, this agent is protected, concealed from opponents, untargetable, excluded from aura effects, and limited to movement. It phases through agents until body collision resumes at the endpoint of its expiring transition.";
  const v2Mechanics = {
    availability_kind: "available_v2",
    configured_duration_steps: 3,
    movement_speed: 2,
    protection_effect: "invulnerable",
    visibility_effect: "concealed_from_opponents",
    targetability_effect: "untargetable",
    action_scope: "movement_only",
    aura_effect: "excluded_as_emitter_and_beneficiary",
    agent_collision_effect: "phased_until_expiring_endpoint_rejoin",
    ordinary_application_mechanism: "end_of_transition_respawn_lifecycle",
  };
  const audiences = ["Oracle", "NoShared", "Shared"];
  const variants = [
    {
      name: "V2",
      mechanics: v2Mechanics,
      summary: v2Summary,
      rows: /** @param {string} owner */ (owner) => [
        ["Protection Effect", "Invulnerable"],
        ["Movement Speed", "2"],
        ["Visibility Effect", "Concealed from opponents"],
        ["Targetability Effect", "Untargetable"],
        ["Action Effect", "Movement only"],
        ["Aura Effect", "Excluded as emitter and beneficiary"],
        ["Agent Collision Effect", "Phased until expiring endpoint rejoin"],
        ["Effect Duration", "3 Ticks"],
        ["Duration Remaining", "3 Ticks"],
        ["Owner", owner],
        ["Source", "Not recorded"],
        ["Ordinary Application", "End-of-transition respawn lifecycle"],
      ],
    },
    {
      name: "V1",
      mechanics: {
        availability_kind: "available",
        configured_duration_steps: 3,
        movement_speed: 2,
      },
      summary: null,
      rows: /** @param {string} owner */ (owner) => [
        ["Movement Speed", "2"],
        ["Effect Duration", "3 Ticks"],
        ["Duration Remaining", "3 Ticks"],
        ["Owner", owner],
        ["Source", "Not recorded"],
      ],
    },
    {
      name: "unavailable",
      mechanics: { availability_kind: "unavailable" },
      summary: null,
      rows: /** @param {string} owner */ (owner) => [
        ["Duration Remaining", "3 Ticks"],
        ["Owner", owner],
        ["Source", "Not recorded"],
      ],
    },
  ];

  for (const audience of audiences) {
    const publicAgentId = `${audience.toLowerCase()}-shielded`;
    const agent = {
      presentation_key: `${audience.toLowerCase()}:shielded`,
      public_agent_id: publicAgentId,
      class_id: 1,
      team_id: 1,
      spawn_shield_remaining: 3,
      audience_sentinel: audience,
    };
    const owner = `Agent ID ${publicAgentId} · Mage · Team A`;
    for (const variant of variants) {
      const view = createSpawnShieldView(agent, variant.mechanics);
      const descriptor = view.descriptor;
      assert.equal(Object.isFrozen(view), true, `${audience}/${variant.name}`);
      assert.equal(view.active, true, `${audience}/${variant.name}`);
      assert.equal(view.badgeText, "S3", `${audience}/${variant.name}`);
      assert.equal(view.remainingTicks, 3, `${audience}/${variant.name}`);
      assert.equal(
        view.rootAriaLabel,
        "Spawn Shield active, 3 ticks remaining",
        `${audience}/${variant.name}`,
      );
      assert.equal(view.shieldAriaLabel, "Spawn Shield");
      assert.equal(descriptor.kind, "status");
      assert.equal(descriptor.title, "Spawn Shield");
      assert.equal(descriptor.summary, variant.summary, `${audience}/${variant.name}`);
      assert.deepEqual(
        descriptor.rows.map(({ label, value }) => [label, value]),
        variant.rows(owner),
        `${audience}/${variant.name}`,
      );
      assert.equal(
        JSON.stringify(explainSpawnShield(agent, variant.mechanics)),
        JSON.stringify(descriptor),
      );
      assert.doesNotMatch(fullText(descriptor), /audience_sentinel/u);
      if (variant.name !== "V2") {
        assert.doesNotMatch(
          fullText(descriptor),
          /Invulnerable|Concealed|Untargetable|Movement only|Aura Effect|Collision|Ordinary Application/u,
          `${audience}/${variant.name}`,
        );
      }
    }
  }
});

test("malformed spawn shield mechanics fail closed without invoking accessors", () => {
  const agent = {
    presentation_key: "shield:malformed",
    public_agent_id: "malformed-shielded",
    class_id: 2,
    team_id: 2,
    spawn_shield_remaining: 1,
  };
  const unavailable = createSpawnShieldView(agent, {
    availability_kind: "unavailable",
  }).descriptor;
  let accessorReads = 0;
  const accessor = {};
  Object.defineProperty(accessor, "availability_kind", {
    enumerable: true,
    get() {
      accessorReads += 1;
      return "available_v2";
    },
  });
  const extraV2 = {
    availability_kind: "available_v2",
    configured_duration_steps: 3,
    movement_speed: 2,
    protection_effect: "invulnerable",
    visibility_effect: "concealed_from_opponents",
    targetability_effect: "untargetable",
    action_scope: "movement_only",
    aura_effect: "excluded_as_emitter_and_beneficiary",
    agent_collision_effect: "phased_until_expiring_endpoint_rejoin",
    ordinary_application_mechanism: "end_of_transition_respawn_lifecycle",
    browser_invented_claim: true,
  };
  const hostileV2 = new Proxy(extraV2, {
    ownKeys() {
      throw new Error("must fail closed");
    },
  });
  for (const malformed of [null, accessor, extraV2, hostileV2]) {
    assert.equal(
      JSON.stringify(createSpawnShieldView(agent, malformed).descriptor),
      JSON.stringify(unavailable),
    );
  }
  assert.equal(accessorReads, 0);
});

test("five cooldown cards use canonical owners, Ultimate tokens, and exact rows", () => {
  /** @type {Array<[number, string, string, string]>} */
  const cases = [
    [1, "Mage", "mage", "Burst"],
    [2, "Warrior", "warrior", "Charge"],
    [3, "Hunter", "hunter", "Freezing Trap"],
    [4, "Rogue", "rogue", "Crippling Poison"],
    [5, "Priest", "priest", "Holy Word: Salvation"],
  ];
  for (const [classId, className, accent, ultimateName] of cases) {
    const owner = {
      presentation_key: `cooldown:${classId}`,
      public_agent_id: `opaque-${classId * 17}`,
      class_id: classId,
      team_id: classId % 2 === 0 ? 2 : 1,
    };
    const ticks = classId - 1;
    const descriptor = explainCooldown(
      {
        presentation_key: owner.presentation_key,
        public_agent_id: owner.public_agent_id,
        ultimate_cooldown_remaining: ticks,
        current_health: 999,
      },
      owner,
    );
    assert.ok(descriptor);
    const team = owner.team_id === 1 ? "Team A" : "Team B";
    const canonical = `Agent ID ${owner.public_agent_id} · ${className} · ${team}`;
    assert.equal(descriptor.title, `${ultimateName} Cooldown · ${canonical}`);
    assert.equal(descriptor.summary, null);
    assert.equal(descriptor.accent, accent);
    assert.deepEqual(
      descriptor.rows.map(({ label, value }) => [label, value]),
      ticks === 0
        ? [
            ["Ultimate Status", "Ready"],
            ["Source", canonical],
          ]
        : [
            ["Remaining Cooldown", `${ticks} ${ticks === 1 ? "Tick" : "Ticks"}`],
            ["Source", canonical],
          ],
    );
  }
  assert.equal(
    explainCooldown(
      {
        presentation_key: "wrong",
        public_agent_id: AUTHORIZED_RECIPIENT.public_agent_id,
        ultimate_cooldown_remaining: 1,
      },
      AUTHORIZED_RECIPIENT,
    ),
    null,
  );
});

test("three range cards require exact owner joins and exact four-row copy", () => {
  for (const [kind, title] of [
    ["observation", "Observation Range"],
    [
      "basic",
      `Basic Range · Agent ID ${AUTHORIZED_RECIPIENT.public_agent_id} · Priest · Team B`,
    ],
    [
      "ultimate",
      `Ultimate Range · Agent ID ${AUTHORIZED_RECIPIENT.public_agent_id} · Priest · Team B`,
    ],
  ]) {
    const descriptor = explainRange(
      {
        presentation_key: AUTHORIZED_RECIPIENT.presentation_key,
        public_agent_id: AUTHORIZED_RECIPIENT.public_agent_id,
        kind,
        radius: Math.PI,
      },
      AUTHORIZED_RECIPIENT,
    );
    assert.ok(descriptor);
    assert.equal(descriptor.title, title);
    assert.equal(descriptor.summary, null);
    assert.equal(descriptor.accent, "priest");
    assert.deepEqual(
      descriptor.rows.map(({ label, value }) => [label, value]),
      [
        ["Radius", "3.14"],
        ["Owner ID", `Agent ID ${AUTHORIZED_RECIPIENT.public_agent_id}`],
        ["Team", "Team B"],
        ["Class", "Priest"],
      ],
    );
  }

  for (const malformed of [
    {
      presentation_key: "wrong",
      public_agent_id: AUTHORIZED_RECIPIENT.public_agent_id,
      kind: "basic",
      radius: 2.5,
    },
    {
      presentation_key: AUTHORIZED_RECIPIENT.presentation_key,
      public_agent_id: "wrong",
      kind: "ultimate",
      radius: 2.5,
    },
    {
      global_slot: 8,
      kind: "observation",
      radius: 2.5,
    },
  ]) {
    assert.equal(explainRange(malformed, AUTHORIZED_RECIPIENT), null);
  }
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

test("aura fields use exact copy and key-plus-public-ID source joins only", () => {
  const cases = [
    {
      aura_id: "mage_damage_amplification",
      token_id: "mage_amplification",
      owner: {
        presentation_key: "aura:mage",
        public_agent_id: "mage/alpha",
        class_id: 1,
        team_id: 1,
      },
      radius: 4.567,
      multiplier: 1.2,
      title: "Sorcerer's Empowerment · Mage Damage Amplification Aura",
      summary:
        "This Mage radiates arcane magic, amplifying outgoing damage for eligible unshielded same-team agents in its radius, including itself.",
      label: "Damage Amplification Effect",
      value: "20% more damage dealt per emitter",
      accent: "mage",
    },
    {
      aura_id: "warrior_damage_mitigation",
      token_id: "warrior_mitigation",
      owner: {
        presentation_key: "aura:warrior",
        public_agent_id: "warrior/beta",
        class_id: 2,
        team_id: 2,
      },
      radius: 3,
      multiplier: 0.8,
      title: "Guardian's Barrier · Warrior Damage Mitigation Aura",
      summary:
        "This Warrior emanates a defensive aura, mitigating incoming damage for eligible unshielded same-team agents in its radius, including itself.",
      label: "Damage Mitigation Effect",
      value: "20% less damage received per emitter",
      accent: "warrior",
    },
  ];
  for (const auraCase of cases) {
    const field = {
      aura_id: auraCase.aura_id,
      token_id: auraCase.token_id,
      source_presentation_key: auraCase.owner.presentation_key,
      source_public_agent_id: auraCase.owner.public_agent_id,
      radius: auraCase.radius,
      per_emitter_multiplier: auraCase.multiplier,
      source_global_slot: 999,
      source_class_id: 999,
      beneficiary_relation: "browser-must-not-infer",
      center: [999, -42],
    };
    const descriptor = explainAura(field, auraCase.owner);
    assert.ok(descriptor);
    assert.equal(descriptor.title, auraCase.title);
    assert.equal(descriptor.summary, auraCase.summary);
    assert.equal(descriptor.accent, auraCase.accent);
    assert.deepEqual(
      descriptor.rows.map(({ label, value }) => [label, value]),
      [
        [auraCase.label, auraCase.value],
        ["Effect Radius", format(auraCase.radius)],
        [
          "Source",
          `Agent ID ${auraCase.owner.public_agent_id} · ${auraCase.accent === "mage" ? "Mage · Team A" : "Warrior · Team B"}`,
        ],
      ],
    );
    assert.deepEqual(descriptor.sections, []);

    const injected = explainAura(
      {
        ...field,
        source_global_slot: -1,
        source_class_id: -1,
        source_class_name: "Secret",
        eligible_recipient_slots: [1, 2, 3],
        proximity_result: true,
      },
      auraCase.owner,
    );
    assert.equal(JSON.stringify(injected), JSON.stringify(descriptor));

    const mismatched = explainAura(
      { ...field, source_public_agent_id: "wrong-public-id" },
      auraCase.owner,
    );
    assert.ok(mismatched);
    assert.equal(rowValue(mismatched, "Source"), "Unavailable in this artifact");
    assert.doesNotMatch(fullText(mismatched), /wrong-public-id/u);
  }
});

test("Agent POV aura attribution is source-inert and redacts before hidden reads", () => {
  let hiddenSourceReads = 0;
  const rawField = new Proxy(
    {
      aura_id: "mage_damage_amplification",
      token_id: "mage_amplification",
      radius: 2,
      per_emitter_multiplier: 1.15,
      source_presentation_key: "hidden:presentation:key",
      source_public_agent_id: "hidden-public-id",
    },
    {
      get(target, key, receiver) {
        if (key === "source_presentation_key" || key === "source_public_agent_id") {
          hiddenSourceReads += 1;
          throw new Error("Agent POV must not read hidden aura source values");
        }
        return Reflect.get(target, key, receiver);
      },
      getOwnPropertyDescriptor(target, key) {
        if (key === "source_presentation_key" || key === "source_public_agent_id") {
          hiddenSourceReads += 1;
          throw new Error("Agent POV must not inspect hidden aura source descriptors");
        }
        return Reflect.getOwnPropertyDescriptor(target, key);
      },
      ownKeys() {
        hiddenSourceReads += 1;
        throw new Error("Agent POV must not enumerate the hidden aura source record");
      },
    },
  );
  const hiddenSourceAgent = new Proxy(
    {
      presentation_key: "hidden:presentation:key",
      public_agent_id: "hidden-public-id",
      class_id: 1,
      team_id: 1,
    },
    {
      get() {
        hiddenSourceReads += 1;
        throw new Error("Agent POV must not read the hidden source agent");
      },
      getOwnPropertyDescriptor() {
        hiddenSourceReads += 1;
        throw new Error("Agent POV must not inspect the hidden source agent");
      },
      ownKeys() {
        hiddenSourceReads += 1;
        throw new Error("Agent POV must not enumerate the hidden source agent");
      },
    },
  );

  const descriptor = explainAura(rawField, hiddenSourceAgent, "agent_pov");
  assert.ok(descriptor);
  assert.equal(rowValue(descriptor, "Source"), "Not disclosed in Agent POV");
  assert.equal(descriptor.id, "aura:agent-pov:mage_amplification");
  assert.doesNotMatch(
    fullText(descriptor),
    /hidden-public-id|hidden:presentation:key/u,
  );
  assert.equal(hiddenSourceReads, 0);
});

test("C2 helpers snapshot every consumed field and fail closed on accessors", () => {
  let sourceAccessorReads = 0;
  const accessorField = {
    aura_id: "mage_damage_amplification",
    token_id: "mage_amplification",
    radius: 3,
    per_emitter_multiplier: 1.2,
  };
  Object.defineProperties(accessorField, {
    source_presentation_key: {
      enumerable: true,
      get() {
        sourceAccessorReads += 1;
        return "aura:mage";
      },
    },
    source_public_agent_id: {
      enumerable: true,
      get() {
        sourceAccessorReads += 1;
        return "mage/alpha";
      },
    },
  });
  const owner = {
    presentation_key: "aura:mage",
    public_agent_id: "mage/alpha",
    class_id: 1,
    team_id: 1,
  };
  const descriptor = explainAura(accessorField, owner);
  assert.ok(descriptor);
  assert.equal(sourceAccessorReads, 0);
  assert.equal(rowValue(descriptor, "Source"), "Unavailable in this artifact");

  let scalarReads = 0;
  const auraRadiusAccessor = {
    aura_id: "mage_damage_amplification",
    token_id: "mage_amplification",
    per_emitter_multiplier: 1.15,
    source_presentation_key: owner.presentation_key,
    source_public_agent_id: owner.public_agent_id,
  };
  Object.defineProperty(auraRadiusAccessor, "radius", {
    enumerable: true,
    get() {
      scalarReads += 1;
      throw new Error("aura radius getter must remain unread");
    },
  });
  const radiusDescriptor = explainAura(auraRadiusAccessor, owner);
  assert.ok(radiusDescriptor);
  assert.equal(rowValue(radiusDescriptor, "Effect Radius"), "Unavailable");

  const aggregateMultiplierAccessor = {
    aura_id: "mage_damage_amplification",
    token_id: "mage_amplification",
  };
  Object.defineProperty(aggregateMultiplierAccessor, "multiplier", {
    enumerable: true,
    get() {
      scalarReads += 1;
      throw new Error("aggregate multiplier getter must remain unread");
    },
  });
  const aggregateDescriptor = explainModifier(
    aggregateMultiplierAccessor,
    AUTHORIZED_RECIPIENT,
  );
  assert.ok(aggregateDescriptor);
  assert.equal(
    rowValue(aggregateDescriptor, "Aggregated Damage Amplification Effect"),
    "Effect unavailable",
  );

  const cooldownAccessor = {
    presentation_key: owner.presentation_key,
    public_agent_id: owner.public_agent_id,
  };
  for (const key of ["ultimate_cooldown_remaining", "ultimate_cooldown"]) {
    Object.defineProperty(cooldownAccessor, key, {
      enumerable: true,
      get() {
        scalarReads += 1;
        throw new Error("cooldown getter must remain unread");
      },
    });
  }
  assert.equal(explainCooldown(cooldownAccessor, owner), null);

  const rangeRadiusAccessor = {
    presentation_key: owner.presentation_key,
    public_agent_id: owner.public_agent_id,
    kind: "basic",
  };
  Object.defineProperty(rangeRadiusAccessor, "radius", {
    enumerable: true,
    get() {
      scalarReads += 1;
      throw new Error("range radius getter must remain unread");
    },
  });
  assert.equal(explainRange(rangeRadiusAccessor, owner), null);

  const legalityAccessor = {
    owner_presentation_key: owner.presentation_key,
    owner_public_agent_id: owner.public_agent_id,
    lane_1_available: true,
  };
  Object.defineProperty(legalityAccessor, "lane_0_available", {
    enumerable: true,
    get() {
      scalarReads += 1;
      throw new Error("legality getter must remain unread");
    },
  });
  assert.equal(explainLegality(legalityAccessor, 0, owner), null);
  const untouchedOtherLane = explainLegality(legalityAccessor, 1, owner);
  assert.ok(untouchedOtherLane);
  assert.equal(rowValue(untouchedOtherLane, "Status"), "True");
  assert.equal(scalarReads, 0);

  let vocabularyReads = 0;
  const auraVocabularyAccessor = {
    token_id: "mage_amplification",
    radius: 2,
    per_emitter_multiplier: 1.15,
    source_presentation_key: owner.presentation_key,
    source_public_agent_id: owner.public_agent_id,
  };
  Object.defineProperty(auraVocabularyAccessor, "aura_id", {
    enumerable: true,
    get() {
      vocabularyReads += 1;
      throw new Error("aura vocabulary getter must remain unread");
    },
  });
  assert.doesNotThrow(() => explainAura(auraVocabularyAccessor, owner));

  const rangeKindAccessor = {
    presentation_key: owner.presentation_key,
    public_agent_id: owner.public_agent_id,
    radius: 2,
  };
  Object.defineProperty(rangeKindAccessor, "kind", {
    enumerable: true,
    get() {
      vocabularyReads += 1;
      throw new Error("range kind getter must remain unread");
    },
  });
  assert.equal(explainRange(rangeKindAccessor, owner), null);

  const classAccessorOwner = {
    presentation_key: owner.presentation_key,
    public_agent_id: owner.public_agent_id,
    team_id: owner.team_id,
  };
  Object.defineProperty(classAccessorOwner, "class_id", {
    enumerable: true,
    get() {
      vocabularyReads += 1;
      throw new Error("class getter must remain unread");
    },
  });
  assert.equal(
    explainCooldown(
      {
        presentation_key: owner.presentation_key,
        public_agent_id: owner.public_agent_id,
        ultimate_cooldown: 1,
      },
      classAccessorOwner,
    ),
    null,
  );
  assert.equal(vocabularyReads, 0);
});

test("C2 helpers absorb hostile scientific-record proxy traps", () => {
  const owner = {
    presentation_key: "hostile:owner",
    public_agent_id: "owner/hostile",
    class_id: 1,
    team_id: 1,
  };
  const throwingRecord = new Proxy(
    {},
    {
      getOwnPropertyDescriptor() {
        throw new Error("scientific record must fail closed");
      },
    },
  );
  assert.equal(explainAura(throwingRecord, owner), null);
  assert.equal(explainModifier(throwingRecord, AUTHORIZED_RECIPIENT), null);
  assert.equal(explainCooldown(throwingRecord, owner), null);
  assert.equal(explainRange(throwingRecord, owner), null);
  assert.equal(explainLegality(throwingRecord, 0, owner), null);
});

test("POV agent builder is byte-noninterfering with researcher-only extras", () => {
  const authorized = {
    ...RECIPIENT,
    current_health: 50,
    max_health: 100,
    effective_movement_speed: 1.25,
    ultimate_cooldown_remaining: 2,
    steps_until_out_of_combat: 2,
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
  const direct = explainPovAgent(authorized, {
    controlled: true,
    inspected: true,
  });
  const descriptor = explainAgent(
    {
      ...authorized,
      aura_modifiers: [{ multiplier: 9 }],
      effective_speed: 99,
      ultimate_cooldown: 99,
      direct_source_evidence: [{ source_public_agent_id: "secret" }],
      life_state: "SECRET_RESEARCHER_STATE",
    },
    {
      audience: "agent_pov",
      controlled: true,
      selected: true,
      reference: true,
      inspected: true,
    },
  );
  assert.equal(JSON.stringify(descriptor), JSON.stringify(direct));
  assert.equal(descriptor.summary, null);
  assert.deepEqual(
    descriptor.rows.map((row) => row.label),
    [
      "Health",
      "Effective Speed",
      "Ultimate Status",
      "Combat Status",
      "Steps until OOC",
    ],
  );
  assert.equal(rowValue(descriptor, "Effective Speed"), "1.25");
  assert.equal(rowValue(descriptor, "Ultimate Status"), "On cooldown (2 Ticks)");
  assert.equal(rowValue(descriptor, "Combat Status"), "IC");
  assert.equal(rowValue(descriptor, "Steps until OOC"), "2 Ticks");
  assert.equal(descriptor.sections.length, 0);
  assert.doesNotMatch(
    fullText(descriptor),
    /secret|\u00d79|99|selection|selected target|reference|controlled|inspected|current status details|source agent/iu,
  );
});

test("POV status overflow is byte-noninterfering and discloses no source identity", () => {
  const hunterSlow = STATUS_CASES.find(
    (candidate) => candidate.statusId === "hunter_basic_slow",
  );
  const rogueStun = STATUS_CASES.find(
    (candidate) => candidate.statusId === "rogue_poison_stun",
  );
  assert.ok(hunterSlow);
  assert.ok(rogueStun);
  const authorized = [durableStatus(hunterSlow), durableStatus(rogueStun)];
  const baseline = explainPovOverflow(authorized, RECIPIENT);
  const injected = explainPovOverflow(
    authorized.map((status) => ({
      ...status,
      source_public_agent_id: "secret-overflow-source",
      direct_sources: [{ event_id: "secret-overflow-event" }],
      accessible_name: "secret-overflow-accessible-name",
    })),
    { ...RECIPIENT, global_slot: 999, life_state: "secret" },
  );
  assert.equal(JSON.stringify(injected), JSON.stringify(baseline));
  assert.equal(baseline.title, "2 Hidden Statuses");
  assert.equal(baseline.rows.length, 2);
  assert.match(fullText(baseline), /Source not disclosed in Agent POV\./u);
  assert.doesNotMatch(
    fullText(injected),
    /secret-overflow-source|secret-overflow-event|secret-overflow-accessible-name/iu,
  );
});

test("researcher status explanation ignores an injected POV discriminator", () => {
  const status = researcherStatus({
    direct_sources: [SOURCE_REFERENCE_A],
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

test("legality is owner-bound, class-accented, and Status-only", () => {
  const owner = {
    presentation_key: "legality:owner",
    public_agent_id: "owner/42",
    class_id: 4,
    team_id: 2,
  };
  /** @type {Array<[0 | 1, boolean, string]>} */
  const cases = [
    [0, true, "Basic"],
    [1, false, "Ultimate"],
  ];
  for (const [lane, available, laneName] of cases) {
    const descriptor = explainLegality(
      {
        owner_presentation_key: owner.presentation_key,
        owner_public_agent_id: owner.public_agent_id,
        target_global_slot: 99,
        lane_0_available: lane === 0 ? available : false,
        lane_1_available: lane === 1 ? available : true,
        armed_lane: lane,
        armed_pair_legal: false,
        python_mask: "secret",
      },
      lane,
      owner,
    );
    assert.ok(descriptor);
    assert.equal(descriptor.title, `${laneName} Legality · Agent ID owner/42`);
    assert.equal(descriptor.summary, null);
    assert.equal(descriptor.accent, "rogue");
    assert.equal(rowValue(descriptor, "Status"), available ? "True" : "False");
    assert.deepEqual(
      descriptor.rows.map((row) => row.label),
      ["Status"],
    );
    assert.doesNotMatch(fullText(descriptor), /mask|armed|pair|target|Python|99/iu);
  }
  assert.equal(explainLegality({ lane_0_available: true }, 0, owner), null);
  assert.equal(
    explainLegality(
      {
        owner_presentation_key: owner.presentation_key,
        owner_public_agent_id: "wrong",
        lane_0_available: true,
      },
      0,
      owner,
    ),
    null,
  );
  assert.equal(
    explainLegality(
      {
        owner_presentation_key: owner.presentation_key,
        owner_public_agent_id: owner.public_agent_id,
        lane_0_available: 1,
      },
      0,
      owner,
    ),
    null,
  );
});

test("action route uses epoch-neutral copy and exact Source and Target public IDs", () => {
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
    "Authorized action route; no physical path is implied.",
  );
  assert.deepEqual(
    descriptor.rows.map(({ label, value }) => [label, value]),
    [
      ["Source", "Agent ID source::<x>"],
      ["Target", "Agent ID target&y"],
    ],
  );
  assert.doesNotMatch(fullText(descriptor), /currently selected|Selected Target/iu);
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
  assert.equal(
    visibility.summary,
    "Oracle View visibility diagnostic copied from the normalized scene.",
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
  const trap = STATUS_CASES.find(
    (candidate) => candidate.statusId === "hunter_trap_stun",
  );
  assert.ok(trap);
  const descriptor = explainOverflow(
    [
      researcherStatus({
        configured_duration_steps: 9,
        remaining_duration: 7,
        direct_sources: [SOURCE_REFERENCE_A],
      }),
      durableStatus(trap, { direct_sources: [] }),
    ],
    "status",
    RECIPIENT,
    [SOURCE_A],
  );
  assert.equal(descriptor.rows.length, 2);
  assert.equal(descriptor.title, "2 Hidden Statuses");
  assert.match(descriptor.rows[0].value, /Effect Duration: 9 Ticks/u);
  assert.match(descriptor.rows[0].value, /Duration Remaining: 7 Ticks/u);
  assert.match(
    descriptor.rows[0].value,
    /Source: Agent ID alpha\/9001 · Hunter · Team A/u,
  );
  assert.match(descriptor.rows[1].value, /Duration Remaining: 3 Ticks/u);
  assert.match(descriptor.rows[1].value, /Source: Unavailable in this artifact/u);
  assert.doesNotMatch(fullText(descriptor), /id_8|global.slot/u);
});

function researcherStatus(overrides = {}) {
  return {
    status_channel: 1,
    status_id: "hunter_basic_slow",
    configured_duration_steps: 5,
    remaining_duration: 2,
    magnitude_kind: "movement_multiplier",
    magnitude: 0.8,
    breaks_on_positive_damage: false,
    direct_sources: [],
    ...overrides,
  };
}

/** @param {number} value */
function format(value) {
  return Number(value.toFixed(2)).toString();
}
