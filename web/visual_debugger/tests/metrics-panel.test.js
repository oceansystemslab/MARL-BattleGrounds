/**
 * @file Check metric ordering, missing/applicable values, source-to-recipient views
 * and exact/natural-language searches.
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
  buildMetricSearchIndex as buildIndex,
  findMeasurements,
  metricLabel,
  metricTooltipRows,
  selectedMetricRows,
} from "../src/metrics-panel.js";

const classNames = ["Neutral", "Mage", "Warrior", "Hunter", "Rogue", "Priest"];
const testAgents = [1, 2, 2, 4, 5, 1, 2, 2, 4, 5].map((class_id, slot) => ({
  slot,
  class_id,
  class_name: classNames[class_id],
  team_id: slot < 5 ? 1 : 2,
  active: true,
}));

/** @param {string} kind @param {string} subject
 * @param {Record<string, any>} [extra] */
function facts(kind, subject, extra = {}) {
  return {
    kind,
    subject,
    ability: null,
    status: null,
    status_subject: extra.status ? "recipient" : null,
    source_class_id: null,
    recipient_class_id: null,
    source: subject === "source" ? "agent" : "team",
    relation: null,
    qualifiers: [],
    ...extra,
  };
}

/** @type {Record<string, Record<string, any>>} */
const searchRows = {
  agent_7_basic_kills: {
    scope: "agent",
    subjects: [7],
    search_facts: facts("kill", "source", { ability: "basic", relation: "enemy" }),
  },
  other: {
    scope: "team",
    subjects: [1],
    search_facts: facts("coordination", "source", { source: "team" }),
  },
  first_basic: {
    scope: "agent",
    subjects: [0],
    search_facts: facts("activation", "source", { ability: "basic" }),
  },
  agent_3_ultimate_damage_done: {
    scope: "agent",
    subjects: [3],
    search_facts: facts("damage", "source", { ability: "ultimate", relation: "enemy" }),
  },
  agent_8_healing_prevented_by_poison: {
    scope: "agent",
    subjects: [8],
    search_facts: facts("poison_prevention", "recipient", {
      source: "unknown",
      status: "rogue_poison_anti_heal",
    }),
  },
  agent_4_ultimate_healing_done: {
    scope: "agent",
    subjects: [4],
    search_facts: facts("healing", "source", {
      ability: "ultimate",
      source_class_id: 5,
      relation: "ally",
      qualifiers: ["priest"],
    }),
  },
  agent_4_regenerated_healing: {
    scope: "agent",
    subjects: [4],
    search_facts: facts("regeneration", "recipient", { source: "none" }),
  },
  agent_2_damage_received_while_rogue_poison_slow: {
    scope: "agent",
    subjects: [2],
    search_facts: facts("damage", "recipient", {
      status: "rogue_poison_slow",
      relation: "enemy",
    }),
  },
  agent_2_damage_done: {
    scope: "agent",
    subjects: [2],
    search_facts: facts("damage", "source", { relation: "enemy" }),
  },
  agent_2_healing_done: {
    scope: "agent",
    subjects: [2],
    search_facts: facts("healing", "source", {
      source_class_id: 5,
      relation: "ally",
      qualifiers: ["priest"],
    }),
  },
  agent_2_healing_received: {
    scope: "agent",
    subjects: [2],
    search_facts: facts("healing", "recipient", {
      qualifiers: ["combined"],
      relation: "ally",
    }),
  },
  agent_0_deaths_while_warrior_charge_slow: {
    scope: "agent",
    subjects: [0],
    search_facts: facts("death", "recipient", {
      status: "warrior_charge_slow",
      relation: "enemy",
    }),
  },
  agent_2_rescues: {
    scope: "agent",
    subjects: [2],
    search_facts: facts("rescue", "recipient", {
      source: "class",
      source_class_id: 5,
      relation: "ally",
      qualifiers: ["priest"],
    }),
  },
  agent_2_kill_contributions_per_death: {
    scope: "agent",
    subjects: [2],
    search_facts: facts("kill", "source", {
      relation: "enemy",
      qualifiers: ["contribution"],
    }),
  },
  agent_4_kill_contributions: {
    scope: "agent",
    subjects: [4],
    search_facts: facts("kill", "source", {
      relation: "enemy",
      qualifiers: ["contribution"],
    }),
  },
  agent_2_deaths_while_hunter_basic_slow: {
    scope: "agent",
    subjects: [2],
    search_facts: facts("death", "recipient", {
      status: "hunter_basic_slow",
      relation: "enemy",
    }),
  },
  ...Object.fromEntries(
    [0, 2, 7].map((slot) => [
      `agent_${slot}_deaths`,
      {
        scope: "agent",
        subjects: [slot],
        search_facts: facts("death", "recipient", { relation: "enemy" }),
      },
    ]),
  ),
};

/** @param {Record<string, any>[]} rows @param {Record<string, any>[]} [topics] */
function buildMetricSearchIndex(rows, topics = []) {
  return buildIndex(
    rows.map((row) => {
      assert(searchRows[row.name], `Missing test facts for ${row.name}`);
      return { ...row, ...searchRows[row.name] };
    }),
    topics,
    testAgents,
    classNames,
  );
}

test("metric locations keep paired views disjoint and use the server order", () => {
  const rows = [
    {
      name: "agent_9_healing",
      applicable: true,
      order: 4,
      subject: "Agent ID 9 · Mage · Team B",
      valid: true,
      value: 0,
      locations: [{ topic: "healing_done", view: "totals" }],
      topic_order: { healing_done: [3] },
    },
    {
      name: "agent_2_healing",
      label: "Ultimate Healing",
      description: "Healing from this agent's Ultimate ability.",
      topic_text: {
        ultimate_priest: {
          label: "Salvation Healing",
          description: "How much healing this Priest gave with Salvation.",
          subtitle: "Holy Word: Salvation",
          value: 99,
        },
      },
      applicable: true,
      order: 3,
      subject: "Agent ID 2 · Priest · Team A",
      valid: true,
      value: 12.5,
      locations: [
        { topic: "healing_done", view: "totals" },
        { topic: "ultimate_priest", view: "totals" },
      ],
      topic_order: { healing_done: [2], ultimate_priest: [0] },
    },
    {
      name: "team_b_healing",
      applicable: true,
      order: 2,
      locations: [{ topic: "healing_done", view: "totals" }],
      topic_order: { healing_done: [0] },
    },
    {
      name: "team_a_healing",
      applicable: true,
      order: 1,
      locations: [{ topic: "healing_done", view: "totals" }],
      topic_order: { healing_done: [1] },
    },
    {
      name: "pair",
      applicable: true,
      order: 5,
      locations: [{ topic: "healing_done", view: "recipients" }],
      topic_order: { healing_done: [0] },
    },
  ];
  const before = structuredClone(rows);
  const selected = selectedMetricRows({ statistics: rows }, "healing_done", "totals");
  assert.deepEqual(
    selected.map((/** @type {Record<string, any>} */ row) => row.name),
    ["team_b_healing", "team_a_healing", "agent_2_healing", "agent_9_healing"],
  );
  assert.equal(selected[2].subject, "Agent ID 2 · Priest · Team A");
  assert.equal(selected[2].value, 12.5);
  assert.equal(selected[3].value, 0);
  assert.deepEqual(
    selectedMetricRows({ statistics: rows }, "healing_done", "recipients"),
    [rows[4]],
  );
  assert.deepEqual(
    selectedMetricRows({ statistics: rows }, "ultimate_priest", "totals"),
    [
      {
        ...rows[1],
        label: "Salvation Healing",
        description: "How much healing this Priest gave with Salvation.",
        subtitle: "Holy Word: Salvation",
      },
    ],
  );
  assert.deepEqual(selectedMetricRows({ statistics: rows }, "unknown", "totals"), []);
  assert.deepEqual(rows, before);
  const activation = {
    name: "team_a_warrior_charge_slow_applications",
    label: "Warrior Charge Slow Applications",
    status: "Warrior Charge Slow",
    applicable: true,
    order: 1,
    locations: [
      { topic: "status_applications", view: "totals" },
      { topic: "ultimate_warrior", view: "totals" },
    ],
    topic_order: { status_applications: [0], ultimate_warrior: [0] },
    topic_text: {
      ultimate_warrior: {
        label: "Ultimate Ability Activations",
        subtitle: "Charge",
      },
    },
  };
  const summary = { statistics: [activation] };
  const ultimate = selectedMetricRows(summary, "ultimate_warrior", "totals")[0];
  assert.equal(ultimate.subtitle, "Charge");
  assert.equal(ultimate.label, "Ultimate Ability Activations");
  assert.equal(ultimate.status, "Warrior Charge Slow");
  assert.deepEqual(selectedMetricRows(summary, "status_applications", "totals"), [
    activation,
  ]);
  assert.equal(metricLabel("agent_steps"), "Agent Steps");
});

test("undefined fractions remain visible while inapplicable rows are omitted", () => {
  const common = {
    locations: [{ topic: "abilities", view: "totals" }],
    applicable: true,
  };
  const rows = [
    {
      ...common,
      name: "undefined_fraction",
      topic_order: { abilities: [1, 0] },
      order: 1,
      valid: false,
      value: null,
    },
    {
      ...common,
      name: "zero_count",
      topic_order: { abilities: [0, 9] },
      order: 2,
      valid: true,
      value: 0,
    },
    {
      ...common,
      name: "not_applicable",
      topic_order: { abilities: [0, 0] },
      order: 0,
      applicable: false,
      valid: false,
      value: null,
    },
  ];
  const selected = selectedMetricRows({ statistics: rows }, "abilities", "totals");
  assert.deepEqual(
    selected.map((/** @type {Record<string, any>} */ row) => row.name),
    ["zero_count", "undefined_fraction"],
  );
  assert.equal(selected[0].value, 0);
  assert.equal(selected[1].value, null);
});

test("measurement search includes inactive rows and ranks exact CSV names before other matches", () => {
  const named = {
    name: "agent_7_basic_kills",
    label: "Basic Kill Contributions",
    description: "Help from Basic abilities.",
    order: 9,
    applicable: false,
  };
  const rows = [
    {
      name: "other",
      label: "Related to Agent 7 Basic Kills",
      description: "The supporting detail for that figure.",
      order: 0,
      applicable: true,
    },
    named,
    { ...named, locations: [{ topic: "another", view: "totals" }] },
    {
      name: "first_basic",
      label: "Basic uses",
      description: "How many times an ability was used.",
      order: 1,
      applicable: true,
    },
  ];
  const index = buildMetricSearchIndex(rows);
  assert.equal(index.entries.length, 3);
  assert.deepEqual(
    findMeasurements(index, "  AGENT_7_BASIC_KILLS ").map((row) => row.name),
    [named.name],
  );
  assert.deepEqual(
    findMeasurements(index, "agent 7 basic kills").map((row) => row.name),
    [named.name],
  );
  assert.equal(findMeasurements(index, "agent 7 basic kills")[0].applicable, false);
  assert.deepEqual(
    findMeasurements(index, "basic").map((row) => row.name),
    ["first_basic", named.name],
  );
  assert.deepEqual(
    findMeasurements(index, "ability used").map((row) => row.name),
    ["first_basic"],
  );
  assert.deepEqual(findMeasurements(index, "missing measurement"), []);
  assert.deepEqual(findMeasurements(index, ""), []);
  assert.deepEqual(
    findMeasurements(index, "supporting detail").map((row) => row.name),
    ["other"],
  );

  const topicIndex = buildMetricSearchIndex(
    [
      {
        name: "agent_3_ultimate_damage_done",
        label: "Ultimate Damage",
        description: "Damage from this agent's Ultimate ability.",
        subject: "Agent ID 3 · Rogue · Team A",
        locations: [{ topic: "damage_done" }, { topic: "ultimate_rogue" }],
        order: 0,
        applicable: true,
        primary_topic: "damage_done",
        topic_text: {
          ultimate_rogue: {
            label: "Invented Measurement Alias",
            description:
              "Direct Crippling Poison damage from this Rogue's poison strike.",
          },
        },
      },
      {
        name: "agent_8_healing_prevented_by_poison",
        label: "Priest Healing Prevented",
        description: "Healing blocked by Poison.",
        locations: [{ topic: "ultimate_rogue" }],
        order: 1,
        applicable: false,
        primary_topic: "ultimate_rogue",
        topic_text: {
          ultimate_rogue: { label: "Priest Healing Blocked by Poison" },
        },
      },
      {
        name: "agent_4_ultimate_healing_done",
        label: "Ultimate Healing",
        description:
          "Healing from this agent's Ultimate ability. Excludes regeneration.",
        locations: [{ topic: "ultimate_priest" }],
        order: 2,
        applicable: true,
      },
      {
        name: "agent_4_regenerated_healing",
        label: "Health Restored by Regeneration",
        description: "Health restored automatically.",
        search_terms: ["passive healing", "natural health recovery"],
        locations: [],
        order: 3,
        applicable: true,
      },
    ],
    [
      { name: "damage_done", label: "Damage Done" },
      { name: "ultimate_rogue", label: "Crippling Poison (Rogue Ultimate)" },
      { name: "ultimate_priest", label: "Holy Word: Salvation (Priest Ultimate)" },
    ],
  );
  for (const query of ["crippling poison", "CRIPPL POIS", '"Crippling Poison"']) {
    assert.deepEqual(
      findMeasurements(topicIndex, query).map((row) => row.name),
      ["agent_3_ultimate_damage_done", "agent_8_healing_prevented_by_poison"],
    );
  }
  const alternate = findMeasurements(topicIndex, "direct crippling poison damage");
  assert.equal(alternate[0].name, "agent_3_ultimate_damage_done");
  assert.equal(alternate[0].label, "Ultimate Damage");
  assert.deepEqual(findMeasurements(topicIndex, "invented measurement alias"), []);
  assert.equal(
    findMeasurements(topicIndex, "poison strike")[0].name,
    "agent_3_ultimate_damage_done",
  );
  assert.equal(
    findMeasurements(topicIndex, "agent_8_healing_prevented_by_poison")[0].label,
    "Priest Healing Prevented",
  );
  assert.deepEqual(
    findMeasurements(topicIndex, "agent 3 rogue").map((row) => row.name),
    ["agent_3_ultimate_damage_done"],
  );
  assert.deepEqual(
    findMeasurements(topicIndex, "holy word salvation").map((row) => row.name),
    ["agent_4_ultimate_healing_done"],
  );
  for (const query of ["regen", "passive heal", "natural health rec"]) {
    assert.deepEqual(
      findMeasurements(topicIndex, query).map((row) => row.name),
      ["agent_4_regenerated_healing"],
    );
  }
  assert.deepEqual(findMeasurements(topicIndex, "timate"), []);
  assert.equal(
    findMeasurements(topicIndex, "agent_8_healing_prevented_by_poison")[0].name,
    "agent_8_healing_prevented_by_poison",
  );
  for (const query of ["by the", "about an", "is this", "a"]) {
    assert.deepEqual(findMeasurements(topicIndex, query), []);
  }

  const relationIndex = buildMetricSearchIndex([
    {
      name: "agent_2_damage_received_while_rogue_poison_slow",
      label: "Damage Received While Rogue Poison Slow",
      subject: "Agent ID 2 · Warrior · Team A",
      search_terms: ["damage received while slowed"],
      description: "Damage this agent took while already slowed by Poison.",
      order: 0,
    },
    ...[
      ["damage_done", "Damage Done", "Damage dealt by this agent."],
      ["healing_done", "Healing Done", "Healing provided by this agent."],
      ["healing_received", "Healing Received", "Healing this agent received."],
    ].map(([name, label, description], order) => ({
      name: `agent_2_${name}`,
      label,
      description,
      subject: "Agent ID 2 · Warrior · Team A",
      order: order + 1,
    })),
  ]);
  const poisonDamage = "agent_2_damage_received_while_rogue_poison_slow";
  for (const query of [
    "damage received while slowed rogue poison",
    "damage received while slowed by rogue poison",
    "the damage taken while slowed by the rogue poison",
    "incoming damage while slowed by rogue poison",
  ]) {
    assert.deepEqual(
      findMeasurements(relationIndex, query).map((row) => row.name),
      [poisonDamage],
    );
  }
  for (const [queries, name] of [
    [
      [
        "damage received",
        "damage taken",
        "damage tak",
        "damage suffered",
        "incoming dam",
      ],
      poisonDamage,
    ],
    [
      [
        "damage done",
        "damage dealt",
        "damage dea",
        "damage inflicted",
        "outgoing damage",
      ],
      "agent_2_damage_done",
    ],
    [
      ["healing received", "healing taken", "incoming healing"],
      "agent_2_healing_received",
    ],
    [
      ["healing done", "healing given", "healing provided", "outgoing heal"],
      "agent_2_healing_done",
    ],
  ]) {
    for (const query of queries) {
      assert.deepEqual(
        findMeasurements(relationIndex, query).map((row) => row.name),
        [name],
      );
    }
  }
  assert.equal(findMeasurements(relationIndex, "damage").length, 2);
  assert.equal(findMeasurements(relationIndex, "team a damage").length, 2);
  assert.equal(findMeasurements(relationIndex, "team b damage").length, 0);
  for (const query of [
    "damage without poison",
    "damage not received",
    "damage after poison",
    "healing from rogue",
  ]) {
    assert.deepEqual(findMeasurements(relationIndex, query), []);
  }
  for (const query of [
    "damage",
    "healing",
    "crippling poison",
    "regen",
    "holy word salvation",
  ]) {
    const expected = findMeasurements(topicIndex, query);
    for (const decorated of [
      `the ${query}`,
      `about the ${query}`,
      `the ${query} by this`,
    ]) {
      assert.deepEqual(findMeasurements(topicIndex, decorated), expected);
    }
  }

  const countIndex = buildMetricSearchIndex([
    {
      name: "agent_0_deaths_while_warrior_charge_slow",
      label: "Deaths While Warrior Charge Slow",
      subject: "Agent ID 0 · Mage · Team A",
      unit: "count",
      description: "Deaths while slowed by Charge.",
      order: 0,
    },
    {
      name: "agent_2_rescues",
      label: "Times Rescued",
      subject: "Agent ID 2 · Warrior · Team A",
      unit: "count",
      search_terms: ["death prevention"],
      description: "Times healing stopped this agent from dying.",
      order: 1,
    },
    {
      name: "agent_2_kill_contributions_per_death",
      label: "Kill Contributions per Death",
      subject: "Agent ID 2 · Warrior · Team A",
      unit: "ratio",
      description: "Kill contributions compared with deaths.",
      order: 2,
    },
    ...[
      [0, "Mage"],
      [2, "Warrior"],
      [7, "Warrior"],
    ].map(([slot, agentClass], order) => ({
      name: `agent_${slot}_deaths`,
      label: "Deaths",
      subject: `Agent ID ${slot} · ${agentClass} · Team ${Number(slot) < 5 ? "A" : "B"}`,
      unit: "count",
      search_terms: ["death count", "repeated deaths"],
      description: "How many times this agent has died.",
      order: order + 3,
    })),
    {
      name: "agent_4_kill_contributions",
      label: "Kill Contributions",
      subject: "Agent ID 4 · Priest · Team A",
      unit: "count",
      description: "Kills this Priest helped through useful healing.",
      order: 6,
    },
    {
      name: "agent_2_deaths_while_hunter_basic_slow",
      label: "Deaths While Hunter Basic Slow",
      subject: "Agent ID 2 · Warrior · Team A",
      unit: "count",
      description: "Deaths while slowed by a Hunter Basic attack.",
      order: 7,
    },
  ]);
  for (const query of ["warrior deaths", "warrior death", "warrior death counts"]) {
    const matches = findMeasurements(countIndex, query).map((row) => row.name);
    assert.deepEqual(matches.slice(0, 2), ["agent_2_deaths", "agent_7_deaths"]);
    assert(matches.includes("agent_2_deaths_while_hunter_basic_slow"));
    assert(!matches.includes("agent_0_deaths_while_warrior_charge_slow"));
  }
  assert.equal(
    findMeasurements(countIndex, "warrior deaths").some(
      (row) => row.name === "agent_2_rescues" || row.name === "agent_0_deaths",
    ),
    false,
  );
  assert.equal(
    findMeasurements(countIndex, "warrior death counts").some(
      (row) => row.name === "agent_2_kill_contributions_per_death",
    ),
    false,
  );
  assert.equal(
    findMeasurements(countIndex, "agent_0_deaths_while_warrior_charge_slow")[0].name,
    "agent_0_deaths_while_warrior_charge_slow",
  );
  for (const query of ["priest kill counts", "priest kill contribution counts"]) {
    assert.deepEqual(
      findMeasurements(countIndex, query).map((row) => row.name),
      ["agent_4_kill_contributions"],
    );
  }

  // Topic context can show a total beside an ability without giving that ability
  // credit for the total. These facts also keep the status on its actual owner.
  const attributionIndex = buildIndex(
    [
      ...["win", "draw", "loss"].flatMap((outcome) =>
        [1, 2].map((team) => ({
          name: `team_${team === 1 ? "a" : "b"}_${outcome}`,
          label: metricLabel(outcome),
          unit: "indicator",
          scope: "team",
          subjects: [team],
          search_facts: facts("outcome", "team", {
            source: "none",
            qualifiers: [outcome],
          }),
        })),
      ),
      {
        name: "team_a_return",
        label: "Team Return",
        scope: "team",
        subjects: [1],
        search_facts: facts("return", "team", { source: "none" }),
      },
      ...[
        { kind: "damage", stem: "damage_done", relation: "enemy" },
        { kind: "healing", stem: "healing_done", relation: "ally" },
      ].flatMap(({ kind, stem, relation }) => [
        {
          name: `team_a_${stem}`,
          label: metricLabel(stem),
          unit: "health",
          scope: "team",
          subjects: [1],
          search_facts: facts(kind, "source", { source: "team", relation }),
        },
        {
          name: `agent_4_to_agent_0_${stem}_allocation_fraction`,
          label: `${metricLabel(stem)} Allocation`,
          unit: "fraction",
          scope: "source_recipient",
          subjects: [4, 0],
          search_facts: facts(kind, "source", {
            relation,
            qualifiers: ["allocation", "fraction"],
          }),
        },
      ]),
      ...[
        [0, 9],
        [4, 5],
      ].map(([source, recipient]) => ({
        name: `agent_${source}_to_agent_${recipient}_damage_done`,
        label: "Damage Done to Recipient",
        unit: "health",
        scope: "source_recipient",
        subjects: [source, recipient],
        search_facts: facts("damage", "source", { relation: "enemy" }),
      })),
      {
        name: "team_a_kills",
        label: "Total Kills",
        scope: "team",
        subjects: [1],
        locations: [{ topic: "ultimate_warrior" }],
        search_facts: facts("kill", "source", { source: "team", relation: "enemy" }),
      },
      {
        name: "team_a_warrior_charge_slow_applications",
        label: "Warrior Charge Slow Applications",
        scope: "team",
        subjects: [1],
        search_facts: facts("status_application", "source", {
          source: "team",
          source_class_id: 2,
          ability: "ultimate",
          status: "warrior_charge_slow",
          status_subject: "recipient",
        }),
      },
      {
        name: "team_a_warrior_charge_applications",
        label: "Warrior Charge Applications",
        scope: "team",
        subjects: [1],
        locations: [{ topic: "ultimate_warrior" }],
        search_facts: facts("activation", "source", {
          source: "team",
          source_class_id: 2,
          ability: "ultimate",
        }),
      },
      {
        name: "team_a_damage_to_warrior_charge_slow_recipient",
        label: "Damage to Enemies Slowed by Warrior Charge",
        unit: "health",
        scope: "team",
        subjects: [1],
        search_facts: facts("damage", "source", {
          source: "team",
          relation: "enemy",
          status: "warrior_charge_slow",
          status_subject: "recipient",
        }),
      },
      {
        name: "agent_6_to_agent_0_ultimate_applications",
        label: "Ultimate Ability Activations on Recipient",
        scope: "source_recipient",
        subjects: [6, 0],
        locations: [{ topic: "ultimate_warrior" }],
        search_facts: facts("activation", "source", {
          ability: "ultimate",
          source_class_id: 2,
          relation: "enemy",
        }),
      },
      {
        name: "team_a_warrior_ultimate_kills",
        label: "Kills Helped by Charge",
        scope: "team",
        subjects: [1],
        locations: [{ topic: "ultimate_warrior" }],
        search_facts: facts("kill", "source", {
          source: "team",
          relation: "enemy",
          ability: "ultimate",
          source_class_id: 2,
        }),
      },
      {
        name: "agent_0_burst_damage_received",
        label: "Damage Received From Mages With Burst",
        scope: "agent",
        subjects: [0],
        locations: [{ topic: "ultimate_mage" }],
        search_facts: facts("damage", "recipient", {
          source: "class",
          source_class_id: 1,
          relation: "enemy",
          ability: "burst",
          status: "mage_burst",
          status_subject: "source",
        }),
      },
      {
        name: "agent_0_rescue_opportunities",
        label: "Chances to Save This Agent",
        scope: "agent",
        subjects: [0],
        locations: [{ topic: "ultimate_priest" }],
        search_facts: facts("rescue", "recipient", {
          source: "class",
          source_class_id: 5,
          relation: "ally",
          qualifiers: ["priest", "opportunity"],
        }),
      },
      {
        name: "agent_0_rescues",
        label: "Times Rescued",
        scope: "agent",
        subjects: [0],
        locations: [{ topic: "ultimate_priest" }],
        search_facts: facts("rescue", "recipient", {
          source: "class",
          source_class_id: 5,
          relation: "ally",
          qualifiers: ["priest"],
        }),
      },
      {
        name: "agent_4_to_agent_0_ultimate_rescue_contributions",
        label: "Ultimate Rescue Contributions to Recipient",
        scope: "source_recipient",
        subjects: [4, 0],
        locations: [{ topic: "ultimate_priest" }],
        search_facts: facts("rescue", "source", {
          ability: "ultimate",
          source_class_id: 5,
          relation: "ally",
          qualifiers: ["priest", "contribution"],
        }),
      },
      {
        name: "agent_4_basic_rescue_participation",
        label: "Share of Saves Helped by Basic Healing",
        unit: "fraction",
        scope: "agent",
        subjects: [4],
        search_facts: facts("rescue", "source", {
          ability: "basic",
          source_class_id: 5,
          relation: "ally",
          qualifiers: ["priest", "participation", "fraction"],
        }),
      },
      {
        name: "agent_4_basic_rescue_contributions",
        label: "Basic Rescue Contributions",
        scope: "agent",
        subjects: [4],
        search_facts: facts("rescue", "source", {
          ability: "basic",
          source_class_id: 5,
          relation: "ally",
          qualifiers: ["priest", "contribution"],
        }),
      },
    ].map((row, order) => ({
      applicable: true,
      unit: "count",
      locations: [],
      ...row,
      order,
    })),
    [],
    testAgents,
    classNames,
  );
  /** @type {[string, string[]][]} */
  const attributionQueries = [
    [
      "warrior charge count",
      [
        "team_a_warrior_charge_applications",
        "agent_6_to_agent_0_ultimate_applications",
      ],
    ],
    ["team a warrior charge count", ["team_a_warrior_charge_applications"]],
    ["team b warrior charge count", ["agent_6_to_agent_0_ultimate_applications"]],
    [
      "team a warrior charge",
      [
        "team_a_warrior_charge_applications",
        "team_a_warrior_ultimate_kills",
        "team_a_kills",
      ],
    ],
    ["warrior charge slow count", ["team_a_warrior_charge_slow_applications"]],
    ["warrior charge slow applications", ["team_a_warrior_charge_slow_applications"]],
    ["warrior charge kills", ["team_a_warrior_ultimate_kills"]],
    ["damage received by mage while burst", []],
    ["damage taken while burst is active", []],
    ["mage rescue opportunities", ["agent_0_rescue_opportunities"]],
    [
      "priest salvation saves to mage",
      ["agent_4_to_agent_0_ultimate_rescue_contributions"],
    ],
    ["priest salvation rescue opportunities", []],
    ["rescue opportunities from priest to mage", []],
    ["priest basic save share", ["agent_4_basic_rescue_participation"]],
    ["agent 4 basic rescue share", ["agent_4_basic_rescue_participation"]],
    ["win", ["team_a_win", "team_b_win"]],
    ["draw", ["team_a_draw", "team_b_draw"]],
    ["loss", ["team_a_loss", "team_b_loss"]],
    ["draw for team a", ["team_a_draw"]],
    ["loss by team b", ["team_b_loss"]],
    ["reward for team a", ["team_a_return"]],
    ["damage from mage and priest", []],
    ["damage to mage and priest", []],
  ];
  for (const [query, expected] of attributionQueries) {
    assert.deepEqual(
      findMeasurements(attributionIndex, query).map((row) => row.name),
      expected,
      query,
    );
  }
  for (const query of ["damage", "healing", "kills"]) {
    const ordinary = findMeasurements(attributionIndex, query);
    assert(ordinary.some((row) => row.scope === "team" && row.unit !== "fraction"));
    assert.deepEqual(findMeasurements(attributionIndex, `all ${query}`), ordinary);
  }
});

test("metric tooltips keep identities and explain both parts of a fraction without repeated help", () => {
  const common = {
    name: "a_name_that_cannot_supply_identities",
    unit: "count",
    direction: "descriptive",
    guidance: "Context dependent for Team A.",
    missing_when: "Not collected.",
  };
  /** @param {Record<string, any>} row */
  const values = (row) =>
    Object.fromEntries(
      metricTooltipRows(row).map(({ label, value }) => [label, value]),
    );
  const source = values({
    ...common,
    scope: "source_recipient",
    subjects: [5, 1],
    subject_role: "source",
    recipient_role: "recipient",
    numerator: "How many times Agent ID 5 used its Basic ability on Agent ID 1.",
    denominator: "How many times Agent ID 5 used its Basic ability on any enemy.",
  });
  assert.equal(source.From, "Agent ID 5");
  assert.equal(source.To, "Agent ID 1");
  assert.equal(source["CSV Column"], common.name);
  assert.equal(
    source.Numerator,
    "How many times Agent ID 5 used its Basic ability on Agent ID 1.",
  );
  assert.equal(
    source.Denominator,
    "How many times Agent ID 5 used its Basic ability on any enemy.",
  );
  const contribution = values({
    ...common,
    numerator: source.Numerator,
    denominator: "How many times Team B used Basic abilities on Agent ID 1.",
  });
  assert.equal(contribution.Numerator, source.Numerator);
  assert.equal(
    contribution.Denominator,
    "How many times Team B used Basic abilities on Agent ID 1.",
  );
  assert.notEqual(contribution.Denominator, source.Denominator);

  const team = values({
    ...common,
    scope: "team_recipient",
    subjects: [2, 1],
    subject_role: "recipient",
    recipient_role: "recipient",
  });
  assert.equal(team.From, "Team B");
  assert.equal(team.To, "Agent ID 1");

  const recipient = values({
    ...common,
    scope: "agent",
    subjects: [1],
    subject_role: "recipient",
    recipient_role: "recipient",
  });
  assert.equal(recipient.From, undefined);
  assert.equal(recipient.To, undefined);

  const allies = values({
    ...common,
    scope: "ally_pair",
    subjects: [0, 1],
    subject_role: "ally_pair",
  });
  assert.equal(allies.From, undefined);
  assert.equal(allies.To, undefined);
  assert.equal(allies["How to Read It"], "Context dependent for Team A.");
  assert.equal(
    values({ ...common, guidance: "Lower is better for Team B." })["How to Read It"],
    "Lower is better for Team B.",
  );

  for (const role of ["source", "recipient", "emitter"]) {
    const total = values({
      ...common,
      scope: "team",
      subjects: [1],
      subject_role: role,
    });
    assert.deepEqual(Object.keys(total), [
      "CSV Column",
      "Unit",
      "How to Read It",
      "Blank When",
    ]);
  }
  for (const row of [source, contribution, team, recipient, allies]) {
    for (const removed of [
      "About",
      "Who Acts",
      "Who Is Affected",
      "Who This Describes",
      "Divided By",
    ]) {
      assert.equal(row[removed], undefined);
    }
  }
  assert.equal(values({ ...common, unit: "fraction" }).Unit, "Share: 1 means 100%");
  assert.equal(values(common)["Blank When"], "Not collected.");
});
