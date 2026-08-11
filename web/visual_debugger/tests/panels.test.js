import assert from "node:assert/strict";
import test from "node:test";

import {
  actionTupleCombatLabel,
  eventSummary,
  pendingActionDisplayFacts,
  rosterStatusDurationLabel,
} from "../src/panels.js";

test("source-output summaries retain exact aura evidence without exposing slots", () => {
  const damage = eventSummary({
    event_type: "source_damage_output",
    source_global_slot: 3,
    raw_damage_output: 10,
    source_modified_damage_output: 12,
    recipient_damage_modifier: 0.8,
    mage_damage_aura_covering_emitter_global_slots: [0, 5],
    warrior_mitigation_aura_covering_emitter_global_slots: [1],
  });
  assert.match(damage, /Mage aura emitters 2/u);
  assert.match(damage, /Warrior mitigation emitters 1/u);
  assert.doesNotMatch(damage, /\[(?:0|1|5)/u);

  const healing = eventSummary({
    event_type: "source_healing_output",
    source_global_slot: 4,
    raw_healing_output: 10,
    source_modified_healing_output: 10,
    recipient_healing_modifier: 1,
  });
  assert.match(healing, /aura emitter evidence not recorded/u);
});

test("roster duration labels abbreviate only the human-facing extreme value", () => {
  assert.equal(rosterStatusDurationLabel(5), "5");
  assert.equal(rosterStatusDurationLabel(123456789), "123M");
  assert.equal(rosterStatusDurationLabel(null), "?");
});

test("pending no-combat copy hides transport vocabulary without dropping facts", () => {
  const facts = pendingActionDisplayFacts({
    move_action: 0,
    movement_mask_value: true,
    target_action: 0,
    target: { disclosure: "target_none", global_slot: null },
    armed_lane: 0,
    pair_mask_value: null,
  });

  assert.deepEqual(facts, {
    movement: "Movement · Stay (0) · Available",
    target: "Target · None",
    action: "Action · No combat",
    legality: "Legality · Not applicable",
  });
  assert.doesNotMatch(Object.values(facts).join(" "), /target-none|Lane 0\/B/u);
  assert.equal(Object.isFrozen(facts), true);
});

test("pending combat copy keeps exact target, lane shortcut, and legality", () => {
  assert.deepEqual(
    pendingActionDisplayFacts({
      move_action: 5,
      movement_mask_value: false,
      target_action: 3,
      target: { disclosure: "public", global_slot: 7 },
      armed_lane: 1,
      pair_mask_value: false,
    }),
    {
      movement: "Movement · Northeast (5) · Unavailable",
      target: "Target · id_7",
      action: "Action · Ultimate (1/U)",
      legality: "Legality · Unavailable",
    },
  );
});

test("pending target-none retains an explicitly armed source-local Ultimate", () => {
  assert.deepEqual(
    pendingActionDisplayFacts({
      move_action: 0,
      movement_mask_value: true,
      target: { target_action: 0, public_agent_id: null },
      armed_lane: 1,
      pair_mask_value: true,
    }),
    {
      movement: "Movement · Stay (0) · Available",
      target: "Target · None",
      action: "Action · Ultimate (1/U)",
      legality: "Legality · Available",
    },
  );
});

test("actor-POV action tuples recognize the recipient-local no-target action", () => {
  assert.equal(
    actionTupleCombatLabel({
      target: { target_action: 0, public_agent_id: null },
      use_ultimate_action: 0,
    }),
    "No combat",
  );
  assert.equal(
    actionTupleCombatLabel({
      target: { target_action: 3, public_agent_id: "visible-target" },
      use_ultimate_action: 0,
    }),
    "0/B · Basic",
  );
  assert.equal(
    actionTupleCombatLabel({
      target: { target_action: 3, public_agent_id: "visible-target" },
      use_ultimate_action: 1,
    }),
    "1/U · Ultimate",
  );
  assert.equal(
    actionTupleCombatLabel({
      target: { target_action: 0, public_agent_id: null },
      use_ultimate_action: 1,
    }),
    "1/U · Ultimate",
  );
});
