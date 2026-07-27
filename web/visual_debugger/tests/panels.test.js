import assert from "node:assert/strict";
import test from "node:test";

import { pendingActionDisplayFacts, rosterStatusDurationLabel } from "../src/panels.js";

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
