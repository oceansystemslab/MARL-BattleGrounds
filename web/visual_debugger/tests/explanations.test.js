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
  explainRange,
  explainStatus,
  explainVisibility,
} from "../src/explanations.js";

test("authorized scene facts produce concise two-decimal explanations", () => {
  const agent = {
    global_slot: 2,
    class_id: 3,
    team_id: 1,
    current_health: 87.654321,
    max_health: 100,
    effective_speed: 0.333333,
    ultimate_cooldown: 3,
  };
  assert.deepEqual(explainAgent(agent, { controlled: true }), {
    kind: "agent",
    id: "agent:2",
    title: "id_2 · Hunter",
    details: [
      "Team A",
      "Health 87.65 / 100",
      "Effective speed 0.33",
      "Ultimate cooldown 3 ticks",
      "Controlled actor",
    ],
    anchor: "element",
  });
  assert.equal(explainCooldown(agent).details[0], "3 ticks remaining");
  assert.equal(
    explainModifier(
      {
        token_id: "warrior_mitigation",
        multiplier: 0.812345,
      },
      2,
    ).details[1],
    "Multiplier ×0.81",
  );
  assert.equal(
    explainAura({
      source_global_slot: 2,
      token_id: "mage_amplification",
      radius: 4.56789,
    }).details[1],
    "Public field radius 4.57",
  );
  assert.equal(
    explainRange({ global_slot: 2, kind: "ultimate", radius: Math.PI }).details[0],
    "Radius 3.14",
  );
  assert.deepEqual(
    explainStatus(
      {
        token_id: "slow_hunter_basic",
        source_class_id: 3,
        duration: 2,
      },
      5,
    ).details,
    ["Hunter Basic slow", "Source class Hunter", "Duration 2 ticks", "Recipient id_5"],
  );
  assert.equal(
    explainObstacle({
      obstacle_id: "wall-a",
      kind: "wall",
      width: 1.2345,
      height: 6,
    }).details.join(" · "),
    "Kind Wall · Width 1.23 · Height 6",
  );
});

test("legality and pending explanations report exact returned booleans", () => {
  assert.deepEqual(
    explainLegality(
      {
        target_global_slot: 7,
        lane_0_available: false,
        lane_1_available: true,
        armed_lane: 0,
        armed_pair_legal: false,
      },
      0,
    ).details,
    [
      "Exact Python mask value false",
      "Currently armed",
      "Staged pair legal false",
      "Target id_7",
    ],
  );
  assert.deepEqual(
    explainPendingRoute({
      source_global_slot: 2,
      target_global_slot: 7,
      lane: 1,
      legal: false,
    }).details,
    ["Source id_2", "Target id_7", "Ultimate lane", "Exact staged pair legal false"],
  );
});

test("activation explanations never reconstruct a redacted endpoint or amount", () => {
  const explanation = explainActivation({
    eventId: "pov-redacted",
    tokenId: "rogue_poison",
    sourceSlot: 3,
    targetSlot: null,
    targetDisclosure: "redacted",
    source: { x: 10, y: 10 },
    target: null,
  });
  assert.deepEqual(explanation.details, [
    "Accepted Rogue Poison activation",
    "Source id_3",
    "Target endpoint not disclosed in this view",
    "No per-source health amount is available",
  ]);
  const serialized = JSON.stringify(explanation);
  assert.equal(serialized.includes("target_anchor"), false);
  assert.equal(serialized.includes('"amount":'), false);
  assert.equal(serialized.includes('"x":10'), false);
});

test("recipient NET explanation is the only numeric health attribution", () => {
  const explanation = explainNetHealth({
    eventId: "net-1",
    recipientSlot: 5,
    netDelta: -12.34567,
    outcome: "damage",
  });
  assert.deepEqual(explanation.details, [
    "Recipient id_5",
    "NET -12.35",
    "Recipient-level before/after outcome; not source attribution",
  ]);
});

test("tiny recipient NET values preserve direction without signed zero", () => {
  const explanation = explainNetHealth({
    eventId: "net-tiny",
    recipientSlot: 5,
    netDelta: -0.004,
    outcome: "damage",
  });
  assert.equal(explanation.details[1], "NET −<0.01");
});

test("overflow names every hidden item without inheriting a class identity", () => {
  const explanation = explainOverflow(
    [
      {
        token_id: "slow_hunter_basic",
        source_class_id: 3,
        duration: 2,
      },
      {
        token_id: "stun_rogue_poison",
        source_class_id: 4,
        duration: 1,
      },
    ],
    "status",
    5,
  );

  assert.equal(explanation.kind, "status-overflow");
  assert.equal(explanation.title, "id_5 · 2 hidden statuses");
  assert.deepEqual(explanation.details, [
    "Hunter slow · Hunter Basic slow · source Hunter · 2 ticks",
    "Poison stun · Rogue Poison stun · source Rogue · 1 tick",
  ]);
});

test("visibility explanation labels privileged truth without deriving it", () => {
  assert.deepEqual(
    explainVisibility({
      observer_global_slot: 1,
      candidate_global_slot: 6,
      visible: false,
    }).details,
    [
      "Observer id_1",
      "Candidate id_6",
      "Candidate hidden",
      "Privileged researcher diagnostic",
    ],
  );
});
