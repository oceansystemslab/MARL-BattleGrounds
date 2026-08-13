import assert from "node:assert/strict";
import test from "node:test";

import {
  actionTupleCombatLabel,
  disclosurePanelInitiallyOpen,
  eventDescriptor,
  eventSummary,
  panelDisclosureAuthorityKey,
  pendingActionDisplayFacts,
  publicAgentIdMap,
  replayDiagnosticFacts,
  rosterControlDescriptor,
  rosterStatusDurationLabel,
} from "../src/panels.js";
import { projectSemanticDescriptor, semanticDescriptorText } from "../src/tooltip.js";

const PUBLIC_IDS = new Map([
  [0, "zero:<opaque>"],
  [1, "one/agent&x"],
  [3, "three.300"],
  [4, "four four"],
  [5, "five#five"],
  [7, "seven:semicolon;"],
]);

test("native disclosure state resets only at scenario, artifact, or audience boundaries", () => {
  const live = {
    viewer_mode: "live",
    revision: 1,
    scenario: { name: "arena" },
    scene: { audience: "researcher", agents: [] },
  };
  assert.equal(
    panelDisclosureAuthorityKey({ ...live, revision: 99, transition_id: "later" }),
    panelDisclosureAuthorityKey(live),
  );
  assert.notEqual(
    panelDisclosureAuthorityKey({ ...live, scenario: { name: "other" } }),
    panelDisclosureAuthorityKey(live),
  );

  const replay = {
    viewer_mode: "replay",
    replay_audience: "researcher",
    cursor: { frame_index: 0 },
    artifact_summary: {
      replay_reference: {
        artifact_id: "artifact-a",
        canonical_digest_sha256: "digest-a",
      },
    },
    scene: { audience: "researcher", agents: [] },
  };
  assert.equal(
    panelDisclosureAuthorityKey({ ...replay, cursor: { frame_index: 200 } }),
    panelDisclosureAuthorityKey(replay),
  );
  assert.notEqual(
    panelDisclosureAuthorityKey({
      ...replay,
      replay_audience: "actor_pov",
      pov_global_slot: 0,
    }),
    panelDisclosureAuthorityKey(replay),
  );
  assert.notEqual(
    panelDisclosureAuthorityKey({
      ...replay,
      artifact_summary: {
        replay_reference: {
          artifact_id: "artifact-b",
          canonical_digest_sha256: "digest-b",
        },
      },
    }),
    panelDisclosureAuthorityKey(replay),
  );
  assert.equal(panelDisclosureAuthorityKey(null), null);
});

test("native disclosure defaults keep only operational live panels and roster open", () => {
  assert.equal(disclosurePanelInitiallyOpen("command-deck", false), true);
  assert.equal(disclosurePanelInitiallyOpen("command-deck", true), false);
  assert.equal(disclosurePanelInitiallyOpen("roster-details", false), true);
  assert.equal(disclosurePanelInitiallyOpen("roster-details", true), true);
  for (const id of [
    "agent-details",
    "pending-turn-details",
    "latest-transition-details",
    "events-details",
    "visual-key",
    "technical-frame-details",
  ]) {
    assert.equal(disclosurePanelInitiallyOpen(id, false), false);
    assert.equal(disclosurePanelInitiallyOpen(id, true), false);
  }
  assert.throws(() => disclosurePanelInitiallyOpen("unknown", false), /Unknown/u);
});

test("replay diagnostics display recorded movement scale only for researchers", () => {
  const base = {
    viewer_mode: "replay",
    frame_kind: "researcher_replay_viewer",
    timeline_id: "artifact:timeline:researcher",
    cursor: { cursor_generation: 3, choreography_generation: 4 },
    artifact_summary: { metric_report_availability: "available" },
  };
  const researcher = replayDiagnosticFacts({
    ...base,
    replay_audience: "researcher",
    recorded_ordinary_movement_distance_scale: 0.375,
  });
  const pov = replayDiagnosticFacts({
    ...base,
    replay_audience: "actor_pov",
  });

  assert.deepEqual(researcher.at(-1), {
    label: "Recorded movement scale",
    value: "0.38",
  });
  assert.equal(
    pov.some((fact) => fact.label === "Recorded movement scale"),
    false,
  );
  assert.equal(Object.isFrozen(researcher), true);
  assert.deepEqual(replayDiagnosticFacts({ viewer_mode: "live" }), []);
});

test("roster buttons own audience- and availability-specific control help", () => {
  /** @type {Array<["target" | "control", "live" | "researcher_replay", boolean, string, string, string]>} */
  const cases = [
    ["target", "live", false, "Target", "staged action", "Available"],
    ["control", "live", true, "Control", "currently unavailable", "Unavailable"],
    [
      "target",
      "researcher_replay",
      false,
      "Reference",
      "does not change the immutable range anchor",
      "Available",
    ],
    ["control", "researcher_replay", false, "POV actor", "point of view", "Available"],
  ];
  for (const [role, mode, disabled, title, copy, availability] of cases) {
    const descriptor = rosterControlDescriptor(
      role,
      "arbitrary:<agent>&7",
      mode,
      disabled,
    );
    assert.equal(descriptor.title, title);
    assert.match(descriptor.summary, new RegExp(copy, "u"));
    assert.equal(descriptor.rows[0].value, "Agent ID arbitrary:<agent>&7");
    assert.equal(descriptor.rows[1].value, availability);
    assert.equal(descriptor.metadata.full, false);
  }
  assert.throws(
    () => rosterControlDescriptor("target", "agent", "live", /** @type {never} */ (1)),
    /boolean/u,
  );
});

test("same-root public-ID map joins scene and batch and fails closed on conflicts", () => {
  const joined = publicAgentIdMap({
    scene: {
      agents: [
        { global_slot: 1, public_agent_id: "one/agent&x" },
        { global_slot: 7, public_agent_id: "seven:semicolon;" },
        { global_slot: "4", public_agent_id: "coerced-slot-must-not-join" },
      ],
    },
    event_batch: {
      public_agent_id_by_global_slot: ["zero:<opaque>", "one/agent&x", "batch-two"],
    },
  });
  assert.deepEqual(
    [...joined],
    [
      [1, "one/agent&x"],
      [7, "seven:semicolon;"],
      [0, "zero:<opaque>"],
      [2, "batch-two"],
    ],
  );
  assert.equal(joined.has(4), false);

  const conflict = publicAgentIdMap({
    scene: { agents: [{ global_slot: 1, public_agent_id: "scene-one" }] },
    event_batch: {
      public_agent_id_by_global_slot: ["zero", "different-one"],
    },
  });
  assert.equal(conflict.has(1), false);
  assert.equal(conflict.get(0), "zero");
});

test("all 21 researcher event kinds use arbitrary public IDs and never slot labels", () => {
  const events = [
    {
      event_type: "action_rejected",
      actor_global_slot: 1,
      rejection_component: "movement",
    },
    {
      event_type: "ability_activated",
      source_global_slot: 1,
      recipient_global_slot: 7,
      ability_component: "warrior_charge",
    },
    {
      event_type: "source_damage_output",
      source_global_slot: 1,
      raw_damage_output: 10,
      source_modified_damage_output: 12,
      recipient_damage_modifier: 0.8,
      mage_damage_aura_covering_emitter_global_slots: [0, 5],
      warrior_mitigation_aura_covering_emitter_global_slots: [3],
    },
    {
      event_type: "source_healing_output",
      source_global_slot: 4,
      raw_healing_output: 10,
      source_modified_healing_output: 11,
      recipient_healing_modifier: 1,
    },
    {
      event_type: "recipient_health_resolution",
      recipient_global_slot: 7,
      realized_net_health_change: -3,
      transition_start_health: 10,
      health_after_combat_resolution: 7,
    },
    { event_type: "combat_countdown_reset", agent_global_slot: 1 },
    {
      event_type: "health_regenerated",
      agent_global_slot: 3,
      actual_health_regenerated: 1.25,
    },
    { event_type: "cooldown_started", agent_global_slot: 4 },
    { event_type: "cooldown_ready", agent_global_slot: 5 },
    {
      event_type: "charge_phase_displacement",
      agent_global_slot: 1,
      start_anchor: { position: [1, 2] },
      end_anchor: { position: [3, 4] },
    },
    {
      event_type: "ordinary_movement_phase_displacement",
      agent_global_slot: 7,
      start_anchor: { position: [4, 5] },
      end_anchor: { position: [6, 7] },
    },
    { event_type: "agent_died", recipient_global_slot: 7 },
    {
      event_type: "lethal_damage_contribution",
      source_global_slot: 1,
      recipient_global_slot: 7,
      attributed_death_damage: 4.5,
    },
    {
      event_type: "status_aged_to_zero",
      recipient_global_slot: 7,
      status_id: "hunter_basic_slow",
    },
    {
      event_type: "status_broken_by_damage",
      recipient_global_slot: 7,
      status_id: "warrior_charge_stun",
    },
    {
      event_type: "status_applied",
      source_global_slot: 1,
      recipient_global_slot: 7,
      status_id: "rogue_poison_slow",
    },
    {
      event_type: "status_refreshed_or_extended",
      recipient_global_slot: 7,
      status_id: "mage_burst",
    },
    {
      event_type: "status_cleared_by_new_death",
      recipient_global_slot: 7,
      status_id: "priest_freedom",
    },
    { event_type: "spawn_shield_expired", agent_global_slot: 3 },
    { event_type: "respawn_wave_occurred", team_id: 2 },
    {
      event_type: "agent_respawned",
      agent_global_slot: 5,
      realized_successor_position: [8, 9],
    },
  ];
  assert.equal(events.length, 21);
  const summaries = events.map((event) => eventSummary(event, PUBLIC_IDS));
  assert.equal(
    summaries.every((summary) => !summary.includes("id_")),
    true,
  );
  assert.equal(
    summaries
      .filter((_, index) => index !== 19)
      .every((summary) => !summary.includes("Agent ID unavailable")),
    true,
  );
  assert.match(summaries[0], /Agent ID one\/agent&x/u);
  assert.match(summaries[1], /Agent ID seven:semicolon;/u);
  assert.match(summaries[2], /Sorcerer’s Empowerment emitters 2/u);
  assert.match(summaries[2], /Guardian’s Barrier emitters 1/u);
  assert.doesNotMatch(summaries[2], /zero:<opaque>|five#five|three\.300/u);
  assert.match(summaries[3], /aura emitter evidence not recorded/u);
});

test("missing event joins fail closed instead of deriving a public ID from a slot", () => {
  const summary = eventSummary(
    { event_type: "agent_died", recipient_global_slot: 999 },
    PUBLIC_IDS,
  );
  assert.equal(summary, "Agent died · Agent ID unavailable");
  assert.doesNotMatch(summary, /999|id_/u);
});

test("event semantic descriptor and accessible text never consume canonical event IDs", () => {
  const technicalEventId = "event:canonical:<private>&9001";
  const event = {
    event_id: technicalEventId,
    event_type: "status_applied",
    status_id: "rogue_poison_slow",
    source_global_slot: 1,
    recipient_global_slot: 7,
  };
  const descriptor = eventDescriptor(event, 4, PUBLIC_IDS);
  const serialized = JSON.stringify(descriptor);
  const compactAccessibleText = semanticDescriptorText(
    projectSemanticDescriptor(descriptor, "compact"),
  ).join(" ");
  const fullAccessibleText = semanticDescriptorText(
    projectSemanticDescriptor(descriptor, "full"),
  ).join(" ");

  assert.equal(descriptor.id, "event:status_applied:4");
  assert.doesNotMatch(serialized, /event:canonical:<private>&9001/u);
  assert.doesNotMatch(compactAccessibleText, /event:canonical:<private>&9001/u);
  assert.doesNotMatch(fullAccessibleText, /event:canonical:<private>&9001/u);
  assert.match(compactAccessibleText, /Agent ID one\/agent&x/u);
  assert.match(fullAccessibleText, /Agent ID seven:semicolon;/u);
  assert.throws(() => eventDescriptor(event, -1, PUBLIC_IDS), /non-negative integer/u);
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

test("pending combat copy requires the same-root resolver for its public target", () => {
  const pending = {
    move_action: 5,
    movement_mask_value: false,
    target_action: 3,
    target: { disclosure: "public", global_slot: 7, target_action: 3 },
    armed_lane: 1,
    pair_mask_value: false,
  };
  assert.deepEqual(pendingActionDisplayFacts(pending, PUBLIC_IDS), {
    movement: "Movement · Northeast (5) · Unavailable",
    target: "Target · Agent ID seven:semicolon; (action 3)",
    action: "Action · Ultimate (1/U)",
    legality: "Legality · Unavailable",
  });
  assert.equal(
    pendingActionDisplayFacts(pending).target,
    "Target · Agent ID unavailable",
  );
});

test("pending combat copy retains the envelope target action", () => {
  const facts = pendingActionDisplayFacts(
    {
      move_action: 3,
      movement_mask_value: true,
      target_action: 6,
      target: { disclosure: "public", global_slot: 6 },
      armed_lane: null,
      pair_mask_value: null,
    },
    new Map([[6, "opaque-six"]]),
  );
  assert.equal(facts.target, "Target · Agent ID opaque-six (action 6)");
  assert.doesNotMatch(facts.target, /unavailable|id_6/u);
});

test("pending public ID and slot must agree when both are supplied", () => {
  const facts = pendingActionDisplayFacts(
    {
      move_action: 0,
      movement_mask_value: true,
      target: {
        disclosure: "public",
        global_slot: 7,
        target_action: 3,
        public_agent_id: "mismatched-public-id",
      },
      armed_lane: 0,
      pair_mask_value: true,
    },
    PUBLIC_IDS,
  );
  assert.equal(facts.target, "Target · Agent ID unavailable");
  assert.doesNotMatch(JSON.stringify(facts), /mismatched-public-id|id_7/u);
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
