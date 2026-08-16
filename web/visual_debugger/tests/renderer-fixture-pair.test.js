import assert from "node:assert/strict";
import test from "node:test";

import {
  loadRendererFixture,
  syntheticFixturePresentationPair,
} from "../e2e/support/renderer-fixture.js";
import { extractFrame, extractReplayTimeline } from "../src/api.js";

const EVENT_V2_KINDS = new Set([
  "action_rejected",
  "ability_activated",
  "source_damage_output",
  "source_healing_output",
  "recipient_health_resolution",
  "combat_countdown_reset",
  "health_regenerated",
  "cooldown_started",
  "cooldown_ready",
  "charge_phase_displacement",
  "ordinary_movement_phase_displacement",
  "agent_died",
  "lethal_damage_contribution",
  "status_aged_to_zero",
  "status_broken_by_damage",
  "status_applied",
  "status_refreshed_or_extended",
  "status_cleared_by_new_death",
  "spawn_shield_expired",
  "respawn_wave_occurred",
  "agent_respawned",
  "team_deathmatch_score_changed",
  "team_deathmatch_completed",
]);

const POV_CUE_KINDS = new Set([
  "own_action_outcome",
  "own_position_changed",
  "own_health_changed",
  "own_status_changed",
  "own_cooldown_changed",
  "own_lifecycle_changed",
  "visible_body_observation_changed",
  "episode_ended",
]);

/** @type {Promise<Record<string, Record<string, any>>> | undefined} */
let fixturesPromise;

async function exhaustiveFixtures() {
  fixturesPromise ??= Promise.all(
    [
      "canonical_event_vocabulary",
      "visual_vocabulary",
      "crowded_teamfight",
      "mixed_net_zero",
      "pov_redaction",
    ].map(async (name) => [name, await loadRendererFixture(name)]),
  ).then(Object.fromEntries);
  return fixturesPromise;
}

test("fixture pairs cross exact live and loaded-replay browser boundaries", async () => {
  const fixtures = await exhaustiveFixtures();
  for (const [name, fixture] of Object.entries(fixtures)) {
    const pair = syntheticFixturePresentationPair(fixture);
    const live = extractFrame(pair.liveFrame);
    const replay = extractFrame(pair.replayFrame);
    const timeline = extractReplayTimeline(pair.replayTimeline);

    assert.ok(live, `${name} live frame must normalize`);
    assert.ok(replay, `${name} replay frame must normalize`);
    assert.deepEqual(replay.projection, live.projection);
    assert.equal(replay.viewer_mode, "replay");
    assert.equal(replay.cursor.frame_index, 1);
    assert.equal(timeline.rows[1].simulator_step_count, replay.simulator_step_count);
    assert.equal(
      timeline.timeline_kind,
      pair.audience === "researcher" ? "researcher" : "actor_pov",
    );
    assert.equal(Object.isFrozen(live), true);
    assert.equal(Object.isFrozen(replay), true);
    assert.equal(Object.isFrozen(timeline), true);
  }
});

test("researcher fixture pairs own the exhaustive accepted vocabulary", async () => {
  const fixtures = await exhaustiveFixtures();
  const grammar = syntheticFixturePresentationPair(fixtures.canonical_event_vocabulary);
  const vocabulary = syntheticFixturePresentationPair(fixtures.visual_vocabulary);
  const crowded = syntheticFixturePresentationPair(fixtures.crowded_teamfight);

  for (const pair of [grammar, vocabulary, crowded]) {
    const live = extractFrame(pair.liveFrame);
    const replay = extractFrame(pair.replayFrame);
    assert.ok(live);
    assert.ok(replay);
    assert.deepEqual(replay.projection, live.projection);
  }

  const events = /** @type {Array<Record<string, any>>} */ (
    grammar.liveFrame.projection.incoming_events.events
  );
  assert.equal(events.length, 25);
  assert.deepEqual(new Set(events.map((event) => event.event_type)), EVENT_V2_KINDS);
  assert.equal(grammar.replayTimeline.rows[1].incoming_event_count, 25);

  const vocabularyScene = vocabulary.liveFrame.projection.scene;
  const vocabularyAgents = /** @type {Array<Record<string, any>>} */ (
    vocabularyScene.agents
  );
  const classMechanics = /** @type {Array<Record<string, any>>} */ (
    vocabularyScene.class_mechanics
  );
  const vocabularyRanges = /** @type {Array<Record<string, any>>} */ (
    vocabularyScene.ranges
  );
  const auraFields = /** @type {Array<Record<string, any>>} */ (
    vocabularyScene.aura_fields
  );
  assert.deepEqual(
    new Set(vocabularyAgents.map((agent) => agent.class_id)),
    new Set([1, 2, 3, 4, 5]),
  );
  assert.deepEqual(
    new Set(
      classMechanics.flatMap((mechanics) =>
        /** @type {Array<Record<string, any>>} */ (mechanics.status_mechanics).map(
          (status) => status.status_channel,
        ),
      ),
    ),
    new Set([0, 1, 2, 3, 4, 5, 6, 7, 8]),
  );
  assert.deepEqual(
    new Set(vocabularyRanges.map((range) => range.kind)),
    new Set(["observation", "basic", "ultimate"]),
  );
  assert.deepEqual(
    new Set(auraFields.map((field) => field.aura_id)),
    new Set(["mage_damage_amplification", "warrior_damage_mitigation"]),
  );
  assert.deepEqual(
    new Set(
      vocabularyAgents.flatMap((agent) =>
        /** @type {Array<Record<string, any>>} */ (agent.aura_modifiers).map(
          (modifier) => modifier.aura_id,
        ),
      ),
    ),
    new Set(["mage_damage_amplification", "warrior_damage_mitigation"]),
  );

  const crowdedScene = crowded.liveFrame.projection.scene;
  const crowdedObstacles = /** @type {Array<Record<string, any>>} */ (
    crowdedScene.map.obstacles
  );
  const crowdedAgents = /** @type {Array<Record<string, any>>} */ (crowdedScene.agents);
  assert.deepEqual(
    new Set(crowdedObstacles.map((obstacle) => obstacle.kind)),
    new Set(["pillar", "wall"]),
  );
  assert.deepEqual(
    new Set(
      crowdedAgents.flatMap((agent) =>
        /** @type {Array<Record<string, any>>} */ (agent.statuses).map(
          (status) => status.status_channel,
        ),
      ),
    ),
    new Set([0, 1, 2, 3, 4, 5, 6, 7, 8]),
  );
});

test("POV fixture pair owns every authorized cue kind without privileged events", async () => {
  const fixture = (await exhaustiveFixtures()).pov_redaction;
  const pair = syntheticFixturePresentationPair(fixture);
  const live = extractFrame(pair.liveFrame);
  const replay = extractFrame(pair.replayFrame);
  assert.ok(live);
  assert.ok(replay);

  const cues = /** @type {Array<Record<string, any>>} */ (
    pair.liveFrame.projection.incoming_cues
  );
  assert.equal(cues.length, 8);
  assert.deepEqual(new Set(cues.map((cue) => cue.cue_type)), POV_CUE_KINDS);
  assert.deepEqual(replay.projection, live.projection);
  assert.equal(pair.replayTimeline.rows[1].incoming_cue_count, 8);
  assert.equal(
    pair.replayFrame.artifact_summary.metric_report_availability,
    "not_available_in_actor_pov",
  );
  assert.equal(
    pair.replayFrame.processing_disclosure.disclosure,
    "not_available_in_actor_pov",
  );
  assert.equal(Object.hasOwn(pair.replayFrame.projection, "incoming_events"), false);
});
