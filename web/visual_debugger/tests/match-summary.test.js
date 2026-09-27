/**
 * @file Check captured team names, scores and outcomes, including missing summaries
 * and wrong-episode facts.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { authorizedPresentationTechnicalFacts } from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { matchSummaryView } from "../src/match-summary.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

test("match summary displays captured names, configured scores and per-team results", async () => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const summary = raw.match_summary;
  summary.task_mode = 1;
  summary.score_threshold = 7;
  summary.scores = [6, 2];
  summary.outcome = "in_progress";
  summary.teams[0].display_name = "learned-model-a";
  summary.teams[1].display_name = "learned-model-b";
  const view = matchSummaryView(await normalizeAuthorizedPresentationFrameV1(raw));
  assert.ok(view);
  assert.equal(view.task, "TDM");
  assert.deepEqual(
    view.teams.map((/** @type {Record<string, any>} */ team) => [
      team.name,
      team.score,
      team.result,
    ]),
    [
      ["learned-model-a", "6/7", null],
      ["learned-model-b", "2/7", null],
    ],
  );
  for (const [outcome, results] of [
    ["team_a_win", ["VICTORY", "DEFEAT"]],
    ["team_b_win", ["DEFEAT", "VICTORY"]],
    ["draw", ["DRAW", "DRAW"]],
  ]) {
    summary.outcome = outcome;
    const result = matchSummaryView(await normalizeAuthorizedPresentationFrameV1(raw));
    assert.ok(result);
    assert.deepEqual(
      result.teams.map((/** @type {Record<string, any>} */ team) => team.result),
      results,
    );
  }
  assert.equal(matchSummaryView(raw), null);
});

test("legacy diagnostic labels and missing compatibility summaries never imply TDM", async () => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const view = matchSummaryView(await normalizeAuthorizedPresentationFrameV1(raw));
  assert.ok(view);
  assert.equal(view.task, "Combat diagnostic");
  assert.ok(
    view.teams.every(
      (/** @type {Record<string, any>} */ team) =>
        team.score === null && team.result === null,
    ),
  );
  raw.match_summary = null;
  assert.equal(
    matchSummaryView(await normalizeAuthorizedPresentationFrameV1(raw)),
    null,
  );
});

test("match summary rejects facts from another episode, frame, or tick", async () => {
  for (const [field, value] of [
    ["episode_id", "other"],
    ["source_frame_index", 999],
    ["simulator_step_count", 999],
  ]) {
    const raw = structuredClone(fixture.presentations.replay_oracle);
    raw.match_summary[field] = value;
    await assert.rejects(
      () => normalizeAuthorizedPresentationFrameV1(raw),
      /match summary/iu,
    );
  }
});

test("all six production presentation pairs retain the same source-bound match envelope", async () => {
  for (const pair of Object.values(fixture.pairs)) {
    const frame = await normalizeAuthorizedPresentationFrameV1(pair.presentation);
    assert.ok(frame.match_summary);
    assert.equal(frame.match_summary.episode_id, frame.source.episode_id);
    assert.equal(
      frame.match_summary.source_frame_index,
      frame.source.source_frame_index,
    );
    assert.equal(
      frame.match_summary.simulator_step_count,
      frame.source.source_simulator_step_count,
    );
    assert.ok(matchSummaryView(frame));
    const facts = new Map(
      authorizedPresentationTechnicalFacts(frame).map((row) => [row.id, row.value]),
    );
    assert.equal(facts.get("episode"), frame.source.episode_id);
    assert.equal(facts.get("task_mode"), "Combat diagnostic");
    assert.equal(
      facts.get("observation_mode"),
      frame.match_summary.observation_mode === "shared_obs"
        ? "SharedObs"
        : "NoSharedObs",
    );
    assert.equal(
      facts.get("episode_limit"),
      `${frame.match_summary.episode_limit} ticks`,
    );
    assert.equal(facts.get("map"), frame.match_summary.map.technical_name);
    assert.equal(
      facts.get("seeds"),
      `Root ${frame.match_summary.root_seed} · Episode stream ${frame.match_summary.episode_seed}`,
    );
  }
});

test("friendly map display and technical identity retain the explicitly recorded split", async () => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  raw.match_summary.map = {
    map_id: 48,
    technical_name: "tdm_map_id_48_three_body_problem_test",
    display_name: "Three Body Problem",
    split: "test",
  };
  raw.match_summary.root_seed = null;
  raw.match_summary.episode_seed = null;
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  assert.equal(matchSummaryView(frame)?.map, "Three Body Problem (Test Map)");
  const facts = new Map(
    authorizedPresentationTechnicalFacts(frame).map((row) => [row.id, row.value]),
  );
  assert.equal(facts.get("map"), "tdm_map_id_48_three_body_problem_test");
  assert.equal(facts.get("seeds"), "Root unknown · Episode stream unknown");
  raw.match_summary.map.split = null;
  assert.equal(
    matchSummaryView(await normalizeAuthorizedPresentationFrameV1(raw))?.map,
    "Three Body Problem",
  );
});

test("recorded spawn sides change layout without swapping team scores or results", async () => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  raw.match_summary.teams[0].display_side = "right";
  raw.match_summary.teams[1].display_side = "left";
  raw.match_summary.task_mode = 1;
  raw.match_summary.score_threshold = 7;
  raw.match_summary.scores = [6, 2];
  raw.match_summary.outcome = "team_a_win";
  const view = matchSummaryView(await normalizeAuthorizedPresentationFrameV1(raw));
  assert.ok(view);
  assert.deepEqual(
    view.teams.map((/** @type {Record<string, any>} */ team) => [
      team.teamId,
      team.side,
      team.score,
      team.result,
    ]),
    [
      [1, "right", "6/7", "VICTORY"],
      [2, "left", "2/7", "DEFEAT"],
    ],
  );
  delete raw.match_summary.teams[0].display_side;
  delete raw.match_summary.teams[1].display_side;
  const legacy = matchSummaryView(await normalizeAuthorizedPresentationFrameV1(raw));
  assert.deepEqual(
    legacy?.teams.map((/** @type {Record<string, any>} */ team) => team.side),
    ["left", "right"],
  );
  raw.match_summary.teams[0].display_side = "right";
  await assert.rejects(
    () => normalizeAuthorizedPresentationFrameV1(raw),
    /display sides/u,
  );
  raw.match_summary.teams[1].display_side = "right";
  await assert.rejects(
    () => normalizeAuthorizedPresentationFrameV1(raw),
    /display sides/u,
  );
});
