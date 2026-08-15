import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  authorizedPresentationAudience,
  authorizedPresentationIdentityRows,
  authorizedPresentationIncomingRows,
  authorizedPresentationInspection,
  authorizedPresentationInspectionState,
  authorizedPresentationSceneView,
  authorizedPresentationTechnicalFacts,
  authorizedPresentationTransitionRows,
  authorizedOracleCommandSlotForPublicAgentId,
  authorizedOracleCommandSlotForPresentationKey,
  isAuthorizedPresentationFrame,
  scopedPresentationKey,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { buildChoreographyPlan } from "../src/choreography-plan.js";
import {
  explainChoreographyEvent,
  statusApplicationRoutes,
} from "../src/choreography-painter.js";
import {
  explainActivation,
  explainLegality,
  explainNetHealth,
  explainOverflow,
  explainPendingRoute,
  explainPovOverflow,
  explainPovStatus,
  explainSpawnShield,
} from "../src/explanations.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @param {keyof typeof fixture.presentations} kind */
async function normalized(kind) {
  return await normalizeAuthorizedPresentationFrameV1(fixture.presentations[kind]);
}

/** @param {keyof typeof fixture.pairs} kind */
async function normalizedPairPresentation(kind) {
  return await normalizeAuthorizedPresentationFrameV1(fixture.pairs[kind].presentation);
}

const surface = Object.freeze({
  /** @param {readonly [number, number] | {x: number, y: number}} point */
  worldToScreen: (point) => ({
    x: ("x" in point ? Number(point.x) : Number(point[0])) * 10,
    y: ("y" in point ? Number(point.y) : Number(point[1])) * 10,
  }),
  /** @param {number} length */
  worldLengthToScreen: (length) => length * 10,
  viewportBounds: Object.freeze({
    left: 0,
    top: 0,
    right: 640,
    bottom: 480,
    width: 640,
    height: 480,
  }),
  protectedRects: Object.freeze([]),
});

test("only unforgeably normalized roots enter the presentation adapter", async () => {
  const raw = fixture.presentations.replay_shared_obs_agent_pov;
  const frame = await normalized("replay_shared_obs_agent_pov");
  assert.equal(isAuthorizedPresentationFrame(raw), false);
  assert.equal(isAuthorizedPresentationFrame(Object.freeze({ ...frame })), false);
  assert.equal(isAuthorizedPresentationFrame(frame), true);
  assert.equal(authorizedPresentationAudience(frame), "agent_pov");
  assert.deepEqual(authorizedPresentationIdentityRows(raw), []);
});

test("opaque display identity is session-scoped and Agent rows gain no slot", async () => {
  const oracle = await normalized("replay_oracle");
  const shared = await normalized("replay_shared_obs_agent_pov");
  const oracleIdentity = authorizedPresentationIdentityRows(oracle).find(
    ({ public_agent_id }) => public_agent_id === "agent-slot-0",
  );
  const sharedIdentity = authorizedPresentationIdentityRows(shared).find(
    ({ public_agent_id }) => public_agent_id === "agent-slot-0",
  );
  assert.ok(oracleIdentity);
  assert.ok(sharedIdentity);
  assert.notEqual(oracleIdentity.display_key, sharedIdentity.display_key);
  assert.equal(sharedIdentity.command_global_slot, null);
  assert.equal(Object.hasOwn(sharedIdentity.agent, "global_slot"), false);
  assert.equal(
    scopedPresentationKey(shared, sharedIdentity.presentation_key),
    sharedIdentity.display_key,
  );
  assert.equal(Object.isFrozen(sharedIdentity), true);
});

test("only Oracle directory topology yields command transport slots", async () => {
  const oracle = await normalized("live_oracle");
  const rows = authorizedPresentationIdentityRows(oracle);
  assert.equal(authorizedPresentationAudience(oracle), "researcher");
  assert.deepEqual(
    rows.map(({ command_global_slot }) => command_global_slot),
    [0, 1, 2, 5, 6],
  );
  assert.deepEqual(
    oracle.action_axis.target_actions
      .slice(1)
      .map((/** @type {Record<string, any>} */ target) =>
        authorizedOracleCommandSlotForPublicAgentId(
          oracle,
          target.target_public_agent_id,
        ),
      ),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  );
  assert.equal(
    authorizedOracleCommandSlotForPresentationKey(oracle, rows[3].presentation_key),
    5,
  );
  assert.equal(
    rows.every(({ agent }) => !Object.hasOwn(agent, "global_slot")),
    true,
  );

  for (const kind of [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    assert.equal(
      authorizedPresentationIdentityRows(await normalized(kind)).every(
        ({ command_global_slot }) => command_global_slot === null,
      ),
      true,
    );
    const agentFrame = await normalized(kind);
    assert.equal(
      authorizedOracleCommandSlotForPresentationKey(
        agentFrame,
        authorizedPresentationIdentityRows(agentFrame)[0].presentation_key,
      ),
      null,
    );
    assert.equal(
      authorizedOracleCommandSlotForPublicAgentId(agentFrame, "agent-slot-0"),
      null,
    );
  }
});

test("scene adapter preserves exact mask and inspection-owned settled overlays", async () => {
  for (const kind of [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const frame = await normalized(kind);
    const inspection = authorizedPresentationInspection(frame);
    const scene = authorizedPresentationSceneView(frame);
    assert.ok(inspection);
    assert.ok(scene);
    assert.equal(
      scene.agents.every(
        (/** @type {Record<string, any>} */ agent) =>
          !Object.hasOwn(agent, "global_slot"),
      ),
      true,
    );
    assert.equal(inspection.decision_mask.movement_action_mask.length, 9);
    assert.equal(inspection.decision_mask.target_action_mask.length, 11);
    assert.equal(inspection.decision_mask.use_ultimate_action_mask.length, 2);
    assert.deepEqual(
      inspection.decision_mask.target_use_ultimate_joint_mask.map(
        (/** @type {unknown[]} */ row) => row.length,
      ),
      Array(11).fill(2),
    );
    assert.deepEqual(
      [
        ...new Set(
          inspection.decision_mask.target_actions.map(
            (/** @type {Record<string, any>} */ target) => target.target_kind,
          ),
        ),
      ].sort(),
      ["axis_only_authorized_agent", "no_target", "visible_authorized_agent"],
    );
    assert.equal(scene.ranges.length, 2);
    assert.equal(
      scene.ranges.every(
        (/** @type {Record<string, any>} */ range) =>
          range.presentation_key === inspection.actor_presentation_key &&
          !Object.hasOwn(range, "global_slot"),
      ),
      true,
    );
    if (scene.pending_route !== null) {
      assert.equal(
        scene.pending_route.source_presentation_key,
        inspection.actor_presentation_key,
      );
    }
  }
});

test("Latest Events, Submitted/Accepted, Technical Frame, and inspection stay disjoint", async () => {
  const frame = await normalized("replay_shared_obs_agent_pov");
  const incoming = authorizedPresentationIncomingRows(frame);
  const transition = authorizedPresentationTransitionRows(frame);
  const technical = authorizedPresentationTechnicalFacts(frame);
  const inspection = authorizedPresentationInspection(frame);
  assert.ok(incoming.length > 0);
  assert.equal(
    incoming.every(({ vocabulary }) => vocabulary === "observation_delta"),
    true,
  );
  assert.equal(transition.length, 1);
  assert.equal(transition[0], frame.latest_transition.action_rows[0]);
  assert.equal(inspection, frame.inspection);
  assert.deepEqual(
    technical.map(({ label }) => label),
    Object.keys(frame.technical_frame),
  );
  assert.equal(
    incoming.some(({ payload }) => payload === inspection),
    false,
  );
});

test("live scope and replay inspection states remain exact", async () => {
  const liveOracle = authorizedPresentationInspectionState(
    await normalized("live_oracle"),
  );
  const liveAgent = authorizedPresentationInspectionState(
    await normalized("live_no_shared_obs_agent_pov"),
  );
  const replayFrame = await normalized("replay_oracle");
  const replay = authorizedPresentationInspectionState(replayFrame);
  const replayNoneFrame = await normalizedPairPresentation("replay_oracle");
  const replayNone = authorizedPresentationInspectionState(replayNoneFrame);
  assert.equal(liveOracle.state_kind, "live_editable");
  assert.equal(liveOracle.submission_scope, "joint_turn");
  assert.equal(liveOracle.inspection?.inspection_kind, "live_draft_action");
  assert.equal(liveAgent.submission_scope, "controlled_actor");
  assert.equal(replay.state_kind, "replay_outgoing");
  assert.equal(replay.submission_scope, null);
  assert.equal(replay.inspection?.combat_lane, replayFrame.inspection.combat_lane);
  assert.equal(replayNone.state_kind, "replay_none");
  assert.equal(replayNone.submission_scope, null);
  assert.equal(replayNone.inspection, null);
  assert.deepEqual(authorizedPresentationSceneView(replayNoneFrame)?.ranges, []);
});

test("Latest Events alone schedule opaque-key choreography", async () => {
  const oracle = await normalized("replay_oracle");
  const oraclePlan = buildChoreographyPlan(oracle, surface);
  assert.ok(oraclePlan);
  assert.deepEqual(
    oraclePlan.events.map(({ eventId }) => eventId),
    authorizedPresentationIncomingRows(oracle).map(({ id }) => id),
  );
  assert.equal(
    oraclePlan.events.some((event) =>
      Object.keys(event).some((key) => key === "global_slot" || key.endsWith("Slot")),
    ),
    false,
  );
  assert.equal(
    oraclePlan.events.some(
      (event) =>
        typeof event.sourcePresentationKey === "string" ||
        typeof event.recipientPresentationKey === "string" ||
        typeof event.agentPresentationKey === "string",
    ),
    true,
  );

  const shared = await normalized("replay_shared_obs_agent_pov");
  const sharedPlan = buildChoreographyPlan(shared, surface);
  assert.ok(sharedPlan);
  assert.equal(
    sharedPlan.events.length,
    authorizedPresentationIncomingRows(shared).length,
  );
  assert.equal(
    sharedPlan.events.every(
      (event) =>
        event.authorityVocabulary === "observation_delta" &&
        event.noncausal === true &&
        event.kind === "feed_only" &&
        event.spatial === false,
    ),
    true,
  );
  assert.equal(
    JSON.stringify(sharedPlan).includes(
      shared.inspection.transition_reference.transition_id,
    ),
    false,
  );

  const noShared = await normalized("replay_no_shared_obs_agent_pov");
  const noSharedPlan = buildChoreographyPlan(noShared, surface);
  assert.ok(noSharedPlan);
  assert.equal(
    noSharedPlan.events.every(
      (event) =>
        event.authorityVocabulary === "recipient_cue" && event.noncausal === true,
    ),
    true,
  );
});

test("painter metadata retains opaque keys without inventing slots", async () => {
  const oracle = await normalized("replay_oracle");
  const plan = buildChoreographyPlan(oracle, surface);
  const lifecycle = plan?.events.find(({ kind }) => kind === "status_lifecycle");
  assert.ok(lifecycle);
  const routes = statusApplicationRoutes(lifecycle);
  assert.equal(routes.length, 1);
  assert.equal(routes[0].sourcePresentationKey, lifecycle.sourcePresentationKey);
  assert.equal(routes[0].sourceSlot, null);

  const samePublicIdentity = {
    kind: "net_health",
    eventType: "recipient_health_resolution",
    recipientPublicAgentId: "agent-slot-0",
  };
  const firstIdentity = authorizedPresentationIdentityRows(oracle)[0];
  const shared = await normalized("replay_shared_obs_agent_pov");
  const secondIdentity = authorizedPresentationIdentityRows(shared)[0];
  assert.notEqual(
    explainChoreographyEvent({
      ...samePublicIdentity,
      recipientPresentationKey: firstIdentity.presentation_key,
    }).id,
    explainChoreographyEvent({
      ...samePublicIdentity,
      recipientPresentationKey: secondIdentity.presentation_key,
    }).id,
  );
});

test("tooltip owner identities cannot be reused across presentation sessions", async () => {
  const oracle = await normalized("replay_oracle");
  const shared = await normalized("replay_shared_obs_agent_pov");
  const first = authorizedPresentationIdentityRows(oracle)[0].agent;
  const second = authorizedPresentationIdentityRows(shared)[0].agent;
  const status = {
    token_id: "priest_freedom",
    status_feature_index: 0,
    duration: 1,
  };
  assert.equal(first.public_agent_id, second.public_agent_id);
  assert.notEqual(
    explainPovStatus(status, first).id,
    explainPovStatus(status, second).id,
  );
  assert.notEqual(
    explainPovOverflow([status], first).id,
    explainPovOverflow([status], second).id,
  );
  assert.notEqual(
    explainOverflow([status], "status", first).id,
    explainOverflow([status], "status", second).id,
  );
  assert.notEqual(explainSpawnShield(first).id, explainSpawnShield(second).id);
  assert.notEqual(
    explainLegality(
      { target_presentation_key: first.presentation_key, lane_0_available: true },
      0,
    ).id,
    explainLegality(
      { target_presentation_key: second.presentation_key, lane_0_available: true },
      0,
    ).id,
  );
  assert.notEqual(
    explainPendingRoute({
      source_presentation_key: first.presentation_key,
      target_presentation_key: first.presentation_key,
      lane: 0,
    }).id,
    explainPendingRoute({
      source_presentation_key: second.presentation_key,
      target_presentation_key: second.presentation_key,
      lane: 0,
    }).id,
  );
  assert.notEqual(
    explainActivation({
      tokenId: "basic_damage",
      sourcePresentationKey: first.presentation_key,
    }).id,
    explainActivation({
      tokenId: "basic_damage",
      sourcePresentationKey: second.presentation_key,
    }).id,
  );
  assert.notEqual(
    explainNetHealth({
      recipientPresentationKey: first.presentation_key,
      outcome: "damage",
    }).id,
    explainNetHealth({
      recipientPresentationKey: second.presentation_key,
      outcome: "damage",
    }).id,
  );
});
