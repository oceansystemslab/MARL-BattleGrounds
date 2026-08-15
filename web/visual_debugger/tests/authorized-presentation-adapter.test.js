import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  authorizedOracleCommandSlotForPresentationKey,
  authorizedOracleCommandSlotForPublicAgentId,
  authorizedPresentationAudience,
  authorizedPresentationIdentityRows,
  authorizedPresentationIncomingRows,
  authorizedPresentationInspection,
  authorizedPresentationInspectionState,
  authorizedPresentationPreferenceKey,
  authorizedPresentationSceneView,
  authorizedPresentationTechnicalFacts,
  authorizedPresentationTransitionRows,
  isAuthorizedPresentationFrame,
  projectCertifiedInspectionLegality,
  projectCertifiedInspectionRoute,
  sameAuthorizedPresentationPreferenceKey,
  scopedPresentationKey,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  explainChoreographyEvent,
  statusApplicationRoutes,
} from "../src/choreography-painter.js";
import { buildChoreographyPlan } from "../src/choreography-plan.js";
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

/** @param {readonly unknown[]} tuple */
function preferenceKeyFromTuple(tuple) {
  const frozenTuple = Object.freeze([...tuple]);
  return Object.freeze({
    tuple: frozenTuple,
    serialized: JSON.stringify(frozenTuple),
  });
}

/** @param {unknown} value @param {Set<object>} [seen] */
function assertRecursivelyFrozen(value, seen = new Set()) {
  if (value === null || typeof value !== "object" || seen.has(value)) {
    return;
  }
  seen.add(value);
  assert.equal(Object.isFrozen(value), true);
  for (const child of Object.values(value)) {
    assertRecursivelyFrozen(child, seen);
  }
}

/** @param {unknown} value @param {string} forbiddenKey @returns {boolean} */
function containsOwnKey(value, forbiddenKey) {
  if (value === null || typeof value !== "object") {
    return false;
  }
  return (
    Object.hasOwn(value, forbiddenKey) ||
    Object.values(value).some((child) => containsOwnKey(child, forbiddenKey))
  );
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
  assert.equal(authorizedPresentationSceneView(raw, "raw-agent-key"), null);
  assert.equal(authorizedPresentationSceneView(raw, null), null);
});

test("preference authority keys project the exact frozen tuple for all five leaves", async () => {
  for (const [kind, raw] of Object.entries(fixture.presentations)) {
    const frame = await normalized(kind);
    const key = authorizedPresentationPreferenceKey(frame);
    assert.ok(key);
    const expected = [
      raw.product_kind,
      raw.source.source_session_id,
      raw.source.episode_id,
      raw.presentation_kind,
      raw.authority.authority_kind,
      raw.authority.observation_mode ?? null,
      raw.source.source_artifact_id ?? null,
      raw.authority.recipient_public_agent_id ?? null,
      raw.authority.recipient_presentation_key ?? null,
    ];
    assert.deepEqual(key.tuple, expected);
    assert.equal(key.serialized, JSON.stringify(expected));
    assert.equal(Object.isFrozen(key), true);
    assert.equal(Object.isFrozen(key.tuple), true);

    const second = authorizedPresentationPreferenceKey(await normalized(kind));
    assert.equal(sameAuthorizedPresentationPreferenceKey(key, second), true);
  }

  const raw = fixture.presentations.replay_shared_obs_agent_pov;
  const normalizedFrame = await normalized("replay_shared_obs_agent_pov");
  assert.equal(authorizedPresentationPreferenceKey(raw), null);
  assert.equal(authorizedPresentationPreferenceKey({ ...normalizedFrame }), null);
  assert.equal(authorizedPresentationPreferenceKey(null), null);
  assert.equal(sameAuthorizedPresentationPreferenceKey(null, null), false);
});

test("preference authority keys ignore revision, epoch, and replay cursor generations", async () => {
  for (const [kind, raw] of Object.entries(fixture.presentations)) {
    const before = await normalized(kind);
    const changed = structuredClone(raw);
    changed.source.source_revision += 17;
    changed.source.source_authority_epoch = changed.source.source_revision;
    if (changed.live_inspection) {
      changed.live_inspection.source_revision = changed.source.source_revision;
      changed.live_inspection.source_authority_epoch =
        changed.source.source_authority_epoch;
    }
    const after = await normalizeAuthorizedPresentationFrameV1(changed);
    assert.equal(
      sameAuthorizedPresentationPreferenceKey(
        authorizedPresentationPreferenceKey(before),
        authorizedPresentationPreferenceKey(after),
      ),
      true,
    );
  }

  const replayBefore = await normalized("replay_oracle");
  const replayChanged = structuredClone(fixture.presentations.replay_oracle);
  replayChanged.source.source_cursor_generation += 2;
  replayChanged.source.source_choreography_generation += 1;
  const replayAfter = await normalizeAuthorizedPresentationFrameV1(replayChanged);
  assert.equal(
    sameAuthorizedPresentationPreferenceKey(
      authorizedPresentationPreferenceKey(replayBefore),
      authorizedPresentationPreferenceKey(replayAfter),
    ),
    true,
  );
});

test("every preference authority tuple slot is a reset boundary", () => {
  const baseTuple = [
    "combat_debugger",
    "session-a",
    "episode:a",
    "live_no_shared_obs_agent_pov",
    "agent_pov",
    "no_shared_obs",
    null,
    "recipient:a",
    "pov_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  ];
  const replacements = [
    "replay_viewer",
    "session-b",
    "episode:b",
    "replay_no_shared_obs_agent_pov",
    "oracle",
    "shared_obs_visual_union",
    "episode:a:replay",
    "recipient:b",
    "pov_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  ];
  const base = preferenceKeyFromTuple(baseTuple);
  for (const [index, replacement] of replacements.entries()) {
    const changedTuple = [...baseTuple];
    changedTuple[index] = replacement;
    assert.equal(
      sameAuthorizedPresentationPreferenceKey(
        base,
        preferenceKeyFromTuple(changedTuple),
      ),
      false,
      `tuple slot ${index} must reset preferences`,
    );
  }
});

test("structured preference keys reject colon collisions and distinguish A to B to A", async () => {
  const colonLeftTuple = [
    "combat_debugger",
    "session:a",
    "b",
    "live_oracle",
    "oracle",
    null,
    null,
    null,
    null,
  ];
  const colonRightTuple = [
    "combat_debugger",
    "session",
    "a:b",
    "live_oracle",
    "oracle",
    null,
    null,
    null,
    null,
  ];
  assert.equal(colonLeftTuple.join(":"), colonRightTuple.join(":"));
  assert.equal(
    sameAuthorizedPresentationPreferenceKey(
      preferenceKeyFromTuple(colonLeftTuple),
      preferenceKeyFromTuple(colonRightTuple),
    ),
    false,
  );

  const authorityA = authorizedPresentationPreferenceKey(
    await normalized("live_oracle"),
  );
  const authorityB = authorizedPresentationPreferenceKey(
    await normalized("live_no_shared_obs_agent_pov"),
  );
  const authorityAAgain = authorizedPresentationPreferenceKey(
    await normalized("live_oracle"),
  );
  assert.equal(sameAuthorizedPresentationPreferenceKey(authorityA, authorityB), false);
  assert.equal(
    sameAuthorizedPresentationPreferenceKey(authorityB, authorityAAgain),
    false,
  );
  assert.equal(
    sameAuthorizedPresentationPreferenceKey(authorityA, authorityAAgain),
    true,
  );

  const forged = {
    ...authorityA,
    serialized: authorityB?.serialized,
  };
  assert.equal(sameAuthorizedPresentationPreferenceKey(authorityA, forged), false);
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

test("scene adapter preserves exact mask and owner-centered settled overlays", async () => {
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
    const owner = scene.agents.find(
      (/** @type {Record<string, any>} */ agent) =>
        agent.presentation_key === inspection.actor_presentation_key,
    );
    assert.ok(owner);
    assert.deepEqual(
      scene.ranges.map((/** @type {Record<string, any>} */ range) => [
        range.kind,
        range.center,
        range.radius,
      ]),
      [
        ["observation", owner.position, owner.observation_radius],
        ["basic", owner.position, owner.basic_interaction_radius],
        ["ultimate", owner.position, owner.ultimate_interaction_radius],
      ],
    );
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

test("replay selection names the inspection owner while live selection remains the draft target", async () => {
  for (const kind of [
    "replay_oracle",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const frame = await normalized(kind);
    const inspection = authorizedPresentationInspection(frame);
    const scene = authorizedPresentationSceneView(frame);
    assert.ok(inspection);
    assert.ok(scene);
    assert.equal(
      scene.selection.inspection_owner_presentation_key,
      inspection.actor_presentation_key,
    );
    assert.equal(
      scene.selection.selected_presentation_key,
      inspection.actor_presentation_key,
    );
    assert.notEqual(
      scene.selection.selected_presentation_key,
      inspection.accepted_target.target_presentation_key ?? null,
    );
  }

  for (const kind of ["live_oracle", "live_no_shared_obs_agent_pov"]) {
    const frame = await normalized(kind);
    const inspection = authorizedPresentationInspection(frame);
    const scene = authorizedPresentationSceneView(frame);
    assert.ok(inspection);
    assert.ok(scene);
    assert.equal(
      scene.selection.inspection_owner_presentation_key,
      inspection.actor_presentation_key,
    );
    assert.equal(
      scene.selection.selected_presentation_key,
      inspection.draft_target.target_presentation_key ?? null,
    );
  }
});

test("Agent local inspection keeps the no-argument view exact and preserves action-owner overlays", async () => {
  for (const kind of [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const frame = await normalized(kind);
    const baseline = authorizedPresentationSceneView(frame);
    const explicitDefault = authorizedPresentationSceneView(frame, undefined);
    assert.ok(baseline);
    assert.deepEqual(explicitDefault, baseline);

    const actionOwnerKey = baseline.selection.inspection_owner_presentation_key;
    assert.equal(typeof actionOwnerKey, "string");
    const inspected = authorizedPresentationSceneView(frame, actionOwnerKey);
    assert.ok(inspected);
    assert.equal(
      inspected.selection.controlled_presentation_key,
      baseline.selection.controlled_presentation_key,
    );
    assert.equal(inspected.selection.inspection_owner_presentation_key, actionOwnerKey);
    assert.equal(inspected.selection.selected_presentation_key, actionOwnerKey);
    assert.deepEqual(inspected.ranges, baseline.ranges);
    assert.deepEqual(inspected.selected_legality, baseline.selected_legality);
    assert.deepEqual(inspected.pending_route, baseline.pending_route);
    assertRecursivelyFrozen(inspected);
  }
});

test("Agent local inspection selects an authorized nonrecipient without transferring outgoing authority", async () => {
  for (const kind of [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const frame = await normalized(kind);
    const before = JSON.stringify(frame);
    const baseline = authorizedPresentationSceneView(frame);
    assert.ok(baseline);
    const actionOwnerKey = baseline.selection.inspection_owner_presentation_key;
    const localAgent = baseline.agents.find(
      (/** @type {Record<string, any>} */ agent) =>
        agent.presentation_key !== actionOwnerKey,
    );
    assert.ok(localAgent);

    const inspected = authorizedPresentationSceneView(
      frame,
      localAgent.presentation_key,
    );
    assert.ok(inspected);
    assert.equal(
      inspected.selection.controlled_presentation_key,
      baseline.selection.controlled_presentation_key,
    );
    assert.equal(
      inspected.selection.inspection_owner_presentation_key,
      localAgent.presentation_key,
    );
    assert.equal(
      inspected.selection.selected_presentation_key,
      localAgent.presentation_key,
    );
    assert.deepEqual(
      inspected.ranges.map((/** @type {Record<string, any>} */ range) => [
        range.kind,
        range.presentation_key,
        range.public_agent_id,
        range.center,
        range.radius,
      ]),
      [
        [
          "observation",
          localAgent.presentation_key,
          localAgent.public_agent_id,
          localAgent.position,
          localAgent.observation_radius,
        ],
        [
          "basic",
          localAgent.presentation_key,
          localAgent.public_agent_id,
          localAgent.position,
          localAgent.basic_interaction_radius,
        ],
        [
          "ultimate",
          localAgent.presentation_key,
          localAgent.public_agent_id,
          localAgent.position,
          localAgent.ultimate_interaction_radius,
        ],
      ],
    );
    assert.equal(inspected.selected_legality, null);
    assert.equal(inspected.pending_route, null);
    assert.deepEqual(inspected.agents, baseline.agents);
    assert.equal(containsOwnKey(inspected, "global_slot"), false);
    assert.equal(JSON.stringify(frame), before);
    assertRecursivelyFrozen(inspected);
  }
});

test("Agent explicit clear and disappeared keys fail closed without hiding authorized bodies", async () => {
  const noShared = await normalized("replay_no_shared_obs_agent_pov");
  const shared = await normalized("replay_shared_obs_agent_pov");
  const sharedView = authorizedPresentationSceneView(shared);
  assert.ok(sharedView);
  const foreignPresentationKey = sharedView.agents[0]?.presentation_key;
  assert.equal(typeof foreignPresentationKey, "string");

  for (const [frame, unavailableKey] of [
    [noShared, foreignPresentationKey],
    [shared, "disappeared-agent-presentation-key"],
  ]) {
    const baseline = authorizedPresentationSceneView(frame);
    assert.ok(baseline);
    for (const localKey of [null, unavailableKey]) {
      const inspected = authorizedPresentationSceneView(frame, localKey);
      assert.ok(inspected);
      assert.deepEqual(inspected.agents, baseline.agents);
      assert.equal(
        inspected.selection.controlled_presentation_key,
        baseline.selection.controlled_presentation_key,
      );
      assert.equal(inspected.selection.inspection_owner_presentation_key, null);
      assert.equal(inspected.selection.selected_presentation_key, null);
      assert.deepEqual(inspected.ranges, []);
      assert.equal(inspected.selected_legality, null);
      assert.equal(inspected.pending_route, null);
      assert.equal(containsOwnKey(inspected, "global_slot"), false);
      assertRecursivelyFrozen(inspected);
    }
  }
});

test("researcher views ignore Agent-local inspection preferences", async () => {
  for (const kind of ["live_oracle", "replay_oracle"]) {
    const frame = await normalized(kind);
    const baseline = authorizedPresentationSceneView(frame);
    assert.ok(baseline);
    const alternateKey = baseline.agents.at(-1)?.presentation_key;
    assert.equal(typeof alternateKey, "string");
    for (const localKey of [null, alternateKey, "missing-presentation-key"]) {
      assert.deepEqual(authorizedPresentationSceneView(frame, localKey), baseline);
    }
  }
});

test("final replay selection falls back only to an authorized action-axis owner", async () => {
  for (const kind of [
    "replay_oracle_final_selected",
    "replay_no_shared_final",
    "replay_shared_final",
  ]) {
    const frame = await normalizeAuthorizedPresentationFrameV1(
      fixture.state_cases[kind],
    );
    const scene = authorizedPresentationSceneView(frame);
    assert.ok(scene);
    assert.equal(authorizedPresentationInspection(frame), null);
    assert.equal(
      scene.selection.inspection_owner_presentation_key,
      frame.action_axis.owner_presentation_key,
    );
    assert.equal(
      scene.selection.selected_presentation_key,
      frame.action_axis.owner_presentation_key,
    );
    const owner = scene.agents.find(
      (/** @type {Record<string, any>} */ agent) =>
        agent.presentation_key === frame.action_axis.owner_presentation_key,
    );
    assert.ok(owner);
    assert.deepEqual(
      scene.ranges.map((/** @type {Record<string, any>} */ range) => range.kind),
      ["observation", "basic", "ultimate"],
    );
    assert.equal(
      scene.ranges.every(
        (/** @type {Record<string, any>} */ range) =>
          range.presentation_key === owner.presentation_key &&
          range.public_agent_id === owner.public_agent_id &&
          range.center === owner.position,
      ),
      true,
    );
    assert.equal(scene.pending_route, null);
    assert.equal(scene.selected_legality, null);
  }

  const frame = await normalizeAuthorizedPresentationFrameV1(
    fixture.state_cases.replay_oracle_final_unselected,
  );
  const scene = authorizedPresentationSceneView(frame);
  assert.ok(scene);
  assert.equal(frame.action_axis, null);
  assert.deepEqual(scene.selection, {
    controlled_presentation_key: null,
    inspection_owner_presentation_key: null,
    selected_presentation_key: null,
  });
  assert.deepEqual(scene.ranges, []);
  assert.equal(scene.pending_route, null);
  assert.equal(scene.selected_legality, null);
});

test("certified legality projection crosses every target disclosure and combat lane", async () => {
  const frame = await normalized("replay_oracle");
  const inspection = authorizedPresentationInspection(frame);
  const scene = authorizedPresentationSceneView(frame);
  assert.ok(inspection);
  assert.ok(scene);
  const decisionMask = inspection.decision_mask;
  const owner = scene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key === inspection.actor_presentation_key,
  );
  assert.ok(owner);
  const targets = [
    "no_target",
    "visible_authorized_agent",
    "axis_only_authorized_agent",
  ].map((targetKind) =>
    decisionMask.target_actions.find(
      (/** @type {Record<string, any>} */ target) => target.target_kind === targetKind,
    ),
  );
  assert.equal(targets.every(Boolean), true);
  const lanes = Object.freeze([
    Object.freeze({ name: "none", index: null }),
    Object.freeze({ name: "basic", index: 0 }),
    Object.freeze({ name: "ultimate", index: 1 }),
  ]);

  for (const target of targets) {
    assert.ok(target);
    const row = decisionMask.target_use_ultimate_joint_mask[target.target_action];
    /** @type {Record<string, any> | null} */
    const visibleAgent =
      target.target_kind === "visible_authorized_agent"
        ? scene.agents.find(
            (/** @type {Record<string, any>} */ agent) =>
              agent.presentation_key === target.target_presentation_key,
          )
        : null;
    if (target.target_kind === "visible_authorized_agent") {
      assert.ok(visibleAgent);
    }
    for (const lane of lanes) {
      const legality = projectCertifiedInspectionLegality(
        decisionMask,
        owner,
        target,
        lane.name,
      );
      assert.ok(legality);
      assert.equal(legality.owner_presentation_key, owner.presentation_key);
      assert.equal(legality.owner_public_agent_id, owner.public_agent_id);
      assert.equal(legality.target_kind, target.target_kind);
      assert.equal(legality.target_action, target.target_action);
      assert.equal(
        legality.target_presentation_key,
        target.target_kind === "visible_authorized_agent"
          ? visibleAgent?.presentation_key
          : null,
      );
      assert.equal(
        legality.target_public_agent_id,
        target.target_kind === "no_target" ? null : target.target_public_agent_id,
      );
      assert.deepEqual([legality.lane_0_available, legality.lane_1_available], row);
      assert.equal(legality.armed_lane, lane.index);
      assert.equal(
        legality.armed_pair_legal,
        lane.index === null ? null : row[lane.index],
      );
      assert.equal(Object.isFrozen(legality), true);

      const route = projectCertifiedInspectionRoute(
        inspection,
        owner,
        visibleAgent,
        target,
        legality,
      );
      const routeExpected =
        target.target_kind === "visible_authorized_agent" && lane.index !== null;
      assert.equal(route !== null, routeExpected);
      if (route !== null) {
        if (lane.index !== 0 && lane.index !== 1) {
          assert.fail("A drawable route requires an armed Basic or Ultimate lane.");
        }
        assert.equal(route.source_presentation_key, owner.presentation_key);
        assert.equal(route.target_presentation_key, visibleAgent?.presentation_key);
        assert.equal(route.lane, lane.index);
        assert.equal(route.legal, row[lane.index]);
        assert.equal(Object.isFrozen(route), true);
      }
    }
  }
});

test("inspection projections fail closed on mismatched authority or target joins", () => {
  const owner = {
    presentation_key: "owner-key",
    public_agent_id: "agent-owner",
    radius: 0.5,
  };
  const target = {
    display_name: "Visible target",
    target_action: 0,
    target_kind: "visible_authorized_agent",
    target_presentation_key: "visible-key",
    target_public_agent_id: "agent-visible",
    target_anchor: [7, 8],
  };
  const matchingMask = {
    owner_presentation_key: owner.presentation_key,
    owner_public_agent_id: owner.public_agent_id,
    target_actions: [target],
    target_use_ultimate_joint_mask: [[true, false]],
  };
  assert.equal(
    projectCertifiedInspectionLegality(
      { ...matchingMask, owner_public_agent_id: "wrong-owner" },
      owner,
      target,
      "basic",
    ),
    null,
  );
  assert.equal(
    projectCertifiedInspectionLegality(
      matchingMask,
      owner,
      { ...target, target_action: 4 },
      "basic",
    ),
    null,
  );
  assert.equal(
    projectCertifiedInspectionLegality(
      matchingMask,
      owner,
      { ...target, display_name: "Mismatched target row" },
      "basic",
    ),
    null,
  );

  const legality = projectCertifiedInspectionLegality(
    matchingMask,
    owner,
    target,
    "basic",
  );
  assert.ok(legality);
  assert.equal(
    projectCertifiedInspectionRoute(
      { actor_anchor: [2, 3] },
      owner,
      {
        presentation_key: "different-visible-key",
        public_agent_id: target.target_public_agent_id,
        radius: 0.75,
      },
      target,
      legality,
    ),
    null,
  );
  assert.equal(
    projectCertifiedInspectionRoute(
      { actor_anchor: [2, 3] },
      owner,
      {
        presentation_key: target.target_presentation_key,
        public_agent_id: target.target_public_agent_id,
        radius: 0.75,
      },
      target,
      { ...legality, owner_public_agent_id: "wrong-owner" },
    ),
    null,
  );
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
