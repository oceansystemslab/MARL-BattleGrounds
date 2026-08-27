import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  authorizedOracleCommandSlotForPresentationKey,
  authorizedOracleCommandSlotForPublicAgentId,
  authorizedPresentationAudience,
  authorizedPresentationHasResearcherSpace,
  authorizedPresentationIdentityRows,
  authorizedPresentationIncomingRows,
  authorizedPresentationInspection,
  authorizedPresentationInspectionState,
  authorizedPresentationPendingJointActionRows,
  authorizedPresentationPreferenceKey,
  authorizedPresentationResearcherInspectionState,
  authorizedPresentationResearcherSceneView,
  authorizedPresentationSceneView,
  authorizedPresentationTechnicalFacts,
  authorizedPresentationTransitionRows,
  authorizedPresentationUpcomingTransitionRows,
  isAuthorizedPresentationFrame,
  projectCertifiedInCombatDurationStatus,
  projectCertifiedInspectionLegality,
  projectCertifiedInspectionRoute,
  sameAuthorizedPresentationPreferenceKey,
  scopedPresentationKey,
} from "../src/authorized-presentation-adapter.js";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { explainChoreographyEvent } from "../src/choreography-painter.js";
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

const EXPECTED_CLASS_LABELS = new Map([
  [1, "Mage"],
  [2, "Warrior"],
  [3, "Hunter"],
  [4, "Rogue"],
  [5, "Priest"],
]);

/** @param {Record<string, any>} identity */
function expectedAuthorizedIdentityTitle(identity) {
  return `Agent ID ${identity.public_agent_id} · ${EXPECTED_CLASS_LABELS.get(Number(identity.class_id))} · Team ${identity.team_id === 1 ? "A" : "B"}`;
}

/** @param {keyof typeof fixture.presentations} kind */
async function normalized(kind) {
  return await normalizeAuthorizedPresentationFrameV1(fixture.presentations[kind]);
}

/** @param {keyof typeof fixture.state_cases} kind */
async function normalizedState(kind) {
  return await normalizeAuthorizedPresentationFrameV1(fixture.state_cases[kind]);
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

/** @type {Readonly<Record<string, number>>} */
const ORACLE_EVENT_PHASE_RANK = Object.freeze({
  action_rejected: 10,
  ability_activated: 20,
  recipient_health_resolution: 40,
  combat_countdown_reset: 50,
  agent_left_combat: 50,
  health_regenerated: 50,
  cooldown_started: 60,
  cooldown_ready: 60,
  charge_phase_displacement: 70,
  ordinary_movement_phase_displacement: 80,
  agent_died: 90,
  status_aged_to_zero: 100,
  status_broken_by_damage: 100,
  status_applied: 100,
  status_refreshed_or_extended: 100,
  status_cleared_by_new_death: 100,
  spawn_shield_expired: 110,
  respawn_wave_occurred: 120,
  agent_respawned: 120,
});

const ORACLE_STATUS_ID_BY_CHANNEL = Object.freeze([
  "warrior_charge_slow",
  "hunter_basic_slow",
  "rogue_poison_slow",
  "warrior_charge_stun",
  "hunter_trap_stun",
  "rogue_poison_stun",
  "rogue_poison_anti_heal",
  "mage_burst_damage_amplification",
  "priest_blessing_of_freedom_movement_floor",
]);

/** @param {Record<string, any>} raw @param {number} classId */
function oracleTrajectoryForClass(raw, classId) {
  const agent = raw.current_endpoint.scene.agents.find(
    (/** @type {Record<string, any>} */ candidate) => candidate.class_id === classId,
  );
  assert.ok(agent, `missing class ${classId} scene agent`);
  const trajectory = raw.latest_events.agent_phase_trajectories.find(
    (/** @type {Record<string, any>} */ candidate) =>
      candidate.agent_presentation_key === agent.presentation_key,
  );
  assert.ok(trajectory, `missing class ${classId} trajectory`);
  return trajectory;
}

/**
 * Give every represented class distinct start/post-Charge/successor points
 * without changing the digest-owned current endpoint.
 */
function movingOracleRaw() {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const startsByClass = new Map([
    [1, [2.6, 1.2]],
    [2, [2.7, 3.4]],
    [3, [17.4, 1.2]],
    [4, [17.3, 3.4]],
    [5, [2.8, 5.7]],
  ]);
  const postChargeByClass = new Map([
    [1, [2.1, 1.35]],
    [2, [2.2, 3.55]],
    [3, [17.9, 1.35]],
    [4, [17.8, 3.55]],
    [5, [2.2, 5.85]],
  ]);
  for (const agent of raw.current_endpoint.scene.agents) {
    const trajectory = oracleTrajectoryForClass(raw, agent.class_id);
    trajectory.transition_start.position = startsByClass.get(agent.class_id);
    trajectory.post_charge.position = postChargeByClass.get(agent.class_id);
  }
  return raw;
}

/** @param {Record<string, any>} raw @param {number} classId @param {string} phase */
function oracleAnchor(raw, classId, phase) {
  return structuredClone(oracleTrajectoryForClass(raw, classId)[phase]);
}

/** @param {Record<string, any>} raw @param {Record<string, any>[]} events */
function installOracleEvents(raw, events) {
  const transitionId = raw.latest_events.incoming_transition_id;
  raw.latest_events.events = events.map(
    (/** @type {Record<string, any>} */ event, ordinal) => ({
      ...event,
      event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
      ordinal,
      phase_rank: ORACLE_EVENT_PHASE_RANK[event.event_kind],
    }),
  );
  raw.latest_events.event_count = raw.latest_events.events.length;
  raw.latest_events.ordered_event_ids = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  raw.latest_events.ordered_event_kinds = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );
  return raw;
}

/** @param {Record<string, any>} raw @param {number} classId @param {string} phase */
function projectedTrajectoryPoint(raw, classId, phase) {
  const [x, y] = oracleTrajectoryForClass(raw, classId)[phase].position;
  return { x: x * 10, y: y * 10 };
}

/**
 * @param {Record<string, any>} raw
 * @param {string} kind
 * @param {{channel?: number, recipientClass?: number, sourceClass?: number}} [options]
 */
function oracleStatusEvent(raw, kind, options = {}) {
  const channel = options.channel ?? 4;
  const event = {
    event_kind: kind,
    recipient_anchor: oracleAnchor(raw, options.recipientClass ?? 2, "successor"),
    status_channel: channel,
    status_id: ORACLE_STATUS_ID_BY_CHANNEL[channel],
  };
  return kind === "status_applied"
    ? {
        ...event,
        source_anchor: oracleAnchor(raw, options.sourceClass ?? 3, "successor"),
      }
    : event;
}

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
    if (
      changed.researcher_space?.researcher_space_kind === "global_live_researcher_space"
    ) {
      changed.researcher_space.source_revision = changed.source.source_revision;
      changed.researcher_space.source_authority_epoch =
        changed.source.source_authority_epoch;
    }
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

test("opaque display identity is session-scoped and Agent researcher rows expose only internal navigation slots", async () => {
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
  assert.equal(sharedIdentity.command_global_slot, 0);
  assert.equal(sharedIdentity.activation_kind, "replay_pov_global");
  assert.equal(typeof sharedIdentity.visible_in_snapshot, "boolean");
  assert.equal(Object.hasOwn(sharedIdentity.agent, "global_slot"), false);
  assert.equal(
    scopedPresentationKey(shared, sharedIdentity.presentation_key),
    sharedIdentity.display_key,
  );
  assert.equal(Object.isFrozen(sharedIdentity), true);
});

test("Oracle, live Agent control, and Replay Agent navigation keep separate capabilities", async () => {
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

  const liveAgent = await normalized("live_no_shared_obs_agent_pov");
  const liveAgentRows = authorizedPresentationIdentityRows(liveAgent);
  assert.equal(authorizedPresentationHasResearcherSpace(liveAgent), true);
  assert.equal(
    liveAgentRows.every(
      ({ command_global_slot, activation_kind }) =>
        Number.isInteger(command_global_slot) && activation_kind === "live_pov_global",
    ),
    true,
  );
  assert.equal(
    authorizedOracleCommandSlotForPresentationKey(
      liveAgent,
      liveAgentRows[0].presentation_key,
    ),
    null,
  );
  assert.equal(liveAgentRows[0].command_global_slot, 0);
  assert.deepEqual(
    liveAgent.researcher_space.pending_inspection.draft.decision_mask.target_actions
      .slice(1)
      .map((/** @type {Record<string, any>} */ target) =>
        authorizedOracleCommandSlotForPublicAgentId(
          liveAgent,
          target.target_public_agent_id,
        ),
      ),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  );
  for (const kind of [
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const agentFrame = await normalized(kind);
    const agentRows = authorizedPresentationIdentityRows(agentFrame);
    assert.equal(
      agentRows.every(
        ({ command_global_slot, activation_kind }) =>
          Number.isInteger(command_global_slot) &&
          activation_kind === "replay_pov_global",
      ),
      true,
    );
    assert.equal(
      authorizedOracleCommandSlotForPresentationKey(
        agentFrame,
        agentRows[0].presentation_key,
      ),
      null,
    );
    assert.equal(
      authorizedOracleCommandSlotForPublicAgentId(agentFrame, "agent-slot-0"),
      null,
    );
  }
});

test("Agent researcher roster is global while live and replay battlefields remain fog-scoped", async () => {
  for (const kind of [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const frame = await normalized(kind);
    const battlefield = authorizedPresentationSceneView(frame);
    const researcher = authorizedPresentationResearcherSceneView(frame);
    const rows = authorizedPresentationIdentityRows(frame);
    assert.ok(battlefield);
    assert.ok(researcher);
    assert.equal(researcher.agents.length, frame.researcher_space.roster_agents.length);
    assert.equal(rows.length, researcher.agents.length);
    assert.equal(authorizedPresentationHasResearcherSpace(frame), true);
    assert.deepEqual(
      rows
        .filter((/** @type {Record<string, any>} */ row) => row.visible_in_snapshot)
        .map((/** @type {Record<string, any>} */ row) => row.public_agent_id),
      battlefield.agents.map(
        (/** @type {Record<string, any>} */ agent) => agent.public_agent_id,
      ),
    );
    assert.equal(
      rows.some(({ visible_in_snapshot }) => !visible_in_snapshot),
      true,
    );
    assert.equal(
      frame.researcher_space.roster_agents.some(
        (/** @type {Record<string, any>} */ agent) => Object.hasOwn(agent, "position"),
      ),
      false,
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
    assert.equal(scene.class_mechanics, frame.scene.class_mechanics);
    assert.equal(scene.spawn_shield_mechanics, frame.scene.spawn_shield_mechanics);
    assertRecursivelyFrozen(scene.class_mechanics);
    assertRecursivelyFrozen(scene.spawn_shield_mechanics);
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

test("all five leaves project the combat countdown as one durable display status", async () => {
  for (const kind of Object.keys(fixture.presentations)) {
    const baseline = authorizedPresentationSceneView(
      await normalized(/** @type {keyof typeof fixture.presentations} */ (kind)),
    );
    assert.ok(baseline);
    assert.equal(
      baseline.agents.every((/** @type {Record<string, any>} */ agent) =>
        agent.statuses.every(
          (/** @type {Record<string, any>} */ status) =>
            status.token_id !== "in_combat",
        ),
      ),
      true,
      kind,
    );

    const raw = structuredClone(
      fixture.presentations[/** @type {keyof typeof fixture.presentations} */ (kind)],
    );
    const rawScene = raw.current_endpoint.scene ?? raw.current_endpoint.parts?.scene;
    assert.ok(rawScene, kind);
    const rawAgent = rawScene.agents[0];
    assert.ok(rawAgent, kind);
    rawAgent.steps_until_out_of_combat = 2;
    const projectedStatus = projectCertifiedInCombatDurationStatus(rawAgent);
    assert.deepEqual(projectedStatus, {
      status_id: "in_combat",
      token_id: "in_combat",
      configured_duration_steps: rawAgent.out_of_combat_delay_steps,
      remaining_duration: 2,
      duration: 2,
    });
    assert.equal(Object.isFrozen(projectedStatus), true, kind);
    assert.equal(
      rawAgent.statuses.some(
        (/** @type {Record<string, any>} */ status) => status.status_id === "in_combat",
      ),
      false,
      kind,
    );
  }
  assert.equal(
    projectCertifiedInCombatDurationStatus({
      steps_until_out_of_combat: 0,
      out_of_combat_delay_steps: 5,
    }),
    null,
  );
  assert.equal(projectCertifiedInCombatDurationStatus(null), null);
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

test("live Agent local preferences cannot replace the controlled draft target or overlays", async () => {
  const frame = await normalized("live_no_shared_obs_agent_pov");
  const baseline = authorizedPresentationSceneView(frame);
  assert.ok(baseline);
  const localAgent = baseline.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key !== baseline.selection.inspection_owner_presentation_key,
  );
  assert.ok(localAgent);

  for (const localKey of [
    baseline.selection.inspection_owner_presentation_key,
    localAgent.presentation_key,
    null,
    "disappeared-agent-presentation-key",
  ]) {
    const projected = authorizedPresentationSceneView(frame, localKey);
    assert.ok(projected);
    assert.deepEqual(projected.selection, baseline.selection);
    assert.deepEqual(projected.ranges, baseline.ranges);
    assert.deepEqual(projected.selected_legality, baseline.selected_legality);
    assert.deepEqual(projected.pending_route, baseline.pending_route);
    assertRecursivelyFrozen(projected);
  }
});

test("live Oracle and Agent expose one interaction model before fog projection", async () => {
  const oracle = authorizedPresentationSceneView(await normalized("live_oracle"));
  const agentFrame = await normalized("live_no_shared_obs_agent_pov");
  const agentBaseline = authorizedPresentationSceneView(agentFrame);
  assert.ok(oracle);
  assert.ok(agentBaseline);
  const alternateLocalAgent = agentBaseline.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key !==
      agentBaseline.selection.inspection_owner_presentation_key,
  );
  assert.ok(alternateLocalAgent);
  const agent = authorizedPresentationSceneView(
    agentFrame,
    alternateLocalAgent.presentation_key,
  );
  assert.ok(agent);

  const publicSelection = (/** @type {Record<string, any>} */ scene) => {
    const byKey = new Map(
      scene.agents.map((/** @type {Record<string, any>} */ row) => [
        row.presentation_key,
        row.public_agent_id,
      ]),
    );
    return {
      controlled: byKey.get(scene.selection.controlled_presentation_key) ?? null,
      inspectionOwner:
        byKey.get(scene.selection.inspection_owner_presentation_key) ?? null,
      selected: byKey.get(scene.selection.selected_presentation_key) ?? null,
      legality: {
        ...scene.selected_legality,
        owner_presentation_key: undefined,
        target_presentation_key: undefined,
      },
      route:
        scene.pending_route === null
          ? null
          : {
              ...scene.pending_route,
              source_presentation_key: undefined,
              target_presentation_key: undefined,
            },
      ranges: scene.ranges.map((/** @type {Record<string, any>} */ range) => ({
        kind: range.kind,
        public_agent_id: range.public_agent_id,
        center: range.center,
        radius: range.radius,
      })),
    };
  };
  assert.deepEqual(publicSelection(agent), publicSelection(oracle));
});

test("replay Agent local inspection keeps the no-argument view exact and preserves action-owner overlays", async () => {
  for (const kind of [
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

test("replay Agent local inspection selects an authorized nonrecipient without transferring outgoing authority", async () => {
  for (const kind of [
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
      assert.equal(legality.basic_available, target.target_action > 0 && row[0]);
      assert.equal(legality.ultimate_available, row[1]);
      if (target.target_kind === "no_target") {
        assert.equal(row[0], true);
        assert.equal(legality.basic_available, false);
      }
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

test("Agent visual events, local evidence, Submitted/Accepted, Technical Frame, and inspection stay disjoint", async () => {
  const frame = await normalized("replay_shared_obs_agent_pov");
  const incoming = authorizedPresentationIncomingRows(frame);
  const transition = authorizedPresentationTransitionRows(frame);
  const technical = authorizedPresentationTechnicalFacts(frame);
  const inspection = authorizedPresentationInspection(frame);
  assert.ok(incoming.length > 0);
  assert.equal(
    incoming.every(({ vocabulary }) => vocabulary === "event"),
    true,
  );
  assert.equal(
    frame.latest_events.summary_kind,
    "shared_obs_recipient_observation_deltas",
  );
  assert.equal(
    frame.visual_events.summary_kind,
    "agent_pov_fog_filtered_visual_events",
  );
  assert.equal(
    transition.length,
    frame.researcher_space.latest_transition.action_rows.length,
  );
  assert.deepEqual(Object.keys(transition[0]), [
    "actor_title",
    "actor_accent",
    "actor_team",
    "submitted_action",
    "accepted_action",
  ]);
  assert.match(
    transition[0].actor_title,
    /^Agent ID .+ · (?:Mage|Warrior|Hunter|Rogue|Priest) · Team [AB]$/u,
  );
  assert.notEqual(
    transition[0].submitted_action,
    frame.latest_transition.action_rows[0].submitted_action,
  );
  assert.notEqual(
    transition[0].accepted_action,
    frame.latest_transition.action_rows[0].accepted_action,
  );
  assert.equal(inspection, frame.inspection);
  assert.deepEqual(
    technical.map(({ id, label, value }) => [id, label, value]),
    [
      ["frame", "Frame", 1],
      ["simulator_step", "Simulator Step", 1],
      [
        "incoming_transition",
        "Incoming Transition",
        "episode-001:shared-obs-visual-union:agent-slot-0:transition:0",
      ],
    ],
  );
  assert.equal(
    incoming.some(({ payload }) => payload === inspection),
    false,
  );
});

test("Latest Transition is exact for all five leaves and empty at frame zero", async () => {
  const laterCases = [
    ["live_oracle", 5],
    ["live_no_shared_obs_agent_pov", 5],
    ["replay_oracle", 5],
    ["replay_no_shared_obs_agent_pov", 5],
    ["replay_shared_obs_agent_pov", 5],
  ];
  const exactKeys = [
    "actor_title",
    "actor_accent",
    "actor_team",
    "submitted_action",
    "accepted_action",
  ];
  for (const [kind, expectedCount] of laterCases) {
    const rows = authorizedPresentationTransitionRows(
      await normalized(/** @type {keyof typeof fixture.presentations} */ (kind)),
    );
    assert.equal(rows.length, expectedCount, String(kind));
    for (const row of rows) {
      assert.deepEqual(Object.keys(row), exactKeys, String(kind));
      assert.match(
        row.actor_title,
        /^Agent ID .+ · (?:Mage|Warrior|Hunter|Rogue|Priest) · Team [AB]$/u,
      );
      assert.match(row.actor_accent, /^(?:mage|warrior|hunter|rogue|priest)$/u);
      assert.match(row.actor_team, /^team-[ab]$/u);
      assert.equal(Object.isFrozen(row.submitted_action), true);
      assert.equal(Object.isFrozen(row.accepted_action), true);
      assert.deepEqual(Object.keys(row.submitted_action), [
        "move_action",
        "target_action",
        "use_ultimate_action",
      ]);
      assert.deepEqual(Object.keys(row.accepted_action), [
        "move_action",
        "target_action",
        "use_ultimate_action",
      ]);
    }
    const serialized = JSON.stringify(rows);
    for (const forbidden of [
      "mask",
      "movement_accepted",
      "combat_result",
      "submission_kind",
      "revision",
      "generation",
      "transition_id",
      "presentation_key",
      "public_agent_id",
      "target_action_recipient_public_agent_id_by_id",
    ]) {
      assert.equal(serialized.includes(forbidden), false, `${kind}: ${forbidden}`);
    }
  }

  for (const kind of [
    "live_oracle_frame_zero",
    "live_no_shared_frame_zero",
    "replay_oracle_frame_zero",
    "replay_no_shared_frame_zero",
    "replay_shared_frame_zero",
  ]) {
    assert.deepEqual(
      authorizedPresentationTransitionRows(
        await normalizedState(/** @type {keyof typeof fixture.state_cases} */ (kind)),
      ),
      [],
      String(kind),
    );
  }
});

test("Upcoming Transition is the exact authority-scoped transition out of the current frame", async () => {
  const replayCases = [
    await normalized("replay_oracle"),
    await normalized("replay_no_shared_obs_agent_pov"),
    await normalized("replay_shared_obs_agent_pov"),
    await normalizedState("replay_oracle_frame_zero"),
    await normalizedState("replay_no_shared_frame_zero"),
    await normalizedState("replay_shared_frame_zero"),
  ];
  for (const frame of replayCases) {
    const rows = authorizedPresentationUpcomingTransitionRows(frame);
    const transition =
      frame.authority.authority_kind === "oracle"
        ? frame.upcoming_transition
        : frame.researcher_space.upcoming_transition;
    assert.ok(transition);
    assert.equal(rows.length, transition.action_rows.length);
    for (const [index, row] of rows.entries()) {
      const source = transition.action_rows[index];
      assert.deepEqual(Object.keys(row), [
        "actor_title",
        "actor_accent",
        "actor_team",
        "submitted_action",
        "accepted_action",
      ]);
      assert.deepEqual(row.submitted_action, source.submitted_action);
      assert.deepEqual(row.accepted_action, source.accepted_action);
      assert.notEqual(row.submitted_action, source.submitted_action);
      assert.notEqual(row.accepted_action, source.accepted_action);
      assert.match(
        row.actor_title,
        /^Agent ID .+ · (?:Mage|Warrior|Hunter|Rogue|Priest) · Team [AB]$/u,
      );
      assert.match(row.actor_team, /^team-[ab]$/u);
      assert.equal(Object.isFrozen(row), true);
    }
    assert.equal(rows.length, 5, frame.presentation_kind);
    assert.equal(Object.isFrozen(rows), true);
  }

  for (const kind of ["live_oracle", "live_no_shared_obs_agent_pov"]) {
    assert.deepEqual(
      authorizedPresentationUpcomingTransitionRows(
        await normalized(/** @type {keyof typeof fixture.presentations} */ (kind)),
      ),
      [],
    );
  }
  for (const kind of [
    "replay_oracle_final_selected",
    "replay_oracle_final_unselected",
    "replay_no_shared_final",
    "replay_shared_final",
  ]) {
    assert.deepEqual(
      authorizedPresentationUpcomingTransitionRows(
        await normalizedState(/** @type {keyof typeof fixture.state_cases} */ (kind)),
      ),
      [],
      kind,
    );
  }
  assert.deepEqual(
    authorizedPresentationUpcomingTransitionRows({ ...replayCases[0] }),
    [],
  );
});

test("Pending Joint Action is one exact researcher-space row per live actor", async () => {
  const oracle = await normalized("live_oracle");
  const agent = await normalized("live_no_shared_obs_agent_pov");
  const oracleRows = authorizedPresentationPendingJointActionRows(oracle);
  const agentRows = authorizedPresentationPendingJointActionRows(agent);
  assert.equal(oracleRows.length, oracle.pending_joint_action.action_rows.length);
  assert.equal(
    agentRows.length,
    agent.researcher_space.pending_joint_action.action_rows.length,
  );
  assert.deepEqual(agentRows, oracleRows);
  for (const row of oracleRows) {
    assert.deepEqual(Object.keys(row), [
      "actor_title",
      "actor_accent",
      "actor_team",
      "pending_action",
    ]);
    assert.match(
      row.actor_title,
      /^Agent ID .+ · (?:Mage|Warrior|Hunter|Rogue|Priest) · Team [AB]$/u,
    );
    assert.match(row.actor_accent, /^(?:mage|warrior|hunter|rogue|priest)$/u);
    assert.match(row.actor_team, /^team-[ab]$/u);
    assert.deepEqual(Object.keys(row.pending_action), [
      "move_action",
      "target_action",
      "use_ultimate_action",
    ]);
    assert.equal(Object.isFrozen(row), true);
    assert.equal(Object.isFrozen(row.pending_action), true);
  }
  assert.equal(Object.isFrozen(oracleRows), true);
  assert.equal(Object.isFrozen(agentRows), true);
  for (const kind of [
    "replay_oracle",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    assert.deepEqual(
      authorizedPresentationPendingJointActionRows(
        await normalized(/** @type {keyof typeof fixture.presentations} */ (kind)),
      ),
      [],
      kind,
    );
  }
});

test("Technical Frame projects the exact final five-leaf allowlist atomically", async () => {
  const cases = [
    [
      "live_oracle",
      [
        ["episode", "Episode", "episode-001"],
        ["frame", "Frame", 1],
        ["simulator_step", "Simulator Step", 1],
        ["incoming_transition", "Incoming Transition", "episode-001:transition:0"],
      ],
    ],
    [
      "live_no_shared_obs_agent_pov",
      [
        ["episode", "Episode", "episode-001"],
        ["frame", "Frame", 1],
        ["simulator_step", "Simulator Step", 1],
        ["incoming_transition", "Incoming Transition", "episode-001:transition:0"],
      ],
    ],
    [
      "replay_oracle",
      [
        ["artifact_digest_prefix", "Artifact Digest Prefix", "cccccccccccc"],
        ["frame", "Frame", 1],
        ["simulator_step", "Simulator Step", 1],
        ["incoming_transition", "Incoming Transition", "episode-001:transition:0"],
        ["ordinary_movement_distance_scale", "Ordinary Movement Distance Scale", 1],
      ],
    ],
    [
      "replay_no_shared_obs_agent_pov",
      [
        ["frame", "Frame", 1],
        ["simulator_step", "Simulator Step", 1],
        [
          "incoming_transition",
          "Incoming Transition",
          "episode-001:actor-pov:agent-slot-0:transition:0",
        ],
      ],
    ],
    [
      "replay_shared_obs_agent_pov",
      [
        ["frame", "Frame", 1],
        ["simulator_step", "Simulator Step", 1],
        [
          "incoming_transition",
          "Incoming Transition",
          "episode-001:shared-obs-visual-union:agent-slot-0:transition:0",
        ],
      ],
    ],
  ];
  for (const [kind, expected] of cases) {
    const frame = await normalized(
      /** @type {keyof typeof fixture.presentations} */ (kind),
    );
    const before = JSON.stringify(frame);
    const facts = authorizedPresentationTechnicalFacts(frame);
    assert.deepEqual(
      facts.map(({ id, label, value }) => [id, label, value]),
      expected,
      String(kind),
    );
    assert.equal(JSON.stringify(frame), before, String(kind));
    assertRecursivelyFrozen(facts);
    assert.equal(
      facts.every(
        (fact) =>
          Object.keys(fact).length === 3 &&
          Object.keys(fact).every(
            (key, index) => key === ["id", "label", "value"][index],
          ),
      ),
      true,
      String(kind),
    );
    const serialized = JSON.stringify(facts);
    for (const forbidden of [
      "technical_kind",
      "episode_id",
      "incoming_transition_id",
      "incoming_recipient_transition_id",
      "evaluation_frame_index",
      "recipient_frame_index",
      "simulator_step_count",
      "recorded_ordinary_movement_distance_scale",
      "revision",
      "generation",
      "timeline_id",
      "session_id",
      "presentation_key",
      "global_slot",
    ]) {
      assert.equal(
        serialized.includes(`"${forbidden}"`),
        false,
        `${kind}: ${forbidden}`,
      );
    }
  }

  const frameZeroCases = [
    [
      "live_oracle_frame_zero",
      [
        ["episode", "Episode", "episode-001"],
        ["frame", "Frame", 0],
        ["simulator_step", "Simulator Step", 0],
      ],
    ],
    [
      "live_no_shared_frame_zero",
      [
        ["episode", "Episode", "episode-001"],
        ["frame", "Frame", 0],
        ["simulator_step", "Simulator Step", 0],
      ],
    ],
    [
      "replay_oracle_frame_zero",
      [
        ["artifact_digest_prefix", "Artifact Digest Prefix", "cccccccccccc"],
        ["frame", "Frame", 0],
        ["simulator_step", "Simulator Step", 0],
        ["ordinary_movement_distance_scale", "Ordinary Movement Distance Scale", 1],
      ],
    ],
    [
      "replay_no_shared_frame_zero",
      [
        ["frame", "Frame", 0],
        ["simulator_step", "Simulator Step", 0],
      ],
    ],
    [
      "replay_shared_frame_zero",
      [
        ["frame", "Frame", 0],
        ["simulator_step", "Simulator Step", 0],
      ],
    ],
  ];
  for (const [kind, expected] of frameZeroCases) {
    const facts = authorizedPresentationTechnicalFacts(
      await normalizedState(/** @type {keyof typeof fixture.state_cases} */ (kind)),
    );
    assert.deepEqual(
      facts.map(({ id, label, value }) => [id, label, value]),
      expected,
      String(kind),
    );
    assert.equal(
      facts.some(({ id }) => id === "incoming_transition"),
      false,
    );
    assertRecursivelyFrozen(facts);
  }

  const normalizedOracle = await normalized("replay_oracle");
  assert.deepEqual(
    authorizedPresentationTechnicalFacts(fixture.presentations.replay_oracle),
    [],
  );
  assert.deepEqual(
    authorizedPresentationTechnicalFacts(Object.freeze({ ...normalizedOracle })),
    [],
  );
  let accessorReads = 0;
  const accessorRoot = structuredClone(fixture.presentations.replay_oracle);
  Object.defineProperty(accessorRoot, "technical_frame", {
    enumerable: true,
    get() {
      accessorReads += 1;
      throw new Error("untrusted Technical Frame accessor was read");
    },
  });
  assert.deepEqual(authorizedPresentationTechnicalFacts(accessorRoot), []);
  assert.equal(accessorReads, 0);

  for (const technicalFrame of [
    { ...normalizedOracle.technical_frame, technical_kind: "forbidden_technical" },
    { ...normalizedOracle.technical_frame, artifact_digest_prefix: "ABCDEF012345" },
    { ...normalizedOracle.technical_frame, incoming_transition_id: "" },
    { ...normalizedOracle.technical_frame, frame_index: -1 },
    {
      ...normalizedOracle.technical_frame,
      recorded_ordinary_movement_distance_scale: 0,
    },
  ]) {
    assert.deepEqual(
      authorizedPresentationTechnicalFacts(
        Object.freeze({ ...normalizedOracle, technical_frame: technicalFrame }),
      ),
      [],
    );
  }
});

test("live scope and replay inspection states remain exact", async () => {
  const liveOracle = authorizedPresentationInspectionState(
    await normalized("live_oracle"),
  );
  const liveAgent = authorizedPresentationInspectionState(
    await normalized("live_no_shared_obs_agent_pov"),
  );
  const liveAgentResearcher = authorizedPresentationResearcherInspectionState(
    await normalized("live_no_shared_obs_agent_pov"),
  );
  const replayFrame = await normalized("replay_oracle");
  const replay = authorizedPresentationInspectionState(replayFrame);
  const replayNoneFrame = await normalizedPairPresentation("replay_oracle");
  const replayNone = authorizedPresentationInspectionState(replayNoneFrame);
  assert.equal(liveOracle.state_kind, "live_editable");
  assert.equal(liveOracle.submission_scope, "joint_turn");
  assert.equal(liveOracle.inspection?.inspection_kind, "live_draft_action");
  assert.equal(liveAgent.submission_scope, "joint_turn");
  assert.equal(liveAgentResearcher.state_kind, "live_editable");
  assert.equal(liveAgentResearcher.submission_scope, "joint_turn");
  assert.equal(liveAgentResearcher.inspection?.actor_public_agent_id, "agent-slot-0");
  assert.equal(replay.state_kind, "replay_outgoing");
  assert.equal(replay.submission_scope, null);
  assert.equal(replay.inspection?.combat_lane, replayFrame.inspection.combat_lane);
  assert.equal(replayNone.state_kind, "replay_none");
  assert.equal(replayNone.submission_scope, null);
  assert.equal(replayNone.inspection, null);
  assert.deepEqual(authorizedPresentationSceneView(replayNoneFrame)?.ranges, []);
});

test("authority-selected visual events alone schedule opaque-key choreography", async () => {
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
      (event) => event.authorityVocabulary === "event" && event.noncausal !== true,
    ),
    true,
  );
  assert.equal(
    sharedPlan.events.some(
      (event) => event.kind === "activation" && event.spatial === true,
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
      (event) => event.authorityVocabulary === "event" && event.noncausal !== true,
    ),
    true,
  );
});

test("Oracle choreography exact-joins moving trajectories and presents successor endpoints", async () => {
  const raw = movingOracleRaw();
  installOracleEvents(raw, [
    {
      event_kind: "ability_activated",
      ability_component: "ultimate",
      source_anchor: oracleAnchor(raw, 1, "transition_start"),
      recipient_anchor: null,
    },
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: oracleAnchor(raw, 2, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 3, "transition_start"),
    },
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: oracleAnchor(raw, 5, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 2, "transition_start"),
    },
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: oracleAnchor(raw, 3, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 1, "transition_start"),
    },
    {
      event_kind: "ability_activated",
      ability_component: "ultimate",
      source_anchor: oracleAnchor(raw, 4, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 5, "transition_start"),
    },
    {
      event_kind: "ability_activated",
      ability_component: "ultimate",
      source_anchor: oracleAnchor(raw, 2, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 4, "transition_start"),
    },
    {
      event_kind: "recipient_health_resolution",
      recipient_anchor: oracleAnchor(raw, 3, "transition_start"),
      transition_start_health: 100,
      total_effective_damage: 7,
      total_effective_healing: 0,
      health_after_combat_resolution: 93,
      realized_net_health_change: -7,
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: oracleAnchor(raw, 4, "transition_start"),
      actual_health_regenerated: 3,
    },
    {
      event_kind: "cooldown_started",
      agent_anchor: oracleAnchor(raw, 1, "transition_start"),
    },
    {
      event_kind: "cooldown_ready",
      agent_anchor: oracleAnchor(raw, 5, "transition_start"),
    },
    {
      event_kind: "charge_phase_displacement",
      realized_displacement: [
        oracleTrajectoryForClass(raw, 2).post_charge.position[0] -
          oracleTrajectoryForClass(raw, 2).transition_start.position[0],
        oracleTrajectoryForClass(raw, 2).post_charge.position[1] -
          oracleTrajectoryForClass(raw, 2).transition_start.position[1],
      ],
      start_anchor: oracleAnchor(raw, 2, "transition_start"),
      end_anchor: oracleAnchor(raw, 2, "post_charge"),
    },
    {
      event_kind: "ordinary_movement_phase_displacement",
      realized_displacement: [
        oracleTrajectoryForClass(raw, 2).successor.position[0] -
          oracleTrajectoryForClass(raw, 2).post_charge.position[0],
        oracleTrajectoryForClass(raw, 2).successor.position[1] -
          oracleTrajectoryForClass(raw, 2).post_charge.position[1],
      ],
      start_anchor: oracleAnchor(raw, 2, "post_charge"),
      end_anchor: oracleAnchor(raw, 2, "successor"),
    },
  ]);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const incomingRows = authorizedPresentationIncomingRows(frame);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(plan.events.length, raw.latest_events.event_count);
  assert.deepEqual(
    plan.events.map((event) => event.eventId),
    incomingRows.map((row) => row.id),
  );
  assert.deepEqual(
    plan.events.map((event) => event.eventType),
    incomingRows.map((row) => row.kind),
  );

  /** @type {ReadonlyArray<readonly [number, number, number | null, string]>} */
  const abilityCases = [
    [0, 1, null, "mage_burst"],
    [1, 2, 3, "basic_damage"],
    [2, 5, 2, "basic_heal"],
    [3, 3, 1, "basic_damage"],
    [4, 4, 5, "rogue_poison"],
  ];
  for (const [eventIndex, sourceClass, targetClass, tokenId] of abilityCases) {
    const plannedEvent = /** @type {Record<string, any>} */ (plan.events[eventIndex]);
    const sourceTrajectory = oracleTrajectoryForClass(raw, sourceClass);
    assert.equal(plannedEvent.kind, "activation");
    assert.equal(plannedEvent.tokenId, tokenId);
    assert.equal(
      plannedEvent.sourcePresentationKey,
      sourceTrajectory.agent_presentation_key,
    );
    assert.equal(
      plannedEvent.sourcePublicAgentId,
      sourceTrajectory.agent_public_agent_id,
    );
    const expectedClassMechanics = raw.current_endpoint.scene.class_mechanics.find(
      (/** @type {Record<string, any>} */ mechanics) =>
        mechanics.class_id === sourceClass,
    );
    assert.deepEqual(
      plannedEvent.authorizedClassMechanics,
      plannedEvent.component === "ultimate" ? expectedClassMechanics : null,
    );
    assert.deepEqual(
      plannedEvent.source,
      projectedTrajectoryPoint(raw, sourceClass, "successor"),
    );
    if (targetClass === null) {
      assert.equal(plannedEvent.targetPresentationKey, null);
      assert.equal(plannedEvent.targetPublicAgentId, null);
      assert.equal(plannedEvent.target, null);
      assert.equal(plannedEvent.route, null);
      assert.equal(plannedEvent.presentationKind, "source_local");
    } else {
      const targetTrajectory = oracleTrajectoryForClass(raw, targetClass);
      assert.equal(
        plannedEvent.targetPresentationKey,
        targetTrajectory.agent_presentation_key,
      );
      assert.equal(
        plannedEvent.targetPublicAgentId,
        targetTrajectory.agent_public_agent_id,
      );
      assert.deepEqual(
        plannedEvent.target,
        projectedTrajectoryPoint(raw, targetClass, "successor"),
      );
      assert.ok(plannedEvent.route);
      assert.equal(plannedEvent.presentationKind, "routed");
    }
  }

  const chargeActivation = plan.events[5];
  assert.equal(chargeActivation.tokenId, "warrior_charge");
  assert.deepEqual(
    chargeActivation.source,
    projectedTrajectoryPoint(raw, 2, "transition_start"),
  );
  assert.deepEqual(
    chargeActivation.target,
    projectedTrajectoryPoint(raw, 4, "transition_start"),
  );

  const health = plan.events[6];
  assert.equal(health.kind, "net_health");
  assert.deepEqual(health.recipient, projectedTrajectoryPoint(raw, 3, "successor"));
  assert.equal(
    health.recipientPresentationKey,
    oracleTrajectoryForClass(raw, 3).agent_presentation_key,
  );
  assert.equal(
    health.recipientPublicAgentId,
    oracleTrajectoryForClass(raw, 3).agent_public_agent_id,
  );

  const regeneration = plan.events[7];
  const regenerationTrajectory = oracleTrajectoryForClass(raw, 4);
  assert.equal(regeneration.kind, "regeneration");
  assert.equal(regeneration.cueSemantic, "health_regenerated");
  assert.equal(regeneration.value, 3);
  assert.deepEqual(
    regeneration.recipient,
    projectedTrajectoryPoint(raw, 4, "successor"),
  );
  assert.equal(
    regeneration.agentPresentationKey,
    regenerationTrajectory.agent_presentation_key,
  );
  assert.equal(
    regeneration.agentPublicAgentId,
    regenerationTrajectory.agent_public_agent_id,
  );

  const cooldownStarted = /** @type {Record<string, any>} */ (plan.events[8]);
  const cooldownStartedTrajectory = oracleTrajectoryForClass(raw, 1);
  assert.equal(cooldownStarted.kind, "semantic_pulse");
  assert.equal(cooldownStarted.cueSemantic, "cooldown_started");
  assert.equal(cooldownStarted.anchor, null);
  assert.deepEqual(cooldownStarted.paintParts, { effect: false });
  assert.equal(cooldownStarted.presentationSuppressed, true);
  assert.equal(cooldownStarted.spatial, false);
  assert.equal(
    cooldownStarted.agentPresentationKey,
    cooldownStartedTrajectory.agent_presentation_key,
  );
  assert.equal(
    cooldownStarted.agentPublicAgentId,
    cooldownStartedTrajectory.agent_public_agent_id,
  );

  const cooldownReady = /** @type {Record<string, any>} */ (plan.events[9]);
  const cooldownReadyTrajectory = oracleTrajectoryForClass(raw, 5);
  assert.equal(cooldownReady.kind, "semantic_pulse");
  assert.equal(cooldownReady.cueSemantic, "cooldown_ready");
  assert.deepEqual(cooldownReady.anchor, projectedTrajectoryPoint(raw, 5, "successor"));
  assert.deepEqual(cooldownReady.paintParts, { effect: true });
  assert.equal(cooldownReady.presentationSuppressed, false);
  assert.equal(cooldownReady.spatial, true);
  assert.equal(
    cooldownReady.agentPresentationKey,
    cooldownReadyTrajectory.agent_presentation_key,
  );
  assert.equal(
    cooldownReady.agentPublicAgentId,
    cooldownReadyTrajectory.agent_public_agent_id,
  );

  const charge = plan.events[10];
  assert.equal(charge.eventType, "charge_phase_displacement");
  assert.equal(charge.kind, "feed_only");
  assert.equal(charge.spatial, false);
  for (const field of [
    "sourcePresentationKey",
    "sourcePublicAgentId",
    "start",
    "end",
    "route",
    "paintParts",
    "persistent",
  ]) {
    assert.equal(Object.hasOwn(charge, field), false, field);
  }

  const movement = plan.events[11];
  assert.equal(movement.eventType, "ordinary_movement_phase_displacement");
  assert.equal(movement.kind, "feed_only");
  assert.equal(movement.spatial, false);
  assert.equal(Object.hasOwn(movement, "start"), false);
  assert.equal(Object.hasOwn(movement, "end"), false);
  assert.deepEqual(
    raw.latest_events.ordered_event_ids,
    plan.events.map(({ eventId }) => eventId),
  );
  assert.deepEqual(
    raw.latest_events.ordered_event_kinds,
    plan.events.map(({ eventType }) => eventType),
  );
});

test("moving Warrior Charge activation stays a direct transition-start underlay", async () => {
  const raw = movingOracleRaw();
  oracleTrajectoryForClass(raw, 2).transition_start.position = [3.2, 3.4];
  oracleTrajectoryForClass(raw, 4).transition_start.position = [16.8, 3.4];
  installOracleEvents(raw, [
    {
      event_kind: "ability_activated",
      ability_component: "ultimate",
      source_anchor: oracleAnchor(raw, 2, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 4, "transition_start"),
    },
  ]);
  /** @param {number} classId */
  const successorBody = (classId) => {
    const agent = raw.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ candidate) => candidate.class_id === classId,
    );
    assert.ok(agent);
    const center = projectedTrajectoryPoint(raw, classId, "successor");
    const extent = Number(agent.radius) * 10 + 3;
    return Object.freeze({
      layoutKey: `body:successor:${classId}`,
      protectedKind: "body",
      ownerPresentationKey: agent.presentation_key,
      bounds: Object.freeze({
        left: center.x - extent,
        top: center.y - extent,
        right: center.x + extent,
        bottom: center.y + extent,
        width: extent * 2,
        height: extent * 2,
      }),
    });
  };
  const successorBodies = Object.freeze([successorBody(2), successorBody(4)]);
  const movingSurface = Object.freeze({
    ...surface,
    protectedRects: Object.freeze([
      ...successorBodies,
      Object.freeze({
        layoutKey: "durable:route-wall-top",
        bounds: Object.freeze({
          left: 120,
          top: 0,
          right: 145,
          bottom: 30,
          width: 25,
          height: 30,
        }),
      }),
      Object.freeze({
        layoutKey: "durable:route-wall-bottom",
        bounds: Object.freeze({
          left: 120,
          top: 38,
          right: 145,
          bottom: 200,
          width: 25,
          height: 162,
        }),
      }),
    ]),
  });
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const unobstructedPlan = buildChoreographyPlan(frame, surface);
  assert.ok(unobstructedPlan);
  const unobstructedActivation = unobstructedPlan.events[0];
  const plan = buildChoreographyPlan(frame, movingSurface);
  assert.ok(plan);
  const activation = plan.events[0];
  const startSource = projectedTrajectoryPoint(raw, 2, "transition_start");
  const startTarget = projectedTrajectoryPoint(raw, 4, "transition_start");

  assert.equal(activation.tokenId, "warrior_charge");
  assert.equal(activation.sourceEndpointPhase, "transition_start");
  assert.equal(activation.targetEndpointPhase, "transition_start");
  assert.equal(activation.endpointPhase, "transition_start");
  assert.notDeepEqual(startSource, projectedTrajectoryPoint(raw, 2, "successor"));
  assert.notDeepEqual(startTarget, projectedTrajectoryPoint(raw, 4, "successor"));
  assert.equal(unobstructedActivation.route?.kind, "curve");
  assert.deepEqual(unobstructedActivation.route?.start, startSource);
  assert.deepEqual(unobstructedActivation.route?.end, startTarget);
  assert.deepEqual(activation.source, startSource);
  assert.deepEqual(activation.target, startTarget);
  const { markerProgresses: activationMarkerProgresses, ...activationRoute } =
    activation.route;
  const { markerProgresses: unobstructedMarkerProgresses, ...unobstructedRoute } =
    unobstructedActivation.route;
  assert.deepEqual(activationRoute, unobstructedRoute);
  assert.deepEqual(unobstructedMarkerProgresses, [1 / 3, 2 / 3]);
  assert.deepEqual(activationMarkerProgresses, [3 / 4]);
  assert.equal(activation.route?.kind, "curve");
  assert.equal(activation.route?.offset, 0);
  assert.equal(activation.route?.lane, 0);
  assert.deepEqual(activation.route?.bridgeGaps, []);
  assert.deepEqual(activation.route?.start, startSource);
  assert.deepEqual(activation.route?.end, startTarget);
  assert.equal(activation.ownershipCueCollisionFree, true);
});

test("activation trajectories and impacts ignore foreground collision allocation", async () => {
  const raw = movingOracleRaw();
  installOracleEvents(raw, [
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: oracleAnchor(raw, 1, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 3, "transition_start"),
    },
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: oracleAnchor(raw, 1, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 3, "transition_start"),
    },
    {
      event_kind: "ability_activated",
      ability_component: "ultimate",
      source_anchor: oracleAnchor(raw, 5, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 2, "transition_start"),
    },
    {
      event_kind: "ability_activated",
      ability_component: "basic",
      source_anchor: oracleAnchor(raw, 5, "transition_start"),
      recipient_anchor: oracleAnchor(raw, 4, "transition_start"),
    },
  ]);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const bodyProtectedRects = raw.current_endpoint.scene.agents.map(
    (/** @type {Record<string, any>} */ agent) => {
      const center = projectedTrajectoryPoint(raw, agent.class_id, "successor");
      const extent = Number(agent.radius) * 10 + 3;
      return Object.freeze({
        layoutKey: `body:${agent.presentation_key}`,
        protectedKind: "body",
        ownerPresentationKey: agent.presentation_key,
        bounds: Object.freeze({
          left: center.x - extent,
          top: center.y - extent,
          right: center.x + extent,
          bottom: center.y + extent,
          width: extent * 2,
          height: extent * 2,
        }),
      });
    },
  );
  const unobstructedSurface = Object.freeze({
    ...surface,
    protectedRects: Object.freeze(bodyProtectedRects),
  });
  const unobstructed = buildChoreographyPlan(frame, unobstructedSurface);
  assert.ok(unobstructed);
  const unobstructedActivations = unobstructed.events.filter(
    (event) => event.kind === "activation",
  );
  assert.equal(unobstructedActivations.length, 4);
  assert.deepEqual(
    unobstructedActivations.map(({ tokenId }) => tokenId),
    ["basic_damage", "basic_damage", "holy_word", "basic_heal"],
  );
  assert.equal(
    unobstructedActivations[0].route.path,
    unobstructedActivations[1].route.path,
  );
  assert.equal(unobstructedActivations[0].routeMultiplicity, 2);
  assert.equal(unobstructedActivations[1].routeMultiplicity, 2);
  for (const event of unobstructedActivations) {
    assert.equal(event.route.kind, "curve");
    assert.equal(event.route.offset, 0);
    assert.equal(event.route.lane, 0);
    assert.deepEqual(event.route.bridgeGaps, []);
    assert.notDeepEqual(event.route.start, event.source);
    assert.notDeepEqual(event.route.end, event.target);
    assert.equal(event.impactCue ?? null, null);
    assert.equal(event.impactBounds ?? null, null);
    assert.equal(event.impactLeader ?? null, null);
    assert.equal(event.impactLayoutKey ?? null, null);
    assert.equal(event.impactDisposition ?? null, null);
    assert.equal(event.impactCueCollisionFree ?? null, null);
  }

  const protectedRects = unobstructedActivations.map((event, index) => {
    const route = event.route;
    const progress = 0.5;
    const remainder = 1 - progress;
    const center = {
      x:
        remainder * remainder * route.start.x +
        2 * remainder * progress * route.control.x +
        progress * progress * route.end.x,
      y:
        remainder * remainder * route.start.y +
        2 * remainder * progress * route.control.y +
        progress * progress * route.end.y,
    };
    return Object.freeze({
      layoutKey: `foreground:${index}`,
      bounds: Object.freeze({
        left: center.x - 4,
        top: center.y - 4,
        right: center.x + 4,
        bottom: center.y + 4,
        width: 8,
        height: 8,
      }),
    });
  });
  const obstructed = buildChoreographyPlan(
    frame,
    Object.freeze({
      ...surface,
      protectedRects: Object.freeze([...bodyProtectedRects, ...protectedRects]),
    }),
  );
  assert.ok(obstructed);
  const obstructedActivations = obstructed.events.filter(
    (event) => event.kind === "activation",
  );
  assert.deepEqual(
    obstructedActivations.map(({ route }) => route),
    unobstructedActivations.map(({ route }) => route),
  );
  assert.equal(
    obstructedActivations.every(
      (event) =>
        (event.impactCue ?? null) === null &&
        (event.impactBounds ?? null) === null &&
        (event.impactLeader ?? null) === null &&
        (event.impactLayoutKey ?? null) === null &&
        (event.impactDisposition ?? null) === null &&
        (event.impactCueCollisionFree ?? null) === null,
    ),
    true,
  );
});

test("Charge displacement remains authorized but creates no visual overlay", async () => {
  const raw = movingOracleRaw();
  const trajectory = oracleTrajectoryForClass(raw, 2);
  installOracleEvents(raw, [
    {
      event_kind: "charge_phase_displacement",
      realized_displacement: [
        trajectory.post_charge.position[0] - trajectory.transition_start.position[0],
        trajectory.post_charge.position[1] - trajectory.transition_start.position[1],
      ],
      start_anchor: oracleAnchor(raw, 2, "transition_start"),
      end_anchor: oracleAnchor(raw, 2, "post_charge"),
    },
  ]);
  const sourceAgent = raw.current_endpoint.scene.agents.find(
    (/** @type {Record<string, any>} */ candidate) => candidate.class_id === 2,
  );
  assert.ok(sourceAgent);
  const scientificStart = projectedTrajectoryPoint(raw, 2, "transition_start");
  const scientificEnd = projectedTrajectoryPoint(raw, 2, "post_charge");
  const protectedExtent = Number(sourceAgent.radius) * 10 + 4;
  const targetBounds = Object.freeze({
    left: scientificEnd.x - protectedExtent,
    top: scientificEnd.y - protectedExtent,
    right: scientificEnd.x + protectedExtent,
    bottom: scientificEnd.y + protectedExtent,
    width: protectedExtent * 2,
    height: protectedExtent * 2,
  });
  const targetProtectedKey = "durable:charge-post-charge-body";
  const roleSurface = Object.freeze({
    ...surface,
    protectedRects: Object.freeze([
      Object.freeze({
        layoutKey: targetProtectedKey,
        protectedKind: "body",
        ownerPresentationKey: sourceAgent.presentation_key,
        bounds: targetBounds,
      }),
    ]),
  });
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const plan = buildChoreographyPlan(frame, roleSurface);
  assert.ok(plan);
  const displacement = plan.events[0];

  assert.equal(displacement.eventType, "charge_phase_displacement");
  assert.equal(displacement.kind, "feed_only");
  assert.equal(displacement.spatial, false);
  const allocatorFields = [
    "start",
    "end",
    "sourcePresentationKey",
    "sourcePublicAgentId",
    "paintParts",
    "persistent",
    "route",
    "routeLayoutKey",
    "routeLane",
    "routeBridgeGaps",
    "startCue",
    "startBounds",
    "startCueBounds",
    "startLeader",
    "startCueLeader",
    "startLayoutKey",
    "startCueLayoutKey",
    "startDisposition",
    "startCueDisposition",
    "startCueCollisionFree",
    "endCue",
    "endBounds",
    "endCueBounds",
    "endLeader",
    "endCueLeader",
    "endLayoutKey",
    "endCueLayoutKey",
    "endDisposition",
    "endCueDisposition",
    "endCueCollisionFree",
  ];
  assert.equal(
    allocatorFields.every((field) => !Object.hasOwn(displacement, field)),
    true,
  );

  const routeMidpoint = {
    x: (scientificStart.x + scientificEnd.x) / 2,
    y: (scientificStart.y + scientificEnd.y) / 2,
  };
  const blocker = Object.freeze({
    layoutKey: "durable:charge-displacement-blocker",
    bounds: Object.freeze({
      left: routeMidpoint.x - 2,
      top: routeMidpoint.y - 2,
      right: routeMidpoint.x + 2,
      bottom: routeMidpoint.y + 2,
      width: 4,
      height: 4,
    }),
  });
  const blockedPlan = buildChoreographyPlan(
    frame,
    Object.freeze({
      ...roleSurface,
      protectedRects: Object.freeze([...roleSurface.protectedRects, blocker]),
    }),
  );
  assert.ok(blockedPlan);
  const blockedDisplacement = blockedPlan.events[0];
  assert.deepEqual(blockedDisplacement, displacement);
  assert.equal(
    allocatorFields.every((field) => !Object.hasOwn(blockedDisplacement, field)),
    true,
  );
});

test("combat reset stays feed-only while successor regeneration cues pack deterministically", async () => {
  const resetRaw = movingOracleRaw();
  installOracleEvents(resetRaw, [
    {
      event_kind: "combat_countdown_reset",
      agent_anchor: oracleAnchor(resetRaw, 1, "transition_start"),
    },
  ]);
  const resetFrame = await normalizeAuthorizedPresentationFrameV1(resetRaw);
  const resetPlan = buildChoreographyPlan(resetFrame, surface);
  assert.ok(resetPlan);
  assert.equal(resetPlan.events.length, 1);
  assert.equal(resetPlan.events[0].eventId, resetRaw.latest_events.events[0].event_id);
  assert.equal(resetPlan.events[0].eventType, "combat_countdown_reset");
  assert.equal(resetPlan.events[0].kind, "feed_only");
  assert.equal(resetPlan.events[0].spatial, false);
  assert.equal(Object.hasOwn(resetPlan.events[0], "cueSemantic"), false);
  assert.equal(Object.hasOwn(resetPlan.events[0], "anchor"), false);
  assert.equal(resetPlan.phases.total, 0);

  const raw = movingOracleRaw();
  installOracleEvents(raw, [
    {
      event_kind: "recipient_health_resolution",
      recipient_anchor: oracleAnchor(raw, 4, "transition_start"),
      transition_start_health: 100,
      total_effective_damage: 5,
      total_effective_healing: 0,
      health_after_combat_resolution: 95,
      realized_net_health_change: -5,
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: oracleAnchor(raw, 1, "transition_start"),
      actual_health_regenerated: 4,
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: oracleAnchor(raw, 2, "transition_start"),
      actual_health_regenerated: 2,
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: oracleAnchor(raw, 5, "transition_start"),
      actual_health_regenerated: 1,
    },
    oracleStatusEvent(raw, "status_applied", {
      recipientClass: 4,
      sourceClass: 3,
    }),
  ]);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const protectedRect = Object.freeze({
    left: 0,
    top: 100,
    right: 100,
    bottom: 200,
    width: 100,
    height: 100,
  });
  const packedSurface = Object.freeze({
    ...surface,
    protectedRects: Object.freeze([protectedRect]),
  });
  const plan = buildChoreographyPlan(frame, packedSurface);
  const repeated = buildChoreographyPlan(frame, packedSurface);
  assert.ok(plan);
  assert.ok(repeated);
  const regenerations = plan.events.filter((event) => event.kind === "regeneration");
  const repeatedRegenerations = repeated.events.filter(
    (event) => event.kind === "regeneration",
  );
  assert.equal(regenerations.length, 3);
  assert.deepEqual(
    regenerations.map(({ cue, cueBounds }) => ({ cue, cueBounds })),
    repeatedRegenerations.map(({ cue, cueBounds }) => ({ cue, cueBounds })),
  );
  assert.deepEqual(
    regenerations.map(({ value }) => value),
    [4, 2, 1],
  );
  assert.deepEqual(
    regenerations.map(({ recipient }) => recipient),
    [1, 2, 5].map((classId) => projectedTrajectoryPoint(raw, classId, "successor")),
  );
  assert.equal(
    regenerations.every(
      (event) =>
        event.spatial === true &&
        event.cueCollisionFree === true &&
        event.spatialDisposition === "rendered" &&
        event.cue !== null &&
        event.cueBounds !== null,
    ),
    true,
  );
  const intersectionArea = (
    /** @type {Record<string, number>} */ left,
    /** @type {Record<string, number>} */ right,
  ) =>
    Math.max(0, Math.min(left.right, right.right) - Math.max(left.left, right.left)) *
    Math.max(0, Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top));
  for (const [index, event] of regenerations.entries()) {
    assert.equal(intersectionArea(event.cueBounds, protectedRect), 0);
    for (const prior of regenerations.slice(0, index)) {
      assert.equal(intersectionArea(event.cueBounds, prior.cueBounds), 0);
    }
  }
  const health = plan.events.find((event) => event.kind === "net_health");
  const status = plan.events.find((event) => event.kind === "status_lifecycle");
  assert.ok(health);
  assert.ok(status);
  assert.ok(health.phaseEnd <= regenerations[0].phaseStart);
  assert.equal(
    regenerations.every(
      (event) =>
        event.phaseStart === regenerations[0].phaseStart &&
        event.phaseEnd === regenerations[0].phaseEnd,
    ),
    true,
  );
  assert.ok(regenerations[0].phaseEnd <= status.phaseStart);
});

test("regeneration cue leaders allow only their exact protected agent body", async () => {
  const raw = movingOracleRaw();
  installOracleEvents(raw, [
    {
      event_kind: "health_regenerated",
      agent_anchor: oracleAnchor(raw, 4, "transition_start"),
      actual_health_regenerated: 4,
    },
  ]);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const trajectory = oracleTrajectoryForClass(raw, 4);
  const anchor = projectedTrajectoryPoint(raw, 4, "successor");
  const bodyBounds = Object.freeze({
    left: anchor.x - 17,
    top: anchor.y - 17,
    right: anchor.x + 17,
    bottom: anchor.y + 17,
    width: 34,
    height: 34,
  });
  const protectedSurface = Object.freeze({
    ...surface,
    protectedRects: Object.freeze([
      Object.freeze({
        layoutKey: JSON.stringify(["body", trajectory.agent_presentation_key]),
        protectedKind: "body",
        ownerPresentationKey: trajectory.agent_presentation_key,
        bounds: bodyBounds,
      }),
    ]),
  });

  const plan = buildChoreographyPlan(frame, protectedSurface);
  assert.ok(plan);
  const regeneration = plan.events.find((event) => event.kind === "regeneration");
  assert.ok(regeneration);
  assert.deepEqual(regeneration.recipient, anchor);
  assert.deepEqual(regeneration.cueLeader?.start, anchor);
  assert.equal(regeneration.cueCollisionFree, true);
});

test("Oracle lifecycle cues retain successor and strict team-anchor authority", async () => {
  const raw = movingOracleRaw();
  const deathAnchor = oracleAnchor(raw, 4, "successor");
  const shieldAnchor = oracleAnchor(raw, 1, "successor");
  const respawnAnchor = oracleAnchor(raw, 3, "successor");
  installOracleEvents(raw, [
    {
      event_kind: "agent_died",
      recipient_anchor: deathAnchor,
    },
    {
      event_kind: "spawn_shield_expired",
      agent_anchor: shieldAnchor,
    },
    {
      event_kind: "respawn_wave_occurred",
      team_anchor: { phase: "successor", team_index: 0, team_id: 1 },
    },
    {
      event_kind: "respawn_wave_occurred",
      team_anchor: { phase: "successor", team_index: 1, team_id: 2 },
    },
    {
      event_kind: "agent_respawned",
      agent_anchor: respawnAnchor,
      team_id: 2,
      realized_successor_position: structuredClone(respawnAnchor.position),
    },
  ]);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const rows = authorizedPresentationIncomingRows(frame);
  const rowsBefore = JSON.stringify(rows);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(JSON.stringify(authorizedPresentationIncomingRows(frame)), rowsBefore);
  assert.deepEqual(
    plan.events.map(({ eventId, eventType }) => [eventId, eventType]),
    rows.map(({ id, kind }) => [id, kind]),
  );

  const death = plan.events[0];
  assert.equal(death.kind, "semantic_pulse");
  assert.equal(death.cueSemantic, "agent_died");
  assert.equal(death.persistent, true);
  assert.deepEqual(death.anchor, projectedTrajectoryPoint(raw, 4, "successor"));
  assert.equal(death.agentPresentationKey, deathAnchor.presentation_key);
  assert.equal(death.agentPublicAgentId, deathAnchor.public_agent_id);

  const shield = plan.events[1];
  assert.equal(shield.kind, "status_lifecycle");
  assert.equal(shield.eventType, "spawn_shield_expired");
  assert.equal(shield.tokenId, "spawn_shield");
  assert.equal(shield.lifecycle, "expired");
  assert.deepEqual(shield.recipient, projectedTrajectoryPoint(raw, 1, "successor"));
  assert.equal(shield.recipientPresentationKey, shieldAnchor.presentation_key);
  assert.equal(shield.recipientPublicAgentId, shieldAnchor.public_agent_id);

  const waves = plan.events.slice(2, 4);
  assert.deepEqual(
    waves.map(
      ({ cueSemantic, teamIndex, teamId, teamSide, label, anchor, persistent }) => ({
        cueSemantic,
        teamIndex,
        teamId,
        teamSide,
        label,
        anchor,
        persistent,
      }),
    ),
    [
      {
        cueSemantic: "respawn_wave_occurred",
        teamIndex: 0,
        teamId: 1,
        teamSide: "left",
        label: "EVENT: Team A Respawn",
        anchor: { x: 112, y: 24 },
        persistent: true,
      },
      {
        cueSemantic: "respawn_wave_occurred",
        teamIndex: 1,
        teamId: 2,
        teamSide: "right",
        label: "EVENT: Team B Respawn",
        anchor: { x: 528, y: 24 },
        persistent: true,
      },
    ],
  );
  assert.ok(waves.every((wave) => !Object.hasOwn(wave, "agentPresentationKey")));
  assert.equal(waves[0].phaseStart, waves[1].phaseStart);
  assert.equal(waves[0].phaseEnd, waves[1].phaseEnd);

  const respawn = plan.events[4];
  assert.equal(respawn.kind, "semantic_pulse");
  assert.equal(respawn.cueSemantic, "agent_respawned");
  assert.equal(respawn.persistent, true);
  assert.deepEqual(respawn.anchor, projectedTrajectoryPoint(raw, 3, "successor"));
  assert.equal(respawn.agentPresentationKey, respawnAnchor.presentation_key);
  assert.equal(respawn.agentPublicAgentId, respawnAnchor.public_agent_id);
  assert.equal(respawn.phaseEnd, plan.phases.total);
  for (const lifecycleEvent of [death, respawn]) {
    for (const allocationField of [
      "cue",
      "cueLayoutKey",
      "cueBounds",
      "cueDisposition",
      "cueCollisionFree",
      "cueLeader",
    ]) {
      assert.equal(Object.hasOwn(lifecycleEvent, allocationField), false);
    }
  }
  assert.ok(shield.cue);
  assert.ok(waves.every((wave) => wave.cue));

  const lifecycleCopy = [death, ...waves, respawn].map((event) =>
    explainChoreographyEvent(event),
  );
  assert.deepEqual(
    lifecycleCopy.map(({ title }) => title),
    ["Agent Died", "EVENT: Team A Respawn", "EVENT: Team B Respawn", "Agent Respawned"],
  );
  assert.doesNotMatch(
    JSON.stringify(
      lifecycleCopy.map(({ title, summary, rows }) => ({ title, summary, rows })),
    ),
    /agent_died|agent_respawned|respawn_wave_occurred|semantic pulse|authoritative semantic/iu,
  );
  assert.equal(plan.bounds.persistentNodes, 28);
});

test("NoShared and Shared authorized clocks never synthesize a respawn wave", async () => {
  for (const kind of [
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const frame = await normalized(kind);
    const serializedBefore = JSON.stringify(frame);
    const plan = buildChoreographyPlan(frame, surface);
    assert.ok(plan, kind);
    assert.equal(JSON.stringify(frame), serializedBefore, kind);
    assert.equal(
      plan.events.some(
        (event) =>
          event.eventType === "respawn_wave_occurred" ||
          event.cueSemantic === "respawn_wave_occurred",
      ),
      false,
      kind,
    );
    assert.ok(
      plan.events.every((event) => event.authorityVocabulary === "event"),
      kind,
    );
  }
});

test("rejection remains serialized feed evidence without spatial choreography", async () => {
  const raw = structuredClone(fixture.state_cases.replay_oracle_final_selected);
  const rejection = raw.latest_events.events[0];
  const trajectory = raw.latest_events.agent_phase_trajectories.find(
    (/** @type {Record<string, any>} */ candidate) =>
      candidate.agent_presentation_key === rejection.actor_anchor.presentation_key,
  );
  assert.ok(trajectory);
  trajectory.transition_start.position = [3.2, 3.1];
  rejection.actor_anchor = structuredClone(trajectory.transition_start);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(plan.events.length, 1);
  assert.equal(plan.events[0].eventId, rejection.event_id);
  assert.equal(plan.events[0].kind, "feed_only");
  assert.equal(plan.events[0].feedKind, "rejected_action");
  assert.equal(plan.events[0].spatial, false);
  assert.equal(plan.events[0].presentationSuppressed, true);
  assert.equal(Object.hasOwn(plan.events[0], "actor"), false);
  assert.equal(Object.hasOwn(plan.events[0], "route"), false);
});

test("missing, duplicate, phase-discordant, and identity-discordant trajectories fail closed", async () => {
  /** @type {ReadonlyArray<readonly [string, (raw: Record<string, any>) => void]>} */
  const mutations = [
    [
      "missing",
      (raw) => {
        raw.latest_events.agent_phase_trajectories.pop();
      },
    ],
    [
      "duplicate",
      (raw) =>
        raw.latest_events.agent_phase_trajectories.push(
          structuredClone(raw.latest_events.agent_phase_trajectories[0]),
        ),
    ],
    [
      "phase discordant",
      (raw) => {
        raw.latest_events.agent_phase_trajectories[0].transition_start.phase =
          "successor";
      },
    ],
    [
      "identity discordant",
      (raw) => {
        raw.latest_events.agent_phase_trajectories[0].transition_start.public_agent_id =
          "unique-discordant-public-agent";
      },
    ],
  ];
  for (const [label, mutate] of mutations) {
    const raw = structuredClone(fixture.presentations.replay_oracle);
    mutate(raw);
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(raw), TypeError, label);
  }
});

test("Oracle status compositor applies exact precedence and preserves every atomic identity", async () => {
  const cases = [
    {
      label: "apply",
      kinds: ["status_applied"],
      lifecycle: "applied",
      primaryIndex: 0,
    },
    {
      label: "multiple apply",
      kinds: ["status_applied", "status_applied"],
      lifecycle: "applied",
      primaryIndex: 0,
    },
    {
      label: "refresh",
      kinds: ["status_refreshed_or_extended"],
      lifecycle: "applied",
      primaryIndex: 0,
    },
    {
      label: "refresh plus apply",
      kinds: ["status_applied", "status_refreshed_or_extended"],
      lifecycle: "applied",
      primaryIndex: 0,
    },
    {
      label: "break",
      kinds: ["status_broken_by_damage"],
      lifecycle: "trap_broken",
      primaryIndex: 0,
    },
    {
      label: "break plus apply",
      kinds: ["status_broken_by_damage", "status_applied"],
      lifecycle: "applied",
      primaryIndex: 1,
    },
    {
      label: "age",
      kinds: ["status_aged_to_zero"],
      lifecycle: "expired",
      primaryIndex: 0,
    },
    {
      label: "age plus apply",
      kinds: ["status_aged_to_zero", "status_applied"],
      lifecycle: "applied",
      primaryIndex: 1,
    },
    {
      label: "death clear precedence",
      kinds: ["status_aged_to_zero", "status_applied", "status_cleared_by_new_death"],
      lifecycle: "cleared_by_death",
      primaryIndex: 2,
    },
  ];

  for (const expected of cases) {
    const raw = structuredClone(fixture.presentations.replay_oracle);
    const events = expected.kinds.map((kind, index) =>
      oracleStatusEvent(raw, kind, {
        sourceClass: expected.label === "multiple apply" ? [3, 5][index] : 3,
      }),
    );
    installOracleEvents(raw, events);
    const frame = await normalizeAuthorizedPresentationFrameV1(raw);
    const serializedBefore = JSON.stringify(frame);
    const incomingRows = authorizedPresentationIncomingRows(frame);
    const incomingRowsBefore = JSON.stringify(incomingRows);
    const plan = buildChoreographyPlan(frame, surface);
    assert.ok(plan, expected.label);
    assert.equal(JSON.stringify(frame), serializedBefore, expected.label);
    assert.equal(
      JSON.stringify(authorizedPresentationIncomingRows(frame)),
      incomingRowsBefore,
      expected.label,
    );
    const atomicEventIds = incomingRows.map(({ id }) => id);
    const applicationRows = incomingRows.filter(
      ({ kind }) => kind === "status_applied",
    );
    const scene = authorizedPresentationSceneView(frame);
    assert.ok(scene);
    const expectedApplicationSources = applicationRows.map(({ id, payload }) => {
      const sourceAgent = scene.agents.find(
        (/** @type {Record<string, any>} */ agent) =>
          agent.presentation_key === payload.source_anchor.presentation_key &&
          agent.public_agent_id === payload.source_anchor.public_agent_id,
      );
      assert.ok(sourceAgent);
      return {
        eventId: id,
        sourcePresentationKey: payload.source_anchor.presentation_key,
        sourcePublicAgentId: payload.source_anchor.public_agent_id,
        sourceIdentity: {
          presentation_key: sourceAgent.presentation_key,
          public_agent_id: sourceAgent.public_agent_id,
          class_id: sourceAgent.class_id,
          team_id: sourceAgent.team_id,
        },
      };
    });
    assert.equal(plan.events.length, 1, expected.label);
    assert.equal(plan.bounds.nodes, 33, expected.label);
    const [lifecycle] = plan.events;
    assert.deepEqual(
      [lifecycle.eventId, lifecycle.eventType],
      [atomicEventIds[expected.primaryIndex], incomingRows[expected.primaryIndex].kind],
      expected.label,
    );
    assert.equal(lifecycle.kind, "status_lifecycle", expected.label);
    assert.equal(lifecycle.lifecycle, expected.lifecycle, expected.label);
    assert.deepEqual(lifecycle.atomicEventIds, atomicEventIds, expected.label);
    assert.deepEqual(
      lifecycle.applicationEventIds,
      applicationRows.map(({ id }) => id),
      expected.label,
    );
    assert.deepEqual(
      lifecycle.applicationSources,
      expectedApplicationSources,
      expected.label,
    );
    assert.equal(lifecycle.presentationSuppressed, false, expected.label);
    assert.equal(lifecycle.spatial, true, expected.label);
    assert.ok(
      lifecycle.applicationSources.every(
        (/** @type {Record<string, any>} */ source) =>
          typeof source.sourcePresentationKey === "string" &&
          typeof source.sourcePublicAgentId === "string" &&
          typeof source.sourceIdentity === "object" &&
          !Object.hasOwn(source, "source") &&
          !Object.hasOwn(source, "sourceSlot"),
      ),
      expected.label,
    );
    if (expectedApplicationSources.length > 0) {
      const sourceRow = explainChoreographyEvent(lifecycle).rows.find(
        (/** @type {Record<string, any>} */ row) =>
          row.label ===
          (expectedApplicationSources.length === 1 ? "Source" : "Sources"),
      );
      assert.deepEqual(
        sourceRow,
        {
          label: expectedApplicationSources.length === 1 ? "Source" : "Sources",
          value: expectedApplicationSources
            .map(({ sourceIdentity }) =>
              expectedAuthorizedIdentityTitle(sourceIdentity),
            )
            .join("; "),
          metadata: { compact: true, full: true },
        },
        expected.label,
      );
    }
    assertRecursivelyFrozen(lifecycle);
    if (expected.lifecycle === "applied") {
      assert.equal(lifecycle.lifecycleToken.label, "Applied");
      assert.equal(lifecycle.lifecycleToken.accessibleName, "Status applied");
      assert.equal(lifecycle.lifecycleToken.glyphKey, "lifecycle-applied");
      assert.doesNotMatch(JSON.stringify(lifecycle.lifecycleToken), /expir/iu);
    }
  }
});

test("status groups keep first-atomic plan order and use the nearest collision-safe layout when precedence favors a later group", async () => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const events = [
    oracleStatusEvent(raw, "status_applied", { channel: 8, sourceClass: 5 }),
    oracleStatusEvent(raw, "status_aged_to_zero", { channel: 4 }),
    oracleStatusEvent(raw, "status_applied", { channel: 4 }),
    oracleStatusEvent(raw, "status_applied", { channel: 0 }),
    oracleStatusEvent(raw, "status_applied", { channel: 1 }),
  ];
  installOracleEvents(raw, events);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const incomingRows = authorizedPresentationIncomingRows(frame);
  const serializedBefore = JSON.stringify(frame);
  const incomingRowsBefore = JSON.stringify(incomingRows);
  const layoutSurface = Object.freeze({
    ...surface,
    /** @param {readonly [number, number] | {x: number, y: number}} point */
    worldToScreen: (point) => ({
      x: ("x" in point ? Number(point.x) : Number(point[0])) * 10 + 200,
      y: ("y" in point ? Number(point.y) : Number(point[1])) * 10 + 160,
    }),
  });
  const plan = buildChoreographyPlan(frame, layoutSurface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(
    JSON.stringify(authorizedPresentationIncomingRows(frame)),
    incomingRowsBefore,
  );
  assert.equal(plan.events.length, 4);
  assert.equal(plan.bounds.nodes, 123);
  assert.deepEqual(
    plan.events.map(({ eventId }) => eventId),
    [incomingRows[0].id, incomingRows[2].id, incomingRows[3].id, incomingRows[4].id],
  );
  assert.deepEqual(
    plan.events.map(({ lane, laneCount, statusLayoutOrder }) => [
      lane,
      laneCount,
      statusLayoutOrder,
    ]),
    [
      [0, 4, 0],
      [1, 4, 1],
      [2, 4, 3],
      [3, 4, 4],
    ],
  );
  assert.deepEqual(plan.events[0].atomicEventIds, [incomingRows[0].id]);
  assert.deepEqual(plan.events[1].atomicEventIds, [
    incomingRows[1].id,
    incomingRows[2].id,
  ]);
  assert.equal(plan.events[0].lifecycle, "applied");
  assert.equal(plan.events[1].lifecycle, "applied");
  const earlierApply = plan.events[0];
  const laterApplication = plan.events[1];
  assert.ok(earlierApply.cue);
  assert.ok(laterApplication.cue);
  assert.ok(earlierApply.cueBounds);
  assert.ok(laterApplication.cueBounds);
  assert.equal(earlierApply.cueCollisionFree, true);
  assert.equal(laterApplication.cueCollisionFree, true);
  assert.equal(
    earlierApply.cueLayoutKey,
    JSON.stringify(["event", incomingRows[0].id, "cue"]),
  );
  assert.equal(
    laterApplication.cueLayoutKey,
    JSON.stringify(["event", incomingRows[2].id, "cue"]),
  );
  const cueDistance = (/** @type {Record<string, any>} */ event) =>
    Math.hypot(event.cue.x - event.recipient.x, event.cue.y - event.recipient.y);
  assert.ok(Math.abs(cueDistance(earlierApply) - cueDistance(laterApplication)) < 1e-9);
  assert.equal(earlierApply.cueDisposition, "recipient_stack");
  assert.equal(laterApplication.cueDisposition, "recipient_stack");
  const overlapWidth = Math.max(
    0,
    Math.min(earlierApply.cueBounds.right, laterApplication.cueBounds.right) -
      Math.max(earlierApply.cueBounds.left, laterApplication.cueBounds.left),
  );
  const overlapHeight = Math.max(
    0,
    Math.min(earlierApply.cueBounds.bottom, laterApplication.cueBounds.bottom) -
      Math.max(earlierApply.cueBounds.top, laterApplication.cueBounds.top),
  );
  assert.equal(overlapWidth * overlapHeight, 0);
});

test("durable current status without an atomic status event creates no lifecycle cue", async () => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  installOracleEvents(raw, []);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  assert.ok(
    authorizedPresentationSceneView(frame)?.agents.some(
      (/** @type {Record<string, any>} */ agent) =>
        Array.isArray(agent.statuses) && agent.statuses.length > 0,
    ),
  );
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.deepEqual(plan.events, []);
  assert.equal(plan.phases.total, 0);
});

test("canonical in-combat expiry paints one self-contained duration expiration", async () => {
  const raw = movingOracleRaw();
  const agentAnchor = oracleAnchor(raw, 2, "successor");
  installOracleEvents(raw, [
    {
      event_kind: "agent_left_combat",
      agent_anchor: agentAnchor,
    },
  ]);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const firstPlan = buildChoreographyPlan(frame, surface);
  const directSeekPlan = buildChoreographyPlan(frame, surface);
  assert.ok(firstPlan);
  assert.ok(directSeekPlan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.deepEqual(directSeekPlan, firstPlan);
  assert.equal(firstPlan.events.length, 1);
  const [expiration] = firstPlan.events;
  assert.equal(expiration.eventType, "agent_left_combat");
  assert.equal(expiration.kind, "status_lifecycle");
  assert.equal(expiration.tokenId, "in_combat");
  assert.equal(expiration.token.glyphKey, "combat-in-progress");
  assert.equal(expiration.lifecycle, "expired");
  assert.equal(expiration.durationBefore, 1);
  assert.equal(expiration.durationAfter, 0);
  assert.deepEqual(expiration.applicationSources, []);
  assert.deepEqual(expiration.applicationEventIds, []);
  assert.deepEqual(expiration.atomicEventIds, [raw.latest_events.ordered_event_ids[0]]);
  assert.deepEqual(expiration.recipient, projectedTrajectoryPoint(raw, 2, "successor"));
  assert.equal(expiration.recipientPresentationKey, agentAnchor.presentation_key);
  assert.equal(expiration.recipientPublicAgentId, agentAnchor.public_agent_id);
  const recipientAgent = authorizedPresentationSceneView(frame)?.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key === agentAnchor.presentation_key &&
      agent.public_agent_id === agentAnchor.public_agent_id,
  );
  assert.ok(recipientAgent);
  assert.equal(
    expiration.outOfCombatRegenerationPerTick,
    recipientAgent.maximum_health *
      recipientAgent.out_of_combat_health_regeneration_fraction_per_step,
  );
  const explanation = explainChoreographyEvent(expiration);
  assert.equal(explanation.title, "Out of Combat");
  assert.equal(
    explanation.summary,
    "This agent can regenerate up to 8 health per tick while it remains out of combat.",
  );
  assert.deepEqual(
    explanation.rows.map(({ label, value }) => [label, value]),
    [["Recipient", expectedAuthorizedIdentityTitle(recipientAgent)]],
  );

  const resetRaw = movingOracleRaw();
  installOracleEvents(resetRaw, [
    {
      event_kind: "combat_countdown_reset",
      agent_anchor: oracleAnchor(resetRaw, 2, "transition_start"),
    },
  ]);
  const resetFrame = await normalizeAuthorizedPresentationFrameV1(resetRaw);
  const resetPlan = buildChoreographyPlan(resetFrame, surface);
  assert.ok(resetPlan);
  assert.equal(resetPlan.events.length, 1);
  assert.equal(resetPlan.events[0].kind, "feed_only");
  assert.equal(resetPlan.events[0].spatial, false);
});

test("in-combat expiry retains its canonical phase before concurrent death", async () => {
  const raw = movingOracleRaw();
  installOracleEvents(raw, [
    {
      event_kind: "agent_left_combat",
      agent_anchor: oracleAnchor(raw, 2, "successor"),
    },
    {
      event_kind: "agent_died",
      recipient_anchor: oracleAnchor(raw, 4, "successor"),
    },
  ]);

  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  const expiration = plan.events.find(
    (event) => event.eventType === "agent_left_combat",
  );
  const death = plan.events.find((event) => event.eventType === "agent_died");
  assert.ok(expiration);
  assert.ok(death);
  assert.equal(expiration.kind, "status_lifecycle");
  assert.ok(expiration.phaseStart < death.phaseStart);
  assert.equal(expiration.phaseEnd, death.phaseStart);
});

test("lifecycle metadata retains opaque keys without inventing source routes", async () => {
  const oracle = await normalized("replay_oracle");
  const plan = buildChoreographyPlan(oracle, surface);
  const lifecycle = plan?.events.find(({ kind }) => kind === "status_lifecycle");
  assert.ok(lifecycle);
  assert.equal(lifecycle.applicationSources.length, 1);
  assert.equal(
    lifecycle.applicationSources[0].sourcePresentationKey,
    lifecycle.sourcePresentationKey,
  );
  assert.equal(Object.hasOwn(lifecycle.applicationSources[0], "source"), false);
  assert.equal(Object.hasOwn(lifecycle.applicationSources[0], "sourceSlot"), false);

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
  const firstLegality = explainLegality(
    {
      owner_presentation_key: first.presentation_key,
      owner_public_agent_id: first.public_agent_id,
      lane_0_available: true,
    },
    0,
    first,
  );
  const secondLegality = explainLegality(
    {
      owner_presentation_key: second.presentation_key,
      owner_public_agent_id: second.public_agent_id,
      lane_0_available: true,
    },
    0,
    second,
  );
  assert.ok(firstLegality);
  assert.ok(secondLegality);
  assert.notEqual(firstLegality.id, secondLegality.id);
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
