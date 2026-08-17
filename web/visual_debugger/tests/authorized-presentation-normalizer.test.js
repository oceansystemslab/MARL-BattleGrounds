import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  isJoinedTransportAndAuthorizedPresentationV1,
  isNormalizedAuthorizedPresentationFrameV1,
  isPresentationJoinRace,
  joinReplayTransportAndTimelineV1,
  joinTransportAndAuthorizedPresentationV1,
  normalizeAuthorizedPresentationFrameV1,
  normalizePresentationApiErrorV1,
  normalizeSharedObsAgentPovReplayTimelineTransportV1,
  normalizeSharedObsAgentPovReplayTransportV1,
  validateReplayTransportContinuityV1,
} from "../src/authorized-presentation-normalizer.js";
import { AUTHORIZED_PRESENTATION_SCHEMA_V1 } from "../src/authorized-presentation-schema.js";

const fixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

const kinds = [
  "live_oracle",
  "live_no_shared_obs_agent_pov",
  "replay_oracle",
  "replay_no_shared_obs_agent_pov",
  "replay_shared_obs_agent_pov",
];

/**
 * @template T
 * @param {T} value
 * @returns {T}
 */
function clone(value) {
  return structuredClone(value);
}

/** @param {Record<string, any>} presentation */
function presentationScene(presentation) {
  return (
    presentation.current_endpoint.scene ?? presentation.current_endpoint.parts.scene
  );
}

/** @param {unknown} value */
function assertRecursivelyFrozen(value) {
  if (!value || typeof value !== "object") return;
  assert.equal(Object.isFrozen(value), true);
  for (const child of Object.values(value)) assertRecursivelyFrozen(child);
}

/**
 * @param {any} value
 * @param {(text: string) => string} transform
 * @returns {any}
 */
function transformStrings(value, transform) {
  if (typeof value === "string") return transform(value);
  if (Array.isArray(value)) {
    return value.map((item) => transformStrings(item, transform));
  }
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [
      key,
      transformStrings(item, transform),
    ]),
  );
}

/**
 * @typedef {{
 *   label: string,
 *   makeTransport: (transport: Record<string, any>) => Record<string, any>,
 * }} JoinIdentityMutation
 */

/**
 * @param {string} label
 * @param {(transport: Record<string, any>) => void} mutate
 * @returns {JoinIdentityMutation}
 */
function identityMutation(label, mutate) {
  return {
    label,
    makeTransport(transport) {
      const candidate = clone(transport);
      mutate(candidate);
      return candidate;
    },
  };
}

/**
 * Clone a Python-produced presentation and tripwire its first nested endpoint
 * branch. Identity preflight must reject without touching this property.
 *
 * @param {Record<string, any>} source
 */
function withEndpointReadTripwire(source) {
  const presentation = clone(source);
  const endpoint = presentation.current_endpoint;
  const nestedKey = Object.hasOwn(endpoint, "scene") ? "scene" : "parts";
  let reads = 0;
  Object.defineProperty(endpoint, nestedKey, {
    configurable: true,
    enumerable: true,
    get() {
      reads += 1;
      throw new Error("must not read endpoint after identity rejection");
    },
  });
  return { presentation, readCount: () => reads };
}

/**
 * @param {Record<string, any>} pair
 * @param {Record<string, any>} transport
 * @param {string} label
 */
async function assertJoinRaceBeforeEndpoint(pair, transport, label) {
  const { presentation, readCount } = withEndpointReadTripwire(pair.presentation);
  await assert.rejects(
    joinTransportAndAuthorizedPresentationV1(transport, presentation),
    (error) => {
      assert.equal(isPresentationJoinRace(error), true, label);
      return true;
    },
  );
  assert.equal(readCount(), 0, label);
}

/**
 * @param {Record<string, any>} pair
 * @param {Record<string, any>} transport
 * @param {string} label
 */
async function assertProtocolPoisonBeforeEndpoint(pair, transport, label) {
  const { presentation, readCount } = withEndpointReadTripwire(pair.presentation);
  await assert.rejects(
    joinTransportAndAuthorizedPresentationV1(transport, presentation),
    (error) => {
      assert.equal(error instanceof TypeError, true, label);
      assert.equal(isPresentationJoinRace(error), false, label);
      return true;
    },
  );
  assert.equal(readCount(), 0, label);
}

test("all five exact Python presentation leaves normalize repeatably and freeze", async () => {
  assert.deepEqual(Object.keys(fixture.presentations).sort(), [...kinds].sort());
  for (const kind of kinds) {
    const source = fixture.presentations[kind];
    const before = JSON.stringify(source);
    const first = await normalizeAuthorizedPresentationFrameV1(source);
    const second = await normalizeAuthorizedPresentationFrameV1(source);
    assert.equal(isNormalizedAuthorizedPresentationFrameV1(source), false);
    assert.equal(isNormalizedAuthorizedPresentationFrameV1(first), true);
    assert.equal(isNormalizedAuthorizedPresentationFrameV1({ ...first }), false);
    assert.equal(first.presentation_kind, kind);
    assert.deepEqual(first, second);
    assert.equal(JSON.stringify(source), before);
    assertRecursivelyFrozen(first);
  }
  for (const source of Object.values(fixture.state_cases)) {
    const normalized = await normalizeAuthorizedPresentationFrameV1(source);
    assert.equal(isNormalizedAuthorizedPresentationFrameV1(normalized), true);
    assertRecursivelyFrozen(normalized);
  }
  assertRecursivelyFrozen(AUTHORIZED_PRESENTATION_SCHEMA_V1);
});

test("Python-certified V2 class and shield mechanics project exactly and freeze", async () => {
  const expectedProfile = {
    availability_kind: "available",
    profile_id: "marl_battlegrounds.class_documentation.canonical_v1",
  };
  for (const kind of kinds) {
    const sourceScene = presentationScene(fixture.presentations[kind]);
    const normalized = await normalizeAuthorizedPresentationFrameV1(
      fixture.presentations[kind],
    );
    assert.deepEqual(normalized.scene.class_mechanics, sourceScene.class_mechanics);
    assert.deepEqual(
      normalized.scene.spawn_shield_mechanics,
      sourceScene.spawn_shield_mechanics,
    );
    for (const row of normalized.scene.class_mechanics) {
      assert.equal(row.mechanics_version, 2);
      assert.deepEqual(row.documentation_profile, expectedProfile);
    }
    assert.deepEqual(normalized.scene.spawn_shield_mechanics, {
      availability_kind: "available_v2",
      configured_duration_steps:
        sourceScene.spawn_shield_mechanics.configured_duration_steps,
      movement_speed: sourceScene.spawn_shield_mechanics.movement_speed,
      protection_effect: "invulnerable",
      visibility_effect: "concealed_from_opponents",
      targetability_effect: "untargetable",
      action_scope: "movement_only",
      aura_effect: "excluded_as_emitter_and_beneficiary",
      agent_collision_effect: "phased_until_expiring_endpoint_rejoin",
      ordinary_application_mechanism: "end_of_transition_respawn_lifecycle",
    });
    assertRecursivelyFrozen(normalized.scene.class_mechanics);
    assertRecursivelyFrozen(normalized.scene.spawn_shield_mechanics);
  }
});

test("legacy V1 class and shield mechanics remain exact and gain no V2 claims", async () => {
  const legacy = fixture.compatibility_cases.legacy_v1;
  const normalized = await normalizeAuthorizedPresentationFrameV1(legacy);
  const sourceScene = presentationScene(legacy);
  assert.deepEqual(normalized.scene.class_mechanics, sourceScene.class_mechanics);
  assert.deepEqual(
    normalized.scene.spawn_shield_mechanics,
    sourceScene.spawn_shield_mechanics,
  );
  assert.equal(
    normalized.scene.class_mechanics.every(
      (/** @type {Record<string, any>} */ row) =>
        !Object.hasOwn(row, "mechanics_version") &&
        !Object.hasOwn(row, "documentation_profile"),
    ),
    true,
  );
  assert.deepEqual(Object.keys(normalized.scene.spawn_shield_mechanics).sort(), [
    "availability_kind",
    "configured_duration_steps",
    "movement_speed",
  ]);
  assert.equal(normalized.scene.spawn_shield_mechanics.availability_kind, "available");
  assertRecursivelyFrozen(normalized);
});

test("V2 class and shield mechanics reject missing, extra, and coerced fields", async () => {
  const base = fixture.presentations.replay_oracle;
  const cases = [];
  const shieldPath =
    "Authorized presentation frame.current_endpoint.scene.spawn_shield_mechanics";
  const classPath =
    "Authorized presentation frame.current_endpoint.scene.class_mechanics[0]";

  const missingShield = clone(base);
  delete presentationScene(missingShield).spawn_shield_mechanics.action_scope;
  cases.push({
    label: "missing required V2 shield field",
    poisoned: missingShield,
    message: `${shieldPath}.action_scope is required.`,
  });

  const extraShield = clone(base);
  presentationScene(extraShield).spawn_shield_mechanics.derived_comparison = "faster";
  cases.push({
    label: "extra V2 shield field",
    poisoned: extraShield,
    message: `${shieldPath} contains an unknown field.`,
  });

  const coercedShield = clone(base);
  presentationScene(coercedShield).spawn_shield_mechanics.configured_duration_steps =
    "3";
  cases.push({
    label: "coerced V2 shield integer",
    poisoned: coercedShield,
    message: `${shieldPath}.configured_duration_steps must be a safe integer.`,
  });

  const missingClassVersion = clone(base);
  delete presentationScene(missingClassVersion).class_mechanics[0].mechanics_version;
  cases.push({
    label: "missing required V2 class version",
    poisoned: missingClassVersion,
    message: `${classPath} does not match any allowed strict variant.`,
  });

  const extraClass = clone(base);
  presentationScene(extraClass).class_mechanics[0].comparative_rank = 1;
  cases.push({
    label: "extra V2 class field",
    poisoned: extraClass,
    message: `${classPath} does not match any allowed strict variant.`,
  });

  const coercedClassVersion = clone(base);
  presentationScene(coercedClassVersion).class_mechanics[0].mechanics_version = "2";
  cases.push({
    label: "coerced V2 class version",
    poisoned: coercedClassVersion,
    message: `${classPath} does not match any allowed strict variant.`,
  });

  const missingProfileId = clone(base);
  delete presentationScene(missingProfileId).class_mechanics[0].documentation_profile
    .profile_id;
  cases.push({
    label: "missing required documentation profile ID",
    poisoned: missingProfileId,
    message: `${classPath} does not match any allowed strict variant.`,
  });

  const extraProfile = clone(base);
  presentationScene(extraProfile).class_mechanics[0].documentation_profile.extra = true;
  cases.push({
    label: "extra documentation profile field",
    poisoned: extraProfile,
    message: `${classPath} does not match any allowed strict variant.`,
  });

  const coercedProfileId = clone(base);
  presentationScene(
    coercedProfileId,
  ).class_mechanics[0].documentation_profile.profile_id = true;
  cases.push({
    label: "coerced documentation profile ID",
    poisoned: coercedProfileId,
    message: `${classPath} does not match any allowed strict variant.`,
  });

  for (const { label, poisoned, message } of cases) {
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1(poisoned),
      { name: "TypeError", message },
      label,
    );
  }
});

test("class mechanics reject mixed versions and discordant V2 profile decisions", async () => {
  const base = fixture.presentations.replay_oracle;

  const mixed = clone(base);
  const legacyRow = presentationScene(mixed).class_mechanics[0];
  delete legacyRow.mechanics_version;
  delete legacyRow.documentation_profile;
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(mixed),
    /Scene class mechanics must be entirely V1 or entirely V2\./u,
  );

  const discordant = clone(base);
  presentationScene(discordant).class_mechanics[0].documentation_profile = {
    availability_kind: "unavailable",
  };
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(discordant),
    /Scene V2 class mechanics must share one documentation profile\./u,
  );
});

test("V2 spawn shield retains configured duration and speed validation", async () => {
  const poisoned = clone(fixture.presentations.replay_oracle);
  presentationScene(poisoned).spawn_shield_mechanics.movement_speed = 0;
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(poisoned),
    /Scene spawn-shield remaining duration exceeds configuration\./u,
  );
});

test("closed roots reject missing, extra, coercion, discriminator, and nonfinite poison", async () => {
  const base = fixture.presentations.replay_oracle;
  const cases = [];

  const missing = clone(base);
  delete missing.technical_frame;
  cases.push(missing);

  const extra = clone(base);
  extra.current_endpoint.scene.extra = true;
  cases.push(extra);

  const coercion = clone(base);
  coercion.source.source_frame_index = "1";
  cases.push(coercion);

  const discriminator = clone(base);
  discriminator.presentation_kind = "future_oracle";
  cases.push(discriminator);

  const nestedDiscriminator = clone(base);
  nestedDiscriminator.current_endpoint.action_axis.target_actions[1].target_kind =
    "future_target";
  cases.push(nestedDiscriminator);

  const nonfinite = clone(base);
  nonfinite.current_endpoint.scene.map.width = Number.POSITIVE_INFINITY;
  cases.push(nonfinite);

  for (const poisoned of cases) {
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(poisoned), TypeError);
  }
});

test("axis, mask, epoch, identity, and digest semantic poison rejects", async () => {
  const oracle = fixture.presentations.replay_oracle;
  const agent = fixture.presentations.replay_shared_obs_agent_pov;

  const axisOrder = clone(oracle);
  axisOrder.current_endpoint.action_axis.movement_actions.reverse();

  const maskMarginal = clone(agent);
  maskMarginal.current_endpoint.parts.next_decision_action_mask.select_target[0] =
    !maskMarginal.current_endpoint.parts.next_decision_action_mask.select_target[0];

  const wrongFrameId = clone(agent);
  wrongFrameId.source.source_recipient_frame_id = "episode-001:wrong:frame:1";

  const staleEpoch = clone(oracle);
  staleEpoch.source.source_authority_epoch += 1;

  const digest = clone(oracle);
  digest.source.source_authorized_endpoint_digest_sha256 = "f".repeat(64);

  const latest = clone(agent);
  latest.latest_transition.incoming_successor_frame_id = "wrong";

  for (const poisoned of [
    axisOrder,
    maskMarginal,
    wrongFrameId,
    staleEpoch,
    digest,
    latest,
  ]) {
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(poisoned), TypeError);
  }

  for (const kind of kinds) {
    const content = clone(fixture.presentations[kind]);
    const endpoint = content.current_endpoint;
    const scene = endpoint.scene ?? endpoint.parts.scene;
    scene.map.width += 0.125;
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(content), TypeError);

    const coherentFakeDigest = clone(fixture.presentations[kind]);
    coherentFakeDigest.current_endpoint.authorized_endpoint_digest_sha256 = "e".repeat(
      64,
    );
    coherentFakeDigest.source.source_authorized_endpoint_digest_sha256 = "e".repeat(64);
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1(coherentFakeDigest),
      TypeError,
    );
  }
});

test("per-leaf history, Technical Frame, and inspection branches join source epoch", async () => {
  const liveAgent = fixture.presentations.live_no_shared_obs_agent_pov;
  const replayShared = fixture.presentations.replay_shared_obs_agent_pov;

  const inspectionSession = clone(liveAgent);
  inspectionSession.live_inspection.source_session_id = "other-session";

  const eventEpisode = clone(liveAgent);
  eventEpisode.latest_events.source_episode_id = "other-episode";

  const technicalIncoming = clone(replayShared);
  technicalIncoming.technical_frame.incoming_recipient_transition_id =
    "other:valid:transition:0";

  const outgoingIndex = clone(replayShared);
  outgoingIndex.replay_inspection.outgoing_transition_index += 7;

  const invertedPrefix = clone(replayShared);
  invertedPrefix.source.source_final_frame_index = 0;

  for (const poisoned of [
    inspectionSession,
    eventEpisode,
    technicalIncoming,
    outgoingIndex,
    invertedPrefix,
  ]) {
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(poisoned), TypeError);
  }
});

test("action semantics, endpoint identity, and authority key derivation match Python", async () => {
  const replayAgent = fixture.presentations.replay_no_shared_obs_agent_pov;
  const liveAgent = fixture.presentations.live_no_shared_obs_agent_pov;
  const replayOracle = fixture.presentations.replay_oracle;

  const combatLane = clone(replayAgent);
  combatLane.replay_inspection.combat_lane =
    combatLane.replay_inspection.combat_lane === "basic" ? "ultimate" : "basic";

  const draftLegality = clone(liveAgent);
  draftLegality.live_inspection.inspection.draft.draft_legality.move_action_is_legal =
    !draftLegality.live_inspection.inspection.draft.draft_legality.move_action_is_legal;

  const targetVariant = clone(replayAgent);
  const visibleIndex =
    targetVariant.replay_inspection.decision_mask.target_actions.findIndex(
      (/** @type {any} */ row) => row.target_kind === "visible_authorized_agent",
    );
  const visible =
    targetVariant.replay_inspection.decision_mask.target_actions[visibleIndex];
  targetVariant.replay_inspection.decision_mask.target_actions[visibleIndex] = {
    target_kind: "axis_only_authorized_agent",
    target_action: visible.target_action,
    display_name: visible.display_name,
    target_public_agent_id: visible.target_public_agent_id,
  };

  const directoryClass = clone(replayOracle);
  const activeDirectory =
    directoryClass.current_endpoint.identity_directory.identities.find(
      (/** @type {any} */ row) => row.configured_active,
    );
  activeDirectory.class_name = `${activeDirectory.class_name} changed`;

  const axisSwap = clone(replayOracle);
  axisSwap.source.source_frame_index = axisSwap.source.source_final_frame_index;
  axisSwap.source.source_frame_id = `${axisSwap.source.episode_id}:frame:${axisSwap.source.source_frame_index}`;
  axisSwap.current_endpoint.frame_index = axisSwap.source.source_frame_index;
  axisSwap.current_endpoint.frame_id = axisSwap.source.source_frame_id;
  axisSwap.replay_inspection = null;
  const first = axisSwap.current_endpoint.action_axis.target_actions[1];
  const second = axisSwap.current_endpoint.action_axis.target_actions[2];
  [first.target_public_agent_id, second.target_public_agent_id] = [
    second.target_public_agent_id,
    first.target_public_agent_id,
  ];

  const acceptedOutOfRange = clone(replayAgent);
  acceptedOutOfRange.latest_transition.action_rows[0].submitted_action.move_action = 99;
  acceptedOutOfRange.latest_transition.action_rows[0].accepted_action.move_action = 99;
  acceptedOutOfRange.latest_events.cues[0].outcome = "accepted";

  const phaseRank = clone(replayOracle);
  const ability = phaseRank.latest_events.events.find(
    (/** @type {any} */ event) => event.event_kind === "ability_activated",
  );
  ability.phase_rank += 1;

  const forgedAnchor = clone(replayOracle);
  const anchoredEvent = forgedAnchor.latest_events.events.find(
    (/** @type {any} */ event) => event.source_anchor,
  );
  anchoredEvent.source_anchor.position[0] += 0.25;

  for (const poisoned of [
    combatLane,
    draftLegality,
    targetVariant,
    directoryClass,
    axisSwap,
    acceptedOutOfRange,
    phaseRank,
    forgedAnchor,
  ]) {
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(poisoned), TypeError);
  }

  for (const kind of [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ]) {
    const wrongNamespace = transformStrings(fixture.presentations[kind], (value) =>
      value.replace(/^pov_(?=[0-9a-f]{64}$)/u, "oracle_"),
    );
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1(wrongNamespace),
      TypeError,
    );

    const wrongAuthorityHash = transformStrings(fixture.presentations[kind], (value) =>
      /^pov_[0-9a-f]{64}$/u.test(value) ? `pov_${"f".repeat(64)}` : value,
    );
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1(wrongAuthorityHash),
      TypeError,
    );
  }
});

test("Python custom incoming, history, and inspection validators stay closed", async () => {
  for (const kind of kinds) {
    const frame = fixture.presentations[kind];
    const wrongEpisode = clone(frame);
    wrongEpisode.latest_transition.episode_id += "-other";
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1(wrongEpisode),
      TypeError,
    );

    const shortTargets = clone(frame);
    const decisionMask = kind.startsWith("live_")
      ? shortTargets.live_inspection.inspection.draft.decision_mask
      : shortTargets.replay_inspection.decision_mask;
    decisionMask.target_actions.pop();
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1(shortTargets),
      TypeError,
    );
  }

  for (const kind of [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
  ]) {
    const wrongCueTransition = clone(fixture.presentations[kind]);
    wrongCueTransition.latest_events.cues[1].pov_transition_id += "-other";

    const unchangedFlag = clone(fixture.presentations[kind]);
    unchangedFlag.latest_events.cues[1].observed_payload_changed = false;

    const changedStaticProfile = clone(fixture.presentations[kind]);
    changedStaticProfile.latest_events.cues[1].start_observation.radius += 0.25;

    const duplicateAura = clone(fixture.presentations[kind]);
    duplicateAura.latest_events.cues[1].start_observation.aura_modifiers.push(
      clone(duplicateAura.latest_events.cues[1].start_observation.aura_modifiers[0]),
    );

    for (const poisoned of [
      wrongCueTransition,
      unchangedFlag,
      changedStaticProfile,
      duplicateAura,
    ]) {
      await assert.rejects(normalizeAuthorizedPresentationFrameV1(poisoned), TypeError);
    }
  }

  const shared = fixture.presentations.replay_shared_obs_agent_pov;
  const wrongDeltaTransition = clone(shared);
  wrongDeltaTransition.latest_events.deltas[0].recipient_transition_id += "-other";

  const missingDynamicField = clone(shared);
  missingDynamicField.latest_events.deltas[0].changed_dynamic_fields.pop();

  const duplicatedDynamicField = clone(shared);
  duplicatedDynamicField.latest_events.deltas[0].changed_dynamic_fields.push(
    duplicatedDynamicField.latest_events.deltas[0].changed_dynamic_fields[0],
  );

  const changedStartPosition = clone(shared);
  changedStartPosition.latest_events.deltas[0].start_observation.position.reverse();

  for (const poisoned of [
    wrongDeltaTransition,
    missingDynamicField,
    duplicatedDynamicField,
    changedStartPosition,
  ]) {
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(poisoned), TypeError);
  }

  const statusIdentity = clone(fixture.presentations.replay_oracle);
  const statusEvent = statusIdentity.latest_events.events.find(
    (/** @type {any} */ event) => event.event_kind === "status_applied",
  );
  statusEvent.status_id = "warrior_charge_slow";

  const replayIdentity = clone(fixture.presentations.replay_oracle);
  replayIdentity.source.source_timeline_id += "-other";

  const replayGeneration = clone(fixture.presentations.replay_oracle);
  replayGeneration.source.source_choreography_generation += 1;

  for (const poisoned of [statusIdentity, replayIdentity, replayGeneration]) {
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(poisoned), TypeError);
  }
});

test("all five exact raw/presentation pairs join identity-first and timelines stay transport-only", async () => {
  for (const kind of kinds) {
    const pair = fixture.pairs[kind];
    const before = JSON.stringify(pair);
    const joined = await joinTransportAndAuthorizedPresentationV1(
      pair.transport,
      pair.presentation,
    );
    assert.equal(isJoinedTransportAndAuthorizedPresentationV1(joined), true);
    assert.equal(isJoinedTransportAndAuthorizedPresentationV1({ ...joined }), false);
    assert.equal(joined.presentation.presentation_kind, kind);
    assert.equal(JSON.stringify(pair), before);
    assertRecursivelyFrozen(joined);
    if (pair.timeline) {
      const installed = joinReplayTransportAndTimelineV1(joined, pair.timeline);
      assert.equal(isJoinedTransportAndAuthorizedPresentationV1(installed), true);
      assert.equal(installed.timeline.timeline_id, joined.transport.timeline_id);
      assertRecursivelyFrozen(installed);
    }
  }
});

test("every coherent raw identity tuple mismatch is a retryable race before endpoint reads", async () => {
  /** @type {Record<string, string>} */
  const modePeer = {
    live_oracle: "live_no_shared_obs_agent_pov",
    live_no_shared_obs_agent_pov: "live_oracle",
    replay_oracle: "replay_no_shared_obs_agent_pov",
    replay_no_shared_obs_agent_pov: "replay_shared_obs_agent_pov",
    replay_shared_obs_agent_pov: "replay_no_shared_obs_agent_pov",
  };
  /** @type {Record<string, string>} */
  const productPeer = {
    live_oracle: "replay_oracle",
    live_no_shared_obs_agent_pov: "replay_no_shared_obs_agent_pov",
    replay_oracle: "live_oracle",
    replay_no_shared_obs_agent_pov: "live_no_shared_obs_agent_pov",
    replay_shared_obs_agent_pov: "live_no_shared_obs_agent_pov",
  };

  for (const kind of kinds) {
    const pair = fixture.pairs[kind];
    const live = kind.startsWith("live_");
    const oracle = kind.endsWith("oracle");
    const shared = kind === "replay_shared_obs_agent_pov";
    const sessionKey = live ? "session_id" : "viewer_session_id";
    /** @type {JoinIdentityMutation[]} */
    const mutations = [
      identityMutation("session", (transport) => {
        transport[sessionKey] += "-race";
      }),
      identityMutation("revision and authority epoch", (transport) => {
        transport.revision += 1;
      }),
      identityMutation("simulator tick", (transport) => {
        transport.simulator_step_count += 1;
      }),
      {
        label: "authority mode and frame kind",
        makeTransport: () => clone(fixture.pairs[modePeer[kind]].transport),
      },
      {
        label: "product and root schema",
        makeTransport: () => clone(fixture.pairs[productPeer[kind]].transport),
      },
      {
        label: live
          ? "episode and canonical frame ID"
          : oracle
            ? "episode, artifact, timeline, and canonical frame ID"
            : "episode and recipient-local frame ID",
        makeTransport(transport) {
          const episode = live
            ? transport.episode_id
            : shared
              ? transport.artifact_summary.episode_id
              : transport.artifact_summary.replay_reference.episode_id;
          const replacement = `${episode}-race`;
          return transformStrings(transport, (text) =>
            text === episode || text.startsWith(`${episode}:`)
              ? `${replacement}${text.slice(episode.length)}`
              : text,
          );
        },
      },
      identityMutation("frame index and canonical/local frame ID", (transport) => {
        if (live) {
          transport.frame_index += 1;
          transport.frame_id = `${transport.episode_id}:frame:${transport.frame_index}`;
          return;
        }
        transport.cursor.frame_index = transport.cursor.frame_index === 0 ? 1 : 0;
        const episode = shared
          ? transport.artifact_summary.episode_id
          : transport.artifact_summary.replay_reference.episode_id;
        if (oracle) {
          transport.frame_id = `${episode}:frame:${transport.cursor.frame_index}`;
          transport.incoming_transition_id = null;
          transport.incoming_transition_index = null;
        } else {
          const mode = shared ? "shared-obs-visual-union" : "actor-pov";
          const frameKey = shared ? "recipient_frame_id" : "pov_frame_id";
          transport[frameKey] =
            `${episode}:${mode}:${transport.public_agent_id}:frame:${transport.cursor.frame_index}`;
          const incomingKey = shared
            ? "incoming_recipient_transition_id"
            : "incoming_pov_transition_id";
          transport[incomingKey] = null;
        }
      }),
    ];

    if (live) {
      mutations.push(
        identityMutation("run generation", (transport) => {
          transport.run_generation += 1;
        }),
        identityMutation("submission scope", (transport) => {
          transport.hud.pending_submission_scope =
            transport.hud.pending_submission_scope === "scripted_playback"
              ? "joint_turn"
              : "scripted_playback";
        }),
      );
    } else {
      mutations.push(
        identityMutation("final frame index", (transport) => {
          transport.cursor.final_frame_index += 1;
        }),
      );
    }

    if (!oracle) {
      mutations.push({
        label: "recipient and recipient-local projection identity",
        makeTransport(transport) {
          const recipient = live
            ? transport.hud.controlled_public_agent_id
            : transport.public_agent_id;
          const replacement = `${recipient}-race`;
          return transformStrings(transport, (text) => {
            if (text === recipient) return replacement;
            return text.replaceAll(`:${recipient}:`, `:${replacement}:`);
          });
        },
      });
    }

    if (kind === "replay_oracle") {
      mutations.push(
        identityMutation("timeline", (transport) => {
          transport.timeline_id += ":race";
        }),
        identityMutation("context digest", (transport) => {
          transport.artifact_summary.replay_reference.context_digest_sha256 =
            "f".repeat(64);
        }),
        identityMutation("trajectory digest", (transport) => {
          transport.artifact_summary.replay_reference.trajectory_content_digest_sha256 =
            "e".repeat(64);
        }),
        identityMutation("artifact digest", (transport) => {
          transport.artifact_summary.replay_reference.canonical_digest_sha256 =
            "d".repeat(64);
        }),
        identityMutation("cursor generation", (transport) => {
          transport.cursor.cursor_generation += 1;
        }),
        identityMutation("cursor and choreography generations", (transport) => {
          transport.cursor.cursor_generation += 1;
          transport.cursor.choreography_generation = transport.cursor.cursor_generation;
        }),
        identityMutation("recorded movement scale", (transport) => {
          transport.recorded_ordinary_movement_distance_scale = 0.5;
        }),
      );
    }

    for (const mutation of mutations) {
      await assertJoinRaceBeforeEndpoint(
        pair,
        mutation.makeTransport(pair.transport),
        `${kind}: ${mutation.label}`,
      );
    }

    const wrongPreset = clone(pair.transport);
    wrongPreset.preset = "presentation";
    await assertProtocolPoisonBeforeEndpoint(
      pair,
      wrongPreset,
      `${kind}: preset must remain analysis`,
    );
  }

  const replayOracle = fixture.pairs.replay_oracle;
  const choreographyRaceRaw = clone(replayOracle.transport);
  choreographyRaceRaw.cursor.cursor_generation += 1;
  choreographyRaceRaw.cursor.choreography_generation += 1;
  const choreographyRaceSource = clone(replayOracle.presentation);
  choreographyRaceSource.source.source_cursor_generation += 1;
  const { presentation: choreographyRacePresentation, readCount } =
    withEndpointReadTripwire(choreographyRaceSource);
  await assert.rejects(
    joinTransportAndAuthorizedPresentationV1(
      choreographyRaceRaw,
      choreographyRacePresentation,
    ),
    (error) => {
      assert.equal(
        isPresentationJoinRace(error),
        true,
        "replay_oracle: choreography generation",
      );
      return true;
    },
  );
  assert.equal(readCount(), 0, "replay_oracle: choreography generation");

  const wrongReplaySchema = clone(replayOracle.transport);
  wrongReplaySchema.artifact_summary.replay_reference.replay_schema_version = 2;
  await assertProtocolPoisonBeforeEndpoint(
    replayOracle,
    wrongReplaySchema,
    "replay_oracle: unsupported replay schema is not a retryable race",
  );

  const liveOracle = fixture.pairs.live_oracle;
  const incoherentLive = clone(liveOracle.transport);
  incoherentLive.episode_id += "-incoherent";
  await assertProtocolPoisonBeforeEndpoint(
    liveOracle,
    incoherentLive,
    "live: episode without its canonical frame ID",
  );

  const incoherentReplay = clone(replayOracle.transport);
  incoherentReplay.cursor.frame_index = incoherentReplay.cursor.final_frame_index + 1;
  await assertProtocolPoisonBeforeEndpoint(
    replayOracle,
    incoherentReplay,
    "replay: frame index beyond final frame",
  );
});

test("private Shared transport completion and timeline mirror exact Python invariants", () => {
  const pair = fixture.pairs.replay_shared_obs_agent_pov;
  const valid = normalizeSharedObsAgentPovReplayTransportV1(pair.transport);
  const timeline = normalizeSharedObsAgentPovReplayTimelineTransportV1(pair.timeline);
  assertRecursivelyFrozen(valid);
  assertRecursivelyFrozen(timeline);

  const capturedPrefixBasis = clone(pair.transport);
  capturedPrefixBasis.completion.completion_bases = ["captured_prefix"];
  assert.throws(
    () => normalizeSharedObsAgentPovReplayTransportV1(capturedPrefixBasis),
    TypeError,
  );

  const incompleteWithBases = clone(pair.transport);
  incompleteWithBases.completion.completion_state = "partial";
  incompleteWithBases.completion.public_end_or_failure_reason = "captured_prefix";
  assert.throws(
    () => normalizeSharedObsAgentPovReplayTransportV1(incompleteWithBases),
    TypeError,
  );

  const bothDone = clone(pair.transport);
  bothDone.completion.terminated = true;
  bothDone.completion.truncated = true;
  bothDone.completion.completion_bases = ["task_terminal", "declared_horizon"];
  assert.doesNotThrow(() => normalizeSharedObsAgentPovReplayTransportV1(bothDone));

  const accessor = clone(pair.timeline);
  let reads = 0;
  Object.defineProperty(accessor.rows[0], "recipient_frame_id", {
    configurable: true,
    enumerable: true,
    get() {
      reads += 1;
      throw new Error("must not run");
    },
  });
  assert.throws(
    () => normalizeSharedObsAgentPovReplayTimelineTransportV1(accessor),
    TypeError,
  );
  assert.equal(reads, 0);
});

test("replay continuity spans branded legacy, private, and audience-switch pairs", async () => {
  const legacyPair = fixture.pairs.replay_oracle;
  const privatePair = fixture.pairs.replay_shared_obs_agent_pov;
  const legacy = await joinTransportAndAuthorizedPresentationV1(
    legacyPair.transport,
    legacyPair.presentation,
  );
  const privateShared = await joinTransportAndAuthorizedPresentationV1(
    privatePair.transport,
    privatePair.presentation,
  );
  assert.equal(validateReplayTransportContinuityV1(legacy, legacy, "no_op"), legacy);
  assert.equal(
    validateReplayTransportContinuityV1(privateShared, privateShared, "no_op"),
    privateShared,
  );
  assert.equal(
    validateReplayTransportContinuityV1(privateShared, privateShared, "duplicate"),
    privateShared,
  );
  assert.equal(
    validateReplayTransportContinuityV1(privateShared, privateShared, "stale_resync"),
    privateShared,
  );

  /**
   * @param {Record<string, any>} pair
   * @param {number} revision
   * @param {{cursor?: number, choreography?: number}} generations
   */
  async function withRevision(pair, revision, generations = {}) {
    const transport = clone(pair.transport);
    const presentation = clone(pair.presentation);
    transport.revision = revision;
    if (generations.cursor !== undefined) {
      transport.cursor.cursor_generation = generations.cursor;
    }
    if (generations.choreography !== undefined) {
      transport.cursor.choreography_generation = generations.choreography;
    }
    presentation.source.source_revision = revision;
    presentation.source.source_authority_epoch = revision;
    return joinTransportAndAuthorizedPresentationV1(transport, presentation);
  }
  const legacyApplied = await withRevision(legacyPair, 1);
  const privateApplied = await withRevision(privatePair, 1);
  assert.equal(
    validateReplayTransportContinuityV1(legacy, legacyApplied, "applied"),
    legacyApplied,
  );
  assert.equal(
    validateReplayTransportContinuityV1(legacy, legacyApplied, "duplicate"),
    legacyApplied,
  );
  assert.equal(
    validateReplayTransportContinuityV1(legacy, legacyApplied, "stale_resync"),
    legacyApplied,
  );
  assert.equal(
    validateReplayTransportContinuityV1(privateShared, privateApplied, "applied"),
    privateApplied,
  );
  const privateSettled = await withRevision(privatePair, 1, {
    cursor: 2,
    choreography: 1,
  });
  assert.equal(
    validateReplayTransportContinuityV1(privateShared, privateSettled, "duplicate"),
    privateSettled,
  );
  assert.equal(
    validateReplayTransportContinuityV1(privateShared, privateSettled, "stale_resync"),
    privateSettled,
  );
  const privateGenerationBase = await withRevision(privatePair, 0, {
    cursor: 2,
    choreography: 0,
  });
  const invalidGenerationDelta = await withRevision(privatePair, 1, {
    cursor: 3,
    choreography: 2,
  });
  assert.throws(
    () =>
      validateReplayTransportContinuityV1(
        privateGenerationBase,
        invalidGenerationDelta,
        "stale_resync",
      ),
    TypeError,
  );
  assert.throws(
    () =>
      validateReplayTransportContinuityV1(
        privateApplied,
        privateShared,
        "stale_resync",
      ),
    TypeError,
  );
  assert.throws(
    () => validateReplayTransportContinuityV1({ ...legacy }, privateShared, "no_op"),
    TypeError,
  );

  const switchOraclePair = fixture.continuity_pairs.oracle;
  const switchSharedPair = fixture.continuity_pairs.shared_obs;
  const switchOracle = await joinTransportAndAuthorizedPresentationV1(
    switchOraclePair.transport,
    switchOraclePair.presentation,
  );
  const switchShared = await joinTransportAndAuthorizedPresentationV1(
    switchSharedPair.transport,
    switchSharedPair.presentation,
  );
  assert.equal(
    validateReplayTransportContinuityV1(switchOracle, switchShared, "stale_resync"),
    switchShared,
  );
  assert.equal(
    validateReplayTransportContinuityV1(switchShared, switchOracle, "stale_resync"),
    switchOracle,
  );
  assert.throws(
    () => validateReplayTransportContinuityV1(legacy, privateShared, "stale_resync"),
    TypeError,
  );
});

test("prototype, accessor, symbol, and sparse JSON tricks reject without getter reads", async () => {
  const prototype = clone(fixture.presentations.live_oracle);
  Object.setPrototypeOf(prototype.current_endpoint, { poisoned: true });

  const accessor = clone(fixture.presentations.live_oracle);
  let reads = 0;
  Object.defineProperty(accessor, "technical_frame", {
    enumerable: true,
    get() {
      reads += 1;
      throw new Error("must not run");
    },
  });

  const symbol = clone(fixture.presentations.live_oracle);
  symbol[Symbol("poison")] = true;

  const sparse = clone(fixture.presentations.live_oracle);
  delete sparse.current_endpoint.scene.agents[0];

  const topProtoKey = clone(fixture.presentations.live_oracle);
  Object.defineProperty(topProtoKey, "__proto__", {
    configurable: true,
    enumerable: true,
    value: { poison: true },
    writable: true,
  });

  const nestedProtoKey = clone(fixture.presentations.replay_shared_obs_agent_pov);
  Object.defineProperty(nestedProtoKey.source, "__proto__", {
    configurable: true,
    enumerable: true,
    value: { poison: true },
    writable: true,
  });

  for (const poisoned of [
    prototype,
    accessor,
    symbol,
    sparse,
    topProtoKey,
    nestedProtoKey,
  ]) {
    await assert.rejects(normalizeAuthorizedPresentationFrameV1(poisoned), TypeError);
  }
  assert.equal(reads, 0);
});

test("raw Shared and retired diagnostic roots are never presentation leaves", async () => {
  for (const frameKind of [
    "shared_obs_agent_pov_replay_viewer",
    "shared_obs_source_material_replay_viewer",
  ]) {
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1({
        schema_version: 1,
        frame_kind: frameKind,
      }),
      TypeError,
    );
  }
});

test("Agent leaves contain no forbidden Oracle or diagnostic transport keys", async () => {
  for (const kind of kinds.filter((value) => value.includes("agent_pov"))) {
    const normalized = await normalizeAuthorizedPresentationFrameV1(
      fixture.presentations[kind],
    );
    const encoded = JSON.stringify(normalized);
    for (const forbidden of [
      "global_slot",
      "source_material",
      "metric_report",
      "canonical_digest_sha256",
      "trajectory_content_digest_sha256",
      "oracle_",
    ]) {
      assert.equal(encoded.includes(forbidden), false, `${kind}: ${forbidden}`);
    }

    const oracleValue = clone(fixture.presentations[kind]);
    oracleValue.current_endpoint.action_axis.movement_actions[0].display_name = `${oracleValue.source.episode_id}:frame:${oracleValue.source.source_frame_index}`;
    await assert.rejects(
      normalizeAuthorizedPresentationFrameV1(oracleValue),
      TypeError,
    );
    for (const forbiddenValue of [
      `${oracleValue.source.episode_id}:transition:0:event:0000`,
      `${oracleValue.source.episode_id}:replay`,
      `${oracleValue.source.episode_id}:shared-obs-visual-union:${oracleValue.authority.recipient_public_agent_id}:timeline`,
    ]) {
      const rawIdentity = clone(fixture.presentations[kind]);
      rawIdentity.current_endpoint.action_axis.movement_actions[0].display_name =
        forbiddenValue;
      await assert.rejects(
        normalizeAuthorizedPresentationFrameV1(rawIdentity),
        TypeError,
      );
    }
  }

  const shared = clone(fixture.presentations.replay_shared_obs_agent_pov);
  const source = shared.current_endpoint.parts.authorized_sensor_sources[1];
  const ownProvenance = shared.current_endpoint.parts.agent_observation_provenance.find(
    (/** @type {any} */ row) =>
      row.agent_public_agent_id === source.source_public_agent_id,
  );
  ownProvenance.observation_sources = ownProvenance.observation_sources.filter(
    (/** @type {any} */ row) =>
      row.source_public_agent_id !== source.source_public_agent_id,
  );
  await assert.rejects(normalizeAuthorizedPresentationFrameV1(shared), TypeError);
});

test("presentation unavailable error accepts only the exact 422 payload root", () => {
  const exact = {
    schema_version: 1,
    error_code: "audience_unavailable",
    message: "Authorized presentation is unavailable for the active audience.",
  };
  const normalized = normalizePresentationApiErrorV1(exact);
  assert.deepEqual(normalized, exact);
  assertRecursivelyFrozen(normalized);

  for (const poisoned of [
    { ...exact, schema_version: true },
    { ...exact, error_code: "internal_error" },
    { ...exact, message: "details" },
    { ...exact, extra: true },
  ]) {
    assert.throws(() => normalizePresentationApiErrorV1(poisoned), TypeError);
  }
});
