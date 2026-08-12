import { statusTokenIdFromCatalogId } from "./vocabulary.js";

const RESEARCHER_EVENT_TYPES_V2 = new Set([
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
]);

const POV_CUE_TYPES_V1 = new Set([
  "own_action_outcome",
  "own_position_changed",
  "own_health_changed",
  "own_status_changed",
  "own_cooldown_changed",
  "own_lifecycle_changed",
  "visible_body_observation_changed",
  "episode_ended",
]);

const RECORDING_STATUS_KEYS_V1 = Object.freeze([
  "captured_transition_count",
  "completion_reason",
  "completion_state",
  "discard_available",
  "expected_transition_count",
  "finish_available",
  "lifecycle",
  "persistence_error_code",
  "restart_fenced",
  "retry_available",
  "review_available",
  "save_as_available",
  "schema_version",
]);

const RECORDING_LIFECYCLES_V1 = new Set([
  "recording",
  "sealed",
  "finalized_unsaved",
  "persistence_failed",
  "saved",
  "reviewing",
  "discarded",
]);

const RECORDING_COMPLETION_STATES_V1 = new Set([
  "complete",
  "partial",
  "interrupted",
  "failed",
]);

const RECORDING_PERSISTENCE_ERRORS_V1 = new Set([
  "target_unavailable",
  "publication_failed",
  "verification_failed",
]);

const RESEARCHER_LIVE_FRAME_KEYS_V2 = Object.freeze([
  "available_scenarios",
  "episode_id",
  "frame_id",
  "frame_index",
  "frame_kind",
  "hud",
  "incoming_transition_id",
  "incoming_transition_index",
  "preset",
  "projection",
  "recording",
  "revision",
  "run_generation",
  "scenario",
  "schema_version",
  "session_id",
  "simulator_step_count",
  "terminal",
  "view_mode",
]);

const ACTOR_POV_LIVE_FRAME_KEYS_V2 = Object.freeze([
  "episode_id",
  "frame_id",
  "frame_index",
  "frame_kind",
  "hud",
  "incoming_pov_transition_id",
  "preset",
  "projection",
  "recording",
  "revision",
  "run_generation",
  "schema_version",
  "session_id",
  "simulator_step_count",
  "terminal",
  "view_mode",
]);

// ActorPovSelfSceneV1 retains observation columns 15..28 exactly.  Only the
// nine duration columns denote durable status presence; multiplier/fraction
// columns remain policy input but must not be misread as additional statuses.
// Keep this table in the renderer's canonical presentation order while
// preserving each absolute V1 feature index as recipient-visible evidence.
const POV_STATUS_DURATION_FEATURES_V1 = Object.freeze([
  Object.freeze({
    offset: 6,
    featureIndex: 21,
    tokenId: "stun_warrior_charge",
    effectClassId: 2,
  }),
  Object.freeze({
    offset: 7,
    featureIndex: 22,
    tokenId: "stun_hunter_trap",
    effectClassId: 3,
  }),
  Object.freeze({
    offset: 8,
    featureIndex: 23,
    tokenId: "stun_rogue_poison",
    effectClassId: 4,
  }),
  Object.freeze({
    offset: 0,
    featureIndex: 15,
    tokenId: "slow_warrior_charge",
    effectClassId: 2,
  }),
  Object.freeze({
    offset: 1,
    featureIndex: 16,
    tokenId: "slow_hunter_basic",
    effectClassId: 3,
  }),
  Object.freeze({
    offset: 2,
    featureIndex: 17,
    tokenId: "slow_rogue_poison",
    effectClassId: 4,
  }),
  Object.freeze({
    offset: 9,
    featureIndex: 24,
    tokenId: "anti_heal_rogue_poison",
    effectClassId: 4,
  }),
  Object.freeze({
    offset: 12,
    featureIndex: 27,
    tokenId: "priest_freedom",
    effectClassId: 5,
  }),
  Object.freeze({
    offset: 11,
    featureIndex: 26,
    tokenId: "mage_burst",
    effectClassId: 1,
  }),
]);

/** @type {Readonly<Record<string, string>>} */
const AURA_TOKEN_BY_ID = Object.freeze({
  mage_damage_amplification: "mage_amplification",
  warrior_damage_mitigation: "warrior_mitigation",
});

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * @param {unknown} value
 * @param {string} message
 * @returns {Record<string, any>}
 */
function requireRecord(value, message) {
  if (!isRecord(value)) {
    throw new TypeError(message);
  }
  return value;
}

/**
 * @param {Record<string, any>} value
 * @param {readonly string[]} expected
 * @param {string} message
 */
function requireExactKeys(value, expected, message) {
  const actual = Object.keys(value).sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new TypeError(message);
  }
}

/**
 * Strictly normalize the path-free recording lifecycle shared by both live
 * audiences. Availability is joined here rather than inferred later by the
 * UI, so malformed or over-disclosing state never reaches a control surface.
 *
 * @param {unknown} value
 * @param {number} frameIndex
 * @returns {Readonly<Record<string, any>> | null}
 */
export function normalizeRecordingStatusV1(value, frameIndex) {
  if (value === null) {
    return null;
  }
  const status = requireRecord(
    value,
    "Live debugger recording status must be an object or null.",
  );
  requireExactKeys(
    status,
    RECORDING_STATUS_KEYS_V1,
    "Live debugger recording status has unknown or missing fields.",
  );
  if (status.schema_version !== 1) {
    throw new TypeError("Live debugger recording status must use schema version 1.");
  }
  if (!RECORDING_LIFECYCLES_V1.has(status.lifecycle)) {
    throw new TypeError("Live debugger recording lifecycle is unknown.");
  }
  if (
    !Number.isInteger(status.captured_transition_count) ||
    status.captured_transition_count < 0 ||
    !Number.isInteger(status.expected_transition_count) ||
    status.expected_transition_count <= 0 ||
    status.captured_transition_count > status.expected_transition_count ||
    status.captured_transition_count !== frameIndex
  ) {
    throw new TypeError(
      "Live debugger recording progress does not join its frame and horizon.",
    );
  }
  if (
    status.completion_state !== null &&
    !RECORDING_COMPLETION_STATES_V1.has(status.completion_state)
  ) {
    throw new TypeError("Live debugger recording completion state is unknown.");
  }
  const reasonLength =
    typeof status.completion_reason === "string"
      ? [...status.completion_reason].length
      : 0;
  if (
    status.completion_reason !== null &&
    (typeof status.completion_reason !== "string" ||
      reasonLength < 1 ||
      reasonLength > 256)
  ) {
    throw new TypeError("Live debugger recording completion reason is invalid.");
  }
  const booleanFields = [
    "restart_fenced",
    "finish_available",
    "review_available",
    "retry_available",
    "save_as_available",
    "discard_available",
  ];
  if (booleanFields.some((field) => typeof status[field] !== "boolean")) {
    throw new TypeError("Live debugger recording availability must be boolean.");
  }
  if (
    status.persistence_error_code !== null &&
    !RECORDING_PERSISTENCE_ERRORS_V1.has(status.persistence_error_code)
  ) {
    throw new TypeError("Live debugger recording persistence error is unknown.");
  }

  const finalized = [
    "sealed",
    "finalized_unsaved",
    "persistence_failed",
    "saved",
    "reviewing",
  ].includes(status.lifecycle);
  const reasonRequired = ["partial", "interrupted", "failed"].includes(
    status.completion_state,
  );
  const restartFenced =
    status.captured_transition_count > 0 || status.lifecycle !== "recording";
  const finishAvailable = status.lifecycle === "recording";
  const reviewAvailable = status.lifecycle === "saved";
  const retryAvailable = status.lifecycle === "persistence_failed";
  const discardAvailable =
    status.lifecycle === "recording" && status.captured_transition_count > 0;
  if (
    finalized !== (status.completion_state !== null) ||
    reasonRequired !== (status.completion_reason !== null) ||
    status.restart_fenced !== restartFenced ||
    status.finish_available !== finishAvailable ||
    status.review_available !== reviewAvailable ||
    status.retry_available !== retryAvailable ||
    status.save_as_available !== retryAvailable ||
    status.discard_available !== discardAvailable ||
    (status.persistence_error_code !== null) !== retryAvailable
  ) {
    throw new TypeError(
      "Live debugger recording lifecycle and availability are not canonical.",
    );
  }

  return Object.freeze({
    schema_version: status.schema_version,
    lifecycle: status.lifecycle,
    captured_transition_count: status.captured_transition_count,
    expected_transition_count: status.expected_transition_count,
    completion_state: status.completion_state,
    completion_reason: status.completion_reason,
    restart_fenced: status.restart_fenced,
    finish_available: status.finish_available,
    review_available: status.review_available,
    retry_available: status.retry_available,
    save_as_available: status.save_as_available,
    discard_available: status.discard_available,
    persistence_error_code: status.persistence_error_code,
  });
}

/**
 * @param {unknown} value
 * @param {string} message
 * @returns {any[]}
 */
function requireArray(value, message) {
  if (!Array.isArray(value)) {
    throw new TypeError(message);
  }
  return value;
}

/**
 * @param {Record<string, any>} status
 */
function normalizeStatus(status) {
  return Object.freeze({
    ...status,
    token_id: statusTokenIdFromCatalogId(status.status_id),
    duration: status.remaining_duration,
  });
}

/**
 * @param {Record<string, any>} modifier
 */
function normalizeAuraModifier(modifier) {
  const tokenId = AURA_TOKEN_BY_ID[modifier.aura_id];
  if (typeof tokenId !== "string") {
    throw new TypeError(`Unknown V2 aura identifier: ${modifier.aura_id}.`);
  }
  return Object.freeze({
    ...modifier,
    token_id: tokenId,
  });
}

/**
 * @param {unknown} value
 * @returns {value is readonly [number, number]}
 */
function isFinitePoint(value) {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    value.every(
      (coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate),
    )
  );
}

/**
 * Compose a live-only pending route from exact researcher HUD identity and the
 * already-authorized settled-scene body anchors. This projection never enters
 * the renderer-neutral SceneV2 wire contract.
 *
 * @param {Record<string, any>} scene
 * @param {Record<string, any>} frame
 */
function researcherPendingRoute(scene, frame) {
  const hud = isRecord(frame.hud) ? frame.hud : null;
  const pending = hud && isRecord(hud.pending_action) ? hud.pending_action : null;
  const target = pending && isRecord(pending.target) ? pending.target : null;
  const lane = pending?.armed_lane;
  if (
    !pending ||
    !target ||
    (lane !== 0 && lane !== 1) ||
    target.disclosure !== "public" ||
    !Number.isInteger(pending.actor_global_slot) ||
    !Number.isInteger(target.global_slot)
  ) {
    return null;
  }
  const agents = requireArray(
    scene.agents,
    "Researcher scene agents must be an array.",
  );
  const source = agents.find(
    (agent) => isRecord(agent) && agent.global_slot === pending.actor_global_slot,
  );
  const recipient = agents.find(
    (agent) => isRecord(agent) && agent.global_slot === target.global_slot,
  );
  if (
    !isRecord(source) ||
    !isRecord(recipient) ||
    typeof source.public_agent_id !== "string" ||
    typeof recipient.public_agent_id !== "string" ||
    !isFinitePoint(source.position) ||
    !isFinitePoint(recipient.position) ||
    typeof source.radius !== "number" ||
    !Number.isFinite(source.radius) ||
    source.radius <= 0 ||
    typeof recipient.radius !== "number" ||
    !Number.isFinite(recipient.radius) ||
    recipient.radius <= 0
  ) {
    return null;
  }
  return Object.freeze({
    audience: "researcher",
    source_global_slot: pending.actor_global_slot,
    target_global_slot: target.global_slot,
    source_public_agent_id: source.public_agent_id,
    target_public_agent_id: recipient.public_agent_id,
    target_action: pending.target_action,
    source_anchor: source.position,
    target_anchor: recipient.position,
    source_radius: source.radius,
    target_radius: recipient.radius,
    lane,
    legal:
      typeof pending.pair_mask_value === "boolean" ? pending.pair_mask_value : null,
  });
}

/**
 * Compose the same live-only affordance for a recipient POV without assigning
 * global identities to visible observation rows.
 *
 * @param {Record<string, any>} scene
 * @param {Record<string, any>} frame
 */
function povPendingRoute(scene, frame) {
  const hud = isRecord(frame.hud) ? frame.hud : null;
  const pending = hud && isRecord(hud.pending_action) ? hud.pending_action : null;
  const target = pending && isRecord(pending.target) ? pending.target : null;
  const lane = pending?.armed_lane;
  const targetAction = target?.target_action;
  if (
    !pending ||
    !target ||
    (lane !== 0 && lane !== 1) ||
    !Number.isInteger(targetAction) ||
    targetAction <= 0 ||
    typeof pending.actor_public_agent_id !== "string" ||
    typeof target.public_agent_id !== "string"
  ) {
    return null;
  }
  const source = requireArray(scene.agents, "POV self agents must be an array.").find(
    (agent) =>
      isRecord(agent) && agent.public_agent_id === pending.actor_public_agent_id,
  );
  const bodies = [
    ...requireArray(scene.agents, "POV self agents must be an array."),
    ...requireArray(scene.observed_bodies, "POV observed bodies must be an array."),
  ];
  const recipient = bodies.find(
    (body) => isRecord(body) && body.public_agent_id === target.public_agent_id,
  );
  if (
    !isRecord(source) ||
    !isRecord(recipient) ||
    !isFinitePoint(source.position) ||
    !isFinitePoint(recipient.position) ||
    typeof source.radius !== "number" ||
    !Number.isFinite(source.radius) ||
    source.radius <= 0 ||
    typeof recipient.radius !== "number" ||
    !Number.isFinite(recipient.radius) ||
    recipient.radius <= 0
  ) {
    return null;
  }
  return Object.freeze({
    audience: "agent_pov",
    source_public_agent_id: source.public_agent_id,
    target_public_agent_id: recipient.public_agent_id,
    target_action: targetAction,
    source_anchor: source.position,
    target_anchor: recipient.position,
    source_radius: source.radius,
    target_radius: recipient.radius,
    lane,
    legal:
      typeof pending.pair_mask_value === "boolean" ? pending.pair_mask_value : null,
  });
}

/**
 * Preserve the complete V2 agent row while publishing the stable names used
 * by the SVG/HUD presentation layer. These aliases are display vocabulary;
 * they do not recompute health, cooldown, status, or lifecycle truth.
 *
 * @param {Record<string, any>} agent
 * @returns {Readonly<Record<string, any>>}
 */
function normalizeResearcherAgent(agent) {
  const statuses = requireArray(
    agent.statuses,
    "Researcher V2 agent statuses must be an array.",
  ).map((status) => normalizeStatus(requireRecord(status, "Invalid V2 status row.")));
  const modifiers = requireArray(
    agent.aura_modifiers,
    "Researcher V2 agent aura modifiers must be an array.",
  ).map((modifier) =>
    normalizeAuraModifier(requireRecord(modifier, "Invalid V2 aura modifier row.")),
  );
  return Object.freeze({
    ...agent,
    alive: agent.life_state === "alive",
    effective_speed: agent.effective_movement_speed,
    ultimate_cooldown: agent.ultimate_cooldown_remaining,
    statuses: Object.freeze(statuses),
    modifiers: Object.freeze(modifiers),
  });
}

/**
 * @param {Record<string, any>} projection
 * @param {Record<string, any>} frame
 */
function normalizeResearcherProjection(projection, frame) {
  if (projection.schema_version !== 2) {
    throw new TypeError("Researcher projection must use schema version 2.");
  }
  const sourceScene = requireRecord(
    projection.scene,
    "Researcher projection is missing its V2 scene.",
  );
  if (sourceScene.schema_version !== 2 || sourceScene.audience !== "researcher") {
    throw new TypeError("Researcher projection must contain BattlefieldSceneV2.");
  }
  if (
    sourceScene.episode_id !== frame.episode_id ||
    sourceScene.frame_id !== frame.frame_id ||
    sourceScene.frame_index !== frame.frame_index ||
    sourceScene.simulator_step_count !== frame.simulator_step_count ||
    sourceScene.incoming_transition_id !== frame.incoming_transition_id
  ) {
    throw new TypeError("Researcher projection does not join its live frame.");
  }

  const agents = requireArray(
    sourceScene.agents,
    "Researcher V2 scene agents must be an array.",
  ).map((agent) =>
    normalizeResearcherAgent(requireRecord(agent, "Invalid V2 agent row.")),
  );
  const auraFields = requireArray(
    sourceScene.aura_fields,
    "Researcher V2 aura fields must be an array.",
  ).map((field) => {
    const row = requireRecord(field, "Invalid V2 aura field row.");
    const tokenId = AURA_TOKEN_BY_ID[row.aura_id];
    if (typeof tokenId !== "string") {
      throw new TypeError(`Unknown V2 aura identifier: ${row.aura_id}.`);
    }
    return Object.freeze({ ...row, token_id: tokenId });
  });
  const selection = sourceScene.selection;
  const observerVisibility = requireArray(
    sourceScene.observer_visibility,
    "Researcher V2 observer visibility must be an array.",
  ).map((rawFact, index) => {
    const fact = requireRecord(rawFact, "Invalid V2 observer-visibility row.");
    const candidate = agents[index];
    if (
      !isRecord(selection) ||
      fact.observer_global_slot !== selection.controlled_global_slot ||
      fact.candidate_global_slot !== candidate?.global_slot ||
      typeof fact.visible !== "boolean"
    ) {
      throw new TypeError(
        "Researcher V2 observer visibility must join the controlled ordered roster.",
      );
    }
    return Object.freeze({ ...fact });
  });
  if (
    (isRecord(selection) && observerVisibility.length !== agents.length) ||
    (!isRecord(selection) && observerVisibility.length !== 0)
  ) {
    throw new TypeError(
      "Researcher V2 observer visibility must match scene selection authority.",
    );
  }
  const sceneShell = {
    ...sourceScene,
    agents: Object.freeze(agents),
    aura_fields: Object.freeze(auraFields),
    selected_legality: sourceScene.next_decision_selected_legality ?? null,
    observer_visibility: Object.freeze(observerVisibility),
  };
  const scene = Object.freeze({
    ...sceneShell,
    pending_route: researcherPendingRoute(sceneShell, frame),
  });

  const sourceBatch = projection.incoming_events;
  if (sourceBatch === null || sourceBatch === undefined) {
    if (frame.frame_index !== 0 || frame.incoming_transition_id !== null) {
      throw new TypeError("Only researcher frame zero may omit incoming events.");
    }
    return { scene, eventBatch: null };
  }
  const batch = requireRecord(sourceBatch, "Invalid researcher V2 event batch.");
  if (
    batch.schema_version !== 2 ||
    batch.episode_id !== frame.episode_id ||
    batch.transition_id !== frame.incoming_transition_id ||
    batch.successor_frame_id !== frame.frame_id ||
    batch.successor_simulator_step_count !== frame.simulator_step_count
  ) {
    throw new TypeError("Researcher V2 event batch does not join its live frame.");
  }
  const events = requireArray(batch.events, "V2 event batch events must be an array.");
  const normalizedEvents = events.map((rawEvent, ordinal) => {
    const event = requireRecord(rawEvent, "Invalid V2 event row.");
    if (!RESEARCHER_EVENT_TYPES_V2.has(event.event_type)) {
      throw new TypeError(`Unknown V2 researcher event type: ${event.event_type}.`);
    }
    const expectedEventId = `${batch.transition_id}:event:${String(ordinal).padStart(4, "0")}`;
    if (
      event.ordinal !== ordinal ||
      event.transition_id !== batch.transition_id ||
      event.event_id !== expectedEventId
    ) {
      throw new TypeError("V2 event order or canonical identity is invalid.");
    }
    return Object.freeze({ ...event });
  });
  const sceneEventIds = requireArray(
    sourceScene.incoming_event_ids,
    "Researcher V2 scene incoming event IDs must be an array.",
  );
  if (
    sceneEventIds.length !== normalizedEvents.length ||
    sceneEventIds.some(
      (eventId, ordinal) => eventId !== normalizedEvents[ordinal].event_id,
    )
  ) {
    throw new TypeError("Researcher scene and event batch IDs do not join.");
  }
  return {
    scene,
    eventBatch: Object.freeze({
      ...batch,
      simulator_step: batch.successor_simulator_step_count,
      events: Object.freeze(normalizedEvents),
    }),
  };
}

/**
 * Decode only recipient-visible V1 status-duration columns. The fixed effect
 * class follows from the published feature channel; no source actor identity
 * or researcher-only attribution enters this boundary.
 *
 * @param {unknown} rawValues
 */
function normalizePovStatuses(rawValues) {
  const statusFeatureValues = requireArray(
    rawValues,
    "POV status features must be the exact V1 vector.",
  );
  if (
    statusFeatureValues.length !== 14 ||
    statusFeatureValues.some(
      (value) => typeof value !== "number" || !Number.isFinite(value) || value < 0,
    )
  ) {
    throw new TypeError(
      "POV status features must retain 14 finite nonnegative values.",
    );
  }
  return Object.freeze(
    POV_STATUS_DURATION_FEATURES_V1.flatMap(
      ({ offset, featureIndex, tokenId, effectClassId }) => {
        const duration = statusFeatureValues[offset];
        if (!Number.isInteger(duration)) {
          throw new TypeError(
            "POV status durations must be integer-valued V1 features.",
          );
        }
        return duration === 0
          ? []
          : [
              Object.freeze({
                token_id: tokenId,
                duration,
                status_feature_index: featureIndex,
                source_class_id: effectClassId,
                source_evidence: "effect_channel_only",
              }),
            ];
      },
    ),
  );
}

/**
 * Convert the already recipient-sliced POV projection into the common visual
 * shell without assigning global identities to visible observation rows.
 *
 * @param {Record<string, any>} projection
 * @param {Record<string, any>} frame
 */
function normalizePovProjection(projection, frame) {
  const sourceScene = requireRecord(
    projection.scene,
    "POV projection is missing its authorized scene.",
  );
  const selfActor = requireRecord(
    sourceScene.self_actor,
    "POV scene is missing its self actor.",
  );
  if (
    sourceScene.schema_version !== 1 ||
    sourceScene.observation_materialization !== "exact_no_shared_obs_actor_input" ||
    !isRecord(projection.next_decision_action_mask)
  ) {
    throw new TypeError("POV projection must contain the exact V1 actor-input roots.");
  }
  const expectedPovFrameId = `${frame.episode_id}:actor-pov:${selfActor.public_agent_id}:frame:${frame.frame_index}`;
  if (
    sourceScene.episode_id !== frame.episode_id ||
    sourceScene.source_frame_id !== frame.frame_id ||
    sourceScene.frame_index !== frame.frame_index ||
    sourceScene.simulator_step_count !== frame.simulator_step_count ||
    sourceScene.pov_frame_id !== expectedPovFrameId
  ) {
    throw new TypeError("POV projection does not join its live frame.");
  }
  const selfStatuses = normalizePovStatuses(selfActor.status_feature_values);
  const selfAgent = Object.freeze({
    ...selfActor,
    effective_speed: selfActor.effective_movement_speed,
    ultimate_cooldown: selfActor.ultimate_cooldown_remaining,
    statuses: Object.freeze(selfStatuses),
    modifiers: Object.freeze([]),
  });
  const visibleBodies = requireArray(
    sourceScene.visible_bodies,
    "POV visible bodies must be an array.",
  ).map((body) => {
    const row = requireRecord(body, "Invalid POV body row.");
    if (
      (row.relation !== "ally" && row.relation !== "enemy") ||
      !Number.isInteger(row.observation_row) ||
      typeof row.public_agent_id !== "string" ||
      Object.hasOwn(row, "global_slot")
    ) {
      throw new TypeError("POV visible bodies require recipient-relative identity.");
    }
    return Object.freeze({
      ...row,
      observation_key: `${row.relation}:${row.observation_row}`,
      effective_speed: row.effective_movement_speed,
      ultimate_cooldown: row.ultimate_cooldown_remaining,
      statuses: normalizePovStatuses(row.status_feature_values),
      modifiers: Object.freeze([]),
    });
  });
  const sceneShell = {
    ...sourceScene,
    audience: "agent_pov",
    frame_id: sourceScene.source_frame_id,
    incoming_transition_id: frame.incoming_pov_transition_id,
    agents: Object.freeze([selfAgent]),
    observed_bodies: Object.freeze(visibleBodies),
    aura_fields: Object.freeze([]),
    ranges: Object.freeze([]),
    observer_visibility: Object.freeze([]),
    selection: Object.freeze({
      controlled_global_slot: selfActor.global_slot,
      selected_global_slot: null,
    }),
    selected_legality: null,
  };
  const scene = Object.freeze({
    ...sceneShell,
    pending_route: povPendingRoute(sceneShell, frame),
  });

  const incomingTransitionId = projection.incoming_transition_id;
  const cues = requireArray(
    projection.incoming_cues,
    "POV incoming cues must be an array.",
  );
  if (incomingTransitionId === null) {
    if (
      frame.frame_index !== 0 ||
      frame.incoming_pov_transition_id !== null ||
      cues.length !== 0
    ) {
      throw new TypeError("Only POV frame zero may omit its incoming transition.");
    }
    return { scene, eventBatch: null };
  }
  const expectedPovTransitionId = `${frame.episode_id}:actor-pov:${selfActor.public_agent_id}:transition:${frame.frame_index - 1}`;
  if (incomingTransitionId !== expectedPovTransitionId) {
    throw new TypeError("POV incoming transition identity is not canonical.");
  }
  if (frame.incoming_pov_transition_id !== expectedPovTransitionId) {
    throw new TypeError("POV live envelope does not join its local transition.");
  }
  const normalizedCues = cues.map((rawCue, ordinal) => {
    const cue = requireRecord(rawCue, "Invalid POV cue row.");
    if (!POV_CUE_TYPES_V1.has(cue.cue_type)) {
      throw new TypeError(`Unknown actor-POV cue type: ${cue.cue_type}.`);
    }
    if (
      cue.ordinal !== ordinal ||
      cue.pov_transition_id !== incomingTransitionId ||
      cue.cue_id !== `${incomingTransitionId}:cue:${ordinal}`
    ) {
      throw new TypeError("POV cue order or local identity is invalid.");
    }
    return Object.freeze({
      ...cue,
      event_id: cue.cue_id,
      event_type: cue.cue_type,
      transition_id: cue.pov_transition_id,
    });
  });
  return {
    scene,
    eventBatch: Object.freeze({
      schema_version: 1,
      audience: "agent_pov",
      transition_id: incomingTransitionId,
      simulator_step: frame.simulator_step_count,
      events: Object.freeze(normalizedCues),
    }),
  };
}

/**
 * Normalize one validated LiveDebuggerFrameV2 at the API boundary.
 *
 * All downstream browser components consume only `scene` and `event_batch`.
 * The original audience-specific `projection` remains available for the
 * Technical panel, but no downstream component needs to guess projection
 * aliases or filter a privileged researcher frame into a POV frame.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
export function normalizeLiveDebuggerFrameV2(value) {
  const frame = requireRecord(value, "Live debugger frame must be an object.");
  if (frame.schema_version !== 2) {
    throw new TypeError("Live debugger frame must use schema version 2.");
  }
  if (
    (frame.frame_kind !== "researcher_live_debugger" ||
      frame.view_mode !== "researcher") &&
    (frame.frame_kind !== "actor_pov_live_debugger" || frame.view_mode !== "pov")
  ) {
    throw new TypeError("Live debugger frame has an unknown audience mode.");
  }
  if (!Object.hasOwn(frame, "recording")) {
    throw new TypeError("Live debugger frame is missing recording authority.");
  }
  requireExactKeys(
    frame,
    frame.frame_kind === "researcher_live_debugger"
      ? RESEARCHER_LIVE_FRAME_KEYS_V2
      : ACTOR_POV_LIVE_FRAME_KEYS_V2,
    "Live debugger frame has unknown or missing top-level fields.",
  );
  if (
    !Number.isInteger(frame.frame_index) ||
    typeof frame.episode_id !== "string" ||
    frame.frame_id !== `${frame.episode_id}:frame:${frame.frame_index}`
  ) {
    throw new TypeError("Live debugger frame identity is not canonical.");
  }
  const recording = normalizeRecordingStatusV1(frame.recording, frame.frame_index);
  if (frame.frame_kind === "researcher_live_debugger") {
    const expectedTransitionIndex =
      frame.frame_index === 0 ? null : frame.frame_index - 1;
    const expectedTransitionId =
      expectedTransitionIndex === null
        ? null
        : `${frame.episode_id}:transition:${expectedTransitionIndex}`;
    if (
      frame.incoming_transition_index !== expectedTransitionIndex ||
      frame.incoming_transition_id !== expectedTransitionId
    ) {
      throw new TypeError(
        "Live debugger incoming transition identity is not canonical.",
      );
    }
  }
  const projection = requireRecord(
    frame.projection,
    "Live debugger frame is missing its audience projection.",
  );
  const normalized =
    frame.frame_kind === "researcher_live_debugger"
      ? normalizeResearcherProjection(projection, frame)
      : normalizePovProjection(projection, frame);
  const presentationTransitionId =
    frame.frame_kind === "researcher_live_debugger"
      ? frame.incoming_transition_id
      : frame.incoming_pov_transition_id;
  return Object.freeze({
    ...frame,
    recording,
    simulator_step: frame.simulator_step_count,
    transition_id: presentationTransitionId,
    scene: normalized.scene,
    event_batch: normalized.eventBatch,
  });
}

export const researcherEventTypesV2 = Object.freeze([...RESEARCHER_EVENT_TYPES_V2]);

/**
 * Resolve live scripted authority from the audience-owned envelope. POV
 * deliberately omits researcher scenario metadata and exposes this fact only
 * through its recipient-safe HUD submission scope.
 *
 * @param {unknown} value
 */
export function liveDebuggerFrameIsScripted(value) {
  if (!isRecord(value)) {
    return false;
  }
  if (value.frame_kind === "actor_pov_live_debugger") {
    return (
      isRecord(value.hud) && value.hud.pending_submission_scope === "scripted_playback"
    );
  }
  return isRecord(value.scenario) && value.scenario.mode === "scripted";
}

/**
 * Researcher scenario controls are intentionally absent from recipient POV.
 * This answers only transport authority; it never infers scenario metadata.
 *
 * @param {unknown} value
 */
export function liveDebuggerScenarioControlsAvailable(value) {
  return isRecord(value) && value.frame_kind === "researcher_live_debugger";
}
