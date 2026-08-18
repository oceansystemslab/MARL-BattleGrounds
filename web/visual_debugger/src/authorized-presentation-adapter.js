import { exactAuthorizedAgentIdentityV1 } from "./agent-identity.js";
import { isNormalizedAuthorizedPresentationFrameV1 } from "./authorized-presentation-normalizer.js";

/**
 * @typedef {Readonly<Record<string, any>> & {
 *   readonly presentation_kind: string,
 *   readonly viewer_mode: "live" | "replay",
 *   readonly session_id: string,
 *   readonly episode_id: string,
 *   readonly scene: Readonly<Record<string, any>>,
 * }} AuthorizedPresentationFrame
 */

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Identify only the already-normalized five-leaf presentation root. Validation
 * remains owned by `authorized-presentation-normalizer.js`; this adapter never
 * accepts an untrusted wire payload directly.
 *
 * @param {unknown} value
 * @returns {value is AuthorizedPresentationFrame}
 */
export function isAuthorizedPresentationFrame(value) {
  return isNormalizedAuthorizedPresentationFrameV1(value);
}

/**
 * Return the authority-neutral display audience without consulting transport.
 *
 * @param {unknown} value
 * @returns {"researcher" | "agent_pov" | null}
 */
export function authorizedPresentationAudience(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return null;
  }
  return value.presentation_kind.endsWith("oracle") ||
    value.presentation_kind === "live_oracle" ||
    value.presentation_kind === "replay_oracle"
    ? "researcher"
    : "agent_pov";
}

/**
 * @typedef {readonly [
 *   string,
 *   string,
 *   string,
 *   string,
 *   string,
 *   string | null,
 *   string | null,
 *   string | null,
 *   string | null,
 * ]} AuthorizedPresentationPreferenceTuple
 */

/**
 * Project the exact authority identity that scopes inert browser preferences.
 * Scientific refresh epochs are deliberately absent. The structured tuple is
 * retained beside its canonical JSON encoding so callers never need a
 * delimiter-based key.
 *
 * @param {unknown} value
 * @returns {Readonly<{
 *   tuple: AuthorizedPresentationPreferenceTuple,
 *   serialized: string,
 * }> | null}
 */
export function authorizedPresentationPreferenceKey(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return null;
  }
  const tuple = /** @type {AuthorizedPresentationPreferenceTuple} */ (
    Object.freeze([
      value.product_kind,
      value.source.source_session_id,
      value.source.episode_id,
      value.presentation_kind,
      value.authority.authority_kind,
      value.authority.observation_mode ?? null,
      value.source.source_artifact_id ?? null,
      value.authority.recipient_public_agent_id ?? null,
      value.authority.recipient_presentation_key ?? null,
    ])
  );
  return Object.freeze({ tuple, serialized: JSON.stringify(tuple) });
}

/**
 * Compare only well-formed structured preference keys. Rechecking each
 * canonical encoding prevents a caller-supplied stale string from overriding
 * the tuple fields that define the authority boundary.
 *
 * @param {unknown} left
 * @param {unknown} right
 * @returns {boolean}
 */
export function sameAuthorizedPresentationPreferenceKey(left, right) {
  if (!isRecord(left) || !isRecord(right)) {
    return false;
  }
  const leftTuple = left.tuple;
  const rightTuple = right.tuple;
  if (
    !Array.isArray(leftTuple) ||
    !Array.isArray(rightTuple) ||
    leftTuple.length !== 9 ||
    rightTuple.length !== 9 ||
    typeof left.serialized !== "string" ||
    typeof right.serialized !== "string" ||
    left.serialized !== JSON.stringify(leftTuple) ||
    right.serialized !== JSON.stringify(rightTuple) ||
    left.serialized !== right.serialized
  ) {
    return false;
  }
  return leftTuple.every((entry, index) => Object.is(entry, rightTuple[index]));
}

/**
 * Scope one opaque Python-owned presentation key to its installed browser
 * authority. The returned string is an internal map key only; it is never a
 * simulator slot or a command capability.
 *
 * @param {unknown} value
 * @param {unknown} presentationKey
 * @returns {string | null}
 */
export function scopedPresentationKey(value, presentationKey) {
  if (
    !isAuthorizedPresentationFrame(value) ||
    typeof presentationKey !== "string" ||
    presentationKey.length === 0
  ) {
    return null;
  }
  return JSON.stringify([value.session_id, presentationKey]);
}

/**
 * Enumerate exact presentation identities for retained scene/roster nodes.
 * Oracle command slots are derived only from the fixed ten-row public identity
 * directory and remain outside body records. Agent rows never gain a command
 * slot.
 *
 * @param {unknown} value
 * @returns {ReadonlyArray<Readonly<{
 *   display_key: string,
 *   presentation_key: string,
 *   public_agent_id: string,
 *   command_global_slot: number | null,
 *   agent: Readonly<Record<string, any>>,
 * }>>}
 */
export function authorizedPresentationIdentityRows(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return Object.freeze([]);
  }
  const audience = authorizedPresentationAudience(value);
  const directory =
    audience === "researcher" && isRecord(value.current_endpoint)
      ? value.current_endpoint.identity_directory
      : null;
  const commandSlotByPublicId = new Map();
  if (isRecord(directory) && Array.isArray(directory.identities)) {
    for (const row of directory.identities) {
      if (
        isRecord(row) &&
        row.configured_active === true &&
        typeof row.public_agent_id === "string" &&
        (row.team_id === 1 || row.team_id === 2) &&
        Number.isInteger(row.team_local_slot) &&
        row.team_local_slot >= 0 &&
        row.team_local_slot < 5
      ) {
        commandSlotByPublicId.set(
          row.public_agent_id,
          (row.team_id - 1) * 5 + row.team_local_slot,
        );
      }
    }
  }

  const rows = [];
  for (const agent of Array.isArray(value.scene.agents) ? value.scene.agents : []) {
    if (
      !isRecord(agent) ||
      typeof agent.presentation_key !== "string" ||
      typeof agent.public_agent_id !== "string"
    ) {
      continue;
    }
    const displayKey = scopedPresentationKey(value, agent.presentation_key);
    if (displayKey === null) {
      continue;
    }
    rows.push(
      Object.freeze({
        display_key: displayKey,
        presentation_key: agent.presentation_key,
        public_agent_id: agent.public_agent_id,
        command_global_slot:
          audience === "researcher"
            ? (commandSlotByPublicId.get(agent.public_agent_id) ?? null)
            : null,
        agent,
      }),
    );
  }
  return Object.freeze(rows);
}

/**
 * Resolve an Oracle command-transport slot from the validated fixed identity
 * directory. Agent POV presentation keys never resolve to command slots.
 *
 * @param {unknown} value
 * @param {unknown} presentationKey
 * @returns {number | null}
 */
export function authorizedOracleCommandSlotForPresentationKey(value, presentationKey) {
  if (typeof presentationKey !== "string") {
    return null;
  }
  const row = authorizedPresentationIdentityRows(value).find(
    (identity) => identity.presentation_key === presentationKey,
  );
  return row?.command_global_slot ?? null;
}

/**
 * Resolve an Oracle target-axis public identity through the accepted fixed
 * identity directory. Unlike visible presentation-key rows, the 11-category
 * target axis deliberately includes axis-only inactive identities. Requiring
 * the public ID to occur on that exact axis prevents this directory lookup
 * from becoming a general capability mint. Agent POV roots always return null.
 *
 * @param {unknown} value
 * @param {unknown} publicAgentId
 * @returns {number | null}
 */
export function authorizedOracleCommandSlotForPublicAgentId(value, publicAgentId) {
  if (
    !isAuthorizedPresentationFrame(value) ||
    authorizedPresentationAudience(value) !== "researcher" ||
    typeof publicAgentId !== "string" ||
    !isRecord(value.action_axis) ||
    !Array.isArray(value.action_axis.target_actions) ||
    !value.action_axis.target_actions.some(
      (target) => isRecord(target) && target.target_public_agent_id === publicAgentId,
    ) ||
    !isRecord(value.current_endpoint?.identity_directory) ||
    !Array.isArray(value.current_endpoint.identity_directory.identities)
  ) {
    return null;
  }
  const identity = value.current_endpoint.identity_directory.identities.find(
    (/** @type {unknown} */ row) =>
      isRecord(row) && row.public_agent_id === publicAgentId,
  );
  return isRecord(identity) &&
    (identity.team_id === 1 || identity.team_id === 2) &&
    Number.isInteger(identity.team_local_slot) &&
    identity.team_local_slot >= 0 &&
    identity.team_local_slot < 5
    ? (identity.team_id - 1) * 5 + identity.team_local_slot
    : null;
}

/** @param {unknown} value */
function presentationInspection(value) {
  return authorizedPresentationInspectionState(value).inspection;
}

/**
 * Preserve the four exact inspection states instead of treating absence as
 * scripted playback or collapsing Oracle joint-turn scope to controlled actor.
 *
 * @param {unknown} value
 * @returns {Readonly<{
 *   state_kind: "live_editable" | "live_scripted" | "replay_outgoing" | "replay_none" | "unavailable",
 *   submission_scope: "joint_turn" | "controlled_actor" | "scripted_playback" | null,
 *   inspection: Readonly<Record<string, any>> | null,
 * }>}
 */
export function authorizedPresentationInspectionState(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return Object.freeze({
      state_kind: "unavailable",
      submission_scope: null,
      inspection: null,
    });
  }
  if (value.viewer_mode === "live") {
    const live = isRecord(value.live_inspection)
      ? value.live_inspection.inspection
      : null;
    if (isRecord(live) && live.inspection_kind === "editable_live_draft") {
      return Object.freeze({
        state_kind: "live_editable",
        submission_scope: live.submission_scope,
        inspection: isRecord(live.draft) ? live.draft : null,
      });
    }
    return Object.freeze({
      state_kind: "live_scripted",
      submission_scope: isRecord(live) ? live.submission_scope : null,
      inspection: null,
    });
  }
  return isRecord(value.inspection)
    ? Object.freeze({
        state_kind: "replay_outgoing",
        submission_scope: null,
        inspection: value.inspection,
      })
    : Object.freeze({
        state_kind: "replay_none",
        submission_scope: null,
        inspection: null,
      });
}

/**
 * @param {unknown} rawStatus
 * @returns {Readonly<Record<string, any>> | null}
 */
function statusView(rawStatus) {
  if (!isRecord(rawStatus)) {
    return null;
  }
  return Object.freeze({
    ...rawStatus,
    token_id: rawStatus.status_id,
    duration: rawStatus.remaining_duration,
  });
}

/**
 * @param {unknown} rawModifier
 * @returns {Readonly<Record<string, any>> | null}
 */
function modifierView(rawModifier) {
  if (!isRecord(rawModifier)) {
    return null;
  }
  return Object.freeze({ ...rawModifier, token_id: rawModifier.aura_id });
}

/**
 * @param {unknown} rawAgent
 * @param {AuthorizedPresentationFrame} presentation
 * @returns {Readonly<Record<string, any>> | null}
 */
function agentView(rawAgent, presentation) {
  if (!isRecord(rawAgent)) {
    return null;
  }
  const displayKey = scopedPresentationKey(presentation, rawAgent.presentation_key);
  if (displayKey === null) {
    return null;
  }
  return Object.freeze({
    ...rawAgent,
    display_key: displayKey,
    alive: rawAgent.life_state === "alive",
    max_health: rawAgent.maximum_health,
    ultimate_cooldown: rawAgent.ultimate_cooldown_remaining,
    statuses: Object.freeze(
      (Array.isArray(rawAgent.statuses) ? rawAgent.statuses : [])
        .map(statusView)
        .filter((status) => status !== null),
    ),
    modifiers: Object.freeze(
      (Array.isArray(rawAgent.aura_modifiers) ? rawAgent.aura_modifiers : [])
        .map(modifierView)
        .filter((modifier) => modifier !== null),
    ),
  });
}

/**
 * @param {unknown} rawField
 * @returns {Readonly<Record<string, any>> | null}
 */
function auraFieldView(rawField) {
  return isRecord(rawField)
    ? Object.freeze({ ...rawField, token_id: rawField.aura_id })
    : null;
}

/**
 * Return the exact outgoing inspection owned by the presentation, unwrapping
 * only the live editable-draft envelope. Scripted playback has no outgoing
 * inspection overlays.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>> | null}
 */
export function authorizedPresentationInspection(value) {
  return presentationInspection(value);
}

/**
 * Project the exact actor-owned legality row from one already-certified
 * outgoing inspection. This helper does not authorize input; the production
 * caller invokes it only after the five-leaf presentation brand has passed.
 * Keeping the projection pure makes the target-disclosure/lane cross-product
 * testable without forging a presentation root.
 *
 * @param {unknown} rawDecisionMask
 * @param {unknown} rawOwner
 * @param {unknown} rawTarget
 * @param {unknown} rawLaneName
 * @returns {Readonly<Record<string, any>> | null}
 */
export function projectCertifiedInspectionLegality(
  rawDecisionMask,
  rawOwner,
  rawTarget,
  rawLaneName,
) {
  const decisionMask = isRecord(rawDecisionMask) ? rawDecisionMask : null;
  const owner = isRecord(rawOwner) ? rawOwner : null;
  const target = isRecord(rawTarget) ? rawTarget : null;
  const targetAction =
    target !== null && Number.isInteger(target.target_action)
      ? Number(target.target_action)
      : null;
  const certifiedTarget =
    decisionMask !== null && targetAction !== null
      ? decisionMask.target_actions?.[targetAction]
      : null;
  const pairRow =
    decisionMask !== null && targetAction !== null
      ? decisionMask.target_use_ultimate_joint_mask?.[targetAction]
      : null;
  if (
    decisionMask === null ||
    owner === null ||
    target === null ||
    typeof owner.presentation_key !== "string" ||
    typeof owner.public_agent_id !== "string" ||
    decisionMask.owner_presentation_key !== owner.presentation_key ||
    decisionMask.owner_public_agent_id !== owner.public_agent_id ||
    targetAction === null ||
    !isRecord(certifiedTarget) ||
    certifiedTarget.target_action !== targetAction ||
    certifiedTarget.target_kind !== target.target_kind ||
    certifiedTarget.display_name !== target.display_name ||
    !Array.isArray(pairRow) ||
    pairRow.length !== 2 ||
    typeof pairRow[0] !== "boolean" ||
    typeof pairRow[1] !== "boolean" ||
    !["no_target", "visible_authorized_agent", "axis_only_authorized_agent"].includes(
      target.target_kind,
    ) ||
    typeof rawLaneName !== "string" ||
    !["none", "basic", "ultimate"].includes(rawLaneName) ||
    (target.target_kind === "visible_authorized_agent" &&
      (certifiedTarget.target_presentation_key !== target.target_presentation_key ||
        certifiedTarget.target_public_agent_id !== target.target_public_agent_id ||
        !Array.isArray(certifiedTarget.target_anchor) ||
        !Array.isArray(target.target_anchor) ||
        certifiedTarget.target_anchor.length !== target.target_anchor.length ||
        certifiedTarget.target_anchor.some(
          (coordinate, index) => coordinate !== target.target_anchor[index],
        ))) ||
    (target.target_kind === "axis_only_authorized_agent" &&
      certifiedTarget.target_public_agent_id !== target.target_public_agent_id)
  ) {
    return null;
  }
  const lane = rawLaneName === "basic" ? 0 : rawLaneName === "ultimate" ? 1 : null;
  return Object.freeze({
    owner_presentation_key: owner.presentation_key,
    owner_public_agent_id: owner.public_agent_id,
    target_action: targetAction,
    target_kind: target.target_kind,
    target_display_name:
      typeof target.display_name === "string" ? target.display_name : null,
    target_presentation_key:
      target.target_kind === "visible_authorized_agent" &&
      typeof target.target_presentation_key === "string"
        ? target.target_presentation_key
        : null,
    target_public_agent_id:
      target.target_kind !== "no_target" &&
      typeof target.target_public_agent_id === "string"
        ? target.target_public_agent_id
        : null,
    lane_0_available: pairRow[0],
    lane_1_available: pairRow[1],
    armed_lane: lane,
    armed_pair_legal: lane === null ? null : pairRow[lane],
  });
}

/**
 * Project one visible-target route from already-certified inspection facts.
 * No-target and axis-only disclosures deliberately have no drawable route.
 *
 * @param {unknown} rawInspection
 * @param {unknown} rawOwner
 * @param {unknown} rawTargetAgent
 * @param {unknown} rawTarget
 * @param {unknown} rawLegality
 * @returns {Readonly<Record<string, any>> | null}
 */
export function projectCertifiedInspectionRoute(
  rawInspection,
  rawOwner,
  rawTargetAgent,
  rawTarget,
  rawLegality,
) {
  const inspection = isRecord(rawInspection) ? rawInspection : null;
  const owner = isRecord(rawOwner) ? rawOwner : null;
  const targetAgent = isRecord(rawTargetAgent) ? rawTargetAgent : null;
  const target = isRecord(rawTarget) ? rawTarget : null;
  const legality = isRecord(rawLegality) ? rawLegality : null;
  if (
    inspection === null ||
    owner === null ||
    targetAgent === null ||
    target === null ||
    legality === null ||
    target.target_kind !== "visible_authorized_agent" ||
    target.target_presentation_key !== targetAgent.presentation_key ||
    target.target_public_agent_id !== targetAgent.public_agent_id ||
    legality.owner_presentation_key !== owner.presentation_key ||
    legality.owner_public_agent_id !== owner.public_agent_id ||
    legality.target_presentation_key !== targetAgent.presentation_key ||
    legality.target_public_agent_id !== targetAgent.public_agent_id ||
    legality.target_action !== target.target_action ||
    !Array.isArray(inspection.actor_anchor) ||
    !Array.isArray(target.target_anchor) ||
    (legality.armed_lane !== 0 && legality.armed_lane !== 1) ||
    typeof legality.armed_pair_legal !== "boolean"
  ) {
    return null;
  }
  return Object.freeze({
    source_presentation_key: owner.presentation_key,
    source_public_agent_id: owner.public_agent_id,
    source_anchor: inspection.actor_anchor,
    source_radius: owner.radius,
    target_presentation_key: targetAgent.presentation_key,
    target_public_agent_id: targetAgent.public_agent_id,
    target_anchor: target.target_anchor,
    target_radius: targetAgent.radius,
    lane: legality.armed_lane,
    legal: legality.armed_pair_legal,
  });
}

/**
 * Adapt exact authorized scene/inspection field names to the retained visual
 * components. Every numeric or categorical fact is copied from one accepted
 * presentation branch. This view adds no global slot and no command field.
 *
 * @param {unknown} value
 * @param {string | null | undefined} [localInspectedPresentationKey]
 * @returns {Readonly<Record<string, any>> | null}
 */
export function authorizedPresentationSceneView(
  value,
  localInspectedPresentationKey = undefined,
) {
  if (!isAuthorizedPresentationFrame(value)) {
    return null;
  }
  const audience = authorizedPresentationAudience(value);
  const agents = Object.freeze(
    (Array.isArray(value.scene.agents) ? value.scene.agents : [])
      .map((agent) => agentView(agent, value))
      .filter((agent) => agent !== null),
  );
  const agentByKey = new Map(agents.map((agent) => [agent.presentation_key, agent]));
  const inspectionState = authorizedPresentationInspectionState(value);
  const inspection = inspectionState.inspection;
  const inspectionOwnerKey =
    inspection && typeof inspection.actor_presentation_key === "string"
      ? inspection.actor_presentation_key
      : null;
  const inspectionOwnerPublicId =
    inspection && typeof inspection.actor_public_agent_id === "string"
      ? inspection.actor_public_agent_id
      : null;
  const axisOwnerKey =
    value.viewer_mode === "replay" &&
    isRecord(value.action_axis) &&
    typeof value.action_axis.owner_presentation_key === "string"
      ? value.action_axis.owner_presentation_key
      : null;
  const axisOwnerPublicId =
    value.viewer_mode === "replay" &&
    isRecord(value.action_axis) &&
    typeof value.action_axis.owner_public_agent_id === "string"
      ? value.action_axis.owner_public_agent_id
      : null;
  const ownerCandidateKey = inspectionOwnerKey ?? axisOwnerKey;
  const ownerCandidatePublicId = inspectionOwnerPublicId ?? axisOwnerPublicId;
  const ownerCandidate =
    ownerCandidateKey === null ? null : (agentByKey.get(ownerCandidateKey) ?? null);
  const actor =
    ownerCandidate !== null && ownerCandidate.public_agent_id === ownerCandidatePublicId
      ? ownerCandidate
      : null;
  const ownerKey = actor?.presentation_key ?? null;
  const hasLocalInspection =
    localInspectedPresentationKey !== undefined &&
    (audience === "agent_pov" || inspectionState.state_kind === "live_scripted");
  const inspectedActor = hasLocalInspection
    ? typeof localInspectedPresentationKey === "string"
      ? (agentByKey.get(localInspectedPresentationKey) ?? null)
      : null
    : actor;
  const inspectedOwnerKey = inspectedActor?.presentation_key ?? null;
  const target =
    inspection &&
    isRecord(
      inspection.inspection_kind === "live_draft_action"
        ? inspection.draft_target
        : inspection.accepted_target,
    )
      ? inspection.inspection_kind === "live_draft_action"
        ? inspection.draft_target
        : inspection.accepted_target
      : null;
  const targetKey =
    isRecord(target) &&
    target.target_kind === "visible_authorized_agent" &&
    typeof target.target_presentation_key === "string"
      ? target.target_presentation_key
      : null;
  const targetAgent = targetKey === null ? null : (agentByKey.get(targetKey) ?? null);
  const decisionMask =
    inspection && isRecord(inspection.decision_mask) ? inspection.decision_mask : null;
  const action =
    inspection?.inspection_kind === "live_draft_action"
      ? inspection.draft_action
      : inspection?.inspection_kind === "replay_recorded_outgoing_action"
        ? inspection.accepted_action
        : null;
  const laneName =
    inspection?.inspection_kind === "live_draft_action"
      ? action?.armed_lane
      : inspection?.combat_lane;
  const actionOwnerLegality =
    inspection && actor && decisionMask && target
      ? projectCertifiedInspectionLegality(decisionMask, actor, target, laneName)
      : null;
  const actionOwnerRoute = projectCertifiedInspectionRoute(
    inspection,
    actor,
    targetAgent,
    target,
    actionOwnerLegality,
  );
  const inspectedOwnerOwnsOutgoingAction =
    !hasLocalInspection ||
    (actor !== null &&
      inspectedActor !== null &&
      inspectedActor.presentation_key === actor.presentation_key &&
      inspectedActor.public_agent_id === actor.public_agent_id);
  const selectedLegality = inspectedOwnerOwnsOutgoingAction
    ? actionOwnerLegality
    : null;
  const pendingRoute = inspectedOwnerOwnsOutgoingAction ? actionOwnerRoute : null;
  const ranges =
    inspectedActor === null
      ? Object.freeze([])
      : Object.freeze(
          [
            ["observation", inspectedActor.observation_radius],
            ["basic", inspectedActor.basic_interaction_radius],
            ["ultimate", inspectedActor.ultimate_interaction_radius],
          ].map(([kind, radius]) =>
            Object.freeze({
              kind,
              presentation_key: inspectedActor.presentation_key,
              public_agent_id: inspectedActor.public_agent_id,
              center: inspectedActor.position,
              radius,
            }),
          ),
        );
  const selectedKey = hasLocalInspection
    ? inspectedOwnerKey
    : value.viewer_mode === "replay"
      ? ownerKey
      : targetKey;

  return Object.freeze({
    audience,
    map: value.scene.map,
    agents,
    aura_fields: Object.freeze(
      (Array.isArray(value.scene.aura_fields) ? value.scene.aura_fields : [])
        .map(auraFieldView)
        .filter((field) => field !== null),
    ),
    class_mechanics: value.scene.class_mechanics,
    spawn_shield_mechanics: value.scene.spawn_shield_mechanics,
    spawn_pads: value.scene.spawn_pads,
    respawn_waves: value.scene.respawn_waves,
    selection: Object.freeze({
      controlled_presentation_key: value.viewer_mode === "live" ? ownerKey : null,
      inspection_owner_presentation_key: inspectedOwnerKey,
      selected_presentation_key: selectedKey,
    }),
    ranges,
    pending_route: pendingRoute,
    selected_legality: selectedLegality,
  });
}

/**
 * Flatten only the exact Latest Events branch for panel/choreography dispatch.
 * Shared observation deltas retain an explicitly noncausal vocabulary.
 *
 * @param {unknown} value
 * @returns {ReadonlyArray<Readonly<{
 *   id: string,
 *   kind: string,
 *   vocabulary: "event" | "recipient_cue" | "observation_delta",
 *   payload: Readonly<Record<string, any>>,
 * }>>}
 */
export function authorizedPresentationIncomingRows(value) {
  if (!isAuthorizedPresentationFrame(value) || !isRecord(value.latest_events)) {
    return Object.freeze([]);
  }
  const latest = value.latest_events;
  /** @type {[unknown[], string, string, "event" | "recipient_cue" | "observation_delta"] | null} */
  const source = Array.isArray(latest.events)
    ? [latest.events, "event_id", "event_kind", "event"]
    : Array.isArray(latest.cues)
      ? [latest.cues, "cue_id", "cue_type", "recipient_cue"]
      : Array.isArray(latest.deltas)
        ? [latest.deltas, "cue_id", "delta_kind", "observation_delta"]
        : null;
  if (source === null) {
    return Object.freeze([]);
  }
  const [rawRows, idField, kindField, vocabulary] = source;
  return Object.freeze(
    rawRows.flatMap((payload) =>
      isRecord(payload) &&
      typeof payload[idField] === "string" &&
      typeof payload[kindField] === "string"
        ? [
            Object.freeze({
              id: payload[idField],
              kind: payload[kindField],
              vocabulary,
              payload,
            }),
          ]
        : [],
    ),
  );
}

/** @param {unknown} value */
export function authorizedPresentationTransitionRows(value) {
  if (
    !isAuthorizedPresentationFrame(value) ||
    !isRecord(value.latest_transition) ||
    !Array.isArray(value.latest_transition.action_rows)
  ) {
    return Object.freeze([]);
  }
  const identityByPresentationKey = new Map(
    authorizedPresentationIdentityRows(value).map((identity) => [
      identity.presentation_key,
      identity,
    ]),
  );
  const rows = [];
  for (const rawRow of value.latest_transition.action_rows) {
    const identity = isRecord(rawRow)
      ? identityByPresentationKey.get(rawRow.actor_presentation_key)
      : null;
    const actorIdentity = exactAuthorizedAgentIdentityV1(identity?.agent);
    if (
      !isRecord(rawRow) ||
      identity === null ||
      identity === undefined ||
      actorIdentity === null ||
      actorIdentity.presentationKey !== rawRow.actor_presentation_key ||
      actorIdentity.publicAgentId !== rawRow.actor_public_agent_id ||
      identity.public_agent_id !== rawRow.actor_public_agent_id ||
      !isRecord(rawRow.submitted_action) ||
      !isRecord(rawRow.accepted_action)
    ) {
      return Object.freeze([]);
    }
    const submittedAction = Object.freeze({
      move_action: rawRow.submitted_action.move_action,
      target_action: rawRow.submitted_action.target_action,
      use_ultimate_action: rawRow.submitted_action.use_ultimate_action,
    });
    const acceptedAction = Object.freeze({
      move_action: rawRow.accepted_action.move_action,
      target_action: rawRow.accepted_action.target_action,
      use_ultimate_action: rawRow.accepted_action.use_ultimate_action,
    });
    if (
      !Object.values(submittedAction).every(Number.isInteger) ||
      !Object.values(acceptedAction).every(Number.isInteger)
    ) {
      return Object.freeze([]);
    }
    rows.push(
      Object.freeze({
        actor_title: actorIdentity.title,
        actor_accent: actorIdentity.accent,
        submitted_action: submittedAction,
        accepted_action: acceptedAction,
      }),
    );
  }
  return Object.freeze(rows);
}

/** @typedef {"nonnegative_integer" | "positive_finite_number" | "scientific_id" | "optional_scientific_id" | "sha256_prefix"} TechnicalFactValueKind */
/** @typedef {readonly [string, string, string, TechnicalFactValueKind]} TechnicalFactSpecification */

const TECHNICAL_FIELD_UNAVAILABLE = Symbol("technical-field-unavailable");
const SCIENTIFIC_ID_PATTERN = /^[-A-Za-z0-9_.:/+]+$/u;
const SHA256_PREFIX_PATTERN = /^[0-9a-f]{12}$/u;

/**
 * @param {string} id
 * @param {string} label
 * @param {string} field
 * @param {TechnicalFactValueKind} valueKind
 * @returns {TechnicalFactSpecification}
 */
function technicalFactSpecification(id, label, field, valueKind) {
  return Object.freeze([id, label, field, valueKind]);
}

const TECHNICAL_FACT_SPECIFICATIONS = Object.freeze({
  live_oracle_technical_frame: Object.freeze([
    technicalFactSpecification("episode", "Episode", "episode_id", "scientific_id"),
    technicalFactSpecification(
      "frame",
      "Frame",
      "evaluation_frame_index",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "simulator_step",
      "Simulator step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming transition",
      "incoming_transition_id",
      "optional_scientific_id",
    ),
  ]),
  live_no_shared_obs_technical_frame: Object.freeze([
    technicalFactSpecification("episode", "Episode", "episode_id", "scientific_id"),
    technicalFactSpecification(
      "frame",
      "Frame",
      "recipient_frame_index",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "simulator_step",
      "Simulator step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming transition",
      "incoming_recipient_transition_id",
      "optional_scientific_id",
    ),
  ]),
  replay_oracle_technical_frame: Object.freeze([
    technicalFactSpecification(
      "artifact_digest_prefix",
      "Artifact digest prefix",
      "artifact_digest_prefix",
      "sha256_prefix",
    ),
    technicalFactSpecification("frame", "Frame", "frame_index", "nonnegative_integer"),
    technicalFactSpecification(
      "simulator_step",
      "Simulator step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming transition",
      "incoming_transition_id",
      "optional_scientific_id",
    ),
    technicalFactSpecification(
      "ordinary_movement_distance_scale",
      "Ordinary movement distance scale",
      "recorded_ordinary_movement_distance_scale",
      "positive_finite_number",
    ),
  ]),
  replay_no_shared_obs_technical_frame: Object.freeze([
    technicalFactSpecification("frame", "Frame", "frame_index", "nonnegative_integer"),
    technicalFactSpecification(
      "simulator_step",
      "Simulator step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming transition",
      "incoming_recipient_transition_id",
      "optional_scientific_id",
    ),
  ]),
  replay_shared_obs_technical_frame: Object.freeze([
    technicalFactSpecification("frame", "Frame", "frame_index", "nonnegative_integer"),
    technicalFactSpecification(
      "simulator_step",
      "Simulator step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming transition",
      "incoming_recipient_transition_id",
      "optional_scientific_id",
    ),
  ]),
});

/**
 * Read one finite, own, enumerable data field without invoking an accessor.
 * The normalized root already rejects extras; this adapter deliberately never
 * enumerates the Technical Frame or consults a broader source as fallback.
 *
 * @param {Record<string, any>} value
 * @param {string} field
 */
function technicalFrameDataValue(value, field) {
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, field);
    return descriptor?.enumerable === true && Object.hasOwn(descriptor, "value")
      ? descriptor.value
      : TECHNICAL_FIELD_UNAVAILABLE;
  } catch {
    return TECHNICAL_FIELD_UNAVAILABLE;
  }
}

/** @param {unknown} value */
function isScientificId(value) {
  return (
    typeof value === "string" &&
    value.length >= 1 &&
    value.length <= 512 &&
    SCIENTIFIC_ID_PATTERN.test(value)
  );
}

/**
 * @param {TechnicalFactValueKind} valueKind
 * @param {unknown} value
 */
function isTechnicalFactValue(valueKind, value) {
  if (valueKind === "nonnegative_integer") {
    return Number.isInteger(value) && Number(value) >= 0;
  }
  if (valueKind === "positive_finite_number") {
    return typeof value === "number" && Number.isFinite(value) && value > 0;
  }
  if (valueKind === "scientific_id") {
    return isScientificId(value);
  }
  if (valueKind === "optional_scientific_id") {
    return value === null || isScientificId(value);
  }
  return typeof value === "string" && SHA256_PREFIX_PATTERN.test(value);
}

/** @param {unknown} value */
export function authorizedPresentationTechnicalFacts(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return Object.freeze([]);
  }
  const technicalFrame = technicalFrameDataValue(value, "technical_frame");
  if (technicalFrame === TECHNICAL_FIELD_UNAVAILABLE || !isRecord(technicalFrame)) {
    return Object.freeze([]);
  }
  const technicalKind = technicalFrameDataValue(technicalFrame, "technical_kind");
  if (
    typeof technicalKind !== "string" ||
    !Object.hasOwn(TECHNICAL_FACT_SPECIFICATIONS, technicalKind)
  ) {
    return Object.freeze([]);
  }
  const specification =
    TECHNICAL_FACT_SPECIFICATIONS[
      /** @type {keyof typeof TECHNICAL_FACT_SPECIFICATIONS} */ (technicalKind)
    ];
  /** @type {Array<readonly [string, string, unknown]>} */
  const snapshot = [];
  for (const [id, label, field, valueKind] of specification) {
    const factValue = technicalFrameDataValue(technicalFrame, field);
    if (
      factValue === TECHNICAL_FIELD_UNAVAILABLE ||
      !isTechnicalFactValue(valueKind, factValue)
    ) {
      return Object.freeze([]);
    }
    snapshot.push(Object.freeze([id, label, factValue]));
  }
  return Object.freeze(
    snapshot
      .filter(([, , factValue]) => factValue !== null)
      .map(([id, label, factValue]) => Object.freeze({ id, label, value: factValue })),
  );
}
