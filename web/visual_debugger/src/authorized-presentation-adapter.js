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
 * Adapt exact authorized scene/inspection field names to the retained visual
 * components. Every numeric or categorical fact is copied from one accepted
 * presentation branch. This view adds no global slot and no command field.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>> | null}
 */
export function authorizedPresentationSceneView(value) {
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
  const inspection = presentationInspection(value);
  const ownerKey =
    inspection && typeof inspection.actor_presentation_key === "string"
      ? inspection.actor_presentation_key
      : null;
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
  const actor = ownerKey === null ? null : (agentByKey.get(ownerKey) ?? null);
  const selected = targetKey === null ? null : (agentByKey.get(targetKey) ?? null);
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
  const lane = laneName === "basic" ? 0 : laneName === "ultimate" ? 1 : null;
  const targetAction = Number.isInteger(target?.target_action)
    ? Number(target.target_action)
    : null;
  const pairMask =
    decisionMask && targetAction !== null && lane !== null
      ? decisionMask.target_use_ultimate_joint_mask?.[targetAction]?.[lane]
      : null;
  const routeLegal =
    inspection?.inspection_kind === "live_draft_action"
      ? inspection.draft_legality?.combat_pair_is_legal
      : inspection?.inspection_kind === "replay_recorded_outgoing_action"
        ? true
        : null;
  const pendingRoute =
    inspection && actor && selected && target && lane !== null
      ? Object.freeze({
          source_presentation_key: actor.presentation_key,
          source_public_agent_id: actor.public_agent_id,
          source_anchor: inspection.actor_anchor,
          source_radius: actor.radius,
          target_presentation_key: selected.presentation_key,
          target_public_agent_id: selected.public_agent_id,
          target_anchor: target.target_anchor,
          target_radius: selected.radius,
          lane,
          legal: routeLegal,
        })
      : null;
  const ranges =
    actor === null || inspection === null
      ? Object.freeze([])
      : Object.freeze(
          [
            ["basic", actor.basic_interaction_radius],
            ["ultimate", actor.ultimate_interaction_radius],
          ].map(([kind, radius]) =>
            Object.freeze({
              kind,
              presentation_key: actor.presentation_key,
              public_agent_id: actor.public_agent_id,
              center: inspection.actor_anchor,
              radius,
            }),
          ),
        );
  const selectedLegality =
    selected && decisionMask && targetAction !== null
      ? Object.freeze({
          target_presentation_key: selected.presentation_key,
          target_public_agent_id: selected.public_agent_id,
          lane_0_available:
            decisionMask.target_use_ultimate_joint_mask[targetAction][0],
          lane_1_available:
            decisionMask.target_use_ultimate_joint_mask[targetAction][1],
          armed_lane: lane,
          armed_pair_legal: pairMask,
        })
      : null;

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
      inspection_owner_presentation_key: ownerKey,
      selected_presentation_key: targetKey,
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
  return Object.freeze([...value.latest_transition.action_rows]);
}

/** @param {unknown} value */
export function authorizedPresentationTechnicalFacts(value) {
  if (!isAuthorizedPresentationFrame(value) || !isRecord(value.technical_frame)) {
    return Object.freeze([]);
  }
  return Object.freeze(
    Object.entries(value.technical_frame).map(([label, factValue]) =>
      Object.freeze({ label, value: factValue }),
    ),
  );
}
