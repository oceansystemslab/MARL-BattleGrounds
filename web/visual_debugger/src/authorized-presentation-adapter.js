/**
 * @file Turn an already approved presentation into browser display rows and overlays.
 * The normalizer owns wire validation and the private approval marker. This module
 * accepts only its marked roots, copies allowed facts, and keeps battlefield geometry
 * separate from the global researcher panels attached to Agent POV presentations.
 * Public helpers return frozen display objects, empty arrays, or null when a view is
 * unavailable. They do not send commands, fetch data, run simulation, or grant authority.
 * Helpers named projectCertified require the caller to supply already approved facts.
 */
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
 * Return whether value is a non-null object that is not an array.
 * This small shape check does not prove that an object has presentation authority.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return whether value is the exact object marked by the presentation normalizer.
 * A look-alike wire object or copied object is false. This check delegates to the
 * normalizer; it does not validate or approve new input.
 *
 * @param {unknown} value
 * @returns {value is AuthorizedPresentationFrame}
 */
export function isAuthorizedPresentationFrame(value) {
  return isNormalizedAuthorizedPresentationFrameV1(value);
}

/**
 * Return the display audience of an approved value: researcher for Oracle, otherwise
 * agent_pov. Return null for an unmarked value. Attached researcher panels do not
 * change the actor-view audience or authorize wider battlefield geometry.
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
 * Build a frozen authority key for inert browser preferences from approved value.
 * Return null for unmarked input. The tuple holds product, source session, episode,
 * presentation kind, authority kind, observation mode, source artifact, recipient
 * public ID, and recipient presentation key; absent optional fields become null.
 * The result includes both tuple and its JSON string. Refresh revisions are excluded
 * so ordinary updates preserve preferences. This key is not permission to act.
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
 * Compare left and right preference-key records, returning false if either is
 * malformed or its serialized string differs from JSON.stringify of its own tuple.
 * Both tuples must have nine entries; their strings and every entry must match.
 * This compares key structure and values, not whether either key came from an approved
 * presentation. Call the preference-key builder at the authority boundary.
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
 * Return a JSON-encoded pair of value.session_id and presentationKey.
 * value must be an approved root and presentationKey a nonempty string; otherwise
 * return null. The string is a browser map key, not a simulator slot or command
 * permission. This helper does not check that the key names an existing body.
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
 * Return value's separately approved global researcher branch for an Agent POV,
 * or null when absent, wrong for live/replay mode, or not an Agent POV. The caller
 * supplies an approved presentation. This branch is for researcher panels; it is not
 * a replacement for the actor's battlefield scene.
 *
 * @param {AuthorizedPresentationFrame} value
 */
function researcherSpace(value) {
  if (
    authorizedPresentationAudience(value) !== "agent_pov" ||
    !isRecord(value.researcher_space)
  ) {
    return null;
  }
  const expectedKind =
    value.viewer_mode === "live"
      ? "global_live_researcher_space"
      : "global_replay_researcher_space";
  return value.researcher_space.researcher_space_kind === expectedKind
    ? value.researcher_space
    : null;
}

/**
 * Return true only when value is approved and has the mode-matching global
 * researcher branch. A false result means no such branch is available; it does not
 * identify the display audience by itself.
 *
 * @param {unknown} value
 */
export function authorizedPresentationHasResearcherSpace(value) {
  return isAuthorizedPresentationFrame(value) && researcherSpace(value) !== null;
}

/** @type {WeakMap<object, Map<string, number>>} */
const identitySlotCache = new WeakMap();

/**
 * Return the cached public-ID-to-global-slot map for immutable approved value.
 * Prefer its researcher identity directory, then the current endpoint directory.
 * Only configured-active rows with Team 1/2 and local slots 0 through 4 enter the map.
 * The derived slot is 0 through 9. The WeakMap cache is shared by display and command
 * lookups; callers must not mutate the returned Map.
 *
 * @param {AuthorizedPresentationFrame} value
 */
function identitySlots(value) {
  const cached = identitySlotCache.get(value);
  if (cached) return cached;
  const directory =
    researcherSpace(value)?.identity_directory ??
    value.current_endpoint?.identity_directory;
  /** @type {Map<string, number>} */
  const slots = new Map();
  for (const row of isRecord(directory) && Array.isArray(directory.identities)
    ? directory.identities
    : []) {
    if (
      isRecord(row) &&
      row.configured_active === true &&
      typeof row.public_agent_id === "string" &&
      (row.team_id === 1 || row.team_id === 2) &&
      Number.isInteger(row.team_local_slot) &&
      row.team_local_slot >= 0 &&
      row.team_local_slot < 5
    )
      slots.set(row.public_agent_id, (row.team_id - 1) * 5 + row.team_local_slot);
  }
  identitySlotCache.set(value, slots);
  return slots;
}

/**
 * Return the numeric researcher display label for publicAgentId in approved value.
 * The label is the configured-active global slot written as a string, from "0" through
 * "9". Return null for an unapproved root, non-string ID, or missing active identity.
 * The saved public ID is never rewritten, and a display label alone grants no command.
 *
 * @param {unknown} value @param {unknown} publicAgentId
 * @returns {string | null}
 */
export function authorizedPresentationAgentDisplayId(value, publicAgentId) {
  if (!isAuthorizedPresentationFrame(value) || typeof publicAgentId !== "string")
    return null;
  const slot = identitySlots(value).get(publicAgentId);
  return slot === undefined ? null : String(slot);
}

/**
 * Return frozen display-identity rows for the approved presentation value.
 * Use the attached researcher roster when present, otherwise the visible scene roster.
 * Each row includes a session-scoped display key, original presentation/public IDs,
 * optional command slot, activation kind, visibility in the actual scene, and a frozen
 * agent copy with its numeric display label. Invalid rows are skipped; unmarked input
 * returns an empty frozen array. Global slots stay in adapter rows, not body records.
 * A row from an Agent researcher panel is not permission to draw hidden geometry or
 * to send a command; the command-resolution helpers enforce their own audience rules.
 *
 * @param {unknown} value
 * @returns {ReadonlyArray<Readonly<{
 *   display_key: string,
 *   presentation_key: string,
 *   public_agent_id: string,
 *   command_global_slot: number | null,
 *   activation_kind: "scene_agent" | "live_pov_global" | "replay_pov_global",
 *   visible_in_snapshot: boolean,
 *   agent: Readonly<Record<string, any>>,
 * }>>}
 */
export function authorizedPresentationIdentityRows(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return Object.freeze([]);
  }
  const audience = authorizedPresentationAudience(value);
  const globalResearcherSpace = researcherSpace(value);
  const commandSlotByPublicId = identitySlots(value);

  const rows = [];
  const rosterAgents =
    globalResearcherSpace !== null && Array.isArray(globalResearcherSpace.roster_agents)
      ? globalResearcherSpace.roster_agents
      : Array.isArray(value.scene.agents)
        ? value.scene.agents
        : [];
  const visiblePublicIds = new Set(
    (Array.isArray(value.scene.agents) ? value.scene.agents : []).flatMap((agent) =>
      isRecord(agent) && typeof agent.public_agent_id === "string"
        ? [agent.public_agent_id]
        : [],
    ),
  );
  for (const agent of rosterAgents) {
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
          audience === "researcher" || globalResearcherSpace !== null
            ? (commandSlotByPublicId.get(agent.public_agent_id) ?? null)
            : null,
        activation_kind:
          globalResearcherSpace === null
            ? "scene_agent"
            : value.viewer_mode === "live"
              ? "live_pov_global"
              : "replay_pov_global",
        visible_in_snapshot: visiblePublicIds.has(agent.public_agent_id),
        agent: Object.freeze({
          ...agent,
          display_agent_id: authorizedPresentationAgentDisplayId(
            value,
            agent.public_agent_id,
          ),
        }),
      }),
    );
  }
  return Object.freeze(rows);
}

/**
 * Resolve presentationKey to a command transport slot only for an approved Oracle
 * value. Return a global slot from 0 through 9, or null for a non-string key, absent
 * identity, or Agent POV audience. The lookup uses the approved identity rows and does
 * not infer a slot from the opaque key's text. This helper does not send the command.
 *
 * @param {unknown} value
 * @param {unknown} presentationKey
 * @returns {number | null}
 */
export function authorizedOracleCommandSlotForPresentationKey(value, presentationKey) {
  if (
    typeof presentationKey !== "string" ||
    authorizedPresentationAudience(value) !== "researcher"
  ) {
    return null;
  }
  const row = authorizedPresentationIdentityRows(value).find(
    (identity) => identity.presentation_key === presentationKey,
  );
  return row?.command_global_slot ?? null;
}

/**
 * Resolve publicAgentId only when it occurs on the approved target-action axis.
 * Use Oracle's current directory and action axis, or a live Agent POV's separately
 * approved global researcher directory and pending decision mask. Return a global slot
 * from 0 through 9, including an inactive axis-only identity, or null when the ID or
 * branch is unavailable. Replay Agent POV has no such command route. This lookup does
 * not authorize a hidden battlefield body or send a command.
 *
 * @param {unknown} value
 * @param {unknown} publicAgentId
 * @returns {number | null}
 */
export function authorizedOracleCommandSlotForPublicAgentId(value, publicAgentId) {
  const globalResearcherSpace = isAuthorizedPresentationFrame(value)
    ? researcherSpace(value)
    : null;
  const liveResearcherSpace =
    isAuthorizedPresentationFrame(value) &&
    value.viewer_mode === "live" &&
    globalResearcherSpace !== null
      ? globalResearcherSpace
      : null;
  const directory =
    liveResearcherSpace !== null
      ? liveResearcherSpace.identity_directory
      : isAuthorizedPresentationFrame(value) &&
          authorizedPresentationAudience(value) === "researcher"
        ? value.current_endpoint?.identity_directory
        : null;
  const targetActions =
    liveResearcherSpace !== null &&
    isRecord(liveResearcherSpace.pending_inspection?.draft?.decision_mask)
      ? liveResearcherSpace.pending_inspection.draft.decision_mask.target_actions
      : isAuthorizedPresentationFrame(value) &&
          authorizedPresentationAudience(value) === "researcher" &&
          isRecord(value.action_axis)
        ? value.action_axis.target_actions
        : null;
  if (
    !isAuthorizedPresentationFrame(value) ||
    typeof publicAgentId !== "string" ||
    !Array.isArray(targetActions) ||
    !targetActions.some(
      (target) => isRecord(target) && target.target_public_agent_id === publicAgentId,
    ) ||
    !isRecord(directory) ||
    !Array.isArray(directory.identities)
  ) {
    return null;
  }
  const identity = directory.identities.find(
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

/**
 * Return the inspection field from authorizedPresentationInspectionState(value).
 * The result is the approved outgoing inspection or null; unavailable and scripted
 * states remain distinguishable only through the full state helper.
 *
 * @param {unknown} value
 */
function presentationInspection(value) {
  return authorizedPresentationInspectionState(value).inspection;
}

/**
 * Return a frozen inspection state for value without guessing from missing data.
 * Unmarked input returns unavailable. A live editable draft returns live_editable,
 * its declared submission_scope, and draft inspection; scripted live playback returns
 * live_scripted with no inspection. Replay returns replay_outgoing when an outgoing
 * inspection exists, otherwise replay_none. Replay submission_scope is null.
 * The joint_turn and scripted_playback scopes retain their exact server meaning.
 *
 * @param {unknown} value
 * @returns {Readonly<{
 *   state_kind: "live_editable" | "live_scripted" | "replay_outgoing" | "replay_none" | "unavailable",
 *   submission_scope: "joint_turn" | "scripted_playback" | null,
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
 * Return the inspection state used by researcher panels and live controls.
 * For an approved live Agent POV with a researcher branch, read its global pending
 * inspection. Otherwise delegate to the ordinary inspection-state helper. The frozen
 * state distinguishes editable, scripted, replay-outgoing, replay-none, and unavailable.
 * The global live draft has no battlefield geometry; SVG consumers must use
 *  authorizedPresentationInspectionState instead.
 *
 * @param {unknown} value
 * @returns {ReturnType<typeof authorizedPresentationInspectionState>}
 */
export function authorizedPresentationResearcherInspectionState(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return authorizedPresentationInspectionState(value);
  }
  const globalResearcherSpace = researcherSpace(value);
  if (value.viewer_mode !== "live" || globalResearcherSpace === null) {
    return authorizedPresentationInspectionState(value);
  }
  const pending = globalResearcherSpace.pending_inspection;
  if (isRecord(pending) && pending.inspection_kind === "editable_live_draft") {
    return Object.freeze({
      state_kind: "live_editable",
      submission_scope: pending.submission_scope,
      inspection: isRecord(pending.draft) ? pending.draft : null,
    });
  }
  return Object.freeze({
    state_kind: "live_scripted",
    submission_scope: isRecord(pending) ? pending.submission_scope : null,
    inspection: null,
  });
}

/**
 * Copy a record rawStatus into a frozen display object, adding token_id from
 * status_id and duration from remaining_duration. Return null for non-record input.
 * The caller supplies approved status facts; this rename does not validate semantics.
 *
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
 * Build a display-only In Combat status from already approved rawAgent facts.
 * Return null unless steps_until_out_of_combat is a positive integer and
 * out_of_combat_delay_steps is an integer at least as large. Otherwise return a frozen
 * status with the configured and remaining durations in ticks. This does not alter the
 * scientific status schema or approve an untrusted agent; the caller owns that check.
 *
 * @param {unknown} rawAgent
 * @returns {Readonly<Record<string, any>> | null}
 */
export function projectCertifiedInCombatDurationStatus(rawAgent) {
  if (!isRecord(rawAgent)) {
    return null;
  }
  const remainingDuration = rawAgent.steps_until_out_of_combat;
  const configuredDuration = rawAgent.out_of_combat_delay_steps;
  if (
    !Number.isInteger(remainingDuration) ||
    remainingDuration <= 0 ||
    !Number.isInteger(configuredDuration) ||
    configuredDuration < remainingDuration
  ) {
    return null;
  }
  return Object.freeze({
    status_id: "in_combat",
    token_id: "in_combat",
    configured_duration_steps: configuredDuration,
    remaining_duration: remainingDuration,
    duration: remainingDuration,
  });
}

/**
 * Return a frozen copy of approved rawModifier with token_id copied from aura_id,
 * or null for non-record input. No effect value is recomputed.
 *
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
 * Adapt approved rawAgent facts for components using the owning presentation.
 * Return null for a non-record agent or unusable scoped presentation key. Otherwise
 * return a frozen copy with display identity, alive/max-health/cooldown aliases,
 * normalized statuses plus an active combat countdown, and aura modifiers. Nested
 * source facts remain shared from the immutable approved root; no simulator slot is
 * added to this body record.
 *
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
  const durableStatuses = (Array.isArray(rawAgent.statuses) ? rawAgent.statuses : [])
    .map(statusView)
    .filter((status) => status !== null);
  const inCombatStatus = projectCertifiedInCombatDurationStatus(rawAgent);
  if (inCombatStatus !== null) {
    durableStatuses.push(inCombatStatus);
  }
  return Object.freeze({
    ...rawAgent,
    display_agent_id: authorizedPresentationAgentDisplayId(
      presentation,
      rawAgent.public_agent_id,
    ),
    display_key: displayKey,
    alive: rawAgent.life_state === "alive",
    max_health: rawAgent.maximum_health,
    ultimate_cooldown: rawAgent.ultimate_cooldown_remaining,
    statuses: Object.freeze(durableStatuses),
    modifiers: Object.freeze(
      (Array.isArray(rawAgent.aura_modifiers) ? rawAgent.aura_modifiers : [])
        .map(modifierView)
        .filter((modifier) => modifier !== null),
    ),
  });
}

/**
 * Return a frozen copy of approved rawField with token_id copied from aura_id,
 * or null for non-record input. Geometry and effect facts stay unchanged.
 *
 * @param {unknown} rawField
 * @returns {Readonly<Record<string, any>> | null}
 */
function auraFieldView(rawField) {
  return isRecord(rawField)
    ? Object.freeze({ ...rawField, token_id: rawField.aura_id })
    : null;
}

/**
 * Return the exact outgoing inspection for approved value, unwrapping a live
 * editable draft. Return null for unmarked input, scripted playback, or a replay frame
 * without an outgoing action. This view does not substitute the global researcher branch.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>> | null}
 */
export function authorizedPresentationInspection(value) {
  return presentationInspection(value);
}

/**
 * Project a target/Ultimate mask row from already approved inspection facts.
 * rawDecisionMask must belong to rawOwner. rawTarget must match the same certified
 * category, disclosure kind, label, identity, and any visible anchor. rawLaneName is
 * none, basic, or ultimate. Return null when these joins or the two-boolean joint row
 * are invalid. Otherwise return a frozen row with raw lane availability, Basic
 * availability excluding target-none, Ultimate availability, target identity/disclosure,
 * and armed lane 0/1 or null. armed_pair_legal is null when no lane is armed.
 * The caller must already have approved the containing presentation; this pure helper
 * does not itself grant information or action authority.
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
    basic_available: targetAction > 0 && pairRow[0],
    ultimate_available: pairRow[1],
    armed_lane: lane,
    armed_pair_legal: lane === null ? null : pairRow[lane],
  });
}

/**
 * Build a drawable route from already approved inspection, owner, target, and legality.
 * rawInspection supplies the actor anchor; rawOwner and rawTargetAgent supply matching
 * public/presentation identities and radii. rawTarget must disclose a visible target
 * anchor, and rawLegality must join both identities, category, and an armed lane 0/1.
 * Return a frozen source/target anchor-and-radius route with lane and legal flag, or
 * null for a failed join, target-none, or axis-only target. Coordinates are map units.
 * The caller owns approval and finite geometry checks; no hidden anchor is inferred.
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
 * Adapt the approved battlefield scene and its outgoing inspection for rendering.
 * value must carry the normalizer's approval marker; otherwise return null.
 * localInspectedPresentationKey defaults to undefined, which keeps the server-selected
 * owner. In replay Agent POV or scripted live playback, a supplied string selects a
 * visible scene body locally and null clears it. This local selection does not take
 * ownership of another actor's outgoing action or reveal a hidden target.
 * Return a frozen view with audience, map, adapted agents/auras, class/spawn mechanics,
 * spawn pads/waves, selection keys, ranges, pending route, and selected legality.
 * Unavailable routes/legality are null; no selected body gives empty ranges. Nested
 * approved source values are shared. No global slot or command field is added.
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
    ((audience === "agent_pov" && value.viewer_mode === "replay") ||
      inspectionState.state_kind === "live_scripted");
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
 * Return the non-spatial view used by researcher panels for approved value.
 * When an Agent POV has a global researcher branch, adapt its roster and class facts,
 * selected identity, and any selected legality. Ranges are empty and pending_route is
 * null. Otherwise return the ordinary scene view; unmarked input returns null.
 * Battlefield renderers must use authorizedPresentationSceneView so hidden roster
 * facts cannot become hidden-body geometry.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>> | null}
 */
export function authorizedPresentationResearcherSceneView(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return null;
  }
  const researcher = researcherSpace(value);
  if (researcher === null) {
    return authorizedPresentationSceneView(value);
  }
  const agents = Object.freeze(
    (Array.isArray(researcher.roster_agents) ? researcher.roster_agents : [])
      .map((agent) => agentView(agent, value))
      .filter((agent) => agent !== null),
  );
  const selected = agents.find(
    (agent) => agent.public_agent_id === researcher.selected_public_agent_id,
  );
  const selectedKey = selected?.presentation_key ?? null;
  const inspectionState = authorizedPresentationResearcherInspectionState(value);
  const inspection = inspectionState.inspection;
  const target = isRecord(inspection?.draft_target) ? inspection.draft_target : null;
  const selectedLegality =
    selected !== undefined &&
    isRecord(inspection) &&
    isRecord(inspection.decision_mask) &&
    target !== null
      ? projectCertifiedInspectionLegality(
          inspection.decision_mask,
          selected,
          target,
          inspection.draft_action?.armed_lane,
        )
      : null;
  return Object.freeze({
    audience: "researcher",
    agents,
    class_mechanics: researcher.class_mechanics,
    selection: Object.freeze({
      controlled_presentation_key: null,
      inspection_owner_presentation_key: selectedKey,
      selected_presentation_key: selectedKey,
    }),
    ranges: Object.freeze([]),
    pending_route: null,
    selected_legality: selectedLegality,
  });
}

/**
 * Return frozen incoming visual rows in their declared source order.
 * For approved Oracle value, read latest_events; for Agent POV, read its separately
 * approved visual_events. Each row has id, kind, vocabulary, and the original immutable
 * payload. Prefer event rows, then recipient cues, then observation deltas according
 * to the available branch. Invalid entries are skipped; absent or unmarked input
 * returns an empty array. This does not substitute global researcher events into fog.
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
  if (!isAuthorizedPresentationFrame(value)) {
    return Object.freeze([]);
  }
  const latest =
    authorizedPresentationAudience(value) === "agent_pov"
      ? value.visual_events
      : value.latest_events;
  if (!isRecord(latest)) {
    return Object.freeze([]);
  }
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

/**
 * Build frozen submitted/accepted action table rows from approved value and transition.
 * Resolve every actor through the exact approved identity directory. Return an empty
 * array if transition has no action_rows, or if any row has an identity/action mismatch.
 * Successful rows contain actor title/accent/team plus copied integer action triples;
 * row order is preserved. The caller supplies the approved transition branch.
 *
 * @param {Record<string, any>} value
 * @param {unknown} transition
 */
function authorizedPresentationActionRows(value, transition) {
  if (!isRecord(transition) || !Array.isArray(transition.action_rows)) {
    return Object.freeze([]);
  }
  const identityByPresentationKey = new Map(
    authorizedPresentationIdentityRows(value).map((identity) => [
      identity.presentation_key,
      identity,
    ]),
  );
  const rows = [];
  for (const rawRow of transition.action_rows) {
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
        actor_team: identity.agent.team_id === 1 ? "team-a" : "team-b",
        submitted_action: submittedAction,
        accepted_action: acceptedAction,
      }),
    );
  }
  return Object.freeze(rows);
}

/**
 * Return the latest completed transition's action table for approved value.
 * Prefer its attached global researcher latest_transition, otherwise the top-level
 * transition. The result is frozen submitted/accepted action rows; absent, unmarked,
 * or inconsistent input gives an empty array. This panel view does not feed animation.
 *
 * @param {unknown} value
 */
export function authorizedPresentationTransitionRows(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return Object.freeze([]);
  }
  const researcher = researcherSpace(value);
  return authorizedPresentationActionRows(
    value,
    researcher?.latest_transition ?? value.latest_transition,
  );
}

/**
 * Return the incoming transition ID used by the Latest Transition panel.
 * For approved value, prefer the attached researcher branch, then the top-level latest
 * transition. Return null when unavailable. Battlefield animation remains tied to the
 * separate fog-authorized transition, even when the researcher panel names a global one.
 *
 * @param {unknown} value
 * @returns {string | null}
 */
export function authorizedPresentationLatestTransitionId(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return null;
  }
  const researcher = researcherSpace(value);
  const transition = researcher?.latest_transition ?? value.latest_transition;
  return isRecord(transition) && typeof transition.incoming_transition_id === "string"
    ? transition.incoming_transition_id
    : null;
}

/**
 * Return frozen pending-action table rows for an approved live value.
 * Prefer the attached researcher pending_joint_action, then the top-level branch.
 * Rows keep actor title/accent/team and copied integer action triples in source order.
 * Return an empty array for replay, unmarked input, absence, or any identity/action
 * mismatch. This displays the pending joint turn; it neither accepts nor sends actions.
 *
 * @param {unknown} value
 */
export function authorizedPresentationPendingJointActionRows(value) {
  if (!isAuthorizedPresentationFrame(value) || value.viewer_mode !== "live") {
    return Object.freeze([]);
  }
  const researcher = researcherSpace(value);
  const pending = researcher?.pending_joint_action ?? value.pending_joint_action;
  if (!isRecord(pending) || !Array.isArray(pending.action_rows)) {
    return Object.freeze([]);
  }
  const identityByPresentationKey = new Map(
    authorizedPresentationIdentityRows(value).map((identity) => [
      identity.presentation_key,
      identity,
    ]),
  );
  const rows = [];
  for (const rawRow of pending.action_rows) {
    const identity = isRecord(rawRow)
      ? identityByPresentationKey.get(rawRow.actor_presentation_key)
      : null;
    const actorIdentity = exactAuthorizedAgentIdentityV1(identity?.agent);
    const rawAction = isRecord(rawRow) ? rawRow.pending_action : null;
    if (
      !isRecord(rawRow) ||
      identity === null ||
      identity === undefined ||
      actorIdentity === null ||
      actorIdentity.presentationKey !== rawRow.actor_presentation_key ||
      actorIdentity.publicAgentId !== rawRow.actor_public_agent_id ||
      identity.public_agent_id !== rawRow.actor_public_agent_id ||
      !isRecord(rawAction)
    ) {
      return Object.freeze([]);
    }
    const pendingAction = Object.freeze({
      move_action: rawAction.move_action,
      target_action: rawAction.target_action,
      use_ultimate_action: rawAction.use_ultimate_action,
    });
    if (!Object.values(pendingAction).every(Number.isInteger)) {
      return Object.freeze([]);
    }
    rows.push(
      Object.freeze({
        actor_title: actorIdentity.title,
        actor_accent: actorIdentity.accent,
        actor_team: identity.agent.team_id === 1 ? "team-a" : "team-b",
        pending_action: pendingAction,
      }),
    );
  }
  return Object.freeze(rows);
}

/**
 * Return frozen submitted/accepted rows for the recorded outgoing replay transition.
 * value must be an approved replay presentation. Prefer its global researcher branch
 * when attached, otherwise the top-level upcoming_transition. Live, absent, unmarked,
 * or inconsistent input returns an empty array; no future action is predicted.
 *
 * @param {unknown} value
 */
export function authorizedPresentationUpcomingTransitionRows(value) {
  if (!isAuthorizedPresentationFrame(value) || value.viewer_mode !== "replay") {
    return Object.freeze([]);
  }
  const researcher = researcherSpace(value);
  return authorizedPresentationActionRows(
    value,
    researcher?.upcoming_transition ?? value.upcoming_transition,
  );
}

/** @typedef {"nonnegative_integer" | "positive_finite_number" | "scientific_id" | "optional_scientific_id" | "sha256_prefix"} TechnicalFactValueKind */
/** @typedef {readonly [string, string, string, TechnicalFactValueKind]} TechnicalFactSpecification */

const TECHNICAL_FIELD_UNAVAILABLE = Symbol("technical-field-unavailable");
const SCIENTIFIC_ID_PATTERN = /^[-A-Za-z0-9_.:/+]+$/u;
const SHA256_PREFIX_PATTERN = /^[0-9a-f]{12}$/u;

/**
 * Return a frozen four-entry specification: stable id, display label, source field,
 * and valueKind validation rule. All arguments are required and come from this module's
 * fixed technical-field table. This helper does not inspect a presentation.
 *
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
      "Simulator Step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming Transition",
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
      "Simulator Step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming Transition",
      "incoming_recipient_transition_id",
      "optional_scientific_id",
    ),
  ]),
  live_shared_obs_technical_frame: Object.freeze([
    technicalFactSpecification("episode", "Episode", "episode_id", "scientific_id"),
    technicalFactSpecification(
      "frame",
      "Frame",
      "recipient_frame_index",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "simulator_step",
      "Simulator Step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming Transition",
      "incoming_recipient_transition_id",
      "optional_scientific_id",
    ),
  ]),
  replay_oracle_technical_frame: Object.freeze([
    technicalFactSpecification(
      "artifact_digest_prefix",
      "Artifact Digest Prefix",
      "artifact_digest_prefix",
      "sha256_prefix",
    ),
    technicalFactSpecification("frame", "Frame", "frame_index", "nonnegative_integer"),
    technicalFactSpecification(
      "simulator_step",
      "Simulator Step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming Transition",
      "incoming_transition_id",
      "optional_scientific_id",
    ),
    technicalFactSpecification(
      "ordinary_movement_distance_scale",
      "Ordinary Movement Distance Scale",
      "recorded_ordinary_movement_distance_scale",
      "positive_finite_number",
    ),
  ]),
  replay_no_shared_obs_technical_frame: Object.freeze([
    technicalFactSpecification("frame", "Frame", "frame_index", "nonnegative_integer"),
    technicalFactSpecification(
      "simulator_step",
      "Simulator Step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming Transition",
      "incoming_recipient_transition_id",
      "optional_scientific_id",
    ),
  ]),
  replay_shared_obs_technical_frame: Object.freeze([
    technicalFactSpecification("frame", "Frame", "frame_index", "nonnegative_integer"),
    technicalFactSpecification(
      "simulator_step",
      "Simulator Step",
      "simulator_step_count",
      "nonnegative_integer",
    ),
    technicalFactSpecification(
      "incoming_transition",
      "Incoming Transition",
      "incoming_recipient_transition_id",
      "optional_scientific_id",
    ),
  ]),
});

/**
 * Read value's own enumerable data property named field without calling a getter.
 * Return the field value, or the private unavailable sentinel when the property is
 * missing, non-enumerable, an accessor, or descriptor lookup throws. The caller owns
 * value validation. No other field is enumerated and no broader source is consulted.
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

/**
 * Return whether value is a 1-to-512-character string containing only the admitted
 * scientific-ID characters. This checks syntax, not whether the identity exists.
 *
 * @param {unknown} value
 */
function isScientificId(value) {
  return (
    typeof value === "string" &&
    value.length >= 1 &&
    value.length <= 512 &&
    SCIENTIFIC_ID_PATTERN.test(value)
  );
}

/**
 * Return whether value fits valueKind from the fixed technical-field table.
 * Rules admit nonnegative integers, positive finite numbers, scientific IDs, nullable
 * scientific IDs, or a 12-character lowercase SHA-256 prefix. A null value is accepted
 * only by optional_scientific_id; it means an absent optional fact.
 *
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

/**
 * Return frozen technical fact rows from the approved, explicitly selected source.
 * For live Agent POV with researcher metadata, use that branch's technical_frame;
 * otherwise use value's top-level technical_frame. Read only the fixed field list for
 * its technical_kind, without invoking getters. Any missing/invalid required fact
 * returns an empty array. Optional null facts are omitted. Rows contain id, Title Case
 * label, and value in declared order, plus permitted match-summary facts such as map,
 * mode, horizon, and recorded seeds when available. No spatial or actor-input fallback
 * is used; unmarked input returns an empty array.
 *
 * @param {unknown} value
 */
export function authorizedPresentationTechnicalFacts(value) {
  if (!isAuthorizedPresentationFrame(value)) {
    return Object.freeze([]);
  }
  const researcher = researcherSpace(value);
  const technicalOwner =
    value.viewer_mode === "live" && researcher !== null ? researcher : value;
  const technicalFrame = technicalFrameDataValue(technicalOwner, "technical_frame");
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
  const facts = snapshot
    .filter(([, , factValue]) => factValue !== null)
    .map(([id, label, factValue]) => Object.freeze({ id, label, value: factValue }));
  // Common episode facts are explicitly admitted researcher HUD metadata. They
  // are shared across POVs and never taken from a spatial or actor-input fallback.
  const match = value.match_summary;
  if (match) {
    if (!facts.some((fact) => fact.id === "episode")) {
      facts.unshift(
        Object.freeze({ id: "episode", label: "Episode", value: match.episode_id }),
      );
    }
    facts.push(
      Object.freeze({
        id: "task_mode",
        label: "Task Mode",
        value: match.task_mode === 1 ? "TDM" : "Combat diagnostic",
      }),
    );
    if (match.map) {
      facts.push(
        Object.freeze({ id: "map", label: "Map", value: match.map.technical_name }),
      );
    }
    if (match.observation_mode) {
      facts.push(
        Object.freeze({
          id: "observation_mode",
          label: "Observation Mode",
          value: match.observation_mode === "shared_obs" ? "SharedObs" : "NoSharedObs",
        }),
      );
    }
    if (match.episode_limit !== null && match.episode_limit !== undefined) {
      facts.push(
        Object.freeze({
          id: "episode_limit",
          label: "Episode Limit",
          value: `${match.episode_limit} ticks`,
        }),
      );
    }
    facts.push(
      Object.freeze({
        id: "seeds",
        label: "Seeds",
        value: `Root ${match.root_seed ?? "unknown"} · Episode stream ${match.episode_seed ?? "unknown"}`,
      }),
    );
  }
  return Object.freeze(facts);
}
