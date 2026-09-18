/**
 * @file Recognize a replay view that changes only its authorized agent recipient.
 * The helpers compare already joined transport/presentation identities. Both
 * recipient identifiers must change while artifact, cursor and display
 * conditions remain equal. They do not load a replay or grant access to one.
 */
import { isJoinedTransportAndAuthorizedPresentationV1 } from "./authorized-presentation-normalizer.js";

/**
 * @typedef {Readonly<{
 *   scope: string,
 *   recipientPublicAgentId: string,
 *   recipientPresentationKey: string,
 * }>} ReplayAgentRecipientRotationIdentity
 */

/**
 * Return true for a non-null, non-array object. value is otherwise unchecked;
 * this helper alone does not establish a valid replay identity.
 *
 * @param {unknown} value @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Extract the continuity key and recipient IDs from a joined replay Agent view.
 *
 * value must be a registered transport/presentation pair for a SharedObs or
 * NoSharedObs agent-POV replay. Return null for another kind or a missing/wrongly
 * typed required field. Success returns a frozen record: scope serializes the
 * artifact, source, cursor and display conditions in a fixed order; the two
 * recipient strings are separate. Inputs are unchanged and nothing is fetched.
 * Nested scientific validation belongs to the normalizer that made the join.
 *
 * @param {unknown} value
 * @returns {ReplayAgentRecipientRotationIdentity | null}
 */
export function replayAgentRecipientRotationIdentity(value) {
  if (!isJoinedTransportAndAuthorizedPresentationV1(value)) {
    return null;
  }
  const joined = /** @type {Readonly<Record<string, any>>} */ (value);
  const transport = joined.transport;
  const presentation = joined.presentation;
  if (
    !isRecord(transport) ||
    !isRecord(presentation) ||
    transport.viewer_mode !== "replay" ||
    presentation.product_kind !== "replay_viewer" ||
    presentation.viewer_mode !== "replay" ||
    presentation.authority?.authority_kind !== "agent_pov" ||
    !["replay_no_shared_obs_agent_pov", "replay_shared_obs_agent_pov"].includes(
      presentation.presentation_kind,
    )
  ) {
    return null;
  }
  const artifactFacts = transport.artifact_facts;
  const artifactReference = artifactFacts?.artifact_summary?.replay_reference;
  const cursor = transport.cursor;
  const source = presentation.source;
  const authority = presentation.authority;
  if (
    !isRecord(artifactReference) ||
    !isRecord(cursor) ||
    !isRecord(source) ||
    !isRecord(authority) ||
    typeof transport.session_id !== "string" ||
    typeof source.source_session_id !== "string" ||
    typeof source.episode_id !== "string" ||
    typeof artifactReference.episode_id !== "string" ||
    typeof artifactReference.artifact_id !== "string" ||
    typeof artifactReference.context_digest_sha256 !== "string" ||
    typeof artifactReference.trajectory_content_digest_sha256 !== "string" ||
    typeof artifactReference.canonical_digest_sha256 !== "string" ||
    !Number.isSafeInteger(artifactReference.replay_schema_version) ||
    !Number.isSafeInteger(cursor.schema_version) ||
    !Number.isSafeInteger(cursor.frame_index) ||
    !Number.isSafeInteger(cursor.final_frame_index) ||
    !Number.isSafeInteger(cursor.cursor_generation) ||
    !Number.isSafeInteger(cursor.choreography_generation) ||
    typeof transport.preset !== "string" ||
    typeof transport.verbose !== "boolean" ||
    typeof authority.observation_mode !== "string" ||
    typeof authority.projection_basis !== "string" ||
    typeof authority.exact_actor_input_export_available !== "boolean" ||
    typeof authority.recipient_public_agent_id !== "string" ||
    typeof authority.recipient_presentation_key !== "string"
  ) {
    return null;
  }
  const scope = JSON.stringify([
    presentation.product_kind,
    presentation.viewer_mode,
    transport.frame_kind,
    transport.session_id,
    source.source_session_id,
    source.episode_id,
    artifactReference.episode_id,
    artifactReference.artifact_id,
    artifactReference.replay_schema_version,
    artifactReference.context_digest_sha256,
    artifactReference.trajectory_content_digest_sha256,
    artifactReference.canonical_digest_sha256,
    cursor.schema_version,
    cursor.frame_index,
    cursor.final_frame_index,
    cursor.cursor_generation,
    cursor.choreography_generation,
    authority.observation_mode,
    authority.projection_basis,
    authority.exact_actor_input_export_available,
    presentation.presentation_kind,
    transport.preset,
    transport.verbose,
  ]);
  return Object.freeze({
    scope,
    recipientPublicAgentId: authority.recipient_public_agent_id,
    recipientPresentationKey: authority.recipient_presentation_key,
  });
}

/**
 * Compare two extracted identities for a recipient-only change.
 *
 * previous and next may be any inputs. Return true only when their scope
 * strings match and both recipientPublicAgentId and recipientPresentationKey
 * are strings that differ. This checks the supplied identity records, not
 * their provenance; use isReplayAgentRecipientRotation for joined frames.
 * No values are changed.
 *
 * @param {unknown} previous
 * @param {unknown} next
 */
export function isReplayAgentRecipientIdentityRotation(previous, next) {
  return (
    isRecord(previous) &&
    isRecord(next) &&
    typeof previous.scope === "string" &&
    typeof next.scope === "string" &&
    previous.scope === next.scope &&
    typeof previous.recipientPublicAgentId === "string" &&
    typeof next.recipientPublicAgentId === "string" &&
    previous.recipientPublicAgentId !== next.recipientPublicAgentId &&
    typeof previous.recipientPresentationKey === "string" &&
    typeof next.recipientPresentationKey === "string" &&
    previous.recipientPresentationKey !== next.recipientPresentationKey
  );
}

/**
 * Return whether two joined replay frames differ only by agent recipient.
 *
 * previous and next are each checked by replayAgentRecipientRotationIdentity.
 * An invalid/non-replay input returns false. Both recipient IDs must change
 * while every extracted continuity field remains equal. This pure check does
 * not install the new view or change playback.
 *
 * @param {unknown} previous
 * @param {unknown} next
 */
export function isReplayAgentRecipientRotation(previous, next) {
  return isReplayAgentRecipientIdentityRotation(
    replayAgentRecipientRotationIdentity(previous),
    replayAgentRecipientRotationIdentity(next),
  );
}
