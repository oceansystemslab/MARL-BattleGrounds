import { isJoinedTransportAndAuthorizedPresentationV1 } from "./authorized-presentation-normalizer.js";

/**
 * @typedef {Readonly<{
 *   scope: string,
 *   recipientPublicAgentId: string,
 *   recipientPresentationKey: string,
 * }>} ReplayAgentRecipientRotationIdentity
 */

/** @param {unknown} value @returns {value is Record<string, any>} */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Extract the immutable replay identity that may survive one recipient
 * rotation. The input must already be an unforgeable transport/presentation
 * pair, so this projection never accepts a raw wire root on its own.
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
 * Compare two validated replay installation identities. Every continuity field
 * must remain byte-identical while both recipient identities rotate.
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
 * Recognize only a same-artifact, same-cursor Replay Agent recipient rotation.
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
