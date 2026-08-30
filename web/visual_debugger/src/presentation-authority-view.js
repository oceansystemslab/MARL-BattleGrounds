import {
  isJoinedTransportAndAuthorizedPresentationV1,
  isNormalizedAuthorizedPresentationFrameV1,
} from "./authorized-presentation-normalizer.js";
import { REPLAY_PLAYBACK_RATES } from "./replay-controls.js";

export const PENDING_PRESENTATION_COPY = "Unavailable while authority is pending";

/** @param {unknown} value @returns {value is Readonly<Record<string, any>>} */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Resolve the only transport/presentation pair eligible to populate browser
 * scientific or operational surfaces. Retaining an old transport during a
 * request is useful for protocol accounting, but never makes it display
 * authority after the joined root has been cleared.
 *
 * @param {unknown} authority
 * @param {unknown} transport
 * @param {unknown} presentation
 * @returns {Readonly<{
 *   transport: Readonly<Record<string, any>>,
 *   presentation: Readonly<Record<string, any>>,
 * }> | null}
 */
export function resolveInstalledPresentationAuthorityV1(
  authority,
  transport,
  presentation,
) {
  if (
    !isJoinedTransportAndAuthorizedPresentationV1(authority) ||
    !isRecord(transport) ||
    !isNormalizedAuthorizedPresentationFrameV1(presentation) ||
    authority.transport !== transport ||
    authority.presentation !== presentation
  ) {
    return null;
  }
  return Object.freeze({ transport, presentation });
}

/**
 * Produce the exact fail-closed chrome contract used both at synchronous clear
 * time and during every pending render. No authority, cursor, continuation, or
 * preview field from the previous transport snapshot is copied into the
 * pending timeline. The validated playback rate is a page-local presentation
 * preference and remains inert while transport is offline.
 *
 * @param {ReturnType<import("./replay-controls.js").ReplayPlaybackController["snapshot"]>} replaySnapshot
 */
export function pendingPresentationSurfaceView(replaySnapshot) {
  const playbackRate = REPLAY_PLAYBACK_RATES.includes(replaySnapshot.playbackRate)
    ? replaySnapshot.playbackRate
    : 1;
  return Object.freeze({
    presentation: null,
    transport: null,
    scenarioDescription: "Waiting for an authorized presentation.",
    viewMode: "",
    terminal: Object.freeze({ hidden: true, text: "Terminal" }),
    replay: Object.freeze({
      artifactReference: PENDING_PRESENTATION_COPY,
      completion: PENDING_PRESENTATION_COPY,
      processing: PENDING_PRESENTATION_COPY,
      endReason: PENDING_PRESENTATION_COPY,
      timeline: Object.freeze({
        transportState: "OFFLINE",
        generation: 0,
        presentationIntent: null,
        cursor: null,
        connected: false,
        hidden: true,
        playbackRate,
        requestPending: false,
        presentationPending: false,
        playing: false,
        pauseReason: "presentation_pending",
        atStart: true,
        atEnd: true,
      }),
    }),
    recording: Object.freeze({
      hidden: true,
      badgeText: PENDING_PRESENTATION_COPY,
      lifecycle: PENDING_PRESENTATION_COPY,
      progress: PENDING_PRESENTATION_COPY,
      completion: PENDING_PRESENTATION_COPY,
      persistence: PENDING_PRESENTATION_COPY,
      status: PENDING_PRESENTATION_COPY,
    }),
  });
}
