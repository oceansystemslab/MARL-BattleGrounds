/**
 * @file Choose which installed facts the browser may display while requests change.
 * resolveInstalledPresentationAuthorityV1 checks exact joined object ownership.
 * pendingPresentationSurfaceView returns safe unavailable labels and a stopped
 * timeline. Neither helper fetches data, installs authority or changes the DOM.
 */
import {
  isJoinedTransportAndAuthorizedPresentationV1,
  isNormalizedAuthorizedPresentationFrameV1,
} from "./authorized-presentation-normalizer.js";
import { REPLAY_PLAYBACK_RATES } from "./replay-controls.js";

export const PENDING_PRESENTATION_COPY = "Unavailable while authority is pending";

/**
 * Return whether value is a non-null object that is not an array.
 * This shape check does not validate fields, prototypes or authority.
 *
 * @param {unknown} value @returns {value is Readonly<Record<string, any>>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return the exact installed transport/presentation pair, or null.
 *
 * authority must be a registered joined result; transport and presentation
 * must be the very objects held by that result. Equal-looking copies do not
 * qualify. presentation must also be a registered normalized presentation.
 * Any mismatch returns null. Success returns a new frozen pair referencing
 * the original objects. This does not install data or grant new information
 * rights, and a retained old transport alone is never display authority.
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
 * Build an unavailable display state while no presentation is installed.
 *
 * replaySnapshot is a playback-controller snapshot. Only its playbackRate is
 * retained, if supported; otherwise the rate becomes 1. All cursor, artifact,
 * authority, recording and continuation facts are cleared or replaced with
 * unavailable labels. The timeline is hidden, disconnected and stopped.
 * Returns frozen display records without mutating the snapshot or DOM.
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
