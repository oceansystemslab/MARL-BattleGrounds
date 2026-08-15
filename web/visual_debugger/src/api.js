import {
  joinTransportAndAuthorizedPresentationV1,
  normalizePresentationApiErrorV1,
  normalizeSharedObsAgentPovReplayTransportV1,
} from "./authorized-presentation-normalizer.js";
import { normalizeLiveDebuggerFrameV2 } from "./frame-normalizer.js";
import {
  isReplayViewerFrame,
  normalizeReplayApiErrorV1,
  normalizeReplayCommandResponseV1,
  normalizeReplayTimelineV1,
  normalizeReplayViewerFrameV1,
} from "./replay-frame-normalizer.js";

const TOKEN_STORAGE_KEY = "marl-battlegrounds.debugger-token";
const CLIENT_STORAGE_KEY = "marl-battlegrounds.debugger-client-id";
const TOKEN_HEADER = "X-MARL-Debugger-Token";
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;
const UNJOINED_SHARED_REPLAY_FRAME_KINDS = new Set([
  "shared_obs_source_material_replay_viewer",
  "shared_obs_agent_pov_replay_viewer",
]);
const LIVE_JOIN_FRAME_KINDS = new Set([
  "researcher_live_debugger",
  "actor_pov_live_debugger",
]);
const REPLAY_JOIN_FRAME_KINDS = new Set([
  "researcher_replay_viewer",
  "actor_pov_replay_viewer",
  "shared_obs_agent_pov_replay_viewer",
]);
const COMMAND_RESULTS = new Set([
  "applied",
  "duplicate",
  "no_op",
  "shutdown_scheduled",
]);
const INSTALLABLE_JOIN_ERROR_CODES = new Set(["stale_revision", "command_id_conflict"]);
const REPLAY_API_ERROR_CODES = new Set([
  "invalid_request",
  "invalid_cursor",
  "audience_unavailable",
  "unauthorized",
  "forbidden_origin",
  "not_found",
  "method_not_allowed",
  "payload_too_large",
  "unsupported_media_type",
  "stale_revision",
  "command_id_conflict",
  "server_shutting_down",
  "internal_error",
]);

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Shared replay roots are not live-frame candidates. The diagnostic V1 root
 * is never a product input, and the private V1 root is unusable until CP2.7
 * joins it to a separately authorized presentation.
 *
 * @param {unknown} value
 */
function rejectUnjoinedSharedReplayFrame(value) {
  if (
    isRecord(value) &&
    value.schema_version === 1 &&
    UNJOINED_SHARED_REPLAY_FRAME_KINDS.has(value.frame_kind)
  ) {
    throw new TypeError(
      "Shared replay frame is unavailable without an authorized presentation join.",
    );
  }
}

/**
 * @param {string} key
 * @returns {string | null}
 */
function readSessionValue(key) {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

/**
 * @param {string} key
 * @param {string} value
 */
function writeSessionValue(key, value) {
  try {
    window.sessionStorage.setItem(key, value);
  } catch {
    // The in-memory return value remains usable when storage is unavailable.
  }
}

/**
 * Move the loopback capability from the URL fragment into tab-local storage.
 * URL fragments are not transmitted in HTTP requests or Referer headers.
 */
export function acquireCapabilityToken() {
  const fragment = new URLSearchParams(window.location.hash.slice(1));
  const fragmentToken = fragment.get("token");
  if (fragmentToken) {
    writeSessionValue(TOKEN_STORAGE_KEY, fragmentToken);
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}`,
    );
    return fragmentToken;
  }
  return readSessionValue(TOKEN_STORAGE_KEY);
}

/**
 * Return the stable identity for this browser tab. It is not simulator state.
 */
export function acquireClientId() {
  const stored = readSessionValue(CLIENT_STORAGE_KEY);
  if (stored && /^[0-9a-f-]{36}$/i.test(stored)) {
    return stored;
  }
  const clientId = window.crypto.randomUUID();
  writeSessionValue(CLIENT_STORAGE_KEY, clientId);
  return clientId;
}

export class DebuggerApiError extends Error {
  /**
   * @param {string} message
   * @param {{status?: number, payload?: unknown}} options
   */
  constructor(message, { status = 0, payload = null } = {}) {
    super(message);
    this.name = "DebuggerApiError";
    this.status = status;
    this.payload = payload;
  }
}

/**
 * @param {Response} response
 * @returns {Promise<any>}
 */
async function decodeResponse(response) {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().includes("application/json")) {
    const text = await response.text();
    throw new DebuggerApiError(
      text.trim() || `Debugger service returned HTTP ${response.status}.`,
      { status: response.status },
    );
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new DebuggerApiError("Debugger service returned invalid JSON.", {
      status: response.status,
    });
  }
  if (!response.ok) {
    const errorRecord = isRecord(payload?.error) ? payload.error : null;
    const message =
      (typeof errorRecord?.message === "string" && errorRecord.message) ||
      (typeof payload?.message === "string" && payload.message) ||
      `Debugger service returned HTTP ${response.status}.`;
    throw new DebuggerApiError(message, {
      status: response.status,
      payload,
    });
  }
  return payload;
}

/**
 * Decode one replay HTTP response without ever exposing an unvalidated replay
 * error body to callers. Non-JSON failures remain transport/protocol errors;
 * JSON failures must match the exact ReplayApiErrorV1 root.
 *
 * @param {Response} response
 * @returns {Promise<any>}
 */
async function decodeReplayResponse(response) {
  try {
    return await decodeResponse(response);
  } catch (error) {
    if (!(error instanceof DebuggerApiError) || !isRecord(error.payload)) {
      throw error;
    }
    let normalized;
    try {
      const payload = snapshotApiRecord(error.payload, "Replay API error");
      requireExactEnvelopeKeys(
        payload,
        ["schema_version", "error_code", "message", "latest_frame"],
        "Replay API error",
      );
      if (
        payload.schema_version !== 1 ||
        !REPLAY_API_ERROR_CODES.has(payload.error_code) ||
        typeof payload.message !== "string" ||
        payload.message.length < 1 ||
        payload.message.length > 2048
      ) {
        throw new TypeError("Replay API error scalar contract is invalid.");
      }
      if (
        isRecord(payload.latest_frame) &&
        payload.latest_frame.frame_kind === "shared_obs_agent_pov_replay_viewer"
      ) {
        normalizeSharedObsAgentPovReplayTransportV1(payload.latest_frame);
        normalized = Object.freeze({ ...payload });
      } else {
        normalizeReplayApiErrorV1(payload);
        normalized = Object.freeze({ ...payload });
      }
    } catch (protocolError) {
      throw new DebuggerApiError(
        protocolError instanceof Error
          ? `Replay service returned an invalid error envelope: ${protocolError.message}`
          : "Replay service returned an invalid error envelope.",
        { status: error.status },
      );
    }
    throw new DebuggerApiError(normalized.message, {
      status: error.status,
      payload: normalized,
    });
  }
}

/**
 * Decode only the exact HTTP 422 presentation-unavailable response.
 *
 * @param {Response} response
 * @returns {Promise<any>}
 */
async function decodePresentationResponse(response) {
  try {
    return await decodeResponse(response);
  } catch (error) {
    if (
      !(error instanceof DebuggerApiError) ||
      error.status !== 422 ||
      !isRecord(error.payload)
    ) {
      throw error;
    }
    try {
      const normalized = normalizePresentationApiErrorV1(error.payload);
      throw new DebuggerApiError(normalized.message, {
        status: 422,
        payload: normalized,
      });
    } catch (protocolError) {
      if (protocolError instanceof DebuggerApiError) throw protocolError;
      throw new DebuggerApiError(
        protocolError instanceof Error
          ? `Presentation service returned an invalid error envelope: ${protocolError.message}`
          : "Presentation service returned an invalid error envelope.",
        { status: 422 },
      );
    }
  }
}

/**
 * The current-frame route is shared by live and replay launchers. Schema V1
 * failures are replay-only and must cross the exact replay error boundary;
 * live Schema V2 failures retain their existing live error handling.
 *
 * @param {Response} response
 * @returns {Promise<any>}
 */
async function decodeCurrentFrameResponse(response) {
  try {
    return await decodeResponse(response);
  } catch (error) {
    if (
      !(error instanceof DebuggerApiError) ||
      !isRecord(error.payload) ||
      error.payload.schema_version !== 1
    ) {
      throw error;
    }
    let normalized;
    try {
      normalized = normalizeReplayApiErrorV1(error.payload);
    } catch (protocolError) {
      throw new DebuggerApiError(
        protocolError instanceof Error
          ? `Replay service returned an invalid error envelope: ${protocolError.message}`
          : "Replay service returned an invalid error envelope.",
        { status: error.status },
      );
    }
    throw new DebuggerApiError(normalized.message, {
      status: error.status,
      payload: normalized,
    });
  }
}

/**
 * @param {string | null | undefined} token
 * @returns {Record<string, string>}
 */
function authorizationHeaders(token) {
  if (!token) {
    throw new DebuggerApiError(
      "No loopback capability token is available. Reopen the printed debugger URL.",
    );
  }
  return {
    [TOKEN_HEADER]: token,
  };
}

/**
 * @param {string} path
 * @param {RequestInit} options
 * @returns {Promise<Response>}
 */
async function fetchWithTimeout(path, options) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => {
    controller.abort(
      new DOMException("The local debugger request timed out.", "TimeoutError"),
    );
  }, REQUEST_TIMEOUT_MS);
  try {
    return await window.fetch(path, {
      ...options,
      signal: controller.signal,
    });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

/**
 * @param {string | null} token
 * @returns {Promise<any>}
 */
export async function getCurrentFrame(token) {
  let response;
  try {
    response = await fetchWithTimeout("/api/frame", {
      method: "GET",
      headers: authorizationHeaders(token),
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
    });
  } catch (error) {
    throw new DebuggerApiError(
      error instanceof Error
        ? `Could not reach the local debugger service: ${error.message}`
        : "Could not reach the local debugger service.",
    );
  }
  return decodeCurrentFrameResponse(response);
}

/**
 * Fetch one raw authorized-presentation candidate. It remains unrenderable
 * until `extractJoinedFrame` completes the identity-first two-root join.
 *
 * @param {string | null} token
 */
export async function getCurrentPresentation(token) {
  let response;
  try {
    response = await fetchWithTimeout("/api/presentation/frame", {
      method: "GET",
      headers: authorizationHeaders(token),
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
    });
  } catch (error) {
    throw new DebuggerApiError(
      error instanceof Error
        ? `Could not load the authorized presentation: ${error.message}`
        : "Could not load the authorized presentation.",
    );
  }
  return decodePresentationResponse(response);
}

/**
 * Start one raw GET and one presentation GET as one bounded install attempt.
 *
 * @param {string | null} token
 * @returns {Promise<Readonly<Record<string, any>> | null>}
 */
export async function getCurrentFrameAndPresentation(token) {
  const [rawPayload, presentationPayload] = await Promise.all([
    getCurrentFrame(token),
    getCurrentPresentation(token),
  ]);
  return extractJoinedFrame(rawPayload, presentationPayload);
}

/**
 * @param {string | null} token
 * @returns {Promise<any>}
 */
export async function getReplayTimeline(token) {
  let response;
  try {
    response = await fetchWithTimeout("/api/replay/timeline", {
      method: "GET",
      headers: authorizationHeaders(token),
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
    });
  } catch (error) {
    throw new DebuggerApiError(
      error instanceof Error
        ? `Could not load the replay timeline: ${error.message}`
        : "Could not load the replay timeline.",
    );
  }
  return decodeReplayResponse(response);
}

/**
 * Send one command exactly once. This function deliberately has no retry path.
 *
 * @param {string | null} token
 * @param {unknown} request
 * @returns {Promise<any>}
 */
export async function postCommand(token, request) {
  let response;
  try {
    response = await fetchWithTimeout("/api/command", {
      method: "POST",
      headers: {
        ...authorizationHeaders(token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
    });
  } catch (error) {
    throw new DebuggerApiError(
      error instanceof Error
        ? `Command outcome is unknown because the connection failed: ${error.message}`
        : "Command outcome is unknown because the connection failed.",
    );
  }
  return decodeResponse(response);
}

/**
 * Send one replay command exactly once. Replay requests have a separate route
 * and can never enter the live debugger dispatcher.
 *
 * @param {string | null} token
 * @param {unknown} request
 * @returns {Promise<any>}
 */
export async function postReplayCommand(token, request) {
  let response;
  try {
    response = await fetchWithTimeout("/api/replay/command", {
      method: "POST",
      headers: {
        ...authorizationHeaders(token),
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
    });
  } catch (error) {
    throw new DebuggerApiError(
      error instanceof Error
        ? `Replay command outcome is unknown because the connection failed: ${error.message}`
        : "Replay command outcome is unknown because the connection failed.",
    );
  }
  return decodeReplayResponse(response);
}

/** @param {unknown} value @param {string} label */
function snapshotApiRecord(value, label) {
  if (!isRecord(value)) throw new TypeError(`${label} must be an object.`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    throw new TypeError(`${label} must use a plain JSON object prototype.`);
  }
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    throw new TypeError(`${label} must not contain symbol fields.`);
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const snapshot = Object.create(null);
  for (const [key, descriptor] of Object.entries(descriptors)) {
    if (!("value" in descriptor) || !descriptor.enumerable) {
      throw new TypeError(`${label}.${key} must be an enumerable data field.`);
    }
    snapshot[key] = descriptor.value;
  }
  return snapshot;
}

/** @param {Record<string, any>} value @param {string[]} expected @param {string} label */
function requireExactEnvelopeKeys(value, expected, label) {
  const actual = Object.keys(value).sort();
  const canonical = [...expected].sort();
  if (
    actual.length !== canonical.length ||
    actual.some((key, index) => key !== canonical[index])
  ) {
    throw new TypeError(`${label} has unknown or missing fields.`);
  }
}

/**
 * Bind an exact command/error envelope schema to its raw product family before
 * the two-root join can inspect any presentation endpoint branch.
 *
 * @param {unknown} value
 * @param {1 | 2} envelopeSchema
 * @param {string} label
 */
function requireEnvelopeFrameFamily(value, envelopeSchema, label) {
  const frame = snapshotApiRecord(value, `${label} frame`);
  const expectedKinds =
    envelopeSchema === 2 ? LIVE_JOIN_FRAME_KINDS : REPLAY_JOIN_FRAME_KINDS;
  if (!expectedKinds.has(frame.frame_kind)) {
    throw new TypeError(`${label} frame belongs to the wrong product family.`);
  }
}

/**
 * Select one exact direct/success/error raw candidate and join it to one raw
 * presentation candidate. The presentation argument is intentionally not
 * normalized before the top-level identity preflight.
 *
 * @param {unknown} rawPayload
 * @param {unknown} rawPresentationPayload
 */
export async function extractJoinedFrame(rawPayload, rawPresentationPayload) {
  const envelope = snapshotApiRecord(rawPayload, "Debugger frame payload");
  let candidate = rawPayload;
  let animateIncoming = false;
  if (!Object.hasOwn(envelope, "frame_kind")) {
    if (Object.hasOwn(envelope, "frame")) {
      const replay = envelope.schema_version === 1;
      requireExactEnvelopeKeys(
        envelope,
        replay
          ? ["schema_version", "result", "frame", "notice", "animate_incoming"]
          : ["schema_version", "result", "frame", "notice"],
        replay ? "Replay command response" : "Live command response",
      );
      if (
        (!replay && envelope.schema_version !== 2) ||
        !COMMAND_RESULTS.has(envelope.result) ||
        (envelope.notice !== null && typeof envelope.notice !== "string") ||
        (replay &&
          typeof envelope.notice === "string" &&
          (envelope.notice.length < 1 || envelope.notice.length > 2048)) ||
        (replay && typeof envelope.animate_incoming !== "boolean") ||
        (replay && envelope.animate_incoming && envelope.result !== "applied")
      ) {
        throw new TypeError("Command response scalar contract is invalid.");
      }
      candidate = envelope.frame;
      requireEnvelopeFrameFamily(
        candidate,
        replay ? 1 : 2,
        replay ? "Replay command response" : "Live command response",
      );
      animateIncoming = replay ? envelope.animate_incoming : false;
    } else if (Object.hasOwn(envelope, "latest_frame")) {
      requireExactEnvelopeKeys(
        envelope,
        ["schema_version", "error_code", "message", "latest_frame"],
        "Debugger error response",
      );
      if (
        ![1, 2].includes(envelope.schema_version) ||
        !INSTALLABLE_JOIN_ERROR_CODES.has(envelope.error_code) ||
        typeof envelope.message !== "string" ||
        envelope.message.trim() === ""
      ) {
        throw new TypeError("Debugger error response scalar contract is invalid.");
      }
      candidate = envelope.latest_frame;
      if (candidate === null) return null;
      requireEnvelopeFrameFamily(
        candidate,
        /** @type {1 | 2} */ (envelope.schema_version),
        envelope.schema_version === 1 ? "Replay error response" : "Live error response",
      );
    } else {
      return null;
    }
  }
  return joinTransportAndAuthorizedPresentationV1(
    candidate,
    rawPresentationPayload,
    animateIncoming,
  );
}

/**
 * Accept the direct frame contract and the response envelopes used by success
 * and stale-client responses.
 *
 * @param {unknown} payload
 * @returns {Record<string, any> | null}
 */
export function extractFrame(payload) {
  if (!isRecord(payload)) {
    return null;
  }
  rejectUnjoinedSharedReplayFrame(payload);
  const responseFrame = payload.frame;
  rejectUnjoinedSharedReplayFrame(responseFrame);
  if (isRecord(responseFrame) && isReplayViewerFrame(responseFrame)) {
    return normalizeReplayCommandResponseV1(payload).frame;
  }
  if (
    payload.schema_version === 1 &&
    Object.hasOwn(payload, "result") &&
    Object.hasOwn(payload, "frame") &&
    Object.hasOwn(payload, "animate_incoming")
  ) {
    return normalizeReplayCommandResponseV1(payload).frame;
  }
  const latestFrame = payload.latest_frame;
  rejectUnjoinedSharedReplayFrame(latestFrame);
  if (isRecord(latestFrame) && isReplayViewerFrame(latestFrame)) {
    return normalizeReplayApiErrorV1(payload).latest_frame;
  }
  if (
    payload.schema_version === 1 &&
    Object.hasOwn(payload, "error_code") &&
    Object.hasOwn(payload, "latest_frame")
  ) {
    return normalizeReplayApiErrorV1(payload).latest_frame;
  }
  if (isReplayViewerFrame(payload)) {
    return normalizeReplayViewerFrameV1(payload);
  }
  const envelopeCandidates = [responseFrame, latestFrame];
  for (const candidate of envelopeCandidates) {
    if (isRecord(candidate)) {
      if (payload.schema_version !== 2) {
        throw new TypeError("Debugger response envelope must use schema version 2.");
      }
      return normalizeLiveDebuggerFrameV2(candidate);
    }
  }
  if (
    Number.isInteger(payload.revision) &&
    (isRecord(payload.scene) || isRecord(payload.projection))
  ) {
    return normalizeLiveDebuggerFrameV2(payload);
  }
  return null;
}

/**
 * @param {unknown} payload
 * @returns {Readonly<Record<string, any>>}
 */
export function extractReplayTimeline(payload) {
  return normalizeReplayTimelineV1(payload);
}

/**
 * @param {unknown} payload
 * @returns {Readonly<Record<string, any>>}
 */
export function extractReplayError(payload) {
  return normalizeReplayApiErrorV1(payload);
}

/**
 * @param {unknown} payload
 * @returns {string | null}
 */
export function extractNotice(payload) {
  if (!isRecord(payload)) {
    return null;
  }
  if (typeof payload.notice === "string" && payload.notice.trim()) {
    return payload.notice;
  }
  return null;
}
