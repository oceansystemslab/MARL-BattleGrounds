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

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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
  if (isRecord(payload.frame) && isReplayViewerFrame(payload.frame)) {
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
  if (isRecord(payload.latest_frame) && isReplayViewerFrame(payload.latest_frame)) {
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
  const envelopeCandidates = [payload.frame, payload.latest_frame];
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
