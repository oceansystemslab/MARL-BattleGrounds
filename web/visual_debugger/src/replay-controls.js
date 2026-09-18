/**
 * @file Validate replay commands and own browser playback/navigation state.
 * The controller sends at most one request at a time and advances playback after
 * the current presentation completes. Frame indices select captured states;
 * simulator ticks come from the joined timeline. DOM helpers bind controls and
 * render availability without changing recorded replay data.
 */
/** @type {Readonly<Record<string, string>>} */
const REPLAY_NAVIGATION_COMMAND_TYPES = Object.freeze({
  first: "first_frame",
  previous: "previous_frame",
  next: "next_frame",
  last: "last_frame",
});

/**
 * @typedef {{
 *   setTimeout: (callback: () => void | Promise<void>, delay: number) => any,
 *   clearTimeout: (id: any) => void,
 * }} ReplayClock
 */

export const REPLAY_SLIDER_DEBOUNCE_MS = 160;
export const REPLAY_AUTOPLAY_CADENCE_MS = Object.freeze({
  normal: 500,
  reduced: 750,
  off: 1_000,
});

export const REPLAY_PLAYBACK_RATES = Object.freeze([
  0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,
]);

export const REPLAY_TRANSPORT_STATES = Object.freeze({
  OFFLINE: "OFFLINE",
  SETTLED: "SETTLED",
  PLAYING: "PLAYING",
  ADVANCING: "ADVANCING",
});

/**
 * Resolve frameIndex through elements.tickForFrameIndex when provided. Return a
 * nonnegative integer simulator tick or null. The frame index is never substituted
 * for missing scientific time; callback errors propagate.
 *
 * @param {{tickForFrameIndex?: (frameIndex: number) => unknown}} elements
 * @param {number} frameIndex
 */
function simulatorTickForFrameIndex(elements, frameIndex) {
  if (typeof elements.tickForFrameIndex !== "function") {
    return null;
  }
  const tick = elements.tickForFrameIndex(frameIndex);
  return Number.isInteger(tick) && Number(tick) >= 0 ? Number(tick) : null;
}

/**
 * Build frozen accessible and visible tick labels for current and final frame indices.
 * Use the optional timeline resolver in elements; missing ticks display an em dash.
 * This formats text only and changes no DOM.
 *
 * @param {{tickForFrameIndex?: (frameIndex: number) => unknown}} elements
 * @param {number} frameIndex
 * @param {number} finalFrameIndex
 */
function replayTickText(elements, frameIndex, finalFrameIndex) {
  const currentTick = simulatorTickForFrameIndex(elements, frameIndex);
  const finalTick = simulatorTickForFrameIndex(elements, finalFrameIndex);
  return Object.freeze({
    aria: `Tick ${currentTick ?? "—"} of ${finalTick ?? "—"}`,
    visible: `Tick ${currentTick ?? "—"} / ${finalTick ?? "—"}`,
  });
}

/**
 * Return value as a number only when it is already a nonnegative integer.
 * Throw TypeError naming the field otherwise; numeric strings are not accepted.
 *
 * @param {unknown} value
 * @param {string} name
 */
function nonNegativeInteger(value, name) {
  if (!Number.isInteger(value) || Number(value) < 0) {
    throw new TypeError(`${name} must be a non-negative integer.`);
  }
  return Number(value);
}

/**
 * Require a finite numeric playback rate from the eight supported quarter-step values
 * between 0.25 and 2. Return it unchanged or throw RangeError.
 *
 * @param {unknown} value
 */
function replayPlaybackRate(value) {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    !REPLAY_PLAYBACK_RATES.includes(value)
  ) {
    throw new RangeError(
      "Replay playback rate must be one of 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, or 2.00.",
    );
  }
  return value;
}

/**
 * Return whether value is a nonnull, nonarray object. Field validation is separate.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Look up rawFrameIndex in rawTimeline.rows and return its matching simulator step.
 * The index is converted with Number, then required to be a nonnegative integer.
 * Missing rows, mismatched indices, or invalid steps return null. This does not
 * claim that a frame index and simulator tick are the same value.
 *
 * @param {unknown} rawTimeline
 * @param {unknown} rawFrameIndex
 */
export function replayTimelineSimulatorStep(rawTimeline, rawFrameIndex) {
  const timeline = isRecord(rawTimeline) ? rawTimeline : null;
  const frameIndex = Number(rawFrameIndex);
  if (
    !Number.isInteger(frameIndex) ||
    frameIndex < 0 ||
    !Array.isArray(timeline?.rows)
  ) {
    return null;
  }
  const row = timeline.rows[frameIndex];
  if (
    !isRecord(row) ||
    row.frame_index !== frameIndex ||
    !Number.isInteger(row.simulator_step_count) ||
    row.simulator_step_count < 0
  ) {
    return null;
  }
  return Number(row.simulator_step_count);
}

/**
 * Require value to have exactly the expected own enumerable field names.
 * Throw TypeError naming label on missing or extra keys; neither input is changed.
 *
 * @param {Record<string, any>} value
 * @param {readonly string[]} expected
 * @param {string} label
 */
function exactKeys(value, expected, label) {
  const keys = Object.keys(value).sort();
  const canonical = [...expected].sort();
  if (
    keys.length !== canonical.length ||
    keys.some((key, index) => key !== canonical[index])
  ) {
    throw new TypeError(`${label} must use its exact V1 fields.`);
  }
}

/**
 * Require a 1..128 character string of letters, digits, underscores, or hyphens.
 * Return the identity or throw TypeError using label.
 *
 * @param {unknown} value @param {string} label
 */
function opaqueIdentifier(value, label) {
  if (typeof value !== "string" || !/^[A-Za-z0-9_-]{1,128}$/u.test(value)) {
    throw new TypeError(`${label} must be a canonical opaque identifier.`);
  }
  return value;
}

/**
 * Require an integer simulator slot from 0 through 9. Return it, throwing TypeError
 * for a noninteger or negative value and RangeError when it exceeds the slot axis.
 *
 * @param {unknown} value @param {string} label
 */
function replayGlobalSlot(value, label) {
  const slot = nonNegativeInteger(value, label);
  if (slot >= 10) {
    throw new RangeError(`${label} must be on the ten-slot V1 axis.`);
  }
  return slot;
}

/**
 * Require the opaque actor display-key form pov_ followed by 64 lowercase hex digits.
 * Return the key or throw TypeError using label; no global slot is inferred.
 *
 * @param {unknown} value @param {string} label
 */
function povPresentationKey(value, label) {
  if (typeof value !== "string" || !/^pov_[0-9a-f]{64}$/u.test(value)) {
    throw new TypeError(`${label} must be an opaque Agent POV presentation key.`);
  }
  return value;
}

/**
 * Validate value against the complete outbound replay command union and return a
 * frozen normalized command. Require exact fields and supported IDs/categories.
 * Preset aliases become analysis and verbosity becomes false; other invalid input
 * throws TypeError or RangeError. This sends no request.
 *
 * @param {unknown} value
 */
export function normalizeReplayCommand(value) {
  if (!isRecord(value) || typeof value.command_type !== "string") {
    throw new TypeError("Replay command must be an object with a discriminator.");
  }
  const type = value.command_type;
  if (
    type === "first_frame" ||
    type === "previous_frame" ||
    type === "next_frame" ||
    type === "last_frame" ||
    type === "exit"
  ) {
    exactKeys(value, ["command_type"], "Replay command");
    return Object.freeze({ command_type: type });
  }
  if (type === "absolute_seek") {
    exactKeys(value, ["command_type", "frame_index"], "Replay seek command");
    return Object.freeze({
      command_type: type,
      frame_index: nonNegativeInteger(value.frame_index, "frame_index"),
    });
  }
  if (type === "select_agent") {
    exactKeys(
      value,
      ["command_type", "selected_global_slot"],
      "Replay Reference command",
    );
    return Object.freeze({
      command_type: type,
      selected_global_slot:
        value.selected_global_slot === null
          ? null
          : replayGlobalSlot(value.selected_global_slot, "selected_global_slot"),
    });
  }
  if (type === "set_pov_actor") {
    if (Object.hasOwn(value, "presentation_key")) {
      exactKeys(
        value,
        ["command_type", "presentation_key"],
        "Replay POV actor command",
      );
      return Object.freeze({
        command_type: type,
        presentation_key: povPresentationKey(
          value.presentation_key,
          "presentation_key",
        ),
      });
    }
    exactKeys(value, ["command_type", "global_slot"], "Replay POV actor command");
    return Object.freeze({
      command_type: type,
      global_slot: replayGlobalSlot(value.global_slot, "global_slot"),
    });
  }
  if (type === "set_view") {
    exactKeys(value, ["command_type", "view_mode"], "Replay view command");
    if (value.view_mode !== "researcher" && value.view_mode !== "pov") {
      throw new TypeError("Replay view mode is invalid.");
    }
    return Object.freeze({ command_type: type, view_mode: value.view_mode });
  }
  if (type === "set_preset") {
    exactKeys(value, ["command_type", "preset"], "Replay preset command");
    if (!["presentation", "analysis", "technical", "debug"].includes(value.preset)) {
      throw new TypeError("Replay preset is invalid.");
    }
    return Object.freeze({
      command_type: type,
      preset: "analysis",
    });
  }
  if (type === "set_ranges") {
    exactKeys(value, ["command_type", "show_ranges"], "Replay boolean command");
    if (typeof value.show_ranges !== "boolean") {
      throw new TypeError("Replay show_ranges must be a boolean.");
    }
    return Object.freeze({ command_type: type, show_ranges: value.show_ranges });
  }
  if (type === "set_verbosity") {
    exactKeys(value, ["command_type", "verbose"], "Replay boolean command");
    if (typeof value.verbose !== "boolean") {
      throw new TypeError("Replay verbose must be a boolean.");
    }
    return Object.freeze({ command_type: type, verbose: false });
  }
  throw new TypeError(`Unknown replay command type ${type}.`);
}

/**
 * Validate the exact version-1 cursor fields and return a frozen copy. Indices and
 * generations must be nonnegative integers; the frame cannot exceed the final frame
 * and choreography generation cannot exceed cursor generation. Invalid fields
 * throw TypeError; out-of-range relationships throw RangeError.
 *
 * @param {unknown} value
 */
export function normalizeReplayCursor(value) {
  if (!isRecord(value)) {
    throw new TypeError("Replay cursor must be an object.");
  }
  const keys = Object.keys(value).sort();
  const expected = [
    "choreography_generation",
    "cursor_generation",
    "final_frame_index",
    "frame_index",
    "schema_version",
  ];
  if (
    keys.length !== expected.length ||
    keys.some((key, index) => key !== expected[index]) ||
    value.schema_version !== 1
  ) {
    throw new TypeError("Replay cursor must use the exact V1 contract.");
  }
  const cursor = Object.freeze({
    schema_version: 1,
    frame_index: nonNegativeInteger(value.frame_index, "cursor.frame_index"),
    final_frame_index: nonNegativeInteger(
      value.final_frame_index,
      "cursor.final_frame_index",
    ),
    cursor_generation: nonNegativeInteger(
      value.cursor_generation,
      "cursor.cursor_generation",
    ),
    choreography_generation: nonNegativeInteger(
      value.choreography_generation,
      "cursor.choreography_generation",
    ),
  });
  if (
    cursor.frame_index > cursor.final_frame_index ||
    cursor.choreography_generation > cursor.cursor_generation
  ) {
    throw new RangeError("Replay cursor exceeds its final frame index.");
  }
  return cursor;
}

/**
 * Convert first, previous, next, or last intent to a frozen replay command.
 * Unknown intents throw RangeError. No request or cursor mutation occurs.
 *
 * @param {unknown} intent
 */
export function replayNavigationCommand(intent) {
  const commandType = REPLAY_NAVIGATION_COMMAND_TYPES[String(intent)];
  if (typeof commandType !== "string") {
    throw new RangeError(`Unknown replay navigation intent ${String(intent)}.`);
  }
  return Object.freeze({
    command_type: commandType,
  });
}

/**
 * Build a frozen absolute-seek command for nonnegative integer frameIndex.
 * TypeError rejects invalid indices; the server or controller checks final bounds.
 *
 * @param {unknown} frameIndex
 */
export function replaySeekCommand(frameIndex) {
  return Object.freeze({
    command_type: "absolute_seek",
    frame_index: nonNegativeInteger(frameIndex, "frame_index"),
  });
}

/**
 * Build a frozen version-1 request from clientId, commandId, baseRevision, and command.
 * IDs must be canonical opaque strings and revision a nonnegative integer. The
 * command is normalized through the shared union; invalid input throws TypeError
 * or RangeError. The caller owns assigning a new command ID and sending the result.
 *
 * @param {{
 *   clientId: string,
 *   commandId: string,
 *   baseRevision: number,
 *   command: Readonly<Record<string, any>>,
 * }} value
 */
export function replayCommandRequest({ clientId, commandId, baseRevision, command }) {
  const normalizedClientId = opaqueIdentifier(clientId, "Replay client ID");
  const normalizedCommandId = opaqueIdentifier(commandId, "Replay command ID");
  if (!Number.isInteger(baseRevision) || baseRevision < 0) {
    throw new TypeError("Replay base revision must be a non-negative integer.");
  }
  const normalizedCommand = normalizeReplayCommand(command);
  return Object.freeze({
    schema_version: 1,
    client_id: normalizedClientId,
    command_id: normalizedCommandId,
    base_revision: baseRevision,
    command: normalizedCommand,
  });
}

/**
 * Read and normalize the cursor from result.frame or a direct result frame.
 * Throw TypeError or RangeError for an absent or invalid cursor; mutate no input.
 *
 * @param {unknown} result
 */
function cursorFromResult(result) {
  if (!isRecord(result)) {
    throw new TypeError("Replay command result must be an object.");
  }
  const frame = isRecord(result.frame) ? result.frame : result;
  return normalizeReplayCursor(frame.cursor);
}

/**
 * Require next to represent the exact navigation command applied to previous.
 * The final frame stays fixed; cursor generation rises once, and only next_frame
 * raises choreography generation. Return next or throw Error on mismatch.
 *
 * @param {Readonly<Record<string, any>>} command
 * @param {ReturnType<typeof normalizeReplayCursor>} previous
 * @param {ReturnType<typeof normalizeReplayCursor>} next
 */
export function validateReplayCursorTransition(command, previous, next) {
  const type = command.command_type;
  const expectedFrameIndex =
    type === "first_frame"
      ? 0
      : type === "previous_frame"
        ? previous.frame_index - 1
        : type === "next_frame"
          ? previous.frame_index + 1
          : type === "last_frame"
            ? previous.final_frame_index
            : type === "absolute_seek"
              ? command.frame_index
              : null;
  if (
    !Number.isInteger(expectedFrameIndex) ||
    next.final_frame_index !== previous.final_frame_index ||
    next.frame_index !== expectedFrameIndex ||
    next.cursor_generation !== previous.cursor_generation + 1 ||
    next.choreography_generation !==
      previous.choreography_generation + (type === "next_frame" ? 1 : 0)
  ) {
    throw new Error("Replay response carries an incoherent cursor generation.");
  }
  return next;
}

/**
 * Require animation only for an applied next_frame response. Return result unchanged
 * when valid. Nonobjects throw TypeError; inconsistent animation intent throws Error.
 * This checks response intent, not the animation renderer.
 *
 * @param {Readonly<Record<string, any>>} command
 * @param {unknown} result
 */
export function validateReplayAnimationIntent(command, result) {
  if (!isRecord(result)) {
    throw new TypeError("Replay response must be an object.");
  }
  if (Object.hasOwn(result, "result") || Object.hasOwn(result, "animate_incoming")) {
    const expected =
      result.result === "applied" && command.command_type === "next_frame";
    if (result.animate_incoming !== expected) {
      throw new Error(
        "Replay animation must be true exactly for an applied next-frame response.",
      );
    }
  } else if (
    result.animate_incoming === true &&
    command.command_type !== "next_frame"
  ) {
    throw new Error("Only an exact next-frame response may animate incoming replay.");
  }
  return result;
}

/**
 * Validate result before replacing a caller-owned cursor. Applied navigation must
 * move exactly as commanded; duplicate outcomes may move forward without regressing
 * generations; other outcomes keep the cursor unchanged. Return the normalized next
 * cursor. Invalid shape or intent throws TypeError, RangeError, or Error.
 *
 * @param {Readonly<Record<string, any>>} command
 * @param {unknown} result
 * @param {ReturnType<typeof normalizeReplayCursor>} previous
 */
export function validateReplayCommandOutcome(command, result, previous) {
  if (!isRecord(result) || !isRecord(result.frame)) {
    throw new TypeError("Replay command outcome must contain a frame.");
  }
  validateReplayAnimationIntent(command, result);
  const next = normalizeReplayCursor(result.frame.cursor);
  const navigation = [
    "absolute_seek",
    "first_frame",
    "previous_frame",
    "next_frame",
    "last_frame",
  ].includes(String(command.command_type));
  const exactEnvelope =
    Object.hasOwn(result, "result") && Object.hasOwn(result, "animate_incoming");
  if (!exactEnvelope && navigation) {
    return validateReplayCursorTransition(command, previous, next);
  }
  if (result.result === "duplicate") {
    const cursorAdvance = next.cursor_generation - previous.cursor_generation;
    const choreographyAdvance =
      next.choreography_generation - previous.choreography_generation;
    if (
      next.final_frame_index !== previous.final_frame_index ||
      cursorAdvance < 0 ||
      choreographyAdvance < 0 ||
      choreographyAdvance > cursorAdvance
    ) {
      throw new Error("A duplicate replay outcome regressed its authoritative cursor.");
    }
    return next;
  }
  if (result.result === "applied" && navigation) {
    return validateReplayCursorTransition(command, previous, next);
  }
  if (
    next.frame_index !== previous.frame_index ||
    next.final_frame_index !== previous.final_frame_index ||
    next.cursor_generation !== previous.cursor_generation ||
    next.choreography_generation !== previous.choreography_generation
  ) {
    throw new Error("A non-navigation replay outcome changed the durable cursor.");
  }
  return next;
}

/**
 * Serialize replay requests and track OFFLINE, SETTLED, PLAYING, or ADVANCING state.
 * A cursor belongs to one installed authority generation. Late callbacks cannot
 * replace newer authority. Snapshot booleans derive from the transport state and
 * active request, so they are not independent state-machine authorities.
 */
export class ReplayPlaybackController {
  /**
   * Create an offline controller using options.request to send replay commands.
   * waitForPresentation defaults to an already-resolved promise; state/error callbacks
   * default to no-ops; playbackRate defaults to 1. getMotionMode is a retained unused
   * compatibility option. Missing request throws TypeError and an unsupported rate
   * throws RangeError. Construction starts no request or timer.
   *
   * @param {{
   *   request: (command: Readonly<Record<string, any>>) => Promise<unknown>,
   *   waitForPresentation?: () => Promise<unknown>,
   *   getMotionMode?: () => "normal" | "reduced" | "off",
   *   playbackRate?: number,
   *   onStateChange?: (state: ReturnType<ReplayPlaybackController["snapshot"]>) => void,
   *   onError?: (error: unknown) => void,
   * }} options
   */
  constructor(options) {
    if (!options || typeof options.request !== "function") {
      throw new TypeError("ReplayPlaybackController requires a request function.");
    }
    this.request = options.request;
    this.waitForPresentation = options.waitForPresentation ?? (() => Promise.resolve());
    void options.getMotionMode;
    this.onStateChange = options.onStateChange ?? (() => {});
    this.onError = options.onError ?? (() => {});
    /** @type {ReturnType<typeof normalizeReplayCursor> | null} */
    this.cursor = null;
    /** @type {"OFFLINE" | "SETTLED" | "PLAYING" | "ADVANCING"} */
    this.transportState = REPLAY_TRANSPORT_STATES.OFFLINE;
    this.connected = false;
    this.hidden = false;
    this.playbackRate = replayPlaybackRate(options.playbackRate ?? 1);
    /** @type {string | null} */
    this.pauseReason = "offline";
    this.generation = 0;
    this.authorityGeneration = 0;
    this.presentationGeneration = 0;
    /** @type {Readonly<{generation: number, renderPolicy: "replay_static" | "replay_animated", restartAnimated: boolean}> | null} */
    this.presentationIntent = null;
    /** @type {{generation: number, authorityGeneration: number, playback: boolean} | null} */
    this.activeRequest = null;
    this.disposed = false;
  }

  /**
   * Return a frozen view of cursor, connection, playback, pause reason, and derived
   * control flags. No request or state change occurs; retained cursor and presentation
   * intent are already immutable.
   */
  snapshot() {
    const playing = this.#playbackIsActive();
    const requestPending = this.transportState === REPLAY_TRANSPORT_STATES.ADVANCING;
    return Object.freeze({
      transportState: this.transportState,
      generation: this.generation,
      presentationIntent: this.presentationIntent,
      cursor: this.cursor,
      playing,
      requestPending,
      presentationPending: false,
      connected: this.connected,
      hidden: this.hidden,
      playbackRate: this.playbackRate,
      pauseReason: this.pauseReason,
      atStart: this.cursor === null || this.cursor.frame_index === 0,
      atEnd:
        this.cursor === null ||
        this.cursor.frame_index === this.cursor.final_frame_index,
    });
  }

  /**
   * Invalidate previous request ownership and install a normalized authoritative cursor.
   * Set static presentation and settle when connected, otherwise remain offline.
   * Return the published snapshot. Invalid cursor input throws TypeError or RangeError;
   * generation invalidation occurs before cursor validation.
   *
   * @param {unknown} value
   */
  installCursor(value) {
    this.authorityGeneration += 1;
    this.generation += 1;
    this.activeRequest = null;
    this.cursor = normalizeReplayCursor(value);
    this.pauseReason = null;
    this.transportState = this.connected
      ? REPLAY_TRANSPORT_STATES.SETTLED
      : REPLAY_TRANSPORT_STATES.OFFLINE;
    this.#setPresentationIntent("replay_static", false);
    this.#publish();
    return this.snapshot();
  }

  /**
   * Set connection availability and publish a snapshot. False clears old authority
   * through setAuthorityPending; true settles only if a cursor is installed. Return
   * the current snapshot without starting playback.
   *
   * @param {boolean} connected
   */
  setConnected(connected) {
    if (!connected) {
      return this.setAuthorityPending("disconnect");
    }
    if (this.connected) {
      this.#publish();
      return this.snapshot();
    }
    this.connected = true;
    this.transportState = this.cursor
      ? REPLAY_TRANSPORT_STATES.SETTLED
      : REPLAY_TRANSPORT_STATES.OFFLINE;
    this.pauseReason = this.cursor ? null : "offline";
    this.#publish();
    return this.snapshot();
  }

  /**
   * Invalidate old callbacks, clear the cursor and request owner, and go offline.
   * reason defaults to presentation_pending. Clear presentation intent, notify the
   * state callback, and return the new snapshot. Existing network work is not aborted.
   *
   * @param {string} reason
   */
  setAuthorityPending(reason = "presentation_pending") {
    this.authorityGeneration += 1;
    this.generation += 1;
    this.connected = false;
    this.cursor = null;
    this.activeRequest = null;
    this.transportState = REPLAY_TRANSPORT_STATES.OFFLINE;
    this.pauseReason = reason;
    this.presentationIntent = null;
    this.#publish();
    return this.snapshot();
  }

  /**
   * Store the boolean hidden state. Hidden pages pause playback with reason hidden;
   * showing the page publishes state without resuming playback. Return the snapshot.
   *
   * @param {boolean} hidden
   */
  setHidden(hidden) {
    this.hidden = Boolean(hidden);
    if (this.hidden) {
      this.pause("hidden");
    } else {
      this.#publish();
    }
    return this.snapshot();
  }

  /**
   * Validate and store a supported presentation speed. Return the snapshot and publish
   * only if the rate changes. Throw RangeError for unsupported values. Cursor, request
   * ownership, and presentation intent remain unchanged.
   *
   * @param {unknown} rate
   */
  setPlaybackRate(rate) {
    const nextRate = replayPlaybackRate(rate);
    if (nextRate === this.playbackRate) {
      return this.snapshot();
    }
    this.playbackRate = nextRate;
    this.#publish();
    return this.snapshot();
  }

  /**
   * Start or continue playback if connected, visible, undisposed, and holding a cursor.
   * restartCurrent defaults to true and restarts the current incoming animation when
   * already past frame zero. Frame zero first requests next_frame. Return whether
   * playback was accepted; unavailable or empty replays return false. Later request
   * errors are handled by the controller callback, not retried automatically.
   *
   * @param {{restartCurrent?: boolean}} [options]
   */
  play({ restartCurrent = true } = {}) {
    if (
      this.disposed ||
      !this.cursor ||
      !this.connected ||
      this.hidden ||
      this.transportState === REPLAY_TRANSPORT_STATES.OFFLINE
    ) {
      this.pause("unavailable");
      return false;
    }
    if (this.#playbackIsActive()) {
      return true;
    }
    if (this.transportState === REPLAY_TRANSPORT_STATES.ADVANCING) {
      return false;
    }
    if (this.cursor.final_frame_index === 0) {
      this.pause("endpoint");
      return false;
    }
    this.generation += 1;
    this.pauseReason = null;
    const generation = this.generation;
    if (this.cursor.frame_index === 0) {
      this.transportState = REPLAY_TRANSPORT_STATES.ADVANCING;
      this.#setPresentationIntent("replay_animated", false);
      this.#publish();
      void this.#sendCommand(replayNavigationCommand("next"), generation, true);
    } else {
      this.transportState = REPLAY_TRANSPORT_STATES.PLAYING;
      this.#setPresentationIntent(
        restartCurrent ? "replay_animated" : "replay_static",
        restartCurrent,
      );
      this.#publish();
      void this.#waitForPlaybackCompletion(generation, this.cursor);
    }
    return true;
  }

  /**
   * Pause active playback and return false, otherwise call play and return whether
   * playback starts. State changes notify the configured callback.
   */
  toggle() {
    if (this.#playbackIsActive()) {
      this.pause("user_pause");
      return false;
    }
    return this.play();
  }

  /**
   * Invalidate playback continuation and request static presentation. reason defaults
   * to user_pause. An in-flight command still owns its transaction and may finish,
   * but its completion cannot restart the paused playback generation. Return snapshot.
   *
   * @param {string} reason
   */
  pause(reason = "user_pause") {
    this.generation += 1;
    this.pauseReason = reason;
    if (this.transportState === REPLAY_TRANSPORT_STATES.PLAYING) {
      this.transportState = this.connected
        ? REPLAY_TRANSPORT_STATES.SETTLED
        : REPLAY_TRANSPORT_STATES.OFFLINE;
    }
    if (this.cursor) {
      this.#setPresentationIntent("replay_static", false);
    }
    this.#publish();
    return this.snapshot();
  }

  /**
   * Request one clamped absolute seek to the first captured frame. Return a promise
   * for whether it was accepted; unavailable navigation resolves false.
   */
  first() {
    return this.#userNavigation("first");
  }

  /**
   * Request one clamped absolute seek to the preceding captured frame. Return a
   * promise for acceptance; this is a frame offset, not a recomputed simulator step.
   */
  previous() {
    return this.#userNavigation("previous");
  }

  /**
   * Request one clamped absolute seek to the next captured frame for manual navigation.
   * Return a promise for acceptance. Manual navigation remains static; autoplay uses
   * the distinct next_frame command to request incoming animation.
   */
  next() {
    return this.#userNavigation("next");
  }

  /**
   * Request one absolute seek to the final captured frame. Return a promise for
   * acceptance; unavailable navigation resolves false.
   */
  last() {
    return this.#userNavigation("last");
  }

  /**
   * Move by the nonzero integer tickDelta using one clamped absolute-seek request.
   * Despite the retained argument name, the offset counts captured frame indices.
   * Return a promise for acceptance; invalid offsets reject with TypeError. No
   * sequence of repeated next/previous requests is created.
   *
   * @param {unknown} tickDelta
   */
  jump(tickDelta) {
    if (!Number.isInteger(tickDelta) || Number(tickDelta) === 0) {
      return Promise.reject(new TypeError("Replay jump must be a non-zero integer."));
    }
    const frameIndex = this.cursor?.frame_index;
    return this.#navigateTo(
      frameIndex === undefined ? undefined : frameIndex + Number(tickDelta),
    );
  }

  /**
   * Request a clamped absolute seek to nonnegative integer frameIndex. Invalid input
   * throws TypeError before a promise is returned; unavailable navigation resolves
   * false. Navigation publishes ADVANCING and a static presentation intent.
   *
   * @param {unknown} frameIndex
   */
  seek(frameIndex) {
    const index = nonNegativeInteger(frameIndex, "frame_index");
    return this.#navigateTo(index);
  }

  /**
   * Mark the controller disposed, clear authority, and notify an offline snapshot.
   * Future playback/navigation is fenced; an already sent network request is not aborted.
   */
  dispose() {
    this.disposed = true;
    this.setAuthorityPending("disposed");
  }

  /**
   * Resolve the named first/previous/next/last destination from the current cursor
   * and delegate to the single absolute-seek route. Return its acceptance promise.
   *
   * @param {"first" | "previous" | "next" | "last"} intent
   */
  #userNavigation(intent) {
    const frameIndex = this.cursor?.frame_index;
    const destination =
      intent === "first"
        ? 0
        : intent === "previous"
          ? frameIndex === undefined
            ? undefined
            : frameIndex - 1
          : intent === "next"
            ? frameIndex === undefined
              ? undefined
              : frameIndex + 1
            : this.cursor?.final_frame_index;
    return this.#navigateTo(destination);
  }

  /**
   * Reject unavailable navigation with a resolved false promise. Otherwise convert
   * and clamp destination to a retained integer frame, pause autoplay, publish
   * ADVANCING, and send one static absolute-seek command. Nonfinite destinations
   * resolve false; input conversion errors can propagate.
   *
   * @param {unknown} destination
   */
  #navigateTo(destination) {
    if (
      this.disposed ||
      this.transportState === REPLAY_TRANSPORT_STATES.ADVANCING ||
      !this.cursor ||
      !this.connected ||
      this.hidden
    ) {
      return Promise.resolve(false);
    }
    const numericDestination = Number(destination);
    if (!Number.isFinite(numericDestination)) {
      return Promise.resolve(false);
    }
    const frameIndex = Math.max(
      0,
      Math.min(this.cursor.final_frame_index, Math.trunc(numericDestination)),
    );
    this.generation += 1;
    const generation = this.generation;
    this.pauseReason = "user_seek";
    this.transportState = REPLAY_TRANSPORT_STATES.ADVANCING;
    this.#setPresentationIntent("replay_static", false);
    this.#publish();
    return this.#sendCommand(replaySeekCommand(frameIndex), generation, false);
  }

  /**
   * Await presentation completion for the exact generation and cursor that began playing.
   * Ignore stale completion, hidden/disconnected state, or changed cursor. At the end
   * settle playback; otherwise send one next_frame request. Current-generation errors
   * pause and call onError. Return whether further navigation was accepted.
   *
   * @param {number} generation
   * @param {ReturnType<typeof normalizeReplayCursor>} cursor
   */
  async #waitForPlaybackCompletion(generation, cursor) {
    try {
      await this.waitForPresentation();
    } catch (error) {
      if (generation === this.generation) {
        this.generation += 1;
        this.transportState = this.connected
          ? REPLAY_TRANSPORT_STATES.SETTLED
          : REPLAY_TRANSPORT_STATES.OFFLINE;
        this.pauseReason = "error";
        this.#setPresentationIntent("replay_static", false);
        this.#publish();
        this.onError(error);
      }
      return false;
    }
    if (
      generation !== this.generation ||
      this.transportState !== REPLAY_TRANSPORT_STATES.PLAYING ||
      this.cursor !== cursor ||
      this.hidden ||
      !this.connected
    ) {
      return false;
    }
    if (cursor.frame_index === cursor.final_frame_index) {
      this.generation += 1;
      this.transportState = REPLAY_TRANSPORT_STATES.SETTLED;
      this.pauseReason = "endpoint";
      this.#setPresentationIntent("replay_static", false);
      this.#publish();
      return false;
    }
    this.transportState = REPLAY_TRANSPORT_STATES.ADVANCING;
    this.#setPresentationIntent("replay_animated", false);
    this.#publish();
    return this.#sendCommand(replayNavigationCommand("next"), generation, true);
  }

  /**
   * Own one command transaction and await its checked result. Ignore callbacks after
   * authority replacement. Successful current playback continues only after its
   * presentation completes; duplicates or handled resync settle without replaying.
   * Return true for an accepted ordinary result or false for stale/rejected work.
   * Current errors clear request ownership, publish a pause, and call onError.
   *
   * @param {Readonly<Record<string, any>>} command
   * @param {number} generation
   * @param {boolean} playback
   */
  async #sendCommand(command, generation, playback) {
    if (this.activeRequest !== null || !this.cursor) {
      return false;
    }
    const previous = this.cursor;
    const transaction = {
      generation,
      authorityGeneration: this.authorityGeneration,
      playback,
    };
    this.activeRequest = transaction;
    try {
      const result = await this.request(command);
      if (
        this.activeRequest !== transaction ||
        transaction.authorityGeneration !== this.authorityGeneration ||
        !this.connected ||
        this.disposed
      ) {
        return false;
      }
      if (isRecord(result) && result.handled_resync === true) {
        const resynchronized = cursorFromResult(result);
        this.activeRequest = null;
        this.cursor = resynchronized;
        this.generation += 1;
        this.transportState = REPLAY_TRANSPORT_STATES.SETTLED;
        this.pauseReason = "resync";
        this.#setPresentationIntent("replay_static", false);
        this.#publish();
        return false;
      }
      const next = cursorFromResult(result);
      validateReplayCommandOutcome(command, result, previous);
      this.activeRequest = null;
      this.cursor = next;
      if (isRecord(result) && result.result === "duplicate") {
        this.generation += 1;
        this.transportState = REPLAY_TRANSPORT_STATES.SETTLED;
        this.pauseReason = "resync";
        this.#setPresentationIntent("replay_static", false);
        this.#publish();
        return false;
      }
      if (
        playback &&
        generation === this.generation &&
        !this.hidden &&
        this.connected
      ) {
        this.transportState = REPLAY_TRANSPORT_STATES.PLAYING;
        this.pauseReason = null;
        this.#publish();
        void this.#waitForPlaybackCompletion(generation, next);
      } else {
        this.transportState = REPLAY_TRANSPORT_STATES.SETTLED;
        if (this.presentationIntent?.renderPolicy !== "replay_static") {
          this.#setPresentationIntent("replay_static", false);
        }
        this.#publish();
      }
      return true;
    } catch (error) {
      if (
        this.activeRequest === transaction &&
        transaction.authorityGeneration === this.authorityGeneration
      ) {
        this.activeRequest = null;
        this.generation += 1;
        this.transportState = this.connected
          ? REPLAY_TRANSPORT_STATES.SETTLED
          : REPLAY_TRANSPORT_STATES.OFFLINE;
        this.pauseReason = "error";
        if (this.cursor) {
          this.#setPresentationIntent("replay_static", false);
        }
        this.#publish();
        this.onError(error);
      }
      return false;
    }
  }

  /**
   * Return whether transport is playing or an animated playback request is advancing.
   */
  #playbackIsActive() {
    return (
      this.transportState === REPLAY_TRANSPORT_STATES.PLAYING ||
      (this.transportState === REPLAY_TRANSPORT_STATES.ADVANCING &&
        this.presentationIntent?.renderPolicy === "replay_animated")
    );
  }

  /**
   * Increase presentation generation and store a frozen renderPolicy/restartAnimated
   * request. This changes local render intent only and does not publish by itself.
   *
   * @param {"replay_static" | "replay_animated"} renderPolicy
   * @param {boolean} restartAnimated
   */
  #setPresentationIntent(renderPolicy, restartAnimated) {
    this.presentationGeneration += 1;
    this.presentationIntent = Object.freeze({
      generation: this.presentationGeneration,
      renderPolicy,
      restartAnimated,
    });
  }

  /**
   * Call onStateChange with the current frozen snapshot. Callback exceptions propagate.
   */
  #publish() {
    this.onStateChange(this.snapshot());
  }
}

/**
 * Map unmodified arrow keys to navigation, fresh Space to toggle, and fresh Escape
 * to clear_selection. Repeated arrows are allowed; modifiers and other keys return
 * null. This only classifies input and does not prevent browser defaults.
 *
 * @param {{key?: string, repeat?: boolean, shiftKey?: boolean, ctrlKey?: boolean, altKey?: boolean, metaKey?: boolean}} event
 */
export function replayKeyboardIntent(event) {
  if (!event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey) {
    if (event.key === "Escape" && !event.repeat) {
      return "clear_selection";
    }
    if (event.key === "ArrowLeft") {
      return "previous";
    }
    if (event.key === "ArrowRight") {
      return "next";
    }
    if (event.key === " " && !event.repeat) {
      return "toggle";
    }
  }
  return null;
}

/**
 * Attach timeline buttons, slider, rate, and keyboard handlers to controller.
 * elements supplies required DOM controls plus current authority/keyboard checks.
 * clock defaults to globalThis but is retained unused; slider input previews locally
 * and change commits one seek. Return a cleanup function removing every installed
 * listener. No simulator data or replay file is changed.
 *
 * @param {{
 *   root: HTMLElement,
 *   keyboardTarget: EventTarget,
 *   keyboardAuthorityInstalled: () => boolean,
 *   keyboardEnabled: () => boolean,
 *   clearSelection: () => void,
 *   firstButton: HTMLButtonElement,
 *   backTenButton: HTMLButtonElement,
 *   previousButton: HTMLButtonElement,
 *   playPauseButton: HTMLButtonElement,
 *   nextButton: HTMLButtonElement,
 *   forwardTenButton: HTMLButtonElement,
 *   lastButton: HTMLButtonElement,
 *   slider: HTMLInputElement,
 *   position: HTMLOutputElement,
 *   rateSelect: HTMLSelectElement,
 *   status: HTMLOutputElement,
 *   tickForFrameIndex?: (frameIndex: number) => unknown,
 *   incomingTransitionForFrameIndex?: (frameIndex: number) => unknown,
 * }} elements
 * @param {ReplayPlaybackController} controller
 * @param {ReplayClock} [clock]
 */
export function bindReplayTimelineControls(elements, controller, clock = globalThis) {
  void clock;
  /**
   * Read the slider value and send one seek only when it converts to an integer.
   */
  const seekSlider = () => {
    const intendedIndex = Number(elements.slider.value);
    if (Number.isInteger(intendedIndex)) {
      void controller.seek(intendedIndex);
    }
  };
  /**
   * Preview the slider's current/final tick labels and accessible text without a request.
   */
  const onSliderInput = () => {
    const previewIndex = Number(elements.slider.value);
    elements.slider.value = String(previewIndex);
    const tickText = replayTickText(
      elements,
      previewIndex,
      Number(elements.slider.max),
    );
    elements.slider.setAttribute("aria-valuetext", tickText.aria);
    elements.position.value = tickText.visible;
    elements.position.textContent = elements.position.value;
  };
  /**
   * Commit the slider's chosen frame through the shared seek helper.
   */
  const onSliderChange = () => seekSlider();
  /**
   * Read the rate selector and apply the controller's supported-speed validation.
   */
  const onRateChange = () => {
    controller.setPlaybackRate(Number(elements.rateSelect.value));
  };
  /**
   * Handle replay shortcuts only when this visible control set owns installed authority.
   * Respect editable controls and dialog focus. Prevent scrolling defaults for owned
   * Space/arrows even while navigation is fenced; dispatch only when keyboardEnabled.
   *
   * @param {Event} event
   */
  const onKeyDown = (event) => {
    if (elements.root.hidden) {
      return;
    }
    const keyboardEvent = /** @type {KeyboardEvent} */ (event);
    const target = event.target;
    const intent = replayKeyboardIntent(keyboardEvent);
    const clearsSelection = intent === "clear_selection";
    if (
      target instanceof Element &&
      !clearsSelection &&
      (target.closest?.(
        'button, input, select, textarea, a[href], summary, dialog, [contenteditable]:not([contenteditable="false"]), [role="button"], [role="slider"], [role="textbox"], [role="combobox"], [role="spinbutton"], [role="menuitem"]',
      ) ??
        target.matches(
          'button, input, select, textarea, a[href], summary, dialog, [contenteditable]:not([contenteditable="false"]), [role="button"], [role="slider"], [role="textbox"], [role="combobox"], [role="spinbutton"], [role="menuitem"]',
        ))
    ) {
      return;
    }
    if (
      clearsSelection &&
      target instanceof Element &&
      (target.closest?.(
        'dialog, input, select, textarea, [contenteditable]:not([contenteditable="false"])',
      ) ?? null) !== null
    ) {
      return;
    }
    const ownsSpaceDefault =
      keyboardEvent.key === " " &&
      !keyboardEvent.shiftKey &&
      !keyboardEvent.ctrlKey &&
      !keyboardEvent.altKey &&
      !keyboardEvent.metaKey;
    const ownsArrowDefault = intent === "previous" || intent === "next";
    const state = controller.snapshot();
    const ownsReplayKeyboardDefault =
      elements.keyboardAuthorityInstalled() &&
      state.connected &&
      state.transportState !== REPLAY_TRANSPORT_STATES.OFFLINE &&
      state.cursor !== null &&
      !state.hidden;
    // Space and replay arrows must never become page/disclosure scrolling,
    // including while a coherent successor is pending and navigation is
    // fenced. Escape has no scrolling default and remains inert while fenced.
    if (ownsReplayKeyboardDefault && (ownsSpaceDefault || ownsArrowDefault)) {
      event.preventDefault();
    }
    if (!ownsReplayKeyboardDefault || !elements.keyboardEnabled()) {
      return;
    }
    if (!intent) {
      return;
    }
    event.preventDefault();
    if (intent === "clear_selection") {
      elements.clearSelection();
    } else if (intent === "toggle") {
      controller.toggle();
    } else {
      void controller[intent]();
    }
  };
  /** @type {Array<[EventTarget, string, EventListener]>} */
  const handlers = [
    [elements.firstButton, "click", () => void controller.first()],
    [elements.backTenButton, "click", () => void controller.jump(-10)],
    [elements.previousButton, "click", () => void controller.previous()],
    [elements.playPauseButton, "click", () => controller.toggle()],
    [elements.nextButton, "click", () => void controller.next()],
    [elements.forwardTenButton, "click", () => void controller.jump(10)],
    [elements.lastButton, "click", () => void controller.last()],
    [elements.slider, "input", onSliderInput],
    [elements.slider, "change", onSliderChange],
    [elements.rateSelect, "change", onRateChange],
    [elements.keyboardTarget, "keydown", onKeyDown],
  ];
  for (const [target, type, handler] of handlers) {
    target.addEventListener(type, handler);
  }
  return () => {
    for (const [target, type, handler] of handlers) {
      target.removeEventListener(type, handler);
    }
  };
}

/**
 * Render cursor, tick labels, rate, transport status, and button availability from state.
 * elements supplies DOM controls and optional authoritative tick/transition resolvers.
 * Pending navigation disables seek buttons; speed can remain editable. Return
 * undefined and perform no network work. Missing scientific ticks display an em dash.
 *
 * @param {{
 *   firstButton: HTMLButtonElement,
 *   backTenButton: HTMLButtonElement,
 *   previousButton: HTMLButtonElement,
 *   playPauseButton: HTMLButtonElement,
 *   nextButton: HTMLButtonElement,
 *   forwardTenButton: HTMLButtonElement,
 *   lastButton: HTMLButtonElement,
 *   slider: HTMLInputElement,
 *   position: HTMLOutputElement,
 *   rateSelect: HTMLSelectElement,
 *   status: HTMLOutputElement,
 *   tickForFrameIndex?: (frameIndex: number) => unknown,
 *   incomingTransitionForFrameIndex?: (frameIndex: number) => unknown,
 * }} elements
 * @param {ReturnType<ReplayPlaybackController["snapshot"]>} state
 */
export function renderReplayTimelineControls(elements, state) {
  const cursor = state.cursor;
  const unavailable =
    cursor === null ||
    state.transportState === REPLAY_TRANSPORT_STATES.OFFLINE ||
    state.hidden ||
    state.transportState === REPLAY_TRANSPORT_STATES.ADVANCING;
  const frameIndex = cursor?.frame_index ?? 0;
  const finalFrameIndex = cursor?.final_frame_index ?? 0;
  elements.slider.min = "0";
  elements.slider.max = String(finalFrameIndex);
  elements.slider.step = "1";
  elements.slider.value = String(frameIndex);
  elements.slider.disabled = unavailable;
  const tickText = replayTickText(elements, frameIndex, finalFrameIndex);
  elements.slider.setAttribute("aria-valuetext", tickText.aria);
  elements.position.value = tickText.visible;
  elements.position.textContent = elements.position.value;
  elements.rateSelect.value = String(state.playbackRate);
  elements.rateSelect.disabled =
    cursor === null ||
    state.transportState === REPLAY_TRANSPORT_STATES.OFFLINE ||
    state.hidden;
  const incomingTransition =
    cursor !== null && typeof elements.incomingTransitionForFrameIndex === "function"
      ? elements.incomingTransitionForFrameIndex(frameIndex)
      : null;
  const statusParts = [
    cursor === null ? "Frame — / —" : `Frame ${frameIndex} / ${finalFrameIndex}`,
    tickText.visible,
    ...(typeof incomingTransition === "string" && incomingTransition.length > 0
      ? [`Incoming Transition ${incomingTransition}`]
      : []),
    `${state.playbackRate.toFixed(2)}×`,
    state.transportState,
  ];
  elements.status.value = statusParts.join(" · ");
  elements.status.textContent = elements.status.value;
  elements.firstButton.disabled = unavailable;
  elements.backTenButton.disabled = unavailable;
  elements.previousButton.disabled = unavailable;
  elements.nextButton.disabled = unavailable;
  elements.forwardTenButton.disabled = unavailable;
  elements.lastButton.disabled = unavailable;
  elements.playPauseButton.disabled =
    cursor === null ||
    state.transportState === REPLAY_TRANSPORT_STATES.OFFLINE ||
    state.hidden ||
    (state.transportState === REPLAY_TRANSPORT_STATES.ADVANCING && !state.playing) ||
    (cursor.final_frame_index === 0 && !state.playing);
  elements.playPauseButton.textContent = state.playing ? "Pause" : "Play";
  elements.playPauseButton.setAttribute("aria-pressed", String(state.playing));
  elements.playPauseButton.setAttribute(
    "aria-label",
    state.playing ? "Pause replay" : "Play replay",
  );
}
