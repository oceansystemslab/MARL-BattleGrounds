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

/**
 * Resolve one simulator tick from the joined timeline without ever treating a
 * transport frame index as scientific time.
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
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Join a transport frame index to its authoritative simulator step. Timeline
 * order remains transport authority; the returned value is display-only time.
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

/** @param {unknown} value @param {string} label */
function opaqueIdentifier(value, label) {
  if (typeof value !== "string" || !/^[A-Za-z0-9_-]{1,128}$/u.test(value)) {
    throw new TypeError(`${label} must be a canonical opaque identifier.`);
  }
  return value;
}

/** @param {unknown} value @param {string} label */
function replayGlobalSlot(value, label) {
  const slot = nonNegativeInteger(value, label);
  if (slot >= 10) {
    throw new RangeError(`${label} must be on the ten-slot V1 axis.`);
  }
  return slot;
}

/**
 * Normalize the complete outbound replay command union. The browser never
 * emits compatibility aliases or audience-dependent extra fields.
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
    if (!["presentation", "analysis", "debug"].includes(value.preset)) {
      throw new TypeError("Replay preset is invalid.");
    }
    return Object.freeze({
      command_type: type,
      preset: value.preset === "debug" ? "analysis" : value.preset,
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
 * @param {unknown} frameIndex
 */
export function replaySeekCommand(frameIndex) {
  return Object.freeze({
    command_type: "absolute_seek",
    frame_index: nonNegativeInteger(frameIndex, "frame_index"),
  });
}

/**
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
 * Verify the service-owned durable cursor and choreography epochs for one
 * accepted navigation. Only exact next-frame motion advances choreography.
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
 * Animation intent belongs only to the response to an exact next-frame
 * command. The durable frame and every other command remain settled.
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
 * Validate one complete authoritative command outcome before any caller-owned
 * frame, timeline, or cursor reference may be replaced.
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
 * @param {unknown} value
 * @returns {"normal" | "reduced" | "off"}
 */
function motionMode(value) {
  if (value === "normal" || value === "reduced" || value === "off") {
    return value;
  }
  throw new RangeError(`Unknown replay motion mode ${String(value)}.`);
}

/**
 * Serialized replay playback. It owns only cursor requests and pacing; the
 * server remains authoritative for every accepted cursor and generation.
 */
export class ReplayPlaybackController {
  /**
   * @param {{
   *   request: (command: Readonly<Record<string, any>>) => Promise<unknown>,
   *   waitForPresentation?: () => Promise<unknown>,
   *   getMotionMode?: () => "normal" | "reduced" | "off",
   *   onStateChange?: (state: ReturnType<ReplayPlaybackController["snapshot"]>) => void,
   *   onError?: (error: unknown) => void,
   *   clock?: ReplayClock,
   * }} options
   */
  constructor(options) {
    if (!options || typeof options.request !== "function") {
      throw new TypeError("ReplayPlaybackController requires a request function.");
    }
    this.request = options.request;
    this.waitForPresentation = options.waitForPresentation ?? (() => Promise.resolve());
    this.getMotionMode = options.getMotionMode ?? (() => "normal");
    this.onStateChange = options.onStateChange ?? (() => {});
    this.onError = options.onError ?? (() => {});
    this.clock = options.clock ?? globalThis;
    /** @type {ReturnType<typeof normalizeReplayCursor> | null} */
    this.cursor = null;
    this.playing = false;
    this.requestPending = false;
    this.presentationPending = false;
    this.connected = true;
    this.hidden = false;
    this.pauseReason = null;
    /** @type {any} */
    this.timer = null;
    this.disposed = false;
  }

  snapshot() {
    return Object.freeze({
      cursor: this.cursor,
      playing: this.playing,
      requestPending: this.requestPending,
      presentationPending: this.presentationPending,
      connected: this.connected,
      hidden: this.hidden,
      pauseReason: this.pauseReason,
      atStart: this.cursor === null || this.cursor.frame_index === 0,
      atEnd:
        this.cursor === null ||
        this.cursor.frame_index === this.cursor.final_frame_index,
    });
  }

  /** @param {unknown} value */
  installCursor(value) {
    this.cursor = normalizeReplayCursor(value);
    if (this.playing && this.cursor.frame_index === this.cursor.final_frame_index) {
      this.pause("endpoint");
    } else {
      this.#publish();
    }
    return this.snapshot();
  }

  /** @param {boolean} connected */
  setConnected(connected) {
    this.connected = Boolean(connected);
    if (!this.connected) {
      this.pause("disconnect");
    } else {
      this.#publish();
    }
    return this.snapshot();
  }

  /** @param {boolean} hidden */
  setHidden(hidden) {
    this.hidden = Boolean(hidden);
    if (this.hidden) {
      this.pause("hidden");
    } else {
      this.#publish();
    }
    return this.snapshot();
  }

  play() {
    if (
      this.disposed ||
      !this.cursor ||
      !this.connected ||
      this.hidden ||
      this.cursor.frame_index >= this.cursor.final_frame_index
    ) {
      this.pause(
        this.cursor?.frame_index === this.cursor?.final_frame_index
          ? "endpoint"
          : "unavailable",
      );
      return false;
    }
    if (this.playing) {
      return true;
    }
    this.playing = true;
    this.pauseReason = null;
    this.#scheduleAutoplay();
    this.#publish();
    return true;
  }

  toggle() {
    if (this.playing) {
      this.pause("user_pause");
      return false;
    }
    return this.play();
  }

  /** @param {string} reason */
  pause(reason = "user_pause") {
    this.playing = false;
    this.pauseReason = reason;
    this.#clearTimer();
    this.#publish();
    return this.snapshot();
  }

  first() {
    return this.#userNavigation("first");
  }

  previous() {
    return this.#userNavigation("previous");
  }

  next() {
    return this.#userNavigation("next");
  }

  last() {
    return this.#userNavigation("last");
  }

  /**
   * Seek by a signed number of ticks with one clamped absolute-seek request.
   * This deliberately does not expand a jump into repeated next/previous calls.
   *
   * @param {unknown} tickDelta
   */
  jump(tickDelta) {
    if (!Number.isInteger(tickDelta) || Number(tickDelta) === 0) {
      return Promise.reject(new TypeError("Replay jump must be a non-zero integer."));
    }
    this.pause("user_seek");
    if (!this.cursor) {
      return Promise.resolve(false);
    }
    const frameIndex = Math.max(
      0,
      Math.min(
        this.cursor.final_frame_index,
        this.cursor.frame_index + Number(tickDelta),
      ),
    );
    if (frameIndex === this.cursor.frame_index) {
      return Promise.resolve(false);
    }
    return this.#send(replaySeekCommand(frameIndex));
  }

  /** @param {unknown} frameIndex */
  seek(frameIndex) {
    const index = nonNegativeInteger(frameIndex, "frame_index");
    this.pause("user_seek");
    if (!this.cursor || index > this.cursor.final_frame_index) {
      return Promise.reject(new RangeError("Replay seek is outside the timeline."));
    }
    return this.#send(replaySeekCommand(index));
  }

  dispose() {
    this.disposed = true;
    this.pause("disposed");
  }

  /** @param {"first" | "previous" | "next" | "last"} intent */
  #userNavigation(intent) {
    this.pause("user_seek");
    if (!this.cursor) {
      return Promise.resolve(false);
    }
    if (
      (intent === "first" || intent === "previous") &&
      this.cursor.frame_index === 0
    ) {
      return Promise.resolve(false);
    }
    if (
      (intent === "next" || intent === "last") &&
      this.cursor.frame_index === this.cursor.final_frame_index
    ) {
      return Promise.resolve(false);
    }
    return this.#send(replayNavigationCommand(intent));
  }

  #scheduleAutoplay() {
    if (
      !this.playing ||
      this.timer !== null ||
      this.requestPending ||
      this.presentationPending ||
      !this.cursor ||
      !this.connected ||
      this.hidden ||
      this.disposed
    ) {
      return;
    }
    const delay = REPLAY_AUTOPLAY_CADENCE_MS[motionMode(this.getMotionMode())];
    this.timer = this.clock.setTimeout(() => {
      this.timer = null;
      void this.#autoplayNext();
    }, delay);
  }

  async #autoplayNext() {
    if (!this.playing || !this.cursor) {
      return false;
    }
    if (this.cursor.frame_index >= this.cursor.final_frame_index) {
      this.pause("endpoint");
      return false;
    }
    const moved = await this.#send(replayNavigationCommand("next"));
    if (moved && this.playing) {
      this.#scheduleAutoplay();
      this.#publish();
    }
    return moved;
  }

  /**
   * @param {Readonly<Record<string, any>>} command
   */
  async #send(command) {
    if (
      this.disposed ||
      this.requestPending ||
      this.presentationPending ||
      !this.connected ||
      this.hidden
    ) {
      return false;
    }
    const previous = this.cursor;
    if (!previous) {
      return false;
    }
    this.requestPending = true;
    this.#publish();
    try {
      const result = await this.request(command);
      if (isRecord(result) && result.handled_resync === true) {
        this.cursor = cursorFromResult(result);
        this.requestPending = false;
        this.presentationPending = false;
        this.pause("resync");
        return false;
      }
      const next = cursorFromResult(result);
      validateReplayCommandOutcome(command, result, previous);
      this.cursor = next;
      if (isRecord(result) && result.result === "duplicate") {
        this.requestPending = false;
        this.presentationPending = false;
        this.pause("resync");
        return false;
      }
      this.requestPending = false;
      this.presentationPending = true;
      this.#publish();
      await this.waitForPresentation();
      this.presentationPending = false;
      if (next.frame_index === next.final_frame_index) {
        this.pause("endpoint");
      } else {
        this.#publish();
      }
      return true;
    } catch (error) {
      this.requestPending = false;
      this.presentationPending = false;
      this.pause("error");
      this.onError(error);
      return false;
    }
  }

  #clearTimer() {
    if (this.timer === null) {
      return;
    }
    this.clock.clearTimeout(this.timer);
    this.timer = null;
  }

  #publish() {
    this.onStateChange(this.snapshot());
  }
}

/**
 * @param {{key?: string, repeat?: boolean, shiftKey?: boolean, ctrlKey?: boolean, altKey?: boolean, metaKey?: boolean}} event
 */
export function replayKeyboardIntent(event) {
  if (event.ctrlKey || event.altKey || event.metaKey) {
    return null;
  }
  if (event.key === "Home") {
    return "first";
  }
  if (event.key === "End") {
    return "last";
  }
  if (event.key === "ArrowLeft") {
    return event.shiftKey ? "back_ten" : "previous";
  }
  if (event.key === "ArrowRight") {
    return event.shiftKey ? "forward_ten" : "next";
  }
  if ((event.key === " " || event.key === "Spacebar") && !event.repeat) {
    return "toggle";
  }
  return null;
}

/**
 * @param {{
 *   root: HTMLElement,
 *   firstButton: HTMLButtonElement,
 *   backTenButton: HTMLButtonElement,
 *   previousButton: HTMLButtonElement,
 *   playPauseButton: HTMLButtonElement,
 *   nextButton: HTMLButtonElement,
 *   forwardTenButton: HTMLButtonElement,
 *   lastButton: HTMLButtonElement,
 *   slider: HTMLInputElement,
 *   position: HTMLOutputElement,
 *   tickForFrameIndex?: (frameIndex: number) => unknown,
 * }} elements
 * @param {ReplayPlaybackController} controller
 * @param {ReplayClock} [clock]
 */
export function bindReplayTimelineControls(elements, controller, clock = globalThis) {
  /** @type {any} */
  let sliderTimer = null;
  /** @type {number | null} */
  let pendingSliderIndex = null;
  const clearSliderTimer = () => {
    if (sliderTimer !== null) {
      clock.clearTimeout(sliderTimer);
      sliderTimer = null;
    }
  };
  const seekSlider = () => {
    const intendedIndex = pendingSliderIndex;
    pendingSliderIndex = null;
    clearSliderTimer();
    if (intendedIndex !== null) {
      void controller.seek(intendedIndex);
    }
  };
  const onSliderInput = () => {
    pendingSliderIndex = Number(elements.slider.value);
    controller.pause("user_seek");
    elements.slider.value = String(pendingSliderIndex);
    const tickText = replayTickText(
      elements,
      pendingSliderIndex,
      Number(elements.slider.max),
    );
    elements.slider.setAttribute("aria-valuetext", tickText.aria);
    elements.position.value = tickText.visible;
    elements.position.textContent = elements.position.value;
    clearSliderTimer();
    sliderTimer = clock.setTimeout(seekSlider, REPLAY_SLIDER_DEBOUNCE_MS);
  };
  const onSliderChange = () => seekSlider();
  /** @param {Event} event */
  const onKeyDown = (event) => {
    const keyboardEvent = /** @type {KeyboardEvent} */ (event);
    const target = event.target;
    if (
      target instanceof Element &&
      (target.matches("button, input, select, textarea, a[href]") ||
        target.getAttribute("contenteditable") === "true")
    ) {
      return;
    }
    const intent = replayKeyboardIntent(keyboardEvent);
    if (!intent) {
      return;
    }
    event.preventDefault();
    if (intent === "toggle") {
      controller.toggle();
    } else if (intent === "back_ten") {
      void controller.jump(-10);
    } else if (intent === "forward_ten") {
      void controller.jump(10);
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
    [elements.root, "keydown", onKeyDown],
  ];
  for (const [target, type, handler] of handlers) {
    target.addEventListener(type, handler);
  }
  return () => {
    pendingSliderIndex = null;
    clearSliderTimer();
    for (const [target, type, handler] of handlers) {
      target.removeEventListener(type, handler);
    }
  };
}

/**
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
 *   tickForFrameIndex?: (frameIndex: number) => unknown,
 * }} elements
 * @param {ReturnType<ReplayPlaybackController["snapshot"]>} state
 */
export function renderReplayTimelineControls(elements, state) {
  const cursor = state.cursor;
  const unavailable =
    cursor === null ||
    !state.connected ||
    state.hidden ||
    state.requestPending ||
    state.presentationPending;
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
  elements.firstButton.disabled = unavailable || state.atStart;
  elements.backTenButton.disabled = unavailable || state.atStart;
  elements.previousButton.disabled = unavailable || state.atStart;
  elements.nextButton.disabled = unavailable || state.atEnd;
  elements.forwardTenButton.disabled = unavailable || state.atEnd;
  elements.lastButton.disabled = unavailable || state.atEnd;
  elements.playPauseButton.disabled =
    cursor === null ||
    !state.connected ||
    state.hidden ||
    ((state.requestPending || state.presentationPending) && !state.playing) ||
    (state.atEnd && !state.playing);
  elements.playPauseButton.textContent = state.playing ? "Pause" : "Play";
  elements.playPauseButton.setAttribute("aria-pressed", String(state.playing));
  elements.playPauseButton.setAttribute(
    "aria-label",
    state.playing ? "Pause replay" : "Play replay",
  );
}
