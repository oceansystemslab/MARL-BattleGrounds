const GAME_KEYS = new Set([
  "Tab",
  "Escape",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  " ",
  "Enter",
  "0",
  "1",
  "2",
  "[",
  "]",
  "w",
  "a",
  "s",
  "d",
  "q",
  "e",
  "z",
  "c",
  "x",
  "n",
  "r",
  "g",
  "p",
  "?",
]);

const RECORDING_LIFECYCLE_COMMANDS = new Set([
  "finish_and_review",
  "review_replay",
  "retry_save",
  "save_as",
  "confirm_discard_and_replace",
  "exit",
]);

const RECORDING_PRESENTATION_KEYS = new Set(["g", "p", "?"]);

const RECORDING_SAVE_AS_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*\.marlbg-replay\.json$/u;

/**
 * @typedef {{
 *   shiftKey?: boolean,
 *   ctrlKey?: boolean,
 *   altKey?: boolean,
 *   metaKey?: boolean,
 * }} ModifierSource
 */

/**
 * @param {ModifierSource} source
 * @returns {{
 *   shift_key: boolean,
 *   ctrl_key: boolean,
 *   alt_key: boolean,
 *   meta_key: boolean,
 * }}
 */
function modifierFields(source) {
  return {
    shift_key: Boolean(source.shiftKey),
    ctrl_key: Boolean(source.ctrlKey),
    alt_key: Boolean(source.altKey),
    meta_key: Boolean(source.metaKey),
  };
}

/**
 * @param {string} key
 * @param {ModifierSource & {repeat?: boolean}} options
 */
export function keyboardCommand(
  key,
  {
    shiftKey = false,
    ctrlKey = false,
    altKey = false,
    metaKey = false,
    repeat = false,
  } = {},
) {
  return {
    command_type: "keyboard",
    key,
    shift_key: Boolean(shiftKey),
    ctrl_key: Boolean(ctrlKey),
    alt_key: Boolean(altKey),
    meta_key: Boolean(metaKey),
    repeat: Boolean(repeat),
  };
}

/**
 * Decide whether Submit must synchronously settle local presentation before
 * entering the normal request fence. The readable submission gate may already
 * be open while animations are still active, and paused animations remain
 * active regardless of that gate.
 *
 * @param {unknown} rawPresentation
 */
export function presentationRequiresSubmissionSettle(rawPresentation) {
  if (
    typeof rawPresentation !== "object" ||
    rawPresentation === null ||
    Array.isArray(rawPresentation)
  ) {
    return false;
  }
  const presentation = /** @type {Record<string, unknown>} */ (rawPresentation);
  return (
    presentation.submissionBlocked === true ||
    presentation.paused === true ||
    (Number.isInteger(presentation.animationCount) &&
      Number(presentation.animationCount) > 0)
  );
}

/**
 * Convert the target select's audience-specific option value into the one
 * command authorized for that audience. Actor POV values deliberately carry
 * only the recipient-relative target-action axis; this boundary must never
 * manufacture or transmit a researcher global slot.
 *
 * @param {string} value
 * @param {{actorPov?: boolean}} options
 * @returns {Record<string, unknown> | null}
 */
export function targetSelectionCommand(value, { actorPov = false } = {}) {
  if (actorPov) {
    const match = /^pov-target-action:(0|[1-9]|10)$/u.exec(value);
    if (!match) {
      return null;
    }
    return {
      command_type: "actor_pov_target_action",
      target_action: Number(match[1]),
    };
  }
  if (value === "") {
    return keyboardCommand("Escape");
  }
  if (!/^(0|[1-9])$/u.test(value)) {
    return null;
  }
  return {
    command_type: "roster_selection",
    role: "target",
    global_slot: Number(value),
  };
}

/**
 * Resolve only effective episode replacements to the exact non-keyboard
 * command accepted by ConfirmDiscardAndReplaceCommandV1. This is advisory UX;
 * Python repeats the same classification against authoritative session state.
 *
 * @param {Record<string, any>} frame
 * @param {Record<string, unknown>} command
 * @returns {Readonly<Record<string, unknown>> | null}
 */
export function recordingReplacementCommand(frame, command) {
  if (command.command_type === "reset") {
    return Object.freeze({ command_type: "reset" });
  }
  if (command.command_type === "scenario_switch") {
    const scenarioName = command.scenario_name;
    if (
      typeof scenarioName !== "string" ||
      scenarioName === frame.scenario?.name ||
      !Array.isArray(frame.available_scenarios) ||
      !frame.available_scenarios.some(
        (entry) => (typeof entry === "string" ? entry : entry?.name) === scenarioName,
      )
    ) {
      return null;
    }
    return Object.freeze({
      command_type: "scenario_switch",
      scenario_name: scenarioName,
    });
  }
  if (
    command.command_type !== "keyboard" ||
    typeof command.key !== "string" ||
    command.ctrl_key === true ||
    command.alt_key === true ||
    command.meta_key === true
  ) {
    return null;
  }
  const key = command.key.toLowerCase();
  if (key === "r" && command.shift_key !== true) {
    return Object.freeze({ command_type: "reset" });
  }
  if (key !== "[" && key !== "]") {
    return null;
  }
  const names = Array.isArray(frame.available_scenarios)
    ? frame.available_scenarios.flatMap((entry) => {
        const name = typeof entry === "string" ? entry : entry?.name;
        return typeof name === "string" ? [name] : [];
      })
    : [];
  const currentName = frame.scenario?.name;
  const currentIndex = names.indexOf(currentName);
  if (currentIndex < 0 || names.length < 2) {
    return null;
  }
  const direction = key === "[" ? -1 : 1;
  const nextIndex = (currentIndex + direction + names.length) % names.length;
  return Object.freeze({
    command_type: "scenario_switch",
    scenario_name: names[nextIndex],
  });
}

/**
 * @param {Record<string, any>} frame
 * @param {Record<string, unknown>} command
 * @returns {Readonly<{
 *   action: "allow" | "block" | "confirm",
 *   command?: Record<string, unknown>,
 *   replacement?: Record<string, unknown>,
 *   notice?: string,
 * }>}
 */
export function recordingCommandDecision(frame, command) {
  const status = frame.recording;
  if (!status || typeof status !== "object" || Array.isArray(status)) {
    return Object.freeze({ action: "allow", command });
  }
  if (RECORDING_LIFECYCLE_COMMANDS.has(String(command.command_type))) {
    return Object.freeze({ action: "allow", command });
  }
  const replacement = recordingReplacementCommand(frame, command);
  if (replacement) {
    if (status.discard_available === true) {
      return Object.freeze({
        action: "confirm",
        replacement,
        notice:
          "Replacing this episode discards the captured in-memory prefix. Confirm the exact replacement to continue.",
      });
    }
    if (status.restart_fenced === true) {
      return Object.freeze({
        action: "block",
        notice:
          "Episode replacement is fenced after recording closeout. Review or recover the replay first.",
      });
    }
    return Object.freeze({ action: "allow", command });
  }

  const presentationOnly =
    command.command_type === "set_view" ||
    command.command_type === "set_preset" ||
    (command.command_type === "keyboard" &&
      typeof command.key === "string" &&
      RECORDING_PRESENTATION_KEYS.has(command.key.toLowerCase()));
  if (status.lifecycle !== "recording" && !presentationOnly) {
    return Object.freeze({
      action: "block",
      notice:
        "Scientific controls are fenced because this recording is no longer capturing transitions.",
    });
  }
  return Object.freeze({ action: "allow", command });
}

/**
 * @param {unknown} value
 * @returns {Readonly<{command_type: "save_as", file_name: string}> | null}
 */
export function recordingSaveAsCommand(value) {
  if (
    typeof value !== "string" ||
    value.length < 20 ||
    value.length > 160 ||
    !RECORDING_SAVE_AS_PATTERN.test(value)
  ) {
    return null;
  }
  return Object.freeze({ command_type: "save_as", file_name: value });
}

/**
 * @param {Record<string, unknown>} command
 * @param {unknown} payload
 */
export function commandResponseSchedulesShutdown(command, payload) {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    return false;
  }
  const response = /** @type {Record<string, unknown>} */ (payload);
  return command.command_type === "exit" && response.result === "shutdown_scheduled";
}

/** @param {unknown} frame */
export function recordingReviewHandoffRequired(frame) {
  if (typeof frame !== "object" || frame === null || Array.isArray(frame)) {
    return false;
  }
  const candidate = /** @type {Record<string, any>} */ (frame);
  return (
    candidate.viewer_mode !== "replay" &&
    typeof candidate.recording === "object" &&
    candidate.recording !== null &&
    !Array.isArray(candidate.recording) &&
    candidate.recording.lifecycle === "reviewing"
  );
}

/**
 * @param {KeyboardEvent} event
 */
function keyboardCommandFromEvent(event) {
  return {
    command_type: "keyboard",
    key: event.key,
    ...modifierFields(event),
    repeat: Boolean(event.repeat),
  };
}

/**
 * @param {{
 *   key: string,
 *   shiftKey?: boolean,
 *   ctrlKey?: boolean,
 *   altKey?: boolean,
 *   metaKey?: boolean,
 * }} event
 */
export function isDebuggerKey(event) {
  if (event.ctrlKey || event.altKey || event.metaKey) {
    return false;
  }
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  return GAME_KEYS.has(key);
}

/**
 * Presentation pause is local and edge-triggered. Key repeat must never
 * oscillate the controller or leave a future submission gate paused.
 *
 * @param {{key?: string, repeat?: boolean}} event
 */
export function isPresentationPauseEvent(event) {
  return event.key?.toLowerCase() === "p" && !event.repeat;
}

/**
 * @param {SVGSVGElement} svg
 * @param {PointerEvent} event
 * @returns {{x: number, y: number} | null}
 */
function pointInSvg(svg, event) {
  const matrix = svg.getScreenCTM();
  if (!matrix) {
    return null;
  }
  const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(
    matrix.inverse(),
  );
  return { x: point.x, y: point.y };
}

/**
 * Bind only to the focusable battlefield. No document-level key listener is
 * installed, so native controls and panel inputs retain ordinary Tab behavior.
 *
 * @param {{
 *   battlefield: SVGSVGElement,
 *   toWorldPoint: (point: {x: number, y: number}) =>
 *     {world_x: number, world_y: number} | null,
 *   onCommand: (command: Record<string, unknown>) => void | Promise<void>,
 *   onPointerCommand?: (
 *     target: EventTarget | null,
 *     command: Readonly<Record<string, unknown>>,
 *   ) => void,
 *   onHelp: () => void,
 *   onPresentationKey?: (key: "toggle-pause") => void,
 *   onReleaseFocus: () => void,
 *   isInteractive?: () => boolean,
 * }} bindings
 */
export function bindBattlefieldControls({
  battlefield,
  toWorldPoint,
  onCommand,
  onPointerCommand = () => {},
  onHelp,
  onPresentationKey = () => {},
  onReleaseFocus,
  isInteractive = () => true,
}) {
  battlefield.addEventListener("keydown", async (event) => {
    if (!isInteractive()) {
      return;
    }
    if (!isDebuggerKey(event)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (event.key === "?") {
      onHelp();
      return;
    }
    if (event.key.toLowerCase() === "p") {
      if (isPresentationPauseEvent(event)) {
        onPresentationKey("toggle-pause");
      }
      return;
    }
    if (event.key === "Escape") {
      try {
        await onCommand(keyboardCommandFromEvent(event));
      } finally {
        onReleaseFocus();
      }
      return;
    }
    onCommand(keyboardCommandFromEvent(event));
  });

  battlefield.addEventListener("pointerdown", (event) => {
    if (!isInteractive()) {
      return;
    }
    if (event.button !== 0 && event.button !== 2) {
      return;
    }
    event.preventDefault();
    battlefield.focus({ preventScroll: true });
    const svgPoint = pointInSvg(battlefield, event);
    if (!svgPoint) {
      return;
    }
    const worldPoint = toWorldPoint(svgPoint);
    if (
      !worldPoint ||
      !Number.isFinite(worldPoint.world_x) ||
      !Number.isFinite(worldPoint.world_y)
    ) {
      return;
    }
    const command = {
      command_type: "battlefield_pointer",
      world_x: worldPoint.world_x,
      world_y: worldPoint.world_y,
      button: event.button === 0 ? "primary" : "secondary",
      ...modifierFields(event),
    };
    onPointerCommand(event.target, command);
    onCommand(command);
  });

  battlefield.addEventListener("contextmenu", (event) => {
    if (isInteractive()) {
      event.preventDefault();
    }
  });
}
