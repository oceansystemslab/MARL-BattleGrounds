/**
 * @file Translate battlefield UI input into commands and recording notices.
 * Python remains the authority for command legality and recording lifecycle.
 * These browser helpers build requests and decide local focus/confirmation
 * behavior; they do not execute a simulator transition or persist a replay.
 */
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
  "w",
  "a",
  "s",
  "d",
  "q",
  "e",
  "z",
  "c",
  "x",
  "r",
  "g",
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

const RECORDING_PRESENTATION_KEYS = new Set(["g", "?"]);

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
 * Copy source's optional keyboard modifiers into snake-case boolean fields.
 * Missing values become false. Return a new mutable record without changing
 * source; this helper does not decide whether a shortcut belongs to the game.
 *
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
 * Build a keyboard request from key and optional modifier/repeat flags.
 * All flags default to false and are converted to booleans; key is retained
 * exactly. Return a new mutable command record without interpreting the key
 * or checking action legality. Python handles the command's meaning.
 *
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
 * Return whether rawPresentation says local animation must settle before
 * submission: submissionBlocked or paused is true, or animationCount is a
 * positive integer. Nonobject input returns false. This checks the supplied
 * local state only; it does not stop animation or send a command.
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
 * Translate a validated selector string into a target command. Empty value
 * returns an Escape keyboard command; one digit 0–9 returns a roster_selection
 * request with that global slot. Other strings return null. No targetability
 * or permission is granted by this conversion.
 *
 * @param {string} value
 * @returns {Record<string, unknown> | null}
 */
export function targetSelectionCommand(value) {
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
 * Return whether value is manual, reactive_tdm or random_valid. This exact
 * name check excludes Team B's separately handled scenario_5 controller.
 *
 * @param {unknown} value
 */
function isTeamController(value) {
  return value === "manual" || value === "reactive_tdm" || value === "random_valid";
}

/**
 * Project an effective episode replacement from frame and command, or null.
 *
 * Recognize reset, a changed valid combat configuration, an available changed
 * scenario, or an unmodified R keyboard reset. Reactive controllers require
 * SharedObs; scenario_5 is accepted only for Team B. Return a frozen exact
 * replacement request. This is an advisory browser check against the current
 * frame; Python repeats validation against the actual session. Inputs stay
 * unchanged and no reset or recording discard occurs here.
 *
 * @param {Record<string, any>} frame
 * @param {Record<string, unknown>} command
 * @returns {Readonly<Record<string, unknown>> | null}
 */
export function recordingReplacementCommand(frame, command) {
  if (command.command_type === "reset") {
    return Object.freeze({ command_type: "reset" });
  }
  if (command.command_type === "set_combat_configuration") {
    const installed = frame.combat_configuration;
    const teamAController = command.team_a_controller;
    const teamBController = command.team_b_controller;
    const informationMode = command.execution_information_mode;
    if (
      !installed ||
      typeof installed !== "object" ||
      Array.isArray(installed) ||
      !isTeamController(teamAController) ||
      (!isTeamController(teamBController) && teamBController !== "scenario_5") ||
      (informationMode !== "shared_obs" && informationMode !== "no_shared_obs") ||
      ((teamAController === "reactive_tdm" ||
        teamBController === "reactive_tdm" ||
        teamBController === "scenario_5") &&
        informationMode !== "shared_obs") ||
      (installed.team_a_controller === teamAController &&
        installed.team_b_controller === teamBController &&
        installed.execution_information_mode === informationMode)
    ) {
      return null;
    }
    return Object.freeze({
      command_type: "set_combat_configuration",
      team_a_controller: teamAController,
      team_b_controller: teamBController,
      execution_information_mode: informationMode,
    });
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
  return null;
}

/**
 * Return a frozen allow, block or confirm decision for command in frame.
 *
 * Missing recording state and recording-lifecycle commands are allowed. An
 * episode replacement asks for confirmation when discard_available is true,
 * or blocks when restart_fenced is true. Outside active recording, block
 * other scientific commands while allowing the declared display-only ones.
 * An allow result retains the command reference; notices explain other
 * results. This does not send commands or replace Python lifecycle checks.
 *
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
 * Return a frozen save_as request for a valid replay file name, or null.
 * value must be 20–160 characters, begin with an ASCII letter/digit, contain
 * only letters/digits/dot/underscore/hyphen and end in .marlbg-replay.json.
 * Paths and surrounding whitespace are rejected. No file is written here.
 *
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
 * Return true only when command is exit and object payload reports
 * result=shutdown_scheduled. Other payload shapes return false. This checks
 * the response marker; it does not close the browser or stop the service.
 *
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

/**
 * Return whether frame is still a live view whose recording lifecycle says
 * reviewing. Nonobjects, replay views and missing/other recording states return
 * false. The caller performs the handoff; this check changes no state.
 *
 * @param {unknown} frame
 */
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
 * Build a new keyboard request from event.key, modifiers and repeat.
 * Keep the raw key meaning for Python. Do not cancel the event or change focus.
 *
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
 * Return whether event.key belongs to the battlefield key set. Ctrl, Alt
 * or Meta combinations always return false; Shift is allowed. One-character
 * keys are lowercased for lookup. No event is cancelled by this check.
 *
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
 * Convert event client-pixel coordinates into svg's local coordinates.
 * Use the inverse screen transform; return null when no transform exists.
 * Return a new x/y record without moving the pointer or DOM. The caller
 * checks world conversion and finiteness; matrix errors may propagate.
 *
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
 * Attach keyboard and primary-pointer listeners to one focusable battlefield.
 *
 * bindings.battlefield receives the listeners. toWorldPoint maps SVG points
 * to world_x/world_y or null. onCommand handles built requests; onHelp shows
 * help and onReleaseFocus follows Escape, even if its awaited command fails.
 * onPointerCommand defaults to a no-op and observes the pointer target/request
 * before submission. isInteractive defaults to true; onFencedCommand defaults
 * to false and may consume blocked keys; ownsFencedSpaceDefault defaults to
 * true to suppress page scrolling for a blocked Space key.
 *
 * Only events targeted at the battlefield are captured. Pointer input focuses
 * it and rejects missing/nonfinite world coordinates. Ordinary command
 * promises are not awaited here; callers own their error handling. Return
 * undefined. No disposer is returned, so bind once per battlefield element
 * and release its owning DOM when finished. No document-level listener or
 * Python action logic is added.
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
 *   onReleaseFocus: () => void,
 *   isInteractive?: () => boolean,
 *   onFencedCommand?: (command: Record<string, unknown>) => boolean,
 *   ownsFencedSpaceDefault?: () => boolean,
 * }} bindings
 */
export function bindBattlefieldControls({
  battlefield,
  toWorldPoint,
  onCommand,
  onPointerCommand = () => {},
  onHelp,
  onReleaseFocus,
  isInteractive = () => true,
  onFencedCommand = () => false,
  ownsFencedSpaceDefault = () => true,
}) {
  battlefield.addEventListener("keydown", async (event) => {
    if (event.target !== battlefield || !isDebuggerKey(event)) {
      return;
    }
    if (!isInteractive()) {
      if (onFencedCommand(keyboardCommandFromEvent(event))) {
        event.preventDefault();
        event.stopPropagation();
      } else if (event.key === " " && ownsFencedSpaceDefault()) {
        event.preventDefault();
      }
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (event.key === "?") {
      onHelp();
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
    if (event.button !== 0) {
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
      button: "primary",
      ...modifierFields(event),
    };
    onPointerCommand(event.target, command);
    onCommand(command);
  });
}
