const GAME_KEYS = new Set([
  "Tab",
  "Escape",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  " ",
  "Enter",
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
  "n",
  "r",
  "g",
  "v",
  "p",
  "?",
]);

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
 *   onHelp: () => void,
 *   onPresentationKey?: (key: "toggle-pause") => void,
 *   onReleaseFocus: () => void,
 * }} bindings
 */
export function bindBattlefieldControls({
  battlefield,
  toWorldPoint,
  onCommand,
  onHelp,
  onPresentationKey = () => {},
  onReleaseFocus,
}) {
  battlefield.addEventListener("keydown", async (event) => {
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
    onCommand({
      command_type: "battlefield_pointer",
      world_x: worldPoint.world_x,
      world_y: worldPoint.world_y,
      button: event.button === 0 ? "primary" : "secondary",
      ...modifierFields(event),
    });
  });

  battlefield.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });
}
