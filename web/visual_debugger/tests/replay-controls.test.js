import assert from "node:assert/strict";
import test from "node:test";

import {
  bindReplayTimelineControls,
  normalizeReplayCommand,
  normalizeReplayCursor,
  REPLAY_AUTOPLAY_CADENCE_MS,
  REPLAY_PLAYBACK_RATES,
  REPLAY_TRANSPORT_STATES,
  ReplayPlaybackController,
  renderReplayTimelineControls,
  replayCommandRequest,
  replayKeyboardIntent,
  replayNavigationCommand,
  replaySeekCommand,
  replayTimelineSimulatorStep,
  validateReplayAnimationIntent,
  validateReplayCommandOutcome,
  validateReplayCursorTransition,
} from "../src/replay-controls.js";

function deferred() {
  /** @type {(value?: any) => void} */
  let resolve = () => {};
  /** @type {(reason?: any) => void} */
  let reject = () => {};
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

async function flushMicrotasks() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

/**
 * @param {ReplayPlaybackController} controller
 * @param {ReturnType<typeof cursor>} value
 */
function installConnected(controller, value) {
  controller.setConnected(true);
  controller.installCursor(value);
}

class FakeElement {
  constructor() {
    /** @type {Map<string, Set<(event: any) => void>>} */
    this.handlers = new Map();
    this.attributes = new Map();
    this.disabled = false;
    this.value = "";
    this.min = "";
    this.max = "";
    this.step = "";
    this.textContent = "";
    this.interactive = false;
    this.hidden = false;
  }

  /** @param {string} type @param {(event: any) => void} handler */
  addEventListener(type, handler) {
    const handlers = this.handlers.get(type) ?? new Set();
    handlers.add(handler);
    this.handlers.set(type, handlers);
  }

  /** @param {string} type @param {(event: any) => void} handler */
  removeEventListener(type, handler) {
    this.handlers.get(type)?.delete(handler);
  }

  /** @param {string} name @param {unknown} value */
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  /** @param {string} name */
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  matches() {
    return this.interactive;
  }

  /** @param {string} type @param {Record<string, any>} event */
  dispatch(type, event = {}) {
    const fullEvent = {
      target: this,
      defaultPrevented: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
      ...event,
    };
    for (const handler of this.handlers.get(type) ?? []) {
      handler(fullEvent);
    }
    return fullEvent;
  }
}

/** @param {(frameIndex: number) => unknown} [tickForFrameIndex] */
function controlElements(
  tickForFrameIndex = (/** @type {number} */ frameIndex) => frameIndex,
) {
  return {
    root: new FakeElement(),
    keyboardTarget: new FakeElement(),
    keyboardEnabled: () => true,
    clearSelection: () => {},
    firstButton: new FakeElement(),
    backTenButton: new FakeElement(),
    previousButton: new FakeElement(),
    playPauseButton: new FakeElement(),
    nextButton: new FakeElement(),
    forwardTenButton: new FakeElement(),
    lastButton: new FakeElement(),
    slider: new FakeElement(),
    position: new FakeElement(),
    rateSelect: new FakeElement(),
    status: new FakeElement(),
    tickForFrameIndex,
    incomingTransitionForFrameIndex: (/** @type {number} */ frameIndex) =>
      frameIndex === 0 ? null : `episode:test:transition:${frameIndex - 1}`,
  };
}

/**
 * @param {number} frameIndex
 * @param {number} finalFrameIndex
 * @param {{cursor?: number, choreography?: number}} generations
 * @returns {{schema_version: 1, frame_index: number, final_frame_index: number, cursor_generation: number, choreography_generation: number}}
 */
function cursor(frameIndex, finalFrameIndex = 3, generations = {}) {
  return {
    schema_version: 1,
    frame_index: frameIndex,
    final_frame_index: finalFrameIndex,
    cursor_generation: generations.cursor ?? frameIndex,
    choreography_generation: generations.choreography ?? frameIndex,
  };
}

test("cursor and replay command constructors reject malformed authority inputs", () => {
  assert.deepEqual(normalizeReplayCursor(cursor(1)), cursor(1));
  assert.throws(() => normalizeReplayCursor(cursor(4, 3)), /exceeds/u);
  assert.throws(
    () => normalizeReplayCursor({ ...cursor(1), cursor_generation: -1 }),
    /cursor_generation/u,
  );
  assert.deepEqual(replayNavigationCommand("next"), {
    command_type: "next_frame",
  });
  assert.deepEqual(replaySeekCommand(2), {
    command_type: "absolute_seek",
    frame_index: 2,
  });
  assert.throws(() => replayNavigationCommand("random"), /Unknown/u);
  assert.throws(() => replaySeekCommand(1.5), /non-negative integer/u);
  assert.deepEqual(
    replayCommandRequest({
      clientId: "client",
      commandId: "command",
      baseRevision: 4,
      command: replayNavigationCommand("next"),
    }),
    {
      schema_version: 1,
      client_id: "client",
      command_id: "command",
      base_revision: 4,
      command: { command_type: "next_frame" },
    },
  );
  assert.throws(
    () =>
      replayCommandRequest({
        clientId: "client",
        commandId: "command",
        baseRevision: -1,
        command: replayNavigationCommand("next"),
      }),
    /base revision/u,
  );
  assert.deepEqual(
    normalizeReplayCommand({
      command_type: "select_agent",
      selected_global_slot: null,
    }),
    { command_type: "select_agent", selected_global_slot: null },
  );
  for (const preset of ["presentation", "analysis", "technical", "debug"]) {
    assert.deepEqual(normalizeReplayCommand({ command_type: "set_preset", preset }), {
      command_type: "set_preset",
      preset: "analysis",
    });
  }
  assert.deepEqual(
    normalizeReplayCommand({ command_type: "set_verbosity", verbose: true }),
    { command_type: "set_verbosity", verbose: false },
  );
  assert.throws(
    () =>
      normalizeReplayCommand({
        command_type: "next_frame",
        navigation: "next",
      }),
    /exact V1 fields/u,
  );
  assert.throws(
    () =>
      normalizeReplayCommand({
        command_type: "set_pov_actor",
        global_slot: 10,
      }),
    /ten-slot/u,
  );
  const presentationKey = `pov_${"a".repeat(64)}`;
  assert.deepEqual(
    normalizeReplayCommand({
      command_type: "set_pov_actor",
      presentation_key: presentationKey,
    }),
    { command_type: "set_pov_actor", presentation_key: presentationKey },
  );
  assert.throws(
    () =>
      normalizeReplayCommand({
        command_type: "set_pov_actor",
        presentation_key: "pov_not-opaque",
      }),
    /opaque Agent POV presentation key/u,
  );
  assert.throws(
    () =>
      normalizeReplayCommand({
        command_type: "set_pov_actor",
        global_slot: 1,
        presentation_key: presentationKey,
      }),
    /exact V1 fields/u,
  );
  assert.throws(
    () =>
      replayCommandRequest({
        clientId: "contains spaces",
        commandId: "command",
        baseRevision: 0,
        command: replayNavigationCommand("next"),
      }),
    /opaque identifier/u,
  );
});

test("keyboard intent is bounded to unmodified replay navigation and edge-triggered play", () => {
  assert.equal(replayKeyboardIntent({ key: "Home" }), null);
  assert.equal(replayKeyboardIntent({ key: "End" }), null);
  assert.equal(replayKeyboardIntent({ key: "ArrowLeft" }), "previous");
  assert.equal(replayKeyboardIntent({ key: "ArrowRight" }), "next");
  assert.equal(replayKeyboardIntent({ key: "ArrowLeft", shiftKey: true }), null);
  assert.equal(replayKeyboardIntent({ key: "ArrowRight", shiftKey: true }), null);
  assert.equal(replayKeyboardIntent({ key: " ", repeat: false }), "toggle");
  assert.equal(replayKeyboardIntent({ key: " ", repeat: true }), null);
  assert.equal(
    replayKeyboardIntent({ key: "Escape", repeat: false }),
    "clear_selection",
  );
  assert.equal(replayKeyboardIntent({ key: "Escape", repeat: true }), null);
  assert.equal(replayKeyboardIntent({ key: "Escape", shiftKey: true }), null);
  assert.equal(replayKeyboardIntent({ key: "Spacebar", repeat: false }), null);
  assert.equal(replayKeyboardIntent({ key: "ArrowRight", ctrlKey: true }), null);
  assert.equal(replayKeyboardIntent({ key: "n" }), null);
});

test("replay-owned Escape clears selection once without moving the replay cursor", () => {
  const originalElement = Object.getOwnPropertyDescriptor(globalThis, "Element");
  Object.defineProperty(globalThis, "Element", {
    configurable: true,
    value: FakeElement,
  });
  const elements = controlElements();
  let clearCount = 0;
  elements.clearSelection = () => {
    clearCount += 1;
  };
  const controller = new ReplayPlaybackController({
    onStateChange: () => {},
    request: async () => ({ frame: { cursor: cursor(1) } }),
  });
  installConnected(controller, cursor(1));
  const unbind = bindReplayTimelineControls(/** @type {any} */ (elements), controller);

  try {
    const initialCursor = controller.snapshot().cursor;
    const firstEscape = elements.keyboardTarget.dispatch("keydown", {
      key: "Escape",
      repeat: false,
      target: elements.root,
    });
    assert.equal(firstEscape.defaultPrevented, true);
    assert.equal(clearCount, 1);
    assert.deepEqual(controller.snapshot().cursor, initialCursor);

    const repeatedEscape = elements.keyboardTarget.dispatch("keydown", {
      key: "Escape",
      repeat: true,
      target: elements.root,
    });
    assert.equal(repeatedEscape.defaultPrevented, false);
    assert.equal(clearCount, 1);

    elements.keyboardEnabled = () => false;
    const fencedEscape = elements.keyboardTarget.dispatch("keydown", {
      key: "Escape",
      repeat: false,
      target: elements.root,
    });
    assert.equal(fencedEscape.defaultPrevented, false);
    assert.equal(clearCount, 1);
    assert.deepEqual(controller.snapshot().cursor, initialCursor);
  } finally {
    unbind();
    if (originalElement) {
      Object.defineProperty(globalThis, "Element", originalElement);
    } else {
      Reflect.deleteProperty(globalThis, "Element");
    }
  }
});

test("replay-owned Space suppresses scroll without repeating toggles or stealing native controls", () => {
  const originalElement = Object.getOwnPropertyDescriptor(globalThis, "Element");
  Object.defineProperty(globalThis, "Element", {
    configurable: true,
    value: FakeElement,
  });
  const elements = controlElements();
  const controller = new ReplayPlaybackController({
    onStateChange: () => {},
    request: async () => ({ frame: { cursor: cursor(0) } }),
  });
  installConnected(controller, cursor(0));
  const unbind = bindReplayTimelineControls(/** @type {any} */ (elements), controller);

  try {
    const firstSpace = elements.keyboardTarget.dispatch("keydown", {
      key: " ",
      repeat: false,
      target: elements.root,
    });
    assert.equal(firstSpace.defaultPrevented, true);
    assert.equal(controller.snapshot().playing, true);

    const repeatedSpace = elements.keyboardTarget.dispatch("keydown", {
      key: " ",
      repeat: true,
      target: elements.root,
    });
    assert.equal(repeatedSpace.defaultPrevented, true);
    assert.equal(controller.snapshot().playing, true);

    elements.keyboardEnabled = () => false;
    const fencedSpace = elements.keyboardTarget.dispatch("keydown", {
      key: " ",
      repeat: false,
      target: elements.root,
    });
    assert.equal(fencedSpace.defaultPrevented, true);
    assert.equal(controller.snapshot().playing, true);

    elements.playPauseButton.interactive = true;
    const nativeSpace = elements.keyboardTarget.dispatch("keydown", {
      key: " ",
      repeat: false,
      target: elements.playPauseButton,
    });
    assert.equal(nativeSpace.defaultPrevented, false);
    assert.equal(controller.snapshot().playing, true);
  } finally {
    unbind();
    if (originalElement) {
      Object.defineProperty(globalThis, "Element", originalElement);
    } else {
      Reflect.deleteProperty(globalThis, "Element");
    }
  }
});

test("timeline binding previews slider input without a request and commits once", async () => {
  const originalElement = Object.getOwnPropertyDescriptor(globalThis, "Element");
  Object.defineProperty(globalThis, "Element", {
    configurable: true,
    value: FakeElement,
  });
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  const elements = controlElements();
  const controller = new ReplayPlaybackController({
    onStateChange: (state) => {
      renderReplayTimelineControls(/** @type {any} */ (elements), state);
    },
    request: async (command) => {
      commands.push(command);
      return {
        frame: {
          cursor: cursor(Number(command.frame_index ?? 0), 3, {
            cursor: commands.length,
            choreography: 0,
          }),
        },
      };
    },
  });
  installConnected(controller, cursor(0));
  elements.slider.value = "2";
  elements.slider.max = "3";
  elements.playPauseButton.interactive = true;
  const unbind = bindReplayTimelineControls(/** @type {any} */ (elements), controller);

  try {
    elements.slider.dispatch("input");
    assert.equal(controller.snapshot().pauseReason, null);
    assert.equal(elements.slider.value, "2");
    assert.equal(elements.position.textContent, "Tick 2 / 3");
    assert.deepEqual(commands, []);
    elements.slider.dispatch("change");
    await flushMicrotasks();
    assert.deepEqual(commands, [replaySeekCommand(2)]);

    elements.slider.value = "3";
    elements.slider.dispatch("input");
    assert.equal(elements.position.textContent, "Tick 3 / 3");
    renderReplayTimelineControls(/** @type {any} */ (elements), controller.snapshot());
    assert.equal(elements.slider.value, "2");
    elements.slider.dispatch("change");
    await flushMicrotasks();
    assert.deepEqual(commands, [replaySeekCommand(2), replaySeekCommand(2)]);

    elements.rateSelect.value = "1.75";
    elements.rateSelect.dispatch("change");
    assert.equal(controller.snapshot().playbackRate, 1.75);
    assert.deepEqual(commands, [replaySeekCommand(2), replaySeekCommand(2)]);
    assert.match(elements.status.textContent, /1\.75× · SETTLED$/u);

    elements.keyboardTarget.dispatch("keydown", {
      key: " ",
      repeat: false,
      target: elements.playPauseButton,
    });
    assert.equal(controller.snapshot().playing, false);
    elements.keyboardTarget.dispatch("keydown", {
      key: "ArrowRight",
      shiftKey: true,
      target: elements.root,
    });
    assert.equal(commands.length, 2);
    elements.keyboardTarget.dispatch("keydown", {
      key: "ArrowRight",
      target: elements.root,
    });
    await flushMicrotasks();
    assert.deepEqual(commands.at(-1), replaySeekCommand(3));
    assert.equal(commands.length, 3);
    elements.root.hidden = true;
    elements.keyboardTarget.dispatch("keydown", {
      key: "ArrowRight",
      target: elements.root,
    });
    assert.equal(commands.length, 3);
    elements.root.hidden = false;
    elements.keyboardEnabled = () => false;
    elements.keyboardTarget.dispatch("keydown", {
      key: "ArrowLeft",
      target: elements.root,
    });
    assert.equal(commands.length, 3);
    elements.keyboardEnabled = () => true;
    controller.setAuthorityPending("disconnect");
    elements.keyboardTarget.dispatch("keydown", {
      key: "ArrowLeft",
      target: elements.root,
    });
    assert.equal(commands.length, 3);
  } finally {
    unbind();
    if (originalElement) {
      Object.defineProperty(globalThis, "Element", originalElement);
    } else {
      Reflect.deleteProperty(globalThis, "Element");
    }
  }
});

test("timeline rendering publishes bounded controls and accessible cursor text", () => {
  const elements = controlElements();
  renderReplayTimelineControls(/** @type {any} */ (elements), {
    transportState: REPLAY_TRANSPORT_STATES.SETTLED,
    generation: 1,
    presentationIntent: null,
    cursor: cursor(2, 3),
    playing: false,
    requestPending: false,
    presentationPending: false,
    connected: true,
    hidden: false,
    playbackRate: 1,
    pauseReason: null,
    atStart: false,
    atEnd: false,
  });
  assert.equal(elements.slider.value, "2");
  assert.equal(elements.slider.max, "3");
  assert.equal(elements.slider.getAttribute("aria-valuetext"), "Tick 2 of 3");
  assert.equal(elements.position.textContent, "Tick 2 / 3");
  assert.equal(elements.rateSelect.value, "1");
  assert.equal(
    elements.status.textContent,
    "Frame 2 / 3 · Tick 2 / 3 · Incoming transition episode:test:transition:1 · 1.00× · SETTLED",
  );
  assert.equal(elements.backTenButton.disabled, false);
  assert.equal(elements.previousButton.disabled, false);
  assert.equal(elements.nextButton.disabled, false);
  assert.equal(elements.forwardTenButton.disabled, false);
  assert.equal(elements.playPauseButton.getAttribute("aria-label"), "Play replay");
});

test("timeline labels use nonzero simulator ticks while slider seeks frame indices", () => {
  const timeline = {
    rows: Array.from({ length: 4 }, (_, frameIndex) => ({
      frame_index: frameIndex,
      simulator_step_count: 40 + frameIndex,
    })),
  };
  const elements = controlElements((frameIndex) =>
    replayTimelineSimulatorStep(timeline, frameIndex),
  );
  renderReplayTimelineControls(/** @type {any} */ (elements), {
    transportState: REPLAY_TRANSPORT_STATES.SETTLED,
    generation: 1,
    presentationIntent: null,
    cursor: cursor(2, 3),
    playing: false,
    requestPending: false,
    presentationPending: false,
    connected: true,
    hidden: false,
    playbackRate: 1,
    pauseReason: null,
    atStart: false,
    atEnd: false,
  });
  assert.equal(elements.slider.value, "2");
  assert.equal(elements.slider.max, "3");
  assert.equal(elements.slider.getAttribute("aria-valuetext"), "Tick 42 of 43");
  assert.equal(elements.position.textContent, "Tick 42 / 43");
  assert.equal(
    elements.status.textContent,
    "Frame 2 / 3 · Tick 42 / 43 · Incoming transition episode:test:transition:1 · 1.00× · SETTLED",
  );
  assert.equal(replayTimelineSimulatorStep(timeline, 2), 42);
  assert.equal(replayTimelineSimulatorStep(timeline, 4), null);
  assert.equal(
    replayTimelineSimulatorStep(
      { rows: [{ frame_index: 1, simulator_step_count: 40 }] },
      0,
    ),
    null,
  );
});

test("durable generations advance only for exact next and reject animation aliases", () => {
  const previous = cursor(1, 3, { cursor: 7, choreography: 2 });
  assert.deepEqual(
    validateReplayCursorTransition(
      replayNavigationCommand("next"),
      previous,
      cursor(2, 3, { cursor: 8, choreography: 3 }),
    ),
    cursor(2, 3, { cursor: 8, choreography: 3 }),
  );
  assert.deepEqual(
    validateReplayCursorTransition(
      replaySeekCommand(1),
      previous,
      cursor(1, 3, { cursor: 8, choreography: 2 }),
    ),
    cursor(1, 3, { cursor: 8, choreography: 2 }),
  );
  assert.throws(
    () =>
      validateReplayCursorTransition(
        replaySeekCommand(1),
        previous,
        cursor(1, 3, { cursor: 8, choreography: 3 }),
      ),
    /cursor generation/u,
  );
  assert.throws(
    () =>
      validateReplayAnimationIntent(replaySeekCommand(1), {
        animate_incoming: true,
      }),
    /applied next-frame/u,
  );
  assert.throws(
    () =>
      validateReplayAnimationIntent(replayNavigationCommand("next"), {
        result: "applied",
        animate_incoming: false,
      }),
    /exactly for an applied next-frame/u,
  );
  assert.doesNotThrow(() =>
    validateReplayAnimationIntent(replayNavigationCommand("next"), {
      result: "applied",
      animate_incoming: true,
    }),
  );
});

test("malformed command outcomes cannot mutate an installed cursor candidate", () => {
  const previous = cursor(1, 3, { cursor: 7, choreography: 2 });
  let installed = previous;
  const malformed = {
    result: "applied",
    animate_incoming: true,
    frame: { cursor: cursor(3, 3, { cursor: 8, choreography: 3 }) },
  };
  assert.throws(() => {
    const candidate = validateReplayCommandOutcome(
      replayNavigationCommand("next"),
      malformed,
      previous,
    );
    installed = candidate;
  }, /cursor generation/u);
  assert.equal(installed, previous);
});

test("an interleaved duplicate installs settled authority without retry or animation", async () => {
  const previous = cursor(1, 3, { cursor: 7, choreography: 2 });
  const authoritative = cursor(3, 3, { cursor: 9, choreography: 3 });
  assert.deepEqual(
    validateReplayCommandOutcome(
      replayNavigationCommand("next"),
      {
        result: "duplicate",
        animate_incoming: false,
        frame: { cursor: authoritative },
      },
      previous,
    ),
    authoritative,
  );
  assert.throws(
    () =>
      validateReplayCommandOutcome(
        replayNavigationCommand("next"),
        {
          result: "duplicate",
          animate_incoming: false,
          frame: {
            cursor: cursor(2, 3, { cursor: 6, choreography: 2 }),
          },
        },
        previous,
      ),
    /regressed/u,
  );

  /** @type {Array<Record<string, any>>} */
  const requests = [];
  /** @type {Array<unknown>} */
  const errors = [];
  let presentations = 0;
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      requests.push(command);
      return {
        result: "duplicate",
        animate_incoming: false,
        frame: { cursor: authoritative },
      };
    },
    waitForPresentation: async () => {
      presentations += 1;
    },
    onError: (error) => errors.push(error),
  });
  installConnected(controller, previous);
  assert.equal(await controller.next(), false);
  assert.equal(requests.length, 1);
  assert.equal(presentations, 0);
  assert.deepEqual(errors, []);
  assert.deepEqual(controller.snapshot().cursor, authoritative);
  assert.equal(controller.snapshot().playing, false);
  assert.equal(controller.snapshot().pauseReason, "resync");
});

test("explicit navigation is first-wins while one absolute seek is advancing", async () => {
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  /** @type {(value: unknown) => void} */
  let resolveRequest = () => {
    throw new Error("Request resolver was not installed.");
  };
  const requestPending = new Promise((resolve) => {
    resolveRequest = resolve;
  });
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      commands.push(command);
      return requestPending;
    },
  });
  installConnected(controller, cursor(1));
  const first = controller.next();
  const second = await controller.last();
  assert.equal(controller.snapshot().playing, false);
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.ADVANCING);
  assert.equal(controller.snapshot().pauseReason, "user_seek");
  assert.equal(second, false);
  assert.deepEqual(commands, [replaySeekCommand(2)]);
  resolveRequest({ frame: { cursor: cursor(2, 3, { cursor: 2, choreography: 1 }) } });
  assert.equal(await first, true);
  assert.equal(controller.snapshot().cursor?.frame_index, 2);
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.SETTLED);
});

test("same-frame and out-of-range destinations each send one clamped absolute seek", async () => {
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  let authoritative = cursor(1, 3, { cursor: 7, choreography: 2 });
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      commands.push(command);
      authoritative = cursor(Number(command.frame_index), 3, {
        cursor: authoritative.cursor_generation + 1,
        choreography: authoritative.choreography_generation,
      });
      return { frame: { cursor: authoritative } };
    },
  });
  installConnected(controller, authoritative);
  assert.equal(await controller.seek(1), true);
  assert.deepEqual(commands, [replaySeekCommand(1)]);
  assert.equal(await controller.seek(4), true);
  assert.deepEqual(commands, [replaySeekCommand(1), replaySeekCommand(3)]);
});

test("ten-tick jumps clamp and send one absolute seek instead of repeated requests", async () => {
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  let authoritative = cursor(12, 25, { cursor: 7, choreography: 2 });
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      commands.push(command);
      authoritative = cursor(Number(command.frame_index), 25, {
        cursor: authoritative.cursor_generation + 1,
        choreography: authoritative.choreography_generation,
      });
      return { frame: { cursor: authoritative } };
    },
  });
  installConnected(controller, authoritative);

  assert.equal(await controller.jump(-10), true);
  assert.deepEqual(commands, [replaySeekCommand(2)]);
  assert.equal(await controller.jump(10), true);
  assert.deepEqual(commands, [replaySeekCommand(2), replaySeekCommand(12)]);

  authoritative = cursor(24, 25, {
    cursor: authoritative.cursor_generation,
    choreography: authoritative.choreography_generation,
  });
  installConnected(controller, authoritative);
  assert.equal(await controller.jump(10), true);
  assert.deepEqual(commands.at(-1), replaySeekCommand(25));
  assert.equal(commands.length, 3);
  assert.equal(await controller.jump(10), true);
  assert.deepEqual(commands.at(-1), replaySeekCommand(25));
  assert.equal(commands.length, 4);
  await assert.rejects(controller.jump(0), /non-zero integer/u);
});

test("controller starts offline and reconnect settles only after a coherent cursor install", () => {
  const controller = new ReplayPlaybackController({ request: async () => null });
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.OFFLINE);
  assert.equal(controller.snapshot().cursor, null);

  controller.setConnected(true);
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.OFFLINE);
  controller.installCursor(cursor(1));
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.SETTLED);

  controller.setAuthorityPending();
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.OFFLINE);
  assert.equal(controller.snapshot().cursor, null);
  assert.equal(controller.snapshot().presentationIntent, null);
  controller.setConnected(true);
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.OFFLINE);
  controller.installCursor(cursor(2));
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.SETTLED);

  assert.deepEqual(REPLAY_AUTOPLAY_CADENCE_MS, {
    normal: 500,
    reduced: 750,
    off: 1_000,
  });
});

test("playback rate is exact controller state and never changes transport authority", () => {
  /** @type {unknown[]} */
  const requests = [];
  /** @type {Array<ReturnType<ReplayPlaybackController["snapshot"]>>} */
  const publications = [];
  const controller = new ReplayPlaybackController({
    playbackRate: 0.5,
    request: async (command) => {
      requests.push(command);
      return null;
    },
    onStateChange: (state) => publications.push(state),
  });
  installConnected(controller, cursor(1, 3));
  const baseline = controller.snapshot();
  const baselinePublicationCount = publications.length;

  assert.deepEqual(REPLAY_PLAYBACK_RATES, [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]);
  assert.equal(Object.isFrozen(REPLAY_PLAYBACK_RATES), true);
  assert.equal(baseline.playbackRate, 0.5);

  controller.setPlaybackRate(0.5);
  assert.equal(publications.length, baselinePublicationCount);
  for (const rate of REPLAY_PLAYBACK_RATES) {
    const before = controller.snapshot();
    const after = controller.setPlaybackRate(rate);
    assert.equal(after.playbackRate, rate);
    assert.equal(after.transportState, before.transportState);
    assert.equal(after.generation, before.generation);
    assert.equal(after.cursor, before.cursor);
    assert.equal(after.presentationIntent, before.presentationIntent);
    assert.equal(after.pauseReason, before.pauseReason);
    assert.equal(after.requestPending, false);
  }
  assert.deepEqual(requests, []);

  const accepted = controller.snapshot();
  for (const invalid of [
    0,
    0.1,
    0.3,
    2.25,
    Number.NaN,
    Number.POSITIVE_INFINITY,
    "1",
  ]) {
    assert.throws(
      () => controller.setPlaybackRate(/** @type {any} */ (invalid)),
      /Replay playback rate must be one of/u,
    );
    assert.equal(controller.snapshot().playbackRate, accepted.playbackRate);
    assert.equal(controller.snapshot().generation, accepted.generation);
    assert.equal(controller.snapshot().cursor, accepted.cursor);
  }
  assert.throws(
    () =>
      new ReplayPlaybackController({
        playbackRate: 1.1,
        request: async () => null,
      }),
    /Replay playback rate must be one of/u,
  );
});

test("every named navigation control sends one clamped absolute seek", async () => {
  /** @type {Array<Record<string, any>>} */
  const requests = [];
  let authoritative = cursor(2, 4, { cursor: 10, choreography: 3 });
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      requests.push(command);
      authoritative = cursor(Number(command.frame_index), 4, {
        cursor: authoritative.cursor_generation + 1,
        choreography: authoritative.choreography_generation,
      });
      return { frame: { cursor: authoritative } };
    },
  });
  installConnected(controller, authoritative);

  await controller.first();
  await controller.previous();
  await controller.next();
  await controller.last();

  assert.deepEqual(requests, [
    replaySeekCommand(0),
    replaySeekCommand(0),
    replaySeekCommand(1),
    replaySeekCommand(4),
  ]);
});

test("Play at frame zero advances privately, animates the successor, then settles", async () => {
  const presentation = deferred();
  /** @type {Array<Record<string, any>>} */
  const requests = [];
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      requests.push(command);
      return { frame: { cursor: cursor(1, 1, { cursor: 1, choreography: 1 }) } };
    },
    waitForPresentation: () => presentation.promise,
  });
  installConnected(controller, cursor(0, 1));
  assert.equal(controller.play(), true);
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.ADVANCING);
  assert.deepEqual(requests, [replayNavigationCommand("next")]);
  await flushMicrotasks();
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.PLAYING);
  assert.deepEqual(controller.snapshot().presentationIntent, {
    generation: 2,
    renderPolicy: "replay_animated",
    restartAnimated: false,
  });
  presentation.resolve();
  await flushMicrotasks();
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.SETTLED);
  assert.equal(controller.snapshot().playing, false);
  assert.equal(controller.snapshot().pauseReason, "endpoint");
  assert.equal(requests.length, 1);
});

test("Pause, filters, and hidden state invalidate a current replay generation", async () => {
  for (const stop of ["user_pause", "visual_filter_changed", "hidden"]) {
    const presentation = deferred();
    let requests = 0;
    const controller = new ReplayPlaybackController({
      request: async () => {
        requests += 1;
        return { frame: { cursor: cursor(2, 3) } };
      },
      waitForPresentation: () => presentation.promise,
    });
    installConnected(controller, cursor(1, 3));
    assert.equal(controller.play(), true);
    assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.PLAYING);
    assert.equal(controller.snapshot().presentationIntent?.restartAnimated, true);
    if (stop === "hidden") {
      controller.setHidden(true);
    } else {
      controller.pause(stop);
    }
    assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.SETTLED);
    assert.equal(
      controller.snapshot().presentationIntent?.renderPolicy,
      "replay_static",
    );
    presentation.resolve();
    await flushMicrotasks();
    assert.equal(requests, 0);
    assert.equal(controller.snapshot().pauseReason, stop);
  }
});

test("play-owned navigation rejects an incoherent private-next response", async () => {
  /** @type {Array<Record<string, any>>} */
  const requests = [];
  /** @type {unknown[]} */
  const errors = [];
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      requests.push(command);
      return { frame: { cursor: cursor(2, 3) } };
    },
    onError: (error) => errors.push(error),
  });
  installConnected(controller, cursor(0));
  controller.play();
  await flushMicrotasks();
  assert.deepEqual(requests, [replayNavigationCommand("next")]);
  assert.equal(controller.snapshot().playing, false);
  assert.equal(controller.snapshot().pauseReason, "error");
  assert.match(String(errors[0]), /cursor generation/u);
});

test("a validated stale resync installs its cursor without retry, animation, or error", async () => {
  /** @type {Array<Record<string, any>>} */
  const requests = [];
  /** @type {unknown[]} */
  const errors = [];
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      requests.push(command);
      return {
        handled_resync: true,
        frame: { cursor: cursor(2, 3, { cursor: 9, choreography: 4 }) },
      };
    },
    onError: (error) => errors.push(error),
  });
  installConnected(controller, cursor(1, 3, { cursor: 7, choreography: 2 }));

  assert.equal(await controller.next(), false);
  assert.deepEqual(requests, [replaySeekCommand(2)]);
  assert.equal(controller.snapshot().cursor?.frame_index, 2);
  assert.equal(controller.snapshot().pauseReason, "resync");
  assert.equal(controller.snapshot().playing, false);
  assert.deepEqual(errors, []);
});

test("Pause during a play-owned request settles the arriving successor without continuation", async () => {
  const response = deferred();
  /** @type {Array<Record<string, any>>} */
  const requests = [];
  let presentations = 0;
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      requests.push(command);
      return response.promise;
    },
    waitForPresentation: async () => {
      presentations += 1;
    },
  });
  installConnected(controller, cursor(0));

  assert.equal(controller.play(), true);
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.ADVANCING);
  assert.equal(controller.snapshot().playing, true);
  controller.pause("user_pause");
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.ADVANCING);
  assert.equal(controller.snapshot().playing, false);
  assert.equal(controller.snapshot().presentationIntent?.renderPolicy, "replay_static");
  assert.equal(await controller.next(), false);
  assert.deepEqual(requests, [replayNavigationCommand("next")]);

  response.resolve({
    frame: { cursor: cursor(1, 3, { cursor: 1, choreography: 1 }) },
  });
  await flushMicrotasks();
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.SETTLED);
  assert.equal(controller.snapshot().cursor?.frame_index, 1);
  assert.equal(controller.snapshot().playing, false);
  assert.equal(presentations, 0);
  assert.equal(requests.length, 1);
});

test("authority replacement fences a late request result and cannot revive playback", async () => {
  const response = deferred();
  const controller = new ReplayPlaybackController({
    request: async () => response.promise,
  });
  installConnected(controller, cursor(1));
  const advancing = controller.next();
  controller.setAuthorityPending("presentation_pending");
  controller.setConnected(true);

  response.resolve({ frame: { cursor: cursor(2) } });
  assert.equal(await advancing, false);
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.OFFLINE);
  assert.equal(controller.snapshot().cursor, null);
  assert.equal(controller.snapshot().presentationIntent, null);
});

test("Play replays middle and final incoming transitions before any continuation", async () => {
  const currentPresentation = deferred();
  const successorPresentation = deferred();
  const presentations = [currentPresentation, successorPresentation];
  let presentationIndex = 0;
  /** @type {Array<Record<string, any>>} */
  const requests = [];
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      requests.push(command);
      return {
        frame: {
          cursor: cursor(2, 3, { cursor: 2, choreography: 2 }),
        },
      };
    },
    waitForPresentation: () => presentations[presentationIndex++].promise,
  });
  installConnected(controller, cursor(1, 3));

  assert.equal(controller.play(), true);
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.PLAYING);
  assert.equal(controller.snapshot().presentationIntent?.restartAnimated, true);
  assert.deepEqual(requests, []);
  currentPresentation.resolve();
  await flushMicrotasks();
  assert.deepEqual(requests, [replayNavigationCommand("next")]);
  assert.equal(controller.snapshot().cursor?.frame_index, 2);
  assert.equal(controller.snapshot().transportState, REPLAY_TRANSPORT_STATES.PLAYING);
  assert.equal(controller.snapshot().presentationIntent?.restartAnimated, false);
  controller.pause();
  successorPresentation.resolve();
  await flushMicrotasks();
  assert.equal(requests.length, 1);

  const finalPresentation = deferred();
  const finalController = new ReplayPlaybackController({
    request: async () => {
      throw new Error("Final-frame replay must not request another cursor.");
    },
    waitForPresentation: () => finalPresentation.promise,
  });
  installConnected(finalController, cursor(3, 3));
  assert.equal(finalController.play(), true);
  assert.equal(finalController.snapshot().presentationIntent?.restartAnimated, true);
  finalPresentation.resolve();
  await flushMicrotasks();
  assert.equal(
    finalController.snapshot().transportState,
    REPLAY_TRANSPORT_STATES.SETTLED,
  );
  assert.equal(finalController.snapshot().pauseReason, "endpoint");

  const emptyController = new ReplayPlaybackController({ request: async () => null });
  installConnected(emptyController, cursor(0, 0));
  assert.equal(emptyController.play(), false);
  assert.equal(
    emptyController.snapshot().transportState,
    REPLAY_TRANSPORT_STATES.SETTLED,
  );
});
