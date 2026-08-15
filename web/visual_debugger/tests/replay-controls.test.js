import assert from "node:assert/strict";
import test from "node:test";

import {
  bindReplayTimelineControls,
  normalizeReplayCommand,
  normalizeReplayCursor,
  REPLAY_AUTOPLAY_CADENCE_MS,
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

class FakeClock {
  constructor() {
    this.now = 0;
    this.nextId = 1;
    /** @type {Map<number, {at: number, callback: () => void | Promise<void>}>} */
    this.timers = new Map();
  }

  /**
   * @param {() => void | Promise<void>} callback
   * @param {number} delay
   */
  setTimeout = (callback, delay) => {
    const id = this.nextId;
    this.nextId += 1;
    this.timers.set(id, { at: this.now + Number(delay), callback });
    return id;
  };

  /** @param {number} id */
  clearTimeout = (id) => {
    this.timers.delete(id);
  };

  /** @param {number} milliseconds */
  async advance(milliseconds) {
    const target = this.now + milliseconds;
    while (true) {
      const due = [...this.timers.entries()]
        .filter(([, timer]) => timer.at <= target)
        .sort((left, right) => left[1].at - right[1].at)[0];
      if (!due) {
        break;
      }
      const [id, timer] = due;
      this.timers.delete(id);
      this.now = timer.at;
      await timer.callback();
      await Promise.resolve();
    }
    this.now = target;
    await Promise.resolve();
  }
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
      preventDefault() {},
      ...event,
    };
    for (const handler of this.handlers.get(type) ?? []) {
      handler(fullEvent);
    }
  }
}

/** @param {(frameIndex: number) => unknown} [tickForFrameIndex] */
function controlElements(
  tickForFrameIndex = (/** @type {number} */ frameIndex) => frameIndex,
) {
  return {
    root: new FakeElement(),
    firstButton: new FakeElement(),
    backTenButton: new FakeElement(),
    previousButton: new FakeElement(),
    playPauseButton: new FakeElement(),
    nextButton: new FakeElement(),
    forwardTenButton: new FakeElement(),
    lastButton: new FakeElement(),
    slider: new FakeElement(),
    position: new FakeElement(),
    tickForFrameIndex,
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
  assert.equal(replayKeyboardIntent({ key: "Home" }), "first");
  assert.equal(replayKeyboardIntent({ key: "End" }), "last");
  assert.equal(replayKeyboardIntent({ key: "ArrowLeft" }), "previous");
  assert.equal(replayKeyboardIntent({ key: "ArrowRight" }), "next");
  assert.equal(replayKeyboardIntent({ key: "ArrowLeft", shiftKey: true }), "back_ten");
  assert.equal(
    replayKeyboardIntent({ key: "ArrowRight", shiftKey: true }),
    "forward_ten",
  );
  assert.equal(replayKeyboardIntent({ key: " ", repeat: false }), "toggle");
  assert.equal(replayKeyboardIntent({ key: " ", repeat: true }), null);
  assert.equal(replayKeyboardIntent({ key: "ArrowRight", ctrlKey: true }), null);
  assert.equal(replayKeyboardIntent({ key: "n" }), null);
});

test("timeline binding debounces slider seeks and preserves native control keys", async () => {
  const originalElement = Object.getOwnPropertyDescriptor(globalThis, "Element");
  Object.defineProperty(globalThis, "Element", {
    configurable: true,
    value: FakeElement,
  });
  const clock = new FakeClock();
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  const elements = controlElements();
  const controller = new ReplayPlaybackController({
    clock,
    onStateChange: (state) => {
      renderReplayTimelineControls(/** @type {any} */ (elements), state);
    },
    request: async (command) => {
      commands.push(command);
      return {
        cursor: cursor(Number(command.frame_index ?? 0), 3, {
          cursor: 1,
          choreography: 0,
        }),
      };
    },
  });
  controller.installCursor(cursor(0));
  elements.slider.value = "2";
  elements.slider.max = "3";
  elements.playPauseButton.interactive = true;
  const unbind = bindReplayTimelineControls(
    /** @type {any} */ (elements),
    controller,
    clock,
  );

  try {
    elements.slider.dispatch("input");
    assert.equal(controller.snapshot().pauseReason, "user_seek");
    assert.equal(elements.slider.value, "2");
    assert.equal(elements.position.textContent, "Tick 2 / 3");
    await clock.advance(159);
    assert.deepEqual(commands, []);
    await clock.advance(1);
    await Promise.resolve();
    assert.deepEqual(commands, [replaySeekCommand(2)]);

    elements.root.dispatch("keydown", {
      key: " ",
      repeat: false,
      target: elements.playPauseButton,
    });
    assert.equal(controller.snapshot().playing, false);
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
    cursor: cursor(2, 3),
    playing: false,
    requestPending: false,
    presentationPending: false,
    connected: true,
    hidden: false,
    pauseReason: null,
    atStart: false,
    atEnd: false,
  });
  assert.equal(elements.slider.value, "2");
  assert.equal(elements.slider.max, "3");
  assert.equal(elements.slider.getAttribute("aria-valuetext"), "Tick 2 of 3");
  assert.equal(elements.position.textContent, "Tick 2 / 3");
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
    cursor: cursor(2, 3),
    playing: false,
    requestPending: false,
    presentationPending: false,
    connected: true,
    hidden: false,
    pauseReason: null,
    atStart: false,
    atEnd: false,
  });
  assert.equal(elements.slider.value, "2");
  assert.equal(elements.slider.max, "3");
  assert.equal(elements.slider.getAttribute("aria-valuetext"), "Tick 42 of 43");
  assert.equal(elements.position.textContent, "Tick 42 / 43");
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
  controller.installCursor(previous);
  controller.play();
  assert.equal(await controller.next(), false);
  assert.equal(requests.length, 1);
  assert.equal(presentations, 0);
  assert.deepEqual(errors, []);
  assert.deepEqual(controller.snapshot().cursor, authoritative);
  assert.equal(controller.snapshot().playing, false);
  assert.equal(controller.snapshot().pauseReason, "resync");
});

test("explicit navigation pauses autoplay and serializes one request", async () => {
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
  controller.installCursor(cursor(1));
  controller.play();
  const first = controller.next();
  const second = await controller.last();
  assert.equal(controller.snapshot().playing, false);
  assert.equal(controller.snapshot().pauseReason, "user_seek");
  assert.equal(second, false);
  assert.deepEqual(commands, [replayNavigationCommand("next")]);
  resolveRequest({ frame: { cursor: cursor(2) } });
  assert.equal(await first, true);
  assert.equal(controller.snapshot().cursor?.frame_index, 2);
});

test("same-frame absolute seek is sent while out-of-range seek is rejected locally", async () => {
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  const controller = new ReplayPlaybackController({
    request: async (command) => {
      commands.push(command);
      return { frame: { cursor: cursor(1, 3, { cursor: 8, choreography: 2 }) } };
    },
  });
  controller.installCursor(cursor(1, 3, { cursor: 7, choreography: 2 }));
  assert.equal(await controller.seek(1), true);
  assert.deepEqual(commands, [replaySeekCommand(1)]);
  await assert.rejects(controller.seek(4), /outside/u);
  assert.equal(commands.length, 1);
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
  controller.installCursor(authoritative);

  assert.equal(await controller.jump(-10), true);
  assert.deepEqual(commands, [replaySeekCommand(2)]);
  assert.equal(await controller.jump(10), true);
  assert.deepEqual(commands, [replaySeekCommand(2), replaySeekCommand(12)]);

  authoritative = cursor(24, 25, {
    cursor: authoritative.cursor_generation,
    choreography: authoritative.choreography_generation,
  });
  controller.installCursor(authoritative);
  assert.equal(await controller.jump(10), true);
  assert.deepEqual(commands.at(-1), replaySeekCommand(25));
  assert.equal(commands.length, 3);
  assert.equal(await controller.jump(10), false);
  assert.equal(commands.length, 3);
  await assert.rejects(controller.jump(0), /non-zero integer/u);
});

test("autoplay waits for both response and presentation before scheduling another request", async () => {
  const clock = new FakeClock();
  /** @type {Array<Record<string, any>>} */
  const commands = [];
  let frameIndex = 0;
  /** @type {() => void} */
  let releasePresentation = () => {
    throw new Error("Presentation resolver was not installed.");
  };
  /** @type {Promise<void>} */
  let presentation = new Promise((resolve) => {
    releasePresentation = resolve;
  });
  const controller = new ReplayPlaybackController({
    clock,
    getMotionMode: () => "normal",
    request: async (command) => {
      commands.push(command);
      frameIndex += 1;
      return { frame: { cursor: cursor(frameIndex) } };
    },
    waitForPresentation: () => presentation,
  });
  controller.installCursor(cursor(0));
  assert.equal(controller.play(), true);
  await clock.advance(REPLAY_AUTOPLAY_CADENCE_MS.normal);
  assert.equal(commands.length, 1);
  assert.equal(controller.snapshot().presentationPending, true);
  await clock.advance(REPLAY_AUTOPLAY_CADENCE_MS.normal * 5);
  assert.equal(commands.length, 1);
  releasePresentation();
  await Promise.resolve();
  await Promise.resolve();
  presentation = Promise.resolve();
  await clock.advance(REPLAY_AUTOPLAY_CADENCE_MS.normal);
  assert.equal(commands.length, 2);
  assert.deepEqual(commands, [
    replayNavigationCommand("next"),
    replayNavigationCommand("next"),
  ]);
});

test("motion off autoplay retains a bounded human-visible cadence", async () => {
  const clock = new FakeClock();
  let requests = 0;
  const controller = new ReplayPlaybackController({
    clock,
    getMotionMode: () => "off",
    request: async () => {
      requests += 1;
      return { frame: { cursor: cursor(requests, 4) } };
    },
  });
  controller.installCursor(cursor(0, 4));
  controller.play();
  await clock.advance(REPLAY_AUTOPLAY_CADENCE_MS.off - 1);
  assert.equal(requests, 0);
  await clock.advance(1);
  assert.equal(requests, 1);
  await clock.advance(REPLAY_AUTOPLAY_CADENCE_MS.off - 1);
  assert.equal(requests, 1);
  await clock.advance(1);
  assert.equal(requests, 2);
});

test("autoplay pauses at endpoint and never spins a tight loop", async () => {
  const clock = new FakeClock();
  let requests = 0;
  const controller = new ReplayPlaybackController({
    clock,
    request: async () => {
      requests += 1;
      return { frame: { cursor: cursor(1, 1) } };
    },
  });
  controller.installCursor(cursor(0, 1));
  controller.play();
  await clock.advance(REPLAY_AUTOPLAY_CADENCE_MS.normal * 20);
  assert.equal(requests, 1);
  assert.equal(controller.snapshot().playing, false);
  assert.equal(controller.snapshot().pauseReason, "endpoint");
  assert.equal(clock.timers.size, 0);
});

test("disconnect, hidden page, and request errors stop autoplay", async () => {
  for (const stop of ["disconnect", "hidden", "error"]) {
    const clock = new FakeClock();
    let requests = 0;
    const controller = new ReplayPlaybackController({
      clock,
      request: async () => {
        requests += 1;
        throw new Error("synthetic failure");
      },
    });
    controller.installCursor(cursor(0));
    controller.play();
    if (stop === "disconnect") {
      controller.setConnected(false);
    } else if (stop === "hidden") {
      controller.setHidden(true);
    } else {
      await clock.advance(REPLAY_AUTOPLAY_CADENCE_MS.normal);
    }
    await clock.advance(REPLAY_AUTOPLAY_CADENCE_MS.normal * 3);
    assert.equal(controller.snapshot().playing, false);
    assert.equal(controller.snapshot().pauseReason, stop);
    assert.equal(requests, stop === "error" ? 1 : 0);
  }
});

test("autoplay rejects a non-next authoritative cursor and pauses", async () => {
  const clock = new FakeClock();
  /** @type {unknown[]} */
  const errors = [];
  const controller = new ReplayPlaybackController({
    clock,
    request: async () => ({ frame: { cursor: cursor(2, 3) } }),
    onError: (error) => errors.push(error),
  });
  controller.installCursor(cursor(0));
  controller.play();
  await clock.advance(REPLAY_AUTOPLAY_CADENCE_MS.normal);
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
  controller.installCursor(cursor(1, 3, { cursor: 7, choreography: 2 }));

  assert.equal(await controller.next(), false);
  assert.deepEqual(requests, [replayNavigationCommand("next")]);
  assert.equal(controller.snapshot().cursor?.frame_index, 2);
  assert.equal(controller.snapshot().pauseReason, "resync");
  assert.equal(controller.snapshot().playing, false);
  assert.deepEqual(errors, []);
});
