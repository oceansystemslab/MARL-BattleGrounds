import assert from "node:assert/strict";
import test from "node:test";

import { CombatChoreographer, ConsumedTransitionLedger } from "../src/choreography.js";

class FakeAnimation {
  /**
   * @param {string} id
   */
  constructor(id) {
    this.id = id;
    this.currentTime = 0;
    this.playbackRate = 1;
    this.playState = "running";
    this.finished = new Promise((resolve, reject) => {
      this.resolveFinished = resolve;
      this.rejectFinished = reject;
    });
  }

  pause() {
    this.playState = "paused";
  }

  play() {
    this.playState = "running";
  }

  /**
   * @param {number} rate
   */
  updatePlaybackRate(rate) {
    this.playbackRate = rate;
  }

  finish() {
    this.playState = "finished";
    this.currentTime = 900;
    this.resolveFinished(this);
  }

  cancel() {
    this.playState = "idle";
    this.rejectFinished(new Error("cancelled"));
  }
}

class FakeAnimationFactory {
  constructor() {
    /** @type {FakeAnimation[]} */
    this.created = [];
  }

  /**
   * @param {{id: string}} spec
   */
  create(spec) {
    const animation = new FakeAnimation(spec.id);
    this.created.push(animation);
    return animation;
  }

  /**
   * @param {Element} _element
   * @param {number} _duration
   * @param {string} id
   */
  createClock(_element, _duration, id) {
    const animation = new FakeAnimation(id);
    this.created.push(animation);
    return animation;
  }

  /**
   * @param {string} suffix
   */
  find(suffix) {
    return this.created.find(({ id }) => id.endsWith(suffix));
  }
}

class FakePainter {
  constructor() {
    /** @type {any[][]} */
    this.calls = [];
  }

  /**
   * @param {Record<string, any>} plan
   * @param {Record<string, any>} surface
   * @param {{
   *   settled: boolean,
   *   persistentOnly: boolean,
   *   motionMode: "normal" | "reduced" | "off",
   * }} options
   */
  install(plan, surface, options) {
    this.calls.push(["install", plan.epochKey, surface.viewportKey, options]);
    return {
      root: /** @type {any} */ ({}),
      nodeCount: options.persistentOnly ? 2 : 6,
      persistentNodeCount: plan.persistentNodeCount ?? 2,
      animationSpecs:
        options.settled ||
        options.persistentOnly ||
        options.motionMode === "off" ||
        plan.animationSpecCount === 0
          ? []
          : [
              {
                element: /** @type {any} */ ({}),
                keyframes: [],
                options: {},
                id: `mbg:${plan.epochKey}:effect-0`,
              },
              {
                element: /** @type {any} */ ({}),
                keyframes: [],
                options: {},
                id: `mbg:${plan.epochKey}:effect-1`,
              },
            ],
    };
  }

  /**
   * @param {unknown} _installation
   * @param {string} reason
   */
  clear(_installation, reason) {
    this.calls.push(["clear", reason]);
  }

  /**
   * @param {unknown} _installation
   */
  settle(_installation) {
    this.calls.push(["settle"]);
  }

  /**
   * @param {unknown} _installation
   * @param {Record<string, any>} plan
   * @param {Record<string, any>} surface
   */
  reproject(_installation, plan, surface) {
    this.calls.push(["reproject", plan.epochKey, surface.viewportKey]);
  }
}

class FakeStorage {
  constructor() {
    this.values = new Map();
  }

  /**
   * @param {string} key
   */
  getItem(key) {
    return this.values.get(key) ?? null;
  }

  /**
   * @param {string} key
   * @param {string} value
   */
  setItem(key, value) {
    this.values.set(key, value);
  }
}

/**
 * @param {string} epoch
 * @param {string} [authorization]
 * @param {string} [fingerprint]
 * @returns {Record<string, any>}
 */
function plan(epoch, authorization = "researcher", fingerprint = `events-${epoch}`) {
  return {
    epochKey: epoch,
    authorizationKey: authorization,
    fingerprint,
    phases: {
      submissionRelease: 450,
      reducedTotal: 220,
      total: 900,
    },
    bounds: {
      nodes: 16,
      animations: 8,
      persistentNodes: 2,
    },
    events: [],
  };
}

/**
 * @param {{
 *   ledger?: ConsumedTransitionLedger,
 *   storage?: FakeStorage | null,
 *   motionMode?: "normal" | "reduced" | "off",
 *   playbackRate?: number,
 * }} [options]
 */
function harness(options = {}) {
  const painter = new FakePainter();
  const animationFactory = new FakeAnimationFactory();
  const ledger =
    options.ledger ??
    new ConsumedTransitionLedger({
      storage: options.storage ?? null,
      storageKey: "test-ledger",
      limit: 4,
    });
  const controller = new CombatChoreographer({
    painter,
    animationFactory,
    ledger,
    motionMode: options.motionMode ?? "normal",
    playbackRate: options.playbackRate,
    planBuilder: (frame) =>
      /** @type {Record<string, any> | null | undefined} */ (
        /** @type {any} */ (frame)?.plan
      ) ?? null,
  });
  return { animationFactory, controller, ledger, painter };
}

const surfaceA = /** @type {any} */ ({ viewportKey: "16:12:800:600" });
const surfaceSameSizeProtected = /** @type {any} */ ({
  viewportKey: "16:12:800:600:protected-layout-b",
});
const surfaceB = /** @type {any} */ ({ viewportKey: "16:12:600:450" });

test("same transition renders once and resize reprojects without replay", async () => {
  const { animationFactory, controller, painter } = harness();
  controller.presentFrame({ plan: plan("epoch-1") }, surfaceA);
  assert.equal(controller.snapshot().submissionBlocked, true);
  assert.equal(painter.calls.filter(([kind]) => kind === "install").length, 1);
  const createdCount = animationFactory.created.length;

  controller.presentFrame({ plan: plan("epoch-1") }, surfaceA);
  assert.equal(animationFactory.created.length, createdCount);
  assert.equal(painter.calls.filter(([kind]) => kind === "install").length, 1);

  animationFactory.created.forEach((animation, index) => {
    animation.currentTime = 230 + index;
  });
  const animationReferences = [...animationFactory.created];
  const animationTimes = animationFactory.created.map(({ currentTime }) => currentTime);
  controller.presentFrame({ plan: plan("epoch-1") }, surfaceSameSizeProtected);
  assert.deepEqual(painter.calls.at(-1), [
    "reproject",
    "epoch-1",
    surfaceSameSizeProtected.viewportKey,
  ]);
  assert.equal(painter.calls.filter(([kind]) => kind === "install").length, 1);
  assert.deepEqual(animationFactory.created, animationReferences);
  assert.deepEqual(
    animationFactory.created.map(({ currentTime }) => currentTime),
    animationTimes,
  );

  controller.reproject({ plan: plan("epoch-1") }, surfaceB);
  assert.deepEqual(painter.calls.at(-1), [
    "reproject",
    "epoch-1",
    surfaceB.viewportKey,
  ]);
  assert.equal(animationFactory.created.length, createdCount);
  assert.deepEqual(animationFactory.created, animationReferences);
  assert.deepEqual(
    animationFactory.created.map(({ currentTime }) => currentTime),
    animationTimes,
  );

  const gate = animationFactory.find(":gate");
  assert.ok(gate);
  gate.finish();
  await Promise.resolve();
  assert.equal(controller.snapshot().submissionBlocked, false);
});

test("reprojection clears before rebuilding changed event disclosure", () => {
  const { controller, painter } = harness();
  controller.presentFrame(
    { plan: plan("epoch-1", "researcher", "public-events") },
    surfaceA,
  );
  const callCount = painter.calls.length;

  controller.reproject(
    { plan: plan("epoch-1", "researcher", "redacted-events") },
    surfaceB,
  );

  assert.deepEqual(
    painter.calls.slice(callCount).map(([kind]) => kind),
    ["clear", "install", "settle"],
  );
  assert.equal(controller.snapshot().fingerprint, "redacted-events");
});

test("authorization change clears before settled safe rebuild without replay", () => {
  const { animationFactory, controller, painter } = harness();
  controller.presentFrame({ plan: plan("epoch-1", "researcher") }, surfaceA);
  const createdCount = animationFactory.created.length;
  const callCount = painter.calls.length;

  controller.presentFrame({ plan: plan("epoch-1", "pov-0", "pov-events") }, surfaceA);
  assert.deepEqual(
    painter.calls.slice(callCount).map(([kind]) => kind),
    ["clear", "install", "settle"],
  );
  assert.equal(animationFactory.created.length, createdCount);
  assert.equal(controller.snapshot().authorizationKey, "pov-0");
  assert.equal(controller.snapshot().submissionBlocked, false);
});

test("new transitions replace once and absent batches clear", () => {
  const { controller, painter } = harness();
  controller.presentFrame({ plan: plan("epoch-1") }, surfaceA);
  controller.presentFrame({ plan: plan("epoch-2") }, surfaceA);
  assert.equal(painter.calls.filter(([kind]) => kind === "install").length, 2);
  assert.ok(
    painter.calls.some(
      ([kind, reason]) => kind === "clear" && reason === "new_transition",
    ),
  );
  controller.presentFrame({ plan: null }, surfaceA);
  assert.equal(controller.snapshot().active, false);
  assert.equal(controller.snapshot().epochKey, null);
});

test("a plan without spatial animation does not create clocks or gate submission", () => {
  const { animationFactory, controller, painter } = harness();
  const quietPlan = plan("epoch-quiet");
  quietPlan.animationSpecCount = 0;

  controller.presentFrame({ plan: quietPlan }, surfaceA);

  assert.equal(animationFactory.created.length, 0);
  assert.equal(controller.snapshot().submissionBlocked, false);
  assert.ok(painter.calls.some(([kind]) => kind === "settle"));
});

test("persistent node bounds fail closed before animation starts", () => {
  const { animationFactory, controller, painter } = harness();
  const oversized = plan("epoch-oversized");
  oversized.persistentNodeCount = 3;
  oversized.bounds.persistentNodes = 2;

  assert.throws(
    () => controller.presentFrame({ plan: oversized }, surfaceA),
    /persistent node bound/,
  );
  assert.equal(animationFactory.created.length, 0);
  assert.ok(
    painter.calls.some(
      ([kind, reason]) =>
        kind === "clear" && reason === "persistent_node_bound_exceeded",
    ),
  );
});

test("partial animation creation is cancelled and its DOM is removed", () => {
  const painter = new FakePainter();
  const factory = new FakeAnimationFactory();
  const create = factory.create.bind(factory);
  let createCount = 0;
  factory.create = (spec) => {
    createCount += 1;
    if (createCount === 2) {
      throw new Error("synthetic animation failure");
    }
    return create(spec);
  };
  const controller = new CombatChoreographer({
    painter,
    animationFactory: factory,
    planBuilder: (frame) =>
      /** @type {Record<string, any> | null | undefined} */ (
        /** @type {any} */ (frame)?.plan
      ) ?? null,
  });

  assert.throws(
    () => controller.presentFrame({ plan: plan("epoch-failure") }, surfaceA),
    /synthetic animation failure/,
  );
  assert.equal(factory.created[0]?.playState, "idle");
  assert.equal(controller.snapshot().active, false);
  assert.ok(
    painter.calls.some(
      ([kind, reason]) => kind === "clear" && reason === "animation_creation_failed",
    ),
  );
});

test("same-tab restoration installs only settled persistent cues", () => {
  const storage = new FakeStorage();
  const first = harness({ storage });
  first.controller.presentFrame({ plan: plan("epoch-1") }, surfaceA);

  const restored = harness({ storage });
  restored.controller.presentFrame({ plan: plan("epoch-1") }, surfaceA);
  const install = restored.painter.calls.find(([kind]) => kind === "install");
  assert.ok(install);
  assert.equal(install[3].settled, true);
  assert.equal(install[3].persistentOnly, true);
  assert.equal(restored.animationFactory.created.length, 0);
  assert.equal(restored.controller.snapshot().submissionBlocked, false);
});

test("replay animates only response-authorized incoming frames and exposes settle completion", async () => {
  const settled = harness();
  settled.controller.presentFrame(
    {
      viewer_mode: "replay",
      plan: plan("replay-direct-get"),
    },
    surfaceA,
    { animateIncoming: false },
  );
  const settledInstall = settled.painter.calls.find(([kind]) => kind === "install");
  assert.ok(settledInstall);
  assert.equal(settledInstall[3].settled, true);
  assert.equal(settled.controller.snapshot().animationCount, 0);
  await settled.controller.whenSettled();

  const animated = harness();
  animated.controller.presentFrame(
    {
      viewer_mode: "replay",
      plan: plan("replay-exact-next"),
    },
    surfaceA,
    { animateIncoming: true },
  );
  assert.ok(animated.controller.snapshot().animationCount > 0);
  let didSettle = false;
  const presentation = animated.controller.whenSettled().then(() => {
    didSettle = true;
  });
  await Promise.resolve();
  assert.equal(didSettle, false);
  animated.animationFactory.find(":cleanup")?.finish();
  await presentation;
  assert.equal(didSettle, true);
  assert.equal(animated.controller.snapshot().animationCount, 0);

  const scientificInputCannotAuthorize = harness();
  scientificInputCannotAuthorize.controller.presentFrame(
    {
      viewer_mode: "replay",
      animate_incoming: true,
      plan: plan("replay-scientific-input-cannot-authorize"),
    },
    surfaceA,
  );
  assert.equal(scientificInputCannotAuthorize.controller.snapshot().animationCount, 0);
});

test("a direct replay refresh settles an in-flight presentation of the same epoch", () => {
  const { controller } = harness();
  const replayPlan = plan("replay-refresh");
  controller.presentFrame({ viewer_mode: "replay", plan: replayPlan }, surfaceA, {
    animateIncoming: true,
  });
  assert.ok(controller.snapshot().animationCount > 0);

  controller.presentFrame({ viewer_mode: "replay", plan: replayPlan }, surfaceA, {
    animateIncoming: false,
  });

  assert.equal(controller.snapshot().animationCount, 0);
  assert.equal(controller.snapshot().submissionBlocked, false);
});

test("reproject fallbacks preserve explicit replay intent and otherwise fail settled", () => {
  const animated = harness();
  animated.controller.reproject(
    { viewer_mode: "replay", plan: plan("replay-reproject-authorized") },
    surfaceA,
    { animateIncoming: true },
  );
  assert.ok(animated.controller.snapshot().animationCount > 0);

  const settled = harness();
  settled.controller.reproject(
    {
      viewer_mode: "replay",
      animate_incoming: true,
      plan: plan("replay-reproject-default"),
    },
    surfaceA,
  );
  assert.equal(settled.controller.snapshot().animationCount, 0);
  assert.equal(settled.controller.snapshot().submissionBlocked, false);
});

test("pause, fixed-rate compatibility, Skip, reduced motion, and Off remain presentation-only", async () => {
  const idle = harness();
  assert.equal(idle.controller.togglePaused().paused, false);

  const normal = harness();
  normal.controller.presentFrame({ plan: plan("epoch-1") }, surfaceA);
  normal.controller.togglePaused();
  assert.equal(
    normal.animationFactory.created.every(({ playState }) => playState === "paused"),
    true,
  );
  normal.controller.setPlaybackRate(2);
  assert.equal(
    normal.animationFactory.created.every(({ playbackRate }) => playbackRate === 1),
    true,
  );
  assert.equal(normal.controller.snapshot().playbackRate, 1);
  normal.controller.togglePaused();
  normal.controller.skip();
  assert.equal(normal.controller.snapshot().submissionBlocked, false);
  assert.ok(normal.painter.calls.some(([kind]) => kind === "settle"));
  assert.equal(normal.controller.snapshot().animationCount, 0);

  const reduced = harness({ motionMode: "reduced" });
  reduced.controller.presentFrame({ plan: plan("epoch-reduced") }, surfaceA);
  assert.equal(reduced.controller.snapshot().submissionBlocked, false);
  assert.equal(reduced.animationFactory.find(":gate"), undefined);

  const off = harness({ motionMode: "off" });
  off.controller.presentFrame({ plan: plan("epoch-off") }, surfaceA);
  assert.equal(off.controller.snapshot().submissionBlocked, false);
  assert.equal(off.animationFactory.find(":gate"), undefined);
  assert.ok(off.animationFactory.find(":cleanup"));
  assert.equal(off.controller.togglePaused().paused, false);
  off.animationFactory.find(":cleanup")?.finish();
  await Promise.resolve();
  assert.ok(off.painter.calls.some(([kind]) => kind === "settle"));
  assert.equal(off.controller.snapshot().animationCount, 0);
});

test("legacy graphics-speed requests are canonical fixed-rate no-ops", () => {
  for (const legacy of [0.01, 0.5, 2, 0, 2.001, Number.NaN, Number.POSITIVE_INFINITY]) {
    const fixed = harness();
    assert.doesNotThrow(() => fixed.controller.setPlaybackRate(legacy));
    assert.equal(fixed.controller.snapshot().playbackRate, 1);
  }
});

test("legacy constructor playback rate is canonicalized to 1.0", () => {
  const fixed = harness({ playbackRate: 0.25 });
  fixed.controller.presentFrame({ plan: plan("fixed-constructor-rate") }, surfaceA);
  assert.equal(fixed.controller.snapshot().playbackRate, 1);
  assert.equal(
    fixed.animationFactory.created.every(({ playbackRate }) => playbackRate === 1),
    true,
  );
});

test("switching an active explanation Off reinstalls the authorized batch until bounded cleanup", async () => {
  const { animationFactory, controller, painter } = harness();
  const authorizedPlan = plan("epoch-mid-off", "researcher", "authorized-events");
  controller.presentFrame({ plan: authorizedPlan }, surfaceA);
  const oldAnimations = [...animationFactory.created];
  const callCount = painter.calls.length;

  controller.setMotionMode("off");

  assert.deepEqual(painter.calls.slice(callCount), [
    ["clear", "motion_disabled_static_reinstall"],
    [
      "install",
      "epoch-mid-off",
      surfaceA.viewportKey,
      {
        motionMode: "off",
        settled: false,
        persistentOnly: false,
      },
    ],
  ]);
  assert.equal(
    oldAnimations.every(({ playState }) => playState === "idle"),
    true,
  );
  assert.deepEqual(controller.snapshot(), {
    active: true,
    animationCount: 1,
    epochKey: "epoch-mid-off",
    authorizationKey: "researcher",
    fingerprint: "authorized-events",
    logicalTime: 0,
    motionMode: "off",
    paused: false,
    playbackRate: 1,
    submissionBlocked: false,
  });
  const offAnimations = animationFactory.created.slice(oldAnimations.length);
  assert.equal(offAnimations.length, 1);
  assert.match(offAnimations[0].id, /:cleanup$/);

  const installCount = painter.calls.filter(([kind]) => kind === "install").length;
  const createdCount = animationFactory.created.length;
  controller.setMotionMode("normal");
  controller.setPlaybackRate(0.5);
  assert.equal(controller.snapshot().motionMode, "normal");
  assert.equal(controller.snapshot().playbackRate, 1);
  assert.equal(
    painter.calls.filter(([kind]) => kind === "install").length,
    installCount,
  );
  assert.equal(animationFactory.created.length, createdCount);

  offAnimations[0].finish();
  await Promise.resolve();
  assert.ok(painter.calls.some(([kind]) => kind === "settle"));
  assert.equal(controller.snapshot().animationCount, 0);
});

test("legacy rate calls preserve an active reduced-motion explanation at 1.0", () => {
  const { animationFactory, controller, painter } = harness({
    motionMode: "reduced",
  });
  controller.presentFrame({ plan: plan("epoch-reduced-rate") }, surfaceA);
  const installCount = painter.calls.filter(([kind]) => kind === "install").length;
  const createdCount = animationFactory.created.length;

  controller.setPlaybackRate(2);

  assert.equal(controller.snapshot().motionMode, "reduced");
  assert.equal(controller.snapshot().playbackRate, 1);
  assert.equal(
    animationFactory.created.every(({ playbackRate }) => playbackRate === 1),
    true,
  );
  assert.equal(
    painter.calls.filter(([kind]) => kind === "install").length,
    installCount,
  );
  assert.equal(animationFactory.created.length, createdCount);
});

test("ledger is bounded and storage failures never affect authority", () => {
  const productionBound = new ConsumedTransitionLedger();
  for (let index = 0; index < 257; index += 1) {
    productionBound.record(`epoch-${index}`, `fingerprint-${index}`);
  }
  assert.equal(productionBound.has("epoch-0"), false);
  for (let index = 1; index < 257; index += 1) {
    assert.equal(productionBound.has(`epoch-${index}`), true);
  }

  const storage = new FakeStorage();
  const ledger = new ConsumedTransitionLedger({
    storage,
    storageKey: "bounded",
    limit: 2,
  });
  ledger.record("one", "1");
  ledger.record("two", "2");
  ledger.record("three", "3");
  assert.equal(ledger.has("one"), false);
  assert.equal(ledger.has("two"), true);
  assert.equal(ledger.has("three"), true);
  assert.equal(ledger.fingerprintFor("three"), "3");
  assert.equal(ledger.fingerprintFor("missing"), null);

  const brokenStorage = {
    getItem() {
      throw new Error("unavailable");
    },
    setItem() {
      throw new Error("unavailable");
    },
  };
  const fallback = new ConsumedTransitionLedger({
    storage: brokenStorage,
    limit: 2,
  });
  fallback.record("safe", "fingerprint");
  assert.equal(fallback.has("safe"), true);
});
