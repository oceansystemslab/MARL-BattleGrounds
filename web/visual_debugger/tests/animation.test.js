import assert from "node:assert/strict";
import test from "node:test";

import { CombatChoreographer, ConsumedTransitionLedger } from "../src/choreography.js";

class FakeAnimation {
  /**
   * @param {string} id
   * @param {KeyframeAnimationOptions} [timing]
   */
  constructor(id, timing = {}) {
    this.id = id;
    this.currentTime = 0;
    this.playbackRate = 1;
    this.playState = "running";
    this.delay = Number(timing.delay ?? 0);
    this.duration = Number(timing.duration ?? 0);
    this.endDelay = Number(timing.endDelay ?? 0);
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
    this.currentTime = this.delay + this.duration + this.endDelay;
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
   * @param {{id: string, options: KeyframeAnimationOptions}} spec
   */
  create(spec) {
    const animation = new FakeAnimation(spec.id, spec.options);
    this.created.push(animation);
    return animation;
  }

  /**
   * @param {Element} _element
   * @param {number} duration
   * @param {string} id
   */
  createClock(_element, duration, id) {
    const animation = new FakeAnimation(id, { duration });
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
   *   retainTransientOnSettle?: boolean,
   *   motionMode: "normal" | "reduced" | "off",
   *   renderPolicy: "live_once" | "replay_animated" | "replay_static",
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
 * @param {string} [paintKey]
 * @returns {Record<string, any>}
 */
function plan(
  epoch,
  authorization = "researcher",
  fingerprint = `events-${epoch}`,
  paintKey = "visual-filters-v2:all-on",
) {
  return {
    epochKey: epoch,
    authorizationKey: authorization,
    fingerprint,
    paintKey,
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

test("presentation controls forward the same visual-filter state to planning", () => {
  const painter = new FakePainter();
  const visualFilters = Object.freeze({ synthetic_filter: false });
  /** @type {Array<[
   *   string | undefined,
   *   Record<string, boolean> | undefined,
   *   string | undefined,
   * ]>} */
  const received = [];
  const controller = new CombatChoreographer({
    painter,
    animationFactory: new FakeAnimationFactory(),
    planBuilder(frame, surface, filters, renderPolicy) {
      received.push([surface?.viewportKey, filters, renderPolicy]);
      return /** @type {any} */ (frame).plan;
    },
  });
  const current = plan("epoch-filter-forwarding");

  controller.presentFrame({ plan: current }, surfaceA, {
    renderPolicy: "live_once",
    visualFilters,
  });
  controller.reproject({ plan: current }, surfaceB, {
    renderPolicy: "live_once",
    visualFilters,
  });

  assert.deepEqual(received, [
    [surfaceA.viewportKey, visualFilters, "live_once"],
    [surfaceB.viewportKey, visualFilters, "live_once"],
  ]);
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

test("live paint-only changes settle locally without replay or ledger mutation", () => {
  const { animationFactory, controller, ledger, painter } = harness();
  const originalRecord = ledger.record.bind(ledger);
  let recordCount = 0;
  ledger.record = (epochKey, fingerprint) => {
    recordCount += 1;
    originalRecord(epochKey, fingerprint);
  };
  const allOn = plan(
    "epoch-paint-live",
    "researcher",
    "scientific-events",
    "paint-all-on",
  );
  const filtered = plan(
    "epoch-paint-live",
    "researcher",
    "scientific-events",
    "paint-filtered",
  );
  controller.presentFrame({ plan: allOn }, surfaceA);
  const createdCount = animationFactory.created.length;
  const recordedCount = recordCount;

  for (const nextPlan of [filtered, allOn]) {
    const callCount = painter.calls.length;
    controller.presentFrame({ plan: nextPlan }, surfaceA);
    assert.deepEqual(
      painter.calls.slice(callCount).map(([kind]) => kind),
      ["clear", "install", "settle"],
    );
    const install = painter.calls.at(-2);
    assert.deepEqual(install?.[3], {
      motionMode: "normal",
      renderPolicy: "live_once",
      settled: true,
      persistentOnly: true,
    });
    assert.equal(animationFactory.created.length, createdCount);
    assert.equal(recordCount, recordedCount);
    assert.equal(ledger.fingerprintFor(allOn.epochKey), allOn.fingerprint);
    assert.equal(controller.snapshot().paintKey, nextPlan.paintKey);
    assert.equal(controller.snapshot().animationCount, 0);
    assert.equal(controller.snapshot().submissionBlocked, false);
  }
});

test("replay paint-only changes pause and reinstall the current static summary", () => {
  const { animationFactory, controller, ledger, painter } = harness();
  const originalRecord = ledger.record.bind(ledger);
  let recordCount = 0;
  ledger.record = (epochKey, fingerprint) => {
    recordCount += 1;
    originalRecord(epochKey, fingerprint);
  };
  const allOn = plan(
    "epoch-paint-replay",
    "researcher",
    "scientific-events",
    "paint-all-on",
  );
  const filtered = plan(
    "epoch-paint-replay",
    "researcher",
    "scientific-events",
    "paint-filtered",
  );
  controller.presentFrame({ viewer_mode: "replay", plan: allOn }, surfaceA, {
    renderPolicy: "replay_animated",
  });
  const createdCount = animationFactory.created.length;
  const recordedCount = recordCount;

  for (const nextPlan of [filtered, allOn]) {
    const callCount = painter.calls.length;
    controller.presentFrame({ viewer_mode: "replay", plan: nextPlan }, surfaceA, {
      renderPolicy: "replay_animated",
    });
    assert.deepEqual(
      painter.calls.slice(callCount).map(([kind]) => kind),
      ["clear", "install", "settle"],
    );
    const install = painter.calls.at(-2);
    assert.deepEqual(install?.[3], {
      motionMode: "normal",
      renderPolicy: "replay_static",
      settled: true,
      persistentOnly: false,
      retainTransientOnSettle: true,
    });
    assert.equal(animationFactory.created.length, createdCount);
    assert.equal(recordCount, recordedCount);
    assert.equal(ledger.fingerprintFor(allOn.epochKey), null);
    assert.equal(controller.snapshot().paintKey, nextPlan.paintKey);
    assert.equal(controller.snapshot().paused, true);
    assert.equal(controller.snapshot().animationCount, 0);
    assert.equal(controller.snapshot().submissionBlocked, false);
  }
  controller.presentFrame(
    { viewer_mode: "replay", plan: plan("epoch-paint-replay-next") },
    surfaceA,
    { renderPolicy: "replay_animated" },
  );
  assert.equal(controller.snapshot().renderPolicy, "replay_animated");
  assert.equal(controller.snapshot().paused, false);
  assert.ok(controller.snapshot().animationCount > 0);
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
    { renderPolicy: "replay_static" },
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
    { renderPolicy: "replay_animated" },
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
    { renderPolicy: "replay_static" },
  );
  assert.equal(scientificInputCannotAuthorize.controller.snapshot().animationCount, 0);
});

test("explicit replay policies bypass consumption and static uses zero clocks", () => {
  let lookupCount = 0;
  let recordCount = 0;
  const forbiddenReplayLedger = /** @type {any} */ ({
    fingerprintFor() {
      lookupCount += 1;
      throw new Error("replay consulted the consumed ledger");
    },
    record() {
      recordCount += 1;
      throw new Error("replay mutated the consumed ledger");
    },
  });

  const animated = harness({ ledger: forbiddenReplayLedger });
  animated.controller.presentFrame({ plan: plan("policy-replay-animated") }, surfaceA, {
    renderPolicy: "replay_animated",
  });
  assert.equal(animated.controller.snapshot().renderPolicy, "replay_animated");
  assert.ok(animated.controller.snapshot().animationCount > 0);
  assert.deepEqual(animated.painter.calls[0]?.[3], {
    motionMode: "normal",
    renderPolicy: "replay_animated",
    settled: false,
    persistentOnly: false,
  });

  const staticSummary = harness({ ledger: forbiddenReplayLedger });
  staticSummary.controller.presentFrame(
    { plan: plan("policy-replay-static") },
    surfaceA,
    { renderPolicy: "replay_static" },
  );
  const firstInstallCount = staticSummary.painter.calls.filter(
    ([kind]) => kind === "install",
  ).length;
  staticSummary.controller.presentFrame(
    { plan: plan("policy-replay-static") },
    surfaceA,
    { renderPolicy: "replay_static" },
  );
  assert.deepEqual(staticSummary.painter.calls[0]?.[3], {
    motionMode: "normal",
    renderPolicy: "replay_static",
    settled: true,
    persistentOnly: false,
    retainTransientOnSettle: true,
  });
  assert.equal(staticSummary.animationFactory.created.length, 0);
  assert.equal(
    staticSummary.painter.calls.filter(([kind]) => kind === "install").length,
    firstInstallCount,
  );
  assert.equal(staticSummary.controller.snapshot().logicalTime, 0);
  assert.equal(staticSummary.controller.snapshot().animationCount, 0);
  assert.equal(staticSummary.controller.snapshot().renderPolicy, "replay_static");
  assert.equal(staticSummary.controller.snapshot().paused, true);
  assert.equal(lookupCount, 0);
  assert.equal(recordCount, 0);
});

test("a direct replay refresh settles an in-flight presentation of the same epoch", () => {
  const { controller } = harness();
  const replayPlan = plan("replay-refresh");
  controller.presentFrame({ viewer_mode: "replay", plan: replayPlan }, surfaceA, {
    renderPolicy: "replay_animated",
  });
  assert.ok(controller.snapshot().animationCount > 0);

  controller.presentFrame({ viewer_mode: "replay", plan: replayPlan }, surfaceA, {
    renderPolicy: "replay_static",
  });

  assert.equal(controller.snapshot().animationCount, 0);
  assert.equal(controller.snapshot().submissionBlocked, false);
});

test("an explicit same-identity replay restart bypasses sticky static exactly at the presentation seam", async () => {
  let lookupCount = 0;
  let recordCount = 0;
  const forbiddenReplayLedger = /** @type {any} */ ({
    fingerprintFor() {
      lookupCount += 1;
      throw new Error("replay consulted the consumed ledger");
    },
    record() {
      recordCount += 1;
      throw new Error("replay mutated the consumed ledger");
    },
  });
  const { animationFactory, controller, painter } = harness({
    ledger: forbiddenReplayLedger,
  });
  const replayPlan = plan("replay-current-restart");

  controller.presentFrame({ viewer_mode: "replay", plan: replayPlan }, surfaceA, {
    renderPolicy: "replay_static",
  });
  controller.presentFrame({ viewer_mode: "replay", plan: replayPlan }, surfaceA, {
    renderPolicy: "replay_animated",
  });
  assert.equal(controller.snapshot().renderPolicy, "replay_static");
  assert.equal(controller.snapshot().animationCount, 0);

  controller.presentFrame({ viewer_mode: "replay", plan: replayPlan }, surfaceA, {
    renderPolicy: "replay_animated",
    restartAnimated: true,
  });
  assert.equal(controller.snapshot().renderPolicy, "replay_animated");
  assert.ok(controller.snapshot().animationCount > 0);
  const createdCount = animationFactory.created.length;
  const installCount = painter.calls.filter(([kind]) => kind === "install").length;

  controller.presentFrame({ viewer_mode: "replay", plan: replayPlan }, surfaceA, {
    renderPolicy: "replay_animated",
  });
  assert.equal(animationFactory.created.length, createdCount);
  assert.equal(
    painter.calls.filter(([kind]) => kind === "install").length,
    installCount,
  );

  const obsoleteAnimations = [...animationFactory.created];
  controller.presentFrame({ viewer_mode: "replay", plan: replayPlan }, surfaceA, {
    renderPolicy: "replay_static",
  });
  assert.equal(controller.snapshot().renderPolicy, "replay_static");
  assert.equal(controller.snapshot().animationCount, 0);
  assert.equal(
    obsoleteAnimations.every(({ playState }) => playState === "idle"),
    true,
  );
  obsoleteAnimations.find(({ id }) => id.endsWith(":cleanup"))?.finish();
  await Promise.resolve();
  assert.equal(controller.snapshot().renderPolicy, "replay_static");
  assert.equal(controller.snapshot().animationCount, 0);
  assert.equal(lookupCount, 0);
  assert.equal(recordCount, 0);
});

test("replay restart intent cannot cross paint, disclosure, or authorization identity", () => {
  const original = plan(
    "replay-restart-fence",
    "researcher",
    "full-events",
    "paint-all",
  );
  const changedPlans = [
    plan("replay-restart-fence", "researcher", "full-events", "paint-filtered"),
    plan("replay-restart-fence", "researcher", "redacted-events", "paint-all"),
    plan("replay-restart-fence", "pov-0", "pov-events", "paint-all"),
  ];

  for (const changed of changedPlans) {
    const { animationFactory, controller } = harness();
    controller.presentFrame({ viewer_mode: "replay", plan: original }, surfaceA, {
      renderPolicy: "replay_static",
    });
    controller.presentFrame({ viewer_mode: "replay", plan: changed }, surfaceA, {
      renderPolicy: "replay_animated",
      restartAnimated: true,
    });
    assert.equal(controller.snapshot().renderPolicy, "replay_static");
    assert.equal(controller.snapshot().animationCount, 0);
    assert.equal(animationFactory.created.length, 0);
  }
});

test("reproject fallbacks preserve explicit replay intent and otherwise fail settled", () => {
  const animated = harness();
  animated.controller.reproject(
    { viewer_mode: "replay", plan: plan("replay-reproject-authorized") },
    surfaceA,
    { renderPolicy: "replay_animated" },
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
    { renderPolicy: "replay_static" },
  );
  assert.equal(settled.controller.snapshot().animationCount, 0);
  assert.equal(settled.controller.snapshot().submissionBlocked, false);
});

test("pause, whole-clock rate changes, Skip, reduced motion, and Off remain presentation-only", async () => {
  const idle = harness();
  assert.equal(idle.controller.togglePaused().paused, false);

  const normal = harness();
  normal.controller.presentFrame({ plan: plan("epoch-1") }, surfaceA);
  normal.controller.togglePaused();
  const animationReferences = [...normal.animationFactory.created];
  normal.animationFactory.created.forEach((animation, index) => {
    animation.currentTime = 120 + index;
  });
  const animationTimes = normal.animationFactory.created.map(
    ({ currentTime }) => currentTime,
  );
  assert.equal(
    normal.animationFactory.created.every(({ playState }) => playState === "paused"),
    true,
  );
  normal.controller.setPlaybackRate(2);
  assert.equal(
    normal.animationFactory.created.every(({ playbackRate }) => playbackRate === 2),
    true,
  );
  assert.deepEqual(normal.animationFactory.created, animationReferences);
  assert.deepEqual(
    normal.animationFactory.created.map(({ currentTime }) => currentTime),
    animationTimes,
  );
  assert.equal(normal.controller.snapshot().playbackRate, 2);
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

test("all eight rates scale current and future effect, gate, cleanup, and hold clocks", () => {
  for (const rate of [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]) {
    const current = harness();
    current.controller.presentFrame(
      { viewer_mode: "replay", plan: plan(`current-rate-${rate}`) },
      surfaceA,
      { renderPolicy: "replay_animated" },
    );
    const currentAnimations = [...current.animationFactory.created];
    current.controller.setPlaybackRate(rate);
    assert.equal(current.controller.snapshot().playbackRate, rate);
    assert.deepEqual(current.animationFactory.created, currentAnimations);
    assert.equal(
      current.animationFactory.created.every(
        ({ playbackRate }) => playbackRate === rate,
      ),
      true,
    );
    assert.equal(current.animationFactory.find(":gate")?.playbackRate, rate);
    assert.equal(current.animationFactory.find(":cleanup")?.playbackRate, rate);
    assert.equal(current.animationFactory.find(":cleanup")?.duration, 1_500);
    assert.equal(
      Number(current.animationFactory.find(":cleanup")?.duration) / rate,
      1_500 / rate,
    );

    const future = harness({ playbackRate: rate });
    future.controller.presentFrame(
      { viewer_mode: "replay", plan: plan(`future-rate-${rate}`) },
      surfaceA,
      { renderPolicy: "replay_animated" },
    );
    assert.equal(future.controller.snapshot().playbackRate, rate);
    assert.equal(
      future.animationFactory.created.every(
        ({ playbackRate }) => playbackRate === rate,
      ),
      true,
    );
  }
});

test("replay terminal hold is clock-owned and live timing is unchanged", () => {
  const ordinaryPlan = plan("ordinary-replay-hold");
  ordinaryPlan.phases.total = 1_200;
  const replay = harness();
  replay.controller.presentFrame(
    { viewer_mode: "replay", plan: ordinaryPlan },
    surfaceA,
    { renderPolicy: "replay_animated" },
  );
  assert.equal(replay.animationFactory.find(":cleanup")?.duration, 1_800);
  assert.equal(replay.animationFactory.find(":gate")?.duration, 450);
  const replayCleanup = replay.animationFactory.find(":cleanup");
  assert.ok(replayCleanup);
  replayCleanup.currentTime = 1_650;
  replay.controller.setPlaybackRate(1.5);
  assert.equal(replay.controller.snapshot().logicalTime, 1_200);
  assert.equal(replayCleanup.currentTime, 1_650);
  assert.equal(replayCleanup.playbackRate, 1.5);

  const live = harness();
  live.controller.presentFrame({ plan: ordinaryPlan }, surfaceA);
  assert.equal(live.animationFactory.find(":cleanup")?.duration, 1_200);

  const reduced = harness({ motionMode: "reduced" });
  reduced.controller.presentFrame(
    { viewer_mode: "replay", plan: ordinaryPlan },
    surfaceA,
    { renderPolicy: "replay_animated" },
  );
  assert.equal(reduced.animationFactory.find(":cleanup")?.duration, 820);
  assert.equal(reduced.animationFactory.find(":gate"), undefined);

  const densePlan = plan("dense-replay-hold");
  densePlan.phases.total = 2_400;
  const dense = harness();
  dense.controller.presentFrame({ viewer_mode: "replay", plan: densePlan }, surfaceA, {
    renderPolicy: "replay_animated",
  });
  assert.equal(dense.animationFactory.find(":cleanup")?.duration, 3_000);
});

test("eventless animated replay retains one rate-aware terminal-hold clock", async () => {
  const emptyPlan = plan("empty-replay-family");
  emptyPlan.animationSpecCount = 0;
  const empty = harness();
  empty.controller.presentFrame({ viewer_mode: "replay", plan: emptyPlan }, surfaceA, {
    renderPolicy: "replay_animated",
  });

  const cleanup = empty.animationFactory.find(":cleanup");
  assert.ok(cleanup);
  assert.equal(empty.animationFactory.created.length, 1);
  assert.equal(empty.animationFactory.find(":gate"), undefined);
  assert.equal(cleanup.duration, 1_500);
  assert.equal(empty.controller.snapshot().animationCount, 1);
  assert.equal(empty.controller.snapshot().submissionBlocked, false);
  assert.equal(empty.painter.calls.filter(([kind]) => kind === "settle").length, 0);

  let didSettle = false;
  const presentation = empty.controller.whenSettled().then(() => {
    didSettle = true;
  });
  await Promise.resolve();
  assert.equal(didSettle, false);

  cleanup.currentTime = 1_350;
  empty.controller.setPlaybackRate(1.5);
  assert.equal(cleanup.currentTime, 1_350);
  assert.equal(cleanup.playbackRate, 1.5);
  assert.equal(empty.controller.snapshot().logicalTime, 900);
  assert.equal(empty.animationFactory.created.length, 1);

  cleanup.finish();
  await presentation;
  assert.equal(didSettle, true);
  assert.equal(empty.controller.snapshot().animationCount, 0);
  assert.equal(empty.painter.calls.filter(([kind]) => kind === "settle").length, 1);
});

test("eventless live and static replay presentations still settle immediately", async () => {
  const emptyPlan = plan("empty-non-animated-family");
  emptyPlan.animationSpecCount = 0;

  for (const renderPolicy of /** @type {const} */ (["live_once", "replay_static"])) {
    const immediate = harness();
    immediate.controller.presentFrame(
      { viewer_mode: "replay", plan: emptyPlan },
      surfaceA,
      { renderPolicy },
    );
    assert.equal(immediate.animationFactory.created.length, 0);
    assert.equal(immediate.controller.snapshot().animationCount, 0);
    assert.equal(immediate.controller.snapshot().submissionBlocked, false);
    assert.equal(
      immediate.painter.calls.filter(([kind]) => kind === "settle").length,
      1,
    );
    await immediate.controller.whenSettled();
  }
});

test("unsupported playback rates fail closed without changing an active clock", () => {
  const fixed = harness();
  fixed.controller.presentFrame(
    { viewer_mode: "replay", plan: plan("invalid-rate") },
    surfaceA,
    { renderPolicy: "replay_animated" },
  );
  const animations = [...fixed.animationFactory.created];
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
      () => fixed.controller.setPlaybackRate(/** @type {any} */ (invalid)),
      /playback rate must be one of/u,
    );
    assert.equal(fixed.controller.snapshot().playbackRate, 1);
    assert.deepEqual(fixed.animationFactory.created, animations);
    assert.equal(
      fixed.animationFactory.created.every(({ playbackRate }) => playbackRate === 1),
      true,
    );
    assert.throws(
      () => harness({ playbackRate: /** @type {any} */ (invalid) }),
      /playback rate must be one of/u,
    );
  }
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
        renderPolicy: "live_once",
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
    paintKey: "visual-filters-v2:all-on",
    renderPolicy: "live_once",
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
  assert.equal(controller.snapshot().playbackRate, 0.5);
  assert.equal(
    offAnimations.every(({ playbackRate }) => playbackRate === 0.5),
    true,
  );
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

test("whole-clock rate changes preserve an active reduced-motion explanation", () => {
  const { animationFactory, controller, painter } = harness({
    motionMode: "reduced",
  });
  controller.presentFrame({ plan: plan("epoch-reduced-rate") }, surfaceA);
  const installCount = painter.calls.filter(([kind]) => kind === "install").length;
  const createdCount = animationFactory.created.length;

  controller.setPlaybackRate(2);

  assert.equal(controller.snapshot().motionMode, "reduced");
  assert.equal(controller.snapshot().playbackRate, 2);
  assert.equal(
    animationFactory.created.every(({ playbackRate }) => playbackRate === 2),
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
