import { buildChoreographyPlan } from "./choreography-plan.js";

const DEFAULT_LEDGER_KEY = "marl-battlegrounds.visual-debugger.consumed-transitions.v1";
const DEFAULT_LEDGER_LIMIT = 256;
const MAX_ACTIVE_ANIMATIONS = 512;
const REPLAY_TERMINAL_HOLD_MS = 600;
const SUPPORTED_PLAYBACK_RATES = Object.freeze([
  0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,
]);

/**
 * @typedef {"normal" | "reduced" | "off"} MotionMode
 * @typedef {"live_once" | "replay_animated" | "replay_static"} RenderPolicy
 * @typedef {{
 *   element: Element,
 *   keyframes: Keyframe[] | PropertyIndexedKeyframes,
 *   options: KeyframeAnimationOptions,
 *   id: string,
 * }} AnimationSpec
 * @typedef {{
 *   id: string,
 *   currentTime: CSSNumberish | null,
 *   playbackRate: number,
 *   finished: Promise<unknown>,
 *   pause: () => void,
 *   play: () => void,
 *   finish: () => void,
 *   cancel: () => void,
 *   updatePlaybackRate?: (rate: number) => void,
 * }} AnimationHandle
 * @typedef {{
 *   create: (spec: AnimationSpec) => AnimationHandle,
 *   createClock: (
 *     element: Element,
 *     duration: number,
 *     id: string,
 *   ) => AnimationHandle,
 * }} AnimationFactory
 * @typedef {{
 *   layer: SVGElement,
 *   ownerDocument: Document,
 *   viewportKey: string,
 *   worldToScreen: (
 *     point: {x: number, y: number} | readonly [number, number],
 *   ) => {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 * }} ChoreographySurface
 * @typedef {{
 *   root: Element,
 *   animationSpecs?: readonly AnimationSpec[],
 *   nodeCount?: number,
 *   persistentNodeCount?: number,
 * }} ChoreographyInstallation
 * @typedef {{
 *   getItem: (key: string) => string | null,
 *   setItem: (key: string, value: string) => void,
 * }} PresentationStorage
 */

/**
 * Small bounded tab-local record of already-presented transition epochs.
 * Storage failures degrade to in-memory replay suppression.
 */
export class ConsumedTransitionLedger {
  /**
   * @param {{
   *   storage?: PresentationStorage | null,
   *   storageKey?: string,
   *   limit?: number,
   * }} [options]
   */
  constructor(options = {}) {
    this.storage = options.storage ?? null;
    this.storageKey = options.storageKey ?? DEFAULT_LEDGER_KEY;
    this.limit = positiveInteger(options.limit ?? DEFAULT_LEDGER_LIMIT, "limit");
    /** @type {Array<{epochKey: string, fingerprint: string}>} */
    this.entries = this.#load();
  }

  /**
   * @param {string} epochKey
   */
  has(epochKey) {
    return this.entries.some((entry) => entry.epochKey === epochKey);
  }

  /**
   * @param {string} epochKey
   * @returns {string | null}
   */
  fingerprintFor(epochKey) {
    return (
      this.entries.find((entry) => entry.epochKey === epochKey)?.fingerprint ?? null
    );
  }

  /**
   * @param {string} epochKey
   * @param {string} fingerprint
   */
  record(epochKey, fingerprint) {
    const withoutEpoch = this.entries.filter((entry) => entry.epochKey !== epochKey);
    withoutEpoch.push({ epochKey, fingerprint });
    this.entries = withoutEpoch.slice(-this.limit);
    this.#save();
  }

  #load() {
    if (!this.storage) {
      return [];
    }
    try {
      const raw = this.storage.getItem(this.storageKey);
      const parsed = raw === null ? [] : JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed
        .filter(
          (entry) =>
            entry &&
            typeof entry === "object" &&
            typeof entry.epochKey === "string" &&
            typeof entry.fingerprint === "string",
        )
        .slice(-this.limit)
        .map((entry) => ({
          epochKey: entry.epochKey,
          fingerprint: entry.fingerprint,
        }));
    } catch {
      return [];
    }
  }

  #save() {
    if (!this.storage) {
      return;
    }
    try {
      this.storage.setItem(this.storageKey, JSON.stringify(this.entries));
    } catch {
      // Presentation persistence is optional; authority is unaffected.
    }
  }
}

/**
 * Browser Web Animations adapter. The controller also accepts a fake adapter
 * with the same methods for deterministic unit tests.
 */
export class BrowserAnimationFactory {
  /**
   * @param {AnimationSpec} spec
   */
  create(spec) {
    const animation = spec.element.animate(spec.keyframes, spec.options);
    animation.id = spec.id;
    return animation;
  }

  /**
   * @param {Element} element
   * @param {number} duration
   * @param {string} id
   */
  createClock(element, duration, id) {
    return this.create({
      element,
      keyframes: [{ opacity: 1 }, { opacity: 1 }],
      options: { duration, fill: "both" },
      id,
    });
  }
}

/**
 * Own presentation time and only the controller's child beneath the transient
 * SVG layer. It never calls the debugger command API.
 */
export class CombatChoreographer {
  /**
   * @param {{
   *   painter: {
   *     install: (
   *       plan: Record<string, any>,
   *       surface: ChoreographySurface,
   *       options: {
   *         motionMode: MotionMode,
   *         renderPolicy: RenderPolicy,
   *         settled: boolean,
   *         persistentOnly: boolean,
   *         retainTransientOnSettle?: boolean,
   *       },
   *     ) => ChoreographyInstallation,
   *     clear: (installation: any, reason: string) => void,
   *     settle: (installation: any) => void,
   *     reproject: (
   *       installation: any,
   *       plan: Record<string, any>,
   *       surface: ChoreographySurface,
   *     ) => void,
   *   },
   *   planBuilder?: (
   *     frame: unknown,
   *     surface: ChoreographySurface | null,
   *     visualFilters?: Record<string, boolean>,
   *     renderPolicy?: RenderPolicy,
   *   ) => Record<string, any> | null,
   *   animationFactory?: AnimationFactory,
   *   ledger?: ConsumedTransitionLedger,
   *   onStateChange?: (state: ReturnType<CombatChoreographer["snapshot"]>) => void,
   *   motionMode?: MotionMode,
   *   playbackRate?: number,
   * }} options
   */
  constructor(options) {
    if (!options || typeof options !== "object" || !options.painter) {
      throw new TypeError("CombatChoreographer requires a painter.");
    }
    this.painter = options.painter;
    this.planBuilder = options.planBuilder ?? buildChoreographyPlan;
    this.animationFactory = options.animationFactory ?? new BrowserAnimationFactory();
    this.ledger = options.ledger ?? new ConsumedTransitionLedger();
    this.onStateChange = options.onStateChange ?? (() => {});
    this.motionMode = normalizeMotionMode(options.motionMode ?? "normal");
    this.playbackRate = normalizePlaybackRate(options.playbackRate ?? 1);
    this.paused = false;
    this.submissionBlocked = false;
    this.logicalTime = 0;
    this.generation = 0;
    /** @type {Record<string, any> | null} */
    this.plan = null;
    /** @type {RenderPolicy | null} */
    this.renderPolicy = null;
    /** @type {ChoreographySurface | null} */
    this.surface = null;
    /** @type {ChoreographyInstallation | null} */
    this.installation = null;
    /** @type {AnimationHandle[]} */
    this.animations = [];
    /** @type {AnimationHandle | null} */
    this.gateClock = null;
    /** @type {AnimationHandle | null} */
    this.cleanupClock = null;
    /** @type {Set<(value?: unknown) => void>} */
    this.settleWaiters = new Set();
  }

  snapshot() {
    return Object.freeze({
      active: this.installation !== null,
      animationCount: this.#allAnimations().length,
      epochKey: this.plan?.epochKey ?? null,
      authorizationKey: this.plan?.authorizationKey ?? null,
      fingerprint: this.plan?.fingerprint ?? null,
      paintKey: this.plan?.paintKey ?? null,
      renderPolicy: this.renderPolicy,
      logicalTime: this.logicalTime,
      motionMode: this.motionMode,
      paused: this.paused,
      playbackRate: this.playbackRate,
      submissionBlocked: this.submissionBlocked,
    });
  }

  /**
   * Resolve after the current authorized presentation has reached its durable
   * settled state. Replay autoplay uses this boundary so it cannot outrun the
   * explanation clock or overlap requests.
   */
  whenSettled() {
    if (this.#isSettled()) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.settleWaiters.add(resolve);
    });
  }

  /**
   * Install or reconcile one already-authorized frame after durable rendering.
   *
   * @param {unknown} frame
   * @param {ChoreographySurface | null} surface
   * @param {{
   *   renderPolicy?: RenderPolicy,
   *   visualFilters?: Record<string, boolean>,
   *   restartAnimated?: boolean,
   * }} [presentationControl]
   */
  presentFrame(frame, surface, presentationControl = {}) {
    const requestedPolicy = normalizeRenderPolicy(
      presentationControl.renderPolicy ?? "live_once",
    );
    const initialPolicy = requestedPolicy;
    let nextPlan = surface
      ? this.planBuilder(
          frame,
          surface,
          presentationControl.visualFilters,
          initialPolicy,
        )
      : null;
    if (!nextPlan || !surface) {
      this.clear("absent_scene_or_event_batch");
      return this.snapshot();
    }

    const currentPlan = this.plan;
    const sameEpoch = currentPlan?.epochKey === nextPlan.epochKey;
    const sameAuthorization =
      currentPlan?.authorizationKey === nextPlan.authorizationKey;
    const sameFingerprint = currentPlan?.fingerprint === nextPlan.fingerprint;
    const samePaint = currentPlan?.paintKey === nextPlan.paintKey;
    const replayPaintChange =
      sameEpoch &&
      sameAuthorization &&
      sameFingerprint &&
      !samePaint &&
      isReplayPolicy(initialPolicy);
    const explicitSamePlanReplayRestart =
      presentationControl.restartAnimated === true &&
      sameEpoch &&
      sameAuthorization &&
      sameFingerprint &&
      samePaint &&
      this.renderPolicy === "replay_static" &&
      requestedPolicy === "replay_animated";
    const stickyStatic =
      sameEpoch &&
      this.renderPolicy === "replay_static" &&
      requestedPolicy === "replay_animated" &&
      !explicitSamePlanReplayRestart;
    const nextPolicy =
      replayPaintChange || stickyStatic ? "replay_static" : initialPolicy;
    if (nextPolicy !== initialPolicy) {
      const policyPlan = this.planBuilder(
        frame,
        surface,
        presentationControl.visualFilters,
        nextPolicy,
      );
      if (!samePlanIdentity(nextPlan, policyPlan)) {
        throw new Error("render policy changed the authorized plan identity.");
      }
      nextPlan = policyPlan;
    }
    const samePolicy = this.renderPolicy === nextPolicy;

    if (sameEpoch && sameAuthorization && sameFingerprint && samePaint && samePolicy) {
      if (this.surface?.viewportKey !== surface.viewportKey && this.installation) {
        this.painter.reproject(this.installation, nextPlan, surface);
      }
      this.plan = nextPlan;
      this.surface = surface;
      this.#publish();
      return this.snapshot();
    }

    if (sameEpoch && sameAuthorization && sameFingerprint && !samePaint) {
      this.#clearOwned("visual_filters_changed");
      this.plan = nextPlan;
      this.surface = surface;
      this.#installForPolicy(nextPlan, surface, nextPolicy, {
        liveSafeRebuild: true,
      });
      this.#publish();
      return this.snapshot();
    }

    if (sameEpoch && sameAuthorization && sameFingerprint && !samePolicy) {
      this.#clearOwned("render_policy_changed");
      this.plan = nextPlan;
      this.surface = surface;
      this.#installForPolicy(nextPlan, surface, nextPolicy);
      this.#publish();
      return this.snapshot();
    }

    if (sameEpoch) {
      // Clear first: the previous authorization may contain privileged nodes.
      this.#clearOwned("authorization_or_disclosure_changed");
      this.plan = nextPlan;
      this.surface = surface;
      this.#installForPolicy(nextPlan, surface, nextPolicy, {
        liveSafeRebuild: true,
      });
      if (nextPolicy === "live_once") {
        this.ledger.record(nextPlan.epochKey, nextPlan.fingerprint);
      }
      this.#publish();
      return this.snapshot();
    }

    const consumedFingerprint =
      nextPolicy === "live_once" ? this.ledger.fingerprintFor(nextPlan.epochKey) : null;
    const alreadyConsumed = consumedFingerprint !== null;
    this.#clearOwned("new_transition");
    this.plan = nextPlan;
    this.surface = surface;
    this.#installForPolicy(nextPlan, surface, nextPolicy, {
      liveSafeRebuild: alreadyConsumed,
    });
    if (nextPolicy === "live_once" && consumedFingerprint !== nextPlan.fingerprint) {
      this.ledger.record(nextPlan.epochKey, nextPlan.fingerprint);
    }
    this.#publish();
    return this.snapshot();
  }

  /**
   * Recompute active geometry after a durable resize without replaying time.
   *
   * @param {unknown} frame
   * @param {ChoreographySurface | null} surface
   * @param {{
   *   renderPolicy?: RenderPolicy,
   *   visualFilters?: Record<string, boolean>,
   * }} [presentationControl]
   */
  reproject(frame, surface, presentationControl = {}) {
    if (!surface || !this.installation || !this.plan) {
      return this.presentFrame(frame, surface, presentationControl);
    }
    const requestedPolicy = normalizeRenderPolicy(
      presentationControl.renderPolicy ?? "live_once",
    );
    let nextPlan = this.planBuilder(
      frame,
      surface,
      presentationControl.visualFilters,
      requestedPolicy,
    );
    if (
      !nextPlan ||
      nextPlan.epochKey !== this.plan.epochKey ||
      nextPlan.authorizationKey !== this.plan.authorizationKey ||
      nextPlan.fingerprint !== this.plan.fingerprint ||
      nextPlan.paintKey !== this.plan.paintKey
    ) {
      return this.presentFrame(frame, surface, presentationControl);
    }
    const nextPolicy =
      this.renderPolicy === "replay_static" && requestedPolicy === "replay_animated"
        ? "replay_static"
        : requestedPolicy;
    if (nextPolicy !== this.renderPolicy) {
      return this.presentFrame(frame, surface, {
        ...presentationControl,
        renderPolicy: nextPolicy,
      });
    }
    if (nextPolicy !== requestedPolicy) {
      nextPlan = this.planBuilder(
        frame,
        surface,
        presentationControl.visualFilters,
        nextPolicy,
      );
      if (!samePlanIdentity(this.plan, nextPlan)) {
        return this.presentFrame(frame, surface, presentationControl);
      }
    }
    this.painter.reproject(this.installation, nextPlan, surface);
    this.plan = nextPlan;
    this.surface = surface;
    this.#publish();
    return this.snapshot();
  }

  togglePaused() {
    if (this.motionMode === "off" || this.#allAnimations().length === 0) {
      return this.snapshot();
    }
    this.paused = !this.paused;
    for (const animation of this.#allAnimations()) {
      if (this.paused) {
        animation.pause();
      } else {
        animation.play();
      }
    }
    this.#captureLogicalTime();
    this.#publish();
    return this.snapshot();
  }

  /**
   * @param {number} rate
   */
  setPlaybackRate(rate) {
    const nextRate = normalizePlaybackRate(rate);
    if (nextRate === this.playbackRate) {
      return this.snapshot();
    }
    this.#captureLogicalTime();
    this.playbackRate = nextRate;
    for (const animation of this.#allAnimations()) {
      applyPlaybackRate(animation, nextRate);
    }
    this.#publish();
    return this.snapshot();
  }

  /**
   * @param {MotionMode} mode
   */
  setMotionMode(mode) {
    const nextMode = normalizeMotionMode(mode);
    if (nextMode === this.motionMode) {
      return this.snapshot();
    }

    const plan = this.plan;
    const surface = this.surface;
    const hasActiveExplanation =
      this.installation !== null &&
      plan !== null &&
      surface !== null &&
      this.#allAnimations().length > 0;
    this.motionMode = nextMode;

    if (this.motionMode === "off" && hasActiveExplanation) {
      this.paused = false;
      this.#clearOwned("motion_disabled_static_reinstall");
      this.#install(plan, surface, {
        renderPolicy: this.renderPolicy ?? "live_once",
        settled: false,
        persistentOnly: false,
      });
      this.#publish();
    } else if (this.motionMode !== "normal") {
      this.paused = false;
      this.skip();
    } else {
      this.#publish();
    }
    return this.snapshot();
  }

  skip() {
    const installation = this.installation;
    if (!installation) {
      this.submissionBlocked = false;
      this.#publish();
      return this.snapshot();
    }
    for (const animation of this.#allAnimations()) {
      safeFinish(animation);
    }
    this.logicalTime = Number(this.plan?.phases?.total ?? this.logicalTime);
    this.painter.settle(installation);
    this.#cancelAnimations();
    this.submissionBlocked = false;
    this.#publish();
    return this.snapshot();
  }

  /**
   * @param {string} reason
   */
  clear(reason = "explicit_clear") {
    this.#clearOwned(reason);
    this.plan = null;
    this.renderPolicy = null;
    this.surface = null;
    this.logicalTime = 0;
    this.#publish();
    return this.snapshot();
  }

  dispose() {
    return this.clear("dispose");
  }

  /**
   * @param {Record<string, any>} plan
   * @param {ChoreographySurface} surface
   * @param {RenderPolicy} renderPolicy
   * @param {{liveSafeRebuild?: boolean}} [options]
   */
  #installForPolicy(plan, surface, renderPolicy, options = {}) {
    if (renderPolicy === "replay_static") {
      this.paused = true;
      this.#install(plan, surface, {
        renderPolicy,
        settled: true,
        persistentOnly: false,
        retainTransientOnSettle: true,
      });
      return;
    }
    this.paused = false;
    const liveSafeRebuild =
      renderPolicy === "live_once" && options.liveSafeRebuild === true;
    this.#install(plan, surface, {
      renderPolicy,
      settled: liveSafeRebuild,
      persistentOnly: liveSafeRebuild,
    });
  }

  /**
   * @param {Record<string, any>} plan
   * @param {ChoreographySurface} surface
   * @param {{
   *   renderPolicy: RenderPolicy,
   *   settled: boolean,
   *   persistentOnly: boolean,
   *   retainTransientOnSettle?: boolean,
   * }} options
   */
  #install(plan, surface, options) {
    const persistentOnly = Boolean(options.persistentOnly);
    const settled = Boolean(options.settled);
    /** @type {{
     *   motionMode: MotionMode,
     *   renderPolicy: RenderPolicy,
     *   settled: boolean,
     *   persistentOnly: boolean,
     *   retainTransientOnSettle?: boolean,
     * }} */
    const painterOptions = {
      motionMode: this.motionMode,
      renderPolicy: options.renderPolicy,
      settled,
      persistentOnly,
    };
    if (options.retainTransientOnSettle === true) {
      painterOptions.retainTransientOnSettle = true;
    }
    const installation = this.painter.install(plan, surface, painterOptions);
    const nodeCount = nonNegativeInteger(installation.nodeCount ?? 0, "nodeCount");
    const animationSpecs = Array.isArray(installation.animationSpecs)
      ? installation.animationSpecs
      : [];
    if (nodeCount > Number(plan.bounds?.nodes ?? 512)) {
      this.painter.clear(installation, "node_bound_exceeded");
      throw new RangeError("choreography painter exceeded the planned node bound.");
    }
    const persistentNodeCount = nonNegativeInteger(
      installation.persistentNodeCount ?? 0,
      "persistentNodeCount",
    );
    if (persistentNodeCount > Number(plan.bounds?.persistentNodes ?? 64)) {
      this.painter.clear(installation, "persistent_node_bound_exceeded");
      throw new RangeError("choreography painter exceeded the persistent node bound.");
    }
    const staticOff =
      this.motionMode === "off" && !settled && !persistentOnly && nodeCount > 1;
    const eventlessReplayAnimated =
      options.renderPolicy === "replay_animated" &&
      animationSpecs.length === 0 &&
      !settled &&
      !persistentOnly;
    const clockBudget =
      settled || persistentOnly
        ? 0
        : eventlessReplayAnimated || staticOff
          ? 1
          : animationSpecs.length === 0
            ? 0
            : this.motionMode === "normal"
              ? 2
              : 1;
    const activeAnimationCount = animationSpecs.length + clockBudget;
    const plannedAnimationLimit = nonNegativeInteger(
      plan.bounds?.animations ?? MAX_ACTIVE_ANIMATIONS,
      "plan.bounds.animations",
    );
    if (
      activeAnimationCount > MAX_ACTIVE_ANIMATIONS ||
      activeAnimationCount > plannedAnimationLimit
    ) {
      this.painter.clear(installation, "animation_bound_exceeded");
      throw new RangeError("choreography painter exceeded the animation bound.");
    }
    this.installation = installation;
    this.renderPolicy = options.renderPolicy;
    this.logicalTime =
      options.renderPolicy === "replay_static"
        ? 0
        : settled
          ? Number(plan.phases?.total ?? 0)
          : 0;
    if (settled || persistentOnly) {
      this.painter.settle(installation);
      this.submissionBlocked = false;
      return;
    }
    if (animationSpecs.length === 0 && !staticOff && !eventlessReplayAnimated) {
      this.logicalTime = Number(plan.phases?.total ?? 0);
      this.painter.settle(installation);
      this.submissionBlocked = false;
      return;
    }

    const authoredDuration =
      this.motionMode === "reduced"
        ? Number(plan.phases?.reducedTotal ?? 0)
        : Number(plan.phases?.total ?? 0);
    const duration =
      authoredDuration +
      (options.renderPolicy === "replay_animated" ? REPLAY_TERMINAL_HOLD_MS : 0);
    /** @type {AnimationHandle[]} */
    const createdAnimations = [];
    /** @type {AnimationHandle | null} */
    let cleanupClock = null;
    /** @type {AnimationHandle | null} */
    let gateClock = null;
    try {
      for (const spec of animationSpecs) {
        const animation = this.animationFactory.create(spec);
        ignoreCancellation(animation.finished);
        createdAnimations.push(animation);
      }
      cleanupClock = this.animationFactory.createClock(
        installation.root,
        duration,
        `mbg:${plan.epochKey}:cleanup`,
      );
      ignoreCancellation(cleanupClock.finished);
      if (this.motionMode === "normal" && !eventlessReplayAnimated) {
        gateClock = this.animationFactory.createClock(
          installation.root,
          Number(plan.phases?.submissionRelease ?? 0),
          `mbg:${plan.epochKey}:gate`,
        );
        ignoreCancellation(gateClock.finished);
      }
    } catch (error) {
      for (const animation of [
        ...createdAnimations,
        ...(cleanupClock ? [cleanupClock] : []),
        ...(gateClock ? [gateClock] : []),
      ]) {
        safeCancel(animation);
      }
      this.painter.clear(installation, "animation_creation_failed");
      this.installation = null;
      throw error;
    }
    this.animations = createdAnimations;
    this.cleanupClock = cleanupClock;
    this.gateClock = gateClock;
    this.submissionBlocked = this.motionMode === "normal" && !eventlessReplayAnimated;
    for (const animation of this.#allAnimations()) {
      applyPlaybackRate(animation, this.playbackRate);
      if (this.paused) {
        animation.pause();
      }
    }
    const generation = this.generation;
    if (this.gateClock) {
      ignoreCancellation(
        this.gateClock.finished.then(() => {
          if (generation !== this.generation) {
            return;
          }
          this.submissionBlocked = false;
          this.#captureLogicalTime();
          this.#publish();
        }),
      );
    }
    if (this.cleanupClock) {
      ignoreCancellation(
        this.cleanupClock.finished.then(() => {
          if (generation !== this.generation || !this.installation) {
            return;
          }
          this.logicalTime = Number(this.plan?.phases?.total ?? this.logicalTime);
          this.painter.settle(this.installation);
          this.#cancelAnimations();
          this.submissionBlocked = false;
          this.#publish();
        }),
      );
    }
  }

  /**
   * @param {string} reason
   */
  #clearOwned(reason) {
    this.generation += 1;
    this.#captureLogicalTime();
    this.#cancelAnimations();
    if (this.installation) {
      this.painter.clear(this.installation, reason);
    }
    this.installation = null;
    this.submissionBlocked = false;
  }

  #captureLogicalTime() {
    const candidate =
      this.cleanupClock?.currentTime ?? this.gateClock?.currentTime ?? this.logicalTime;
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      const authoredTotal = Number(this.plan?.phases?.total ?? candidate);
      this.logicalTime = Number.isFinite(authoredTotal)
        ? Math.min(Math.max(candidate, 0), Math.max(authoredTotal, 0))
        : candidate;
    }
  }

  #allAnimations() {
    return [
      ...this.animations,
      ...(this.gateClock ? [this.gateClock] : []),
      ...(this.cleanupClock ? [this.cleanupClock] : []),
    ];
  }

  #cancelAnimations() {
    for (const animation of this.#allAnimations()) {
      safeCancel(animation);
    }
    this.animations = [];
    this.gateClock = null;
    this.cleanupClock = null;
  }

  #publish() {
    const snapshot = this.snapshot();
    this.onStateChange(snapshot);
    if (this.#isSettled()) {
      const waiters = [...this.settleWaiters];
      this.settleWaiters.clear();
      for (const resolve of waiters) {
        resolve();
      }
    }
  }

  #isSettled() {
    return this.#allAnimations().length === 0 && !this.submissionBlocked;
  }
}

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * @param {unknown} value
 * @returns {RenderPolicy}
 */
function normalizeRenderPolicy(value) {
  if (
    value === "live_once" ||
    value === "replay_animated" ||
    value === "replay_static"
  ) {
    return value;
  }
  throw new RangeError(`unknown render policy ${String(value)}.`);
}

/** @param {RenderPolicy} policy */
function isReplayPolicy(policy) {
  return policy === "replay_animated" || policy === "replay_static";
}

/**
 * @param {Record<string, any>} first
 * @param {unknown} second
 * @returns {second is Record<string, any>}
 */
function samePlanIdentity(first, second) {
  return (
    isRecord(second) &&
    second.epochKey === first.epochKey &&
    second.authorizationKey === first.authorizationKey &&
    second.fingerprint === first.fingerprint &&
    second.paintKey === first.paintKey
  );
}

/**
 * @param {Promise<unknown>} promise
 */
function ignoreCancellation(promise) {
  promise.catch(() => {});
}

/**
 * @param {AnimationHandle} animation
 */
function safeCancel(animation) {
  try {
    animation.cancel();
  } catch {
    // Cancellation is best-effort presentation cleanup.
  }
}

/**
 * @param {AnimationHandle} animation
 */
function safeFinish(animation) {
  try {
    animation.finish();
  } catch {
    // An idle/cancelled animation needs no further presentation work.
  }
}

/**
 * @param {unknown} value
 * @returns {MotionMode}
 */
function normalizeMotionMode(value) {
  if (value === "normal" || value === "reduced" || value === "off") {
    return value;
  }
  throw new RangeError(`unknown motion mode ${String(value)}.`);
}

/** @param {unknown} value */
function normalizePlaybackRate(value) {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    !SUPPORTED_PLAYBACK_RATES.includes(value)
  ) {
    throw new RangeError(
      "playback rate must be one of 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, or 2.00.",
    );
  }
  return value;
}

/** @param {AnimationHandle} animation @param {number} rate */
function applyPlaybackRate(animation, rate) {
  if (typeof animation.updatePlaybackRate === "function") {
    animation.updatePlaybackRate(rate);
  } else {
    animation.playbackRate = rate;
  }
}

/**
 * @param {unknown} value
 * @param {string} name
 */
function positiveInteger(value, name) {
  if (!Number.isInteger(value) || Number(value) <= 0) {
    throw new RangeError(`${name} must be a positive integer.`);
  }
  return Number(value);
}

/**
 * @param {unknown} value
 * @param {string} name
 */
function nonNegativeInteger(value, name) {
  if (!Number.isInteger(value) || Number(value) < 0) {
    throw new RangeError(`${name} must be a non-negative integer.`);
  }
  return Number(value);
}
