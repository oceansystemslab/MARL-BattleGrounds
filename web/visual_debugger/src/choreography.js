/**
 * @file Control presentation time for already-authorized combat frames.
 * CombatChoreographer installs painter-owned SVG, controls Web Animations and
 * suppresses repeated live transitions with a bounded optional storage ledger.
 * Replay can animate or show settled explanations. This module never chooses
 * actions, advances the simulator or calls the debugger command API.
 */
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
 * Remember a bounded list of presented live transition epochs.
 *
 * Entries pair an epoch key with its disclosed-content fingerprint. This mutable
 * presentation cache suppresses repeated animation; it grants no information rights.
 * Optional storage failures fall back to the current in-memory entries.
 */
export class ConsumedTransitionLedger {
  /**
   * Load the optional presentation cache and keep its newest bounded entries.
   *
   * options defaults to an empty object. storage defaults to null (memory only),
   * storageKey to the module's versioned ledger key, and limit to 256. A nonpositive
   * or noninteger limit throws RangeError. Storage/JSON errors are ignored; invalid
   * stored rows are removed. The supplied storage object remains caller-owned.
   *
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
   * Return whether epochKey has an entry, regardless of its fingerprint.
   *
   * This exact string lookup neither updates recency nor reads storage again.
   *
   * @param {string} epochKey
   */
  has(epochKey) {
    return this.entries.some((entry) => entry.epochKey === epochKey);
  }

  /**
   * Return the stored fingerprint for epochKey, or null if it is absent.
   *
   * An empty stored string remains an empty string. This lookup does not mutate the
   * ledger or reread optional storage.
   *
   * @param {string} epochKey
   * @returns {string | null}
   */
  fingerprintFor(epochKey) {
    return (
      this.entries.find((entry) => entry.epochKey === epochKey)?.fingerprint ?? null
    );
  }

  /**
   * Remember epochKey with fingerprint as the newest consumed transition.
   *
   * Replace any older entry for the same key, retain at most limit entries, then
   * attempt persistence. Inputs are caller-supplied strings; this method adds no
   * validation. Storage failure is ignored. Returns undefined.
   *
   * @param {string} epochKey
   * @param {string} fingerprint
   */
  record(epochKey, fingerprint) {
    const withoutEpoch = this.entries.filter((entry) => entry.epochKey !== epochKey);
    withoutEpoch.push({ epochKey, fingerprint });
    this.entries = withoutEpoch.slice(-this.limit);
    this.#save();
  }

  /**
   * Read and sanitize optional storage once during construction.
   *
   * Return a new array of the newest limit records having string epochKey and
   * fingerprint fields. Missing storage, invalid JSON/root or storage errors yield
   * an empty array. Stored identities are not treated as authorization.
   */
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

  /**
   * Attempt to persist the current entries under this ledger's storage key.
   *
   * Return undefined; absent storage does nothing and write/serialization failures
   * are ignored because animation suppression must not control authority.
   */
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
 * Adapt the browser Web Animations API to the choreographer's clock interface.
 *
 * Methods create live browser animations. Tests may supply another adapter with
 * the same methods; the controller owns cancellation and settlement.
 */
export class BrowserAnimationFactory {
  /**
   * Start one browser animation from the supplied spec and assign its ID.
   *
   * spec provides the target Element, keyframes, timing options and id. Return the
   * browser Animation; Element.animate errors propagate. The element is not removed.
   *
   * @param {AnimationSpec} spec
   */
  create(spec) {
    const animation = spec.element.animate(spec.keyframes, spec.options);
    animation.id = spec.id;
    return animation;
  }

  /**
   * Create a timer animation whose two opacity values are both one.
   *
   * element hosts the clock, duration is browser animation time in milliseconds,
   * and id names it for inspection. Return the live Animation with fill both.
   * Browser animation errors propagate; the caller owns its lifetime.
   *
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
 * Own one authorized combat explanation and its presentation clocks.
 *
 * The controller mutates only its painter installation under the supplied surface,
 * tracks pause/rate/settlement, and reports immutable state snapshots. New authority
 * clears old nodes before installing replacements. It never submits an action.
 * Call dispose when the owner no longer needs the controller.
 */
export class CombatChoreographer {
  /**
   * Create an idle controller with a required painter and optional adapters.
   *
   * options.painter supplies install, clear, settle and reproject. planBuilder defaults
   * to buildChoreographyPlan, animationFactory to BrowserAnimationFactory, ledger to
   * a memory-only ConsumedTransitionLedger, and onStateChange to a no-op. motionMode
   * defaults to normal and playbackRate to 1. Missing painter throws TypeError;
   * unknown modes or unsupported rates throw RangeError. No frame is installed yet.
   *
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

  /**
   * Return a frozen view of the controller's current presentation state.
   *
   * The result includes active installation, animation count, nullable plan identity,
   * render policy, logical milliseconds, motion mode, pause/rate and submission gate.
   * It reports the last captured logical time; it does not sample the clock itself.
   */
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
   * Return a promise resolved when this controller next reports settlement.
   *
   * Resolve immediately when there are no owned animations and submission is not
   * blocked. Otherwise register a waiter until publish observes that state, including
   * a clear or skip. Replay autoplay uses this boundary to avoid overlapping requests.
   * The promise carries no frame data and never advances the simulator.
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
   * Install or reconcile an already-authorized frame after durable rendering.
   *
   * frame is passed to the plan builder; surface supplies the transient SVG layer and
   * projection, or null to clear. presentationControl defaults to live_once with
   * builder-default visualFilters. restartAnimated=false normally preserves settled
   * replay state; true permits an explicit same-plan static-to-animated restart.
   *
   * Return the frozen controller snapshot. Identical plans preserve time; a viewport
   * change reprojects. Filter changes rebuild live plans without replaying consumed
   * motion and make replay explanations static. Changed authority/disclosure clears
   * old nodes first. Absent scene/plan clears the installation. Live epochs use the
   * ledger; replay does not consume it. Invalid policy, plans, painter bounds or
   * browser animations may throw; a policy-induced identity change throws Error.
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
   * Update installed geometry after a durable resize without replaying time.
   *
   * frame and surface have the presentFrame contract. presentationControl defaults
   * to live_once and builder-default filters. Return a frozen snapshot. If the plan,
   * authority, paint choice or installation no longer matches, delegate to presentFrame.
   * A settled replay remains static. Plan/painter errors propagate.
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

  /**
   * Toggle all owned animation clocks and return the resulting snapshot.
   *
   * Motion off or no animations is a no-op. Otherwise pause/play every handle, capture
   * logical time and publish. This pauses presentation only; it sends no game command.
   */
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
   * Apply a supported presentation speed while retaining current clock progress.
   *
   * rate must be 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75 or 2; otherwise throw RangeError.
   * An unchanged rate is a no-op. Update every owned animation and publish, then return
   * a snapshot. Simulation timing and recorded transition values are unchanged.
   *
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
   * Set normal, reduced or off motion and reconcile the active explanation.
   *
   * mode outside that set throws RangeError. An unchanged mode is a no-op. Off can
   * reinstall an active explanation statically; reduced settles current motion through
   * skip. Returning to normal does not replay an already settled transition. Return a
   * snapshot; painter/browser failures propagate.
   *
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

  /**
   * Settle the current explanation immediately and release submission gating.
   *
   * Finish owned handles where possible, settle the painter, cancel remaining clocks
   * and publish the final logical time. With no installation, simply release the gate.
   * Return a snapshot; this does not request or skip a simulator transition.
   */
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
   * Remove this controller's installation and clear its active frame identity.
   *
   * reason defaults to explicit_clear and is passed to the painter. Cancel clocks,
   * reset logical time, release the gate and publish, resolving settlement waiters.
   * Return a snapshot. The consumed-transition ledger is retained.
   *
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

  /**
   * Release owned SVG and clocks using the dispose clear reason.
   *
   * Return the cleared snapshot and resolve settlement waiters. The ledger survives;
   * this method does not destroy caller-owned storage or prevent later reuse.
   */
  dispose() {
    return this.clear("dispose");
  }

  /**
   * Choose settled/static or animated installation for a validated plan.
   *
   * plan and surface are ready for the painter; renderPolicy selects live/replay.
   * options.liveSafeRebuild defaults to false and suppresses repeated live transients
   * when true. replay_static keeps settled explanatory transients and marks paused.
   * Mutate controller installation state; return undefined. Installation errors propagate.
   *
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
   * Install bounded painter nodes and create the required animation clocks.
   *
   * plan supplies phases in milliseconds and resource bounds; surface owns projection
   * and the transient layer. options gives renderPolicy, settled and persistentOnly;
   * retainTransientOnSettle is forwarded only when true. Validate node, persistent-node
   * and animation counts; excess bounds clear the new installation and throw RangeError.
   *
   * Settled or persistent-only output needs no clocks. Animated replay adds a 600 ms
   * terminal hold, including eventless transitions. Normal motion can gate submission
   * until its release phase. Creation failures cancel newly created handles, clear the
   * installation and rethrow. Completion handlers use generation checks so stale clocks
   * cannot settle a later authority. Return undefined.
   *
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
   * Cancel this installation before an authority or presentation replacement.
   *
   * reason is passed to painter.clear. Increment generation to invalidate old promise
   * handlers, capture logical time, cancel handles and release the gate. Keep the plan,
   * surface and render policy for the caller to replace. Return undefined.
   *
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

  /**
   * Capture finite cleanup-clock time, falling back to gate or saved time.
   *
   * Clamp a numeric clock to zero through the authored total when that total is finite.
   * Non-numeric clock values leave saved time unchanged. Mutate logicalTime only.
   */
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

  /**
   * Return a new array of owned visual handles followed by gate and cleanup clocks.
   *
   * Absent clocks are omitted. Handles remain the controller's mutable browser objects;
   * this helper does not advance, cancel or copy them.
   */
  #allAnimations() {
    return [
      ...this.animations,
      ...(this.gateClock ? [this.gateClock] : []),
      ...(this.cleanupClock ? [this.cleanupClock] : []),
    ];
  }

  /**
   * Best-effort cancel every owned handle and clear all handle references.
   *
   * Cancellation errors are ignored. Painter nodes, plan identity, logical time and
   * submission gating are left to the caller. Return undefined.
   */
  #cancelAnimations() {
    for (const animation of this.#allAnimations()) {
      safeCancel(animation);
    }
    this.animations = [];
    this.gateClock = null;
    this.cleanupClock = null;
  }

  /**
   * Notify the owner with a snapshot and release waiters when settled.
   *
   * Call onStateChange synchronously, then resolve and clear all queued settlement
   * waiters if no animations or submission gate remain. Callback errors propagate.
   */
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

  /**
   * Return true exactly when there are no owned animations and no submission gate.
   *
   * A persistent or static installation may still be visible; settlement concerns its
   * presentation clock, not removal of durable explanatory nodes.
   */
  #isSettled() {
    return this.#allAnimations().length === 0 && !this.submissionBlocked;
  }
}

/**
 * Return whether value is a non-null object other than an array.
 *
 * This broad shape guard does not validate a plan or its authority.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return a recognized live_once, replay_animated or replay_static value.
 *
 * Any other value throws RangeError; no coercion or fallback is applied.
 *
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

/**
 * Return whether policy selects either replay presentation mode.
 *
 * The caller supplies a validated render policy; live_once returns false.
 *
 * @param {RenderPolicy} policy
 */
function isReplayPolicy(policy) {
  return policy === "replay_animated" || policy === "replay_static";
}

/**
 * Compare the four stable identity fields without changing either plan.
 *
 * first is a plan; second may be unknown. Return false unless second is a record
 * with equal epochKey, authorizationKey, fingerprint and paintKey. This is an
 * identity comparison, not full plan or information-rights validation.
 *
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
 * Attach a no-op rejection handler to a presentation promise.
 *
 * promise is normally animation.finished. All rejections are swallowed, not only
 * cancellation; return undefined. This avoids unhandled cleanup rejections.
 *
 * @param {Promise<unknown>} promise
 */
function ignoreCancellation(promise) {
  promise.catch(() => {});
}

/**
 * Best-effort cancel animation and ignore cancellation errors.
 *
 * Return undefined; this helper does not remove the animation from owner arrays.
 *
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
 * Best-effort finish animation and ignore finish errors.
 *
 * Idle or already cancelled browser handles may reject finish; return undefined.
 *
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
 * Return normal, reduced or off unchanged, otherwise throw RangeError.
 *
 * value is checked exactly; there is no string coercion or default here.
 *
 * @param {unknown} value
 * @returns {MotionMode}
 */
function normalizeMotionMode(value) {
  if (value === "normal" || value === "reduced" || value === "off") {
    return value;
  }
  throw new RangeError(`unknown motion mode ${String(value)}.`);
}

/**
 * Return a finite supported quarter-step rate from 0.25 through 2.
 *
 * value must be a number in the fixed rate list; otherwise throw RangeError.
 *
 * @param {unknown} value
 */
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

/**
 * Update animation to rate using its browser rate method when available.
 *
 * rate is already validated. Fall back to assigning playbackRate for compatible
 * adapters. Return undefined; adapter errors propagate.
 *
 * @param {AnimationHandle} animation @param {number} rate
 */
function applyPlaybackRate(animation, rate) {
  if (typeof animation.updatePlaybackRate === "function") {
    animation.updatePlaybackRate(rate);
  } else {
    animation.playbackRate = rate;
  }
}

/**
 * Return value as a number when it is an integer greater than zero.
 *
 * name labels RangeError otherwise. This checks Number.isInteger, not safe-integer
 * precision, and does not accept numeric strings.
 *
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
 * Return value as a number when it is an integer at least zero.
 *
 * name labels RangeError otherwise. No numeric strings or safe-integer guarantee
 * are accepted/implied by this check.
 *
 * @param {unknown} value
 * @param {string} name
 */
function nonNegativeInteger(value, name) {
  if (!Number.isInteger(value) || Number(value) < 0) {
    throw new RangeError(`${name} must be a non-negative integer.`);
  }
  return Number(value);
}
