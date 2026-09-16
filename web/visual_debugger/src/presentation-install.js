/**
 * @file Keep asynchronous presentation installs tied to the latest browser request.
 * The coordinator numbers attempts, drops stale results and permits one
 * GET-only recovery after a classified join race. Callers own network calls,
 * authorization joins, DOM updates and pending-view behavior.
 */
/**
 * Coordinate freshness for one browser page's presentation installation.
 *
 * The instance owns a mutable generation counter and three caller callbacks.
 * It never decodes transport or decides information rights. A newer begin call
 * supersedes older asynchronous work; work is not cancelled, but stale results
 * and failures cannot install a view. Reuse the instance for that page.
 */
export class PresentationInstallCoordinator {
  /**
   * Create a coordinator with three required callbacks.
   *
   * options.onAttemptBegin(reason, pendingPolicy) updates the caller's pending
   * state; options.install(joined) installs an already authorized pair;
   * options.isJoinRace(error) decides whether one GET-only recovery is allowed.
   * Each must be a function or construction throws TypeError. No callback is
   * called during construction. generation starts at zero; callbacks are retained.
   *
   * @param {{
   *   onAttemptBegin: (
   *     reason: string,
   *     pendingPolicy: "retain_last_authorized" | "clear",
   *   ) => void,
   *   install: (joined: Readonly<Record<string, any>>) => void,
   *   isJoinRace: (error: unknown) => boolean,
   * }} options
   */
  constructor({ onAttemptBegin, install, isJoinRace }) {
    if (
      typeof onAttemptBegin !== "function" ||
      typeof install !== "function" ||
      typeof isJoinRace !== "function"
    ) {
      throw new TypeError("Presentation installation callbacks are required.");
    }
    this.onAttemptBegin = onAttemptBegin;
    this.install = install;
    this.isJoinRace = isJoinRace;
    this.generation = 0;
  }

  /**
   * Start a new attempt and return its generation number.
   *
   * reason is passed through to onAttemptBegin. pendingPolicy must be
   * retain_last_authorized or clear, otherwise throw TypeError before advancing.
   * Increment the counter, call onAttemptBegin synchronously and return the new
   * number. An exception from that callback propagates after the increment.
   *
   * @param {string} reason
   * @param {"retain_last_authorized" | "clear"} pendingPolicy
   */
  begin(reason, pendingPolicy) {
    if (pendingPolicy !== "retain_last_authorized" && pendingPolicy !== "clear") {
      throw new TypeError("A valid presentation pending policy is required.");
    }
    this.generation += 1;
    const generation = this.generation;
    this.onAttemptBegin(reason, pendingPolicy);
    return generation;
  }

  /**
   * Return whether generation exactly equals this instance's latest counter.
   * No validation or state change occurs; a stale attempt receives false.
   *
   * @param {number} generation
   */
  isCurrent(generation) {
    return generation === this.generation;
  }

  /**
   * Fetch and install a joined view if this attempt is still current.
   *
   * options supplies reason, pendingPolicy and getJoined(), which resolves an
   * already checked transport/presentation pair. A missing getJoined function
   * rejects with TypeError. begin applies the pending policy. A first error
   * classified by isJoinRace permits exactly one new getJoined call. Other
   * errors, including a failed retry, reject while current. Superseded work
   * resolves a frozen {status: superseded} record instead. Success calls install
   * once and returns a frozen {status: installed, joined, resynchronized}
   * record. The coordinator performs no network call except through callbacks.
   *
   * @param {{
   *   reason: string,
   *   pendingPolicy: "retain_last_authorized" | "clear",
   *   getJoined: () => Promise<Readonly<Record<string, any>>>,
   * }} options
   */
  async installFromGet({ reason, pendingPolicy, getJoined }) {
    if (typeof getJoined !== "function") {
      throw new TypeError("A joined GET callback is required.");
    }
    const generation = this.begin(reason, pendingPolicy);
    try {
      let resynchronized = false;
      let joined;
      try {
        joined = await getJoined();
      } catch (error) {
        if (!this.isCurrent(generation)) {
          return Object.freeze({ status: "superseded" });
        }
        if (!this.isJoinRace(error)) {
          throw error;
        }
        resynchronized = true;
        joined = await getJoined();
      }
      if (!this.isCurrent(generation)) {
        return Object.freeze({ status: "superseded" });
      }
      this.install(joined);
      return Object.freeze({
        status: "installed",
        joined,
        resynchronized,
      });
    } catch (error) {
      if (!this.isCurrent(generation)) {
        return Object.freeze({ status: "superseded" });
      }
      throw error;
    }
  }

  /**
   * Send a command once, then install its joined result if still current.
   *
   * options supplies reason, pendingPolicy, sendCommand(),
   * joinCommandResult(commandResult) and getJoined(). All three work callbacks
   * must be functions or the promise rejects with TypeError. After begin, call
   * sendCommand exactly once. A classified race from joinCommandResult permits
   * one fresh getJoined call; the command is never repeated. Current failures
   * reject. Superseded work resolves {status: superseded}. Success calls install
   * and resolves a frozen {status: installed, commandResult, joined,
   * resynchronized} record. Fetching and authorization remain caller-owned.
   *
   * @param {{
   *   reason: string,
   *   pendingPolicy: "retain_last_authorized" | "clear",
   *   sendCommand: () => Promise<any>,
   *   joinCommandResult: (commandResult: any) => Promise<Readonly<Record<string, any>>>,
   *   getJoined: () => Promise<Readonly<Record<string, any>>>,
   * }} options
   */
  async installFromCommand({
    reason,
    pendingPolicy,
    sendCommand,
    joinCommandResult,
    getJoined,
  }) {
    if (
      typeof sendCommand !== "function" ||
      typeof joinCommandResult !== "function" ||
      typeof getJoined !== "function"
    ) {
      throw new TypeError("Command installation callbacks are required.");
    }
    const generation = this.begin(reason, pendingPolicy);
    try {
      const commandResult = await sendCommand();
      if (!this.isCurrent(generation)) {
        return Object.freeze({ status: "superseded" });
      }
      let resynchronized = false;
      let joined;
      try {
        joined = await joinCommandResult(commandResult);
      } catch (error) {
        if (!this.isCurrent(generation)) {
          return Object.freeze({ status: "superseded" });
        }
        if (!this.isJoinRace(error)) {
          throw error;
        }
        resynchronized = true;
        joined = await getJoined();
      }
      if (!this.isCurrent(generation)) {
        return Object.freeze({ status: "superseded" });
      }
      this.install(joined);
      return Object.freeze({
        status: "installed",
        commandResult,
        joined,
        resynchronized,
      });
    } catch (error) {
      if (!this.isCurrent(generation)) {
        return Object.freeze({ status: "superseded" });
      }
      throw error;
    }
  }
}
