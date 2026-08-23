/**
 * Coordinate browser-owned presentation installation freshness without owning
 * transport decoding or raw/presentation identity joins. Callers supply those
 * authority boundaries and retain ownership of all DOM/state changes.
 */
export class PresentationInstallCoordinator {
  /**
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

  /** @param {number} generation */
  isCurrent(generation) {
    return generation === this.generation;
  }

  /**
   * Install one GET pair, allowing exactly one fresh GET-only resynchronization
   * when the supplied identity join classifies the first attempt as a race.
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
   * Send a command once, join its returned raw candidate to a separately
   * fetched presentation, and fall back to one fresh GET pair only when that
   * join races. The command callback is deliberately outside the retry branch.
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
