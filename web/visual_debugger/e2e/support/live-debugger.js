/**
 * @file Launch and control a local live debugger for browser tests, with fixed inputs
 * and explicit process cleanup. Startup waits for the launch URL and shutdown sends
 * SIGINT and waits for exit, both with no time limit.
 *
 * This file also owns the lifecycle of every browser-test server. The startup
 * helpers (`startDebuggerProcess` here, `startRecordingDebugger` and
 * `startReplayViewer`) give each server a fresh random owner token in
 * `MARL_BG_E2E_SERVER_OWNER` and register the server at spawn, before any other
 * `exit` listener, so the registry sees each exit first. The `uv` launcher, the
 * Python server and anything it starts inherit the token; the product ignores it.
 * A process is owned only when it runs as this user and its `/proc/<pid>/environ`
 * holds that exact token, so this works on Linux only. This process and zombies
 * never count. Ownership and the Linux process start time are checked again just
 * before each signal. Cleanup remembers PID plus start time, so reuse of a PID
 * cannot hide a new owned process. A final check followed by a numeric-PID signal
 * still has a small race; these helpers do not provide atomic kernel handles.
 * `signalOwnedProcess` returns false for missing or changed ownership and throws
 * actual signal errors for its caller to report. `killOwnedProcesses` may exclude
 * a paused launcher by PID while a failure test kills only its server children.
 *
 * After startup, a server may exit only in an allowed way:
 * - `stopDebugger` marks a requested stop, so any outcome is allowed, then sends
 *   SIGINT to the launcher and waits for it to exit.
 * - `expectServerShutdown` allows one product shutdown that ends with exit code 0
 *   and no signal. Its `response` resolves with the matching page response, before
 *   or after the exit. It rejects at once on a matching `requestfailed`, a page
 *   crash or a page close, without waiting for the exit. Its `exit` resolves with
 *   the exit outcome, whatever it is.
 *
 * Any other exit runs the failure path at once. It records the original failure
 * (label, launcher process ID, exit code, signal and the last 4,000 characters of
 * output). It then sends SIGKILL to every owned process of the failed server and
 * of every other registered server, whose exits are then allowed. It repeats `/proc`
 * scans until one finds nothing new. A process that has already ended (ESRCH) is
 * not an error; other kill errors are kept. It never waits on an ordinary stop or
 * for a process to die. A pending shutdown `response` is rejected, and a
 * `ServerFailureError` is thrown on the next tick. Its message starts with the
 * original failure, its first inner error is that failure, and cleanup errors
 * follow. Playwright 1.62's worker fails the running test with it, or reports a
 * worker error between tests. A server that dies while still starting is reported
 * by its startup helper's rejection instead, and its surviving processes are
 * killed too.
 *
 * `captureServerFailure` is for tests of the failure path only: it hands back the
 * failure instead of throwing it, and cleanup still runs. `setServerFailureHooks`
 * and `workerExitCleanupResult` also exist only for unit tests. The worker `exit`
 * hook is a fallback only: when the worker ends, it kills any owned process still
 * left, but it cannot run when the worker itself is killed with SIGKILL.
 */
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPOSITORY_ROOT = resolve(HERE, "../../../..");
export const COMBAT_DEBUGGER_ENTRYPOINT = "scripts/dev/debug_renderer.py";
export const SCRIPTED_DEBUGGER_HARNESS =
  "tests/visual_debugger_scripted_browser_harness.py";
export const DEVCLIENT_BROWSER_HARNESS =
  "tests/visual_debugger_dev_client_browser_harness.py";

export const SERVER_OWNER_VARIABLE = "MARL_BG_E2E_SERVER_OWNER";

const OUTPUT_TAIL_CHARACTERS = 4000;

/**
 * @typedef {object} ServerExitOutcome
 * @property {number | null} exitCode
 * @property {NodeJS.Signals | null} signalCode
 */

/**
 * @typedef {object} OwnedProcessCleanup
 * @property {number[]} killedProcessIds
 * @property {number[]} alreadyExitedProcessIds
 * @property {number[]} failedProcessIds
 * @property {number[]} stillPresentProcessIds
 * @property {Error[]} errors
 */

/**
 * @typedef {object} ServerFailureHooks
 * @property {(error: ServerFailureError) => void} onFailure
 * @property {(pid: number, signal: NodeJS.Signals) => void} killProcess
 */

/**
 * @typedef {object} PendingShutdown
 * @property {ServerExitOutcome | null} exitOutcome
 * @property {(error: Error) => void} rejectResponse
 * @property {(outcome: ServerExitOutcome) => void} resolveExit
 */

/**
 * @typedef {object} ServerRecord
 * @property {string} label
 * @property {string} token
 * @property {number | undefined} launcherPid
 * @property {"starting" | "started"} phase
 * @property {"none" | "requested_stop" | "clean_shutdown" | "failure_cleanup"} allowedExit
 * @property {string} outputTail
 * @property {((error: ServerFailureError) => void) | null} capture
 * @property {PendingShutdown | null} shutdown
 * @property {string} startupCleanupNote
 * @property {string} failureCause
 */

export class ServerFailureError extends AggregateError {
  /**
   * @param {Error} originalFailure
   * @param {OwnedProcessCleanup} cleanup
   * @param {Error[]} extraErrors
   * @param {string} label
   * @param {ServerExitOutcome} outcome
   */
  constructor(originalFailure, cleanup, extraErrors, label, outcome) {
    const secondaryErrors = [...cleanup.errors, ...extraErrors];
    super(
      [originalFailure, ...secondaryErrors],
      `${originalFailure.message}\n${describeCleanup(cleanup)}${
        secondaryErrors.length > 0
          ? `\n${secondaryErrors.length} cleanup error(s) follow the original failure.`
          : ""
      }`,
    );
    this.name = "ServerFailureError";
    this.label = label;
    this.outcome = outcome;
    this.cleanup = cleanup;
  }
}

/** @type {WeakMap<import("node:child_process").ChildProcess, ServerRecord>} */
const serverRecords = new WeakMap();
/** @type {Map<import("node:child_process").ChildProcess, ServerRecord>} */
const liveServers = new Map();
/** @type {Set<string>} */
const issuedTokens = new Set();
/** @type {OwnedProcessCleanup | null} */
let workerExitCleanup = null;

/** @param {number} pid @param {NodeJS.Signals} signal */
function killWithProcessApi(pid, signal) {
  process.kill(pid, signal);
}

/** @param {ServerFailureError} error */
function throwServerFailure(error) {
  // Throw on the next tick, not inside the exit event, so the launcher's other exit
  // listeners and Node's own exit handling still run. The error then reaches
  // Playwright's uncaughtException handler, which fails the running test.
  process.nextTick(() => {
    throw error;
  });
}

/** @type {ServerFailureHooks} */
let failureHooks = { onFailure: throwServerFailure, killProcess: killWithProcessApi };

process.once("exit", runWorkerExitCleanup);

/** @returns {string} */
export function createServerOwnerToken() {
  const token = randomBytes(16).toString("hex");
  issuedTokens.add(token);
  return token;
}

/**
 * @param {string} token
 * @returns {NodeJS.ProcessEnv}
 */
export function serverEnvironment(token) {
  if (!issuedTokens.has(token)) {
    throw new TypeError("Server owner tokens must come from createServerOwnerToken.");
  }
  return { ...process.env, [SERVER_OWNER_VARIABLE]: token };
}

/**
 * @param {import("node:child_process").ChildProcess} child
 * @param {{label: string, token: string}} options
 * @returns {void}
 */
export function registerServer(child, { label, token }) {
  if (!issuedTokens.has(token)) {
    throw new TypeError("Server owner tokens must come from createServerOwnerToken.");
  }
  if (serverRecords.has(child)) {
    throw new TypeError(`${label} is already registered.`);
  }
  /** @type {ServerRecord} */
  const record = {
    label,
    token,
    launcherPid: child.pid,
    phase: "starting",
    allowedExit: "none",
    outputTail: "",
    capture: null,
    shutdown: null,
    startupCleanupNote: "",
    failureCause: "",
  };
  serverRecords.set(child, record);
  liveServers.set(child, record);
  /** @param {unknown} chunk */
  const keepOutput = (chunk) => {
    record.outputTail = `${record.outputTail}${String(chunk)}`.slice(
      -OUTPUT_TAIL_CHARACTERS,
    );
  };
  child.stdout?.on("data", keepOutput);
  child.stderr?.on("data", keepOutput);
  child.once("exit", (exitCode, signalCode) => {
    handleServerExit(child, record, { exitCode, signalCode });
  });
  child.once("error", () => {
    if (child.pid === undefined) {
      liveServers.delete(child);
    }
  });
}

/**
 * @param {import("node:child_process").ChildProcess} child
 * @returns {void}
 */
export function markServerStarted(child) {
  requiredRecord(child).phase = "started";
}

/**
 * @param {import("node:child_process").ChildProcess} child
 * @returns {string}
 */
export function startupCleanupNote(child) {
  return serverRecords.get(child)?.startupCleanupNote ?? "";
}

/**
 * @param {import("node:child_process").ChildProcess} child
 * @returns {string}
 */
export function serverOwnerToken(child) {
  return requiredRecord(child).token;
}

/** @param {number} pid @returns {string | null} */
function processStartTime(pid) {
  const stat = readFileSync(`/proc/${pid}/stat`, "latin1");
  const fields = stat
    .slice(stat.lastIndexOf(")") + 2)
    .trim()
    .split(/\s+/u);
  const start = fields[19];
  return fields[0] !== "Z" && fields[0] !== "X" && /^\d+$/u.test(start ?? "")
    ? start
    : null;
}

/** @param {number} pid @param {Iterable<string>} tokens
 * @returns {string | null} */
function ownedProcessIdentity(pid, tokens) {
  if (pid === process.pid) {
    return null;
  }
  try {
    const start = processStartTime(pid);
    if (start === null || statSync(`/proc/${pid}`).uid !== process.getuid?.()) {
      return null;
    }
    const entries = new Set(
      [...tokens].map((token) => `${SERVER_OWNER_VARIABLE}=${token}`),
    );
    const environment = readFileSync(`/proc/${pid}/environ`, "latin1");
    if (
      !environment.split("\0").some((entry) => entries.has(entry)) ||
      processStartTime(pid) !== start
    ) {
      return null;
    }
    return `${pid}:${start}`;
  } catch {
    // An exited or unreadable process is not proof of ownership.
    return null;
  }
}

/**
 * @param {Iterable<string>} tokens
 * @returns {number[]}
 */
export function findOwnedProcessIds(tokens) {
  const tokenList = [...tokens];
  if (tokenList.length === 0) {
    return [];
  }
  return readdirSync("/proc")
    .filter((name) => /^\d+$/u.test(name))
    .map(Number)
    .filter((pid) => ownedProcessIdentity(pid, tokenList) !== null)
    .sort((left, right) => left - right);
}

/**
 * @param {number} pid
 * @param {Iterable<string>} tokens
 * @param {NodeJS.Signals} signal
 * @param {(pid: number, signal: NodeJS.Signals) => void} [killProcess]
 * @param {string} [expectedIdentity]
 * @returns {boolean}
 */
export function signalOwnedProcess(
  pid,
  tokens,
  signal,
  killProcess = killWithProcessApi,
  expectedIdentity,
) {
  const identity = ownedProcessIdentity(pid, tokens);
  if (
    identity === null ||
    (expectedIdentity !== undefined && identity !== expectedIdentity)
  ) {
    return false;
  }
  killProcess(pid, signal);
  return true;
}

/**
 * @param {Iterable<string>} tokens
 * @param {(pid: number, signal: NodeJS.Signals) => void} [killProcess]
 * @param {number[]} [excludedPids]
 * @returns {OwnedProcessCleanup}
 */
export function killOwnedProcesses(
  tokens,
  killProcess = killWithProcessApi,
  excludedPids = [],
) {
  const tokenList = [...tokens];
  const excluded = new Set(excludedPids);
  /** @type {OwnedProcessCleanup} */
  const cleanup = {
    killedProcessIds: [],
    alreadyExitedProcessIds: [],
    failedProcessIds: [],
    stillPresentProcessIds: [],
    errors: [],
  };
  /** @type {Set<string>} */
  const signalled = new Set();
  for (;;) {
    const present = findOwnedProcessIds(tokenList).filter((pid) => !excluded.has(pid));
    const fresh = present.flatMap((pid) => {
      const identity = ownedProcessIdentity(pid, tokenList);
      return identity !== null && !signalled.has(identity) ? [{ pid, identity }] : [];
    });
    if (fresh.length === 0) {
      cleanup.stillPresentProcessIds = present;
      return cleanup;
    }
    for (const { pid, identity } of fresh) {
      signalled.add(identity);
      try {
        if (signalOwnedProcess(pid, tokenList, "SIGKILL", killProcess, identity)) {
          cleanup.killedProcessIds.push(pid);
        }
      } catch (error) {
        if (errorCode(error) === "ESRCH") {
          cleanup.alreadyExitedProcessIds.push(pid);
        } else {
          cleanup.failedProcessIds.push(pid);
          cleanup.errors.push(
            new Error(
              `SIGKILL of owned process ${pid} failed: ${errorMessage(error)}`,
              { cause: error },
            ),
          );
        }
      }
    }
  }
}

/**
 * @param {Partial<ServerFailureHooks>} hooks
 * @returns {ServerFailureHooks}
 */
export function setServerFailureHooks(hooks) {
  const previous = failureHooks;
  failureHooks = { ...previous, ...hooks };
  return previous;
}

/** @returns {OwnedProcessCleanup | null} */
export function workerExitCleanupResult() {
  return workerExitCleanup;
}

function runWorkerExitCleanup() {
  if (issuedTokens.size === 0) {
    return;
  }
  try {
    workerExitCleanup = killOwnedProcesses(issuedTokens);
  } catch {
    workerExitCleanup = null;
  }
}

/** @param {unknown} error */
function errorCode(error) {
  return typeof error === "object" && error !== null && "code" in error
    ? error.code
    : undefined;
}

/** @param {unknown} error */
function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

/** @param {import("node:child_process").ChildProcess} child */
function requiredRecord(child) {
  const record = serverRecords.get(child);
  if (!record) {
    throw new TypeError("This process was not registered as a browser-test server.");
  }
  return record;
}

/** @param {ServerExitOutcome} outcome */
function describeOutcome(outcome) {
  return `exit code ${outcome.exitCode ?? "none"} and signal ${outcome.signalCode ?? "none"}`;
}

/** @param {ServerRecord["allowedExit"]} allowedExit */
function describeAllowedExit(allowedExit) {
  return allowedExit === "clean_shutdown"
    ? "exit code 0 and no signal, after a product shutdown request"
    : "none; the test did not ask this server to stop";
}

/** @param {OwnedProcessCleanup} cleanup */
function describeCleanup(cleanup) {
  const lines = [
    cleanup.killedProcessIds.length > 0
      ? `Cleanup sent SIGKILL to owned processes ${cleanup.killedProcessIds.join(", ")}.`
      : "Cleanup found no owned process to kill.",
  ];
  if (cleanup.alreadyExitedProcessIds.length > 0) {
    lines.push(
      `Already ended before SIGKILL reached them: ${cleanup.alreadyExitedProcessIds.join(", ")}.`,
    );
  }
  if (cleanup.stillPresentProcessIds.length > 0) {
    lines.push(
      `Still present at the last scan after SIGKILL, not waited on: ${cleanup.stillPresentProcessIds.join(", ")}.`,
    );
  }
  return lines.join("\n");
}

/**
 * @param {ServerRecord} record
 * @returns {string}
 */
function cleanUpAfterStartupDeath(record) {
  let cleanup;
  try {
    cleanup = killOwnedProcesses([record.token], failureHooks.killProcess);
  } catch (error) {
    return `\nCleanup could not scan for this server's surviving processes: ${errorMessage(error)}`;
  }
  if (
    cleanup.killedProcessIds.length === 0 &&
    cleanup.errors.length === 0 &&
    cleanup.stillPresentProcessIds.length === 0
  ) {
    return "";
  }
  const errors = cleanup.errors.map((error) => `\n${error.message}`).join("");
  return `\n${describeCleanup(cleanup)}${errors}`;
}

/**
 * @param {import("node:child_process").ChildProcess} child
 * @param {ServerRecord} record
 * @param {ServerExitOutcome} outcome
 */
function handleServerExit(child, record, outcome) {
  liveServers.delete(child);
  if (record.allowedExit === "failure_cleanup") {
    record.shutdown?.rejectResponse(new Error(record.failureCause));
    record.shutdown?.resolveExit(outcome);
    return;
  }
  if (record.phase === "starting") {
    record.startupCleanupNote = cleanUpAfterStartupDeath(record);
    return;
  }
  if (record.allowedExit === "requested_stop") {
    record.shutdown?.rejectResponse(
      new Error(
        `The test stopped ${record.label} on request before its shutdown response arrived.`,
      ),
    );
    record.shutdown?.resolveExit(outcome);
    return;
  }
  if (
    record.allowedExit === "clean_shutdown" &&
    outcome.exitCode === 0 &&
    outcome.signalCode === null
  ) {
    record.shutdown?.resolveExit(outcome);
    return;
  }
  runFailurePath(record, outcome);
}

/**
 * @param {ServerRecord} record
 * @param {ServerExitOutcome} outcome
 */
function runFailurePath(record, outcome) {
  const originalFailure = new Error(
    `${record.label} (launcher process ${record.launcherPid}) exited during the test ` +
      `with ${describeOutcome(outcome)}. Allowed exit: ${describeAllowedExit(record.allowedExit)}.\n` +
      `Last output (up to ${OUTPUT_TAIL_CHARACTERS} characters):\n${record.outputTail || "(none)"}`,
  );
  const peers = [...liveServers.values()];
  for (const peer of peers) {
    peer.allowedExit = "failure_cleanup";
    peer.failureCause = `Cleanup killed ${peer.label} because ${record.label} failed.`;
  }
  /** @type {OwnedProcessCleanup} */
  let cleanup = {
    killedProcessIds: [],
    alreadyExitedProcessIds: [],
    failedProcessIds: [],
    stillPresentProcessIds: [],
    errors: [],
  };
  /** @type {Error[]} */
  const extraErrors = [];
  try {
    cleanup = killOwnedProcesses(
      [record.token, ...peers.map((peer) => peer.token)],
      failureHooks.killProcess,
    );
  } catch (error) {
    extraErrors.push(
      new Error(`Owned-process cleanup could not scan /proc: ${errorMessage(error)}`, {
        cause: error,
      }),
    );
  }
  const failure = new ServerFailureError(
    originalFailure,
    cleanup,
    extraErrors,
    record.label,
    outcome,
  );
  record.shutdown?.rejectResponse(failure);
  record.shutdown?.resolveExit(outcome);
  if (record.capture) {
    record.capture(failure);
  } else {
    failureHooks.onFailure(failure);
  }
}

/** @param {string[]} extraArgs */
export function combatDebuggerArguments(extraArgs = []) {
  return [
    "run",
    "python",
    "-u",
    COMBAT_DEBUGGER_ENTRYPOINT,
    "--no-open",
    "--port",
    "0",
    ...extraArgs,
  ];
}

/** @param {import("node:child_process").ChildProcess} child */
function hasExited(child) {
  return child.exitCode !== null || child.signalCode !== null;
}

/** @param {import("node:child_process").ChildProcess} child
 * @returns {Promise<void>} */
function waitForExit(child) {
  if (hasExited(child)) {
    return Promise.resolve();
  }
  return new Promise((resolveExit) => {
    child.once("exit", () => resolveExit());
  });
}

/** @param {{scenario?: string, extraArgs?: string[]}} options
 * @returns {Promise<{
 *   process: import("node:child_process").ChildProcess,
 *   url: string,
 * }>} */
export function startDebugger({ scenario, extraArgs = [] } = {}) {
  if (scenario !== undefined && scenario !== "arena_5v5") {
    throw new TypeError(
      "Scripted demonstrations must be launched through the Replay Viewer support.",
    );
  }
  return startDebuggerProcess(combatDebuggerArguments(extraArgs));
}

/** @param {{scenario?: string}} options */
export function startScriptedDebugger({ scenario = "aura_crossfire" } = {}) {
  return startDebuggerProcess([
    "run",
    "python",
    "-u",
    SCRIPTED_DEBUGGER_HARNESS,
    "--port",
    "0",
    "--scenario",
    scenario,
  ]);
}

/** @param {{artifactRoot: string, seedMapCount?: number, seedScenarioCount?: number, offeredSystems?: string[]}} options */
export function startIsolatedDevClient({
  artifactRoot,
  seedMapCount = 0,
  seedScenarioCount = 0,
  offeredSystems = [],
}) {
  if (typeof artifactRoot !== "string" || artifactRoot.length === 0) {
    throw new TypeError("Isolated DevClient tests require an artifact root.");
  }
  if (!Number.isSafeInteger(seedMapCount) || seedMapCount < 0) {
    throw new TypeError("seedMapCount must be a nonnegative safe integer.");
  }
  if (!Number.isSafeInteger(seedScenarioCount) || seedScenarioCount < 0) {
    throw new TypeError("seedScenarioCount must be a nonnegative safe integer.");
  }
  return startDebuggerProcess([
    "run",
    "python",
    "-u",
    DEVCLIENT_BROWSER_HARNESS,
    "--artifact-root",
    artifactRoot,
    "--seed-map-count",
    String(seedMapCount),
    "--seed-scenario-count",
    String(seedScenarioCount),
    ...offeredSystems.flatMap((value) => ["--offer-system", value]),
    "--port",
    "0",
  ]);
}

/**
 * @param {string[]} arguments_
 * @returns {Promise<{
 *   process: import("node:child_process").ChildProcess,
 *   url: string,
 * }>}
 */
function startDebuggerProcess(arguments_) {
  return new Promise((resolveUrl, reject) => {
    const token = createServerOwnerToken();
    const child = spawn("uv", arguments_, {
      cwd: REPOSITORY_ROOT,
      env: serverEnvironment(token),
      stdio: ["ignore", "pipe", "pipe"],
    });
    registerServer(child, {
      label: `Live debugger (${arguments_[3] ?? "uv"})`,
      token,
    });
    let settled = false;
    let stdout = "";
    let stderr = "";

    child.once("error", (error) => {
      if (settled) {
        return;
      }
      settled = true;
      reject(error);
    });
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
      const match = stdout.match(
        /MARL-BattleGrounds DevClient: (http:\/\/127\.0\.0\.1:\d+\/#token=[A-Za-z0-9_-]+)/,
      );
      if (!match || settled) {
        return;
      }
      settled = true;
      markServerStarted(child);
      resolveUrl({ process: child, url: match[1] });
    });
    child.once("exit", (code) => {
      if (!settled) {
        settled = true;
        reject(
          new Error(
            `Debugger exited before startup with code ${code}.\n${stderr}${startupCleanupNote(child)}`,
          ),
        );
      }
    });
  });
}

/**
 * @param {import("node:child_process").ChildProcess | null} child
 * @returns {Promise<void>}
 */
export async function stopDebugger(child) {
  if (!child) {
    return;
  }
  const record = serverRecords.get(child);
  if (record && record.allowedExit !== "failure_cleanup") {
    record.allowedExit = "requested_stop";
  }
  if (hasExited(child)) {
    return;
  }
  child.kill("SIGINT");
  await waitForExit(child);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {import("node:child_process").ChildProcess} child
 * @param {(request: import("@playwright/test").Request) => boolean} isShutdownRequest
 * @returns {{
 *   response: Promise<import("@playwright/test").Response>,
 *   exit: Promise<ServerExitOutcome>,
 * }}
 */
export function expectServerShutdown(page, child, isShutdownRequest) {
  const record = requiredRecord(child);
  if (
    record.phase !== "started" ||
    !liveServers.has(child) ||
    record.allowedExit !== "none"
  ) {
    throw new Error(
      `${record.label} must be started, still running and without another allowed exit.`,
    );
  }
  record.allowedExit = "clean_shutdown";
  /** @type {(response: import("@playwright/test").Response) => void} */
  let resolveResponse = () => {};
  /** @type {(error: Error) => void} */
  let rejectResponse = () => {};
  /** @type {(outcome: ServerExitOutcome) => void} */
  let resolveExitPromise = () => {};
  /** @type {Promise<import("@playwright/test").Response>} */
  const response = new Promise((resolvePromise, rejectPromise) => {
    resolveResponse = resolvePromise;
    rejectResponse = rejectPromise;
  });
  response.catch(() => {});
  /** @type {Promise<ServerExitOutcome>} */
  const exit = new Promise((resolvePromise) => {
    resolveExitPromise = resolvePromise;
  });
  let settled = false;
  const settle = () => {
    settled = true;
    page.off("response", onResponse);
    page.off("requestfailed", onRequestFailed);
    page.off("crash", onPageCrash);
    page.off("close", onPageClose);
  };
  const exitText = () =>
    pending.exitOutcome === null
      ? "not reported yet"
      : describeOutcome(pending.exitOutcome);
  /** @type {PendingShutdown} */
  const pending = {
    exitOutcome: null,
    rejectResponse: (error) => {
      if (!settled) {
        settle();
        rejectResponse(error);
      }
    },
    resolveExit: (outcome) => {
      pending.exitOutcome = outcome;
      resolveExitPromise(outcome);
    },
  };
  /** @param {import("@playwright/test").Response} candidate */
  function onResponse(candidate) {
    if (settled || !isShutdownRequest(candidate.request())) {
      return;
    }
    settle();
    resolveResponse(candidate);
  }
  /** @param {import("@playwright/test").Request} request */
  function onRequestFailed(request) {
    if (settled || !isShutdownRequest(request)) {
      return;
    }
    pending.rejectResponse(
      new Error(
        `The shutdown request ${request.method()} ${request.url()} failed before its ` +
          `response arrived (${request.failure()?.errorText ?? "no error text"}). ` +
          `${record.label} exit: ${exitText()}.`,
      ),
    );
  }
  /** @param {"crash" | "close"} event @param {string} happened */
  function rejectForPageEvent(event, happened) {
    pending.rejectResponse(
      new Error(
        `The page ${happened} before the shutdown response arrived (page "${event}" ` +
          `event). ${record.label} exit: ${exitText()}.`,
      ),
    );
  }
  function onPageCrash() {
    rejectForPageEvent("crash", "crashed");
  }
  function onPageClose() {
    rejectForPageEvent("close", "closed");
  }
  record.shutdown = pending;
  page.on("response", onResponse);
  page.on("requestfailed", onRequestFailed);
  page.on("crash", onPageCrash);
  page.on("close", onPageClose);
  return { response, exit };
}

/**
 * @param {import("node:child_process").ChildProcess} child
 * @returns {Promise<ServerFailureError>}
 */
export function captureServerFailure(child) {
  const record = requiredRecord(child);
  if (!liveServers.has(child)) {
    throw new Error(`${record.label} has already exited.`);
  }
  return new Promise((resolveFailure) => {
    record.capture = resolveFailure;
  });
}
