import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPOSITORY_ROOT = resolve(HERE, "../../../..");
export const DEBUGGER_STOP_TIMEOUT_MS = 30_000;
export const COMBAT_DEBUGGER_ENTRYPOINT = "scripts/dev/debug_renderer.py";
export const SCRIPTED_DEBUGGER_HARNESS =
  "tests/visual_debugger_scripted_browser_harness.py";
export const DEVCLIENT_BROWSER_HARNESS =
  "tests/visual_debugger_dev_client_browser_harness.py";

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

/**
 * Node keeps exitCode null when a child exits because of a signal, so both
 * fields participate in the lifecycle check.
 *
 * @param {import("node:child_process").ChildProcess} child
 */
function hasExited(child) {
  return child.exitCode !== null || child.signalCode !== null;
}

/**
 * Wait a bounded period for one child process to emit its exit event.
 *
 * @param {import("node:child_process").ChildProcess} child
 * @param {number} timeoutMs
 * @returns {Promise<boolean>}
 */
function waitForExit(child, timeoutMs) {
  if (hasExited(child)) {
    return Promise.resolve(true);
  }
  return new Promise((resolveExit) => {
    /** @param {boolean} exited */
    const finish = (exited) => {
      clearTimeout(timeout);
      child.off("exit", onExit);
      resolveExit(exited);
    };
    const onExit = () => finish(true);
    const timeout = setTimeout(() => finish(false), timeoutMs);
    child.once("exit", onExit);
  });
}

/**
 * Start the real loopback debugger for browser integration tests.
 *
 * @param {{scenario?: string, extraArgs?: string[]}} options
 * @returns {Promise<{
 *   process: import("node:child_process").ChildProcess,
 *   url: string,
 * }>}
 */
export function startDebugger({ scenario, extraArgs = [] } = {}) {
  if (scenario !== undefined && scenario !== "arena_5v5") {
    throw new TypeError(
      "Scripted demonstrations must be launched through the Replay Viewer support.",
    );
  }
  return startDebuggerProcess(combatDebuggerArguments(extraArgs));
}

/**
 * Start a registered scripted DebuggerService used only by browser causal
 * tests. This bypasses neither the real HTTP server nor service logic, and
 * deliberately adds no scenario option to the public launcher.
 *
 * @param {{scenario?: string}} options
 */
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

/**
 * Start the public DevClient launcher path with saved drafts isolated in
 * one caller-owned temporary directory. Reusing the directory across process
 * restarts proves persisted discovery without touching developer assets.
 *
 * @param {{artifactRoot: string}} options
 */
export function startIsolatedDevClient({ artifactRoot }) {
  if (typeof artifactRoot !== "string" || artifactRoot.length === 0) {
    throw new TypeError("Isolated DevClient tests require an artifact root.");
  }
  return startDebuggerProcess([
    "run",
    "python",
    "-u",
    DEVCLIENT_BROWSER_HARNESS,
    "--artifact-root",
    artifactRoot,
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
    const child = spawn("uv", arguments_, {
      cwd: REPOSITORY_ROOT,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let settled = false;
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Debugger startup timed out.\n${stderr}`));
    }, 60_000);

    child.once("error", (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
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
      clearTimeout(timeout);
      resolveUrl({ process: child, url: match[1] });
    });
    child.once("exit", (code) => {
      clearTimeout(timeout);
      if (!settled) {
        settled = true;
        reject(
          new Error(`Debugger exited before startup with code ${code}.\n${stderr}`),
        );
      }
    });
  });
}

/**
 * Stop one loopback debugger process without relying on browser-close signals.
 *
 * @param {import("node:child_process").ChildProcess | null} child
 */
export async function stopDebugger(child) {
  if (!child || hasExited(child)) {
    return;
  }
  child.kill("SIGINT");
  if (await waitForExit(child, DEBUGGER_STOP_TIMEOUT_MS)) {
    return;
  }

  child.kill("SIGKILL");
  const cleanupCompleted = await waitForExit(child, DEBUGGER_STOP_TIMEOUT_MS);
  throw new Error(
    `Debugger did not exit within ${DEBUGGER_STOP_TIMEOUT_MS} ms after SIGINT; ` +
      `SIGKILL cleanup ${cleanupCompleted ? "completed" : "did not complete"}.`,
  );
}
