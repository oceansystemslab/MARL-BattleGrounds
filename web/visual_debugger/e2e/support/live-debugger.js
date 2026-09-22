/**
 * @file Launch and control a local live debugger for browser tests, with fixed inputs
 * and explicit process cleanup. Startup waits for the launch URL and shutdown sends
 * SIGINT and waits for exit, both with no time limit.
 */
import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPOSITORY_ROOT = resolve(HERE, "../../../..");
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

/** @param {{artifactRoot: string, seedMapCount?: number, seedScenarioCount?: number}} options */
export function startIsolatedDevClient({
  artifactRoot,
  seedMapCount = 0,
  seedScenarioCount = 0,
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
    "--port",
    "0",
  ]);
}

/** @param {string[]} arguments_
 * @returns {Promise<{
 *   process: import("node:child_process").ChildProcess,
 *   url: string,
 * }>} */
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
      resolveUrl({ process: child, url: match[1] });
    });
    child.once("exit", (code) => {
      if (!settled) {
        settled = true;
        reject(
          new Error(`Debugger exited before startup with code ${code}.\n${stderr}`),
        );
      }
    });
  });
}

/** @param {import("node:child_process").ChildProcess | null} child */
export async function stopDebugger(child) {
  if (!child || hasExited(child)) {
    return;
  }
  child.kill("SIGINT");
  await waitForExit(child);
}
