import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";

import { REPOSITORY_ROOT, stopDebugger } from "./live-debugger.js";

const RECORDING_TEMP_PREFIX = "marl-battlegrounds-recording-e2e-";
const REPLAY_FILE_SUFFIX = ".marlbg-replay.json";
const METRIC_FILE_SUFFIX = ".marlbg-metrics.json";
const STARTUP_TIMEOUT_MS = 60_000;

/**
 * Derive the canonical report companion used by the production replay saver.
 *
 * @param {string} replayPath
 */
export function metricReportPathForReplay(replayPath) {
  if (!replayPath.endsWith(REPLAY_FILE_SUFFIX)) {
    throw new TypeError(`Replay path must end with ${REPLAY_FILE_SUFFIX}.`);
  }
  return `${replayPath.slice(0, -REPLAY_FILE_SUFFIX.length)}${METRIC_FILE_SUFFIX}`;
}

/**
 * Remove only a unique recording directory allocated by this support module.
 *
 * @param {string | null | undefined} outputDirectory
 */
export async function removeRecordingArtifacts(outputDirectory) {
  if (!outputDirectory) {
    return;
  }
  const resolvedDirectory = resolve(outputDirectory);
  if (
    dirname(resolvedDirectory) !== resolve(tmpdir()) ||
    !basename(resolvedDirectory).startsWith(RECORDING_TEMP_PREFIX)
  ) {
    throw new Error("Refusing to remove a directory outside the recording E2E prefix.");
  }
  await rm(resolvedDirectory, { force: true, recursive: true });
}

/**
 * Start the production CLI with a structurally preflighted, initially absent
 * recording destination. Startup failures synchronously reap the child and
 * remove the unique directory before they escape.
 *
 * @param {{stem?: string}} options
 * @returns {Promise<{
 *   process: import("node:child_process").ChildProcess,
 *   url: string,
 *   outputDirectory: string,
 *   replayPath: string,
 *   metricReportPath: string,
 * }>}
 */
export async function startRecordingDebugger({ stem = "episode" } = {}) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(stem)) {
    throw new TypeError("Recording E2E stems must be safe filename components.");
  }
  const outputDirectory = await mkdtemp(join(tmpdir(), RECORDING_TEMP_PREFIX));
  const replayPath = join(outputDirectory, `${stem}${REPLAY_FILE_SUFFIX}`);
  const metricReportPath = metricReportPathForReplay(replayPath);
  const child = spawn(
    "uv",
    [
      "run",
      "python",
      "-u",
      "scripts/dev/debug_renderer.py",
      "--no-open",
      "--port",
      "0",
      "--record-replay",
      replayPath,
    ],
    {
      cwd: REPOSITORY_ROOT,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  try {
    const url = await new Promise((resolveUrl, reject) => {
      let settled = false;
      let stdout = "";
      let stderr = "";
      /** @param {() => void} callback */
      const finish = (callback) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timeout);
        callback();
      };
      const timeout = setTimeout(() => {
        finish(() => {
          child.kill("SIGTERM");
          reject(new Error(`Recording debugger startup timed out.\n${stderr}`));
        });
      }, STARTUP_TIMEOUT_MS);

      child.once("error", (error) => finish(() => reject(error)));
      child.stderr?.setEncoding("utf8");
      child.stderr?.on("data", (chunk) => {
        stderr += String(chunk);
      });
      child.stdout?.setEncoding("utf8");
      child.stdout?.on("data", (chunk) => {
        stdout += String(chunk);
        const match = stdout.match(
          /MARL-BattleGrounds Combat Debugger: (http:\/\/127\.0\.0\.1:\d+\/#token=[A-Za-z0-9_-]+)/,
        );
        if (match) {
          finish(() => resolveUrl(match[1]));
        }
      });
      child.once("exit", (code, signal) => {
        finish(() =>
          reject(
            new Error(
              `Recording debugger exited before startup with code ${code} and signal ${signal}.\n${stderr}`,
            ),
          ),
        );
      });
    });
    if (typeof url !== "string") {
      throw new TypeError("Recording debugger returned an invalid launch URL.");
    }
    return {
      process: child,
      url,
      outputDirectory,
      replayPath,
      metricReportPath,
    };
  } catch (error) {
    /** @type {unknown[]} */
    const cleanupErrors = [];
    try {
      await stopDebugger(child);
    } catch (cleanupError) {
      cleanupErrors.push(cleanupError);
    }
    try {
      await removeRecordingArtifacts(outputDirectory);
    } catch (cleanupError) {
      cleanupErrors.push(cleanupError);
    }
    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        [error, ...cleanupErrors],
        "Recording debugger startup and cleanup both failed.",
      );
    }
    throw error;
  }
}

/**
 * Stop one recording debugger and remove its exact temporary directory.
 *
 * @param {Awaited<ReturnType<typeof startRecordingDebugger>> | null} started
 */
export async function stopRecordingDebugger(started) {
  if (!started) {
    return;
  }
  /** @type {unknown[]} */
  const cleanupErrors = [];
  try {
    await stopDebugger(started.process);
  } catch (error) {
    cleanupErrors.push(error);
  }
  try {
    await removeRecordingArtifacts(started.outputDirectory);
  } catch (error) {
    cleanupErrors.push(error);
  }
  if (cleanupErrors.length > 0) {
    throw new AggregateError(cleanupErrors, "Recording E2E cleanup failed.");
  }
}

/**
 * Wait for a debugger child to terminate without sleeping or losing the exact
 * exit code/signal pair. This is used to prove Exit and Ctrl-C flush durable
 * recording bytes before process teardown.
 *
 * @param {import("node:child_process").ChildProcess} child
 * @param {number} [timeoutMs]
 */
export async function waitForRecordingDebuggerExit(child, timeoutMs = 30_000) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return { exitCode: child.exitCode, signalCode: child.signalCode };
  }
  return new Promise((resolveExit, reject) => {
    const timeout = setTimeout(() => {
      child.off("exit", onExit);
      reject(new Error("Recording debugger did not exit within the deadline."));
    }, timeoutMs);
    /** @param {number | null} exitCode @param {NodeJS.Signals | null} signalCode */
    const onExit = (exitCode, signalCode) => {
      clearTimeout(timeout);
      resolveExit({ exitCode, signalCode });
    };
    child.once("exit", onExit);
  });
}

/**
 * Create the deliberate post-preflight target race used by recovery tests.
 *
 * @param {Awaited<ReturnType<typeof startRecordingDebugger>>} started
 * @param {Uint8Array} [sentinel]
 */
export async function createReplayTargetRace(
  started,
  sentinel = new TextEncoder().encode("recording-e2e-target-race"),
) {
  await writeFile(started.replayPath, sentinel, { flag: "wx" });
  return sentinel;
}

/**
 * Read a materialized artifact without changing or reserializing its bytes.
 *
 * @param {string} path
 */
export async function readJsonArtifact(path) {
  const bytes = await readFile(path);
  return { bytes, value: JSON.parse(bytes.toString("utf8")) };
}
