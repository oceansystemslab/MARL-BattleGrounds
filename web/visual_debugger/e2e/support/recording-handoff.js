/**
 * @file Provide browser-test setup and inspection for live-recording handoff to an
 * immutable replay. Waits for debugger startup and exit have no time limit.
 *
 * `startRecordingDebugger` records one episode into a fresh temporary folder and
 * follows the server lifecycle that live-debugger.js owns. Each recording debugger
 * gets a fresh owner token in `MARL_BG_E2E_SERVER_OWNER` and is registered at
 * spawn. A process is owned only when it runs as this user and its
 * `/proc/<pid>/environ` holds that exact token, so this works on Linux only. After
 * startup, `stopDebugger` (also called by `stopRecordingDebugger`, which then
 * removes the folder) marks a requested stop. `expectServerShutdown` allows only
 * exit code 0 with no signal, and rejects its `response` at once on a matching
 * `requestfailed`, a page crash or a page close. Any other exit sends SIGKILL to
 * every owned process through repeated `/proc` scans, without waiting on an
 * ordinary stop, and then fails the running test with the original failure first.
 * A death during startup rejects the startup promise instead. The server is then
 * stopped and the folder removed; if that cleanup fails too, an `AggregateError`
 * lists the startup error first. The test-only `captureServerFailure` and the
 * fallback worker exit hook live in live-debugger.js.
 */
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";

import {
  createServerOwnerToken,
  markServerStarted,
  REPOSITORY_ROOT,
  registerServer,
  serverEnvironment,
  startupCleanupNote,
  stopDebugger,
} from "./live-debugger.js";

const RECORDING_TEMP_PREFIX = "marl-battlegrounds-recording-e2e-";
const REPLAY_FILE_SUFFIX = ".marlbg-replay.json";
const METRIC_FILE_SUFFIX = ".marlbg-metrics.json";

/** @param {string} replayPath */
export function metricReportPathForReplay(replayPath) {
  if (!replayPath.endsWith(REPLAY_FILE_SUFFIX)) {
    throw new TypeError(`Replay path must end with ${REPLAY_FILE_SUFFIX}.`);
  }
  return `${replayPath.slice(0, -REPLAY_FILE_SUFFIX.length)}${METRIC_FILE_SUFFIX}`;
}

/** @param {string | null | undefined} outputDirectory */
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

/** @param {{stem?: string}} options
 * @returns {Promise<{
 *   process: import("node:child_process").ChildProcess,
 *   url: string,
 *   outputDirectory: string,
 *   replayPath: string,
 *   metricReportPath: string,
 * }>} */
export async function startRecordingDebugger({ stem = "episode" } = {}) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(stem)) {
    throw new TypeError("Recording E2E stems must be safe filename components.");
  }
  const outputDirectory = await mkdtemp(join(tmpdir(), RECORDING_TEMP_PREFIX));
  const replayPath = join(outputDirectory, `${stem}${REPLAY_FILE_SUFFIX}`);
  const metricReportPath = metricReportPathForReplay(replayPath);
  const token = createServerOwnerToken();
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
      env: serverEnvironment(token),
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  registerServer(child, { label: `Recording debugger (${replayPath})`, token });

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
        callback();
      };

      child.once("error", (error) => finish(() => reject(error)));
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
        if (match) {
          finish(() => {
            markServerStarted(child);
            resolveUrl(match[1]);
          });
        }
      });
      child.once("exit", (code, signal) => {
        finish(() =>
          reject(
            new Error(
              `Recording debugger exited before startup with code ${code} and signal ${signal}.\n${stderr}${startupCleanupNote(child)}`,
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

/** @param {Awaited<ReturnType<typeof startRecordingDebugger>> | null} started */
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

/** @param {Awaited<ReturnType<typeof startRecordingDebugger>>} started
 * @param {Uint8Array} [sentinel] */
export async function createReplayTargetRace(
  started,
  sentinel = new TextEncoder().encode("recording-e2e-target-race"),
) {
  await writeFile(started.replayPath, sentinel, { flag: "wx" });
  return sentinel;
}

/** @param {string} path */
export async function readJsonArtifact(path) {
  const bytes = await readFile(path);
  return { bytes, value: JSON.parse(bytes.toString("utf8")) };
}
