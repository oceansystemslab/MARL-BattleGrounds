import { execFile, spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { promisify } from "node:util";

import {
  DEBUGGER_STOP_TIMEOUT_MS,
  REPOSITORY_ROOT,
  stopDebugger,
} from "./live-debugger.js";

const execFileAsync = promisify(execFile);
const STARTUP_TIMEOUT_MS = 60_000;
export const REPLAY_VIEWER_ENTRYPOINT = "scripts/dev/replay_viewer.py";

/**
 * Generate real canonical artifacts through the public Python capture,
 * observer, replay, and persistence APIs.
 *
 * @returns {Promise<{
 *   outputDirectory: string,
 *   complete: string,
 *   partial: string,
 *   shared: string,
 *   missingMetric: string,
 * }>}
 */
export async function exportReplayArtifacts() {
  const outputDirectory = await mkdtemp(
    join(tmpdir(), "marl-battlegrounds-replay-e2e-"),
  );
  try {
    const result = await execFileAsync(
      "uv",
      [
        "run",
        "python",
        "-m",
        "tests.export_visual_debugger_replay_artifacts",
        "--output-directory",
        outputDirectory,
      ],
      {
        cwd: REPOSITORY_ROOT,
        env: process.env,
        maxBuffer: 1024 * 1024,
        timeout: 120_000,
      },
    );
    const payload = JSON.parse(result.stdout.trim());
    if (
      typeof payload.complete !== "string" ||
      typeof payload.partial !== "string" ||
      typeof payload.shared !== "string" ||
      typeof payload.missing_metric !== "string"
    ) {
      throw new TypeError("Replay exporter returned an invalid path manifest.");
    }
    return {
      outputDirectory,
      complete: payload.complete,
      partial: payload.partial,
      shared: payload.shared,
      missingMetric: payload.missing_metric,
    };
  } catch (error) {
    await rm(outputDirectory, { force: true, recursive: true });
    throw error;
  }
}

/**
 * Remove only the unique temporary directory created by exportReplayArtifacts.
 *
 * @param {string | null | undefined} outputDirectory
 */
export async function removeReplayArtifacts(outputDirectory) {
  if (!outputDirectory) {
    return;
  }
  const resolvedDirectory = resolve(outputDirectory);
  if (
    dirname(resolvedDirectory) !== resolve(tmpdir()) ||
    !basename(resolvedDirectory).startsWith("marl-battlegrounds-replay-e2e-")
  ) {
    throw new Error("Refusing to remove a directory outside the replay E2E prefix.");
  }
  await rm(resolvedDirectory, { force: true, recursive: true });
}

/**
 * Start the production CLI against one canonical file, checked sample, or
 * isolated scripted-scenario materialization.
 *
 * @param {{
 *   replayPath?: string,
 *   sampleReplay?: string,
 *   scenario?: string,
 *   seed?: number,
 *   includeStress?: boolean,
 *   frameIndex?: number,
 *   view?: "researcher" | "pov",
 *   povSlot?: number,
 *   preset?: "presentation" | "analysis" | "debug",
 *   ranges?: boolean,
 * }} options
 * @returns {string[]}
 */
export function replayViewerArguments({
  replayPath,
  sampleReplay,
  scenario,
  seed,
  includeStress,
  frameIndex,
  view,
  povSlot,
  preset,
  ranges,
}) {
  const selectors = [replayPath, sampleReplay, scenario].filter(
    (value) => typeof value === "string",
  );
  if (selectors.length !== 1) {
    throw new TypeError(
      "Replay viewer startup requires exactly one replayPath, sampleReplay, or scenario.",
    );
  }
  const replayValue =
    typeof replayPath === "string"
      ? replayPath
      : typeof sampleReplay === "string"
        ? sampleReplay
        : scenario;
  if (typeof replayValue !== "string") {
    throw new TypeError("Replay viewer startup requires a replay selector.");
  }
  /** @type {string[]} */
  const replayArguments = [
    typeof replayPath === "string"
      ? "--replay"
      : typeof sampleReplay === "string"
        ? "--sample-replay"
        : "--scenario",
    replayValue,
    "--no-open",
    "--port",
    "0",
  ];
  if (seed !== undefined) {
    if (typeof scenario !== "string" || !Number.isInteger(seed)) {
      throw new TypeError("Replay scenario seed must be an integer.");
    }
    replayArguments.push("--seed", String(seed));
  }
  if (includeStress === true) {
    if (typeof scenario !== "string") {
      throw new TypeError("includeStress is available only for replay scenarios.");
    }
    replayArguments.push("--include-stress");
  }
  if (Number.isInteger(frameIndex)) {
    replayArguments.push("--frame-index", String(frameIndex));
  }
  if (view) {
    replayArguments.push("--view", view);
  }
  if (Number.isInteger(povSlot)) {
    replayArguments.push("--pov-slot", String(povSlot));
  }
  if (preset) {
    replayArguments.push("--preset", preset);
  }
  if (typeof ranges === "boolean") {
    replayArguments.push(ranges ? "--ranges" : "--no-ranges");
  }
  return ["run", "python", "-u", REPLAY_VIEWER_ENTRYPOINT, ...replayArguments];
}

/**
 * @param {Parameters<typeof replayViewerArguments>[0]} options
 * @returns {Promise<{
 *   process: import("node:child_process").ChildProcess,
 *   url: string,
 * }>}
 */
export function startReplayViewer(options) {
  const replayArguments = replayViewerArguments(options);
  return new Promise((resolveUrl, reject) => {
    const child = spawn("uv", replayArguments, {
      cwd: REPOSITORY_ROOT,
      env: process.env,
    });
    let settled = false;
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      child.kill("SIGTERM");
      reject(new Error(`Replay viewer startup timed out.\n${stderr}`));
    }, STARTUP_TIMEOUT_MS);

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
        /MARL-BattleGrounds Replay Viewer: (http:\/\/127\.0\.0\.1:\d+\/#token=[A-Za-z0-9_-]+)/,
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
          new Error(
            `Replay viewer exited before startup with code ${code}.\n${stderr}`,
          ),
        );
      }
    });
  });
}

/**
 * Read one authenticated replay route from inside the real browser origin.
 *
 * @param {import("@playwright/test").Page} page
 * @param {"/api/frame" | "/api/replay/timeline"} path
 * @returns {Promise<Record<string, any>>}
 */
async function authenticatedReplayGet(page, path) {
  return page.evaluate(async (requestPath) => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Replay capability token is unavailable.");
    }
    const response = await fetch(requestPath, {
      cache: "no-store",
      credentials: "omit",
      headers: { "X-MARL-Debugger-Token": token },
      redirect: "error",
    });
    if (!response.ok) {
      throw new Error(`${requestPath} failed with HTTP ${response.status}.`);
    }
    return response.json();
  }, path);
}

/** @param {import("@playwright/test").Page} page */
export function currentReplayFrame(page) {
  return authenticatedReplayGet(page, "/api/frame");
}

/** @param {import("@playwright/test").Page} page */
export function currentReplayTimeline(page) {
  return authenticatedReplayGet(page, "/api/replay/timeline");
}

/**
 * Wait for the browser to install a replay frame at one exact cursor index.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} frameIndex
 */
export async function expectReplayFrameIndex(page, frameIndex) {
  const expected = String(frameIndex);
  await page
    .locator("#replay-frame-slider")
    .waitFor({ state: "visible", timeout: 30_000 });
  await page.waitForFunction(
    ([selector, value]) => {
      const slider = document.querySelector(selector);
      return slider instanceof HTMLInputElement && slider.value === value;
    },
    ["#replay-frame-slider", expected],
    { timeout: 30_000 },
  );
}

export { DEBUGGER_STOP_TIMEOUT_MS, stopDebugger };
