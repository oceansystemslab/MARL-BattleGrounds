/**
 * @file Launch local replay-viewer test sessions and expose their URLs/process
 * lifecycle for browser checks. Replay export and viewer startup wait with no
 * time limit, and frame waits set no limit of their own. A process that fails
 * to start or exits early still rejects at once.
 *
 * `startReplayViewer` checks its options with `replayViewerArguments`, which throws
 * a `TypeError` for invalid options before any process starts, for example when not
 * exactly one of `replayPath`, `sampleReplay` or `scenario` is given.
 *
 * Each viewer follows the server lifecycle that live-debugger.js owns. It gets a
 * fresh owner token in `MARL_BG_E2E_SERVER_OWNER` and is registered at spawn. A
 * process is owned only when it runs as this user and its `/proc/<pid>/environ`
 * holds that exact token, so this works on Linux only. After startup,
 * `stopDebugger` marks a requested stop. `expectServerShutdown` allows only exit
 * code 0 with no signal, and rejects its `response` at once on a matching
 * `requestfailed`, a page crash or a page close. Any other exit sends SIGKILL to
 * every owned process through repeated `/proc` scans, without waiting on an
 * ordinary stop, and then fails the running test with the original failure first.
 * A death during startup rejects the startup promise instead. The test-only
 * `captureServerFailure` and the fallback worker exit hook live in
 * live-debugger.js.
 */
import { execFile, spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { promisify } from "node:util";

import {
  createServerOwnerToken,
  markServerStarted,
  REPOSITORY_ROOT,
  registerServer,
  serverEnvironment,
  startupCleanupNote,
  stopDebugger,
} from "./live-debugger.js";

const execFileAsync = promisify(execFile);
export const REPLAY_VIEWER_ENTRYPOINT = "scripts/dev/replay_viewer.py";

/** @returns {Promise<{
 *   outputDirectory: string,
 *   complete: string,
 *   partial: string,
 *   shared: string,
 *   tdm: string,
 *   tdmShared: string,
 *   corpseNoShared: string,
 *   corpseShared: string,
 *   missingMetric: string,
 * }>} */
export async function exportReplayArtifacts() {
  const { outputDirectory, payload } = await exportReplayPaths();
  try {
    if (
      typeof payload.complete !== "string" ||
      typeof payload.partial !== "string" ||
      typeof payload.shared !== "string" ||
      typeof payload.tdm !== "string" ||
      typeof payload.tdm_shared !== "string" ||
      typeof payload.corpse_no_shared !== "string" ||
      typeof payload.corpse_shared !== "string" ||
      typeof payload.missing_metric !== "string"
    ) {
      throw new TypeError("Replay exporter returned an invalid path manifest.");
    }
    return {
      outputDirectory,
      complete: payload.complete,
      partial: payload.partial,
      shared: payload.shared,
      tdm: payload.tdm,
      tdmShared: payload.tdm_shared,
      corpseNoShared: payload.corpse_no_shared,
      corpseShared: payload.corpse_shared,
      missingMetric: payload.missing_metric,
    };
  } catch (error) {
    await removeReplayArtifacts(outputDirectory);
    throw error;
  }
}

export async function exportReplayAcceptanceArtifacts() {
  const { outputDirectory, payload } = await exportReplayPaths(["--c7-acceptance"]);
  if (
    !Array.isArray(payload.scenarios) ||
    payload.scenarios.length !== 8 ||
    !payload.scenarios.every(
      (/** @type {unknown} */ path) => typeof path === "string",
    ) ||
    typeof payload.dense !== "string"
  ) {
    await removeReplayArtifacts(outputDirectory);
    throw new TypeError("Replay acceptance exporter returned an invalid manifest.");
  }
  return {
    outputDirectory,
    scenarios: /** @type {string[]} */ (payload.scenarios),
    dense: payload.dense,
  };
}

/**
 * Export the one-tick Red Zone metric replay and return its path.
 *
 * The Python exporter (`--red-zone`) writes a current version-4 DevClient capture
 * at Red Zone depth 5.0: Team A kills agents 5 and 6 inside Team B's Red Zone and
 * Team B kills agent 3 outside Team A's Red Zone, so the scores are 4 and 1. It
 * fails before writing a manifest unless those deaths and scores happen. The
 * caller owns the returned temporary directory and removes it with
 * `removeReplayArtifacts`. The export waits with no time limit.
 *
 * @returns {Promise<{ outputDirectory: string, redZone: string }>}
 * @throws {TypeError} When the exporter's manifest has no `red_zone` path; the
 *   temporary directory is removed first.
 */
export async function exportRedZoneReplayArtifacts() {
  const { outputDirectory, payload } = await exportReplayPaths(["--red-zone"]);
  if (typeof payload.red_zone !== "string") {
    await removeReplayArtifacts(outputDirectory);
    throw new TypeError("Red Zone replay exporter returned an invalid manifest.");
  }
  return { outputDirectory, redZone: payload.red_zone };
}

/** @param {string[]} args */
async function exportReplayPaths(args = []) {
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
        ...args,
        "--output-directory",
        outputDirectory,
      ],
      {
        cwd: REPOSITORY_ROOT,
        env: process.env,
        maxBuffer: 1024 * 1024,
      },
    );
    const payload = JSON.parse(result.stdout.trim());
    return { outputDirectory, payload };
  } catch (error) {
    await rm(outputDirectory, { force: true, recursive: true });
    throw error;
  }
}

/** @param {string | null | undefined} outputDirectory */
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

/** @param {{
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
 * @returns {string[]} */
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

/** @param {Parameters<typeof replayViewerArguments>[0]} options
 * @returns {Promise<{
 *   process: import("node:child_process").ChildProcess,
 *   url: string,
 * }>} */
export function startReplayViewer(options) {
  const replayArguments = replayViewerArguments(options);
  return new Promise((resolveUrl, reject) => {
    const token = createServerOwnerToken();
    const child = spawn("uv", replayArguments, {
      cwd: REPOSITORY_ROOT,
      env: serverEnvironment(token),
    });
    registerServer(child, {
      label: `Replay viewer (${replayArguments[4]} ${replayArguments[5]})`,
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
        /MARL-BattleGrounds Replay Viewer: (http:\/\/127\.0\.0\.1:\d+\/#token=[A-Za-z0-9_-]+)/,
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
            `Replay viewer exited before startup with code ${code}.\n${stderr}${startupCleanupNote(child)}`,
          ),
        );
      }
    });
  });
}

/** @param {import("@playwright/test").Page} page
 * @param {"/api/frame" | "/api/replay/timeline"} path
 * @returns {Promise<Record<string, any>>} */
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

/** @param {import("@playwright/test").Page} page
 * @param {number} frameIndex */
export async function expectReplayFrameIndex(page, frameIndex) {
  const expected = String(frameIndex);
  await page.locator("#replay-frame-slider").waitFor({ state: "visible" });
  await page.waitForFunction(
    ([selector, value]) => {
      const slider = document.querySelector(selector);
      return slider instanceof HTMLInputElement && slider.value === value;
    },
    ["#replay-frame-slider", expected],
  );
}

export { stopDebugger };
