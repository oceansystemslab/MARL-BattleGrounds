import { execFile } from "node:child_process";
import { promisify } from "node:util";

import { normalizeLiveDebuggerFrameV2 } from "../../src/frame-normalizer.js";
import {
  joinReplayFrameAndTimeline,
  normalizeReplayTimelineV1,
  normalizeReplayViewerFrameV1,
} from "../../src/replay-frame-normalizer.js";
import { REPOSITORY_ROOT } from "./live-debugger.js";

const execFileAsync = promisify(execFile);

/**
 * Load one registered synthetic renderer fixture through the narrow Python
 * exporter. This helper is test-only and never mutates a live debugger session.
 *
 * @param {string} name
 * @returns {Promise<Record<string, any>>}
 */
export async function loadRendererFixture(name) {
  const { stdout, stderr } = await execFileAsync(
    "uv",
    [
      "run",
      "--quiet",
      "python",
      "-m",
      "scripts.dev.visual_debugger.export_renderer_fixture",
      name,
    ],
    {
      cwd: REPOSITORY_ROOT,
      encoding: "utf8",
      env: {
        ...process.env,
        JAX_PLATFORMS: "cpu",
        UV_CACHE_DIR:
          process.env.UV_CACHE_DIR ?? "/tmp/marl-battlegrounds-renderer-fixture-uv",
      },
      maxBuffer: 8 * 1024 * 1024,
    },
  );
  if (stderr.trim()) {
    throw new Error(`Renderer fixture exporter wrote to stderr:\n${stderr}`);
  }
  const parsed = JSON.parse(stdout);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Renderer fixture exporter returned a non-object payload.");
  }
  return parsed;
}

/**
 * Load the one Python-mutated catalog payload shared by the host and browser
 * propagation proofs. Unlike the synthetic renderer vocabulary, this envelope
 * is rebuilt from the validated evaluation catalog before it crosses JSON.
 *
 * @returns {Promise<{live_frame: Record<string, any>, expected: Record<string, number>}>}
 */
export async function loadCatalogPropagationFixture() {
  const { stdout, stderr } = await execFileAsync(
    "uv",
    ["run", "--quiet", "python", "-m", "tests.catalog_propagation_fixture"],
    {
      cwd: REPOSITORY_ROOT,
      encoding: "utf8",
      env: {
        ...process.env,
        JAX_PLATFORMS: "cpu",
        UV_CACHE_DIR:
          process.env.UV_CACHE_DIR ?? "/tmp/marl-battlegrounds-renderer-fixture-uv",
      },
      maxBuffer: 8 * 1024 * 1024,
    },
  );
  if (stderr.trim()) {
    throw new Error(`Catalog propagation exporter wrote to stderr:\n${stderr}`);
  }
  const parsed = JSON.parse(stdout);
  if (
    !parsed ||
    typeof parsed !== "object" ||
    Array.isArray(parsed) ||
    !parsed.live_frame ||
    typeof parsed.live_frame !== "object" ||
    Array.isArray(parsed.live_frame) ||
    !parsed.expected ||
    typeof parsed.expected !== "object" ||
    Array.isArray(parsed.expected)
  ) {
    throw new Error("Catalog propagation exporter returned an invalid payload.");
  }
  normalizeLiveDebuggerFrameV2(parsed.live_frame);
  return parsed;
}

/**
 * Return the exact Python-produced outbound live-frame envelope after proving
 * that the production browser normalizer accepts it. The returned object is
 * intentionally not decorated with presentation aliases.
 *
 * @param {Record<string, any>} fixture
 * @returns {Record<string, any>}
 */
export function syntheticDebuggerWireFrame(fixture) {
  const scene = fixture?.scene;
  if (!scene || typeof scene !== "object" || Array.isArray(scene)) {
    throw new Error("Renderer fixture is missing its scene.");
  }
  const liveFrame = fixture.live_frame;
  if (!liveFrame || typeof liveFrame !== "object" || Array.isArray(liveFrame)) {
    throw new Error("Renderer fixture is missing its validated live frame.");
  }
  if (
    fixture.audience === "researcher" &&
    (scene.schema_version !== 2 ||
      scene.audience !== "researcher" ||
      liveFrame.frame_kind !== "researcher_live_debugger")
  ) {
    throw new Error("Researcher fixture must contain exact V2 presentation roots.");
  }
  if (
    fixture.audience === "agent_pov" &&
    (scene.schema_version !== 1 ||
      liveFrame.frame_kind !== "actor_pov_live_debugger" ||
      liveFrame.projection?.scene?.pov_frame_id !== scene.pov_frame_id)
  ) {
    throw new Error("POV fixture must contain exact recipient presentation roots.");
  }
  normalizeLiveDebuggerFrameV2(liveFrame);
  return liveFrame;
}

/**
 * Build a presentation-only view of the Python-validated production V2 fixture.
 * The returned object contains browser-owned aliases such as `scene` and
 * `event_batch`; it is for local assertions only and must never be serialized
 * into an intercepted HTTP response. Use `syntheticDebuggerWireFrame` for that.
 *
 * @param {Record<string, any>} fixture
 * @returns {Record<string, any>}
 */
export function syntheticDebuggerPresentationFrame(fixture) {
  return normalizeLiveDebuggerFrameV2(syntheticDebuggerWireFrame(fixture));
}

/**
 * Return one fixture-owned live/replay transport pair after both production
 * browser boundaries and the replay frame/timeline join accept it. These are
 * existing protocol envelopes around one synthetic authorized projection;
 * this helper does not fabricate a replay from a live frame.
 *
 * @param {Record<string, any>} fixture
 * @returns {{
 *   audience: "researcher" | "agent_pov",
 *   liveFrame: Record<string, any>,
 *   replayFrame: Record<string, any>,
 *   replayTimeline: Record<string, any>,
 * }}
 */
export function syntheticFixturePresentationPair(fixture) {
  const pair = fixture?.synthetic_presentation_pair;
  if (!pair || typeof pair !== "object" || Array.isArray(pair)) {
    throw new Error("Renderer fixture has no synthetic presentation pair.");
  }
  const keys = Object.keys(pair).sort();
  if (
    keys.join("|") !==
    ["audience", "live_frame", "replay_frame", "replay_timeline"].sort().join("|")
  ) {
    throw new Error("Synthetic presentation pair has an invalid fixture shape.");
  }
  if (pair.audience !== fixture.audience) {
    throw new Error("Synthetic presentation pair audience does not match its fixture.");
  }
  const liveFrame = syntheticDebuggerWireFrame({
    audience: pair.audience,
    scene: pair.live_frame?.projection?.scene,
    live_frame: pair.live_frame,
  });
  const replayFrame = normalizeReplayViewerFrameV1(pair.replay_frame);
  const replayTimeline = normalizeReplayTimelineV1(pair.replay_timeline);
  joinReplayFrameAndTimeline(replayFrame, replayTimeline);
  if (
    replayFrame.projection.scene.episode_id !== liveFrame.projection.scene.episode_id ||
    replayFrame.projection.scene.frame_index !== liveFrame.projection.scene.frame_index
  ) {
    throw new Error("Synthetic live and replay projections do not share one epoch.");
  }
  return Object.freeze({
    audience: pair.audience,
    liveFrame,
    replayFrame: pair.replay_frame,
    replayTimeline: pair.replay_timeline,
  });
}
