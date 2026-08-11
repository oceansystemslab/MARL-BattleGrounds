import { execFile } from "node:child_process";
import { promisify } from "node:util";

import { normalizeLiveDebuggerFrameV2 } from "../../src/frame-normalizer.js";
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
 * Pass the Python-validated production V2 fixture envelope through the same
 * strict normalization boundary used for loopback API responses.
 *
 * @param {Record<string, any>} fixture
 * @returns {Record<string, any>}
 */
export function syntheticDebuggerFrame(fixture) {
  return normalizeLiveDebuggerFrameV2(syntheticDebuggerWireFrame(fixture));
}
