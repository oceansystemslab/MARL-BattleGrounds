import { execFile } from "node:child_process";
import { promisify } from "node:util";

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
      },
      maxBuffer: 4 * 1024 * 1024,
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
 * Wrap a renderer-only fixture in the smallest explicitly synthetic browser
 * presentation envelope. No live command can install this frame.
 *
 * @param {Record<string, any>} fixture
 * @returns {Record<string, any>}
 */
export function syntheticDebuggerFrame(fixture) {
  const scene = fixture.scene;
  if (!scene || typeof scene !== "object" || Array.isArray(scene)) {
    throw new Error("Renderer fixture is missing its scene.");
  }
  const name = typeof fixture.name === "string" ? fixture.name : "renderer_fixture";
  const description =
    typeof fixture.description === "string"
      ? fixture.description
      : "SYNTHETIC: renderer-only fixture.";
  const audience = scene.audience === "agent_pov" ? "agent_pov" : "researcher";
  const eventBatch =
    fixture.event_batch &&
    typeof fixture.event_batch === "object" &&
    !Array.isArray(fixture.event_batch)
      ? fixture.event_batch
      : null;
  const transitionId = eventBatch?.transition_id ?? 0;
  return {
    schema_version: "renderer_fixture_v1",
    frame_kind: "synthetic_renderer_fixture",
    session_id: `synthetic:${name}`,
    run_generation: 0,
    revision: 0,
    simulator_step: eventBatch?.simulator_step ?? 0,
    transition_id: transitionId,
    scenario: {
      name,
      title: `SYNTHETIC · ${name}`,
      description,
      audience: "renderer_fixture",
      frame_index: 0,
      frame_count: 1,
    },
    available_scenarios: [],
    view_mode: audience === "agent_pov" ? "pov" : "researcher",
    preset: "analysis",
    terminal: false,
    scene,
    event_batch: eventBatch,
    hud: {
      controlled_global_slot: scene.selection?.controlled_global_slot ?? null,
      pending_submission_scope: "scripted_playback",
      pending_actions: [],
      pending_action: null,
      latest_transition:
        eventBatch === null
          ? null
          : {
              label: "SYNTHETIC EVENT BATCH",
              transition_id: transitionId,
              submission_kind: "renderer_fixture",
              actors: [],
            },
      diagnostics: [],
    },
  };
}
