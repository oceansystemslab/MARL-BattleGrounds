import assert from "node:assert/strict";
import test from "node:test";

import {
  combatDebuggerArguments,
  COMBAT_DEBUGGER_ENTRYPOINT,
} from "../e2e/support/live-debugger.js";
import {
  replayViewerArguments,
  REPLAY_VIEWER_ENTRYPOINT,
} from "../e2e/support/replay-viewer.js";

test("combat browser support launches only the fixed manual debugger", () => {
  const args = combatDebuggerArguments();

  assert.ok(args.includes(COMBAT_DEBUGGER_ENTRYPOINT));
  assert.equal(args.includes("--scenario"), false);
  assert.equal(args.includes("--replay"), false);
  assert.equal(args.includes("--preset"), false);
});

test("replay browser support uses the dedicated replay viewer entrypoint", () => {
  const args = replayViewerArguments({
    replayPath: "/tmp/example.marlbg-replay.json",
    frameIndex: 3,
    view: "researcher",
    preset: "presentation",
  });

  assert.ok(args.includes(REPLAY_VIEWER_ENTRYPOINT));
  assert.equal(args.includes(COMBAT_DEBUGGER_ENTRYPOINT), false);
  assert.deepEqual(args.slice(-6), [
    "--frame-index",
    "3",
    "--view",
    "researcher",
    "--preset",
    "presentation",
  ]);
});

test("replay browser support materializes scripted scenarios through the viewer", () => {
  const args = replayViewerArguments({
    scenario: "charge_convergence",
    seed: 17,
    includeStress: true,
    frameIndex: 2,
  });

  assert.ok(args.includes(REPLAY_VIEWER_ENTRYPOINT));
  assert.equal(args.includes(COMBAT_DEBUGGER_ENTRYPOINT), false);
  assert.deepEqual(args.slice(-10), [
    "--scenario",
    "charge_convergence",
    "--no-open",
    "--port",
    "0",
    "--seed",
    "17",
    "--include-stress",
    "--frame-index",
    "2",
  ]);
});
