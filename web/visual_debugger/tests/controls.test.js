import assert from "node:assert/strict";
import test from "node:test";

import {
  commandResponseSchedulesShutdown,
  isDebuggerKey,
  isPresentationPauseEvent,
  keyboardCommand,
  presentationRequiresSubmissionSettle,
  recordingCommandDecision,
  recordingReplacementCommand,
  recordingReviewHandoffRequired,
  recordingSaveAsCommand,
  targetSelectionCommand,
} from "../src/controls.js";

test("keyboardCommand forwards raw key and modifier state without semantics", () => {
  assert.deepEqual(
    keyboardCommand("Enter", {
      shiftKey: true,
      repeat: true,
    }),
    {
      command_type: "keyboard",
      key: "Enter",
      shift_key: true,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: true,
    },
  );
});

test("Submit settlement covers gated, post-gate active, and paused explanations", () => {
  assert.equal(
    presentationRequiresSubmissionSettle({
      submissionBlocked: true,
      animationCount: 3,
      paused: false,
    }),
    true,
  );
  assert.equal(
    presentationRequiresSubmissionSettle({
      submissionBlocked: false,
      animationCount: 2,
      paused: false,
    }),
    true,
  );
  assert.equal(
    presentationRequiresSubmissionSettle({
      submissionBlocked: false,
      animationCount: 2,
      paused: true,
    }),
    true,
  );
  assert.equal(
    presentationRequiresSubmissionSettle({
      submissionBlocked: false,
      animationCount: 0,
      paused: false,
    }),
    false,
  );
  assert.equal(presentationRequiresSubmissionSettle(null), false);
});

test("debugger key capture leaves modified browser shortcuts native", () => {
  assert.equal(isDebuggerKey({ key: "r" }), true);
  assert.equal(isDebuggerKey({ key: "p" }), true);
  assert.equal(isDebuggerKey({ key: "v" }), false);
  assert.equal(isDebuggerKey({ key: "P", shiftKey: true }), true);
  assert.equal(isDebuggerKey({ key: "x" }), true);
  assert.equal(isDebuggerKey({ key: "X" }), true);
  assert.equal(isDebuggerKey({ key: "0" }), true);
  assert.equal(isDebuggerKey({ key: "Tab", shiftKey: true }), true);
  assert.equal(isDebuggerKey({ key: "r", ctrlKey: true }), false);
  assert.equal(isDebuggerKey({ key: "p", ctrlKey: true }), false);
  assert.equal(isDebuggerKey({ key: "w", metaKey: true }), false);
  assert.equal(isDebuggerKey({ key: "ArrowLeft", altKey: true }), false);
  assert.equal(isDebuggerKey({ key: "F5" }), false);
});

test("presentation pause is edge-triggered and ignores key repeat", () => {
  assert.equal(isPresentationPauseEvent({ key: "p" }), true);
  assert.equal(isPresentationPauseEvent({ key: "P", repeat: false }), true);
  assert.equal(isPresentationPauseEvent({ key: "p", repeat: true }), false);
  assert.equal(isPresentationPauseEvent({ key: "Enter" }), false);
});

test("target selection keeps researcher and actor-POV identity domains separate", () => {
  assert.deepEqual(targetSelectionCommand("7"), {
    command_type: "roster_selection",
    role: "target",
    global_slot: 7,
  });
  assert.deepEqual(targetSelectionCommand("pov-target-action:7", { actorPov: true }), {
    command_type: "actor_pov_target_action",
    target_action: 7,
  });
  const povCommand = targetSelectionCommand("pov-target-action:7", {
    actorPov: true,
  });
  assert.ok(povCommand);
  assert.equal(Object.hasOwn(povCommand, "global_slot"), false);
  assert.deepEqual(targetSelectionCommand("pov-target-action:0", { actorPov: true }), {
    command_type: "actor_pov_target_action",
    target_action: 0,
  });
  assert.equal(targetSelectionCommand("7", { actorPov: true }), null);
  assert.equal(targetSelectionCommand("pov-target-action:7"), null);
});

function recordingFrame({ lifecycle = "recording", captured = 1 } = {}) {
  const persistenceFailed = lifecycle === "persistence_failed";
  const finalized = [
    "sealed",
    "finalized_unsaved",
    "persistence_failed",
    "saved",
    "reviewing",
  ].includes(lifecycle);
  return {
    scenario: {
      name: "alpha",
      ordinary_movement_distance_scale: 0.5,
    },
    available_scenarios: [{ name: "alpha" }, { name: "bravo" }, { name: "charlie" }],
    recording: {
      schema_version: 1,
      lifecycle,
      captured_transition_count: captured,
      expected_transition_count: 5,
      completion_state: finalized ? "partial" : null,
      completion_reason: finalized ? "user_finish_and_review" : null,
      restart_fenced: captured > 0 || lifecycle !== "recording",
      finish_available: lifecycle === "recording",
      review_available: lifecycle === "saved",
      retry_available: persistenceFailed,
      save_as_available: persistenceFailed,
      discard_available: lifecycle === "recording" && captured > 0,
      persistence_error_code: persistenceFailed ? "publication_failed" : null,
    },
  };
}

test("recording restart resolution canonicalizes direct and keyboard replacements", () => {
  const frame = recordingFrame();
  assert.deepEqual(recordingReplacementCommand(frame, { command_type: "reset" }), {
    command_type: "reset",
  });
  assert.deepEqual(recordingReplacementCommand(frame, keyboardCommand("[")), {
    command_type: "scenario_switch",
    scenario_name: "charlie",
  });
  assert.deepEqual(recordingReplacementCommand(frame, keyboardCommand("]")), {
    command_type: "scenario_switch",
    scenario_name: "bravo",
  });
  assert.equal(
    recordingReplacementCommand(frame, keyboardCommand("R", { shiftKey: true })),
    null,
  );
});

test("captured recording prefixes require exact discard confirmation for replacement", () => {
  const frame = recordingFrame();
  for (const command of [
    { command_type: "reset" },
    { command_type: "scenario_switch", scenario_name: "bravo" },
    keyboardCommand("r"),
  ]) {
    const decision = recordingCommandDecision(frame, command);
    assert.equal(decision.action, "confirm");
    assert.ok(decision.replacement);
    assert.equal(Object.isFrozen(decision.replacement), true);
  }

  const frameZero = recordingFrame({ captured: 0 });
  assert.equal(
    recordingCommandDecision(frameZero, { command_type: "reset" }).action,
    "allow",
  );
  assert.equal(
    recordingCommandDecision({ ...frame, recording: null }, { command_type: "reset" })
      .action,
    "allow",
  );
  for (const command of [
    keyboardCommand("w"),
    keyboardCommand("Enter"),
    { command_type: "battlefield_pointer", world_x: 1, world_y: 2 },
    { command_type: "roster_selection", role: "target", global_slot: 1 },
  ]) {
    assert.equal(recordingCommandDecision(frame, command).action, "allow");
  }
});

test("closed recording lifecycles fence scientific controls but retain presentation and recovery", () => {
  const failed = recordingFrame({ lifecycle: "persistence_failed" });
  for (const command of [
    keyboardCommand("w"),
    { command_type: "battlefield_pointer", world_x: 1, world_y: 2 },
    { command_type: "roster_selection", role: "control", global_slot: 0 },
    { command_type: "reset" },
  ]) {
    assert.equal(recordingCommandDecision(failed, command).action, "block");
  }
  for (const command of [
    { command_type: "set_view", view_mode: "pov" },
    { command_type: "set_preset", preset: "analysis" },
    keyboardCommand("g"),
    { command_type: "retry_save" },
    { command_type: "save_as", file_name: "copy.marlbg-replay.json" },
    { command_type: "exit" },
  ]) {
    assert.equal(recordingCommandDecision(failed, command).action, "allow");
  }
});

test("Save As emits only the protocol basename and rejects path-shaped input", () => {
  assert.deepEqual(recordingSaveAsCommand("copy.marlbg-replay.json"), {
    command_type: "save_as",
    file_name: "copy.marlbg-replay.json",
  });
  for (const value of [
    "../copy.marlbg-replay.json",
    "/tmp/copy.marlbg-replay.json",
    ".hidden.marlbg-replay.json",
    "copy replay.marlbg-replay.json",
    "copy.json",
    "",
  ]) {
    assert.equal(recordingSaveAsCommand(value), null);
  }
});

test("only an accepted Exit schedules shutdown and only reviewing live state requests handoff", () => {
  const exit = { command_type: "exit" };
  assert.equal(
    commandResponseSchedulesShutdown(exit, { result: "shutdown_scheduled" }),
    true,
  );
  assert.equal(commandResponseSchedulesShutdown(exit, { result: "no_op" }), false);
  assert.equal(
    commandResponseSchedulesShutdown(
      { command_type: "reset" },
      {
        result: "shutdown_scheduled",
      },
    ),
    false,
  );

  assert.equal(
    recordingReviewHandoffRequired({
      recording: { lifecycle: "reviewing" },
    }),
    true,
  );
  assert.equal(
    recordingReviewHandoffRequired({
      viewer_mode: "replay",
      recording: { lifecycle: "reviewing" },
    }),
    false,
  );
  assert.equal(
    recordingReviewHandoffRequired({ recording: { lifecycle: "saved" } }),
    false,
  );
});
