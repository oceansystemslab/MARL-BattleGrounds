import assert from "node:assert/strict";
import test from "node:test";

import {
  isDebuggerKey,
  isPresentationPauseEvent,
  keyboardCommand,
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

test("debugger key capture leaves modified browser shortcuts native", () => {
  assert.equal(isDebuggerKey({ key: "r" }), true);
  assert.equal(isDebuggerKey({ key: "p" }), true);
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
