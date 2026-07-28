import assert from "node:assert/strict";
import test from "node:test";

import {
  DebuggerApiError,
  extractFrame,
  extractNotice,
  postCommand,
} from "../src/api.js";

test("extractFrame accepts success, stale, and direct frame envelopes", () => {
  const frame = { revision: 3, scene: {} };

  assert.equal(extractFrame({ frame }), frame);
  assert.equal(extractFrame({ latest_frame: frame }), frame);
  assert.equal(extractFrame(frame), frame);
  assert.equal(extractFrame({ revision: 3 }), null);
  assert.equal(extractFrame(null), null);
});

test("extractNotice accepts only non-empty protocol notices", () => {
  assert.equal(extractNotice({ notice: "Applied." }), "Applied.");
  assert.equal(extractNotice({ notice: { message: "Stale." } }), "Stale.");
  assert.equal(extractNotice({ notice: "  " }), null);
  assert.equal(extractNotice({}), null);
});

test("postCommand reports an unknown outcome and never retries", async () => {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  let fetchCalls = 0;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      clearTimeout: globalThis.clearTimeout,
      fetch: async () => {
        fetchCalls += 1;
        throw new Error("synthetic disconnect");
      },
      setTimeout: globalThis.setTimeout,
    },
  });

  try {
    await assert.rejects(
      postCommand("capability", {
        schema_version: 1,
        client_id: "client",
        command_id: "command",
        base_revision: 0,
        command: { command_type: "reset" },
      }),
      (error) =>
        error instanceof DebuggerApiError &&
        error.status === 0 &&
        error.message.includes("outcome is unknown"),
    );
    assert.equal(fetchCalls, 1);
  } finally {
    if (originalWindow) {
      Object.defineProperty(globalThis, "window", originalWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});
