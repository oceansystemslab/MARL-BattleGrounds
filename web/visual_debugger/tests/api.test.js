import assert from "node:assert/strict";
import test from "node:test";

import {
  DebuggerApiError,
  extractFrame,
  extractNotice,
  postCommand,
} from "../src/api.js";

test("extractFrame rejects stale and unknown frame schemas at every envelope", () => {
  const stale = { schema_version: 1, revision: 3, scene: {} };
  const unknown = { schema_version: 99, revision: 3, projection: {} };

  assert.throws(
    () => extractFrame({ schema_version: 2, frame: stale }),
    /schema version 2/,
  );
  assert.throws(
    () => extractFrame({ schema_version: 2, latest_frame: unknown }),
    /schema version 2/,
  );
  assert.throws(() => extractFrame(stale), /schema version 2/);
  assert.equal(extractFrame({ schema_version: 2, current_frame: stale }), null);
  assert.equal(extractFrame({ revision: 3 }), null);
  assert.equal(extractFrame(null), null);
});

test("extractFrame rejects a V2 frame carried by a stale response envelope", () => {
  const v2Frame = { schema_version: 2 };
  assert.throws(
    () => extractFrame({ schema_version: 1, frame: v2Frame }),
    /response envelope must use schema version 2/u,
  );
  assert.throws(
    () => extractFrame({ frame: v2Frame }),
    /response envelope must use schema version 2/u,
  );
});

test("extractNotice accepts only non-empty protocol notices", () => {
  assert.equal(extractNotice({ notice: "Applied." }), "Applied.");
  assert.equal(extractNotice({ notice: { message: "Stale." } }), null);
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
