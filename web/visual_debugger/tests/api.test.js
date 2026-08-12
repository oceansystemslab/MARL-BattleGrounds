import assert from "node:assert/strict";
import test from "node:test";
import {
  loadRendererFixture,
  syntheticDebuggerFrame,
} from "../e2e/support/renderer-fixture.js";
import {
  DebuggerApiError,
  extractFrame,
  extractNotice,
  getCurrentFrame,
  getReplayTimeline,
  postCommand,
  postReplayCommand,
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

test("schema-one replay responses cannot fall through the live-frame boundary", () => {
  assert.throws(
    () =>
      extractFrame({
        schema_version: 1,
        result: "applied",
        frame: { schema_version: 1, frame_kind: "unknown_replay_root" },
        notice: null,
        animate_incoming: false,
      }),
    /Unknown replay viewer frame kind/u,
  );
});

test("extractFrame enforces the exact replay error envelope even without a latest frame", () => {
  assert.equal(
    extractFrame({
      schema_version: 1,
      error_code: "invalid_request",
      message: "invalid replay request",
      latest_frame: null,
    }),
    null,
  );
  assert.throws(
    () =>
      extractFrame({
        schema_version: 1,
        error_code: "invalid_request",
        message: "invalid replay request",
        latest_frame: null,
        animate_incoming: false,
      }),
    /unknown or missing/u,
  );
});

test("live V2 and replay V1 error envelopes retain separate frame boundaries", async () => {
  const liveFrame = syntheticDebuggerFrame(
    await loadRendererFixture("canonical_event_vocabulary"),
  );
  const liveLatest = extractFrame({
    schema_version: 2,
    error_code: "stale_revision",
    message: "stale live tab",
    latest_frame: liveFrame,
  });
  assert.equal(liveLatest?.frame_kind, "researcher_live_debugger");
  assert.equal(liveLatest?.scene?.audience, "researcher");

  assert.equal(
    extractFrame({
      schema_version: 1,
      error_code: "stale_revision",
      message: "stale replay tab",
      latest_frame: null,
    }),
    null,
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

test("replay timeline and command use separate exact routes and send once", async () => {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  /** @type {Array<{path: string, options: RequestInit}>} */
  const calls = [];
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      clearTimeout: globalThis.clearTimeout,
      /**
       * @param {string} path
       * @param {RequestInit} options
       */
      fetch: async (path, options) => {
        calls.push({ path: String(path), options });
        return {
          headers: { get: () => "application/json" },
          json: async () => ({ route: String(path) }),
          ok: true,
          status: 200,
        };
      },
      setTimeout: globalThis.setTimeout,
    },
  });

  try {
    assert.deepEqual(await getReplayTimeline("capability"), {
      route: "/api/replay/timeline",
    });
    const request = {
      schema_version: 1,
      client_id: "client",
      command_id: "command",
      base_revision: 4,
      command: { command_type: "next_frame" },
    };
    assert.deepEqual(await postReplayCommand("capability", request), {
      route: "/api/replay/command",
    });
    assert.equal(calls.length, 2);
    assert.equal(calls[0].path, "/api/replay/timeline");
    assert.equal(calls[0].options.method, "GET");
    assert.equal(calls[1].path, "/api/replay/command");
    assert.equal(calls[1].options.method, "POST");
    assert.equal(calls[1].options.body, JSON.stringify(request));
  } finally {
    if (originalWindow) {
      Object.defineProperty(globalThis, "window", originalWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});

test("replay HTTP errors cross the exact error boundary before callers can inspect them", async () => {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  /** @type {Record<string, any>} */
  let payload = {
    schema_version: 1,
    error_code: "audience_unavailable",
    message: "Reference is unavailable in actor POV.",
    latest_frame: null,
  };
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      clearTimeout: globalThis.clearTimeout,
      fetch: async () => ({
        headers: { get: () => "application/json" },
        json: async () => payload,
        ok: false,
        status: 400,
      }),
      setTimeout: globalThis.setTimeout,
    },
  });

  try {
    await assert.rejects(
      postReplayCommand("capability", {
        schema_version: 1,
        client_id: "client",
        command_id: "command",
        base_revision: 0,
        command: { command_type: "select_agent", selected_global_slot: null },
      }),
      (error) => {
        assert.ok(error instanceof DebuggerApiError);
        assert.equal(error.status, 400);
        assert.deepEqual(error.payload, payload);
        assert.equal(Object.isFrozen(error.payload), true);
        return true;
      },
    );

    await assert.rejects(getCurrentFrame("capability"), (error) => {
      assert.ok(error instanceof DebuggerApiError);
      assert.equal(error.status, 400);
      assert.deepEqual(error.payload, payload);
      assert.equal(Object.isFrozen(error.payload), true);
      return true;
    });

    payload = { ...payload, animate_incoming: false };
    await assert.rejects(getReplayTimeline("capability"), (error) => {
      assert.ok(error instanceof DebuggerApiError);
      assert.equal(error.status, 400);
      assert.equal(error.payload, null);
      assert.match(error.message, /invalid error envelope/u);
      return true;
    });
  } finally {
    if (originalWindow) {
      Object.defineProperty(globalThis, "window", originalWindow);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  }
});
