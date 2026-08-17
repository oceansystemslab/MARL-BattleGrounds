import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  DebuggerApiError,
  extractFrame,
  extractJoinedFrame,
  extractNotice,
  getCurrentFrame,
  getCurrentFrameAndPresentation,
  getCurrentPresentation,
  getReplayTimeline,
  postCommand,
  postReplayCommand,
} from "../src/api.js";

const presentationFixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

const presentationKinds = [
  "live_oracle",
  "live_no_shared_obs_agent_pov",
  "replay_oracle",
  "replay_no_shared_obs_agent_pov",
  "replay_shared_obs_agent_pov",
];

/**
 * @template T
 * @param {T} value
 * @returns {T}
 */
function clone(value) {
  return structuredClone(value);
}

/** @param {Record<string, any>} source */
function withEndpointReadTripwire(source) {
  const presentation = clone(source);
  const endpoint = presentation.current_endpoint;
  const nestedKey = Object.hasOwn(endpoint, "scene") ? "scene" : "parts";
  let reads = 0;
  Object.defineProperty(endpoint, nestedKey, {
    configurable: true,
    enumerable: true,
    get() {
      reads += 1;
      throw new Error("must not read presentation endpoint after envelope mismatch");
    },
  });
  return { presentation, readCount: () => reads };
}

/**
 * @param {unknown} payload
 * @param {{ok?: boolean, status?: number, contentType?: string}} options
 * @returns {Response}
 */
function jsonResponse(
  payload,
  { ok = true, status = 200, contentType = "application/json" } = {},
) {
  return /** @type {Response} */ (
    /** @type {unknown} */ ({
      headers: { get: () => contentType },
      json: async () => payload,
      ok,
      status,
      text: async () => "",
    })
  );
}

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

test("diagnostic and unjoined private Shared replay frames reject before live fallback", () => {
  for (const frameKind of [
    "shared_obs_source_material_replay_viewer",
    "shared_obs_agent_pov_replay_viewer",
  ]) {
    let nestedReads = 0;
    const frame = {
      schema_version: 1,
      frame_kind: frameKind,
      revision: 0,
    };
    Object.defineProperty(frame, "projection", {
      configurable: true,
      enumerable: true,
      get() {
        nestedReads += 1;
        throw new Error("Shared nested source material must remain unread");
      },
    });

    assert.throws(
      () => extractFrame(frame),
      /without an authorized presentation join/u,
    );
    assert.throws(
      () =>
        extractFrame({
          schema_version: 1,
          result: "no_op",
          frame,
          notice: null,
          animate_incoming: false,
        }),
      /without an authorized presentation join/u,
    );
    assert.throws(
      () =>
        extractFrame({
          schema_version: 1,
          error_code: "stale_revision",
          message: "stale replay cursor",
          latest_frame: frame,
        }),
      /without an authorized presentation join/u,
    );
    assert.equal(nestedReads, 0);
  }
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

test("live V2 and replay V1 error envelopes retain separate frame boundaries", () => {
  const liveFrame = clone(presentationFixture.pairs.live_oracle.transport);
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

test("extractJoinedFrame accepts all five direct Python pairs", async () => {
  for (const kind of presentationKinds) {
    const pair = presentationFixture.pairs[kind];
    const joined = await extractJoinedFrame(pair.transport, pair.presentation);
    assert.ok(joined);
    assert.equal(joined.transport.frame_kind, pair.transport.frame_kind);
    assert.equal(joined.presentation.presentation_kind, kind);
    assert.equal(Object.isFrozen(joined), true);
  }
});

test("extractJoinedFrame accepts exact success and 409 envelopes only", async () => {
  const livePair = presentationFixture.pairs.live_oracle;
  const liveSuccess = await extractJoinedFrame(
    {
      schema_version: 2,
      result: "no_op",
      frame: livePair.transport,
      notice: null,
    },
    livePair.presentation,
  );
  assert.ok(liveSuccess);
  assert.equal(liveSuccess.presentation.presentation_kind, "live_oracle");

  const sharedPair = presentationFixture.pairs.replay_shared_obs_agent_pov;
  const replaySuccess = await extractJoinedFrame(
    {
      schema_version: 1,
      result: "no_op",
      frame: sharedPair.transport,
      notice: "Replay cursor unchanged.",
      animate_incoming: false,
    },
    sharedPair.presentation,
  );
  assert.ok(replaySuccess);
  assert.equal(
    replaySuccess.presentation.presentation_kind,
    "replay_shared_obs_agent_pov",
  );

  for (const errorCode of ["stale_revision", "command_id_conflict"]) {
    const joined = await extractJoinedFrame(
      {
        schema_version: 1,
        error_code: errorCode,
        message: "The client raced the current replay epoch.",
        latest_frame: sharedPair.transport,
      },
      sharedPair.presentation,
    );
    assert.ok(joined);
    assert.equal(joined.transport.frame_kind, sharedPair.transport.frame_kind);
  }

  const liveStale = await extractJoinedFrame(
    {
      schema_version: 2,
      error_code: "stale_revision",
      message: "The client raced the current live epoch.",
      latest_frame: livePair.transport,
    },
    livePair.presentation,
  );
  assert.ok(liveStale);
  assert.equal(liveStale.transport.frame_kind, livePair.transport.frame_kind);

  const crossFamilyCases = [
    {
      label: "live pair in replay success envelope",
      envelope: {
        schema_version: 1,
        result: "no_op",
        frame: livePair.transport,
        notice: null,
        animate_incoming: false,
      },
      pair: livePair,
    },
    {
      label: "replay pair in live success envelope",
      envelope: {
        schema_version: 2,
        result: "no_op",
        frame: sharedPair.transport,
        notice: null,
      },
      pair: sharedPair,
    },
    {
      label: "live pair in replay stale-error envelope",
      envelope: {
        schema_version: 1,
        error_code: "stale_revision",
        message: "Cross-family candidate.",
        latest_frame: livePair.transport,
      },
      pair: livePair,
    },
    {
      label: "replay pair in live stale-error envelope",
      envelope: {
        schema_version: 2,
        error_code: "stale_revision",
        message: "Cross-family candidate.",
        latest_frame: sharedPair.transport,
      },
      pair: sharedPair,
    },
  ];
  for (const { label, envelope, pair } of crossFamilyCases) {
    const { presentation, readCount } = withEndpointReadTripwire(pair.presentation);
    await assert.rejects(extractJoinedFrame(envelope, presentation), TypeError);
    assert.equal(readCount(), 0, label);
  }

  assert.equal(
    await extractJoinedFrame(
      {
        schema_version: 1,
        error_code: "stale_revision",
        message: "No latest candidate.",
        latest_frame: null,
      },
      sharedPair.presentation,
    ),
    null,
  );

  for (const errorCode of ["internal_error", "invalid_request", "unauthorized"]) {
    await assert.rejects(
      extractJoinedFrame(
        {
          schema_version: 1,
          error_code: errorCode,
          message: "Must never install.",
          latest_frame: sharedPair.transport,
        },
        sharedPair.presentation,
      ),
      TypeError,
    );
  }
});

test("extractJoinedFrame enforces replay notice and animation scalars", async () => {
  const pair = presentationFixture.pairs.replay_shared_obs_agent_pov;
  for (const notice of ["", "x".repeat(2049)]) {
    await assert.rejects(
      extractJoinedFrame(
        {
          schema_version: 1,
          result: "no_op",
          frame: pair.transport,
          notice,
          animate_incoming: false,
        },
        pair.presentation,
      ),
      TypeError,
    );
  }
  await assert.rejects(
    extractJoinedFrame(
      {
        schema_version: 1,
        result: "no_op",
        frame: pair.transport,
        notice: null,
        animate_incoming: true,
      },
      pair.presentation,
    ),
    TypeError,
  );

  const animatedTransport = clone(pair.transport);
  animatedTransport.cursor.cursor_generation = 1;
  animatedTransport.cursor.choreography_generation = 1;
  const animated = await extractJoinedFrame(
    {
      schema_version: 1,
      result: "applied",
      frame: animatedTransport,
      notice: "Applied.",
      animate_incoming: true,
    },
    pair.presentation,
  );
  assert.ok(animated);
  assert.equal(animated.transport.animate_incoming, true);
});

test("presentation GET is exact, atomic with frame GET, and never installs errors", async () => {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  const pair = presentationFixture.pairs.replay_oracle;
  /** @type {Map<string, unknown>} */
  const payloads = new Map([
    ["/api/frame", pair.transport],
    ["/api/presentation/frame", pair.presentation],
  ]);
  /** @type {string[]} */
  const calls = [];
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      clearTimeout: globalThis.clearTimeout,
      /** @param {string} path */
      fetch: async (path) => {
        calls.push(String(path));
        return jsonResponse(payloads.get(String(path)));
      },
      setTimeout: globalThis.setTimeout,
    },
  });

  try {
    assert.equal(await getCurrentPresentation("capability"), pair.presentation);
    const joined = await getCurrentFrameAndPresentation("capability");
    assert.ok(joined);
    assert.equal(joined.presentation.presentation_kind, "replay_oracle");
    assert.deepEqual(calls, [
      "/api/presentation/frame",
      "/api/frame",
      "/api/presentation/frame",
    ]);

    const unavailable = {
      schema_version: 1,
      error_code: "audience_unavailable",
      message: "Authorized presentation is unavailable for the active audience.",
    };
    globalThis.window.fetch = async () =>
      jsonResponse(unavailable, { ok: false, status: 422 });
    await assert.rejects(getCurrentPresentation("capability"), (error) => {
      assert.ok(error instanceof DebuggerApiError);
      assert.equal(error.status, 422);
      assert.deepEqual(error.payload, unavailable);
      assert.equal(Object.isFrozen(error.payload), true);
      return true;
    });

    globalThis.window.fetch = async () =>
      jsonResponse({ ...unavailable, extra: true }, { ok: false, status: 422 });
    await assert.rejects(getCurrentPresentation("capability"), (error) => {
      assert.ok(error instanceof DebuggerApiError);
      assert.equal(error.status, 422);
      assert.equal(error.payload, null);
      assert.match(error.message, /invalid error envelope/u);
      return true;
    });

    globalThis.window.fetch = async () =>
      jsonResponse(unavailable, { ok: false, status: 500 });
    await assert.rejects(getCurrentPresentation("capability"), (error) => {
      assert.ok(error instanceof DebuggerApiError);
      assert.equal(error.status, 500);
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
