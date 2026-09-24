/**
 * @file Check that browser-test launch helpers use the intended live or replay entry
 * point and fixed scenario route, and that the browser-test server lifecycle in
 * e2e/support/live-debugger.js handles server deaths with real processes and the
 * real /proc owner-token scan (Linux only):
 *
 * - When only a started server's launcher is killed with SIGKILL, the failure path
 *   reports that original failure first. It kills the launcher's surviving child
 *   (which ignores SIGINT), a peer server that is still starting, and a started peer
 *   with its own grandchild. An unrelated process without a token stays alive and is
 *   never signalled.
 * - A server that dies while still starting is left to its startup helper to
 *   report: no failure is thrown, its surviving child is killed and the cleanup is
 *   described for the startup rejection message.
 * - A process that is already dead when cleanup reaches it (ESRCH) is ignored. Any
 *   other kill error is listed after the original failure, and cleanup still kills
 *   the remaining owned processes. The dead process's live parent, which has no
 *   token, reaps it, and the test probes it with signal 0, so the test never
 *   signals a process ID that could have been reused.
 * - Signal-free fake /proc cases change ownership between selection and signalling:
 *   a lost token, changed user or unreadable process is skipped. Reused PIDs are
 *   tracked by start time, and excluded launchers are left alone. Direct signals
 *   check their expected identity as well as current ownership.
 * - A requested stop and a clean product shutdown are not failures.
 * - An allowed shutdown that ends with exit code 1 or by SIGKILL is a failure naming
 *   that outcome, and captureServerFailure captures it instead of throwing.
 * - expectServerShutdown resolves when the shutdown response comes before or after
 *   the clean exit, ignores requests that do not match, and removes all four of its
 *   page listeners once the response settles.
 * - A matching requestfailed rejects the shutdown response at once, without waiting
 *   for the server to exit.
 * - A page crash or close rejects the shutdown response at once and removes all four
 *   page listeners. The message names the page event and gives the server's exit
 *   outcome only when Node has already reported it, such as after a clean exit.
 * - The worker exit hook kills the owned processes that are still left when a worker
 *   process ends.
 *
 * The failure hooks are swapped so a failure is captured instead of crashing the
 * test runner. The registry settles failures, responses and exits inside the
 * launcher's exit event or a page event, before later listeners run. Tests
 * therefore await only a real process exit, then read each promise's state, so a
 * broken registry fails a test instead of leaving it waiting. Every test kills the
 * processes it starts before it ends.
 */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import fs, {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  COMBAT_DEBUGGER_ENTRYPOINT,
  captureServerFailure,
  combatDebuggerArguments,
  createServerOwnerToken,
  expectServerShutdown,
  findOwnedProcessIds,
  killOwnedProcesses,
  markServerStarted,
  registerServer,
  SERVER_OWNER_VARIABLE,
  ServerFailureError,
  serverEnvironment,
  setServerFailureHooks,
  signalOwnedProcess,
  startupCleanupNote,
  stopDebugger,
} from "../e2e/support/live-debugger.js";
import {
  REPLAY_VIEWER_ENTRYPOINT,
  replayViewerArguments,
} from "../e2e/support/replay-viewer.js";

const LIVE_DEBUGGER_MODULE = new URL(
  "../e2e/support/live-debugger.js",
  import.meta.url,
);
const workDirectory = mkdtempSync(join(tmpdir(), "marl-bg-launcher-support-"));
const treeScript = join(workDirectory, "process-tree.cjs");

// One small program for every test process. Its JSON argument says which role it
// prints, which children it starts and how it reacts to signals and stdin. A parent
// exits with code 1 when one of its children ends, unless its spec says it keeps
// running. A child spec's ownerToken adds that owner token to that child's
// environment only.
writeFileSync(
  treeScript,
  `const { spawn } = require("node:child_process");
const spec = JSON.parse(process.argv[2]);
if (spec.ignoreSigint) process.on("SIGINT", () => {});
if (spec.sigintExitCode !== undefined) {
  process.on("SIGINT", () => process.exit(spec.sigintExitCode));
}
if (spec.reportSignals) {
  for (const name of ["SIGINT", "SIGTERM", "SIGHUP", "SIGUSR1", "SIGUSR2"]) {
    process.on(name, () => console.log("signal " + name));
  }
}
for (const childSpec of spec.children ?? []) {
  const env =
    childSpec.ownerToken === undefined
      ? process.env
      : { ...process.env, ${SERVER_OWNER_VARIABLE}: childSpec.ownerToken };
  const child = spawn(process.execPath, [__filename, JSON.stringify(childSpec)], {
    env,
    stdio: ["ignore", "inherit", "inherit"],
  });
  child.on("exit", () => {
    if (!spec.keepRunningWhenChildrenEnd) process.exit(1);
  });
}
console.log("ready " + spec.role + " " + process.pid);
if (spec.exitOnStdin) {
  process.stdin.once("data", (data) => process.exit(Number(String(data).trim())));
} else setInterval(() => {}, 1000);
`,
);

test.after(() => {
  rmSync(workDirectory, { force: true, recursive: true });
});

/**
 * @typedef {{
 *   role: string,
 *   children?: TreeSpec[],
 *   ignoreSigint?: boolean,
 *   sigintExitCode?: number,
 *   reportSignals?: boolean,
 *   exitOnStdin?: boolean,
 *   keepRunningWhenChildrenEnd?: boolean,
 *   ownerToken?: string,
 * }} TreeSpec
 */

/** @param {TreeSpec} spec @param {NodeJS.ProcessEnv} environment */
function startTree(spec, environment) {
  return spawn(process.execPath, [treeScript, JSON.stringify(spec)], {
    env: environment,
    stdio: [spec.exitOnStdin ? "pipe" : "ignore", "pipe", "inherit"],
  });
}

/** @param {import("node:child_process").ChildProcess} child */
function collectOutput(child) {
  let output = "";
  child.stdout?.on("data", (chunk) => {
    output += String(chunk);
  });
  return () => output;
}

/** @param {import("node:child_process").ChildProcess} child
 * @param {string[]} roles
 * @returns {Promise<Record<string, number>>} */
function waitForReady(child, roles) {
  return new Promise((resolveReady, rejectReady) => {
    let output = "";
    /** @type {Record<string, number>} */
    const pids = {};
    /** @param {unknown} chunk */
    const onData = (chunk) => {
      output += String(chunk);
      for (const match of output.matchAll(/^ready (\S+) (\d+)$/gmu)) {
        pids[match[1]] = Number(match[2]);
      }
      if (roles.every((role) => role in pids)) {
        stopListening();
        resolveReady(pids);
      }
    };
    /** @param {number | null} code @param {NodeJS.Signals | null} signal */
    const onExit = (code, signal) => {
      stopListening();
      rejectReady(
        new Error(`Tree root exited before readiness (${code}, ${signal}):\n${output}`),
      );
    };
    const stopListening = () => {
      child.stdout?.off("data", onData);
      child.off("exit", onExit);
    };
    child.stdout?.on("data", onData);
    child.once("exit", onExit);
  });
}

/** @param {import("node:child_process").ChildProcess} child
 * @returns {Promise<{exitCode: number | null, signalCode: NodeJS.Signals | null}>} */
function exited(child) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve({ exitCode: child.exitCode, signalCode: child.signalCode });
  }
  return new Promise((resolveExit) => {
    child.once("exit", (exitCode, signalCode) => resolveExit({ exitCode, signalCode }));
  });
}

/** @param {number} milliseconds */
function sleep(milliseconds) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, milliseconds));
}

/** @param {number} pid */
function isRunning(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** @param {number} pid
 * @returns {string | null} */
function processState(pid) {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, "latin1");
    const end = stat.lastIndexOf(")");
    return stat.slice(end + 2, end + 3);
  } catch {
    return null;
  }
}

/** @param {string[]} tokens @param {number[]} signalled */
async function waitUntilOwnedProcessesAreGone(tokens, signalled) {
  for (;;) {
    const present = findOwnedProcessIds(tokens);
    if (present.length === 0) {
      return;
    }
    // Every process still present was already sent SIGKILL; only its death is awaited.
    assert.deepEqual(
      present.filter((pid) => !signalled.includes(pid)),
      [],
    );
    await sleep(20);
  }
}

/** @param {string[]} tokens
 * @param {import("node:child_process").ChildProcess[]} children */
async function removeEverything(tokens, children) {
  for (;;) {
    killOwnedProcesses(tokens);
    if (findOwnedProcessIds(tokens).length === 0) {
      break;
    }
    await sleep(20);
  }
  for (const child of children) {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill("SIGKILL");
    }
    await exited(child);
  }
}

/** @param {(pid: number, signal: NodeJS.Signals) => void} [killProcess] */
function installCapturingHooks(
  killProcess = (pid, signal) => process.kill(pid, signal),
) {
  /** @type {ServerFailureError[]} */
  const failures = [];
  /** @type {number[]} */
  const kills = [];
  const previous = setServerFailureHooks({
    onFailure: (error) => {
      failures.push(error);
    },
    killProcess: (pid, signal) => {
      kills.push(pid);
      killProcess(pid, signal);
    },
  });
  return {
    failures,
    kills,
    restore: () => {
      setServerFailureHooks(previous);
    },
  };
}

/** @param {Promise<unknown>} promise */
function track(promise) {
  /** @type {{status: "pending" | "fulfilled" | "rejected", value: any}} */
  const state = { status: "pending", value: undefined };
  promise.then(
    (value) => {
      state.status = "fulfilled";
      state.value = value;
    },
    (error) => {
      state.status = "rejected";
      state.value = error;
    },
  );
  return state;
}

function nextTurn() {
  return new Promise((resolveTurn) => setImmediate(resolveTurn));
}

/** @param {string} url */
function fakeRequest(url) {
  return {
    method: () => "POST",
    url: () => url,
    failure: () => ({ errorText: "net::ERR_CONNECTION_REFUSED" }),
  };
}

/** @param {ReturnType<typeof fakeRequest>} request */
function fakeResponse(request) {
  return { request: () => request };
}

/** @returns {any} */
function fakePage() {
  return new EventEmitter();
}

/** @param {EventEmitter} page */
function pageListenerCounts(page) {
  return Object.fromEntries(
    ["response", "requestfailed", "crash", "close"].map((event) => [
      event,
      page.listenerCount(event),
    ]),
  );
}

/** @param {import("@playwright/test").Request} request */
function isShutdownCommand(request) {
  return request.url() === "http://127.0.0.1:9/api/command";
}

/** @param {number[]} values */
function sorted(values) {
  return [...values].sort((left, right) => left - right);
}

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

/**
 * @typedef {{uid: number, start: string, environment: string, unreadable?: boolean}} FakeProcess
 */

/** @param {import("node:test").TestContext} context
 * @param {(rows: Map<number, FakeProcess>, token: string) => void} check */
function withFakeOwnedProcesses(context, check) {
  const token = createServerOwnerToken();
  const environment = `${SERVER_OWNER_VARIABLE}=${token}\0`;
  /** @type {Map<number, FakeProcess>} */
  const rows = new Map(
    [2100000001, 2100000002].map((pid) => [
      pid,
      { uid: Number(process.getuid?.()), start: "1", environment },
    ]),
  );
  /** @param {unknown} path */
  const rowAt = (path) => {
    const pid = Number(String(path).split("/")[2]);
    const row = rows.get(pid);
    if (row === undefined || row.unreadable) {
      throw Object.assign(new Error("Fake process is unavailable"), { code: "ENOENT" });
    }
    return row;
  };
  context.mock.method(fs, "readdirSync", () => [...rows.keys()].map(String));
  context.mock.method(fs, "statSync", (/** @type {unknown} */ path) => ({
    uid: rowAt(path).uid,
  }));
  context.mock.method(fs, "readFileSync", (/** @type {unknown} */ path) => {
    const row = rowAt(path);
    if (String(path).endsWith("/environ")) {
      return row.environment;
    }
    const fields = ["S", ...Array(18).fill("0"), row.start];
    return `0 (fake process) ${fields.join(" ")}`;
  });
  syncBuiltinESMExports();
  try {
    check(rows, token);
  } finally {
    context.mock.restoreAll();
    syncBuiltinESMExports();
  }
}

for (const change of ["token", "user", "unreadable"]) {
  test(`cleanup skips a selected process after its ${change} changes`, (context) => {
    withFakeOwnedProcesses(context, (rows, token) => {
      /** @type {number[]} */
      const signals = [];
      const cleanup = killOwnedProcesses([token], (pid) => {
        signals.push(pid);
        rows.delete(pid);
        const next = rows.get(2100000002);
        assert.ok(next);
        if (change === "token") next.environment = "UNRELATED=1\0";
        if (change === "user") next.uid += 1;
        if (change === "unreadable") next.unreadable = true;
      });
      assert.deepEqual(signals, [2100000001]);
      assert.deepEqual(cleanup.killedProcessIds, signals);
      assert.deepEqual(cleanup.failedProcessIds, []);
      assert.deepEqual(cleanup.stillPresentProcessIds, []);
      assert.deepEqual(cleanup.errors, []);
    });
  });
}

test("cleanup handles a new owned process reusing an already signalled PID", (context) => {
  withFakeOwnedProcesses(context, (rows, token) => {
    rows.delete(2100000002);
    /** @type {string[]} */
    const signals = [];
    const cleanup = killOwnedProcesses([token], (pid) => {
      const row = rows.get(pid);
      assert.ok(row);
      signals.push(row.start);
      if (row.start === "1") row.start = "2";
      else rows.delete(pid);
    });
    assert.deepEqual(signals, ["1", "2"]);
    assert.deepEqual(cleanup.killedProcessIds, [2100000001, 2100000001]);
    assert.deepEqual(cleanup.stillPresentProcessIds, []);
  });
});

test("direct signals require current ownership and the expected start identity", (context) => {
  withFakeOwnedProcesses(context, (rows, token) => {
    const pid = 2100000001;
    /** @type {number[]} */
    const signals = [];
    /** @param {number} selected */
    const recordSignal = (selected) => signals.push(selected);
    assert.equal(
      signalOwnedProcess(pid, [token], "SIGSTOP", recordSignal, `${pid}:0`),
      false,
    );
    assert.deepEqual(signals, []);
    assert.equal(
      signalOwnedProcess(pid, [token], "SIGSTOP", recordSignal, `${pid}:1`),
      true,
    );
    const row = rows.get(pid);
    assert.ok(row);
    row.environment = "UNRELATED=1\0";
    assert.equal(signalOwnedProcess(pid, [token], "SIGKILL", recordSignal), false);
    assert.deepEqual(signals, [pid]);
  });
});

test("cleanup leaves an excluded launcher alone while signalling its owned server", (context) => {
  withFakeOwnedProcesses(context, (rows, token) => {
    /** @type {number[]} */
    const signals = [];
    const cleanup = killOwnedProcesses(
      [token],
      (pid) => {
        signals.push(pid);
        rows.delete(pid);
      },
      [2100000001],
    );
    assert.deepEqual(signals, [2100000002]);
    assert.ok(rows.has(2100000001));
    assert.deepEqual(cleanup.stillPresentProcessIds, []);
  });
});

test("a launcher-only SIGKILL reports the original failure first and kills every owned process but no other", async () => {
  const hooks = installCapturingHooks();
  const tokens = [
    createServerOwnerToken(),
    createServerOwnerToken(),
    createServerOwnerToken(),
  ];
  const [tokenA, tokenB, tokenC] = tokens;
  /** @type {import("node:child_process").ChildProcess[]} */
  const children = [];
  try {
    assert.equal(process.env[SERVER_OWNER_VARIABLE], undefined);
    const launcherA = startTree(
      { role: "launcher-a", children: [{ role: "child-a", ignoreSigint: true }] },
      serverEnvironment(tokenA),
    );
    children.push(launcherA);
    registerServer(launcherA, { label: "Server A", token: tokenA });
    const launcherB = startTree({ role: "launcher-b" }, serverEnvironment(tokenB));
    children.push(launcherB);
    registerServer(launcherB, { label: "Server B", token: tokenB });
    const launcherC = startTree(
      {
        role: "launcher-c",
        children: [{ role: "child-c", children: [{ role: "grandchild-c" }] }],
      },
      serverEnvironment(tokenC),
    );
    children.push(launcherC);
    registerServer(launcherC, { label: "Server C", token: tokenC });
    const unrelated = startTree(
      { role: "unrelated", reportSignals: true },
      process.env,
    );
    children.push(unrelated);
    const unrelatedOutput = collectOutput(unrelated);
    const [a, b, c, u] = await Promise.all([
      waitForReady(launcherA, ["launcher-a", "child-a"]),
      waitForReady(launcherB, ["launcher-b"]),
      waitForReady(launcherC, ["launcher-c", "child-c", "grandchild-c"]),
      waitForReady(unrelated, ["unrelated"]),
    ]);
    // Server B stays "starting"; servers A and C have printed their launch URL.
    markServerStarted(launcherA);
    markServerStarted(launcherC);

    process.kill(a["launcher-a"], "SIGKILL");
    await exited(launcherA);

    assert.equal(hooks.failures.length, 1);
    const failure = hooks.failures[0];
    assert.ok(failure instanceof ServerFailureError);
    assert.equal(failure.errors.length, 1);
    assert.match(
      failure.errors[0].message,
      /^Server A \(launcher process \d+\) exited during the test with exit code none and signal SIGKILL\. Allowed exit: none; the test did not ask this server to stop\.\n/u,
    );
    assert.ok(failure.message.startsWith(failure.errors[0].message));
    assert.deepEqual(failure.outcome, { exitCode: null, signalCode: "SIGKILL" });
    const owned = [
      a["child-a"],
      b["launcher-b"],
      c["launcher-c"],
      c["child-c"],
      c["grandchild-c"],
    ];
    assert.deepEqual(
      sorted([
        ...failure.cleanup.killedProcessIds,
        ...failure.cleanup.alreadyExitedProcessIds,
      ]),
      sorted(owned),
    );
    assert.deepEqual(sorted(hooks.kills), sorted(owned));
    assert.equal(hooks.kills.includes(u.unrelated), false);

    await waitUntilOwnedProcessesAreGone(tokens, hooks.kills);
    await Promise.all([exited(launcherB), exited(launcherC)]);
    assert.equal(hooks.failures.length, 1);
    assert.equal(isRunning(u.unrelated), true);
    assert.equal(unrelated.exitCode, null);
    assert.equal(unrelated.signalCode, null);
    assert.doesNotMatch(unrelatedOutput(), /signal/u);
  } finally {
    await removeEverything(tokens, children);
    hooks.restore();
  }
});

test("a server that dies while starting is reported by its startup helper and its survivors are killed", async () => {
  const hooks = installCapturingHooks();
  const token = createServerOwnerToken();
  /** @type {import("node:child_process").ChildProcess[]} */
  const children = [];
  try {
    const launcher = startTree(
      { role: "launcher", children: [{ role: "child", ignoreSigint: true }] },
      serverEnvironment(token),
    );
    children.push(launcher);
    registerServer(launcher, { label: "Server M", token });
    const pids = await waitForReady(launcher, ["launcher", "child"]);

    process.kill(pids.launcher, "SIGKILL");
    await exited(launcher);

    assert.deepEqual(hooks.failures, []);
    assert.deepEqual(hooks.kills, [pids.child]);
    const note = startupCleanupNote(launcher);
    assert.ok(
      note.startsWith(`\nCleanup sent SIGKILL to owned processes ${pids.child}.`),
      note,
    );
    await waitUntilOwnedProcessesAreGone([token], hooks.kills);
  } finally {
    await removeEverything([token], children);
    hooks.restore();
  }
});

test("cleanup keeps the original failure first, lists other kill errors, ignores ESRCH and kills the rest", async () => {
  /** @type {number | null} */
  let alreadyDeadPid = null;
  /** @type {number | null} */
  let reaperPid = null;
  /** @type {number | null} */
  let injectedPid = null;
  const hooks = installCapturingHooks((pid, signal) => {
    if (pid === alreadyDeadPid) {
      // Kill it first. Its parent is a live process without a token, which reaps it
      // at once, so the signal-0 probe below meets the real ESRCH. Signal 0 sends
      // nothing, so a reused process ID would never be signalled.
      process.kill(pid, "SIGKILL");
      while (existsSync(`/proc/${pid}`)) {
        const reaperState = processState(Number(reaperPid));
        if (reaperState === null || reaperState === "Z") {
          throw new Error(`Reaper ${reaperPid} ended before it reaped process ${pid}.`);
        }
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 1);
      }
      process.kill(pid, 0);
      return;
    }
    if (injectedPid === null) {
      injectedPid = pid;
      throw Object.assign(new Error("Injected kill failure"), { code: "EPERM" });
    }
    process.kill(pid, signal);
  });
  const token = createServerOwnerToken();
  /** @type {import("node:child_process").ChildProcess[]} */
  const children = [];
  try {
    const launcher = startTree(
      { role: "launcher", children: [{ role: "first" }, { role: "second" }] },
      serverEnvironment(token),
    );
    children.push(launcher);
    registerServer(launcher, { label: "Server D", token });
    // The already-dead process holds the server's token, but its parent is a live
    // process without a token. That parent, not init, reaps it once it is killed.
    const reaper = startTree(
      {
        role: "reaper",
        keepRunningWhenChildrenEnd: true,
        children: [{ role: "doomed", ownerToken: token }],
      },
      process.env,
    );
    children.push(reaper);
    const [pids, reaperPids] = await Promise.all([
      waitForReady(launcher, ["launcher", "first", "second"]),
      waitForReady(reaper, ["reaper", "doomed"]),
    ]);
    markServerStarted(launcher);
    alreadyDeadPid = reaperPids.doomed;
    reaperPid = reaperPids.reaper;

    process.kill(pids.launcher, "SIGKILL");
    await exited(launcher);

    assert.equal(hooks.failures.length, 1);
    const failure = hooks.failures[0];
    assert.ok(failure instanceof ServerFailureError);
    assert.notEqual(injectedPid, null);
    const failedPid = Number(injectedPid);
    assert.deepEqual(
      failure.errors.slice(1).map((error) => error.message),
      [`SIGKILL of owned process ${failedPid} failed: Injected kill failure`],
    );
    assert.match(
      failure.errors[0].message,
      /^Server D \(launcher process \d+\) exited during the test with exit code none and signal SIGKILL\./u,
    );
    assert.ok(failure.message.startsWith(failure.errors[0].message));
    assert.match(
      failure.message,
      /1 cleanup error\(s\) follow the original failure\./u,
    );
    assert.equal(
      /** @type {{code?: string}} */ (/** @type {Error} */ (failure.errors[1]).cause)
        .code,
      "EPERM",
    );
    assert.deepEqual(failure.cleanup.alreadyExitedProcessIds, [reaperPids.doomed]);
    assert.equal(hooks.kills.includes(reaperPids.reaper), false);
    assert.equal(isRunning(reaperPids.reaper), true);
    assert.deepEqual(failure.cleanup.failedProcessIds, [failedPid]);
    assert.ok(failure.cleanup.stillPresentProcessIds.includes(failedPid));
    assert.ok([pids.first, pids.second].includes(failedPid));
    const remainingPid = failedPid === pids.first ? pids.second : pids.first;
    assert.deepEqual(failure.cleanup.killedProcessIds, [remainingPid]);
    assert.ok(hooks.kills.indexOf(remainingPid) > hooks.kills.indexOf(failedPid));

    for (;;) {
      const present = findOwnedProcessIds([token]);
      assert.deepEqual(
        present.filter((pid) => pid !== failedPid && pid !== remainingPid),
        [],
      );
      if (!present.includes(remainingPid)) {
        break;
      }
      // The remaining process was already sent SIGKILL; only its death is awaited.
      await sleep(20);
    }
    assert.equal(isRunning(failedPid), true);
    assert.equal(hooks.failures.length, 1);
  } finally {
    await removeEverything([token], children);
    hooks.restore();
  }
});

test("a requested stop and a clean product shutdown are not failures", async () => {
  const hooks = installCapturingHooks();
  const tokens = [
    createServerOwnerToken(),
    createServerOwnerToken(),
    createServerOwnerToken(),
  ];
  /** @type {import("node:child_process").ChildProcess[]} */
  const children = [];
  try {
    const stoppedBySignal = startTree(
      { role: "stopped" },
      serverEnvironment(tokens[0]),
    );
    const stoppedWithCode = startTree(
      { role: "stopped-with-code", sigintExitCode: 1 },
      serverEnvironment(tokens[1]),
    );
    const shutdownServer = startTree(
      { role: "shutdown", exitOnStdin: true },
      serverEnvironment(tokens[2]),
    );
    children.push(stoppedBySignal, stoppedWithCode, shutdownServer);
    registerServer(stoppedBySignal, { label: "Server E", token: tokens[0] });
    registerServer(stoppedWithCode, { label: "Server F", token: tokens[1] });
    registerServer(shutdownServer, { label: "Server G", token: tokens[2] });
    await Promise.all([
      waitForReady(stoppedBySignal, ["stopped"]),
      waitForReady(stoppedWithCode, ["stopped-with-code"]),
      waitForReady(shutdownServer, ["shutdown"]),
    ]);
    for (const child of children) {
      markServerStarted(child);
    }

    await stopDebugger(stoppedBySignal);
    await stopDebugger(stoppedWithCode);
    assert.equal(stoppedBySignal.signalCode, "SIGINT");
    assert.equal(stoppedWithCode.exitCode, 1);

    const page = fakePage();
    const shutdown = expectServerShutdown(page, shutdownServer, isShutdownCommand);
    const response = track(shutdown.response);
    const exit = track(shutdown.exit);
    const matching = fakeResponse(fakeRequest("http://127.0.0.1:9/api/command"));
    page.emit("response", matching);
    shutdownServer.stdin?.write("0\n");
    await exited(shutdownServer);
    await nextTurn();
    assert.deepEqual(response, { status: "fulfilled", value: matching });
    assert.deepEqual(exit, {
      status: "fulfilled",
      value: { exitCode: 0, signalCode: null },
    });
    assert.deepEqual(hooks.failures, []);
    assert.deepEqual(hooks.kills, []);
  } finally {
    await removeEverything(tokens, children);
    hooks.restore();
  }
});

test("an allowed shutdown that exits with code 1 or by SIGKILL is a failure naming the outcome", async () => {
  const hooks = installCapturingHooks();
  const tokens = [createServerOwnerToken(), createServerOwnerToken()];
  /** @type {import("node:child_process").ChildProcess[]} */
  const children = [];
  try {
    // One server at a time: a failure path also kills every other registered server.
    const exitsWithOne = startTree(
      { role: "exits-with-one", exitOnStdin: true },
      serverEnvironment(tokens[0]),
    );
    children.push(exitsWithOne);
    registerServer(exitsWithOne, { label: "Server H", token: tokens[0] });
    await waitForReady(exitsWithOne, ["exits-with-one"]);
    markServerStarted(exitsWithOne);

    const firstShutdown = expectServerShutdown(
      fakePage(),
      exitsWithOne,
      isShutdownCommand,
    );
    const firstResponse = track(firstShutdown.response);
    const firstExit = track(firstShutdown.exit);
    exitsWithOne.stdin?.write("1\n");
    await exited(exitsWithOne);
    await nextTurn();
    assert.equal(hooks.failures.length, 1);
    const firstFailure = hooks.failures[0];
    assert.ok(firstFailure instanceof ServerFailureError);
    assert.match(
      firstFailure.errors[0].message,
      /^Server H \(launcher process \d+\) exited during the test with exit code 1 and signal none\. Allowed exit: exit code 0 and no signal, after a product shutdown request\./u,
    );
    assert.deepEqual(firstResponse, { status: "rejected", value: firstFailure });
    assert.deepEqual(firstExit, {
      status: "fulfilled",
      value: { exitCode: 1, signalCode: null },
    });

    const killed = startTree({ role: "killed" }, serverEnvironment(tokens[1]));
    children.push(killed);
    registerServer(killed, { label: "Server I", token: tokens[1] });
    const killedPids = await waitForReady(killed, ["killed"]);
    markServerStarted(killed);
    const captured = track(captureServerFailure(killed));
    const secondShutdown = expectServerShutdown(fakePage(), killed, isShutdownCommand);
    const secondResponse = track(secondShutdown.response);
    const secondExit = track(secondShutdown.exit);
    process.kill(killedPids.killed, "SIGKILL");
    await exited(killed);
    await nextTurn();
    assert.equal(captured.status, "fulfilled");
    const secondFailure = captured.value;
    assert.ok(secondFailure instanceof ServerFailureError);
    assert.match(
      secondFailure.errors[0].message,
      /^Server I \(launcher process \d+\) exited during the test with exit code none and signal SIGKILL\. Allowed exit: exit code 0 and no signal, after a product shutdown request\./u,
    );
    assert.deepEqual(secondResponse, { status: "rejected", value: secondFailure });
    assert.deepEqual(secondExit, {
      status: "fulfilled",
      value: { exitCode: null, signalCode: "SIGKILL" },
    });
    // The captured failure was not also handed to the throwing failure action.
    assert.deepEqual(hooks.failures, [firstFailure]);
  } finally {
    await removeEverything(tokens, children);
    hooks.restore();
  }
});

test("a clean shutdown resolves when its response comes before or after the exit", async () => {
  const hooks = installCapturingHooks();
  const tokens = [createServerOwnerToken(), createServerOwnerToken()];
  /** @type {import("node:child_process").ChildProcess[]} */
  const children = [];
  try {
    const responseFirst = startTree(
      { role: "response-first", exitOnStdin: true },
      serverEnvironment(tokens[0]),
    );
    const exitFirst = startTree(
      { role: "exit-first", exitOnStdin: true },
      serverEnvironment(tokens[1]),
    );
    children.push(responseFirst, exitFirst);
    registerServer(responseFirst, { label: "Server J", token: tokens[0] });
    registerServer(exitFirst, { label: "Server K", token: tokens[1] });
    await Promise.all([
      waitForReady(responseFirst, ["response-first"]),
      waitForReady(exitFirst, ["exit-first"]),
    ]);
    markServerStarted(responseFirst);
    markServerStarted(exitFirst);

    const firstPage = fakePage();
    const first = expectServerShutdown(firstPage, responseFirst, isShutdownCommand);
    const firstResponse = track(first.response);
    const firstExit = track(first.exit);
    const firstMatching = fakeResponse(fakeRequest("http://127.0.0.1:9/api/command"));
    firstPage.emit("response", firstMatching);
    await nextTurn();
    assert.deepEqual(firstResponse, { status: "fulfilled", value: firstMatching });
    assert.equal(firstExit.status, "pending");
    responseFirst.stdin?.write("0\n");
    await exited(responseFirst);
    await nextTurn();
    assert.deepEqual(firstExit, {
      status: "fulfilled",
      value: { exitCode: 0, signalCode: null },
    });

    const secondPage = fakePage();
    const second = expectServerShutdown(secondPage, exitFirst, isShutdownCommand);
    const secondResponse = track(second.response);
    const secondExit = track(second.exit);
    exitFirst.stdin?.write("0\n");
    await exited(exitFirst);
    await nextTurn();
    assert.deepEqual(secondExit, {
      status: "fulfilled",
      value: { exitCode: 0, signalCode: null },
    });
    assert.equal(secondResponse.status, "pending");
    const unrelatedRequest = fakeRequest("http://127.0.0.1:9/api/frame");
    secondPage.emit("response", fakeResponse(unrelatedRequest));
    secondPage.emit("requestfailed", unrelatedRequest);
    await nextTurn();
    assert.equal(secondResponse.status, "pending");
    const secondMatching = fakeResponse(fakeRequest("http://127.0.0.1:9/api/command"));
    secondPage.emit("response", secondMatching);
    await nextTurn();
    assert.deepEqual(secondResponse, { status: "fulfilled", value: secondMatching });
    assert.deepEqual(pageListenerCounts(secondPage), {
      response: 0,
      requestfailed: 0,
      crash: 0,
      close: 0,
    });
    assert.deepEqual(hooks.failures, []);
  } finally {
    await removeEverything(tokens, children);
    hooks.restore();
  }
});

test("a matching requestfailed rejects the shutdown response at once without waiting for the exit", async () => {
  const hooks = installCapturingHooks();
  const token = createServerOwnerToken();
  /** @type {import("node:child_process").ChildProcess[]} */
  const children = [];
  try {
    const server = startTree({ role: "never-answers" }, serverEnvironment(token));
    children.push(server);
    registerServer(server, { label: "Server L", token });
    await waitForReady(server, ["never-answers"]);
    markServerStarted(server);

    const page = fakePage();
    const shutdown = expectServerShutdown(page, server, isShutdownCommand);
    const response = track(shutdown.response);
    const exit = track(shutdown.exit);
    page.emit("requestfailed", fakeRequest("http://127.0.0.1:9/api/command"));
    await nextTurn();
    assert.equal(response.status, "rejected");
    assert.equal(
      response.value.message,
      "The shutdown request POST http://127.0.0.1:9/api/command failed before its " +
        "response arrived (net::ERR_CONNECTION_REFUSED). Server L exit: not reported yet.",
    );
    assert.equal(exit.status, "pending");
    assert.equal(server.exitCode, null);
    assert.equal(server.signalCode, null);

    await stopDebugger(server);
    await nextTurn();
    assert.deepEqual(exit, {
      status: "fulfilled",
      value: { exitCode: null, signalCode: "SIGINT" },
    });
    assert.deepEqual(hooks.failures, []);
  } finally {
    await removeEverything([token], children);
    hooks.restore();
  }
});

test("a page crash or close rejects the shutdown response at once and removes every page listener", async () => {
  const hooks = installCapturingHooks();
  const tokens = [createServerOwnerToken(), createServerOwnerToken()];
  /** @type {import("node:child_process").ChildProcess[]} */
  const children = [];
  try {
    const running = startTree({ role: "still-running" }, serverEnvironment(tokens[0]));
    const exitsCleanly = startTree(
      { role: "exits-cleanly", exitOnStdin: true },
      serverEnvironment(tokens[1]),
    );
    children.push(running, exitsCleanly);
    registerServer(running, { label: "Server N", token: tokens[0] });
    registerServer(exitsCleanly, { label: "Server O", token: tokens[1] });
    await Promise.all([
      waitForReady(running, ["still-running"]),
      waitForReady(exitsCleanly, ["exits-cleanly"]),
    ]);
    markServerStarted(running);
    markServerStarted(exitsCleanly);
    const everyListener = { response: 1, requestfailed: 1, crash: 1, close: 1 };
    const noListener = { response: 0, requestfailed: 0, crash: 0, close: 0 };

    // The page crashes while its server is still running.
    const crashPage = fakePage();
    const crashed = expectServerShutdown(crashPage, running, isShutdownCommand);
    const crashedResponse = track(crashed.response);
    const crashedExit = track(crashed.exit);
    assert.deepEqual(pageListenerCounts(crashPage), everyListener);
    crashPage.emit("crash", crashPage);
    await nextTurn();
    assert.equal(crashedResponse.status, "rejected");
    assert.equal(
      crashedResponse.value.message,
      'The page crashed before the shutdown response arrived (page "crash" event). ' +
        "Server N exit: not reported yet.",
    );
    assert.deepEqual(pageListenerCounts(crashPage), noListener);
    assert.equal(crashedExit.status, "pending");
    assert.equal(running.exitCode, null);
    assert.equal(running.signalCode, null);

    // The server exits cleanly first, and then the page closes before the response.
    const closePage = fakePage();
    const closed = expectServerShutdown(closePage, exitsCleanly, isShutdownCommand);
    const closedResponse = track(closed.response);
    const closedExit = track(closed.exit);
    assert.deepEqual(pageListenerCounts(closePage), everyListener);
    exitsCleanly.stdin?.write("0\n");
    await exited(exitsCleanly);
    await nextTurn();
    assert.deepEqual(closedExit, {
      status: "fulfilled",
      value: { exitCode: 0, signalCode: null },
    });
    assert.equal(closedResponse.status, "pending");
    closePage.emit("close", closePage);
    await nextTurn();
    assert.equal(closedResponse.status, "rejected");
    assert.equal(
      closedResponse.value.message,
      'The page closed before the shutdown response arrived (page "close" event). ' +
        "Server O exit: exit code 0 and signal none.",
    );
    assert.deepEqual(pageListenerCounts(closePage), noListener);

    await stopDebugger(running);
    await nextTurn();
    assert.deepEqual(crashedExit, {
      status: "fulfilled",
      value: { exitCode: null, signalCode: "SIGINT" },
    });
    assert.match(crashedResponse.value.message, /^The page crashed /u);
    assert.deepEqual(hooks.failures, []);
    assert.deepEqual(hooks.kills, []);
  } finally {
    await removeEverything(tokens, children);
    hooks.restore();
  }
});

test("the worker exit hook kills the owned processes that remain", async () => {
  const workerScript = join(workDirectory, "exit-hook-worker.mjs");
  writeFileSync(
    workerScript,
    `import { spawn } from "node:child_process";
import { writeSync } from "node:fs";
import {
  createServerOwnerToken,
  markServerStarted,
  registerServer,
  serverEnvironment,
  signalOwnedProcess,
  workerExitCleanupResult,
} from ${JSON.stringify(LIVE_DEBUGGER_MODULE.href)};

const token = createServerOwnerToken();
const spec = { role: "launcher", children: [{ role: "child", ignoreSigint: true }] };
const launcher = spawn(process.execPath, [${JSON.stringify(treeScript)}, JSON.stringify(spec)], {
  env: serverEnvironment(token),
  stdio: ["ignore", "pipe", "inherit"],
});
registerServer(launcher, { label: "Worker server", token });
let output = "";
launcher.stdout.on("data", (chunk) => {
  output += String(chunk);
  const pids = [...output.matchAll(/^ready (\\S+) (\\d+)$/gmu)].map((match) => Number(match[2]));
  if (pids.length < 2) return;
  markServerStarted(launcher);
  process.on("exit", () => {
    writeSync(1, JSON.stringify({ token, pids, cleanup: workerExitCleanupResult() }) + "\\n");
  });
  process.exit(0);
});
launcher.once("exit", (code, signal) => {
  writeSync(2, "Worker server exited before readiness: " + code + " " + signal + "\\n");
  process.exit(3);
});
`,
  );
  const worker = spawn(process.execPath, [workerScript], {
    stdio: ["ignore", "pipe", "inherit"],
  });
  const workerOutput = collectOutput(worker);
  /** @type {string[]} */
  let tokens = [];
  try {
    assert.deepEqual(await exited(worker), { exitCode: 0, signalCode: null });
    const report = JSON.parse(workerOutput().trim());
    tokens = [report.token];
    assert.equal(report.pids.length, 2);
    assert.notEqual(report.cleanup, null);
    assert.deepEqual(sorted(report.cleanup.killedProcessIds), sorted(report.pids));
    assert.deepEqual(report.cleanup.errors, []);
    await waitUntilOwnedProcessesAreGone(tokens, report.cleanup.killedProcessIds);
  } finally {
    await removeEverything(tokens, [worker]);
  }
});
