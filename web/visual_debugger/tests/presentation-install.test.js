import assert from "node:assert/strict";
import test from "node:test";

import { PresentationInstallCoordinator } from "../src/presentation-install.js";

class JoinRaceError extends Error {}

/**
 * @returns {{
 *   promise: Promise<any>,
 *   resolve: (value: any) => void,
 *   reject: (reason?: any) => void,
 * }}
 */
function deferred() {
  /** @type {(value: any) => void} */
  let resolve = () => {};
  /** @type {(reason?: any) => void} */
  let reject = () => {};
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function harness() {
  /** @type {string[]} */
  const clears = [];
  /** @type {Readonly<Record<string, any>>[]} */
  const installs = [];
  const coordinator = new PresentationInstallCoordinator({
    clear: (reason) => clears.push(reason),
    install: (joined) => installs.push(joined),
    isJoinRace: (error) => error instanceof JoinRaceError,
  });
  return { clears, coordinator, installs };
}

test("a newer authority generation discards an older completed GET pair", async () => {
  const { clears, coordinator, installs } = harness();
  const oldAttempt = deferred();
  const current = Object.freeze({ transport: { revision: 2 }, presentation: {} });
  const oldPromise = coordinator.installFromGet({
    reason: "initial",
    getJoined: () => oldAttempt.promise,
  });
  const currentResult = await coordinator.installFromGet({
    reason: "recipient_change",
    getJoined: async () => current,
  });
  oldAttempt.resolve(Object.freeze({ transport: { revision: 1 }, presentation: {} }));
  const oldResult = await oldPromise;

  assert.deepEqual(clears, ["initial", "recipient_change"]);
  assert.equal(currentResult.status, "installed");
  assert.equal(oldResult.status, "superseded");
  assert.deepEqual(installs, [current]);
});

test("a GET join race performs one fresh GET and installs only its pair", async () => {
  const { coordinator, installs } = harness();
  const current = Object.freeze({ transport: { revision: 2 }, presentation: {} });
  let gets = 0;
  const result = await coordinator.installFromGet({
    reason: "reconnect",
    getJoined: async () => {
      gets += 1;
      if (gets === 1) {
        throw new JoinRaceError("mixed revision");
      }
      return current;
    },
  });

  assert.equal(gets, 2);
  assert.equal(result.status, "installed");
  if (result.status !== "installed") {
    throw new Error("Expected the resynchronized GET pair to install.");
  }
  assert.equal(result.resynchronized, true);
  assert.deepEqual(installs, [current]);
});

test("a second GET join race fails closed without a third attempt", async () => {
  const { coordinator, installs } = harness();
  let gets = 0;
  await assert.rejects(
    coordinator.installFromGet({
      reason: "reconnect",
      getJoined: async () => {
        gets += 1;
        throw new JoinRaceError(`mixed revision ${gets}`);
      },
    }),
    /mixed revision 2/u,
  );
  assert.equal(gets, 2);
  assert.deepEqual(installs, []);
});

test("a command join race resynchronizes with GET and never retries the command", async () => {
  const { coordinator, installs } = harness();
  const current = Object.freeze({ transport: { revision: 4 }, presentation: {} });
  let commands = 0;
  let gets = 0;
  const result = await coordinator.installFromCommand({
    reason: "replay_command",
    sendCommand: async () => {
      commands += 1;
      return Object.freeze({ result: "applied", frame: { revision: 3 } });
    },
    joinCommandResult: async () => {
      throw new JoinRaceError("presentation advanced");
    },
    getJoined: async () => {
      gets += 1;
      return current;
    },
  });

  assert.equal(commands, 1);
  assert.equal(gets, 1);
  assert.equal(result.status, "installed");
  if (result.status !== "installed") {
    throw new Error("Expected the command resynchronization pair to install.");
  }
  assert.equal(result.resynchronized, true);
  assert.deepEqual(installs, [current]);
});

test("a newer generation discards an older command response before joining it", async () => {
  const { coordinator, installs } = harness();
  const oldCommand = deferred();
  const current = Object.freeze({ transport: { revision: 9 }, presentation: {} });
  let commandJoins = 0;
  let resyncGets = 0;
  const oldPromise = coordinator.installFromCommand({
    reason: "live_command",
    sendCommand: () => oldCommand.promise,
    joinCommandResult: async () => {
      commandJoins += 1;
      return Object.freeze({});
    },
    getJoined: async () => {
      resyncGets += 1;
      return Object.freeze({});
    },
  });
  await coordinator.installFromGet({
    reason: "recipient_change",
    getJoined: async () => current,
  });
  oldCommand.resolve(Object.freeze({ result: "applied" }));
  const oldResult = await oldPromise;

  assert.equal(oldResult.status, "superseded");
  assert.equal(commandJoins, 0);
  assert.equal(resyncGets, 0);
  assert.deepEqual(installs, [current]);
});

test("a newer generation discards an old command's pending GET without exposing its result", async () => {
  const { coordinator, installs } = harness();
  const oldGet = deferred();
  const oldGetStarted = deferred();
  const commandResult = Object.freeze({ result: "applied", frame: { revision: 8 } });
  const current = Object.freeze({ transport: { revision: 10 }, presentation: {} });
  let commands = 0;
  let gets = 0;
  const oldPromise = coordinator.installFromCommand({
    reason: "live_command",
    sendCommand: async () => {
      commands += 1;
      return commandResult;
    },
    joinCommandResult: async () => {
      throw new JoinRaceError("presentation advanced");
    },
    getJoined: () => {
      gets += 1;
      oldGetStarted.resolve(undefined);
      return oldGet.promise;
    },
  });
  await oldGetStarted.promise;
  await coordinator.installFromGet({
    reason: "recipient_change",
    getJoined: async () => current,
  });
  oldGet.resolve(Object.freeze({ transport: { revision: 9 }, presentation: {} }));
  const oldResult = await oldPromise;

  assert.deepEqual(oldResult, { status: "superseded" });
  assert.equal(commands, 1);
  assert.equal(gets, 1);
  assert.deepEqual(installs, [current]);
});

test("a command race followed by a raced GET fails without a second GET or command", async () => {
  const { coordinator, installs } = harness();
  let commands = 0;
  let gets = 0;
  await assert.rejects(
    coordinator.installFromCommand({
      reason: "replay_command",
      sendCommand: async () => {
        commands += 1;
        return Object.freeze({ result: "applied" });
      },
      joinCommandResult: async () => {
        throw new JoinRaceError("command pair raced");
      },
      getJoined: async () => {
        gets += 1;
        throw new JoinRaceError("fresh GET pair also raced");
      },
    }),
    /fresh GET pair also raced/u,
  );

  assert.equal(commands, 1);
  assert.equal(gets, 1);
  assert.deepEqual(installs, []);
});

test("non-race command failures clear authority without command or GET retry", async () => {
  const { clears, coordinator, installs } = harness();
  let commands = 0;
  let gets = 0;
  await assert.rejects(
    coordinator.installFromCommand({
      reason: "live_command",
      sendCommand: async () => {
        commands += 1;
        return Object.freeze({ result: "applied" });
      },
      joinCommandResult: async () => {
        throw new TypeError("invalid presentation payload");
      },
      getJoined: async () => {
        gets += 1;
        return Object.freeze({});
      },
    }),
    /invalid presentation payload/u,
  );
  assert.deepEqual(clears, ["live_command"]);
  assert.equal(commands, 1);
  assert.equal(gets, 0);
  assert.deepEqual(installs, []);
});
