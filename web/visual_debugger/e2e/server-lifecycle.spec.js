import { expect, test } from "@playwright/test";

import {
  DEBUGGER_STOP_TIMEOUT_MS,
  startDebugger,
  stopDebugger,
} from "./support/live-debugger.js";

test("normal launcher exits cleanly within the graceful shutdown window", async () => {
  const started = await startDebugger();
  const shutdownStartedAt = performance.now();

  await stopDebugger(started.process);

  const shutdownElapsedMs = performance.now() - shutdownStartedAt;
  expect(shutdownElapsedMs).toBeLessThan(DEBUGGER_STOP_TIMEOUT_MS);
  expect(started.process.exitCode).toBe(0);
  expect(started.process.signalCode).toBeNull();
});
