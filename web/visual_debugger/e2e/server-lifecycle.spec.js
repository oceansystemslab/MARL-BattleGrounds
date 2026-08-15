import { expect, test } from "@playwright/test";

import {
  DEBUGGER_STOP_TIMEOUT_MS,
  startDebugger,
  stopDebugger,
} from "./support/live-debugger.js";
import { startReplayViewer } from "./support/replay-viewer.js";

test("normal launcher exits cleanly within the graceful shutdown window", async () => {
  const started = await startDebugger();
  const shutdownStartedAt = performance.now();

  await stopDebugger(started.process);

  const shutdownElapsedMs = performance.now() - shutdownStartedAt;
  expect(shutdownElapsedMs).toBeLessThan(DEBUGGER_STOP_TIMEOUT_MS);
  expect(started.process.exitCode).toBe(0);
  expect(started.process.signalCode).toBeNull();
});

test("standalone launchers install their exact product identity", async ({ page }) => {
  const live = await startDebugger();
  try {
    await page.goto(live.url);
    await expect(page).toHaveTitle("MARL-BattleGrounds Combat Debugger");
    await expect(page.locator("#app-title")).toHaveText(
      "MARL-BattleGrounds Combat Debugger",
    );
    await expect(page.locator("html")).toHaveAttribute(
      "data-product-kind",
      "combat_debugger",
    );
    await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "live");
  } finally {
    await stopDebugger(live.process);
  }

  const replay = await startReplayViewer({
    sampleReplay: "death-respawn-shield",
  });
  try {
    await page.goto(replay.url);
    await expect(page).toHaveTitle("MARL-BattleGrounds Replay Viewer");
    await expect(page.locator("#app-title")).toHaveText(
      "MARL-BattleGrounds Replay Viewer",
    );
    await expect(page.locator("html")).toHaveAttribute(
      "data-product-kind",
      "replay_viewer",
    );
    await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
  } finally {
    await stopDebugger(replay.process);
  }
});

test("invalid bootstrap shapes block startup before the first frame request", async ({
  context,
}) => {
  const started = await startDebugger();
  const cases = [
    ["missing", "delete globalThis.__MARL_DEBUGGER_BOOTSTRAP__;"],
    [
      "extra-field",
      'globalThis.__MARL_DEBUGGER_BOOTSTRAP__ = {schema_version: 1, product_kind: "combat_debugger", extra: true};',
    ],
    [
      "unsupported-kind",
      'globalThis.__MARL_DEBUGGER_BOOTSTRAP__ = {schema_version: 1, product_kind: "combined_debugger"};',
    ],
  ];
  try {
    for (const [name, bootstrapBody] of cases) {
      const page = await context.newPage();
      let frameRequests = 0;
      page.on("request", (request) => {
        if (new URL(request.url()).pathname === "/api/frame") {
          frameRequests += 1;
        }
      });
      await page.route("**/bootstrap.js", (route) =>
        route.fulfill({
          body: bootstrapBody,
          contentType: "text/javascript",
          status: 200,
        }),
      );

      await page.goto(started.url);

      await expect(page.locator("#notice")).toContainText("Startup blocked:");
      await expect(page).toHaveTitle("MARL-BattleGrounds");
      await expect(page.locator("html")).not.toHaveAttribute(
        "data-product-kind",
        /.+/u,
      );
      expect(frameRequests, name).toBe(0);
      await page.close();
    }
  } finally {
    await stopDebugger(started.process);
  }
});
