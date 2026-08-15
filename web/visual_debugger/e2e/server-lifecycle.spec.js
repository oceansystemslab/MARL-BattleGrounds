import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import { startReplayViewer } from "./support/replay-viewer.js";

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
