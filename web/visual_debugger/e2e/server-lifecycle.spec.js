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
  let releaseMain = () => {};
  /** @type {Promise<void>} */
  const mainRelease = new Promise((resolve) => {
    releaseMain = () => resolve();
  });
  let markMainHeld = () => {};
  /** @type {Promise<void>} */
  const mainHeld = new Promise((resolve) => {
    markMainHeld = () => resolve();
  });

  await page.route("**/src/main.js", async (route) => {
    const response = await route.fetch();
    markMainHeld();
    await mainRelease;
    await route.fulfill({ response });
  });

  try {
    await page.goto(replay.url, { waitUntil: "commit" });
    await mainHeld;
    await page.locator("#battlefield").waitFor({ state: "attached" });
    await expect(page.locator("html")).not.toHaveAttribute("data-product-kind", /.+/u);
    await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
    await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
    await expect(page.locator("#battlefield")).toHaveAttribute(
      "aria-label",
      "Battlefield unavailable until product identity is validated.",
    );
    await expect(page.locator("#battlefield-instructions")).toBeHidden();
    await expect(page.locator("#battlefield-instructions")).toHaveText(
      "Product-specific Battlefield instructions are loading.",
    );

    const visibleProductRoots = await page
      .locator("[data-live-only], [data-replay-only]")
      .evaluateAll((elements) =>
        elements
          .filter((element) => getComputedStyle(element).display !== "none")
          .map((element) => element.id || element.className),
      );
    expect(visibleProductRoots).toEqual([]);

    releaseMain();
    await page.waitForLoadState("load");
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
    releaseMain();
    await stopDebugger(replay.process);
  }
});
