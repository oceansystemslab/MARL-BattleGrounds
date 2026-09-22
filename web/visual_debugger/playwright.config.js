/**
 * @file Configure serial Chromium browser tests for the two native clients.
 * Each shard has one worker. Nothing has a time limit: a whole test, each
 * action, each navigation and each web-first assertion waits until it
 * finishes. A test that hangs stays hung so the cause can be found and fixed.
 * CI rejects focused tests; failed runs retain traces and screenshots. Run
 * through check_frontend.sh for the complete inventory or Playwright for a
 * selected local check.
 */
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  workers: 1,
  // In Playwright 1.62, 0 means "no time limit" for every timeout below.
  // A whole test has no time limit.
  timeout: 0,
  expect: {
    // Web-first assertions such as toHaveText keep retrying until they pass,
    // unless a check of the product's own timing sets its own limit.
    timeout: 0,
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.001,
      scale: "css",
    },
  },
  reporter: "line",
  outputDir: "test-results",
  use: {
    // No limit: full gates run many browsers and servers at once, so a slow
    // action waits instead of failing. This is also the default for
    // waitForResponse, waitForEvent and route or request fetches.
    actionTimeout: 0,
    browserName: "chromium",
    colorScheme: "dark",
    deviceScaleFactor: 1,
    headless: true,
    locale: "en-GB",
    // No limit: page.goto and reload wait until the page loads.
    navigationTimeout: 0,
    reducedMotion: "no-preference",
    screenshot: "only-on-failure",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
});
