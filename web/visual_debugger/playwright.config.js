import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  workers: 1,
  // A whole test has no time limit. Individual failed waits still report errors.
  timeout: 0,
  expect: {
    timeout: 15_000,
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.001,
      scale: "css",
    },
  },
  reporter: "line",
  outputDir: "test-results",
  use: {
    actionTimeout: 10_000,
    browserName: "chromium",
    colorScheme: "dark",
    deviceScaleFactor: 1,
    headless: true,
    locale: "en-GB",
    navigationTimeout: 30_000,
    reducedMotion: "no-preference",
    screenshot: "only-on-failure",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
});
