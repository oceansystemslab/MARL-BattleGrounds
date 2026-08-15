import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  workers: 1,
  timeout: 180_000,
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
    browserName: "chromium",
    colorScheme: "dark",
    deviceScaleFactor: 1,
    headless: true,
    locale: "en-GB",
    reducedMotion: "no-preference",
    screenshot: "only-on-failure",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
});
