import { spawn } from "node:child_process";
import { once } from "node:events";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPOSITORY_ROOT = resolve(HERE, "../../..");

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";

/**
 * @returns {Promise<string>}
 */
function startDebugger() {
  return new Promise((resolveUrl, reject) => {
    const child = spawn(
      "uv",
      [
        "run",
        "python",
        "-u",
        "scripts/dev/debug_renderer.py",
        "--ui",
        "browser",
        "--no-open",
        "--port",
        "0",
        "--scenario",
        "arena_5v5",
      ],
      {
        cwd: REPOSITORY_ROOT,
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    serverProcess = child;
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Debugger startup timed out.\n${stderr}`));
    }, 60_000);

    child.once("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
      const match = stdout.match(
        /Visual debugger: (http:\/\/127\.0\.0\.1:\d+\/#token=[A-Za-z0-9_-]+)/,
      );
      if (match) {
        clearTimeout(timeout);
        resolveUrl(match[1]);
      }
    });
    child.once("exit", (code) => {
      clearTimeout(timeout);
      if (!debuggerUrl) {
        reject(
          new Error(`Debugger exited before startup with code ${code}.\n${stderr}`),
        );
      }
    });
  });
}

async function stopDebugger() {
  const child = serverProcess;
  serverProcess = null;
  if (!child || child.exitCode !== null) {
    return;
  }
  child.kill("SIGINT");
  await Promise.race([
    once(child, "exit"),
    new Promise((resolveDelay) => setTimeout(resolveDelay, 5_000)),
  ]);
  if (child.exitCode === null) {
    child.kill("SIGKILL");
    await once(child, "exit");
  }
}

test.beforeAll(async () => {
  debuggerUrl = await startDebugger();
});

test.afterAll(stopDebugger);

/**
 * @param {import("@playwright/test").Page} page
 * @returns {Promise<number>}
 */
async function currentRevision(page) {
  return Number(await page.locator("#revision-value").textContent());
}

/**
 * @param {import("@playwright/test").Page} page
 * @returns {Promise<number>}
 */
async function currentStep(page) {
  return Number(await page.locator("#step-value").textContent());
}

test("battlefield commands preserve UI and simulator revision boundaries", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText("0");
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(page.locator("#battlefield .agent")).toHaveCount(10);
  await expect(page.locator("#roster .roster-row")).toHaveCount(10);
  expect(page.url()).not.toContain("token=");

  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Move northwest" })).toBeFocused();
  await expect(page.locator("#revision-value")).toHaveText("0");
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("button", { name: "Move north", exact: true }),
  ).toBeFocused();

  await battlefield.focus();
  await page.keyboard.press("d");
  await expect(page.locator("#revision-value")).toHaveText("1");
  await expect(page.locator("#step-value")).toHaveText("0");

  await battlefield.evaluate((element) => {
    element.dispatchEvent(
      new KeyboardEvent("keydown", {
        bubbles: true,
        cancelable: true,
        key: "Enter",
        repeat: true,
      }),
    );
  });
  await expect(page.locator("#notice")).toContainText(
    "Repeated submission input ignored.",
  );
  await expect(page.locator("#revision-value")).toHaveText("1");
  await expect(page.locator("#step-value")).toHaveText("0");

  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#revision-value")).toHaveText("2", {
    timeout: 120_000,
  });
  await expect(page.locator("#step-value")).toHaveText("1");
  await expect(page.locator("#transition-value")).toHaveText("1");

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText("2");
  await expect(page.locator("#step-value")).toHaveText("1");
  expect(page.url()).not.toContain("token=");
});

test("pointer, roster, toolbar, and command-deck controls use the live service", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  let revision = await currentRevision(page);

  const targetButton = page.getByRole("button", { name: "Target id_6" });
  await expect(targetButton).toBeVisible();
  await targetButton.click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator('.roster-row[data-selected="true"]')).toHaveAttribute(
    "aria-label",
    "Agent id_6",
  );

  await page.getByRole("button", { name: "Control id_1" }).click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator('.roster-row[data-controlled="true"]')).toHaveAttribute(
    "aria-label",
    "Agent id_1",
  );

  await page
    .locator('#battlefield .agent[data-slot="2"] .agent-body')
    .click({ modifiers: ["Shift"] });
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator('.roster-row[data-controlled="true"]')).toHaveAttribute(
    "aria-label",
    "Agent id_2",
  );

  await page.locator('#battlefield .agent[data-slot="7"] .agent-body').click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator('.roster-row[data-selected="true"]')).toHaveAttribute(
    "aria-label",
    "Agent id_7",
  );

  await page.locator("#preset-select").selectOption("debug");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#preset-select")).toHaveValue("debug");

  await page.locator("#view-select").selectOption("pov");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#view-select")).toHaveValue("pov");

  await page.locator("#view-select").selectOption("researcher");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#view-select")).toHaveValue("researcher");

  await page.locator("#scenario-select").selectOption("basic_support");
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#scenario-select")).toHaveValue("basic_support");
  await expect(page.locator("#step-value")).toHaveText("0");

  await page.getByRole("button", { name: "Reset" }).click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#step-value")).toHaveText("0");

  await page.getByRole("button", { name: "Move east" }).click();
  revision += 1;
  await expect(page.locator("#revision-value")).toHaveText(String(revision));
  await expect(page.locator("#step-value")).toHaveText("0");
});

test("a stale tab adopts the latest frame without replaying its command", async ({
  context,
  page,
}) => {
  await page.goto(debuggerUrl);
  const stalePage = await context.newPage();
  await stalePage.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(stalePage.locator("#connection-status")).toHaveText("Online");
  const baseRevision = await currentRevision(page);
  await expect(stalePage.locator("#revision-value")).toHaveText(String(baseRevision));

  await page.getByRole("button", { name: "Ranges" }).click();
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 1));

  await stalePage.getByRole("button", { name: "Verbosity" }).click();
  await expect(stalePage.locator("#revision-value")).toHaveText(
    String(baseRevision + 1),
  );
  await expect(stalePage.locator("#notice")).toContainText("stale");
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 1));

  await stalePage.getByRole("button", { name: "Verbosity" }).click();
  await expect(stalePage.locator("#revision-value")).toHaveText(
    String(baseRevision + 2),
  );

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 2));
  await stalePage.close();
});

test("a lost applied response requires GET resync and never replays submit", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const baseRevision = await currentRevision(page);
  const baseStep = await currentStep(page);
  let interceptedCommands = 0;
  let appliedStatus = 0;
  await page.route("**/api/command", async (route) => {
    interceptedCommands += 1;
    const response = await route.fetch();
    appliedStatus = response.status();
    await route.abort("failed");
  });

  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect.poll(() => appliedStatus).toBe(200);
  await page.keyboard.press("v");
  await expect.poll(() => interceptedCommands).toBe(1);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Help" })).toBeFocused();

  await page.unroute("**/api/command");
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 1));
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));

  await battlefield.focus();
  await page.keyboard.press("g");
  await expect(page.locator("#revision-value")).toHaveText(String(baseRevision + 2));
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
});

test("Exit button reaches the authenticated shutdown path", async ({ page }) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await page.getByRole("button", { name: "Exit debugger" }).click();
  await expect(page.locator("#connection-status")).toHaveText("Shutting down");
  await expect(page.locator("#notice")).toContainText("Exit accepted");
});
