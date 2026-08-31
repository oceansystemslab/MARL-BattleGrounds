import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test } from "@playwright/test";

import { startIsolatedDevClient, stopDebugger } from "./support/live-debugger.js";

/** @param {import("@playwright/test").Response} response @param {string} commandType */
function isAuthoringResponse(response, commandType) {
  if (
    response.request().method() !== "POST" ||
    new URL(response.url()).pathname !== "/api/dev/authoring/command"
  ) {
    return false;
  }
  try {
    return response.request().postDataJSON()?.command_type === commandType;
  } catch {
    return false;
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} commandType
 * @param {() => Promise<unknown>} activate
 */
async function applyAuthoringCommand(page, commandType, activate) {
  const responsePromise = page.waitForResponse((response) =>
    isAuthoringResponse(response, commandType),
  );
  await activate();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const payload = await response.json();
  expect(payload.command_type).toBe(commandType);
  return payload;
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {() => Promise<unknown>} activate
 */
async function applyLiveCommand(page, activate) {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
    { timeout: 60_000 },
  );
  await activate();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  await expect(page.locator("#connection-status")).toHaveText("Online");
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {readonly string[]} answers
 * @param {() => Promise<unknown>} activate
 */
async function answerPrompts(page, answers, activate) {
  let index = 0;
  /** @type {string[]} */
  const unexpected = [];
  /** @param {import("@playwright/test").Dialog} dialog */
  const handleDialog = async (dialog) => {
    const answer = answers[index];
    index += 1;
    if (answer === undefined) {
      unexpected.push(dialog.message());
      await dialog.dismiss();
      return;
    }
    await dialog.accept(answer);
  };
  page.on("dialog", handleDialog);
  try {
    await activate();
    await expect.poll(() => index).toBe(answers.length);
    expect(unexpected).toEqual([]);
  } finally {
    page.off("dialog", handleDialog);
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} label
 * @param {string | number} value
 */
async function editField(page, label, value) {
  return applyAuthoringCommand(page, "validate", async () => {
    const input = page.getByLabel(label, { exact: true });
    await expect(input).toBeEnabled();
    await input.fill(String(value));
    await input.press("Tab");
  });
}

/** @param {import("@playwright/test").Page} page @param {string} assetId */
async function selectPersistedScenario(page, assetId) {
  const value = await page
    .locator("#devclient-scenario-select")
    .evaluate((select, requestedAssetId) => {
      if (!(select instanceof HTMLSelectElement)) {
        return null;
      }
      for (const option of select.options) {
        try {
          if (JSON.parse(option.value).asset_id === requestedAssetId) {
            return option.value;
          }
        } catch {
          // The built-in arena has an intentionally empty non-JSON value.
        }
      }
      return null;
    }, assetId);
  expect(value).not.toBeNull();
  await page.locator("#devclient-scenario-select").selectOption(value);
}

test("authoring persists through restart and drives same-start Combat comparisons", async ({
  page,
}) => {
  const artifactRoot = await mkdtemp(join(tmpdir(), "marl-devclient-e2e-"));
  const mapId = "e2e-authored-map";
  const scenarioId = "e2e-authored-scenario";
  /** @type {Awaited<ReturnType<typeof startIsolatedDevClient>> | null} */
  let devClient = null;

  try {
    devClient = await startIsolatedDevClient({ artifactRoot });
    await page.goto(devClient.url);
    await expect(page).toHaveTitle("MARL-BattleGrounds DevClient");
    await expect(page.locator("#connection-status")).toHaveText("Online");

    await applyAuthoringCommand(page, "new_map", () =>
      answerPrompts(page, [mapId], () =>
        page.getByRole("button", { name: "Maps", exact: true }).click(),
      ),
    );
    await expect(page.locator("#authoring-shell")).toBeVisible();
    await applyAuthoringCommand(page, "validate", () =>
      page.getByRole("button", { name: "Wall", exact: true }).click(),
    );
    await editField(page, "Center X", 9.5);
    await editField(page, "Center Y", 5.5);

    const savedMap = await applyAuthoringCommand(page, "save", () =>
      page.locator("#authoring-save").click(),
    );
    expect(savedMap.draft.revision).toBe(1);
    await expect(page.locator("#authoring-freeze")).toBeEnabled();
    const frozenMap = await applyAuthoringCommand(page, "freeze", () =>
      page.locator("#authoring-freeze").click(),
    );
    expect(frozenMap.candidate.candidate_id).toMatch(/^[a-f0-9]{64}$/u);

    await applyAuthoringCommand(page, "list", () =>
      page.getByRole("button", { name: "Combat Debugger", exact: true }).click(),
    );
    await page.getByRole("button", { name: "Maps", exact: true }).click();
    await applyAuthoringCommand(page, "open", () =>
      answerPrompts(page, [mapId], () => page.locator("#authoring-open").click()),
    );
    await expect(page.locator("#authoring-title")).toHaveText("Untitled map");
    await expect(
      page.locator('.authoring-svg-object[data-object-id="wall-1"]'),
    ).toBeVisible();

    await applyAuthoringCommand(page, "list", () =>
      page.getByRole("button", { name: "Combat Debugger", exact: true }).click(),
    );
    await page
      .locator("#authoring-new-scenario-mode")
      .selectOption("copy_saved_map", { force: true });
    await applyAuthoringCommand(page, "new_scenario", () =>
      answerPrompts(page, [scenarioId, mapId], () =>
        page.getByRole("button", { name: "Scenarios", exact: true }).click(),
      ),
    );
    await expect(
      page.locator('.authoring-svg-object[data-object-id="wall-1"]'),
    ).toBeVisible();
    await editField(page, "Step count", 7);
    await editField(page, "Team A score", 1);
    await editField(page, "Team B score", 2);

    /** @type {ReadonlyArray<readonly [string, string | number]>} */
    const studyFields = [
      ["Purpose / research question", "Exercise authored TDM pressure."],
      ["Hypothesis", "The scripted opponent reproduces the staged pressure."],
      ["Expected public behavior", "Team B advances while Team A holds."],
      ["Focal role template", "Hold the center lane."],
      ["Cooperative role template", "Protect the focal agent."],
      ["Adversarial role template", "Apply deterministic TDM pressure."],
      ["Matched seeds", "7, 11"],
      ["Primary measurement", "focal survival"],
      ["Violation declarations", "none"],
      ["Completion / censoring", "Complete at terminal; otherwise right-censor."],
      ["Success policy ID", "tdm-success"],
      ["Success policy version", 1],
      ["Completion policy ID", "tdm-completion"],
      ["Completion policy version", 1],
      ["Partial-result policy ID", "tdm-partial"],
      ["Partial-result policy version", 1],
      ["Team B pressure protocol ID", "scripted-tdm-pressure"],
      ["Team B pressure protocol version", 1],
      [
        "Team B pressure protocol digest",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ],
    ];
    let finalValidation = null;
    for (const [label, value] of studyFields) {
      finalValidation = await editField(page, label, value);
    }
    expect(finalValidation.validation.execution_valid).toBe(true);
    expect(finalValidation.validation.freeze_qualified).toBe(true);
    const focalEffectiveSpeed = finalValidation.validation.effective_movement_speeds[0];
    expect(focalEffectiveSpeed).toEqual(expect.any(Number));
    await page.getByRole("button", { name: "A1 · mage", exact: true }).click();
    await expect(
      page.getByLabel("Current effective speed", { exact: true }),
    ).toHaveValue(String(focalEffectiveSpeed));

    const savedScenario = await applyAuthoringCommand(page, "save", () =>
      page.locator("#authoring-save").click(),
    );
    expect(savedScenario.draft.revision).toBe(1);
    await expect(page.locator("#authoring-freeze")).toBeEnabled();
    const frozenScenario = await applyAuthoringCommand(page, "freeze", () =>
      page.locator("#authoring-freeze").click(),
    );
    expect(frozenScenario.candidate.candidate_id).toMatch(/^[a-f0-9]{64}$/u);

    await applyAuthoringCommand(page, "open_in_debug", () =>
      page.locator("#authoring-open-debug").click(),
    );
    await expect(
      page.getByRole("button", { name: "Combat Debugger", exact: true }),
    ).toHaveAttribute("aria-current", "page");
    await expect(page.locator("#step-value")).toHaveText("7");
    await expect(page.locator("#devclient-team-b-controller")).toHaveValue("manual");
    await expect(page.locator("#devclient-information-mode")).toHaveValue("shared_obs");

    await applyLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText("8");
    await applyLiveCommand(page, () => page.locator("#reset-button").click());
    await expect(page.locator("#step-value")).toHaveText("7");

    await applyLiveCommand(page, () =>
      page.locator("#devclient-team-b-controller").selectOption("scripted_tdm"),
    );
    await expect(page.locator("#devclient-team-b-controller")).toHaveValue(
      "scripted_tdm",
    );
    await expect(page.locator("#step-value")).toHaveText("7");
    await applyLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText("8");
    await applyLiveCommand(page, () => page.locator("#reset-button").click());
    await expect(page.locator("#step-value")).toHaveText("7");

    await applyLiveCommand(page, () =>
      page.locator("#devclient-information-mode").selectOption("no_shared_obs"),
    );
    await expect(page.locator("#devclient-information-mode")).toHaveValue(
      "no_shared_obs",
    );
    await expect(page.locator("#step-value")).toHaveText("7");

    await stopDebugger(devClient.process);
    devClient = await startIsolatedDevClient({ artifactRoot });
    await page.goto(devClient.url);
    await expect(page.locator("#connection-status")).toHaveText("Online");
    await expect
      .poll(async () =>
        page.locator("#devclient-scenario-select option").evaluateAll((options) =>
          options.some((option) => {
            if (!(option instanceof HTMLOptionElement)) {
              return false;
            }
            try {
              return JSON.parse(option.value).asset_id === "e2e-authored-scenario";
            } catch {
              return false;
            }
          }),
        ),
      )
      .toBe(true);
    await selectPersistedScenario(page, scenarioId);
    await applyAuthoringCommand(page, "open_in_debug", () =>
      page.locator("#devclient-scenario-load").click(),
    );
    await expect(page.locator("#step-value")).toHaveText("7");
    await applyLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText("8");
  } finally {
    await stopDebugger(devClient?.process ?? null);
    await rm(artifactRoot, { recursive: true, force: true });
  }
});
