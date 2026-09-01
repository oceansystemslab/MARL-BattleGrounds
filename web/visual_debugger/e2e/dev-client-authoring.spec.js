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
 * @template T
 * @param {import("@playwright/test").Page} page
 * @param {() => Promise<T>} activate
 * @returns {Promise<T>}
 */
async function expectNoPrompts(page, activate) {
  /** @type {string[]} */
  const messages = [];
  /** @param {import("@playwright/test").Dialog} dialog */
  const handleDialog = async (dialog) => {
    messages.push(dialog.message());
    await dialog.accept();
  };
  page.on("dialog", handleDialog);
  let result;
  try {
    result = await activate();
  } finally {
    page.off("dialog", handleDialog);
  }
  expect(messages).toEqual([]);
  return result;
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {() => Promise<unknown>} activate
 */
async function dismissOnePrompt(page, activate) {
  /** @type {string[]} */
  const messages = [];
  /** @param {import("@playwright/test").Dialog} dialog */
  const handleDialog = async (dialog) => {
    messages.push(dialog.message());
    await dialog.dismiss();
  };
  page.on("dialog", handleDialog);
  try {
    await activate();
    await expect.poll(() => messages.length).toBe(1);
  } finally {
    page.off("dialog", handleDialog);
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {() => Promise<unknown>} activate
 */
async function expectNoAuthoringCommands(page, activate) {
  /** @type {string[]} */
  const commandTypes = [];
  /** @param {import("@playwright/test").Request} request */
  const recordRequest = (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/dev/authoring/command"
    ) {
      try {
        commandTypes.push(request.postDataJSON()?.command_type ?? "unknown");
      } catch {
        commandTypes.push("malformed");
      }
    }
  };
  page.on("request", recordRequest);
  try {
    await activate();
    await page.evaluate(
      () => new Promise((resolve) => requestAnimationFrame(() => resolve(null))),
    );
  } finally {
    page.off("request", recordRequest);
  }
  expect(commandTypes).toEqual([]);
}

/** @param {import("@playwright/test").Page} page */
async function authoringViewBox(page) {
  return page.locator("#authoring-canvas").getAttribute("viewBox");
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
    await expect(page.locator("#devclient-nav")).toBeVisible();
    await expect(page.locator("#devclient-combat-config")).toBeVisible();
    await expect(page.locator("#workspace")).toBeVisible();
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await page.locator("#reconnect-button").click();
    await expect(page.locator("#connection-status")).toHaveText("Online");
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await page.evaluate(() => window.scrollTo(0, 0));

    const initialMap = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "new_map", () =>
        page.getByRole("button", { name: "Maps", exact: true }).click(),
      ),
    );
    expect(initialMap.draft.asset_id).toBe("untitled-map");
    expect(initialMap.draft.revision).toBe(0);
    await expect(page.locator("#authoring-shell")).toBeVisible();
    await expect(page.locator("#workspace")).toBeHidden();
    await expect(page.locator("#devclient-combat-config")).toBeHidden();
    const freshMap = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "new_map", () =>
        page.locator("#authoring-new").click(),
      ),
    );
    expect(freshMap.draft.asset_id).toBe("untitled-map");
    expect(freshMap.draft.revision).toBe(0);

    await page.getByRole("button", { name: "map document", exact: true }).click();
    const mapName = page.getByLabel("Name", { exact: true });
    const originalMapName = await mapName.inputValue();
    await mapName.press("End");
    await mapName.press("r");
    await expect(mapName).toHaveValue(`${originalMapName}r`);
    await mapName.fill(originalMapName);

    const canvas = page.locator("#authoring-canvas");
    await expect(canvas.locator("title")).toHaveCount(0);
    await canvas.hover({ position: { x: 12, y: 12 } });
    await expect(page.locator("#visual-tooltip")).toBeHidden();
    await applyAuthoringCommand(page, "validate", () =>
      page.getByRole("button", { name: "Wall", exact: true }).click(),
    );
    await editField(page, "Center X", 9.5);
    await editField(page, "Center Y", 5.5);
    const wall = page.locator('.authoring-svg-object[data-object-id="wall-1"]');
    await expect(wall.locator("title")).toHaveCount(0);
    await wall.hover();
    await expect(page.locator("#visual-tooltip")).toBeHidden();
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-reset").click(),
    );
    await expect(wall).toHaveCount(0);
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-undo").click(),
    );
    await expect(wall).toBeVisible();
    await page.getByRole("button", { name: "wall-1", exact: true }).click();
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("9.5");

    await expectNoAuthoringCommands(page, () =>
      dismissOnePrompt(page, () => page.locator("#authoring-save").click()),
    );
    await expectNoAuthoringCommands(page, () =>
      answerPrompts(page, ["Invalid ID"], () =>
        page.locator("#authoring-save").click(),
      ),
    );
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("9.5");

    const savedMap = await applyAuthoringCommand(page, "save_as", () =>
      answerPrompts(page, [mapId], () => page.locator("#authoring-save").click()),
    );
    expect(savedMap.draft.asset_id).toBe(mapId);
    expect(savedMap.draft.revision).toBe(1);
    await editField(page, "Center X", 10.5);
    const updatedMap = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "save", () =>
        page.locator("#authoring-save").click(),
      ),
    );
    expect(updatedMap.draft.asset_id).toBe(mapId);
    expect(updatedMap.draft.revision).toBe(2);

    const fullMapViewBox = await authoringViewBox(page);
    const selectedCenter = await page
      .getByLabel("Center X", { exact: true })
      .inputValue();
    const undoWasDisabled = await page.locator("#authoring-undo").isDisabled();
    await canvas.scrollIntoViewIfNeeded();
    const pageScrollBeforeWheel = await page.evaluate(() => window.scrollY);
    await canvas.hover({ position: { x: 240, y: 160 } });
    await page.mouse.wheel(0, 360);
    const zoomedMapViewBox = await authoringViewBox(page);
    expect(zoomedMapViewBox).not.toBe(fullMapViewBox);
    expect(await page.evaluate(() => window.scrollY)).toBe(pageScrollBeforeWheel);

    await expectNoAuthoringCommands(page, () =>
      page.locator("#authoring-recenter").click(),
    );
    expect(await authoringViewBox(page)).toBe(fullMapViewBox);
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue(
      selectedCenter,
    );
    expect(await page.locator("#authoring-undo").isDisabled()).toBe(undoWasDisabled);

    await editField(page, "Center X", 13.5);
    await canvas.hover({ position: { x: 240, y: 160 } });
    await page.mouse.wheel(0, -360);
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-reset").click(),
    );
    expect(await authoringViewBox(page)).toBe(fullMapViewBox);
    await expect(page.getByLabel("Center X", { exact: true })).toHaveCount(0);
    await page.getByRole("button", { name: "wall-1", exact: true }).click();
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("10.5");
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-undo").click(),
    );
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("13.5");
    await canvas.focus();
    await applyAuthoringCommand(page, "validate", () => page.keyboard.press("r"));
    await page.getByRole("button", { name: "wall-1", exact: true }).click();
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("10.5");
    await canvas.hover({ position: { x: 240, y: 160 } });
    await page.mouse.wheel(0, 360);
    await expectNoAuthoringCommands(page, () =>
      page.locator("#authoring-reset").click(),
    );
    expect(await authoringViewBox(page)).toBe(fullMapViewBox);
    await canvas.hover({ position: { x: 240, y: 160 } });
    await page.mouse.wheel(0, -360);
    const retainedMapViewBox = await authoringViewBox(page);

    await expect(page.locator("#authoring-freeze")).toBeEnabled();
    const frozenMap = await applyAuthoringCommand(page, "freeze", () =>
      page.locator("#authoring-freeze").click(),
    );
    expect(frozenMap.candidate.candidate_id).toMatch(/^[a-f0-9]{64}$/u);

    await applyAuthoringCommand(page, "list", () =>
      page.getByRole("button", { name: "Combat Debugger", exact: true }).click(),
    );
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await expect(page.locator("#workspace")).toBeVisible();
    await expect(page.locator("#devclient-combat-config")).toBeVisible();
    await expectNoAuthoringCommands(page, () =>
      page.getByRole("button", { name: "Maps", exact: true }).click(),
    );
    expect(await authoringViewBox(page)).toBe(retainedMapViewBox);
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("10.5");
    await applyAuthoringCommand(page, "open", () =>
      answerPrompts(page, [mapId], () => page.locator("#authoring-open").click()),
    );
    await expect(page.locator("#authoring-title")).toHaveText("Untitled map");
    await expect(
      page.locator('.authoring-svg-object[data-object-id="wall-1"]'),
    ).toBeVisible();
    await page.getByRole("button", { name: "wall-1", exact: true }).click();
    await editField(page, "Center X", 14.5);
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-reset").click(),
    );
    await page.getByRole("button", { name: "wall-1", exact: true }).click();
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("10.5");
    await canvas.hover({ position: { x: 240, y: 160 } });
    await page.mouse.wheel(0, -360);
    const contextualMapViewBox = await authoringViewBox(page);
    const mapValidationEvidence = await page
      .locator("#authoring-problem-list")
      .innerText();

    await applyAuthoringCommand(page, "list", () =>
      page.getByRole("button", { name: "Combat Debugger", exact: true }).click(),
    );
    await page
      .locator("#authoring-new-scenario-mode")
      .selectOption("blank", { force: true });
    const initialScenario = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "new_scenario", () =>
        page.getByRole("button", { name: "Scenarios", exact: true }).click(),
      ),
    );
    expect(initialScenario.draft.asset_id).toBe("untitled-scenario");
    expect(initialScenario.draft.revision).toBe(0);
    await expect(page.locator("#devclient-combat-config")).toBeHidden();
    const freshScenario = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "new_scenario", () =>
        page.locator("#authoring-new").click(),
      ),
    );
    expect(freshScenario.draft.asset_id).toBe("untitled-scenario");
    expect(freshScenario.draft.revision).toBe(0);
    await page.locator("#authoring-new-scenario-mode").selectOption("copy_saved_map");
    await applyAuthoringCommand(page, "new_scenario", () =>
      answerPrompts(page, [mapId], () => page.locator("#authoring-new").click()),
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

    const savedScenario = await applyAuthoringCommand(page, "save_as", () =>
      answerPrompts(page, [scenarioId], () => page.locator("#authoring-save").click()),
    );
    expect(savedScenario.draft.asset_id).toBe(scenarioId);
    expect(savedScenario.draft.revision).toBe(1);
    const updatedScenario = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "save", () =>
        page.locator("#authoring-save").click(),
      ),
    );
    expect(updatedScenario.draft.asset_id).toBe(scenarioId);
    expect(updatedScenario.draft.revision).toBe(2);

    await page.getByRole("button", { name: "scenario document", exact: true }).click();
    await editField(page, "Team B score", 3);
    const fullScenarioViewBox = await authoringViewBox(page);
    const scenarioUndoWasDisabled = await page.locator("#authoring-undo").isDisabled();
    const scenarioScrollBeforeWheel = await page.evaluate(() => window.scrollY);
    await canvas.hover({ position: { x: 240, y: 160 } });
    await page.mouse.wheel(0, 360);
    expect(await authoringViewBox(page)).not.toBe(fullScenarioViewBox);
    expect(await page.evaluate(() => window.scrollY)).toBe(scenarioScrollBeforeWheel);
    await expectNoAuthoringCommands(page, () =>
      page.locator("#authoring-recenter").click(),
    );
    expect(await authoringViewBox(page)).toBe(fullScenarioViewBox);
    await expect(page.getByLabel("Team B score", { exact: true })).toHaveValue("3");
    expect(await page.locator("#authoring-undo").isDisabled()).toBe(
      scenarioUndoWasDisabled,
    );
    await canvas.hover({ position: { x: 240, y: 160 } });
    await page.mouse.wheel(0, 360);
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-reset").click(),
    );
    await expect(page.getByLabel("Team B score", { exact: true })).toHaveValue("2");
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-undo").click(),
    );
    await expect(page.getByLabel("Team B score", { exact: true })).toHaveValue("3");
    await canvas.focus();
    await applyAuthoringCommand(page, "validate", () => page.keyboard.press("r"));
    await expect(page.getByLabel("Team B score", { exact: true })).toHaveValue("2");
    await page.getByRole("button", { name: "A1 · mage", exact: true }).click();
    await canvas.hover({ position: { x: 240, y: 160 } });
    await page.mouse.wheel(0, 360);
    const retainedScenarioViewBox = await authoringViewBox(page);
    const scenarioValidationEvidence = await page
      .locator("#authoring-problem-list")
      .innerText();

    await expectNoAuthoringCommands(page, () =>
      page.getByRole("button", { name: "Maps", exact: true }).click(),
    );
    expect(await authoringViewBox(page)).toBe(contextualMapViewBox);
    await expect(page.locator("#authoring-problem-list")).toHaveText(
      mapValidationEvidence,
    );
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("10.5");
    await expectNoAuthoringCommands(page, () =>
      page.getByRole("button", { name: "Scenarios", exact: true }).click(),
    );
    expect(await authoringViewBox(page)).toBe(retainedScenarioViewBox);
    await expect(page.locator("#authoring-problem-list")).toHaveText(
      scenarioValidationEvidence,
    );
    await expect(
      page.getByRole("button", { name: "A1 · mage", exact: true }),
    ).toHaveAttribute("aria-current", "true");
    await expect(
      page.getByLabel("Current effective speed", { exact: true }),
    ).toHaveValue(String(focalEffectiveSpeed));

    await applyAuthoringCommand(page, "list", () =>
      page.getByRole("button", { name: "Combat Debugger", exact: true }).click(),
    );
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await expectNoAuthoringCommands(page, () =>
      page.getByRole("button", { name: "Scenarios", exact: true }).click(),
    );
    await expect(page.locator("#devclient-combat-config")).toBeHidden();
    await expect(page.locator("#authoring-problem-list")).toHaveText(
      scenarioValidationEvidence,
    );

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
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await expect(page.locator("#workspace")).toBeVisible();
    await expect(page.locator("#devclient-combat-config")).toBeVisible();
    await expect(page.locator("#step-value")).toHaveText("7");
    await expect(page.locator("#devclient-team-b-controller")).toHaveValue("manual");
    await expect(page.locator("#devclient-information-mode")).toHaveValue("shared_obs");

    await applyLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText("8");
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await applyLiveCommand(page, () => page.locator("#reset-button").click());
    await expect(page.locator("#step-value")).toHaveText("7");
    await expect(page.locator("#authoring-shell")).toBeHidden();

    await applyLiveCommand(page, () =>
      page.locator("#devclient-team-b-controller").selectOption("scripted_tdm"),
    );
    await expect(page.locator("#devclient-team-b-controller")).toHaveValue(
      "scripted_tdm",
    );
    await expect(page.locator("#step-value")).toHaveText("7");
    await applyLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText("8");
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await applyLiveCommand(page, () => page.locator("#reset-button").click());
    await expect(page.locator("#step-value")).toHaveText("7");
    await expect(page.locator("#authoring-shell")).toBeHidden();

    await applyLiveCommand(page, () =>
      page.locator("#devclient-information-mode").selectOption("no_shared_obs"),
    );
    await expect(page.locator("#devclient-information-mode")).toHaveValue(
      "no_shared_obs",
    );
    await expect(page.locator("#step-value")).toHaveText("7");
    await expect(page.locator("#authoring-shell")).toBeHidden();

    await stopDebugger(devClient.process);
    devClient = await startIsolatedDevClient({ artifactRoot });
    await page.goto(devClient.url);
    await expect(page.locator("#connection-status")).toHaveText("Online");
    await expect(page.locator("#authoring-shell")).toBeHidden();
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
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await applyLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText("8");
    await expect(page.locator("#authoring-shell")).toBeHidden();
  } finally {
    await stopDebugger(devClient?.process ?? null);
    await rm(artifactRoot, { recursive: true, force: true });
  }
});
