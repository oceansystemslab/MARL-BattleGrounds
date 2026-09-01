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

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} selector
 * @param {string} assetId
 */
async function selectPersistedAsset(page, selector, assetId) {
  const value = await page.locator(selector).evaluate((select, requestedAssetId) => {
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
  await page.locator(selector).selectOption(value);
}

/** @param {import("@playwright/test").Page} page */
async function expectAuthoringIdle(page) {
  await expect(page.locator("#authoring-shell")).toHaveAttribute("aria-busy", "false");
}

/** @param {import("@playwright/test").Page} page */
async function expectSpawnPadRadius(page) {
  const radii = await page
    .locator('#authoring-canvas circle[data-kind="spawn_pad"]')
    .evaluateAll((circles) => circles.map((circle) => circle.getAttribute("r")));
  expect(radii.length).toBeGreaterThan(0);
  expect(new Set(radii)).toEqual(new Set(["0.5"]));
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
    await expect(page.locator("#authoring-persistence-status")).toHaveText(
      "Unsaved map draft",
    );
    await expectSpawnPadRadius(page);

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
    const wall = page.locator('.authoring-svg-object[data-object-id="obstacle-0"]');
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
    await page.getByRole("button", { name: "obstacle-0", exact: true }).click();
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("9.5");
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-duplicate").click(),
    );
    await expect(
      page.getByRole("button", { name: "obstacle-1", exact: true }),
    ).toBeVisible();
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-undo").click(),
    );
    await expect(
      page.getByRole("button", { name: "obstacle-1", exact: true }),
    ).toHaveCount(0);
    await page.getByRole("button", { name: "obstacle-0", exact: true }).click();

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
    await expectAuthoringIdle(page);
    await expect(page.locator("#authoring-persistence-status")).toHaveText(
      `Saved map ${mapId} · revision 1 · artifacts/dev_client/drafts/maps/${mapId}/r1.json`,
    );
    await editField(page, "Center X", 10.5);
    await expect(page.locator("#authoring-persistence-status")).toHaveText(
      `Unsaved changes · last saved map ${mapId} revision 1`,
    );
    await page.route(
      "**/api/dev/authoring/command",
      async (route) => {
        await route.fulfill({
          status: 409,
          contentType: "application/json",
          body: JSON.stringify({ detail: "simulated stale revision" }),
        });
      },
      { times: 1 },
    );
    await page.locator("#authoring-save").click();
    await expectAuthoringIdle(page);
    await expect(page.locator("#authoring-persistence-status")).toHaveText(
      `Unsaved changes · last saved map ${mapId} revision 1`,
    );
    const updatedMap = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "save", () =>
        page.locator("#authoring-save").click(),
      ),
    );
    expect(updatedMap.draft.asset_id).toBe(mapId);
    expect(updatedMap.draft.revision).toBe(2);
    await expectAuthoringIdle(page);
    await expect(page.locator("#authoring-persistence-status")).toHaveText(
      `Saved map ${mapId} · revision 2 · artifacts/dev_client/drafts/maps/${mapId}/r2.json`,
    );

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
    await page.getByRole("button", { name: "obstacle-0", exact: true }).click();
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("10.5");
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-undo").click(),
    );
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("13.5");
    await canvas.focus();
    await applyAuthoringCommand(page, "validate", () => page.keyboard.press("r"));
    await page.getByRole("button", { name: "obstacle-0", exact: true }).click();
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
    await expectAuthoringIdle(page);
    await expect(page.locator("#authoring-persistence-status")).toHaveText(
      `Frozen candidate ${frozenMap.candidate.candidate_id} · artifacts/dev_client/candidates/map-${frozenMap.candidate.candidate_id}.json`,
    );

    const mapPreviewRequest = page.waitForRequest(
      (request) =>
        request.method() === "POST" &&
        new URL(request.url()).pathname === "/api/dev/authoring/command" &&
        request.postDataJSON()?.command_type === "open_in_debug",
    );
    await applyAuthoringCommand(page, "open_in_debug", () =>
      page.locator("#authoring-open-debug").click(),
    );
    expect((await mapPreviewRequest).postDataJSON()?.source).toMatchObject({
      source_kind: "current_buffer",
      asset_kind: "map",
    });
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await expect(page.locator("#workspace")).toBeVisible();
    await expect(page.locator("#devclient-combat-config")).toBeVisible();
    await expect(
      page
        .locator("#devclient-scenario-select option")
        .filter({ hasText: "Map preview" })
        .first(),
    ).toContainText("default 5v5 TDM");
    await applyAuthoringCommand(page, "list", () =>
      page.getByRole("button", { name: "Maps", exact: true }).click(),
    );
    expect(await authoringViewBox(page)).toBe(retainedMapViewBox);
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("10.5");
    await selectPersistedAsset(page, "#authoring-saved-draft-select", mapId);
    await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "open", () =>
        page.locator("#authoring-open").click(),
      ),
    );
    await expect(page.locator("#authoring-title")).toHaveText("Untitled map");
    await expect(
      page.locator('.authoring-svg-object[data-object-id="obstacle-0"]'),
    ).toBeVisible();
    await page.getByRole("button", { name: "obstacle-0", exact: true }).click();
    await editField(page, "Center X", 14.5);
    await applyAuthoringCommand(page, "validate", () =>
      page.locator("#authoring-reset").click(),
    );
    await page.getByRole("button", { name: "obstacle-0", exact: true }).click();
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
    await selectPersistedAsset(page, "#authoring-new-scenario-source", mapId);
    const copiedScenario = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "new_scenario", () =>
        page.locator("#authoring-new").click(),
      ),
    );
    expect(copiedScenario.draft.content.source_map_provenance).toMatchObject({
      asset_id: mapId,
      revision: 2,
    });
    await expect(
      page.locator('.authoring-svg-object[data-object-id="obstacle-0"]'),
    ).toBeVisible();
    await expectSpawnPadRadius(page);
    await page.getByRole("button", { name: "scenario document", exact: true }).click();
    await expect(page.getByLabel("Notes", { exact: true })).toBeVisible();
    await expect(page.getByText("Role", { exact: true })).toHaveCount(0);
    await expect(page.getByText("Controlled Study", { exact: true })).toHaveCount(0);
    await expect(page.getByText("Study Identities", { exact: true })).toHaveCount(0);
    await editField(page, "Step count", 7);
    await editField(page, "Team A score", 1);
    const finalValidation = await editField(page, "Team B score", 2);
    expect(finalValidation.validation.execution_valid).toBe(true);
    expect(finalValidation.validation.freeze_qualified).toBe(true);
    const firstAgentEffectiveSpeed =
      finalValidation.validation.effective_movement_speeds[0];
    expect(firstAgentEffectiveSpeed).toEqual(expect.any(Number));
    await page.getByRole("button", { name: "A1 · mage", exact: true }).click();
    const effectiveSpeedField = page.getByLabel("Current effective speed", {
      exact: true,
    });
    const displayedEffectiveSpeed = await effectiveSpeedField.inputValue();
    expect(displayedEffectiveSpeed).toMatch(/^-?\d+(?:\.\d{1,2})?$/u);
    expect(Number(displayedEffectiveSpeed)).toBeCloseTo(firstAgentEffectiveSpeed, 2);
    const [inspectorBox, effectiveSpeedBox] = await Promise.all([
      page.locator("#authoring-inspector-form").boundingBox(),
      effectiveSpeedField.boundingBox(),
    ]);
    if (inspectorBox === null || effectiveSpeedBox === null) {
      throw new Error("The effective-speed field must have measurable layout boxes.");
    }
    expect(effectiveSpeedBox.x).toBeGreaterThanOrEqual(inspectorBox.x);
    expect(effectiveSpeedBox.x + effectiveSpeedBox.width).toBeLessThanOrEqual(
      inspectorBox.x + inspectorBox.width + 0.5,
    );
    const mechanicsPresentation = await page
      .locator("#authoring-inspector-form")
      .evaluate((form) =>
        [...form.querySelectorAll("fieldset")]
          .filter((fieldset) =>
            fieldset.querySelector("legend")?.textContent?.includes("mechanics"),
          )
          .flatMap((fieldset) => [
            fieldset.textContent ?? "",
            ...[...fieldset.querySelectorAll("input, select, textarea")].map(
              (control) =>
                control instanceof HTMLInputElement ||
                control instanceof HTMLSelectElement ||
                control instanceof HTMLTextAreaElement
                  ? control.value
                  : "",
            ),
          ])
          .join(" "),
      );
    expect(mechanicsPresentation).not.toContain("_");
    const inspectorOverflow = await page
      .locator("#authoring-inspector-form")
      .evaluate((form) => ({
        scrollWidth: form.scrollWidth,
        clientWidth: form.clientWidth,
      }));
    expect(inspectorOverflow.scrollWidth).toBeLessThanOrEqual(
      inspectorOverflow.clientWidth,
    );

    const savedScenario = await applyAuthoringCommand(page, "save_as", () =>
      answerPrompts(page, [scenarioId], () => page.locator("#authoring-save").click()),
    );
    expect(savedScenario.draft.asset_id).toBe(scenarioId);
    expect(savedScenario.draft.revision).toBe(1);
    await expectAuthoringIdle(page);
    await expect(page.locator("#authoring-persistence-status")).toHaveText(
      `Saved scenario ${scenarioId} · revision 1 · artifacts/dev_client/drafts/scenarios/${scenarioId}/r1.json`,
    );
    const updatedScenario = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "save", () =>
        page.locator("#authoring-save").click(),
      ),
    );
    expect(updatedScenario.draft.asset_id).toBe(scenarioId);
    expect(updatedScenario.draft.revision).toBe(2);
    await expectAuthoringIdle(page);
    await expect(page.locator("#authoring-persistence-status")).toHaveText(
      `Saved scenario ${scenarioId} · revision 2 · artifacts/dev_client/drafts/scenarios/${scenarioId}/r2.json`,
    );

    await page
      .locator("#authoring-new-scenario-mode")
      .selectOption("duplicate_saved_scenario");
    await selectPersistedAsset(page, "#authoring-new-scenario-source", scenarioId);
    const duplicatedScenario = await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "new_scenario", () =>
        page.locator("#authoring-new").click(),
      ),
    );
    expect(duplicatedScenario.draft.asset_id).toBe("untitled-scenario");
    expect(duplicatedScenario.draft.revision).toBe(0);
    expect(duplicatedScenario.draft.content.global_state).toEqual(
      updatedScenario.draft.content.global_state,
    );
    await selectPersistedAsset(page, "#authoring-saved-draft-select", scenarioId);
    await expectNoPrompts(page, () =>
      applyAuthoringCommand(page, "open", () =>
        page.locator("#authoring-open").click(),
      ),
    );

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

    await applyAuthoringCommand(page, "list", () =>
      page.getByRole("button", { name: "Maps", exact: true }).click(),
    );
    expect(await authoringViewBox(page)).toBe(contextualMapViewBox);
    await expect(page.locator("#authoring-problem-list")).toHaveText(
      mapValidationEvidence,
    );
    await expect(page.getByLabel("Center X", { exact: true })).toHaveValue("10.5");
    await applyAuthoringCommand(page, "list", () =>
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
    ).toHaveValue(displayedEffectiveSpeed);

    await applyAuthoringCommand(page, "list", () =>
      page.getByRole("button", { name: "Combat Debugger", exact: true }).click(),
    );
    await expect(page.locator("#authoring-shell")).toBeHidden();
    await applyAuthoringCommand(page, "list", () =>
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
    await expectAuthoringIdle(page);
    await expect(page.locator("#authoring-persistence-status")).toHaveText(
      `Frozen candidate ${frozenScenario.candidate.candidate_id} · artifacts/dev_client/candidates/scenario-${frozenScenario.candidate.candidate_id}.json`,
    );

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
    await selectPersistedAsset(page, "#devclient-scenario-select", scenarioId);
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
