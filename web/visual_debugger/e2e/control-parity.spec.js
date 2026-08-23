import { expect, test } from "@playwright/test";

import { startDebugger, stopDebugger } from "./support/live-debugger.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";

const SCIENTIFIC_DISCLOSURES = Object.freeze([
  "#command-deck",
  "#roster-details",
  "#agent-details",
  "#pending-turn-details",
  "#latest-transition-details",
  "#events-details",
  "#technical-frame-details",
]);

test.beforeAll(async () => {
  const started = await startDebugger();
  serverProcess = started.process;
  debuggerUrl = started.url;
});

test.afterAll(async () => {
  const child = serverProcess;
  serverProcess = null;
  await stopDebugger(child);
});

/**
 * @param {import("@playwright/test").Page} page
 * @returns {Promise<number>}
 */
async function currentStep(page) {
  return Number(await page.locator("#step-value").textContent());
}

/**
 * Hold exactly one real authorized-presentation response after the browser has
 * synchronously crossed into pending authority.
 *
 * @param {import("@playwright/test").Page} page
 */
async function holdNextPresentation(page) {
  let markHeld = () => {};
  let releaseResponse = () => {};
  let released = false;
  const held = new Promise((resolve) => {
    markHeld = () => resolve(undefined);
  });
  const release = new Promise((resolve) => {
    releaseResponse = () => resolve(undefined);
  });
  const handler = async (/** @type {import("@playwright/test").Route} */ route) => {
    const response = await route.fetch();
    markHeld();
    await release;
    await route.fulfill({ response });
  };
  await page.route("**/api/presentation/frame", handler, { times: 1 });
  return {
    held,
    release() {
      if (!released) {
        released = true;
        releaseResponse();
      }
    },
    async dispose() {
      if (!released) {
        released = true;
        releaseResponse();
      }
      await page.unroute("**/api/presentation/frame", handler);
    },
  };
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} selector
 * @param {boolean} open
 */
async function setDisclosureOpen(page, selector, open) {
  const disclosure = page.locator(selector);
  if ((await disclosure.getAttribute("open")) !== (open ? "" : null)) {
    await disclosure.locator(":scope > summary").click();
  }
  if (open) {
    await expect(disclosure).toHaveAttribute("open", "");
  } else {
    await expect(disclosure).not.toHaveAttribute("open", "");
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{frameZeroEvents?: boolean}} [options]
 */
async function expectExactDisclosureDefaults(page, { frameZeroEvents = false } = {}) {
  await expect(page.locator("#command-deck")).toHaveAttribute("open", "");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  await expect(page.locator("#events-details")).toHaveAttribute("open", "");
  for (const selector of [
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
  }
  if (frameZeroEvents) {
    await expect(page.locator("#step-value")).toHaveText("0");
    await expect(page.locator("#event-count")).toHaveText("0");
    await expect(page.locator("#event-feed .event-item")).toHaveCount(0);
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @returns {Promise<Readonly<Record<string, boolean>>>}
 */
async function disclosureOpenState(page) {
  /** @type {Record<string, boolean>} */
  const result = {};
  for (const selector of SCIENTIFIC_DISCLOSURES) {
    result[selector] = (await page.locator(selector).getAttribute("open")) !== null;
  }
  return Object.freeze(result);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} previousPresentationKey
 * @param {Readonly<Record<string, boolean>>} expectedDisclosureOpen
 */
async function expectRetainedDisclosureAuthority(
  page,
  previousPresentationKey,
  expectedDisclosureOpen,
) {
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "retained",
  );
  await expect(page.locator("#connection-status")).toHaveText("Command in flight");
  await expect(page.locator("#battlefield-shell")).toHaveAttribute("aria-busy", "true");
  await expect(page.locator("#scenario-description")).toContainText("Last confirmed");
  await expect(page.locator("#scenario-description")).toContainText("update pending");
  await expect(page.locator("#battlefield-empty")).toBeHidden();
  await expect(
    page.locator(
      `#battlefield .agent[data-presentation-key="${previousPresentationKey}"]`,
    ),
  ).toHaveCount(1);
  await expect(page.locator("#submit-turn-button")).toBeDisabled();
  await expect(page.locator("#command-target-select")).toBeDisabled();
  await expect(page.locator("#view-select")).toBeDisabled();
  for (const selector of SCIENTIFIC_DISCLOSURES) {
    const disclosure = page.locator(selector);
    await expect(disclosure).toHaveAttribute("inert", "");
    await expect(disclosure.locator(":scope > [data-disclosure-body]")).toHaveAttribute(
      "inert",
      "",
    );
    await expect(disclosure.locator(":scope > summary")).toHaveAttribute(
      "aria-disabled",
      "true",
    );
    await expect(disclosure.locator(":scope > summary")).toHaveAttribute(
      "tabindex",
      "-1",
    );
    expect((await disclosure.getAttribute("open")) !== null).toBe(
      expectedDisclosureOpen[selector],
    );
  }
}

test("authorized draft edit and rapid Submit install exactly one successor", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commands = [];
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    commands.push(payload?.command ?? {});
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-inspection-state",
    "live_editable",
  );
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-submission-scope",
    "joint_turn",
  );
  await expect(page.locator("#command-controlled-actor")).toContainText(
    "Actor · Agent ID ",
  );
  await expect(
    page.locator("#battlefield .agent[data-presentation-key]"),
  ).not.toHaveCount(0);

  const draftLabel = page.locator("#pending-card .action-card__label");
  const draftBefore = await draftLabel.textContent();
  if (draftBefore === null || draftBefore.length === 0) {
    throw new Error("Authorized draft label is unavailable.");
  }
  const movement = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"]:not([aria-pressed="true"])',
    )
    .first();
  await expect(movement).toBeVisible();
  const movementAction = await movement.getAttribute("data-move-action");
  if (movementAction === null) {
    throw new Error("Authorized movement control did not publish its action identity.");
  }
  const selectedMovement = page.locator(
    `#command-deck button[data-move-action=${JSON.stringify(movementAction)}]`,
  );
  await movement.click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(selectedMovement).toHaveAttribute("aria-pressed", "true");
  await expect(draftLabel).not.toHaveText(draftBefore);
  await expect.poll(() => commands.length).toBe(1);

  const successorStep = await currentStep(page);
  const transitionBefore = await page.locator("#transition-value").textContent();
  const retainedPresentationKey = await page
    .locator("#battlefield .agent[data-presentation-key]")
    .first()
    .getAttribute("data-presentation-key");
  if (retainedPresentationKey === null) {
    throw new Error("Installed battlefield has no presentation key.");
  }
  const retainedDisclosureOpen = await disclosureOpenState(page);
  const heldSubmitPresentation = await holdNextPresentation(page);
  try {
    await page.locator("#submit-turn-button").evaluate((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        throw new TypeError("Submit button is unavailable.");
      }
      button.click();
      button.click();
    });
    await heldSubmitPresentation.held;
    await expectRetainedDisclosureAuthority(
      page,
      retainedPresentationKey,
      retainedDisclosureOpen,
    );
    await page.locator("#submit-turn-button").evaluate((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        throw new TypeError("Submit button is unavailable.");
      }
      button.click();
    });
    await page
      .locator(
        `#roster .roster-primary-action[data-presentation-key="${retainedPresentationKey}"]`,
      )
      .evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Retained roster action is unavailable.");
        }
        button.click();
      });
    await page
      .locator(
        `#battlefield .agent[data-presentation-key="${retainedPresentationKey}"]`,
      )
      .dispatchEvent("pointerdown", { button: 0 });
    await page.locator("#battlefield").focus();
    await page.keyboard.press("w");
    await expect.poll(() => commands.length).toBe(2);
    heldSubmitPresentation.release();
  } finally {
    heldSubmitPresentation.release();
    await heldSubmitPresentation.dispose();
  }
  await expect(page.locator("#step-value")).toHaveText(String(successorStep + 1), {
    timeout: 120_000,
  });
  await expect.poll(() => commands.length).toBe(2);
  expect(
    commands.filter(
      (command) =>
        command.command_type === "keyboard" &&
        String(command.key).toLowerCase() === "enter",
    ),
  ).toHaveLength(1);

  const transitionId = await page.locator("#transition-value").textContent();
  expect(transitionId).not.toBeNull();
  expect(transitionId).not.toBe("—");
  expect(transitionId).not.toBe(transitionBefore);
  await expect(page.locator("#accepted-card")).not.toHaveAttribute(
    "data-transition-id",
    /.+/u,
  );
  await expect(page.locator("#accepted-card .accepted-action-row")).not.toHaveCount(0);
  await expect(page.locator("#accepted-card")).toContainText("Submitted");
  await expect(page.locator("#accepted-card")).toContainText("Accepted");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await page.unroute("**/api/command");
});

test("pointer draft keeps battlefield focus and queues one Enter exactly once", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  const movement = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"]:not([aria-pressed="true"])',
    )
    .first();
  await expect(movement).toBeVisible();

  const heldDraftPresentation = await holdNextPresentation(page);
  let heldSubmitPresentation = null;
  try {
    await movement.click();
    await heldDraftPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await expect.poll(() => requests.length).toBe(1);

    await page.keyboard.press("Enter");
    await battlefield.dispatchEvent("keydown", {
      key: "Enter",
      repeat: true,
    });
    await expect.poll(() => requests.length).toBe(1);
    await expect(page.locator("#connection-status")).toHaveText("Submit queued");

    heldSubmitPresentation = await holdNextPresentation(page);
    heldDraftPresentation.release();
    await heldSubmitPresentation.held;
    await expect.poll(() => requests.length).toBe(2);
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await battlefield.dispatchEvent("keydown", {
      key: "Enter",
      repeat: true,
    });
    await expect.poll(() => requests.length).toBe(2);

    expect(requests[0].command).toMatchObject({ command_type: "keyboard" });
    expect(String(requests[0].command.key).toLowerCase()).not.toBe("enter");
    expect(requests[1].command).toEqual({
      command_type: "keyboard",
      key: "Enter",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    });
    expect(requests[1].command_id).not.toBe(requests[0].command_id);
    expect(requests[1].base_revision).toBe(requests[0].base_revision + 1);
    heldSubmitPresentation.release();
  } finally {
    heldDraftPresentation.release();
    await heldDraftPresentation.dispose();
    heldSubmitPresentation?.release();
    await heldSubmitPresentation?.dispose();
  }

  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1), {
    timeout: 120_000,
  });
  await expect.poll(() => requests.length).toBe(2);
  await expect(battlefield).toBeFocused();

  const keyboardMovementCandidate = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"]:not([aria-pressed="true"])',
    )
    .first();
  const keyboardMovementAction =
    await keyboardMovementCandidate.getAttribute("data-move-action");
  if (keyboardMovementAction === null) {
    throw new Error("Keyboard movement control has no stable action identity.");
  }
  const keyboardMovement = page.locator(
    `#command-deck button[data-move-action=${JSON.stringify(keyboardMovementAction)}]`,
  );
  const keyboardDraftResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  const heldKeyboardPresentation = await holdNextPresentation(page);
  try {
    await keyboardMovement.press("Enter");
    await heldKeyboardPresentation.held;
    expect((await keyboardDraftResponse).status()).toBe(200);
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(3);
    heldKeyboardPresentation.release();
  } finally {
    heldKeyboardPresentation.release();
    await heldKeyboardPresentation.dispose();
  }
  await expect.poll(() => requests.length).toBe(3);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(keyboardMovement).toBeFocused();
  expect(String(requests[2].command.key).toLowerCase()).not.toBe("enter");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await page.unroute("**/api/command");
});

test("Enter follows a coherent idempotent draft exactly once", async ({ page }) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  const selectedMovement = page
    .locator(
      '#command-deck button[data-move-action][aria-disabled="false"][aria-pressed="true"]',
    )
    .first();
  await expect(selectedMovement).toBeVisible();
  const heldNoOpPresentation = await holdNextPresentation(page);
  try {
    await selectedMovement.click();
    await heldNoOpPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(1);
    heldNoOpPresentation.release();
  } finally {
    heldNoOpPresentation.release();
    await heldNoOpPresentation.dispose();
  }

  await expect.poll(() => requests.length).toBe(2);
  expect(String(requests[0].command.key).toLowerCase()).not.toBe("enter");
  expect(String(requests[1].command.key).toLowerCase()).toBe("enter");
  expect(requests[1].command_id).not.toBe(requests[0].command_id);
  expect(requests[1].base_revision).toBe(requests[0].base_revision);
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1), {
    timeout: 120_000,
  });
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(battlefield).toBeFocused();
  await page.unroute("**/api/command");
});

test("target selection preserves pointer Submit and keyboard-native focus", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  const targetSelect = page.locator("#command-target-select");
  const pointerTarget = await targetSelect.evaluate((select) => {
    if (!(select instanceof HTMLSelectElement)) {
      throw new TypeError("Target control is unavailable.");
    }
    return [...select.options].find(
      (option) => !option.disabled && option.value !== select.value,
    )?.value;
  });
  if (pointerTarget === undefined) {
    throw new Error("Target control has no alternative authorized option.");
  }

  const heldPointerPresentation = await holdNextPresentation(page);
  try {
    await targetSelect.focus();
    await targetSelect.dispatchEvent("pointerdown", {
      button: 0,
      pointerType: "mouse",
    });
    await targetSelect.selectOption(pointerTarget);
    await heldPointerPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(1);
    heldPointerPresentation.release();
  } finally {
    heldPointerPresentation.release();
    await heldPointerPresentation.dispose();
  }

  await expect.poll(() => requests.length).toBe(2);
  expect(requests[0].command).toMatchObject({
    command_type: "roster_selection",
    role: "target",
  });
  expect(String(requests[1].command.key).toLowerCase()).toBe("enter");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1), {
    timeout: 120_000,
  });
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(battlefield).toBeFocused();

  const keyboardDirection = await targetSelect.evaluate((select) => {
    if (!(select instanceof HTMLSelectElement)) {
      throw new TypeError("Target control is unavailable.");
    }
    const enabled = [...select.options].filter((option) => !option.disabled);
    const currentIndex = enabled.findIndex((option) => option.value === select.value);
    if (enabled.length < 2 || currentIndex < 0) {
      throw new Error("Target control has no keyboard alternative.");
    }
    return currentIndex < enabled.length - 1 ? "ArrowDown" : "ArrowUp";
  });
  const heldKeyboardPresentation = await holdNextPresentation(page);
  try {
    await targetSelect.focus();
    await page.keyboard.press(keyboardDirection);
    await heldKeyboardPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(3);
    heldKeyboardPresentation.release();
  } finally {
    heldKeyboardPresentation.release();
    await heldKeyboardPresentation.dispose();
  }
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect.poll(() => requests.length).toBe(3);
  expect(requests[2].command.command_type).not.toBe("keyboard");
  await expect(targetSelect).toBeFocused();
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await page.unroute("**/api/command");
});

test("WASD followed immediately by Enter submits the installed draft once", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const requests = [];
  await page.route("**/api/command", async (route) => {
    requests.push(route.request().postDataJSON());
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  const heldDraftPresentation = await holdNextPresentation(page);
  try {
    await page.keyboard.press("w");
    await heldDraftPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(1);
    heldDraftPresentation.release();
  } finally {
    heldDraftPresentation.release();
    await heldDraftPresentation.dispose();
  }

  await expect.poll(() => requests.length).toBe(2);
  expect(String(requests[0].command.key).toLowerCase()).toBe("w");
  expect(String(requests[1].command.key).toLowerCase()).toBe("enter");
  expect(requests[1].command_id).not.toBe(requests[0].command_id);
  expect(requests[1].base_revision).toBe(requests[0].base_revision + 1);
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1), {
    timeout: 120_000,
  });
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(battlefield).toBeFocused();
  await page.unroute("**/api/command");
});

test("battlefield Space submits once and consumes busy repeats without scrolling", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commands = [];
  await page.route("**/api/command", async (route) => {
    const payload = route.request().postDataJSON();
    commands.push(payload?.command ?? {});
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const stepBefore = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  await battlefield.focus();
  const scrollBeforeSpace = await page.evaluate(() => {
    window.scrollTo(0, 0);
    return window.scrollY;
  });
  const heldPresentation = await holdNextPresentation(page);
  try {
    await page.keyboard.press("Space");
    await heldPresentation.held;
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "retained",
    );
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await expect.poll(() => commands.length).toBe(1);
    expect(commands[0]).toEqual({
      command_type: "keyboard",
      key: " ",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    });

    await page.keyboard.press("Space");
    await page.evaluate(
      () =>
        new Promise((resolve) => {
          window.requestAnimationFrame(() => {
            window.requestAnimationFrame(resolve);
          });
        }),
    );
    expect(await page.evaluate(() => window.scrollY)).toBe(scrollBeforeSpace);
    await expect(battlefield).toBeFocused();
    await expect.poll(() => commands.length).toBe(1);
    heldPresentation.release();
  } finally {
    heldPresentation.release();
    await heldPresentation.dispose();
  }

  await expect(page.locator("#step-value")).toHaveText(String(stepBefore + 1), {
    timeout: 120_000,
  });
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect.poll(() => commands.length).toBe(1);

  const helpButton = page.locator("#help-button");
  await helpButton.focus();
  await page.keyboard.press("Space");
  await expect(page.locator("#help-dialog")).toHaveAttribute("open", "");
  await expect.poll(() => commands.length).toBe(1);
  await page.locator("#help-close-button").click();
  await page.unroute("**/api/command");
});

test("a stale tab adopts the latest ranges state without replaying its command", async ({
  context,
  page,
}) => {
  await page.goto(debuggerUrl);
  const stalePage = await context.newPage();
  await stalePage.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(stalePage.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(stalePage.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const ranges = page.locator("#live-ranges-button");
  const staleRanges = stalePage.locator("#live-ranges-button");
  const initialRangesPressed = await ranges.getAttribute("aria-pressed");
  expect(["true", "false"]).toContain(initialRangesPressed);
  if (initialRangesPressed === null) {
    throw new Error("Live range control did not publish pressed state.");
  }

  await ranges.click();
  await expect(ranges).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed === "true" ? "false" : "true",
  );
  await expect(staleRanges).toHaveAttribute("aria-pressed", initialRangesPressed);

  await staleRanges.click();
  await expect(stalePage.locator("#notice")).toContainText("stale");
  await expect(staleRanges).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed === "true" ? "false" : "true",
  );
  await staleRanges.click();
  await expect(staleRanges).toHaveAttribute("aria-pressed", initialRangesPressed);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#live-ranges-button")).toHaveAttribute(
    "aria-pressed",
    initialRangesPressed,
  );
  await stalePage.close();
});

test("a lost applied response requires GET resync and never replays submit", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#pending-card")).toHaveAttribute(
    "data-inspection-state",
    "live_editable",
  );
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
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "retained",
  );
  await expect(page.locator("#battlefield-empty")).toBeHidden();
  await expect(
    page.locator("#battlefield .agent[data-presentation-key]"),
  ).not.toHaveCount(0);
  await expect(page.locator("#scenario-description")).toContainText("Last confirmed");
  await expect(page.locator("#scenario-description")).toContainText(
    "reconnect required",
  );
  await expect.poll(() => appliedStatus).toBe(200);
  await battlefield.focus();
  await page.keyboard.press("Enter");
  await expect.poll(() => interceptedCommands).toBe(1);

  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
  await expect.poll(() => interceptedCommands).toBe(1);
  await page.unroute("**/api/command");

  const ranges = page.locator("#live-ranges-button");
  const rangesBefore = await ranges.getAttribute("aria-pressed");
  await ranges.click();
  await expect(ranges).toHaveAttribute(
    "aria-pressed",
    rangesBefore === "true" ? "false" : "true",
  );
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1));
});

test("a command authorization failure clears the retained battlefield", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const retainedPresentationKey = await page
    .locator("#battlefield .agent[data-presentation-key]")
    .first()
    .getAttribute("data-presentation-key");
  if (retainedPresentationKey === null) {
    throw new Error("Installed battlefield has no presentation key.");
  }
  let markCommandHeld = () => {};
  let releaseCommand = () => {};
  const commandHeld = new Promise((resolve) => {
    markCommandHeld = () => resolve(undefined);
  });
  const commandRelease = new Promise((resolve) => {
    releaseCommand = () => resolve(undefined);
  });
  let interceptedCommands = 0;
  await page.route("**/api/command", async (route) => {
    interceptedCommands += 1;
    markCommandHeld();
    await commandRelease;
    await route.fulfill({
      status: 403,
      contentType: "application/json",
      body: JSON.stringify({ message: "Denied by browser regression proof." }),
    });
  });

  try {
    await page.locator("#battlefield").focus();
    await page.keyboard.press("w");
    await commandHeld;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await page.keyboard.press("Enter");
  } finally {
    releaseCommand();
  }
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect.poll(() => interceptedCommands).toBe(1);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  await expect(page.locator("#battlefield .agent")).toHaveCount(0);
  await expect(
    page.locator(`[data-presentation-key="${retainedPresentationKey}"]`),
  ).toHaveCount(0);
  await expect(page.locator("#battlefield-empty")).toBeVisible();
  await page.unroute("**/api/command");
});

test("an accepted Exit cannot retain a scene after sidecar authorization fails", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const currentFrame = await page.evaluate(async () => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    const response = await window.fetch("/api/frame", {
      headers: { "X-MARL-Debugger-Token": token ?? "" },
    });
    if (!response.ok) {
      throw new Error(`Could not obtain the live frame: HTTP ${response.status}.`);
    }
    return response.json();
  });
  await page.route(
    "**/api/command",
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          schema_version: 2,
          result: "shutdown_scheduled",
          frame: currentFrame,
        }),
      });
    },
    { times: 1 },
  );
  await page.route(
    "**/api/presentation/frame",
    async (route) => {
      await route.fulfill({
        status: 403,
        contentType: "application/json",
        body: JSON.stringify({ message: "Presentation capability denied." }),
      });
    },
    { times: 1 },
  );

  await page.locator("#exit-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Shutting down");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  await expect(page.locator("#battlefield .agent")).toHaveCount(0);
  await expect(page.locator("#battlefield-empty")).toBeVisible();
  await page.unroute("**/api/command");
  await page.unroute("**/api/presentation/frame");
});

test("native panels preserve user state only within exact authority", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#view-select")).toHaveValue("researcher");
  await expectExactDisclosureDefaults(page, {
    frameZeroEvents: (await currentStep(page)) === 0,
  });

  const firstActivation = page
    .locator(
      '#roster .roster-primary-action[data-presentation-key]:not([aria-pressed="true"])',
    )
    .first();
  await expect(firstActivation).toBeVisible();
  const firstCommand = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await firstActivation.press("Enter");
  expect((await firstCommand).status()).toBe(200);
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");

  await setDisclosureOpen(page, "#agent-details", false);
  const ranges = page.locator("#live-ranges-button");
  if ((await ranges.getAttribute("aria-pressed")) !== "true") {
    await ranges.click();
    await expect(ranges).toHaveAttribute("aria-pressed", "true");
  }
  await expect(page.locator("#battlefield .range-ring-owner").first()).toBeVisible();
  await page.locator("#battlefield .range-ring-owner").first().dispatchEvent("click");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");

  await setDisclosureOpen(page, "#events-details", false);
  await setDisclosureOpen(page, "#technical-frame-details", true);
  const rosterBody = page.locator("#roster-details-body");
  let savedScrollTop = await rosterBody.evaluate((body) => {
    body.style.height = "56px";
    body.style.maxHeight = "56px";
    body.style.overflowY = "scroll";
    body.scrollTop = 24;
    return body.scrollTop;
  });
  expect(savedScrollTop).toBeGreaterThan(0);

  const focusedAction = page
    .locator(
      '#roster .roster-primary-action[data-presentation-key]:not([aria-pressed="true"])',
    )
    .first();
  const focusedPresentationKey = await focusedAction.getAttribute(
    "data-presentation-key",
  );
  if (focusedPresentationKey === null) {
    throw new Error("Authorized roster action has no presentation key.");
  }
  await focusedAction.focus();
  savedScrollTop = await rosterBody.evaluate((body) => body.scrollTop);
  const commandDisclosureOpen = await disclosureOpenState(page);

  /** @type {import("@playwright/test").Request[]} */
  const commandRequests = [];
  const recordCommands = (
    /** @type {import("@playwright/test").Request} */ request,
  ) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/command"
    ) {
      commandRequests.push(request);
    }
  };
  page.on("request", recordCommands);
  const heldCommandPresentation = await holdNextPresentation(page);
  try {
    await focusedAction.press("Enter");
    await heldCommandPresentation.held;
    await expectRetainedDisclosureAuthority(
      page,
      focusedPresentationKey,
      commandDisclosureOpen,
    );
    heldCommandPresentation.release();
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
  } finally {
    heldCommandPresentation.release();
    await heldCommandPresentation.dispose();
    page.off("request", recordCommands);
  }
  expect(commandRequests).toHaveLength(1);
  expect(commandRequests[0].postDataJSON().command).toMatchObject({
    command_type: "roster_selection",
    role: "control",
  });
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  await expect(page.locator("#events-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#technical-frame-details")).toHaveAttribute("open", "");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect
    .poll(() => rosterBody.evaluate((body) => body.scrollTop))
    .toBe(savedScrollTop);
  await expect
    .poll(() =>
      page.evaluate(() => ({
        surface: document.activeElement?.closest("#roster")?.id ?? null,
        presentationKey:
          document.activeElement
            ?.closest("[data-presentation-key]")
            ?.getAttribute("data-presentation-key") ?? null,
      })),
    )
    .toEqual({ surface: "roster", presentationKey: focusedPresentationKey });

  const refreshDisclosureOpen = await disclosureOpenState(page);
  const heldRefresh = await holdNextPresentation(page);
  try {
    await page.locator("#reconnect-button").evaluate((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        throw new TypeError("Reconnect button is unavailable.");
      }
      button.click();
    });
    await heldRefresh.held;
    await expectRetainedDisclosureAuthority(
      page,
      focusedPresentationKey,
      refreshDisclosureOpen,
    );
    await page.locator("#help-button").click();
    await expect(page.locator("#help-dialog")).toHaveAttribute("open", "");
    heldRefresh.release();
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
  } finally {
    heldRefresh.release();
    await heldRefresh.dispose();
  }
  await expect
    .poll(() =>
      page.evaluate(() => document.activeElement?.closest("dialog")?.id ?? null),
    )
    .toBe("help-dialog");
  await page.locator("#help-close-button").click();
  await expect(page.locator("#help-dialog")).not.toHaveAttribute("open", "");
  await expect
    .poll(() => rosterBody.evaluate((body) => body.scrollTop))
    .toBe(savedScrollTop);

  await setDisclosureOpen(page, "#roster-details", false);
  await page.locator("#view-select").selectOption("pov");
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expectExactDisclosureDefaults(page);
  await expect.poll(() => rosterBody.evaluate((body) => body.scrollTop)).toBe(0);
  await expect
    .poll(() =>
      page.evaluate(
        (oldKey) =>
          document.activeElement
            ?.closest("[data-presentation-key]")
            ?.getAttribute("data-presentation-key") !== oldKey,
        focusedPresentationKey,
      ),
    )
    .toBe(true);

  await setDisclosureOpen(page, "#events-details", false);
  await setDisclosureOpen(page, "#technical-frame-details", true);
  await page.locator("#view-select").selectOption("researcher");
  await expect(page.locator("#view-select")).toHaveValue("researcher");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expectExactDisclosureDefaults(page);
  await expect.poll(() => rosterBody.evaluate((body) => body.scrollTop)).toBe(0);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
});
