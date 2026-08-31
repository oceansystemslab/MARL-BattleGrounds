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
  "#technical-frame-details",
]);

/** @typedef {{target_public_agent_id?: string, target_action: number}} TargetActionRow */
/** @typedef {{move_action: number, target_action: number, use_ultimate_action: number}} ActionTuple */
/** @typedef {{actor_public_agent_id: string, submitted_action: ActionTuple, accepted_action: ActionTuple}} TransitionActionRow */
/** @typedef {{status_id: string}} IncomingStatus */
/** @typedef {{delta_kind: string, agent_public_agent_id: string, changed_dynamic_fields: string[], start_observation: {statuses: IncomingStatus[]}, successor_observation: {statuses: IncomingStatus[]}}} IncomingStatusDelta */

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
 * Read the exact currently installed authorized presentation from the real
 * loopback service. This does not bypass the browser capability boundary.
 *
 * @param {import("@playwright/test").Page} page
 * @returns {Promise<Record<string, any>>}
 */
async function currentAuthorizedPresentation(page) {
  return page.evaluate(async () => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Debugger capability token is unavailable.");
    }
    const response = await window.fetch("/api/presentation/frame", {
      cache: "no-store",
      credentials: "omit",
      headers: { "X-MARL-Debugger-Token": token },
      redirect: "error",
    });
    if (!response.ok) {
      throw new Error(
        `Could not obtain the authorized presentation: HTTP ${response.status}.`,
      );
    }
    return response.json();
  });
}

/**
 * Wait for one ordinary live command and its complete presentation successor.
 *
 * @param {import("@playwright/test").Page} page
 * @param {() => Promise<unknown>} activate
 */
async function activateLiveCommand(page, activate) {
  const response = page.waitForResponse(
    (candidate) =>
      candidate.request().method() === "POST" &&
      new URL(candidate.url()).pathname === "/api/command",
  );
  await activate();
  expect((await response).status()).toBe(200);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#battlefield-empty")).toBeHidden();
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} publicAgentId
 */
async function controlLiveAgent(page, publicAgentId) {
  await activateLiveCommand(page, () =>
    page
      .getByRole("button", {
        name: `Control and inspect Agent ID ${publicAgentId}`,
        exact: true,
      })
      .click(),
  );
  await expect(page.locator("#command-controlled-actor")).toContainText(
    `Agent ID ${publicAgentId}`,
  );
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} publicAgentId
 */
async function liveTargetOption(page, publicAgentId) {
  const option = page
    .locator("#command-target-select option")
    .filter({ hasText: `Agent ID ${publicAgentId} ·` });
  await expect(option).toHaveCount(1);
  const value = await option.getAttribute("value");
  const text = await option.textContent();
  if (value === null || text === null) {
    throw new Error(`Agent ID ${publicAgentId} has no target option.`);
  }
  return { option, text, value };
}

/**
 * Return one global live-researcher roster actor that the current Agent POV
 * battlefield does not expose. Fog must not be encoded into the roster DOM.
 *
 * @param {import("@playwright/test").Page} page
 */
async function liveRosterActorOutsideBattlefield(page) {
  const publicId = (/** @type {string | null} */ label) =>
    label?.match(/Agent ID ([^,.]+)/u)?.[1] ?? null;
  const visiblePublicIds = new Set(
    (
      await page
        .locator("#battlefield .agent")
        .evaluateAll((agents) =>
          agents.map((agent) => agent.getAttribute("aria-label")),
        )
    ).map(publicId),
  );
  const rosterActors = page.locator(
    '#roster .roster-primary-action:not([aria-pressed="true"])',
  );
  const labels = await rosterActors.evaluateAll((buttons) =>
    buttons.map((button) => button.getAttribute("aria-label")),
  );
  const index = labels.findIndex((label) => {
    const candidate = publicId(label);
    return candidate !== null && !visiblePublicIds.has(candidate);
  });
  if (index < 0) {
    throw new Error("The live researcher roster has no actor outside this snapshot.");
  }
  return rosterActors.nth(index);
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
 */
async function expectExactDisclosureDefaults(page) {
  await expect(page.locator("#command-deck")).toHaveAttribute("open", "");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  for (const selector of [
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
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

  const controlledActor = await page.locator("#command-controlled-actor").textContent();
  const controlledPublicId = controlledActor?.match(/Agent ID ([^ ·]+)/u)?.[1];
  if (controlledPublicId === undefined) {
    throw new Error("Controlled actor identity is unavailable.");
  }
  const draftLabel = page
    .locator("#pending-card .accepted-action-row")
    .filter({ hasText: `Agent ID ${controlledPublicId} ·` })
    .locator(".accepted-action-tuple__value");
  await expect(draftLabel).toHaveCount(1);
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
    await expect(page.locator("#connection-status")).toHaveText("Input queued");
    await expect.poll(() => commands.length).toBe(2);
    heldSubmitPresentation.release();
  } finally {
    heldSubmitPresentation.release();
    await heldSubmitPresentation.dispose();
  }
  await expect(page.locator("#step-value")).toHaveText(String(successorStep + 1), {
    timeout: 120_000,
  });
  await expect.poll(() => commands.length).toBe(3);
  expect(
    commands.filter(
      (command) =>
        command.command_type === "keyboard" &&
        String(command.key).toLowerCase() === "enter",
    ),
  ).toHaveLength(1);
  expect(commands[2]).toMatchObject({
    command_type: "keyboard",
    key: "w",
    repeat: false,
  });

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
  let heldKeyboardSubmitPresentation = null;
  try {
    await keyboardMovement.press("Enter");
    await heldKeyboardPresentation.held;
    expect((await keyboardDraftResponse).status()).toBe(200);
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(3);
    await expect(page.locator("#connection-status")).toHaveText("Submit queued");
    heldKeyboardSubmitPresentation = await holdNextPresentation(page);
    heldKeyboardPresentation.release();
    await heldKeyboardSubmitPresentation.held;
    await expect.poll(() => requests.length).toBe(4);
    expect(String(requests[2].command.key).toLowerCase()).not.toBe("enter");
    expect(requests[3].command).toEqual({
      command_type: "keyboard",
      key: "Enter",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    });
    expect(requests[3].command_id).not.toBe(requests[2].command_id);
    expect(requests[3].base_revision).toBe(requests[2].base_revision + 1);
    heldKeyboardSubmitPresentation.release();
  } finally {
    heldKeyboardPresentation.release();
    await heldKeyboardPresentation.dispose();
    heldKeyboardSubmitPresentation?.release();
    await heldKeyboardSubmitPresentation?.dispose();
  }
  await expect.poll(() => requests.length).toBe(4);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(keyboardMovement).toBeFocused();
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 2));
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
  const basicButton = page.locator("#basic-button");
  await expect(targetSelect).toHaveValue("");
  expect(
    await targetSelect
      .locator('option[value=""]')
      .evaluate((option) => option.textContent?.includes("B ×")),
  ).toBe(true);
  await expect(basicButton).toHaveAttribute("data-authoritative-available", "false");
  await expect(basicButton).toHaveAttribute("aria-disabled", "true");

  const authorizedNonzeroTarget = await targetSelect.evaluate((select) => {
    if (!(select instanceof HTMLSelectElement)) {
      throw new TypeError("Target control is unavailable.");
    }
    const option = [...select.options].find(
      (candidate) => !candidate.disabled && candidate.value !== "",
    );
    return option
      ? {
          value: option.value,
          basicAvailable: option.textContent?.includes("B ✓") === true,
        }
      : null;
  });
  if (authorizedNonzeroTarget === null) {
    throw new Error("Target control has no alternative authorized option.");
  }

  await targetSelect.selectOption(authorizedNonzeroTarget.value);
  await expect.poll(() => requests.length).toBe(1);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(targetSelect).toHaveValue(authorizedNonzeroTarget.value);
  await expect(basicButton).toHaveAttribute(
    "data-authoritative-available",
    String(authorizedNonzeroTarget.basicAvailable),
  );
  await expect(basicButton).toHaveAttribute(
    "aria-disabled",
    String(!authorizedNonzeroTarget.basicAvailable),
  );

  await targetSelect.selectOption("");
  await expect.poll(() => requests.length).toBe(2);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(targetSelect).toHaveValue("");
  await expect(basicButton).toHaveAttribute("data-authoritative-available", "false");
  await expect(basicButton).toHaveAttribute("aria-disabled", "true");
  requests.length = 0;

  const pointerTarget = authorizedNonzeroTarget.value;

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
  let heldKeyboardSubmitPresentation = null;
  try {
    await targetSelect.focus();
    await page.keyboard.press(keyboardDirection);
    await heldKeyboardPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await page.keyboard.press("Enter");
    await expect.poll(() => requests.length).toBe(3);
    await expect(page.locator("#connection-status")).toHaveText("Submit queued");
    heldKeyboardSubmitPresentation = await holdNextPresentation(page);
    heldKeyboardPresentation.release();
    await heldKeyboardSubmitPresentation.held;
    await expect.poll(() => requests.length).toBe(4);
    expect(requests[2].command.command_type).not.toBe("keyboard");
    expect(String(requests[3].command.key).toLowerCase()).toBe("enter");
    expect(requests[3].command_id).not.toBe(requests[2].command_id);
    expect(requests[3].base_revision).toBe(requests[2].base_revision + 1);
    heldKeyboardSubmitPresentation.release();
  } finally {
    heldKeyboardPresentation.release();
    await heldKeyboardPresentation.dispose();
    heldKeyboardSubmitPresentation?.release();
    await heldKeyboardSubmitPresentation?.dispose();
  }
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect.poll(() => requests.length).toBe(4);
  await expect(targetSelect).toBeFocused();
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 2));
  await page.unroute("**/api/command");
});

test("Agent pointer roster selection queues movement before one Enter", async ({
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
  const viewSelect = page.locator("#view-select");
  await viewSelect.selectOption("pov");
  await expect(viewSelect).toHaveValue("pov");
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const initialRosterActorLabel = await page
    .locator('#roster .roster-primary-action[aria-pressed="true"]')
    .getAttribute("aria-label");
  if (initialRosterActorLabel === null) {
    throw new Error("Agent POV has no initially controlled roster actor.");
  }
  requests.length = 0;

  const baseStep = await currentStep(page);
  const battlefield = page.locator("#battlefield");
  const hiddenRosterActor = await liveRosterActorOutsideBattlefield(page);
  await expect(hiddenRosterActor).toBeVisible();

  const heldRosterPresentation = await holdNextPresentation(page);
  try {
    await hiddenRosterActor.click();
    await heldRosterPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect(battlefield).toBeFocused();
    await expect.poll(() => requests.length).toBe(1);

    await page.keyboard.press("w");
    await expect(page.locator("#connection-status")).toHaveText("Input queued");
    await page.keyboard.press("Enter");
    await expect(page.locator("#connection-status")).toHaveText("Submit queued");
    await expect.poll(() => requests.length).toBe(1);

    heldRosterPresentation.release();
  } finally {
    heldRosterPresentation.release();
    await heldRosterPresentation.dispose();
  }

  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1), {
    timeout: 120_000,
  });
  await expect.poll(() => requests.length).toBe(3);
  expect(requests[0].command).toMatchObject({
    command_type: "roster_selection",
    role: "control",
  });
  expect(requests[1].command).toMatchObject({
    command_type: "keyboard",
    key: "w",
    repeat: false,
  });
  expect(requests[2].command).toEqual({
    command_type: "keyboard",
    key: "Enter",
    shift_key: false,
    ctrl_key: false,
    alt_key: false,
    meta_key: false,
    repeat: false,
  });
  expect(requests[1].command_id).not.toBe(requests[0].command_id);
  expect(requests[2].command_id).not.toBe(requests[1].command_id);
  expect(requests[1].base_revision).toBe(requests[0].base_revision + 1);
  expect(requests[2].base_revision).toBe(requests[1].base_revision + 1);
  await expect(battlefield).toBeFocused();

  await page.unroute("**/api/command");
  await page
    .getByRole("button", { name: initialRosterActorLabel, exact: true })
    .click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await viewSelect.selectOption("researcher");
  await expect(viewSelect).toHaveValue("researcher");
  await expect(page.locator("#connection-status")).toHaveText("Online");
});

test("Agent keyboard roster selection retains researcher focus without submitting", async ({
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
  const viewSelect = page.locator("#view-select");
  await viewSelect.selectOption("pov");
  await expect(viewSelect).toHaveValue("pov");
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const initialRosterActorLabel = await page
    .locator('#roster .roster-primary-action[aria-pressed="true"]')
    .getAttribute("aria-label");
  if (initialRosterActorLabel === null) {
    throw new Error("Agent POV has no initially controlled roster actor.");
  }
  const hiddenRosterActor = await liveRosterActorOutsideBattlefield(page);
  await expect(hiddenRosterActor).toBeVisible();
  const focusedPresentationKey = await hiddenRosterActor.getAttribute(
    "data-presentation-key",
  );
  if (focusedPresentationKey === null) {
    throw new Error("Agent researcher roster action has no presentation key.");
  }
  const baseStep = await currentStep(page);
  requests.length = 0;

  const heldRosterPresentation = await holdNextPresentation(page);
  try {
    await hiddenRosterActor.press("Enter");
    await heldRosterPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect.poll(() => requests.length).toBe(1);
    heldRosterPresentation.release();
  } finally {
    heldRosterPresentation.release();
    await heldRosterPresentation.dispose();
  }

  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep));
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0].command).toMatchObject({
    command_type: "roster_selection",
    role: "control",
  });
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

  await page.unroute("**/api/command");
  await viewSelect.evaluate((select) => {
    if (!(select instanceof HTMLSelectElement)) {
      throw new TypeError("View mode control is unavailable.");
    }
    select.value = "researcher";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(viewSelect).toHaveValue("researcher");
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectExactDisclosureDefaults(page);
  await expect
    .poll(() =>
      page.evaluate(() => document.activeElement?.closest("#roster") === null),
    )
    .toBe(true);

  await page
    .getByRole("button", { name: initialRosterActorLabel, exact: true })
    .click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
});

test("Agent keyboard battlefield activation rebinds focus to the authorized successor", async ({
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
  const viewSelect = page.locator("#view-select");
  await viewSelect.selectOption("pov");
  await expect(viewSelect).toHaveValue("pov");
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const initialRosterActorLabel = await page
    .locator('#roster .roster-primary-action[aria-pressed="true"]')
    .getAttribute("aria-label");
  const initialPublicId = initialRosterActorLabel?.match(/Agent ID ([^.]+)/u)?.[1];
  if (initialRosterActorLabel === null || initialPublicId === undefined) {
    throw new Error("Agent POV has no parseable initially controlled actor.");
  }
  const battlefieldActors = page.locator("#battlefield .agent[data-presentation-key]");
  const labels = await battlefieldActors.evaluateAll((actors) =>
    actors.map((actor) => actor.getAttribute("aria-label")),
  );
  const targetIndex = labels.findIndex(
    (label) =>
      typeof label === "string" && !label.startsWith(`Agent ID ${initialPublicId}.`),
  );
  if (targetIndex < 0) {
    throw new Error("Agent POV has no visible non-controlled battlefield actor.");
  }
  const target = battlefieldActors.nth(targetIndex);
  const targetLabel = await target.getAttribute("aria-label");
  const oldPresentationKey = await target.getAttribute("data-presentation-key");
  const targetPublicId = targetLabel?.match(/Agent ID ([^.]+)/u)?.[1];
  if (
    targetLabel === null ||
    oldPresentationKey === null ||
    targetPublicId === undefined
  ) {
    throw new Error("Visible Agent battlefield actor lacks its public identity.");
  }
  await setDisclosureOpen(page, "#technical-frame-details", true);
  await setDisclosureOpen(page, "#visual-filters", true);
  const ranges = page.locator("#live-ranges-button");
  if ((await ranges.getAttribute("aria-pressed")) !== "false") {
    await ranges.click();
  }
  await expect(ranges).toHaveAttribute("aria-pressed", "false");
  await setDisclosureOpen(page, "#visual-filters", false);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  const baseStep = await currentStep(page);
  requests.length = 0;

  const heldPresentation = await holdNextPresentation(page);
  try {
    await target.press("Enter");
    await heldPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect.poll(() => requests.length).toBe(1);
    heldPresentation.release();
  } finally {
    heldPresentation.release();
    await heldPresentation.dispose();
  }

  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#step-value")).toHaveText(String(baseStep));
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0].command).toMatchObject({
    command_type: "roster_selection",
    role: "control",
  });
  await expect(page.locator("#command-controlled-actor")).toContainText(
    `Agent ID ${targetPublicId}`,
  );
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await expect(page.locator("#technical-frame-details")).toHaveAttribute("open", "");
  await expect(ranges).toHaveAttribute("aria-pressed", "false");
  const restoredFocus = await page.evaluate(() => ({
    inBattlefield: Boolean(document.activeElement?.closest("#battlefield")),
    label: document.activeElement?.getAttribute("aria-label") ?? null,
    presentationKey:
      document.activeElement?.getAttribute("data-presentation-key") ?? null,
  }));
  expect(restoredFocus).toMatchObject({
    inBattlefield: true,
    label: targetLabel,
  });
  expect(restoredFocus.presentationKey).not.toBe(oldPresentationKey);

  await page.unroute("**/api/command");
  await viewSelect.evaluate((select) => {
    if (!(select instanceof HTMLSelectElement)) {
      throw new TypeError("View mode control is unavailable.");
    }
    select.value = "researcher";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(viewSelect).toHaveValue("researcher");
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#command-controlled-actor")).toContainText(
    `Agent ID ${targetPublicId}`,
  );
  await expectExactDisclosureDefaults(page);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.activeElement?.matches("#battlefield .agent") ?? false,
      ),
    )
    .toBe(false);

  await page
    .getByRole("button", { name: initialRosterActorLabel, exact: true })
    .click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
});

test("post-Charge Agent history stays installable through a reciprocal Charge", async ({
  page,
}) => {
  test.setTimeout(300_000);
  /** @type {Record<string, any>[]} */
  const commandRequests = [];
  /** @type {{path: string, status: number}[]} */
  const endpointFailures = [];
  /** @type {string[]} */
  const pageErrors = [];
  const recordResponse = (
    /** @type {import("@playwright/test").Response} */ response,
  ) => {
    const path = new URL(response.url()).pathname;
    if (
      (path === "/api/command" || path === "/api/presentation/frame") &&
      response.status() >= 400
    ) {
      endpointFailures.push({ path, status: response.status() });
    }
  };
  const recordPageError = (/** @type {Error} */ error) => {
    pageErrors.push(error.message);
  };
  page.on("response", recordResponse);
  page.on("pageerror", recordPageError);
  await page.route("**/api/command", async (route) => {
    commandRequests.push(route.request().postDataJSON());
    await route.continue();
  });

  const battlefield = page.locator("#battlefield");
  const viewSelect = page.locator("#view-select");
  const targetSelect = page.locator("#command-target-select");
  let initialControlledPublicId = null;
  let primaryError = null;
  const expectHealthyInstallation = async () => {
    await expect(page.locator("#connection-status")).toHaveText("Online");
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
    await expect(page.locator("#battlefield-empty")).toBeHidden();
    await expect(page.locator("#notice")).not.toContainText(
      /could not process|safe fault|reconnect before sending|no authorized battlefield scene/iu,
    );
  };
  /** @param {string} key */
  const selectMovement = async (key) => {
    const button = page.locator(`#command-deck button[data-key="${key}"]`);
    await expect(button).toBeEnabled();
    if ((await button.getAttribute("aria-pressed")) !== "true") {
      await activateLiveCommand(page, () => button.click());
    }
    await expect(button).toHaveAttribute("aria-pressed", "true");
  };

  try {
    await page.goto(debuggerUrl);
    await expectHealthyInstallation();
    const initialControlledLabel = await page
      .locator("#command-controlled-actor")
      .textContent();
    initialControlledPublicId =
      initialControlledLabel?.match(/Agent ID ([^\s]+)/u)?.[1] ?? null;
    if (initialControlledPublicId === null) {
      throw new Error("The initial controlled actor identity is unavailable.");
    }
    await activateLiveCommand(page, () => page.locator("#reset-button").click());
    await expect(page.locator("#step-value")).toHaveText("0");

    // Use only ordinary browser drafts and joint submissions to converge the
    // two Warriors. Stop as soon as Agent 1's exact mask exposes Charge.
    let stagedTurns = 0;
    /** @type {Awaited<ReturnType<typeof liveTargetOption>> | null} */
    let chargeTarget = null;
    while (stagedTurns < 9) {
      await controlLiveAgent(page, 1);
      chargeTarget = await liveTargetOption(page, 6);
      if (chargeTarget.text.includes("U ✓")) {
        break;
      }
      await selectMovement("d");
      await controlLiveAgent(page, 6);
      await selectMovement("z");
      const stepBeforeMovement = await currentStep(page);
      await activateLiveCommand(page, () =>
        page.locator("#submit-turn-button").click(),
      );
      await expect(page.locator("#step-value")).toHaveText(
        String(stepBeforeMovement + 1),
        { timeout: 120_000 },
      );
      stagedTurns += 1;
    }
    expect(stagedTurns).toBeGreaterThan(0);
    expect(chargeTarget).not.toBeNull();
    if (chargeTarget === null) {
      throw new Error("The Warriors could not be staged into Charge range.");
    }
    expect(chargeTarget.text).toContain("U ✓");

    // Freeze both movement drafts at the converged geometry before combat.
    await selectMovement("x");
    await controlLiveAgent(page, 6);
    await selectMovement("x");
    await controlLiveAgent(page, 1);
    chargeTarget = await liveTargetOption(page, 6);
    await activateLiveCommand(page, () =>
      targetSelect.selectOption(chargeTarget.value),
    );
    await expect(targetSelect).toHaveValue(chargeTarget.value);
    await expect(page.locator("#ultimate-button")).toHaveAttribute(
      "data-authoritative-available",
      "true",
    );
    await expect(battlefield.locator('.agent[data-selected="true"]')).toHaveCount(1);
    await expect(battlefield.locator('.agent[data-selected="true"]')).toHaveAttribute(
      "aria-label",
      /^Agent ID 6[,.]/u,
    );
    const firstChargeDraftPresentation = await currentAuthorizedPresentation(page);
    const firstChargeDraft =
      firstChargeDraftPresentation.live_inspection.inspection.draft;
    const firstChargeTargetAction = /** @type {TargetActionRow[]} */ (
      firstChargeDraft.decision_mask.target_actions
    ).find((target) => target.target_public_agent_id === "6")?.target_action;
    expect(firstChargeTargetAction).toEqual(expect.any(Number));
    if (typeof firstChargeTargetAction !== "number") {
      throw new Error("Agent 6 is absent from Agent 1's authorized target axis.");
    }
    expect(firstChargeDraft.draft_action.target_action).toBe(firstChargeTargetAction);

    await activateLiveCommand(page, () => page.locator("#ultimate-button").click());
    const firstChargeStep = await currentStep(page);
    await activateLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText(String(firstChargeStep + 1), {
      timeout: 120_000,
    });
    const afterFirstCharge = await currentAuthorizedPresentation(page);
    const firstChargeRow = /** @type {TransitionActionRow[]} */ (
      afterFirstCharge.latest_transition.action_rows
    ).find((row) => row.actor_public_agent_id === "1");
    expect(firstChargeRow).toBeTruthy();
    if (firstChargeRow === undefined) {
      throw new Error("Agent 1 is absent from the first Charge transition.");
    }
    expect(firstChargeRow.submitted_action).toMatchObject({
      target_action: firstChargeTargetAction,
      use_ultimate_action: 1,
    });
    expect(firstChargeRow.accepted_action).toEqual(firstChargeRow.submitted_action);

    // Age the one-tick stun while retaining Charge's five-tick slow. This is
    // the exact transition whose historical status ordering used to make the
    // subsequent Agent presentation unbuildable.
    await activateLiveCommand(page, () => page.locator("#no-combat-button").click());
    const settlingStep = await currentStep(page);
    await activateLiveCommand(page, () => page.locator("#submit-turn-button").click());
    await expect(page.locator("#step-value")).toHaveText(String(settlingStep + 1), {
      timeout: 120_000,
    });

    await controlLiveAgent(page, 6);
    const stepBeforeAgentInstall = await currentStep(page);
    await activateLiveCommand(page, () => viewSelect.selectOption("pov"));
    await expect(viewSelect).toHaveValue("pov");
    await expect(page.locator("#step-value")).toHaveText(
      String(stepBeforeAgentInstall),
    );
    await expectHealthyInstallation();

    const agentAfterSettling = await currentAuthorizedPresentation(page);
    expect(agentAfterSettling.presentation_kind).toBe("live_shared_obs_agent_pov");
    expect(agentAfterSettling.authority.recipient_public_agent_id).toBe("6");
    const statusHistory = /** @type {IncomingStatusDelta[]} */ (
      agentAfterSettling.latest_events.deltas
    ).find(
      (delta) =>
        delta.delta_kind === "observed_values_change" &&
        delta.agent_public_agent_id === "6" &&
        delta.changed_dynamic_fields.includes("statuses"),
    );
    expect(statusHistory).toBeTruthy();
    if (statusHistory === undefined) {
      throw new Error("Agent 6 has no post-Charge status history delta.");
    }
    expect(
      statusHistory.start_observation.statuses.map((status) => status.status_id),
    ).toEqual(["warrior_charge_stun", "warrior_charge_slow"]);
    expect(
      statusHistory.successor_observation.statuses.map((status) => status.status_id),
    ).toEqual(["warrior_charge_slow"]);
    for (const status of [
      ...statusHistory.start_observation.statuses,
      ...statusHistory.successor_observation.statuses,
    ]) {
      expect("direct_sources" in status).toBe(false);
      expect("source_public_agent_id" in status).toBe(false);
      expect("source_presentation_key" in status).toBe(false);
    }

    const reverseTarget = await liveTargetOption(page, 1);
    expect(reverseTarget.text).toContain("U ✓");
    await activateLiveCommand(page, () =>
      targetSelect.selectOption(reverseTarget.value),
    );
    await expect(targetSelect).toHaveValue(reverseTarget.value);
    await expect(battlefield.locator('.agent[data-selected="true"]')).toHaveAttribute(
      "aria-label",
      /^Agent ID 1[,.]/u,
    );
    const reverseDraftPresentation = await currentAuthorizedPresentation(page);
    const reverseDraft =
      reverseDraftPresentation.researcher_space.pending_inspection.draft;
    const reverseTargetAction = /** @type {TargetActionRow[]} */ (
      reverseDraft.decision_mask.target_actions
    ).find((target) => target.target_public_agent_id === "1")?.target_action;
    expect(reverseDraft.actor_public_agent_id).toBe("6");
    expect(reverseTargetAction).toEqual(expect.any(Number));
    if (typeof reverseTargetAction !== "number") {
      throw new Error("Agent 1 is absent from Agent 6's authorized target axis.");
    }
    expect(reverseDraft.draft_action.target_action).toBe(reverseTargetAction);

    // Reproduce the user's rapid path: arm Ultimate and press Enter before the
    // Agent successor has finished installing. The queue must serialize both
    // commands and advance exactly once.
    commandRequests.length = 0;
    const reverseChargeStep = await currentStep(page);
    await page.locator("#ultimate-button").click();
    await page.keyboard.press("Enter");
    await expect.poll(() => commandRequests.length).toBe(2);
    expect(commandRequests.map((request) => request.command.key)).toEqual([
      "2",
      "Enter",
    ]);
    expect(commandRequests[1].base_revision).toBe(commandRequests[0].base_revision + 1);
    await expect(page.locator("#step-value")).toHaveText(
      String(reverseChargeStep + 1),
      { timeout: 120_000 },
    );
    await expectHealthyInstallation();
    await expect.poll(() => commandRequests.length).toBe(2);

    const afterReverseCharge = await currentAuthorizedPresentation(page);
    const reverseChargeRow = /** @type {TransitionActionRow[]} */ (
      afterReverseCharge.researcher_space.latest_transition.action_rows
    ).find((row) => row.actor_public_agent_id === "6");
    expect(reverseChargeRow).toBeTruthy();
    if (reverseChargeRow === undefined) {
      throw new Error("Agent 6 is absent from the reciprocal Charge transition.");
    }
    expect(reverseChargeRow.submitted_action).toMatchObject({
      target_action: reverseTargetAction,
      use_ultimate_action: 1,
    });
    expect(reverseChargeRow.accepted_action).toEqual(reverseChargeRow.submitted_action);

    const fixedStep = await currentStep(page);
    for (let cycle = 0; cycle < 3; cycle += 1) {
      for (const mode of ["researcher", "pov"]) {
        await activateLiveCommand(page, () => viewSelect.selectOption(mode));
        await expect(viewSelect).toHaveValue(mode);
        await expect(page.locator("#step-value")).toHaveText(String(fixedStep));
        await expect(targetSelect).toHaveValue(reverseTarget.value);
        await expect(
          battlefield.locator('.agent[data-controlled="true"]'),
        ).toHaveAttribute("aria-label", /^Agent ID 6[,.]/u);
        await expect(battlefield.locator('.agent[data-selected="true"]')).toHaveCount(
          1,
        );
        await expect(
          battlefield.locator('.agent[data-selected="true"]'),
        ).toHaveAttribute("aria-label", /^Agent ID 1[,.]/u);
        await expectHealthyInstallation();
      }
    }

    expect(endpointFailures).toEqual([]);
    expect(pageErrors).toEqual([]);
  } catch (error) {
    primaryError = error;
  }

  let cleanupError = null;
  try {
    page.off("response", recordResponse);
    page.off("pageerror", recordPageError);
    await page.unroute("**/api/command");
    if (!page.isClosed()) {
      if ((await viewSelect.inputValue()) !== "researcher") {
        await activateLiveCommand(page, () => viewSelect.selectOption("researcher"));
      }
      await activateLiveCommand(page, () => page.locator("#reset-button").click());
      if (initialControlledPublicId !== null) {
        await controlLiveAgent(page, Number(initialControlledPublicId));
      }
    }
  } catch (error) {
    cleanupError = error;
  }
  if (primaryError !== null) {
    throw primaryError;
  }
  if (cleanupError !== null) {
    throw cleanupError;
  }
});

test("right click stays native while Escape alone clears the target and releases focus", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commands = [];
  await page.route("**/api/command", async (route) => {
    commands.push(route.request().postDataJSON()?.command ?? {});
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  const battlefield = page.locator("#battlefield");
  const targetSelect = page.locator("#command-target-select");
  const target = await targetSelect.evaluate((select) => {
    if (!(select instanceof HTMLSelectElement)) {
      throw new TypeError("Target control is unavailable.");
    }
    return [...select.options].find((option) => !option.disabled && option.value !== "")
      ?.value;
  });
  if (typeof target !== "string") {
    throw new Error("Target control has no authorized nonzero target.");
  }
  await targetSelect.selectOption(target);
  await expect.poll(() => commands.length).toBe(1);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(targetSelect).toHaveValue(target);
  commands.length = 0;

  await page.locator("#help-button").focus();
  const nativeEvents = await battlefield.evaluate((surface) => {
    const pointerdown = new PointerEvent("pointerdown", {
      bubbles: true,
      cancelable: true,
      button: 2,
      clientX: 100,
      clientY: 100,
      pointerType: "mouse",
    });
    const contextmenu = new MouseEvent("contextmenu", {
      bubbles: true,
      cancelable: true,
      button: 2,
      clientX: 100,
      clientY: 100,
    });
    const pointerDispatched = surface.dispatchEvent(pointerdown);
    const contextmenuDispatched = surface.dispatchEvent(contextmenu);
    return {
      pointerDispatched,
      pointerDefaultPrevented: pointerdown.defaultPrevented,
      contextmenuDispatched,
      contextmenuDefaultPrevented: contextmenu.defaultPrevented,
      activeElementId: document.activeElement?.id ?? null,
    };
  });
  expect(nativeEvents).toEqual({
    pointerDispatched: true,
    pointerDefaultPrevented: false,
    contextmenuDispatched: true,
    contextmenuDefaultPrevented: false,
    activeElementId: "help-button",
  });
  await page.waitForTimeout(100);
  expect(commands).toEqual([]);
  await expect(targetSelect).toHaveValue(target);

  await battlefield.focus();
  await page.keyboard.press("Escape");
  await expect.poll(() => commands.length).toBe(1);
  expect(commands[0]).toMatchObject({
    command_type: "keyboard",
    key: "Escape",
  });
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(targetSelect).toHaveValue("");
  await expect(battlefield).not.toBeFocused();
  await page.unroute("**/api/command");
});

test("Live Mage Burst retains every authorized Mage and Warrior aura modifier", async ({
  page,
}) => {
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await page.locator("#reset-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#step-value")).toHaveText("0");
  const stepBefore = await currentStep(page);
  await page.locator("#ultimate-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#ultimate-button")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.locator("#submit-turn-button").click();
  await expect(page.locator("#step-value")).toHaveText(String(stepBefore + 1), {
    timeout: 120_000,
  });
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const inventories = await page.evaluate(() => {
    const identity = (/** @type {Element} */ element) =>
      element
        .closest("[data-presentation-key]")
        ?.getAttribute("data-presentation-key") ?? null;
    const factKey = (/** @type {Element} */ element) =>
      `${identity(element)}|${element.getAttribute("data-token-id")}`;
    return {
      roster: Array.from(
        document.querySelectorAll(".roster-fact-token--modifier"),
        factKey,
      ).sort(),
      battlefield: Array.from(
        document.querySelectorAll("#battlefield .modifier-cell"),
        factKey,
      ).sort(),
      controlledKey: document
        .querySelector('#battlefield .agent[data-controlled="true"]')
        ?.getAttribute("data-presentation-key"),
      suppressed: document
        .querySelector('#battlefield [data-layer="durable-status-modifier"]')
        ?.getAttribute("data-suppressed-modifier-presentation-keys"),
    };
  });
  expect(inventories.roster).toHaveLength(10);
  expect(inventories.battlefield).toEqual(inventories.roster);
  expect(inventories.suppressed).toBe("");
  expect(inventories.controlledKey).toBeTruthy();
  await expect(
    page.locator(
      `#battlefield .status-cell[data-presentation-key="${inventories.controlledKey}"]`,
    ),
  ).not.toHaveCount(0);
  await expect(
    page.locator(
      `#battlefield .cooldown-cell[data-presentation-key="${inventories.controlledKey}"]`,
    ),
  ).not.toHaveCount(0);
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

for (const [viewLabel, viewValue] of [
  ["Oracle", "researcher"],
  ["Agent", "pov"],
]) {
  test(`${viewLabel} queues next-turn WASD and Enter behind an in-flight Submit`, async ({
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
    const viewSelect = page.locator("#view-select");
    if (viewValue === "pov") {
      await viewSelect.selectOption(viewValue);
      await expect(viewSelect).toHaveValue(viewValue);
      await expect(page.locator("#connection-status")).toHaveText("Online");
    }
    requests.length = 0;

    const baseStep = await currentStep(page);
    const battlefield = page.locator("#battlefield");
    await battlefield.focus();
    const heldSubmitPresentation = await holdNextPresentation(page);
    let heldDraftPresentation = null;
    try {
      await page.keyboard.press("Enter");
      await heldSubmitPresentation.held;
      await expect(page.locator("#connection-status")).toHaveText("Command in flight");
      await expect.poll(() => requests.length).toBe(1);
      await expect(battlefield).toBeFocused();

      await page.keyboard.press("w");
      await expect(page.locator("#connection-status")).toHaveText("Input queued");
      await page.keyboard.press("Enter");
      await expect(page.locator("#connection-status")).toHaveText("Submit queued");
      await expect.poll(() => requests.length).toBe(1);
      heldDraftPresentation = await holdNextPresentation(page);
      heldSubmitPresentation.release();
      await heldDraftPresentation.held;
      await expect.poll(() => requests.length).toBe(2);
      expect(String(requests[1].command.key).toLowerCase()).toBe("w");
      await expect(page.locator("#connection-status")).toHaveText("Submit queued");

      await page.keyboard.press("a");
      await expect.poll(() => requests.length).toBe(2);
      await expect(page.locator("#connection-status")).toHaveText("Submit queued");
      await page.keyboard.press("Escape");
      await expect.poll(() => requests.length).toBe(2);
      await expect(page.locator("#connection-status")).toHaveText("Submit queued");
      heldDraftPresentation.release();
    } finally {
      heldSubmitPresentation.release();
      await heldSubmitPresentation.dispose();
      heldDraftPresentation?.release();
      await heldDraftPresentation?.dispose();
    }

    await expect.poll(() => requests.length, { timeout: 120_000 }).toBe(3);
    expect(
      requests.map((request) => String(request.command.key).toLowerCase()),
    ).toEqual(["enter", "w", "enter"]);
    expect(requests[1].command_id).not.toBe(requests[0].command_id);
    expect(requests[2].command_id).not.toBe(requests[1].command_id);
    expect(requests[1].base_revision).toBe(requests[0].base_revision + 1);
    expect(requests[2].base_revision).toBe(requests[1].base_revision + 1);
    await expect(page.locator("#step-value")).toHaveText(String(baseStep + 2), {
      timeout: 120_000,
    });
    await expect(page.locator("#connection-status")).toHaveText("Online");
    await expect(battlefield).not.toBeFocused();
    await page.unroute("**/api/command");
    if (viewValue === "pov") {
      await viewSelect.selectOption("researcher");
      await expect(viewSelect).toHaveValue("researcher");
      await expect(page.locator("#connection-status")).toHaveText("Online");
    }
    await page.locator("#reset-button").click();
    await expect(page.locator("#connection-status")).toHaveText("Online");
    await expect(page.locator("#step-value")).toHaveText("0");
  });
}

test("queued Escape preserves its direct battlefield focus release", async ({
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
  const heldSubmitPresentation = await holdNextPresentation(page);
  try {
    await page.keyboard.press("Enter");
    await heldSubmitPresentation.held;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await expect.poll(() => requests.length).toBe(1);

    await page.keyboard.press("Escape");
    await expect(page.locator("#connection-status")).toHaveText("Input queued");
    await expect.poll(() => requests.length).toBe(1);
    heldSubmitPresentation.release();
  } finally {
    heldSubmitPresentation.release();
    await heldSubmitPresentation.dispose();
  }

  await expect.poll(() => requests.length, { timeout: 120_000 }).toBe(2);
  expect(requests.map((request) => String(request.command.key).toLowerCase())).toEqual([
    "enter",
    "escape",
  ]);
  await expect(page.locator("#step-value")).toHaveText(String(baseStep + 1), {
    timeout: 120_000,
  });
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(battlefield).not.toBeFocused();
  await page.unroute("**/api/command");
  await page.locator("#reset-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#step-value")).toHaveText("0");
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
  await setDisclosureOpen(page, "#visual-filters", true);
  await setDisclosureOpen(stalePage, "#visual-filters", true);
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
  let markAppliedResponseHeld = () => {};
  let releaseAppliedResponse = () => {};
  const appliedResponseHeld = new Promise((resolve) => {
    markAppliedResponseHeld = () => resolve(undefined);
  });
  const appliedResponseRelease = new Promise((resolve) => {
    releaseAppliedResponse = () => resolve(undefined);
  });
  let interceptedCommands = 0;
  let appliedStatus = 0;
  await page.route("**/api/command", async (route) => {
    interceptedCommands += 1;
    const response = await route.fetch();
    appliedStatus = response.status();
    markAppliedResponseHeld();
    await appliedResponseRelease;
    await route.abort("failed");
  });

  const battlefield = page.locator("#battlefield");
  try {
    await battlefield.focus();
    await page.keyboard.press("Enter");
    await appliedResponseHeld;
    await expect(page.locator("#connection-status")).toHaveText("Command in flight");
    await page.keyboard.press("Escape");
    await expect(page.locator("#connection-status")).toHaveText("Input queued");
    await expect.poll(() => interceptedCommands).toBe(1);
  } finally {
    releaseAppliedResponse();
  }
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect(battlefield).not.toBeFocused();
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

  await setDisclosureOpen(page, "#visual-filters", true);
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

test("class details change only through explicit agent activation", async ({
  page,
}) => {
  /** @type {Record<string, any>[]} */
  const commands = [];
  await page.route("**/api/command", async (route) => {
    commands.push(route.request().postDataJSON().command ?? {});
    await route.continue();
  });

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const details = page.locator("#selection-card");
  const detailsText = () =>
    details.evaluate((card) => card.textContent?.replace(/\s+/gu, " ").trim() ?? "");
  await expect(details).toHaveText(
    "Activate an agent to inspect its comprehensive class details.",
  );
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("data-accent");
  const baseStep = await currentStep(page);

  const obstacle = page.locator("#battlefield .obstacle[data-tooltip-owner]").first();
  await expect(obstacle).toBeVisible();
  await obstacle.hover();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await obstacle.focus();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await obstacle.click();
  await expect(details).toHaveText(
    "Activate an agent to inspect its comprehensive class details.",
  );
  expect(commands).toHaveLength(0);

  const targetSelect = page.locator("#command-target-select");
  const originalTargetValue = await targetSelect.inputValue();
  const nextTargetValue = await targetSelect.evaluate((select, currentValue) => {
    if (!(select instanceof HTMLSelectElement)) {
      throw new TypeError("Target control is unavailable.");
    }
    const alternate = [...select.options].find(
      (option) => option.value !== currentValue && option.value !== "",
    );
    return alternate?.value ?? (currentValue === "" ? null : "");
  }, originalTargetValue);
  if (nextTargetValue === null) {
    throw new Error("Target control has no alternate authorized choice.");
  }
  await activateLiveCommand(page, () => targetSelect.selectOption(nextTargetValue));
  expect(commands.at(-1)).toMatchObject({
    command_type: "roster_selection",
    role: "target",
  });
  await expect(details).toHaveText(
    "Activate an agent to inspect its comprehensive class details.",
  );
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");

  const initialRosterActor = page.locator(
    '#roster .roster-primary-action[aria-pressed="true"]',
  );
  const initialRosterActorLabel = await initialRosterActor.getAttribute("aria-label");
  const initialPublicId =
    initialRosterActorLabel?.match(/Agent ID ([^,.]+)/u)?.[1] ?? null;
  if (initialRosterActorLabel === null || initialPublicId === null) {
    throw new Error("The initially controlled actor has no public identity.");
  }
  const rosterLabels = await page
    .locator("#roster .roster-primary-action")
    .evaluateAll((buttons) =>
      buttons
        .map((button) => button.getAttribute("aria-label"))
        .filter((label) => typeof label === "string"),
    );
  const activationLabels = rosterLabels.filter(
    (label) => label !== initialRosterActorLabel,
  );
  if (activationLabels.length < 3) {
    throw new Error("The live roster has fewer than three alternate agents.");
  }
  const publicIdFromLabel = (/** @type {string} */ label) => {
    const publicId = label.match(/Agent ID ([^,.]+)/u)?.[1];
    if (publicId === undefined) {
      throw new Error(`Could not parse a public identity from ${label}.`);
    }
    return publicId;
  };
  const [battlefieldLabel, rosterPointerLabel, rosterKeyboardLabel] = activationLabels;
  const battlefieldPublicId = publicIdFromLabel(battlefieldLabel);
  const rosterPointerPublicId = publicIdFromLabel(rosterPointerLabel);
  const rosterKeyboardPublicId = publicIdFromLabel(rosterKeyboardLabel);

  const battlefieldActor = page.locator(
    `#battlefield .agent[aria-label^="Agent ID ${battlefieldPublicId}."]`,
  );
  await expect(battlefieldActor).toHaveCount(1);
  await activateLiveCommand(page, () => battlefieldActor.click());
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await expect(details).toContainText(`Agent ID ${battlefieldPublicId}`);
  const battlefieldDetails = await detailsText();
  expect(battlefieldDetails).not.toContain("Activate an agent");

  const retainedObstacle = page
    .locator("#battlefield .obstacle[data-tooltip-owner]")
    .first();
  await retainedObstacle.hover();
  await retainedObstacle.focus();
  await retainedObstacle.click();
  await expect.poll(detailsText).toBe(battlefieldDetails);
  const commandsAfterObstacle = commands.length;
  const retainedTargetValue = await targetSelect.evaluate((select) => {
    if (!(select instanceof HTMLSelectElement)) {
      throw new TypeError("Target control is unavailable.");
    }
    const currentValue = select.value;
    const alternate = [...select.options].find(
      (option) => option.value !== "" && option.value !== currentValue,
    );
    return alternate?.value ?? null;
  });
  if (retainedTargetValue === null) {
    throw new Error("Target control has no alternate nonzero choice.");
  }
  await activateLiveCommand(page, () => targetSelect.selectOption(retainedTargetValue));
  expect(commands).toHaveLength(commandsAfterObstacle + 1);
  expect(commands.at(-1)).toMatchObject({
    command_type: "roster_selection",
    role: "target",
  });
  await expect.poll(detailsText).toBe(battlefieldDetails);

  await activateLiveCommand(page, () =>
    page.getByRole("button", { name: rosterPointerLabel, exact: true }).click(),
  );
  await expect(details).toContainText(`Agent ID ${rosterPointerPublicId}`);
  const rosterPointerDetails = await detailsText();
  expect(rosterPointerDetails).not.toBe(battlefieldDetails);

  await activateLiveCommand(page, () =>
    page.getByRole("button", { name: rosterKeyboardLabel, exact: true }).press("Enter"),
  );
  await expect(details).toContainText(`Agent ID ${rosterKeyboardPublicId}`);
  const rosterKeyboardDetails = await detailsText();
  expect(rosterKeyboardDetails).not.toBe(rosterPointerDetails);

  const finalObstacle = page
    .locator("#battlefield .obstacle[data-tooltip-owner]")
    .first();
  await finalObstacle.focus();
  await finalObstacle.click();
  await expect.poll(detailsText).toBe(rosterKeyboardDetails);
  await expect(page.locator("#step-value")).toHaveText(String(baseStep));

  await activateLiveCommand(page, () =>
    page.getByRole("button", { name: initialRosterActorLabel, exact: true }).click(),
  );
  if ((await targetSelect.inputValue()) !== originalTargetValue) {
    await activateLiveCommand(page, () =>
      targetSelect.selectOption(originalTargetValue),
    );
  }
  await expect(page.locator("#step-value")).toHaveText(String(baseStep));
  await page.unroute("**/api/command");
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
  await expectExactDisclosureDefaults(page);

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
  await setDisclosureOpen(page, "#visual-filters", true);
  const ranges = page.locator("#live-ranges-button");
  if ((await ranges.getAttribute("aria-pressed")) !== "true") {
    await ranges.click();
    await expect(ranges).toHaveAttribute("aria-pressed", "true");
  }
  await expect(page.locator("#battlefield .range-ring-owner").first()).toBeVisible();
  await page.locator("#battlefield .range-ring-owner").first().dispatchEvent("click");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");

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
