import { expect, test } from "@playwright/test";

import { REPLAY_AUTOPLAY_CADENCE_MS } from "../src/replay-controls.js";
import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  expectReplayFrameIndex,
  exportReplayArtifacts,
  removeReplayArtifacts,
  startReplayViewer,
} from "./support/replay-viewer.js";

test.describe.configure({ mode: "serial" });
test.use({ screenshot: "off" });

const FILTERS = Object.freeze([
  ["aura_fields", "Aura Fields"],
  ["aura_modifier_badges", "Aura Modifier Badges"],
  ["duration_status_badges", "Duration Status Badges"],
  ["spawn_shield", "Spawn Shield"],
  ["rejected_action_feedback", "Rejected Action Feedback"],
  ["basic_ability_effects", "Basic Ability Effects"],
  ["ultimate_ability_effects", "Ultimate Ability Effects"],
  ["damage_effects", "Damage Effects"],
  ["healing_effects", "Healing Effects"],
  ["regeneration_effects", "Regeneration Effects"],
  ["cooldown_effects", "Cooldown Effects"],
  ["charge_movement", "Charge Movement"],
  ["status_application", "Status Application"],
  ["status_reapplication", "Status Reapplication"],
  ["status_refresh_extension", "Status Refresh/Extension"],
  ["natural_status_expiry", "Natural Status Expiry"],
  ["freezing_trap_break", "Freezing Trap Break"],
  ["status_clear_on_death", "Status Clear on Death"],
  ["death_effects", "Death Effects"],
  ["respawn_wave", "Respawn Wave"],
  ["resurrection_effects", "Resurrection Effects"],
  ["spawn_shield_expiry", "Spawn-Shield Expiry"],
  ["scrolling_battle_text", "Scrolling Battle Text"],
]);

const FILTER_IDS = FILTERS.map(([id]) => id);
const FILTER_INPUT = 'input[type="checkbox"][data-visual-filter-id]';
const CHOREOGRAPHY_ROOTS =
  "#battlefield .combat-choreography, #battlefield .combat-choreography-routes";

/** @type {Awaited<ReturnType<typeof exportReplayArtifacts>> | null} */
let artifacts = null;
/** @type {Awaited<ReturnType<typeof startDebugger>> | null} */
let liveDebugger = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let sharedReplay = null;

test.beforeAll(async () => {
  /** @type {import("node:child_process").ChildProcess[]} */
  const startedProcesses = [];
  try {
    artifacts = await exportReplayArtifacts();
    liveDebugger = await startDebugger();
    startedProcesses.push(liveDebugger.process);
    sharedReplay = await startReplayViewer({
      replayPath: artifacts.shared,
      frameIndex: 1,
    });
    startedProcesses.push(sharedReplay.process);
  } catch (error) {
    const cleanupResults = await Promise.allSettled(
      startedProcesses.map((process) => stopDebugger(process)),
    );
    const cleanupErrors = cleanupResults.flatMap((result) =>
      result.status === "rejected" ? [result.reason] : [],
    );
    liveDebugger = null;
    sharedReplay = null;
    try {
      await removeReplayArtifacts(artifacts?.outputDirectory);
    } catch (cleanupError) {
      cleanupErrors.push(cleanupError);
    }
    artifacts = null;
    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        [error, ...cleanupErrors],
        "Visual-filter E2E startup and cleanup both failed.",
      );
    }
    throw error;
  }
});

test.afterAll(async () => {
  const processes = [liveDebugger?.process ?? null, sharedReplay?.process ?? null];
  liveDebugger = null;
  sharedReplay = null;
  const cleanupResults = await Promise.allSettled(
    processes.map((process) => stopDebugger(process)),
  );
  const cleanupErrors = cleanupResults.flatMap((result) =>
    result.status === "rejected" ? [result.reason] : [],
  );
  try {
    await removeReplayArtifacts(artifacts?.outputDirectory);
  } catch (error) {
    cleanupErrors.push(error);
  }
  artifacts = null;
  if (cleanupErrors.length > 0) {
    throw new AggregateError(cleanupErrors, "Visual-filter E2E cleanup failed.");
  }
});

/**
 * @template T
 * @param {T | null} value
 * @param {string} label
 * @returns {T}
 */
function requiredService(value, label) {
  if (value === null) {
    throw new Error(`${label} was not started.`);
  }
  return value;
}

/** @param {import("@playwright/test").Page} page */
function captureBrowserErrors(page) {
  /** @type {string[]} */
  const errors = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
  return errors;
}

/** @param {import("@playwright/test").Page} page */
function captureApiRequests(page) {
  /** @type {{method: string, path: string}[]} */
  const requests = [];
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname;
    if (path.startsWith("/api/")) {
      requests.push({ method: request.method(), path });
    }
  });
  return requests;
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} path
 */
async function authenticatedText(page, path) {
  return page.evaluate(async (requestPath) => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Debugger capability token is unavailable.");
    }
    const response = await fetch(requestPath, {
      cache: "no-store",
      credentials: "omit",
      headers: { "X-MARL-Debugger-Token": token },
      redirect: "error",
    });
    if (!response.ok) {
      throw new Error(`${requestPath} failed with HTTP ${response.status}.`);
    }
    return response.text();
  }, path);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 * @param {{viewerMode: "live" | "replay", audience: string, presentationKind: string}} expected
 */
async function openInstalled(page, url, expected) {
  await page.goto(url);
  await expect(page.locator("#connection-status")).toHaveText("Online", {
    timeout: 30_000,
  });
  const root = page.locator("html");
  await expect(root).toHaveAttribute("data-viewer-mode", expected.viewerMode);
  await expect(root).toHaveAttribute("data-audience", expected.audience);
  await expect(root).toHaveAttribute("data-presentation-authority", "installed");
  await expect(root).toHaveAttribute("data-submission-blocked", "false");
  const presentation = JSON.parse(
    await authenticatedText(page, "/api/presentation/frame"),
  );
  expect(presentation.presentation_kind).toBe(expected.presentationKind);
  expect(page.url()).not.toContain("token=");
}

/** @param {import("@playwright/test").Page} page */
async function settleLocalRender(page) {
  await page.evaluate(
    () =>
      new Promise((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      }),
  );
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{method: string, path: string}[]} apiRequests
 * @param {() => Promise<void>} action
 * @param {{label: string, delayMs?: number}} options
 */
async function expectLocalOnly(page, apiRequests, action, options) {
  const mark = apiRequests.length;
  await action();
  await settleLocalRender(page);
  const delayMs = options.delayMs;
  if (delayMs !== undefined && delayMs > 0) {
    await page.waitForTimeout(delayMs);
  }
  expect(apiRequests.slice(mark), `${options.label} caused an API request`).toEqual([]);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {"/api/command" | "/api/replay/command"} path
 * @param {() => Promise<unknown>} action
 */
async function performCommand(page, path, action) {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === path,
    { timeout: 30_000 },
  );
  await action();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );
}

/** @param {import("@playwright/test").Page} page */
async function openVisualFilters(page) {
  await page.locator("#visual-filters").evaluate((details) => {
    if (!(details instanceof HTMLDetailsElement)) {
      throw new TypeError("Visual Filters disclosure is unavailable.");
    }
    details.open = true;
  });
  await expect(page.locator("#visual-filter-options")).toBeVisible();
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string[]} disabledIds
 */
async function expectFilterSurface(page, disabledIds = []) {
  const expectedDisabled = new Set(disabledIds);
  const rows = await page
    .locator(`#visual-filter-options ${FILTER_INPUT}`)
    .evaluateAll((inputs) =>
      inputs.map((input) => {
        if (!(input instanceof HTMLInputElement)) {
          throw new TypeError("Visual filter is not an input.");
        }
        return {
          id: input.dataset.visualFilterId,
          label: input.parentElement?.textContent?.trim(),
          value: input.value,
          type: input.type,
          autocomplete: input.autocomplete,
          defaultChecked: input.defaultChecked,
          checked: input.checked,
        };
      }),
    );
  expect(rows).toEqual(
    FILTERS.map(([id, label]) => ({
      id,
      label,
      value: id,
      type: "checkbox",
      autocomplete: "off",
      defaultChecked: true,
      checked: !expectedDisabled.has(id),
    })),
  );
  const enabledCount = FILTERS.length - expectedDisabled.size;
  await expect(page.locator("#visual-filter-count")).toHaveText(
    `${enabledCount} enabled`,
  );
  const restoreAll = page.locator("#restore-all-visual-filters-button");
  if (enabledCount === FILTERS.length) {
    await expect(restoreAll).toBeDisabled();
  } else {
    await expect(restoreAll).toBeEnabled();
  }
  expect(
    await page.locator("#visual-filters button").evaluateAll((buttons) =>
      buttons.map((button) => ({
        id: button.id,
        text: button.textContent?.trim(),
      })),
    ),
  ).toEqual([{ id: "restore-all-visual-filters-button", text: "Restore All" }]);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{method: string, path: string}[]} apiRequests
 * @param {string} filterId
 * @param {boolean} enabled
 * @param {string} label
 */
async function setFilter(page, apiRequests, filterId, enabled, label) {
  const input = page.locator(`${FILTER_INPUT}[data-visual-filter-id="${filterId}"]`);
  await expectLocalOnly(
    page,
    apiRequests,
    async () => {
      if (enabled) {
        await input.check();
      } else {
        await input.uncheck();
      }
    },
    { label },
  );
  await expect(input).toBeChecked({ checked: enabled });
}

/**
 * Exact science, inspector, legality, authority, and accessible-feed bytes.
 * Deliberately excludes filter paint and playback state.
 *
 * @param {import("@playwright/test").Page} page
 * @param {boolean} replay
 */
async function scientificSignature(page, replay) {
  const apiPaths = replay
    ? ["/api/frame", "/api/presentation/frame", "/api/replay/timeline"]
    : ["/api/frame", "/api/presentation/frame"];
  const api = Object.fromEntries(
    await Promise.all(
      apiPaths.map(async (path) => [path, await authenticatedText(page, path)]),
    ),
  );
  const dom = await page.evaluate(() => {
    /** @param {string} selector */
    const exactHtml = (selector) => {
      const element = document.querySelector(selector);
      if (!(element instanceof HTMLElement)) {
        throw new Error(`Missing scientific surface ${selector}.`);
      }
      return element.innerHTML;
    };
    const controls = Array.from(
      document.querySelectorAll(
        "#view-select, #command-deck button[data-key], #command-target-select, #reset-button",
      ),
    ).map((control) => ({
      id: control.id,
      tag: control.tagName,
      key: control.getAttribute("data-key"),
      value:
        control instanceof HTMLInputElement || control instanceof HTMLSelectElement
          ? control.value
          : null,
      disabled:
        control instanceof HTMLButtonElement ||
        control instanceof HTMLInputElement ||
        control instanceof HTMLSelectElement
          ? control.disabled
          : null,
      pressed: control.getAttribute("aria-pressed"),
      description: control.getAttribute("aria-description"),
      text: control.textContent?.trim() ?? "",
    }));
    return {
      authority: {
        productKind: document.documentElement.dataset.productKind,
        viewerMode: document.documentElement.dataset.viewerMode,
        audience: document.documentElement.dataset.audience,
        preset: document.documentElement.dataset.preset,
        presentationAuthority: document.documentElement.dataset.presentationAuthority,
      },
      step: document.querySelector("#step-value")?.textContent,
      transition: document.querySelector("#transition-value")?.textContent,
      feed: exactHtml("#event-feed"),
      selection: exactHtml("#selection-card"),
      pending: exactHtml("#pending-card"),
      accepted: exactHtml("#accepted-card"),
      diagnostics: exactHtml("#diagnostics-card"),
      disclosures: [
        "roster-details",
        "agent-details",
        "pending-turn-details",
        "latest-transition-details",
        "events-details",
        "technical-frame-details",
      ].map((id) => ({
        id,
        open: /** @type {HTMLDetailsElement | null} */ (document.getElementById(id))
          ?.open,
      })),
      controls,
      rosterAuthority: Array.from(
        document.querySelectorAll("#roster [data-presentation-key]"),
      ).map((element) => ({
        key: element.getAttribute("data-presentation-key"),
        controlled: element.getAttribute("data-controlled"),
        selected: element.getAttribute("data-selected"),
        pressed: element.getAttribute("aria-pressed"),
        disabled: element.getAttribute("aria-disabled"),
      })),
      battlefieldAuthority: Array.from(
        document.querySelectorAll("#battlefield .agent[data-presentation-key]"),
      ).map((element) => ({
        key: element.getAttribute("data-presentation-key"),
        controlled: element.getAttribute("data-controlled"),
        selected: element.getAttribute("data-selected"),
      })),
    };
  });
  return { api, dom };
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {"#live-ranges-button" | "#replay-ranges-button"} buttonSelector
 */
async function rangeSignature(page, buttonSelector) {
  return page.evaluate((selector) => {
    const button = document.querySelector(selector);
    const layer = document.querySelector('#battlefield [data-layer="debug-range"]');
    if (!(button instanceof HTMLButtonElement) || !(layer instanceof Element)) {
      throw new Error("Range presentation surface is unavailable.");
    }
    return {
      id: button.id,
      pressed: button.getAttribute("aria-pressed"),
      disabled: button.disabled,
      description: button.getAttribute("aria-description"),
      markup: layer.innerHTML,
    };
  }, buttonSelector);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {"live" | "replay"} mode
 */
async function ensureRangesOn(page, mode) {
  const selector = mode === "live" ? "#live-ranges-button" : "#replay-ranges-button";
  const path = mode === "live" ? "/api/command" : "/api/replay/command";
  const button = page.locator(selector);
  if ((await button.getAttribute("aria-pressed")) !== "true") {
    await performCommand(page, path, () => button.click());
  }
  await expect(button).toHaveAttribute("aria-pressed", "true");
}

/** @param {import("@playwright/test").Page} page */
function auraMarkup(page) {
  return page.locator('#battlefield [data-layer="aura"]').innerHTML();
}

/** @param {import("@playwright/test").Page} page */
async function expectTooltipCleared(page) {
  await page.evaluate(() => {
    const active = document.activeElement;
    if (active instanceof HTMLElement || active instanceof SVGElement) {
      active.blur();
    }
  });
  await page.mouse.move(1, 1);
  await settleLocalRender(page);
  await expect(page.locator("#visual-tooltip")).toBeHidden();
  await expect(page.locator("#visual-tooltip-title")).toHaveText("");
  await expect(page.locator("#visual-tooltip-details")).toHaveText("");
  await expect(
    page.locator('#battlefield [aria-describedby~="visual-tooltip"]'),
  ).toHaveCount(0);
}

/** @param {import("@playwright/test").Page} page */
async function expectAuraAbsent(page) {
  await expect(page.locator('#battlefield [data-layer="aura"]')).toBeEmpty();
  await expect(
    page.locator(
      "#battlefield .aura-field, #battlefield .aura-field-owner, #battlefield .aura-field-hit",
    ),
  ).toHaveCount(0);
  await expectTooltipCleared(page);
}

/** @param {import("@playwright/test").Page} page */
async function ultimateInventory(page) {
  return page
    .locator('#battlefield .combat-effect--activation[data-component="ultimate"]')
    .evaluateAll((elements) =>
      elements.map((element) => ({
        eventId: element.getAttribute("data-event-id"),
        eventType: element.getAttribute("data-event-type"),
        component: element.getAttribute("data-component"),
        markup: element.outerHTML,
      })),
    );
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string[]} eventIds
 */
async function expectUltimateAbsent(page, eventIds) {
  const residue = await page.locator("#battlefield").evaluate((battlefield, ids) => {
    const markup = battlefield.innerHTML;
    return {
      ownedEventIds: Array.from(
        battlefield.querySelectorAll("[data-event-id]"),
        (element) => element.getAttribute("data-event-id"),
      ).filter((id) => id !== null && ids.includes(id)),
      serializedIds: ids.filter((id) => markup.includes(id)),
    };
  }, eventIds);
  expect(residue).toEqual({ ownedEventIds: [], serializedIds: [] });
  await expectTooltipCleared(page);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{method: string, path: string}[]} apiRequests
 */
async function disableEveryFilter(page, apiRequests) {
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator(`#visual-filter-options ${FILTER_INPUT}`).evaluateAll((inputs) => {
        for (const input of inputs) {
          if (input instanceof HTMLInputElement && input.checked) {
            input.click();
          }
        }
      }),
    { label: "disabling all 23 filters" },
  );
}

/** @param {import("@playwright/test").Page} page */
async function expectAllPaintAbsentWithoutDwell(page) {
  await expect(
    page.locator(
      [
        "#battlefield .aura-field",
        "#battlefield .modifier-cell",
        "#battlefield .modifier-overflow",
        "#battlefield .status-cell",
        "#battlefield .status-overflow",
        '#battlefield .required-dock-fallback[data-kind="status"]',
        "#battlefield .pov-observed-status",
        "#battlefield .agent-spawn-shield",
        "#battlefield .cooldown-cell",
        '#battlefield .required-dock-fallback[data-kind="cooldown"]',
        "#battlefield .combat-effect",
        "#battlefield .combat-route-effect",
      ].join(", "),
    ),
  ).toHaveCount(0);
  const state = await page.evaluate(() => ({
    blocked: document.documentElement.dataset.submissionBlocked,
    roots: Array.from(
      document.querySelectorAll(
        "#battlefield .combat-choreography, #battlefield .combat-choreography-routes",
      ),
    ).map((root) => ({
      state: root.getAttribute("data-state"),
      paintKey: root.getAttribute("data-paint-key"),
      childCount: root.childElementCount,
    })),
    animations: document
      .getAnimations()
      .map((animation) => animation.id)
      .filter((id) => id.startsWith("mbg:")),
    agentAria: Array.from(document.querySelectorAll("#battlefield .agent"), (agent) =>
      agent.getAttribute("aria-label"),
    ),
  }));
  expect(state.blocked).toBe("false");
  expect(state.roots).toEqual([
    {
      state: "settled",
      paintKey: `visual-filters-v1:${"0".repeat(23)}`,
      childCount: 0,
    },
    {
      state: "settled",
      paintKey: `visual-filters-v1:${"0".repeat(23)}`,
      childCount: 0,
    },
  ]);
  expect(state.animations).toEqual([]);
  for (const ariaLabel of state.agentAria) {
    expect(ariaLabel).not.toMatch(/combat status|spawn shield/iu);
  }
  await expectTooltipCleared(page);
}

test("visual filters remain page-local across live Oracle/NoShared and replay Oracle/Shared authority", async ({
  page,
}) => {
  const browserErrors = captureBrowserErrors(page);
  const apiRequests = captureApiRequests(page);

  const live = requiredService(liveDebugger, "Live debugger");
  await openInstalled(page, live.url, {
    viewerMode: "live",
    audience: "researcher",
    presentationKind: "live_oracle",
  });
  await openVisualFilters(page);
  await expectFilterSurface(page);
  await ensureRangesOn(page, "live");

  const liveScience = await scientificSignature(page, false);
  const liveRanges = await rangeSignature(page, "#live-ranges-button");
  const liveAuras = await auraMarkup(page);
  expect(liveAuras).not.toBe("");
  await expect(
    page.locator(
      '#battlefield [data-layer="aura"] .aura-field[role="img"][data-tooltip-owner]',
    ),
  ).not.toHaveCount(0);

  await setFilter(page, apiRequests, "aura_fields", false, "live aura off");
  await expectFilterSurface(page, ["aura_fields"]);
  await expectAuraAbsent(page);
  expect(await scientificSignature(page, false)).toEqual(liveScience);
  expect(await rangeSignature(page, "#live-ranges-button")).toEqual(liveRanges);

  await setFilter(page, apiRequests, "aura_fields", true, "live aura on");
  expect(await auraMarkup(page)).toBe(liveAuras);
  expect(await scientificSignature(page, false)).toEqual(liveScience);
  expect(await rangeSignature(page, "#live-ranges-button")).toEqual(liveRanges);

  await setFilter(page, apiRequests, "aura_fields", false, "live durable filter");
  await setFilter(
    page,
    apiRequests,
    "ultimate_ability_effects",
    false,
    "live transient filter",
  );
  const disabledPair = ["aura_fields", "ultimate_ability_effects"];
  await expectFilterSurface(page, disabledPair);

  const initialStep = Number(await page.locator("#step-value").textContent());
  await performCommand(page, "/api/command", () =>
    page.locator("#submit-turn-button").click(),
  );
  await expect(page.locator("#step-value")).toHaveText(String(initialStep + 1));
  await expectFilterSurface(page, disabledPair);

  await performCommand(page, "/api/command", () =>
    page.locator("#view-select").selectOption("pov"),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  const livePov = JSON.parse(await authenticatedText(page, "/api/presentation/frame"));
  expect(livePov.presentation_kind).toBe("live_no_shared_obs_agent_pov");
  await expectFilterSurface(page, disabledPair);

  await performCommand(page, "/api/command", () =>
    page.locator("#reset-button").click(),
  );
  await expect(page.locator("#step-value")).toHaveText("0");
  await expectFilterSurface(page, disabledPair);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await openVisualFilters(page);
  await expectFilterSurface(page);
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("live_no_shared_obs_agent_pov");

  const replay = requiredService(sharedReplay, "Shared replay viewer");
  await openInstalled(page, replay.url, {
    viewerMode: "replay",
    audience: "researcher",
    presentationKind: "replay_oracle",
  });
  await expectReplayFrameIndex(page, 1);
  await openVisualFilters(page);
  await expectFilterSurface(page);
  await ensureRangesOn(page, "replay");

  const replayScience = await scientificSignature(page, true);
  const replayRanges = await rangeSignature(page, "#replay-ranges-button");
  const replayAuras = await auraMarkup(page);
  expect(replayAuras).not.toBe("");

  const playPause = page.locator("#replay-play-pause-button");
  const replayAuraInput = page.locator(
    `${FILTER_INPUT}[data-visual-filter-id="aura_fields"]`,
  );
  await expectLocalOnly(
    page,
    apiRequests,
    async () => {
      await playPause.click();
      await expect(playPause).toHaveText("Pause");
      await expect(playPause).toHaveAttribute("aria-pressed", "true");
      await replayAuraInput.evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) {
          throw new TypeError("Aura filter is unavailable.");
        }
        input.click();
      });
    },
    {
      label: "replay auto-pause aura toggle",
      delayMs: REPLAY_AUTOPLAY_CADENCE_MS.normal + 125,
    },
  );
  await expect(playPause).toHaveText("Play");
  await expect(playPause).toHaveAttribute("aria-pressed", "false");
  await expectReplayFrameIndex(page, 1);
  await expectAuraAbsent(page);
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(replayRanges);

  await setFilter(page, apiRequests, "aura_fields", true, "replay aura on");
  expect(await auraMarkup(page)).toBe(replayAuras);
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(replayRanges);

  const ultimateBefore = await ultimateInventory(page);
  expect(ultimateBefore.length).toBeGreaterThan(0);
  expect(ultimateBefore.every((row) => row.component === "ultimate")).toBe(true);
  const ultimateIds = ultimateBefore.map(({ eventId }) => {
    if (typeof eventId !== "string" || eventId.length === 0) {
      throw new Error("Real replay ultimate has no event ID.");
    }
    return eventId;
  });
  const ultimateOwner = page
    .locator(
      '#battlefield .combat-effect--activation[data-component="ultimate"][data-tooltip-owner]',
    )
    .first();
  await ultimateOwner.dispatchEvent("focusin");
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "activation",
  );
  await expect(page.locator("#visual-tooltip-title")).not.toHaveText("");

  const ultimateInput = page.locator(
    `${FILTER_INPUT}[data-visual-filter-id="ultimate_ability_effects"]`,
  );
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      ultimateInput.evaluate((input) => {
        if (!(input instanceof HTMLInputElement)) {
          throw new TypeError("Ultimate filter is unavailable.");
        }
        input.checked = false;
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }),
    { label: "real replay ultimate off" },
  );
  await expect(ultimateInput).not.toBeChecked();
  await expect(page.locator("#visual-tooltip")).toBeHidden();
  await expectUltimateAbsent(page, ultimateIds);
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(replayRanges);

  await setFilter(
    page,
    apiRequests,
    "ultimate_ability_effects",
    true,
    "real replay ultimate on",
  );
  expect(await ultimateInventory(page)).toEqual(ultimateBefore);
  expect(await scientificSignature(page, true)).toEqual(replayScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(replayRanges);

  await setFilter(
    page,
    apiRequests,
    "aura_fields",
    false,
    "replay durable persistence",
  );
  await setFilter(
    page,
    apiRequests,
    "ultimate_ability_effects",
    false,
    "replay transient persistence",
  );
  await expectFilterSurface(page, disabledPair);

  await performCommand(page, "/api/replay/command", () =>
    page.locator("#view-select").selectOption("pov"),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("replay_shared_obs_agent_pov");
  await expectFilterSurface(page, disabledPair);

  await performCommand(page, "/api/replay/command", () =>
    page.locator("#replay-frame-slider").fill("2"),
  );
  await expectReplayFrameIndex(page, 2);
  await expectFilterSurface(page, disabledPair);
  await performCommand(page, "/api/replay/command", () =>
    page.locator("#replay-frame-slider").fill("1"),
  );
  await expectReplayFrameIndex(page, 1);
  await expectFilterSurface(page, disabledPair);

  await performCommand(page, "/api/replay/command", () =>
    page.locator("#view-select").selectOption("researcher"),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("replay_oracle");
  await expectFilterSurface(page, disabledPair);

  const allOffScience = await scientificSignature(page, true);
  const allOffRanges = await rangeSignature(page, "#replay-ranges-button");
  await disableEveryFilter(page, apiRequests);
  await expectFilterSurface(page, FILTER_IDS);
  await expectAllPaintAbsentWithoutDwell(page);
  expect(await scientificSignature(page, true)).toEqual(allOffScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(allOffRanges);

  await expectLocalOnly(
    page,
    apiRequests,
    () => page.locator("#restore-all-visual-filters-button").click(),
    { label: "Restore All" },
  );
  await expectFilterSurface(page);
  expect(await scientificSignature(page, true)).toEqual(allOffScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(allOffRanges);
  expect(await auraMarkup(page)).toBe(replayAuras);
  expect(await ultimateInventory(page)).toEqual(ultimateBefore);
  const restoredChoreographyRoots = page.locator(CHOREOGRAPHY_ROOTS);
  await expect(restoredChoreographyRoots).toHaveCount(2);
  expect(
    await restoredChoreographyRoots.evaluateAll((roots) =>
      roots.map((root) => root.getAttribute("data-paint-key")),
    ),
  ).toEqual(Array(2).fill(`visual-filters-v1:${"1".repeat(23)}`));

  await setFilter(page, apiRequests, "aura_fields", false, "pre-reload replay aura");
  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(page, 1);
  await openVisualFilters(page);
  await expectFilterSurface(page);
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("replay_oracle");

  expect(browserErrors).toEqual([]);
});
