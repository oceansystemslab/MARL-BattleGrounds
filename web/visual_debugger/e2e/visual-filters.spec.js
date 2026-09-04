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
  ["target_selection_visuals", "Target Selection Visuals"],
  ["basic_ability_effects", "Basic Ability Effects"],
  ["ultimate_ability_effects", "Ultimate Ability Effects"],
  ["regeneration_effects", "Regeneration Effects"],
  ["cooldown_effects", "Cooldown Effects"],
  ["status_application", "Status Application"],
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
  "#battlefield .combat-choreography, #battlefield .combat-choreography-connectors, #battlefield .combat-choreography-routes";

/** @type {Awaited<ReturnType<typeof exportReplayArtifacts>> | null} */
let artifacts = null;
/** @type {Awaited<ReturnType<typeof startDebugger>> | null} */
let liveDebugger = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let sharedReplay = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let noSharedReplay = null;

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
    noSharedReplay = await startReplayViewer({
      replayPath: artifacts.complete,
      frameIndex: 1,
    });
    startedProcesses.push(noSharedReplay.process);
  } catch (error) {
    const cleanupResults = await Promise.allSettled(
      startedProcesses.map((process) => stopDebugger(process)),
    );
    const cleanupErrors = cleanupResults.flatMap((result) =>
      result.status === "rejected" ? [result.reason] : [],
    );
    liveDebugger = null;
    sharedReplay = null;
    noSharedReplay = null;
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
  const processes = [
    liveDebugger?.process ?? null,
    sharedReplay?.process ?? null,
    noSharedReplay?.process ?? null,
  ];
  liveDebugger = null;
  sharedReplay = null;
  noSharedReplay = null;
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
 * @param {boolean} rangesEnabled
 */
async function expectFilterSurface(page, disabledIds = [], rangesEnabled = true) {
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
  const enabledCount = FILTERS.length - expectedDisabled.size + (rangesEnabled ? 1 : 0);
  await expect(page.locator("#visual-filter-count")).toHaveText(
    `${enabledCount} enabled`,
  );
  const enableAll = page.locator("#enable-all-visual-filters-button");
  const disableAll = page.locator("#disable-all-visual-filters-button");
  if (enabledCount === FILTERS.length + 1) {
    await expect(enableAll).toBeDisabled();
  } else {
    await expect(enableAll).toBeEnabled();
  }
  if (enabledCount === 0) {
    await expect(disableAll).toBeDisabled();
  } else {
    await expect(disableAll).toBeEnabled();
  }
  expect(
    await page.locator("#visual-filters button").evaluateAll((buttons) =>
      buttons.map((button) => ({
        id: button.id,
        text: button.textContent?.trim(),
      })),
    ),
  ).toEqual([
    { id: "live-ranges-button", text: "Ranges" },
    { id: "replay-ranges-button", text: "Ranges" },
    { id: "enable-all-visual-filters-button", text: "Enable All" },
    { id: "disable-all-visual-filters-button", text: "Disable All" },
  ]);
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
 * Exact science, inspector, legality, authority, and authorized-panel bytes.
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
      selection: exactHtml("#selection-card"),
      pending: exactHtml("#pending-card"),
      accepted: exactHtml("#accepted-card"),
      diagnostics: exactHtml("#diagnostics-card"),
      disclosures: [
        "roster-details",
        "agent-details",
        "pending-turn-details",
        "latest-transition-details",
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
 * Remove one presentation source's service revision pair.
 *
 * @param {unknown} value
 */
function omitPresentationRevision(value) {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    Reflect.deleteProperty(value, "source_revision");
    Reflect.deleteProperty(value, "source_authority_epoch");
  }
}

/**
 * Normalize only the exact transport paths changed by the existing Ranges
 * service command. No scientific scene, action, lifecycle, panel, or authority
 * content is excluded.
 *
 * @param {string} path
 * @param {string} serialized
 * @returns {unknown}
 */
function rangeIndependentApiPayload(path, serialized) {
  /** @type {Record<string, any>} */
  const payload = JSON.parse(serialized);
  if (path === "/api/frame") {
    delete payload.revision;
    delete payload.show_ranges;
    if (payload.projection?.scene) {
      delete payload.projection.scene.ranges;
    }
    if (Array.isArray(payload.hud?.diagnostics)) {
      for (const fact of payload.hud.diagnostics) {
        if (fact?.fact_id === "revision") {
          fact.value = "<range-presentation revision>";
        }
      }
    }
    return payload;
  }
  if (path === "/api/presentation/frame") {
    omitPresentationRevision(payload.source);
    omitPresentationRevision(payload.live_inspection);
    omitPresentationRevision(payload.researcher_space);
  }
  return payload;
}

/**
 * Exact scientific, authority, and panel bytes with only the expected Ranges
 * presentation toggle normalized away.
 *
 * @param {import("@playwright/test").Page} page
 * @param {boolean} replay
 */
async function rangeIndependentScientificSignature(page, replay) {
  const signature = await scientificSignature(page, replay);
  return {
    api: Object.fromEntries(
      Object.entries(signature.api).map(([path, payload]) => [
        path,
        rangeIndependentApiPayload(path, payload),
      ]),
    ),
    dom: signature.dom,
  };
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

/** @param {import("@playwright/test").Page} page */
async function targetSelectionPaintSignature(page) {
  return page.locator("#battlefield").evaluate((battlefield) => ({
    visibleReticles: battlefield.querySelectorAll(".selected-reticle:not([hidden])")
      .length,
    legality: Array.from(
      battlefield.querySelectorAll(".legality-pill"),
      (node) => node.outerHTML,
    ),
    controlled: battlefield.querySelector(".controlled-halo:not([hidden])")?.outerHTML,
    pendingRoute: battlefield.querySelector('[data-layer="pending-route"]')?.innerHTML,
    actionRoutes: battlefield.querySelector(".combat-choreography-routes")?.innerHTML,
    ranges: battlefield.querySelector('[data-layer="debug-range"]')?.innerHTML,
    bodies: battlefield.querySelector('[data-layer="body"]')?.innerHTML,
  }));
}

/**
 * Exercise the target-selection paint boundary through one real Replay Agent
 * authority. Replay selection changes the inspected POV, so hidden-target
 * retention is intentionally proved only by the live draft path.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{method: string, path: string}[]} apiRequests
 * @param {{url: string}} replay
 * @param {string} expectedPresentationKind
 * @param {string} label
 */
async function expectReplayAgentTargetFilter(
  page,
  apiRequests,
  replay,
  expectedPresentationKind,
  label,
) {
  await openInstalled(page, replay.url, {
    viewerMode: "replay",
    audience: "researcher",
    presentationKind: "replay_oracle",
  });
  await expectReplayFrameIndex(page, 1);
  const candidate = page
    .locator('#roster .roster-primary-action:not([aria-pressed="true"])')
    .first();
  await expect(candidate).toBeEnabled();
  await performCommand(page, "/api/replay/command", () => candidate.click());
  await openVisualFilters(page);
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    `${label} enable replay-default target visuals`,
  );
  const oracleScience = await scientificSignature(page, true);
  const oraclePaint = await targetSelectionPaintSignature(page);
  expect(oraclePaint.visibleReticles, `${label} Oracle enabled reticle`).toBe(1);
  expect(oraclePaint.legality, `${label} Oracle enabled legality`).toHaveLength(2);
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    `${label} Oracle target visuals off`,
  );
  const filteredOracle = await targetSelectionPaintSignature(page);
  expect(filteredOracle.visibleReticles).toBe(0);
  expect(filteredOracle.legality).toEqual([]);
  expect(filteredOracle.controlled).toBe(oraclePaint.controlled);
  expect(filteredOracle.pendingRoute).toBe(oraclePaint.pendingRoute);
  expect(filteredOracle.actionRoutes).toBe(oraclePaint.actionRoutes);
  expect(filteredOracle.ranges).toBe(oraclePaint.ranges);
  expect(filteredOracle.bodies).toBe(oraclePaint.bodies);
  expect(await scientificSignature(page, true)).toEqual(oracleScience);
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    `${label} Oracle target visuals on`,
  );
  expect(await targetSelectionPaintSignature(page)).toEqual(oraclePaint);
  await performCommand(page, "/api/replay/command", () =>
    page.locator("#view-select").selectOption("pov"),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe(expectedPresentationKind);
  const input = page.locator(
    `${FILTER_INPUT}[data-visual-filter-id="target_selection_visuals"]`,
  );
  if (!(await input.isChecked())) {
    await setFilter(
      page,
      apiRequests,
      "target_selection_visuals",
      true,
      `${label} target visuals initial restore`,
    );
  }
  const science = await scientificSignature(page, true);
  const paint = await targetSelectionPaintSignature(page);
  expect(paint.visibleReticles, `${label} enabled reticle`).toBe(1);
  expect(paint.legality, `${label} enabled legality`).toHaveLength(2);

  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    `${label} target visuals off`,
  );
  const filtered = await targetSelectionPaintSignature(page);
  expect(filtered.visibleReticles).toBe(0);
  expect(filtered.legality).toEqual([]);
  expect(filtered.controlled).toBe(paint.controlled);
  expect(filtered.pendingRoute).toBe(paint.pendingRoute);
  expect(filtered.actionRoutes).toBe(paint.actionRoutes);
  expect(filtered.ranges).toBe(paint.ranges);
  expect(filtered.bodies).toBe(paint.bodies);
  expect(await scientificSignature(page, true)).toEqual(science);

  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    `${label} target visuals on`,
  );
  expect(await targetSelectionPaintSignature(page)).toEqual(paint);
  expect(await scientificSignature(page, true)).toEqual(science);
  await performCommand(page, "/api/replay/command", () =>
    page.locator("#view-select").selectOption("researcher"),
  );
  await page.goto("about:blank");
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
        tokenId: element.getAttribute("data-token-id"),
        sourceClass: element.getAttribute("data-source-class"),
        ariaLabel: element.getAttribute("aria-label"),
        ariaDescription: element.getAttribute("aria-description"),
        icons: Array.from(element.querySelectorAll("[data-icon]"), (icon) =>
          icon.getAttribute("data-icon"),
        ),
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
 */
async function disableAllFilters(page) {
  const replay =
    (await page.locator("html").getAttribute("data-viewer-mode")) === "replay";
  await performCommand(page, replay ? "/api/replay/command" : "/api/command", () =>
    page.locator("#disable-all-visual-filters-button").click(),
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
        "#battlefield .combat-effect",
        "#battlefield .combat-connector-effect",
        "#battlefield .combat-route-effect",
      ].join(", "),
    ),
  ).toHaveCount(0);
  const state = await page.evaluate(() => ({
    blocked: document.documentElement.dataset.submissionBlocked,
    roots: Array.from(
      document.querySelectorAll(
        "#battlefield .combat-choreography, #battlefield .combat-choreography-connectors, #battlefield .combat-choreography-routes",
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
      paintKey: `visual-filters-v2:${"0".repeat(18)}`,
      childCount: 0,
    },
    {
      state: "settled",
      paintKey: `visual-filters-v2:${"0".repeat(18)}`,
      childCount: 0,
    },
    {
      state: "settled",
      paintKey: `visual-filters-v2:${"0".repeat(18)}`,
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
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#enable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Enable All is unavailable.");
        }
        button.click();
      }),
    { label: "disabled Enable All idempotency" },
  );
  await expectFilterSurface(page);
  await ensureRangesOn(page, "live");

  const liveBulkScience = await rangeIndependentScientificSignature(page, false);
  const liveBulkRanges = await rangeSignature(page, "#live-ranges-button");
  const liveBulkStep = await page.locator("#step-value").textContent();
  const liveDisableMark = apiRequests.length;
  await disableAllFilters(page);
  expect(
    apiRequests.slice(liveDisableMark).filter(({ method }) => method === "POST"),
    "Live Disable All range request count",
  ).toEqual([{ method: "POST", path: "/api/command" }]);
  await expectFilterSurface(page, FILTER_IDS, false);
  expect(await rangeIndependentScientificSignature(page, false)).toEqual(
    liveBulkScience,
  );
  await expect(page.locator("#step-value")).toHaveText(liveBulkStep ?? "0");
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#disable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Disable All is unavailable.");
        }
        button.click();
      }),
    { label: "live disabled Disable All idempotency" },
  );

  const liveEnableMark = apiRequests.length;
  await performCommand(page, "/api/command", () =>
    page.locator("#enable-all-visual-filters-button").click(),
  );
  expect(
    apiRequests.slice(liveEnableMark).filter(({ method }) => method === "POST"),
    "Live Enable All range request count",
  ).toEqual([{ method: "POST", path: "/api/command" }]);
  await expectFilterSurface(page);
  expect(await rangeIndependentScientificSignature(page, false)).toEqual(
    liveBulkScience,
  );
  expect(await rangeSignature(page, "#live-ranges-button")).toEqual(liveBulkRanges);
  await expect(page.locator("#step-value")).toHaveText(liveBulkStep ?? "0");
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#enable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Enable All is unavailable.");
        }
        button.click();
      }),
    { label: "live disabled Enable All idempotency" },
  );

  await performCommand(page, "/api/command", () =>
    page
      .getByRole("button", {
        name: "Control and inspect Agent ID 4",
        exact: true,
      })
      .click(),
  );
  await performCommand(page, "/api/command", () =>
    page.locator('#command-deck button[data-key="d"]').click(),
  );
  const targetSelect = page.locator("#command-target-select");
  const targetOption = targetSelect
    .locator("option")
    .filter({ hasText: "Agent ID 3 ·" });
  await expect(targetOption).toHaveCount(1);
  const targetValue = await targetOption.getAttribute("value");
  if (targetValue === null) {
    throw new Error("Live target-selection filter proof has no Priest ally target.");
  }
  await performCommand(page, "/api/command", () =>
    targetSelect.selectOption(targetValue),
  );
  await performCommand(page, "/api/command", () =>
    page.locator('#command-deck button[data-key="1"]').click(),
  );
  await expect(targetSelect).toHaveValue(targetValue);
  const targetScience = await scientificSignature(page, false);
  const targetPaint = await targetSelectionPaintSignature(page);
  expect(targetPaint.visibleReticles).toBe(1);
  expect(targetPaint.legality).toHaveLength(2);
  expect(targetPaint.controlled).toBeDefined();
  expect(targetPaint.pendingRoute).not.toBe("");

  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    "live target-selection visuals off",
  );
  const targetPaintHidden = await targetSelectionPaintSignature(page);
  expect(targetPaintHidden.visibleReticles).toBe(0);
  expect(targetPaintHidden.legality).toEqual([]);
  expect(targetPaintHidden.controlled).toBe(targetPaint.controlled);
  expect(targetPaintHidden.pendingRoute).toBe(targetPaint.pendingRoute);
  expect(targetPaintHidden.actionRoutes).toBe(targetPaint.actionRoutes);
  expect(targetPaintHidden.ranges).toBe(targetPaint.ranges);
  expect(targetPaintHidden.bodies).toBe(targetPaint.bodies);
  expect(await scientificSignature(page, false)).toEqual(targetScience);
  await expect(targetSelect).toHaveValue(targetValue);
  await expectTooltipCleared(page);

  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    "live target-selection visuals on",
  );
  expect(await targetSelectionPaintSignature(page)).toEqual(targetPaint);
  expect(await scientificSignature(page, false)).toEqual(targetScience);
  await performCommand(page, "/api/command", () => targetSelect.selectOption(""));
  await expect(targetSelect).toHaveValue("");

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
  expect(livePov.presentation_kind).toBe("live_shared_obs_agent_pov");
  await expectFilterSurface(page, disabledPair);

  const visiblePovTarget = targetSelect
    .locator("option")
    .filter({ hasText: "Agent ID 3 ·" });
  const visiblePovTargetValue = await visiblePovTarget.getAttribute("value");
  if (visiblePovTargetValue === null) {
    throw new Error("Live Agent target-filter proof has no visible ally target.");
  }
  await performCommand(page, "/api/command", () =>
    targetSelect.selectOption(visiblePovTargetValue),
  );
  await performCommand(page, "/api/command", () =>
    page.locator('#command-deck button[data-key="1"]').click(),
  );
  const visiblePovScience = await scientificSignature(page, false);
  const visiblePovPaint = await targetSelectionPaintSignature(page);
  expect(visiblePovPaint.visibleReticles).toBe(1);
  expect(visiblePovPaint.legality).toHaveLength(2);
  expect(visiblePovPaint.pendingRoute).not.toBe("");
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    "live Agent visible target visuals off",
  );
  const hiddenVisiblePovPaint = await targetSelectionPaintSignature(page);
  expect(hiddenVisiblePovPaint.visibleReticles).toBe(0);
  expect(hiddenVisiblePovPaint.legality).toEqual([]);
  expect(hiddenVisiblePovPaint.controlled).toBe(visiblePovPaint.controlled);
  expect(hiddenVisiblePovPaint.pendingRoute).toBe(visiblePovPaint.pendingRoute);
  expect(hiddenVisiblePovPaint.actionRoutes).toBe(visiblePovPaint.actionRoutes);
  expect(hiddenVisiblePovPaint.ranges).toBe(visiblePovPaint.ranges);
  expect(hiddenVisiblePovPaint.bodies).toBe(visiblePovPaint.bodies);
  expect(await scientificSignature(page, false)).toEqual(visiblePovScience);
  await expect(targetSelect).toHaveValue(visiblePovTargetValue);
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    "live Agent visible target visuals on",
  );
  expect(await targetSelectionPaintSignature(page)).toEqual(visiblePovPaint);

  const fogHiddenTarget = targetSelect
    .locator("option")
    .filter({ hasText: "Agent ID 9 ·" });
  const fogHiddenTargetValue = await fogHiddenTarget.getAttribute("value");
  if (fogHiddenTargetValue === null) {
    throw new Error("Live Agent target-filter proof has no fog-hidden target.");
  }
  await performCommand(page, "/api/command", () =>
    targetSelect.selectOption(fogHiddenTargetValue),
  );
  const fogHiddenPresentation = JSON.parse(
    await authenticatedText(page, "/api/presentation/frame"),
  );
  expect(
    fogHiddenPresentation.current_endpoint.parts.scene.agents.some(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "9",
    ),
  ).toBe(false);
  await expect(targetSelect).toHaveValue(fogHiddenTargetValue);
  const fogHiddenScience = await scientificSignature(page, false);
  const fogHiddenPaint = await targetSelectionPaintSignature(page);
  expect(fogHiddenPaint.visibleReticles).toBe(0);
  expect(fogHiddenPaint.legality).toHaveLength(2);
  expect(fogHiddenPaint.pendingRoute).toBe("");
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    false,
    "live Agent hidden target visuals off",
  );
  const filteredFogHiddenPaint = await targetSelectionPaintSignature(page);
  expect(filteredFogHiddenPaint.visibleReticles).toBe(0);
  expect(filteredFogHiddenPaint.legality).toEqual([]);
  expect(filteredFogHiddenPaint.controlled).toBe(fogHiddenPaint.controlled);
  expect(filteredFogHiddenPaint.pendingRoute).toBe(fogHiddenPaint.pendingRoute);
  expect(filteredFogHiddenPaint.actionRoutes).toBe(fogHiddenPaint.actionRoutes);
  expect(filteredFogHiddenPaint.ranges).toBe(fogHiddenPaint.ranges);
  expect(filteredFogHiddenPaint.bodies).toBe(fogHiddenPaint.bodies);
  expect(await scientificSignature(page, false)).toEqual(fogHiddenScience);
  await expect(targetSelect).toHaveValue(fogHiddenTargetValue);
  await setFilter(
    page,
    apiRequests,
    "target_selection_visuals",
    true,
    "live Agent hidden target visuals on",
  );
  expect(await targetSelectionPaintSignature(page)).toEqual(fogHiddenPaint);
  await performCommand(page, "/api/command", () => targetSelect.selectOption(""));

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
  ).toBe("live_shared_obs_agent_pov");

  await expectReplayAgentTargetFilter(
    page,
    apiRequests,
    requiredService(noSharedReplay, "NoShared replay viewer"),
    "replay_no_shared_obs_agent_pov",
    "Replay NoShared",
  );
  await expectReplayAgentTargetFilter(
    page,
    apiRequests,
    requiredService(sharedReplay, "Shared replay viewer"),
    "replay_shared_obs_agent_pov",
    "Replay Shared",
  );

  const replay = requiredService(sharedReplay, "Shared replay viewer");
  await openInstalled(page, replay.url, {
    viewerMode: "replay",
    audience: "researcher",
    presentationKind: "replay_oracle",
  });
  await expectReplayFrameIndex(page, 1);
  await openVisualFilters(page);
  await expectFilterSurface(page, ["target_selection_visuals"]);
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
  const replayDisabled = [...disabledPair, "target_selection_visuals"];
  await expectFilterSurface(page, replayDisabled);

  await performCommand(page, "/api/replay/command", () =>
    page.locator("#view-select").selectOption("pov"),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("replay_shared_obs_agent_pov");
  await expectFilterSurface(page, replayDisabled);

  const replayAgentRangesButton = page.locator("#replay-ranges-button");
  await expectLocalOnly(page, apiRequests, () => replayAgentRangesButton.click(), {
    label: "Replay Agent ranges off before recipient switch",
  });
  await expect(replayAgentRangesButton).toHaveAttribute("aria-pressed", "false");
  const recipientSwitchCursor = structuredClone(
    JSON.parse(await authenticatedText(page, "/api/frame")).cursor,
  );
  const nextReplayRecipient = page
    .locator('#roster .roster-primary-action:not([aria-pressed="true"])')
    .first();
  await expect(nextReplayRecipient).toBeEnabled();
  await performCommand(page, "/api/replay/command", () => nextReplayRecipient.click());
  expect(JSON.parse(await authenticatedText(page, "/api/frame")).cursor).toEqual(
    recipientSwitchCursor,
  );
  await expect(replayAgentRangesButton).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expectFilterSurface(page, replayDisabled, false);

  await performCommand(page, "/api/replay/command", () =>
    page.locator("#replay-frame-slider").fill("2"),
  );
  await expectReplayFrameIndex(page, 2);
  await expectFilterSurface(page, replayDisabled, false);
  await performCommand(page, "/api/replay/command", () =>
    page.locator("#replay-frame-slider").fill("1"),
  );
  await expectReplayFrameIndex(page, 1);
  await expectFilterSurface(page, replayDisabled, false);

  await performCommand(page, "/api/replay/command", () =>
    page.locator("#view-select").selectOption("researcher"),
  );
  await expect(page.locator("html")).toHaveAttribute("data-audience", "researcher");
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("replay_oracle");
  await expectFilterSurface(page, replayDisabled);

  await performCommand(page, "/api/replay/command", () =>
    page.locator("#view-select").selectOption("pov"),
  );
  await expect(replayAgentRangesButton).toHaveAttribute("aria-pressed", "false");
  await expectFilterSurface(page, replayDisabled, false);
  await performCommand(page, "/api/replay/command", () =>
    page.locator("#view-select").selectOption("researcher"),
  );
  await expect(replayAgentRangesButton).toHaveAttribute("aria-pressed", "true");
  await expectFilterSurface(page, replayDisabled);

  const bulkCursor = structuredClone(
    JSON.parse(await authenticatedText(page, "/api/frame")).cursor,
  );
  const bulkScience = await rangeIndependentScientificSignature(page, true);
  const allOffRanges = await rangeSignature(page, "#replay-ranges-button");
  const replayDisableMark = apiRequests.length;
  await disableAllFilters(page);
  expect(
    apiRequests.slice(replayDisableMark).filter(({ method }) => method === "POST"),
    "Replay Disable All range request count",
  ).toEqual([{ method: "POST", path: "/api/replay/command" }]);
  await expectFilterSurface(page, FILTER_IDS, false);
  await expectAllPaintAbsentWithoutDwell(page);
  expect(JSON.parse(await authenticatedText(page, "/api/frame")).cursor).toEqual(
    bulkCursor,
  );
  expect(await rangeIndependentScientificSignature(page, true)).toEqual(bulkScience);
  await expect(page.locator("#replay-ranges-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#disable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Disable All is unavailable.");
        }
        button.click();
      }),
    { label: "disabled Disable All idempotency" },
  );
  await expectFilterSurface(page, FILTER_IDS, false);

  const replayEnableMark = apiRequests.length;
  await performCommand(page, "/api/replay/command", () =>
    page.locator("#enable-all-visual-filters-button").click(),
  );
  expect(
    apiRequests.slice(replayEnableMark).filter(({ method }) => method === "POST"),
    "Replay Enable All range request count",
  ).toEqual([{ method: "POST", path: "/api/replay/command" }]);
  await expectFilterSurface(page);
  expect(JSON.parse(await authenticatedText(page, "/api/frame")).cursor).toEqual(
    bulkCursor,
  );
  expect(await rangeIndependentScientificSignature(page, true)).toEqual(bulkScience);
  expect(await rangeSignature(page, "#replay-ranges-button")).toEqual(allOffRanges);
  expect(await auraMarkup(page)).toBe(replayAuras);
  expect(await ultimateInventory(page)).toEqual(ultimateBefore);
  const enabledChoreographyRoots = page.locator(CHOREOGRAPHY_ROOTS);
  await expect(enabledChoreographyRoots).toHaveCount(3);
  expect(
    await enabledChoreographyRoots.evaluateAll((roots) =>
      roots.map((root) => root.getAttribute("data-paint-key")),
    ),
  ).toEqual(Array(3).fill(`visual-filters-v2:${"1".repeat(18)}`));
  await expectLocalOnly(
    page,
    apiRequests,
    () =>
      page.locator("#enable-all-visual-filters-button").evaluate((button) => {
        if (!(button instanceof HTMLButtonElement)) {
          throw new TypeError("Enable All is unavailable.");
        }
        button.click();
      }),
    { label: "post-enable Enable All idempotency" },
  );
  await expectFilterSurface(page);

  await setFilter(page, apiRequests, "aura_fields", false, "pre-reload replay aura");
  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(page, 1);
  await openVisualFilters(page);
  await expectFilterSurface(page, ["target_selection_visuals"]);
  expect(
    JSON.parse(await authenticatedText(page, "/api/presentation/frame"))
      .presentation_kind,
  ).toBe("replay_oracle");

  expect(browserErrors).toEqual([]);
});

test("real maximum-status replay renders +2 without losing owner semantics", async ({
  page,
}) => {
  test.setTimeout(120_000);
  const replay = await startReplayViewer({
    scenario: "max_status_stack",
    includeStress: true,
    frameIndex: 1,
  });

  try {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(replay.url);
    await expect(page.locator("#connection-status")).toHaveText("Online", {
      timeout: 30_000,
    });
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );

    await expect(page.locator("#battlefield .modifier-overflow")).toHaveCount(0);
    await expect(page.locator("#battlefield .modifier-cell__value")).toHaveText([
      "×1.15",
      "×0.85",
    ]);

    const presentation = JSON.parse(
      await authenticatedText(page, "/api/presentation/frame"),
    );
    const ownerAgent = presentation.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "0",
    );
    if (!ownerAgent) {
      throw new Error("The maximum-status replay has no Agent ID 0 body.");
    }
    const ownerBody = page.locator(
      `#battlefield .agent[data-presentation-key="${ownerAgent.presentation_key}"]`,
    );
    if ((await ownerBody.getAttribute("data-selected")) !== "true") {
      await performCommand(page, "/api/replay/command", () =>
        page
          .locator(
            `#roster .roster-primary-action[data-presentation-key="${ownerAgent.presentation_key}"]`,
          )
          .click(),
      );
    }
    await expect(ownerBody).toHaveAttribute("data-selected", "true");

    const overflow = page.locator("#battlefield .status-overflow").filter({
      has: page.locator(".status-overflow__label", { hasText: /^\+2$/u }),
    });
    await expect(overflow).toHaveCount(1);
    await expect(overflow.locator(".status-overflow__label")).toHaveText("+2");
    expect(await overflow.locator(".status-overflow__label").innerHTML()).toBe("+2");

    const owner = "Agent ID 0";
    const fullOwner = "Agent ID 0 · Mage · Team A";
    await expect(overflow).toHaveAttribute("data-zone", "status-overflow");
    await expect(overflow).toHaveAttribute("data-hidden-count", "2");
    await expect(overflow).toHaveAttribute("data-owner-label", owner);
    await expect(overflow).toHaveAttribute("aria-label", new RegExp(owner, "u"));
    await expect(overflow).toHaveAttribute("data-presentation-key", /^oracle_/u);

    await overflow.dispatchEvent("focusin");
    await expect(page.locator("#visual-tooltip")).toBeVisible();
    await expect(page.locator("#visual-tooltip-title")).toHaveText("2 Hidden Statuses");
    await expect(page.locator("#visual-tooltip")).toContainText(fullOwner);

    await page.setViewportSize({ width: 480, height: 360 });
    await page.evaluate(async () => {
      await document.fonts.ready;
      await new Promise((resolve) => {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            requestAnimationFrame(resolve);
          });
        });
      });
    });
    const compactStatusOverflows = page.locator(
      [
        "#battlefield .status-overflow",
        '#battlefield .required-dock-fallback[data-kind="status"]',
      ].join(", "),
    );
    await expect(compactStatusOverflows).not.toHaveCount(0);
    const compactRows = await compactStatusOverflows.evaluateAll((overflows) =>
      overflows.map((overflow) => ({
        visibleText: overflow.textContent,
        value: overflow.querySelector(
          ".status-overflow__label, .required-dock-fallback__value",
        )?.textContent,
        ownerNodes: overflow.querySelectorAll(
          ".status-overflow__owner, .required-dock-fallback__owner",
        ).length,
        ownerLabel: overflow.getAttribute("data-owner-label"),
        ariaLabel: overflow.getAttribute("aria-label"),
        presentationKey: overflow.getAttribute("data-presentation-key"),
      })),
    );
    for (const row of compactRows) {
      expect(row.value).toMatch(/^\+\d+$/u);
      expect(row.visibleText).toBe(row.value);
      expect(row.ownerNodes).toBe(0);
      expect(row.ownerLabel).toMatch(/^Agent ID /u);
      expect(row.ariaLabel).toContain(row.ownerLabel);
      expect(row.presentationKey).toMatch(/^oracle_/u);
    }

    await expect(
      page.locator('#battlefield .required-dock-fallback[data-kind="cooldown"]'),
    ).toHaveCount(0);
  } finally {
    await stopDebugger(replay.process);
  }
});

test("amended regular and stress scenarios retain Oracle-Agent presentation continuity", async ({
  page,
}) => {
  test.setTimeout(600_000);
  const browserErrors = captureBrowserErrors(page);
  const scenarios = [
    { name: "ultimate_showcase", includeStress: false },
    { name: "max_status_stack", includeStress: true },
    { name: "lifecycle_density", includeStress: true },
  ];
  const playingChoreography = [
    "#battlefield .combat-choreography[data-state=playing]",
    "#battlefield .combat-choreography-connectors[data-state=playing]",
    "#battlefield .combat-choreography-routes[data-state=playing]",
  ].join(", ");

  /**
   * Agent observation facts are serialized at their native binary32 precision.
   * Normalize Oracle numbers to that same public wire precision before comparing
   * the otherwise exact public structures.
   *
   * @param {any} value
   * @returns {any}
   */
  const atAuthorizedWirePrecision = (value) => {
    if (typeof value === "number" && Number.isFinite(value)) {
      return Math.fround(value);
    }
    if (Array.isArray(value)) {
      return value.map(atAuthorizedWirePrecision);
    }
    if (value !== null && typeof value === "object") {
      return Object.fromEntries(
        Object.entries(value).map(([key, item]) => [
          key,
          atAuthorizedWirePrecision(item),
        ]),
      );
    }
    return value;
  };

  /** @param {Record<string, any>} agent */
  const publicAgentFacts = (agent) => {
    const {
      presentation_key: _presentationKey,
      relation: _relation,
      statuses,
      ...facts
    } = agent;
    return atAuthorizedWirePrecision({
      ...facts,
      statuses: statuses.map((/** @type {Record<string, any>} */ status) => {
        const { direct_sources: _directSources, ...publicStatus } = status;
        return publicStatus;
      }),
    });
  };

  /**
   * @param {"researcher" | "agent_pov"} audience
   * @param {number} frameIndex
   * @param {Record<string, any> | null} oracle
   */
  const inspectInstalledFrame = async (audience, frameIndex, oracle) => {
    await expect(page.locator("#connection-status")).toHaveText("Online");
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
    await expect(page.locator("html")).toHaveAttribute(
      "data-submission-blocked",
      "false",
    );
    await expect(page.locator("#battlefield-empty")).toBeHidden();
    await expect(page.locator(playingChoreography)).toHaveCount(0, {
      timeout: 15_000,
    });
    const presentation = JSON.parse(
      await authenticatedText(page, "/api/presentation/frame"),
    );
    expect(presentation.source.source_frame_index).toBe(frameIndex);
    expect(presentation.presentation_kind).toBe(
      audience === "researcher" ? "replay_oracle" : "replay_no_shared_obs_agent_pov",
    );
    const scene =
      audience === "researcher"
        ? presentation.current_endpoint.scene
        : presentation.current_endpoint.parts.scene;
    const authorizedKeys = scene.agents
      .map((/** @type {Record<string, any>} */ agent) => agent.presentation_key)
      .sort();
    const bodyKeys = await page
      .locator("#battlefield .agent[data-presentation-key]")
      .evaluateAll((agents) =>
        agents.map((agent) => agent.getAttribute("data-presentation-key")).sort(),
      );
    expect(bodyKeys).toEqual(authorizedKeys);
    await expect(
      page.locator(
        "#battlefield [data-global-slot], #battlefield [data-team-local-slot]",
      ),
    ).toHaveCount(0);
    const notice = page.locator("#notice:not([hidden])");
    if ((await notice.count()) > 0) {
      expect(await notice.innerText()).not.toMatch(
        /could not process|safe fault|reconnect|no authorized battlefield scene/iu,
      );
    }

    if (audience === "agent_pov") {
      if (oracle === null) {
        throw new Error("Agent scenario smoke requires its Oracle frame.");
      }
      const oracleByPublicId = new Map(
        oracle.current_endpoint.scene.agents.map(
          (/** @type {Record<string, any>} */ agent) => [agent.public_agent_id, agent],
        ),
      );
      for (const agent of scene.agents) {
        const oracleAgent = oracleByPublicId.get(agent.public_agent_id);
        expect(oracleAgent).toBeDefined();
        expect(publicAgentFacts(agent)).toEqual(publicAgentFacts(oracleAgent));
      }
      const battlefieldMarkup = await page.locator("#battlefield").innerHTML();
      for (const oracleAgent of oracle.current_endpoint.scene.agents) {
        expect(battlefieldMarkup).not.toContain(oracleAgent.presentation_key);
      }
    }
    return presentation;
  };

  for (const scenario of scenarios) {
    const replay = await startReplayViewer({
      scenario: scenario.name,
      includeStress: scenario.includeStress,
    });
    try {
      await openInstalled(page, replay.url, {
        viewerMode: "replay",
        audience: "researcher",
        presentationKind: "replay_oracle",
      });
      await page.locator("#replay-playback-rate").selectOption("2");
      const timeline = JSON.parse(
        await authenticatedText(page, "/api/replay/timeline"),
      );
      const oracleFrames = [];
      oracleFrames.push(await inspectInstalledFrame("researcher", 0, null));
      for (
        let frameIndex = 1;
        frameIndex <= timeline.final_frame_index;
        frameIndex += 1
      ) {
        await performCommand(page, "/api/replay/command", () =>
          page.locator("#replay-next-button").click(),
        );
        await expectReplayFrameIndex(page, frameIndex);
        oracleFrames.push(await inspectInstalledFrame("researcher", frameIndex, null));
      }

      await performCommand(page, "/api/replay/command", () =>
        page.locator("#replay-frame-slider").fill("0"),
      );
      await expectReplayFrameIndex(page, 0);
      await performCommand(page, "/api/replay/command", () =>
        page.locator("#view-select").selectOption("pov"),
      );
      await inspectInstalledFrame("agent_pov", 0, oracleFrames[0]);
      for (
        let frameIndex = 1;
        frameIndex <= timeline.final_frame_index;
        frameIndex += 1
      ) {
        await performCommand(page, "/api/replay/command", () =>
          page.locator("#replay-next-button").click(),
        );
        await expectReplayFrameIndex(page, frameIndex);
        await inspectInstalledFrame("agent_pov", frameIndex, oracleFrames[frameIndex]);
      }
      await page.goto("about:blank");
    } finally {
      await stopDebugger(replay.process);
    }
  }
  expect(browserErrors).toEqual([]);
});
