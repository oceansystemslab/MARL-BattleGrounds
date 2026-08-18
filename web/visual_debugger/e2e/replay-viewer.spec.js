import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { copyFile, mkdir, readFile, rm } from "node:fs/promises";
import { basename, dirname, join } from "node:path";

import { expect, test } from "@playwright/test";
import {
  currentReplayFrame,
  currentReplayTimeline,
  expectReplayFrameIndex,
  exportReplayArtifacts,
  removeReplayArtifacts,
  startReplayViewer,
  stopDebugger,
} from "./support/replay-viewer.js";
import {
  expectVisibleInteractiveHelpInventory,
  waitForStablePresentation,
} from "./support/visual-regression.js";

test.describe.configure({ mode: "serial" });

const CP9_REPLAY_ARTIFACT_TEST_TITLE =
  "all replay authorities export canonical battlefield PNGs and preserve metric privacy";
const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const PNG_PROVENANCE_KEYWORD = "MARL-BattleGrounds Replay Provenance";
const METRIC_REPORT_ROUTE = "/api/replay/metric-report";
const REPLAY_SUFFIX = ".marlbg-replay.json";
const METRIC_SUFFIX = ".marlbg-metrics.json";
const FIXED_EXPORT_FONT_PATHS = new Set([
  "/assets/fonts/AtkinsonHyperlegible-Regular.woff2",
  "/assets/fonts/AtkinsonHyperlegible-Bold.woff2",
]);
const REPRESENTATIVE_DISABLED_FILTERS = Object.freeze([
  "aura_fields",
  "ultimate_ability_effects",
  "scrolling_battle_text",
]);
const CP9_VISUAL_FILTER_IDS = Object.freeze([
  "aura_fields",
  "aura_modifier_badges",
  "duration_status_badges",
  "spawn_shield",
  "combat_status_icon",
  "rejected_action_feedback",
  "basic_ability_effects",
  "ultimate_ability_effects",
  "damage_effects",
  "healing_effects",
  "regeneration_effects",
  "cooldown_effects",
  "charge_movement",
  "status_application",
  "status_reapplication",
  "status_refresh_extension",
  "natural_status_expiry",
  "freezing_trap_break",
  "status_clear_on_death",
  "death_effects",
  "respawn_wave",
  "resurrection_effects",
  "spawn_shield_expiry",
  "scrolling_battle_text",
]);

/** @type {Awaited<ReturnType<typeof exportReplayArtifacts>> | null} */
let artifacts = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let completeViewer = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let partialViewer = null;
/** @type {Record<string, any> | null} */
let liveFrameCandidate = null;

/** @type {WeakMap<import("@playwright/test").Page, string[]>} */
const browserErrors = new WeakMap();

test.beforeAll(async () => {
  artifacts = await exportReplayArtifacts();
  /** @type {import("node:child_process").ChildProcess[]} */
  const startedProcesses = [];
  try {
    const fixture = JSON.parse(
      readFileSync(
        new URL("../tests/fixtures/authorized-presentations-v1.json", import.meta.url),
        "utf8",
      ),
    );
    liveFrameCandidate = structuredClone(fixture.pairs.live_oracle.transport);
    completeViewer = await startReplayViewer({ replayPath: artifacts.complete });
    startedProcesses.push(completeViewer.process);
    partialViewer = await startReplayViewer({
      replayPath: artifacts.partial,
      frameIndex: 2,
    });
    startedProcesses.push(partialViewer.process);
  } catch (error) {
    const stopResults = await Promise.allSettled(
      startedProcesses.map((process) => stopDebugger(process)),
    );
    /** @type {unknown[]} */
    const cleanupErrors = stopResults.flatMap((result) =>
      result.status === "rejected" ? [result.reason] : [],
    );
    completeViewer = null;
    partialViewer = null;
    liveFrameCandidate = null;
    try {
      await removeReplayArtifacts(artifacts.outputDirectory);
    } catch (cleanupError) {
      cleanupErrors.push(cleanupError);
    }
    artifacts = null;
    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        [error, ...cleanupErrors],
        "Replay E2E startup and cleanup both failed.",
      );
    }
    throw error;
  }
});

test.afterAll(async () => {
  const processes = [completeViewer?.process ?? null, partialViewer?.process ?? null];
  completeViewer = null;
  partialViewer = null;
  liveFrameCandidate = null;
  const stopResults = await Promise.allSettled(
    processes.map((process) => stopDebugger(process)),
  );
  /** @type {unknown[]} */
  const cleanupErrors = stopResults.flatMap((result) =>
    result.status === "rejected" ? [result.reason] : [],
  );
  try {
    await removeReplayArtifacts(artifacts?.outputDirectory);
  } catch (error) {
    cleanupErrors.push(error);
  }
  artifacts = null;
  if (cleanupErrors.length > 0) {
    throw new AggregateError(cleanupErrors, "Replay E2E cleanup failed.");
  }
});

/** @param {import("@playwright/test").Page} page */
function captureBrowserErrors(page) {
  if (!browserErrors.has(page)) {
    /** @type {string[]} */
    const errors = [];
    browserErrors.set(page, errors);
    page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
    page.on("console", (message) => {
      if (message.type() === "error") {
        errors.push(`console: ${message.text()}`);
      }
    });
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 */
async function openReplay(page, url) {
  captureBrowserErrors(page);
  await page.goto(url);
  await expect(page.locator("#connection-status")).toHaveText("Online", {
    timeout: 30_000,
  });
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
  await expect(page.locator("#replay-timeline")).toBeVisible();
  expect(page.url()).not.toContain("token=");
}

/** @param {import("@playwright/test").Page} page */
function expectNoBrowserErrors(page) {
  expect(browserErrors.get(page) ?? []).toEqual([]);
}

/**
 * Chromium reports an HTTP failure at the network-console layer even when the
 * application intentionally handles the exact replay conflict response.
 *
 * @param {import("@playwright/test").Page} page
 * @param {import("@playwright/test").Response} response
 */
function expectOnlyHandledReplayConflictConsole(page, response) {
  expect(response.status()).toBe(409);
  expect(response.request().method()).toBe("POST");
  expect(new URL(response.url()).pathname).toBe("/api/replay/command");
  expect(browserErrors.get(page) ?? []).toEqual([
    "console: Failed to load resource: the server responded with a status of 409 (Conflict)",
  ]);
}

/**
 * @param {import("@playwright/test").Page} page
 */
function nextReplayResponse(page) {
  return page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/replay/command",
    { timeout: 30_000 },
  );
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} selector
 */
async function clickReplayCommand(page, selector) {
  const responsePromise = nextReplayResponse(page);
  await page.locator(selector).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  return response.json();
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} key
 */
async function pressReplayCommand(page, key) {
  await expect(page.locator("#replay-frame-slider")).toBeEnabled({
    timeout: 30_000,
  });
  const responsePromise = nextReplayResponse(page);
  await page.locator("#battlefield").focus();
  await page.keyboard.press(key);
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  return response.json();
}

/**
 * @param {Record<string, any>} frame
 * @param {Record<string, any>} timeline
 * @param {number} frameIndex
 */
function expectResearcherJoin(frame, timeline, frameIndex) {
  const episodeId = frame.artifact_summary.replay_reference.episode_id;
  const row = timeline.rows[frameIndex];
  expect(frame).toMatchObject({
    schema_version: 1,
    frame_kind: "researcher_replay_viewer",
    view_mode: "researcher",
    timeline_id: timeline.timeline_id,
    frame_id: `${episodeId}:frame:${frameIndex}`,
    cursor: {
      frame_index: frameIndex,
      final_frame_index: timeline.final_frame_index,
    },
  });
  expect(row).toMatchObject({
    frame_index: frameIndex,
    frame_id: frame.frame_id,
    simulator_step_count: frame.simulator_step_count,
    incoming_transition_id: frame.incoming_transition_id,
  });
  expect(frame.projection.scene).toMatchObject({
    episode_id: episodeId,
    frame_index: frameIndex,
    frame_id: frame.frame_id,
    simulator_step_count: frame.simulator_step_count,
    incoming_transition_id: frame.incoming_transition_id,
  });
  if (frame.incoming_transition_id === null) {
    expect(frame.projection.incoming_events).toBeNull();
  } else {
    expect(frame.projection.incoming_events.transition_id).toBe(
      frame.incoming_transition_id,
    );
  }
}

/**
 * Install frame zero through the user-facing replay keyboard boundary.
 *
 * @param {import("@playwright/test").Page} page
 */
async function installFirstFrame(page) {
  const slider = page.locator("#replay-frame-slider");
  if ((await slider.inputValue()) !== "0") {
    const result = await clickReplayCommand(page, "#replay-first-button");
    expect(result).toMatchObject({
      animate_incoming: false,
      frame: { cursor: { frame_index: 0 } },
    });
  }
  await expectReplayFrameIndex(page, 0);
}

/**
 * Prove the public presentation clock has no active replay animation. Motion
 * editing controls are intentionally absent from the product shell.
 *
 * @param {import("@playwright/test").Page} page
 */
async function expectReplayChoreographySettled(page) {
  await expect(
    page.locator("#battlefield .combat-choreography[data-state=playing]"),
  ).toHaveCount(0);
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );
}

/**
 * Freeze and prove the complete replay transport at one supported viewport.
 * The element screenshot is intentionally paired with its semantic ordering,
 * cursor, range, and overflow contract so a visually plausible but incomplete
 * transport cannot refresh the baseline.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{width: number, height: number}} viewport
 */
async function captureReplayTransportBaseline(page, viewport) {
  await page.setViewportSize(viewport);
  await waitForStablePresentation(page);

  const transport = page.locator("#replay-timeline .replay-timeline__transport");
  await expect(transport).toBeVisible();
  expect(
    await transport.locator(":scope > button").evaluateAll((buttons) =>
      buttons.map((button) => ({
        id: button.id,
        label: button.textContent?.trim(),
      })),
    ),
  ).toEqual([
    { id: "replay-first-button", label: "Start" },
    { id: "replay-back-ten-button", label: "−10" },
    { id: "replay-previous-button", label: "−1" },
    { id: "replay-play-pause-button", label: "Play" },
    { id: "replay-next-button", label: "+1" },
    { id: "replay-forward-ten-button", label: "+10" },
    { id: "replay-last-button", label: "End" },
  ]);
  await expect(page.locator("#replay-frame-slider")).toHaveAttribute("min", "0");
  await expect(page.locator("#replay-frame-slider")).toHaveAttribute("max", "5");
  await expect(page.locator("#replay-frame-slider")).toHaveValue("0");
  await expect(page.locator("#replay-frame-slider")).toBeEnabled();
  for (const selector of [
    "#replay-first-button",
    "#replay-back-ten-button",
    "#replay-previous-button",
    "#replay-play-pause-button",
    "#replay-next-button",
    "#replay-forward-ten-button",
    "#replay-last-button",
  ]) {
    await expect(page.locator(selector)).toBeEnabled();
  }
  await expect(page.locator("#replay-play-pause-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await expect(page.locator("#replay-frame-position")).toHaveText("Tick 0 / 5");
  await expect(page.locator("#replay-playback-rate")).toHaveValue("1");
  await expect(page.locator("#replay-playback-rate option")).toHaveText([
    "0.25×",
    "0.50×",
    "0.75×",
    "1.00×",
    "1.25×",
    "1.50×",
    "1.75×",
    "2.00×",
  ]);
  await expect(page.locator("#replay-transport-status")).toHaveText(
    "Frame 0 / 5 · Tick 0 / 5 · 1.00× · SETTLED",
  );
  const layout = await transport.evaluate((element) => ({
    horizontalOverflow: element.scrollWidth - element.clientWidth,
    viewportEscape: {
      left: Math.max(0, -element.getBoundingClientRect().left),
      right: Math.max(
        0,
        element.getBoundingClientRect().right - document.documentElement.clientWidth,
      ),
    },
  }));
  expect(layout.horizontalOverflow).toBeLessThanOrEqual(1);
  expect(layout.viewportEscape).toEqual({ left: 0, right: 0 });

  await expect(transport).toHaveScreenshot(
    `replay-transport-${viewport.width}x${viewport.height}.png`,
    { animations: "disabled" },
  );
}

/** @param {number} milliseconds */
function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

/**
 * @param {Awaited<ReturnType<typeof startReplayViewer>> | null} viewer
 * @param {string} name
 * @returns {Awaited<ReturnType<typeof startReplayViewer>>}
 */
function requiredViewer(viewer, name) {
  if (!viewer) {
    throw new Error(`${name} replay viewer was not started.`);
  }
  return viewer;
}

/** @returns {Record<string, any>} */
function requiredLiveFrameCandidate() {
  if (!liveFrameCandidate) {
    throw new Error("The validated live-frame reconnect candidate was not loaded.");
  }
  return liveFrameCandidate;
}

/** @param {Buffer | Uint8Array} bytes */
function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

/** @param {Buffer | Uint8Array} bytes */
function pngCrc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** @param {unknown} value @returns {string} */
function independentCanonicalJson(value) {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError(
        "Canonical replay provenance cannot contain nonfinite numbers.",
      );
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(independentCanonicalJson).join(",")}]`;
  }
  if (
    typeof value !== "object" ||
    value === null ||
    Object.getPrototypeOf(value) !== Object.prototype
  ) {
    throw new TypeError("Canonical replay provenance must contain plain JSON data.");
  }
  const record = /** @type {Record<string, unknown>} */ (value);
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${independentCanonicalJson(record[key])}`)
    .join(",")}}`;
}

/**
 * Independently validate the downloaded container rather than trusting the
 * production PNG inspector under test.
 *
 * @param {Buffer} bytes
 */
function inspectDownloadedReplayPng(bytes) {
  expect(bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)).toBe(true);
  /** @type {Array<{type: string, data: Buffer}>} */
  const chunks = [];
  let offset = PNG_SIGNATURE.length;
  while (offset < bytes.length) {
    expect(bytes.length - offset).toBeGreaterThanOrEqual(12);
    const length = bytes.readUInt32BE(offset);
    const end = offset + 12 + length;
    expect(Number.isSafeInteger(end)).toBe(true);
    expect(end).toBeLessThanOrEqual(bytes.length);
    const typeBytes = bytes.subarray(offset + 4, offset + 8);
    const type = typeBytes.toString("ascii");
    expect(type).toMatch(/^[A-Za-z]{4}$/u);
    const data = bytes.subarray(offset + 8, offset + 8 + length);
    expect(bytes.readUInt32BE(offset + 8 + length)).toBe(
      pngCrc32(Buffer.concat([typeBytes, data])),
    );
    chunks.push({ type, data });
    offset = end;
    if (type === "IEND") {
      expect(offset).toBe(bytes.length);
    }
  }
  expect(chunks[0]?.type).toBe("IHDR");
  expect(chunks.at(-1)?.type).toBe("IEND");
  expect(chunks.filter(({ type }) => type === "IHDR")).toHaveLength(1);
  expect(chunks.filter(({ type }) => type === "IEND")).toHaveLength(1);
  expect(chunks.at(-1)?.data).toHaveLength(0);
  expect(chunks.some(({ type }) => type === "IDAT")).toBe(true);
  const header = chunks[0].data;
  expect(header).toHaveLength(13);
  const width = header.readUInt32BE(0);
  const height = header.readUInt32BE(4);
  expect(width).toBeGreaterThan(0);
  expect(height).toBeGreaterThan(0);

  const textChunks = chunks.filter(({ type }) =>
    ["iTXt", "tEXt", "zTXt"].includes(type),
  );
  const provenanceChunks = textChunks.filter(({ data }) => {
    const terminator = data.indexOf(0);
    return (
      terminator >= 0 &&
      data.subarray(0, terminator).toString("latin1") === PNG_PROVENANCE_KEYWORD
    );
  });
  expect(provenanceChunks).toHaveLength(1);
  expect(chunks[1]).toBe(provenanceChunks[0]);
  expect(provenanceChunks[0].type).toBe("iTXt");
  const payload = provenanceChunks[0].data;
  const keywordLength = Buffer.byteLength(PNG_PROVENANCE_KEYWORD, "latin1");
  expect([...payload.subarray(keywordLength, keywordLength + 5)]).toEqual([
    0, 0, 0, 0, 0,
  ]);
  const canonicalJson = new TextDecoder("utf-8", { fatal: true }).decode(
    payload.subarray(keywordLength + 5),
  );
  const provenance = /** @type {Record<string, any>} */ (JSON.parse(canonicalJson));
  expect(independentCanonicalJson(provenance)).toBe(canonicalJson);
  return Object.freeze({
    width,
    height,
    chunkTypes: Object.freeze(chunks.map(({ type }) => type)),
    provenance,
    canonicalJson,
  });
}

/** @param {unknown} value @param {number} limit @param {string} fallback */
function independentSafeComponent(value, limit, fallback) {
  if (typeof value !== "string") {
    throw new TypeError("Replay filename component must be a string.");
  }
  let safe = value.replace(/[^A-Za-z0-9._-]+/gu, "-");
  safe = safe.replace(/^[._-]+|[._-]+$/gu, "").slice(0, limit);
  safe = safe.replace(/^[._-]+|[._-]+$/gu, "");
  safe = safe.replaceAll(METRIC_SUFFIX, "-").replace(/^[._-]+|[._-]+$/gu, "");
  return safe || fallback;
}

/** @param {Record<string, any>} provenance */
function expectedReplayPngFilename(provenance) {
  const episode = independentSafeComponent(provenance.source.episode_id, 64, "replay");
  const authority =
    provenance.authority.audience === "oracle"
      ? `oracle-${provenance.source.artifact_digest_sha256.slice(0, 8)}`
      : `agent-pov-${independentSafeComponent(
          provenance.authority.recipient_public_agent_id,
          64,
          "agent",
        )}`;
  const frame = String(provenance.frame.frame_index).padStart(6, "0");
  const tick = String(provenance.frame.simulator_step_count).padStart(6, "0");
  return `${episode}__${authority}__frame-${frame}__tick-${tick}__presentation.png`;
}

/** @param {import("@playwright/test").Page} page */
async function currentReplayPresentation(page) {
  return page.evaluate(async () => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Replay capability token is unavailable.");
    }
    const response = await fetch("/api/presentation/frame", {
      cache: "no-store",
      credentials: "omit",
      headers: { "X-MARL-Debugger-Token": token },
      redirect: "error",
    });
    if (!response.ok) {
      throw new Error(`/api/presentation/frame failed with HTTP ${response.status}.`);
    }
    return response.json();
  });
}

/** @param {import("@playwright/test").Page} page */
async function replayCapabilityToken(page) {
  const token = await page.evaluate(() =>
    window.sessionStorage.getItem("marl-battlegrounds.debugger-token"),
  );
  if (!token) {
    throw new Error("Replay capability token is unavailable.");
  }
  return token;
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {"researcher" | "pov"} view
 */
async function installReplayView(page, view) {
  if ((await page.locator("#view-select").inputValue()) === view) {
    return;
  }
  const responsePromise = nextReplayResponse(page);
  await page.locator("#view-select").selectOption(view);
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
}

/** @param {import("@playwright/test").Page} page */
async function waitForSettledReplayArtifactActions(page) {
  await expect(page.locator("#replay-transport-status")).toContainText("SETTLED", {
    timeout: 30_000,
  });
  await expect(page.locator("#battlefield-shell")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "data-render-policy",
    "replay_static",
  );
  await expect(page.locator("#replay-export-png-button")).toBeEnabled({
    timeout: 30_000,
  });
}

/** @param {import("@playwright/test").Page} page */
async function replayExportSnapshot(page) {
  const [presentation, transport, dom] = await Promise.all([
    currentReplayPresentation(page),
    currentReplayFrame(page),
    page.evaluate(() => {
      const battlefield = document.querySelector("#battlefield");
      const shell = document.querySelector("#battlefield-shell");
      if (!(battlefield instanceof SVGSVGElement) || !(shell instanceof HTMLElement)) {
        throw new TypeError("Replay battlefield export roots are unavailable.");
      }
      const shellStyle = getComputedStyle(shell);
      const selected = battlefield.querySelectorAll(
        '.agent[data-selected="true"][data-presentation-key]',
      );
      if (selected.length > 1) {
        throw new TypeError("Replay battlefield has more than one painted selection.");
      }
      const visualFilters = Object.fromEntries(
        [
          ...document.querySelectorAll(
            '#visual-filter-options input[type="checkbox"][data-visual-filter-id]',
          ),
        ].map((input) => {
          if (!(input instanceof HTMLInputElement)) {
            throw new TypeError("Visual filter control is not an input.");
          }
          return [input.dataset.visualFilterId, input.checked];
        }),
      );
      const ranges = document.querySelector("#replay-ranges-button");
      if (!(ranges instanceof HTMLButtonElement)) {
        throw new TypeError("Replay ranges control is unavailable.");
      }
      const battlefieldRect = battlefield.getBoundingClientRect();
      const resolvedRasterCue = (
        /** @type {Element} */ element,
        /** @type {"fill" | "stroke"} */ property,
        /** @type {string} */ label,
      ) => {
        const rect = element.getBoundingClientRect();
        const color = getComputedStyle(element)[property];
        const match = /^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/u.exec(color);
        if (!match || rect.width <= 0 || rect.height <= 0) {
          throw new TypeError(`Replay raster cue ${label} is unavailable.`);
        }
        return {
          label,
          property,
          color: match.slice(1).map(Number),
          left: rect.left - battlefieldRect.left,
          top: rect.top - battlefieldRect.top,
          width: rect.width,
          height: rect.height,
        };
      };
      const agentBodies = [...battlefield.querySelectorAll(".agent .agent-body")].slice(
        0,
        2,
      );
      if (agentBodies.length < 2) {
        throw new TypeError("Replay raster proof requires two painted agent bodies.");
      }
      const rasterCues = agentBodies.map((body, index) =>
        resolvedRasterCue(body, "stroke", `agent-body-stroke-${index}`),
      );
      if (ranges.getAttribute("aria-pressed") === "true") {
        const rangeRing = battlefield.querySelector(".range-ring");
        if (rangeRing instanceof SVGElement) {
          rasterCues.push(resolvedRasterCue(rangeRing, "stroke", "range-ring"));
        }
      }
      return {
        clientWidth: battlefield.clientWidth,
        clientHeight: battlefield.clientHeight,
        battlefieldRect: battlefieldRect.toJSON(),
        shellRect: shell.getBoundingClientRect().toJSON(),
        shellBorder: {
          top: shellStyle.borderTopWidth,
          right: shellStyle.borderRightWidth,
          bottom: shellStyle.borderBottomWidth,
          left: shellStyle.borderLeftWidth,
          color: shellStyle.borderTopColor,
        },
        shellBackground: {
          color: shellStyle.backgroundColor,
          image: shellStyle.backgroundImage,
          size: shellStyle.backgroundSize,
          rootFontSize: getComputedStyle(document.documentElement).fontSize,
        },
        selectedPresentationKey:
          selected[0]?.getAttribute("data-presentation-key") ?? null,
        showRanges: ranges.getAttribute("aria-pressed") === "true",
        visualFilters,
        rasterCues,
        battlefieldMarkup: battlefield.outerHTML,
      };
    }),
  ]);
  return { presentation, transport, dom };
}

/** @param {Awaited<ReturnType<typeof replayExportSnapshot>>} snapshot */
function expectedReplayPngProvenance(snapshot) {
  const { presentation, dom } = snapshot;
  const oracle = presentation.presentation_kind === "replay_oracle";
  const scene =
    presentation.current_endpoint.scene ??
    presentation.current_endpoint.parts?.scene ??
    null;
  const selectedAgent =
    dom.selectedPresentationKey === null
      ? null
      : scene?.agents?.find(
          (/** @type {Record<string, any>} */ agent) =>
            agent.presentation_key === dom.selectedPresentationKey,
        );
  if (dom.selectedPresentationKey !== null && !selectedAgent) {
    throw new TypeError("Painted replay selection has no authorized public identity.");
  }
  const authority = oracle
    ? { audience: "oracle" }
    : {
        audience: "agent_pov",
        observation_mode: presentation.authority.observation_mode,
        recipient_public_agent_id: presentation.authority.recipient_public_agent_id,
      };
  const source = oracle
    ? {
        episode_id: presentation.source.episode_id,
        authorized_endpoint_digest_sha256:
          presentation.source.source_authorized_endpoint_digest_sha256,
        artifact_id: presentation.source.source_artifact_id,
        replay_schema_version: presentation.source.source_replay_schema_version,
        artifact_digest_sha256: presentation.source.source_artifact_digest_sha256,
      }
    : {
        episode_id: presentation.source.episode_id,
        authorized_endpoint_digest_sha256:
          presentation.source.source_authorized_endpoint_digest_sha256,
      };
  return {
    schema_id: "marl_battlegrounds.replay_battlefield_png_provenance",
    schema_version: 1,
    product_kind: "replay_viewer",
    presentation_kind: presentation.presentation_kind,
    authority,
    source,
    frame: {
      frame_index: presentation.technical_frame.frame_index,
      simulator_step_count: presentation.technical_frame.simulator_step_count,
      incoming_transition_id: oracle
        ? presentation.technical_frame.incoming_transition_id
        : presentation.technical_frame.incoming_recipient_transition_id,
    },
    presentation: {
      render_policy: "replay_static",
      scale_factor: 2,
      css_width: dom.clientWidth,
      css_height: dom.clientHeight,
      pixel_width: dom.clientWidth * 2,
      pixel_height: dom.clientHeight * 2,
      show_ranges: dom.showRanges,
      selected_public_agent_id: selectedAgent?.public_agent_id ?? null,
      visual_filters: dom.visualFilters,
    },
  };
}

/**
 * Decode the downloaded file through Chromium and return only bounded pixel
 * evidence rather than transferring the complete RGBA buffer back to Node.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Buffer} bytes
 * @param {readonly [number, number, number]} resolvedShellBorderRgb
 * @param {Array<Record<string, any>>} rasterCues
 */
async function decodeReplayPngPixels(page, bytes, resolvedShellBorderRgb, rasterCues) {
  return page.evaluate(
    async ({ base64, shellBorderRgb, cues }) => {
      const binary = atob(base64);
      const encoded = Uint8Array.from(binary, (character) => character.charCodeAt(0));
      const bitmap = await createImageBitmap(
        new Blob([encoded], { type: "image/png" }),
      );
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext("2d", { alpha: true });
      if (!context) {
        throw new Error("PNG pixel proof requires a 2D Canvas context.");
      }
      context.drawImage(bitmap, 0, 0);
      bitmap.close();
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      let transparentPixels = 0;
      let basePixels = 0;
      let gridPixels = 0;
      let nonBackdropPixels = 0;
      let shellBorderEdgePixels = 0;
      const isRgb = (
        /** @type {number} */ offset,
        /** @type {number} */ red,
        /** @type {number} */ green,
        /** @type {number} */ blue,
        /** @type {number} */ tolerance = 0,
      ) =>
        Math.abs(pixels[offset] - red) <= tolerance &&
        Math.abs(pixels[offset + 1] - green) <= tolerance &&
        Math.abs(pixels[offset + 2] - blue) <= tolerance;
      for (let offset = 0; offset < pixels.length; offset += 4) {
        if (pixels[offset + 3] !== 255) transparentPixels += 1;
        if (isRgb(offset, 17, 24, 39)) basePixels += 1;
        if (isRgb(offset, 21, 30, 47, 1) || isRgb(offset, 25, 35, 53, 1)) {
          gridPixels += 1;
        }
        if (
          !isRgb(offset, 17, 24, 39, 1) &&
          !isRgb(offset, 21, 30, 47, 2) &&
          !isRgb(offset, 25, 35, 53, 2)
        ) {
          nonBackdropPixels += 1;
        }
      }
      const edgeOffsets = new Set();
      for (let x = 0; x < canvas.width; x += 1) {
        edgeOffsets.add(x * 4);
        edgeOffsets.add(((canvas.height - 1) * canvas.width + x) * 4);
      }
      for (let y = 0; y < canvas.height; y += 1) {
        edgeOffsets.add(y * canvas.width * 4);
        edgeOffsets.add((y * canvas.width + canvas.width - 1) * 4);
      }
      for (const offset of edgeOffsets) {
        if (isRgb(offset, shellBorderRgb[0], shellBorderRgb[1], shellBorderRgb[2])) {
          shellBorderEdgePixels += 1;
        }
      }
      const colorAt = (/** @type {number} */ x, /** @type {number} */ y) => {
        const offset = (y * canvas.width + x) * 4;
        return [...pixels.slice(offset, offset + 4)];
      };
      const cueMatches = cues.map((cue) => {
        const left = Math.max(0, Math.floor(cue.left * 2));
        const top = Math.max(0, Math.floor(cue.top * 2));
        const right = Math.min(canvas.width, Math.ceil((cue.left + cue.width) * 2));
        const bottom = Math.min(canvas.height, Math.ceil((cue.top + cue.height) * 2));
        const centerX = (cue.left + cue.width / 2) * 2;
        const centerY = (cue.top + cue.height / 2) * 2;
        const radiusX = cue.width;
        const radiusY = cue.height;
        let matchingPixels = 0;
        const quadrantMatches = [0, 0, 0, 0];
        for (let y = top; y < bottom; y += 1) {
          for (let x = left; x < right; x += 1) {
            if (cue.label === "range-ring") {
              const normalizedRadius = Math.hypot(
                (x + 0.5 - centerX) / radiusX,
                (y + 0.5 - centerY) / radiusY,
              );
              const boundaryDistance =
                Math.abs(normalizedRadius - 1) * Math.min(radiusX, radiusY);
              if (boundaryDistance > 4) continue;
            }
            const offset = (y * canvas.width + x) * 4;
            if (isRgb(offset, cue.color[0], cue.color[1], cue.color[2], 8)) {
              matchingPixels += 1;
              const quadrant =
                (y + 0.5 >= centerY ? 2 : 0) + (x + 0.5 >= centerX ? 1 : 0);
              quadrantMatches[quadrant] += 1;
            }
          }
        }
        return { label: cue.label, matchingPixels, quadrantMatches };
      });
      return {
        width: canvas.width,
        height: canvas.height,
        transparentPixels,
        basePixels,
        gridPixels,
        nonBackdropPixels,
        shellBorderEdgePixels,
        edgePixelCount: edgeOffsets.size,
        corners: [
          colorAt(0, 0),
          colorAt(canvas.width - 1, 0),
          colorAt(0, canvas.height - 1),
          colorAt(canvas.width - 1, canvas.height - 1),
        ],
        cueMatches,
      };
    },
    {
      base64: bytes.toString("base64"),
      shellBorderRgb: resolvedShellBorderRgb,
      cues: rasterCues,
    },
  );
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{label: string, expectRangeCue?: boolean}} options
 */
async function exportAndInspectReplayPng(page, options) {
  await waitForSettledReplayArtifactActions(page);
  const before = await replayExportSnapshot(page);
  expect(Object.keys(before.dom.visualFilters).sort()).toEqual(
    [...CP9_VISUAL_FILTER_IDS].sort(),
  );
  expect(before.dom.shellBorder.top).toBe("1px");
  expect(before.dom.shellBorder.right).toBe("1px");
  expect(before.dom.shellBorder.bottom).toBe("1px");
  expect(before.dom.shellBorder.left).toBe("1px");
  const shellBorderMatch = /^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/u.exec(
    before.dom.shellBorder.color,
  );
  expect(shellBorderMatch).not.toBeNull();
  const shellBorderRgb = /** @type {[number, number, number]} */ (
    shellBorderMatch?.slice(1).map(Number)
  );
  expect(before.dom.shellBackground.color).toBe("rgb(17, 24, 39)");
  expect(before.dom.shellBackground.image.match(/linear-gradient\(/gu)).toHaveLength(2);
  expect(before.dom.shellBackground.image).toContain("1px");
  expect(before.dom.shellBackground.size.split(",")[0]?.trim()).toBe("32px 32px");
  expect(before.dom.shellBackground.rootFontSize).toBe("16px");
  if (options.expectRangeCue === true) {
    expect(before.dom.rasterCues.map(({ label }) => label)).toContain("range-ring");
  }
  expect(
    Math.abs(before.dom.shellRect.width - before.dom.battlefieldRect.width - 2),
  ).toBeLessThanOrEqual(0.01);
  expect(
    Math.abs(before.dom.shellRect.height - before.dom.battlefieldRect.height - 2),
  ).toBeLessThanOrEqual(0.01);

  /** @type {Array<{method: string, path: string}>} */
  const requests = [];
  const recordRequest = (/** @type {import("@playwright/test").Request} */ request) => {
    requests.push({
      method: request.method(),
      path: new URL(request.url()).pathname,
    });
  };
  page.on("request", recordRequest);
  let download;
  try {
    const downloadPromise = page.waitForEvent("download", { timeout: 30_000 });
    await page.locator("#replay-export-png-button").click();
    download = await downloadPromise;
    await expect(page.locator("#replay-export-png-button")).toHaveAttribute(
      "aria-busy",
      "false",
    );
    await expect(page.locator("#notice")).toContainText("Exported");
  } finally {
    page.off("request", recordRequest);
  }
  const apiRequests = requests.filter(({ path }) => path.startsWith("/api/"));
  expect(apiRequests, `${options.label} scientific request ledger`).toEqual([]);
  const nonFontRequests = requests.filter(
    ({ method, path }) => method !== "GET" || !FIXED_EXPORT_FONT_PATHS.has(path),
  );
  expect(nonFontRequests, `${options.label} external request ledger`).toEqual([]);
  const downloadPath = await download.path();
  if (downloadPath === null) {
    throw new Error(`${options.label} PNG download has no local path.`);
  }
  expect(await download.failure()).toBeNull();
  const bytes = await readFile(downloadPath);
  const container = inspectDownloadedReplayPng(bytes);
  const expectedProvenance = expectedReplayPngProvenance(before);
  expect(container.provenance).toEqual(expectedProvenance);
  expect(container.canonicalJson).toBe(independentCanonicalJson(expectedProvenance));
  expect(container.width).toBe(before.dom.clientWidth * 2);
  expect(container.height).toBe(before.dom.clientHeight * 2);
  expect(container.provenance.presentation.pixel_width).toBe(container.width);
  expect(container.provenance.presentation.pixel_height).toBe(container.height);
  const expectedFilename = expectedReplayPngFilename(expectedProvenance);
  expect(download.suggestedFilename()).toBe(expectedFilename);
  expect(Buffer.byteLength(expectedFilename, "ascii")).toBeLessThanOrEqual(240);
  expect(expectedFilename).toMatch(/^[A-Za-z0-9][A-Za-z0-9._-]*\.png$/u);
  expect(expectedFilename).not.toMatch(/[\\/]/u);

  const pixelEvidence = await decodeReplayPngPixels(
    page,
    bytes,
    shellBorderRgb,
    before.dom.rasterCues,
  );
  expect(pixelEvidence.width).toBe(container.width);
  expect(pixelEvidence.height).toBe(container.height);
  expect(pixelEvidence.transparentPixels).toBe(0);
  expect(pixelEvidence.basePixels).toBeGreaterThan(100);
  expect(pixelEvidence.gridPixels).toBeGreaterThan(100);
  expect(pixelEvidence.nonBackdropPixels).toBeGreaterThan(1_000);
  expect(pixelEvidence.shellBorderEdgePixels).toBe(0);
  expect(pixelEvidence.corners).not.toContainEqual([...shellBorderRgb, 255]);
  expect(pixelEvidence.cueMatches).toHaveLength(before.dom.rasterCues.length);
  for (const cue of pixelEvidence.cueMatches) {
    const minimumMatches = cue.label === "range-ring" ? 8 : 4;
    expect(
      cue.matchingPixels,
      `${options.label} ${cue.label} raster cue`,
    ).toBeGreaterThan(minimumMatches);
    if (cue.label === "range-ring") {
      expect(
        cue.quadrantMatches.filter((count) => count > 0).length,
        `${options.label} range-ring separated arc proof`,
      ).toBeGreaterThanOrEqual(2);
    }
  }

  const after = await replayExportSnapshot(page);
  expect(after.presentation).toEqual(before.presentation);
  expect(after.transport).toEqual(before.transport);
  expect(after.dom.visualFilters).toEqual(before.dom.visualFilters);
  expect(after.dom.showRanges).toBe(before.dom.showRanges);
  expect(after.dom.selectedPresentationKey).toBe(before.dom.selectedPresentationKey);
  expect(after.dom.battlefieldMarkup).toBe(before.dom.battlefieldMarkup);
  return Object.freeze({
    label: options.label,
    filename: expectedFilename,
    bytes,
    sha256: sha256(bytes),
    requests: Object.freeze(requests),
    provenance: expectedProvenance,
    pixelEvidence,
  });
}

/** @param {import("@playwright/test").Page} page */
async function restoreAllReplayVisualFilters(page) {
  const restore = page.locator("#restore-all-visual-filters-button");
  if (await restore.isEnabled()) {
    await restore.click();
  }
  await expect(page.locator("#visual-filter-count")).toHaveText("24 enabled");
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{proveVisibleUltimate?: boolean}} [options]
 */
async function installRepresentativeReplayPresentationState(page, options = {}) {
  await page.locator("#visual-filters").evaluate((details) => {
    if (!(details instanceof HTMLDetailsElement)) {
      throw new TypeError("Replay visual-filter disclosure is unavailable.");
    }
    details.open = true;
  });
  await expect(page.locator("#visual-filter-options")).toBeVisible();
  const ultimateEffects = page.locator(
    '#battlefield .combat-effect--activation[data-component="ultimate"]',
  );
  const agentCountBefore = await page.locator("#battlefield .agent").count();
  const rosterCountBefore = await page
    .locator("#roster .roster-row--authorized")
    .count();
  let keyedUltimateEventId = null;
  if (options.proveVisibleUltimate === true) {
    expect(await ultimateEffects.count()).toBeGreaterThan(0);
    const eventIds = await ultimateEffects.evaluateAll((effects) =>
      effects.map((effect) => effect.getAttribute("data-event-id")),
    );
    expect(eventIds).toEqual(expect.arrayContaining([expect.any(String)]));
    keyedUltimateEventId = eventIds.find((eventId) => eventId !== null) ?? null;
    expect(keyedUltimateEventId).not.toBeNull();
  }
  for (const id of REPRESENTATIVE_DISABLED_FILTERS) {
    const input = page.locator(
      `#visual-filter-options input[data-visual-filter-id="${id}"]`,
    );
    await expect(input).toBeChecked();
    await input.uncheck();
  }
  await expect(page.locator("#visual-filter-count")).toHaveText("21 enabled");
  if (options.proveVisibleUltimate === true) {
    await expect(ultimateEffects).toHaveCount(0);
    expect(
      await page
        .locator('#battlefield .combat-effect--activation[data-component="ultimate"]')
        .evaluateAll(
          (effects, eventId) =>
            effects.filter((effect) => effect.getAttribute("data-event-id") === eventId)
              .length,
          keyedUltimateEventId,
        ),
    ).toBe(0);
    await expect(page.locator("#battlefield .agent")).toHaveCount(agentCountBefore);
    await expect(page.locator("#roster .roster-row--authorized")).toHaveCount(
      rosterCountBefore,
    );
    expect(agentCountBefore).toBeGreaterThan(0);
    expect(rosterCountBefore).toBeGreaterThan(0);
  }

  const ranges = page.locator("#replay-ranges-button");
  if ((await ranges.getAttribute("aria-pressed")) !== "true") {
    const audience = await page.locator("#audience-badge").innerText();
    if (audience.includes("Oracle")) {
      const responsePromise = nextReplayResponse(page);
      await ranges.click();
      expect((await responsePromise).status()).toBe(200);
    } else {
      /** @type {import("@playwright/test").Request[]} */
      const commands = [];
      const record = (/** @type {import("@playwright/test").Request} */ request) => {
        if (new URL(request.url()).pathname === "/api/replay/command") {
          commands.push(request);
        }
      };
      page.on("request", record);
      try {
        await ranges.click();
      } finally {
        page.off("request", record);
      }
      expect(commands).toEqual([]);
    }
  }
  await expect(ranges).toHaveAttribute("aria-pressed", "true");

  const bodies = page.locator('#battlefield .agent[role="button"]');
  expect(await bodies.count()).toBeGreaterThan(1);
  const selected = page.locator('#battlefield .agent[data-selected="true"]');
  const selectedKey =
    (await selected.count()) === 0
      ? null
      : await selected.getAttribute("data-presentation-key");
  const keys = await bodies.evaluateAll((nodes) =>
    nodes.map((node) => node.getAttribute("data-presentation-key")),
  );
  const nextKey = keys.find((key) => key !== null && key !== selectedKey);
  if (!nextKey) {
    throw new Error("Representative replay selection target is unavailable.");
  }
  const target = page.locator(
    `#battlefield .agent[data-presentation-key="${nextKey}"]`,
  );
  const audience = await page.locator("#audience-badge").innerText();
  if (audience.includes("Oracle")) {
    const responsePromise = nextReplayResponse(page);
    await target.click();
    expect((await responsePromise).status()).toBe(200);
  } else {
    /** @type {import("@playwright/test").Request[]} */
    const commands = [];
    const record = (/** @type {import("@playwright/test").Request} */ request) => {
      if (new URL(request.url()).pathname === "/api/replay/command") {
        commands.push(request);
      }
    };
    page.on("request", record);
    try {
      await target.click();
    } finally {
      page.off("request", record);
    }
    expect(commands).toEqual([]);
  }
  await expect(target).toHaveAttribute("data-selected", "true");
  await waitForSettledReplayArtifactActions(page);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {{
 *   url: string,
 *   view: "researcher" | "pov",
 *   presentationKind: string,
 *   label: string,
 *   repeatRepresentative?: boolean,
 * }} options
 */
async function proveReplayLeafExports(page, options) {
  await openReplay(page, options.url);
  await installReplayView(page, options.view);
  await installFirstFrame(page);
  await restoreAllReplayVisualFilters(page);
  await page.setViewportSize({ width: 960, height: 600 });
  await waitForStablePresentation(page);
  await waitForSettledReplayArtifactActions(page);
  expect((await currentReplayPresentation(page)).presentation_kind).toBe(
    options.presentationKind,
  );
  const allOn = await exportAndInspectReplayPng(page, {
    label: `${options.label} 960x600 all-on`,
  });
  expect(Object.values(allOn.provenance.presentation.visual_filters)).toEqual(
    Array.from({ length: 24 }, () => true),
  );

  const laterFrame = await clickReplayCommand(page, "#replay-next-button");
  expect(laterFrame).toMatchObject({
    animate_incoming: false,
    frame: { cursor: { frame_index: 1 } },
  });
  await expectReplayFrameIndex(page, 1);
  await waitForSettledReplayArtifactActions(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await waitForStablePresentation(page);
  await installRepresentativeReplayPresentationState(page, {
    proveVisibleUltimate: options.repeatRepresentative === true,
  });
  const representative = await exportAndInspectReplayPng(page, {
    label: `${options.label} 1440x900 representative`,
    expectRangeCue: true,
  });
  expect(representative.provenance.presentation.show_ranges).toBe(true);
  expect(representative.provenance.presentation.selected_public_agent_id).toEqual(
    expect.any(String),
  );
  for (const id of CP9_VISUAL_FILTER_IDS) {
    expect(representative.provenance.presentation.visual_filters[id]).toBe(
      !REPRESENTATIVE_DISABLED_FILTERS.includes(id),
    );
  }

  let repeated = null;
  if (options.repeatRepresentative === true) {
    repeated = await exportAndInspectReplayPng(page, {
      label: `${options.label} repeated representative`,
      expectRangeCue: true,
    });
    expect(repeated.filename).toBe(representative.filename);
    expect(repeated.sha256).toBe(representative.sha256);
    expect(repeated.bytes.equals(representative.bytes)).toBe(true);
  }
  return Object.freeze({ allOn, representative, repeated });
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} sourceMetricPath
 */
async function downloadCanonicalMetricReport(page, sourceMetricPath) {
  await waitForSettledReplayArtifactActions(page);
  await expect(page.locator("#replay-download-metrics-button")).toBeEnabled();
  /** @type {Array<{method: string, path: string}>} */
  const requests = [];
  const record = (/** @type {import("@playwright/test").Request} */ request) => {
    requests.push({ method: request.method(), path: new URL(request.url()).pathname });
  };
  page.on("request", record);
  let download;
  let response;
  try {
    const responsePromise = page.waitForResponse(
      (candidate) =>
        candidate.request().method() === "GET" &&
        new URL(candidate.url()).pathname === METRIC_REPORT_ROUTE,
      { timeout: 30_000 },
    );
    const downloadPromise = page.waitForEvent("download", { timeout: 30_000 });
    await page.locator("#replay-download-metrics-button").click();
    [response, download] = await Promise.all([responsePromise, downloadPromise]);
    await expect(page.locator("#notice")).toContainText("Downloaded");
  } finally {
    page.off("request", record);
  }
  expect(response.status()).toBe(200);
  expect(requests).toEqual([{ method: "GET", path: METRIC_REPORT_ROUTE }]);
  const path = await download.path();
  if (path === null) {
    throw new Error("Metric download has no local path.");
  }
  const [downloadedBytes, sourceBytes] = await Promise.all([
    readFile(path),
    readFile(sourceMetricPath),
  ]);
  expect(downloadedBytes.equals(sourceBytes)).toBe(true);
  expect(sha256(downloadedBytes)).toBe(sha256(sourceBytes));
  expect(download.suggestedFilename()).toMatch(
    /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\.marlbg-metrics\.json$/u,
  );
  return Object.freeze({
    filename: download.suggestedFilename(),
    sha256: sha256(downloadedBytes),
    byteLength: downloadedBytes.length,
    requests: Object.freeze(requests),
  });
}

/** @param {import("@playwright/test").Page} page */
async function proveMissingOracleMetricUi(page) {
  await waitForSettledReplayArtifactActions(page);
  const button = page.locator("#replay-download-metrics-button");
  await expect(button).toBeEnabled();
  /** @type {Array<{method: string, path: string}>} */
  const requests = [];
  let downloadCount = 0;
  const recordRequest = (/** @type {import("@playwright/test").Request} */ request) => {
    requests.push({ method: request.method(), path: new URL(request.url()).pathname });
  };
  const recordDownload = () => {
    downloadCount += 1;
  };
  const errors = browserErrors.get(page) ?? [];
  const errorOffset = errors.length;
  page.on("request", recordRequest);
  page.on("download", recordDownload);
  let response;
  try {
    const responsePromise = page.waitForResponse(
      (candidate) =>
        candidate.request().method() === "GET" &&
        new URL(candidate.url()).pathname === METRIC_REPORT_ROUTE,
      { timeout: 30_000 },
    );
    await button.click();
    response = await responsePromise;
    await expect(page.locator("#notice")).toHaveText(
      "No metric report is available for this replay.",
    );
    await expect(button).toBeEnabled();
    await page.waitForTimeout(250);
  } finally {
    page.off("request", recordRequest);
    page.off("download", recordDownload);
  }
  expect(response.status()).toBe(404);
  expect(requests).toEqual([{ method: "GET", path: METRIC_REPORT_ROUTE }]);
  expect(downloadCount).toBe(0);
  expect(errors.slice(errorOffset)).toEqual([
    "console: Failed to load resource: the server responded with a status of 404 (Not Found)",
  ]);
  errors.splice(errorOffset);
}

/** @param {import("@playwright/test").Page} page */
async function directForbiddenMetricProbe(page) {
  const token = await replayCapabilityToken(page);
  const origin = new URL(page.url()).origin;
  const response = await page.request.get(`${origin}${METRIC_REPORT_ROUTE}`, {
    failOnStatusCode: false,
    headers: {
      "X-MARL-Debugger-Token": token,
      Origin: origin,
    },
    maxRedirects: 0,
  });
  const headers = { ...response.headers() };
  delete headers.date;
  return Object.freeze({
    ledger: Object.freeze([{ method: "GET", path: METRIC_REPORT_ROUTE }]),
    status: response.status(),
    body: Buffer.from(await response.body()),
    headers: Object.freeze(headers),
  });
}

/** @param {import("@playwright/test").Page} page */
async function agentMetricDenialSurface(page) {
  const button = page.locator("#replay-download-metrics-button");
  await waitForSettledReplayArtifactActions(page);
  await expect(button).toBeDisabled();
  return Object.freeze({
    text: await button.innerText(),
    describedBy: await button.getAttribute("aria-describedby"),
    help: await page.locator("#replay-download-metrics-help").innerText(),
  });
}

test("canonical complete and partial artifacts join their frame-zero and captured endpoints", async ({
  page,
  context,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  const partial = requiredViewer(partialViewer, "partial");
  await openReplay(page, complete.url);

  const [frame, timeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  expect(timeline).toMatchObject({
    schema_version: 1,
    timeline_kind: "researcher",
    final_frame_index: 5,
    completion: { completion_state: "complete" },
  });
  expect(timeline.rows).toHaveLength(6);
  expect(timeline.rows[5].endpoint_kind).toBe("declared_horizon");
  expectResearcherJoin(frame, timeline, 0);
  expect(frame.incoming_transition_index).toBeNull();
  expect(frame.incoming_transition_id).toBeNull();
  expect(frame.artifact_summary).toMatchObject({
    expected_transition_count: 5,
    recorded_transition_count: 5,
    recorded_frame_count: 6,
    metric_report_availability: "available",
  });
  await expect(page.locator("#replay-frame-position")).toHaveText("Tick 0 / 5");
  expect(
    await page
      .locator("#replay-timeline .replay-timeline__transport")
      .evaluate((transport) =>
        [...transport.querySelectorAll(":scope > button")].map((button) => button.id),
      ),
  ).toEqual([
    "replay-first-button",
    "replay-back-ten-button",
    "replay-previous-button",
    "replay-play-pause-button",
    "replay-next-button",
    "replay-forward-ten-button",
    "replay-last-button",
  ]);
  await expect(
    page.locator("#battlefield-utilities > #replay-ranges-button"),
  ).toHaveCount(1);
  await expect(
    page.locator("#battlefield-utilities > #replay-clear-reference-button"),
  ).toHaveCount(1);
  await expect(page.locator("#replay-clear-reference-button")).toHaveText(
    "Clear Selection",
  );
  await expect(page.locator("#replay-timeline #replay-ranges-button")).toHaveCount(0);
  await expect(
    page.locator("#replay-timeline #replay-clear-reference-button"),
  ).toHaveCount(0);
  await expect(page.locator("#battlefield-shell #replay-ranges-button")).toHaveCount(0);
  await expect(page.locator("#replay-visual-key > dt")).toHaveText([
    "Team A",
    "Team B",
    "Selected agent",
  ]);
  await expect(page.locator("#live-visual-key")).toHaveAttribute("hidden", "");
  await expect(page.locator("#replay-visual-key")).not.toHaveAttribute("hidden", "");
  await expect(page.locator("#roster-details")).toHaveAttribute("open", "");
  await expect(page.locator("#events-details")).toHaveAttribute("open", "");
  await expect(page.locator("#event-feed .event-item")).toHaveCount(0);
  for (const selector of [
    "#agent-details",
    "#visual-key",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
  }
  await expect(page.locator("#replay-completion-badge")).toHaveText("Complete");
  await expect(page.locator("#replay-processing-badge")).toHaveText(
    "Authorized replay",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  const researcherHelp = await expectVisibleInteractiveHelpInventory(page);
  expect(researcherHelp.disabled.length).toBeGreaterThan(0);
  expect(researcherHelp.registered).toContain("#replay-timeline");
  await expect(page.locator("#replay-timeline")).toHaveAttribute(
    "aria-description",
    "Use the read-only transport and presentation controls to inspect recorded frames.",
  );

  const replayBoundary = await page.evaluate(() => {
    const liveRoots = [...document.querySelectorAll("[data-live-only]")];
    const focusable = [
      ...document.querySelectorAll(
        'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])',
      ),
    ].filter((element) => {
      if (!(element instanceof HTMLElement)) {
        return false;
      }
      const style = getComputedStyle(element);
      return (
        !element.hidden &&
        !element.hasAttribute("disabled") &&
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        element.getClientRects().length > 0
      );
    });
    return {
      everyLiveRootHidden: liveRoots.every(
        (element) => element instanceof HTMLElement && element.hidden,
      ),
      focusableInsideLiveRoot: focusable
        .filter((element) => element.closest("[data-live-only]") !== null)
        .map((element) => element.id || element.textContent?.trim()),
    };
  });
  expect(replayBoundary.everyLiveRootHidden).toBe(true);
  expect(replayBoundary.focusableInsideLiveRoot).toEqual([]);
  await expect(page.getByRole("button", { name: "Submit joint turn" })).toHaveCount(0);

  const frameIdentityBeforeResize = frame.frame_id;
  await page.setViewportSize({ width: 960, height: 600 });
  await page.evaluate(
    () =>
      new Promise((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(resolve));
      }),
  );
  const resized = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(resized.scrollWidth).toBeLessThanOrEqual(resized.clientWidth);
  expect((await currentReplayFrame(page)).frame_id).toBe(frameIdentityBeforeResize);
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  expect((await currentReplayFrame(page)).frame_id).toBe(frameIdentityBeforeResize);

  const partialPage = await context.newPage();
  await openReplay(partialPage, partial.url);
  const [partialFrame, partialTimeline] = await Promise.all([
    currentReplayFrame(partialPage),
    currentReplayTimeline(partialPage),
  ]);
  expect(partialTimeline).toMatchObject({
    timeline_kind: "researcher",
    final_frame_index: 2,
    completion: {
      completion_state: "partial",
      end_or_failure_reason: "browser_test_capture_stopped",
      validated_transition_count: 2,
    },
  });
  expect(partialTimeline.rows).toHaveLength(3);
  expect(partialTimeline.rows[2].endpoint_kind).toBe("captured_prefix");
  expectResearcherJoin(partialFrame, partialTimeline, 2);
  await expect(partialPage.locator("#replay-frame-position")).toHaveText("Tick 2 / 2");
  await expect(partialPage.locator("#terminal-badge")).toHaveText(
    "End of captured prefix",
  );
  await expect(partialPage.locator("#replay-completion-badge")).toHaveText("Partial");
  await expect(partialPage.locator("#replay-end-reason")).toHaveText(
    "browser_test_capture_stopped",
  );
  expectNoBrowserErrors(page);
  expectNoBrowserErrors(partialPage);
  await partialPage.close();
});

test("replay transport has permanent visual proof at the minimum supported viewport", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);

  const [frame, timeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  expectResearcherJoin(frame, timeline, 0);
  expect(timeline.final_frame_index).toBe(5);
  await captureReplayTransportBaseline(page, { width: 960, height: 600 });
  expectNoBrowserErrors(page);
});

test("replay reconnect rejects a valid live frame without crossing the viewer boundary", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  const liveFrame = requiredLiveFrameCandidate();
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  const [installedFrame, installedTimeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  expectResearcherJoin(installedFrame, installedTimeline, 0);
  expect(liveFrame).toMatchObject({
    schema_version: 2,
    frame_kind: "researcher_live_debugger",
  });
  const installedDom = {
    framePosition: await page.locator("#replay-frame-position").textContent(),
    view: await page.locator("#view-select").inputValue(),
  };

  let frameRequestCount = 0;
  let timelineRequestCount = 0;
  /** @param {import("@playwright/test").Request} request */
  const recordTimelineRequest = (request) => {
    if (
      request.method() === "GET" &&
      new URL(request.url()).pathname === "/api/replay/timeline"
    ) {
      timelineRequestCount += 1;
    }
  };
  page.on("request", recordTimelineRequest);
  await page.route("**/api/frame", async (route) => {
    frameRequestCount += 1;
    await route.fulfill({ json: structuredClone(liveFrame), status: 200 });
  });
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect(page.locator("#connection-status")).toHaveAttribute(
    "data-state",
    "offline",
  );
  await expect(page.locator("#notice")).toContainText(
    "Raw transport and presentation kinds raced between GET responses",
  );
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
  await expect(page.locator("#replay-timeline")).toBeVisible();
  await expect(page.locator("#replay-artifact-reference")).toHaveText(
    "Unavailable while authority is pending",
  );
  await expect(page.locator("#replay-frame-position")).toHaveText("Tick — / —");
  await expect(page.locator("#view-select")).toHaveValue("");
  await expect(page.locator("#view-select")).toBeDisabled();
  await expect(page.locator("[data-live-only]:not([hidden])")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Submit joint turn" })).toHaveCount(0);
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-label",
    "Read-only live battlefield. Simulator and actor activation controls are unavailable.",
  );
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  expect(frameRequestCount).toBe(2);
  expect(timelineRequestCount).toBe(0);

  await page.unroute("**/api/frame");
  page.off("request", recordTimelineRequest);
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(page, 0);
  await expect(page.locator("#replay-frame-position")).toHaveText(
    installedDom.framePosition ?? "",
  );
  await expect(page.locator("#view-select")).toHaveValue(installedDom.view);
  await expect(page.locator("#view-select")).toBeEnabled();
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expectReplayChoreographySettled(page);
  const recoveredFrame = await currentReplayFrame(page);
  expect(recoveredFrame.frame_id).toBe(installedFrame.frame_id);
  expect(recoveredFrame.timeline_id).toBe(installedTimeline.timeline_id);
  expectNoBrowserErrors(page);
});

test("one replay transport trajectory keeps static seeks, playback, rates, and races coherent", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);

  /** @type {Record<string, any>[]} */
  const replayPosts = [];
  /** @param {import("@playwright/test").Request} request */
  const recordReplayPost = (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      replayPosts.push(request.postDataJSON());
    }
  };
  page.on("request", recordReplayPost);

  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 960, height: 600 },
  ]) {
    await page.setViewportSize(viewport);
    const transport = page.locator("#replay-timeline .replay-timeline__transport");
    const bounds = await transport.evaluate((element) => ({
      horizontalOverflow: element.scrollWidth - element.clientWidth,
      left: element.getBoundingClientRect().left,
      right: element.getBoundingClientRect().right,
      viewportWidth: document.documentElement.clientWidth,
    }));
    expect(bounds.horizontalOverflow, JSON.stringify(viewport)).toBeLessThanOrEqual(1);
    expect(bounds.left, JSON.stringify(viewport)).toBeGreaterThanOrEqual(0);
    expect(bounds.right, JSON.stringify(viewport)).toBeLessThanOrEqual(
      bounds.viewportWidth,
    );
  }

  const next = await clickReplayCommand(page, "#replay-next-button");
  expect(next).toMatchObject({
    result: "applied",
    animate_incoming: false,
    frame: { cursor: { frame_index: 1 } },
  });
  expect(replayPosts.at(-1)?.command).toEqual({
    command_type: "absolute_seek",
    frame_index: 1,
  });
  await expectReplayFrameIndex(page, 1);
  await expectReplayChoreographySettled(page);
  const frameOne = await currentReplayFrame(page);
  await expect(page.locator("#replay-transport-status")).toHaveText(
    `Frame 1 / 5 · Tick 1 / 5 · Incoming transition ${frameOne.incoming_transition_id} · 1.00× · SETTLED`,
  );

  const postsBeforePreview = replayPosts.length;
  const statusBeforePreview = await page
    .locator("#replay-transport-status")
    .textContent();
  const battlefieldBeforePreview = await page.locator("#battlefield").innerHTML();
  await page.locator("#replay-frame-slider").evaluate((element) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new TypeError("Replay slider is unavailable.");
    }
    element.value = "4";
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(replayPosts).toHaveLength(postsBeforePreview);
  await expect(page.locator("#replay-frame-position")).toHaveText("Tick 4 / 5");
  await expect(page.locator("#replay-transport-status")).toHaveText(
    statusBeforePreview ?? "",
  );
  expect(await page.locator("#battlefield").innerHTML()).toBe(battlefieldBeforePreview);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );

  const seekResponsePromise = nextReplayResponse(page);
  await page.locator("#replay-frame-slider").dispatchEvent("change");
  const seekResponse = await seekResponsePromise;
  expect(seekResponse.status()).toBe(200);
  expect(replayPosts.at(-1)?.command).toEqual({
    command_type: "absolute_seek",
    frame_index: 4,
  });
  const seek = await seekResponse.json();
  expect(seek).toMatchObject({
    result: "applied",
    animate_incoming: false,
    frame: { cursor: { frame_index: 4 } },
  });
  await expectReplayFrameIndex(page, 4);
  await expectReplayChoreographySettled(page);
  const frameFour = await currentReplayFrame(page);

  const postsBeforeRate = replayPosts.length;
  await page.locator("#replay-playback-rate").selectOption("2");
  expect(replayPosts).toHaveLength(postsBeforeRate);
  await expect(page.locator("#replay-transport-status")).toHaveText(
    `Frame 4 / 5 · Tick 4 / 5 · Incoming transition ${frameFour.incoming_transition_id} · 2.00× · SETTLED`,
  );

  await page.locator("#battlefield").focus();
  await page.keyboard.press("Shift+ArrowLeft");
  expect(replayPosts).toHaveLength(postsBeforeRate);
  await page.locator("#replay-playback-rate").focus();
  await page.keyboard.press("ArrowLeft");
  expect(replayPosts).toHaveLength(postsBeforeRate);
  await page.locator("#replay-playback-rate").selectOption("2");

  const keyboardPrevious = await pressReplayCommand(page, "ArrowLeft");
  expect(keyboardPrevious).toMatchObject({
    animate_incoming: false,
    frame: { cursor: { frame_index: 3 } },
  });
  expect(replayPosts.at(-1)?.command).toEqual({
    command_type: "absolute_seek",
    frame_index: 3,
  });
  await expectReplayFrameIndex(page, 3);

  await clickReplayCommand(page, "#replay-first-button");
  await clickReplayCommand(page, "#replay-next-button");
  await expectReplayFrameIndex(page, 1);

  const postsBeforePausedReplay = replayPosts.length;
  await page.locator("#replay-play-pause-button").click();
  await expect(page.locator("#replay-transport-status")).toHaveText(
    `Frame 1 / 5 · Tick 1 / 5 · Incoming transition ${frameOne.incoming_transition_id} · 2.00× · PLAYING`,
  );
  await page.locator("#replay-play-pause-button").click();
  expect(replayPosts).toHaveLength(postsBeforePausedReplay);
  await expectReplayChoreographySettled(page);
  await expect(page.locator("#replay-transport-status")).toContainText(
    "2.00× · SETTLED",
  );

  await page.route("**/api/replay/command", async (route) => {
    await delay(250);
    await route.continue();
  });
  const continuedRequestPromise = page.waitForRequest(
    (request) =>
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command",
    { timeout: 30_000 },
  );
  const continuedResponsePromise = nextReplayResponse(page);
  await page.locator("#replay-play-pause-button").click();
  const continuedRequest = await continuedRequestPromise;
  expect(continuedRequest.postDataJSON().command).toEqual({
    command_type: "next_frame",
  });
  const cancellationFilter = page.locator('input[data-visual-filter-id="aura_fields"]');
  await cancellationFilter.evaluate((element) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new TypeError("Visual filter input is unavailable.");
    }
    element.checked = false;
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const continuedResponse = await continuedResponsePromise;
  await page.unroute("**/api/replay/command");
  expect((await continuedResponse.json()).frame.cursor.frame_index).toBe(2);
  expect(replayPosts.at(-1)?.command).toEqual({ command_type: "next_frame" });
  await expectReplayFrameIndex(page, 2);
  await expectReplayChoreographySettled(page);
  await expect(page.locator("#replay-transport-status")).toContainText(
    "2.00× · SETTLED",
  );
  await cancellationFilter.evaluate((element) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new TypeError("Visual filter input is unavailable.");
    }
    element.checked = true;
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });

  await page.route("**/api/replay/command", async (route) => {
    await delay(250);
    await route.continue();
  });
  const postsBeforeRace = replayPosts.length;
  const endResponsePromise = nextReplayResponse(page);
  await page.locator("#replay-last-button").click();
  await page.locator("#battlefield").focus();
  await page.keyboard.press("ArrowLeft");
  expect((await endResponsePromise).status()).toBe(200);
  await page.unroute("**/api/replay/command");
  expect(replayPosts).toHaveLength(postsBeforeRace + 1);
  expect(replayPosts.at(-1)?.command).toEqual({
    command_type: "absolute_seek",
    frame_index: 5,
  });
  await expectReplayFrameIndex(page, 5);
  await expectReplayChoreographySettled(page);

  const postsBeforeFinalReplay = replayPosts.length;
  await page.locator("#replay-play-pause-button").click();
  await expect(page.locator("#replay-transport-status")).toContainText(
    "2.00× · PLAYING",
  );
  await expect(page.locator("#replay-transport-status")).toContainText(
    "2.00× · SETTLED",
    { timeout: 30_000 },
  );
  expect(replayPosts).toHaveLength(postsBeforeFinalReplay);
  await expect(page.locator("#replay-play-pause-button")).toBeEnabled();

  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(page, 5);
  await expect(page.locator("#replay-playback-rate")).toHaveValue("2");
  expect(replayPosts).toHaveLength(postsBeforeFinalReplay);
  page.off("request", recordReplayPost);
  expectNoBrowserErrors(page);
});

test("a stale cross-tab command atomically installs the latest audience without retry or animation", async ({
  page,
  context,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  if ((await page.locator("#view-select").inputValue()) !== "researcher") {
    const researcherResponse = nextReplayResponse(page);
    await page.locator("#view-select").selectOption("researcher");
    await researcherResponse;
  }

  const secondPage = await context.newPage();
  await openReplay(secondPage, complete.url);
  const audienceChangeResponse = nextReplayResponse(secondPage);
  await secondPage.locator("#view-select").selectOption("pov");
  expect((await audienceChangeResponse).status()).toBe(200);
  await expect(secondPage.locator("#view-select")).toHaveValue("pov");

  let stalePostCount = 0;
  /** @param {import("@playwright/test").Request} request */
  const recordStalePost = (request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      stalePostCount += 1;
    }
  };
  page.on("request", recordStalePost);
  const staleResponsePromise = page.waitForResponse(
    (response) =>
      response.status() === 409 &&
      new URL(response.url()).pathname === "/api/replay/command",
    { timeout: 30_000 },
  );
  const matchingTimelinePromise = page.waitForResponse(
    (response) =>
      response.status() === 200 &&
      response.request().method() === "GET" &&
      new URL(response.url()).pathname === "/api/replay/timeline",
    { timeout: 30_000 },
  );
  await page.locator("#replay-next-button").click();
  const staleResponse = await staleResponsePromise;
  const matchingTimelineResponse = await matchingTimelinePromise;
  const [stalePayload, matchingTimeline] = await Promise.all([
    staleResponse.json(),
    matchingTimelineResponse.json(),
  ]);
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expect(page.locator("#notice")).toContainText(
    "coherent latest pair was installed; the command was not retried",
  );
  await expect(page.locator("#replay-play-pause-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await expectReplayChoreographySettled(page);
  expect(stalePayload.error_code).toBe("stale_revision");
  expect(stalePayload).not.toHaveProperty("animate_incoming");
  expect(matchingTimeline.timeline_id).toBe(stalePayload.latest_frame.timeline_id);
  expect(stalePostCount).toBe(1);
  page.off("request", recordStalePost);

  const [latestFrame, latestTimeline] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
  ]);
  expect(latestFrame.frame_kind).toBe("actor_pov_replay_viewer");
  expect(latestTimeline.timeline_kind).toBe("actor_pov");
  expect(latestTimeline.timeline_id).toBe(latestFrame.timeline_id);

  const restoreResponse = nextReplayResponse(page);
  await page.locator("#view-select").selectOption("researcher");
  expect((await restoreResponse).status()).toBe(200);
  await expect(page.locator("#view-select")).toHaveValue("researcher");
  expectOnlyHandledReplayConflictConsole(page, staleResponse);
  expectNoBrowserErrors(secondPage);
  await secondPage.close();
});

test("accessible playback pauses on hidden/error/endpoint and keeps one request in flight", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  await installFirstFrame(page);
  const timeline = page.locator("#replay-timeline");
  const play = page.locator("#replay-play-pause-button");

  await clickReplayCommand(page, "#replay-next-button");
  await expectReplayFrameIndex(page, 1);

  await timeline.focus();
  await page.keyboard.press("Space");
  await expect(play).toHaveAttribute("aria-label", "Pause replay");
  await expect(play).toHaveAttribute("aria-pressed", "true");
  await page.keyboard.press("Space");
  await expect(play).toHaveAttribute("aria-label", "Play replay");
  await expect(play).toHaveAttribute("aria-pressed", "false");

  await timeline.focus();
  await page.keyboard.press("Space");
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      get: () => true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(play).toHaveAttribute("aria-pressed", "false");
  await page.evaluate(() => {
    Reflect.deleteProperty(document, "hidden");
    document.dispatchEvent(new Event("visibilitychange"));
  });

  await clickReplayCommand(page, "#replay-first-button");
  await expectReplayFrameIndex(page, 0);

  await page.route("**/api/replay/command", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "{",
    });
  });
  await page.locator("#replay-next-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Resync required", {
    timeout: 10_000,
  });
  await expect(page.locator("#connection-status")).toHaveAttribute(
    "data-state",
    "offline",
  );
  await expect(play).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator("#notice")).toContainText(
    "Reconnect before sending another replay command",
  );
  await page.unroute("**/api/replay/command");
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expectReplayFrameIndex(page, 0);

  let activeRequests = 0;
  let maximumActiveRequests = 0;
  /** @type {Record<string, any>[]} */
  const playbackRequests = [];
  /** @type {Record<string, any>[]} */
  const playbackResponses = [];
  await page.route("**/api/replay/command", async (route) => {
    activeRequests += 1;
    maximumActiveRequests = Math.max(maximumActiveRequests, activeRequests);
    playbackRequests.push(route.request().postDataJSON());
    await delay(225);
    const response = await route.fetch();
    playbackResponses.push(await response.json());
    activeRequests -= 1;
    await route.fulfill({ response });
  });
  await play.click();
  await expect(play).toHaveAttribute("aria-label", "Pause replay");
  await expectReplayFrameIndex(page, 5);
  await expect(play).toHaveAttribute("aria-label", "Play replay");
  await expect(play).toHaveAttribute("aria-pressed", "false");
  await expect(play).toBeEnabled();
  await expect(page.locator("#terminal-badge")).toHaveText("Declared horizon");
  await page.unroute("**/api/replay/command");

  expect(maximumActiveRequests).toBe(1);
  expect(playbackRequests).toHaveLength(5);
  expect(playbackRequests.map((request) => request.command.command_type)).toEqual(
    Array(5).fill("next_frame"),
  );
  expect(playbackResponses).toHaveLength(5);
  expect(playbackResponses.every((response) => response.animate_incoming)).toBe(true);
  expectNoBrowserErrors(page);
});

test("Actor POV all-surface scan excludes researcher authority and host secrets", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  const artifactManifest = artifacts;
  if (!artifactManifest) {
    throw new Error("Replay artifacts are unavailable for the POV isolation scan.");
  }

  await openReplay(page, complete.url);
  await installFirstFrame(page);
  if ((await page.locator("#view-select").inputValue()) !== "researcher") {
    const researcherResponse = nextReplayResponse(page);
    await page.locator("#view-select").selectOption("researcher");
    expect((await researcherResponse).status()).toBe(200);
  }
  const next = await clickReplayCommand(page, "#replay-next-button");
  expect(next.frame.cursor.frame_index).toBe(1);
  await expectReplayFrameIndex(page, 1);
  await expectReplayChoreographySettled(page);
  const researcherFrame = await currentReplayFrame(page);
  expect(researcherFrame.frame_kind).toBe("researcher_replay_viewer");

  const povResponse = nextReplayResponse(page);
  await page.locator("#view-select").selectOption("pov");
  expect((await povResponse).status()).toBe(200);
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expectReplayFrameIndex(page, 1);
  await expect(page.locator("#preset-select")).toHaveCount(0);
  await expect(page.locator("html")).toHaveAttribute("data-preset", "analysis");
  await page.locator("#technical-frame-details").evaluate((details) => {
    if (!(details instanceof HTMLDetailsElement)) {
      throw new TypeError("Technical frame details are unavailable.");
    }
    details.open = true;
  });

  const [povFrame, povTimeline, capabilityToken] = await Promise.all([
    currentReplayFrame(page),
    currentReplayTimeline(page),
    page.evaluate(() =>
      window.sessionStorage.getItem("marl-battlegrounds.debugger-token"),
    ),
  ]);
  expect(povFrame.frame_kind).toBe("actor_pov_replay_viewer");
  expect(povTimeline.timeline_kind).toBe("actor_pov");
  expect(capabilityToken).toMatch(/^[A-Za-z0-9_-]+$/u);

  /** @param {unknown} value @param {string} field @returns {unknown[]} */
  const recursiveFieldValues = (value, field) => {
    /** @type {unknown[]} */
    const found = [];
    /** @param {unknown} candidate */
    const visit = (candidate) => {
      if (Array.isArray(candidate)) {
        candidate.forEach(visit);
        return;
      }
      if (!candidate || typeof candidate !== "object") {
        return;
      }
      for (const [key, child] of Object.entries(candidate)) {
        if (key === field) {
          found.push(child);
        }
        visit(child);
      }
    };
    visit(value);
    return found;
  };
  /** @param {unknown} value @returns {Set<string>} */
  const recursiveKeys = (value) => {
    /** @type {Set<string>} */
    const found = new Set();
    /** @param {unknown} candidate */
    const visit = (candidate) => {
      if (Array.isArray(candidate)) {
        candidate.forEach(visit);
        return;
      }
      if (!candidate || typeof candidate !== "object") {
        return;
      }
      for (const [key, child] of Object.entries(candidate)) {
        found.add(key);
        visit(child);
      }
    };
    visit(value);
    return found;
  };

  const authorizedPublicIds = new Set(
    recursiveFieldValues([povFrame, povTimeline], "public_agent_id").filter(
      (value) => typeof value === "string",
    ),
  );
  expect(authorizedPublicIds.has(povFrame.public_agent_id)).toBe(true);
  /** @type {Array<Record<string, any>>} */
  const researcherAgents = researcherFrame.projection.scene.agents;
  const hiddenAgents = researcherAgents.filter(
    (agent) => !authorizedPublicIds.has(agent.public_agent_id),
  );
  expect(hiddenAgents.length).toBeGreaterThan(0);
  const hiddenSlots = new Set(hiddenAgents.map((agent) => agent.global_slot));
  const hiddenPublicIds = hiddenAgents.map((agent) => agent.public_agent_id);
  const researcherEventIds = recursiveFieldValues(
    researcherFrame.projection.incoming_events,
    "event_id",
  ).filter((value) => typeof value === "string");
  expect(researcherEventIds.length).toBeGreaterThan(0);

  const selfRoster = page.locator("#roster .roster-primary-action").first();
  await expect(selfRoster).toBeVisible();
  await selfRoster.hover();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  const tooltipSnapshot = await page.locator("#visual-tooltip").evaluate((root) => ({
    text: root.textContent ?? "",
    attributes: [...root.querySelectorAll("*")].flatMap((element) =>
      [...element.attributes].map((attribute) => ({
        name: attribute.name,
        value: attribute.value,
      })),
    ),
  }));
  const selfAgent = page.locator("#battlefield .agent").first();
  await expect(page.locator('#battlefield .agent[role="button"]')).toHaveCount(
    await page.locator("#battlefield .agent").count(),
  );
  await expect(selfAgent).toHaveAttribute("role", "button");
  await expect(selfAgent).toHaveAttribute("tabindex", "0");
  expect(await selfAgent.getAttribute("data-slot")).toBeNull();
  await expect(selfAgent).toHaveAttribute("data-presentation-key", /.+/u);
  await selfAgent.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#agent-details")).toHaveAttribute("open", "");
  await expect(page.locator("#event-feed .event-item")).not.toHaveCount(0);

  const surfaces = await page.evaluate(() => {
    /** @param {Element} element */
    const isVisible = (element) => {
      if (element.closest("[hidden], [aria-hidden='true']")) return false;
      const style = getComputedStyle(element);
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        element.getClientRects().length > 0
      );
    };
    /** @param {Element} root */
    const snapshot = (root) => ({
      text: root.textContent ?? "",
      attributes: [root, ...root.querySelectorAll("*")].flatMap((element) =>
        [...element.attributes].map((attribute) => ({
          name: attribute.name,
          value: attribute.value,
        })),
      ),
    });
    /** @param {string} selector @returns {Element} */
    const requiredRoot = (selector) => {
      const root = document.querySelector(selector);
      if (!root) {
        throw new Error(`Required isolation surface is missing: ${selector}`);
      }
      return root;
    };
    const visibleElements = [
      document.body,
      ...document.body.querySelectorAll("*"),
    ].filter(isVisible);
    const visibleAttributes = visibleElements.flatMap((element) =>
      [...element.attributes].map((attribute) => ({
        name: attribute.name,
        value: attribute.value,
      })),
    );
    const descriptorText = visibleElements
      .filter((element) => element.hasAttribute("data-tooltip-owner"))
      .flatMap((element) => [
        element.getAttribute("aria-label") ?? "",
        element.getAttribute("aria-description") ?? "",
      ])
      .join("\n");
    const technicalRoot = requiredRoot("#diagnostics-card");
    return {
      normalText: document.body.innerText,
      visibleAttributes,
      descriptorText,
      feed: snapshot(requiredRoot("#event-feed")),
      inspector: snapshot(requiredRoot("#agent-details")),
      technical: snapshot(technicalRoot),
      technicalFacts: [
        ...technicalRoot.querySelectorAll(".fact[data-technical-fact]"),
      ].map((node) => ({
        id: node.getAttribute("data-technical-fact"),
        label: node.querySelector("span")?.textContent ?? "",
        value: node.querySelector("strong")?.textContent ?? "",
      })),
      technicalJson: [...document.querySelectorAll(".technical-json")].map(
        (node) => node.textContent ?? "",
      ),
    };
  });

  const allAttributes = [
    ...surfaces.visibleAttributes,
    ...tooltipSnapshot.attributes,
    ...surfaces.feed.attributes,
    ...surfaces.inspector.attributes,
    ...surfaces.technical.attributes,
  ];
  const semanticSurfaceText = [
    surfaces.normalText,
    surfaces.descriptorText,
    tooltipSnapshot.text,
    surfaces.feed.text,
    surfaces.inspector.text,
    surfaces.technical.text,
    ...allAttributes
      .filter(({ name }) => name.startsWith("aria-") || name === "title")
      .map(({ value }) => value),
  ].join("\n");
  const completeSurfaceBytes = JSON.stringify({
    surfaces,
    tooltipSnapshot,
  });
  const povWireBytes = JSON.stringify([povFrame, povTimeline]);

  for (const authorizedId of authorizedPublicIds) {
    if (authorizedId === povFrame.public_agent_id) {
      expect(completeSurfaceBytes).toContain(authorizedId);
    }
  }
  for (const forbiddenValue of [
    ...hiddenPublicIds,
    ...researcherEventIds,
    artifactManifest.outputDirectory,
    artifactManifest.complete,
    artifactManifest.missingMetric,
    capabilityToken,
  ]) {
    if (typeof forbiddenValue === "string" && forbiddenValue.length > 0) {
      expect(completeSurfaceBytes).not.toContain(forbiddenValue);
      expect(povWireBytes).not.toContain(forbiddenValue);
    }
  }
  for (const slot of hiddenSlots) {
    expect(semanticSurfaceText).not.toMatch(
      new RegExp(`\\b(?:global[ _-]?slot|slot|id_)\\s*[:#= -]?\\s*${slot}\\b`, "iu"),
    );
  }
  const hiddenSlotAttributes = allAttributes.filter(
    ({ name, value }) =>
      /slot/iu.test(name) && /^\d+$/u.test(value) && hiddenSlots.has(Number(value)),
  );
  expect(hiddenSlotAttributes).toEqual([]);

  for (const researcherOnlyPhrase of [
    "PRIVILEGED RESEARCHER",
    "Exact Class Mechanics",
    "Observer visibility",
    "Status source evidence",
    "Researcher selection",
  ]) {
    expect(semanticSurfaceText).not.toContain(researcherOnlyPhrase);
  }
  const actorKeys = recursiveKeys([povFrame, povTimeline]);
  for (const researcherOnlyKey of [
    "agent_phase_trajectories",
    "aura_fields",
    "class_mechanics",
    "configured_active_by_global_slot",
    "event_id",
    "incoming_event_ids",
    "metric_report_reference",
    "observer_visibility",
    "processing",
    "public_agent_id_by_global_slot",
    "status_source_evidence",
  ]) {
    expect(actorKeys.has(researcherOnlyKey), researcherOnlyKey).toBe(false);
  }
  expect(surfaces.technicalFacts).toEqual([
    {
      id: "frame",
      label: "Frame",
      value: String(povFrame.cursor.frame_index),
    },
    {
      id: "simulator_step",
      label: "Simulator step",
      value: String(povFrame.simulator_step_count),
    },
  ]);
  expect(completeSurfaceBytes).not.toContain("technical_kind");
  expect(completeSurfaceBytes).not.toContain("replay_no_shared_obs_technical_frame");
  expect(surfaces.normalText).toContain("Agent POV");
  expect(completeSurfaceBytes).not.toContain("actor_pov_replay_viewer");
  expect(surfaces.technicalJson).toEqual([]);
  expect(completeSurfaceBytes).not.toMatch(/(?:token|secret|password)\s*[:=]/iu);
  expectNoBrowserErrors(page);
});

test(CP9_REPLAY_ARTIFACT_TEST_TITLE, async ({ page }) => {
  const replayArtifacts = artifacts;
  if (!replayArtifacts) {
    throw new Error("CP9 replay artifacts are unavailable.");
  }
  const complete = requiredViewer(completeViewer, "complete");
  if (!replayArtifacts.complete.endsWith(REPLAY_SUFFIX)) {
    throw new Error("Complete replay path does not use the canonical suffix.");
  }
  const sourceMetricPath = `${replayArtifacts.complete.slice(
    0,
    -REPLAY_SUFFIX.length,
  )}${METRIC_SUFFIX}`;
  const sharedMissingDirectory = join(
    dirname(replayArtifacts.shared),
    "shared-missing-sidecar",
  );
  const sharedMissingPath = join(
    sharedMissingDirectory,
    basename(replayArtifacts.shared),
  );
  /** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
  let missingViewer = null;
  /** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
  let sharedViewer = null;
  /** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
  let sharedMissingViewer = null;
  /** @type {unknown} */
  let testError = null;

  try {
    await mkdir(sharedMissingDirectory, { recursive: false });
    await copyFile(replayArtifacts.shared, sharedMissingPath);
    missingViewer = await startReplayViewer({
      replayPath: replayArtifacts.missingMetric,
    });
    sharedViewer = await startReplayViewer({
      replayPath: replayArtifacts.shared,
      view: "pov",
      povSlot: 0,
    });
    sharedMissingViewer = await startReplayViewer({
      replayPath: sharedMissingPath,
      view: "pov",
      povSlot: 0,
    });

    const oracleExports = await proveReplayLeafExports(page, {
      url: complete.url,
      view: "researcher",
      presentationKind: "replay_oracle",
      label: "Replay Oracle",
      repeatRepresentative: true,
    });
    expect(oracleExports.allOn.provenance.authority).toEqual({
      audience: "oracle",
    });
    expect(Object.keys(oracleExports.allOn.provenance.source).sort()).toEqual(
      [
        "artifact_digest_sha256",
        "artifact_id",
        "authorized_endpoint_digest_sha256",
        "episode_id",
        "replay_schema_version",
      ].sort(),
    );
    const metricDownload = await downloadCanonicalMetricReport(page, sourceMetricPath);
    expect(metricDownload.requests).toEqual([
      { method: "GET", path: METRIC_REPORT_ROUTE },
    ]);

    await openReplay(page, missingViewer.url);
    await installReplayView(page, "researcher");
    await installFirstFrame(page);
    await proveMissingOracleMetricUi(page);
    await installReplayView(page, "pov");
    const noSharedMissingSurface = await agentMetricDenialSurface(page);
    const noSharedMissingProbe = await directForbiddenMetricProbe(page);

    const noSharedExports = await proveReplayLeafExports(page, {
      url: complete.url,
      view: "pov",
      presentationKind: "replay_no_shared_obs_agent_pov",
      label: "Replay NoShared Agent POV",
    });
    for (const artifact of [noSharedExports.allOn, noSharedExports.representative]) {
      expect(Object.keys(artifact.provenance.source).sort()).toEqual(
        ["authorized_endpoint_digest_sha256", "episode_id"].sort(),
      );
      expect(artifact.provenance.authority.audience).toBe("agent_pov");
      expect(artifact.provenance.authority.observation_mode).toBe("no_shared_obs");
      expect(artifact.filename).not.toMatch(/[0-9a-f]{8}__frame-/u);
    }
    const noSharedPresentSurface = await agentMetricDenialSurface(page);
    const noSharedPresentProbe = await directForbiddenMetricProbe(page);

    const sharedExports = await proveReplayLeafExports(page, {
      url: sharedViewer.url,
      view: "pov",
      presentationKind: "replay_shared_obs_agent_pov",
      label: "Replay Shared Agent POV",
    });
    for (const artifact of [sharedExports.allOn, sharedExports.representative]) {
      expect(Object.keys(artifact.provenance.source).sort()).toEqual(
        ["authorized_endpoint_digest_sha256", "episode_id"].sort(),
      );
      expect(artifact.provenance.authority).toMatchObject({
        audience: "agent_pov",
        observation_mode: "shared_obs_visual_union",
      });
      expect(artifact.filename).not.toMatch(/[0-9a-f]{8}__frame-/u);
    }
    const sharedPresentSurface = await agentMetricDenialSurface(page);
    const sharedPresentProbe = await directForbiddenMetricProbe(page);

    await openReplay(page, sharedMissingViewer.url);
    await installFirstFrame(page);
    const sharedMissingSurface = await agentMetricDenialSurface(page);
    const sharedMissingProbe = await directForbiddenMetricProbe(page);

    expect(noSharedMissingSurface).toEqual(noSharedPresentSurface);
    expect(sharedMissingSurface).toEqual(sharedPresentSurface);
    expect(sharedPresentSurface).toEqual(noSharedPresentSurface);
    const denialProbes = [
      noSharedPresentProbe,
      noSharedMissingProbe,
      sharedPresentProbe,
      sharedMissingProbe,
    ];
    for (const probe of denialProbes) {
      expect(probe.ledger).toEqual([{ method: "GET", path: METRIC_REPORT_ROUTE }]);
      expect(probe.status).toBe(403);
    }
    for (const probe of denialProbes.slice(1)) {
      expect(probe.body.equals(denialProbes[0].body)).toBe(true);
      expect(probe.headers).toEqual(denialProbes[0].headers);
    }
    expectNoBrowserErrors(page);
  } catch (error) {
    testError = error;
  }

  await page.goto("about:blank").catch(() => {});
  const cleanupResults = await Promise.allSettled([
    stopDebugger(missingViewer?.process ?? null),
    stopDebugger(sharedViewer?.process ?? null),
    stopDebugger(sharedMissingViewer?.process ?? null),
    rm(sharedMissingDirectory, { force: true, recursive: true }),
  ]);
  const cleanupErrors = cleanupResults.flatMap((result) =>
    result.status === "rejected" ? [result.reason] : [],
  );
  if (testError !== null || cleanupErrors.length > 0) {
    throw new AggregateError(
      [...(testError === null ? [] : [testError]), ...cleanupErrors],
      "CP9 replay export/metric proof or runtime-artifact cleanup failed.",
    );
  }
});

test("Exit flushes its replay response before clean server shutdown", async ({
  page,
}) => {
  const complete = requiredViewer(completeViewer, "complete");
  await openReplay(page, complete.url);
  const exitPromise = new Promise((resolve, reject) => {
    const process = complete.process;
    if (process.exitCode !== null || process.signalCode !== null) {
      resolve(undefined);
      return;
    }
    const timeout = setTimeout(
      () => reject(new Error("Replay viewer did not exit after its exit response.")),
      5_000,
    );
    process.once("exit", () => {
      clearTimeout(timeout);
      resolve(undefined);
    });
  });
  const response = await clickReplayCommand(page, "#exit-button");
  expect(response).toMatchObject({
    result: "shutdown_scheduled",
    animate_incoming: false,
  });
  await exitPromise;
  expect(complete.process.exitCode).toBe(0);
  expect(complete.process.signalCode).toBeNull();
});
