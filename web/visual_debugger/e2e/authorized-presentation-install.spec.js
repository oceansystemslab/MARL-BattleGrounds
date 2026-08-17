import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";

import { expect, test } from "@playwright/test";

import {
  REPOSITORY_ROOT,
  startDebugger,
  stopDebugger,
} from "./support/live-debugger.js";
import {
  exportReplayArtifacts,
  removeReplayArtifacts,
  startReplayViewer,
} from "./support/replay-viewer.js";

test.describe.configure({ mode: "serial" });

const CP4_E_CAPTURE_FILENAMES = Object.freeze([
  "live-oracle-1440x900.png",
  "live-no-shared-agent-960x600.png",
  "replay-oracle-960x600.png",
  "replay-shared-agent-1440x900.png",
]);
const CP4_E_OWNER_PATHS = Object.freeze([
  "web/visual_debugger/e2e/authorized-presentation-install.spec.js",
  "web/visual_debugger/e2e/authorized-presentation-renderer.spec.js",
  "web/visual_debugger/index.html",
  "web/visual_debugger/src/authorized-presentation-adapter.js",
  "web/visual_debugger/src/authorized-presentation-schema.js",
  "web/visual_debugger/src/explanations.js",
  "web/visual_debugger/src/main.js",
  "web/visual_debugger/src/panels.js",
  "web/visual_debugger/src/scene.js",
  "web/visual_debugger/src/tooltip.js",
  "web/visual_debugger/tests/fixtures/authorized-presentations-v1.json",
]);
const requestedCaptureDirectory = process.env.MARL_CP4_E_CAPTURE_DIR?.trim() || null;
const cp4ECaptureDirectory =
  requestedCaptureDirectory === null ? null : resolve(requestedCaptureDirectory);
if (
  requestedCaptureDirectory !== null &&
  (!isAbsolute(requestedCaptureDirectory) ||
    cp4ECaptureDirectory === null ||
    !cp4ECaptureDirectory.startsWith("/tmp/m6-cp4-e-") ||
    cp4ECaptureDirectory === "/tmp/m6-cp4-e-")
) {
  throw new TypeError(
    "MARL_CP4_E_CAPTURE_DIR must resolve to a uniquely named /tmp/m6-cp4-e-* directory.",
  );
}
const cp4C3ShieldOnly = process.env.MARL_CP4_C3_SHIELD_ONLY === "1";

/** @type {Array<Record<string, unknown>>} */
const cp4ENativeCaptures = [];
/** @type {WeakMap<import("@playwright/test").Page, Array<Record<string, unknown>>>} */
const evidenceRequests = new WeakMap();
/** @type {WeakMap<import("@playwright/test").Page, number>} */
const evidenceRequestOffsets = new WeakMap();
/** @type {string[]} */
const cp4EBrowserErrors = [];
/** @type {Array<Record<string, unknown>>} */
const cp4ENetworkFailures = [];

/** @type {Awaited<ReturnType<typeof exportReplayArtifacts>> | null} */
let artifacts = null;
/** @type {Awaited<ReturnType<typeof startDebugger>> | null} */
let liveDebugger = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let noSharedReplay = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let sharedReplay = null;
/** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
let deathReplay = null;

/** @type {WeakMap<import("@playwright/test").Page, string[]>} */
const browserErrors = new WeakMap();

test.beforeAll(async () => {
  /** @type {import("node:child_process").ChildProcess[]} */
  const startedProcesses = [];
  try {
    if (cp4C3ShieldOnly) {
      deathReplay = await startReplayViewer({
        sampleReplay: "death-respawn-shield",
      });
      startedProcesses.push(deathReplay.process);
    } else {
      artifacts = await exportReplayArtifacts();
      liveDebugger = await startDebugger();
      startedProcesses.push(liveDebugger.process);
      noSharedReplay = await startReplayViewer({ replayPath: artifacts.complete });
      startedProcesses.push(noSharedReplay.process);
      sharedReplay = await startReplayViewer({
        replayPath: artifacts.shared,
        view: "pov",
        povSlot: 0,
      });
      startedProcesses.push(sharedReplay.process);
      if (cp4ECaptureDirectory === null) {
        deathReplay = await startReplayViewer({
          sampleReplay: "death-respawn-shield",
        });
        startedProcesses.push(deathReplay.process);
      }
    }
    if (cp4ECaptureDirectory !== null) {
      await mkdir(cp4ECaptureDirectory, { recursive: false });
    }
  } catch (error) {
    await Promise.allSettled(startedProcesses.map((child) => stopDebugger(child)));
    await removeReplayArtifacts(artifacts?.outputDirectory);
    artifacts = null;
    throw error;
  }
});

test.afterAll(async () => {
  const processes = [
    liveDebugger?.process ?? null,
    noSharedReplay?.process ?? null,
    sharedReplay?.process ?? null,
    deathReplay?.process ?? null,
  ];
  liveDebugger = null;
  noSharedReplay = null;
  sharedReplay = null;
  deathReplay = null;
  const stopResults = await Promise.allSettled(
    processes.map((child) => stopDebugger(child)),
  );
  const cleanupErrors = stopResults.flatMap((result) =>
    result.status === "rejected" ? [result.reason] : [],
  );
  try {
    await removeReplayArtifacts(artifacts?.outputDirectory);
  } catch (error) {
    cleanupErrors.push(error);
  }
  artifacts = null;
  if (cp4ECaptureDirectory !== null) {
    try {
      await writeCp4EEvidenceReport(cp4ECaptureDirectory);
    } catch (error) {
      cleanupErrors.push(error);
    }
  }
  if (cleanupErrors.length > 0) {
    throw new AggregateError(
      cleanupErrors,
      "Authorized-presentation E2E cleanup failed.",
    );
  }
});

/** @param {import("@playwright/test").Page} page */
function captureBrowserErrors(page) {
  if (browserErrors.has(page)) {
    return;
  }
  /** @type {string[]} */
  const errors = [];
  browserErrors.set(page, errors);
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
  if (cp4ECaptureDirectory !== null) {
    page.on("pageerror", (error) =>
      cp4EBrowserErrors.push(`pageerror: ${error.message}`),
    );
    page.on("console", (message) => {
      if (message.type() === "error") {
        cp4EBrowserErrors.push(`console: ${message.text()}`);
      }
    });
    page.on("requestfailed", (request) => {
      cp4ENetworkFailures.push({
        kind: "requestfailed",
        method: request.method(),
        path: new URL(request.url()).pathname,
        error: request.failure()?.errorText ?? "unknown",
      });
    });
    page.on("response", (response) => {
      if (!response.ok()) {
        cp4ENetworkFailures.push({
          kind: "response",
          method: response.request().method(),
          path: new URL(response.url()).pathname,
          status: response.status(),
        });
      }
    });
    /** @type {Array<Record<string, unknown>>} */
    const requests = [];
    evidenceRequests.set(page, requests);
    evidenceRequestOffsets.set(page, 0);
    page.on("request", (request) => {
      const path = new URL(request.url()).pathname;
      if (!path.startsWith("/api/")) {
        return;
      }
      let command = null;
      if (request.method() === "POST") {
        try {
          command = request.postDataJSON()?.command ?? null;
        } catch {
          command = null;
        }
      }
      requests.push({ method: request.method(), path, command });
    });
  }
}

/** @param {string} path */
async function sha256File(path) {
  return createHash("sha256")
    .update(await readFile(path))
    .digest("hex");
}

/** @param {string} captureDirectory */
async function writeCp4EEvidenceReport(captureDirectory) {
  expect(cp4ENativeCaptures.map(({ filename }) => filename)).toEqual(
    CP4_E_CAPTURE_FILENAMES,
  );
  expect(cp4EBrowserErrors).toEqual([]);
  expect(cp4ENetworkFailures).toEqual([]);
  /** @type {Record<string, string>} */
  const captureHashes = {};
  for (const filename of CP4_E_CAPTURE_FILENAMES) {
    captureHashes[filename] = await sha256File(join(captureDirectory, filename));
  }
  /** @type {Record<string, string>} */
  const ownerHashes = {};
  for (const path of CP4_E_OWNER_PATHS) {
    ownerHashes[path] = await sha256File(join(REPOSITORY_ROOT, path));
  }
  const report = {
    schema: "marl-battlegrounds.cp4-e-native-evidence.v1",
    checkpoint: "CP4-E",
    generated_at_utc: new Date().toISOString(),
    repository_root: REPOSITORY_ROOT,
    git_head: execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: REPOSITORY_ROOT,
      encoding: "utf8",
    }).trim(),
    git_head_tree: execFileSync("git", ["rev-parse", "HEAD^{tree}"], {
      cwd: REPOSITORY_ROOT,
      encoding: "utf8",
    }).trim(),
    runtime: {
      node: process.version,
      platform: process.platform,
      architecture: process.arch,
      user_agent: cp4ENativeCaptures[0]?.user_agent ?? null,
    },
    service_inventory: [
      "real live Combat Debugger",
      "real complete Oracle replay",
      "real SharedObs Agent POV replay",
    ],
    capture_inventory: [...CP4_E_CAPTURE_FILENAMES],
    capture_sha256: captureHashes,
    owner_sha256: ownerHashes,
    captures: cp4ENativeCaptures,
    browser_errors: cp4EBrowserErrors,
    network_failures: cp4ENetworkFailures,
  };
  await writeFile(
    join(captureDirectory, "cp4-e-native-evidence.json"),
    `${JSON.stringify(report, null, 2)}\n`,
    { encoding: "utf8", flag: "wx" },
  );
}

/** @type {Readonly<Record<number, Readonly<{label: string, accent: string}>>>} */
const EXPECTED_AGENT_CLASSES = Object.freeze({
  1: Object.freeze({ label: "Mage", accent: "mage" }),
  2: Object.freeze({ label: "Warrior", accent: "warrior" }),
  3: Object.freeze({ label: "Hunter", accent: "hunter" }),
  4: Object.freeze({ label: "Rogue", accent: "rogue" }),
  5: Object.freeze({ label: "Priest", accent: "priest" }),
});
/** @type {Readonly<Record<number, string>>} */
const EXPECTED_AGENT_TEAMS = Object.freeze({ 1: "Team A", 2: "Team B" });

/** @param {Record<string, any>} agent */
function expectedAgentIdentity(agent) {
  const classIdentity = EXPECTED_AGENT_CLASSES[agent.class_id];
  const teamLabel = EXPECTED_AGENT_TEAMS[agent.team_id];
  expect(classIdentity).toBeTruthy();
  expect(teamLabel).toBeTruthy();
  expect(typeof agent.public_agent_id).toBe("string");
  return {
    title: `Agent ID ${agent.public_agent_id} · ${classIdentity.label} · ${teamLabel}`,
    accent: classIdentity.accent,
  };
}

/**
 * Prove the compact agent card is current-fact-only and cannot replace the
 * persistent certified class-documentation card.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} agent
 * @param {unknown} persistentCardBefore
 */
async function expectCompactAgentTooltip(page, agent, persistentCardBefore) {
  const identity = expectedAgentIdentity(agent);
  const expectedLabels = [
    "Health",
    "Effective Speed",
    "Ultimate Status",
    "Combat Status",
    ...(agent.steps_until_out_of_combat > 0 ? ["Steps until OOC"] : []),
  ];
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await expect(page.locator("#visual-tooltip-title")).toHaveText(identity.title);
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-accent",
    identity.accent,
  );
  await expect(
    page.locator("#visual-tooltip .semantic-explanation__summary"),
  ).toHaveCount(0);
  await expect(page.locator("#visual-tooltip .semantic-explanation__label")).toHaveText(
    expectedLabels,
  );
  await expect(
    page.locator("#visual-tooltip .semantic-explanation__value").nth(3),
  ).toHaveText(agent.steps_until_out_of_combat > 0 ? "IC" : "OOC");
  const tooltipText = await page.locator("#visual-tooltip").innerText();
  for (const forbidden of [
    "Ultimate Name",
    "Controlled actor",
    "Selected target",
    "Reference",
    "Inspected agent",
    "Current effect",
    "Spawn Shield",
    "Now",
  ]) {
    expect(tooltipText).not.toContain(forbidden);
  }
  expect(await page.locator("#selection-card").evaluate((node) => node.innerHTML)).toBe(
    persistentCardBefore,
  );
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} agent
 */
async function expectCertifiedDocumentationCard(page, agent) {
  const identity = expectedAgentIdentity(agent);
  await expect(page.locator("#selection-card > .sr-only")).toHaveText(identity.title);
  await expect(
    page.locator("#selection-card .semantic-explanation__heading"),
  ).toHaveText(["Class Overview", "Authored Tactical Guide", "Class Mechanics"]);
  await expect(
    page.locator(
      "#selection-card > .semantic-inspector__details > .semantic-explanation__summary",
    ),
  ).toHaveCount(0);
  await expect(page.locator("#selection-card .selected-outgoing-target")).toHaveCount(
    0,
  );
  await expect(page.locator("#selection-card .selected-legality")).toHaveCount(0);
  const forbiddenLabels = [
    "Health",
    "Effective Speed",
    "Ultimate Status",
    "Combat Status",
    "Steps until OOC",
  ];
  for (const label of forbiddenLabels) {
    await expect(
      page.locator("#selection-card .semantic-explanation__label", {
        hasText: new RegExp(`^${label}$`, "u"),
      }),
    ).toHaveCount(0);
  }
}

/** @param {import("@playwright/test").Page} page */
async function expectRetiredMetadataAbsent(page) {
  await expect(page.locator("#revision-value")).toHaveCount(0);
  await expect(page.locator("#replay-incoming-value")).toHaveCount(0);
  await expect(page.locator("#transition-value")).toHaveCount(1);
}

const TECHNICAL_HELP = Object.freeze({
  frame: Object.freeze({
    label: "Frame",
    summary: "The zero-based authorized frame index represented by this presentation.",
  }),
  simulator_step: Object.freeze({
    label: "Simulator step",
    summary: "The simulator decision step represented by this authorized frame.",
  }),
  ordinary_movement_distance_scale: Object.freeze({
    label: "Ordinary movement distance scale",
    summary:
      "The recorded multiplier applied to ordinary voluntary movement distance. Spawn Shield uses its separately authorized absolute movement speed.",
  }),
});

/**
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 */
async function expectTechnicalFrameDom(page, presentation) {
  const technical = presentation.technical_frame;
  const technicalDetailsWasOpen =
    (await page.locator("#technical-frame-details").getAttribute("open")) === "";
  if (!technicalDetailsWasOpen) {
    await openDetails(page, ["#technical-frame-details"]);
  }
  /** @type {Record<string, Array<[keyof typeof TECHNICAL_HELP, string]>>} */
  const specifications = {
    live_oracle: [
      ["frame", "evaluation_frame_index"],
      ["simulator_step", "simulator_step_count"],
    ],
    live_no_shared_obs_agent_pov: [
      ["frame", "recipient_frame_index"],
      ["simulator_step", "simulator_step_count"],
    ],
    replay_oracle: [
      ["frame", "frame_index"],
      ["simulator_step", "simulator_step_count"],
      ["ordinary_movement_distance_scale", "recorded_ordinary_movement_distance_scale"],
    ],
    replay_no_shared_obs_agent_pov: [
      ["frame", "frame_index"],
      ["simulator_step", "simulator_step_count"],
    ],
    replay_shared_obs_agent_pov: [
      ["frame", "frame_index"],
      ["simulator_step", "simulator_step_count"],
    ],
  };
  const expected = specifications[presentation.presentation_kind];
  if (!expected) {
    throw new TypeError(
      `Unsupported Technical Frame presentation kind ${String(presentation.presentation_kind)}.`,
    );
  }
  const facts = page.locator("#diagnostics-card .fact[data-technical-fact]");
  await expect(facts).toHaveCount(expected.length);
  expect(
    await facts.evaluateAll((nodes) =>
      nodes.map((node) => ({
        id: node.getAttribute("data-technical-fact"),
        label: node.querySelector("span")?.textContent,
        value: node.querySelector("strong")?.textContent,
        tabindex: node.getAttribute("tabindex"),
      })),
    ),
  ).toEqual(
    expected.map(([id, field]) => ({
      id,
      label: TECHNICAL_HELP[id].label,
      value: String(technical[field]),
      tabindex: "0",
    })),
  );
  for (const [id] of expected) {
    const owner = page.locator(`#diagnostics-card .fact[data-technical-fact="${id}"]`);
    await page.mouse.move(1, 1);
    await owner.focus();
    await expect(page.locator("#visual-tooltip-title")).toHaveText(
      TECHNICAL_HELP[id].label,
    );
    await expect(
      page.locator("#visual-tooltip .semantic-explanation__summary"),
    ).toHaveText(TECHNICAL_HELP[id].summary);
  }
  if (presentation.product_kind === "replay_viewer") {
    for (const [selector, title, summary] of [
      [
        "#replay-completion-badge",
        "Completion",
        "How the captured rollout ended. Rollout completion is independent of host-side processing success.",
      ],
      [
        "#replay-processing-badge",
        "Processing",
        "Whether host-side evaluation output was produced successfully. Processing does not change how the rollout ended.",
      ],
    ]) {
      const owner = page.locator(selector);
      await expect(owner).toHaveAttribute("tabindex", "0");
      await page.mouse.move(1, 1);
      await owner.focus();
      await expect(page.locator("#visual-tooltip-title")).toHaveText(title);
      await expect(
        page.locator("#visual-tooltip .semantic-explanation__summary"),
      ).toHaveText(summary);
    }
  }
  if (!technicalDetailsWasOpen) {
    await closeDetails(page, ["#technical-frame-details"]);
  }
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 */
async function expectLatestTransitionDom(page, presentation) {
  const rawRows = presentation.latest_transition?.action_rows ?? [];
  const rows = page.locator("#accepted-card .accepted-action-row");
  await expect(rows).toHaveCount(rawRows.length);
  await expect(page.locator("#accepted-card")).not.toHaveAttribute(
    "data-transition-id",
    /.+/u,
  );
  await expect(
    page.locator("#accepted-card .accepted-action-row[data-presentation-key]"),
  ).toHaveCount(0);
  if (rawRows.length === 0) {
    expect((await page.locator("#accepted-card").textContent())?.trim()).toBe("");
    return;
  }
  const scene =
    presentation.current_endpoint.scene ?? presentation.current_endpoint.parts?.scene;
  expect(Array.isArray(scene?.agents)).toBe(true);
  for (const [index, row] of rawRows.entries()) {
    const identity = scene.agents.find(
      (/** @type {Record<string, any>} */ candidate) =>
        candidate.presentation_key === row.actor_presentation_key &&
        candidate.public_agent_id === row.actor_public_agent_id,
    );
    expect(identity).toBeTruthy();
    const expectedIdentity = expectedAgentIdentity(identity);
    const rendered = rows.nth(index);
    await expect(rendered.locator(".accepted-action-row__title")).toHaveText(
      expectedIdentity.title,
    );
    await expect(rendered.locator(".accepted-action-row__title")).toHaveAttribute(
      "data-class",
      expectedIdentity.accent,
    );
    await expect(rendered.locator(".accepted-action-tuple")).toHaveCount(2);
    await expect(rendered.locator(".accepted-action-tuple > h4")).toHaveText([
      "Submitted",
      "Accepted",
    ]);
    expect(
      await rendered
        .locator(".accepted-action-tuple")
        .evaluateAll((tuples) =>
          tuples.map((tuple) => tuple.getAttribute("data-kind")),
        ),
    ).toEqual(["submitted", "accepted"]);
    const tupleText = (/** @type {Record<string, any>} */ action) =>
      `Move ${action.move_action} · Target ${action.target_action} · Ultimate ${action.use_ultimate_action}`;
    await expect(rendered.locator(".accepted-action-tuple__value")).toHaveText([
      tupleText(row.submitted_action),
      tupleText(row.accepted_action),
    ]);
  }
}

/** @param {import("@playwright/test").Page} page @param {string[]} selectors */
async function openDetails(page, selectors) {
  for (const selector of selectors) {
    const details = page.locator(selector);
    if ((await details.getAttribute("open")) !== "") {
      await details.locator(":scope > summary").click();
    }
    await expect(details).toHaveAttribute("open", "");
  }
}

/** @param {import("@playwright/test").Page} page @param {string[]} selectors */
async function closeDetails(page, selectors) {
  for (const selector of selectors) {
    const details = page.locator(selector);
    if ((await details.getAttribute("open")) === "") {
      await details.locator(":scope > summary").click();
    }
    await expect(details).not.toHaveAttribute("open", "");
  }
}

/**
 * Restore the retained trajectory's native baseline after one screenshot-only
 * viewport and disclosure state.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string[]} openedDetails
 */
async function cleanupAfterCp4ECapture(page, openedDetails) {
  await page.setViewportSize({ width: 1440, height: 900 });
  await closeDetails(page, openedDetails);
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  });
  await page.mouse.move(1, 1);
  await page.evaluate(() => window.scrollTo(0, 0));
}

/**
 * Capture one prescribed native state only when the CP4-E evidence run opts in.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{filename: string, width: number, height: number, presentation: Record<string, any>, selectedAgent: Record<string, any>}} options
 */
async function captureCp4ENativeState(page, options) {
  if (cp4ECaptureDirectory === null) {
    return;
  }
  expect(CP4_E_CAPTURE_FILENAMES).toContain(options.filename);
  expect(cp4ENativeCaptures.some(({ filename }) => filename === options.filename)).toBe(
    false,
  );
  await page.setViewportSize({ width: options.width, height: options.height });
  await page.evaluate(async () => {
    await document.fonts.ready;
    await new Promise((resolveAnimation) =>
      requestAnimationFrame(() => requestAnimationFrame(resolveAnimation)),
    );
  });
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const selectedIdentity = expectedAgentIdentity(options.selectedAgent);
  await page
    .locator(
      `#battlefield .agent[data-presentation-key="${options.selectedAgent.presentation_key}"]`,
    )
    .hover();
  await expect(page.locator("#visual-tooltip-title")).toHaveText(
    selectedIdentity.title,
  );
  const trace = await page.evaluate(() => {
    const values = (/** @type {string} */ selector) =>
      [...document.querySelectorAll(selector)].map((node) => node.textContent?.trim());
    const rectangle = (/** @type {string} */ selector) => {
      const node = document.querySelector(selector);
      if (!(node instanceof Element)) return null;
      const bounds = node.getBoundingClientRect();
      return {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
        right: bounds.right,
        bottom: bounds.bottom,
      };
    };
    const documentElement = document.documentElement;
    return {
      product_kind: documentElement.getAttribute("data-product-kind"),
      viewer_mode: documentElement.getAttribute("data-viewer-mode"),
      authority: documentElement.getAttribute("data-presentation-authority"),
      viewport: { width: innerWidth, height: innerHeight },
      horizontal_overflow:
        documentElement.scrollWidth > documentElement.clientWidth ||
        document.body.scrollWidth > document.body.clientWidth,
      rectangles: {
        document: {
          client_width: documentElement.clientWidth,
          client_height: documentElement.clientHeight,
          scroll_width: documentElement.scrollWidth,
          scroll_height: documentElement.scrollHeight,
        },
        body: rectangle("body"),
        workspace: rectangle(".workspace"),
        battlefield_shell: rectangle("#battlefield-shell"),
        battlefield: rectangle("#battlefield"),
        inspector: rectangle(".hud-panel"),
        tooltip: rectangle("#visual-tooltip"),
      },
      open_details: [...document.querySelectorAll("details[open]")].map(
        (node) => node.id,
      ),
      documentation_title:
        document.querySelector("#selection-card > .sr-only")?.textContent ?? null,
      documentation_sections: values("#selection-card .semantic-explanation__heading"),
      pending_heading: document.querySelector("#pending-heading")?.textContent ?? null,
      latest_transition_titles: values(".accepted-action-row__title"),
      latest_transition_tuples: values(".accepted-action-tuple__value"),
      technical_facts: [...document.querySelectorAll("[data-technical-fact]")].map(
        (node) => ({
          id: node.getAttribute("data-technical-fact"),
          text: node.textContent?.trim(),
        }),
      ),
      tooltip: {
        kind: document
          .querySelector("#visual-tooltip")
          ?.getAttribute("data-tooltip-kind"),
        accent: document
          .querySelector("#visual-tooltip")
          ?.getAttribute("data-tooltip-accent"),
        title: document.querySelector("#visual-tooltip-title")?.textContent ?? null,
        labels: values("#visual-tooltip .semantic-explanation__label"),
        values: values("#visual-tooltip .semantic-explanation__value"),
      },
      retired_roots: {
        revision: document.querySelectorAll("#revision-value").length,
        replay_incoming: document.querySelectorAll("#replay-incoming-value").length,
      },
      user_agent: navigator.userAgent,
    };
  });
  expect(trace.viewport).toEqual({ width: options.width, height: options.height });
  expect(trace.horizontal_overflow).toBe(false);
  expect(trace.retired_roots).toEqual({ revision: 0, replay_incoming: 0 });
  expect(trace.documentation_title).toBe(selectedIdentity.title);
  const allRequests = evidenceRequests.get(page) ?? [];
  const requestOffset = evidenceRequestOffsets.get(page) ?? 0;
  const requestDelta = allRequests.slice(requestOffset);
  evidenceRequestOffsets.set(page, allRequests.length);
  const outputPath = join(cp4ECaptureDirectory, options.filename);
  await page.screenshot({ path: outputPath, fullPage: false, animations: "disabled" });
  cp4ENativeCaptures.push({
    filename: options.filename,
    presentation_kind: options.presentation.presentation_kind,
    selected_canonical_identity: selectedIdentity,
    source_frame_index: options.presentation.source.source_frame_index,
    simulator_step: options.presentation.technical_frame.simulator_step_count,
    ...trace,
    api_request_delta: requestDelta,
    post_command_delta: requestDelta.filter(({ method }) => method === "POST"),
    browser_errors: [...(browserErrors.get(page) ?? [])],
  });
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 * @param {"live" | "replay"} mode
 */
async function openProduct(page, url, mode) {
  captureBrowserErrors(page);
  await page.goto(url);
  await expect(page.locator("#connection-status")).toHaveText("Online", {
    timeout: 30_000,
  });
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", mode);
  expect(page.url()).not.toContain("token=");
}

/**
 * Hold only the initial authorized-presentation response after the synchronous
 * route bootstrap has established product identity. The pending shell must
 * remain product-correct without exposing any previously or partially joined
 * scientific presentation.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 * @param {"live" | "replay"} mode
 * @param {"combat_debugger" | "replay_viewer"} productKind
 */
async function openProductWithHeldInitialPresentation(page, url, mode, productKind) {
  let releasePresentation = () => {};
  let markPresentationHeld = () => {};
  const presentationHeld = new Promise((resolve) => {
    markPresentationHeld = () => resolve(undefined);
  });
  const presentationRelease = new Promise((resolve) => {
    releasePresentation = () => resolve(undefined);
  });
  let holdNextPresentation = true;
  await page.route("**/api/presentation/frame", async (route) => {
    if (!holdNextPresentation) {
      await route.continue();
      return;
    }
    holdNextPresentation = false;
    const response = await route.fetch();
    markPresentationHeld();
    await presentationRelease;
    await route.fulfill({ response });
  });

  const opening = openProduct(page, url, mode);
  try {
    await presentationHeld;
    await expect(page.locator("html")).toHaveAttribute(
      "data-product-kind",
      productKind,
    );
    await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", mode);
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "pending",
    );
    await expect(page.locator("#battlefield .agent")).toHaveCount(0);
    await expect(page.locator("[data-presentation-key]")).toHaveCount(0);
    await expect(page.locator("#step-value")).toHaveText("—");
    await expect(page.locator("#transition-value")).toHaveText("—");
    await expect(page.locator("#view-select")).toBeDisabled();
    await expect(page.locator("#exit-button")).toBeDisabled();

    const oppositeSelector =
      productKind === "replay_viewer" ? "[data-live-only]" : "[data-replay-only]";
    const visibleOppositeRoots = await page
      .locator(oppositeSelector)
      .evaluateAll((elements) =>
        elements
          .filter((element) => getComputedStyle(element).display !== "none")
          .map((element) => element.id || element.className),
      );
    expect(visibleOppositeRoots).toEqual([]);
    await expect(page.locator("#battlefield")).toHaveAttribute("role", "img");
    await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
    await expect(page.locator("#battlefield")).toHaveAttribute(
      "aria-label",
      "Read-only live battlefield. Simulator and actor activation controls are unavailable.",
    );
    await expect(page.locator("#battlefield-instructions")).toHaveText(
      "Live battlefield interaction is unavailable while authority is pending, offline, resynchronizing, shutting down, or terminal.",
    );

    if (productKind === "replay_viewer") {
      await expect(page.locator("#replay-timeline")).toBeVisible();
      await expect(page.locator("#replay-ranges-button")).toBeVisible();
      await expect(page.locator("#replay-ranges-button")).toBeDisabled();
      await expect(page.locator("#replay-clear-reference-button")).toBeVisible();
      await expect(page.locator("#replay-clear-reference-button")).toBeDisabled();
      expect(
        await page
          .locator("#replay-timeline button")
          .evaluateAll((buttons) =>
            buttons.every(
              (button) => button instanceof HTMLButtonElement && button.disabled,
            ),
          ),
      ).toBe(true);
    } else {
      await expect(page.locator("#command-deck")).toBeVisible();
      await expect(page.locator("#live-ranges-button")).toBeDisabled();
      expect(
        await page
          .locator("#command-deck button")
          .evaluateAll((buttons) =>
            buttons.every(
              (button) => button instanceof HTMLButtonElement && button.disabled,
            ),
          ),
      ).toBe(true);
    }
  } finally {
    releasePresentation();
    await Promise.allSettled([opening]);
    await page.unroute("**/api/presentation/frame");
  }
  await opening;
}

/**
 * Prove one replay utility activation emits one exact command and installs the
 * response's joined successor presentation before this helper returns.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} selector
 * @param {Readonly<Record<string, unknown>>} expectedCommand
 */
async function expectSingleReplayUtilityCommand(page, selector, expectedCommand) {
  /** @type {import("@playwright/test").Request[]} */
  const commandRequests = [];
  const recordCommand = (/** @type {import("@playwright/test").Request} */ request) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      commandRequests.push(request);
    }
  };
  page.on("request", recordCommand);
  let responsePayload;
  try {
    const responsePromise = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname === "/api/replay/command",
      { timeout: 30_000 },
    );
    const presentationPromise = page.waitForResponse(
      (response) =>
        response.request().method() === "GET" &&
        new URL(response.url()).pathname === "/api/presentation/frame",
      { timeout: 30_000 },
    );
    await page.locator(selector).click();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    responsePayload = await response.json();
    expect((await presentationPromise).status()).toBe(200);
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
  } finally {
    page.off("request", recordCommand);
  }
  expect(commandRequests).toHaveLength(1);
  expect(commandRequests[0].postDataJSON().command).toEqual(expectedCommand);
  return responsePayload;
}

/**
 * Run one native activation and prove it crosses exactly one product command
 * boundary with the exact existing command body.
 *
 * @param {import("@playwright/test").Page} page
 * @param {"/api/command" | "/api/replay/command"} path
 * @param {() => Promise<void>} activate
 * @param {Readonly<Record<string, unknown>>} expectedCommand
 */
async function expectSingleActivationCommand(page, path, activate, expectedCommand) {
  /** @type {import("@playwright/test").Request[]} */
  const requests = [];
  const record = (/** @type {import("@playwright/test").Request} */ request) => {
    if (
      request.method() === "POST" &&
      ["/api/command", "/api/replay/command"].includes(new URL(request.url()).pathname)
    ) {
      requests.push(request);
    }
  };
  page.on("request", record);
  try {
    const responsePromise = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname === path,
      { timeout: 30_000 },
    );
    await activate();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
    await page.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
  } finally {
    page.off("request", record);
  }
  expect(requests).toHaveLength(1);
  expect(new URL(requests[0].url()).pathname).toBe(path);
  expect(requests[0].postDataJSON().command).toEqual(expectedCommand);
}

/**
 * Run one local interaction through a real browser event and prove neither
 * product command route was touched.
 *
 * @param {import("@playwright/test").Page} page
 * @param {() => Promise<void>} activate
 */
async function expectZeroCommandInteraction(page, activate) {
  /** @type {import("@playwright/test").Request[]} */
  const requests = [];
  const record = (/** @type {import("@playwright/test").Request} */ request) => {
    if (
      request.method() === "POST" &&
      ["/api/command", "/api/replay/command"].includes(new URL(request.url()).pathname)
    ) {
      requests.push(request);
    }
  };
  page.on("request", record);
  try {
    await activate();
    await page.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
  } finally {
    page.off("request", record);
  }
  expect(requests).toEqual([]);
}

/**
 * Exercise the one authorized activation through both public surfaces and all
 * three native activation gestures. Oracle cells cross one exact command
 * boundary; local inspection cells cross none.
 *
 * @param {import("@playwright/test").Page} page
 * @param {{
 *   body: import("@playwright/test").Locator,
 *   row: import("@playwright/test").Locator,
 *   path: "/api/command" | "/api/replay/command" | null,
 *   command: Readonly<Record<string, unknown>> | null,
 * }} options
 */
async function expectNativeAgentActivationMatrix(page, { body, row, path, command }) {
  const scrollY = () => page.evaluate(() => window.scrollY);
  const cells = [
    () => row.click(),
    async () => {
      await row.focus();
      await row.press("Enter");
    },
    async () => {
      await row.focus();
      const before = await scrollY();
      await row.press(" ");
      expect(await scrollY()).toBe(before);
    },
    () => body.click(),
    async () => {
      await body.focus();
      await body.press("Enter");
    },
    async () => {
      await body.focus();
      const before = await scrollY();
      await body.press(" ");
      expect(await scrollY()).toBe(before);
    },
  ];
  for (const activate of cells) {
    if (path === null || command === null) {
      await expectZeroCommandInteraction(page, activate);
    } else {
      await expectSingleActivationCommand(page, path, activate, command);
    }
  }
}

/**
 * Terminal presentation bodies and roster activators remain descriptive but
 * cannot cross either product command route, even under forced DOM events.
 *
 * @param {import("@playwright/test").Page} page
 */
async function expectTerminalAgentActivationInert(page) {
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-label",
    "Read-only terminal replay battlefield snapshot. Agent inspection controls are unavailable.",
  );
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "This replay frame is terminal. Agent inspection controls are unavailable; use the timeline to review another frame.",
  );
  const bodies = page.locator("#battlefield .agent");
  const rows = page.locator("#roster .roster-primary-action");
  expect(await bodies.count()).toBeGreaterThan(0);
  await expect(bodies.first()).toHaveAttribute("role", "img");
  await expect(bodies.first()).toHaveAttribute("tabindex", "-1");
  expect(
    await rows.evaluateAll((buttons) =>
      buttons.every((button) => button instanceof HTMLButtonElement && button.disabled),
    ),
  ).toBe(true);
  const localState = () =>
    page.evaluate(() => ({
      selectedKeys: [
        ...document.querySelectorAll('#battlefield .agent[data-selected="true"]'),
      ].map((agent) => agent.getAttribute("data-presentation-key")),
      pressedRows: [...document.querySelectorAll("#roster .roster-primary-action")].map(
        (button) => button.getAttribute("aria-pressed"),
      ),
      ranges: [...document.querySelectorAll("#battlefield .range-ring")].map(
        (range) => ({
          kind: range.getAttribute("data-kind"),
          owner: range.getAttribute("data-presentation-key"),
        }),
      ),
      detailsOpen: document.querySelector("#agent-details")?.hasAttribute("open"),
      detailsAccent: document
        .querySelector("#agent-details")
        ?.getAttribute("data-accent"),
      selectionText: document.querySelector("#selection-card")?.textContent,
    }));
  const before = await localState();
  await expectZeroCommandInteraction(page, () => bodies.first().click({ force: true }));
  await expectZeroCommandInteraction(page, () =>
    bodies.first().dispatchEvent("keydown", { key: "Enter" }),
  );
  await expectZeroCommandInteraction(page, () =>
    bodies.first().dispatchEvent("keydown", { key: " " }),
  );
  expect(await localState()).toEqual(before);
}

/**
 * Read one exact authenticated product resource without bypassing the real
 * loopback service or hand-authoring an authority payload.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} path
 */
async function authenticatedGet(page, path) {
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
    return response.json();
  }, path);
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {string} transportKind
 * @param {string} presentationKind
 */
async function expectInstalledLeaf(page, transportKind, presentationKind) {
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const [transport, presentation] = await Promise.all([
    authenticatedGet(page, "/api/frame"),
    authenticatedGet(page, "/api/presentation/frame"),
  ]);
  expect(transport.frame_kind).toBe(transportKind);
  expect(presentation.presentation_kind).toBe(presentationKind);
  expect(presentation.source.source_session_id).toBe(
    transport.session_id ?? transport.viewer_session_id,
  );
  await expect(page.locator("#battlefield .agent")).not.toHaveCount(0);
  const bodyKeys = await page
    .locator("#battlefield .agent")
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  expect(bodyKeys.every((key) => typeof key === "string" && key.length > 0)).toBe(true);
  return { transport, presentation };
}

/**
 * Prove the retained session fact uses the installed presentation variant's
 * exact incoming-transition identity. The authoritative frame-zero
 * representation has no latest-events branch.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 */
async function expectAuthorizedIncomingTransitionDom(page, presentation) {
  const latestEvents = presentation.latest_events;
  const oracle =
    presentation.presentation_kind === "live_oracle" ||
    presentation.presentation_kind === "replay_oracle";
  const expected =
    latestEvents === null
      ? null
      : oracle
        ? latestEvents.incoming_transition_id
        : latestEvents.incoming_recipient_transition_id;
  if (latestEvents !== null) {
    expect(typeof expected).toBe("string");
    expect(expected.length).toBeGreaterThan(0);
  }
  await expect(page.locator("#transition-value")).toHaveText(expected ?? "—");
}

/**
 * Prove current selected-owner facts remain separate from the optional outgoing
 * inspection overlay. This reuses the real service trajectory already owned by
 * the five-leaf test; it adds no fixture server or viewport loop.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 */
async function expectReplayInspectionDom(page, presentation) {
  expect(presentation.product_kind).toBe("replay_viewer");
  const inspection = presentation.replay_inspection;
  const endpoint = presentation.current_endpoint;
  const axis = endpoint.action_axis;
  const scene = endpoint.scene ?? endpoint.parts?.scene;
  expect(Array.isArray(scene?.agents)).toBe(true);
  const ownerKey =
    inspection?.actor_presentation_key ?? axis?.owner_presentation_key ?? null;
  const ownerPublicId =
    inspection?.actor_public_agent_id ?? axis?.owner_public_agent_id ?? null;
  const owner =
    typeof ownerKey === "string"
      ? (scene.agents.find(
          (/** @type {Record<string, any>} */ agent) =>
            agent.presentation_key === ownerKey &&
            agent.public_agent_id === ownerPublicId,
        ) ?? null)
      : null;

  await expect(page.locator("#selection-heading")).toHaveText(
    "Comprehensive Agent Details",
  );
  await expect(page.locator('[data-layer="selection-legality"]')).toHaveAttribute(
    "aria-label",
    "Selection and exact actor-owned legality",
  );
  if (owner === null) {
    await expect(page.locator('#battlefield .agent[data-selected="true"]')).toHaveCount(
      0,
    );
    await expect(page.locator("#agent-details")).not.toHaveAttribute("data-accent");
    await expect(page.locator("#selection-card")).toContainText(
      "No authorized agent details are available.",
    );
    await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
    await expect(page.locator("#pending-card .selected-legality__lane")).toHaveCount(0);
    await expect(page.locator("#pending-card .selected-outgoing-target")).toHaveCount(
      0,
    );
    await expect(page.locator("#battlefield .legality-dock")).toHaveCount(0);
    await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
    return;
  }

  const ownerBody = page.locator(
    `#battlefield .agent[data-presentation-key="${owner.presentation_key}"]`,
  );
  await expect(ownerBody).toHaveCount(1);
  await expect(ownerBody).toHaveAttribute("data-selected", "true");
  const classAccent = await ownerBody.getAttribute("data-class");
  expect(classAccent).not.toBeNull();
  await expect(page.locator("#agent-details")).toHaveAttribute(
    "data-accent",
    String(classAccent),
  );
  await expectCertifiedDocumentationCard(page, owner);
  const expectedRangeKinds = [
    ["observation", owner.observation_radius],
    ["basic", owner.basic_interaction_radius],
    ["ultimate", owner.ultimate_interaction_radius],
  ]
    .filter(([, radius]) => typeof radius === "number" && radius > 0)
    .map(([kind]) => kind);
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual(
    expectedRangeKinds.map((kind) => ({
      kind,
      owner: owner.presentation_key,
    })),
  );

  if (inspection === null) {
    await expect(page.locator("#pending-card .selected-outgoing-target")).toHaveCount(
      0,
    );
    await expect(page.locator("#pending-card .selected-legality__lane")).toHaveCount(0);
    await expect(page.locator("#battlefield .legality-dock")).toHaveCount(0);
    await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
    return;
  }

  const targetAction = inspection.accepted_action.target_action;
  const exactRow =
    inspection.decision_mask.target_use_ultimate_joint_mask[targetAction];
  const outgoingTarget = page.locator("#pending-card .selected-outgoing-target");
  await expect(outgoingTarget).toHaveCount(1);
  await expect(outgoingTarget).toHaveAttribute(
    "data-target-kind",
    inspection.accepted_target.target_kind,
  );
  await expect(outgoingTarget).toContainText(inspection.accepted_target.display_name);
  await expect(outgoingTarget).toContainText(
    inspection.accepted_target.target_kind === "no_target"
      ? "No target"
      : `Agent ID ${inspection.accepted_target.target_public_agent_id}`,
  );
  await expect(page.locator("#pending-card .selected-legality__lane h3")).toHaveText([
    `Basic Legality · Agent ID ${owner.public_agent_id}`,
    `Ultimate Legality · Agent ID ${owner.public_agent_id}`,
  ]);
  const legalityDock = page.locator(
    `#battlefield .legality-dock[data-presentation-key="${owner.presentation_key}"]`,
  );
  await expect(legalityDock).toHaveCount(1);
  await expect(legalityDock).toHaveAttribute(
    "aria-label",
    `Exact actor-owned legality for Agent ID ${owner.public_agent_id}`,
  );
  expect(
    await legalityDock.locator(".legality-pill").evaluateAll((pills) =>
      pills.map((pill) => ({
        available: pill.getAttribute("data-available"),
        lane: pill.getAttribute("data-lane"),
      })),
    ),
  ).toEqual([
    { available: String(exactRow[0]), lane: "0" },
    { available: String(exactRow[1]), lane: "1" },
  ]);
  const routeExpected =
    inspection.accepted_target.target_kind === "visible_authorized_agent" &&
    (inspection.combat_lane === "basic" || inspection.combat_lane === "ultimate");
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(
    routeExpected ? 1 : 0,
  );
}

/**
 * Snapshot every browser-owned DOM string and attribute at the authority
 * boundary. Source JavaScript is intentionally outside this scan.
 *
 * @param {import("@playwright/test").Page} page
 */
async function authorityDomSnapshot(page) {
  return page.evaluate(() => ({
    attributes: [document.body, ...document.body.querySelectorAll("*")].flatMap(
      (element) =>
        [...element.attributes].map((attribute) => ({
          name: attribute.name,
          value: attribute.value,
        })),
    ),
    text: document.body.textContent ?? "",
  }));
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 * @param {string[]} forbiddenValues
 * @param {boolean} [activationEnabled]
 */
async function expectAgentAuthoritySurface(
  page,
  presentation,
  forbiddenValues,
  activationEnabled = true,
) {
  expect(presentation.authority.authority_kind).toBe("agent_pov");
  const recipientKey = presentation.authority.recipient_presentation_key;
  const bodyKeys = await page.locator("#battlefield .agent").evaluateAll((agents) =>
    agents.map((agent) => ({
      key: agent.getAttribute("data-presentation-key"),
      role: agent.getAttribute("role"),
      slot: agent.getAttribute("data-slot"),
    })),
  );
  expect(bodyKeys.every(({ slot }) => slot === null)).toBe(true);
  expect(bodyKeys.every(({ key }) => typeof key === "string")).toBe(true);
  expect(
    bodyKeys.every(({ role }) => role === (activationEnabled ? "button" : "img")),
  ).toBe(true);
  await expect(page.locator("#roster .roster-row--authorized")).toHaveCount(
    bodyKeys.length,
  );
  await expect(page.locator("#roster .roster-primary-action")).toHaveCount(
    bodyKeys.length,
  );
  if (activationEnabled) {
    await expect(page.locator("#roster .roster-primary-action").first()).toBeEnabled();
  } else {
    expect(
      await page
        .locator("#roster .roster-primary-action")
        .evaluateAll((buttons) =>
          buttons.every(
            (button) => button instanceof HTMLButtonElement && button.disabled,
          ),
        ),
    ).toBe(true);
  }
  await expect(page.locator("#roster .roster-actions")).toHaveCount(0);
  await expect(page.locator("#roster [data-role]")).toHaveCount(0);
  expect(bodyKeys.some(({ key }) => key === recipientKey)).toBe(true);

  const snapshot = await authorityDomSnapshot(page);
  const slotAttributes = snapshot.attributes.filter(({ name }) =>
    /(?:^|[-_])(?:global[-_]?slot|source[-_]?slot|target[-_]?slot|recipient[-_]?slot|actor[-_]?slot|controlled[-_]?slot)(?:$|[-_])/iu.test(
      name,
    ),
  );
  expect(slotAttributes).toEqual([]);
  const surfaceBytes = JSON.stringify(snapshot);
  for (const forbidden of forbiddenValues) {
    if (forbidden.length > 0 && !JSON.stringify(presentation).includes(forbidden)) {
      expect(surfaceBytes).not.toContain(forbidden);
    }
  }
  for (const forbiddenPhrase of [
    "shared_obs_source_material",
    "source_artifact_digest_sha256",
    "source_trajectory_content_digest_sha256",
    "source_context_digest_sha256",
    "global_slot",
  ]) {
    expect(surfaceBytes).not.toContain(forbiddenPhrase);
  }
  const agentUtilitySelectors =
    presentation.product_kind === "replay_viewer"
      ? ["#replay-ranges-button", "#replay-clear-reference-button"]
      : ["#live-ranges-button"];
  for (const selector of agentUtilitySelectors) {
    await expect(page.locator(selector)).toHaveAttribute(
      "aria-description",
      /locally authorized|local inspected-agent/u,
    );
    await expect(page.locator(selector)).toHaveAttribute(
      "aria-description",
      /sends no (?:replay )?command/u,
    );
    await expect(page.locator(selector)).toHaveAttribute(
      "aria-description",
      /fixed recipient/u,
    );
    await expect(page.locator(selector)).not.toHaveAttribute(
      "aria-description",
      /Oracle View|server-authored/u,
    );
  }
}

/** @param {import("@playwright/test").Page} page */
async function expectAuthorizedRosterColors(page) {
  const colors = await page.locator("#roster").evaluate((roster) => {
    const resolvedColor = (/** @type {string} */ variable) => {
      const probe = document.createElement("span");
      probe.style.color = `var(${variable})`;
      document.body.append(probe);
      const color = getComputedStyle(probe).color;
      probe.remove();
      return color;
    };
    const classColors = Object.fromEntries(
      [...roster.querySelectorAll(".roster-id[data-class]")].map((element) => [
        element.getAttribute("data-class"),
        getComputedStyle(element).color,
      ]),
    );
    const expectedClassColors = Object.fromEntries(
      ["mage", "warrior", "hunter", "rogue", "priest"].map((className) => [
        className,
        resolvedColor(`--class-${className}`),
      ]),
    );
    const teamBorders = Object.fromEntries(
      [...roster.querySelectorAll(".roster-row--authorized[data-team]")].map(
        (element) => [
          element.getAttribute("data-team"),
          getComputedStyle(element).borderLeftColor,
        ],
      ),
    );
    return {
      classColors,
      expectedClassColors,
      teamBorders,
      expectedTeamBorders: {
        "team-a": resolvedColor("--team-a"),
        "team-b": resolvedColor("--team-b"),
      },
      mutedClassLabels: [
        ...roster.querySelectorAll(".roster-row--authorized .roster-class"),
      ].map((element) => getComputedStyle(element).color),
      expectedMuted: resolvedColor("--text-muted"),
    };
  });
  expect(colors.classColors).toEqual(colors.expectedClassColors);
  expect(colors.teamBorders).toEqual(colors.expectedTeamBorders);
  expect(new Set(colors.mutedClassLabels)).toEqual(new Set([colors.expectedMuted]));
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} frameIndex
 */
async function seekReplay(page, frameIndex) {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/replay/command",
    { timeout: 30_000 },
  );
  await page.locator("#replay-frame-slider").evaluate((element, value) => {
    if (!(element instanceof HTMLInputElement)) {
      throw new TypeError("Replay slider is unavailable.");
    }
    element.value = String(value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
  }, frameIndex);
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  await expect(page.locator("#replay-frame-slider")).toHaveValue(String(frameIndex));
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
}

/**
 * Prove the synchronous clear boundary while one real authority response is
 * deliberately held after the server has accepted the audience command.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} oldScientificSentinel
 * @param {string} oldPresentationKey
 */
async function expectPendingAuthorityIsEmpty(
  page,
  oldScientificSentinel,
  oldPresentationKey,
) {
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  await expect(page.locator("#battlefield .agent")).toHaveCount(0);
  await expect(page.locator("[data-presentation-key]")).toHaveCount(0);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  const pendingTooltip = await page.locator("#visual-tooltip").evaluate((tooltip) => ({
    hidden: tooltip instanceof HTMLElement && tooltip.hidden,
    kind: tooltip.getAttribute("data-tooltip-kind"),
  }));
  expect(pendingTooltip.hidden || pendingTooltip.kind === "control").toBe(true);
  const scientificSurfaces = [
    "#battlefield",
    "#roster",
    "#agent-details",
    "#pending-card",
    "#accepted-card",
    "#event-feed",
    "#diagnostics-card",
  ];
  const descendantsWith = (/** @type {string} */ attribute) =>
    scientificSurfaces.map((surface) => `${surface} [${attribute}]`).join(",");
  await expect(page.locator(descendantsWith("data-tooltip-owner"))).toHaveCount(0);
  await expect(page.locator(descendantsWith("aria-describedby"))).toHaveCount(0);
  await expect(page.locator(descendantsWith("aria-description"))).toHaveCount(0);
  await expect(page.locator("#replay-artifact-reference")).not.toHaveAttribute(
    "data-tooltip-owner",
  );
  await expect(page.locator("#replay-artifact-reference")).not.toHaveAttribute(
    "aria-description",
  );
  await expect(page.locator("[data-authoritative-available]")).toHaveCount(0);
  await expect(page.locator("#battlefield")).not.toHaveAttribute(
    "aria-activedescendant",
  );
  await expect(page.locator("#command-controlled-actor")).not.toHaveAttribute(
    "aria-label",
  );
  await expect(page.locator("#command-controlled-actor")).not.toHaveAttribute(
    "data-controlled-slot",
  );
  await expect(page.locator("#agent-details")).not.toHaveAttribute("data-tone");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("data-accent");
  await expect(page.locator("#selection-heading")).toHaveText(
    "Comprehensive Agent Details",
  );
  for (const selector of [
    "#command-deck",
    "#roster-details",
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#events-details",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
  }
  for (const selector of [
    "#roster",
    "#event-feed",
    "#pending-card",
    "#accepted-card",
    "#diagnostics-card",
    "#visual-tooltip",
  ]) {
    await expect(page.locator(selector)).not.toContainText(oldScientificSentinel);
  }
  const pendingDomBytes = JSON.stringify(await authorityDomSnapshot(page));
  expect(pendingDomBytes).not.toContain(oldScientificSentinel);
  expect(pendingDomBytes).not.toContain(oldPresentationKey);
  await expect(page.locator(".combat-effect")).toHaveCount(0);
  await expect(page.locator(".combat-choreography-routes > *")).toHaveCount(0);
  for (const selector of [
    "#recording-finish-button",
    "#recording-review-button",
    "#recording-retry-button",
    "#recording-save-as-input",
    "#recording-save-as-button",
    "#recording-discard-confirm-button",
    "#view-select",
  ]) {
    await expect(page.locator(selector)).toBeDisabled();
  }
}

test("all five real service leaves install and live authority clears atomically", async ({
  page,
}) => {
  if (!liveDebugger || !noSharedReplay) {
    throw new Error("Authorized-presentation test services are unavailable.");
  }

  await openProductWithHeldInitialPresentation(
    page,
    liveDebugger.url,
    "live",
    "combat_debugger",
  );
  const liveOracle = await expectInstalledLeaf(
    page,
    "researcher_live_debugger",
    "live_oracle",
  );
  await expectAuthorizedRosterColors(page);
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Live Oracle View is interactive. Activate an authorized actor to control it; Shift-click selects an authorized target; right-click clears the target. Battlefield keyboard commands apply only while this surface has focus.",
  );
  await expect(page.locator("#pending-scope")).toHaveText(
    "This panel shows only the authorized pending draft for the next submission.",
  );
  await expect(
    page.locator("#battlefield-utilities > #live-ranges-button"),
  ).toHaveCount(1);
  await expect(page.locator("#live-ranges-button")).toHaveAttribute(
    "aria-description",
    "Toggle server-authored Oracle View range presentation.",
  );
  await expect(
    page.locator('#command-deck button[data-key="Tab"]:not([data-shift])'),
  ).toHaveAttribute(
    "aria-description",
    "Move Oracle View control to the next active actor.",
  );
  await expect(page.locator("#command-deck #live-ranges-button")).toHaveCount(0);
  await expect(page.locator("#battlefield-shell #live-ranges-button")).toHaveCount(0);
  await expect(page.locator("#live-visual-key > dt")).toHaveText([
    "Team A",
    "Team B",
    "Controlled",
    "Selected target",
  ]);
  await expect(page.locator("#live-visual-key")).not.toHaveAttribute("hidden", "");
  await expect(page.locator("#replay-visual-key")).toHaveAttribute("hidden", "");
  const oldScientificSentinel = `Agent ID ${liveOracle.presentation.current_endpoint.scene.agents[0].public_agent_id}`;
  const oldPresentationKey =
    liveOracle.presentation.current_endpoint.scene.agents[0].presentation_key;
  const controlledAgent = page.locator('#battlefield .agent[data-controlled="true"]');
  await expect(controlledAgent).toHaveCount(1);
  const controlledKey = await controlledAgent.getAttribute("data-presentation-key");
  const controlledAgentFacts =
    liveOracle.presentation.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) =>
        agent.presentation_key === controlledKey,
    );
  expect(controlledAgentFacts).toBeTruthy();
  await expectCertifiedDocumentationCard(page, controlledAgentFacts);
  const liveOracleDocumentationBefore = await page
    .locator("#selection-card")
    .evaluate((node) => node.innerHTML);
  await controlledAgent.hover();
  await expectCompactAgentTooltip(
    page,
    controlledAgentFacts,
    liveOracleDocumentationBefore,
  );
  await expectTechnicalFrameDom(page, liveOracle.presentation);
  await expectLatestTransitionDom(page, liveOracle.presentation);
  await expectRetiredMetadataAbsent(page);
  await openDetails(page, [
    "#agent-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]);
  await controlledAgent.hover();
  await expectCompactAgentTooltip(
    page,
    controlledAgentFacts,
    liveOracleDocumentationBefore,
  );
  await captureCp4ENativeState(page, {
    filename: "live-oracle-1440x900.png",
    width: 1440,
    height: 900,
    presentation: liveOracle.presentation,
    selectedAgent: controlledAgentFacts,
  });
  await cleanupAfterCp4ECapture(page, [
    "#agent-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]);
  const installedScientificBytes = JSON.stringify(await authorityDomSnapshot(page));
  expect(installedScientificBytes).toContain(oldScientificSentinel);
  expect(installedScientificBytes).toContain(oldPresentationKey);

  let releasePresentation = () => {};
  let markPresentationHeld = () => {};
  const presentationHeld = new Promise((resolve) => {
    markPresentationHeld = () => resolve(undefined);
  });
  const presentationRelease = new Promise((resolve) => {
    releasePresentation = () => resolve(undefined);
  });
  let holdNextPresentation = true;
  await page.route("**/api/presentation/frame", async (route) => {
    if (!holdNextPresentation) {
      await route.continue();
      return;
    }
    holdNextPresentation = false;
    const response = await route.fetch();
    markPresentationHeld();
    await presentationRelease;
    await route.fulfill({ response });
  });

  const switchAction = page.locator("#view-select").selectOption("pov");
  await presentationHeld;
  await expectPendingAuthorityIsEmpty(page, oldScientificSentinel, oldPresentationKey);
  releasePresentation();
  await switchAction;
  await page.unroute("**/api/presentation/frame");

  const liveAgent = await expectInstalledLeaf(
    page,
    "actor_pov_live_debugger",
    "live_no_shared_obs_agent_pov",
  );
  await expectTechnicalFrameDom(page, liveAgent.presentation);
  await expectLatestTransitionDom(page, liveAgent.presentation);
  await expectRetiredMetadataAbsent(page);
  await expectAgentAuthoritySurface(page, liveAgent.presentation, [oldPresentationKey]);
  await expect(page.locator("#battlefield")).not.toHaveAttribute(
    "aria-description",
    /.+/u,
  );
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-describedby",
    "battlefield-instructions",
  );
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Live Agent POV keeps one fixed recipient. Bodies are passive inspection targets; use the authorized draft controls to prepare that recipient's action.",
  );
  await page.locator("#help-button").click();
  await expect(page.locator("#help-dialog")).toBeVisible();
  const visibleLiveHelp = page.locator(
    "#help-dialog [data-live-help-mode]:not([hidden])",
  );
  await expect(visibleLiveHelp).toHaveCount(1);
  await expect(visibleLiveHelp).toHaveAttribute("data-live-help-mode", "agent");
  const agentHelpText = await visibleLiveHelp.innerText();
  expect(agentHelpText).toContain("fixed recipient");
  expect(agentHelpText).toContain("Toggle locally authorized ranges without a command");
  expect(agentHelpText).not.toContain("Cycle the controlled actor");
  expect(agentHelpText).not.toContain("Control the clicked authorized actor");
  expect(agentHelpText).not.toContain("Toggle Oracle View ranges");
  expect(agentHelpText).not.toContain("drafts persist when you cycle");
  await page.locator("#help-close-button").click();
  await expect(page.locator("#help-dialog")).toBeHidden();

  const agentRecipientKey = liveAgent.presentation.authority.recipient_presentation_key;
  const agentBodyKeys = await page
    .locator("#battlefield .agent")
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  const passiveAgentIndex = agentBodyKeys.findIndex(
    (presentationKey) => presentationKey !== agentRecipientKey,
  );
  expect(passiveAgentIndex).toBeGreaterThanOrEqual(0);
  const passiveAgentKey = agentBodyKeys[passiveAgentIndex];
  expect(typeof passiveAgentKey).toBe("string");
  const liveAgentScene =
    liveAgent.presentation.current_endpoint.scene ??
    liveAgent.presentation.current_endpoint.parts?.scene;
  const passiveAgent = liveAgentScene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key === passiveAgentKey,
  );
  expect(passiveAgent).toBeTruthy();
  const passiveAgentBody = page.locator(
    `#battlefield .agent[data-presentation-key="${passiveAgentKey}"]`,
  );
  const passiveRowButton = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${passiveAgentKey}"]`,
  );
  await page.setViewportSize({ width: 960, height: 600 });
  await expectNativeAgentActivationMatrix(page, {
    body: passiveAgentBody,
    row: passiveRowButton,
    path: null,
    command: null,
  });
  await expect(passiveAgentBody).toHaveAttribute("data-selected", "true");
  await expect(passiveRowButton).toHaveAttribute("aria-pressed", "true");
  await expectCertifiedDocumentationCard(page, passiveAgent);
  const liveAgentDocumentationBefore = await page
    .locator("#selection-card")
    .evaluate((node) => node.innerHTML);
  await passiveAgentBody.hover();
  await expectCompactAgentTooltip(page, passiveAgent, liveAgentDocumentationBefore);
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual([
    { kind: "observation", owner: passiveAgentKey },
    { kind: "basic", owner: passiveAgentKey },
    { kind: "ultimate", owner: passiveAgentKey },
  ]);
  await expect(page.locator("#pending-card .selected-legality__lane")).toHaveCount(0);
  await expect(page.locator("#pending-card .selected-outgoing-target")).toHaveCount(0);
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
  await openDetails(page, [
    "#agent-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]);
  await passiveAgentBody.hover();
  await expectCompactAgentTooltip(page, passiveAgent, liveAgentDocumentationBefore);
  await captureCp4ENativeState(page, {
    filename: "live-no-shared-agent-960x600.png",
    width: 960,
    height: 600,
    presentation: liveAgent.presentation,
    selectedAgent: passiveAgent,
  });
  await cleanupAfterCp4ECapture(page, [
    "#agent-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]);
  await expectAgentAuthoritySurface(page, liveAgent.presentation, [oldPresentationKey]);
  const recipientBody = page.locator(
    `#battlefield .agent[data-presentation-key="${agentRecipientKey}"]`,
  );
  await recipientBody.focus();
  await expectZeroCommandInteraction(page, () => recipientBody.press("Enter"));
  await expect(recipientBody).toHaveAttribute("data-selected", "true");
  const agentTabButtons = page.locator('#command-deck button[data-key="Tab"]');
  await expect(agentTabButtons).toHaveCount(2);
  expect(
    await agentTabButtons.evaluateAll((buttons) =>
      buttons.every((button) => button instanceof HTMLButtonElement && button.disabled),
    ),
  ).toBe(true);
  for (const button of await agentTabButtons.all()) {
    await expect(button).toHaveAttribute(
      "aria-description",
      /unavailable in Agent POV[\s\S]*native browser focus[\s\S]*fixed recipient/u,
    );
    await expect(button).not.toHaveAttribute(
      "aria-description",
      /Move Oracle View control/u,
    );
  }
  await page.locator("#battlefield").focus();
  await expectZeroCommandInteraction(page, () => page.keyboard.press("Tab"));
  await expect(page.locator("#battlefield")).not.toBeFocused();
  for (const button of await agentTabButtons.all()) {
    await expectZeroCommandInteraction(page, () =>
      button.evaluate((element) => {
        if (!(element instanceof HTMLButtonElement)) {
          throw new TypeError("Agent actor-cycle control is not a button.");
        }
        element.click();
      }),
    );
  }
  expect(
    (await authenticatedGet(page, "/api/presentation/frame")).authority
      .recipient_presentation_key,
  ).toBe(agentRecipientKey);
  const liveAgentRanges = page.locator("#live-ranges-button");
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");
  await expectZeroCommandInteraction(page, () => liveAgentRanges.click());
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expectZeroCommandInteraction(page, () => liveAgentRanges.click());
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");
  const currentAgentRanges = await page
    .locator('[data-layer="debug-range"] .range-ring')
    .evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    );
  expect(currentAgentRanges.length).toBeGreaterThan(0);
  expect(currentAgentRanges.every(({ owner }) => owner === agentRecipientKey)).toBe(
    true,
  );
  await page.locator("#battlefield").focus();
  await expectZeroCommandInteraction(page, () => page.keyboard.press("Shift+g"));
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual(currentAgentRanges);
  await expectZeroCommandInteraction(page, () => page.keyboard.press("g"));
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expectZeroCommandInteraction(page, () => page.keyboard.press("g"));
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual(currentAgentRanges);
  const agentPresentationAfterClick = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  expect(agentPresentationAfterClick.authority.recipient_presentation_key).toBe(
    agentRecipientKey,
  );
  const agentKeyboardRequest = page.waitForRequest(
    (request) =>
      request.method() === "POST" && new URL(request.url()).pathname === "/api/command",
  );
  const agentKeyboardResponse = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/command",
  );
  await page.locator("#battlefield").focus();
  await page.keyboard.press("x");
  const [keyboardRequest, keyboardResponse] = await Promise.all([
    agentKeyboardRequest,
    agentKeyboardResponse,
  ]);
  expect(keyboardRequest.postDataJSON().command).toMatchObject({
    command_type: "keyboard",
    key: "x",
  });
  expect(keyboardResponse.status()).toBe(200);
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  const agentPresentationAfterKeyboard = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  expect(agentPresentationAfterKeyboard.authority.recipient_presentation_key).toBe(
    agentRecipientKey,
  );

  await page.locator("#view-select").selectOption("researcher");
  const restoredOracle = await expectInstalledLeaf(
    page,
    "researcher_live_debugger",
    "live_oracle",
  );
  const oracleBody = page.locator("#battlefield .agent").first();
  const oracleKey = await oracleBody.getAttribute("data-presentation-key");
  const oracleAgent = restoredOracle.presentation.current_endpoint.scene.agents.find(
    (/** @type {Record<string, any>} */ agent) => agent.presentation_key === oracleKey,
  );
  const oracleIdentity =
    restoredOracle.presentation.current_endpoint.identity_directory.identities.find(
      (/** @type {Record<string, any>} */ identity) =>
        identity.public_agent_id === oracleAgent.public_agent_id,
    );
  const oracleSlot = (oracleIdentity.team_id - 1) * 5 + oracleIdentity.team_local_slot;
  const expectedOracleActivation = {
    command_type: "roster_selection",
    role: "control",
    global_slot: oracleSlot,
  };
  const oracleRow = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${oracleKey}"]`,
  );
  await expectNativeAgentActivationMatrix(page, {
    body: oracleBody,
    row: oracleRow,
    path: "/api/command",
    command: expectedOracleActivation,
  });
  const beforeScientificOwnerInput = await authenticatedGet(page, "/api/frame");
  const selectedKeysBeforeScientificOwnerInput = await page
    .locator('#battlefield .agent[data-selected="true"]')
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  const rangeOwner = page.locator("#battlefield .range-ring-owner").first();
  await expect(rangeOwner).toBeVisible();
  const scientificOwnerPoint = await page.locator("#battlefield").evaluate(() => {
    const owners = [
      ...document.querySelectorAll("#battlefield [data-tooltip-owner]"),
    ].filter((owner) => owner.closest(".agent") !== owner);
    for (const owner of owners) {
      const bounds = owner.getBoundingClientRect();
      for (const xRatio of [0.1, 0.25, 0.5, 0.75, 0.9]) {
        for (const yRatio of [0.1, 0.25, 0.5, 0.75, 0.9]) {
          const x = bounds.left + bounds.width * xRatio;
          const y = bounds.top + bounds.height * yRatio;
          const topOwner = document
            .elementsFromPoint(x, y)
            .map((element) => element.closest("[data-tooltip-owner]"))
            .find((candidate) => candidate !== null);
          if (topOwner === owner) {
            return { x, y };
          }
        }
      }
    }
    return null;
  });
  expect(scientificOwnerPoint).not.toBeNull();
  if (scientificOwnerPoint === null) {
    throw new Error("No exposed scientific tooltip owner was hit-testable.");
  }
  await expectZeroCommandInteraction(page, () =>
    page.mouse.click(scientificOwnerPoint.x, scientificOwnerPoint.y),
  );
  await rangeOwner.focus();
  await expectZeroCommandInteraction(page, () => rangeOwner.press("Enter"));
  await expectZeroCommandInteraction(page, () => rangeOwner.press(" "));
  await rangeOwner.focus();
  await expectZeroCommandInteraction(page, () => rangeOwner.press("Tab"));
  expect(
    await page
      .locator('#battlefield .agent[data-selected="true"]')
      .evaluateAll((agents) =>
        agents.map((agent) => agent.getAttribute("data-presentation-key")),
      ),
  ).toEqual(selectedKeysBeforeScientificOwnerInput);
  const afterScientificOwnerInput = await authenticatedGet(page, "/api/frame");
  expect(afterScientificOwnerInput.revision).toBe(beforeScientificOwnerInput.revision);
  const stalePresentation = restoredOracle.presentation;
  let commandCount = 0;
  let frameGetCount = 0;
  let presentationGetCount = 0;
  const countRaceRequests = (
    /** @type {import("@playwright/test").Request} */ request,
  ) => {
    const path = new URL(request.url()).pathname;
    if (request.method() === "POST" && path === "/api/command") {
      commandCount += 1;
    }
    if (request.method() === "GET" && path === "/api/frame") {
      frameGetCount += 1;
    }
  };
  page.on("request", countRaceRequests);
  await page.route("**/api/presentation/frame", async (route) => {
    presentationGetCount += 1;
    const response = await route.fetch();
    if (presentationGetCount === 1) {
      await route.fulfill({ response, json: stalePresentation });
      return;
    }
    await route.fulfill({ response });
  });
  await page.locator("#live-ranges-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "installed",
  );
  await expect(page.locator("#notice")).toContainText("command completed once");
  expect(commandCount).toBe(1);
  expect(frameGetCount).toBe(1);
  expect(presentationGetCount).toBe(2);
  await page.unroute("**/api/presentation/frame");
  page.off("request", countRaceRequests);

  await openProductWithHeldInitialPresentation(
    page,
    noSharedReplay.url,
    "replay",
    "replay_viewer",
  );
  const replayOracle = await expectInstalledLeaf(
    page,
    "researcher_replay_viewer",
    "replay_oracle",
  );
  await expectTechnicalFrameDom(page, replayOracle.presentation);
  await expectLatestTransitionDom(page, replayOracle.presentation);
  await expectRetiredMetadataAbsent(page);
  await expect(page.locator("#events-details")).toHaveAttribute("open", "");
  await expect(page.locator("#event-feed .event-item")).toHaveCount(0);
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Replay is read-only. Activate an authorized agent to inspect current facts and its recorded outgoing action; use the timeline to change frames.",
  );
  await expect(
    page.locator("#battlefield-utilities > #replay-ranges-button"),
  ).toHaveCount(1);
  await expect(page.locator("#replay-ranges-button")).toHaveAttribute(
    "aria-description",
    "Toggle recorded Oracle View range presentation.",
  );
  await expect(
    page.locator("#battlefield-utilities > #replay-clear-reference-button"),
  ).toHaveCount(1);
  await expect(page.locator("#replay-clear-reference-button")).toHaveAttribute(
    "aria-description",
    "Clear the selected Oracle View agent and its inspection highlight.",
  );
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

  const replaySelectedBody = page.locator('#battlefield .agent[role="button"]').first();
  const replaySelectedKey = await replaySelectedBody.getAttribute(
    "data-presentation-key",
  );
  const replaySelectedAgent =
    replayOracle.presentation.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) =>
        agent.presentation_key === replaySelectedKey,
    );
  const replaySelectedIdentity =
    replayOracle.presentation.current_endpoint.identity_directory.identities.find(
      (/** @type {Record<string, any>} */ identity) =>
        identity.public_agent_id === replaySelectedAgent.public_agent_id,
    );
  const replaySelectedSlot =
    (replaySelectedIdentity.team_id - 1) * 5 + replaySelectedIdentity.team_local_slot;
  const replaySelectedRow = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${replaySelectedKey}"]`,
  );
  await expectNativeAgentActivationMatrix(page, {
    body: replaySelectedBody,
    row: replaySelectedRow,
    path: "/api/replay/command",
    command: {
      command_type: "select_agent",
      selected_global_slot: replaySelectedSlot,
    },
  });
  await expect(page.locator("#replay-clear-reference-button")).toBeEnabled();
  const replayOracleSelected = await authenticatedGet(page, "/api/presentation/frame");
  await expectReplayInspectionDom(page, replayOracleSelected);
  const replayOracleDocumentationBefore = await page
    .locator("#selection-card")
    .evaluate((node) => node.innerHTML);
  await page.locator('#battlefield .agent[data-selected="true"]').hover();
  await expectCompactAgentTooltip(
    page,
    replaySelectedAgent,
    replayOracleDocumentationBefore,
  );

  await expectAuthorizedIncomingTransitionDom(page, replayOracle.presentation);
  expect(replayOracle.presentation.latest_events).toBeNull();
  await page.locator("#events-details > summary").click();
  await expect(page.locator("#events-details")).not.toHaveAttribute("open", "");
  await page.locator("#technical-frame-details > summary").click();
  await expect(page.locator("#technical-frame-details")).toHaveAttribute("open", "");
  if ((await page.locator("#agent-details").getAttribute("open")) === "") {
    await page.locator("#agent-details > summary").click();
  }
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await replaySelectedRow.focus();
  let replaySeekRequestCount = 0;
  const countReplaySeek = (
    /** @type {import("@playwright/test").Request} */ request,
  ) => {
    if (
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/api/replay/command"
    ) {
      replaySeekRequestCount += 1;
    }
  };
  page.on("request", countReplaySeek);
  try {
    await seekReplay(page, 1);
  } finally {
    page.off("request", countReplaySeek);
  }
  expect(replaySeekRequestCount).toBe(1);
  await expect(page.locator("#events-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#technical-frame-details")).toHaveAttribute("open", "");
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
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
    .toEqual({ surface: "roster", presentationKey: replaySelectedKey });
  const replayOracleMiddle = await authenticatedGet(page, "/api/presentation/frame");
  expect(replayOracleMiddle.presentation_kind).toBe("replay_oracle");
  await expectAuthorizedIncomingTransitionDom(page, replayOracleMiddle);
  await expectReplayInspectionDom(page, replayOracleMiddle);
  await expectTechnicalFrameDom(page, replayOracleMiddle);
  await expectLatestTransitionDom(page, replayOracleMiddle);
  await expectRetiredMetadataAbsent(page);
  expect(replayOracleMiddle.latest_events.incoming_transition_id).toBe(
    await page.locator("#transition-value").textContent(),
  );
  const replayOracleMiddleScene = replayOracleMiddle.current_endpoint.scene;
  const replayOracleMiddleOwner = replayOracleMiddleScene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key ===
        replayOracleMiddle.replay_inspection.actor_presentation_key &&
      agent.public_agent_id ===
        replayOracleMiddle.replay_inspection.actor_public_agent_id,
  );
  expect(replayOracleMiddleOwner).toBeTruthy();
  const replayOracleMiddleDocumentation = await page
    .locator("#selection-card")
    .evaluate((node) => node.innerHTML);
  await openDetails(page, [
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]);
  await page
    .locator(
      `#battlefield .agent[data-presentation-key="${replayOracleMiddleOwner.presentation_key}"]`,
    )
    .hover();
  await expectCompactAgentTooltip(
    page,
    replayOracleMiddleOwner,
    replayOracleMiddleDocumentation,
  );
  await captureCp4ENativeState(page, {
    filename: "replay-oracle-960x600.png",
    width: 960,
    height: 600,
    presentation: replayOracleMiddle,
    selectedAgent: replayOracleMiddleOwner,
  });
  await cleanupAfterCp4ECapture(page, [
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]);
  await seekReplay(page, replayOracle.presentation.source.source_final_frame_index);
  const replayOracleFinal = await authenticatedGet(page, "/api/presentation/frame");
  expect(replayOracleFinal.replay_inspection).toBeNull();
  await expectReplayInspectionDom(page, replayOracleFinal);
  await expectTechnicalFrameDom(page, replayOracleFinal);
  await expectLatestTransitionDom(page, replayOracleFinal);
  await expectTerminalAgentActivationInert(page);
  const nextShowRanges = replayOracle.transport.show_ranges !== true;
  await expectSingleReplayUtilityCommand(page, "#replay-ranges-button", {
    command_type: "set_ranges",
    show_ranges: nextShowRanges,
  });
  await expect(page.locator("#replay-ranges-button")).toHaveAttribute(
    "aria-pressed",
    String(nextShowRanges),
  );
  await expectSingleReplayUtilityCommand(page, "#replay-clear-reference-button", {
    command_type: "select_agent",
    selected_global_slot: null,
  });
  const replayOracleFinalUnselected = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  await expectReplayInspectionDom(page, replayOracleFinalUnselected);
  await seekReplay(page, 0);
  const oracleOnlyValues = [
    replayOracle.presentation.source.source_artifact_id,
    replayOracle.presentation.source.source_artifact_digest_sha256,
    replayOracle.presentation.source.source_trajectory_content_digest_sha256,
    replayOracle.presentation.source.source_context_digest_sha256,
    ...replayOracle.presentation.current_endpoint.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => agent.presentation_key,
    ),
  ].filter((value) => typeof value === "string");

  await page.locator("#view-select").selectOption("pov");
  const replayAgent = await expectInstalledLeaf(
    page,
    "actor_pov_replay_viewer",
    "replay_no_shared_obs_agent_pov",
  );
  await expectAgentAuthoritySurface(page, replayAgent.presentation, oracleOnlyValues);
  await expectReplayInspectionDom(page, replayAgent.presentation);
  await expectTechnicalFrameDom(page, replayAgent.presentation);
  await expectLatestTransitionDom(page, replayAgent.presentation);
  await expectRetiredMetadataAbsent(page);
  const replayAgentSelectedBody = page.locator(
    '#battlefield .agent[data-selected="true"]',
  );
  const replayAgentSelectedKey = await replayAgentSelectedBody.getAttribute(
    "data-presentation-key",
  );
  const replayAgentInitialScene =
    replayAgent.presentation.current_endpoint.scene ??
    replayAgent.presentation.current_endpoint.parts?.scene;
  const replayAgentSelected = replayAgentInitialScene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key === replayAgentSelectedKey,
  );
  expect(replayAgentSelected).toBeTruthy();
  const replayAgentInitialDocumentation = await page
    .locator("#selection-card")
    .evaluate((node) => node.innerHTML);
  await replayAgentSelectedBody.hover();
  await expectCompactAgentTooltip(
    page,
    replayAgentSelected,
    replayAgentInitialDocumentation,
  );
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Replay Agent POV is read-only and keeps one fixed recipient. Activate an authorized visible body to inspect current facts; the replay authority does not change.",
  );
  const replayAgentRecipientKey =
    replayAgent.presentation.authority.recipient_presentation_key;
  const replayAgentBodies = page.locator("#battlefield .agent[role=button]");
  const replayAgentBodyKeys = await replayAgentBodies.evaluateAll((agents) =>
    agents.map((agent) => agent.getAttribute("data-presentation-key")),
  );
  const replayAgentLocalKey = replayAgentBodyKeys.find(
    (key) => key !== replayAgentRecipientKey,
  );
  expect(typeof replayAgentLocalKey).toBe("string");
  const replayAgentScene =
    replayAgent.presentation.current_endpoint.scene ??
    replayAgent.presentation.current_endpoint.parts?.scene;
  const replayAgentLocal = replayAgentScene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key === replayAgentLocalKey,
  );
  expect(replayAgentLocal).toBeTruthy();
  const replayAgentLocalRow = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${replayAgentLocalKey}"]`,
  );
  const replayAgentLocalBody = page.locator(
    `#battlefield .agent[data-presentation-key="${replayAgentLocalKey}"]`,
  );
  await expectNativeAgentActivationMatrix(page, {
    body: replayAgentLocalBody,
    row: replayAgentLocalRow,
    path: null,
    command: null,
  });
  await expect(replayAgentLocalBody).toHaveAttribute("data-selected", "true");
  await expect(replayAgentLocalRow).toHaveAttribute("aria-pressed", "true");
  await expectCertifiedDocumentationCard(page, replayAgentLocal);
  await expect(page.locator("#pending-card .selected-legality__lane")).toHaveCount(0);
  await expect(page.locator("#pending-card .selected-outgoing-target")).toHaveCount(0);
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
  expect(
    await page.locator('[data-layer="debug-range"] .range-ring').evaluateAll((ranges) =>
      ranges.map((range) => ({
        kind: range.getAttribute("data-kind"),
        owner: range.getAttribute("data-presentation-key"),
      })),
    ),
  ).toEqual([
    { kind: "observation", owner: replayAgentLocalKey },
    { kind: "basic", owner: replayAgentLocalKey },
    { kind: "ultimate", owner: replayAgentLocalKey },
  ]);
  await expectAgentAuthoritySurface(page, replayAgent.presentation, oracleOnlyValues);
  const replayAgentRanges = page.locator("#replay-ranges-button");
  await expect(replayAgentRanges).toHaveAttribute("aria-pressed", "true");
  await expectZeroCommandInteraction(page, () => replayAgentRanges.click());
  await expect(replayAgentRanges).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expectZeroCommandInteraction(page, () => replayAgentRanges.click());
  await expectZeroCommandInteraction(page, () =>
    page.locator("#replay-clear-reference-button").click(),
  );
  await expect(page.locator('#battlefield .agent[data-selected="true"]')).toHaveCount(
    0,
  );
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#replay-clear-reference-button")).toBeDisabled();
  const replayAgentPresentationAfterLocalActions = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  expect(
    replayAgentPresentationAfterLocalActions.authority.recipient_presentation_key,
  ).toBe(replayAgentRecipientKey);
  await expect(page.locator("#view-select")).toHaveValue("pov");
  await expectZeroCommandInteraction(page, () =>
    page
      .locator(
        `#roster .roster-primary-action[data-presentation-key="${replayAgentRecipientKey}"]`,
      )
      .click(),
  );
  expect(replayAgent.presentation.source.source_frame_index).toBe(0);
  await expectAuthorizedIncomingTransitionDom(page, replayAgent.presentation);
  for (const frameIndex of [
    1,
    replayAgent.presentation.source.source_final_frame_index,
  ]) {
    await seekReplay(page, frameIndex);
    const presentation = await authenticatedGet(page, "/api/presentation/frame");
    expect(presentation.presentation_kind).toBe("replay_no_shared_obs_agent_pov");
    expect(presentation.source.source_frame_index).toBe(frameIndex);
    await expectAuthorizedIncomingTransitionDom(page, presentation);
    await expectReplayInspectionDom(page, presentation);
    await expectTechnicalFrameDom(page, presentation);
    await expectLatestTransitionDom(page, presentation);
    expect(presentation.latest_events.incoming_recipient_transition_id).toBe(
      await page.locator("#transition-value").textContent(),
    );
    if (frameIndex === replayAgent.presentation.source.source_final_frame_index) {
      await expectTerminalAgentActivationInert(page);
    }
  }
  expect(browserErrors.get(page) ?? []).toEqual([]);
});

test("real Shared replay installs frame zero, middle, final, then rejects a forged raw root", async ({
  page,
}) => {
  if (!sharedReplay) {
    throw new Error("Shared replay service is unavailable.");
  }
  await openProduct(page, sharedReplay.url, "replay");

  const installed = [];
  for (const frameIndex of [0, 1, 2]) {
    if (frameIndex !== 0) {
      await seekReplay(page, frameIndex);
    }
    const leaf = await expectInstalledLeaf(
      page,
      "shared_obs_agent_pov_replay_viewer",
      "replay_shared_obs_agent_pov",
    );
    expect(leaf.transport.cursor).toMatchObject({
      frame_index: frameIndex,
      final_frame_index: 2,
    });
    expect(leaf.presentation.source.source_frame_index).toBe(frameIndex);
    expect(leaf.presentation.source.source_final_frame_index).toBe(2);
    expect(leaf.presentation.source.source_recipient_frame_id).toBe(
      leaf.transport.recipient_frame_id,
    );
    await expectAgentAuthoritySurface(page, leaf.presentation, [], frameIndex < 2);
    await expectAuthorizedIncomingTransitionDom(page, leaf.presentation);
    await expectReplayInspectionDom(page, leaf.presentation);
    await expectTechnicalFrameDom(page, leaf.presentation);
    await expectLatestTransitionDom(page, leaf.presentation);
    await expectRetiredMetadataAbsent(page);
    if (frameIndex === 0) {
      await expect(page.locator("#events-details")).toHaveAttribute("open", "");
      await expect(page.locator("#event-feed .event-item")).toHaveCount(0);
      const recipientKey = leaf.presentation.authority.recipient_presentation_key;
      const scene =
        leaf.presentation.current_endpoint.scene ??
        leaf.presentation.current_endpoint.parts?.scene;
      const localAgent = scene.agents.find(
        (/** @type {Record<string, any>} */ agent) =>
          agent.presentation_key !== recipientKey,
      );
      expect(localAgent).toBeTruthy();
      const localKey = localAgent.presentation_key;
      const localBody = page.locator(
        `#battlefield .agent[data-presentation-key="${localKey}"]`,
      );
      const localRow = page.locator(
        `#roster .roster-primary-action[data-presentation-key="${localKey}"]`,
      );
      await expectNativeAgentActivationMatrix(page, {
        body: localBody,
        row: localRow,
        path: null,
        command: null,
      });
      await expect(localBody).toHaveAttribute("data-selected", "true");
      await expect(localRow).toHaveAttribute("aria-pressed", "true");
      await expectCertifiedDocumentationCard(page, localAgent);
      await expect(page.locator("#pending-card .selected-legality__lane")).toHaveCount(
        0,
      );
      await expect(page.locator("#pending-card .selected-outgoing-target")).toHaveCount(
        0,
      );
      await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
      expect(
        await page
          .locator('[data-layer="debug-range"] .range-ring')
          .evaluateAll((ranges) =>
            ranges.map((range) => ({
              kind: range.getAttribute("data-kind"),
              owner: range.getAttribute("data-presentation-key"),
            })),
          ),
      ).toEqual([
        { kind: "observation", owner: localKey },
        { kind: "basic", owner: localKey },
        { kind: "ultimate", owner: localKey },
      ]);
      await expectAgentAuthoritySurface(page, leaf.presentation, []);
      const rangesButton = page.locator("#replay-ranges-button");
      await expectZeroCommandInteraction(page, () => rangesButton.click());
      await expect(rangesButton).toHaveAttribute("aria-pressed", "false");
      await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(
        0,
      );
      await expectZeroCommandInteraction(page, () => rangesButton.click());
      await expect(rangesButton).toHaveAttribute("aria-pressed", "true");
      expect(
        await page
          .locator('[data-layer="debug-range"] .range-ring')
          .evaluateAll((ranges) =>
            ranges.map((range) => ({
              kind: range.getAttribute("data-kind"),
              owner: range.getAttribute("data-presentation-key"),
            })),
          ),
      ).toEqual([
        { kind: "observation", owner: localKey },
        { kind: "basic", owner: localKey },
        { kind: "ultimate", owner: localKey },
      ]);
      await expectZeroCommandInteraction(page, () =>
        page.locator("#replay-clear-reference-button").click(),
      );
      await expect(
        page.locator('#battlefield .agent[data-selected="true"]'),
      ).toHaveCount(0);
      await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(
        0,
      );
      await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
      await expect(page.locator("#replay-clear-reference-button")).toBeDisabled();
      await expectZeroCommandInteraction(page, () =>
        page
          .locator(
            `#roster .roster-primary-action[data-presentation-key="${recipientKey}"]`,
          )
          .click(),
      );
      const afterLocalActions = await authenticatedGet(page, "/api/presentation/frame");
      expect(afterLocalActions.authority.recipient_presentation_key).toBe(recipientKey);
    }
    if (frameIndex === 1) {
      const scene =
        leaf.presentation.current_endpoint.scene ??
        leaf.presentation.current_endpoint.parts?.scene;
      const owner = scene.agents.find(
        (/** @type {Record<string, any>} */ agent) =>
          agent.presentation_key ===
            leaf.presentation.replay_inspection.actor_presentation_key &&
          agent.public_agent_id ===
            leaf.presentation.replay_inspection.actor_public_agent_id,
      );
      expect(owner).toBeTruthy();
      const documentationBefore = await page
        .locator("#selection-card")
        .evaluate((node) => node.innerHTML);
      await openDetails(page, [
        "#agent-details",
        "#pending-turn-details",
        "#latest-transition-details",
        "#technical-frame-details",
      ]);
      await page
        .locator(
          `#battlefield .agent[data-presentation-key="${owner.presentation_key}"]`,
        )
        .hover();
      await expectCompactAgentTooltip(page, owner, documentationBefore);
      await captureCp4ENativeState(page, {
        filename: "replay-shared-agent-1440x900.png",
        width: 1440,
        height: 900,
        presentation: leaf.presentation,
        selectedAgent: owner,
      });
      await cleanupAfterCp4ECapture(page, [
        "#agent-details",
        "#pending-turn-details",
        "#latest-transition-details",
        "#technical-frame-details",
      ]);
    }
    if (frameIndex === 2) {
      await expectTerminalAgentActivationInert(page);
    }
    installed.push(leaf);
  }

  expect(installed[0].presentation.latest_events).toBeNull();
  expect(installed[0].presentation.latest_transition).toBeNull();
  expect(installed[1].presentation.latest_events).not.toBeNull();
  expect(installed[1].presentation.latest_transition).not.toBeNull();
  expect(installed[1].presentation.replay_inspection).not.toBeNull();
  expect(installed[2].presentation.latest_events).not.toBeNull();
  expect(installed[2].presentation.latest_transition).not.toBeNull();
  expect(installed[2].presentation.replay_inspection).toBeNull();

  await seekReplay(page, 1);
  const middlePresentation = await authenticatedGet(page, "/api/presentation/frame");
  const incomingCount =
    middlePresentation.latest_events.events?.length ??
    middlePresentation.latest_events.deltas?.length ??
    0;
  await expect(page.locator("#event-feed .event-item")).toHaveCount(incomingCount);
  await expect(page.locator("#accepted-card .accepted-action-row")).toHaveCount(
    middlePresentation.latest_transition.action_rows.length,
  );
  await expect(page.locator("#accepted-card")).not.toHaveAttribute(
    "data-transition-id",
    /.+/u,
  );
  await expect(
    page.locator("#accepted-card .accepted-action-row[data-presentation-key]"),
  ).toHaveCount(0);
  await expectLatestTransitionDom(page, middlePresentation);

  let frameRequestCount = 0;
  let presentationRequestCount = 0;
  const countRetiredRequests = (
    /** @type {import("@playwright/test").Request} */ request,
  ) => {
    const path = new URL(request.url()).pathname;
    if (request.method() === "GET" && path === "/api/frame") {
      frameRequestCount += 1;
    }
    if (request.method() === "GET" && path === "/api/presentation/frame") {
      presentationRequestCount += 1;
    }
  };
  page.on("request", countRetiredRequests);
  await page.route("**/api/frame", async (route) => {
    const response = await route.fetch();
    const current = await response.json();
    await route.fulfill({
      response,
      json: {
        schema_version: 1,
        frame_kind: "shared_obs_source_material_replay_viewer",
        revision: current.revision,
        projection: {
          researcher_hidden_truth: "retired-diagnostic-sentinel",
        },
      },
    });
  });
  await page.locator("#reconnect-button").click();
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect(page.locator("html")).toHaveAttribute(
    "data-presentation-authority",
    "pending",
  );
  expect(frameRequestCount).toBe(1);
  expect(presentationRequestCount).toBe(1);
  const middleScene =
    middlePresentation.current_endpoint.scene ??
    middlePresentation.current_endpoint.parts?.scene;
  const middleSentinel = `Agent ID ${middleScene.agents[0].public_agent_id}`;
  await expectPendingAuthorityIsEmpty(
    page,
    middleSentinel,
    middleScene.agents[0].presentation_key,
  );
  await expect(page.locator("[data-presentation-key]")).toHaveCount(0);
  await expect(page.locator("#replay-timeline")).toBeVisible();
  await expect(page.locator("#replay-artifact-reference")).toHaveText(
    "Unavailable while authority is pending",
  );
  for (const selector of [
    "#replay-first-button",
    "#replay-back-ten-button",
    "#replay-previous-button",
    "#replay-play-pause-button",
    "#replay-next-button",
    "#replay-forward-ten-button",
    "#replay-last-button",
    "#replay-frame-slider",
  ]) {
    await expect(page.locator(selector)).toBeDisabled();
  }
  await expect(page.locator("body")).not.toContainText("retired-diagnostic-sentinel");
  await expect(page.locator("body")).not.toContainText(
    /Shared\s*Obs.*Source Material/iu,
  );
  await page.unroute("**/api/frame");
  page.off("request", countRetiredRequests);
  expect(browserErrors.get(page) ?? []).toEqual([]);
});

test("real death and respawn keep one opaque Oracle body identity", async ({
  page,
}) => {
  if (!deathReplay) {
    throw new Error("Death replay service is unavailable.");
  }
  await openProduct(page, deathReplay.url, "replay");
  const initial = await expectInstalledLeaf(
    page,
    "researcher_replay_viewer",
    "replay_oracle",
  );
  const subject = initial.presentation.current_endpoint.scene.agents.find(
    (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
  );
  if (!subject) {
    throw new Error("Death replay subject is absent from frame zero.");
  }
  const subjectSelector = `.agent[data-presentation-key="${subject.presentation_key}"]`;

  await seekReplay(page, 1);
  const corpse = await authenticatedGet(page, "/api/presentation/frame");
  expect(
    corpse.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
    ).presentation_key,
  ).toBe(subject.presentation_key);
  await expect(page.locator(subjectSelector)).toHaveAttribute("data-alive", "false");

  await seekReplay(page, 3);
  const respawn = await authenticatedGet(page, "/api/presentation/frame");
  expect(
    respawn.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
    ).presentation_key,
  ).toBe(subject.presentation_key);
  await expect(page.locator(subjectSelector)).toHaveAttribute("data-alive", "true");
  await expect(page.locator(subjectSelector)).toHaveAttribute(
    "data-spawn-shield-remaining",
    "3",
  );
  expect(respawn.current_endpoint.scene.spawn_shield_mechanics.availability_kind).toBe(
    "available_v2",
  );

  const shield = page.locator(
    `.agent-spawn-shield[data-presentation-key="${subject.presentation_key}"]`,
  );
  await expect(page.locator(subjectSelector)).toHaveAttribute("role", "button");
  await expect(page.locator(subjectSelector)).toHaveAttribute("tabindex", "0");
  expect(
    await page.locator(subjectSelector).evaluate((agent, presentationKey) => {
      const shieldNode = document.querySelector(
        `.agent-spawn-shield[data-presentation-key="${CSS.escape(String(presentationKey))}"]`,
      );
      return {
        contains: shieldNode instanceof Element && agent.contains(shieldNode),
        sameParent:
          shieldNode instanceof Element &&
          agent.parentElement === shieldNode.parentElement,
      };
    }, subject.presentation_key),
  ).toEqual({ contains: false, sameParent: true });
  await expect(shield).toBeVisible();
  await expect(shield).toHaveAttribute("tabindex", "0");
  await expect(shield).toHaveAttribute("aria-label", "Spawn Shield");
  await expect(shield).toHaveAttribute(
    "aria-description",
    "While the spawn shield is active, this agent is protected, concealed from opponents, untargetable, excluded from aura effects, and limited to movement. It phases through agents until body collision resumes at the endpoint of its expiring transition.",
  );
  expect(await page.locator(subjectSelector).getAttribute("aria-label")).not.toMatch(
    /Spawn Shield|invulnerable|concealed|untargetable/iu,
  );

  await expectZeroCommandInteraction(page, () => shield.focus());
  await expect(shield).toBeFocused();
  await expectZeroCommandInteraction(page, () => shield.press("Enter"));
  await expectZeroCommandInteraction(page, () => shield.press(" "));
  await expectZeroCommandInteraction(page, () =>
    shield.locator(".agent-spawn-shield__chip").click(),
  );
  await shield.focus();
  await expect(page.locator("#visual-tooltip")).toBeVisible();
  await expect(page.locator("#visual-tooltip-title")).toHaveText("Spawn Shield");
  await expect(
    page.locator("#visual-tooltip .semantic-explanation__summary"),
  ).toHaveText(
    "While the spawn shield is active, this agent is protected, concealed from opponents, untargetable, excluded from aura effects, and limited to movement. It phases through agents until body collision resumes at the endpoint of its expiring transition.",
  );
  await expect(page.locator("#visual-tooltip .semantic-explanation__label")).toHaveText(
    [
      "Protection Effect",
      "Movement Speed",
      "Visibility Effect",
      "Targetability Effect",
      "Action Effect",
      "Aura Effect",
      "Agent Collision Effect",
      "Effect Duration",
      "Duration Remaining",
      "Owner",
      "Source",
      "Ordinary Application",
    ],
  );
  await expect(page.locator("#visual-tooltip .semantic-explanation__value")).toHaveText(
    [
      "Invulnerable",
      "2",
      "Concealed from opponents",
      "Untargetable",
      "Movement only",
      "Excluded as emitter and beneficiary",
      "Phased until expiring endpoint rejoin",
      "3 Ticks",
      "3 Ticks",
      "Agent ID 5 · Rogue · Team B",
      "Not recorded",
      "End-of-transition respawn lifecycle",
    ],
  );
  const shieldStyles = await shield.evaluate((root) => {
    const shell = root.querySelector(".agent-spawn-shield__shell");
    const chip = root.querySelector(".agent-spawn-shield__chip");
    const ticks = root.querySelector(".agent-spawn-shield__ticks");
    if (!(shell instanceof SVGElement) || !(chip instanceof SVGElement)) {
      throw new TypeError("Spawn Shield shell or chip is unavailable.");
    }
    if (!(ticks instanceof SVGElement)) {
      throw new TypeError("Spawn Shield tick label is unavailable.");
    }
    return {
      chipFill: getComputedStyle(chip).fill,
      chipStroke: getComputedStyle(chip).stroke,
      shellStroke: getComputedStyle(shell).stroke,
      tickFill: getComputedStyle(ticks).fill,
    };
  });
  expect(shieldStyles).toEqual({
    chipFill: "rgb(0, 0, 0)",
    chipStroke: "rgb(255, 255, 255)",
    shellStroke: "rgb(255, 255, 255)",
    tickFill: "rgb(255, 255, 255)",
  });
  for (const [frameIndex, remaining] of [
    [3, 3],
    [4, 2],
    [5, 1],
    [6, 0],
  ]) {
    if (frameIndex !== 3) {
      await seekReplay(page, frameIndex);
    }
    const presentation =
      frameIndex === 3
        ? respawn
        : await authenticatedGet(page, "/api/presentation/frame");
    const currentSubject = presentation.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
    );
    expect(currentSubject.presentation_key).toBe(subject.presentation_key);
    expect(currentSubject.spawn_shield_remaining).toBe(remaining);
    await expect(page.locator(subjectSelector)).toHaveAttribute(
      "data-spawn-shield-remaining",
      String(remaining),
    );
    await expect(shield.locator(".agent-spawn-shield__ticks")).toHaveText(
      `S${remaining}`,
    );
    if (remaining > 0) {
      await expect(shield).toBeVisible();
      await expect(shield).toHaveAttribute("tabindex", "0");
    } else {
      await expect(shield).toBeHidden();
      await expect(shield).toHaveAttribute("hidden", "");
      await expect(shield).toHaveAttribute("tabindex", "-1");
      await expect(shield).not.toHaveAttribute("data-tooltip-owner", "");
      await expect(shield).not.toHaveAttribute("aria-describedby", /./u);
      await expect(shield).not.toHaveAttribute("aria-description", /./u);
      await expect(shield).not.toBeFocused();
      await expect(page.locator("#visual-tooltip")).not.toHaveAttribute(
        "data-tooltip-kind",
        "status",
      );
      await expect(page.locator("#visual-tooltip-title")).not.toHaveText(
        "Spawn Shield",
      );
      expect(
        await page.locator(subjectSelector).getAttribute("aria-label"),
      ).not.toMatch(/Spawn Shield/u);
    }
  }
  expect(browserErrors.get(page) ?? []).toEqual([]);
});
