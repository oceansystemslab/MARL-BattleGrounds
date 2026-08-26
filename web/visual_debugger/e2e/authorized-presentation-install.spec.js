import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";

import { expect, test } from "@playwright/test";
import { CHOREOGRAPHY_ROOT, installWaapiAutopause } from "./support/choreography.js";
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
const CP5_C_SLICE_TEST_TITLE =
  "real moving crossfire and recovery replay expose durable IC and packed +4 regeneration";
const cp5CSliceOnly = process.env.MARL_CP5_C_SLICE_ONLY === "1";
const CP5_SLICE_5_TEST_TITLE =
  "six real scientific trajectories preserve public causality and hidden-root privacy";
const cp5Slice5Only = process.env.MARL_CP5_SLICE_5_ONLY === "1";
const isolatedCp5Proof = cp5CSliceOnly || cp5Slice5Only;
const CP5_SLICE_5_CHECKED_SAMPLE_FILES = Object.freeze([
  "manifest.json",
  "death-respawn-shield.marlbg-replay.json",
  "death-respawn-shield.marlbg-metrics.json",
  "recovery-status-lifecycle.marlbg-replay.json",
  "recovery-status-lifecycle.marlbg-metrics.json",
  "mirrored-five-class-ultimates.marlbg-replay.json",
  "mirrored-five-class-ultimates.marlbg-metrics.json",
]);

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
  if (isolatedCp5Proof) {
    return;
  }
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
  if (isolatedCp5Proof) {
    if (
      artifacts !== null ||
      liveDebugger !== null ||
      noSharedReplay !== null ||
      sharedReplay !== null ||
      deathReplay !== null
    ) {
      throw new Error(
        "Isolated CP5 proof mode must not start shared legacy presentation services or artifact export.",
      );
    }
    return;
  }
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

test.beforeEach(({ browserName: _browserName }, testInfo) => {
  test.skip(
    cp5CSliceOnly && testInfo.title !== CP5_C_SLICE_TEST_TITLE,
    "CP5 Slice C-only mode permits only its self-contained two-service proof.",
  );
  test.skip(
    cp5Slice5Only && testInfo.title !== CP5_SLICE_5_TEST_TITLE,
    "CP5 Slice 5-only mode permits only its self-contained causal/privacy proof.",
  );
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

async function cp5Slice5CheckedSampleSnapshot() {
  /** @type {Record<string, {bytes: Buffer, sha256: string}>} */
  const snapshot = {};
  for (const fileName of CP5_SLICE_5_CHECKED_SAMPLE_FILES) {
    const path = join(REPOSITORY_ROOT, "examples", "replays", "v1", fileName);
    const bytes = await readFile(path);
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    snapshot[fileName] = { bytes, sha256 };
  }
  return snapshot;
}

/**
 * Read the exact authorized response body so hidden raw-transport mutations
 * can be proved byte-inert rather than merely object-equal.
 *
 * @param {import("@playwright/test").Page} page
 */
async function cp5Slice5PresentationBody(page) {
  return page.evaluate(async () => {
    const token = window.sessionStorage.getItem("marl-battlegrounds.debugger-token");
    if (!token) {
      throw new Error("Debugger capability token is unavailable.");
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
    return response.text();
  });
}

/**
 * Project one real authorized response through the pure planner and collect
 * only stable semantic DOM state from the installed product.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} rawPresentation
 */
async function cp5Slice5PlanAndDomSignature(page, rawPresentation) {
  return page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { authorizedPresentationIncomingRows } = await import(
      `${moduleRoot}/authorized-presentation-adapter.js`
    );
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const serializedBefore = JSON.stringify(presentation);
    const surface = Object.freeze({
      /** @param {Record<string, any> | number[]} point */
      worldToScreen: (point) => {
        const x = Array.isArray(point) ? point[0] : point.x;
        const y = Array.isArray(point) ? point[1] : point.y;
        return { x: Number(x) * 10, y: Number(y) * 10 };
      },
      /** @param {number} length */
      worldLengthToScreen: (length) => Number(length) * 10,
      viewportBounds: Object.freeze({
        left: 0,
        top: 0,
        right: 4096,
        bottom: 4096,
        width: 4096,
        height: 4096,
      }),
      protectedRects: Object.freeze([]),
    });
    const incomingRows = /** @type {Array<Record<string, any>>} */ (
      authorizedPresentationIncomingRows(presentation)
    );
    const plan = buildChoreographyPlan(presentation, surface);
    if (!plan && incomingRows.length > 0) {
      throw new Error("Real authorized incoming rows produced no choreography plan.");
    }
    const planEvents =
      plan === null ? [] : /** @type {Array<Record<string, any>>} */ (plan.events);
    const orderedEventIds = incomingRows.map(({ id }) => id);
    const plannedAtomicIds = planEvents.flatMap((event) =>
      Array.isArray(event.atomicEventIds) ? event.atomicEventIds : [event.eventId],
    );
    if (JSON.stringify(plannedAtomicIds) !== JSON.stringify(orderedEventIds)) {
      throw new Error("Authorized plan lost, duplicated, or reordered an incoming ID.");
    }
    const statusLifecycles = [];
    for (const effect of document.querySelectorAll(
      ".combat-effect--status-lifecycle[data-tooltip-owner]",
    )) {
      const hit = effect.querySelector(".combat-lifecycle__hit");
      if (!(hit instanceof SVGElement)) {
        throw new Error("Installed status lifecycle lost its hit target.");
      }
      hit.dispatchEvent(new FocusEvent("focusin", { bubbles: true }));
      await new Promise((resolveFrame) => requestAnimationFrame(resolveFrame));
      statusLifecycles.push({
        id: effect.getAttribute("data-event-id"),
        type: effect.getAttribute("data-event-type"),
        tokenId: effect.getAttribute("data-token-id"),
        lifecycle: effect.getAttribute("data-lifecycle"),
        persistent: effect.getAttribute("data-persistent") === "true",
        atomicEventIds: JSON.parse(
          effect.getAttribute("data-atomic-event-ids") ?? "null",
        ),
        applicationEventIds: JSON.parse(
          effect.getAttribute("data-application-event-ids") ?? "null",
        ),
        tooltipKind:
          document
            .querySelector("#visual-tooltip")
            ?.getAttribute("data-tooltip-kind") ?? null,
        tooltipTitle:
          document.querySelector("#visual-tooltip-title")?.textContent ?? null,
        tooltipSummary:
          document.querySelector("#visual-tooltip .semantic-explanation__summary")
            ?.textContent ?? null,
      });
      hit.dispatchEvent(
        new FocusEvent("focusout", { bubbles: true, relatedTarget: null }),
      );
    }
    return {
      inputUnchanged: JSON.stringify(presentation) === serializedBefore,
      plan: {
        authorizationKey: plan?.authorizationKey ?? null,
        epochKey: plan?.epochKey ?? null,
        fingerprint: plan?.fingerprint ?? null,
        transitionId: plan?.transitionId ?? null,
        events: planEvents.map((event) => ({
          eventId: event.eventId,
          eventType: event.eventType,
          transitionId: event.transitionId,
          authorityVocabulary: event.authorityVocabulary,
          kind: event.kind,
          spatial: event.spatial,
          presentationSuppressed: event.presentationSuppressed ?? false,
          persistent: event.persistent ?? false,
          tokenId: event.tokenId ?? null,
          lifecycle: event.lifecycle ?? null,
          lifecycleLabel: event.lifecycleToken?.label ?? null,
          lifecycleAccessibleName: event.lifecycleToken?.accessibleName ?? null,
          atomicEventIds: event.atomicEventIds ?? null,
          applicationEventIds: event.applicationEventIds ?? null,
          sourcePresentationKey: event.sourcePresentationKey ?? null,
          sourcePublicAgentId: event.sourcePublicAgentId ?? null,
          targetPresentationKey: event.targetPresentationKey ?? null,
          targetPublicAgentId: event.targetPublicAgentId ?? null,
          recipientPresentationKey: event.recipientPresentationKey ?? null,
          recipientPublicAgentId: event.recipientPublicAgentId ?? null,
          agentPresentationKey: event.agentPresentationKey ?? null,
          agentPublicAgentId: event.agentPublicAgentId ?? null,
          source: event.source ?? null,
          target: event.target ?? null,
          start: event.start ?? null,
          end: event.end ?? null,
          route: event.route ?? null,
        })),
      },
      dom: {
        authority: document.documentElement.dataset.presentationAuthority ?? null,
        removedEventSurfaceCount: document.querySelectorAll(
          "#events-details, #event-feed, #event-count",
        ).length,
        agents: [...document.querySelectorAll("#battlefield .agent")].map((agent) => ({
          key: agent.getAttribute("data-presentation-key"),
          alive: agent.getAttribute("data-alive"),
          shield: agent.getAttribute("data-spawn-shield-remaining"),
          statuses: [...agent.querySelectorAll(".status-cell")].map((status) => ({
            token: status.getAttribute("data-token-id"),
            duration: status.getAttribute("data-duration"),
          })),
        })),
        effects: [...document.querySelectorAll(".combat-effect")].map((effect) => ({
          id: effect.getAttribute("data-event-id"),
          type: effect.getAttribute("data-event-type"),
          persistent: effect.getAttribute("data-persistent") === "true",
          sourceKey: effect.getAttribute("data-source-presentation-key"),
          targetKey: effect.getAttribute("data-target-presentation-key"),
          recipientKey: effect.getAttribute("data-recipient-presentation-key"),
          agentKey: effect.getAttribute("data-agent-presentation-key"),
        })),
        statusLifecycles,
        routes: [
          ...document.querySelectorAll(".combat-choreography-routes [data-event-id]"),
        ].map((route) => ({
          id: route.getAttribute("data-event-id"),
          type: route.getAttribute("data-event-type"),
          persistent: route.getAttribute("data-persistent") === "true",
          sourceKey: route.getAttribute("data-source-presentation-key"),
          targetKey: route.getAttribute("data-target-presentation-key"),
          recipientKey: route.getAttribute("data-recipient-presentation-key"),
          agentKey: route.getAttribute("data-agent-presentation-key"),
        })),
        transition: document.querySelector("#transition-value")?.textContent ?? null,
      },
    };
  }, rawPresentation);
}

/**
 * Reinstall one real leaf with a schema-valid private raw transport mutation.
 * The certified presentation response is deliberately left untouched.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} productUrl
 * @param {string} expectedPresentationKind
 */
async function cp5Slice5AssertHiddenTransportNoninterference(
  page,
  productUrl,
  expectedPresentationKind,
) {
  await page.goto("about:blank");
  await openProduct(page, productUrl, "replay");
  const baselineTransport = await authenticatedGet(page, "/api/frame");
  const baselinePresentationBody = await cp5Slice5PresentationBody(page);
  const baselinePresentation = JSON.parse(baselinePresentationBody);
  expect(baselinePresentation.presentation_kind).toBe(expectedPresentationKind);
  const baselineSignature = await cp5Slice5PlanAndDomSignature(
    page,
    baselinePresentation,
  );
  expect(baselineSignature.inputUnchanged).toBe(true);

  await page.goto("about:blank");
  let mutationCount = 0;
  await page.route("**/api/frame", async (route) => {
    const response = await route.fetch();
    const transport = await response.json();
    if (expectedPresentationKind === "replay_oracle") {
      transport.processing = {
        ...transport.processing,
        status: "failed",
        failure_stage: "lifecycle",
        failure_code: "cp5_hidden_transport_probe",
        attempted_transition_index: null,
      };
    } else {
      transport.cursor = {
        ...transport.cursor,
        cursor_generation: transport.cursor.cursor_generation + 1,
      };
    }
    mutationCount += 1;
    await route.fulfill({ response, json: transport });
  });
  try {
    await openProduct(page, productUrl, "replay");
    const mutatedTransport = await authenticatedGet(page, "/api/frame");
    const mutatedPresentationBody = await cp5Slice5PresentationBody(page);
    const mutatedPresentation = JSON.parse(mutatedPresentationBody);
    if (expectedPresentationKind === "replay_oracle") {
      expect(mutatedTransport.processing).toEqual({
        ...baselineTransport.processing,
        status: "failed",
        failure_stage: "lifecycle",
        failure_code: "cp5_hidden_transport_probe",
        attempted_transition_index: null,
      });
      expect({
        ...mutatedTransport,
        processing: baselineTransport.processing,
      }).toEqual(baselineTransport);
    } else {
      expect(mutatedTransport.cursor.cursor_generation).toBe(
        baselineTransport.cursor.cursor_generation + 1,
      );
      expect({
        ...mutatedTransport,
        cursor: baselineTransport.cursor,
      }).toEqual(baselineTransport);
    }
    expect(mutatedPresentationBody).toBe(baselinePresentationBody);
    expect(mutatedPresentation.presentation_kind).toBe(expectedPresentationKind);
    const mutatedSignature = await cp5Slice5PlanAndDomSignature(
      page,
      mutatedPresentation,
    );
    expect(mutatedSignature).toEqual(baselineSignature);
  } finally {
    await page.unroute("**/api/frame");
  }
  expect(mutationCount).toBeGreaterThan(0);
  await page.goto("about:blank");
  await openProduct(page, productUrl, "replay");
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
    ...(agent.steps_until_out_of_combat > 0 ? ["Steps until out of combat"] : []),
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
  ).toHaveText(agent.steps_until_out_of_combat > 0 ? "In combat" : "Out of combat");
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
    "Steps until out of combat",
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
  episode: Object.freeze({
    label: "Episode",
    summary: "Identifies the authorized live episode represented by this frame.",
  }),
  artifact_digest_prefix: Object.freeze({
    label: "Artifact digest prefix",
    summary:
      "These 12 hexadecimal characters locate the canonical Oracle replay without displaying its full hash.",
  }),
  frame: Object.freeze({
    label: "Frame",
    summary: "The zero-based authorized frame index represented by this presentation.",
  }),
  simulator_step: Object.freeze({
    label: "Simulator step",
    summary: "The simulator decision step represented by this authorized frame.",
  }),
  incoming_transition: Object.freeze({
    label: "Incoming transition",
    summary:
      "Identifies the authorized transition that produced this displayed frame. The initial frame has no incoming transition.",
  }),
  ordinary_movement_distance_scale: Object.freeze({
    label: "Ordinary movement distance scale",
    summary:
      "The recorded multiplier applied to ordinary voluntary movement distance. Spawn Shield uses its separately authorized absolute movement speed.",
  }),
});

const FORBIDDEN_TECHNICAL_FACT_IDS = Object.freeze([
  "technical_kind",
  "session",
  "revision",
  "generation",
  "cursor_generation",
  "choreography_generation",
  "artifact_id",
  "artifact_digest",
  "context_digest",
  "trajectory_digest",
  "timeline",
  "processing",
  "completion",
  "metrics",
  "metric_report",
  "path",
  "global_slot",
  "presentation_key",
]);

/**
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 */
async function expectTechnicalFrameDom(page, presentation) {
  const liveResearcherTechnical =
    presentation.presentation_kind === "live_no_shared_obs_agent_pov"
      ? presentation.researcher_space?.technical_frame
      : null;
  const technical = liveResearcherTechnical ?? presentation.technical_frame;
  const technicalDetailsWasOpen =
    (await page.locator("#technical-frame-details").getAttribute("open")) === "";
  if (!technicalDetailsWasOpen) {
    await openDetails(page, ["#technical-frame-details"]);
  }
  /** @type {Record<string, Array<[keyof typeof TECHNICAL_HELP, string]>>} */
  const specifications = {
    live_oracle: [
      ["episode", "episode_id"],
      ["frame", "evaluation_frame_index"],
      ["simulator_step", "simulator_step_count"],
      ["incoming_transition", "incoming_transition_id"],
    ],
    live_no_shared_obs_agent_pov: [
      ["episode", "episode_id"],
      ["frame", "recipient_frame_index"],
      ["simulator_step", "simulator_step_count"],
      ["incoming_transition", "incoming_recipient_transition_id"],
    ],
    replay_oracle: [
      ["artifact_digest_prefix", "artifact_digest_prefix"],
      ["frame", "frame_index"],
      ["simulator_step", "simulator_step_count"],
      ["incoming_transition", "incoming_transition_id"],
      ["ordinary_movement_distance_scale", "recorded_ordinary_movement_distance_scale"],
    ],
    replay_no_shared_obs_agent_pov: [
      ["frame", "frame_index"],
      ["simulator_step", "simulator_step_count"],
      ["incoming_transition", "incoming_recipient_transition_id"],
    ],
    replay_shared_obs_agent_pov: [
      ["frame", "frame_index"],
      ["simulator_step", "simulator_step_count"],
      ["incoming_transition", "incoming_recipient_transition_id"],
    ],
  };
  const specification =
    specifications[
      liveResearcherTechnical === null ? presentation.presentation_kind : "live_oracle"
    ];
  if (!specification) {
    throw new TypeError(
      `Unsupported Technical Frame presentation kind ${String(presentation.presentation_kind)}.`,
    );
  }
  const expected = specification.filter(([, field]) => technical[field] !== null);
  const frameField = specification.find(([id]) => id === "frame")?.[1];
  const incomingField = specification.find(([id]) => id === "incoming_transition")?.[1];
  if (frameField === undefined || incomingField === undefined) {
    throw new TypeError("Technical Frame proof is missing its epoch fields.");
  }
  if (technical[frameField] === 0) {
    expect(technical[incomingField]).toBeNull();
  } else {
    expect(technical[incomingField]).toEqual(expect.any(String));
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
  for (const forbiddenId of FORBIDDEN_TECHNICAL_FACT_IDS) {
    await expect(
      page.locator(`#diagnostics-card .fact[data-technical-fact="${forbiddenId}"]`),
    ).toHaveCount(0);
  }
  const technicalText = await page.locator("#diagnostics-card").innerText();
  expect(technicalText).not.toContain(String(technical.technical_kind));
  for (const sourceField of [
    "source_session_id",
    "source_artifact_id",
    "source_artifact_digest_sha256",
    "source_trajectory_content_digest_sha256",
    "source_context_digest_sha256",
    "source_timeline_id",
  ]) {
    const value = presentation.source?.[sourceField];
    if (typeof value === "string" && value.length > 0) {
      expect(technicalText).not.toContain(value);
    }
  }
  const scene =
    presentation.current_endpoint?.scene ??
    presentation.current_endpoint?.parts?.scene ??
    null;
  const forbiddenPresentationKey = scene?.agents?.[0]?.presentation_key;
  if (typeof forbiddenPresentationKey === "string") {
    expect(technicalText).not.toContain(forbiddenPresentationKey);
  }
  for (const [id] of expected) {
    const owner = page.locator(`#diagnostics-card .fact[data-technical-fact="${id}"]`);
    await expect(owner).toHaveAttribute("aria-description", TECHNICAL_HELP[id].summary);
    await owner.hover();
    await expect(page.locator("#visual-tooltip-title")).toHaveText(
      TECHNICAL_HELP[id].label,
    );
    await expect(
      page.locator("#visual-tooltip .semantic-explanation__summary"),
    ).toHaveText(TECHNICAL_HELP[id].summary);
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
  const researcher = presentation.researcher_space ?? null;
  const rawRows =
    (researcher?.latest_transition ?? presentation.latest_transition)?.action_rows ??
    [];
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
  const identities = researcher?.roster_agents ?? scene.agents;
  for (const [index, row] of rawRows.entries()) {
    const identity = identities.find(
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
 * Prove one native interaction emits exactly one product command while letting
 * the caller validate runtime-derived pointer coordinates.
 *
 * @param {import("@playwright/test").Page} page
 * @param {"/api/command" | "/api/replay/command"} path
 * @param {() => Promise<void>} activate
 * @param {(command: Record<string, any>) => void} verifyCommand
 */
async function expectSingleCommandMatching(page, path, activate, verifyCommand) {
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
    expect((await responsePromise).status()).toBe(200);
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
  } finally {
    page.off("request", record);
  }
  expect(requests).toHaveLength(1);
  expect(new URL(requests[0].url()).pathname).toBe(path);
  verifyCommand(requests[0].postDataJSON().command);
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
 * Pointer activation restores the debugger's battlefield-owned command focus
 * and never exposes the oversized SVG-agent focus outline.
 *
 * @param {import("@playwright/test").Page} page
 */
async function expectBattlefieldRootCommandFocus(page) {
  await expect(page.locator("#battlefield")).toBeFocused();
  await expect(page.locator("#battlefield .agent:focus-visible")).toHaveCount(0);
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
 * A terminal replay has no outgoing transition, but its frozen snapshot stays
 * researcher-selectable without advancing the cursor.
 *
 * @param {import("@playwright/test").Page} page
 */
async function expectTerminalReplayAgentSelection(page) {
  await expect(page.locator("#battlefield")).toHaveAttribute("role", "group");
  await expect(page.locator("#battlefield")).toHaveAttribute("tabindex", "-1");
  await expect(page.locator("#battlefield")).toHaveAttribute(
    "aria-label",
    "Read-only terminal replay battlefield snapshot.",
  );
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "This replay frame is terminal. Activate an agent to inspect current facts, or use the timeline to review another frame.",
  );
  const before = await authenticatedGet(page, "/api/presentation/frame");
  expect(before.presentation_kind).toBe("replay_oracle");
  expect(before.source.source_frame_index).toBe(before.source.source_final_frame_index);
  const selectedKey = await page
    .locator('#battlefield .agent[data-selected="true"]')
    .getAttribute("data-presentation-key");
  const candidate = before.current_endpoint.scene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key !== selectedKey,
  );
  expect(candidate).toBeTruthy();
  const identity = before.current_endpoint.identity_directory.identities.find(
    (/** @type {Record<string, any>} */ row) =>
      row.public_agent_id === candidate.public_agent_id,
  );
  expect(identity).toBeTruthy();
  const globalSlot = (identity.team_id - 1) * 5 + identity.team_local_slot;
  const body = page.locator(
    `#battlefield .agent[data-presentation-key="${candidate.presentation_key}"]`,
  );
  const row = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${candidate.presentation_key}"]`,
  );
  await expect(body).toHaveAttribute("role", "button");
  await expect(row).toBeEnabled();
  await expectSingleActivationCommand(page, "/api/replay/command", () => body.click(), {
    command_type: "select_agent",
    selected_global_slot: globalSlot,
  });
  const after = await authenticatedGet(page, "/api/presentation/frame");
  expect(after.source.source_frame_index).toBe(before.source.source_frame_index);
  expect(after.current_endpoint.action_axis.owner_public_agent_id).toBe(
    candidate.public_agent_id,
  );
  await expect(body).toHaveAttribute("data-selected", "true");
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
 * Prove the retained researcher-space session fact uses the global incoming
 * transition while battlefield choreography keeps its audience-local ID.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} presentation
 */
async function expectAuthorizedIncomingTransitionDom(page, presentation) {
  const latestEvents = presentation.latest_events;
  const researcherSpace = presentation.researcher_space;
  const oracle =
    presentation.presentation_kind === "live_oracle" ||
    presentation.presentation_kind === "replay_oracle";
  const expected =
    researcherSpace !== undefined
      ? (researcherSpace.latest_transition?.incoming_transition_id ?? null)
      : latestEvents === null
        ? null
        : oracle
          ? latestEvents.incoming_transition_id
          : latestEvents.incoming_recipient_transition_id;
  if (expected !== null) {
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
  const researcher = presentation.researcher_space ?? null;
  const endpoint = presentation.current_endpoint;
  const axis = endpoint.action_axis;
  const scene = endpoint.scene ?? endpoint.parts?.scene;
  expect(Array.isArray(scene?.agents)).toBe(true);
  const ownerPublicId =
    researcher?.selected_public_agent_id ??
    inspection?.actor_public_agent_id ??
    axis?.owner_public_agent_id ??
    null;
  const owner =
    typeof ownerPublicId === "string"
      ? (scene.agents.find(
          (/** @type {Record<string, any>} */ agent) =>
            agent.public_agent_id === ownerPublicId,
        ) ?? null)
      : null;

  await expect(page.locator("#selection-heading")).toHaveText(
    "Comprehensive Agent Class Details",
  );
  await expect(page.locator('[data-layer="selection-legality"]')).toHaveAttribute(
    "aria-label",
    "Selection and exact actor-owned legality",
  );
  await expect(page.locator("#accepted-heading")).toHaveText("Latest Transition");
  await expect(page.locator("#visual-key > summary")).toHaveText("Visual Key");
  await expect(page.locator("#pending-heading")).toHaveText("Upcoming Transition");
  await expect(page.locator("#pending-count")).toBeHidden();
  await expect(page.locator("#pending-scope")).toBeHidden();
  const upcoming = researcher?.upcoming_transition ?? presentation.upcoming_transition;
  const rawUpcomingRows = upcoming?.action_rows ?? [];
  const upcomingRows = page.locator("#pending-card .accepted-action-row");
  await expect(upcomingRows).toHaveCount(rawUpcomingRows.length);
  await expect(
    page.locator("#pending-card .accepted-action-row[data-presentation-key]"),
  ).toHaveCount(0);
  const tupleText = (/** @type {Record<string, any>} */ action) =>
    `Move ${action.move_action} · Target ${action.target_action} · Ultimate ${action.use_ultimate_action}`;
  for (const [index, row] of rawUpcomingRows.entries()) {
    const identities = researcher?.roster_agents ?? scene.agents;
    const identity = identities.find(
      (/** @type {Record<string, any>} */ candidate) =>
        candidate.presentation_key === row.actor_presentation_key &&
        candidate.public_agent_id === row.actor_public_agent_id,
    );
    expect(identity).toBeTruthy();
    const expectedIdentity = expectedAgentIdentity(identity);
    const rendered = upcomingRows.nth(index);
    await expect(rendered.locator(".accepted-action-row__title")).toHaveText(
      expectedIdentity.title,
    );
    await expect(rendered.locator(".accepted-action-row__title")).toHaveAttribute(
      "data-class",
      expectedIdentity.accent,
    );
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
    await expect(rendered.locator(".accepted-action-tuple__value")).toHaveText([
      tupleText(row.submitted_action),
      tupleText(row.accepted_action),
    ]);
  }
  await expect(page.locator("#pending-card .selected-legality__lane")).toHaveCount(0);
  await expect(page.locator("#pending-card .selected-outgoing-target")).toHaveCount(0);
  if (owner === null) {
    await expect(page.locator('#battlefield .agent[data-selected="true"]')).toHaveCount(
      0,
    );
    await expect(page.locator("#agent-details")).not.toHaveAttribute("data-accent");
    await expect(page.locator("#selection-card")).toContainText(
      "No authorized agent details are available.",
    );
    await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
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
    await expect(page.locator("#battlefield .legality-dock")).toHaveCount(0);
    await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
    return;
  }

  const inspectionRow = rawUpcomingRows.find(
    (/** @type {Record<string, any>} */ row) =>
      row.actor_public_agent_id === inspection.actor_public_agent_id &&
      (researcher !== null ||
        row.actor_presentation_key === inspection.actor_presentation_key),
  );
  expect(inspectionRow).toBeTruthy();
  expect(inspectionRow.submitted_action).toEqual(inspection.submitted_action);
  expect(inspectionRow.accepted_action).toEqual(inspection.accepted_action);

  const targetAction = inspection.accepted_action.target_action;
  const exactRow =
    inspection.decision_mask.target_use_ultimate_joint_mask[targetAction];
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
    { available: String(targetAction > 0 && exactRow[0]), lane: "0" },
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
  const researcherRoster = presentation.researcher_space?.roster_agents ?? null;
  const expectedRosterCount = researcherRoster?.length ?? bodyKeys.length;
  await expect(page.locator("#roster .roster-row--authorized")).toHaveCount(
    expectedRosterCount,
  );
  await expect(page.locator("#roster .roster-primary-action")).toHaveCount(
    expectedRosterCount,
  );
  if (researcherRoster !== null) {
    if (presentation.presentation_kind.startsWith("replay_")) {
      await expect(page.locator('#roster [data-visibility="visible"]')).toBeVisible();
      await expect(
        page.locator('#roster [data-visibility="not-visible"]'),
      ).toBeVisible();
    } else {
      await expect(page.locator('#roster [data-visibility="visible"]')).toBeHidden();
      await expect(
        page.locator('#roster [data-visibility="not-visible"]'),
      ).toBeHidden();
      await expect(page.locator("#roster .roster-team[data-team-id]")).toHaveCount(2);
      await expect(page.locator("#roster-count")).toHaveText(
        `${expectedRosterCount} visible`,
      );
    }
    expect(expectedRosterCount).toBeGreaterThanOrEqual(bodyKeys.length);
  }
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
  const localUtilityDescription =
    presentation.product_kind === "replay_viewer"
      ? /locally authorized|local inspected-agent/u
      : /fog-authorized/u;
  for (const selector of agentUtilitySelectors) {
    await expect(page.locator(selector)).toHaveAttribute(
      "aria-description",
      localUtilityDescription,
    );
    await expect(page.locator(selector)).toHaveAttribute(
      "aria-description",
      /sends no (?:replay )?command/u,
    );
    await expect(page.locator(selector)).not.toHaveAttribute(
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
    element.dispatchEvent(new Event("change", { bubbles: true }));
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
    "Comprehensive Agent Class Details",
  );
  for (const selector of [
    "#command-deck",
    "#roster-details",
    "#agent-details",
    "#pending-turn-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]) {
    await expect(page.locator(selector)).not.toHaveAttribute("open", "");
  }
  for (const selector of [
    "#roster",
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

test("all five real service leaves install with safe pending continuity", async ({
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
    "Live Oracle View is interactive. Activate an authorized actor to control it; Shift-click selects an authorized target; Escape clears the target and leaves battlefield focus. Battlefield keyboard commands apply only while this surface has focus.",
  );
  await expect(page.locator("#pending-scope")).toHaveText(
    "This panel shows the selected actor's authorized pending draft for the next submission within the global joint turn.",
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
    "Live Agent POV is interactive. Activate a visible authorized actor or choose any active actor in Roster to control it and switch POV; Shift-click selects a visible authorized target; Escape clears the target and leaves battlefield focus.",
  );
  await expect(page.locator("#pending-scope")).toHaveText(
    "This panel shows the selected actor's authorized pending draft for the next submission within the global joint turn.",
  );
  await page.locator("#help-button").click();
  await expect(page.locator("#help-dialog")).toBeVisible();
  const visibleLiveHelp = page.locator(
    "#help-dialog [data-live-help-mode]:not([hidden])",
  );
  await expect(visibleLiveHelp).toHaveCount(1);
  await expect(visibleLiveHelp).toHaveAttribute("data-live-help-mode", "agent");
  const agentHelpText = await visibleLiveHelp.innerText();
  expect(agentHelpText).not.toContain("fixed recipient");
  expect(agentHelpText).toContain("Toggle fog-authorized ranges without a command");
  expect(agentHelpText).toContain("Cycle the controlled actor and Agent POV");
  expect(agentHelpText).toContain("Control that actor and switch Agent POV");
  expect(agentHelpText).not.toContain("Toggle Oracle View ranges");
  expect(agentHelpText).toContain("drafts persist when you cycle");
  await page.locator("#help-close-button").click();
  await expect(page.locator("#help-dialog")).toBeHidden();

  const initialAgentRecipientKey =
    liveAgent.presentation.authority.recipient_presentation_key;
  const agentBodyKeys = await page
    .locator("#battlefield .agent")
    .evaluateAll((agents) =>
      agents.map((agent) => agent.getAttribute("data-presentation-key")),
    );
  const liveAgentScene =
    liveAgent.presentation.current_endpoint.scene ??
    liveAgent.presentation.current_endpoint.parts?.scene;
  const liveResearcher = liveAgent.presentation.researcher_space;
  expect(liveResearcher.researcher_space_kind).toBe("global_live_researcher_space");
  expect(liveResearcher.latest_transition).toBeNull();
  expect(liveAgent.presentation.latest_transition).toBeNull();
  expect(liveResearcher.technical_frame.technical_kind).toBe(
    "live_oracle_technical_frame",
  );
  expect(liveResearcher.pending_inspection.submission_scope).toBe("joint_turn");
  await expect(page.locator("#roster .roster-row--authorized")).toHaveCount(
    liveResearcher.roster_agents.length,
  );
  await expect(page.locator('#roster [data-visibility="visible"]')).toBeHidden();
  await expect(page.locator('#roster [data-visibility="not-visible"]')).toBeHidden();
  await expect(page.locator("#roster .roster-team[data-team-id]")).toHaveCount(2);
  await expect(page.locator("#roster-count")).toHaveText(
    `${liveResearcher.roster_agents.length} visible`,
  );
  await expect(page.locator("#command-target-select")).toBeEnabled();
  await expect(page.locator("#command-target-select option")).toHaveCount(11);
  await expect(page.locator("#submit-turn-button")).toHaveText("Submit joint turn");
  await expect(page.locator("#submit-turn-button")).toBeEnabled();

  const recipientBody = page.locator(
    `#battlefield .agent[data-presentation-key="${initialAgentRecipientKey}"]`,
  );
  const recipientAgent = liveAgentScene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.presentation_key === initialAgentRecipientKey,
  );
  expect(recipientAgent).toBeTruthy();
  await page.setViewportSize({ width: 960, height: 600 });
  await expectCertifiedDocumentationCard(page, recipientAgent);
  const liveAgentDocumentationBefore = await page
    .locator("#selection-card")
    .evaluate((node) => node.innerHTML);
  await recipientBody.hover();
  await expectCompactAgentTooltip(page, recipientAgent, liveAgentDocumentationBefore);
  await expect(page.locator("#battlefield .pending-route")).toHaveCount(0);
  await openDetails(page, [
    "#agent-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]);
  await recipientBody.hover();
  await expectCompactAgentTooltip(page, recipientAgent, liveAgentDocumentationBefore);
  await captureCp4ENativeState(page, {
    filename: "live-no-shared-agent-960x600.png",
    width: 960,
    height: 600,
    presentation: liveAgent.presentation,
    selectedAgent: recipientAgent,
  });
  await cleanupAfterCp4ECapture(page, [
    "#agent-details",
    "#latest-transition-details",
    "#technical-frame-details",
  ]);
  await expectAgentAuthoritySurface(page, liveAgent.presentation, [oldPresentationKey]);

  const liveAgentRanges = page.locator("#live-ranges-button");
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");
  await expectZeroCommandInteraction(page, () => liveAgentRanges.click());
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expectZeroCommandInteraction(page, () => liveAgentRanges.click());
  await expect(liveAgentRanges).toHaveAttribute("aria-pressed", "true");

  const agentTabButtons = page.locator('#command-deck button[data-key="Tab"]');
  await expect(agentTabButtons).toHaveCount(2);
  expect(
    await agentTabButtons.evaluateAll((buttons) =>
      buttons.every(
        (button) => button instanceof HTMLButtonElement && !button.disabled,
      ),
    ),
  ).toBe(true);
  for (const button of await agentTabButtons.all()) {
    await expect(button).toHaveAttribute(
      "aria-description",
      /Move Agent POV control[\s\S]*Staged drafts are preserved/u,
    );
  }

  const visiblePublicIds = new Set(
    liveAgentScene.agents.map(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id,
    ),
  );
  const hiddenAgent = liveResearcher.roster_agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      !visiblePublicIds.has(agent.public_agent_id),
  );
  expect(hiddenAgent).toBeTruthy();
  const hiddenIdentity = liveResearcher.identity_directory.identities.find(
    (/** @type {Record<string, any>} */ identity) =>
      identity.public_agent_id === hiddenAgent.public_agent_id,
  );
  expect(hiddenIdentity).toBeTruthy();
  const hiddenSlot = (hiddenIdentity.team_id - 1) * 5 + hiddenIdentity.team_local_slot;
  const hiddenRow = page.locator(
    `#roster .roster-primary-action[data-presentation-key="${hiddenAgent.presentation_key}"]`,
  );
  await expect(hiddenRow).toBeEnabled();

  let releaseAgentPresentation = () => {};
  let markAgentPresentationHeld = () => {};
  const agentPresentationHeld = new Promise((resolve) => {
    markAgentPresentationHeld = () => resolve(undefined);
  });
  const agentPresentationRelease = new Promise((resolve) => {
    releaseAgentPresentation = () => resolve(undefined);
  });
  await page.route("**/api/presentation/frame", async (route) => {
    const response = await route.fetch();
    markAgentPresentationHeld();
    await agentPresentationRelease;
    await route.fulfill({ response });
  });
  const hiddenSwitch = expectSingleActivationCommand(
    page,
    "/api/command",
    () => hiddenRow.click(),
    {
      command_type: "roster_selection",
      role: "control",
      global_slot: hiddenSlot,
    },
  );
  try {
    await agentPresentationHeld;
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "retained",
    );
    expect(
      await page
        .locator("#battlefield .agent")
        .evaluateAll((agents) =>
          agents.map((agent) => agent.getAttribute("data-presentation-key")),
        ),
    ).toEqual(agentBodyKeys);
    await expect(page.locator("#battlefield-empty")).toBeHidden();
    await expect(page.locator("#command-target-select")).toBeDisabled();
  } finally {
    releaseAgentPresentation();
  }
  await hiddenSwitch;
  await page.unroute("**/api/presentation/frame");
  let switchedAgent = await expectInstalledLeaf(
    page,
    "actor_pov_live_debugger",
    "live_no_shared_obs_agent_pov",
  );
  expect(switchedAgent.presentation.authority.recipient_public_agent_id).toBe(
    hiddenAgent.public_agent_id,
  );
  expect(switchedAgent.presentation.source.source_simulator_step_count).toBe(
    liveAgent.presentation.source.source_simulator_step_count,
  );

  const switchedScene =
    switchedAgent.presentation.current_endpoint.scene ??
    switchedAgent.presentation.current_endpoint.parts?.scene;
  const visibleControl = switchedScene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.public_agent_id !==
      switchedAgent.presentation.authority.recipient_public_agent_id,
  );
  expect(visibleControl).toBeTruthy();
  const visibleControlIdentity =
    switchedAgent.presentation.researcher_space.identity_directory.identities.find(
      (/** @type {Record<string, any>} */ identity) =>
        identity.public_agent_id === visibleControl.public_agent_id,
    );
  expect(visibleControlIdentity).toBeTruthy();
  const visibleControlSlot =
    (visibleControlIdentity.team_id - 1) * 5 + visibleControlIdentity.team_local_slot;
  const visibleControlBody = page.locator(
    `#battlefield .agent[data-presentation-key="${visibleControl.presentation_key}"]`,
  );
  await expectSingleActivationCommand(
    page,
    "/api/command",
    () => visibleControlBody.click(),
    {
      command_type: "roster_selection",
      role: "control",
      global_slot: visibleControlSlot,
    },
  );
  await expectBattlefieldRootCommandFocus(page);
  switchedAgent = await expectInstalledLeaf(
    page,
    "actor_pov_live_debugger",
    "live_no_shared_obs_agent_pov",
  );
  expect(switchedAgent.presentation.authority.recipient_public_agent_id).toBe(
    visibleControl.public_agent_id,
  );

  await expectSingleActivationCommand(
    page,
    "/api/command",
    () => page.keyboard.press("Tab"),
    {
      command_type: "keyboard",
      key: "Tab",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    },
  );
  await expectBattlefieldRootCommandFocus(page);
  await expectSingleActivationCommand(
    page,
    "/api/command",
    () => page.keyboard.press("Shift+Tab"),
    {
      command_type: "keyboard",
      key: "Tab",
      shift_key: true,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    },
  );
  switchedAgent = await expectInstalledLeaf(
    page,
    "actor_pov_live_debugger",
    "live_no_shared_obs_agent_pov",
  );

  const targetScene =
    switchedAgent.presentation.current_endpoint.scene ??
    switchedAgent.presentation.current_endpoint.parts?.scene;
  const visibleTarget = targetScene.agents.find(
    (/** @type {Record<string, any>} */ agent) =>
      agent.public_agent_id !==
      switchedAgent.presentation.authority.recipient_public_agent_id,
  );
  expect(visibleTarget).toBeTruthy();
  const visibleTargetBody = page.locator(
    `#battlefield .agent[data-presentation-key="${visibleTarget.presentation_key}"]`,
  );
  await expectSingleCommandMatching(
    page,
    "/api/command",
    () => visibleTargetBody.click({ modifiers: ["Shift"] }),
    (command) => {
      expect(command).toMatchObject({
        command_type: "battlefield_pointer",
        button: "primary",
        shift_key: true,
        ctrl_key: false,
        alt_key: false,
        meta_key: false,
      });
      expect(Number.isFinite(command.world_x)).toBe(true);
      expect(Number.isFinite(command.world_y)).toBe(true);
    },
  );

  const targetSelect = page.locator("#command-target-select");
  const currentTargetValue = await targetSelect.inputValue();
  const alternateTargetValue = await targetSelect
    .locator("option")
    .evaluateAll(
      (options, current) =>
        options
          .map((option) => option.getAttribute("value") ?? "")
          .find((value) => value.startsWith("pov-target-action:") && value !== current),
      currentTargetValue,
    );
  if (typeof alternateTargetValue !== "string") {
    throw new Error("Live Agent global target selector has no alternate target.");
  }
  expect(alternateTargetValue).toMatch(/^pov-target-action:\d+$/u);
  const alternateTargetAction = Number(alternateTargetValue.split(":").at(-1));
  await expectSingleActivationCommand(
    page,
    "/api/command",
    async () => {
      await targetSelect.selectOption(alternateTargetValue);
    },
    {
      command_type: "actor_pov_target_action",
      target_action: alternateTargetAction,
    },
  );
  await expectSingleActivationCommand(
    page,
    "/api/command",
    async () => {
      await targetSelect.selectOption("pov-target-action:0");
    },
    {
      command_type: "actor_pov_target_action",
      target_action: 0,
    },
  );
  await expect(targetSelect).toHaveValue("pov-target-action:0");
  await page.locator("#battlefield").focus();
  await expectBattlefieldRootCommandFocus(page);

  const agentStepBeforeW = await page.locator("#step-value").innerText();
  await expect(
    page.locator('#command-deck button[data-move-action="1"]'),
  ).toBeEnabled();
  await expectSingleActivationCommand(
    page,
    "/api/command",
    () => page.keyboard.press("w"),
    {
      command_type: "keyboard",
      key: "w",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    },
  );
  await expect(
    page.locator('#command-deck button[data-move-action="1"]'),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#step-value")).toHaveText(agentStepBeforeW);
  const recipientBeforeSubmit = (
    await authenticatedGet(page, "/api/presentation/frame")
  ).authority.recipient_public_agent_id;
  await expect(page.locator("#submit-turn-button")).toHaveText("Submit joint turn");
  await expectSingleActivationCommand(
    page,
    "/api/command",
    () => page.locator("#submit-turn-button").click(),
    {
      command_type: "keyboard",
      key: "Enter",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    },
  );
  const agentPresentationAfterSubmit = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  expect(
    agentPresentationAfterSubmit.researcher_space.latest_transition.action_rows.length,
  ).toBe(agentPresentationAfterSubmit.researcher_space.roster_agents.length);
  expect(agentPresentationAfterSubmit.latest_transition.action_rows).toHaveLength(1);
  expect(
    agentPresentationAfterSubmit.source.source_simulator_step_count,
  ).toBeGreaterThan(Number(agentStepBeforeW));
  expect(agentPresentationAfterSubmit.authority.recipient_public_agent_id).toBe(
    recipientBeforeSubmit,
  );
  await expectTechnicalFrameDom(page, agentPresentationAfterSubmit);

  await page.locator("#view-select").selectOption("researcher");
  const restoredOracle = await expectInstalledLeaf(
    page,
    "researcher_live_debugger",
    "live_oracle",
  );
  await expectTechnicalFrameDom(page, restoredOracle.presentation);
  const oracleCandidate = page
    .locator('#battlefield .agent:not([data-controlled="true"])')
    .first();
  const oracleKey = await oracleCandidate.getAttribute("data-presentation-key");
  if (oracleKey === null) {
    throw new Error("Oracle activation candidate has no presentation key.");
  }
  const oracleBody = page.locator(
    `#battlefield .agent[data-presentation-key="${oracleKey}"]`,
  );
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
  const oracleStepBeforePointerSelections = await page
    .locator("#step-value")
    .innerText();
  const pointerAgents = restoredOracle.presentation.current_endpoint.scene.agents.slice(
    0,
    3,
  );
  for (const pointerAgent of pointerAgents) {
    const pointerIdentity =
      restoredOracle.presentation.current_endpoint.identity_directory.identities.find(
        (/** @type {Record<string, any>} */ identity) =>
          identity.public_agent_id === pointerAgent.public_agent_id,
      );
    if (!pointerIdentity) {
      throw new Error("Oracle pointer candidate has no authorized identity row.");
    }
    const pointerSlot =
      (pointerIdentity.team_id - 1) * 5 + pointerIdentity.team_local_slot;
    const pointerBody = page.locator(
      `#battlefield .agent[data-presentation-key="${pointerAgent.presentation_key}"]`,
    );
    await expectSingleActivationCommand(
      page,
      "/api/command",
      () => pointerBody.click(),
      {
        command_type: "roster_selection",
        role: "control",
        global_slot: pointerSlot,
      },
    );
    await expectBattlefieldRootCommandFocus(page);
    await expect(pointerBody).toHaveAttribute("data-controlled", "true");
    await expect(page.locator("#step-value")).toHaveText(
      oracleStepBeforePointerSelections,
    );
  }
  const oracleStepBeforeW = oracleStepBeforePointerSelections;
  await expect(
    page.locator('#command-deck button[data-move-action="1"]'),
  ).toBeEnabled();
  await expectSingleActivationCommand(
    page,
    "/api/command",
    () => page.keyboard.press("w"),
    {
      command_type: "keyboard",
      key: "w",
      shift_key: false,
      ctrl_key: false,
      alt_key: false,
      meta_key: false,
      repeat: false,
    },
  );
  await expect(
    page.locator('#command-deck button[data-move-action="1"]'),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#step-value")).toHaveText(oracleStepBeforeW);
  const modifierTokens = await page
    .locator("#battlefield .modifier-cell")
    .evaluateAll((cells) =>
      [
        ...new Set(
          cells.map((cell) => {
            const icon = cell.querySelector(".modifier-cell__icon");
            return JSON.stringify({
              tokenId: cell.getAttribute("data-token-id"),
              token: cell.getAttribute("data-token"),
              glyph: icon?.getAttribute("data-icon") ?? null,
            });
          }),
        ),
      ]
        .map((value) => JSON.parse(value))
        .sort((left, right) => left.tokenId.localeCompare(right.tokenId)),
    );
  expect(modifierTokens).toEqual([
    {
      tokenId: "mage_amplification",
      token: "mage-amplification",
      glyph: "modifier-amplification",
    },
    {
      tokenId: "warrior_mitigation",
      token: "warrior-mitigation",
      glyph: "modifier-mitigation",
    },
  ]);
  await expect(
    page.locator('#battlefield .modifier-cell[data-token="unknown"]'),
  ).toHaveCount(0);
  await expect(
    page.locator('#battlefield .modifier-cell__icon[data-icon="unknown"]'),
  ).toHaveCount(0);
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
  await expect(page.locator("#events-details, #event-feed, #event-count")).toHaveCount(
    0,
  );
  await expect(page.locator("#battlefield-instructions")).toHaveText(
    "Replay is read-only. Upcoming Transition shows the authorized recorded joint action out of this frame; activate an agent to inspect current facts, or use the timeline to change frames.",
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
  await expectTerminalReplayAgentSelection(page);
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
  await expectSingleActivationCommand(
    page,
    "/api/replay/command",
    () => replaySelectedRow.click(),
    {
      command_type: "select_agent",
      selected_global_slot: replaySelectedSlot,
    },
  );
  const oracleOnlyValues = [
    replayOracle.presentation.source.source_artifact_id,
    replayOracle.presentation.source.source_artifact_digest_sha256,
    replayOracle.presentation.source.source_trajectory_content_digest_sha256,
    replayOracle.presentation.source.source_context_digest_sha256,
    ...replayOracle.presentation.current_endpoint.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => agent.presentation_key,
    ),
  ].filter((value) => typeof value === "string");

  let releaseViewPresentation = () => {};
  let markViewPresentationHeld = () => {};
  const viewPresentationHeld = new Promise((resolve) => {
    markViewPresentationHeld = () => resolve(undefined);
  });
  const viewPresentationRelease = new Promise((resolve) => {
    releaseViewPresentation = () => resolve(undefined);
  });
  await page.route("**/api/presentation/frame", async (route) => {
    const response = await route.fetch();
    markViewPresentationHeld();
    await viewPresentationRelease;
    await route.fulfill({ response });
  });
  await page.locator("#view-select").selectOption("pov");
  try {
    await viewPresentationHeld;
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "retained",
    );
    await expect(page.locator("#battlefield-empty")).toBeHidden();
    await expect(page.locator("#battlefield .agent")).not.toHaveCount(0);
  } finally {
    releaseViewPresentation();
  }
  const replayAgent = await expectInstalledLeaf(
    page,
    "actor_pov_replay_viewer",
    "replay_no_shared_obs_agent_pov",
  );
  await page.unroute("**/api/presentation/frame");
  const replayAgentArtifactReference =
    replayAgent.transport.artifact_facts.artifact_summary.replay_reference;
  const replayAgentArtifactShellValues = new Set([
    replayAgentArtifactReference.artifact_id,
    replayAgentArtifactReference.canonical_digest_sha256,
  ]);
  const replayAgentBattlefieldForbiddenValues = oracleOnlyValues.filter(
    (value) => !replayAgentArtifactShellValues.has(value),
  );
  await expectAgentAuthoritySurface(
    page,
    replayAgent.presentation,
    replayAgentBattlefieldForbiddenValues,
  );
  await expectReplayInspectionDom(page, replayAgent.presentation);
  await expectTechnicalFrameDom(page, replayAgent.presentation);
  await expectLatestTransitionDom(page, replayAgent.presentation);
  await expectRetiredMetadataAbsent(page);
  expect(replayAgent.presentation.authority.recipient_public_agent_id).toBe(
    replaySelectedAgent.public_agent_id,
  );
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
    "Replay Agent POV is read-only. Activate a visible body or choose any agent in the roster to switch to that agent's fog-of-war view at the same replay tick.",
  );
  const replayAgentRecipientKey =
    replayAgent.presentation.authority.recipient_presentation_key;
  const replayAgentBodies = page.locator("#battlefield .agent[role=button]");
  const replayAgentBodyKeys = await replayAgentBodies.evaluateAll((agents) =>
    agents.map((agent) => agent.getAttribute("data-presentation-key")),
  );
  const replayAgentScene =
    replayAgent.presentation.current_endpoint.scene ??
    replayAgent.presentation.current_endpoint.parts?.scene;
  const replayAgentVisibleIds = new Set(
    replayAgentScene.agents.map(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id,
    ),
  );
  const replayAgentNotVisible =
    replayAgent.presentation.researcher_space.roster_agents.find(
      (/** @type {Record<string, any>} */ agent) =>
        !replayAgentVisibleIds.has(agent.public_agent_id),
    );
  expect(replayAgentNotVisible).toBeTruthy();
  const replayAgentNotVisibleIdentity =
    replayAgent.presentation.researcher_space.identity_directory.identities.find(
      (/** @type {Record<string, any>} */ identity) =>
        identity.public_agent_id === replayAgentNotVisible.public_agent_id,
    );
  expect(replayAgentNotVisibleIdentity).toBeTruthy();
  const replayAgentNotVisibleSlot =
    (replayAgentNotVisibleIdentity.team_id - 1) * 5 +
    replayAgentNotVisibleIdentity.team_local_slot;
  const replayAgentNotVisibleRow = page.locator(
    `#roster [data-visibility="not-visible"] .roster-primary-action[data-presentation-key="${replayAgentNotVisible.presentation_key}"]`,
  );
  await expect(replayAgentNotVisibleRow).toBeEnabled();
  const replayAgentCursorBeforeSwitch = structuredClone(replayAgent.transport.cursor);
  let releasePovPresentation = () => {};
  let markPovPresentationHeld = () => {};
  const povPresentationHeld = new Promise((resolve) => {
    markPovPresentationHeld = () => resolve(undefined);
  });
  const povPresentationRelease = new Promise((resolve) => {
    releasePovPresentation = () => resolve(undefined);
  });
  await page.route("**/api/presentation/frame", async (route) => {
    const response = await route.fetch();
    markPovPresentationHeld();
    await povPresentationRelease;
    await route.fulfill({ response });
  });
  const povSwitch = expectSingleActivationCommand(
    page,
    "/api/replay/command",
    () => replayAgentNotVisibleRow.click(),
    {
      command_type: "set_pov_actor",
      global_slot: replayAgentNotVisibleSlot,
    },
  );
  try {
    await povPresentationHeld;
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "retained",
    );
    await expect(page.locator("#battlefield .agent")).toHaveCount(
      replayAgentBodyKeys.length,
    );
    expect(
      await page
        .locator("#battlefield .agent")
        .evaluateAll((agents) =>
          agents.map((agent) => agent.getAttribute("data-presentation-key")),
        ),
    ).toEqual(replayAgentBodyKeys);
    await expect(page.locator("#battlefield-empty")).toBeHidden();
    await expect(page.locator("#replay-frame-slider")).toBeDisabled();
    expect(
      await page
        .locator("#replay-timeline button")
        .evaluateAll((buttons) =>
          buttons.every(
            (button) => button instanceof HTMLButtonElement && button.disabled,
          ),
        ),
    ).toBe(true);
    await page.keyboard.press("ArrowRight");
  } finally {
    releasePovPresentation();
  }
  await povSwitch;
  await page.unroute("**/api/presentation/frame");
  const switchedReplayAgent = await expectInstalledLeaf(
    page,
    "actor_pov_replay_viewer",
    "replay_no_shared_obs_agent_pov",
  );
  expect(switchedReplayAgent.transport.cursor).toEqual(replayAgentCursorBeforeSwitch);
  expect(switchedReplayAgent.presentation.authority.recipient_public_agent_id).toBe(
    replayAgentNotVisible.public_agent_id,
  );
  expect(
    switchedReplayAgent.presentation.authority.recipient_presentation_key,
  ).not.toBe(replayAgentRecipientKey);
  await expect(
    page.locator(`#battlefield [data-presentation-key="${replayAgentRecipientKey}"]`),
  ).toHaveCount(0);
  await expectAgentAuthoritySurface(
    page,
    switchedReplayAgent.presentation,
    replayAgentBattlefieldForbiddenValues,
  );
  await expect(page.locator("#battlefield")).toBeFocused();
  const replayAgentRanges = page.locator("#replay-ranges-button");
  await expect(replayAgentRanges).toHaveAttribute("aria-pressed", "true");
  await expectZeroCommandInteraction(page, () => replayAgentRanges.click());
  await expect(replayAgentRanges).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expectZeroCommandInteraction(page, () => replayAgentRanges.click());
  const replayAgentPresentationAfterLocalActions = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  expect(
    replayAgentPresentationAfterLocalActions.authority.recipient_presentation_key,
  ).toBe(switchedReplayAgent.presentation.authority.recipient_presentation_key);
  await expect(page.locator("#view-select")).toHaveValue("pov");
  expect(switchedReplayAgent.presentation.source.source_frame_index).toBe(0);
  await expectAuthorizedIncomingTransitionDom(page, switchedReplayAgent.presentation);
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
    expect(presentation.researcher_space.latest_transition.incoming_transition_id).toBe(
      await page.locator("#transition-value").textContent(),
    );
    if (frameIndex === replayAgent.presentation.source.source_final_frame_index) {
      await expect(page.locator("#battlefield .agent[role=button]")).not.toHaveCount(0);
      await expect(page.locator("#battlefield-instructions")).toContainText(
        "switch the fog-of-war recipient",
      );
    }
  }
  const terminalAgent = await expectInstalledLeaf(
    page,
    "actor_pov_replay_viewer",
    "replay_no_shared_obs_agent_pov",
  );
  await page.locator("#view-select").selectOption("researcher");
  const roundTripOracle = await expectInstalledLeaf(
    page,
    "researcher_replay_viewer",
    "replay_oracle",
  );
  expect(roundTripOracle.transport.cursor).toEqual(terminalAgent.transport.cursor);
  expect(
    roundTripOracle.presentation.current_endpoint.action_axis.owner_public_agent_id,
  ).toBe(replayAgentNotVisible.public_agent_id);
  await page.locator("#view-select").selectOption("pov");
  const roundTripAgent = await expectInstalledLeaf(
    page,
    "actor_pov_replay_viewer",
    "replay_no_shared_obs_agent_pov",
  );
  expect(roundTripAgent.transport.cursor).toEqual(terminalAgent.transport.cursor);
  expect(roundTripAgent.presentation.authority.recipient_public_agent_id).toBe(
    replayAgentNotVisible.public_agent_id,
  );
  await expectZeroCommandInteraction(page, () =>
    page.locator("#replay-clear-reference-button").click(),
  );
  await expect(page.locator('#battlefield .agent[data-selected="true"]')).toHaveCount(
    0,
  );
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#replay-clear-reference-button")).toBeDisabled();
  const replayAgentPresentationAfterClear = await authenticatedGet(
    page,
    "/api/presentation/frame",
  );
  expect(replayAgentPresentationAfterClear.authority.recipient_presentation_key).toBe(
    roundTripAgent.presentation.authority.recipient_presentation_key,
  );
  expect(browserErrors.get(page) ?? []).toEqual([]);
});

test(CP5_C_SLICE_TEST_TITLE, async ({ page }) => {
  test.skip(
    cp4C3ShieldOnly || cp4ECaptureDirectory !== null,
    "CP5 Slice C real-service proof is outside bounded CP4 capture modes.",
  );
  /** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
  let movingReplay = null;
  /** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
  let recoveryReplay = null;
  /** @type {unknown} */
  let testError = null;
  try {
    movingReplay = await startReplayViewer({
      scenario: "moving_basic_crossfire",
      includeStress: true,
    });
    recoveryReplay = await startReplayViewer({
      scenario: "recovery_refresh_cycle",
    });

    const combatMatrixViewports = [
      { width: 800, height: 600 },
      { width: 1200, height: 800 },
      { width: 1600, height: 1000 },
    ];
    const classTokenById = new Map([
      [1, "mage"],
      [2, "warrior"],
      [3, "hunter"],
      [4, "rogue"],
      [5, "priest"],
    ]);
    /**
     * @param {Record<string, any>} presentation
     * @param {string} frameLabel
     */
    const expectCombatIdentityMatrix = async (presentation, frameLabel) => {
      const authorizedAgents = /** @type {Record<string, any>[]} */ (
        presentation.current_endpoint.scene.agents
      );
      const authorizedByKey = new Map(
        authorizedAgents.map((agent) => [agent.presentation_key, agent]),
      );
      expect(authorizedByKey.size, `${frameLabel}: authorized identities`).toBe(
        authorizedAgents.length,
      );
      expect(
        new Set(authorizedAgents.map(({ class_id }) => class_id)),
        `${frameLabel}: all five classes`,
      ).toEqual(new Set([1, 2, 3, 4, 5]));

      const passes = [];
      for (const viewport of [...combatMatrixViewports, combatMatrixViewports[0]]) {
        await page.setViewportSize(viewport);
        await page.locator("#battlefield").evaluate(async (battlefield) => {
          if (!(battlefield instanceof SVGSVGElement)) {
            throw new TypeError("Authorized battlefield is unavailable.");
          }
          await new Promise((resolveFrame) =>
            requestAnimationFrame(() => resolveFrame(undefined)),
          );
          await new Promise((resolveFrame) =>
            requestAnimationFrame(() => resolveFrame(undefined)),
          );
        });
        await expect(page.locator("#battlefield .agent")).toHaveCount(
          authorizedAgents.length,
        );
        const pass = await page.locator("#battlefield").evaluate((battlefield) => {
          const states = Array.from(battlefield.querySelectorAll(".agent"))
            .map((agent) => {
              const body = agent.querySelector(".agent-body");
              const classIcon = agent.querySelector(".agent-class-icon");
              const presentationKey = agent.getAttribute("data-presentation-key");
              const inCombatStatusCells =
                presentationKey === null
                  ? []
                  : Array.from(
                      battlefield.querySelectorAll(
                        `.status-cell[data-token-id="in_combat"][data-presentation-key="${CSS.escape(presentationKey)}"]`,
                      ),
                    );
              const inCombatStatus = inCombatStatusCells[0] ?? null;
              const inCombatIcon = inCombatStatus?.querySelector(".status-cell__icon");
              const inCombatValue =
                inCombatStatus?.querySelector(".status-cell__value");
              if (
                !(body instanceof SVGCircleElement) ||
                !(classIcon instanceof SVGSVGElement)
              ) {
                throw new TypeError("Authorized identity geometry is unavailable.");
              }
              const bodyCenter = Number(body.getAttribute("cx"));
              const classX = Number(classIcon.getAttribute("x"));
              const classWidth = Number(classIcon.getAttribute("width"));
              return {
                presentationKey,
                classToken: agent.getAttribute("data-class"),
                projectedRadius: Number(body.getAttribute("r")),
                status: agent.getAttribute("data-combat-status"),
                countdown: Number(agent.getAttribute("data-steps-until-out-of-combat")),
                ariaLabel: agent.getAttribute("aria-label"),
                ariaDescription: agent.getAttribute("aria-description"),
                rootRole: agent.getAttribute("role"),
                rootTabIndex: agent.getAttribute("tabindex"),
                rootOwnsTooltip: agent.hasAttribute("data-tooltip-owner"),
                descendantIdentityOwnerCount: agent.querySelectorAll(
                  "[data-presentation-key]",
                ).length,
                descendantTooltipOwnerCount:
                  agent.querySelectorAll("[data-tooltip-owner]").length,
                descendantHitOwnerCount: agent.querySelectorAll(
                  '[role="button"], [tabindex="0"]',
                ).length,
                identityPartsResolveToRoot: [body, classIcon].every(
                  (part) => part.closest(".agent") === agent,
                ),
                classPointerEvents: getComputedStyle(classIcon).pointerEvents,
                dedicatedCombatIconCount: agent.querySelectorAll(
                  ".agent-combat-state-icon",
                ).length,
                classCentered: Math.abs(classX + classWidth / 2 - bodyCenter) < 0.001,
                inCombatStatusCount: inCombatStatusCells.length,
                inCombatDuration:
                  inCombatStatus === null
                    ? null
                    : Number(inCombatStatus.getAttribute("data-duration")),
                inCombatVisibleValue: inCombatValue?.textContent ?? null,
                inCombatGlyph: inCombatIcon?.getAttribute("data-icon") ?? null,
                inCombatColor:
                  inCombatStatus === null
                    ? null
                    : getComputedStyle(inCombatStatus).color,
                inCombatAriaLabel: inCombatStatus?.getAttribute("aria-label") ?? null,
                inCombatRole: inCombatStatus?.getAttribute("role") ?? null,
                inCombatOwnsTooltip:
                  inCombatStatus?.hasAttribute("data-tooltip-owner") ?? false,
              };
            })
            .sort((left, right) =>
              String(left.presentationKey).localeCompare(String(right.presentationKey)),
            );
          return {
            surface: `${battlefield.clientWidth}x${battlefield.clientHeight}`,
            states,
          };
        });
        passes.push(pass);
      }

      expect(
        new Set(passes.slice(0, 3).map(({ surface }) => surface)).size,
        `${frameLabel}: distinct rendered surfaces`,
      ).toBe(3);
      expect(passes[3], `${frameLabel}: repeated render`).toEqual(passes[0]);
      for (const presentationKey of authorizedByKey.keys()) {
        expect(
          new Set(
            passes
              .slice(0, 3)
              .flatMap(({ states }) =>
                states
                  .filter((state) => state.presentationKey === presentationKey)
                  .map(({ projectedRadius }) => projectedRadius.toFixed(6)),
              ),
          ).size,
          `${frameLabel}: ${presentationKey} representative radii`,
        ).toBe(3);
      }

      for (const { surface, states } of passes.slice(0, 3)) {
        expect(states, `${frameLabel} at ${surface}: DOM identities`).toHaveLength(
          authorizedAgents.length,
        );
        expect(
          new Set(states.map(({ presentationKey }) => presentationKey)),
          `${frameLabel} at ${surface}: exact identity join`,
        ).toEqual(new Set(authorizedByKey.keys()));
        for (const state of states) {
          const authorized = authorizedByKey.get(state.presentationKey);
          expect(
            authorized,
            `${frameLabel} at ${surface}: ${state.presentationKey} is authorized`,
          ).toBeTruthy();
          if (!authorized) {
            throw new Error("Rendered agent is absent from the authorized scene.");
          }
          const inCombat = authorized.steps_until_out_of_combat > 0;
          const status = inCombat ? "IC" : "OOC";
          expect(state.classToken).toBe(classTokenById.get(authorized.class_id));
          expect(state.countdown).toBe(authorized.steps_until_out_of_combat);
          expect(state.status).toBe(status);
          expect(state.dedicatedCombatIconCount).toBe(0);
          expect(state.classCentered).toBe(true);
          expect(state.classPointerEvents).toBe("none");
          expect(state.rootOwnsTooltip).toBe(true);
          expect(["button", "img"]).toContain(state.rootRole);
          expect(["0", "-1"]).toContain(state.rootTabIndex);
          expect(state.descendantIdentityOwnerCount).toBe(0);
          expect(state.descendantTooltipOwnerCount).toBe(0);
          expect(state.descendantHitOwnerCount).toBe(0);
          expect(state.identityPartsResolveToRoot).toBe(true);
          expect(state.ariaLabel).not.toMatch(/combat|steps until out/iu);
          if (inCombat) {
            expect(state.inCombatStatusCount).toBe(1);
            expect(state.inCombatDuration).toBe(authorized.steps_until_out_of_combat);
            expect(state.inCombatVisibleValue).toBe(
              String(authorized.steps_until_out_of_combat),
            );
            expect(state.inCombatGlyph).toBe("combat-in-progress");
            expect(state.inCombatColor).toBe("rgb(255, 255, 255)");
            expect(state.inCombatRole).toBe("img");
            expect(state.inCombatOwnsTooltip).toBe(true);
            expect(state.inCombatAriaLabel).toContain("In combat");
            expect(state.inCombatAriaLabel).toContain(
              `duration ${authorized.steps_until_out_of_combat}`,
            );
          } else {
            expect(state.inCombatStatusCount).toBe(0);
            expect(state.inCombatDuration).toBeNull();
            expect(state.inCombatVisibleValue).toBeNull();
            expect(state.inCombatGlyph).toBeNull();
            expect(state.inCombatColor).toBeNull();
            expect(state.inCombatAriaLabel).toBeNull();
            expect(state.inCombatRole).toBeNull();
            expect(state.inCombatOwnsTooltip).toBe(false);
          }
        }
      }
      return passes.slice(0, 3);
    };

    await openProduct(page, movingReplay.url, "replay");
    const movingFrameZero = await expectInstalledLeaf(
      page,
      "researcher_replay_viewer",
      "replay_oracle",
    );
    const frameZeroAgents = /** @type {Record<string, any>[]} */ (
      movingFrameZero.presentation.current_endpoint.scene.agents
    );
    expect(new Set(frameZeroAgents.map(({ class_id }) => class_id))).toEqual(
      new Set([1, 2, 3, 4, 5]),
    );
    expect(
      frameZeroAgents.every(
        ({ steps_until_out_of_combat }) => steps_until_out_of_combat === 0,
      ),
    ).toBe(true);
    await expectCombatIdentityMatrix(movingFrameZero.presentation, "moving frame zero");
    await expect(page.locator("#battlefield .agent")).toHaveCount(10);
    await expect(page.locator("#battlefield .agent-combat-state-icon")).toHaveCount(0);
    await expect(
      page.locator('#battlefield .status-cell[data-token-id="in_combat"]'),
    ).toHaveCount(0);
    const frameZeroStates = await page
      .locator("#battlefield .agent")
      .evaluateAll((agents) =>
        agents.map((agent) => ({
          status: agent.getAttribute("data-combat-status"),
          countdown: agent.getAttribute("data-steps-until-out-of-combat"),
          ariaLabel: agent.getAttribute("aria-label"),
          ariaDescription: agent.getAttribute("aria-description"),
        })),
      );
    for (const state of frameZeroStates) {
      expect(state.status).toBe("OOC");
      expect(state.countdown).toBe("0");
      expect(state.ariaLabel).toContain("Inspect this authorized agent.");
      expect(state.ariaLabel).not.toMatch(/combat|steps until out/iu);
    }

    await seekReplay(page, 1);
    const movingFrameOne = await authenticatedGet(page, "/api/presentation/frame");
    const frameOneAgents = /** @type {Record<string, any>[]} */ (
      movingFrameOne.current_endpoint.scene.agents
    );
    expect(new Set(frameOneAgents.map(({ class_id }) => class_id))).toEqual(
      new Set([1, 2, 3, 4, 5]),
    );
    const frameOneInCombatAgents = frameOneAgents.filter(
      ({ steps_until_out_of_combat }) => steps_until_out_of_combat > 0,
    );
    const frameOneOutOfCombatAgents = frameOneAgents.filter(
      ({ steps_until_out_of_combat }) => steps_until_out_of_combat === 0,
    );
    expect(new Set(frameOneInCombatAgents.map(({ class_id }) => class_id))).toEqual(
      new Set([1, 2, 3, 4]),
    );
    expect(new Set(frameOneOutOfCombatAgents.map(({ class_id }) => class_id))).toEqual(
      new Set([5]),
    );
    await expectCombatIdentityMatrix(movingFrameOne, "moving frame one");
    await expect(page.locator("#battlefield .agent-combat-state-icon")).toHaveCount(0);
    await expect(
      page.locator('#battlefield .status-cell[data-token-id="in_combat"]'),
    ).toHaveCount(frameOneInCombatAgents.length);
    const frameOneStates = await page
      .locator("#battlefield .agent")
      .evaluateAll((agents) =>
        agents.map((agent) => {
          const presentationKey = agent.getAttribute("data-presentation-key");
          const statusCells =
            presentationKey === null
              ? []
              : Array.from(
                  document.querySelectorAll(
                    `#battlefield .status-cell[data-token-id="in_combat"][data-presentation-key="${CSS.escape(presentationKey)}"]`,
                  ),
                );
          const statusCell = statusCells[0] ?? null;
          const icon = statusCell?.querySelector(".status-cell__icon");
          const value = statusCell?.querySelector(".status-cell__value");
          return {
            presentationKey,
            status: agent.getAttribute("data-combat-status"),
            countdown: Number(agent.getAttribute("data-steps-until-out-of-combat")),
            rootAriaLabel: agent.getAttribute("aria-label"),
            dedicatedCombatIconCount: agent.querySelectorAll(".agent-combat-state-icon")
              .length,
            inCombatStatusCount: statusCells.length,
            duration:
              statusCell === null
                ? null
                : Number(statusCell.getAttribute("data-duration")),
            visibleValue: value?.textContent ?? null,
            glyph: icon?.getAttribute("data-icon") ?? null,
            color: statusCell === null ? null : getComputedStyle(statusCell).color,
            role: statusCell?.getAttribute("role") ?? null,
            inCombatAriaLabel: statusCell?.getAttribute("aria-label") ?? null,
            ownsTooltip: statusCell?.hasAttribute("data-tooltip-owner") ?? false,
          };
        }),
      );
    for (const state of frameOneStates) {
      const authorized = frameOneAgents.find(
        ({ presentation_key }) => presentation_key === state.presentationKey,
      );
      expect(authorized).toBeTruthy();
      if (!authorized) {
        throw new Error(
          "Rendered IC agent is absent from the authorized current scene.",
        );
      }
      expect(state.countdown).toBe(authorized.steps_until_out_of_combat);
      expect(state.dedicatedCombatIconCount).toBe(0);
      expect(state.rootAriaLabel).not.toMatch(/combat|steps until out/iu);
      if (authorized.steps_until_out_of_combat > 0) {
        expect(state.status).toBe("IC");
        expect(state.inCombatStatusCount).toBe(1);
        expect(state.duration).toBe(authorized.steps_until_out_of_combat);
        expect(state.visibleValue).toBe(String(authorized.steps_until_out_of_combat));
        expect(state.glyph).toBe("combat-in-progress");
        expect(state.color).toBe("rgb(255, 255, 255)");
        expect(state.role).toBe("img");
        expect(state.inCombatAriaLabel).toContain("In combat");
        expect(state.inCombatAriaLabel).toContain(
          `duration ${authorized.steps_until_out_of_combat}`,
        );
        expect(state.ownsTooltip).toBe(true);
      } else {
        expect(state.status).toBe("OOC");
        expect(state.inCombatStatusCount).toBe(0);
        expect(state.duration).toBeNull();
        expect(state.visibleValue).toBeNull();
        expect(state.glyph).toBeNull();
        expect(state.color).toBeNull();
        expect(state.role).toBeNull();
        expect(state.inCombatAriaLabel).toBeNull();
        expect(state.ownsTooltip).toBe(false);
      }
    }
    expect(browserErrors.get(page) ?? []).toEqual([]);

    const movingFinalFrameIndex =
      movingFrameZero.presentation.source.source_final_frame_index;
    expect(movingFinalFrameIndex).toBeGreaterThan(1);
    await seekReplay(page, movingFinalFrameIndex);
    const movingFinal = await authenticatedGet(page, "/api/presentation/frame");
    const finalAgents = /** @type {Record<string, any>[]} */ (
      movingFinal.current_endpoint.scene.agents
    );
    const finalPriests = finalAgents.filter(({ class_id }) => class_id === 5);
    expect(finalPriests).toHaveLength(2);
    expect(
      finalPriests.every(
        ({ steps_until_out_of_combat }) => steps_until_out_of_combat > 0,
      ),
    ).toBe(true);
    await expectCombatIdentityMatrix(movingFinal, "moving final frame");
    const renderedPriests = await page
      .locator('#battlefield .agent[data-class="priest"]')
      .evaluateAll((agents) =>
        agents.map((agent) => {
          const presentationKey = agent.getAttribute("data-presentation-key");
          const statusCells =
            presentationKey === null
              ? []
              : Array.from(
                  document.querySelectorAll(
                    `#battlefield .status-cell[data-token-id="in_combat"][data-presentation-key="${CSS.escape(presentationKey)}"]`,
                  ),
                );
          const statusCell = statusCells[0] ?? null;
          return {
            presentationKey,
            status: agent.getAttribute("data-combat-status"),
            countdown: Number(agent.getAttribute("data-steps-until-out-of-combat")),
            rootAriaLabel: agent.getAttribute("aria-label"),
            dedicatedCombatIconCount: agent.querySelectorAll(".agent-combat-state-icon")
              .length,
            inCombatStatusCount: statusCells.length,
            duration:
              statusCell === null
                ? null
                : Number(statusCell.getAttribute("data-duration")),
            visibleValue:
              statusCell?.querySelector(".status-cell__value")?.textContent ?? null,
            glyph:
              statusCell
                ?.querySelector(".status-cell__icon")
                ?.getAttribute("data-icon") ?? null,
            color: statusCell === null ? null : getComputedStyle(statusCell).color,
            inCombatAriaLabel: statusCell?.getAttribute("aria-label") ?? null,
            ownsTooltip: statusCell?.hasAttribute("data-tooltip-owner") ?? false,
          };
        }),
      );
    expect(renderedPriests).toHaveLength(2);
    for (const state of renderedPriests) {
      const authorized = finalPriests.find(
        ({ presentation_key }) => presentation_key === state.presentationKey,
      );
      expect(authorized).toBeTruthy();
      if (!authorized) {
        throw new Error(
          "Rendered Priest is absent from the authorized final moving scene.",
        );
      }
      expect(state.status).toBe("IC");
      expect(state.countdown).toBe(authorized.steps_until_out_of_combat);
      expect(state.rootAriaLabel).not.toMatch(/combat|steps until out/iu);
      expect(state.dedicatedCombatIconCount).toBe(0);
      expect(state.inCombatStatusCount).toBe(1);
      expect(state.duration).toBe(authorized.steps_until_out_of_combat);
      expect(state.visibleValue).toBe(String(authorized.steps_until_out_of_combat));
      expect(state.glyph).toBe("combat-in-progress");
      expect(state.color).toBe("rgb(255, 255, 255)");
      expect(state.inCombatAriaLabel).toContain("In combat");
      expect(state.inCombatAriaLabel).toContain(
        `duration ${authorized.steps_until_out_of_combat}`,
      );
      expect(state.ownsTooltip).toBe(true);
    }
    expect(browserErrors.get(page) ?? []).toEqual([]);

    await page.setViewportSize({ width: 960, height: 600 });
    await installWaapiAutopause(page);
    await openProduct(page, recoveryReplay.url, "replay");
    const recoveryFrameZero = await expectInstalledLeaf(
      page,
      "researcher_replay_viewer",
      "replay_oracle",
    );
    expect(recoveryFrameZero.presentation.source.source_frame_index).toBe(0);
    /** @type {import("@playwright/test").Request[]} */
    const recoveryCommands = [];
    const recordRecoveryCommand = (
      /** @type {import("@playwright/test").Request} */ request,
    ) => {
      if (
        request.method() === "POST" &&
        new URL(request.url()).pathname === "/api/replay/command"
      ) {
        recoveryCommands.push(request);
      }
    };
    const play = page.locator("#replay-play-pause-button");
    const next = page.locator("#replay-next-button");
    const regeneration = page.locator(".combat-effect--regeneration");
    page.on("request", recordRecoveryCommand);
    let recoveryCommandPayload;
    let regenerationDom;
    try {
      const responsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          new URL(response.url()).pathname === "/api/replay/command",
        { timeout: 30_000 },
      );
      await next.click();
      const response = await responsePromise;
      expect(response.status()).toBe(200);
      recoveryCommandPayload = await response.json();
      expect(recoveryCommandPayload).toMatchObject({
        result: "applied",
        animate_incoming: false,
        frame: { cursor: { frame_index: 1 } },
      });
      await expect(regeneration).toHaveCount(1);
      regenerationDom = await regeneration.evaluate((effect) => {
        const eventId = effect.getAttribute("data-event-id");
        return {
          eventId,
          dataValue: effect.getAttribute("data-value"),
          valueText:
            effect.querySelector(".combat-regeneration__value")?.textContent ?? null,
          plusLineCount: effect.querySelectorAll(".combat-regeneration__plus > line")
            .length,
          onionCount: effect.querySelectorAll(
            ".combat-semantic-pulse__ring, .combat-semantic-pulse__core",
          ).length,
          sourceAttributeCount: Array.from(effect.attributes).filter((attribute) =>
            attribute.name.includes("source"),
          ).length,
          resetEffectCount: document.querySelectorAll(
            '.combat-effect[data-event-type="combat_countdown_reset"]',
          ).length,
          resetPulseCount: document.querySelectorAll(
            ".combat-semantic-pulse--combat-countdown-reset",
          ).length,
          routeCount: Array.from(
            document.querySelectorAll(".combat-choreography-routes [data-event-id]"),
          ).filter((route) => route.getAttribute("data-event-id") === eventId).length,
        };
      });
    } finally {
      page.off("request", recordRecoveryCommand);
    }
    await expect(play).toHaveAttribute("aria-label", "Play replay");
    expect(recoveryCommands).toHaveLength(1);
    expect(recoveryCommands[0].postDataJSON().command).toEqual({
      command_type: "absolute_seek",
      frame_index: 1,
    });
    const recoveryFrameOne = await authenticatedGet(page, "/api/presentation/frame");
    const incoming = /** @type {Record<string, any>[]} */ (
      recoveryFrameOne.latest_events.events
    );
    const resetEvents = incoming.filter(
      ({ event_kind }) => event_kind === "combat_countdown_reset",
    );
    const regenerationEvents = incoming.filter(
      ({ event_kind }) => event_kind === "health_regenerated",
    );
    expect(resetEvents).toHaveLength(4);
    expect(regenerationEvents).toHaveLength(1);
    expect(regenerationEvents[0].actual_health_regenerated).toBe(4);
    expect(regenerationDom).toEqual({
      eventId: regenerationEvents[0].event_id,
      dataValue: "4",
      valueText: "+4",
      plusLineCount: 2,
      onionCount: 0,
      sourceAttributeCount: 0,
      resetEffectCount: 0,
      resetPulseCount: 0,
      routeCount: 0,
    });

    /** @type {{frameIndex: number, presentation: Record<string, any>, event: Record<string, any>} | null} */
    let leftCombatFinding = null;
    const recoveryFinalFrameIndex =
      recoveryFrameZero.presentation.source.source_final_frame_index;
    expect(recoveryFinalFrameIndex).toBeGreaterThan(1);
    for (let frameIndex = 2; frameIndex <= recoveryFinalFrameIndex; frameIndex += 1) {
      await seekReplay(page, frameIndex);
      const presentation = await authenticatedGet(page, "/api/presentation/frame");
      expect(presentation.source.source_frame_index).toBe(frameIndex);
      const leftCombatEvent = /** @type {Record<string, any>[]} */ (
        presentation.latest_events.events
      ).find(({ event_kind }) => event_kind === "agent_left_combat");
      if (leftCombatEvent) {
        leftCombatFinding = { frameIndex, presentation, event: leftCombatEvent };
        break;
      }
    }
    expect(leftCombatFinding).not.toBeNull();
    if (leftCombatFinding === null) {
      throw new Error(
        "Recovery trajectory produced no real agent-left-combat transition.",
      );
    }
    expect(leftCombatFinding.presentation.source.source_frame_index).toBe(
      leftCombatFinding.frameIndex,
    );

    const expirationOwner = page.locator(
      `.combat-effect--status-lifecycle[data-event-id="${leftCombatFinding.event.event_id}"][data-event-type="agent_left_combat"][data-token-id="in_combat"][data-lifecycle="expired"]`,
    );
    await expect(expirationOwner).toHaveCount(1);
    const expirationDom = await expirationOwner.evaluate(async (effect) => {
      const hit = effect.querySelector(".combat-lifecycle__hit");
      const icon = effect.querySelector(".combat-lifecycle__status-icon");
      if (!(hit instanceof SVGElement) || !(icon instanceof SVGElement)) {
        throw new Error("In-combat expiration lost its lifecycle paint.");
      }
      hit.dispatchEvent(new FocusEvent("focusin", { bubbles: true }));
      await new Promise((resolveFrame) => requestAnimationFrame(resolveFrame));
      return {
        ariaLabel: effect.getAttribute("aria-label"),
        role: effect.getAttribute("role"),
        glyph: icon.getAttribute("data-icon"),
        color: getComputedStyle(icon).color,
        durationBefore: effect.getAttribute("data-duration-before"),
        durationAfter: effect.getAttribute("data-duration-after"),
        atomicEventIds: JSON.parse(
          effect.getAttribute("data-atomic-event-ids") ?? "null",
        ),
        applicationEventIds: JSON.parse(
          effect.getAttribute("data-application-event-ids") ?? "null",
        ),
        reapplicationPaintCount: effect.querySelectorAll(".combat-lifecycle__reapply")
          .length,
        otherInCombatLifecyclePaintCount: document.querySelectorAll(
          '.combat-effect--status-lifecycle[data-token-id="in_combat"]:not([data-lifecycle="expired"])',
        ).length,
        resetEffectCount: document.querySelectorAll(
          '.combat-effect[data-event-type="combat_countdown_reset"]',
        ).length,
        resetPulseCount: document.querySelectorAll(
          ".combat-semantic-pulse--combat-countdown-reset",
        ).length,
        tooltipTitle:
          document.querySelector("#visual-tooltip-title")?.textContent ?? null,
        tooltipSummary:
          document.querySelector("#visual-tooltip .semantic-explanation__summary")
            ?.textContent ?? null,
      };
    });
    expect(expirationDom).toEqual({
      ariaLabel: "Expired",
      role: "img",
      glyph: "combat-in-progress",
      color: "rgb(255, 255, 255)",
      durationBefore: "1",
      durationAfter: "0",
      atomicEventIds: [leftCombatFinding.event.event_id],
      applicationEventIds: [],
      reapplicationPaintCount: 0,
      otherInCombatLifecyclePaintCount: 0,
      resetEffectCount: 0,
      resetPulseCount: 0,
      tooltipTitle: "Expired",
      tooltipSummary: "Status expired naturally",
    });
    expect(
      `${expirationDom.ariaLabel} ${expirationDom.tooltipTitle} ${expirationDom.tooltipSummary}`,
    ).not.toMatch(/_|semantic pulse/iu);

    expect(browserErrors.get(page) ?? []).toEqual([]);
  } catch (error) {
    testError = error;
  }
  const cleanup = await Promise.allSettled([
    stopDebugger(movingReplay?.process ?? null),
    stopDebugger(recoveryReplay?.process ?? null),
  ]);
  const cleanupErrors = cleanup.flatMap((result) =>
    result.status === "rejected" ? [result.reason] : [],
  );
  if (testError !== null || cleanupErrors.length > 0) {
    throw new AggregateError(
      [...(testError === null ? [] : [testError]), ...cleanupErrors],
      "CP5 Slice C real-service proof or cleanup failed.",
    );
  }
});

test(CP5_SLICE_5_TEST_TITLE, async ({ page }) => {
  test.skip(
    cp4C3ShieldOnly || cp4ECaptureDirectory !== null,
    "CP5 Slice 5 causal/privacy proof is outside bounded CP4 capture modes.",
  );
  test.setTimeout(900_000);
  await installWaapiAutopause(page);
  const checkedSampleBytesBefore = await cp5Slice5CheckedSampleSnapshot();

  /** @type {Readonly<Record<string, number>>} */
  const phaseRankByKind = Object.freeze({
    action_rejected: 10,
    ability_activated: 20,
    source_damage_output: 30,
    source_healing_output: 30,
    recipient_health_resolution: 40,
    agent_left_combat: 50,
    combat_countdown_reset: 50,
    health_regenerated: 50,
    cooldown_started: 60,
    cooldown_ready: 60,
    charge_phase_displacement: 70,
    ordinary_movement_phase_displacement: 80,
    agent_died: 90,
    lethal_damage_contribution: 90,
    status_aged_to_zero: 100,
    status_broken_by_damage: 100,
    status_applied: 100,
    status_refreshed_or_extended: 100,
    status_cleared_by_new_death: 100,
    spawn_shield_expired: 110,
    respawn_wave_occurred: 120,
    agent_respawned: 120,
  });
  const neutral = Object.freeze({
    move_action: 0,
    target_action: 0,
    use_ultimate_action: 0,
  });
  /** @param {number[][]} submitted @param {number[][] | null} [accepted] */
  const transition = (submitted, accepted = null) => ({
    submitted,
    accepted: accepted ?? submitted,
  });
  const basics = [
    ["basic", 0, 5],
    ["basic", 1, 6],
    ["basic", 2, 7],
    ["basic", 3, 8],
    ["basic", 4, 0],
    ["basic", 5, 0],
    ["basic", 6, 1],
    ["basic", 7, 2],
    ["basic", 8, 3],
    ["basic", 9, 5],
  ];
  const contracts = [
    {
      name: "moving_basic_crossfire",
      includeStress: true,
      selectedSlot: 0,
      actions: [
        transition([
          [0, 1, 6, 0],
          [1, 3, 7, 0],
          [2, 1, 8, 0],
          [3, 3, 9, 0],
          [4, 1, 1, 0],
          [5, 1, 6, 0],
          [6, 3, 7, 0],
          [7, 1, 8, 0],
          [8, 3, 9, 0],
          [9, 1, 1, 0],
        ]),
        transition([
          [0, 2, 6, 0],
          [1, 4, 7, 0],
          [2, 2, 8, 0],
          [3, 4, 9, 0],
          [4, 2, 1, 0],
          [5, 2, 6, 0],
          [6, 4, 7, 0],
          [7, 2, 8, 0],
          [8, 4, 9, 0],
          [9, 2, 1, 0],
        ]),
      ],
      abilities: [basics, basics],
      tokens: [
        [
          "basic_damage",
          "basic_damage",
          "basic_damage",
          "basic_damage",
          "basic_heal",
          "basic_damage",
          "basic_damage",
          "basic_damage",
          "basic_damage",
          "basic_heal",
        ],
        [
          "basic_damage",
          "basic_damage",
          "basic_damage",
          "basic_damage",
          "basic_heal",
          "basic_damage",
          "basic_damage",
          "basic_damage",
          "basic_damage",
          "basic_heal",
        ],
      ],
    },
    {
      name: "mirrored_ultimates",
      selectedSlot: 0,
      actions: [
        transition([
          [0, 1, 0, 1],
          [5, 1, 0, 1],
        ]),
        transition([
          [1, 0, 7, 1],
          [6, 0, 7, 1],
        ]),
        transition([
          [2, 1, 8, 1],
          [7, 1, 8, 1],
        ]),
        transition([
          [3, 3, 9, 1],
          [8, 3, 9, 1],
        ]),
        transition([
          [4, 1, 4, 1],
          [9, 1, 4, 1],
        ]),
      ],
      abilities: [
        [
          ["ultimate", 0, null],
          ["ultimate", 5, null],
        ],
        [
          ["ultimate", 1, 6],
          ["ultimate", 6, 1],
        ],
        [
          ["ultimate", 2, 7],
          ["ultimate", 7, 2],
        ],
        [
          ["ultimate", 3, 8],
          ["ultimate", 8, 3],
        ],
        [
          ["ultimate", 4, 3],
          ["ultimate", 9, 8],
        ],
      ],
      tokens: [
        ["mage_burst", "mage_burst"],
        ["warrior_charge", "warrior_charge"],
        ["hunter_trap", "hunter_trap"],
        ["rogue_poison", "rogue_poison"],
        ["holy_word", "holy_word"],
      ],
    },
    {
      name: "charge_convergence",
      includeStress: true,
      selectedSlot: 0,
      actions: [
        transition([
          [0, 0, 6, 1],
          [1, 0, 6, 1],
          [5, 0, 6, 1],
        ]),
      ],
      abilities: [
        [
          ["ultimate", 0, 5],
          ["ultimate", 1, 5],
          ["ultimate", 5, 0],
        ],
      ],
      tokens: [["warrior_charge", "warrior_charge", "warrior_charge"]],
    },
    {
      name: "trap_lifecycle",
      includeStress: true,
      selectedSlot: 0,
      actions: [
        transition([
          [0, 0, 6, 1],
          [1, 0, 7, 1],
          [2, 0, 8, 1],
          [3, 0, 9, 1],
        ]),
        transition([[0, 0, 6, 0]]),
        transition([]),
        transition([[4, 0, 7, 1]]),
        transition([[2, 0, 8, 0]]),
      ],
      abilities: [
        [
          ["ultimate", 0, 5],
          ["ultimate", 1, 6],
          ["ultimate", 2, 7],
          ["ultimate", 3, 8],
        ],
        [["basic", 0, 5]],
        [],
        [["ultimate", 4, 6]],
        [["basic", 2, 7]],
      ],
      tokens: [
        ["hunter_trap", "hunter_trap", "hunter_trap", "hunter_trap"],
        ["basic_damage"],
        [],
        ["hunter_trap"],
        ["basic_damage"],
      ],
      statuses: [
        [
          ["status_applied", 5, "hunter_trap_stun"],
          ["status_applied", 6, "hunter_trap_stun"],
          ["status_applied", 7, "hunter_trap_stun"],
          ["status_applied", 8, "hunter_trap_stun"],
        ],
        [
          ["status_applied", 5, "hunter_basic_slow"],
          ["status_broken_by_damage", 5, "hunter_trap_stun"],
        ],
        [["status_aged_to_zero", 5, "hunter_basic_slow"]],
        [
          ["status_broken_by_damage", 6, "hunter_trap_stun"],
          ["status_applied", 6, "hunter_trap_stun"],
        ],
        [
          ["status_applied", 7, "hunter_basic_slow"],
          ["status_aged_to_zero", 7, "hunter_trap_stun"],
          ["status_aged_to_zero", 8, "hunter_trap_stun"],
        ],
      ],
    },
    {
      name: "recovery_refresh_cycle",
      selectedSlot: 0,
      povSlot: 5,
      povFirst: true,
      actions: [
        transition(
          [
            [0, 0, 6, 1],
            [2, 0, 8, 1],
            [6, 0, 2, 1],
          ],
          [
            [0, 0, 6, 1],
            [2, 0, 8, 1],
          ],
        ),
        transition([
          [1, 0, 6, 1],
          [3, 0, 8, 1],
          [4, 0, 8, 0],
        ]),
        transition([]),
        transition([]),
        transition([]),
        transition([]),
        transition([]),
      ],
      abilities: [
        [
          ["ultimate", 0, 5],
          ["ultimate", 2, 7],
        ],
        [
          ["ultimate", 1, 5],
          ["ultimate", 3, 7],
          ["basic", 4, 7],
        ],
        [],
        [],
        [],
        [],
        [],
      ],
      tokens: [
        ["rogue_poison", "hunter_trap"],
        ["rogue_poison", "hunter_trap", "basic_damage"],
        [],
        [],
        [],
        [],
        [],
      ],
      statuses: [
        [
          ["status_applied", 5, "rogue_poison_slow"],
          ["status_applied", 5, "rogue_poison_stun"],
          ["status_applied", 5, "rogue_poison_anti_heal"],
          ["status_applied", 7, "hunter_trap_stun"],
        ],
        [
          ["status_applied", 5, "rogue_poison_slow"],
          ["status_refreshed_or_extended", 5, "rogue_poison_slow"],
          ["status_aged_to_zero", 5, "rogue_poison_stun"],
          ["status_applied", 5, "rogue_poison_stun"],
          ["status_applied", 5, "rogue_poison_anti_heal"],
          ["status_refreshed_or_extended", 5, "rogue_poison_anti_heal"],
          ["status_broken_by_damage", 7, "hunter_trap_stun"],
          ["status_applied", 7, "hunter_trap_stun"],
        ],
        [["status_aged_to_zero", 5, "rogue_poison_stun"]],
        [],
        [],
        [
          ["status_aged_to_zero", 5, "rogue_poison_anti_heal"],
          ["status_aged_to_zero", 7, "hunter_trap_stun"],
        ],
        [["status_aged_to_zero", 5, "rogue_poison_slow"]],
      ],
    },
    {
      name: "death_respawn_cycle",
      selectedSlot: 5,
      actions: [
        transition([
          [0, 0, 6, 0],
          [1, 0, 6, 0],
        ]),
        transition([]),
        transition([[5, 4, 6, 1]], []),
        transition([[5, 4, 6, 0]], [[5, 4, 0, 0]]),
        transition([[5, 0, 6, 0]], []),
        transition([[5, 0, 6, 0]], []),
        transition([[5, 0, 6, 0]]),
      ],
      abilities: [
        [
          ["basic", 0, 5],
          ["basic", 1, 5],
        ],
        [],
        [],
        [],
        [],
        [],
        [["basic", 5, 0]],
      ],
      tokens: [["basic_damage", "basic_damage"], [], [], [], [], [], ["basic_damage"]],
      statuses: [
        [
          ["status_cleared_by_new_death", 5, "warrior_charge_slow"],
          ["status_applied", 5, "hunter_basic_slow"],
          ["status_cleared_by_new_death", 5, "hunter_basic_slow"],
          ["status_broken_by_damage", 5, "hunter_trap_stun"],
          ["status_cleared_by_new_death", 5, "rogue_poison_anti_heal"],
        ],
        [],
        [],
        [],
        [],
        [],
        [],
      ],
      lifecycleKinds: [
        ["agent_died", "lethal_damage_contribution", "lethal_damage_contribution"],
        [],
        [
          "action_rejected",
          "action_rejected",
          "respawn_wave_occurred",
          "agent_respawned",
        ],
        ["action_rejected"],
        ["action_rejected", "respawn_wave_occurred"],
        ["action_rejected", "spawn_shield_expired"],
        [],
      ],
    },
  ];

  /** @param {Record<string, any>} presentation */
  const slotDirectory = (presentation) =>
    new Map(
      /** @type {Record<string, any>[]} */ (
        presentation.current_endpoint.identity_directory.identities
      ).map((identity) => [
        identity.public_agent_id,
        (identity.team_id - 1) * 5 + identity.team_local_slot,
      ]),
    );
  /** @param {Record<string, any>} presentation */
  const activeOracleSlots = (presentation) =>
    new Set(
      /** @type {Record<string, any>[]} */ (
        presentation.current_endpoint.identity_directory.identities
      )
        .filter(({ configured_active }) => configured_active)
        .map(({ team_id, team_local_slot }) => (team_id - 1) * 5 + team_local_slot),
    );
  /** @param {Record<string, any>} presentation */
  const oracleSceneAgents = (presentation) =>
    /** @type {Record<string, any>[]} */ (presentation.current_endpoint.scene.agents);
  /** @param {Record<string, any>} signature */
  const signaturePlanEvents = (signature) =>
    /** @type {Record<string, any>[]} */ (signature.plan.events);
  /**
   * Prove the exact product-installed effect and route surface against a
   * scenario-owned list whose IDs, vocabulary, and endpoint slots were
   * extracted independently from the real registered trajectory.
   *
   * Row shape: ordinal, type, source slot, target slot, recipient slot,
   * agent slot, persistent.
   *
   * @param {{presentation: Record<string, any>, signature: Record<string, any>}} frame
   * @param {string} label
   * @param {Array<[number, string, number | null, number | null, number | null, number | null, boolean?]>} effectRows
   * @param {number[]} routeOrdinals
   */
  const expectExactInstalledSurface = (frame, label, effectRows, routeOrdinals) => {
    const { presentation, signature } = frame;
    const transitionId = presentation.latest_events?.incoming_transition_id;
    if (typeof transitionId !== "string") {
      throw new Error(`${label} has no incoming transition ID.`);
    }
    const slots = slotDirectory(presentation);
    const keysBySlot = new Map(
      oracleSceneAgents(presentation).map((agent) => [
        slots.get(agent.public_agent_id),
        agent.presentation_key,
      ]),
    );
    /** @param {number | null} slot */
    const keyForSlot = (slot) => {
      if (slot === null) {
        return null;
      }
      const key = keysBySlot.get(slot);
      if (typeof key !== "string") {
        throw new Error(`${label} lost authorized endpoint slot ${slot}.`);
      }
      return key;
    };
    const expectedEffects = effectRows.map(
      ([
        ordinal,
        type,
        sourceSlot,
        targetSlot,
        recipientSlot,
        agentSlot,
        persistent = false,
      ]) => ({
        id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
        type,
        persistent,
        sourceKey: keyForSlot(sourceSlot),
        targetKey: keyForSlot(targetSlot),
        recipientKey: keyForSlot(recipientSlot),
        agentKey: keyForSlot(agentSlot),
      }),
    );
    expect(signature.dom.effects, `${label} exact installed effects`).toEqual(
      expectedEffects,
    );

    const planById = new Map(
      signaturePlanEvents(signature).map((event) => [event.eventId, event]),
    );
    const expectedPlanSurface = expectedEffects.map((expected) => {
      const planned = planById.get(expected.id);
      if (!planned) {
        throw new Error(`${label} lost planned event ${expected.id}.`);
      }
      return {
        id: planned.eventId,
        type: planned.eventType,
        persistent: planned.persistent,
        sourceKey: planned.sourcePresentationKey,
        targetKey: planned.targetPresentationKey,
        recipientKey: planned.recipientPresentationKey,
        agentKey: planned.agentPresentationKey,
      };
    });
    expect(expectedPlanSurface, `${label} exact plan-to-DOM effects`).toEqual(
      expectedEffects,
    );

    const expectedByOrdinal = new Map(
      effectRows.map((row, index) => [row[0], expectedEffects[index]]),
    );
    const expectedRoutes = routeOrdinals.map((ordinal) => {
      const expected = expectedByOrdinal.get(ordinal);
      if (!expected) {
        throw new Error(`${label} route ordinal ${ordinal} has no expected effect.`);
      }
      return expected;
    });
    expect(signature.dom.routes, `${label} exact installed routes`).toEqual(
      expectedRoutes,
    );
    for (const expected of expectedRoutes) {
      const planned = planById.get(expected.id);
      if (!planned) {
        throw new Error(`${label} lost planned route ${expected.id}.`);
      }
      expect(
        planned.route !== null,
        `${label} ${expected.id} remains route-owned`,
      ).toBe(true);
    }
  };
  /** @param {number[]} row */
  const tupleFromRow = (row) => ({
    move_action: row[1],
    target_action: row[2],
    use_ultimate_action: row[3],
  });
  /** @param {number[][]} rows */
  const tuplesBySlot = (rows) =>
    new Map(rows.map((row) => [row[0], tupleFromRow(row)]));
  /** @param {Record<string, any>} event @param {Map<string, number>} slots */
  const eventRecipientSlot = (event, slots) =>
    slots.get(event.recipient_anchor?.public_agent_id) ?? null;
  /** @param {Record<string, any>} event @param {Map<string, number>} slots */
  const eventSourceSlot = (event, slots) =>
    slots.get(event.source_anchor?.public_agent_id) ??
    slots.get(event.agent_anchor?.public_agent_id) ??
    slots.get(event.start_anchor?.public_agent_id) ??
    null;
  /** @param {number} length @param {number[]} trueIndices */
  const indexedMask = (length, trueIndices) =>
    Array.from({ length }, (_, index) => trueIndices.includes(index));
  /** @param {number[][]} trueCells */
  const jointMask = (trueCells) =>
    Array.from({ length: 11 }, (_, target) =>
      Array.from({ length: 2 }, (_, lane) =>
        trueCells.some(
          ([trueTarget, trueLane]) => trueTarget === target && trueLane === lane,
        ),
      ),
    );
  /** @param {Record<string, any>} inspection */
  const inspectedMask = (inspection) => ({
    move: inspection.decision_mask.movement_action_mask,
    select_target: inspection.decision_mask.target_action_mask,
    use_ultimate: inspection.decision_mask.use_ultimate_action_mask,
    select_target_use_ultimate_joint:
      inspection.decision_mask.target_use_ultimate_joint_mask,
  });

  /**
   * Select every named acting owner at the real predecessor, prove its exact
   * tuple against that epoch's mask, then join it to the successor action row.
   *
   * @param {string} contractName
   * @param {number} frameIndex
   * @param {Array<Record<string, any>>} expectations
   */
  const proveOutgoingOwnerSet = async (contractName, frameIndex, expectations) => {
    if (
      (await page.locator("#replay-frame-slider").inputValue()) !== String(frameIndex)
    ) {
      await seekReplay(page, frameIndex);
    }
    const predecessor = await authenticatedGet(page, "/api/presentation/frame");
    const upcoming = predecessor.upcoming_transition;
    expect(upcoming).not.toBeNull();
    expect(upcoming.outgoing_transition_index).toBe(frameIndex);
    /** @type {Array<Record<string, any>>} */
    const outgoing = [];
    for (const expected of expectations) {
      const before = await authenticatedGet(page, "/api/presentation/frame");
      const identity = /** @type {Record<string, any>[]} */ (
        before.current_endpoint.identity_directory.identities
      ).find(
        ({ team_id, team_local_slot }) =>
          (team_id - 1) * 5 + team_local_slot === expected.slot,
      );
      const agent = oracleSceneAgents(before).find(
        ({ public_agent_id }) => public_agent_id === identity?.public_agent_id,
      );
      if (!identity || !agent) {
        throw new Error(`${contractName} F${frameIndex} g${expected.slot} is absent.`);
      }
      const responsePromise = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          new URL(response.url()).pathname === "/api/replay/command",
      );
      await page
        .locator(
          `#roster .roster-primary-action[data-presentation-key="${agent.presentation_key}"]`,
        )
        .click();
      expect((await responsePromise).status()).toBe(200);
      await expect(
        page.locator(
          `#battlefield .agent[data-presentation-key="${agent.presentation_key}"]`,
        ),
      ).toHaveAttribute("data-selected", "true");
      const selected = await authenticatedGet(page, "/api/presentation/frame");
      const inspection = selected.replay_inspection;
      expect(selected.upcoming_transition).toEqual(upcoming);
      expect(inspection.outgoing_transition_index).toBe(frameIndex);
      expect(inspection.actor_public_agent_id).toBe(identity.public_agent_id);
      expect(inspection.submitted_action).toEqual(expected.submitted);
      expect(inspection.accepted_action).toEqual(expected.accepted);
      expect(inspectedMask(inspection)).toEqual(expected.mask);
      const submitted = inspection.submitted_action;
      const accepted = inspection.accepted_action;
      expect(inspection.decision_mask.movement_action_mask[accepted.move_action]).toBe(
        true,
      );
      expect(
        inspection.decision_mask.target_use_ultimate_joint_mask[accepted.target_action][
          accepted.use_ultimate_action
        ],
      ).toBe(true);
      expect(inspection.decision_mask.movement_action_mask[submitted.move_action]).toBe(
        expected.submittedMoveLegal,
      );
      expect(
        inspection.decision_mask.target_use_ultimate_joint_mask[
          submitted.target_action
        ][submitted.use_ultimate_action],
      ).toBe(expected.submittedPairLegal);
      outgoing.push({
        actorPublicAgentId: identity.public_agent_id,
        submitted,
        accepted,
        reference: inspection.transition_reference,
      });
    }
    await seekReplay(page, frameIndex + 1);
    const successor = await authenticatedGet(page, "/api/presentation/frame");
    expect(successor.latest_transition.action_rows).toEqual(upcoming.action_rows);
    for (const expected of outgoing) {
      expect(successor.latest_events.incoming_transition_id).toBe(
        expected.reference.transition_id,
      );
      expect(successor.latest_events.incoming_start_frame_id).toBe(
        expected.reference.start_frame_id,
      );
      expect(successor.latest_events.incoming_successor_frame_id).toBe(
        expected.reference.successor_frame_id,
      );
      const row = /** @type {Record<string, any>[]} */ (
        successor.latest_transition.action_rows
      ).find(
        ({ actor_public_agent_id }) =>
          actor_public_agent_id === expected.actorPublicAgentId,
      );
      if (!row) {
        throw new Error(`${contractName} successor lost an acting-owner row.`);
      }
      expect(row.submitted_action).toEqual(expected.submitted);
      expect(row.accepted_action).toEqual(expected.accepted);
    }
  };

  /** @param {number} frameIndex */
  const advanceToStaticChoreographySummary = async (frameIndex) => {
    const next = page.locator("#replay-next-button");
    const responsePromise = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname === "/api/replay/command",
      { timeout: 30_000 },
    );
    await next.click();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    expect(response.request().postDataJSON().command).toEqual({
      command_type: "absolute_seek",
      frame_index: frameIndex,
    });
    expect(await response.json()).toMatchObject({
      result: "applied",
      animate_incoming: false,
      frame: { cursor: { frame_index: frameIndex } },
    });
    await expect(page.locator("#replay-frame-slider")).toHaveValue(String(frameIndex));
    await expect(page.locator("html")).toHaveAttribute(
      "data-presentation-authority",
      "installed",
    );
  };

  /**
   * @param {string} productUrl
   * @param {Record<string, any>} contract
   */
  const proveOracleTrajectory = async (productUrl, contract) => {
    if (!contract.povFirst) {
      await page.goto("about:blank");
      await openProduct(page, productUrl, "replay");
    }
    const bootstrap = await authenticatedGet(page, "/api/presentation/frame");
    const selectedIdentity = /** @type {Record<string, any>[]} */ (
      bootstrap.current_endpoint.identity_directory.identities
    ).find(
      (identity) =>
        (identity.team_id - 1) * 5 + identity.team_local_slot === contract.selectedSlot,
    );
    const selectedAgent = oracleSceneAgents(bootstrap).find(
      (agent) => agent.public_agent_id === selectedIdentity?.public_agent_id,
    );
    if (!selectedIdentity || !selectedAgent) {
      throw new Error(`${contract.name} default inspection owner is unavailable.`);
    }
    const selectionResponse = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname === "/api/replay/command",
    );
    await page
      .locator(
        `#roster .roster-primary-action[data-presentation-key="${selectedAgent.presentation_key}"]`,
      )
      .click();
    expect((await selectionResponse).status()).toBe(200);
    await expect(
      page.locator(
        `#battlefield .agent[data-presentation-key="${selectedAgent.presentation_key}"]`,
      ),
    ).toHaveAttribute("data-selected", "true");
    /** @type {Array<{presentation: Record<string, any>, signature: Record<string, any>}>} */
    const frames = [];
    for (let frameIndex = 0; frameIndex <= contract.actions.length; frameIndex += 1) {
      if (frameIndex > 0) {
        await advanceToStaticChoreographySummary(frameIndex);
      }
      const presentationBody = await cp5Slice5PresentationBody(page);
      const presentation = JSON.parse(presentationBody);
      expect(presentation.presentation_kind, `${contract.name} F${frameIndex}`).toBe(
        "replay_oracle",
      );
      expect(presentation.source.source_frame_index).toBe(frameIndex);
      expect(presentation.source.source_final_frame_index).toBe(
        contract.actions.length,
      );
      expect(presentation.current_endpoint.frame_index).toBe(frameIndex);
      expect(presentation.current_endpoint.frame_id).toBe(
        presentation.source.source_frame_id,
      );
      if (frameIndex > 0) {
        const incomingTransitionId = presentation.latest_events.incoming_transition_id;
        await expect
          .poll(() => page.locator(CHOREOGRAPHY_ROOT).getAttribute("data-epoch-key"))
          .toContain(incomingTransitionId);
      }
      const signature = await cp5Slice5PlanAndDomSignature(page, presentation);
      expect(signature.inputUnchanged).toBe(true);
      const slots = slotDirectory(presentation);
      const activeSlots = activeOracleSlots(presentation);

      if (frameIndex === 0) {
        expect(presentation.latest_events).toBeNull();
        expect(presentation.latest_transition).toBeNull();
      } else {
        const previous = frames[frameIndex - 1].presentation.replay_inspection;
        const incoming = presentation.latest_events;
        const latest = presentation.latest_transition;
        const incomingEvents = /** @type {Record<string, any>[]} */ (incoming.events);
        const actionRows = /** @type {Record<string, any>[]} */ (latest.action_rows);
        expect(previous).not.toBeNull();
        expect(incoming.incoming_transition_id).toBe(
          previous.transition_reference.transition_id,
        );
        expect(incoming.incoming_start_frame_id).toBe(
          previous.transition_reference.start_frame_id,
        );
        expect(incoming.incoming_successor_frame_id).toBe(
          previous.transition_reference.successor_frame_id,
        );
        expect(latest.incoming_transition_id).toBe(incoming.incoming_transition_id);
        expect(latest.incoming_start_frame_id).toBe(incoming.incoming_start_frame_id);
        expect(latest.incoming_successor_frame_id).toBe(
          incoming.incoming_successor_frame_id,
        );
        expect(incoming.event_count).toBe(incomingEvents.length);
        expect(incoming.ordered_event_ids).toEqual(
          incomingEvents.map(({ event_id }) => event_id),
        );
        expect(incoming.ordered_event_kinds).toEqual(
          incomingEvents.map(({ event_kind }) => event_kind),
        );
        for (const [ordinal, event] of incomingEvents.entries()) {
          expect(event.ordinal).toBe(ordinal);
          expect(event.phase_rank).toBe(phaseRankByKind[event.event_kind]);
          expect(event.event_id).toBe(
            `${incoming.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
          );
        }
        expect(incomingEvents.map(({ phase_rank }) => phase_rank)).toEqual(
          [...incomingEvents.map(({ phase_rank }) => phase_rank)].sort(
            (left, right) => left - right,
          ),
        );
        expect(signature.dom.removedEventSurfaceCount).toBe(0);
        expect(signature.dom.transition).toBe(incoming.incoming_transition_id);
        expect(signature.plan.transitionId).toBe(incoming.incoming_transition_id);
        const planEvents = signaturePlanEvents(signature);
        const incomingKindById = new Map(
          incomingEvents.map((event) => [event.event_id, event.event_kind]),
        );
        for (const event of planEvents) {
          expect(event.transitionId).toBe(incoming.incoming_transition_id);
          expect(event.authorityVocabulary).toBe("event");
          expect(incomingKindById.get(event.eventId)).toBe(event.eventType);
          for (const atomicEventId of event.atomicEventIds ?? [event.eventId]) {
            expect(incomingKindById.has(atomicEventId)).toBe(true);
          }
        }
        const planById = new Map(planEvents.map((event) => [event.eventId, event]));
        expect(new Set(signature.dom.effects.map(({ id }) => id)).size).toBe(
          signature.dom.effects.length,
        );
        for (const effect of signature.dom.effects) {
          const planned = planById.get(effect.id);
          if (!planned) {
            throw new Error(`${contract.name} installed unknown effect ${effect.id}.`);
          }
          expect(effect.type).toBe(planned.eventType);
          expect(effect.persistent).toBe(planned.persistent);
          expect(effect.sourceKey).toBe(planned.sourcePresentationKey);
          expect(effect.targetKey).toBe(planned.targetPresentationKey);
          expect(effect.recipientKey).toBe(planned.recipientPresentationKey);
          expect(effect.agentKey).toBe(planned.agentPresentationKey);
        }
        expect(new Set(signature.dom.routes.map(({ id }) => id)).size).toBe(
          signature.dom.routes.length,
        );
        for (const route of signature.dom.routes) {
          const planned = planById.get(route.id);
          if (!planned) {
            throw new Error(`${contract.name} installed unknown route ${route.id}.`);
          }
          expect(planned.route).not.toBeNull();
          expect(route.type).toBe(planned.eventType);
          expect(route.persistent).toBe(planned.persistent);
          expect(route.sourceKey).toBe(planned.sourcePresentationKey);
          expect(route.targetKey).toBe(planned.targetPresentationKey);
          expect(route.recipientKey).toBe(planned.recipientPresentationKey);
          expect(route.agentKey).toBe(planned.agentPresentationKey);
          expect(signature.dom.effects.some(({ id }) => id === route.id)).toBe(true);
        }

        const expected = contract.actions[frameIndex - 1];
        const submitted = tuplesBySlot(expected.submitted);
        const accepted = tuplesBySlot(expected.accepted);
        expect(actionRows).toHaveLength(activeSlots.size);
        expect(
          new Set(actionRows.map((row) => slots.get(row.actor_public_agent_id))),
        ).toEqual(activeSlots);
        for (const row of actionRows) {
          const slot = slots.get(row.actor_public_agent_id);
          expect(
            row.submitted_action,
            `${contract.name} T${frameIndex - 1} g${slot}`,
          ).toEqual(submitted.get(slot) ?? neutral);
          expect(
            row.accepted_action,
            `${contract.name} T${frameIndex - 1} g${slot}`,
          ).toEqual(accepted.get(slot) ?? neutral);
        }
        const previousSlot = slotDirectory(frames[frameIndex - 1].presentation).get(
          previous.actor_public_agent_id,
        );
        expect(previousSlot).toBe(contract.selectedSlot);
        expect(previous.submitted_action).toEqual(
          submitted.get(contract.selectedSlot) ?? neutral,
        );
        expect(previous.accepted_action).toEqual(
          accepted.get(contract.selectedSlot) ?? neutral,
        );
        const previousRow = actionRows.find(
          (row) => row.actor_public_agent_id === previous.actor_public_agent_id,
        );
        if (!previousRow) {
          throw new Error(`${contract.name} lost its selected incoming action row.`);
        }
        expect(previousRow.submitted_action).toEqual(previous.submitted_action);
        expect(previousRow.accepted_action).toEqual(previous.accepted_action);

        const abilities = incomingEvents
          .filter(({ event_kind }) => event_kind === "ability_activated")
          .map((event) => [
            event.ability_component,
            slots.get(event.source_anchor.public_agent_id),
            eventRecipientSlot(event, slots),
          ]);
        expect(abilities, `${contract.name} T${frameIndex - 1} abilities`).toEqual(
          contract.abilities[frameIndex - 1],
        );
        expect(
          signaturePlanEvents(signature)
            .filter(({ kind }) => kind === "activation")
            .map(({ tokenId }) => tokenId),
          `${contract.name} T${frameIndex - 1} tokens`,
        ).toEqual(contract.tokens[frameIndex - 1]);
        if (contract.statuses) {
          const statuses = incomingEvents
            .filter(({ event_kind }) => event_kind.startsWith("status_"))
            .map((event) => [
              event.event_kind,
              eventRecipientSlot(event, slots),
              event.status_id,
            ]);
          expect(statuses, `${contract.name} T${frameIndex - 1} statuses`).toEqual(
            contract.statuses[frameIndex - 1],
          );
        }
        if (contract.lifecycleKinds) {
          const lifecycleKinds = incomingEvents
            .map(({ event_kind }) => event_kind)
            .filter((kind) =>
              new Set([
                "action_rejected",
                "agent_died",
                "lethal_damage_contribution",
                "respawn_wave_occurred",
                "agent_respawned",
                "spawn_shield_expired",
              ]).has(kind),
            );
          expect(lifecycleKinds).toEqual(contract.lifecycleKinds[frameIndex - 1]);
        }
      }

      const inspection = presentation.replay_inspection;
      if (frameIndex === contract.actions.length) {
        expect(inspection).toBeNull();
      } else {
        expect(inspection.outgoing_transition_index).toBe(frameIndex);
        expect(inspection.current_simulator_step_count).toBe(
          presentation.current_endpoint.simulator_step_count,
        );
        expect(inspection.decision_mask.movement_action_mask).toHaveLength(9);
        expect(inspection.decision_mask.target_action_mask).toHaveLength(11);
        expect(inspection.decision_mask.use_ultimate_action_mask).toHaveLength(2);
        expect(inspection.decision_mask.target_use_ultimate_joint_mask).toHaveLength(
          11,
        );
        expect(
          /** @type {boolean[][]} */ (
            inspection.decision_mask.target_use_ultimate_joint_mask
          ).every((row) => row.length === 2),
        ).toBe(true);
      }
      frames.push({ presentation, signature });
    }
    return frames;
  };

  const controllableMask = Object.freeze({
    move: [true, true, true, true, true, true, true, true, true],
    select_target: [
      true,
      false,
      false,
      false,
      false,
      false,
      true,
      true,
      true,
      true,
      true,
    ],
    use_ultimate: [true, true],
    select_target_use_ultimate_joint: [
      [true, false],
      [false, false],
      [false, false],
      [false, false],
      [false, false],
      [false, false],
      [true, true],
      [true, true],
      [true, false],
      [true, false],
      [true, true],
    ],
  });
  const stunnedMask = Object.freeze({
    move: [true, false, false, false, false, false, false, false, false],
    select_target: [
      true,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
    ],
    use_ultimate: [true, false],
    select_target_use_ultimate_joint: [
      [true, false],
      [false, false],
      [false, false],
      [false, false],
      [false, false],
      [false, false],
      [false, false],
      [false, false],
      [false, false],
      [false, false],
      [false, false],
    ],
  });
  /** @param {string} productUrl */
  const proveDurationOnePov = async (productUrl) => {
    await page.goto("about:blank");
    await openProduct(page, productUrl, "replay");
    for (let frameIndex = 0; frameIndex <= 3; frameIndex += 1) {
      if (frameIndex > 0) {
        await seekReplay(page, frameIndex);
      }
      const presentation = await authenticatedGet(page, "/api/presentation/frame");
      expect(presentation.presentation_kind).toBe("replay_no_shared_obs_agent_pov");
      expect(presentation.authority.recipient_public_agent_id).toBe("5");
      const expected =
        frameIndex === 1 || frameIndex === 2 ? stunnedMask : controllableMask;
      const mask = presentation.current_endpoint.parts.next_decision_action_mask;
      expect({
        move: mask.move,
        select_target: mask.select_target,
        use_ultimate: mask.use_ultimate,
        select_target_use_ultimate_joint: mask.select_target_use_ultimate_joint,
      }).toEqual(expected);
      expect({
        move: presentation.replay_inspection.decision_mask.movement_action_mask,
        select_target: presentation.replay_inspection.decision_mask.target_action_mask,
        use_ultimate:
          presentation.replay_inspection.decision_mask.use_ultimate_action_mask,
        select_target_use_ultimate_joint:
          presentation.replay_inspection.decision_mask.target_use_ultimate_joint_mask,
      }).toEqual(expected);
      expect(presentation.replay_inspection.submitted_action).toEqual(neutral);
      expect(presentation.replay_inspection.accepted_action).toEqual(neutral);
      expect(presentation.replay_inspection.decision_mask.movement_action_mask[0]).toBe(
        true,
      );
      expect(
        presentation.replay_inspection.decision_mask
          .target_use_ultimate_joint_mask[0][0],
      ).toBe(true);
      if (frameIndex === 1 || frameIndex === 2) {
        expect(
          presentation.replay_inspection.decision_mask.movement_action_mask[1],
        ).toBe(false);
        expect(
          presentation.replay_inspection.decision_mask
            .target_use_ultimate_joint_mask[6][1],
        ).toBe(false);
      }
      const self = /** @type {Record<string, any>[]} */ (
        presentation.current_endpoint.parts.scene.agents
      ).find((agent) => agent.relation === "self");
      if (!self) {
        throw new Error("Recovery POV lost its fixed self row.");
      }
      expect(
        /** @type {Record<string, any>[]} */ (self.statuses).some(
          ({ status_id }) => status_id === "rogue_poison_stun",
        ),
      ).toBe(frameIndex === 1 || frameIndex === 2);
      expect(
        /** @type {Record<string, any>[] | undefined} */ (
          presentation.latest_events?.cues
        )?.some(({ cue_type }) => cue_type === "respawn_wave_occurred") ?? false,
      ).toBe(false);
    }
  };

  /** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
  let activeReplay = null;
  /** @type {Awaited<ReturnType<typeof exportReplayArtifacts>> | null} */
  let sliceArtifacts = null;
  /** @type {unknown} */
  let testError = null;
  try {
    for (const contract of contracts) {
      activeReplay = await startReplayViewer({
        scenario: contract.name,
        includeStress: contract.includeStress === true,
        ...(contract.povSlot === undefined ? {} : { povSlot: contract.povSlot }),
        ...(contract.povFirst ? { view: "pov" } : {}),
      });
      if (contract.povFirst) {
        await proveDurationOnePov(activeReplay.url);
        await seekReplay(page, 1);
        await cp5Slice5AssertHiddenTransportNoninterference(
          page,
          activeReplay.url,
          "replay_no_shared_obs_agent_pov",
        );
        await seekReplay(page, 0);
        const researcherResponse = page.waitForResponse(
          (response) =>
            response.request().method() === "POST" &&
            new URL(response.url()).pathname === "/api/replay/command",
          { timeout: 30_000 },
        );
        await page.locator("#view-select").selectOption("researcher");
        expect((await researcherResponse).status()).toBe(200);
        await expect(page.locator("#view-select")).toHaveValue("researcher");
        await expect(page.locator("html")).toHaveAttribute(
          "data-audience",
          "researcher",
        );
      }
      const frames = await proveOracleTrajectory(activeReplay.url, contract);
      if (contract.name === "moving_basic_crossfire") {
        const expectedOutputs = [
          [
            ["source_damage_output", 0, 5, 13, 14.949999809265137],
            ["source_damage_output", 1, 6, 8, 8],
            ["source_damage_output", 2, 7, 6, 6.899999618530273],
            ["source_damage_output", 3, 8, 12, 12],
            ["source_healing_output", 4, 0, 8, 8],
            ["source_damage_output", 5, 0, 13, 14.949999809265137],
            ["source_damage_output", 6, 1, 8, 8],
            ["source_damage_output", 7, 2, 6, 6.899999618530273],
            ["source_damage_output", 8, 3, 12, 12],
            ["source_healing_output", 9, 5, 8, 8],
          ],
          [
            ["source_damage_output", 0, 5, 13, 14.949999809265137],
            ["source_damage_output", 1, 6, 8, 8],
            ["source_damage_output", 2, 7, 6, 6.899999618530273],
            ["source_damage_output", 3, 8, 12, 12],
            ["source_healing_output", 4, 0, 8, 8],
            ["source_damage_output", 5, 0, 13, 14.949999809265137],
            ["source_damage_output", 6, 1, 8, 9.199999809265137],
            ["source_damage_output", 7, 2, 6, 6.899999618530273],
            ["source_damage_output", 8, 3, 12, 12],
            ["source_healing_output", 9, 5, 8, 8],
          ],
        ];
        const expectedHealth = [
          [
            [0, 80, 14.949999809265137, 8, 73.05000305175781],
            [1, 200, 6.800000190734863, 0, 193.1999969482422],
            [2, 100, 6.899999618530273, 0, 93.0999984741211],
            [3, 100, 12, 0, 88],
            [5, 80, 14.949999809265137, 8, 73.05000305175781],
            [6, 200, 6.800000190734863, 0, 193.1999969482422],
            [7, 100, 6.899999618530273, 0, 93.0999984741211],
            [8, 100, 12, 0, 88],
          ],
          [
            [0, 73.05000305175781, 14.949999809265137, 8, 66.10000610351562],
            [1, 193.1999969482422, 7.820000171661377, 0, 185.37998962402344],
            [2, 93.0999984741211, 5.864999771118164, 0, 87.23500061035156],
            [3, 88, 12, 0, 76],
            [5, 73.05000305175781, 12.707500457763672, 8, 68.34249877929688],
            [6, 193.1999969482422, 6.800000190734863, 0, 186.39999389648438],
            [7, 93.0999984741211, 6.899999618530273, 0, 86.19999694824219],
            [8, 88, 12, 0, 76],
          ],
        ];
        expectExactInstalledSurface(
          frames[1],
          "moving Basic F1",
          [
            [0, "ability_activated", 0, 5, null, null],
            [1, "ability_activated", 1, 6, null, null],
            [2, "ability_activated", 2, 7, null, null],
            [3, "ability_activated", 3, 8, null, null],
            [4, "ability_activated", 4, 0, null, null],
            [5, "ability_activated", 5, 0, null, null],
            [6, "ability_activated", 6, 1, null, null],
            [7, "ability_activated", 7, 2, null, null],
            [8, "ability_activated", 8, 3, null, null],
            [9, "ability_activated", 9, 5, null, null],
            [20, "recipient_health_resolution", null, null, 0, null],
            [21, "recipient_health_resolution", null, null, 1, null],
            [22, "recipient_health_resolution", null, null, 2, null],
            [23, "recipient_health_resolution", null, null, 3, null],
            [24, "recipient_health_resolution", null, null, 5, null],
            [25, "recipient_health_resolution", null, null, 6, null],
            [26, "recipient_health_resolution", null, null, 7, null],
            [27, "recipient_health_resolution", null, null, 8, null],
            [46, "status_applied", 4, null, 0, null],
            [47, "status_applied", 7, null, 2, null],
            [48, "status_applied", 9, null, 5, null],
            [49, "status_applied", 2, null, 7, null],
          ],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
        for (let frameIndex = 1; frameIndex <= 2; frameIndex += 1) {
          const previous = frames[frameIndex - 1].presentation;
          const current = frames[frameIndex].presentation;
          const previousSlots = slotDirectory(previous);
          const currentSlots = slotDirectory(current);
          const incomingEvents = /** @type {Record<string, any>[]} */ (
            current.latest_events.events
          );
          expect(
            incomingEvents
              .filter(({ event_kind }) =>
                ["source_damage_output", "source_healing_output"].includes(event_kind),
              )
              .map((event) => [
                event.event_kind,
                eventSourceSlot(event, currentSlots),
                eventRecipientSlot(event, currentSlots),
                event.raw_damage_output ?? event.raw_healing_output,
                event.source_modified_damage_output ??
                  event.source_modified_healing_output,
              ]),
            `moving Basic T${frameIndex - 1} source output rows`,
          ).toEqual(expectedOutputs[frameIndex - 1]);
          expect(
            incomingEvents
              .filter(({ event_kind }) => event_kind === "recipient_health_resolution")
              .map((event) => [
                eventRecipientSlot(event, currentSlots),
                event.transition_start_health,
                event.total_effective_damage,
                event.total_effective_healing,
                event.health_after_combat_resolution,
              ]),
            `moving Basic T${frameIndex - 1} recipient health rows`,
          ).toEqual(expectedHealth[frameIndex - 1]);
          const currentAgents = new Map(
            oracleSceneAgents(current).map((agent) => [
              currentSlots.get(agent.public_agent_id),
              agent,
            ]),
          );
          const previousAgents = new Map(
            oracleSceneAgents(previous).map((agent) => [
              previousSlots.get(agent.public_agent_id),
              agent,
            ]),
          );
          for (const slot of currentAgents.keys()) {
            const currentAgent = currentAgents.get(slot);
            const previousAgent = previousAgents.get(slot);
            if (!currentAgent || !previousAgent) {
              throw new Error(`Moving Basic g${slot} lost its scene join.`);
            }
            expect(currentAgent.position).not.toEqual(previousAgent.position);
          }
          for (const event of signaturePlanEvents(frames[frameIndex].signature).filter(
            ({ kind }) => kind === "activation",
          )) {
            const source = currentAgents.get(
              currentSlots.get(event.sourcePublicAgentId),
            );
            const target = currentAgents.get(
              currentSlots.get(event.targetPublicAgentId),
            );
            if (!source || !target) {
              throw new Error("Moving Basic plan lost an authorized endpoint.");
            }
            expect(event.source).toEqual({
              x: source.position[0] * 10,
              y: source.position[1] * 10,
            });
            expect(event.target).toEqual({
              x: target.position[0] * 10,
              y: target.position[1] * 10,
            });
          }
        }
      }
      if (contract.name === "mirrored_ultimates") {
        const burstEvents = /** @type {Record<string, any>[]} */ (
          frames[1].presentation.latest_events.events
        );
        const burstSlots = slotDirectory(frames[1].presentation);
        expect(
          burstEvents
            .filter(({ event_kind }) => event_kind === "status_applied")
            .map((event) => [
              eventSourceSlot(event, burstSlots),
              eventRecipientSlot(event, burstSlots),
              event.status_id,
            ]),
        ).toEqual([
          [0, 0, "mage_burst_damage_amplification"],
          [5, 5, "mage_burst_damage_amplification"],
        ]);
        expect(
          burstEvents.filter(({ event_kind }) =>
            [
              "source_damage_output",
              "source_healing_output",
              "recipient_health_resolution",
            ].includes(event_kind),
          ),
          "Burst is a source-local status application, not an immediate health event",
        ).toEqual([]);
        expectExactInstalledSurface(
          frames[1],
          "mirrored Burst F1",
          [
            [0, "ability_activated", 0, null, null, null],
            [1, "ability_activated", 5, null, null, null],
            [2, "cooldown_started", null, null, null, 0],
            [3, "cooldown_started", null, null, null, 5],
            [6, "status_applied", 0, null, 0, null],
            [7, "status_applied", 5, null, 5, null],
          ],
          [],
        );
        const fullMove = indexedMask(9, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        /** @param {number[]} targetIndices @param {number[][]} jointCells */
        const legalMask = (targetIndices, jointCells) => ({
          move: fullMove,
          select_target: indexedMask(11, targetIndices),
          use_ultimate: [true, true],
          select_target_use_ultimate_joint: jointMask(jointCells),
        });
        const mirroredLegality = [
          {
            frameIndex: 0,
            cases: [0, 5].map((slot) => ({
              slot,
              submitted: tupleFromRow([slot, 1, 0, 1]),
              accepted: tupleFromRow([slot, 1, 0, 1]),
              mask: legalMask(
                [0],
                [
                  [0, 0],
                  [0, 1],
                ],
              ),
              submittedMoveLegal: true,
              submittedPairLegal: true,
            })),
          },
          {
            frameIndex: 1,
            cases: [1, 6].map((slot) => ({
              slot,
              submitted: tupleFromRow([slot, 0, 7, 1]),
              accepted: tupleFromRow([slot, 0, 7, 1]),
              mask: legalMask(
                [0, 7, 8],
                [
                  [0, 0],
                  [7, 1],
                  [8, 1],
                ],
              ),
              submittedMoveLegal: true,
              submittedPairLegal: true,
            })),
          },
          {
            frameIndex: 2,
            cases: [2, 7].map((slot) => ({
              slot,
              submitted: tupleFromRow([slot, 1, 8, 1]),
              accepted: tupleFromRow([slot, 1, 8, 1]),
              mask: legalMask(
                [0, 8],
                [
                  [0, 0],
                  [8, 0],
                  [8, 1],
                ],
              ),
              submittedMoveLegal: true,
              submittedPairLegal: true,
            })),
          },
          {
            frameIndex: 3,
            cases: [3, 8].map((slot) => ({
              slot,
              submitted: tupleFromRow([slot, 3, 9, 1]),
              accepted: tupleFromRow([slot, 3, 9, 1]),
              mask: legalMask(
                [0, 9],
                [
                  [0, 0],
                  [9, 0],
                  [9, 1],
                ],
              ),
              submittedMoveLegal: true,
              submittedPairLegal: true,
            })),
          },
          {
            frameIndex: 4,
            cases: [
              {
                slot: 4,
                mask: legalMask(
                  [0, 3, 4, 5],
                  [
                    [0, 0],
                    [3, 1],
                    [4, 1],
                    [5, 0],
                    [5, 1],
                  ],
                ),
              },
              {
                slot: 9,
                mask: legalMask(
                  [0, 3, 4, 5],
                  [
                    [0, 0],
                    [3, 1],
                    [4, 0],
                    [4, 1],
                    [5, 0],
                    [5, 1],
                  ],
                ),
              },
            ].map(({ slot, mask }) => ({
              slot,
              submitted: tupleFromRow([slot, 1, 4, 1]),
              accepted: tupleFromRow([slot, 1, 4, 1]),
              mask,
              submittedMoveLegal: true,
              submittedPairLegal: true,
            })),
          },
        ];
        for (const { frameIndex, cases } of mirroredLegality) {
          await proveOutgoingOwnerSet(contract.name, frameIndex, cases);
        }
      }
      if (contract.name === "charge_convergence") {
        const start = frames[0].presentation;
        const successor = frames[1].presentation;
        const startSlots = slotDirectory(start);
        const successorSlots = slotDirectory(successor);
        const incomingEvents = /** @type {Record<string, any>[]} */ (
          successor.latest_events.events
        );
        expect(
          incomingEvents
            .filter(({ event_kind }) => event_kind === "source_damage_output")
            .map((event) => [
              eventSourceSlot(event, successorSlots),
              eventRecipientSlot(event, successorSlots),
              event.raw_damage_output,
              event.source_modified_damage_output,
            ]),
          "Charge source output rows",
        ).toEqual([
          [0, 5, 20, 20],
          [1, 5, 20, 20],
          [5, 0, 20, 20],
        ]);
        expect(
          incomingEvents
            .filter(({ event_kind }) => event_kind === "recipient_health_resolution")
            .map((event) => [
              eventRecipientSlot(event, successorSlots),
              event.transition_start_health,
              event.total_effective_damage,
              event.total_effective_healing,
              event.health_after_combat_resolution,
            ]),
          "Charge recipient health rows",
        ).toEqual([
          [0, 200, 17, 0, 183],
          [5, 200, 34, 0, 166],
        ]);
        expect(
          incomingEvents
            .filter(({ event_kind }) => event_kind === "charge_phase_displacement")
            .map((event) => [
              eventSourceSlot(event, successorSlots),
              event.realized_displacement,
            ]),
          "Charge displacement rows retain acting-agent identity",
        ).toEqual([
          [0, [4.071523189544678, 1.5]],
          [1, [4.071523189544678, -1.5]],
          [5, [-4.071523189544678, -1.6286091804504395]],
        ]);
        expectExactInstalledSurface(
          frames[1],
          "Charge convergence F1",
          [
            [0, "ability_activated", 0, 5, null, null],
            [1, "ability_activated", 1, 5, null, null],
            [2, "ability_activated", 5, 0, null, null],
            [6, "recipient_health_resolution", null, null, 0, null],
            [7, "recipient_health_resolution", null, null, 5, null],
            [11, "cooldown_started", null, null, null, 0],
            [12, "cooldown_started", null, null, null, 1],
            [13, "cooldown_started", null, null, null, 5],
            [17, "status_applied", 5, null, 0, null],
            [18, "status_applied", 5, null, 0, null],
            [19, "status_applied", 0, null, 5, null],
            [21, "status_applied", 0, null, 5, null],
          ],
          [0, 1, 2],
        );
        const startAgents = new Map(
          oracleSceneAgents(start).map((agent) => [
            startSlots.get(agent.public_agent_id),
            agent,
          ]),
        );
        const successorAgents = new Map(
          oracleSceneAgents(successor).map((agent) => [
            successorSlots.get(agent.public_agent_id),
            agent,
          ]),
        );
        for (const event of signaturePlanEvents(frames[1].signature).filter(
          ({ kind }) => kind === "activation",
        )) {
          const sourceSlot = successorSlots.get(event.sourcePublicAgentId);
          const targetSlot = successorSlots.get(event.targetPublicAgentId);
          const startSource = startAgents.get(sourceSlot);
          const startTarget = startAgents.get(targetSlot);
          const successorSource = successorAgents.get(sourceSlot);
          if (!startSource || !startTarget || !successorSource) {
            throw new Error("Charge plan lost an authorized endpoint.");
          }
          expect(successorSource.position).not.toEqual(startSource.position);
          expect(event.source).toEqual({
            x: startSource.position[0] * 10,
            y: startSource.position[1] * 10,
          });
          expect(event.target).toEqual({
            x: startTarget.position[0] * 10,
            y: startTarget.position[1] * 10,
          });
          expect(event.route).not.toBeNull();
        }
        const chargePlan = signaturePlanEvents(frames[1].signature).filter(
          ({ eventType }) => eventType === "charge_phase_displacement",
        );
        expect(
          chargePlan.map((event) => [
            event.eventType,
            event.kind,
            event.spatial,
            event.start,
            event.end,
            event.route,
          ]),
          "Charge phase displacement remains authorized but creates no overlay",
        ).toEqual([
          ["charge_phase_displacement", "feed_only", false, null, null, null],
          ["charge_phase_displacement", "feed_only", false, null, null, null],
          ["charge_phase_displacement", "feed_only", false, null, null, null],
        ]);
        for (const event of chargePlan) {
          expect(event.sourcePublicAgentId).toBeNull();
          expect(event.sourcePresentationKey).toBeNull();
          expect(event.persistent).toBe(false);
        }
      }
      if (contract.name === "recovery_refresh_cycle") {
        await seekReplay(page, 1);
        await cp5Slice5AssertHiddenTransportNoninterference(
          page,
          activeReplay.url,
          "replay_oracle",
        );
        const expectedStatuses = [
          {},
          {
            5: {
              rogue_poison_stun: 1,
              rogue_poison_slow: 5,
              rogue_poison_anti_heal: 4,
            },
            7: { hunter_trap_stun: 4 },
          },
          {
            5: {
              rogue_poison_stun: 1,
              rogue_poison_slow: 5,
              rogue_poison_anti_heal: 4,
            },
            7: { hunter_trap_stun: 4 },
          },
          {
            5: { rogue_poison_slow: 4, rogue_poison_anti_heal: 3 },
            7: { hunter_trap_stun: 3 },
          },
          {
            5: { rogue_poison_slow: 3, rogue_poison_anti_heal: 2 },
            7: { hunter_trap_stun: 2 },
          },
          {
            5: { rogue_poison_slow: 2, rogue_poison_anti_heal: 1 },
            7: { hunter_trap_stun: 1 },
          },
          { 5: { rogue_poison_slow: 1 } },
          {},
        ];
        for (const [frameIndex, frame] of frames.entries()) {
          const slots = slotDirectory(frame.presentation);
          /** @type {Record<string, Record<string, number>>} */
          const actual = {};
          for (const agent of oracleSceneAgents(frame.presentation)) {
            if (agent.statuses.length > 0) {
              actual[String(slots.get(agent.public_agent_id))] = Object.fromEntries(
                /** @type {Record<string, any>[]} */ (agent.statuses).map(
                  ({ status_id, remaining_duration }) => [
                    status_id,
                    remaining_duration,
                  ],
                ),
              );
            }
          }
          expect(actual, `recovery F${frameIndex} duration 0/1/many`).toEqual(
            expectedStatuses[frameIndex],
          );
        }
        /**
         * @param {number} frameIndex
         * @param {number} primaryOrdinal
         * @param {string} eventType
         * @param {string} tokenId
         * @param {string} lifecycle
         * @param {string} lifecycleLabel
         * @param {string} lifecycleAccessibleName
         * @param {number[]} atomicOrdinals
         * @param {number[]} applicationOrdinals
         */
        const lifecycleEvent = (
          frameIndex,
          primaryOrdinal,
          eventType,
          tokenId,
          lifecycle,
          lifecycleLabel,
          lifecycleAccessibleName,
          atomicOrdinals,
          applicationOrdinals,
        ) => {
          const transitionId =
            frames[frameIndex].presentation.latest_events.incoming_transition_id;
          /** @param {number} ordinal */
          const eventId = (ordinal) =>
            `${transitionId}:event:${String(ordinal).padStart(4, "0")}`;
          return {
            eventId: eventId(primaryOrdinal),
            eventType,
            tokenId,
            lifecycle,
            lifecycleLabel,
            lifecycleAccessibleName,
            atomicEventIds: atomicOrdinals.map(eventId),
            applicationEventIds: applicationOrdinals.map(eventId),
          };
        };
        const expectedLifecycleByFrame = [
          [],
          [
            lifecycleEvent(
              1,
              15,
              "status_applied",
              "slow_rogue_poison",
              "applied",
              "Applied",
              "Status applied",
              [15],
              [15],
            ),
            lifecycleEvent(
              1,
              16,
              "status_applied",
              "stun_rogue_poison",
              "applied",
              "Applied",
              "Status applied",
              [16],
              [16],
            ),
            lifecycleEvent(
              1,
              17,
              "status_applied",
              "anti_heal_rogue_poison",
              "applied",
              "Applied",
              "Status applied",
              [17],
              [17],
            ),
            lifecycleEvent(
              1,
              18,
              "status_applied",
              "stun_hunter_trap",
              "applied",
              "Applied",
              "Status applied",
              [18],
              [18],
            ),
          ],
          [
            lifecycleEvent(
              2,
              16,
              "status_applied",
              "slow_rogue_poison",
              "applied",
              "Applied",
              "Status applied",
              [16, 17],
              [16],
            ),
            lifecycleEvent(
              2,
              19,
              "status_applied",
              "stun_rogue_poison",
              "applied",
              "Applied",
              "Status applied",
              [18, 19],
              [19],
            ),
            lifecycleEvent(
              2,
              20,
              "status_applied",
              "anti_heal_rogue_poison",
              "applied",
              "Applied",
              "Status applied",
              [20, 21],
              [20],
            ),
            lifecycleEvent(
              2,
              23,
              "status_applied",
              "stun_hunter_trap",
              "applied",
              "Applied",
              "Status applied",
              [22, 23],
              [23],
            ),
          ],
          [
            lifecycleEvent(
              3,
              1,
              "status_aged_to_zero",
              "stun_rogue_poison",
              "expired",
              "Expired",
              "Status expired naturally",
              [1],
              [],
            ),
          ],
          [
            lifecycleEvent(
              4,
              0,
              "agent_left_combat",
              "in_combat",
              "expired",
              "Expired",
              "Status expired naturally",
              [0],
              [],
            ),
          ],
          [
            lifecycleEvent(
              5,
              0,
              "agent_left_combat",
              "in_combat",
              "expired",
              "Expired",
              "Status expired naturally",
              [0],
              [],
            ),
          ],
          [
            lifecycleEvent(
              6,
              0,
              "agent_left_combat",
              "in_combat",
              "expired",
              "Expired",
              "Status expired naturally",
              [0],
              [],
            ),
            lifecycleEvent(
              6,
              2,
              "status_aged_to_zero",
              "anti_heal_rogue_poison",
              "expired",
              "Expired",
              "Status expired naturally",
              [2],
              [],
            ),
            lifecycleEvent(
              6,
              3,
              "status_aged_to_zero",
              "stun_hunter_trap",
              "expired",
              "Expired",
              "Status expired naturally",
              [3],
              [],
            ),
          ],
          [
            lifecycleEvent(
              7,
              0,
              "agent_left_combat",
              "in_combat",
              "expired",
              "Expired",
              "Status expired naturally",
              [0],
              [],
            ),
            lifecycleEvent(
              7,
              1,
              "agent_left_combat",
              "in_combat",
              "expired",
              "Expired",
              "Status expired naturally",
              [1],
              [],
            ),
            lifecycleEvent(
              7,
              2,
              "agent_left_combat",
              "in_combat",
              "expired",
              "Expired",
              "Status expired naturally",
              [2],
              [],
            ),
            lifecycleEvent(
              7,
              4,
              "agent_left_combat",
              "in_combat",
              "expired",
              "Expired",
              "Status expired naturally",
              [4],
              [],
            ),
            lifecycleEvent(
              7,
              5,
              "status_aged_to_zero",
              "slow_rogue_poison",
              "expired",
              "Expired",
              "Status expired naturally",
              [5],
              [],
            ),
          ],
        ];
        const expectedLifecycleOwnerSlotsByFrame = [
          [],
          [
            [0, 5],
            [0, 5],
            [0, 5],
            [2, 7],
          ],
          [
            [1, 5],
            [1, 5],
            [1, 5],
            [3, 7],
          ],
          [[null, 5]],
          [[null, 0]],
          [[null, 1]],
          [
            [null, 2],
            [null, 5],
            [null, 7],
          ],
          [
            [null, 3],
            [null, 4],
            [null, 5],
            [null, 7],
            [null, 5],
          ],
        ];
        for (const [frameIndex, frame] of frames.entries()) {
          const expectedLifecycle = expectedLifecycleByFrame[frameIndex];
          if (expectedLifecycle.length > 0) {
            expect(
              frame.signature.dom.statusLifecycles.length,
              `recovery F${frameIndex} installs lifecycle DOM`,
            ).toBeGreaterThan(0);
          }
          expect(
            signaturePlanEvents(frame.signature)
              .filter(({ kind }) => kind === "status_lifecycle")
              .map(
                ({
                  eventId,
                  eventType,
                  tokenId,
                  lifecycle,
                  lifecycleLabel,
                  lifecycleAccessibleName,
                  atomicEventIds,
                  applicationEventIds,
                }) => ({
                  eventId,
                  eventType,
                  tokenId,
                  lifecycle,
                  lifecycleLabel,
                  lifecycleAccessibleName,
                  atomicEventIds,
                  applicationEventIds,
                }),
              ),
            `recovery F${frameIndex} plan lifecycle`,
          ).toEqual(expectedLifecycle);
          if (frameIndex === 2) {
            const refreshEventIds = frame.presentation.latest_events.events
              .filter(
                (/** @type {Record<string, any>} */ { event_kind }) =>
                  event_kind === "status_refreshed_or_extended",
              )
              .map((/** @type {Record<string, any>} */ { event_id }) => event_id);
            expect(refreshEventIds).toHaveLength(2);
            expect(
              expectedLifecycle.flatMap(({ atomicEventIds }) => atomicEventIds),
            ).toEqual(expect.arrayContaining(refreshEventIds));
            expect(
              frame.signature.dom.statusLifecycles.filter(
                (/** @type {Record<string, any>} */ { type }) =>
                  type === "status_refreshed_or_extended",
              ),
            ).toEqual([]);
          }
          expect(
            frame.signature.dom.statusLifecycles,
            `recovery F${frameIndex} installed lifecycle DOM/copy`,
          ).toEqual(
            expectedLifecycle.map((event) => ({
              id: event.eventId,
              type: event.eventType,
              tokenId: event.tokenId,
              lifecycle: event.lifecycle,
              persistent: false,
              atomicEventIds: event.atomicEventIds,
              applicationEventIds: event.applicationEventIds,
              tooltipKind: "event",
              tooltipTitle: event.lifecycleLabel,
              tooltipSummary: event.lifecycleAccessibleName,
            })),
          );
          const frameSlots = slotDirectory(frame.presentation);
          const frameKeysBySlot = new Map(
            oracleSceneAgents(frame.presentation).map((agent) => [
              frameSlots.get(agent.public_agent_id),
              agent.presentation_key,
            ]),
          );
          const lifecycleEffectsById = new Map(
            /** @type {Record<string, any>[]} */ (frame.signature.dom.effects).map(
              (effect) => [effect.id, effect],
            ),
          );
          expect(
            expectedLifecycle.map(({ eventId }) => {
              const effect = lifecycleEffectsById.get(eventId);
              if (!effect) {
                throw new Error(
                  `recovery F${frameIndex} lost lifecycle owner ${eventId}.`,
                );
              }
              return {
                id: effect.id,
                sourceKey: effect.sourceKey,
                recipientKey: effect.recipientKey,
              };
            }),
            `recovery F${frameIndex} authorized lifecycle owners`,
          ).toEqual(
            expectedLifecycleOwnerSlotsByFrame[frameIndex].map(
              ([sourceSlot, recipientSlot], index) => ({
                id: expectedLifecycle[index].eventId,
                sourceKey: sourceSlot === null ? null : frameKeysBySlot.get(sourceSlot),
                recipientKey: frameKeysBySlot.get(recipientSlot),
              }),
            ),
          );
        }
        const fullMove = indexedMask(9, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        await proveOutgoingOwnerSet(contract.name, 0, [
          {
            slot: 6,
            submitted: tupleFromRow([6, 0, 2, 1]),
            accepted: neutral,
            mask: {
              move: fullMove,
              select_target: indexedMask(11, [0, 2]),
              use_ultimate: [true, false],
              select_target_use_ultimate_joint: jointMask([
                [0, 0],
                [2, 0],
              ]),
            },
            submittedMoveLegal: true,
            submittedPairLegal: false,
          },
        ]);
        await proveOutgoingOwnerSet(contract.name, 1, [
          {
            slot: 5,
            submitted: neutral,
            accepted: neutral,
            mask: stunnedMask,
            submittedMoveLegal: true,
            submittedPairLegal: true,
          },
        ]);
      }
      if (contract.name === "death_respawn_cycle") {
        const fullMove = indexedMask(9, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        const noOpMask = {
          move: indexedMask(9, [0]),
          select_target: indexedMask(11, [0]),
          use_ultimate: [true, false],
          select_target_use_ultimate_joint: jointMask([[0, 0]]),
        };
        await proveOutgoingOwnerSet(contract.name, 2, [
          {
            slot: 5,
            submitted: tupleFromRow([5, 4, 6, 1]),
            accepted: neutral,
            mask: noOpMask,
            submittedMoveLegal: false,
            submittedPairLegal: false,
          },
        ]);
        await proveOutgoingOwnerSet(contract.name, 3, [
          {
            slot: 5,
            submitted: tupleFromRow([5, 4, 6, 0]),
            accepted: tupleFromRow([5, 4, 0, 0]),
            mask: {
              move: fullMove,
              select_target: indexedMask(11, [0]),
              use_ultimate: [true, false],
              select_target_use_ultimate_joint: jointMask([[0, 0]]),
            },
            submittedMoveLegal: true,
            submittedPairLegal: false,
          },
        ]);
        await proveOutgoingOwnerSet(contract.name, 6, [
          {
            slot: 5,
            submitted: tupleFromRow([5, 0, 6, 0]),
            accepted: tupleFromRow([5, 0, 6, 0]),
            mask: {
              move: fullMove,
              select_target: indexedMask(11, [0, 6]),
              use_ultimate: [true, true],
              select_target_use_ultimate_joint: jointMask([
                [0, 0],
                [6, 0],
                [6, 1],
              ]),
            },
            submittedMoveLegal: true,
            submittedPairLegal: true,
          },
        ]);
      }
      await stopDebugger(activeReplay.process);
      activeReplay = null;
    }

    sliceArtifacts = await exportReplayArtifacts();
    activeReplay = await startReplayViewer({
      replayPath: sliceArtifacts.shared,
      view: "pov",
      povSlot: 0,
    });
    await page.goto("about:blank");
    await openProduct(page, activeReplay.url, "replay");
    await seekReplay(page, 1);
    const shared = await authenticatedGet(page, "/api/presentation/frame");
    expect(shared.presentation_kind).toBe("replay_shared_obs_agent_pov");
    expect(
      /** @type {Record<string, any>[]} */ (shared.latest_events.deltas).some(
        ({ delta_kind }) => delta_kind === "respawn_wave_occurred",
      ),
    ).toBe(false);
    await cp5Slice5AssertHiddenTransportNoninterference(
      page,
      activeReplay.url,
      "replay_shared_obs_agent_pov",
    );
    await stopDebugger(activeReplay.process);
    activeReplay = null;
    await removeReplayArtifacts(sliceArtifacts.outputDirectory);
    sliceArtifacts = null;

    const checkedSampleBytesAfter = await cp5Slice5CheckedSampleSnapshot();
    expect(Object.keys(checkedSampleBytesAfter)).toEqual(
      Object.keys(checkedSampleBytesBefore),
    );
    for (const [fileName, before] of Object.entries(checkedSampleBytesBefore)) {
      expect(checkedSampleBytesAfter[fileName].sha256).toBe(before.sha256);
      expect(checkedSampleBytesAfter[fileName].bytes.equals(before.bytes)).toBe(true);
    }
    expect(browserErrors.get(page) ?? []).toEqual([]);
  } catch (error) {
    testError = error;
  }
  const cleanup = await Promise.allSettled([
    stopDebugger(activeReplay?.process ?? null),
    removeReplayArtifacts(sliceArtifacts?.outputDirectory),
  ]);
  const cleanupErrors = cleanup.flatMap((result) =>
    result.status === "rejected" ? [result.reason] : [],
  );
  if (testError !== null || cleanupErrors.length > 0) {
    throw new AggregateError(
      [...(testError === null ? [] : [testError]), ...cleanupErrors],
      "CP5 Slice 5 public causal/privacy proof or cleanup failed.",
    );
  }
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
    let leaf = await expectInstalledLeaf(
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
    await expectAgentAuthoritySurface(page, leaf.presentation, []);
    await expectAuthorizedIncomingTransitionDom(page, leaf.presentation);
    await expectReplayInspectionDom(page, leaf.presentation);
    await expectTechnicalFrameDom(page, leaf.presentation);
    await expectLatestTransitionDom(page, leaf.presentation);
    await expectRetiredMetadataAbsent(page);
    if (frameIndex === 0) {
      await expect(
        page.locator("#events-details, #event-feed, #event-count"),
      ).toHaveCount(0);
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
      const researcherAgent = leaf.presentation.researcher_space.roster_agents.find(
        (/** @type {Record<string, any>} */ agent) =>
          agent.public_agent_id === localAgent.public_agent_id,
      );
      expect(researcherAgent).toBeTruthy();
      const researcherIdentity =
        leaf.presentation.researcher_space.identity_directory.identities.find(
          (/** @type {Record<string, any>} */ identity) =>
            identity.public_agent_id === localAgent.public_agent_id,
        );
      expect(researcherIdentity).toBeTruthy();
      const researcherSlot =
        (researcherIdentity.team_id - 1) * 5 + researcherIdentity.team_local_slot;
      const localRow = page.locator(
        `#roster .roster-primary-action[data-presentation-key="${researcherAgent.presentation_key}"]`,
      );
      const cursorBeforeSwitch = structuredClone(leaf.transport.cursor);
      await expectSingleActivationCommand(
        page,
        "/api/replay/command",
        () => localRow.click(),
        { command_type: "set_pov_actor", global_slot: researcherSlot },
      );
      leaf = await expectInstalledLeaf(
        page,
        "shared_obs_agent_pov_replay_viewer",
        "replay_shared_obs_agent_pov",
      );
      expect(leaf.transport.cursor).toEqual(cursorBeforeSwitch);
      expect(leaf.presentation.authority.recipient_public_agent_id).toBe(
        localAgent.public_agent_id,
      );
      expect(leaf.presentation.authority.recipient_presentation_key).not.toBe(
        recipientKey,
      );
      await expect(
        page.locator(`#battlefield [data-presentation-key="${localKey}"]`),
      ).toHaveCount(0);
      await expectAgentAuthoritySurface(page, leaf.presentation, []);
      await expect(page.locator("#battlefield")).toBeFocused();
      const rangesButton = page.locator("#replay-ranges-button");
      await expectZeroCommandInteraction(page, () => rangesButton.click());
      await expect(rangesButton).toHaveAttribute("aria-pressed", "false");
      await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(
        0,
      );
      await expectZeroCommandInteraction(page, () => rangesButton.click());
      await expect(rangesButton).toHaveAttribute("aria-pressed", "true");
      const afterLocalActions = await authenticatedGet(page, "/api/presentation/frame");
      expect(afterLocalActions.authority.recipient_presentation_key).toBe(
        leaf.presentation.authority.recipient_presentation_key,
      );
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
      await expect(page.locator("#battlefield .agent[role=button]")).not.toHaveCount(0);
      await expect(page.locator("#battlefield-instructions")).toContainText(
        "switch the fog-of-war recipient",
      );
    }
    installed.push(leaf);
  }

  await expectZeroCommandInteraction(page, () =>
    page.locator("#replay-clear-reference-button").click(),
  );
  await expect(page.locator('#battlefield .agent[data-selected="true"]')).toHaveCount(
    0,
  );
  await expect(page.locator('[data-layer="debug-range"] .range-ring')).toHaveCount(0);
  await expect(page.locator("#agent-details")).not.toHaveAttribute("open", "");
  await expect(page.locator("#replay-clear-reference-button")).toBeDisabled();
  const afterLocalClear = await authenticatedGet(page, "/api/presentation/frame");
  expect(afterLocalClear.authority.recipient_presentation_key).toBe(
    installed[2].presentation.authority.recipient_presentation_key,
  );

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
  await expect(page.locator("#accepted-card .accepted-action-row")).toHaveCount(
    middlePresentation.researcher_space.latest_transition.action_rows.length,
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

test("real death cycle retains truthful outward lifecycle cues at both review viewports", async ({
  page,
}) => {
  if (!deathReplay) {
    throw new Error("Death replay service is unavailable.");
  }
  for (const viewport of [
    { width: 960, height: 600 },
    { width: 1440, height: 900 },
  ]) {
    await page.setViewportSize(viewport);
    await page.goto("about:blank");
    await openProduct(page, deathReplay.url, "replay");
    await seekReplay(page, 0);
    const initial = await authenticatedGet(page, "/api/presentation/frame");
    expect(initial.source.source_frame_index).toBe(0);
    expect(initial.source.source_final_frame_index).toBe(7);
    expect(initial.latest_events).toBeNull();
    const subject = initial.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
    );
    if (!subject) {
      throw new Error("Death replay subject is absent from frame zero.");
    }
    const subjectSelector = `.agent[data-presentation-key="${subject.presentation_key}"]`;
    await expect(page.locator(".combat-lifecycle-ring")).toHaveCount(0);
    await expect(page.locator(".combat-respawn-wave")).toHaveCount(0);

    await seekReplay(page, 1);
    const deathFrame = await authenticatedGet(page, "/api/presentation/frame");
    const deathEvents = /** @type {Record<string, any>[]} */ (
      deathFrame.latest_events.events
    );
    expect(deathFrame.latest_events.ordered_event_ids).toEqual(
      deathEvents.map(({ event_id }) => event_id),
    );
    const deathEvent = deathEvents.find(
      ({ event_kind }) => event_kind === "agent_died",
    );
    if (!deathEvent) {
      throw new Error("Authorized death event is unavailable.");
    }
    const deathEffect = page.locator('.combat-effect[data-event-type="agent_died"]');
    await expect(deathEffect).toHaveCount(1);
    await expect(deathEffect).toHaveAttribute("data-event-id", deathEvent.event_id);
    await expect(deathEffect).toHaveAttribute("data-persistent", "true");
    await expect(deathEffect).toHaveAttribute("data-settled", "true");
    await expect(page.locator(subjectSelector)).toHaveAttribute("data-alive", "false");
    const deathRing = deathEffect.locator(".combat-lifecycle-ring--death");
    expect(
      await deathRing.evaluate((node) =>
        Array.from(node.attributes, ({ name }) => name).filter((name) =>
          name.startsWith("data-layout-"),
        ),
      ),
    ).toEqual([]);
    await expect(deathEffect.locator(".combat-cue__leader--semantic")).toHaveCount(0);
    const deathGeometry = await deathEffect.evaluate((effect, selector) => {
      const ringGroup = effect.querySelector(".combat-lifecycle-ring--death");
      const ring = effect.querySelector(".combat-lifecycle-ring__ring");
      const hit = effect.querySelector(".combat-lifecycle-ring__hit");
      const body = document.querySelector(`${selector} .agent-body`);
      if (
        !(ringGroup instanceof SVGGraphicsElement) ||
        !(ring instanceof SVGCircleElement) ||
        !(hit instanceof SVGCircleElement) ||
        !(body instanceof SVGCircleElement)
      ) {
        throw new Error("Settled death geometry is unavailable.");
      }
      const matrix = ringGroup.getScreenCTM();
      const bodyMatrix = body.getScreenCTM();
      if (matrix === null || bodyMatrix === null) {
        throw new Error("Settled death transform is unavailable.");
      }
      const center = new DOMPoint(0, 0).matrixTransform(matrix);
      const bodyCenter = new DOMPoint(
        body.cx.baseVal.value,
        body.cy.baseVal.value,
      ).matrixTransform(bodyMatrix);
      const centerElements = document.elementsFromPoint(bodyCenter.x, bodyCenter.y);
      return {
        color: getComputedStyle(ringGroup).color,
        radius: ring.getAttribute("r"),
        strokeOpacity: getComputedStyle(ring).strokeOpacity,
        hitPointerEvents: getComputedStyle(hit).pointerEvents,
        hitFill: getComputedStyle(hit).fill,
        hitOwnsBodyCenter: centerElements.includes(hit),
        center: [center.x, center.y],
        bodyCenter: [bodyCenter.x, bodyCenter.y],
      };
    }, subjectSelector);
    expect(deathGeometry.color).toBe("rgb(251, 113, 133)");
    expect(deathGeometry.radius).toBe("32");
    expect(deathGeometry.strokeOpacity).toBe("0.5");
    expect(deathGeometry.hitPointerEvents).toBe("stroke");
    expect(deathGeometry.hitFill).toBe("none");
    expect(deathGeometry.hitOwnsBodyCenter).toBe(false);
    expect(deathGeometry.center.every(Number.isFinite)).toBe(true);
    expect(
      Math.abs(deathGeometry.center[0] - deathGeometry.bodyCenter[0]),
    ).toBeLessThanOrEqual(0.001);
    expect(
      Math.abs(deathGeometry.center[1] - deathGeometry.bodyCenter[1]),
    ).toBeLessThanOrEqual(0.001);
    await expect(deathEffect).toHaveAttribute("aria-label", "Agent died");
    await deathEffect.focus();
    await expect(page.locator("#visual-tooltip-title")).toHaveText("Agent died");
    await expect(
      page.locator("#visual-tooltip .semantic-explanation__summary"),
    ).toHaveText("This agent died on the incoming transition.");
    expect(await deathEffect.getAttribute("aria-label")).not.toMatch(
      /_|semantic pulse/iu,
    );

    await seekReplay(page, 2);
    const waitFrame = await authenticatedGet(page, "/api/presentation/frame");
    expect(waitFrame.latest_events.events).toEqual([]);
    await expect(page.locator(subjectSelector)).toHaveAttribute("data-alive", "false");
    await expect(page.locator(".combat-lifecycle-ring")).toHaveCount(0);
    await expect(page.locator(".combat-respawn-wave")).toHaveCount(0);

    await seekReplay(page, 3);
    const respawnFrame = await authenticatedGet(page, "/api/presentation/frame");
    const respawnEvents = /** @type {Record<string, any>[]} */ (
      respawnFrame.latest_events.events
    );
    expect(respawnFrame.latest_events.ordered_event_ids).toEqual(
      respawnEvents.map(({ event_id }) => event_id),
    );
    expect(respawnEvents.map(({ event_kind }) => event_kind)).toEqual([
      "action_rejected",
      "action_rejected",
      "respawn_wave_occurred",
      "agent_respawned",
    ]);
    const waveEvent = respawnEvents[2];
    const respawnEvent = respawnEvents[3];
    const waveEffect = page.locator(
      '.combat-effect[data-event-type="respawn_wave_occurred"]',
    );
    const wave = waveEffect.locator(".combat-respawn-wave");
    await expect(waveEffect).toHaveCount(1);
    await expect(waveEffect).toHaveAttribute("data-event-id", waveEvent.event_id);
    await expect(waveEffect).toHaveAttribute("data-persistent", "true");
    await expect(waveEffect).toHaveAttribute("data-settled", "true");
    await expect(waveEffect).toHaveAttribute("data-team-index", "1");
    await expect(waveEffect).toHaveAttribute("data-team-id", "2");
    await expect(waveEffect).toHaveAttribute("data-team-side", "right");
    await expect(wave).toHaveAttribute(
      "data-layout-key",
      JSON.stringify(["event", waveEvent.event_id, "cue"]),
    );
    await expect(wave).toHaveAttribute("data-layout-disposition", "perimeter_callout");
    await expect(wave).toHaveAttribute("data-layout-collision-free", "true");
    await expect(wave.locator(".combat-respawn-wave__label")).toHaveText(
      "EVENT: Team B Respawn",
    );
    await expect(waveEffect).toHaveAttribute("aria-label", "EVENT: Team B Respawn");
    await waveEffect.focus();
    await expect(page.locator("#visual-tooltip-title")).toHaveText(
      "EVENT: Team B Respawn",
    );
    await expect(
      page.locator("#visual-tooltip .semantic-explanation__summary"),
    ).toHaveText("This team respawn occurred on the incoming transition.");
    const waveGeometry = await wave.evaluate((node) => {
      const map = document.querySelector(".map-boundary");
      const label = node.querySelector(".combat-respawn-wave__label");
      if (!(map instanceof SVGGraphicsElement) || !(label instanceof SVGTextElement)) {
        throw new Error("Settled wave geometry is unavailable.");
      }
      const waveBounds = node.getBoundingClientRect();
      const mapBounds = map.getBoundingClientRect();
      return {
        color: getComputedStyle(node).color,
        label: label.textContent,
        withinMap:
          waveBounds.left >= mapBounds.left &&
          waveBounds.top >= mapBounds.top &&
          waveBounds.right <= mapBounds.right &&
          waveBounds.bottom <= mapBounds.bottom,
        rightSided:
          waveBounds.left + waveBounds.width / 2 > mapBounds.left + mapBounds.width / 2,
        childTooltipOwners: node.querySelectorAll("[data-tooltip-owner]").length,
      };
    });
    expect(waveGeometry).toEqual({
      color: "rgb(240, 90, 103)",
      label: "EVENT: Team B Respawn",
      withinMap: true,
      rightSided: true,
      childTooltipOwners: 0,
    });

    const respawnEffect = page.locator(
      '.combat-effect[data-event-type="agent_respawned"]',
    );
    await expect(respawnEffect).toHaveCount(1);
    await expect(respawnEffect).toHaveAttribute("data-event-id", respawnEvent.event_id);
    await expect(respawnEffect).toHaveAttribute("data-persistent", "true");
    await expect(respawnEffect).toHaveAttribute("data-settled", "true");
    await expect(page.locator(subjectSelector)).toHaveAttribute("data-alive", "true");
    await expect(page.locator(subjectSelector)).toHaveAttribute(
      "data-spawn-shield-remaining",
      "3",
    );
    const respawnRing = respawnEffect.locator(".combat-lifecycle-ring--resurrection");
    expect(
      await respawnRing.evaluate((node) =>
        Array.from(node.attributes, ({ name }) => name).filter((name) =>
          name.startsWith("data-layout-"),
        ),
      ),
    ).toEqual([]);
    await expect(respawnEffect.locator(".combat-cue__leader--semantic")).toHaveCount(0);
    const respawnGeometry = await respawnEffect.evaluate((effect, selector) => {
      const ringGroup = effect.querySelector(".combat-lifecycle-ring--resurrection");
      const ring = effect.querySelector(".combat-lifecycle-ring__ring");
      const hit = effect.querySelector(".combat-lifecycle-ring__hit");
      const body = document.querySelector(`${selector} .agent-body`);
      if (
        !(ringGroup instanceof SVGGraphicsElement) ||
        !(ring instanceof SVGCircleElement) ||
        !(hit instanceof SVGCircleElement) ||
        !(body instanceof SVGCircleElement)
      ) {
        throw new Error("Settled resurrection geometry is unavailable.");
      }
      const matrix = ringGroup.getScreenCTM();
      const bodyMatrix = body.getScreenCTM();
      if (matrix === null || bodyMatrix === null) {
        throw new Error("Settled resurrection transform is unavailable.");
      }
      const center = new DOMPoint(0, 0).matrixTransform(matrix);
      const bodyCenter = new DOMPoint(
        body.cx.baseVal.value,
        body.cy.baseVal.value,
      ).matrixTransform(bodyMatrix);
      const centerElements = document.elementsFromPoint(bodyCenter.x, bodyCenter.y);
      return {
        color: getComputedStyle(ringGroup).color,
        radius: ring.getAttribute("r"),
        strokeOpacity: getComputedStyle(ring).strokeOpacity,
        hitPointerEvents: getComputedStyle(hit).pointerEvents,
        hitFill: getComputedStyle(hit).fill,
        hitOwnsBodyCenter: centerElements.includes(hit),
        center: [center.x, center.y],
        bodyCenter: [bodyCenter.x, bodyCenter.y],
      };
    }, subjectSelector);
    expect(respawnGeometry.color).toBe("rgb(255, 255, 255)");
    expect(respawnGeometry.radius).toBe("32");
    expect(respawnGeometry.strokeOpacity).toBe("0.5");
    expect(respawnGeometry.hitPointerEvents).toBe("stroke");
    expect(respawnGeometry.hitFill).toBe("none");
    expect(respawnGeometry.hitOwnsBodyCenter).toBe(false);
    expect(respawnGeometry.center.every(Number.isFinite)).toBe(true);
    expect(
      Math.abs(respawnGeometry.center[0] - respawnGeometry.bodyCenter[0]),
    ).toBeLessThanOrEqual(0.001);
    expect(
      Math.abs(respawnGeometry.center[1] - respawnGeometry.bodyCenter[1]),
    ).toBeLessThanOrEqual(0.001);
    await expect(respawnEffect).toHaveAttribute("aria-label", "Agent respawned");
    await respawnEffect.focus();
    await expect(page.locator("#visual-tooltip-title")).toHaveText("Agent respawned");
    await expect(
      page.locator("#visual-tooltip .semantic-explanation__summary"),
    ).toHaveText("This agent respawned on the incoming transition.");
    expect(await respawnEffect.getAttribute("aria-label")).not.toMatch(
      /_|semantic pulse/iu,
    );
    const shield = page.locator(
      `.agent-spawn-shield[data-presentation-key="${subject.presentation_key}"]`,
    );
    const shieldStyles = await shield.evaluate((root) => {
      const shell = root.querySelector(".agent-spawn-shield__shell");
      const chip = root.querySelector(".agent-spawn-shield__chip");
      const ticks = root.querySelector(".agent-spawn-shield__ticks");
      if (
        !(shell instanceof SVGElement) ||
        !(chip instanceof SVGElement) ||
        !(ticks instanceof SVGElement)
      ) {
        throw new Error("Spawn Shield styling is unavailable.");
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

    await seekReplay(page, 4);
    const shieldFrame = await authenticatedGet(page, "/api/presentation/frame");
    const shieldedSubject = shieldFrame.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
    );
    expect(shieldedSubject.spawn_shield_remaining).toBe(2);
    await expect(page.locator(subjectSelector)).toHaveAttribute(
      "data-spawn-shield-remaining",
      "2",
    );
    await expect(page.locator(".combat-lifecycle-ring")).toHaveCount(0);
    await expect(page.locator(".combat-respawn-wave")).toHaveCount(0);

    await seekReplay(page, 5);
    const teamAFrame = await authenticatedGet(page, "/api/presentation/frame");
    const teamAEvents = /** @type {Record<string, any>[]} */ (
      teamAFrame.latest_events.events
    );
    expect(teamAFrame.latest_events.ordered_event_ids).toEqual(
      teamAEvents.map(({ event_id }) => event_id),
    );
    expect(teamAEvents.map(({ event_kind }) => event_kind)).toEqual([
      "action_rejected",
      "respawn_wave_occurred",
    ]);
    const teamAWaveEvent = teamAEvents[1];
    const teamAWaveEffect = page.locator(
      '.combat-effect[data-event-type="respawn_wave_occurred"]',
    );
    const teamAWave = teamAWaveEffect.locator(".combat-respawn-wave");
    await expect(teamAWaveEffect).toHaveCount(1);
    await expect(teamAWaveEffect).toHaveAttribute(
      "data-event-id",
      teamAWaveEvent.event_id,
    );
    await expect(teamAWaveEffect).toHaveAttribute("data-persistent", "true");
    await expect(teamAWaveEffect).toHaveAttribute("data-settled", "true");
    await expect(teamAWaveEffect).toHaveAttribute("data-team-index", "0");
    await expect(teamAWaveEffect).toHaveAttribute("data-team-id", "1");
    await expect(teamAWaveEffect).toHaveAttribute("data-team-side", "left");
    await expect(teamAWave).toHaveAttribute(
      "data-layout-key",
      JSON.stringify(["event", teamAWaveEvent.event_id, "cue"]),
    );
    await expect(teamAWave).toHaveAttribute(
      "data-layout-disposition",
      "perimeter_callout",
    );
    await expect(teamAWave).toHaveAttribute("data-layout-collision-free", "true");
    await expect(teamAWave.locator(".combat-respawn-wave__label")).toHaveText(
      "EVENT: Team A Respawn",
    );
    await expect(teamAWaveEffect).toHaveAttribute(
      "aria-label",
      "EVENT: Team A Respawn",
    );
    await teamAWaveEffect.focus();
    await expect(page.locator("#visual-tooltip-title")).toHaveText(
      "EVENT: Team A Respawn",
    );
    await expect(
      page.locator("#visual-tooltip .semantic-explanation__summary"),
    ).toHaveText("This team respawn occurred on the incoming transition.");
    const teamAGeometry = await teamAWave.evaluate((node) => {
      const map = document.querySelector(".map-boundary");
      const label = node.querySelector(".combat-respawn-wave__label");
      if (!(map instanceof SVGGraphicsElement) || !(label instanceof SVGTextElement)) {
        throw new Error("Settled Team A wave geometry is unavailable.");
      }
      const waveBounds = node.getBoundingClientRect();
      const mapBounds = map.getBoundingClientRect();
      return {
        color: getComputedStyle(node).color,
        label: label.textContent,
        withinMap:
          waveBounds.left >= mapBounds.left &&
          waveBounds.top >= mapBounds.top &&
          waveBounds.right <= mapBounds.right &&
          waveBounds.bottom <= mapBounds.bottom,
        leftSided:
          waveBounds.left + waveBounds.width / 2 < mapBounds.left + mapBounds.width / 2,
        childTooltipOwners: node.querySelectorAll("[data-tooltip-owner]").length,
      };
    });
    expect(teamAGeometry).toEqual({
      color: "rgb(59, 130, 246)",
      label: "EVENT: Team A Respawn",
      withinMap: true,
      leftSided: true,
      childTooltipOwners: 0,
    });
    const teamASubject = teamAFrame.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
    );
    expect(teamASubject.spawn_shield_remaining).toBe(1);
    await expect(page.locator(subjectSelector)).toHaveAttribute(
      "data-spawn-shield-remaining",
      "1",
    );
    await expect(page.locator(".combat-lifecycle-ring")).toHaveCount(0);

    await seekReplay(page, 6);
    const expiryFrame = await authenticatedGet(page, "/api/presentation/frame");
    const expiryEvents = /** @type {Record<string, any>[]} */ (
      expiryFrame.latest_events.events
    );
    expect(
      expiryEvents.some(({ event_kind }) => event_kind === "spawn_shield_expired"),
    ).toBe(true);
    const expiryEvent = expiryEvents.find(
      ({ event_kind }) => event_kind === "spawn_shield_expired",
    );
    if (!expiryEvent) {
      throw new Error("Authorized Spawn Shield expiry event is unavailable.");
    }
    const expiredSubject = expiryFrame.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.public_agent_id === "5",
    );
    expect(expiredSubject.spawn_shield_remaining).toBe(0);
    await expect(shield).toBeHidden();
    const expiryEffect = page.locator(
      '.combat-effect[data-event-type="spawn_shield_expired"]',
    );
    await expect(expiryEffect).toHaveCount(1);
    await expect(expiryEffect).toHaveAttribute("data-event-id", expiryEvent.event_id);
    expect(await expiryEffect.getAttribute("data-persistent")).toBeNull();
    await expect(expiryEffect).toHaveAttribute("data-settled", "true");
    const expiryPulse = expiryEffect.locator(
      ".combat-semantic-pulse--spawn-shield-expired",
    );
    await expect(expiryPulse).toHaveAttribute(
      "data-layout-key",
      JSON.stringify(["event", expiryEvent.event_id, "cue"]),
    );
    await expect(expiryPulse).toHaveAttribute(
      "data-layout-disposition",
      "perimeter_callout",
    );
    await expect(expiryPulse).toHaveAttribute("data-layout-collision-free", "true");
    await expect(page.locator(".combat-lifecycle-ring")).toHaveCount(0);
    await expect(page.locator(".combat-respawn-wave")).toHaveCount(0);

    await seekReplay(page, 7);
    const unshieldedFrame = await authenticatedGet(page, "/api/presentation/frame");
    const unshieldedEvents = /** @type {Record<string, any>[]} */ (
      unshieldedFrame.latest_events.events
    );
    const abilities = unshieldedEvents.filter(
      ({ event_kind }) => event_kind === "ability_activated",
    );
    expect(abilities).toHaveLength(1);
    expect(abilities[0].ability_component).toBe("basic");
    expect(abilities[0].source_anchor.public_agent_id).toBe("5");
    expect(abilities[0].recipient_anchor.public_agent_id).toBe("0");
    expect(
      unshieldedEvents.some(({ event_kind }) => event_kind === "action_rejected"),
    ).toBe(false);
    await expect(expiryEffect).toHaveCount(0);
    await expect(page.locator(".combat-lifecycle-ring")).toHaveCount(0);
    await expect(page.locator(".combat-respawn-wave")).toHaveCount(0);
    expect(browserErrors.get(page) ?? []).toEqual([]);
  }
});
