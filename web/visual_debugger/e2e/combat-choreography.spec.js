import { expect, test } from "@playwright/test";

import {
  assertBoundedChoreography,
  assertTransientSlotsAuthorized,
  CHOREOGRAPHY_ROOT,
  CHOREOGRAPHY_ROUTE_ROOT,
  choreographySnapshot,
  finishControllerClock,
  installWaapiAutopause,
  pauseAtLogicalTime,
} from "./support/choreography.js";
import { startDebugger, stopDebugger } from "./support/live-debugger.js";
import {
  loadRendererFixture,
  syntheticDebuggerFrame,
} from "./support/renderer-fixture.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";
/** @type {Record<string, any>} */
let routeFrame = {};
/** @type {Record<string, any>} */
let mixedFrame = {};
/** @type {Record<string, any>} */
let crowdedFrame = {};
/** @type {Record<string, any>} */
let povFixture = {};

test.beforeAll(async () => {
  const [started, routeFixture, mixedFixture, crowdedFixture, loadedPovFixture] =
    await Promise.all([
      startDebugger(),
      loadRendererFixture("route_collision"),
      loadRendererFixture("mixed_net_zero"),
      loadRendererFixture("crowded_teamfight"),
      loadRendererFixture("pov_redaction"),
    ]);
  serverProcess = started.process;
  debuggerUrl = started.url;
  routeFrame = syntheticDebuggerFrame(routeFixture);
  mixedFrame = syntheticDebuggerFrame(mixedFixture);
  crowdedFrame = syntheticDebuggerFrame(crowdedFixture);
  povFixture = loadedPovFixture;
});

test.afterAll(async () => {
  const child = serverProcess;
  serverProcess = null;
  await stopDebugger(child);
});

/**
 * @param {Record<string, any>} frame
 */
function withoutTransition(frame) {
  const initial = structuredClone(frame);
  initial.revision = 0;
  initial.frame_index = 0;
  initial.frame_id = `${initial.episode_id}:frame:0`;
  initial.simulator_step_count = 0;
  initial.hud.latest_transition = null;
  if (initial.frame_kind === "researcher_live_debugger") {
    initial.incoming_transition_index = null;
    initial.incoming_transition_id = null;
    initial.projection.scene.frame_index = 0;
    initial.projection.scene.frame_id = initial.frame_id;
    initial.projection.scene.simulator_step_count = 0;
    initial.projection.scene.incoming_transition_id = null;
    initial.projection.scene.incoming_event_ids = [];
    initial.projection.incoming_events = null;
  } else {
    const publicAgentId = initial.projection.scene.self_actor.public_agent_id;
    initial.incoming_pov_transition_id = null;
    initial.projection.scene.frame_index = 0;
    initial.projection.scene.source_frame_id = initial.frame_id;
    initial.projection.scene.pov_frame_id = `${initial.episode_id}:actor-pov:${publicAgentId}:frame:0`;
    initial.projection.scene.simulator_step_count = 0;
    initial.projection.incoming_transition_id = null;
    initial.projection.incoming_cues = [];
  }
  return initial;
}

/**
 * Replace one researcher batch with exact canonical V2 event rows while
 * retaining its validated transition envelope and phase trajectories.
 *
 * @param {Record<string, any>} frame
 * @param {Array<Record<string, any>>} payloads
 */
function withResearcherEvents(frame, payloads) {
  const next = structuredClone(frame);
  const batch = next.projection?.incoming_events;
  if (!batch || typeof batch.transition_id !== "string") {
    throw new Error("Synthetic researcher frame is missing its V2 event batch.");
  }
  const events = payloads.map((payload, ordinal) => ({
    ...payload,
    event_id: `${batch.transition_id}:event:${String(ordinal).padStart(4, "0")}`,
    transition_id: batch.transition_id,
    ordinal,
  }));
  next.projection.incoming_events.events = events;
  next.projection.scene.incoming_event_ids = events.map((event) => event.event_id);
  return next;
}

/**
 * @param {Record<string, any>} frame
 * @param {number} globalSlot
 * @param {"transition_start" | "post_charge" | "successor"} phase
 */
function phaseAnchor(frame, globalSlot, phase) {
  const trajectory = frame.projection?.incoming_events?.agent_phase_trajectories.find(
    /** @param {Record<string, any>} row */ (row) => row.global_slot === globalSlot,
  );
  if (!trajectory?.[phase]) {
    throw new Error(`Missing ${phase} anchor for synthetic slot ${globalSlot}.`);
  }
  return structuredClone(trajectory[phase]);
}

/**
 * Intercept one synthetic frame flow without mutating the live Python session.
 *
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} initial
 * @param {(command: Record<string, any>, index: number) => Record<string, any>} next
 */
async function installSyntheticFlow(page, initial, next) {
  const flow = {
    current: initial,
    /** @type {Record<string, any>[]} */
    commands: [],
  };
  await page.route("**/api/frame", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: flow.current,
      status: 200,
    });
  });
  await page.route("**/api/command", async (route) => {
    const request = /** @type {Record<string, any>} */ (route.request().postDataJSON());
    const command = request.command ?? {};
    flow.commands.push(command);
    flow.current = next(command, flow.commands.length);
    await route.fulfill({
      contentType: "application/json",
      json: {
        schema_version: 2,
        result: "applied",
        frame: flow.current,
        notice: null,
      },
      status: 200,
    });
  });
  return flow;
}

test("directional routes share one clock, gate submissions, reproject, and do not replay", async ({
  page,
}) => {
  await installWaapiAutopause(page);
  /** @type {Record<string, any>} */
  const fullFrame = { ...routeFrame, revision: 1 };
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(routeFrame),
    (command) => ({
      ...fullFrame,
      preset: command.command_type === "set_preset" ? command.preset : fullFrame.preset,
      revision: flow.commands.length,
    }),
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "0.5×", exact: true }).click();
  await page.getByRole("button", { name: "Reset" }).click();

  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
  await expect(page.locator("html")).toHaveAttribute("data-submission-blocked", "true");
  expect(flow.commands).toHaveLength(1);
  expect(
    await page
      .locator("#battlefield > [data-layer]")
      .evaluateAll((layers) => layers.map((layer) => layer.getAttribute("data-layer"))),
  ).toEqual([
    "map",
    "aura",
    "debug-range",
    "pending-route",
    "transient-route",
    "obstacle",
    "body",
    "selection-legality",
    "transient-events",
    "durable-status-modifier",
    "accessible-labels",
  ]);

  await page.locator("#battlefield").focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#notice")).toContainText(
    "Scripted playback is inspection-only",
  );
  expect(flow.commands).toHaveLength(1);

  // Sample inside the authored V2 ability phase (60–180 ms). Route and impact
  // geometry must not depend on later health or successor phases.
  await pauseAtLogicalTime(page, 165);
  await expect(page.locator("html")).toHaveAttribute("data-motion-paused", "true");
  await expect(page.locator(".combat-effect--activation")).toHaveCount(9);
  await expect(page.locator(".combat-route__path")).toHaveCount(9);
  await expect(page.locator(".combat-route__arrow")).toHaveCount(9);
  await expect(page.locator(".combat-impact")).toHaveCount(9);
  await expect(page.locator(".combat-effect--activation[data-net-delta]")).toHaveCount(
    0,
  );
  const routePaths = await page
    .locator(".combat-route__path")
    .evaluateAll((paths) => paths.map((path) => path.getAttribute("d")));
  expect(new Set(routePaths).size).toBe(routePaths.length);
  const routeAssociations = await page
    .locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation`)
    .evaluateAll((routes) =>
      routes.map((route) => {
        const eventId = route.getAttribute("data-event-id");
        const arrow = route.querySelector(".combat-route__arrow");
        const impact = [...document.querySelectorAll(".combat-effect--activation")]
          .find((effect) => effect.getAttribute("data-event-id") === eventId)
          ?.querySelector(".combat-impact");
        const arrowMatrix =
          arrow instanceof SVGGraphicsElement
            ? arrow.transform.baseVal.consolidate()?.matrix
            : null;
        const impactMatrix =
          impact instanceof SVGGraphicsElement
            ? impact.transform.baseVal.consolidate()?.matrix
            : null;
        return {
          sourceClass: route.getAttribute("data-source-class"),
          arrowTransform: arrow?.getAttribute("transform"),
          markerImpactDistance:
            arrowMatrix && impactMatrix
              ? Math.hypot(
                  arrowMatrix.e - impactMatrix.e,
                  arrowMatrix.f - impactMatrix.f,
                )
              : null,
        };
      }),
    );
  expect(
    routeAssociations.every(
      ({ sourceClass }) => sourceClass !== null && sourceClass !== "unknown",
    ),
  ).toBe(true);
  expect(
    routeAssociations.every(({ arrowTransform }) =>
      arrowTransform?.includes("rotate("),
    ),
  ).toBe(true);
  const markerImpactDistances = routeAssociations.map(
    ({ markerImpactDistance }) => markerImpactDistance,
  );
  expect(
    Math.min(
      ...markerImpactDistances.map((distance) =>
        distance === null ? Number.NEGATIVE_INFINITY : distance,
      ),
    ),
    JSON.stringify(routeAssociations),
  ).toBeGreaterThan(4);
  expect(
    await page
      .locator(".combat-impact")
      .evaluateAll((impacts) =>
        impacts.every(
          (impact) => Number.parseFloat(getComputedStyle(impact).opacity) > 0,
        ),
      ),
  ).toBe(true);
  await assertBoundedChoreography(page);

  const root = page.locator(CHOREOGRAPHY_ROOT);
  await root.evaluate((element) => {
    element.setAttribute("data-retained-probe", "route-collision");
  });
  const beforeResize = await choreographySnapshot(page);
  await page.setViewportSize({ width: 960, height: 600 });
  await expect
    .poll(async () => root.getAttribute("data-viewport-key"))
    .not.toBe(beforeResize.viewportKey);
  const afterResize = await choreographySnapshot(page);
  await expect(root).toHaveAttribute("data-retained-probe", "route-collision");
  expect(afterResize.epochKey).toBe(beforeResize.epochKey);
  expect(afterResize.animationIds).toEqual(beforeResize.animationIds);
  expect(afterResize.effectIds).toEqual(beforeResize.effectIds);
  expect(
    await page
      .locator(".combat-route__path")
      .evaluateAll((paths) => paths.map((path) => path.getAttribute("d"))),
  ).not.toEqual(routePaths);
  expect(flow.commands).toHaveLength(1);

  await page.locator("#motion-skip-button").click();
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );
  await expect(page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect`)).toHaveCount(0);
  expect((await choreographySnapshot(page)).animationIds).toEqual([]);

  await page.reload();
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
  await expect(page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect`)).toHaveCount(0);
  expect((await choreographySnapshot(page)).animationIds).toEqual([]);
  expect(flow.commands).toHaveLength(1);
});

test("a collision-suppressed NET cue reappears after resize without replay", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await installWaapiAutopause(page);
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(mixedFrame),
    (_command, index) => ({ ...mixedFrame, revision: index }),
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.locator(".combat-effect--net-health")).toHaveCount(1);
  await pauseAtLogicalTime(page, 210);

  const root = page.locator(CHOREOGRAPHY_ROOT);
  const net = page.locator(".combat-effect--net-health");
  await root.evaluate((element) => {
    element.setAttribute("data-retained-probe", "resize-recovery");
  });
  await net.evaluate((element) => {
    element.setAttribute("data-retained-probe", "net-resize-recovery");
  });
  await expect(net).toHaveAttribute("data-spatial-disposition", "rendered");
  const before = await choreographySnapshot(page);

  await page.locator("#battlefield").evaluate((element) => {
    const battlefield = /** @type {SVGElement} */ (element);
    battlefield.style.width = "100px";
    battlefield.style.height = "100px";
  });
  await expect
    .poll(async () => root.getAttribute("data-viewport-key"))
    .not.toBe(before.viewportKey);
  await expect(net).toHaveAttribute("data-spatial-disposition", "suppressed-collision");
  await expect(net).toHaveAttribute("visibility", "hidden");
  await expect(net).toHaveAttribute("aria-hidden", "true");
  const suppressed = await choreographySnapshot(page);
  expect(suppressed.animationIds).toEqual(before.animationIds);
  expect(suppressed.effectIds).toEqual(before.effectIds);

  await page.locator("#battlefield").evaluate((element) => {
    const battlefield = /** @type {SVGElement} */ (element);
    battlefield.style.removeProperty("width");
    battlefield.style.removeProperty("height");
  });
  await expect
    .poll(async () => root.getAttribute("data-viewport-key"))
    .toBe(before.viewportKey);
  await expect(net).toHaveAttribute("data-spatial-disposition", "rendered");
  await expect(net).not.toHaveAttribute("visibility");
  await expect(net).not.toHaveAttribute("aria-hidden");
  await expect(root).toHaveAttribute("data-retained-probe", "resize-recovery");
  await expect(net).toHaveAttribute("data-retained-probe", "net-resize-recovery");
  const restored = await choreographySnapshot(page);
  expect(restored.animationIds).toEqual(before.animationIds);
  expect(restored.effectIds).toEqual(before.effectIds);
  expect(flow.commands).toHaveLength(1);
});

test("same-size protected-layout changes reproject without replay", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await installWaapiAutopause(page);
  /** @type {Record<string, any>} */
  const fullFrame = { ...crowdedFrame, revision: 1 };
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(crowdedFrame),
    (command) => ({
      ...fullFrame,
      preset: command.command_type === "set_preset" ? command.preset : fullFrame.preset,
      revision: flow.commands.length,
    }),
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await pauseAtLogicalTime(page, 210);
  const root = page.locator(CHOREOGRAPHY_ROOT);
  await root.evaluate((element) => {
    element.setAttribute("data-retained-probe", "protected-layout");
  });
  const before = await choreographySnapshot(page);
  const beforeBounds = await page.locator("#battlefield").boundingBox();
  expect(beforeBounds).not.toBeNull();

  await page.locator("#preset-select").selectOption("presentation");
  await expect(page.locator("#preset-select")).toHaveValue("presentation");
  await expect
    .poll(async () => root.getAttribute("data-viewport-key"))
    .not.toBe(before.viewportKey);
  const after = await choreographySnapshot(page);
  const afterBounds = await page.locator("#battlefield").boundingBox();
  expect(afterBounds).not.toBeNull();
  expect(afterBounds?.width).toBeCloseTo(beforeBounds?.width ?? 0, 3);
  expect(afterBounds?.height).toBeCloseTo(beforeBounds?.height ?? 0, 3);
  expect(after.viewportKey?.split(":").slice(0, 8)).toEqual(
    before.viewportKey?.split(":").slice(0, 8),
  );
  expect(after.viewportKey?.split(":").slice(8)).not.toEqual(
    before.viewportKey?.split(":").slice(8),
  );
  expect(after.animationIds).toEqual(before.animationIds);
  expect(after.effectIds).toEqual(before.effectIds);
  await expect(root).toHaveAttribute("data-retained-probe", "protected-layout");
  expect(flow.commands).toHaveLength(2);
});

test("reduced motion preserves damage/heal intent and one recipient NET outcome", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await installWaapiAutopause(page);
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(mixedFrame),
    (_command, index) => ({ ...mixedFrame, revision: index }),
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-motion-mode", "reduced");
  await page.getByRole("button", { name: "2×", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-motion-mode", "reduced");
  await expect(page.locator("html")).toHaveAttribute("data-motion-rate", "2");
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );
  expect(
    (await choreographySnapshot(page)).animationIds.every(
      ({ playbackRate }) => playbackRate === 2,
    ),
  ).toBe(true);
  expect(flow.commands).toHaveLength(1);
  // Reduced motion preserves V2 causal order on a 220 ms clock: ability cues
  // paint first, followed by the independently authoritative health result.
  await pauseAtLogicalTime(page, 30);

  const activations = page.locator(".combat-effect--activation");
  const net = page.locator(".combat-effect--net-health");
  await expect(activations).toHaveCount(2);
  await expect(net).toHaveCount(1);
  await expect(page.locator(".combat-route__path")).toHaveCount(2);
  await expect(page.locator(".combat-impact")).toHaveCount(2);
  await expect(page.locator(".combat-route__particle")).toHaveCount(0);
  expect(
    await page
      .locator(".combat-impact")
      .evaluateAll((nodes) =>
        nodes.every(
          (node) => Number.parseFloat(getComputedStyle(node).opacity) <= 0.001,
        ),
      ),
  ).toBe(true);
  expect(
    await net.evaluate((node) => Number.parseFloat(getComputedStyle(node).opacity)),
  ).toBeLessThanOrEqual(0.001);
  expect(
    await page
      .locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation`)
      .evaluateAll((nodes) =>
        nodes.every((node) => Number.parseFloat(getComputedStyle(node).opacity) > 0),
      ),
  ).toBe(true);
  const routeStyles = await page
    .locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation`)
    .evaluateAll((routes) =>
      Object.fromEntries(
        routes.map((route) => {
          const path = route.querySelector(".combat-route__path");
          const style = path ? getComputedStyle(path) : null;
          return [
            route.getAttribute("data-token-id"),
            {
              strokeWidth: Number.parseFloat(style?.strokeWidth ?? ""),
              dashed:
                style !== null &&
                style.strokeDasharray !== "" &&
                style.strokeDasharray !== "none",
            },
          ];
        }),
      ),
    );
  expect(routeStyles).toEqual({
    basic_damage: { strokeWidth: 2.5, dashed: false },
    basic_heal: { strokeWidth: 3, dashed: true },
  });

  // The compact impact subphase remains inside the ability interval and ends
  // before health resolution begins.
  await pauseAtLogicalTime(page, 42);
  expect(
    await page
      .locator(".combat-impact")
      .evaluateAll((nodes) =>
        nodes.every((node) => Number.parseFloat(getComputedStyle(node).opacity) > 0),
      ),
  ).toBe(true);

  await pauseAtLogicalTime(page, 52);
  await expect(net).toHaveCount(1);
  await expect(net).toHaveAttribute("data-recipient-slot", "5");
  await expect(net).toHaveAttribute("data-net-delta", "0");
  await expect(net.locator(".combat-net__recipient")).toHaveText("id_5");
  await expect(net.locator(".combat-net__label")).toHaveText("HP unchanged");
  expect(
    await net.evaluate((node) => Number.parseFloat(getComputedStyle(node).opacity)),
  ).toBeGreaterThan(0);
  expect(
    await page
      .locator(".combat-impact")
      .evaluateAll((nodes) =>
        nodes.every(
          (node) => Number.parseFloat(getComputedStyle(node).opacity) <= 0.001,
        ),
      ),
  ).toBe(true);
  expect(
    await page
      .locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation`)
      .evaluateAll((nodes) =>
        nodes.every(
          (node) => Number.parseFloat(getComputedStyle(node).opacity) <= 0.001,
        ),
      ),
  ).toBe(true);
  await expect(page.locator(".combat-effect--activation[data-net-delta]")).toHaveCount(
    0,
  );
  expect(
    (await choreographySnapshot(page)).animationIds.some(({ id }) =>
      id.endsWith(":gate"),
    ),
  ).toBe(false);
  await assertBoundedChoreography(page);
  await expect(page).toHaveScreenshot(
    "reduced-motion-mixed-net-health-phase-1440x900.png",
    { animations: "allow" },
  );

  await finishControllerClock(page, "cleanup");
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveAttribute(
    "data-state",
    "settled",
  );
  await expect(page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect`)).toHaveCount(0);
  expect((await choreographySnapshot(page)).animationIds).toEqual([]);
  expect(flow.commands).toHaveLength(1);
});

test("reduced motion keeps Mage Burst static while preserving its identity", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await installWaapiAutopause(page);
  const burstFrame = withResearcherEvents(crowdedFrame, [
    {
      event_type: "ability_activated",
      phase_rank: 20,
      source_global_slot: 0,
      ability_component: "ultimate",
      recipient_global_slot: null,
      source_anchor: phaseAnchor(crowdedFrame, 0, "transition_start"),
      recipient_anchor: null,
    },
  ]);
  burstFrame.revision = 1;
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(burstFrame),
    () => burstFrame,
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-motion-mode", "reduced");
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );

  const wave = page.locator(".combat-burst__wave--outer");
  await expect(wave).toHaveCount(1);
  const keyframes = await wave.evaluate((element) =>
    element
      .getAnimations()
      .flatMap((animation) =>
        animation.effect instanceof KeyframeEffect
          ? animation.effect.getKeyframes()
          : [],
      )
      .map((frame) => ({
        opacity: frame.opacity,
        transform: frame.transform,
      })),
  );
  expect(keyframes.length).toBeGreaterThan(0);
  expect(keyframes.some((frame) => frame.opacity === "1")).toBe(true);
  expect(
    keyframes.every(
      (frame) => frame.transform === undefined || frame.transform === "none",
    ),
  ).toBe(true);
  expect(flow.commands).toHaveLength(1);
});

test("mid-transition Motion Off retains one static batch through bounded cleanup", async ({
  page,
}) => {
  await installWaapiAutopause(page);
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(routeFrame),
    (_command, index) => ({ ...routeFrame, revision: index }),
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await pauseAtLogicalTime(page, 300);
  const animatedSnapshot = await choreographySnapshot(page);
  await page.getByRole("button", { name: "Off", exact: true }).click();

  await expect(page.locator("html")).toHaveAttribute("data-motion-mode", "off");
  await expect(page.locator("html")).toHaveAttribute("data-motion-paused", "false");
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );
  await expect(page.locator("#motion-pause-button")).toBeDisabled();
  await expect(page.locator(".combat-effect--activation")).toHaveCount(9);
  await expect(page.locator(".combat-route__particle")).toHaveCount(0);
  const snapshot = await choreographySnapshot(page);
  expect(snapshot.effectIds).toEqual(animatedSnapshot.effectIds);
  expect(new Set(snapshot.effectIds).size).toBe(snapshot.effectIds.length);
  expect(snapshot.animationIds.map(({ id }) => id)).toEqual([
    expect.stringMatching(/:cleanup$/),
  ]);
  await assertBoundedChoreography(page);
  await expect(page).toHaveScreenshot("motion-off-static-route-batch-1440x900.png", {
    animations: "allow",
  });
  expect(flow.commands).toHaveLength(1);

  await page.getByRole("button", { name: "0.5×", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-motion-mode", "normal");
  await expect(page.locator("html")).toHaveAttribute("data-motion-rate", "0.5");
  const restoredPreference = await choreographySnapshot(page);
  expect(restoredPreference.effectIds).toEqual(snapshot.effectIds);
  expect(restoredPreference.animationIds).toHaveLength(1);
  expect(restoredPreference.animationIds[0].id).toMatch(/:cleanup$/);
  expect(restoredPreference.animationIds[0].playbackRate).toBe(0.5);
  expect(flow.commands).toHaveLength(1);

  await finishControllerClock(page, "cleanup");
  await expect(page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect`)).toHaveCount(0);
  expect((await choreographySnapshot(page)).animationIds).toEqual([]);
  expect(flow.commands).toHaveLength(1);
});

test("rejection and composite Trap lifecycle retain only exact supplied facts", async ({
  page,
}) => {
  await installWaapiAutopause(page);
  const compositeFrame = withResearcherEvents(mixedFrame, [
    {
      event_type: "action_rejected",
      phase_rank: 10,
      actor_global_slot: 0,
      actor_public_agent_id: "0",
      actor_configured_active: true,
      rejection_component: "combat_pair",
      submitted_move_action: 0,
      submitted_select_target_action: 6,
      submitted_use_ultimate_action: 1,
      actor_anchor: phaseAnchor(mixedFrame, 0, "transition_start"),
    },
    {
      event_type: "ability_activated",
      phase_rank: 20,
      source_global_slot: 0,
      ability_component: "ultimate",
      recipient_global_slot: 5,
      source_anchor: phaseAnchor(mixedFrame, 0, "transition_start"),
      recipient_anchor: phaseAnchor(mixedFrame, 5, "transition_start"),
    },
    {
      event_type: "status_broken_by_damage",
      phase_rank: 100,
      recipient_global_slot: 5,
      status_channel: 4,
      status_id: "hunter_trap_stun",
      recipient_anchor: phaseAnchor(mixedFrame, 5, "successor"),
    },
    {
      event_type: "status_applied",
      phase_rank: 100,
      source_global_slot: 0,
      recipient_global_slot: 5,
      status_channel: 4,
      status_id: "hunter_trap_stun",
      source_anchor: phaseAnchor(mixedFrame, 0, "successor"),
      recipient_anchor: phaseAnchor(mixedFrame, 5, "successor"),
    },
  ]);
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(compositeFrame),
    (_command, index) => ({ ...compositeFrame, revision: index }),
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await pauseAtLogicalTime(page, 680);

  const rejection = page.locator(".combat-effect--rejected-action");
  await expect(rejection).toHaveCount(1);
  await expect(rejection).toHaveAttribute("data-actor-slot", "0");
  await expect(rejection).not.toHaveAttribute("data-target-slot", /.+/);
  await expect(rejection).toHaveAttribute("data-component", "combat_pair");
  await expect(rejection.locator(".combat-rejection__ring")).toHaveCount(1);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--rejected-action`),
  ).toHaveCount(0);

  const broken = page.locator(
    '.combat-effect--status-lifecycle[data-event-type="status_broken_by_damage"]',
  );
  const reapplied = page.locator(
    '.combat-effect--status-lifecycle[data-event-type="status_applied"]',
  );
  await expect(broken).toHaveCount(1);
  await expect(reapplied).toHaveCount(1);
  await expect(broken.locator(".combat-lifecycle__shard")).toHaveCount(6);
  await expect(
    broken.locator('.combat-lifecycle__status-icon[data-icon="status-stun"]'),
  ).toHaveCount(1);
  await expect(
    broken.locator('.combat-lifecycle__change-icon[data-icon="lifecycle-trap-broken"]'),
  ).toHaveCount(1);
  await expect(reapplied.locator(".combat-lifecycle__reapply")).toHaveCount(0);
  await expect(broken).toHaveAttribute("data-application-event-ids", "[]");
  await expect(reapplied).toHaveAttribute("data-application-event-ids", "[]");
  await expect(broken).not.toHaveAttribute("data-duration-before", /.+/);
  await expect(reapplied).not.toHaveAttribute("data-duration-after", /.+/);
  await assertBoundedChoreography(page);
  expect(flow.commands).toHaveLength(1);
});

test("same-epoch POV switch clears privileged effects before safe redacted rebuild", async ({
  page,
}) => {
  await installWaapiAutopause(page);
  const researcherFrame = structuredClone(crowdedFrame);
  const safeFrame = syntheticDebuggerFrame(povFixture);
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(researcherFrame),
    (command, index) => {
      if (index === 1) {
        return { ...researcherFrame, revision: 1 };
      }
      if (command.command_type === "set_view") {
        return { ...safeFrame, revision: 2 };
      }
      return { ...safeFrame, revision: 3 };
    },
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await pauseAtLogicalTime(page, 600);
  const privilegedRoot = page.locator(CHOREOGRAPHY_ROOT);
  await expect(privilegedRoot.locator('[data-target-slot="5"]')).not.toHaveCount(0);
  await privilegedRoot.evaluate((element) => {
    element.setAttribute("data-privileged-probe", "present");
  });

  await page.locator("#view-select").selectOption("pov");
  await expect(page.locator("html")).toHaveAttribute("data-audience", "agent_pov");
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT}[data-privileged-probe="present"]`),
  ).toHaveCount(0);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`),
  ).toHaveCount(0);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} [data-event-type="own_health_changed"]`),
  ).toHaveCount(1);
  expect(
    await page
      .locator(`${CHOREOGRAPHY_ROOT} .combat-effect`)
      .evaluateAll((effects) =>
        effects.every((effect) =>
          String(effect.getAttribute("data-event-id")).includes(":actor-pov:"),
        ),
      ),
  ).toBe(true);
  await assertTransientSlotsAuthorized(page);

  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`),
  ).toHaveCount(0);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation`),
  ).toHaveCount(0);
  // POV retains the self actor plus one authorized relative visible-body row;
  // only the self actor is allowed into the identity-bearing roster.
  await expect(page.locator("#battlefield .agent")).toHaveCount(2);
  await expect(page.locator("#battlefield .pov-observed-body")).toHaveCount(1);
  await expect(page.locator("#roster .roster-row")).toHaveCount(1);
  await expect(page.locator('#battlefield .agent[data-slot="5"]')).toHaveCount(0);
  await expect(page.locator(`${CHOREOGRAPHY_ROOT} [data-target-slot="5"]`)).toHaveCount(
    0,
  );
  await assertTransientSlotsAuthorized(page);
  await assertBoundedChoreography(page);
  expect(flow.commands).toHaveLength(2);
});

test("POV bootstrap rejects an impossible split simulator epoch", async ({ page }) => {
  const splitEpoch = withoutTransition(syntheticDebuggerFrame(povFixture));
  splitEpoch.projection.scene.simulator_step_count = 1;
  await installSyntheticFlow(page, splitEpoch, () => splitEpoch);

  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Resync required");
  await expect(page.locator("#notice")).toContainText(
    "POV projection does not join its live frame",
  );
  await expect(page.locator("#battlefield .agent")).toHaveCount(0);
});

test("recipient POV cues never reconstruct a researcher target-only activation", async ({
  page,
}) => {
  await installWaapiAutopause(page);
  const safeFrame = syntheticDebuggerFrame(povFixture);
  const targetOnlyFrame = { ...safeFrame, revision: 1 };
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(targetOnlyFrame),
    () => targetOnlyFrame,
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await pauseAtLogicalTime(page, 600);

  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`),
  ).toHaveCount(0);
  await expect(page.locator(`${CHOREOGRAPHY_ROOT} [data-source-slot="5"]`)).toHaveCount(
    0,
  );
  const cueTypes = await page
    .locator("#event-feed .event-item")
    .evaluateAll((items) => items.map((item) => item.getAttribute("data-event-type")));
  expect(
    cueTypes.every(
      (type) =>
        type?.startsWith("own_") ||
        type === "episode_ended" ||
        type === "visible_body_observation_changed",
    ),
  ).toBe(true);
  await assertBoundedChoreography(page);
  expect(flow.commands).toHaveLength(1);
});
