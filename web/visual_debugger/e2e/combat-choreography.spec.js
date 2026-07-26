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
  return {
    ...frame,
    revision: 0,
    transition_id: null,
    event_batch: null,
    hud: {
      ...frame.hud,
      latest_transition: null,
    },
  };
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
        schema_version: 1,
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
  await expect(page.locator("#notice")).toContainText("still being explained");
  expect(flow.commands).toHaveLength(1);

  await pauseAtLogicalTime(page, 500);
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
  await pauseAtLogicalTime(page, 500);

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
  await pauseAtLogicalTime(page, 500);
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
  await pauseAtLogicalTime(page, 140);

  await expect(page.locator(".combat-effect--activation")).toHaveCount(2);
  await expect(page.locator(".combat-route__path")).toHaveCount(2);
  await expect(page.locator(".combat-impact")).toHaveCount(2);
  await expect(page.locator(".combat-route__particle")).toHaveCount(0);
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
  const net = page.locator(".combat-effect--net-health");
  await expect(net).toHaveCount(1);
  await expect(net).toHaveAttribute("data-recipient-slot", "5");
  await expect(net).toHaveAttribute("data-net-delta", "0");
  await expect(net.locator(".combat-net__label")).toHaveText("HP unchanged");
  await expect(page.locator(".combat-effect--activation[data-net-delta]")).toHaveCount(
    0,
  );
  expect(
    (await choreographySnapshot(page)).animationIds.some(({ id }) =>
      id.endsWith(":gate"),
    ),
  ).toBe(false);
  await assertBoundedChoreography(page);

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
  const source = /** @type {Record<string, any>[]} */ (routeFrame.scene.agents).find(
    (agent) => agent.global_slot === 0,
  );
  expect(source).toBeTruthy();
  if (!source) {
    throw new Error("Synthetic route fixture is missing source id_0.");
  }
  const burstFrame = {
    ...routeFrame,
    revision: 1,
    event_batch: {
      schema_version: 1,
      transition_id: 1,
      simulator_step: 1,
      events: [
        {
          event_type: "accepted_activation",
          event_id: "synthetic:reduced:mage-burst",
          transition_id: 1,
          token_id: "mage_burst",
          source_global_slot: 0,
          target_global_slot: null,
          source_anchor: source.position,
          target_anchor: null,
          target_disclosure: "target_none",
          lane: 1,
          source_class_id: 0,
        },
      ],
    },
  };
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
  const source = mixedFrame.scene.agents.find(
    /** @param {{global_slot: number}} agent */ (agent) => agent.global_slot === 0,
  );
  const target = mixedFrame.scene.agents.find(
    /** @param {{global_slot: number}} agent */ (agent) => agent.global_slot === 5,
  );
  if (!source || !target) {
    throw new Error("Mixed fixture is missing the synthetic source or target.");
  }
  const activationId = "synthetic:composite:activation";
  const compositeFrame = {
    ...mixedFrame,
    event_batch: {
      ...mixedFrame.event_batch,
      events: [
        {
          event_type: "accepted_activation",
          event_id: activationId,
          transition_id: 1,
          token_id: "hunter_trap",
          source_global_slot: 0,
          target_global_slot: 5,
          source_anchor: source.position,
          target_anchor: target.position,
          target_disclosure: "public",
          lane: 1,
          source_class_id: 3,
        },
        {
          event_type: "rejected_action",
          event_id: "synthetic:composite:rejection",
          transition_id: 1,
          actor_global_slot: 0,
          component: "combat",
          actor_anchor: source.position,
          target_global_slot: 5,
          target_anchor: target.position,
          target_disclosure: "public",
          lane: 1,
          movement_mask_value: true,
          pair_mask_value: false,
        },
        {
          event_type: "status_lifecycle",
          event_id: "synthetic:composite:lifecycle",
          transition_id: 1,
          recipient_global_slot: 5,
          recipient_anchor: target.position,
          token_id: "stun_hunter_trap",
          change: "trap_broken_and_reapplied",
          duration_before: 2,
          duration_after: 3,
          source_class_id: 3,
          application_event_ids: [activationId],
        },
      ],
    },
  };
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(compositeFrame),
    (_command, index) => ({ ...compositeFrame, revision: index }),
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await pauseAtLogicalTime(page, 500);

  const rejection = page.locator(".combat-effect--rejected-action");
  await expect(rejection).toHaveCount(1);
  await expect(rejection).toHaveAttribute("data-actor-slot", "0");
  await expect(rejection).toHaveAttribute("data-target-slot", "5");
  await expect(rejection).toHaveAttribute("data-component", "combat");
  await expect(rejection).toHaveAttribute("data-movement-mask-value", "true");
  await expect(rejection).toHaveAttribute("data-pair-mask-value", "false");
  await expect(rejection.locator(".combat-rejection__ring")).toHaveCount(1);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--rejected-action .combat-rejection__route`,
    ),
  ).toHaveCount(1);

  const lifecycle = page.locator(
    '.combat-effect--status-lifecycle[data-lifecycle="trap_broken_and_reapplied"]',
  );
  await expect(lifecycle).toHaveCount(1);
  await expect(lifecycle.locator(".combat-lifecycle__shard")).toHaveCount(6);
  await expect(
    lifecycle.locator('.combat-lifecycle__status-icon[data-icon="status-trap"]'),
  ).toHaveCount(1);
  await expect(
    lifecycle.locator(
      '.combat-lifecycle__change-icon[data-icon="lifecycle-trap-broken-reapplied"]',
    ),
  ).toHaveCount(1);
  await expect(lifecycle.locator(".combat-lifecycle__reapply")).toHaveCount(1);
  await expect(lifecycle).toHaveAttribute(
    "data-application-event-ids",
    JSON.stringify([activationId]),
  );
  await expect(lifecycle).toHaveAttribute("data-duration-before", "2");
  await expect(lifecycle).toHaveAttribute("data-duration-after", "3");
  await assertBoundedChoreography(page);
  expect(flow.commands).toHaveLength(1);
});

test("same-epoch POV switch clears privileged effects before safe redacted rebuild", async ({
  page,
}) => {
  await installWaapiAutopause(page);
  const researcherFrame = syntheticDebuggerFrame({
    ...povFixture,
    scene: povFixture.privileged_source_scene,
    event_batch: povFixture.privileged_source_event_batch,
  });
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
      return {
        ...safeFrame,
        run_generation: 1,
        revision: 3,
      };
    },
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await pauseAtLogicalTime(page, 500);
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
  await expect(page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect`)).toHaveCount(0);
  expect((await choreographySnapshot(page)).animationIds).toEqual([]);
  await assertTransientSlotsAuthorized(page);

  await page.getByRole("button", { name: "Reset" }).click();
  await pauseAtLogicalTime(page, 500);
  const redactedActivation = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--activation`,
  );
  await expect(redactedActivation).toHaveCount(1);
  await expect(redactedActivation).toHaveAttribute("data-source-slot", "0");
  await expect(redactedActivation).not.toHaveAttribute("data-target-slot", /.+/);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation .combat-route__path`,
    ),
  ).toHaveCount(0);
  await expect(redactedActivation.locator(".combat-local")).toHaveCount(1);
  await expect(page.locator("#battlefield .agent")).toHaveCount(2);
  await expect(page.locator("#roster .roster-row")).toHaveCount(2);
  await expect(page.locator(`${CHOREOGRAPHY_ROOT} [data-target-slot="5"]`)).toHaveCount(
    0,
  );
  await assertTransientSlotsAuthorized(page);
  await assertBoundedChoreography(page);
  expect(flow.commands).toHaveLength(3);
});

test("a disclosed POV target with no source anchor renders only at impact phase", async ({
  page,
}) => {
  await installWaapiAutopause(page);
  const safeFrame = syntheticDebuggerFrame(povFixture);
  const safeAgents = /** @type {Record<string, any>[]} */ (safeFrame.scene.agents);
  const target = safeAgents.find((agent) => agent.global_slot === 0);
  expect(target).toBeTruthy();
  if (!target) {
    throw new Error("Synthetic POV fixture is missing target id_0.");
  }
  const targetOnlyFrame = {
    ...safeFrame,
    revision: 1,
    event_batch: {
      schema_version: 1,
      transition_id: 1,
      simulator_step: 1,
      events: [
        {
          event_type: "accepted_activation",
          event_id: "synthetic:pov:target-only",
          transition_id: 1,
          token_id: "holy_word",
          source_global_slot: 5,
          target_global_slot: 0,
          source_anchor: null,
          target_anchor: target.position,
          target_disclosure: "public",
          lane: 1,
          source_class_id: 5,
        },
      ],
    },
  };
  const flow = await installSyntheticFlow(
    page,
    withoutTransition(targetOnlyFrame),
    () => targetOnlyFrame,
  );

  await page.goto(debuggerUrl);
  await page.getByRole("button", { name: "Reset" }).click();
  await pauseAtLogicalTime(page, 500);

  const activation = page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`);
  await expect(activation).toHaveCount(1);
  await expect(activation).toHaveAttribute("data-phase", "impact");
  await expect(activation).toHaveAttribute("data-source-slot", "5");
  await expect(activation).toHaveAttribute("data-target-slot", "0");
  await expect(activation.locator(".combat-local")).toHaveCount(0);
  await expect(activation.locator(".combat-impact--holy-word")).toHaveCount(1);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-event-id="synthetic:pov:target-only"]`,
    ),
  ).toHaveCount(0);
  await assertBoundedChoreography(page);
  expect(flow.commands).toHaveLength(1);
});
