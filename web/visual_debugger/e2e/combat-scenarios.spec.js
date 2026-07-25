import { expect, test } from "@playwright/test";

import {
  assertBoundedChoreography,
  CHOREOGRAPHY_ROOT,
  CHOREOGRAPHY_ROUTE_ROOT,
  choreographySnapshot,
  installWaapiAutopause,
  pauseAtLogicalTime,
} from "./support/choreography.js";
import { startDebugger, stopDebugger } from "./support/live-debugger.js";

/** @type {import("node:child_process").ChildProcess | null} */
let serverProcess = null;
let debuggerUrl = "";

test.describe.configure({ mode: "serial" });

test.beforeAll(async () => {
  const started = await startDebugger({
    scenario: "team_focus_crossfire",
    extraArgs: ["--include-stress"],
  });
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
 */
function trackCommandPosts(page) {
  let count = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/command")) {
      count += 1;
    }
  });
  return Object.freeze({
    count: () => count,
  });
}

/**
 * Install a deterministic page and reset or switch the shared live service to
 * the requested scenario.
 *
 * @param {import("@playwright/test").Page} page
 * @param {string} scenario
 */
async function loadScenario(page, scenario) {
  await installWaapiAutopause(page);
  await page.goto(debuggerUrl);
  await expect(page.locator("#connection-status")).toHaveText("Online");

  const revisionBefore = Number(await page.locator("#revision-value").textContent());
  if ((await page.locator("#scenario-select").inputValue()) === scenario) {
    await page.getByRole("button", { name: "Reset" }).click();
  } else {
    await page.locator("#scenario-select").selectOption(scenario);
  }

  await expect(page.locator("#scenario-select")).toHaveValue(scenario);
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect
    .poll(async () => Number(await page.locator("#revision-value").textContent()), {
      timeout: 120_000,
    })
    .toBeGreaterThan(revisionBefore);
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(0);
}

/**
 * Submit one scripted frame and seek its shared presentation clock.
 *
 * @param {import("@playwright/test").Page} page
 * @param {number} transitionId
 * @param {number} [logicalMs]
 */
async function advanceAnimatedFrame(page, transitionId, logicalMs = 520) {
  await page.locator("#battlefield").focus();
  await page.keyboard.press("n");
  await expect(page.locator("#transition-value")).toHaveText(String(transitionId), {
    timeout: 120_000,
  });
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
  await pauseAtLogicalTime(page, logicalMs);
}

/**
 * Release the current presentation gate when one exists.
 *
 * @param {import("@playwright/test").Page} page
 */
async function skipIfAvailable(page) {
  const skip = page.locator("#motion-skip-button");
  if (!(await skip.isDisabled())) {
    await skip.click();
  }
  await expect(page.locator("html")).toHaveAttribute(
    "data-submission-blocked",
    "false",
  );
}

/**
 * @param {import("@playwright/test").Page} page
 * @param {number} slot
 * @param {string} tokenId
 */
function durableRosterStatus(page, slot, tokenId) {
  return page.locator(
    `.roster-row[data-slot="${slot}"] .roster-fact-token--status[data-token-id="${tokenId}"]`,
  );
}

test("focus crossfire retriggers events and keeps presentation controls local", async ({
  page,
}) => {
  const commandPosts = trackCommandPosts(page);
  await loadScenario(page, "team_focus_crossfire");

  const beforeFirstSubmit = commandPosts.count();
  await page.locator("#battlefield").focus();
  await page.keyboard.press("n");
  await expect(page.locator("#transition-value")).toHaveText("1", {
    timeout: 120_000,
  });
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
  expect(commandPosts.count()).toBe(beforeFirstSubmit + 1);

  const afterFirstSubmit = commandPosts.count();
  await page.locator("#battlefield").focus();
  await page.keyboard.press("p");
  await expect(page.locator("html")).toHaveAttribute("data-motion-paused", "true");
  expect(commandPosts.count()).toBe(afterFirstSubmit);

  await pauseAtLogicalTime(page, 520);
  await page.getByRole("button", { name: "0.5×", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-motion-rate", "0.5");
  expect(commandPosts.count()).toBe(afterFirstSubmit);

  await page.locator("#battlefield").focus();
  await page.keyboard.press("n");
  await expect(page.locator("#notice")).toContainText("still being explained");
  expect(commandPosts.count()).toBe(afterFirstSubmit);

  const root = page.locator(CHOREOGRAPHY_ROOT);
  await root.evaluate((element) => {
    element.setAttribute("data-retained-probe", "same-transition");
  });
  const revisionBeforeRangeToggle = Number(
    await page.locator("#revision-value").textContent(),
  );
  const beforeRangeToggle = await choreographySnapshot(page);
  await page.locator("#battlefield").focus();
  await page.keyboard.press("g");
  await expect(page.locator("#revision-value")).toHaveText(
    String(revisionBeforeRangeToggle + 1),
  );
  expect(commandPosts.count()).toBe(afterFirstSubmit + 1);
  const afterRangeToggle = await choreographySnapshot(page);
  await expect(root).toHaveAttribute("data-retained-probe", "same-transition");
  expect(afterRangeToggle.epochKey).toBe(beforeRangeToggle.epochKey);
  expect(afterRangeToggle.effectIds).toEqual(beforeRangeToggle.effectIds);
  expect(afterRangeToggle.animationIds).toEqual(beforeRangeToggle.animationIds);

  const afterRangeTogglePost = commandPosts.count();
  const firstEventIds = new Set(beforeRangeToggle.effectIds);
  await page.locator("#motion-skip-button").click();
  expect(commandPosts.count()).toBe(afterRangeTogglePost);

  await advanceAnimatedFrame(page, 2);
  const second = await choreographySnapshot(page);
  expect(second.effectIds.every((eventId) => !firstEventIds.has(eventId))).toBe(true);
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await advanceAnimatedFrame(page, 3);
  const activations = page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`);
  await expect(activations).toHaveCount(7);
  expect(
    await activations.evaluateAll((nodes) =>
      nodes
        .map((node) => ({
          source: Number(node.getAttribute("data-source-slot")),
          target: Number(node.getAttribute("data-target-slot")),
        }))
        .sort((left, right) => left.source - right.source),
    ),
  ).toEqual([
    { source: 0, target: 5 },
    { source: 1, target: 5 },
    { source: 2, target: 5 },
    { source: 3, target: 5 },
    { source: 6, target: 5 },
    { source: 7, target: 5 },
    { source: 8, target: 5 },
  ]);
  const net = page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--net-health`);
  await expect(net).toHaveCount(1);
  await expect(net).toHaveAttribute("data-recipient-slot", "5");
  await assertBoundedChoreography(page);
});

test("mirrored Ultimates keep activation identity separate from durable consequences", async ({
  page,
}) => {
  await loadScenario(page, "mirrored_ultimates");

  /** @type {Array<{
   *   tokenId: string,
   *   local: boolean,
   *   grammarSelector: string,
   *   grammarCount: number,
   *   routeStyle?: {strokeWidth: number, dashed: boolean},
   *   statuses: Array<[number, string]>,
   * }>} */
  const frames = [
    {
      tokenId: "mage_burst",
      local: true,
      grammarSelector: ".combat-burst__wave",
      grammarCount: 4,
      statuses: [
        [0, "mage_burst"],
        [5, "mage_burst"],
      ],
    },
    {
      tokenId: "warrior_charge",
      local: false,
      grammarSelector: ".combat-charge__impact",
      grammarCount: 2,
      routeStyle: { strokeWidth: 5, dashed: true },
      statuses: [
        [1, "stun_warrior_charge"],
        [1, "slow_warrior_charge"],
        [6, "stun_warrior_charge"],
        [6, "slow_warrior_charge"],
      ],
    },
    {
      tokenId: "hunter_trap",
      local: false,
      grammarSelector: ".combat-trap__lattice",
      grammarCount: 2,
      routeStyle: { strokeWidth: 3, dashed: true },
      statuses: [
        [2, "stun_hunter_trap"],
        [7, "stun_hunter_trap"],
      ],
    },
    {
      tokenId: "rogue_poison",
      local: false,
      grammarSelector: ".combat-poison__splash",
      grammarCount: 6,
      routeStyle: { strokeWidth: 2, dashed: true },
      statuses: [
        [3, "stun_rogue_poison"],
        [3, "slow_rogue_poison"],
        [3, "anti_heal_rogue_poison"],
        [8, "stun_rogue_poison"],
        [8, "slow_rogue_poison"],
        [8, "anti_heal_rogue_poison"],
      ],
    },
    {
      tokenId: "holy_word",
      local: false,
      grammarSelector: ".combat-holy__pulse",
      grammarCount: 4,
      routeStyle: { strokeWidth: 5, dashed: false },
      statuses: [],
    },
  ];

  for (const [index, frame] of frames.entries()) {
    const transitionId = index + 1;
    await advanceAnimatedFrame(page, transitionId);
    const activations = page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="${frame.tokenId}"]`,
    );
    await expect(activations).toHaveCount(2);
    await expect(
      page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`),
    ).toHaveCount(2);
    await expect(activations.locator(".combat-local")).toHaveCount(frame.local ? 2 : 0);
    await expect(
      page.locator(
        `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="${frame.tokenId}"] .combat-route__path`,
      ),
    ).toHaveCount(frame.local ? 0 : 2);
    await expect(
      page.locator(
        `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="${frame.tokenId}"]`,
      ),
    ).toHaveCount(frame.local ? 0 : 2);
    if (frame.routeStyle) {
      const routeStyles = await page
        .locator(
          `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="${frame.tokenId}"] .combat-route__path`,
        )
        .evaluateAll((paths) =>
          paths.map((path) => {
            const style = getComputedStyle(path);
            return {
              strokeWidth: Number.parseFloat(style.strokeWidth),
              dashed: style.strokeDasharray !== "" && style.strokeDasharray !== "none",
            };
          }),
        );
      expect(routeStyles).toEqual([frame.routeStyle, frame.routeStyle]);
    }
    await expect(page.locator(frame.grammarSelector)).toHaveCount(frame.grammarCount);

    if (frame.tokenId === "warrior_charge") {
      await expect(
        page.locator(
          `${CHOREOGRAPHY_ROOT} .combat-effect--charge-displacement[data-persistent="true"]`,
        ),
      ).toHaveCount(2);
    }
    for (const [slot, tokenId] of frame.statuses) {
      await expect(durableRosterStatus(page, slot, tokenId)).toHaveCount(1);
    }
    await assertBoundedChoreography(page);
    await skipIfAvailable(page);
    for (const [slot, tokenId] of frame.statuses) {
      await expect(durableRosterStatus(page, slot, tokenId)).toHaveCount(1);
    }
  }
});

test("converging Charge routes reproject while displacement chords persist exactly one transition", async ({
  page,
}) => {
  await loadScenario(page, "charge_convergence");
  await advanceAnimatedFrame(page, 1);

  const activations = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="warrior_charge"]`,
  );
  await expect(activations).toHaveCount(3);
  expect(
    await activations.evaluateAll((nodes) =>
      nodes
        .map((node) => [
          Number(node.getAttribute("data-source-slot")),
          Number(node.getAttribute("data-target-slot")),
        ])
        .sort((left, right) => left[0] - right[0]),
    ),
  ).toEqual([
    [0, 5],
    [1, 5],
    [5, 0],
  ]);
  const routePaths = await page
    .locator(
      `${CHOREOGRAPHY_ROUTE_ROOT} .combat-route-effect--activation[data-token-id="warrior_charge"] .combat-route__path`,
    )
    .evaluateAll((paths) => paths.map((path) => path.getAttribute("d")));
  expect(new Set(routePaths).size).toBe(3);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--charge-displacement`),
  ).toHaveCount(3);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROUTE_ROOT} .combat-charge__path`),
  ).toHaveCount(3);
  const mitigation = page.locator(
    '#roster .roster-row[data-slot="0"] .roster-fact-token--modifier[data-token-id="warrior_mitigation"]',
  );
  await expect(mitigation).toContainText("×0.72");
  expect(await mitigation.getAttribute("data-multiplier")).toMatch(/\.\d{3,}/);
  const humanFacingValues = await page
    .locator(
      "#battlefield .modifier-cell__value, #roster .roster-fact-token--modifier, .comparison-agent .fact strong, #event-feed .event-item",
    )
    .allTextContents();
  expect(humanFacingValues.every((value) => !/\.\d{3,}/.test(value))).toBe(true);
  await assertBoundedChoreography(page);

  const root = page.locator(CHOREOGRAPHY_ROOT);
  await root.evaluate((element) => {
    element.setAttribute("data-retained-probe", "charge-resize");
  });
  const beforeResize = await choreographySnapshot(page);
  await page.setViewportSize({ width: 960, height: 600 });
  await expect
    .poll(async () => root.getAttribute("data-viewport-key"))
    .not.toBe(beforeResize.viewportKey);
  const afterResize = await choreographySnapshot(page);
  await expect(root).toHaveAttribute("data-retained-probe", "charge-resize");
  expect(afterResize.epochKey).toBe(beforeResize.epochKey);
  expect(afterResize.animationIds).toEqual(beforeResize.animationIds);
  expect(afterResize.effectIds).toEqual(beforeResize.effectIds);

  await page.locator("#motion-skip-button").click();
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`),
  ).toHaveCount(0);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--charge-displacement`),
  ).toHaveCount(3);
  expect((await choreographySnapshot(page)).animationIds).toEqual([]);

  await page.reload();
  await expect(page.locator("#connection-status")).toHaveText("Online");
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--charge-displacement`),
  ).toHaveCount(3);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect:not(.combat-effect--charge-displacement)`,
    ),
  ).toHaveCount(0);
  expect((await choreographySnapshot(page)).animationIds).toEqual([]);
  await assertBoundedChoreography(page);

  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.locator("#step-value")).toHaveText("0");
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(0);
});

test("Trap lifecycle distinguishes application, break, aging, refresh, and ambiguous ending", async ({
  page,
}) => {
  await loadScenario(page, "trap_lifecycle");

  await advanceAnimatedFrame(page, 1);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--activation[data-token-id="hunter_trap"]`,
    ),
  ).toHaveCount(4);
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-token-id="stun_hunter_trap"][data-lifecycle="applied"]`,
    ),
  ).toHaveCount(4);
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await advanceAnimatedFrame(page, 2);
  const exactBreak = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-token-id="stun_hunter_trap"][data-lifecycle="trap_broken"]`,
  );
  await expect(exactBreak).toHaveCount(1);
  await expect(exactBreak).toHaveAttribute("data-recipient-slot", "5");
  await expect(exactBreak.locator(".combat-lifecycle__shard")).toHaveCount(6);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} [data-lifecycle="decremented"]`),
  ).toHaveCount(0);
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await page.locator("#battlefield").focus();
  await page.keyboard.press("n");
  await expect(page.locator("#transition-value")).toHaveText("3", {
    timeout: 120_000,
  });
  await expect(page.locator(CHOREOGRAPHY_ROOT)).toHaveCount(1);
  const expiredBasicSlow = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-token-id="slow_hunter_basic"][data-lifecycle="expired"]`,
  );
  await expect(expiredBasicSlow).toHaveCount(1);
  await expect(expiredBasicSlow).toHaveAttribute("data-recipient-slot", "5");
  await expect(
    page.locator(
      `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-token-id="stun_hunter_trap"]`,
    ),
  ).toHaveCount(0);
  await expect(
    page.locator('#event-feed [data-event-type="status_lifecycle"]'),
  ).not.toHaveCount(0);
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await advanceAnimatedFrame(page, 4);
  const refreshed = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-token-id="stun_hunter_trap"][data-lifecycle="refreshed"]`,
  );
  await expect(refreshed).toHaveCount(1);
  await expect(refreshed).toHaveAttribute("data-recipient-slot", "6");
  await expect(refreshed).toHaveAttribute("data-application-event-ids", /activation/);
  await assertBoundedChoreography(page);
  await skipIfAvailable(page);

  await advanceAnimatedFrame(page, 5);
  const ambiguous = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-token-id="stun_hunter_trap"][data-lifecycle="cleared_unclassified"]`,
  );
  const expired = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-token-id="stun_hunter_trap"][data-lifecycle="expired"]`,
  );
  await expect(ambiguous).toHaveCount(1);
  await expect(ambiguous).toHaveAttribute("data-recipient-slot", "7");
  await expect(expired).toHaveCount(1);
  await expect(expired).toHaveAttribute("data-recipient-slot", "8");
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-lifecycle__shard`),
  ).toHaveCount(0);
  await assertBoundedChoreography(page);
});

test("maximum status stack keeps nine durable channels and exact activation associations", async ({
  page,
}) => {
  await loadScenario(page, "max_status_stack");
  await advanceAnimatedFrame(page, 1, 600);

  const expectedTokens = [
    "anti_heal_rogue_poison",
    "mage_burst",
    "priest_freedom",
    "slow_hunter_basic",
    "slow_rogue_poison",
    "slow_warrior_charge",
    "stun_hunter_trap",
    "stun_rogue_poison",
    "stun_warrior_charge",
  ];
  const durable = page.locator('#battlefield .status-cell[data-slot="0"]');
  await expect(durable).toHaveCount(9);
  expect(
    (
      await durable.evaluateAll((nodes) =>
        nodes.map((node) => node.getAttribute("data-token-id")),
      )
    ).sort(),
  ).toEqual(expectedTokens);

  const activations = page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--activation`);
  await expect(activations).toHaveCount(6);
  const activationIds = new Set(
    await activations.evaluateAll((nodes) =>
      nodes.map((node) => node.getAttribute("data-event-id")),
    ),
  );
  const lifecycle = page.locator(
    `${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle[data-recipient-slot="0"][data-lifecycle="applied"]`,
  );
  await expect(lifecycle).toHaveCount(9);
  const records = await lifecycle.evaluateAll((nodes) =>
    nodes.map((node) => ({
      applicationIds: JSON.parse(
        node.getAttribute("data-application-event-ids") ?? "[]",
      ),
      tokenId: node.getAttribute("data-token-id"),
      statusIcon: node
        .querySelector(".combat-lifecycle__status-icon")
        ?.getAttribute("data-icon"),
      changeIcon: node
        .querySelector(".combat-lifecycle__change-icon")
        ?.getAttribute("data-icon"),
      collisionFree: node
        .querySelector(".combat-lifecycle")
        ?.getAttribute("data-layout-collision-free"),
      transform: node.querySelector(".combat-lifecycle")?.getAttribute("transform"),
    })),
  );
  expect(records.map(({ tokenId }) => tokenId).sort()).toEqual(expectedTokens);
  expect(new Set(records.map(({ transform }) => transform)).size).toBe(9);
  expect(records.every(({ statusIcon }) => statusIcon?.startsWith("status-"))).toBe(
    true,
  );
  expect(records.every(({ changeIcon }) => changeIcon === "lifecycle-applied")).toBe(
    true,
  );
  expect(records.every(({ collisionFree }) => collisionFree === "true")).toBe(true);
  expect(
    records.every(
      ({ applicationIds }) =>
        applicationIds.length === 1 && activationIds.has(applicationIds[0]),
    ),
  ).toBe(true);
  /** @type {Map<string, number>} */
  const associationCounts = new Map();
  for (const { applicationIds } of records) {
    const eventId = String(applicationIds[0]);
    associationCounts.set(eventId, (associationCounts.get(eventId) ?? 0) + 1);
  }
  expect([...associationCounts.values()].sort((left, right) => left - right)).toEqual([
    1, 1, 1, 1, 2, 3,
  ]);
  await assertBoundedChoreography(page);

  await skipIfAvailable(page);
  await expect(durable).toHaveCount(9);
  await expect(
    page.locator(`${CHOREOGRAPHY_ROOT} .combat-effect--status-lifecycle`),
  ).toHaveCount(0);
});
