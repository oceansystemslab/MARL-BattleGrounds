/**
 * @file Check the Replay Viewer's Red Zone metric topic on one current version-4
 * capture at Red Zone depth 5.0: Team A kills agents 5 and 6 inside Team B's Red
 * Zone (2 points each) and Team B kills agent 3 outside Team A's Red Zone (1
 * point). The catalog is scalar schema 15 with 44 applicable Red Zone rows in one
 * Red Zone topic right after Respawning. The table shows team totals, helpers,
 * then victims under three headings; tick-0 counts are 0 and shares are dashes;
 * Entire Episode shows the counts and 0.5 shares. Episode Results keep kills (2
 * and 1) apart from points (4 and 1). Keyboard search finds Red Zone kills
 * without the ordinary kill count, and focus shows the share help. This spec
 * sets no time limits; the Playwright configuration waits without limit.
 */
import { expect, test } from "@playwright/test";
import {
  exportRedZoneReplayArtifacts,
  removeReplayArtifacts,
  startReplayViewer,
  stopDebugger,
} from "./support/replay-viewer.js";

// Written out independently of the Python catalog, in its table order.
const RED_ZONE_NAMES = Object.freeze([
  "team_a_red_zone_kills",
  "team_a_red_zone_deaths",
  "team_b_red_zone_kills",
  "team_b_red_zone_deaths",
  ...Array.from({ length: 10 }, (_, slot) => [
    `agent_${slot}_red_zone_kill_contributions`,
    `agent_${slot}_red_zone_kill_participation`,
  ]).flat(),
  ...Array.from({ length: 10 }, (_, slot) => [
    `agent_${slot}_red_zone_deaths`,
    `agent_${slot}_red_zone_death_fraction`,
  ]).flat(),
]);
const SHARES = new Set(
  RED_ZONE_NAMES.filter((name) => /_(participation|fraction)$/u.test(name)),
);

/** @param {import("@playwright/test").Page} page @param {string} name */
function metricValue(page, name) {
  return page.locator(`#metric-rows [data-metric="${name}"] .metric-value`);
}

test("Red Zone metrics separate kills from points with keyboard search and help", async ({
  page,
}, testInfo) => {
  const artifacts = await exportRedZoneReplayArtifacts();
  /** @type {Awaited<ReturnType<typeof startReplayViewer>> | null} */
  let viewer = null;
  try {
    viewer = await startReplayViewer({ replayPath: artifacts.redZone, frameIndex: 0 });
    /** @type {(catalog: Record<string, any>) => void} */
    let recordCatalog = () => {};
    /** @type {Promise<Record<string, any>>} */
    const catalogLoaded = new Promise((resolve) => {
      recordCatalog = resolve;
    });
    // Keep the one large catalog body before Chromium can drop it from its cache.
    await page.route("**/api/replay/metrics/catalog.json", async (route) => {
      const response = await route.fetch();
      expect(response.status()).toBe(200);
      recordCatalog(await response.json());
      await route.fulfill({ response });
    });
    await page.goto(viewer.url);
    await expect(page.locator("#connection-status")).toHaveText("Online");
    await expect(page.locator("html")).toHaveAttribute("data-viewer-mode", "replay");
    const firstPrefix = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname === "/api/replay/metrics/0/cursor.json",
    );
    await page.locator("#evaluation-metrics > summary").click();
    expect((await firstPrefix).status()).toBe(200);
    const catalog = await catalogLoaded;
    expect(catalog.metric_schema_version).toBe(15);
    expect(catalog.measurements).toHaveLength(11192);
    const redZone = catalog.measurements.filter(
      (/** @type {Record<string, any>} */ row) => row.family === "red_zone",
    );
    expect(redZone.map((/** @type {Record<string, any>} */ row) => row.name)).toEqual(
      RED_ZONE_NAMES,
    );
    expect(
      redZone.every((/** @type {Record<string, any>} */ row) => row.applicable),
    ).toBe(true);

    const topics = await page
      .locator("#metric-selection option")
      .evaluateAll((options) => options.map((option) => option.getAttribute("value")));
    expect(topics[topics.indexOf("respawn") + 1]).toBe("red_zone");
    expect(
      await page
        .locator(
          '#metric-selection optgroup[label="Kills, Deaths and Respawning"] option',
        )
        .evaluateAll((options) =>
          options.map((option) => [option.getAttribute("value"), option.textContent]),
        ),
    ).toEqual([
      ["kill_contributions", "Kill Contributions"],
      ["controlled_kills", "Kills of Enemies With Harmful Effects"],
      ["deaths", "Deaths and Time Dead"],
      ["respawn", "Respawning"],
      ["red_zone", "Red Zone"],
    ]);

    // Up to Current Tick at tick 0: counts are real zeros, shares are dashes.
    await page.locator("#metric-selection").selectOption("red_zone");
    await expect(page.locator("#metric-view-field")).toBeHidden();
    await expect(page.locator("#metric-status")).toContainText(
      "Up to Current Tick · tick 0",
    );
    expect(
      await page
        .locator("#metric-rows tr[data-metric]")
        .evaluateAll((rows) => rows.map((row) => row.getAttribute("data-metric"))),
    ).toEqual(RED_ZONE_NAMES);
    await expect(page.locator("#metric-rows tr.metric-section th")).toHaveText([
      "Episode and Team Totals",
      "Agent Details",
      "Recipient Totals",
    ]);
    await expect(page.locator("#metric-description")).toContainText(
      "24 of 44 measurements available.",
    );
    for (const name of RED_ZONE_NAMES) {
      if (SHARES.has(name)) {
        await expect(metricValue(page, name)).toHaveText("—");
        await expect(metricValue(page, name)).toHaveAttribute(
          "aria-label",
          /^Unavailable\. Possible reasons: .*is zero/u,
        );
      } else {
        await expect(metricValue(page, name)).toHaveText("0");
      }
    }

    // Entire Episode: two Red Zone kills, one ordinary kill and 0.5 shares.
    await page.locator("#metric-scope").selectOption("final");
    await expect(page.locator("#metric-status")).toContainText(
      "Entire Episode · tick 1",
    );
    /** @type {Record<string, string>} */
    const finalValues = {
      team_a_red_zone_kills: "2",
      team_a_red_zone_deaths: "0",
      team_b_red_zone_kills: "0",
      team_b_red_zone_deaths: "2",
      agent_0_red_zone_kill_contributions: "1",
      agent_0_red_zone_kill_participation: "0.5",
      agent_1_red_zone_kill_contributions: "1",
      agent_1_red_zone_kill_participation: "0.5",
      agent_2_red_zone_kill_participation: "0",
      agent_8_red_zone_kill_contributions: "0",
      agent_8_red_zone_kill_participation: "—",
      agent_3_red_zone_deaths: "0",
      agent_3_red_zone_death_fraction: "—",
      agent_5_red_zone_deaths: "1",
      agent_5_red_zone_death_fraction: "0.5",
      agent_6_red_zone_death_fraction: "0.5",
      agent_7_red_zone_death_fraction: "0",
    };
    for (const [name, text] of Object.entries(finalValues)) {
      await expect(metricValue(page, name)).toHaveText(text);
    }
    await expect(page.locator("#metric-description")).toContainText(
      "34 of 44 measurements available.",
    );
    await page.locator("#evaluation-metrics").screenshot({
      path: testInfo.outputPath("red-zone-entire-episode.png"),
    });

    // Focus shows the share's full help, including both parts of the fraction.
    await page
      .locator('[data-metric="agent_0_red_zone_kill_participation"] .metric-measure')
      .focus();
    const tooltip = page.locator("#visual-tooltip");
    await expect(tooltip).toBeVisible();
    for (const text of [
      "Share of Team A Red Zone Kills",
      "Several teammates can help with the same kill",
      "Share: 1 means 100%",
      "Numerator",
      "Red Zone kills this agent helped with during the selected period",
      "Denominator",
      "All Red Zone kills by Team A during the same period",
      "Context dependent for Team A",
      "Blank When",
      "the Red Zone rule was not recorded",
      "agent_0_red_zone_kill_participation",
    ]) {
      await expect(tooltip).toContainText(text);
    }
    await page.keyboard.press("Escape");

    // Episode Results keep kills apart from the points they gave.
    await page.locator("#metric-selection").selectOption("priority");
    for (const [name, text] of [
      ["team_a_score", "4"],
      ["team_b_score", "1"],
      ["team_a_kills", "2"],
      ["team_b_kills", "1"],
      ["team_a_deaths", "1"],
      ["team_b_deaths", "2"],
    ]) {
      await expect(metricValue(page, name)).toHaveText(text);
    }

    // Keyboard search finds Red Zone kills, not the ordinary kill count.
    const search = page.locator("#metric-search");
    const results = page.locator("#metric-search-results button");
    await search.fill("red zone kills");
    await expect(results.first()).toHaveAttribute(
      "data-measurement",
      "team_a_red_zone_kills",
    );
    const found = await results.evaluateAll((buttons) =>
      buttons.map((button) => button.getAttribute("data-measurement") ?? ""),
    );
    expect(found.length).toBeGreaterThan(0);
    expect(found.every((name) => /_red_zone_kill/u.test(name))).toBe(true);
    await search.press("ArrowDown");
    await expect(results.first()).toHaveAttribute("aria-selected", "true");
    await search.press("Enter");
    await expect(page.locator("#metric-selection")).toHaveValue("red_zone");
    const measure = page.locator(
      '[data-metric="team_a_red_zone_kills"] .metric-measure',
    );
    await expect(measure).toBeFocused();
    await expect(tooltip).toBeVisible();
    await expect(tooltip).toContainText("Red Zone Kills");
    await expect(tooltip).toContainText("Each enemy death counts once");
    await expect(tooltip).toContainText("2 points");
    await expect(tooltip).toContainText("Higher is better for Team A");
    await page.keyboard.press("Escape");
    await search.fill("red zone deaths");
    await expect(results.first()).toBeVisible();
    const deaths = await results.evaluateAll((buttons) =>
      buttons.map((button) => button.getAttribute("data-measurement") ?? ""),
    );
    expect(deaths.every((name) => /_red_zone_death/u.test(name))).toBe(true);
    await search.press("Escape");
  } finally {
    await page.goto("about:blank").catch(() => {});
    await stopDebugger(viewer?.process ?? null);
    await removeReplayArtifacts(artifacts.outputDirectory);
  }
});
