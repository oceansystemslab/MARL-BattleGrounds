import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";
import { MINIMUM_VIEWPORT } from "./support/visual-regression.js";

const WEB_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SOURCE_ROOT = resolve(WEB_ROOT, "src");
const fixture = JSON.parse(
  readFileSync(
    new URL("../tests/fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @type {import("node:http").Server | null} */
let server = null;
let origin = "";

test.beforeAll(async () => {
  server = createServer(async (request, response) => {
    try {
      const pathname = new URL(request.url ?? "/", "http://127.0.0.1").pathname;
      if (pathname === "/") {
        response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        response.end(`<!doctype html>
          <html><head><link rel="stylesheet" href="/styles.css"></head><body>
            <svg id="battlefield" class="battlefield" style="width: 800px; height: 600px"></svg>
            <p id="empty"></p>
            <aside id="visual-tooltip" class="visual-tooltip" hidden>
              <strong id="visual-tooltip-title"></strong>
              <div id="visual-tooltip-details"></div>
            </aside>
          </body></html>`);
        return;
      }
      if (pathname === "/styles.css") {
        response.writeHead(200, { "content-type": "text/css; charset=utf-8" });
        response.end(await readFile(resolve(WEB_ROOT, "styles.css")));
        return;
      }
      const path = resolve(WEB_ROOT, `.${pathname}`);
      if (!path.startsWith(`${SOURCE_ROOT}${sep}`) || extname(path) !== ".js") {
        response.writeHead(404).end();
        return;
      }
      response.writeHead(200, { "content-type": "text/javascript; charset=utf-8" });
      response.end(await readFile(path));
    } catch (error) {
      response.writeHead(500).end(String(error));
    }
  });
  await new Promise((resolveListen, rejectListen) => {
    server?.once("error", rejectListen);
    server?.listen(0, "127.0.0.1", () => resolveListen(undefined));
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("Authorized presentation test server has no TCP address.");
  }
  origin = `http://127.0.0.1:${address.port}`;
});

test.afterAll(async () => {
  if (server === null) {
    return;
  }
  await new Promise((resolveClose, rejectClose) => {
    server?.close((error) => (error ? rejectClose(error) : resolveClose(undefined)));
  });
});

/**
 * @param {import("@playwright/test").Page} page
 * @param {Record<string, any>} rawPresentation
 * @param {boolean} showRanges
 */
async function renderPresentation(page, rawPresentation, showRanges) {
  await page.goto(origin);
  return await page.evaluate(
    async ({ rawPresentation: raw, showRanges: visible }) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { authorizedPresentationSceneView } = await import(
        `${moduleRoot}/authorized-presentation-adapter.js`
      );
      const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
      const authorizedScene = authorizedPresentationSceneView(presentation);
      const authorizedRanges = Array.isArray(authorizedScene?.ranges)
        ? authorizedScene.ranges.filter(
            (/** @type {Record<string, any>} */ range) =>
              typeof range.radius === "number" && range.radius > 0,
          )
        : [];
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Renderer test surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      const painted = renderer.render(presentation, {
        showRanges: visible,
        // Deliberately ignored: transport may control visibility, never geometry.
        ranges: [{ center: [999, 999], radius: 999, global_slot: 999 }],
      });
      const rangeNodes = Array.from(
        battlefield.querySelectorAll('[data-layer="debug-range"] .range-ring'),
      );
      const bodyNodes = Array.from(battlefield.querySelectorAll(".agent"));
      const slotAttributes = Array.from(battlefield.querySelectorAll("*"))
        .flatMap((element) => Array.from(element.attributes))
        .map((attribute) => attribute.name)
        .filter(
          (name) =>
            name === "data-slot" ||
            name === "data-source-slot" ||
            name === "data-target-slot" ||
            name === "data-recipient-slot" ||
            name === "data-actor-slot" ||
            name.endsWith("-slots"),
        );
      return {
        painted,
        audience: presentation.presentation_kind,
        authorizedRangeCount: authorizedRanges.length,
        authorizedRangeKeys: authorizedRanges.map(
          (/** @type {Record<string, any>} */ range) => range.presentation_key,
        ),
        inspectionOwner: presentation.inspection?.actor_presentation_key ?? null,
        rangeCount: rangeNodes.length,
        rangeKeys: rangeNodes.map((node) => node.getAttribute("data-presentation-key")),
        bodyCount: bodyNodes.length,
        bodyKeys: bodyNodes.map((node) => node.getAttribute("data-presentation-key")),
        bodySlots: bodyNodes.map((node) => node.getAttribute("data-slot")),
        slotAttributes,
      };
    },
    { rawPresentation, showRanges },
  );
}

test("replay Oracle range preference changes visibility but not authority", async ({
  page,
}) => {
  const raw = fixture.presentations.replay_oracle;
  const hidden = await renderPresentation(page, raw, false);
  const visible = await renderPresentation(page, raw, true);
  expect(hidden.painted).toBe(true);
  expect(hidden.rangeCount).toBe(0);
  expect(visible.rangeCount).toBe(visible.authorizedRangeCount);
  expect(visible.rangeKeys).toEqual(visible.authorizedRangeKeys);
});

test("Agent DOM identity remains opaque and raw options cannot mint ranges", async ({
  page,
}) => {
  const raw = fixture.presentations.replay_shared_obs_agent_pov;
  const hidden = await renderPresentation(page, raw, false);
  const visible = await renderPresentation(page, raw, true);
  expect(hidden.rangeCount).toBe(0);
  expect(hidden.slotAttributes).toEqual([]);
  expect(visible.rangeCount).toBe(visible.authorizedRangeCount);
  expect(visible.rangeKeys).toEqual(visible.authorizedRangeKeys);
  expect(visible.bodyCount).toBeGreaterThan(0);
  expect(
    visible.bodyKeys.every((key) => typeof key === "string" && key.length > 0),
  ).toBe(true);
  expect(visible.bodySlots).toEqual(Array(visible.bodyCount).fill(null));
  expect(visible.slotAttributes).toEqual([]);
});

test("raw and forged researcher-looking scenes render unavailable and clear accepted paint", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { authorizedPresentationSceneView } = await import(
      `${moduleRoot}/authorized-presentation-adapter.js`
    );
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const authorizedScene = authorizedPresentationSceneView(presentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (
      authorizedScene === null ||
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement)
    ) {
      throw new Error("Raw authority-fence test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    if (!renderer.render(presentation, { showRanges: true })) {
      throw new Error("The accepted control frame did not paint.");
    }
    const acceptedCounts = {
      agents: battlefield.querySelectorAll(".agent").length,
      auras: battlefield.querySelectorAll(".aura-field").length,
    };
    const candidates = [
      { candidate_kind: "raw_scene_envelope", scene: structuredClone(authorizedScene) },
      structuredClone(presentation),
    ];
    const rejected = candidates.map((candidate) => {
      const painted = renderer.render(candidate, { showRanges: true });
      return {
        painted,
        agents: battlefield.querySelectorAll(".agent").length,
        auras: battlefield.querySelectorAll(".aura-field").length,
        audience: battlefield.getAttribute("data-audience"),
        preset: battlefield.getAttribute("data-preset"),
        viewBox: battlefield.getAttribute("viewBox"),
        emptyHidden: empty.hidden,
        emptyText: empty.textContent,
      };
    });
    return { acceptedCounts, rejected };
  }, fixture.presentations.replay_oracle);

  expect(result.acceptedCounts.agents).toBeGreaterThan(0);
  expect(result.acceptedCounts.auras).toBeGreaterThan(0);
  expect(result.rejected).toEqual([
    {
      painted: false,
      agents: 0,
      auras: 0,
      audience: null,
      preset: null,
      viewBox: null,
      emptyHidden: false,
      emptyText: "No authorized battlefield scene was returned.",
    },
    {
      painted: false,
      agents: 0,
      auras: 0,
      audience: null,
      preset: null,
      viewBox: null,
      emptyHidden: false,
      emptyText: "No authorized battlefield scene was returned.",
    },
  ]);
});

test("durable visual filters remove owned paint and restore stable battlefield identity", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(
    async ({ oracleRaw, povRaw }) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { DEFAULT_VISUAL_FILTER_STATE, setVisualFilterEnabled } = await import(
        `${moduleRoot}/visual-filters.js`
      );
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Durable visual-filter test surface is unavailable.");
      }
      const presentation = await normalizeAuthorizedPresentationFrameV1(oracleRaw);
      const povPresentation = await normalizeAuthorizedPresentationFrameV1(povRaw);
      const presentationBytes = JSON.stringify(presentation);
      const povPresentationBytes = JSON.stringify(povPresentation);
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      const filterCases = [
        {
          filterId: "aura_fields",
          ownerSelector: ".aura-field",
          ownsTooltip: true,
          reservesLayout: false,
          layoutAttribute: null,
        },
        {
          filterId: "aura_modifier_badges",
          ownerSelector: ".modifier-cell, .modifier-overflow",
          ownsTooltip: true,
          reservesLayout: true,
          layoutAttribute: "data-suppressed-modifier-presentation-keys",
        },
        {
          filterId: "duration_status_badges",
          ownerSelector:
            '.status-cell, .status-overflow, .required-dock-fallback[data-kind="status"], .pov-observed-status',
          ownsTooltip: true,
          reservesLayout: true,
          layoutAttribute: "data-suppressed-status-presentation-keys",
        },
        {
          filterId: "spawn_shield",
          ownerSelector: ".agent-spawn-shield",
          ownsTooltip: false,
          reservesLayout: false,
          layoutAttribute: null,
        },
        {
          filterId: "cooldown_effects",
          ownerSelector:
            '.cooldown-cell, .required-dock-fallback[data-kind="cooldown"]',
          ownsTooltip: true,
          reservesLayout: true,
          layoutAttribute: "data-suppressed-cooldown-presentation-keys",
        },
      ];
      const viewports = [
        { width: 720, height: 480 },
        { width: 1080, height: 720 },
      ];
      const rows = [];
      const coreSelectors = [
        ".map-boundary",
        ".obstacle",
        ".agent",
        ".agent-body",
        ".agent-health",
        ".agent-team-ring",
        ".agent-class-icon",
        ".agent-dead-mark",
        ".agent-selection",
      ];
      /**
       * @param {{left: number, top: number, right: number, bottom: number}} bounds
       */
      const protectedRectangleKey = ({ left, top, right, bottom }) =>
        [left, top, right, bottom].map((value) => value.toFixed(3)).join(",");
      const protectedKey = () =>
        (renderer.choreographySurface()?.protectedRects ?? [])
          .map(protectedRectangleKey)
          .join(";");
      const coreCounts = () =>
        Object.fromEntries(
          coreSelectors.map((selector) => [
            selector,
            battlefield.querySelectorAll(selector).length,
          ]),
        );
      /** @param {string} ownerSelector */
      const ownedSnapshot = (ownerSelector) => {
        const owners = Array.from(battlefield.querySelectorAll(ownerSelector));
        return {
          count: owners.length,
          tooltipOwnerCount: owners.filter((owner) =>
            owner.hasAttribute("data-tooltip-owner"),
          ).length,
          ariaOwnerCount: owners.filter(
            (owner) =>
              owner.hasAttribute("aria-label") ||
              owner.hasAttribute("aria-hidden") ||
              owner.hasAttribute("role"),
          ).length,
        };
      };

      for (const viewport of viewports) {
        battlefield.style.width = `${viewport.width}px`;
        battlefield.style.height = `${viewport.height}px`;
        renderer.render(presentation, { showRanges: true });
        const defaultAllOnMarkup = battlefield.outerHTML;
        renderer.render(presentation, {
          showRanges: true,
          visualFilterState: DEFAULT_VISUAL_FILTER_STATE,
        });
        const explicitAllOnMatchesDefault =
          battlefield.outerHTML === defaultAllOnMarkup;
        const originalRoot = battlefield;
        const originalAgent = battlefield.querySelector(".agent");
        const originalBodyHit = originalAgent?.querySelector(".agent-body") ?? null;
        if (!(originalAgent instanceof SVGElement) || !originalBodyHit) {
          throw new Error("Stable agent identity surface is unavailable.");
        }
        const baselineCoreCounts = coreCounts();

        for (const filterCase of filterCases) {
          const baselineOwned = ownedSnapshot(filterCase.ownerSelector);
          const baselineProtectedKey = protectedKey();
          const baselineAgentAria = originalAgent.getAttribute("aria-label") ?? "";
          const durableLayer = battlefield.querySelector(
            '[data-layer="durable-status-modifier"]',
          );
          const baselineLayoutAttribute =
            filterCase.layoutAttribute === null
              ? null
              : durableLayer?.hasAttribute(filterCase.layoutAttribute) === true;
          const disabledState = setVisualFilterEnabled(
            DEFAULT_VISUAL_FILTER_STATE,
            filterCase.filterId,
            false,
          );
          renderer.render(presentation, {
            showRanges: true,
            visualFilterState: disabledState,
          });
          const disabledOwned = ownedSnapshot(filterCase.ownerSelector);
          const disabledProtectedKey = protectedKey();
          const disabledAgentAria = originalAgent.getAttribute("aria-label") ?? "";
          const disabledLayoutAttribute =
            filterCase.layoutAttribute === null
              ? null
              : durableLayer?.hasAttribute(filterCase.layoutAttribute) === true;
          const disabledCoreCounts = coreCounts();
          const disabledIdentityStable =
            battlefield === originalRoot &&
            originalAgent.isConnected &&
            originalBodyHit.isConnected &&
            battlefield.contains(originalAgent) &&
            originalAgent.querySelector(".agent-body") === originalBodyHit;
          const disabledAgentAccessible =
            originalAgent.getAttribute("role") === "img" &&
            originalAgent.hasAttribute("data-tooltip-owner") &&
            disabledAgentAria.length > 0;

          renderer.render(presentation, {
            showRanges: true,
            visualFilterState: DEFAULT_VISUAL_FILTER_STATE,
          });
          const restoredOwned = ownedSnapshot(filterCase.ownerSelector);
          const restoredProtectedKey = protectedKey();
          const restoredAgentAria = originalAgent.getAttribute("aria-label") ?? "";
          rows.push({
            viewport: `${viewport.width}x${viewport.height}`,
            ...filterCase,
            explicitAllOnMatchesDefault,
            baselineOwned,
            baselineProtectedKey,
            baselineAgentAria,
            baselineLayoutAttribute,
            baselineCoreCounts,
            disabledOwned,
            disabledProtectedKey,
            disabledAgentAria,
            disabledLayoutAttribute,
            disabledCoreCounts,
            disabledIdentityStable,
            disabledAgentAccessible,
            restoredOwned,
            restoredProtectedKey,
            restoredAgentAria,
            restoredIdentityStable:
              battlefield === originalRoot &&
              originalAgent.isConnected &&
              originalAgent.querySelector(".agent-body") === originalBodyHit,
          });
        }
      }
      const povStatusRows = [];
      const statusOwnerSelector =
        '.status-cell, .status-overflow, .required-dock-fallback[data-kind="status"], .pov-observed-status';
      for (const viewport of viewports) {
        battlefield.style.width = `${viewport.width}px`;
        battlefield.style.height = `${viewport.height}px`;
        renderer.render(povPresentation, {
          showRanges: true,
          visualFilterState: DEFAULT_VISUAL_FILTER_STATE,
        });
        const originalAgent = battlefield.querySelector(".agent");
        const originalBodyHit = originalAgent?.querySelector(".agent-body") ?? null;
        if (!(originalAgent instanceof SVGElement) || !originalBodyHit) {
          throw new Error("Agent POV status identity surface is unavailable.");
        }
        const baselineOwned = ownedSnapshot(statusOwnerSelector);
        renderer.render(povPresentation, {
          showRanges: true,
          visualFilterState: setVisualFilterEnabled(
            DEFAULT_VISUAL_FILTER_STATE,
            "duration_status_badges",
            false,
          ),
        });
        const disabledOwned = ownedSnapshot(statusOwnerSelector);
        const disabledIdentityStable =
          originalAgent.isConnected &&
          originalAgent.querySelector(".agent-body") === originalBodyHit &&
          originalAgent.getAttribute("role") === "img" &&
          originalAgent.hasAttribute("data-tooltip-owner") &&
          (originalAgent.getAttribute("aria-label") ?? "").length > 0;
        renderer.render(povPresentation, {
          showRanges: true,
          visualFilterState: DEFAULT_VISUAL_FILTER_STATE,
        });
        povStatusRows.push({
          viewport: `${viewport.width}x${viewport.height}`,
          audience: battlefield.getAttribute("data-audience"),
          baselineOwned,
          disabledOwned,
          restoredOwned: ownedSnapshot(statusOwnerSelector),
          disabledIdentityStable,
        });
      }
      return {
        presentationUnchanged:
          JSON.stringify(presentation) === presentationBytes &&
          JSON.stringify(povPresentation) === povPresentationBytes,
        stateFrozen: Object.isFrozen(DEFAULT_VISUAL_FILTER_STATE),
        rows,
        povStatusRows,
      };
    },
    {
      oracleRaw: fixture.state_cases.replay_oracle_final_selected,
      povRaw: fixture.state_cases.replay_shared_final,
    },
  );

  expect(result.presentationUnchanged).toBe(true);
  expect(result.stateFrozen).toBe(true);
  expect(result.rows).toHaveLength(10);
  for (const row of result.rows) {
    expect(row.explicitAllOnMatchesDefault, row.viewport).toBe(true);
    expect(row.baselineOwned.count, `${row.viewport} ${row.filterId}`).toBeGreaterThan(
      0,
    );
    expect(row.baselineOwned.ariaOwnerCount, row.filterId).toBeGreaterThan(0);
    if (row.ownsTooltip) {
      expect(row.baselineOwned.tooltipOwnerCount, row.filterId).toBeGreaterThan(0);
    }
    expect(row.disabledOwned).toEqual({
      count: 0,
      tooltipOwnerCount: 0,
      ariaOwnerCount: 0,
    });
    expect(row.disabledCoreCounts).toEqual(row.baselineCoreCounts);
    expect(row.disabledIdentityStable).toBe(true);
    expect(row.disabledAgentAccessible).toBe(true);
    expect(row.restoredOwned).toEqual(row.baselineOwned);
    expect(row.restoredProtectedKey).toBe(row.baselineProtectedKey);
    expect(row.restoredIdentityStable).toBe(true);
    if (row.reservesLayout) {
      expect(row.disabledProtectedKey, row.filterId).not.toBe(row.baselineProtectedKey);
      expect(row.baselineLayoutAttribute, row.filterId).toBe(true);
      expect(row.disabledLayoutAttribute, row.filterId).toBe(false);
    }
  }
  expect(result.povStatusRows).toHaveLength(2);
  for (const row of result.povStatusRows) {
    expect(row.audience).toBe("agent_pov");
    expect(row.baselineOwned.count, row.viewport).toBeGreaterThan(0);
    expect(row.baselineOwned.tooltipOwnerCount, row.viewport).toBeGreaterThan(0);
    expect(row.baselineOwned.ariaOwnerCount, row.viewport).toBeGreaterThan(0);
    expect(row.disabledOwned).toEqual({
      count: 0,
      tooltipOwnerCount: 0,
      ariaOwnerCount: 0,
    });
    expect(row.restoredOwned).toEqual(row.baselineOwned);
    expect(row.disabledIdentityStable).toBe(true);
  }
});

test("selected Replay aura modifiers remain exact beside Mage Burst duration and cooldown", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { authorizedPresentationSceneView } = await import(
      `${moduleRoot}/authorized-presentation-adapter.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { resolveVisualToken } = await import(`${moduleRoot}/vocabulary.js`);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Aura-coexistence test surface is unavailable.");
    }
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const scene = authorizedPresentationSceneView(presentation);
    if (scene === null) {
      throw new Error("Authorized Replay scene is unavailable.");
    }
    const expected = scene.agents
      .flatMap((/** @type {Record<string, any>} */ agent) =>
        (Array.isArray(agent.modifiers) ? agent.modifiers : [])
          .filter(
            (/** @type {Record<string, any>} */ modifier) =>
              typeof modifier.multiplier !== "number" ||
              !Number.isFinite(modifier.multiplier) ||
              modifier.multiplier !== 1,
          )
          .map((/** @type {Record<string, any>} */ modifier) => {
            const token = resolveVisualToken("modifier", modifier.token_id, modifier);
            return `${agent.presentation_key}|${token.tokenId}`;
          }),
      )
      .sort();
    const selectedKey = scene.selection?.selected_presentation_key ?? null;
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    const rows = [];
    for (const viewport of [
      { width: 960, height: 600 },
      { width: 1440, height: 900 },
    ]) {
      battlefield.style.width = `${viewport.width}px`;
      battlefield.style.height = `${viewport.height}px`;
      renderer.render(presentation, { showRanges: true });
      const actual = Array.from(
        battlefield.querySelectorAll(".modifier-cell"),
        (cell) =>
          `${cell.getAttribute("data-presentation-key")}|${cell.getAttribute("data-token-id")}`,
      ).sort();
      const durableLayer = battlefield.querySelector(
        '[data-layer="durable-status-modifier"]',
      );
      rows.push({
        viewport: `${viewport.width}x${viewport.height}`,
        actual,
        suppressed:
          durableLayer?.getAttribute("data-suppressed-modifier-presentation-keys") ??
          null,
        collisionFree: Array.from(
          battlefield.querySelectorAll(".modifier-dock"),
          (dock) => dock.getAttribute("data-collision-free"),
        ),
        selectedStatusCount:
          selectedKey === null
            ? 0
            : battlefield.querySelectorAll(
                `.status-cell[data-presentation-key="${selectedKey}"]`,
              ).length,
        selectedCooldownCount:
          selectedKey === null
            ? 0
            : battlefield.querySelectorAll(
                `.cooldown-cell[data-presentation-key="${selectedKey}"]`,
              ).length,
      });
    }
    return { expected, selectedKey, rows };
  }, fixture.state_cases.replay_oracle_final_selected);

  expect(result.selectedKey).not.toBeNull();
  expect(result.expected).toHaveLength(2);
  expect(
    result.expected.some((/** @type {string} */ key) =>
      key.endsWith("|mage_amplification"),
    ),
  ).toBe(true);
  expect(
    result.expected.some((/** @type {string} */ key) =>
      key.endsWith("|warrior_mitigation"),
    ),
  ).toBe(true);
  for (const row of result.rows) {
    expect(row.actual, row.viewport).toEqual(result.expected);
    expect(row.suppressed, row.viewport).toBe("");
    expect(row.collisionFree, row.viewport).not.toContain("false");
    expect(row.selectedStatusCount, row.viewport).toBeGreaterThan(0);
    expect(row.selectedCooldownCount, row.viewport).toBeGreaterThan(0);
  }
});

test("all three normalized Agent POV leaves keep hidden aura sources byte-inert", async ({
  page,
}) => {
  const leafNames = [
    "live_no_shared_obs_agent_pov",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ];
  const renderings = [];
  for (const leafName of leafNames) {
    const raw = fixture.presentations[leafName];
    await page.goto(origin);
    const installed = await page.evaluate(async (rawPresentation) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
      const { resolveVisualToken } = await import(`${moduleRoot}/vocabulary.js`);
      const presentation =
        await normalizeAuthorizedPresentationFrameV1(rawPresentation);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      const tooltip = document.querySelector("#visual-tooltip");
      const title = document.querySelector("#visual-tooltip-title");
      const details = document.querySelector("#visual-tooltip-details");
      if (
        !(battlefield instanceof SVGSVGElement) ||
        !(empty instanceof HTMLElement) ||
        !(tooltip instanceof HTMLElement) ||
        !(title instanceof HTMLElement) ||
        !(details instanceof HTMLElement)
      ) {
        throw new Error("Agent POV aura tooltip surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      const painted = renderer.render(presentation, { showRanges: true });
      const controller = createTooltipController({
        root: document.body,
        tooltip,
        title,
        details,
      });
      const rawScene = rawPresentation.current_endpoint?.parts?.scene;
      const rawAuraFields = Array.isArray(rawScene?.aura_fields)
        ? rawScene.aura_fields
        : [];
      const expectedModifierRows = (
        Array.isArray(rawScene?.agents) ? rawScene.agents : []
      )
        .flatMap((/** @type {Record<string, any>} */ agent) =>
          (Array.isArray(agent.aura_modifiers) ? agent.aura_modifiers : []).map(
            (/** @type {Record<string, any>} */ modifier) => {
              const token = resolveVisualToken("modifier", modifier.aura_id, modifier);
              return `${agent.presentation_key}|${token.tokenId}|${modifier.multiplier}`;
            },
          ),
        )
        .sort();
      const auraOwners = Array.from(
        battlefield.querySelectorAll(
          '[data-layer="aura"] .aura-field[data-tooltip-owner]',
        ),
      );
      const tooltipBytes = auraOwners.map((owner) => {
        if (!(owner instanceof SVGElement)) {
          throw new Error("Agent POV aura owner is not an SVG element.");
        }
        owner.focus();
        return JSON.stringify({
          title: title.textContent,
          details: details.innerHTML,
          text: details.textContent,
        });
      });
      const auraMarkup = auraOwners.map((owner) => owner.outerHTML);
      const modifierOwners = Array.from(
        battlefield.querySelectorAll(
          '[data-layer="durable-status-modifier"] .modifier-cell[data-tooltip-owner]',
        ),
      );
      const modifierRows = modifierOwners
        .map(
          (owner) =>
            `${owner.getAttribute("data-presentation-key")}|${owner.getAttribute("data-token-id")}|${owner.getAttribute("data-multiplier")}`,
        )
        .sort();
      const modifierTooltipBytes = modifierOwners.map((owner) => {
        if (!(owner instanceof SVGElement)) {
          throw new Error("Agent POV modifier owner is not an SVG element.");
        }
        owner.focus();
        return JSON.stringify({
          title: title.textContent,
          details: details.innerHTML,
          text: details.textContent,
        });
      });
      const modifierMarkup = modifierOwners.map((owner) => owner.outerHTML);
      controller.destroy();
      return {
        auraCount: auraOwners.length,
        audience: battlefield.dataset.audience,
        painted,
        hiddenSourceKeys: rawAuraFields.map(
          (/** @type {Record<string, any>} */ field) => field.source_presentation_key,
        ),
        sourceAttributeCount: battlefield.querySelectorAll(
          '[data-layer="aura"] [data-source-presentation-key]',
        ).length,
        auraMarkup,
        tooltipBytes,
        expectedModifierRows,
        modifierRows,
        modifierMarkup,
        modifierTooltipBytes,
        modifierSourceAttributeCount: battlefield.querySelectorAll(
          '[data-layer="durable-status-modifier"] .modifier-dock [data-source-presentation-key], [data-layer="durable-status-modifier"] .modifier-dock [data-source-public-agent-id], [data-layer="durable-status-modifier"] .modifier-dock [data-source-class-id]',
        ).length,
      };
    }, raw);
    expect(installed.auraCount).toBe(2);
    expect(installed.audience).toBe("agent_pov");
    expect(installed.painted).toBe(true);
    expect(installed.sourceAttributeCount).toBe(0);
    expect(installed.hiddenSourceKeys).toHaveLength(2);
    expect(new Set(installed.hiddenSourceKeys).size).toBe(2);
    for (const hiddenSourceKey of installed.hiddenSourceKeys) {
      expect(typeof hiddenSourceKey).toBe("string");
      expect(installed.auraMarkup.join("\n")).not.toContain(hiddenSourceKey);
      expect(installed.tooltipBytes.join("\n")).not.toContain(hiddenSourceKey);
    }
    expect(installed.tooltipBytes).toHaveLength(2);
    for (const tooltipBytes of installed.tooltipBytes) {
      expect(tooltipBytes).toContain("Source");
      expect(tooltipBytes).toContain("Not disclosed in Agent POV");
      expect(tooltipBytes).not.toContain("agent-slot-");
    }
    expect(installed.expectedModifierRows).toHaveLength(2);
    expect(installed.modifierRows).toEqual(installed.expectedModifierRows);
    expect(installed.modifierSourceAttributeCount).toBe(0);
    expect(installed.modifierTooltipBytes).toHaveLength(2);
    expect(installed.modifierMarkup.join("\n")).not.toContain("data-source-");
    for (const tooltipBytes of installed.modifierTooltipBytes) {
      expect(tooltipBytes).not.toContain('semantic-explanation__label">Source');
      expect(tooltipBytes).not.toContain("Emitter");
    }
    renderings.push(installed);
  }

  const allHiddenSourceKeys = renderings.flatMap(
    (rendering) => rendering.hiddenSourceKeys,
  );
  expect(new Set(allHiddenSourceKeys).size).toBe(allHiddenSourceKeys.length);
  expect(renderings[1].hiddenSourceKeys).not.toEqual(renderings[0].hiddenSourceKeys);
  expect(renderings[2].hiddenSourceKeys).not.toEqual(renderings[0].hiddenSourceKeys);
  expect(renderings[1].auraMarkup).toEqual(renderings[0].auraMarkup);
  expect(renderings[2].auraMarkup).toEqual(renderings[0].auraMarkup);
  expect(renderings[1].tooltipBytes).toEqual(renderings[0].tooltipBytes);
  expect(renderings[2].tooltipBytes).toEqual(renderings[0].tooltipBytes);
  expect(renderings[1].modifierTooltipBytes).toEqual(
    renderings[0].modifierTooltipBytes,
  );
  expect(renderings[2].modifierTooltipBytes).toEqual(
    renderings[0].modifierTooltipBytes,
  );
});

test("Oracle aura attribution remains exact in the tooltip and absent from aura DOM keys", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    const tooltip = document.querySelector("#visual-tooltip");
    const title = document.querySelector("#visual-tooltip-title");
    const details = document.querySelector("#visual-tooltip-details");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement) ||
      !(tooltip instanceof HTMLElement) ||
      !(title instanceof HTMLElement) ||
      !(details instanceof HTMLElement)
    ) {
      throw new Error("Oracle aura tooltip surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    const painted = renderer.render(presentation, { showRanges: true });
    const controller = createTooltipController({
      root: document.body,
      tooltip,
      title,
      details,
    });
    const auraOwners = Array.from(
      battlefield.querySelectorAll(
        '[data-layer="aura"] .aura-field[data-tooltip-owner]',
      ),
    );
    const firstAura = auraOwners[0];
    if (!(firstAura instanceof SVGElement)) {
      throw new Error("Oracle aura owner is unavailable.");
    }
    firstAura.focus();
    const labels = Array.from(
      details.querySelectorAll(".semantic-explanation__label"),
      (node) => node.textContent,
    );
    const values = Array.from(
      details.querySelectorAll(".semantic-explanation__value"),
      (node) => node.textContent,
    );
    const auraMarkup = auraOwners.map((owner) => owner.outerHTML);
    controller.destroy();
    return {
      painted,
      audience: battlefield.dataset.audience,
      auraCount: auraOwners.length,
      sourceAttributeCount: battlefield.querySelectorAll(
        '[data-layer="aura"] [data-source-presentation-key]',
      ).length,
      auraMarkup,
      labels,
      values,
    };
  }, fixture.presentations.replay_oracle);

  expect(result.painted).toBe(true);
  expect(result.audience).toBe("researcher");
  expect(result.auraCount).toBe(2);
  expect(result.sourceAttributeCount).toBe(0);
  expect(result.auraMarkup.join("\n")).not.toContain("data-source-presentation-key");
  const sourceIndex = result.labels.indexOf("Source");
  expect(sourceIndex).toBeGreaterThanOrEqual(0);
  expect(result.values[sourceIndex]).toBe("Agent ID agent-slot-0 · Mage · Team A");
});

test("incoming choreography paints presentation-key metadata and no slots", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Renderer test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, { showRanges: true });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface);
    if (surface === null || plan === null) {
      throw new Error("Authorized choreography surface or plan is unavailable.");
    }
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });
    const eventNodes = Array.from(battlefield.querySelectorAll(".combat-effect"));
    return {
      eventCount: eventNodes.length,
      keyAttributeCount: eventNodes.reduce(
        (count, node) =>
          count +
          Array.from(node.attributes).filter((attribute) =>
            attribute.name.endsWith("-presentation-key"),
          ).length,
        0,
      ),
      slotAttributeCount: Array.from(
        battlefield.querySelectorAll(
          ".combat-choreography *, .combat-choreography-routes *",
        ),
      ).reduce(
        (count, node) =>
          count +
          Array.from(node.attributes).filter(
            (attribute) =>
              attribute.name === "data-slot" || attribute.name.endsWith("-slot"),
          ).length,
        0,
      ),
    };
  }, fixture.presentations.replay_oracle);
  expect(result.eventCount).toBeGreaterThan(0);
  expect(result.keyAttributeCount).toBeGreaterThan(0);
  expect(result.slotAttributeCount).toBe(0);
});

test("live and replay damage/healing marks meet their activation route endpoint", async ({
  page,
}) => {
  const presentations = [
    {
      sourcePresentation: fixture.presentations.live_oracle,
      expectedSemantic: "healing",
    },
    {
      sourcePresentation: fixture.presentations.live_oracle,
      expectedSemantic: "damage",
    },
    {
      sourcePresentation: fixture.presentations.replay_oracle,
      expectedSemantic: "healing",
    },
    {
      sourcePresentation: fixture.presentations.replay_oracle,
      expectedSemantic: "damage",
    },
  ];
  for (const { sourcePresentation, expectedSemantic } of presentations) {
    const raw = structuredClone(sourcePresentation);
    const agentsByKey = new Map(
      raw.current_endpoint.scene.agents.map(
        (/** @type {Record<string, any>} */ agent) => [agent.presentation_key, agent],
      ),
    );
    /** @param {number} classId */
    const anchorForClass = (classId) => {
      const trajectory = raw.latest_events.agent_phase_trajectories.find(
        (/** @type {Record<string, any>} */ candidate) =>
          agentsByKey.get(candidate.agent_presentation_key)?.class_id === classId,
      );
      if (!trajectory) {
        throw new Error(`Oracle class ${classId} trajectory is unavailable.`);
      }
      return structuredClone(trajectory.transition_start);
    };
    const activation = raw.latest_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "ability_activated",
    );
    if (!activation) {
      throw new Error("Oracle activation event is unavailable.");
    }
    if (expectedSemantic === "damage") {
      activation.source_anchor = anchorForClass(1);
      activation.recipient_anchor = anchorForClass(3);
    }
    await page.goto(origin);
    const result = await page.evaluate(async (rawPresentation) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { buildChoreographyPlan } = await import(
        `${moduleRoot}/choreography-plan.js`
      );
      const { SvgChoreographyPainter } = await import(
        `${moduleRoot}/choreography-painter.js`
      );
      const presentation =
        await normalizeAuthorizedPresentationFrameV1(rawPresentation);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Activation-impact test surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      renderer.render(presentation, { showRanges: false });
      const surface = renderer.choreographySurface();
      const plan = buildChoreographyPlan(presentation, surface);
      if (surface === null || plan === null) {
        throw new Error("Authorized activation choreography is unavailable.");
      }
      new SvgChoreographyPainter().install(plan, surface, {
        motionMode: "off",
        settled: false,
        persistentOnly: false,
      });
      const activations = /** @type {Record<string, any>[]} */ (plan.events).filter(
        (event) =>
          event.kind === "activation" &&
          (event.impactSemantic === "damage" || event.impactSemantic === "healing") &&
          event.route,
      );
      return {
        presentationKind: presentation.presentation_kind,
        scrollingTextCount: battlefield.querySelectorAll(".combat-net__label").length,
        activations: activations.map((event) => {
          const selector = `[data-event-id="${CSS.escape(event.eventId)}"]`;
          const effect = battlefield.querySelector(
            `.combat-effect--activation${selector}`,
          );
          const routeEffect = battlefield.querySelector(
            `.combat-route-effect--activation${selector}`,
          );
          const impact = effect?.querySelector(".combat-impact");
          const semantic = impact?.querySelector(".combat-impact__semantic");
          const routePath = routeEffect?.querySelector(".combat-route__path");
          if (
            !(effect instanceof SVGElement) ||
            !(impact instanceof SVGGraphicsElement) ||
            !(semantic instanceof SVGElement) ||
            !(routePath instanceof SVGPathElement)
          ) {
            throw new Error(`Activation impact ${event.eventId} is unavailable.`);
          }
          const matrix = impact.transform.baseVal.consolidate()?.matrix;
          if (!matrix) {
            throw new Error(`Activation impact ${event.eventId} has no transform.`);
          }
          const pathEnd = routePath.getPointAtLength(routePath.getTotalLength());
          return {
            semantic: event.impactSemantic,
            routeEnd: event.route.end,
            transform: { x: matrix.e, y: matrix.f },
            pathEnd: { x: pathEnd.x, y: pathEnd.y },
            semanticLineCount: semantic.querySelectorAll(":scope > line").length,
            impactLayoutAttributes: impact
              .getAttributeNames()
              .filter((name) => name.startsWith("data-layout-")),
            impactLeaderCount: effect.querySelectorAll(".combat-cue__leader--impact")
              .length,
            hasPlannedImpactCue: event.impactCue !== undefined,
            hasPlannedImpactBounds: event.impactBounds !== undefined,
            hasPlannedImpactLeader: event.impactLeader !== undefined,
            hasPlannedImpactLayoutKey: event.impactLayoutKey !== undefined,
            hasPlannedImpactDisposition: event.impactDisposition !== undefined,
            hasPlannedImpactCollisionFlag: event.impactCueCollisionFree !== undefined,
          };
        }),
      };
    }, raw);

    expect(result.presentationKind).toBe(raw.presentation_kind);
    expect(result.activations.length).toBeGreaterThan(0);
    expect(new Set(result.activations.map(({ semantic }) => semantic))).toEqual(
      new Set([expectedSemantic]),
    );
    expect(result.scrollingTextCount).toBeGreaterThan(0);
    for (const activation of result.activations) {
      expect(activation.transform.x).toBeCloseTo(activation.routeEnd.x, 3);
      expect(activation.transform.y).toBeCloseTo(activation.routeEnd.y, 3);
      expect(activation.pathEnd.x).toBeCloseTo(activation.routeEnd.x, 3);
      expect(activation.pathEnd.y).toBeCloseTo(activation.routeEnd.y, 3);
      expect(activation.semanticLineCount).toBe(
        activation.semantic === "healing" ? 2 : 1,
      );
      expect(activation.impactLayoutAttributes).toEqual([]);
      expect(activation.impactLeaderCount).toBe(0);
      expect(activation.hasPlannedImpactCue).toBe(false);
      expect(activation.hasPlannedImpactBounds).toBe(false);
      expect(activation.hasPlannedImpactLeader).toBe(false);
      expect(activation.hasPlannedImpactLayoutKey).toBe(false);
      expect(activation.hasPlannedImpactDisposition).toBe(false);
      expect(activation.hasPlannedImpactCollisionFlag).toBe(false);
    }
  }
});

test("Agent POV damage and healing impacts stop outside the recipient body without intercepting its centre", async ({
  page,
}) => {
  await page.setViewportSize(MINIMUM_VIEWPORT);
  const cases = [
    {
      leafName: "live_no_shared_obs_agent_pov",
      expectedSemantic: "healing",
    },
    {
      leafName: "live_no_shared_obs_agent_pov",
      expectedSemantic: "damage",
    },
    {
      leafName: "replay_no_shared_obs_agent_pov",
      expectedSemantic: "healing",
    },
    {
      leafName: "replay_no_shared_obs_agent_pov",
      expectedSemantic: "damage",
    },
  ];

  for (const { leafName, expectedSemantic } of cases) {
    const raw = structuredClone(fixture.presentations[leafName]);
    const activation = raw.visual_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "ability_activated",
    );
    if (!activation) {
      throw new Error(`${leafName} Agent activation is unavailable.`);
    }
    if (expectedSemantic === "damage") {
      const sourceTrajectory = raw.visual_events.agent_phase_trajectories.find(
        (/** @type {Record<string, any>} */ trajectory) =>
          trajectory.agent_class_id === 1 && trajectory.successor !== null,
      );
      const targetTrajectory = raw.visual_events.agent_phase_trajectories.find(
        (/** @type {Record<string, any>} */ trajectory) =>
          trajectory.agent_class_id === 2 && trajectory.successor !== null,
      );
      if (!sourceTrajectory?.transition_start || !targetTrajectory?.transition_start) {
        throw new Error(`${leafName} Agent damage trajectories are unavailable.`);
      }
      activation.source_anchor = structuredClone(sourceTrajectory.transition_start);
      activation.recipient_anchor = structuredClone(targetTrajectory.transition_start);
    }

    await page.goto(origin);
    const geometry = await page.evaluate(
      async ({ rawPresentation, expectedImpactSemantic }) => {
        const moduleRoot = "/src";
        const { normalizeAuthorizedPresentationFrameV1 } = await import(
          `${moduleRoot}/authorized-presentation-normalizer.js`
        );
        const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
        const { buildChoreographyPlan } = await import(
          `${moduleRoot}/choreography-plan.js`
        );
        const { SvgChoreographyPainter } = await import(
          `${moduleRoot}/choreography-painter.js`
        );
        const presentation =
          await normalizeAuthorizedPresentationFrameV1(rawPresentation);
        const battlefield = document.querySelector("#battlefield");
        const empty = document.querySelector("#empty");
        if (
          !(battlefield instanceof SVGSVGElement) ||
          !(empty instanceof HTMLElement)
        ) {
          throw new Error("Agent impact test surface is unavailable.");
        }
        document.body.style.margin = "0";
        battlefield.style.width = `${window.innerWidth}px`;
        battlefield.style.height = `${window.innerHeight}px`;
        battlefield.style.display = "block";
        const renderer = new BattlefieldRenderer({ battlefield, empty });
        renderer.render(presentation, { showRanges: false });
        const surface = renderer.choreographySurface();
        const plan = buildChoreographyPlan(presentation, surface);
        if (surface === null || plan === null) {
          throw new Error("Agent impact choreography is unavailable.");
        }
        new SvgChoreographyPainter().install(plan, surface, {
          motionMode: "off",
          settled: false,
          persistentOnly: false,
        });
        const event = /** @type {Record<string, any> | undefined} */ (
          plan.events.find(
            (/** @type {Record<string, any>} */ candidate) =>
              candidate.kind === "activation" &&
              candidate.component === "basic" &&
              candidate.impactSemantic === expectedImpactSemantic &&
              candidate.route,
          )
        );
        if (!event || typeof event.targetPresentationKey !== "string") {
          throw new Error(
            `Agent ${expectedImpactSemantic} activation route is unavailable.`,
          );
        }
        const escapedEventId = CSS.escape(event.eventId);
        const effect = battlefield.querySelector(
          `.combat-effect--activation[data-event-id="${escapedEventId}"]`,
        );
        const impact = effect?.querySelector(".combat-impact");
        const semantic = impact?.querySelector(".combat-impact__semantic");
        const targetAgent = battlefield.querySelector(
          `.agent[data-presentation-key="${CSS.escape(event.targetPresentationKey)}"]`,
        );
        const body = targetAgent?.querySelector(".agent-body");
        if (
          !(effect instanceof SVGElement) ||
          !(impact instanceof SVGGraphicsElement) ||
          !(semantic instanceof SVGElement) ||
          !(targetAgent instanceof SVGElement) ||
          !(body instanceof SVGCircleElement)
        ) {
          throw new Error("Agent impact or recipient body is unavailable.");
        }
        const impactMatrix = impact.transform.baseVal.consolidate()?.matrix;
        const bodyMatrix = body.getScreenCTM();
        if (!impactMatrix || bodyMatrix === null) {
          throw new Error("Agent impact or recipient screen geometry is unavailable.");
        }
        const centre = { x: body.cx.baseVal.value, y: body.cy.baseVal.value };
        const bodyRadius = body.r.baseVal.value;
        const routeEndDistance = Math.hypot(
          event.route.end.x - centre.x,
          event.route.end.y - centre.y,
        );
        const impactDistance = Math.hypot(
          impactMatrix.e - centre.x,
          impactMatrix.f - centre.y,
        );
        const targetCentreDistance = Math.hypot(
          event.target.x - centre.x,
          event.target.y - centre.y,
        );
        const screenCentre = new DOMPoint(centre.x, centre.y).matrixTransform(
          bodyMatrix,
        );
        battlefield.addEventListener(
          "pointerdown",
          (pointerEvent) => {
            const pointerTarget =
              pointerEvent.target instanceof Element ? pointerEvent.target : null;
            battlefield.dataset.centrePointerAgentKey =
              pointerTarget
                ?.closest(".agent[data-presentation-key]")
                ?.getAttribute("data-presentation-key") ?? "";
            battlefield.dataset.centrePointerInterceptedByImpact = String(
              pointerTarget?.closest(".combat-impact") !== null,
            );
          },
          { capture: true, once: true },
        );
        return {
          audience: battlefield.dataset.audience,
          component: event.component,
          impactSemantic: event.impactSemantic,
          sourceEndpointPhase: event.sourceEndpointPhase,
          targetEndpointPhase: event.targetEndpointPhase,
          targetPresentationKey: event.targetPresentationKey,
          bodyRadius,
          routeEndDistance,
          impactDistance,
          targetCentreDistance,
          semanticLineCount: semantic.querySelectorAll(":scope > line").length,
          screenCentre: { x: screenCentre.x, y: screenCentre.y },
          viewport: { width: window.innerWidth, height: window.innerHeight },
        };
      },
      { rawPresentation: raw, expectedImpactSemantic: expectedSemantic },
    );

    await page.mouse.click(geometry.screenCentre.x, geometry.screenCentre.y);
    const pointerTarget = await page
      .locator("#battlefield")
      .evaluate((battlefield) => ({
        agentKey: battlefield.dataset.centrePointerAgentKey ?? null,
        interceptedByImpact:
          battlefield.dataset.centrePointerInterceptedByImpact ?? null,
      }));

    expect(geometry.audience, `${leafName} ${expectedSemantic}`).toBe("agent_pov");
    expect(geometry.viewport).toEqual(MINIMUM_VIEWPORT);
    expect(geometry.component).toBe("basic");
    expect(geometry.impactSemantic).toBe(expectedSemantic);
    expect(geometry.sourceEndpointPhase).toBe("successor");
    expect(geometry.targetEndpointPhase).toBe("successor");
    expect(geometry.targetCentreDistance).toBeCloseTo(0, 3);
    expect(geometry.routeEndDistance).toBeCloseTo(geometry.bodyRadius + 3, 3);
    expect(geometry.impactDistance).toBeCloseTo(geometry.routeEndDistance, 3);
    expect(geometry.semanticLineCount).toBe(expectedSemantic === "healing" ? 2 : 1);
    expect(pointerTarget.agentKey).toBe(geometry.targetPresentationKey);
    expect(pointerTarget.interceptedByImpact).toBe("false");
  }
});

test("live and replay keep Charge displacement authorized without painting an overlay", async ({
  page,
}) => {
  for (const leafName of ["live_oracle", "replay_oracle"]) {
    const raw = structuredClone(fixture.presentations[leafName]);
    const warrior = raw.current_endpoint.scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.class_id === 2,
    );
    const trajectory = raw.latest_events.agent_phase_trajectories.find(
      (/** @type {Record<string, any>} */ candidate) =>
        candidate.agent_presentation_key === warrior?.presentation_key,
    );
    if (!warrior || !trajectory) {
      throw new Error(`${leafName} Warrior trajectory is unavailable.`);
    }
    trajectory.transition_start.position = [3, 4];
    trajectory.post_charge.position = [7, 5];
    const event = {
      event_id: `${raw.latest_events.incoming_transition_id}:event:0000`,
      event_kind: "charge_phase_displacement",
      ordinal: 0,
      phase_rank: 70,
      realized_displacement: [4, 1],
      start_anchor: structuredClone(trajectory.transition_start),
      end_anchor: structuredClone(trajectory.post_charge),
    };
    raw.latest_events.events = [event];
    raw.latest_events.event_count = 1;
    raw.latest_events.ordered_event_ids = [event.event_id];
    raw.latest_events.ordered_event_kinds = [event.event_kind];

    await page.goto(origin);
    const result = await page.evaluate(async (rawPresentation) => {
      const moduleRoot = "/src";
      const { normalizeAuthorizedPresentationFrameV1 } = await import(
        `${moduleRoot}/authorized-presentation-normalizer.js`
      );
      const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
      const { buildChoreographyPlan } = await import(
        `${moduleRoot}/choreography-plan.js`
      );
      const { SvgChoreographyPainter } = await import(
        `${moduleRoot}/choreography-painter.js`
      );
      const presentation =
        await normalizeAuthorizedPresentationFrameV1(rawPresentation);
      const battlefield = document.querySelector("#battlefield");
      const empty = document.querySelector("#empty");
      if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
        throw new Error("Charge displacement test surface is unavailable.");
      }
      const renderer = new BattlefieldRenderer({ battlefield, empty });
      renderer.render(presentation, { showRanges: false });
      const surface = renderer.choreographySurface();
      const plan = buildChoreographyPlan(presentation, surface);
      if (surface === null || plan === null) {
        throw new Error("Authorized Charge event is unavailable.");
      }
      new SvgChoreographyPainter().install(plan, surface, {
        motionMode: "off",
        settled: false,
        persistentOnly: false,
      });
      const charge = /** @type {Record<string, any>} */ (plan.events[0]);
      return {
        presentationKind: presentation.presentation_kind,
        eventType: charge.eventType,
        kind: charge.kind,
        spatial: charge.spatial,
        geometryFields: [
          "start",
          "end",
          "route",
          "sourcePresentationKey",
          "sourcePublicAgentId",
          "paintParts",
          "persistent",
        ].filter((field) => Object.hasOwn(charge, field)),
        pathCount: battlefield.querySelectorAll(".combat-charge__path").length,
        endpointCount: battlefield.querySelectorAll(".combat-charge__endpoint").length,
        directionCount: battlefield.querySelectorAll(".combat-charge__direction")
          .length,
        leaderCount: battlefield.querySelectorAll(
          ".combat-cue__leader--charge-start, .combat-cue__leader--charge-end",
        ).length,
      };
    }, raw);

    expect(result.presentationKind).toBe(leafName);
    expect(result.eventType).toBe("charge_phase_displacement");
    expect(result.kind).toBe("feed_only");
    expect(result.spatial).toBe(false);
    expect(result.geometryFields).toEqual([]);
    expect(result.pathCount).toBe(0);
    expect(result.endpointCount).toBe(0);
    expect(result.directionCount).toBe(0);
    expect(result.leaderCount).toBe(0);
  }
});

test("digest-valid OOC state keeps all five class glyphs centered across projected radii", async ({
  page,
}) => {
  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Renderer test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const viewports = [
      { width: 360, height: 240 },
      { width: 720, height: 480 },
      { width: 1080, height: 720 },
    ];
    const states = [];
    for (const viewport of viewports) {
      battlefield.style.width = `${viewport.width}px`;
      battlefield.style.height = `${viewport.height}px`;
      renderer.render(presentation, { showRanges: false });
      states.push(
        ...Array.from(battlefield.querySelectorAll(".agent")).map((agent) => {
          const classIcon = agent.querySelector(".agent-class-icon");
          const body = agent.querySelector(".agent-body");
          if (
            !(classIcon instanceof SVGSVGElement) ||
            !(body instanceof SVGCircleElement)
          ) {
            throw new Error("Agent identity glyphs are unavailable.");
          }
          const classX = Number(classIcon.getAttribute("x"));
          const classWidth = Number(classIcon.getAttribute("width"));
          const bodyCenter = Number(body.getAttribute("cx"));
          return {
            classToken: agent.getAttribute("data-class"),
            viewport: `${battlefield.clientWidth}x${battlefield.clientHeight}`,
            projectedRadius: Number(body.getAttribute("r")),
            status: agent.getAttribute("data-combat-status"),
            countdown: agent.getAttribute("data-steps-until-out-of-combat"),
            ariaLabel: agent.getAttribute("aria-label"),
            dedicatedCombatIconCount: agent.querySelectorAll(".agent-combat-state-icon")
              .length,
            inCombatStatusCount: battlefield.querySelectorAll(
              '.status-cell[data-token-id="in_combat"]',
            ).length,
            classCentered: Math.abs(classX + classWidth / 2 - bodyCenter) < 0.001,
          };
        }),
      );
    }
    return states;
  }, fixture.presentations.replay_oracle);

  expect(new Set(result.map(({ classToken }) => classToken))).toEqual(
    new Set(["mage", "warrior", "hunter", "rogue", "priest"]),
  );
  expect(new Set(result.map(({ viewport }) => viewport)).size).toBe(3);
  expect(
    new Set(result.map(({ projectedRadius }) => projectedRadius.toFixed(3))).size,
  ).toBe(3);
  expect(result).toHaveLength(15);
  for (const state of result) {
    expect(state.dedicatedCombatIconCount).toBe(0);
    expect(state.inCombatStatusCount).toBe(0);
    expect(state.status).toBe("OOC");
    expect(state.countdown).toBe("0");
    expect(state.ariaLabel).not.toMatch(/combat|steps until out/iu);
    expect(state.classCentered).toBe(true);
  }
});

test("authorized regeneration paints packed successor plus cues while reset stays feed-only", async ({
  page,
}) => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const transitionId = raw.latest_events.incoming_transition_id;
  const trajectories = raw.latest_events.agent_phase_trajectories;
  const specifications = [
    {
      event_kind: "combat_countdown_reset",
      agent_anchor: structuredClone(trajectories[0].transition_start),
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: structuredClone(trajectories[0].transition_start),
      actual_health_regenerated: 4,
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: structuredClone(trajectories[1].transition_start),
      actual_health_regenerated: 2,
    },
    {
      event_kind: "health_regenerated",
      agent_anchor: structuredClone(trajectories[2].transition_start),
      actual_health_regenerated: 1,
    },
  ];
  raw.latest_events.events = specifications.map((event, ordinal) => ({
    ...event,
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
    phase_rank: 50,
  }));
  raw.latest_events.event_count = raw.latest_events.events.length;
  raw.latest_events.ordered_event_ids = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  raw.latest_events.ordered_event_kinds = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );

  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Renderer test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, { showRanges: false });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface);
    if (surface === null || plan === null) {
      throw new Error("Authorized regeneration choreography is unavailable.");
    }
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });
    const plannedEvents = /** @type {Record<string, any>[]} */ (plan.events);
    const regenerations = plannedEvents.filter(
      (event) => event.kind === "regeneration",
    );
    const reset = plannedEvents.find(
      (event) => event.eventType === "combat_countdown_reset",
    );
    const effects = Array.from(
      battlefield.querySelectorAll(".combat-effect--regeneration"),
    );
    return {
      planEventIds: plannedEvents.map((event) => event.eventId),
      reset: reset
        ? {
            kind: reset.kind,
            spatial: reset.spatial,
            hasPhase: Object.hasOwn(reset, "phaseStart"),
          }
        : null,
      regenerationValues: regenerations.map((event) => event.value),
      regenerationAnchors: regenerations.map((event) => event.recipient),
      expectedRegenerationAnchors: /** @type {Record<string, any>[]} */ (
        presentation.latest_events.agent_phase_trajectories
      )
        .slice(0, 3)
        .map(({ successor }) => surface.worldToScreen(successor.position)),
      regenerationCues: regenerations.map((event) => event.cue),
      regenerationBounds: regenerations.map((event) => event.cueBounds),
      regenerationCollisionFree: regenerations.map((event) => event.cueCollisionFree),
      regenerationPhases: regenerations.map((event) => [
        event.phaseStart,
        event.phaseEnd,
      ]),
      effectCount: effects.length,
      values: effects.map(
        (effect) =>
          effect.querySelector(".combat-regeneration__value")?.textContent ?? null,
      ),
      plusLineCounts: effects.map(
        (effect) => effect.querySelectorAll(".combat-regeneration__plus > line").length,
      ),
      tooltipOwnerCounts: effects.map((effect) =>
        effect.hasAttribute("data-tooltip-owner") ? 1 : 0,
      ),
      sourceAttributeCount: effects.reduce(
        (count, effect) =>
          count +
          Array.from(effect.attributes).filter((attribute) =>
            attribute.name.includes("source"),
          ).length,
        0,
      ),
      onionCount: battlefield.querySelectorAll(
        ".combat-effect--regeneration .combat-semantic-pulse__ring, .combat-effect--regeneration .combat-semantic-pulse__core",
      ).length,
      resetEffectCount: battlefield.querySelectorAll(
        '.combat-effect[data-event-type="combat_countdown_reset"]',
      ).length,
      routeChildCount: battlefield.querySelectorAll(".combat-choreography-routes > *")
        .length,
    };
  }, raw);

  expect(result.planEventIds).toEqual(raw.latest_events.ordered_event_ids);
  expect(result.reset).toEqual({ kind: "feed_only", spatial: false, hasPhase: false });
  expect(result.regenerationValues).toEqual([4, 2, 1]);
  expect(result.regenerationAnchors).toEqual(result.expectedRegenerationAnchors);
  expect(
    result.regenerationCues.every(
      (/** @type {Record<string, number> | null} */ cue) => cue !== null,
    ),
  ).toBe(true);
  expect(result.regenerationCollisionFree).toEqual([true, true, true]);
  expect(
    new Set(
      result.regenerationPhases.map((/** @type {number[]} */ phase) =>
        JSON.stringify(phase),
      ),
    ),
  ).toEqual(new Set([JSON.stringify(result.regenerationPhases[0])]));
  for (const [index, bounds] of result.regenerationBounds.entries()) {
    for (const prior of result.regenerationBounds.slice(0, index)) {
      const overlap =
        Math.max(
          0,
          Math.min(bounds.right, prior.right) - Math.max(bounds.left, prior.left),
        ) *
        Math.max(
          0,
          Math.min(bounds.bottom, prior.bottom) - Math.max(bounds.top, prior.top),
        );
      expect(overlap).toBe(0);
    }
  }
  expect(result.effectCount).toBe(3);
  expect(result.values).toEqual(["+4", "+2", "+1"]);
  expect(result.plusLineCounts).toEqual([2, 2, 2]);
  expect(result.tooltipOwnerCounts).toEqual([1, 1, 1]);
  expect(result.sourceAttributeCount).toBe(0);
  expect(result.onionCount).toBe(0);
  expect(result.resetEffectCount).toBe(0);
  expect(result.routeChildCount).toBe(0);
});

test("authorized death, team waves, and resurrection retain outward settled geometry", async ({
  page,
}) => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const transitionId = raw.latest_events.incoming_transition_id;
  const trajectories = raw.latest_events.agent_phase_trajectories;
  const deathAnchor = structuredClone(trajectories[4].successor);
  const shieldAnchor = structuredClone(trajectories[0].successor);
  const respawnAnchor = structuredClone(trajectories[3].successor);
  const specifications = [
    {
      event_kind: "agent_died",
      recipient_anchor: deathAnchor,
      phase_rank: 90,
    },
    {
      event_kind: "spawn_shield_expired",
      agent_anchor: shieldAnchor,
      phase_rank: 110,
    },
    {
      event_kind: "respawn_wave_occurred",
      team_anchor: { phase: "successor", team_index: 0, team_id: 1 },
      phase_rank: 120,
    },
    {
      event_kind: "respawn_wave_occurred",
      team_anchor: { phase: "successor", team_index: 1, team_id: 2 },
      phase_rank: 120,
    },
    {
      event_kind: "agent_respawned",
      agent_anchor: respawnAnchor,
      team_id: 2,
      realized_successor_position: structuredClone(respawnAnchor.position),
      phase_rank: 120,
    },
  ];
  raw.latest_events.events = specifications.map((event, ordinal) => ({
    ...event,
    event_id: `${transitionId}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
  }));
  raw.latest_events.event_count = raw.latest_events.events.length;
  raw.latest_events.ordered_event_ids = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  raw.latest_events.ordered_event_kinds = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );

  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const serializedBefore = JSON.stringify(presentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    const tooltip = document.querySelector("#visual-tooltip");
    const title = document.querySelector("#visual-tooltip-title");
    const details = document.querySelector("#visual-tooltip-details");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement) ||
      !(tooltip instanceof HTMLElement) ||
      !(title instanceof HTMLElement) ||
      !(details instanceof HTMLElement)
    ) {
      throw new Error("Renderer test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, { showRanges: false });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface);
    if (surface === null || plan === null) {
      throw new Error("Authorized lifecycle choreography is unavailable.");
    }
    const painter = new SvgChoreographyPainter();
    const normal = painter.install(plan, surface, {
      motionMode: "normal",
      settled: false,
      persistentOnly: false,
    });
    const controller = createTooltipController({
      root: document.body,
      tooltip,
      title,
      details,
    });
    const lifecycleAnimations = normal.animationSpecs
      .filter((/** @type {Record<string, any>} */ spec) =>
        spec.id.endsWith(":lifecycle-ring"),
      )
      .map((/** @type {Record<string, any>} */ spec) =>
        spec.keyframes.map((/** @type {Record<string, any>} */ keyframe) => ({
          radius: String(keyframe.r),
          opacity: Number(keyframe.opacity),
        })),
      );
    const lifecycleNodes = Array.from(
      battlefield.querySelectorAll(".combat-lifecycle-ring"),
    );
    const waves = Array.from(battlefield.querySelectorAll(".combat-respawn-wave"));
    const waveGeometry = waves.map((wave) => {
      const effect = wave.closest(".combat-effect");
      const event = plan.events.find(
        (/** @type {Record<string, any>} */ candidate) =>
          candidate.eventId === effect?.getAttribute("data-event-id"),
      );
      const panel = wave.querySelector(".combat-respawn-wave__panel");
      const label = wave.querySelector(".combat-respawn-wave__label");
      if (!event?.anchor || !(panel instanceof SVGRectElement)) {
        throw new Error("Authorized wave geometry is unavailable.");
      }
      return {
        teamIndex: event.teamIndex,
        teamId: event.teamId,
        side: event.teamSide,
        anchor: event.anchor,
        bounds: {
          left: event.anchor.x + panel.x.baseVal.value,
          top: event.anchor.y + panel.y.baseVal.value,
          right: event.anchor.x + panel.x.baseVal.value + panel.width.baseVal.value,
          bottom: event.anchor.y + panel.y.baseVal.value + panel.height.baseVal.value,
        },
        label: label?.textContent ?? null,
        color: getComputedStyle(wave).color,
        ownerCount: wave.querySelectorAll("[data-tooltip-owner]").length,
        layoutKey: wave.getAttribute("data-layout-key"),
        layoutDisposition: wave.getAttribute("data-layout-disposition"),
        layoutCollisionFree: wave.getAttribute("data-layout-collision-free"),
        ariaLabel: effect?.getAttribute("aria-label") ?? null,
      };
    });
    const lifecycleGeometry = lifecycleNodes.map((node) => {
      const effect = node.closest(".combat-effect");
      const event = plan.events.find(
        (/** @type {Record<string, any>} */ candidate) =>
          candidate.eventId === effect?.getAttribute("data-event-id"),
      );
      const ring = node.querySelector(".combat-lifecycle-ring__ring");
      const hit = node.querySelector(".combat-lifecycle-ring__hit");
      const agent =
        typeof event?.agentPresentationKey === "string"
          ? battlefield.querySelector(
              `.agent[data-presentation-key="${CSS.escape(event.agentPresentationKey)}"]`,
            )
          : null;
      const body = agent?.querySelector(".agent-body");
      if (
        !(node instanceof SVGGraphicsElement) ||
        !(effect instanceof SVGElement) ||
        !event?.anchor ||
        !(ring instanceof SVGCircleElement) ||
        !(hit instanceof SVGCircleElement) ||
        !(agent instanceof SVGElement) ||
        !(body instanceof SVGCircleElement)
      ) {
        throw new Error("Authorized lifecycle geometry is unavailable.");
      }
      const ringMatrix = node.getScreenCTM();
      const bodyMatrix = body.getScreenCTM();
      if (ringMatrix === null || bodyMatrix === null) {
        throw new Error("Authorized lifecycle transform is unavailable.");
      }
      const center = new DOMPoint(0, 0).matrixTransform(ringMatrix);
      const bodyCenter = new DOMPoint(
        body.cx.baseVal.value,
        body.cy.baseVal.value,
      ).matrixTransform(bodyMatrix);
      const centerElements = document.elementsFromPoint(bodyCenter.x, bodyCenter.y);
      effect.focus();
      return {
        eventType: event.eventType,
        lifecycle: node.getAttribute("data-lifecycle-ring"),
        color: getComputedStyle(node).color,
        radius: ring.getAttribute("r"),
        strokeOpacity: getComputedStyle(ring).strokeOpacity,
        center: { x: center.x, y: center.y },
        bodyCenter: { x: bodyCenter.x, y: bodyCenter.y },
        planAnchor: event.anchor,
        planLayoutFields: [
          "cue",
          "cueBounds",
          "cueLeader",
          "cueDisposition",
          "cueCollisionFree",
          "cueLayoutKey",
        ].filter((field) => Object.hasOwn(event, field)),
        domLayoutAttributes: Array.from(node.attributes, ({ name }) => name).filter(
          (name) => name.startsWith("data-layout-"),
        ),
        semanticLeaderCount: effect.querySelectorAll(".combat-cue__leader--semantic")
          .length,
        hitOwnsBodyCenter: centerElements.includes(hit),
        agentOwnsBodyCenter: centerElements.some(
          (element) => element.closest(".agent") === agent,
        ),
        hitPointerEvents: getComputedStyle(hit).pointerEvents,
        hitFill: getComputedStyle(hit).fill,
        ariaLabel: effect.getAttribute("aria-label"),
        tooltipTitle: title.textContent,
        tooltipSummary:
          tooltip.querySelector(".semantic-explanation__summary")?.textContent ?? null,
      };
    });
    controller.destroy();
    const beforeSettle = {
      eventCount: normal.eventNodes.size,
      nodeCount: normal.nodeCount,
      persistentNodeCount: normal.persistentNodeCount,
      persistentNodeBound: plan.bounds.persistentNodes,
      lifecycleGeometry,
      lifecycleAnimations,
      waveGeometry,
      viewportBounds: surface.viewportBounds,
      shieldCount: battlefield.querySelectorAll(
        '.combat-effect[data-event-type="spawn_shield_expired"]',
      ).length,
      eventOwnerCount: battlefield.querySelectorAll(
        ".combat-effect[data-tooltip-owner]",
      ).length,
      childOwnerCount: battlefield.querySelectorAll(
        ".combat-lifecycle-ring [data-tooltip-owner], .combat-respawn-wave [data-tooltip-owner]",
      ).length,
    };
    painter.settle(normal);
    const afterNormalSettle = {
      eventCount: normal.eventNodes.size,
      eventTypes: Array.from(
        battlefield.querySelectorAll(".combat-effect[data-event-type]"),
      ).map((node) => node.getAttribute("data-event-type")),
      settledCount: battlefield.querySelectorAll('.combat-effect[data-settled="true"]')
        .length,
    };
    painter.clear(normal);

    /** @type {Record<string, any>} */
    const endpointModes = {};
    /** @type {Array<[string, {motionMode: "normal" | "reduced" | "off", settled: boolean, persistentOnly: boolean}]>} */
    const modeCases = [
      ["reduced", { motionMode: "reduced", settled: false, persistentOnly: false }],
      ["off", { motionMode: "off", settled: false, persistentOnly: false }],
      ["settled", { motionMode: "normal", settled: true, persistentOnly: false }],
    ];
    for (const [name, options] of modeCases) {
      const installation = painter.install(plan, surface, options);
      endpointModes[name] = {
        ringRadii: Array.from(
          battlefield.querySelectorAll(".combat-lifecycle-ring__ring"),
        ).map((ring) => ring.getAttribute("r")),
        lifecycleAnimationCount: installation.animationSpecs.filter(
          (/** @type {Record<string, any>} */ spec) =>
            spec.id.endsWith(":lifecycle-ring"),
        ).length,
        eventTypes: Array.from(
          battlefield.querySelectorAll(".combat-effect[data-event-type]"),
        ).map((node) => node.getAttribute("data-event-type")),
      };
      painter.clear(installation);
    }
    return {
      serializedUnchanged: JSON.stringify(presentation) === serializedBefore,
      planEventIds: plan.events.map(
        (/** @type {Record<string, any>} */ event) => event.eventId,
      ),
      beforeSettle,
      afterNormalSettle,
      endpointModes,
      remainingEffectCount: battlefield.querySelectorAll(".combat-effect").length,
    };
  }, raw);

  expect(result.serializedUnchanged).toBe(true);
  expect(result.planEventIds).toEqual(raw.latest_events.ordered_event_ids);
  expect(result.beforeSettle.eventCount).toBe(5);
  expect(result.beforeSettle.nodeCount).toBeLessThanOrEqual(
    raw.latest_events.event_count * 30 + 3,
  );
  expect(result.beforeSettle.persistentNodeCount).toBeLessThanOrEqual(
    result.beforeSettle.persistentNodeBound,
  );
  expect(result.beforeSettle.lifecycleGeometry).toEqual([
    expect.objectContaining({
      eventType: "agent_died",
      lifecycle: "death",
      color: "rgb(251, 113, 133)",
      radius: "32",
      strokeOpacity: "0.5",
      planLayoutFields: [],
      domLayoutAttributes: [],
      semanticLeaderCount: 0,
      hitOwnsBodyCenter: false,
      agentOwnsBodyCenter: true,
      hitPointerEvents: "stroke",
      hitFill: "none",
      ariaLabel: "Agent died",
      tooltipTitle: "Agent died",
      tooltipSummary: "This agent died on the incoming transition.",
    }),
    expect.objectContaining({
      eventType: "agent_respawned",
      lifecycle: "resurrection",
      color: "rgb(255, 255, 255)",
      radius: "32",
      strokeOpacity: "0.5",
      planLayoutFields: [],
      domLayoutAttributes: [],
      semanticLeaderCount: 0,
      hitOwnsBodyCenter: false,
      agentOwnsBodyCenter: true,
      hitPointerEvents: "stroke",
      hitFill: "none",
      ariaLabel: "Agent respawned",
      tooltipTitle: "Agent respawned",
      tooltipSummary: "This agent respawned on the incoming transition.",
    }),
  ]);
  for (const lifecycle of result.beforeSettle.lifecycleGeometry) {
    expect(Math.abs(lifecycle.center.x - lifecycle.bodyCenter.x)).toBeLessThanOrEqual(
      0.001,
    );
    expect(Math.abs(lifecycle.center.y - lifecycle.bodyCenter.y)).toBeLessThanOrEqual(
      0.001,
    );
    const copy = [
      lifecycle.ariaLabel,
      lifecycle.tooltipTitle,
      lifecycle.tooltipSummary,
    ].join(" ");
    expect(copy).not.toMatch(/_/u);
    expect(copy).not.toMatch(/semantic pulse/iu);
  }
  expect(result.beforeSettle.lifecycleAnimations).toHaveLength(2);
  for (const keyframes of result.beforeSettle.lifecycleAnimations) {
    expect(
      keyframes.map((/** @type {{radius: string}} */ frame) =>
        Number.parseFloat(frame.radius),
      ),
    ).toEqual([9, 17, 25, 32]);
    expect(keyframes.at(-1)?.opacity).toBe(1);
  }
  expect(result.beforeSettle.waveGeometry).toEqual([
    expect.objectContaining({
      teamIndex: 0,
      teamId: 1,
      side: "left",
      label: "EVENT: Team A Respawn",
      color: "rgb(59, 130, 246)",
      ownerCount: 0,
      layoutDisposition: "perimeter_callout",
      layoutCollisionFree: "true",
      ariaLabel: "EVENT: Team A Respawn",
    }),
    expect.objectContaining({
      teamIndex: 1,
      teamId: 2,
      side: "right",
      label: "EVENT: Team B Respawn",
      color: "rgb(240, 90, 103)",
      ownerCount: 0,
      layoutDisposition: "perimeter_callout",
      layoutCollisionFree: "true",
      ariaLabel: "EVENT: Team B Respawn",
    }),
  ]);
  const [teamA, teamB] = result.beforeSettle.waveGeometry;
  expect(teamA.layoutKey).toBe(
    JSON.stringify(["event", raw.latest_events.events[2].event_id, "cue"]),
  );
  expect(teamB.layoutKey).toBe(
    JSON.stringify(["event", raw.latest_events.events[3].event_id, "cue"]),
  );
  for (const wave of [teamA, teamB]) {
    expect([wave.label, wave.ariaLabel].join(" ")).not.toMatch(/_|semantic pulse/iu);
  }
  const bounds = result.beforeSettle.viewportBounds;
  expect(teamA.bounds.left).toBeGreaterThanOrEqual(bounds.left);
  expect(teamA.bounds.top).toBeGreaterThanOrEqual(bounds.top);
  expect(teamB.bounds.right).toBeLessThanOrEqual(bounds.right);
  expect(teamB.bounds.bottom).toBeLessThanOrEqual(bounds.bottom);
  expect(teamA.bounds.right).toBeLessThan(teamB.bounds.left);
  expect(result.beforeSettle.shieldCount).toBe(1);
  expect(result.beforeSettle.eventOwnerCount).toBe(5);
  expect(result.beforeSettle.childOwnerCount).toBe(0);
  expect(result.afterNormalSettle.eventCount).toBe(4);
  expect(result.afterNormalSettle.eventTypes).toEqual([
    "agent_died",
    "respawn_wave_occurred",
    "respawn_wave_occurred",
    "agent_respawned",
  ]);
  expect(result.afterNormalSettle.settledCount).toBe(4);
  for (const mode of ["reduced", "off", "settled"]) {
    expect(result.endpointModes[mode].ringRadii).toEqual(["32", "32"]);
    expect(result.endpointModes[mode].lifecycleAnimationCount).toBe(0);
  }
  expect(result.endpointModes.settled.eventTypes).toEqual([
    "agent_died",
    "respawn_wave_occurred",
    "respawn_wave_occurred",
    "agent_respawned",
  ]);
  expect(result.remainingEffectCount).toBe(0);
});

test("authorized multi-application status paints one route-free lifecycle", async ({
  page,
}) => {
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const template = raw.latest_events.events.find(
    (/** @type {Record<string, any>} */ event) => event.event_kind === "status_applied",
  );
  if (!template) {
    throw new Error("Authorized status application fixture is unavailable.");
  }
  const sources = [
    raw.latest_events.agent_phase_trajectories[0].successor,
    raw.latest_events.agent_phase_trajectories[2].successor,
  ];
  const applicationIds = sources.map(
    (/** @type {Record<string, any>} */ _source, ordinal) =>
      `${raw.latest_events.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
  );
  raw.latest_events.events = sources.map(
    (/** @type {Record<string, any>} */ sourceAnchor, ordinal) => ({
      ...structuredClone(template),
      event_id: applicationIds[ordinal],
      ordinal,
      source_anchor: structuredClone(sourceAnchor),
    }),
  );
  raw.latest_events.event_count = raw.latest_events.events.length;
  raw.latest_events.ordered_event_ids = [...applicationIds];
  raw.latest_events.ordered_event_kinds = raw.latest_events.events.map(
    (/** @type {Record<string, any>} */ event) => event.event_kind,
  );

  await page.goto(origin);
  const result = await page.evaluate(async (rawPresentation) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const { createTooltipController } = await import(`${moduleRoot}/tooltip.js`);
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    const tooltip = document.querySelector("#visual-tooltip");
    const title = document.querySelector("#visual-tooltip-title");
    const details = document.querySelector("#visual-tooltip-details");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement) ||
      !(tooltip instanceof HTMLElement) ||
      !(title instanceof HTMLElement) ||
      !(details instanceof HTMLElement)
    ) {
      throw new Error("Authorized status component surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, { showRanges: true });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface);
    if (surface === null || plan === null) {
      throw new Error("Authorized status choreography plan is unavailable.");
    }
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });
    const controller = createTooltipController({
      root: document.body,
      tooltip,
      title,
      details,
    });
    const effect = battlefield.querySelector(
      ".combat-effect--status-lifecycle[data-tooltip-owner]",
    );
    if (!(effect instanceof SVGElement)) {
      throw new Error("Authorized status lifecycle owner is unavailable.");
    }
    const hit = effect.querySelector(".combat-lifecycle__hit");
    if (!(hit instanceof SVGElement)) {
      throw new Error("Authorized status lifecycle hit target is unavailable.");
    }
    const hitBounds = hit.getBoundingClientRect();
    hit.dispatchEvent(
      new PointerEvent("pointermove", {
        bubbles: true,
        composed: true,
        clientX: hitBounds.left + hitBounds.width / 2,
        clientY: hitBounds.top + hitBounds.height / 2,
        pointerType: "mouse",
      }),
    );
    await new Promise((resolveFrame) => requestAnimationFrame(resolveFrame));
    const labels = Array.from(
      details.querySelectorAll(".semantic-explanation__label"),
      (node) => node.textContent,
    );
    const values = Array.from(
      details.querySelectorAll(".semantic-explanation__value"),
      (node) => node.textContent,
    );
    const planned = plan.events[0];
    const response = {
      planEventCount: plan.events.length,
      effectCount: battlefield.querySelectorAll(".combat-effect--status-lifecycle")
        .length,
      lifecycleCount: battlefield.querySelectorAll(".combat-lifecycle").length,
      routeChildCount: battlefield.querySelectorAll(".combat-choreography-routes > *")
        .length,
      statusUnderlayCount: battlefield.querySelectorAll(
        ".combat-route-effect--status-lifecycle",
      ).length,
      plannedAtomicIds: planned?.atomicEventIds ?? null,
      plannedApplicationIds: planned?.applicationEventIds ?? null,
      sourcePublicAgentIds:
        planned?.applicationSources?.map(
          (/** @type {Record<string, any>} */ source) => source.sourcePublicAgentId,
        ) ?? null,
      sourceRecordKeys:
        planned?.applicationSources?.map((/** @type {Record<string, any>} */ source) =>
          Object.keys(source).sort(),
        ) ?? null,
      domAtomicIds: JSON.parse(effect.getAttribute("data-atomic-event-ids") ?? "null"),
      domApplicationIds: JSON.parse(
        effect.getAttribute("data-application-event-ids") ?? "null",
      ),
      tooltipRows: labels.map((label, index) => ({
        label,
        value: values[index],
      })),
    };
    controller.destroy();
    return response;
  }, raw);

  expect(result.planEventCount).toBe(1);
  expect(result.effectCount).toBe(1);
  expect(result.lifecycleCount).toBe(1);
  expect(result.routeChildCount).toBe(0);
  expect(result.statusUnderlayCount).toBe(0);
  expect(result.plannedAtomicIds).toEqual(applicationIds);
  expect(result.plannedApplicationIds).toEqual(applicationIds);
  expect(result.domAtomicIds).toEqual(applicationIds);
  expect(result.domApplicationIds).toEqual(applicationIds);
  expect(result.sourcePublicAgentIds).toEqual(["agent-slot-0", "agent-slot-2"]);
  expect(result.sourceRecordKeys).toEqual([
    ["eventId", "sourcePresentationKey", "sourcePublicAgentId"],
    ["eventId", "sourcePresentationKey", "sourcePublicAgentId"],
  ]);
  expect(result.tooltipRows).toContainEqual({
    label: "Application Sources",
    value: "Agent ID agent-slot-0; Agent ID agent-slot-2",
  });
});

test("agent wins real SVG hit arbitration over an overlapping accepted route", async ({
  page,
}) => {
  await page.goto(origin);
  const overlap = await page.evaluate(async (raw) => {
    const moduleRoot = "/src";
    const { normalizeAuthorizedPresentationFrameV1 } = await import(
      `${moduleRoot}/authorized-presentation-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const { buildChoreographyPlan } = await import(
      `${moduleRoot}/choreography-plan.js`
    );
    const { SvgChoreographyPainter } = await import(
      `${moduleRoot}/choreography-painter.js`
    );
    const { createRouteGeometry } = await import(`${moduleRoot}/routes.js`);
    const { createSemanticDescriptor, createTooltipController, registerTooltipOwner } =
      await import(`${moduleRoot}/tooltip.js`);
    const presentation = await normalizeAuthorizedPresentationFrameV1(raw);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    const tooltip = document.querySelector("#visual-tooltip");
    const title = document.querySelector("#visual-tooltip-title");
    const details = document.querySelector("#visual-tooltip-details");
    if (
      !(battlefield instanceof SVGSVGElement) ||
      !(empty instanceof HTMLElement) ||
      !(tooltip instanceof HTMLElement) ||
      !(title instanceof HTMLElement) ||
      !(details instanceof HTMLElement)
    ) {
      throw new Error("Renderer tooltip test surface is unavailable.");
    }
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    renderer.render(presentation, { showRanges: true });
    const surface = renderer.choreographySurface();
    const plan = buildChoreographyPlan(presentation, surface);
    if (surface === null || plan === null) {
      throw new Error("Authorized choreography surface or plan is unavailable.");
    }
    new SvgChoreographyPainter().install(plan, surface, {
      motionMode: "off",
      settled: false,
      persistentOnly: false,
    });

    const routeHit = battlefield.querySelector(".combat-route__hit");
    const routeOwner = routeHit?.closest(".combat-route-effect");
    const targetKey = routeOwner?.getAttribute("data-target-presentation-key");
    const agent =
      typeof targetKey === "string" && targetKey
        ? battlefield.querySelector(
            `.agent[data-presentation-key="${CSS.escape(targetKey)}"]`,
          )
        : battlefield.querySelector(".agent");
    const body = agent?.querySelector(".agent-body");
    const battlefieldMatrix = battlefield.getScreenCTM();
    const bodyMatrix = body instanceof SVGGraphicsElement ? body.getScreenCTM() : null;
    if (
      !(routeHit instanceof SVGPathElement) ||
      !(routeOwner instanceof SVGElement) ||
      !(agent instanceof SVGElement) ||
      !(body instanceof SVGGraphicsElement) ||
      battlefieldMatrix === null ||
      bodyMatrix === null
    ) {
      throw new Error("Authorized route and agent geometry are unavailable.");
    }
    const bodyBounds = body.getBBox();
    const screenCenter = new DOMPoint(
      bodyBounds.x + bodyBounds.width / 2,
      bodyBounds.y + bodyBounds.height / 2,
    ).matrixTransform(bodyMatrix);
    const localCenter = screenCenter.matrixTransform(battlefieldMatrix.inverse());
    const geometry = createRouteGeometry(
      {
        eventId: "component-route-agent-overlap",
        source: { x: localCenter.x - 80, y: localCenter.y },
        target: { x: localCenter.x + 80, y: localCenter.y },
        sourceRadius: 0,
        targetRadius: 0,
        offset: 0,
      },
      { endpointGap: 0 },
    );
    routeHit.setAttribute("d", geometry.path);
    routeOwner.setAttribute("opacity", "1");
    registerTooltipOwner(
      routeOwner,
      createSemanticDescriptor({
        kind: "accepted-route",
        id: "component-route-agent-overlap",
        title: "Accepted Route",
        tone: "information",
        accent: "none",
        summary: "Accepted route component overlap probe.",
        rows: [],
        sections: [],
        metadata: { compact: true, full: false },
        anchor: "pointer",
      }),
    );
    const controller = createTooltipController({
      root: document.body,
      tooltip,
      title,
      details,
    });
    // Retain the controller for the pointer assertion and clean it up on navigation.
    window.addEventListener("pagehide", () => controller.destroy(), { once: true });

    const owners = [
      ...new Set(
        document
          .elementsFromPoint(screenCenter.x, screenCenter.y)
          .map((element) => element.closest("[data-tooltip-owner]"))
          .filter((element) => element instanceof Element),
      ),
    ];
    const routeRoot = routeOwner.parentElement;
    const routeLayer = routeRoot?.parentElement;
    return {
      agentPaintOrder: owners.indexOf(agent),
      routePaintOrder: owners.indexOf(routeOwner),
      activationRoute: routeOwner.classList.contains("combat-route-effect--activation"),
      routeRootOwned:
        routeRoot?.classList.contains("combat-choreography-routes") === true,
      routeLayer: routeLayer?.getAttribute("data-layer") ?? null,
      layerOrder: [...battlefield.children].map((layer) =>
        layer.getAttribute("data-layer"),
      ),
      x: screenCenter.x,
      y: screenCenter.y,
    };
  }, fixture.presentations.replay_oracle);

  expect(overlap.agentPaintOrder).toBeGreaterThanOrEqual(0);
  expect(overlap.routePaintOrder).toBeGreaterThanOrEqual(0);
  expect(overlap.agentPaintOrder).toBeLessThan(overlap.routePaintOrder);
  expect(overlap.activationRoute).toBe(true);
  expect(overlap.routeRootOwned).toBe(true);
  expect(overlap.routeLayer).toBe("transient-route");
  const routeLayerIndex = overlap.layerOrder.indexOf("transient-route");
  expect(routeLayerIndex).toBeGreaterThanOrEqual(0);
  for (const foregroundLayer of [
    "obstacle",
    "body",
    "selection-legality",
    "transient-events",
    "durable-status-modifier",
    "accessible-labels",
  ]) {
    expect(overlap.layerOrder.indexOf(foregroundLayer)).toBeGreaterThan(
      routeLayerIndex,
    );
  }
  await page.mouse.move(overlap.x, overlap.y);
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "agent",
  );
});
