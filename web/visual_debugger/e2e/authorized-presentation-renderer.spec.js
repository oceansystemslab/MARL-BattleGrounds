import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

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
          filterId: "combat_status_icon",
          ownerSelector: ".agent-combat-state-icon",
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
  expect(result.rows).toHaveLength(12);
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
    if (row.filterId === "combat_status_icon") {
      expect(row.baselineAgentAria).toContain("combat status");
      expect(row.disabledAgentAria).not.toContain("combat status");
      expect(row.restoredAgentAria).toContain("combat status");
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
          const combatIcon = agent.querySelector(".agent-combat-state-icon");
          const body = agent.querySelector(".agent-body");
          if (
            !(classIcon instanceof SVGSVGElement) ||
            !(combatIcon instanceof SVGSVGElement) ||
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
            combatHidden: combatIcon.hasAttribute("hidden"),
            combatColor: getComputedStyle(combatIcon).color,
            combatGlyph: combatIcon.getAttribute("data-icon"),
            combatAriaHidden: combatIcon.getAttribute("aria-hidden"),
            combatRole: combatIcon.getAttribute("role"),
            combatOwnsTooltip: combatIcon.hasAttribute("data-tooltip-owner"),
            glyphCount: agent.querySelectorAll(".agent-combat-state-icon").length,
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
    expect(state.glyphCount).toBe(1);
    expect(state.combatGlyph).toBe("combat-in-progress");
    expect(state.combatColor).toBe("rgb(255, 255, 255)");
    expect(state.combatAriaHidden).toBe("true");
    expect(state.combatRole).toBeNull();
    expect(state.combatOwnsTooltip).toBe(false);
    expect(state.status).toBe("OOC");
    expect(state.countdown).toBe("0");
    expect(state.ariaLabel).toContain("combat status OOC");
    expect(state.combatHidden).toBe(true);
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
    const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
    const serializedBefore = JSON.stringify(presentation);
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
      throw new Error("Authorized lifecycle choreography is unavailable.");
    }
    const painter = new SvgChoreographyPainter();
    const normal = painter.install(plan, surface, {
      motionMode: "normal",
      settled: false,
      persistentOnly: false,
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
      };
    });
    const lifecycleColors = lifecycleNodes.map((node) => ({
      lifecycle: node.getAttribute("data-lifecycle-ring"),
      color: getComputedStyle(node).color,
      radius: node.querySelector(".combat-lifecycle-ring__ring")?.getAttribute("r"),
    }));
    const beforeSettle = {
      eventCount: normal.eventNodes.size,
      nodeCount: normal.nodeCount,
      persistentNodeCount: normal.persistentNodeCount,
      persistentNodeBound: plan.bounds.persistentNodes,
      lifecycleColors,
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
    raw.latest_events.event_count * 28 + 2,
  );
  expect(result.beforeSettle.persistentNodeCount).toBeLessThanOrEqual(
    result.beforeSettle.persistentNodeBound,
  );
  expect(result.beforeSettle.lifecycleColors).toEqual([
    { lifecycle: "death", color: "rgb(251, 113, 133)", radius: "32" },
    { lifecycle: "resurrection", color: "rgb(255, 255, 255)", radius: "32" },
  ]);
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
      label: "RESPAWNING · TEAM A",
      color: "rgb(59, 130, 246)",
      ownerCount: 0,
    }),
    expect.objectContaining({
      teamIndex: 1,
      teamId: 2,
      side: "right",
      label: "RESPAWNING · TEAM B",
      color: "rgb(240, 90, 103)",
      ownerCount: 0,
    }),
  ]);
  const [teamA, teamB] = result.beforeSettle.waveGeometry;
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
    return {
      agentPaintOrder: owners.indexOf(agent),
      routePaintOrder: owners.indexOf(routeOwner),
      x: screenCenter.x,
      y: screenCenter.y,
    };
  }, fixture.presentations.replay_oracle);

  expect(overlap.agentPaintOrder).toBeGreaterThanOrEqual(0);
  expect(overlap.routePaintOrder).toBeGreaterThanOrEqual(0);
  expect(overlap.agentPaintOrder).toBeLessThan(overlap.routePaintOrder);
  await page.mouse.move(overlap.x, overlap.y);
  await expect(page.locator("#visual-tooltip")).toHaveAttribute(
    "data-tooltip-kind",
    "agent",
  );
});
