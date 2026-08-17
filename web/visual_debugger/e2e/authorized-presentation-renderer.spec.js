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
