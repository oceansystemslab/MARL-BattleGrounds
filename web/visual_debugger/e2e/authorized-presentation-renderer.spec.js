import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

import { loadRendererFixture } from "./support/renderer-fixture.js";

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

test("compact required cooldown docks retain geometry and owner association", async ({
  page,
}) => {
  const rendererFixture = await loadRendererFixture("required_dock_fallback");
  await page.goto(origin);
  const result = await page.evaluate(async (rawFrame) => {
    const moduleRoot = "/src";
    const { normalizeLiveDebuggerFrameV2 } = await import(
      `${moduleRoot}/frame-normalizer.js`
    );
    const { BattlefieldRenderer } = await import(`${moduleRoot}/scene.js`);
    const battlefield = document.querySelector("#battlefield");
    const empty = document.querySelector("#empty");
    if (!(battlefield instanceof SVGSVGElement) || !(empty instanceof HTMLElement)) {
      throw new Error("Synthetic renderer component surface is unavailable.");
    }
    battlefield.style.width = "600px";
    battlefield.style.height = "420px";
    const renderer = new BattlefieldRenderer({ battlefield, empty });
    const painted = renderer.render(normalizeLiveDebuggerFrameV2(rawFrame), {
      showRanges: true,
    });
    await document.fonts.ready;
    await new Promise((resolveFrame) =>
      requestAnimationFrame(() => requestAnimationFrame(resolveFrame)),
    );
    const battlefieldBounds = battlefield.getBoundingClientRect();
    const layer = battlefield.querySelector('[data-layer="durable-status-modifier"]');
    const rows = [
      ...battlefield.querySelectorAll('.required-dock-fallback[data-kind="cooldown"]'),
    ].map((cell) => {
      const bounds = cell.getBoundingClientRect();
      const layoutKey = cell.getAttribute("data-layout-key") ?? "";
      const slot = Number(layoutKey.split(":")[1]);
      const owner = battlefield.querySelector(`.agent[data-slot="${slot}"]`);
      return {
        ariaLabel: cell.getAttribute("aria-label"),
        collisionFree: cell.parentElement?.getAttribute("data-collision-free"),
        inside:
          bounds.width > 0 &&
          bounds.height > 0 &&
          bounds.left >= battlefieldBounds.left &&
          bounds.top >= battlefieldBounds.top &&
          bounds.right <= battlefieldBounds.right &&
          bounds.bottom <= battlefieldBounds.bottom,
        layoutKey,
        ownerExists: owner instanceof SVGElement,
        ownerLabel: cell.getAttribute("data-owner-label"),
        text: cell.textContent,
        ticks: cell.getAttribute("data-ticks"),
      };
    });
    return {
      compacted: layer?.getAttribute("data-compacted-required-docks") ?? "",
      painted,
      rows,
    };
  }, rendererFixture.live_frame);

  expect(result.painted).toBe(true);
  expect(result.rows.length).toBeGreaterThan(0);
  expect(new Set(result.rows.map(({ layoutKey }) => layoutKey)).size).toBe(
    result.rows.length,
  );
  for (const row of result.rows) {
    const slot = row.layoutKey.split(":")[1];
    expect(result.compacted.split(",")).toContain(row.layoutKey);
    expect(row.collisionFree).toBe("true");
    expect(row.inside).toBe(true);
    expect(row.ownerExists).toBe(true);
    expect(row.ownerLabel).toBe(`Agent ID ${slot}`);
    expect(row.ticks).toBe("30");
    expect(row.text).toContain(`Agent ID ${slot}`);
    expect(row.text).toContain("U30");
    expect(row.ariaLabel).toMatch(/Cooldown Remaining.*30 Ticks/iu);
  }
});
