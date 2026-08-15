import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

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
          <html><body>
            <svg id="battlefield" style="width: 800px; height: 600px"></svg>
            <p id="empty"></p>
          </body></html>`);
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

for (const kind of ["live_oracle", "replay_oracle"]) {
  test(`${kind} range preference changes visibility but not authority`, async ({
    page,
  }) => {
    const raw = fixture.presentations[kind];
    const hidden = await renderPresentation(page, raw, false);
    const visible = await renderPresentation(page, raw, true);
    expect(hidden.painted).toBe(true);
    expect(hidden.rangeCount).toBe(0);
    expect(visible.rangeCount).toBe(visible.authorizedRangeCount);
    expect(visible.rangeKeys).toEqual(visible.authorizedRangeKeys);
  });
}

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
