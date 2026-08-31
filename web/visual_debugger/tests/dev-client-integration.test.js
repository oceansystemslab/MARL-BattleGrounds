import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  createCombatConfigurationController,
  debugScenarioOptionLabel,
  draftAfterAuthoringResponse,
  openableDraftAssets,
} from "../src/dev-client.js";

const devClientUrl = new URL("../src/dev-client.js", import.meta.url);
const authoringRendererUrl = new URL("../src/authoring-renderer.js", import.meta.url);
const mainUrl = new URL("../src/main.js", import.meta.url);
const indexUrl = new URL("../index.html", import.meta.url);

test("Replay startup cannot load the DevClient authoring module", async () => {
  const [markup, main] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(mainUrl, "utf8"),
  ]);

  assert.doesNotMatch(markup, /<script[^>]+src="\/src\/dev-client\.js"/u);
  assert.match(
    main,
    /const startupIdentity = applyProductIdentity\([\s\S]*startupIdentity\.product_kind === "combat_debugger"[\s\S]*startupIdentity\.authoring_available[\s\S]*import\("\.\/dev-client\.js"\)/u,
  );
});

test("general Open lists mutable saved drafts and excludes frozen candidates", () => {
  const savedMap = {
    source_kind: "saved_draft",
    asset_kind: "map",
    asset_id: "arena",
  };
  const assets = [
    savedMap,
    { source_kind: "candidate", asset_kind: "map", candidate_id: "map-digest" },
    {
      source_kind: "saved_draft",
      asset_kind: "scenario",
      asset_id: "crossfire",
    },
    {
      source_kind: "candidate",
      asset_kind: "scenario",
      candidate_id: "scenario-digest",
    },
  ];

  assert.deepEqual(openableDraftAssets(assets, "map"), [savedMap]);
  assert.deepEqual(
    openableDraftAssets(assets, "scenario").map((asset) => asset.asset_id),
    ["crossfire"],
  );
});

test("Combat scenario options expose exact persisted identity and status", () => {
  assert.equal(
    debugScenarioOptionLabel({
      source_kind: "saved_draft",
      name: "Crossfire",
      asset_id: "crossfire",
      revision: 7,
      map_width: 20.5,
      map_height: 10.25,
    }),
    "Crossfire · saved crossfire revision 7 · 20.5 × 10.25 · execution-valid",
  );
  const candidateId = "a".repeat(64);
  assert.equal(
    debugScenarioOptionLabel({
      source_kind: "candidate",
      name: "Crossfire",
      candidate_id: candidateId,
      map_width: 20,
      map_height: 10,
    }),
    `Crossfire · candidate ${candidateId} · 20 × 10 · frozen · execution-valid`,
  );
});

test("new scenarios use one native source selector instead of wire-literal prompts", async () => {
  const [markup, devClient] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(devClientUrl, "utf8"),
  ]);

  assert.match(markup, /id="authoring-new-scenario-mode"/u);
  assert.match(markup, />Blank scenario</u);
  assert.match(markup, />Copy saved map</u);
  assert.match(markup, />Duplicate saved scenario</u);
  assert.match(
    devClient,
    /elements\.newScenarioChoice\.hidden = area !== "scenarios"/u,
  );
  assert.match(devClient, /const creationMode = elements\.newScenarioMode\.value/u);
  assert.doesNotMatch(devClient, /Creation mode:/u);
});

test("a deferred validation response cannot replace newer local edits", () => {
  const submitted = {
    schema: "dev-map-draft@1",
    asset_id: "arena",
    revision: 2,
    content: { name: "Submitted" },
  };
  const current = {
    ...structuredClone(submitted),
    content: { name: "Edited while validation was pending" },
  };

  const retained = draftAfterAuthoringResponse(current, {
    command_type: "validate",
    draft: submitted,
  });

  assert.strictEqual(retained, current);
  assert.equal(retained.content.name, "Edited while validation was pending");
});

test("authoring requests make navigation and mutation surfaces inert", async () => {
  const devClient = await readFile(devClientUrl, "utf8");

  assert.match(devClient, /elements\.shell\.inert = state\.busy/u);
  assert.match(
    devClient,
    /async function selectArea\(area\) \{\s*if \(state\.busy\) \{\s*return;/u,
  );
  assert.match(
    devClient,
    /function commit\(next, before = state\.draft\) \{\s*if \(state\.busy/u,
  );
  assert.match(
    devClient,
    /"drop"[\s\S]*moveAuthoringObjectWithSnap\([\s\S]*fixed_snap_world_units/u,
  );
});

test("the SVG grid uses one repeating pattern instead of dimension-sized nodes", async () => {
  const renderer = await readFile(authoringRendererUrl, "utf8");

  assert.match(renderer, /svgElement\("pattern"/u);
  assert.match(renderer, /fill: "url\(#authoring-grid-pattern\)"/u);
  assert.doesNotMatch(renderer, /for \(let [xy] = gridSpacing/u);
  assert.match(renderer, /Map dimensions must be positive finite numbers\./u);
});

test("Combat configuration remains authoritative until one requested successor installs", () => {
  const teamBController = { value: "manual", disabled: false };
  const informationMode = { value: "shared_obs", disabled: false };
  /** @type {Record<string, string | undefined>} */
  const dataset = {};
  const root = { dataset };
  /** @type {Readonly<Record<string, string>>[]} */
  const emitted = [];
  const controller = createCombatConfigurationController({
    teamBController,
    informationMode,
    root,
    emit: (configuration) => emitted.push(configuration),
  });

  controller.render();
  assert.equal(teamBController.disabled, true);
  assert.equal(informationMode.disabled, true);

  assert.equal(
    controller.install({
      team_b_controller: "manual",
      execution_information_mode: "shared_obs",
    }),
    true,
  );
  teamBController.value = "scripted_tdm";
  informationMode.value = "no_shared_obs";
  assert.equal(controller.request(), true);
  assert.deepEqual(emitted, [
    {
      team_b_controller: "scripted_tdm",
      execution_information_mode: "no_shared_obs",
    },
  ]);
  assert.equal(teamBController.value, "manual");
  assert.equal(informationMode.value, "shared_obs");

  controller.install(emitted[0]);
  assert.equal(controller.request(), false);
  assert.equal(emitted.length, 1);
  assert.equal(teamBController.value, "scripted_tdm");
  assert.equal(informationMode.value, "no_shared_obs");
  assert.equal(root.dataset.teamBController, "scripted_tdm");
  assert.equal(root.dataset.executionInformationMode, "no_shared_obs");
});

test("successful Debug loads emit one event into the existing frame reload path", async () => {
  const [devClient, main] = await Promise.all([
    readFile(devClientUrl, "utf8"),
    readFile(mainUrl, "utf8"),
  ]);

  assert.equal(
    [
      ...devClient.matchAll(
        /new CustomEvent\("marl-devclient-debug-session-replaced"\)/gu,
      ),
    ].length,
    1,
  );
  assert.match(
    devClient,
    /async function openInDebug\([\s\S]*await send\(\{ command_type: "open_in_debug", source \}\);[\s\S]*if \(!response\?\.ok\) \{[\s\S]*return;[\s\S]*\}[\s\S]*notifyDebugSessionReplaced\(\);/u,
  );
  assert.match(
    devClient,
    /elements\.scenarioLoad\.addEventListener\("click", \(\) => \{[\s\S]*void openInDebug\(JSON\.parse\(elements\.scenarioSelect\.value\)\);/u,
  );
  assert.match(
    devClient,
    /elements\.openDebugButton\.addEventListener\("click", async \(\) => \{[\s\S]*await openInDebug\([\s\S]*source_kind: "current_buffer"/u,
  );
  assert.match(
    main,
    /if \(!isReplayMode\(\)\) \{[\s\S]*document\.addEventListener\("marl-devclient-debug-session-replaced", \(\) => \{\s*void loadCurrentFrame\(\);\s*\}\);[\s\S]*\}/u,
  );
  assert.match(
    main,
    /document\.addEventListener\("marl-devclient-combat-configuration", \(event\) => \{[\s\S]*requestedCombatConfigurationCommand\([\s\S]*event\.detail[\s\S]*void dispatchCommand\(command\);/u,
  );
  assert.match(
    main,
    /command_type: "set_combat_configuration",\s*team_b_controller: requested\.team_b_controller,\s*execution_information_mode: requested\.execution_information_mode/u,
  );
  assert.match(main, /publishInstalledCombatConfiguration\(joined\.transport\);/u);
  assert.match(main, /scriptedTeamBBlocksActionEdit\(command\)/u);
});
