import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  authoringPersistenceMessage,
  authoringSourceOptionLabel,
  createCombatConfigurationController,
  debugAssetOptionLabel,
  draftAfterAuthoringResponse,
  frozenAuthoringRecord,
  newScenarioSourceAssets,
  openableDraftAssets,
  persistedAuthoringSource,
  savedDraftOptionLabel,
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

test("DevClient surfaces are fail-closed and authoring hover stays unregistered", async () => {
  const [markup, main, devClient, renderer] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(mainUrl, "utf8"),
    readFile(devClientUrl, "utf8"),
    readFile(authoringRendererUrl, "utf8"),
  ]);

  for (const id of ["devclient-nav", "devclient-combat-config", "authoring-shell"]) {
    assert.match(
      markup,
      new RegExp(`<[^>]+id="${id}"[^>]+data-devclient-only[^>]*>`, "u"),
    );
    assert.doesNotMatch(
      markup,
      new RegExp(`<[^>]+id="${id}"[^>]+data-live-only[^>]*>`, "u"),
    );
  }
  assert.match(markup, /id="authoring-reset"[^>]*>Reset \(R\)<\/button>/u);
  assert.match(markup, /id="authoring-recenter"[^>]*>Recenter<\/button>/u);
  assert.doesNotMatch(main, /\[\s*"#authoring-canvas"/u);
  assert.doesNotMatch(renderer, /svgElement\("title"/u);
  assert.match(devClient, /resetButton:\s*required\("authoring-reset"\)/u);
  assert.match(devClient, /recenterButton:\s*required\("authoring-recenter"\)/u);
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

test("Combat asset options expose exact typed identity, status, and map preview semantics", () => {
  assert.equal(
    debugAssetOptionLabel({
      source_kind: "saved_draft",
      asset_kind: "scenario",
      name: "Crossfire",
      asset_id: "crossfire",
      revision: 7,
      map_width: 20.5,
      map_height: 10.25,
    }),
    "Scenario · Crossfire · saved crossfire revision 7 · 20.5 × 10.25 · execution-valid",
  );
  const candidateId = "a".repeat(64);
  assert.equal(
    debugAssetOptionLabel({
      source_kind: "candidate",
      asset_kind: "map",
      name: "Crossfire",
      candidate_id: candidateId,
      map_width: 20,
      map_height: 10,
    }),
    `Map preview · Crossfire · candidate ${candidateId} · 20 × 10 · frozen · execution-valid · default 5v5 TDM`,
  );
  assert.deepEqual(
    persistedAuthoringSource({
      source_kind: "saved_draft",
      asset_kind: "map",
      asset_id: "arena",
      revision: 3,
    }),
    {
      source_kind: "saved_draft",
      asset_kind: "map",
      asset_id: "arena",
      revision: 3,
    },
  );
});

test("active drafts and new scenarios use typed native asset selectors", async () => {
  const [markup, devClient] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(devClientUrl, "utf8"),
  ]);

  assert.match(markup, /id="authoring-new-scenario-mode"/u);
  assert.match(markup, /id="authoring-new-scenario-source"/u);
  assert.match(markup, /id="authoring-saved-draft-select"/u);
  assert.match(markup, />Blank scenario</u);
  assert.match(markup, />Copy saved map</u);
  assert.match(markup, />Duplicate saved scenario</u);
  assert.match(
    devClient,
    /elements\.newScenarioChoice\.hidden = area !== "scenarios"/u,
  );
  assert.match(devClient, /const creationMode = elements\.newScenarioMode\.value/u);
  assert.doesNotMatch(devClient, /Creation mode:/u);

  const savedMap = {
    source_kind: "saved_draft",
    asset_kind: "map",
    asset_id: "arena",
    revision: 2,
    name: "Arena",
    map_width: 20,
    map_height: 10,
    execution_valid: true,
  };
  const candidateMap = {
    source_kind: "candidate",
    asset_kind: "map",
    candidate_id: "a".repeat(64),
    name: "Frozen arena",
    map_width: 20,
    map_height: 10,
    execution_valid: true,
  };
  const savedScenario = {
    ...savedMap,
    asset_kind: "scenario",
    asset_id: "crossfire",
    name: "Crossfire",
  };
  const assets = [savedMap, candidateMap, savedScenario];
  assert.deepEqual(newScenarioSourceAssets(assets, "blank"), []);
  assert.deepEqual(newScenarioSourceAssets(assets, "copy_saved_map"), [
    savedMap,
    candidateMap,
  ]);
  assert.deepEqual(newScenarioSourceAssets(assets, "duplicate_saved_scenario"), [
    savedScenario,
  ]);
  assert.equal(savedDraftOptionLabel(savedMap), "Arena · arena · revision 2 · 20 × 10");
  assert.equal(
    authoringSourceOptionLabel(candidateMap),
    `Frozen arena · frozen candidate ${"a".repeat(64)} · 20 × 10 · execution-valid`,
  );
});

test("authoring persistence feedback distinguishes unsaved, saved, dirty, and frozen bytes", () => {
  const draft = {
    schema: "dev-map-draft@1",
    asset_id: "arena",
    revision: 0,
    content: { name: "Arena" },
  };
  assert.equal(
    authoringPersistenceMessage(draft, structuredClone(draft.content), null),
    "Unsaved map draft",
  );
  const saved = { ...draft, revision: 2 };
  const savedPath = "artifacts/dev_client/drafts/maps/arena/r2.json";
  assert.equal(
    authoringPersistenceMessage(saved, structuredClone(saved.content), null),
    `Saved map arena · revision 2 · ${savedPath}`,
  );
  const dirty = { ...saved, content: { name: "Changed" } };
  assert.equal(
    authoringPersistenceMessage(dirty, structuredClone(saved.content), null),
    "Unsaved changes · last saved map arena revision 2",
  );
  const candidateId = "b".repeat(64);
  const candidatePath = `artifacts/dev_client/candidates/map-${candidateId}.json`;
  assert.equal(
    authoringPersistenceMessage(saved, structuredClone(saved.content), {
      candidateId,
      content: structuredClone(saved.content),
    }),
    `Frozen candidate ${candidateId} · ${candidatePath}`,
  );
  assert.equal(
    authoringPersistenceMessage(dirty, structuredClone(saved.content), {
      candidateId,
      content: structuredClone(saved.content),
    }),
    `Unsaved changes · last saved map arena revision 2 · Frozen candidate ${candidateId} · ${candidatePath} · preserves an earlier snapshot; current edits are not frozen`,
  );

  const normalizedCandidateResponse = {
    ok: true,
    candidate: {
      candidate_id: candidateId,
      content: { name: "Arena", width: 20.100000381469727 },
    },
  };
  const submitted = { name: "Arena", width: 20.1 };
  const normalizedFreeze = frozenAuthoringRecord(
    normalizedCandidateResponse,
    submitted,
  );
  assert.deepEqual(normalizedFreeze?.content, submitted);
  assert.equal(
    authoringPersistenceMessage(
      { ...saved, content: submitted },
      structuredClone(submitted),
      normalizedFreeze,
    ),
    `Frozen candidate ${candidateId} · ${candidatePath}`,
  );
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
    /function commit\(next, before = state\.editor\.draft\) \{\s*if \(state\.busy/u,
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

  assert.match(
    main,
    /Freeze creates an immutable, content-addressed snapshot of this valid draft in `artifacts\/dev_client\/candidates\/`\. It does not save later edits, promote the asset, or mark it as canonical, training, or evaluation content\./u,
  );

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
