import assert from "node:assert/strict";
import test from "node:test";
import {
  authoringClassAuraMechanics,
  authoringClassMechanics,
  authoringClassStatusMechanics,
  authoringPathMatchesProblem,
  focusAuthoringProblemField,
} from "../src/authoring-inspector.js";
import {
  addAuthoringObstacle,
  authoringContentSnapshot,
  authoringObjects,
  deleteAuthoringObstacle,
  duplicateAuthoringObstacle,
  mapContent,
  moveAuthoringObject,
  moveAuthoringObjectWithSnap,
  reorderAuthoringObstacle,
  restoreAuthoringContent,
  setAgentAlive,
  setAuthoringField,
  setScenarioTeamSize,
  snapAuthoringCoordinate,
} from "../src/authoring-model.js";
import {
  authoringAgentBodyRadius,
  authoringAgentVisual,
  authoringClientPointToWorld,
  authoringGridPattern,
  authoringMapDimensions,
  authoringPaintObjects,
  zoomAuthoringCamera,
} from "../src/authoring-renderer.js";

function pads() {
  return ["A", "B"].flatMap((team) =>
    Array.from({ length: 5 }, (_, index) => ({
      object_id: `pad-${team.toLowerCase()}${index + 1}`,
      team,
      team_local_slot: index + 1,
      position: { x: team === "A" ? 1.5 : 18.5, y: 1.5 + index * 1.75 },
    })),
  );
}

function mapDraft() {
  return {
    schema: "dev-map-draft@1",
    asset_id: "map",
    revision: 0,
    content: {
      schema: "dev-map-content@1",
      name: "Map",
      description: "",
      width: 20,
      height: 10,
      obstacles: [],
      spawn_pads: pads(),
    },
  };
}

function scenarioDraft() {
  const roster = ["A", "B"].flatMap((team, teamIndex) =>
    Array.from({ length: 5 }, (_, index) => ({
      object_id: `agent-${team.toLowerCase()}${index + 1}`,
      team,
      team_local_slot: index + 1,
      global_slot: teamIndex * 5 + index,
      class_name: "mage",
      role: team === "A" ? "focal" : "adversarial_opponent",
    })),
  );
  const embeddedMap = /** @type {any} */ (mapDraft().content);
  embeddedMap.obstacles = [
    {
      kind: "wall",
      object_id: "wall-1",
      center_x: 8,
      center_y: 5,
      width: 2,
      height: 1,
      rotation_degrees: 0,
    },
  ];
  return {
    schema: "dev-scenario-draft@1",
    asset_id: "scenario",
    revision: 0,
    content: {
      embedded_map: embeddedMap,
      team_a_size: 1,
      team_b_size: 1,
      roster,
      agent_states: roster.map((row) => ({
        object_id: row.object_id,
        position: { x: row.team === "A" ? 2 : 18, y: 5 },
        alive: true,
        current_health: 80,
      })),
    },
  };
}

const AUTHORING_CATALOG = {
  maximum_obstacle_slots: 32,
  fixed_grid_world_units: 1,
  fixed_snap_world_units: 0.5,
  canonical_product_movement_scale: 1,
  class_mechanics: [
    {
      class_id: 1,
      class_name: "Mage",
      maximum_health: 80,
      body_radius: 0.5,
    },
    { class_id: 2, class_name: "Warrior", maximum_health: 200 },
    { class_id: 3, class_name: "Hunter", maximum_health: 100 },
    { class_id: 4, class_name: "Rogue", maximum_health: 100 },
    { class_id: 5, class_name: "Priest", maximum_health: 100 },
  ],
  status_channels: [
    { status_id: "mage-burst", source_class_id: 1 },
    { status_id: "warrior-charge", source_class_id: 2 },
  ],
  aura_mechanics: [
    { aura_id: "mage-damage", emitter_class_id: 1 },
    { aura_id: "warrior-mitigation", emitter_class_id: 2 },
  ],
};

test("linked problems focus the exact inspector field after object selection", () => {
  let focused = false;
  const unrelated = {
    dataset: { authoringPath: '["content","agent_states",1,"current_health"]' },
    focus() {},
  };
  const matching = {
    dataset: { authoringPath: '["content","agent_states",0,"current_health"]' },
    focus() {
      focused = true;
    },
  };
  const form = {
    querySelectorAll() {
      return [unrelated, matching];
    },
  };

  assert.equal(
    authoringPathMatchesProblem(
      '["content","embedded_map","obstacles",2,"width"]',
      "obstacles.2.width",
    ),
    true,
  );
  assert.equal(focusAuthoringProblemField(form, "agent_states.0.current_health"), true);
  assert.equal(focused, true);
});

test("obstacle edits preserve ordered fixed-slot semantics", () => {
  const first = addAuthoringObstacle(
    mapDraft(),
    "wall",
    AUTHORING_CATALOG.maximum_obstacle_slots,
  );
  const second = addAuthoringObstacle(
    first.draft,
    "pillar",
    AUTHORING_CATALOG.maximum_obstacle_slots,
  );
  const duplicate = duplicateAuthoringObstacle(
    second.draft,
    first.object_id,
    AUTHORING_CATALOG.maximum_obstacle_slots,
    AUTHORING_CATALOG.fixed_snap_world_units,
  );

  assert.deepEqual(
    mapContent(duplicate.draft).obstacles.map(
      (/** @type {any} */ obstacle) => obstacle.object_id,
    ),
    ["wall-1", "pillar-2", "wall-1-copy"],
  );
  const moved = reorderAuthoringObstacle(duplicate.draft, "wall-1-copy", -1);
  assert.deepEqual(
    mapContent(moved).obstacles.map(
      (/** @type {any} */ obstacle) => obstacle.object_id,
    ),
    ["wall-1", "wall-1-copy", "pillar-2"],
  );
  assert.deepEqual(
    mapContent(deleteAuthoringObstacle(moved, "wall-1")).obstacles.map(
      (/** @type {any} */ obstacle) => obstacle.object_id,
    ),
    ["wall-1-copy", "pillar-2"],
  );
});

test("drag snapping is explicit and exact entry remains available", () => {
  assert.equal(snapAuthoringCoordinate(3.24, 0.5), 3);
  assert.equal(snapAuthoringCoordinate(3.26, 0.5), 3.5);
  assert.equal(snapAuthoringCoordinate(3.26, 0.5, true), 3.26);

  const added = addAuthoringObstacle(
    mapDraft(),
    "pillar",
    AUTHORING_CATALOG.maximum_obstacle_slots,
  );
  const moved = moveAuthoringObject(added.draft, added.object_id, 3.125, 8.875);
  assert.deepEqual(
    [mapContent(moved).obstacles[0].center_x, mapContent(moved).obstacles[0].center_y],
    [3.125, 8.875],
  );
});

test("undo content snapshots retain the latest persisted draft identity", () => {
  const original = { ...mapDraft(), asset_id: "stable-map", revision: 3 };
  const snapshot = authoringContentSnapshot(original);
  const edited = setAuthoringField(original, ["content", "name"], "Edited map");
  const saved = { ...edited, revision: 4 };

  const restored = restoreAuthoringContent(saved, snapshot);

  assert.equal(restored.asset_id, "stable-map");
  assert.equal(restored.revision, 4);
  assert.equal(restored.content.name, "Map");
  assert.equal(Object.hasOwn(snapshot, "asset_id"), false);
  assert.equal(Object.hasOwn(snapshot, "revision"), false);
});

test("pointer and roster drop placement share one snapped move helper", () => {
  const draft = scenarioDraft();
  const moved = moveAuthoringObjectWithSnap(
    draft,
    "agent-a1",
    3.26,
    6.24,
    AUTHORING_CATALOG.fixed_snap_world_units,
  );

  assert.deepEqual(moved.content.agent_states[0].position, { x: 3.5, y: 6 });
  assert.deepEqual(draft.content.agent_states[0].position, { x: 2, y: 5 });
});

test("the authoring paint order keeps agents visible above map content", () => {
  const objects = authoringPaintObjects(scenarioDraft());
  const kinds = objects.map((object) => object.kind);

  assert.deepEqual(kinds.slice(0, 10), Array(10).fill("spawn_pad"));
  assert.deepEqual(kinds.slice(10, 11), ["wall"]);
  assert.deepEqual(kinds.slice(11), ["agent", "agent"]);
});

test("agent rendering resolves body radius from the mechanics catalog", () => {
  const agent = authoringPaintObjects(scenarioDraft()).find(
    (object) => object.kind === "agent",
  );

  assert.equal(authoringAgentBodyRadius(agent, AUTHORING_CATALOG), 0.5);
  assert.equal(authoringAgentBodyRadius(agent, null), 0.45);
});

test("agent rendering uses the shared class vocabulary and explicit life state", () => {
  const draft = scenarioDraft();
  draft.content.roster[0].class_name = "warrior";
  draft.content.agent_states[0].alive = false;
  const agent = authoringPaintObjects(draft).find(
    (object) => object.object_id === "agent-a1",
  );

  assert.deepEqual(authoringAgentVisual(agent), {
    className: "warrior",
    glyphKey: "class-warrior",
    alive: false,
  });
  assert.deepEqual(authoringAgentVisual({ roster: {}, state: {} }), {
    className: "unknown",
    glyphKey: "unknown",
    alive: false,
  });
});

test("the grid model stays constant-size for huge maps and rejects invalid dimensions", () => {
  assert.deepEqual(authoringGridPattern(1), {
    width: 1,
    height: 1,
    path: "M 1 0 H 0 V 1",
  });
  assert.deepEqual(authoringMapDimensions(1e300, 1e300), {
    width: 1e300,
    height: 1e300,
  });
  assert.equal(authoringMapDimensions(Number.NaN, 10), null);
  assert.equal(authoringMapDimensions(20, -1), null);
});

test("catalog class mechanics join by class instead of browser object identity", () => {
  assert.equal(authoringClassMechanics(AUTHORING_CATALOG, "mage")?.maximum_health, 80);
  assert.equal(authoringClassMechanics(AUTHORING_CATALOG, "paladin"), null);
  assert.deepEqual(
    authoringClassStatusMechanics(AUTHORING_CATALOG, 1).map(
      (/** @type {any} */ status) => status.status_id,
    ),
    ["mage-burst"],
  );
  assert.deepEqual(
    authoringClassAuraMechanics(AUTHORING_CATALOG, 2).map(
      (/** @type {any} */ aura) => aura.aura_id,
    ),
    ["warrior-mitigation"],
  );
});

test("typed study identities can be completed from a null draft field", () => {
  const draft = {
    ...mapDraft(),
    content: {
      ...mapDraft().content,
      study: { success_policy_identity: null },
    },
  };
  const withIdentifier = setAuthoringField(
    draft,
    ["content", "study", "success_policy_identity", "identifier"],
    "tdm-success",
  );
  const completed = setAuthoringField(
    withIdentifier,
    ["content", "study", "success_policy_identity", "version"],
    1,
  );
  assert.deepEqual(completed.content.study.success_policy_identity, {
    identifier: "tdm-success",
    version: 1,
  });
});

test("Alive to Dead is one explicit clearing transaction", () => {
  const roster = ["A", "B"].flatMap((team, teamIndex) =>
    Array.from({ length: 5 }, (_, index) => ({
      object_id: `agent-${team.toLowerCase()}${index + 1}`,
      team,
      team_local_slot: index + 1,
      global_slot: teamIndex * 5 + index,
      class_name: "mage",
      role:
        team === "A"
          ? index === 0
            ? "focal"
            : "cooperative_partner"
          : "adversarial_opponent",
    })),
  );
  const states = roster.map((slot) => ({
    object_id: slot.object_id,
    position: { x: 2, y: 3 },
    alive: true,
    current_health: 80,
    ultimate_cooldown_remaining: 7,
    spawn_shield_duration_remaining: 2,
    steps_until_out_of_combat: 4,
    warrior_charge_slow_duration: 3,
    hunter_basic_slow_duration: 0,
    rogue_poison_slow_duration: 0,
    warrior_charge_stun_duration: 0,
    hunter_trap_stun_duration: 0,
    rogue_poison_stun_duration: 0,
    rogue_poison_anti_heal_duration: 0,
    mage_burst_duration: 2,
    priest_blessing_of_freedom_duration: 0,
  }));
  const draft = {
    schema: "dev-scenario-draft@1",
    asset_id: "scenario",
    revision: 0,
    content: {
      embedded_map: mapDraft().content,
      team_a_size: 5,
      team_b_size: 5,
      roster,
      agent_states: states,
    },
  };

  const dead = setAgentAlive(draft, "agent-a1", false);
  const state = dead.content.agent_states[0];
  assert.deepEqual(state.position, { x: 2, y: 3 });
  assert.equal(state.ultimate_cooldown_remaining, 7);
  assert.equal(state.current_health, 0);
  assert.equal(state.spawn_shield_duration_remaining, 0);
  assert.equal(state.steps_until_out_of_combat, 0);
  assert.equal(state.warrior_charge_slow_duration, 0);
  assert.equal(state.mage_burst_duration, 0);
  assert.equal(
    authoringObjects(dead).filter((object) => object.kind === "agent").length,
    10,
  );
});

test("team size edits canonicalize inactive rows and restore one active prefix", () => {
  const roster = ["A", "B"].flatMap((team, teamIndex) =>
    Array.from({ length: 5 }, (_, index) => ({
      object_id: `agent-${team.toLowerCase()}${index + 1}`,
      team,
      team_local_slot: index + 1,
      global_slot: teamIndex * 5 + index,
      class_name: ["mage", "warrior", "hunter", "rogue", "priest"][index],
      role:
        team === "A"
          ? index === 0
            ? "focal"
            : "cooperative_partner"
          : "adversarial_opponent",
    })),
  );
  const draft = {
    schema: "dev-scenario-draft@1",
    asset_id: "roster",
    revision: 0,
    content: {
      embedded_map: mapDraft().content,
      team_a_size: 5,
      team_b_size: 5,
      roster,
      agent_states: roster.map((row, index) => ({
        object_id: row.object_id,
        position: { x: index + 1, y: 2 },
        alive: true,
        current_health: 50,
        ultimate_cooldown_remaining: 3,
      })),
    },
  };

  const reduced = setScenarioTeamSize(draft, "B", 2, AUTHORING_CATALOG);
  assert.equal(reduced.content.roster[7].class_name, "not_applicable");
  assert.equal(reduced.content.roster[7].role, "not_applicable");
  assert.deepEqual(reduced.content.agent_states[7].position, { x: 0, y: 0 });
  assert.equal(reduced.content.agent_states[7].ultimate_cooldown_remaining, 0);

  const restored = setScenarioTeamSize(reduced, "B", 3, AUTHORING_CATALOG);
  assert.equal(restored.content.roster[7].class_name, "hunter");
  assert.equal(restored.content.roster[7].role, "adversarial_opponent");
  assert.deepEqual(
    restored.content.agent_states[7].position,
    mapDraft().content.spawn_pads[7].position,
  );
  assert.equal(restored.content.agent_states[7].current_health, 100);
  assert.equal(restored.content.agent_states[7].alive, true);
});

test("decimal maps use an invertible fitted pointer projection", () => {
  const camera = { x: 0, y: 0, width: 20.5, height: 10.25 };
  const center = authoringClientPointToWorld(
    { left: 10, top: 20, width: 820, height: 410 },
    camera,
    10.25,
    420,
    225,
  );
  assert.deepEqual(center, { x: 10.25, y: 5.125 });

  const zoomed = zoomAuthoringCamera(camera, 20.5, 10.25, center, 0.5);
  assert.deepEqual(zoomed, { x: 5.125, y: 2.5625, width: 10.25, height: 5.125 });
});
