/**
 * @file Check the filter registry/defaults, immutable updates and each filter's
 * allowed display effect: 20 filters, 11 on by default (Cooldown Effects and Red
 * Zone Floors included), the v3 paint key and the Red Zone floor paint part.
 */
import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyVisualPaintPart,
  DEFAULT_VISUAL_FILTER_STATE,
  disableAllVisualFilters,
  enableAllVisualFilters,
  isVisualFilterEnabled,
  isVisualPaintPartEnabled,
  reduceVisualFilterState,
  setVisualFilterEnabled,
  VISUAL_FILTER_IDS,
  VISUAL_FILTER_REGISTRY,
  VISUAL_PAINT_PART_REGISTRY,
  visualFilterPaintKey,
} from "../src/visual-filters.js";

const EXPECTED_FILTERS = Object.freeze([
  ["aura_fields", "Aura Fields"],
  ["aura_modifier_badges", "Aura Modifier Badges"],
  ["duration_status_badges", "Duration Status Badges"],
  ["spawn_shield", "Spawn Shield"],
  ["target_selection_visuals", "Target Selection Visuals"],
  ["basic_ability_effects", "Basic Ability Effects"],
  ["ultimate_ability_effects", "Ultimate Ability Effects"],
  ["regeneration_effects", "Regeneration Effects"],
  ["cooldown_effects", "Cooldown Effects"],
  ["status_application", "Status Application"],
  ["natural_status_expiry", "Natural Status Expiry"],
  ["freezing_trap_break", "Freezing Trap Break"],
  ["status_clear_on_death", "Status Clear on Death"],
  ["death_effects", "Death Effects"],
  ["respawn_wave", "Respawn Wave"],
  ["resurrection_effects", "Resurrection Effects"],
  ["spawn_shield_expiry", "Spawn-Shield Expiry"],
  ["scrolling_battle_text", "Scrolling Battle Text"],
  ["death_announcer", "Death Announcer"],
  ["red_zone_floors", "Red Zone Floors"],
]);

const INITIAL_FILTER_IDS = [
  "spawn_shield",
  "basic_ability_effects",
  "ultimate_ability_effects",
  "regeneration_effects",
  "cooldown_effects",
  "death_effects",
  "respawn_wave",
  "resurrection_effects",
  "scrolling_battle_text",
  "death_announcer",
  "red_zone_floors",
];
const ALL_ENABLED = enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE);

test("locked registry exposes 20 filters and the eleven initial choices", () => {
  assert.deepEqual(
    VISUAL_FILTER_REGISTRY.map(({ id, label }) => [id, label]),
    EXPECTED_FILTERS,
  );
  assert.deepEqual(
    VISUAL_FILTER_IDS,
    EXPECTED_FILTERS.map(([id]) => id),
  );
  assert.equal(new Set(VISUAL_FILTER_IDS).size, 20);
  assert.deepEqual(
    VISUAL_FILTER_REGISTRY.filter(({ defaultEnabled }) => defaultEnabled).map(
      ({ id }) => id,
    ),
    INITIAL_FILTER_IDS,
  );
  assert.equal(Object.isFrozen(VISUAL_FILTER_REGISTRY), true);
  assert.equal(VISUAL_FILTER_REGISTRY.every(Object.isFrozen), true);
});

test("default state is immutable, exact, and enabled through the public helper", () => {
  assert.equal(Object.isFrozen(DEFAULT_VISUAL_FILTER_STATE), true);
  assert.deepEqual(Object.keys(DEFAULT_VISUAL_FILTER_STATE), VISUAL_FILTER_IDS);
  for (const id of VISUAL_FILTER_IDS) {
    assert.equal(DEFAULT_VISUAL_FILTER_STATE[id], INITIAL_FILTER_IDS.includes(id));
    assert.equal(
      isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, id),
      INITIAL_FILTER_IDS.includes(id),
    );
  }
  assert.equal(Reflect.set(DEFAULT_VISUAL_FILTER_STATE, "aura_fields", false), false);
});

test("set and bulk actions return frozen states without mutating their input", () => {
  const before = JSON.stringify(DEFAULT_VISUAL_FILTER_STATE);
  const disabled = setVisualFilterEnabled(
    DEFAULT_VISUAL_FILTER_STATE,
    "spawn_shield",
    false,
  );
  assert.notEqual(disabled, DEFAULT_VISUAL_FILTER_STATE);
  assert.equal(Object.isFrozen(disabled), true);
  assert.equal(disabled.aura_fields, false);
  assert.equal(disabled.spawn_shield, false);
  assert.equal(JSON.stringify(DEFAULT_VISUAL_FILTER_STATE), before);
  assert.equal(setVisualFilterEnabled(disabled, "spawn_shield", false), disabled);
  assert.equal(enableAllVisualFilters(disabled), ALL_ENABLED);
  assert.equal(enableAllVisualFilters(ALL_ENABLED), ALL_ENABLED);
  assert.equal(enableAllVisualFilters(DEFAULT_VISUAL_FILTER_STATE), ALL_ENABLED);
  const allDisabled = disableAllVisualFilters(disabled);
  assert.equal(Object.isFrozen(allDisabled), true);
  assert.equal(
    VISUAL_FILTER_IDS.every((id) => allDisabled[id] === false),
    true,
  );
  assert.equal(disableAllVisualFilters(allDisabled), allDisabled);
});

test("strict reducer accepts only exact set and bulk actions", () => {
  const disabled = reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, {
    type: "set",
    filterId: "scrolling_battle_text",
    enabled: false,
  });
  assert.equal(disabled.scrolling_battle_text, false);
  assert.equal(reduceVisualFilterState(disabled, { type: "enable_all" }), ALL_ENABLED);
  const allDisabled = reduceVisualFilterState(disabled, { type: "disable_all" });
  assert.equal(
    VISUAL_FILTER_IDS.every((id) => allDisabled[id] === false),
    true,
  );
  for (const changed of [disabled, allDisabled, ALL_ENABLED]) {
    assert.equal(
      reduceVisualFilterState(changed, { type: "restore_defaults" }),
      DEFAULT_VISUAL_FILTER_STATE,
    );
  }
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.death_announcer, true);
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.cooldown_effects, true);
  assert.equal(DEFAULT_VISUAL_FILTER_STATE.red_zone_floors, true);
  assert.throws(
    () =>
      reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, {
        type: "set",
        filterId: "aura_fields",
        enabled: false,
        unexpected: true,
      }),
    /invalid shape/u,
  );
  assert.throws(
    () => reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, { type: "unknown" }),
    /Unknown visual filter action/u,
  );
});

test("state validation and paint-key serialization are strict and deterministic", () => {
  const disabled = ["aura_fields", "scrolling_battle_text"].reduce(
    (state, filterId) => setVisualFilterEnabled(state, filterId, false),
    DEFAULT_VISUAL_FILTER_STATE,
  );
  assert.equal(
    visualFilterPaintKey(DEFAULT_VISUAL_FILTER_STATE),
    "visual-filters-v3:00010111100001110111",
  );
  assert.equal(
    visualFilterPaintKey(disabled),
    "visual-filters-v3:00010111100001110011",
  );
  assert.equal(
    visualFilterPaintKey(Object.fromEntries([...Object.entries(disabled)].reverse())),
    visualFilterPaintKey(disabled),
  );
  assert.throws(
    () => visualFilterPaintKey({ ...DEFAULT_VISUAL_FILTER_STATE, extra: true }),
    /invalid shape/u,
  );
  assert.throws(
    () =>
      isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, "future_unregistered_filter"),
    /Unknown visual filter/u,
  );
});

test("every registered paint part has one exact owner and every filter owns a part", () => {
  const serializedParts = new Set();
  const usedFilters = new Set();
  for (const registration of VISUAL_PAINT_PART_REGISTRY) {
    assert.equal(Object.isFrozen(registration), true);
    assert.equal(Object.isFrozen(registration.tag), true);
    const serialized = JSON.stringify(
      Object.entries(registration.tag).sort(([left], [right]) =>
        left.localeCompare(right),
      ),
    );
    assert.equal(serializedParts.has(serialized), false, serialized);
    serializedParts.add(serialized);
    assert.equal(classifyVisualPaintPart(registration.tag), registration.filterId);
    assert.equal(
      isVisualPaintPartEnabled(DEFAULT_VISUAL_FILTER_STATE, registration.tag),
      INITIAL_FILTER_IDS.includes(registration.filterId),
    );
    usedFilters.add(registration.filterId);
  }
  assert.deepEqual([...usedFilters].sort(), [...VISUAL_FILTER_IDS].sort());
});

test("multipart effects retain coherent filter ownership", () => {
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "applied",
      part: "effect",
    }),
    "status_application",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "trap_broken",
      part: "effect",
    }),
    "freezing_trap_break",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "net_health",
      outcome: "unchanged",
      part: "battle_text",
    }),
    "scrolling_battle_text",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "activation",
      component: "basic",
      part: "semantic",
    }),
    "basic_ability_effects",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "activation",
      component: "ultimate",
      part: "semantic",
    }),
    "ultimate_ability_effects",
  );
  for (const outcome of ["damage", "healing", "unchanged"]) {
    assert.equal(
      classifyVisualPaintPart({
        surface: "transient",
        kind: "net_health",
        outcome,
        part: "effect",
      }),
      "scrolling_battle_text",
    );
  }
  for (const removedId of ["damage_effects", "healing_effects"]) {
    assert.throws(
      () => isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, removedId),
      /Unknown visual filter/u,
    );
  }
  assert.equal(
    VISUAL_FILTER_IDS.includes("damage_effects") ||
      VISUAL_FILTER_IDS.includes("healing_effects"),
    false,
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "durable",
      kind: "cooldown_badge",
    }),
    "cooldown_effects",
  );
  assert.equal(
    classifyVisualPaintPart({ surface: "durable", kind: "red_zone_floor" }),
    "red_zone_floors",
  );
  for (const kind of ["selection_reticle", "selected_pair_legality"]) {
    assert.equal(
      classifyVisualPaintPart({ surface: "durable", kind }),
      "target_selection_visuals",
    );
  }
  assert.equal(VISUAL_FILTER_IDS.includes("rejected_action_feedback"), false);
});

test("unknown and malformed future paint parts fail closed", () => {
  for (const malformed of [
    null,
    [],
    {},
    { surface: "transient" },
    { kind: "death_effect" },
    { surface: "transient", kind: 1 },
    { surface: "", kind: "death_effect" },
  ]) {
    assert.throws(() => classifyVisualPaintPart(malformed), TypeError);
  }
  assert.throws(
    () =>
      classifyVisualPaintPart({
        surface: "transient",
        kind: "future_effect",
      }),
    /Unregistered visual paint part/u,
  );
  assert.throws(
    () =>
      classifyVisualPaintPart({
        surface: "transient",
        kind: "death_effect",
        unexpected: "field",
      }),
    /Unregistered visual paint part/u,
  );
});
