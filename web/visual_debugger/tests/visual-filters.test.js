import assert from "node:assert/strict";
import test from "node:test";

import {
  classifyVisualPaintPart,
  DEFAULT_VISUAL_FILTER_STATE,
  isVisualFilterEnabled,
  isVisualPaintPartEnabled,
  reduceVisualFilterState,
  restoreAllVisualFilters,
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
  ["combat_status_icon", "Combat Status Icon"],
  ["rejected_action_feedback", "Rejected Action Feedback"],
  ["basic_ability_effects", "Basic Ability Effects"],
  ["ultimate_ability_effects", "Ultimate Ability Effects"],
  ["damage_effects", "Damage Effects"],
  ["healing_effects", "Healing Effects"],
  ["regeneration_effects", "Regeneration Effects"],
  ["cooldown_effects", "Cooldown Effects"],
  ["charge_movement", "Charge Movement"],
  ["status_application", "Status Application"],
  ["status_reapplication", "Status Reapplication"],
  ["status_refresh_extension", "Status Refresh/Extension"],
  ["natural_status_expiry", "Natural Status Expiry"],
  ["freezing_trap_break", "Freezing Trap Break"],
  ["status_clear_on_death", "Status Clear on Death"],
  ["death_effects", "Death Effects"],
  ["respawn_wave", "Respawn Wave"],
  ["resurrection_effects", "Resurrection Effects"],
  ["spawn_shield_expiry", "Spawn-Shield Expiry"],
  ["scrolling_battle_text", "Scrolling Battle Text"],
]);

test("locked registry exposes exactly 24 ordered all-on filters", () => {
  assert.deepEqual(
    VISUAL_FILTER_REGISTRY.map(({ id, label }) => [id, label]),
    EXPECTED_FILTERS,
  );
  assert.deepEqual(
    VISUAL_FILTER_IDS,
    EXPECTED_FILTERS.map(([id]) => id),
  );
  assert.equal(new Set(VISUAL_FILTER_IDS).size, 24);
  assert.equal(
    VISUAL_FILTER_REGISTRY.every(({ defaultEnabled }) => defaultEnabled),
    true,
  );
  assert.equal(Object.isFrozen(VISUAL_FILTER_REGISTRY), true);
  assert.equal(VISUAL_FILTER_REGISTRY.every(Object.isFrozen), true);
});

test("default state is immutable, exact, and enabled through the public helper", () => {
  assert.equal(Object.isFrozen(DEFAULT_VISUAL_FILTER_STATE), true);
  assert.deepEqual(Object.keys(DEFAULT_VISUAL_FILTER_STATE), VISUAL_FILTER_IDS);
  for (const id of VISUAL_FILTER_IDS) {
    assert.equal(DEFAULT_VISUAL_FILTER_STATE[id], true);
    assert.equal(isVisualFilterEnabled(DEFAULT_VISUAL_FILTER_STATE, id), true);
  }
  assert.equal(Reflect.set(DEFAULT_VISUAL_FILTER_STATE, "aura_fields", false), false);
});

test("set and restore-all return frozen states without mutating their input", () => {
  const before = JSON.stringify(DEFAULT_VISUAL_FILTER_STATE);
  const disabled = setVisualFilterEnabled(
    DEFAULT_VISUAL_FILTER_STATE,
    "aura_fields",
    false,
  );
  assert.notEqual(disabled, DEFAULT_VISUAL_FILTER_STATE);
  assert.equal(Object.isFrozen(disabled), true);
  assert.equal(disabled.aura_fields, false);
  assert.equal(disabled.spawn_shield, true);
  assert.equal(JSON.stringify(DEFAULT_VISUAL_FILTER_STATE), before);
  assert.equal(setVisualFilterEnabled(disabled, "aura_fields", false), disabled);
  assert.equal(restoreAllVisualFilters(disabled), DEFAULT_VISUAL_FILTER_STATE);
});

test("strict reducer accepts only exact set and restore-all actions", () => {
  const disabled = reduceVisualFilterState(DEFAULT_VISUAL_FILTER_STATE, {
    type: "set",
    filterId: "scrolling_battle_text",
    enabled: false,
  });
  assert.equal(disabled.scrolling_battle_text, false);
  assert.equal(
    reduceVisualFilterState(disabled, { type: "restore_all" }),
    DEFAULT_VISUAL_FILTER_STATE,
  );
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
    `visual-filters-v1:${"1".repeat(24)}`,
  );
  assert.equal(visualFilterPaintKey(disabled), `visual-filters-v1:0${"1".repeat(22)}0`);
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
      true,
    );
    usedFilters.add(registration.filterId);
  }
  assert.deepEqual([...usedFilters].sort(), [...VISUAL_FILTER_IDS].sort());
});

test("multipart effects retain disjoint filter ownership", () => {
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "trap_broken_and_reapplied",
      part: "break",
    }),
    "freezing_trap_break",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "trap_broken_and_reapplied",
      part: "reapplication",
    }),
    "status_reapplication",
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
      semantic: "healing",
      part: "semantic",
    }),
    "healing_effects",
  );
  assert.equal(
    classifyVisualPaintPart({
      surface: "durable",
      kind: "cooldown_badge",
    }),
    "cooldown_effects",
  );
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
