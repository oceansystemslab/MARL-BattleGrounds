/**
 * @file Keep page-local visibility preferences separate from scientific authority.
 * The fixed registry maps each drawable part to a filter. Helpers validate
 * filter states and produce display keys/updates; they never authorize hidden
 * facts, change a simulation or enter scientific result fingerprints.
 */
/**
 * @typedef {
 *   | "aura_fields"
 *   | "aura_modifier_badges"
 *   | "duration_status_badges"
 *   | "spawn_shield"
 *   | "target_selection_visuals"
 *   | "basic_ability_effects"
 *   | "ultimate_ability_effects"
 *   | "regeneration_effects"
 *   | "cooldown_effects"
 *   | "status_application"
 *   | "natural_status_expiry"
 *   | "freezing_trap_break"
 *   | "status_clear_on_death"
 *   | "death_effects"
 *   | "respawn_wave"
 *   | "resurrection_effects"
 *   | "spawn_shield_expiry"
 *   | "scrolling_battle_text"
 *   | "death_announcer"
 * } VisualFilterId
 * @typedef {Readonly<Record<VisualFilterId, boolean>>} VisualFilterState
 * @typedef {Readonly<Record<string, string>>} VisualPaintPart
 * @typedef {Readonly<{
 *   tag: VisualPaintPart,
 *   filterId: VisualFilterId,
 * }>} VisualPaintPartRegistration
 */

const INITIAL_VISUAL_FILTER_IDS = new Set([
  "ultimate_ability_effects",
  "spawn_shield",
  "basic_ability_effects",
  "regeneration_effects",
  "death_effects",
  "resurrection_effects",
  "scrolling_battle_text",
  "respawn_wave",
  "death_announcer",
]);

export const VISUAL_FILTER_REGISTRY = Object.freeze(
  [
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
  ].map(([id, label]) =>
    Object.freeze({
      id: /** @type {VisualFilterId} */ (id),
      label,
      defaultEnabled: INITIAL_VISUAL_FILTER_IDS.has(id),
    }),
  ),
);

export const VISUAL_FILTER_IDS = Object.freeze(
  VISUAL_FILTER_REGISTRY.map(({ id }) => id),
);

const VISUAL_FILTER_ID_SET = new Set(VISUAL_FILTER_IDS);

export const DEFAULT_VISUAL_FILTER_STATE = freezeVisualFilterState(
  Object.fromEntries(
    VISUAL_FILTER_REGISTRY.map(({ id, defaultEnabled }) => [id, defaultEnabled]),
  ),
);

const ENABLED_VISUAL_FILTER_STATE = freezeVisualFilterState(
  Object.fromEntries(VISUAL_FILTER_IDS.map((id) => [id, true])),
);

const DISABLED_VISUAL_FILTER_STATE = freezeVisualFilterState(
  Object.fromEntries(VISUAL_FILTER_IDS.map((id) => [id, false])),
);

/**
 * Each entry names one independently suppressible paint part. A visible event
 * may contain several entries, but every individual part has exactly one
 * owner. Accessibility and tooltip content belonging to a part follow that
 * same owner.
 *
 * @type {ReadonlyArray<VisualPaintPartRegistration>}
 */
export const VISUAL_PAINT_PART_REGISTRY = Object.freeze([
  paintPart({ surface: "durable", kind: "aura_field" }, "aura_fields"),
  paintPart(
    { surface: "durable", kind: "aura_modifier_badge" },
    "aura_modifier_badges",
  ),
  paintPart(
    { surface: "durable", kind: "duration_status_badge" },
    "duration_status_badges",
  ),
  paintPart(
    { surface: "durable", kind: "pov_duration_status_badge" },
    "duration_status_badges",
  ),
  paintPart({ surface: "durable", kind: "spawn_shield" }, "spawn_shield"),
  paintPart({ surface: "durable", kind: "cooldown_badge" }, "cooldown_effects"),
  paintPart(
    { surface: "durable", kind: "selection_reticle" },
    "target_selection_visuals",
  ),
  paintPart(
    { surface: "durable", kind: "selected_pair_legality" },
    "target_selection_visuals",
  ),
  paintPart(
    {
      surface: "transient",
      kind: "activation",
      component: "basic",
      part: "ability",
    },
    "basic_ability_effects",
  ),
  paintPart(
    {
      surface: "transient",
      kind: "activation",
      component: "ultimate",
      part: "ability",
    },
    "ultimate_ability_effects",
  ),
  paintPart(
    {
      surface: "transient",
      kind: "activation",
      component: "basic",
      part: "semantic",
    },
    "basic_ability_effects",
  ),
  paintPart(
    {
      surface: "transient",
      kind: "activation",
      component: "ultimate",
      part: "semantic",
    },
    "ultimate_ability_effects",
  ),
  ...["damage", "healing", "unchanged"].flatMap((outcome) => [
    paintPart(
      {
        surface: "transient",
        kind: "net_health",
        outcome,
        part: "effect",
      },
      "scrolling_battle_text",
    ),
    paintPart(
      {
        surface: "transient",
        kind: "net_health",
        outcome,
        part: "battle_text",
      },
      "scrolling_battle_text",
    ),
    paintPart(
      {
        surface: "transient",
        kind: "net_health",
        outcome,
        part: "recipient_text",
      },
      "scrolling_battle_text",
    ),
  ]),
  paintPart(
    { surface: "transient", kind: "regeneration", part: "effect" },
    "regeneration_effects",
  ),
  paintPart(
    { surface: "transient", kind: "regeneration", part: "battle_text" },
    "scrolling_battle_text",
  ),
  paintPart(
    { surface: "transient", kind: "cooldown", semantic: "started" },
    "cooldown_effects",
  ),
  paintPart(
    { surface: "transient", kind: "cooldown", semantic: "ready" },
    "cooldown_effects",
  ),
  paintPart(
    {
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "applied",
      part: "effect",
    },
    "status_application",
  ),
  paintPart(
    {
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "expired",
      part: "effect",
    },
    "natural_status_expiry",
  ),
  paintPart(
    {
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "trap_broken",
      part: "effect",
    },
    "freezing_trap_break",
  ),
  paintPart(
    {
      surface: "transient",
      kind: "status_lifecycle",
      lifecycle: "cleared_by_death",
      part: "effect",
    },
    "status_clear_on_death",
  ),
  paintPart({ surface: "transient", kind: "death_effect" }, "death_effects"),
  paintPart({ surface: "transient", kind: "death_announcement" }, "death_announcer"),
  paintPart({ surface: "transient", kind: "respawn_wave" }, "respawn_wave"),
  paintPart(
    { surface: "transient", kind: "resurrection_effect" },
    "resurrection_effects",
  ),
  paintPart(
    { surface: "transient", kind: "spawn_shield_expiry" },
    "spawn_shield_expiry",
  ),
]);

const FILTER_BY_PAINT_PART_KEY = new Map();
for (const registration of VISUAL_PAINT_PART_REGISTRY) {
  const key = paintPartKey(registration.tag);
  if (FILTER_BY_PAINT_PART_KEY.has(key)) {
    throw new TypeError(`Duplicate visual paint-part registration ${key}.`);
  }
  FILTER_BY_PAINT_PART_KEY.set(key, registration.filterId);
}

/**
 * Return the boolean for filterId in state. Require the registered filter
 * name and exactly the expected enumerable state keys with boolean values.
 * Invalid state throws TypeError; unknown filter IDs throw RangeError. No
 * state copy, freeze or mutation occurs.
 *
 * @param {unknown} state
 * @param {unknown} filterId
 */
export function isVisualFilterEnabled(state, filterId) {
  const normalized = assertVisualFilterState(state);
  const id = assertVisualFilterId(filterId);
  return normalized[id];
}

/**
 * Set one filter without mutating the supplied state. Validate state and
 * filterId, and require enabled to be boolean. Return the original object for
 * a no-op; otherwise return a new frozen state. A valid input need not itself
 * be frozen, so the no-op return is not promised to be frozen. Invalid input
 * throws TypeError or RangeError through the shared validators.
 *
 * @param {unknown} state
 * @param {unknown} filterId
 * @param {unknown} enabled
 * @returns {VisualFilterState}
 */
export function setVisualFilterEnabled(state, filterId, enabled) {
  const normalized = assertVisualFilterState(state);
  const id = assertVisualFilterId(filterId);
  if (typeof enabled !== "boolean") {
    throw new TypeError("Visual filter enabled must be a boolean.");
  }
  if (normalized[id] === enabled) {
    return normalized;
  }
  return freezeVisualFilterState(
    Object.fromEntries(
      VISUAL_FILTER_IDS.map((candidate) => [
        candidate,
        candidate === id ? enabled : normalized[candidate],
      ]),
    ),
  );
}

/**
 * Validate state and return a state with every filter enabled. If all are
 * already true, reuse the input unchanged; otherwise return the shared frozen
 * all-enabled state. Invalid state throws TypeError. A reused input is not
 * automatically frozen.
 *
 * @param {unknown} state
 * @returns {VisualFilterState}
 */
export function enableAllVisualFilters(state) {
  const normalized = assertVisualFilterState(state);
  return VISUAL_FILTER_IDS.every((id) => normalized[id])
    ? normalized
    : ENABLED_VISUAL_FILTER_STATE;
}

/**
 * Validate state and return a state with every filter disabled. If all are
 * already false, reuse the input unchanged; otherwise return the shared frozen
 * all-disabled state. Invalid state throws TypeError. No input is mutated or
 * automatically frozen.
 *
 * @param {unknown} state
 * @returns {VisualFilterState}
 */
export function disableAllVisualFilters(state) {
  const normalized = assertVisualFilterState(state);
  return VISUAL_FILTER_IDS.every((id) => !normalized[id])
    ? normalized
    : DISABLED_VISUAL_FILTER_STATE;
}

/**
 * Apply one validated page-local filter action and return the resulting state.
 *
 * state uses the shared state validator. action.type is set, enable_all,
 * disable_all or restore_defaults. set requires exactly type, filterId and
 * enabled; other actions require only type. Invalid shape/value throws
 * TypeError and unknown action/filter names throw RangeError. No-op updates
 * may reuse the input; changed/default states are frozen. This function does
 * not edit the DOM or scientific data.
 *
 * @param {unknown} state
 * @param {unknown} action
 * @returns {VisualFilterState}
 */
export function reduceVisualFilterState(state, action) {
  const normalized = assertVisualFilterState(state);
  if (!isRecord(action) || typeof action.type !== "string") {
    throw new TypeError("Visual filter action must be a tagged object.");
  }
  if (action.type === "set") {
    assertExactKeys(action, ["enabled", "filterId", "type"], "set action");
    return setVisualFilterEnabled(normalized, action.filterId, action.enabled);
  }
  if (action.type === "enable_all") {
    assertExactKeys(action, ["type"], "enable-all action");
    return enableAllVisualFilters(normalized);
  }
  if (action.type === "disable_all") {
    assertExactKeys(action, ["type"], "disable-all action");
    return disableAllVisualFilters(normalized);
  }
  if (action.type === "restore_defaults") {
    assertExactKeys(action, ["type"], "restore-defaults action");
    return DEFAULT_VISUAL_FILTER_STATE;
  }
  throw new RangeError(`Unknown visual filter action ${action.type}.`);
}

/**
 * Validate state and encode its booleans in the fixed registry order. Return
 * a visual-filters-v2 string for local redraw/cache decisions. Invalid state
 * throws TypeError. The key describes display preferences, not scientific
 * conditions or information rights, and must not enter result fingerprints.
 *
 * @param {unknown} state
 */
export function visualFilterPaintKey(state) {
  const normalized = assertVisualFilterState(state);
  return `visual-filters-v2:${VISUAL_FILTER_IDS.map((id) =>
    normalized[id] ? "1" : "0",
  ).join("")}`;
}

/**
 * Return the registered filter ID for an exact drawable-part tag record.
 * part must contain nonempty string tags including surface and kind. The
 * canonical sorted key must be registered; malformed input throws TypeError
 * and an unknown combination throws RangeError. No future/unrecognized part
 * is silently enabled.
 *
 * @param {unknown} part
 * @returns {VisualFilterId}
 */
export function classifyVisualPaintPart(part) {
  const key = paintPartKey(part);
  const filterId = FILTER_BY_PAINT_PART_KEY.get(key);
  if (filterId === undefined) {
    throw new RangeError(`Unregistered visual paint part ${key}.`);
  }
  return filterId;
}

/**
 * Classify part, validate state and return that part's enabled flag.
 * The classification runs first. Propagate TypeError/RangeError for malformed
 * or unregistered inputs. This controls drawing only and changes nothing.
 *
 * @param {unknown} state
 * @param {unknown} part
 */
export function isVisualPaintPartEnabled(state, part) {
  return isVisualFilterEnabled(state, classifyVisualPaintPart(part));
}

/**
 * Create a frozen registry entry from tag and filterId. Copy and freeze the
 * tag's string properties, then freeze the outer record. The authored caller
 * owns validity; this helper does not look up filterId or mutate tag.
 *
 * @param {Record<string, string>} tag
 * @param {VisualFilterId} filterId
 * @returns {VisualPaintPartRegistration}
 */
function paintPart(tag, filterId) {
  return Object.freeze({ tag: Object.freeze({ ...tag }), filterId });
}

/**
 * Sort value's enumerable string entries and serialize a drawable-part key.
 * Require a non-array object, at least two entries, string surface/kind and
 * nonempty string keys/values. Throw TypeError otherwise. This is not a safe
 * parser for hostile accessors; callers pass normalized/local data. It does
 * not itself check registry membership.
 *
 * @param {unknown} value
 */
function paintPartKey(value) {
  if (!isRecord(value)) {
    throw new TypeError("Visual paint part must be a tagged object.");
  }
  const entries = Object.entries(value).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  if (
    entries.length < 2 ||
    typeof value.surface !== "string" ||
    typeof value.kind !== "string"
  ) {
    throw new TypeError("Visual paint part requires string surface and kind tags.");
  }
  for (const [key, field] of entries) {
    if (typeof field !== "string" || key.length === 0 || field.length === 0) {
      throw new TypeError("Visual paint-part tags must be non-empty strings.");
    }
  }
  return JSON.stringify(entries);
}

/**
 * Validate value's enumerable string keys and registered boolean fields.
 * Return the same object; do not copy or freeze it. Missing/extra enumerable
 * keys or nonboolean fields throw TypeError. This local-state check does not
 * reject inherited/symbol fields or inspect property descriptors.
 *
 * @param {unknown} value @returns {VisualFilterState}
 */
function assertVisualFilterState(value) {
  if (!isRecord(value)) {
    throw new TypeError("Visual filter state must be an object.");
  }
  assertExactKeys(value, [...VISUAL_FILTER_IDS], "visual filter state");
  for (const id of VISUAL_FILTER_IDS) {
    if (typeof value[id] !== "boolean") {
      throw new TypeError(`Visual filter ${id} must be a boolean.`);
    }
  }
  return /** @type {VisualFilterState} */ (value);
}

/**
 * Return value when it is an exact registered filter-name string. Otherwise
 * throw RangeError. No trimming or coercion is used for membership.
 *
 * @param {unknown} value @returns {VisualFilterId}
 */
function assertVisualFilterId(value) {
  if (typeof value !== "string" || !VISUAL_FILTER_ID_SET.has(value)) {
    throw new RangeError(`Unknown visual filter ${String(value)}.`);
  }
  return /** @type {VisualFilterId} */ (value);
}

/**
 * Require value's enumerable string keys to equal expected, ignoring order.
 * Throw TypeError using label when they differ; return undefined on success.
 * This does not validate values or reject symbols/non-enumerable fields.
 *
 * @param {Record<string, unknown>} value
 * @param {ReadonlyArray<string>} expected
 * @param {string} label
 */
function assertExactKeys(value, expected, label) {
  const actual = Object.keys(value).sort();
  const canonical = [...expected].sort();
  if (
    actual.length !== canonical.length ||
    actual.some((key, index) => key !== canonical[index])
  ) {
    throw new TypeError(`${label} has an invalid shape.`);
  }
}

/**
 * Return a frozen shallow copy of value. The caller supplies the complete
 * boolean filter record; this helper does not validate its keys or values.
 *
 * @param {Record<string, boolean>} value @returns {VisualFilterState}
 */
function freezeVisualFilterState(value) {
  return /** @type {VisualFilterState} */ (Object.freeze({ ...value }));
}

/**
 * Return true when value is a non-null object that is not an array.
 * This is a shape check only; no prototype, descriptor or authority validation.
 *
 * @param {unknown} value @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
