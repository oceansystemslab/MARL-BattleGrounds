/**
 * Browser-local visual presentation controls.
 *
 * These values never authorize, redact, or mutate scientific presentation
 * data. The registry order is part of the local paint-key contract and must
 * remain stable.
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
 * } VisualFilterId
 * @typedef {Readonly<Record<VisualFilterId, boolean>>} VisualFilterState
 * @typedef {Readonly<Record<string, string>>} VisualPaintPart
 * @typedef {Readonly<{
 *   tag: VisualPaintPart,
 *   filterId: VisualFilterId,
 * }>} VisualPaintPartRegistration
 */

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
  ].map(([id, label]) =>
    Object.freeze({
      id: /** @type {VisualFilterId} */ (id),
      label,
      defaultEnabled: true,
    }),
  ),
);

export const VISUAL_FILTER_IDS = Object.freeze(
  VISUAL_FILTER_REGISTRY.map(({ id }) => id),
);

const VISUAL_FILTER_ID_SET = new Set(VISUAL_FILTER_IDS);

export const DEFAULT_VISUAL_FILTER_STATE = freezeVisualFilterState(
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
 * Return whether one registered filter is enabled in an exact state.
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
 * Return a new frozen state with exactly one registered filter changed.
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
 * Enable every visual filter after validating the current state.
 *
 * @param {unknown} state
 * @returns {VisualFilterState}
 */
export function enableAllVisualFilters(state) {
  const normalized = assertVisualFilterState(state);
  return VISUAL_FILTER_IDS.every((id) => normalized[id])
    ? normalized
    : DEFAULT_VISUAL_FILTER_STATE;
}

/**
 * Disable every visual filter after validating the current state.
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
 * Strict reducer for the local checkbox surface.
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
  throw new RangeError(`Unknown visual filter action ${action.type}.`);
}

/**
 * Serialize an exact state in locked registry order. The result is local
 * presentation identity only; it must never enter scientific fingerprints.
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
 * Classify one exact tagged paint part. Unknown or malformed future parts fail
 * closed so a new visual cannot bypass the registry silently.
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
 * @param {unknown} state
 * @param {unknown} part
 */
export function isVisualPaintPartEnabled(state, part) {
  return isVisualFilterEnabled(state, classifyVisualPaintPart(part));
}

/**
 * @param {Record<string, string>} tag
 * @param {VisualFilterId} filterId
 * @returns {VisualPaintPartRegistration}
 */
function paintPart(tag, filterId) {
  return Object.freeze({ tag: Object.freeze({ ...tag }), filterId });
}

/** @param {unknown} value */
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

/** @param {unknown} value @returns {VisualFilterState} */
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

/** @param {unknown} value @returns {VisualFilterId} */
function assertVisualFilterId(value) {
  if (typeof value !== "string" || !VISUAL_FILTER_ID_SET.has(value)) {
    throw new RangeError(`Unknown visual filter ${String(value)}.`);
  }
  return /** @type {VisualFilterId} */ (value);
}

/**
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

/** @param {Record<string, boolean>} value @returns {VisualFilterState} */
function freezeVisualFilterState(value) {
  return /** @type {VisualFilterState} */ (Object.freeze({ ...value }));
}

/** @param {unknown} value @returns {value is Record<string, any>} */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
