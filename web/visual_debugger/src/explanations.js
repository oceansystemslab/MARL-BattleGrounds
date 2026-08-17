import {
  canonicalAgentIdentity,
  exactAuthorizedAgentIdentityV1,
} from "./agent-identity.js";
import {
  requiredClassDocumentationValueNamesV1,
  resolveClassDocumentationV1,
} from "./class-documentation.js";
import { formatDisplayNumber } from "./display.js";
import { auraPresentation, statusPresentation } from "./semantic-vocabulary.js";
import { authorizedSourceAttributionV1 } from "./source-attribution.js";
import {
  createSemanticDescriptor,
  projectSemanticDescriptor,
  semanticDescriptorText,
} from "./tooltip.js";
import {
  classTokenFromId,
  resolveVisualToken,
  teamTokenFromId,
  ultimateTokenFromClassId,
} from "./vocabulary.js";

/**
 * Pure semantic-fact builders. Every displayed quantity is copied from an
 * authorized normalized record supplied by the caller. Global slots may form
 * opaque internal descriptor IDs, but never become front-facing identities.
 */

/** @typedef {Record<string, any>} JsonRecord */
/** @typedef {ReturnType<typeof createSemanticDescriptor>} SemanticDescriptor */

const COMPACT_AND_FULL = Object.freeze({ compact: true, full: true });
const FULL_ONLY = Object.freeze({ compact: false, full: true });

const TECHNICAL_FACT_HELP = Object.freeze({
  completion: Object.freeze({
    title: "Completion",
    summary:
      "How the captured rollout ended. Rollout completion is independent of host-side processing success.",
  }),
  processing: Object.freeze({
    title: "Processing",
    summary:
      "Whether host-side evaluation output was produced successfully. Processing does not change how the rollout ended.",
  }),
  frame: Object.freeze({
    title: "Frame",
    summary: "The zero-based authorized frame index represented by this presentation.",
  }),
  simulator_step: Object.freeze({
    title: "Simulator step",
    summary: "The simulator decision step represented by this authorized frame.",
  }),
  ordinary_movement_distance_scale: Object.freeze({
    title: "Ordinary movement distance scale",
    summary:
      "The recorded multiplier applied to ordinary voluntary movement distance. Spawn Shield uses its separately authorized absolute movement speed.",
  }),
});

export { canonicalAgentIdentity } from "./agent-identity.js";

/**
 * Return finite help for one installed operational or Technical Frame fact.
 * Values stay on their owning visible nodes; this descriptor contains only
 * durable explanatory copy and therefore cannot disclose a hidden completion
 * reason, processing error, path, or transport generation.
 *
 * @param {unknown} factId
 */
export function explainTechnicalFact(factId) {
  const key = typeof factId === "string" ? factId : "";
  if (!Object.hasOwn(TECHNICAL_FACT_HELP, key)) {
    throw new RangeError(`Unknown Technical Frame fact ${key || "<empty>"}.`);
  }
  const help =
    TECHNICAL_FACT_HELP[/** @type {keyof typeof TECHNICAL_FACT_HELP} */ (key)];
  return createSemanticDescriptor({
    kind: "technical-help",
    id: `technical-help:${key}`,
    title: help.title,
    tone: "information",
    accent: "none",
    summary: help.summary,
    rows: [],
    sections: [],
    metadata: COMPACT_AND_FULL,
    anchor: "element",
  });
}

/** @param {unknown} value @returns {value is JsonRecord} */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Snapshot only named own enumerable data properties. Accessors are treated as
 * unavailable, and a hostile Proxy fails closed without exposing a trap error
 * to the tooltip or inspector consumer.
 *
 * @param {unknown} value
 * @param {readonly string[]} keys
 * @returns {Readonly<Record<string, unknown>> | null}
 */
function snapshotOwnDataFields(value, keys) {
  try {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    /** @type {Record<string, unknown>} */
    const snapshot = Object.create(null);
    for (const key of keys) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (
        descriptor === undefined ||
        !descriptor.enumerable ||
        !Object.hasOwn(descriptor, "value")
      ) {
        snapshot[key] = undefined;
        continue;
      }
      snapshot[key] = descriptor.value;
    }
    return Object.freeze(snapshot);
  } catch {
    return null;
  }
}

/**
 * Snapshot one exact plain-data record without invoking accessors. This is the
 * fail-closed boundary for mechanics discriminators: missing, extra, symbolic,
 * inherited, accessor-backed, and hostile Proxy fields are all unavailable.
 *
 * @param {unknown} value
 * @param {readonly string[]} keys
 * @returns {Readonly<Record<string, unknown>> | null}
 */
function snapshotExactOwnDataFields(value, keys) {
  try {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return null;
    const actualKeys = Reflect.ownKeys(value);
    if (
      actualKeys.length !== keys.length ||
      actualKeys.some((key) => typeof key !== "string" || !keys.includes(key))
    ) {
      return null;
    }
    /** @type {Record<string, unknown>} */
    const snapshot = Object.create(null);
    const descriptors = Object.getOwnPropertyDescriptors(value);
    for (const key of keys) {
      const field = descriptors[key];
      if (!field?.enumerable || !Object.hasOwn(field, "value")) {
        return null;
      }
      snapshot[key] = field.value;
    }
    return Object.freeze(snapshot);
  } catch {
    return null;
  }
}

/** @param {unknown} value */
function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** @param {unknown} value */
function integer(value) {
  return Number.isInteger(value) ? Number(value) : null;
}

/** @param {unknown} value */
function text(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/** @param {unknown} value */
function humanize(value) {
  return (
    text(value)
      ?.replaceAll("_", " ")
      .replace(/\b\w/g, (c) => c.toUpperCase()) ?? "Unknown"
  );
}

/** @param {unknown} value */
function publicAgentLabel(value) {
  const identity = text(value);
  return identity === null ? "Agent ID unavailable" : `Agent ID ${identity}`;
}

/** @param {unknown} value */
function exactNumber(value) {
  const number = finiteNumber(value);
  return number === null ? "Unavailable" : formatDisplayNumber(number);
}

/** @param {unknown} value */
function tickCount(value) {
  const count = integer(value);
  return count === null ? "Unavailable" : `${count} ${count === 1 ? "Tick" : "Ticks"}`;
}

/** @param {JsonRecord} record */
function semanticIdentity(record) {
  return (
    text(record.presentation_key) ??
    (integer(record.global_slot) === null
      ? null
      : String(integer(record.global_slot))) ??
    text(record.public_agent_id) ??
    "unknown"
  );
}

/**
 * Join one scientific record to one complete authorized scene identity using
 * only its opaque presentation key and public ID. Slots, class guesses, DOM
 * order, and raw-object fallback identity are deliberately excluded.
 *
 * @param {unknown} reference
 * @param {unknown} owner
 * @param {"" | "source_" | "owner_"} prefix
 * @returns {ReturnType<typeof exactAuthorizedAgentIdentityV1>}
 */
function exactJoinedAuthorizedIdentity(reference, owner, prefix) {
  const identity = exactAuthorizedAgentIdentityV1(owner);
  if (identity === null) return null;
  const fields = snapshotOwnDataFields(reference, [
    `${prefix}presentation_key`,
    `${prefix}public_agent_id`,
  ]);
  if (
    fields === null ||
    fields[`${prefix}presentation_key`] !== identity.presentationKey ||
    fields[`${prefix}public_agent_id`] !== identity.publicAgentId
  ) {
    return null;
  }
  return identity;
}

/** @param {unknown} value */
function point(value) {
  return Array.isArray(value) &&
    value.length === 2 &&
    value.every((coordinate) => finiteNumber(coordinate) !== null)
    ? `(${formatDisplayNumber(value[0])}, ${formatDisplayNumber(value[1])})`
    : "Unavailable";
}

/** @param {unknown} multiplier */
function multiplierPercent(multiplier) {
  const exact = finiteNumber(multiplier);
  if (exact === null) {
    return "Unavailable";
  }
  const percent = (exact - 1) * 100;
  const sign = percent > 0 ? "+" : percent < 0 ? "−" : "";
  return `${sign}${formatDisplayNumber(Math.abs(percent))}%`;
}

/**
 * Convert one exact wire multiplier into aura-purpose copy. The presentation
 * registry supplies only the qualitative damage channel; every quantity here
 * comes from the normalized field or aggregate modifier.
 *
 * @param {{effectKind: string}} presentation
 * @param {unknown} multiplier
 * @param {"field" | "recipient"} scope
 */
function auraEffectPresentation(presentation, multiplier, scope) {
  const exact = finiteNumber(multiplier);
  if (exact === null) {
    return "Effect unavailable";
  }
  const difference = `${formatDisplayNumber(Math.abs(exact - 1) * 100)}%`;
  const sourceScope = scope === "field" ? " per emitter" : "";
  if (presentation.effectKind === "damage_dealt") {
    return `${difference} ${exact >= 1 ? "more" : "less"} damage dealt${sourceScope}`;
  }
  if (presentation.effectKind === "damage_received") {
    return `${difference} ${exact <= 1 ? "less" : "more"} damage received${sourceScope}`;
  }
  return `${multiplierPercent(exact)} recorded change${sourceScope}`;
}

/**
 * Translate one wire-authored status magnitude into researcher-readable copy.
 * This is a display conversion of the recorded scalar, never a catalog lookup
 * or browser-owned tuning value.
 *
 * @param {unknown} magnitudeKind
 * @param {unknown} magnitude
 */
function statusMagnitudePresentation(magnitudeKind, magnitude) {
  const kind = text(magnitudeKind);
  const exact = finiteNumber(magnitude);
  if (kind === null || kind === "none" || exact === null) {
    return null;
  }
  const absolutePercent = `${formatDisplayNumber(Math.abs(1 - exact) * 100)}%`;
  if (kind === "movement_multiplier") {
    return {
      label: "Movement Effect",
      value:
        exact <= 1
          ? `${absolutePercent} slower (×${formatDisplayNumber(exact)})`
          : `${absolutePercent} faster (×${formatDisplayNumber(exact)})`,
    };
  }
  if (kind === "healing_multiplier") {
    return {
      label: "Healing Effect",
      value:
        exact <= 1
          ? `${absolutePercent} less healing received (×${formatDisplayNumber(exact)})`
          : `${absolutePercent} more healing received (×${formatDisplayNumber(exact)})`,
    };
  }
  if (kind === "damage_multiplier") {
    return {
      label: "Damage Amplification Effect",
      value:
        exact >= 1
          ? `${absolutePercent} more damage dealt (×${formatDisplayNumber(exact)})`
          : `${absolutePercent} less damage dealt (×${formatDisplayNumber(exact)})`,
    };
  }
  if (kind === "movement_floor") {
    return {
      label: "Movement Floor",
      value: `${formatDisplayNumber(exact * 100)}% of base movement speed (×${formatDisplayNumber(exact)})`,
    };
  }
  return {
    label: `${humanize(kind)} Magnitude`,
    value: formatDisplayNumber(exact),
  };
}

/**
 * @param {string} label
 * @param {unknown} value
 * @param {{compact: boolean, full: boolean}} [metadata]
 */
function row(label, value, metadata = COMPACT_AND_FULL) {
  return { label, value: String(value), metadata };
}

/**
 * @param {string} title
 * @param {Array<ReturnType<typeof row>>} rows
 * @param {string | null} [summary]
 * @param {{compact: boolean, full: boolean}} [metadata]
 */
function section(title, rows, summary = null, metadata = FULL_ONLY) {
  return { title, rows, summary, metadata };
}

/**
 * @param {string} kind
 * @param {string} id
 * @param {string} title
 * @param {string | null} summary
 * @param {Array<ReturnType<typeof row>>} rows
 * @param {Array<ReturnType<typeof section>>} [sections]
 * @param {{tone?: string, accent?: string, anchor?: "element" | "pointer"}} [options]
 */
function descriptor(kind, id, title, summary, rows, sections = [], options = {}) {
  return createSemanticDescriptor({
    kind,
    id,
    title,
    tone: options.tone ?? "neutral",
    accent: options.accent ?? "none",
    summary,
    rows,
    sections,
    metadata: COMPACT_AND_FULL,
    anchor: options.anchor ?? "element",
  });
}

/**
 * @param {unknown} rawAgent
 * @param {{controlled?: boolean, selected?: boolean, reference?: boolean, inspected?: boolean, audience?: string}} [selection]
 * @returns {SemanticDescriptor}
 */
export function explainAgent(rawAgent, selection = {}) {
  if (selection.audience === "agent_pov") {
    return explainPovAgent(rawAgent, selection);
  }
  const agent = isRecord(rawAgent) ? rawAgent : {};
  const identity = canonicalAgentIdentity(agent);
  const currentHealth = finiteNumber(agent.current_health);
  const maxHealth =
    finiteNumber(agent.max_health) ?? finiteNumber(agent.maximum_health);
  const effectiveSpeed =
    finiteNumber(agent.effective_movement_speed) ?? finiteNumber(agent.effective_speed);
  const cooldown =
    integer(agent.ultimate_cooldown_remaining) ?? integer(agent.ultimate_cooldown);
  const combatCountdown = integer(agent.steps_until_out_of_combat);
  const currentRows = [
    row(
      "Health",
      currentHealth === null || maxHealth === null
        ? "Unavailable"
        : `${formatDisplayNumber(currentHealth)} / ${formatDisplayNumber(maxHealth)}`,
    ),
    row(
      "Effective Speed",
      effectiveSpeed === null ? "Unavailable" : formatDisplayNumber(effectiveSpeed),
    ),
    row(
      "Ultimate Status",
      cooldown === null
        ? "Unavailable"
        : cooldown === 0
          ? "Ready"
          : `On cooldown (${tickCount(cooldown)})`,
    ),
    row(
      "Combat Status",
      combatCountdown === null ? "Unavailable" : combatCountdown > 0 ? "IC" : "OOC",
    ),
  ];
  if (combatCountdown !== null && combatCountdown > 0) {
    currentRows.push(row("Steps until OOC", tickCount(combatCountdown)));
  }
  return descriptor(
    "agent",
    `agent:${semanticIdentity(agent)}`,
    identity.title,
    null,
    currentRows,
    [],
    {
      tone: currentHealth === 0 ? "warning" : "information",
      accent: identity.accent,
    },
  );
}

/**
 * Build an agent card through a recipient-authorized field whitelist. Passing
 * researcher mechanics or arbitrary extra fields cannot affect the result.
 *
 * @param {unknown} rawAgent
 * @param {Record<string, unknown>} [_selection]
 * @returns {SemanticDescriptor}
 */
export function explainPovAgent(rawAgent, _selection = {}) {
  const input = isRecord(rawAgent) ? rawAgent : {};
  const reduced = {
    presentation_key: input.presentation_key,
    public_agent_id: input.public_agent_id,
    team_id: input.team_id,
    class_id: input.class_id,
    current_health: input.current_health,
    max_health: input.max_health ?? input.maximum_health,
    effective_movement_speed: input.effective_movement_speed,
    ultimate_cooldown_remaining: input.ultimate_cooldown_remaining,
    steps_until_out_of_combat: input.steps_until_out_of_combat,
  };
  return explainAgent(reduced, { audience: "reduced_agent_pov" });
}

const SPAWN_SHIELD_V1_KEYS = Object.freeze([
  "availability_kind",
  "configured_duration_steps",
  "movement_speed",
]);
const SPAWN_SHIELD_V2_KEYS = Object.freeze([
  ...SPAWN_SHIELD_V1_KEYS,
  "protection_effect",
  "visibility_effect",
  "targetability_effect",
  "action_scope",
  "aura_effect",
  "agent_collision_effect",
  "ordinary_application_mechanism",
]);
const SPAWN_SHIELD_UNAVAILABLE_KEYS = Object.freeze(["availability_kind"]);
const SPAWN_SHIELD_V2_SUMMARY =
  "While the spawn shield is active, this agent is protected, concealed from opponents, untargetable, excluded from aura effects, and limited to movement. It phases through agents until body collision resumes at the endpoint of its expiring transition.";

/**
 * @param {unknown} rawMechanics
 * @returns {Readonly<{kind: "v1" | "v2" | "unavailable", values: Readonly<Record<string, unknown>> | null}>}
 */
function exactSpawnShieldMechanics(rawMechanics) {
  const discriminator = snapshotOwnDataFields(rawMechanics, ["availability_kind"]);
  if (discriminator?.availability_kind === "available") {
    const values = snapshotExactOwnDataFields(rawMechanics, SPAWN_SHIELD_V1_KEYS);
    if (
      values !== null &&
      Number.isSafeInteger(values.configured_duration_steps) &&
      Number(values.configured_duration_steps) >= 0 &&
      finiteNumber(values.movement_speed) !== null &&
      Number(values.movement_speed) > 0
    ) {
      return Object.freeze({ kind: /** @type {const} */ ("v1"), values });
    }
  }
  if (discriminator?.availability_kind === "available_v2") {
    const values = snapshotExactOwnDataFields(rawMechanics, SPAWN_SHIELD_V2_KEYS);
    if (
      values !== null &&
      Number.isSafeInteger(values.configured_duration_steps) &&
      Number(values.configured_duration_steps) >= 0 &&
      finiteNumber(values.movement_speed) !== null &&
      Number(values.movement_speed) > 0 &&
      values.protection_effect === "invulnerable" &&
      values.visibility_effect === "concealed_from_opponents" &&
      values.targetability_effect === "untargetable" &&
      values.action_scope === "movement_only" &&
      values.aura_effect === "excluded_as_emitter_and_beneficiary" &&
      values.agent_collision_effect === "phased_until_expiring_endpoint_rejoin" &&
      values.ordinary_application_mechanism === "end_of_transition_respawn_lifecycle"
    ) {
      return Object.freeze({ kind: /** @type {const} */ ("v2"), values });
    }
  }
  if (discriminator?.availability_kind === "unavailable") {
    const values = snapshotExactOwnDataFields(
      rawMechanics,
      SPAWN_SHIELD_UNAVAILABLE_KEYS,
    );
    if (values !== null) {
      return Object.freeze({ kind: /** @type {const} */ ("unavailable"), values });
    }
  }
  return Object.freeze({
    kind: /** @type {const} */ ("unavailable"),
    values: null,
  });
}

/**
 * Build the single mechanics-discriminated Spawn Shield view consumed by the
 * renderer. V2 alone unlocks categorical semantics; V1 carries only recorded
 * numeric/current facts, and unavailable or malformed mechanics fail closed.
 *
 * @param {unknown} rawAgent
 * @param {unknown} rawMechanics
 * @returns {Readonly<{
 *   active: boolean,
 *   badgeText: string,
 *   descriptor: SemanticDescriptor,
 *   remainingTicks: number,
 *   rootAriaLabel: string | null,
 *   shieldAriaLabel: string,
 * }>}
 */
export function createSpawnShieldView(rawAgent, rawMechanics) {
  const identity = exactAuthorizedAgentIdentityV1(rawAgent);
  const agentFields = snapshotOwnDataFields(rawAgent, ["spawn_shield_remaining"]);
  const recordedRemaining = integer(agentFields?.spawn_shield_remaining);
  const remaining =
    recordedRemaining !== null && recordedRemaining >= 0 ? recordedRemaining : null;
  const remainingTicks = remaining ?? 0;
  const active = remainingTicks > 0;
  const mechanics = exactSpawnShieldMechanics(rawMechanics);
  const owner = identity?.title ?? "Unavailable";
  const currentRows = [
    row(
      "Duration Remaining",
      remaining === null ? "Unavailable" : tickCount(remaining),
    ),
    row("Owner", owner),
    row("Source", "Not recorded"),
  ];
  let summary = null;
  let rows = currentRows;
  if (mechanics.kind === "v1" && mechanics.values !== null) {
    rows = [
      row("Movement Speed", formatDisplayNumber(mechanics.values.movement_speed)),
      row("Effect Duration", tickCount(mechanics.values.configured_duration_steps)),
      ...currentRows,
    ];
  } else if (mechanics.kind === "v2" && mechanics.values !== null) {
    summary = SPAWN_SHIELD_V2_SUMMARY;
    rows = [
      row("Protection Effect", "Invulnerable"),
      row("Movement Speed", formatDisplayNumber(mechanics.values.movement_speed)),
      row("Visibility Effect", "Concealed from opponents"),
      row("Targetability Effect", "Untargetable"),
      row("Action Effect", "Movement only"),
      row("Aura Effect", "Excluded as emitter and beneficiary"),
      row("Agent Collision Effect", "Phased until expiring endpoint rejoin"),
      row("Effect Duration", tickCount(mechanics.values.configured_duration_steps)),
      ...currentRows,
      row("Ordinary Application", "End-of-transition respawn lifecycle"),
    ];
  }
  const explanation = descriptor(
    "status",
    `spawn-shield:${identity?.presentationKey ?? "unavailable"}`,
    "Spawn Shield",
    summary,
    rows,
    [],
    { tone: active ? "positive" : "neutral", accent: "none" },
  );
  return Object.freeze({
    active,
    badgeText: `S${remainingTicks}`,
    descriptor: explanation,
    remainingTicks,
    rootAriaLabel: active
      ? `Spawn Shield active, ${remainingTicks} ${remainingTicks === 1 ? "tick" : "ticks"} remaining`
      : null,
    shieldAriaLabel: explanation.title,
  });
}

/**
 * @param {unknown} rawAgent
 * @param {unknown} [rawMechanics]
 * @returns {SemanticDescriptor}
 */
export function explainSpawnShield(rawAgent, rawMechanics) {
  return createSpawnShieldView(rawAgent, rawMechanics).descriptor;
}

/** @param {unknown} value */
function formattedMechanicNumber(value) {
  const exact = finiteNumber(value);
  return exact === null ? null : formatDisplayNumber(exact);
}

/** @param {string} prefix @param {unknown} value */
function prefixedMechanicNumber(prefix, value) {
  const formatted = formattedMechanicNumber(value);
  return formatted === null ? null : `${prefix}${formatted}`;
}

/** @param {unknown} value */
function formattedMechanicTicks(value) {
  const exact = integer(value);
  return exact === null ? null : tickCount(exact);
}

/**
 * @param {unknown} rawItems
 * @param {string} field
 * @param {string} expected
 * @returns {JsonRecord | null}
 */
function exactNestedMechanic(rawItems, field, expected) {
  if (!Array.isArray(rawItems) || !rawItems.every(isRecord)) return null;
  const matches = rawItems.filter((item) => item[field] === expected);
  return matches.length === 1 ? matches[0] : null;
}

/**
 * @param {JsonRecord} mechanics
 * @param {string} statusId
 * @param {string} magnitudeKind
 */
function formattedStatusMechanicEffect(mechanics, statusId, magnitudeKind) {
  const status = exactNestedMechanic(mechanics.status_mechanics, "status_id", statusId);
  if (status?.magnitude_kind !== magnitudeKind) return null;
  const magnitude = finiteNumber(status.magnitude);
  if (magnitude === null) return null;
  const difference = `${formatDisplayNumber(Math.abs(1 - magnitude) * 100)}%`;
  const multiplier = `×${formatDisplayNumber(magnitude)}`;
  if (magnitudeKind === "damage_multiplier") {
    return `a factor of ${formatDisplayNumber(magnitude)} (${difference} ${magnitude >= 1 ? "more" : "less"} damage dealt)`;
  }
  if (magnitudeKind === "movement_multiplier") {
    return `a ${difference} movement ${magnitude <= 1 ? "reduction" : "increase"} (${multiplier})`;
  }
  if (magnitudeKind === "healing_multiplier") {
    return `a ${difference} ${magnitude <= 1 ? "reduction" : "increase"} (${multiplier})`;
  }
  if (magnitudeKind === "movement_floor") {
    return `a floor of ${formatDisplayNumber(magnitude * 100)}% of base movement speed (${multiplier})`;
  }
  return null;
}

/**
 * @param {JsonRecord} mechanics
 * @param {string} statusId
 */
function formattedStatusMechanicDuration(mechanics, statusId) {
  const status = exactNestedMechanic(mechanics.status_mechanics, "status_id", statusId);
  return status === null ? null : formattedMechanicTicks(status.duration_steps);
}

/**
 * @param {JsonRecord} mechanics
 * @param {string} auraId
 */
function exactAuraMechanic(mechanics, auraId) {
  return exactNestedMechanic(mechanics.aura_mechanics, "aura_id", auraId);
}

/**
 * @param {{effectKind: string}} presentation
 * @param {unknown} multiplier
 */
function formattedAuraDocumentationEffect(presentation, multiplier) {
  const exact = finiteNumber(multiplier);
  if (exact === null) return null;
  const difference = `${formatDisplayNumber(Math.abs(1 - exact) * 100)}%`;
  if (presentation.effectKind === "damage_dealt") {
    return `a ${difference} outgoing-damage ${exact >= 1 ? "increase" : "reduction"} per recorded emitter`;
  }
  if (presentation.effectKind === "damage_received") {
    return `a ${difference} incoming-damage ${exact <= 1 ? "reduction" : "increase"} per recorded emitter`;
  }
  return null;
}

/**
 * Build only the named, preformatted values consumed by one certified authored
 * guide. Missing nested mechanics remain null and can never become an
 * `Unavailable` interpolation.
 *
 * @param {JsonRecord} mechanics
 * @returns {Record<string, string> | null}
 */
function classDocumentationValueMap(mechanics) {
  const classId = integer(mechanics.class_id);
  /** @type {Array<[string, string | null]> | null} */
  let entries = null;
  if (classId === 1) {
    const aura = exactAuraMechanic(mechanics, "mage_damage_amplification");
    const auraMultiplier =
      aura === null ? null : finiteNumber(aura.per_emitter_multiplier);
    entries = [
      [
        "burstDuration",
        formattedStatusMechanicDuration(mechanics, "mage_burst_damage_amplification"),
      ],
      [
        "burstDamageEffect",
        formattedStatusMechanicEffect(
          mechanics,
          "mage_burst_damage_amplification",
          "damage_multiplier",
        ),
      ],
      [
        "auraRadius",
        aura === null ? null : prefixedMechanicNumber("a radius of ", aura.radius),
      ],
      [
        "perEmitterDamageAmplificationEffect",
        auraMultiplier === null
          ? null
          : formattedAuraDocumentationEffect(
              auraPresentation("mage_damage_amplification"),
              auraMultiplier,
            ),
      ],
      [
        "damageAmplificationCeiling",
        aura === null ? null : formattedMechanicNumber(aura.clamp_value),
      ],
    ];
  } else if (classId === 2) {
    const aura = exactAuraMechanic(mechanics, "warrior_damage_mitigation");
    const auraMultiplier =
      aura === null ? null : finiteNumber(aura.per_emitter_multiplier);
    entries = [
      ["ultimateRawDamage", formattedMechanicNumber(mechanics.ultimate_raw_damage)],
      [
        "chargeStunDuration",
        formattedStatusMechanicDuration(mechanics, "warrior_charge_stun"),
      ],
      [
        "chargeSlowEffect",
        formattedStatusMechanicEffect(
          mechanics,
          "warrior_charge_slow",
          "movement_multiplier",
        ),
      ],
      [
        "chargeSlowDuration",
        formattedStatusMechanicDuration(mechanics, "warrior_charge_slow"),
      ],
      [
        "auraRadius",
        aura === null ? null : prefixedMechanicNumber("a radius of ", aura.radius),
      ],
      [
        "perEmitterDamageMitigationEffect",
        auraMultiplier === null
          ? null
          : formattedAuraDocumentationEffect(
              auraPresentation("warrior_damage_mitigation"),
              auraMultiplier,
            ),
      ],
      [
        "damageMitigationFloor",
        aura === null ? null : formattedMechanicNumber(aura.clamp_value),
      ],
    ];
  } else if (classId === 3) {
    entries = [
      ["ultimateRawDamage", formattedMechanicNumber(mechanics.ultimate_raw_damage)],
      [
        "trapStunDuration",
        formattedStatusMechanicDuration(mechanics, "hunter_trap_stun"),
      ],
      [
        "hunterBasicSlowDuration",
        formattedStatusMechanicDuration(mechanics, "hunter_basic_slow"),
      ],
      [
        "hunterBasicMovementEffect",
        formattedStatusMechanicEffect(
          mechanics,
          "hunter_basic_slow",
          "movement_multiplier",
        ),
      ],
    ];
  } else if (classId === 4) {
    entries = [
      ["ultimateRawDamage", formattedMechanicNumber(mechanics.ultimate_raw_damage)],
      [
        "poisonStunDuration",
        formattedStatusMechanicDuration(mechanics, "rogue_poison_stun"),
      ],
      [
        "poisonSlowEffect",
        formattedStatusMechanicEffect(
          mechanics,
          "rogue_poison_slow",
          "movement_multiplier",
        ),
      ],
      [
        "poisonSlowDuration",
        formattedStatusMechanicDuration(mechanics, "rogue_poison_slow"),
      ],
      [
        "poisonAntiHealEffect",
        formattedStatusMechanicEffect(
          mechanics,
          "rogue_poison_anti_heal",
          "healing_multiplier",
        ),
      ],
      [
        "poisonAntiHealDuration",
        formattedStatusMechanicDuration(mechanics, "rogue_poison_anti_heal"),
      ],
      [
        "baseMovementSpeed",
        prefixedMechanicNumber(
          "base movement speed of ",
          mechanics.base_movement_speed,
        ),
      ],
      ["outOfCombatDelay", formattedMechanicTicks(mechanics.out_of_combat_delay_steps)],
    ];
  } else if (classId === 5) {
    entries = [
      ["ultimateRawHealing", formattedMechanicNumber(mechanics.ultimate_raw_healing)],
      [
        "freedomDuration",
        formattedStatusMechanicDuration(
          mechanics,
          "priest_blessing_of_freedom_movement_floor",
        ),
      ],
      [
        "freedomMovementFloor",
        formattedStatusMechanicEffect(
          mechanics,
          "priest_blessing_of_freedom_movement_floor",
          "movement_floor",
        ),
      ],
    ];
  }
  if (
    entries === null ||
    entries.some(
      ([, value]) =>
        typeof value !== "string" ||
        value.length === 0 ||
        /\bunavailable\b/iu.test(value),
    )
  ) {
    return null;
  }
  return Object.fromEntries(/** @type {Array<[string, string]>} */ (entries));
}

/**
 * @param {JsonRecord} mechanics
 * @returns {Array<ReturnType<typeof row>> | null}
 */
function documentationMechanicsRows(mechanics) {
  const classId = integer(mechanics.class_id);
  const classToken = classTokenFromId(classId);
  const maximumHealth = formattedMechanicNumber(mechanics.maximum_health);
  const bodyRadius = formattedMechanicNumber(mechanics.body_radius);
  const baseMovementSpeed = formattedMechanicNumber(mechanics.base_movement_speed);
  const observationRadius = formattedMechanicNumber(mechanics.observation_radius);
  const basicRadius = formattedMechanicNumber(mechanics.basic_interaction_radius);
  const basicDamage = finiteNumber(mechanics.basic_raw_damage);
  const basicHealing = finiteNumber(mechanics.basic_raw_healing);
  const outOfCombatDelay = formattedMechanicTicks(mechanics.out_of_combat_delay_steps);
  const regeneration = finiteNumber(
    mechanics.out_of_combat_health_regeneration_fraction_per_step,
  );
  const ultimateRadius = formattedMechanicNumber(mechanics.ultimate_interaction_radius);
  const ultimateCooldown = formattedMechanicTicks(mechanics.ultimate_cooldown_steps);
  const ultimateDamage = finiteNumber(mechanics.ultimate_raw_damage);
  const ultimateHealing = finiteNumber(mechanics.ultimate_raw_healing);
  const basicTarget = text(mechanics.basic_target_mode);
  const ultimateTarget = text(mechanics.ultimate_target_mode);
  const validBasicTargets = ["unavailable", "ally", "enemy"];
  const validUltimateTargets = ["unavailable", "target_none", "ally", "enemy"];
  if (
    classToken.label === "Unknown" ||
    text(mechanics.class_name) !== classToken.label ||
    [
      maximumHealth,
      bodyRadius,
      baseMovementSpeed,
      observationRadius,
      basicRadius,
      outOfCombatDelay,
      ultimateRadius,
      ultimateCooldown,
    ].some((value) => value === null) ||
    [basicDamage, basicHealing, regeneration, ultimateDamage, ultimateHealing].some(
      (value) => value === null,
    ) ||
    basicTarget === null ||
    !validBasicTargets.includes(basicTarget) ||
    ultimateTarget === null ||
    !validUltimateTargets.includes(ultimateTarget)
  ) {
    return null;
  }
  const rows = [
    row("Maximum Health", maximumHealth, FULL_ONLY),
    row("Body Radius", bodyRadius, FULL_ONLY),
    row("Base Movement Speed", baseMovementSpeed, FULL_ONLY),
    row("Observation Radius", observationRadius, FULL_ONLY),
    row("Basic Target", humanize(basicTarget), FULL_ONLY),
    row("Basic Ability Radius", basicRadius, FULL_ONLY),
  ];
  if (/** @type {number} */ (basicDamage) > 0) {
    rows.push(row("Basic Raw Damage", basicDamage, FULL_ONLY));
  }
  if (/** @type {number} */ (basicHealing) > 0) {
    rows.push(row("Basic Raw Healing", basicHealing, FULL_ONLY));
  }
  rows.push(
    row("Out-of-combat Delay", outOfCombatDelay, FULL_ONLY),
    row(
      "Out-of-combat Regeneration",
      `${formatDisplayNumber(/** @type {number} */ (regeneration) * 100)}% of maximum health per Tick`,
      FULL_ONLY,
    ),
  );
  return rows;
}

/**
 * Build the persistent class-documentation card from one exact owner/mechanics
 * join. Current agent state is deliberately unreachable. V1 and unavailable
 * profiles retain payload mechanics but receive no current authored copy.
 *
 * @param {unknown} rawOwner
 * @param {unknown} rawClassMechanics
 * @returns {SemanticDescriptor | null}
 */
export function explainClassDocumentation(rawOwner, rawClassMechanics) {
  if (!isRecord(rawOwner) || !isRecord(rawClassMechanics)) return null;
  const owner = rawOwner;
  const mechanics = rawClassMechanics;
  const publicId = text(owner.public_agent_id);
  const classId = integer(owner.class_id);
  const classToken = classTokenFromId(classId);
  const teamToken = teamTokenFromId(owner.team_id);
  if (
    publicId === null ||
    classToken.label === "Unknown" ||
    teamToken.label === "Unknown" ||
    text(owner.class_name) !== classToken.label ||
    integer(mechanics.class_id) !== classId ||
    text(mechanics.class_name) !== classToken.label ||
    (mechanics.mechanics_version !== undefined && mechanics.mechanics_version !== 2)
  ) {
    return null;
  }
  const mechanicsRows = documentationMechanicsRows(mechanics);
  if (mechanicsRows === null) return null;

  let authored = null;
  if (mechanics.mechanics_version === 2) {
    const requiredNames = requiredClassDocumentationValueNamesV1(
      mechanics.documentation_profile,
      classId,
    );
    if (requiredNames !== null) {
      const valueMap = classDocumentationValueMap(mechanics);
      if (
        valueMap !== null &&
        Object.keys(valueMap).length === requiredNames.length &&
        requiredNames.every((name) => typeof valueMap[name] === "string")
      ) {
        authored = resolveClassDocumentationV1(
          mechanics.documentation_profile,
          classId,
          Object.fromEntries(requiredNames.map((name) => [name, valueMap[name]])),
        );
      }
    }
  }

  const completeMechanicsRows = [...mechanicsRows];
  if (authored !== null) {
    completeMechanicsRows.push(
      row("Ultimate Name", authored.ultimate.name, FULL_ONLY),
      row("Ultimate Description", authored.ultimate.description, FULL_ONLY),
    );
  }
  completeMechanicsRows.push(
    row("Ultimate Target", humanize(mechanics.ultimate_target_mode), FULL_ONLY),
    row(
      "Ultimate Radius",
      formattedMechanicNumber(mechanics.ultimate_interaction_radius),
      FULL_ONLY,
    ),
    row(
      "Ultimate Cooldown",
      formattedMechanicTicks(mechanics.ultimate_cooldown_steps),
      FULL_ONLY,
    ),
  );
  if (/** @type {number} */ (finiteNumber(mechanics.ultimate_raw_damage)) > 0) {
    completeMechanicsRows.push(
      row("Ultimate Raw Damage", mechanics.ultimate_raw_damage, FULL_ONLY),
    );
  }
  if (/** @type {number} */ (finiteNumber(mechanics.ultimate_raw_healing)) > 0) {
    completeMechanicsRows.push(
      row("Ultimate Raw Healing", mechanics.ultimate_raw_healing, FULL_ONLY),
    );
  }
  if (authored !== null) {
    completeMechanicsRows.push(
      row("Passive Name", authored.passive.name, FULL_ONLY),
      row("Passive Description", authored.passive.description, FULL_ONLY),
    );
  }
  const sections = [];
  if (authored !== null) {
    sections.push(
      section("Class Overview", [], authored.overview),
      section(
        "Authored Tactical Guide",
        authored.tacticalGuideRows.map((guideRow) =>
          row(guideRow.label, guideRow.value, FULL_ONLY),
        ),
      ),
    );
  }
  sections.push(section("Class Mechanics", completeMechanicsRows));
  const identity = canonicalAgentIdentity(owner);
  return descriptor(
    "agent",
    `class-documentation:${publicId}`,
    identity.title,
    null,
    [],
    sections,
    { tone: "information", accent: identity.accent },
  );
}

/**
 * Shared durable-status card. Audience changes only B3 source disclosure; the
 * configured duration, remaining duration, magnitude, and trap break fact are
 * copied from the same authorized status record.
 *
 * @param {unknown} rawStatus
 * @param {unknown} rawRecipient
 * @param {ReadonlyArray<unknown>} rawSourceAgents
 * @param {"researcher" | "agent_pov"} audience
 */
function explainDurableStatus(rawStatus, rawRecipient, rawSourceAgents, audience) {
  const status = isRecord(rawStatus) ? rawStatus : {};
  const recipient = isRecord(rawRecipient) ? rawRecipient : {};
  const token = resolveVisualToken(
    "status",
    status.token_id ?? status.status_id,
    status,
  );
  const profile = statusPresentation(token.tokenId);
  const configuredDuration = integer(status.configured_duration_steps);
  const remainingDuration = integer(status.remaining_duration);
  const magnitude =
    profile.magnitudeKind === status.magnitude_kind
      ? statusMagnitudePresentation(status.magnitude_kind, status.magnitude)
      : null;
  const rows = [];
  if (magnitude !== null) {
    rows.push(row(magnitude.label, magnitude.value));
  }
  rows.push(
    row(
      "Effect Duration",
      configuredDuration === null ? "Unavailable" : tickCount(configuredDuration),
    ),
    row(
      "Duration Remaining",
      remainingDuration === null ? "Unavailable" : tickCount(remainingDuration),
    ),
  );
  if (profile.positiveDamageBreak && status.breaks_on_positive_damage === true) {
    rows.push(row("Break Rule", "Ends when this agent receives positive raw damage"));
  }
  const source = authorizedSourceAttributionV1({
    attribution_kind: "direct",
    audience,
    direct_sources: status.direct_sources,
    authorized_agents: rawSourceAgents,
  });
  if (source !== null) {
    rows.push(row(source.label, source.value));
  }
  return descriptor(
    "status",
    `status:${semanticIdentity(recipient)}:${integer(status.status_channel) ?? token.tokenId}`,
    profile.title,
    profile.effect,
    rows,
    [],
    { tone: "information", accent: profile.accent },
  );
}

/**
 * @param {unknown} rawStatus
 * @param {unknown} [rawRecipient]
 */
export function explainPovStatus(rawStatus, rawRecipient = {}) {
  return explainDurableStatus(rawStatus, rawRecipient, [], "agent_pov");
}

/**
 * @param {unknown} rawStatus
 * @param {unknown} [rawRecipient]
 * @param {ReadonlyArray<unknown>} [rawSourceAgents]
 */
export function explainStatus(rawStatus, rawRecipient = {}, rawSourceAgents = []) {
  return explainDurableStatus(rawStatus, rawRecipient, rawSourceAgents, "researcher");
}

/**
 * @param {unknown} rawModifier
 * @param {unknown} [rawRecipient]
 */
export function explainModifier(rawModifier, rawRecipient = {}) {
  const recipientIdentity = exactAuthorizedAgentIdentityV1(rawRecipient);
  if (recipientIdentity === null) {
    return null;
  }
  const modifier = snapshotOwnDataFields(rawModifier, [
    "token_id",
    "aura_id",
    "multiplier",
    "label",
    "short_label",
    "shortLabel",
    "accessible_name",
    "accessibleName",
  ]);
  if (modifier === null) {
    return null;
  }
  const token = resolveVisualToken(
    "modifier",
    modifier.token_id ?? modifier.aura_id,
    modifier,
  );
  const presentation = auraPresentation(modifier.aura_id ?? modifier.token_id);
  const multiplier = finiteNumber(modifier.multiplier);
  const effect = auraEffectPresentation(presentation, multiplier, "recipient");
  return descriptor(
    "modifier",
    `modifier:${recipientIdentity.presentationKey}:${token.tokenId}`,
    presentation.recipientTitle,
    presentation.aggregateEffect,
    [row(presentation.aggregateEffectLabel, effect)],
    [],
    { tone: "information", accent: presentation.accent },
  );
}

/**
 * @param {ReadonlyArray<unknown>} rawItems
 * @param {"status" | "modifier"} kind
 * @param {unknown} [rawRecipient]
 * @param {ReadonlyArray<unknown>} [rawSourceAgents]
 */
export function explainOverflow(
  rawItems,
  kind,
  rawRecipient = {},
  rawSourceAgents = [],
) {
  const recipient = isRecord(rawRecipient) ? rawRecipient : {};
  const items = Array.isArray(rawItems) ? rawItems : [];
  const rows = items.map((item, index) => {
    const explanation =
      kind === "status"
        ? explainStatus(item, recipient, rawSourceAgents)
        : explainModifier(item, recipient);
    if (explanation === null) {
      return row(`Hidden ${index + 1}`, "Unavailable");
    }
    const compact = projectSemanticDescriptor(explanation, "compact");
    return row(
      `Hidden ${index + 1}`,
      [explanation.title, ...semanticDescriptorText(compact)].join(" · "),
    );
  });
  return descriptor(
    `${kind}-overflow`,
    `${kind}-overflow:${semanticIdentity(recipient)}`,
    `${items.length} Hidden ${kind === "status" ? "Statuses" : "Modifiers"}`,
    `Every hidden ${kind} remains available in canonical display order.`,
    rows.length === 0 ? [row("Hidden Facts", "None")] : rows,
    [],
    { tone: "neutral" },
  );
}

/**
 * Recipient-authorized overflow for POV status docks. Every hidden item passes
 * through the same reduced status projection as a visible POV status cell;
 * researcher attribution cannot alter either the descriptor bytes or copy.
 *
 * @param {ReadonlyArray<unknown>} rawItems
 * @param {unknown} [rawRecipient]
 */
export function explainPovOverflow(rawItems, rawRecipient = {}) {
  const inputRecipient = isRecord(rawRecipient) ? rawRecipient : {};
  const recipient = {
    presentation_key: inputRecipient.presentation_key,
    public_agent_id: inputRecipient.public_agent_id,
  };
  const items = Array.isArray(rawItems) ? rawItems : [];
  const rows = items.map((item, index) => {
    const explanation = explainPovStatus(item, recipient);
    const compact = projectSemanticDescriptor(explanation, "compact");
    return row(
      `Hidden ${index + 1}`,
      [explanation.title, ...semanticDescriptorText(compact)].join(" · "),
    );
  });
  return descriptor(
    "status-overflow",
    typeof recipient.presentation_key === "string"
      ? `pov-status-overflow:${recipient.presentation_key}`
      : `pov-status-overflow:${text(recipient.public_agent_id) ?? "unknown"}`,
    `${items.length} Hidden ${items.length === 1 ? "Status" : "Statuses"}`,
    "Every hidden status remains available in canonical display order. Source not disclosed in Agent POV.",
    rows.length === 0 ? [row("Hidden Facts", "None")] : rows,
    [],
    { tone: "neutral" },
  );
}

/**
 * @param {unknown} rawRecord
 * @param {unknown} [rawOwner]
 * @returns {SemanticDescriptor | null}
 */
export function explainCooldown(rawRecord, rawOwner = null) {
  const record = snapshotOwnDataFields(rawRecord, [
    "presentation_key",
    "public_agent_id",
    "ultimate_cooldown_remaining",
    "ultimate_cooldown",
  ]);
  const owner = snapshotOwnDataFields(rawOwner, ["class_id"]);
  if (record === null || owner === null) {
    return null;
  }
  const identity = exactJoinedAuthorizedIdentity(record, rawOwner, "");
  if (identity === null) {
    return null;
  }
  const ticks =
    integer(record.ultimate_cooldown_remaining) ?? integer(record.ultimate_cooldown);
  if (ticks === null || ticks < 0) {
    return null;
  }
  const ultimate = ultimateTokenFromClassId(owner.class_id);
  return descriptor(
    "cooldown",
    `cooldown:${identity.presentationKey}`,
    `${ultimate.label} Cooldown · ${identity.title}`,
    null,
    [
      ticks === 0
        ? row("Ultimate Status", "Ready")
        : row("Remaining Cooldown", tickCount(ticks)),
      row("Source", identity.title),
    ],
    [],
    {
      tone: ticks === 0 ? "positive" : "information",
      accent: identity.accent,
    },
  );
}

/**
 * @param {unknown} rawField
 * @param {unknown} [rawSourceAgent]
 * @param {"researcher" | "agent_pov"} [audience]
 * @returns {SemanticDescriptor | null}
 */
export function explainAura(rawField, rawSourceAgent = null, audience = "researcher") {
  if (audience !== "researcher" && audience !== "agent_pov") {
    return null;
  }
  const field = snapshotOwnDataFields(rawField, [
    "token_id",
    "aura_id",
    "per_emitter_multiplier",
    "radius",
    "label",
    "short_label",
    "shortLabel",
    "accessible_name",
    "accessibleName",
  ]);
  if (field === null) {
    return null;
  }
  const token = resolveVisualToken("modifier", field.token_id ?? field.aura_id, field);
  const presentation = auraPresentation(field.aura_id ?? field.token_id);
  const sourceIdentity =
    audience === "researcher"
      ? exactJoinedAuthorizedIdentity(rawField, rawSourceAgent, "source_")
      : null;
  const source =
    audience === "researcher" &&
    sourceIdentity !== null &&
    sourceIdentity.accent === presentation.accent
      ? rawSourceAgent
      : null;
  const attribution =
    audience === "agent_pov"
      ? authorizedSourceAttributionV1({
          attribution_kind: "direct",
          audience: "agent_pov",
          direct_sources: [rawField],
          authorized_agents: [rawSourceAgent],
        })
      : authorizedSourceAttributionV1({
          attribution_kind: "direct",
          audience: "researcher",
          direct_sources: [
            {
              source_presentation_key: sourceIdentity?.presentationKey ?? null,
              source_public_agent_id: sourceIdentity?.publicAgentId ?? null,
            },
          ],
          authorized_agents: source === null ? [] : [source],
        });
  const multiplier = finiteNumber(field.per_emitter_multiplier);
  const effect = auraEffectPresentation(presentation, multiplier, "field");
  return descriptor(
    "aura",
    audience === "agent_pov"
      ? `aura:agent-pov:${token.tokenId}`
      : `aura:${sourceIdentity?.presentationKey ?? "unavailable"}:${token.tokenId}`,
    presentation.fieldTitle,
    presentation.fieldEffect,
    [
      row(presentation.fieldEffectLabel, effect),
      row("Effect Radius", exactNumber(field.radius)),
      row(
        attribution?.label ?? "Source",
        attribution?.value ?? "Unavailable in this artifact",
      ),
    ],
    [],
    {
      tone: "information",
      accent: presentation.accent,
      anchor: "pointer",
    },
  );
}

/**
 * @param {unknown} rawRange
 * @param {unknown} [rawOwner]
 * @returns {SemanticDescriptor | null}
 */
export function explainRange(rawRange, rawOwner = null) {
  const range = snapshotOwnDataFields(rawRange, [
    "presentation_key",
    "public_agent_id",
    "kind",
    "radius",
  ]);
  if (range === null) {
    return null;
  }
  const identity = exactJoinedAuthorizedIdentity(range, rawOwner, "");
  if (identity === null) {
    return null;
  }
  const kind = text(range.kind);
  const radius = finiteNumber(range.radius);
  if (!["observation", "basic", "ultimate"].includes(kind ?? "") || radius === null) {
    return null;
  }
  const rangeKind = /** @type {"observation" | "basic" | "ultimate"} */ (kind);
  const title =
    rangeKind === "observation"
      ? "Observation Range"
      : `${rangeKind === "basic" ? "Basic" : "Ultimate"} Range · ${identity.title}`;
  return descriptor(
    `range-${rangeKind}`,
    `range:${identity.presentationKey}:${rangeKind}`,
    title,
    null,
    [
      row("Radius", formatDisplayNumber(radius)),
      row("Owner ID", identity.publicIdentity),
      row("Team", identity.teamLabel),
      row("Class", identity.classLabel),
    ],
    [],
    {
      tone: "information",
      accent: identity.accent,
      anchor: "pointer",
    },
  );
}

/** @param {unknown} rawObstacle */
export function explainObstacle(rawObstacle) {
  const obstacle = isRecord(rawObstacle) ? rawObstacle : {};
  const obstacleId = text(obstacle.obstacle_id) ?? "Unavailable";
  const kind = text(obstacle.kind) ?? "unknown";
  const rows = [row("Obstacle ID", obstacleId)];
  if (kind === "pillar") {
    rows.push(row("Radius", exactNumber(obstacle.radius)));
  } else if (kind === "wall") {
    rows.push(
      row("Width", exactNumber(obstacle.width)),
      row("Height", exactNumber(obstacle.height)),
      row("Rotation", exactNumber(obstacle.theta)),
    );
  }
  rows.push(row("Center", point(obstacle.center)));
  return descriptor(
    "obstacle",
    `obstacle:${obstacleId}`,
    humanize(kind),
    `Exact normalized ${kind} geometry.`,
    rows,
    [],
    { tone: "neutral", anchor: "pointer" },
  );
}

/**
 * @param {unknown} rawFact
 * @param {{observerAgent?: unknown, candidateAgent?: unknown}} [context]
 */
export function explainVisibility(rawFact, context = {}) {
  const fact = isRecord(rawFact) ? rawFact : {};
  const observer = isRecord(context.observerAgent) ? context.observerAgent : {};
  const candidate = isRecord(context.candidateAgent) ? context.candidateAgent : {};
  const visible = typeof fact.visible === "boolean" ? fact.visible : null;
  return descriptor(
    "visibility",
    `visibility:${integer(fact.observer_global_slot) ?? "unknown"}:${integer(fact.candidate_global_slot) ?? "unknown"}`,
    "Observer Visibility",
    "Oracle View visibility diagnostic copied from the normalized scene.",
    [
      row("Observer", publicAgentLabel(observer.public_agent_id)),
      row("Candidate", publicAgentLabel(candidate.public_agent_id)),
      row("Visible", visible === null ? "Unavailable" : visible ? "True" : "False"),
    ],
    [],
    { tone: "information" },
  );
}

/**
 * @param {unknown} rawLegality
 * @param {0 | 1} lane
 * @param {unknown} rawOwner
 * @returns {SemanticDescriptor | null}
 */
export function explainLegality(rawLegality, lane, rawOwner) {
  if (lane !== 0 && lane !== 1) {
    return null;
  }
  const laneProperty = lane === 0 ? "lane_0_available" : "lane_1_available";
  const legality = snapshotOwnDataFields(rawLegality, [
    "owner_presentation_key",
    "owner_public_agent_id",
    laneProperty,
  ]);
  if (legality === null) {
    return null;
  }
  const identity = exactJoinedAuthorizedIdentity(legality, rawOwner, "owner_");
  if (identity === null) {
    return null;
  }
  const laneName = lane === 0 ? "Basic" : "Ultimate";
  const rawAvailable = legality[laneProperty];
  if (typeof rawAvailable !== "boolean") {
    return null;
  }
  return descriptor(
    "legality",
    `legality:${identity.presentationKey}:${lane}:${rawAvailable}`,
    `${laneName} Legality · ${identity.publicIdentity}`,
    null,
    [row("Status", rawAvailable ? "True" : "False")],
    [],
    { tone: rawAvailable ? "positive" : "warning", accent: identity.accent },
  );
}

/** @param {unknown} rawRoute */
export function explainPendingRoute(rawRoute) {
  const route = isRecord(rawRoute) ? rawRoute : {};
  const lane = integer(route.lane);
  const laneName = lane === 0 ? "Basic" : lane === 1 ? "Ultimate" : "Action";
  return descriptor(
    "pending-route",
    `pending:${text(route.source_presentation_key) ?? text(route.source_public_agent_id) ?? "unknown"}:${text(route.target_presentation_key) ?? text(route.target_public_agent_id) ?? "unknown"}:${lane ?? "unknown"}`,
    `${laneName} Action Route`,
    "Authorized action route; no physical path is implied.",
    [
      row("Source", publicAgentLabel(route.source_public_agent_id)),
      row("Target", publicAgentLabel(route.target_public_agent_id)),
    ],
    [],
    { tone: "information", anchor: "pointer" },
  );
}

/** @param {unknown} rawEvent */
export function explainActivation(rawEvent) {
  const event = isRecord(rawEvent) ? rawEvent : {};
  const token = resolveVisualToken(
    "activation",
    event.tokenId ?? event.token_id,
    event,
  );
  const source = event.sourcePublicAgentId ?? event.source_public_agent_id;
  const target = event.targetPublicAgentId ?? event.target_public_agent_id;
  const redacted =
    event.targetDisclosure === "redacted" || event.target_disclosure === "redacted";
  return descriptor(
    "activation",
    `activation:${token.tokenId}:${text(event.sourcePresentationKey ?? event.source_presentation_key) ?? text(source) ?? "source-unavailable"}:${text(event.targetPresentationKey ?? event.target_presentation_key) ?? text(target) ?? (redacted ? "target-redacted" : "source-local")}`,
    token.label,
    token.accessibleName,
    [
      row("Source", publicAgentLabel(source)),
      row(
        "Target",
        text(target) !== null
          ? publicAgentLabel(target)
          : redacted
            ? "Target endpoint not disclosed in this view"
            : "Source-local activation",
      ),
      row("Health Attribution", "No per-source health amount is available"),
    ],
    [],
    { tone: "information", anchor: "pointer" },
  );
}

/** @param {unknown} rawEvent */
export function explainNetHealth(rawEvent) {
  const event = isRecord(rawEvent) ? rawEvent : {};
  const delta = finiteNumber(event.netDelta ?? event.net_delta);
  return descriptor(
    "impact",
    `net:${text(event.recipientPresentationKey ?? event.recipient_presentation_key) ?? text(event.recipientPublicAgentId ?? event.recipient_public_agent_id) ?? "recipient-unavailable"}:${text(event.outcome) ?? "outcome-unavailable"}`,
    `Recipient NET ${humanize(event.outcome)}`,
    "Recipient-level before/after outcome; not source attribution.",
    [
      row(
        "Recipient",
        publicAgentLabel(
          event.recipientPublicAgentId ?? event.recipient_public_agent_id,
        ),
      ),
      row("NET", delta === null ? "Unavailable" : formatNetDelta(delta)),
    ],
    [],
    {
      tone: delta !== null && delta < 0 ? "warning" : "information",
      anchor: "pointer",
    },
  );
}

/** @param {number} delta */
function formatNetDelta(delta) {
  const displayed = formatDisplayNumber(delta);
  if (delta !== 0 && displayed === "0") {
    return `${delta > 0 ? "+" : "−"}<0.01`;
  }
  return `${delta > 0 ? "+" : ""}${displayed}`;
}
