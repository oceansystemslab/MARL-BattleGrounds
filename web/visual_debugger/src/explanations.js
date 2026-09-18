/**
 * @file Build tooltip and inspector descriptions from authorized display records.
 * These pure builders copy quantities from their inputs and use shared
 * display vocabulary. They do not calculate simulator outcomes, grant
 * information access or validate an entire presentation. Callers must pass
 * records from the appropriate normalized audience and perform required
 * joins. Missing facts stay unavailable; descriptor IDs are internal keys.
 */
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
  episode: Object.freeze({
    title: "Episode",
    summary: "Identifies the recorded episode represented by this frame.",
  }),
  task_mode: Object.freeze({
    title: "Task Mode",
    summary: "The task whose rules govern this episode.",
  }),
  map: Object.freeze({
    title: "Map",
    summary: "The recorded technical map name. Its split is shown only when recorded.",
  }),
  observation_mode: Object.freeze({
    title: "Observation Mode",
    summary: "The policy observation mode used when this episode was recorded.",
  }),
  episode_limit: Object.freeze({
    title: "Episode Limit",
    summary: "The maximum number of transitions planned for this episode.",
  }),
  seeds: Object.freeze({
    title: "Seeds",
    summary:
      "The recorded root seed and episode stream coordinate identify the random streams. Unknown means the value was not recorded.",
  }),
  artifact_digest_prefix: Object.freeze({
    title: "Artifact Digest Prefix",
    summary:
      "These 12 hexadecimal characters locate the canonical Oracle replay without displaying its full hash.",
  }),
  incoming_transition: Object.freeze({
    title: "Incoming Transition",
    summary:
      "Identifies the authorized transition that produced this displayed frame. The initial frame has no incoming transition.",
  }),
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
    title: "Simulator Step",
    summary: "The simulator decision step represented by this authorized frame.",
  }),
  ordinary_movement_distance_scale: Object.freeze({
    title: "Ordinary Movement Distance Scale",
    summary:
      "The recorded multiplier applied to ordinary voluntary movement distance. Spawn Shield uses its separately authorized absolute movement speed.",
  }),
});

export { canonicalAgentIdentity } from "./agent-identity.js";

/**
 * Build a descriptor for a normalized grouped death event. event supplies
 * eventId, label and members with title, killingTeamId and contributor rows.
 * Preserve recorded contributor order; null contributors remain explicitly
 * unavailable for historical evidence. Return a semantic descriptor without
 * assigning final-hit credit or changing the event. Caller owns authorization.
 *
 * @param {JsonRecord} event
 */
export function explainDeathAnnouncement(event) {
  return createSemanticDescriptor({
    kind: "event",
    id: `death-announcement:${event.eventId}`,
    title: event.label,
    tone: "information",
    accent: "none",
    summary:
      "Kill contributors include direct damage and useful same-tick Priest healing of a damaging contributor. Credit is shared; it does not identify one final hitter.",
    rows: [],
    sections: event.members.map((/** @type {JsonRecord} */ member) => ({
      title: member.title,
      summary: "This agent died on the incoming transition.",
      rows: [
        {
          label: "Killing Team",
          value:
            member.killingTeamId === null
              ? "Unavailable"
              : `Team ${member.killingTeamId === 1 ? "A" : "B"}`,
          metadata: COMPACT_AND_FULL,
        },
        {
          label: "Kill Contributors",
          value:
            member.contributors === null
              ? "Unavailable — not recorded in this historical evidence."
              : member.contributors
                  .map((/** @type {JsonRecord} */ contributor) => contributor.title)
                  .join("; "),
          metadata: COMPACT_AND_FULL,
        },
      ],
      metadata: COMPACT_AND_FULL,
    })),
    metadata: COMPACT_AND_FULL,
    anchor: "pointer",
  });
}

/**
 * Return a help descriptor for one recognized Technical Frame factId.
 * Throw RangeError for an unknown/nonstring ID. The descriptor contains help
 * only; it does not include a current value, processing error, hidden outcome
 * or source path. Its visible fact node owns the actual recorded value.
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

/**
 * Return whether value is a non-null non-array object. This shallow check
 * accepts class instances and does not validate fields or block accessors.
 *
 * @param {unknown} value @returns {value is JsonRecord}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Copy only keys' own enumerable data values into a frozen record.
 * Missing, accessor-backed or nonenumerable fields become undefined. Return
 * null for nonobjects, arrays or reflection errors; extra fields are ignored.
 * Children remain references. Proxy traps may run during reflection, but
 * their thrown errors are caught rather than shown in the interface.
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
 * Copy a plain record with exactly keys into a frozen record, or return
 * null when its shape cannot be trusted. Require Object.prototype/null,
 * no symbols or extra keys and enumerable own data fields. Catch reflection
 * errors and never read accessor values. Child values remain references;
 * this does not validate their types or prevent Proxy traps from executing.
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

/**
 * Return a finite numeric value unchanged, otherwise null. Do not coerce
 * strings or booleans, round values or enforce a scientific range.
 *
 * @param {unknown} value
 */
function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Return value when Number.isInteger accepts it, otherwise null. This
 * is not a safe-integer or nonnegative check; callers enforce those bounds
 * when their contract requires them.
 *
 * @param {unknown} value
 */
function integer(value) {
  return Number.isInteger(value) ? Number(value) : null;
}

/**
 * Return a trimmed nonempty string, otherwise null. This does not parse
 * markup, validate authority or limit length.
 *
 * @param {unknown} value
 */
function text(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/**
 * Turn a nonempty identifier into display words by replacing underscores
 * and capitalizing word starts. Return Unknown for invalid/empty input.
 * This is display formatting, not a semantic catalog lookup.
 *
 * @param {unknown} value
 */
function humanize(value) {
  return (
    text(value)
      ?.replaceAll("_", " ")
      .replace(/\b\w/g, (c) => c.toUpperCase()) ?? "Unknown"
  );
}

/**
 * Return the title from a complete exact authorized identity, or null.
 * Delegate identity checks to exactAuthorizedAgentIdentityV1; never infer a
 * public identity from a global slot or a partial record.
 *
 * @param {unknown} value
 */
function authorizedAgentIdentityTitle(value) {
  return exactAuthorizedAgentIdentityV1(value)?.title ?? null;
}

/**
 * Trim string value and add a final period unless it already ends in
 * ., ! or ?. Return null for invalid/empty input. No semantic rewriting occurs.
 *
 * @param {unknown} value
 */
function sentence(value) {
  const copy = text(value);
  if (copy === null) return null;
  return /[.!?]$/u.test(copy) ? copy : `${copy}.`;
}

/**
 * Format finite numeric value with the shared display formatter; return
 * Unavailable otherwise. The display may round; the source number is unchanged.
 *
 * @param {unknown} value
 */
function exactNumber(value) {
  const number = finiteNumber(value);
  return number === null ? "Unavailable" : formatDisplayNumber(number);
}

/**
 * Format an integer as N tick/ticks, using singular only for 1. Return
 * Unavailable for a noninteger. This does not reject negative counts or
 * convert seconds to simulator ticks.
 *
 * @param {unknown} value
 */
function tickCount(value) {
  const count = integer(value);
  return count === null ? "Unavailable" : `${count} ${count === 1 ? "tick" : "ticks"}`;
}

/**
 * Build an internal descriptor key from presentation_key, then integer
 * global_slot, then public_agent_id, falling back to unknown. It does not
 * prove identity or authorization. Never display this internal key as a
 * public agent label.
 *
 * @param {JsonRecord} record
 */
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
 * Return owner's complete identity only when reference's prefixed
 * presentation_key and public_agent_id both match it exactly. prefix is
 * empty, source_ or owner_. Return null for incomplete/mismatched identities
 * or unreadable fields. Do not fall back to slots, class names or DOM order.
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

/**
 * Format a two-element array of finite coordinates as (x, y), otherwise
 * return Unavailable. Use the shared number formatter; units and authorized
 * coordinate space are the caller's responsibility.
 *
 * @param {unknown} value
 */
function point(value) {
  return Array.isArray(value) &&
    value.length === 2 &&
    value.every((coordinate) => finiteNumber(coordinate) !== null)
    ? `(${formatDisplayNumber(value[0])}, ${formatDisplayNumber(value[1])})`
    : "Unavailable";
}

/**
 * Format a finite multiplier as its signed percent change from 1.
 * Use a plus sign for an increase and a minus sign for a decrease. Return
 * Unavailable for other inputs. This converts a recorded multiplier for
 * display without calculating the underlying effect.
 *
 * @param {unknown} multiplier
 */
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
 * Describe multiplier using presentation.effectKind's damage channel.
 * scope=field adds per-emitter wording; recipient describes the aggregate.
 * Unknown channels use generic recorded-change text and nonfinite values
 * produce Effect unavailable. Return text only; no aura overlap is calculated.
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
 * Convert recorded magnitude to one labelled display row, or null for
 * missing/none kind or nonfinite magnitude. Recognize movement, healing and
 * damage multipliers plus movement floors; other named kinds use a generic
 * numeric row. Return a plain label/value object without looking up tuning
 * values or validating the input's permitted scientific range.
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
 * Create a label/value row, converting value with String. metadata defaults
 * to visibility in compact and full views and is retained by reference.
 * Return a plain record; final descriptor construction owns normalization.
 *
 * @param {string} label
 * @param {unknown} value
 * @param {{compact: boolean, full: boolean}} [metadata]
 */
function row(label, value, metadata = COMPACT_AND_FULL) {
  return { label, value: String(value), metadata };
}

/**
 * Create a section with title and rows, optional null summary and
 * full-only metadata by default. Preserve references; do not render DOM or
 * validate row contents here.
 *
 * @param {string} title
 * @param {Array<ReturnType<typeof row>>} rows
 * @param {string | null} [summary]
 * @param {{compact: boolean, full: boolean}} [metadata]
 */
function section(title, rows, summary = null, metadata = FULL_ONLY) {
  return { title, rows, summary, metadata };
}

/**
 * Create a normalized semantic descriptor from rows and optional sections.
 * sections defaults to [], options to {}; defaults are neutral tone, no
 * accent and element anchoring. Mark the descriptor visible in compact/full
 * views; each row/section keeps its own visibility rules. Delegate final
 * normalization to createSemanticDescriptor and return its frozen result.
 *
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
 * Describe current health, speed, Ultimate cooldown and combat countdown
 * from an authorized agent record. Missing fields display Unavailable.
 * selection defaults to {}; only audience=agent_pov selects the restricted
 * field route, while other selection flags do not change this card. Return
 * a descriptor without reading simulator state or modifying inputs. Caller
 * must supply the correct audience; this helper is not a privacy filter for
 * arbitrary researcher records.
 *
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
          : `On Cooldown (${tickCount(cooldown)})`,
    ),
    row(
      "Combat Status",
      combatCountdown === null
        ? "Unavailable"
        : combatCountdown > 0
          ? "In Combat"
          : "Out of Combat",
    ),
  ];
  if (combatCountdown !== null && combatCountdown > 0) {
    currentRows.push(row("Steps Until Out of Combat", tickCount(combatCountdown)));
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
 * Build an agent card from the permitted identity/current-state fields.
 * Ignore _selection (default {}) and unrelated fields. Delegate formatting
 * to explainAgent using a reduced record. Return a descriptor; this field
 * projection does not itself prove the input was authorized for a recipient.
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
    display_agent_id: input.display_agent_id,
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

/**
 * Recognize exact available V1, available_v2 or unavailable Spawn Shield
 * records. Require nonnegative safe duration and positive finite speed; V2
 * also requires each declared categorical effect. Return a frozen kind/values
 * record, using unavailable with null values for malformed inputs. This
 * checks the serialized mechanics contract, not a live Core configuration.
 *
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
 * Build Spawn Shield badge state, accessible labels and a descriptor.
 * Use rawAgent's exact identity and nonnegative integer remaining duration.
 * Missing/invalid duration yields inactive/0 for badge control but remains
 * Unavailable in the descriptor. Exact V1 mechanics adds numeric fields;
 * V2 also permits categorical effects. Invalid mechanics does not invent
 * those effects. Return a frozen view without modifying inputs or applying
 * shield logic. A positive recorded duration alone determines active.
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
  const recipient = identity?.title ?? "Unavailable";
  const currentRows = [
    row(
      "Duration Remaining",
      remaining === null ? "Unavailable" : tickCount(remaining),
    ),
    row("Recipient", recipient),
  ];
  const summary = spawnShieldStatusSummary(rawMechanics);
  let rows = currentRows;
  if (mechanics.kind === "v1" && mechanics.values !== null) {
    rows = [
      row("Movement Speed", formatDisplayNumber(mechanics.values.movement_speed)),
      row("Effect Duration", tickCount(mechanics.values.configured_duration_steps)),
      ...currentRows,
    ];
  } else if (mechanics.kind === "v2" && mechanics.values !== null) {
    rows = [
      row("Protection Effect", "Invulnerable"),
      row("Movement Speed", formatDisplayNumber(mechanics.values.movement_speed)),
      row("Visibility Effect", "Concealed from opponents"),
      row("Targetability Effect", "Untargetable"),
      row("Action Effect", "Movement only"),
      row("Aura Effect", "Excluded as emitter and beneficiary"),
      row(
        "Agent Collision Effect",
        "The agent can move through other agents while shielded; collision resumes at the end of the shield's final transition.",
      ),
      row("Effect Duration", tickCount(mechanics.values.configured_duration_steps)),
      ...currentRows,
    ];
  }
  const explanation = descriptor(
    "status",
    `spawn-shield:${identity?.presentationKey ?? "unavailable"}`,
    statusPresentation("spawn_shield").title,
    summary,
    rows,
    [],
    { tone: active ? "positive" : "neutral", accent: "none" },
  );
  return Object.freeze({
    active,
    badgeText: String(remainingTicks),
    descriptor: explanation,
    remainingTicks,
    rootAriaLabel: active
      ? `Spawn Shield active, ${remainingTicks} ${remainingTicks === 1 ? "tick" : "ticks"} remaining`
      : null,
    shieldAriaLabel: explanation.title,
  });
}

/**
 * Return Spawn Shield's categorical summary only for a valid exact V2
 * mechanics record. Return null for V1, unavailable or malformed records.
 * Historical numeric-only evidence does not acquire today's categorical text.
 *
 * @param {unknown} rawMechanics
 * @returns {string | null}
 */
export function spawnShieldStatusSummary(rawMechanics) {
  const mechanics = exactSpawnShieldMechanics(rawMechanics);
  return mechanics.kind === "v2" && mechanics.values !== null
    ? statusPresentation("spawn_shield").effect
    : null;
}

/**
 * Return only the descriptor from createSpawnShieldView. rawAgent provides
 * current duration/identity; omitted or invalid rawMechanics leaves configured
 * effects unavailable. No shield state is created or changed.
 *
 * @param {unknown} rawAgent
 * @param {unknown} [rawMechanics]
 * @returns {SemanticDescriptor}
 */
export function explainSpawnShield(rawAgent, rawMechanics) {
  return createSpawnShieldView(rawAgent, rawMechanics).descriptor;
}

/**
 * Return formatted finite numeric value, or null. Null means an authored
 * guide lacks a required value; do not interpolate the word Unavailable.
 *
 * @param {unknown} value
 */
function formattedMechanicNumber(value) {
  const exact = finiteNumber(value);
  return exact === null ? null : formatDisplayNumber(exact);
}

/**
 * Prefix a formatted finite value with prefix, or return null when value
 * is unavailable. This builds prose only and leaves numeric data unchanged.
 *
 * @param {string} prefix @param {unknown} value
 */
function prefixedMechanicNumber(prefix, value) {
  const formatted = formattedMechanicNumber(value);
  return formatted === null ? null : `${prefix}${formatted}`;
}

/**
 * Format integer value in ticks, or return null. This does not enforce
 * nonnegative/safe bounds; validated mechanics must supply those guarantees.
 *
 * @param {unknown} value
 */
function formattedMechanicTicks(value) {
  const exact = integer(value);
  return exact === null ? null : tickCount(exact);
}

/**
 * Return the sole record in rawItems whose field exactly equals expected.
 * Require an array entirely made of non-null non-array objects. Return null
 * for invalid input, no match or duplicates. The matched object is shared;
 * this helper does not validate the remaining fields.
 *
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
 * Describe the sole matching statusId when its magnitude_kind exactly
 * matches magnitudeKind and its magnitude is finite. Support damage,
 * movement and healing multipliers plus movement floors. Return null for
 * missing, ambiguous, mismatched or unsupported mechanics. All values come
 * from the supplied authorized mechanics bank.
 *
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
    return `a ${difference} ${magnitude >= 1 ? "increase" : "reduction"} (${multiplier})`;
  }
  if (magnitudeKind === "movement_multiplier") {
    return `a ${difference} movement ${magnitude <= 1 ? "reduction" : "increase"} (${multiplier})`;
  }
  if (magnitudeKind === "healing_multiplier") {
    return `a ${difference} ${magnitude <= 1 ? "reduction" : "increase"} (${multiplier})`;
  }
  if (magnitudeKind === "movement_floor") {
    return `${formatDisplayNumber(magnitude * 100)}% of base movement speed (${multiplier})`;
  }
  return null;
}

/**
 * Format duration_steps for the sole matching statusId, or return null
 * when the status or integer duration is unavailable. Preserve the recorded
 * duration rather than looking up a browser-side default.
 *
 * @param {JsonRecord} mechanics
 * @param {string} statusId
 */
function formattedStatusMechanicDuration(mechanics, statusId) {
  const status = exactNestedMechanic(mechanics.status_mechanics, "status_id", statusId);
  return status === null ? null : formattedMechanicTicks(status.duration_steps);
}

/**
 * Return the sole auraId match in mechanics.aura_mechanics through
 * exactNestedMechanic, or null. No copy or aura-effect calculation occurs.
 *
 * @param {JsonRecord} mechanics
 * @param {string} auraId
 */
function exactAuraMechanic(mechanics, auraId) {
  return exactNestedMechanic(mechanics.aura_mechanics, "aura_id", auraId);
}

/**
 * Convert a finite per-emitter multiplier into authored-guide wording.
 * Damage-dealt channels allow increases/reductions; damage-received supports
 * a multiplier at most 1. Return null for unsupported/missing values so the
 * guide cannot silently invent an effect.
 *
 * @param {{effectKind: string}} presentation
 * @param {unknown} multiplier
 */
function formattedAuraDocumentationEffect(presentation, multiplier) {
  const exact = finiteNumber(multiplier);
  if (exact === null) return null;
  const difference = `${formatDisplayNumber(Math.abs(1 - exact) * 100)}%`;
  if (presentation.effectKind === "damage_dealt") {
    return `a ${difference} damage ${exact >= 1 ? "bonus" : "reduction"}`;
  }
  if (presentation.effectKind === "damage_received") {
    return exact <= 1 ? difference : null;
  }
  return null;
}

/**
 * Build the named formatted values needed by the supplied class's guide.
 * Use recorded Ultimate, status and aura mechanics for classes 1–5. Return
 * a plain string map, or null if class/required values are missing, ambiguous
 * or format as unavailable. This prepares substitutions only; the shared
 * profile resolver decides whether that authored guide applies.
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
        aura === null || finiteNumber(aura.clamp_value) === null
          ? null
          : `${formatDisplayNumber(aura.clamp_value * 100)}%`,
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
      ["baseMovementSpeed", formattedMechanicNumber(mechanics.base_movement_speed)],
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
 * Resolve a guide only for V2 mechanics with a recognized matching class
 * ID/name and an accepted documentation profile. Require exactly its named
 * formatted values, then delegate to resolveClassDocumentationV1. Return
 * the resolved guide or null. Persistent class cards and Ultimate cards use
 * this same authority; this does not validate an entire wire presentation.
 *
 * @param {JsonRecord} mechanics
 */
function resolvedAuthorizedClassDocumentationV1(mechanics) {
  const classId = integer(mechanics.class_id);
  const classToken = classTokenFromId(classId);
  if (
    mechanics.mechanics_version !== 2 ||
    classToken.label === "Unknown" ||
    text(mechanics.class_name) !== classToken.label
  ) {
    return null;
  }
  const requiredNames = requiredClassDocumentationValueNamesV1(
    mechanics.documentation_profile,
    classId,
  );
  if (requiredNames === null) return null;
  const valueMap = classDocumentationValueMap(mechanics);
  if (
    valueMap === null ||
    Object.keys(valueMap).length !== requiredNames.length ||
    !requiredNames.every((name) => typeof valueMap[name] === "string")
  ) {
    return null;
  }
  return resolveClassDocumentationV1(
    mechanics.documentation_profile,
    classId,
    Object.fromEntries(requiredNames.map((name) => [name, valueMap[name]])),
  );
}

/**
 * Return the resolved Ultimate name/description for a valid supported
 * V2 class-mechanics guide, otherwise null. Reuse the exact class-card guide
 * resolver; no live cooldown or action availability is inferred.
 *
 * @param {unknown} rawClassMechanics
 * @returns {Readonly<{name: string, description: string}> | null}
 */
export function authorizedUltimatePresentationV1(rawClassMechanics) {
  if (!isRecord(rawClassMechanics)) return null;
  return resolvedAuthorizedClassDocumentationV1(rawClassMechanics)?.ultimate ?? null;
}

/**
 * Build full-view rows from a recognized class's complete numeric mechanics.
 * Require matching class name, finite quantities, integer tick fields and
 * known Basic/Ultimate target modes; return null if any required field is
 * missing. Return base rows for health, radii, movement, Basic and regeneration.
 * The caller appends Ultimate/passive rows. This is a display completeness
 * check, not validation of every legal simulator parameter range.
 *
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
    rows.push(row("Base Basic Damage", basicDamage, FULL_ONLY));
  }
  if (/** @type {number} */ (basicHealing) > 0) {
    rows.push(row("Base Basic Healing", basicHealing, FULL_ONLY));
  }
  rows.push(
    row("Out-of-Combat Delay", outOfCombatDelay, FULL_ONLY),
    row(
      "Out-of-Combat Regeneration",
      `${formatDisplayNumber(/** @type {number} */ (regeneration) * 100)}% of maximum health per tick`,
      FULL_ONLY,
    ),
  );
  return rows;
}

/**
 * Build a persistent class card from matching owner and class mechanics.
 * Require a public ID, recognized team/class names and complete mechanics.
 * Return null for invalid joins or an explicit unsupported mechanics version.
 * V2 with an accepted profile adds authored overview, tactical and ability
 * text; historical version-absent records keep numeric mechanics only.
 * Return a descriptor using no current health/status fields and change no input.
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

  const authored = resolvedAuthorizedClassDocumentationV1(mechanics);

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
      row("Base Ultimate Damage", mechanics.ultimate_raw_damage, FULL_ONLY),
    );
  }
  if (/** @type {number} */ (finiteNumber(mechanics.ultimate_raw_healing)) > 0) {
    completeMechanicsRows.push(
      row("Base Ultimate Healing", mechanics.ultimate_raw_healing, FULL_ONLY),
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
 * Build a status descriptor from authorized status, recipient and sources.
 * Copy configured/remaining durations and matching magnitude values. Add the
 * positive-raw-damage break rule only when both catalog semantics and record
 * permit it. audience controls direct-source disclosure through the shared
 * attribution helper; in_combat has no source row. Return a descriptor; this
 * does not apply a status or prove upstream audience authorization.
 *
 * @param {unknown} rawStatus
 * @param {unknown} rawRecipient
 * @param {ReadonlyArray<unknown>} rawSourceAgents
 * @param {"researcher" | "agent_pov"} audience
 */
function explainDurableStatus(rawStatus, rawRecipient, rawSourceAgents, audience) {
  const status = isRecord(rawStatus) ? rawStatus : {};
  const recipient = isRecord(rawRecipient) ? rawRecipient : {};
  const recipientIdentity = exactAuthorizedAgentIdentityV1(recipient);
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
    rows.push(row("Break Rule", "Ends when this agent receives positive raw damage."));
  }
  if (token.tokenId !== "in_combat") {
    const source = authorizedSourceAttributionV1({
      attribution_kind: "direct",
      audience,
      direct_sources: status.direct_sources,
      authorized_agents: rawSourceAgents,
    });
    if (source !== null) {
      rows.push(row(source.label, source.value));
    }
  }
  if (recipientIdentity !== null) {
    rows.push(row("Recipient", recipientIdentity.title));
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
 * Describe an authorized recipient status without direct-source attribution.
 * rawRecipient defaults to {}. Reuse the same recorded duration/magnitude
 * path as researcher cards, with agent_pov and an empty source list. Return
 * a descriptor; caller supplies recipient-authorized status data.
 *
 * @param {unknown} rawStatus
 * @param {unknown} [rawRecipient]
 */
export function explainPovStatus(rawStatus, rawRecipient = {}) {
  return explainDurableStatus(rawStatus, rawRecipient, [], "agent_pov");
}

/**
 * Describe an authorized status for researcher inspection. rawRecipient
 * defaults to {} and rawSourceAgents to []; incomplete exact source joins
 * produce no attribution row. Return the shared durable-status descriptor
 * without assigning contributor credit or changing any record.
 *
 * @param {unknown} rawStatus
 * @param {unknown} [rawRecipient]
 * @param {ReadonlyArray<unknown>} [rawSourceAgents]
 */
export function explainStatus(rawStatus, rawRecipient = {}, rawSourceAgents = []) {
  return explainDurableStatus(rawStatus, rawRecipient, rawSourceAgents, "researcher");
}

/**
 * Describe an aggregate aura modifier for an exact authorized recipient.
 * rawRecipient defaults to {}; missing recipient identity or unreadable
 * modifier returns null. Copy only named data fields and use the recorded
 * multiplier for the effect text. Return a descriptor without attributing
 * an aggregate multiplier to an individual emitter.
 *
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
 * Build a descriptor for items hidden by a compact status/modifier layout.
 * Preserve array order and include each item's compact explanation; an
 * unavailable item keeps its row. kind must be status or modifier. Recipient
 * and source list default to {}/[]. Non-array input becomes an empty list.
 * Hidden here means not drawn in the dock, not scientifically unauthorized;
 * the caller must already have rights to every supplied item.
 *
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
 * Describe compact-layout overflow through the Agent POV status route.
 * rawRecipient defaults to {}; retain only its identity fields and never
 * add direct-source attribution. Preserve item order; non-array input is
 * empty. Return a descriptor. The caller must supply authorized hidden-in-
 * layout items, not hidden simulator facts.
 *
 * @param {ReadonlyArray<unknown>} rawItems
 * @param {unknown} [rawRecipient]
 */
export function explainPovOverflow(rawItems, rawRecipient = {}) {
  const inputRecipient = isRecord(rawRecipient) ? rawRecipient : {};
  const recipient = {
    presentation_key: inputRecipient.presentation_key,
    public_agent_id: inputRecipient.public_agent_id,
    display_agent_id: inputRecipient.display_agent_id,
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
    "Every hidden status remains available in canonical display order.",
    rows.length === 0 ? [row("Hidden Facts", "None")] : rows,
    [],
    { tone: "neutral" },
  );
}

/**
 * Describe a nonnegative integer cooldown only after the record's exact
 * presentation/public identity joins rawOwner. rawOwner defaults to null,
 * which returns null. Prefer ultimate_cooldown_remaining, then the legacy
 * ultimate_cooldown field. Zero displays Ready. Invalid/unjoined values
 * return null; no availability mask or cooldown progression is calculated.
 *
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
      row("Recipient", identity.title),
    ],
    [],
    {
      tone: ticks === 0 ? "positive" : "information",
      accent: identity.accent,
    },
  );
}

/**
 * Describe an authorized aura field's recorded multiplier and radius.
 * rawSourceAgent defaults to null and audience to researcher; unsupported
 * audiences or unreadable fields return null. Source attribution requires
 * exact source identity and matching class accent. Even an agent_pov call
 * may show a supplied, correctly joined researcher source; the caller must
 * authorize that source separately. Return a pointer-anchored descriptor
 * without computing affected agents or overlap.
 *
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
    audience === "agent_pov" && rawSourceAgent === null
      ? null
      : exactJoinedAuthorizedIdentity(rawField, rawSourceAgent, "source_");
  const source =
    sourceIdentity !== null && sourceIdentity.accent === presentation.accent
      ? rawSourceAgent
      : null;
  const attribution = authorizedSourceAttributionV1({
    attribution_kind: "direct",
    audience: source === null ? audience : "researcher",
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
      ...(attribution === null ? [] : [row(attribution.label, attribution.value)]),
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
 * Describe a finite observation, Basic or Ultimate radius after an exact
 * owner identity join. rawOwner defaults to null, so omitted ownership
 * returns null. Reject unknown kinds/missing values with null. Radius units
 * and nonnegative validity belong to the normalized input; this helper
 * formats the value and does not calculate range or target legality.
 *
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
    [row("Radius", formatDisplayNumber(radius)), row("Source", identity.title)],
    [],
    {
      tone: "information",
      accent: identity.accent,
      anchor: "pointer",
    },
  );
}

/**
 * Describe an authorized obstacle's ID and center, plus pillar radius or
 * wall width/height/rotation. Missing fields display Unavailable. Return a
 * pointer-anchored descriptor without reconstructing geometry or checking
 * collision validity. The caller supplies the recorded coordinate units.
 *
 * @param {unknown} rawObstacle
 */
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
 * Describe one recorded boolean availability lane after an exact owner
 * join. lane=0 means Basic and 1 means Ultimate; any other lane returns null.
 * Prefer named ability availability, falling back to lane_N_available only
 * when the named field is undefined. Invalid/nonboolean/unjoined data returns
 * null. This displays a mask fact; it does not recalculate Core acceptance.
 *
 * @param {unknown} rawLegality
 * @param {0 | 1} lane
 * @param {unknown} rawOwner
 * @returns {SemanticDescriptor | null}
 */
export function explainLegality(rawLegality, lane, rawOwner) {
  if (lane !== 0 && lane !== 1) {
    return null;
  }
  const availabilityProperty = lane === 0 ? "basic_available" : "ultimate_available";
  const laneProperty = lane === 0 ? "lane_0_available" : "lane_1_available";
  const legality = snapshotOwnDataFields(rawLegality, [
    "owner_presentation_key",
    "owner_public_agent_id",
    availabilityProperty,
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
  const rawAvailable =
    legality[availabilityProperty] === undefined
      ? legality[laneProperty]
      : legality[availabilityProperty];
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

/**
 * Describe an authorized planned action route, not a physical path.
 * context defaults to {} and may contain separately authorized sourceAgent
 * and recipientAgent identities. Those titles are displayed without an
 * additional route-to-identity join here; the caller owns that join. Missing
 * identity shows Unavailable. Return a pointer-anchored descriptor only.
 *
 * @param {unknown} rawRoute
 * @param {{sourceAgent?: unknown, recipientAgent?: unknown}} [context]
 */
export function explainPendingRoute(rawRoute, context = {}) {
  const route = isRecord(rawRoute) ? rawRoute : {};
  const lane = integer(route.lane);
  const laneName = lane === 0 ? "Basic" : lane === 1 ? "Ultimate" : "Action";
  const source = authorizedAgentIdentityTitle(context.sourceAgent);
  const recipient = authorizedAgentIdentityTitle(context.recipientAgent);
  return descriptor(
    "pending-route",
    `pending:${text(route.source_presentation_key) ?? text(route.source_public_agent_id) ?? "unknown"}:${text(route.target_presentation_key) ?? text(route.target_public_agent_id) ?? "unknown"}:${lane ?? "unknown"}`,
    `${laneName} Action Route`,
    "Authorized action route; no physical path is implied.",
    [
      row("Source", source ?? "Unavailable in this view"),
      row("Recipient", recipient ?? "Unavailable in this view"),
    ],
    [],
    { tone: "information", anchor: "pointer" },
  );
}

/**
 * Describe a normalized activation event using its authorized identities
 * and display token. A matching source class with an accepted V2 guide may
 * provide authored Ultimate text. Supplied exact identities are displayed;
 * redaction flags suppress missing-identity fallback rows rather than erase
 * an explicitly supplied identity. Caller must enforce those input rights.
 * If no target ID is supplied and the recipient is not redacted, its label
 * falls back to the supplied source identity. No damage amount is calculated.
 *
 * @param {unknown} rawEvent
 */
export function explainActivation(rawEvent) {
  const event = isRecord(rawEvent) ? rawEvent : {};
  const token = resolveVisualToken(
    "activation",
    event.tokenId ?? event.token_id,
    event,
  );
  const source = event.sourcePublicAgentId ?? event.source_public_agent_id;
  const target = event.targetPublicAgentId ?? event.target_public_agent_id;
  const component = text(event.component ?? event.ability_component);
  const sourceClassId = integer(event.sourceClassId ?? event.source_class_id);
  const classMechanics = isRecord(
    event.authorizedClassMechanics ?? event.authorized_class_mechanics,
  )
    ? (event.authorizedClassMechanics ?? event.authorized_class_mechanics)
    : null;
  const ultimate =
    component === "ultimate" &&
    sourceClassId !== null &&
    integer(classMechanics?.class_id) === sourceClassId
      ? authorizedUltimatePresentationV1(classMechanics)
      : null;
  const sourceClass = classTokenFromId(sourceClassId);
  const redacted =
    event.targetDisclosure === "redacted" || event.target_disclosure === "redacted";
  const sourceRedacted =
    event.sourceDisclosure === "redacted" || event.source_disclosure === "redacted";
  const sourceIdentity = authorizedAgentIdentityTitle(
    event.sourceIdentity ?? event.source_identity,
  );
  const recipientIdentity = authorizedAgentIdentityTitle(
    event.recipientIdentity ?? event.recipient_identity,
  );
  const rows = [];
  if (sourceIdentity !== null) {
    rows.push(row("Source", sourceIdentity));
  } else if (!sourceRedacted) {
    rows.push(row("Source", "Unavailable in this view"));
  }
  if (recipientIdentity !== null) {
    rows.push(row("Recipient", recipientIdentity));
  } else if (!redacted) {
    rows.push(
      row(
        "Recipient",
        text(target) === null && sourceIdentity !== null
          ? sourceIdentity
          : "Unavailable in this view",
      ),
    );
  }
  return descriptor(
    "activation",
    `activation:${token.tokenId}:${text(event.sourcePresentationKey ?? event.source_presentation_key) ?? text(source) ?? "source-unavailable"}:${text(event.targetPresentationKey ?? event.target_presentation_key) ?? text(target) ?? (redacted ? "target-redacted" : "source-local")}`,
    ultimate === null || sourceClass.label === "Unknown"
      ? token.label
      : `${ultimate.name} (${sourceClass.label} Ultimate Ability)`,
    sentence(ultimate?.description ?? token.accessibleName),
    rows,
    [],
    { tone: "information", anchor: "pointer" },
  );
}

/**
 * Describe a recipient's recorded before/after net-health outcome.
 * Use supplied exact recipient identity and finite netDelta/net_delta; missing
 * values display Unavailable. Return a descriptor with warning tone for a
 * negative delta. This is recipient-level net change, not individual damage
 * source credit or a calculation from simulator state.
 *
 * @param {unknown} rawEvent
 */
export function explainNetHealth(rawEvent) {
  const event = isRecord(rawEvent) ? rawEvent : {};
  const delta = finiteNumber(event.netDelta ?? event.net_delta);
  const recipientIdentity = authorizedAgentIdentityTitle(
    event.recipientIdentity ?? event.recipient_identity,
  );
  return descriptor(
    "impact",
    `net:${text(event.recipientPresentationKey ?? event.recipient_presentation_key) ?? text(event.recipientPublicAgentId ?? event.recipient_public_agent_id) ?? "recipient-unavailable"}:${text(event.outcome) ?? "outcome-unavailable"}`,
    `Recipient NET ${humanize(event.outcome)}`,
    "Recipient-level before/after outcome; not source attribution.",
    [
      row("Recipient", recipientIdentity ?? "Unavailable in this view"),
      row("NET", delta === null ? "Unavailable" : formatNetDelta(delta)),
    ],
    [],
    {
      tone: delta !== null && delta < 0 ? "warning" : "information",
      anchor: "pointer",
    },
  );
}

/**
 * Format a finite delta with a plus sign for gains. Preserve a nonzero
 * sub-display-unit amount as signed <0.01 when normal formatting shows zero.
 * Return text; caller validates finiteness and keeps the exact source value.
 *
 * @param {number} delta
 */
function formatNetDelta(delta) {
  const displayed = formatDisplayNumber(delta);
  if (delta !== 0 && displayed === "0") {
    return `${delta > 0 ? "+" : "−"}<0.01`;
  }
  return `${delta > 0 ? "+" : ""}${displayed}`;
}
