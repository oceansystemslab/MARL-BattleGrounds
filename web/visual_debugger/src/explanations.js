import { formatDisplayNumber } from "./display.js";
import {
  auraPresentation,
  classPresentation,
  statusPresentation,
} from "./semantic-vocabulary.js";
import {
  createSemanticDescriptor,
  projectSemanticDescriptor,
  semanticDescriptorText,
} from "./tooltip.js";
import {
  classTokenFromId,
  resolveVisualToken,
  statusTokenIdFromCatalogId,
  teamTokenFromId,
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

/** @param {unknown} value @returns {value is JsonRecord} */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** @param {unknown} value @returns {JsonRecord[]} */
function records(value) {
  return Array.isArray(value) ? value.filter(isRecord) : [];
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

/**
 * Neutral aggregate aura rows are serialized scientific truth, but they do not
 * describe an active visual effect. Suppress only the exact multiplicative
 * identity at presentation boundaries; near-neutral recorded values remain
 * visible without tolerance or rounding.
 *
 * @param {JsonRecord} modifier
 */
function isNeutralAuraModifier(modifier) {
  return finiteNumber(modifier.multiplier) === 1;
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
  const sourceScope = scope === "field" ? " per recorded emitter" : "";
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
      label: "Damage Effect",
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
 * @param {string} summary
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
 * @param {{controlled?: boolean, selected?: boolean, reference?: boolean, audience?: string}} [selection]
 * @param {unknown} [rawClassMechanics]
 * @param {ReadonlyArray<unknown>} [rawSourceAgents]
 * @returns {SemanticDescriptor}
 */
export function explainAgent(
  rawAgent,
  selection = {},
  rawClassMechanics = null,
  rawSourceAgents = [],
) {
  if (selection.audience === "agent_pov") {
    return explainPovAgent(rawAgent, selection);
  }
  const reducedAudience = selection.audience === "reduced_agent_pov";
  const agent = isRecord(rawAgent) ? rawAgent : {};
  const mechanics = isRecord(rawClassMechanics) ? rawClassMechanics : null;
  const classToken = classTokenFromId(agent.class_id, mechanics ?? agent);
  const teamToken = teamTokenFromId(agent.team_id, agent);
  const publicIdentity = publicAgentLabel(agent.public_agent_id);
  const currentHealth = finiteNumber(agent.current_health);
  const maxHealth = finiteNumber(agent.max_health);
  const effectiveSpeed =
    finiteNumber(agent.effective_movement_speed) ?? finiteNumber(agent.effective_speed);
  const cooldown =
    integer(agent.ultimate_cooldown_remaining) ?? integer(agent.ultimate_cooldown);
  const profile = classPresentation(mechanics?.class_name ?? classToken.label);
  const statuses = records(agent.statuses);
  const modifiers = records(agent.aura_modifiers ?? agent.modifiers).filter(
    (modifier) => !isNeutralAuraModifier(modifier),
  );
  const modifiersAvailable = Array.isArray(agent.aura_modifiers ?? agent.modifiers);
  const spawnShieldRemaining = integer(agent.spawn_shield_remaining);
  const nowRows = [
    row("Identity", publicIdentity),
    row("Class", classToken.label),
    row("Team", teamToken.label),
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
    row("Ultimate Name", profile.ultimateName),
    row(
      "Ultimate Status",
      cooldown === null
        ? "Unavailable"
        : cooldown === 0
          ? "Ready"
          : `On cooldown (${tickCount(cooldown)})`,
    ),
    row(
      "Spawn Shield",
      spawnShieldRemaining === null
        ? "Unavailable"
        : spawnShieldRemaining > 0
          ? `Invulnerable · ${tickCount(spawnShieldRemaining)} remaining`
          : "Inactive",
    ),
  ];
  if (selection.controlled) {
    nowRows.push(row("Selection", "Controlled actor"));
  }
  if (selection.selected) {
    nowRows.push(row("Selection", "Selected target"));
  }
  if (selection.reference) {
    nowRows.push(row("Selection", "Reference"));
  }

  const fullSections = [];
  if (!reducedAudience && mechanics !== null) {
    fullSections.push(
      section(
        "Class Role",
        [
          row("Role", profile.role, FULL_ONLY),
          row("Strengths", profile.strengths, FULL_ONLY),
          row("Limitations", profile.limitations, FULL_ONLY),
          row("Teamwork", profile.teamwork, FULL_ONLY),
          row("Counterplay", profile.counterplay, FULL_ONLY),
        ],
        null,
      ),
      section("Exact Class Mechanics", classMechanicsRows(mechanics), null),
      ...classAuthoredMechanicSections(mechanics),
    );
  }

  const combatCountdown = integer(agent.steps_until_out_of_combat);
  const regenFraction = finiteNumber(
    mechanics?.out_of_combat_health_regeneration_fraction_per_step,
  );
  fullSections.push(
    section(
      "Current State",
      [
        row(
          "Life State",
          text(agent.life_state) !== null
            ? humanize(agent.life_state)
            : typeof agent.alive === "boolean"
              ? agent.alive
                ? "Alive"
                : "Corpse"
              : "Unavailable",
          FULL_ONLY,
        ),
        row(
          "Combat State",
          combatCountdown === null
            ? "Unavailable"
            : combatCountdown > 0
              ? `In combat; ${tickCount(combatCountdown)} until out of combat`
              : "Out of combat",
          FULL_ONLY,
        ),
        row(
          "Out-of-combat Regeneration",
          regenFraction === null
            ? "Unavailable"
            : `${formatDisplayNumber(regenFraction * 100)}% of maximum health per Tick`,
          FULL_ONLY,
        ),
        row("Persistent Statuses", String(statuses.length), FULL_ONLY),
        row(
          "Aggregate Aura Modifiers",
          modifiersAvailable ? String(modifiers.length) : "Unavailable",
          FULL_ONLY,
        ),
      ],
      null,
    ),
    ...currentEffectSections(
      agent,
      statuses,
      modifiers,
      modifiersAvailable,
      rawSourceAgents,
      reducedAudience ? "agent_pov" : "researcher",
    ),
  );
  return descriptor(
    "agent",
    `agent:${integer(agent.global_slot) ?? text(agent.public_agent_id) ?? "unknown"}`,
    `${publicIdentity} · Now`,
    `${classToken.label} on ${teamToken.label}; exact current normalized state.`,
    nowRows,
    fullSections,
    {
      tone: currentHealth === 0 ? "warning" : "information",
      accent: classAccent(agent.class_id),
    },
  );
}

/**
 * Build an agent card through a recipient-authorized field whitelist. Passing
 * researcher mechanics or arbitrary extra fields cannot affect the result.
 *
 * @param {unknown} rawAgent
 * @param {{controlled?: boolean, selected?: boolean}} [selection]
 * @returns {SemanticDescriptor}
 */
export function explainPovAgent(rawAgent, selection = {}) {
  const input = isRecord(rawAgent) ? rawAgent : {};
  const statuses = records(input.statuses).map((status) => ({
    token_id: status.token_id,
    duration: status.duration,
    status_feature_index: status.status_feature_index,
    source_class_id: status.source_class_id,
    source_evidence: status.source_evidence,
  }));
  const reduced = {
    public_agent_id: input.public_agent_id,
    team_id: input.team_id,
    class_id: input.class_id,
    current_health: input.current_health,
    max_health: input.max_health,
    effective_movement_speed: input.effective_movement_speed,
    ultimate_cooldown_remaining: input.ultimate_cooldown_remaining,
    steps_until_out_of_combat: input.steps_until_out_of_combat,
    spawn_shield_remaining: input.spawn_shield_remaining,
    alive: input.alive,
    statuses,
  };
  return explainAgent(
    reduced,
    {
      controlled: selection.controlled,
      selected: selection.selected,
      audience: "reduced_agent_pov",
    },
    null,
  );
}

/**
 * Explain the exact lifecycle input used by the durable cyan shell. This
 * builder copies only agent identity and the serialized shield countdown, so
 * the same descriptor is safe for researcher agents and the authorized POV
 * self row.
 *
 * @param {unknown} rawAgent
 * @returns {SemanticDescriptor}
 */
export function explainSpawnShield(rawAgent) {
  const agent = isRecord(rawAgent) ? rawAgent : {};
  const remaining = integer(agent.spawn_shield_remaining);
  const active = remaining !== null && remaining > 0;
  return descriptor(
    "status",
    `spawn-shield:${text(agent.public_agent_id) ?? "unknown"}`,
    "Spawn Shield",
    active
      ? `This agent is invulnerable while the spawn shield remains active; ${tickCount(remaining)} remain.`
      : "The spawn shield is inactive.",
    [
      row("Agent", publicAgentLabel(agent.public_agent_id)),
      row("Protection", active ? "Invulnerable" : "Inactive"),
      row("Remaining", remaining === null ? "Unavailable" : tickCount(remaining)),
    ],
    [],
    { tone: active ? "positive" : "neutral", accent: "none" },
  );
}

/** @param {unknown} classId */
function classAccent(classId) {
  const accent = classTokenFromId(classId).cssKey;
  return ["mage", "warrior", "hunter", "rogue", "priest"].includes(accent)
    ? accent
    : "none";
}

/** @param {JsonRecord} mechanics */
function classMechanicsRows(mechanics) {
  const profile = classPresentation(mechanics.class_name);
  const rows = [
    row("Maximum Health", exactNumber(mechanics.maximum_health), FULL_ONLY),
    row("Body Radius", exactNumber(mechanics.body_radius), FULL_ONLY),
    row("Base Movement Speed", exactNumber(mechanics.base_movement_speed), FULL_ONLY),
    row("Observation Radius", exactNumber(mechanics.observation_radius), FULL_ONLY),
    row("Basic Target", humanize(mechanics.basic_target_mode), FULL_ONLY),
    row("Basic Radius", exactNumber(mechanics.basic_interaction_radius), FULL_ONLY),
    row("Ultimate Name", profile.ultimateName, FULL_ONLY),
  ];
  addPositiveOutput(rows, "Basic Raw Damage", mechanics.basic_raw_damage);
  addPositiveOutput(rows, "Basic Raw Healing", mechanics.basic_raw_healing);
  rows.push(
    row("Ultimate Target", humanize(mechanics.ultimate_target_mode), FULL_ONLY),
    row(
      "Ultimate Radius",
      exactNumber(mechanics.ultimate_interaction_radius),
      FULL_ONLY,
    ),
    row(
      "Ultimate Catalog Cooldown",
      tickCount(mechanics.ultimate_cooldown_steps),
      FULL_ONLY,
    ),
  );
  addPositiveOutput(rows, "Ultimate Raw Damage", mechanics.ultimate_raw_damage);
  addPositiveOutput(rows, "Ultimate Raw Healing", mechanics.ultimate_raw_healing);
  rows.push(
    row(
      "Out-of-combat Delay",
      tickCount(mechanics.out_of_combat_delay_steps),
      FULL_ONLY,
    ),
    row(
      "Out-of-combat Regeneration",
      finiteNumber(mechanics.out_of_combat_health_regeneration_fraction_per_step) ===
        null
        ? "Unavailable"
        : `${formatDisplayNumber(mechanics.out_of_combat_health_regeneration_fraction_per_step * 100)}% of maximum health per Tick`,
      FULL_ONLY,
    ),
  );
  return rows;
}

/** @param {JsonRecord} mechanics */
function classAuthoredMechanicSections(mechanics) {
  const statusMechanics = records(mechanics.status_mechanics);
  const auraMechanics = records(mechanics.aura_mechanics);
  const sections = [];
  if (statusMechanics.length > 0) {
    const rows = [];
    for (const status of statusMechanics) {
      const token = statusPresentation(
        status.token_id ?? statusTokenIdFromCatalogId(status.status_id),
      );
      const magnitude = statusMagnitudePresentation(
        status.magnitude_kind,
        status.magnitude,
      );
      const values = [`${tickCount(status.duration_steps)}`];
      if (magnitude !== null) values.push(`${magnitude.label}: ${magnitude.value}`);
      values.push(
        status.breaks_on_positive_damage === true
          ? "Breaks on recorded positive damage"
          : "No positive-damage break rule",
      );
      rows.push(row(token.title, values.join(" · "), FULL_ONLY));
    }
    sections.push(section("Authored Status Mechanics", rows));
  }
  if (auraMechanics.length > 0) {
    const rows = [];
    for (const aura of auraMechanics) {
      const presentation = auraPresentation(aura.aura_id);
      const fieldEffect = auraEffectPresentation(
        presentation,
        finiteNumber(aura.per_emitter_multiplier),
        "field",
      );
      rows.push(
        row(
          presentation.fieldTitle,
          [
            `Radius ${formatDisplayNumber(aura.radius)}`,
            fieldEffect,
            `Stacking ${humanize(aura.stacking_rule)}`,
            `${humanize(aura.clamp_kind)} ×${formatDisplayNumber(aura.clamp_value)}`,
          ].join(" · "),
          FULL_ONLY,
        ),
      );
    }
    sections.push(section("Authored Passive Mechanics", rows));
  }
  return sections;
}

/**
 * Compose full-only realized effects from the same agent/root. The compact card
 * remains bounded; every nested fact reuses the canonical status/modifier
 * explainer rather than recalculating it.
 *
 * @param {JsonRecord} agent
 * @param {JsonRecord[]} statuses
 * @param {JsonRecord[]} modifiers
 * @param {boolean} modifiersAvailable
 * @param {ReadonlyArray<unknown>} sourceAgents
 * @param {"researcher" | "agent_pov"} audience
 */
function currentEffectSections(
  agent,
  statuses,
  modifiers,
  modifiersAvailable,
  sourceAgents,
  audience,
) {
  const sections = [];
  if (statuses.length > 0) {
    const rows = statuses.map((status, index) => {
      const explanation =
        audience === "agent_pov"
          ? explainPovStatus(status, agent)
          : explainStatus(status, agent, sourceAgents);
      const full = projectSemanticDescriptor(explanation, "full");
      return row(
        `${index + 1}. ${explanation.title}`,
        semanticDescriptorText(full).join(" · "),
        FULL_ONLY,
      );
    });
    sections.push(section("Current Status Details", rows));
  }
  if (modifiersAvailable && audience === "researcher") {
    const rows =
      modifiers.length === 0
        ? [row("Modifiers", "None", FULL_ONLY)]
        : modifiers.map((modifier, index) => {
            const explanation = explainModifier(modifier, agent);
            const full = projectSemanticDescriptor(explanation, "full");
            return row(
              `${index + 1}. ${explanation.title}`,
              semanticDescriptorText(full).join(" · "),
              FULL_ONLY,
            );
          });
    sections.push(section("Current Aura Modifier Details", rows));
  }
  return sections;
}

/**
 * @param {Array<ReturnType<typeof row>>} rows
 * @param {string} label
 * @param {unknown} value
 */
function addPositiveOutput(rows, label, value) {
  const exact = finiteNumber(value);
  if (exact !== null && exact > 0) {
    rows.push(row(label, formatDisplayNumber(exact), FULL_ONLY));
  }
}

/**
 * Recipient-authorized POV status explanation. Only reduced fields are copied;
 * researcher-only extras on the input are deliberately unreachable.
 *
 * @param {unknown} rawStatus
 * @param {unknown} [_rawRecipient]
 */
export function explainPovStatus(rawStatus, _rawRecipient = {}) {
  const input = isRecord(rawStatus) ? rawStatus : {};
  const reduced = {
    token_id: input.token_id,
    duration: input.duration,
    status_feature_index: input.status_feature_index,
    source_class_id: input.source_class_id,
    source_evidence: input.source_evidence,
  };
  const token = resolveVisualToken("status", reduced.token_id, reduced);
  const profile = statusPresentation(token.tokenId);
  return descriptor(
    "status",
    `pov-status:${integer(reduced.status_feature_index) ?? token.tokenId}`,
    profile.title,
    `${profile.effect} Source agent identity is not disclosed.`,
    [
      row("Duration", tickCount(reduced.duration)),
      row("Source", "Source agent identity is not disclosed."),
    ],
    [],
    { tone: "information", accent: profile.accent },
  );
}

/**
 * @param {unknown} rawStatus
 * @param {unknown} [rawRecipient]
 * @param {ReadonlyArray<unknown>} [rawSourceAgents]
 */
export function explainStatus(rawStatus, rawRecipient = {}, rawSourceAgents = []) {
  const status = isRecord(rawStatus) ? rawStatus : {};
  const recipient = isRecord(rawRecipient) ? rawRecipient : {};
  const token = resolveVisualToken(
    "status",
    status.token_id ?? status.status_id,
    status,
  );
  const profile = statusPresentation(token.tokenId);
  const duration = integer(status.remaining_duration) ?? integer(status.duration);
  const magnitude = statusMagnitudePresentation(
    status.magnitude_kind,
    status.magnitude,
  );
  const magnitudeRows = [
    row("Duration", duration === null ? "Unavailable" : tickCount(duration)),
  ];
  if (magnitude !== null) {
    magnitudeRows.push(row(magnitude.label, magnitude.value));
  }
  if (typeof status.breaks_on_positive_damage === "boolean") {
    magnitudeRows.push(
      row(
        "Break Rule",
        status.breaks_on_positive_damage
          ? "Ends when the recipient takes recorded positive damage."
          : "No positive-damage break rule is recorded.",
      ),
    );
  }

  const joinedSources = joinStatusSources(status, rawSourceAgents);
  const sourceRows =
    joinedSources.length === 0
      ? [row("Source", "Source agent not recorded.")]
      : joinedSources.map((source) =>
          row(
            "Source",
            `${publicAgentLabel(source.public_agent_id)} · ${teamTokenFromId(source.team_id, source).label} · ${classTokenFromId(source.class_id, source).label}`,
          ),
        );
  const breakSummary =
    typeof status.breaks_on_positive_damage !== "boolean"
      ? ""
      : status.breaks_on_positive_damage
        ? " It ends when the recipient takes recorded positive damage."
        : " No positive-damage break rule is recorded.";
  const magnitudeSummary =
    magnitude === null ? "" : ` Exact effect: ${magnitude.value}.`;
  return descriptor(
    "status",
    `status:${integer(recipient.global_slot) ?? text(recipient.public_agent_id) ?? "unknown"}:${integer(status.status_channel) ?? token.tokenId}`,
    profile.title,
    `${profile.effect}${magnitudeSummary}${breakSummary}`,
    magnitudeRows,
    [section("Direct Source", sourceRows, null, COMPACT_AND_FULL)],
    { tone: "information", accent: profile.accent },
  );
}

/**
 * Preserve upstream evidence order. Authorize a display source only when both
 * slot and public ID match one row in the supplied same-scene roster. Repeated
 * event evidence from the same exact agent collapses for presentation only.
 *
 * @param {JsonRecord} status
 * @param {ReadonlyArray<unknown>} rawSourceAgents
 */
function joinStatusSources(status, rawSourceAgents) {
  const roster = records(rawSourceAgents);
  const seen = new Set();
  const joined = [];
  for (const evidence of records(status.direct_source_evidence)) {
    const slot = integer(evidence.source_global_slot);
    const publicId = text(evidence.source_public_agent_id);
    if (slot === null || publicId === null) {
      continue;
    }
    const key = `${slot}\u0000${publicId}`;
    if (seen.has(key)) {
      continue;
    }
    const source = roster.find(
      (agent) =>
        integer(agent.global_slot) === slot && text(agent.public_agent_id) === publicId,
    );
    if (source === undefined) {
      continue;
    }
    seen.add(key);
    joined.push(source);
  }
  return joined;
}

/**
 * @param {unknown} rawModifier
 * @param {unknown} [rawRecipient]
 */
export function explainModifier(rawModifier, rawRecipient = {}) {
  const modifier = isRecord(rawModifier) ? rawModifier : {};
  const recipient = isRecord(rawRecipient) ? rawRecipient : {};
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
    `modifier:${integer(recipient.global_slot) ?? text(recipient.public_agent_id) ?? "unknown"}:${token.tokenId}`,
    presentation.recipientTitle,
    `This recipient has ${effect} from the exact aggregate aura multiplier.`,
    [
      row(
        "Aggregate Multiplier",
        multiplier === null ? "Unavailable" : `×${formatDisplayNumber(multiplier)}`,
      ),
      row("Recipient Effect", effect),
    ],
    [
      section("Source Scope", [
        row(
          "Source",
          "Aggregate modifier; emitter identity is not recorded.",
          FULL_ONLY,
        ),
      ]),
    ],
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
    const compact = projectSemanticDescriptor(explanation, "compact");
    return row(
      `Hidden ${index + 1}`,
      [explanation.title, ...semanticDescriptorText(compact)].join(" · "),
    );
  });
  return descriptor(
    `${kind}-overflow`,
    `${kind}-overflow:${integer(recipient.global_slot) ?? text(recipient.public_agent_id) ?? "unknown"}`,
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
    `pov-status-overflow:${text(recipient.public_agent_id) ?? "unknown"}`,
    `${items.length} Hidden ${items.length === 1 ? "Status" : "Statuses"}`,
    "Every hidden status remains available in canonical display order. Source agent identity is not disclosed.",
    rows.length === 0 ? [row("Hidden Facts", "None")] : rows,
    [],
    { tone: "neutral" },
  );
}

/**
 * @param {unknown} rawRecord
 * @param {unknown} [rawOwner]
 * @param {unknown} [rawClassMechanics]
 */
export function explainCooldown(rawRecord, rawOwner = null, rawClassMechanics = null) {
  const record = isRecord(rawRecord) ? rawRecord : {};
  const owner = isRecord(rawOwner) ? rawOwner : record;
  const mechanics = isRecord(rawClassMechanics) ? rawClassMechanics : {};
  const classToken = classTokenFromId(owner.class_id ?? record.class_id, mechanics);
  const profile = classPresentation(mechanics.class_name ?? classToken.label);
  const ticks =
    integer(record.ultimate_cooldown_remaining) ??
    integer(record.ultimate_cooldown) ??
    integer(owner.ultimate_cooldown_remaining) ??
    integer(owner.ultimate_cooldown);
  return descriptor(
    "cooldown",
    `cooldown:${integer(owner.global_slot) ?? text(owner.public_agent_id) ?? "unknown"}`,
    `${classToken.label} Ultimate: ${profile.ultimateName} Cooldown`,
    ticks === 0
      ? `${profile.ultimateName} is ready.`
      : `Exact current cooldown remaining for ${profile.ultimateName}.`,
    [
      row("Cooldown Remaining", ticks === null ? "Unavailable" : tickCount(ticks)),
      row("Source", publicAgentLabel(owner.public_agent_id)),
    ],
    [],
    {
      tone: ticks === 0 ? "positive" : "information",
      accent: classAccent(owner.class_id ?? record.class_id),
    },
  );
}

/**
 * @param {unknown} rawField
 * @param {unknown} [rawSourceAgent]
 */
export function explainAura(rawField, rawSourceAgent = null) {
  const field = isRecord(rawField) ? rawField : {};
  const candidateSource = isRecord(rawSourceAgent) ? rawSourceAgent : null;
  const fieldSlot = integer(field.source_global_slot);
  const fieldPublicId = text(field.source_public_agent_id);
  const sourceAgent =
    candidateSource !== null &&
    fieldSlot !== null &&
    fieldPublicId !== null &&
    integer(candidateSource.global_slot) === fieldSlot &&
    text(candidateSource.public_agent_id) === fieldPublicId
      ? candidateSource
      : null;
  const token = resolveVisualToken("modifier", field.token_id ?? field.aura_id, field);
  const presentation = auraPresentation(field.aura_id ?? field.token_id);
  const multiplier = finiteNumber(field.per_emitter_multiplier);
  const effect = auraEffectPresentation(presentation, multiplier, "field");
  const sourcePublicId = sourceAgent?.public_agent_id;
  return descriptor(
    "aura",
    `aura:${integer(field.source_global_slot) ?? text(sourcePublicId) ?? "unknown"}:${token.tokenId}`,
    presentation.fieldTitle,
    `Catalog-declared same-team aura capability: allies within this recorded radius may receive ${effect}. Consult each recipient's exact aggregate modifier for realized effect.`,
    [
      row("Source ID", publicAgentLabel(sourcePublicId)),
      row(
        "Source Class",
        sourceAgent === null
          ? "Unavailable"
          : classTokenFromId(sourceAgent.class_id, sourceAgent).label,
      ),
      row("Radius", exactNumber(field.radius)),
      row(
        "Catalog Multiplier",
        multiplier === null ? "Unavailable" : `×${formatDisplayNumber(multiplier)}`,
      ),
      row("Catalog Effect", effect),
    ],
    [
      section("Field Contract", [
        row("Beneficiary", humanize(field.beneficiary_relation), FULL_ONLY),
        row("Stacking", humanize(field.stacking_rule), FULL_ONLY),
        row(
          "Clamp",
          `${humanize(field.clamp_kind)} ${exactNumber(field.clamp_value)}`,
          FULL_ONLY,
        ),
        row("Center", point(field.center), FULL_ONLY),
      ]),
    ],
    {
      tone: "information",
      accent: presentation.accent,
      anchor: "pointer",
    },
  );
}

const RANGE_PURPOSE = Object.freeze({
  observation: "Shows the exact radius used for the recorded observation range.",
  basic: "Shows the exact radius for the class's Basic interaction.",
  ultimate: "Shows the exact radius for the class's Ultimate interaction.",
});

/**
 * @param {unknown} rawRange
 * @param {unknown} [rawOwner]
 * @param {unknown} [rawClassMechanics]
 */
export function explainRange(rawRange, rawOwner = null, rawClassMechanics = null) {
  const range = isRecord(rawRange) ? rawRange : {};
  const candidateOwner = isRecord(rawOwner) ? rawOwner : null;
  const rangeSlot = integer(range.global_slot);
  const owner =
    candidateOwner !== null &&
    rangeSlot !== null &&
    integer(candidateOwner.global_slot) === rangeSlot &&
    text(candidateOwner.public_agent_id) !== null
      ? candidateOwner
      : {};
  const candidateMechanics = isRecord(rawClassMechanics) ? rawClassMechanics : null;
  const mechanics =
    candidateMechanics !== null &&
    integer(candidateMechanics.class_id) !== null &&
    integer(candidateMechanics.class_id) === integer(owner.class_id)
      ? candidateMechanics
      : {};
  const kind = text(range.kind) ?? "unknown";
  return descriptor(
    `range-${kind}`,
    `range:${integer(range.global_slot) ?? text(owner.public_agent_id) ?? "unknown"}:${kind}`,
    `${humanize(kind)} Range`,
    RANGE_PURPOSE[/** @type {keyof typeof RANGE_PURPOSE} */ (kind)] ??
      "Shows an exact normalized range radius.",
    [
      row("Radius", exactNumber(range.radius)),
      row("Owner ID", publicAgentLabel(owner.public_agent_id)),
      row("Team", teamTokenFromId(owner.team_id, owner).label),
      row(
        "Class",
        text(mechanics.class_name) ?? classTokenFromId(owner.class_id, owner).label,
      ),
    ],
    [],
    {
      tone: "information",
      accent: classAccent(owner.class_id),
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
    "Privileged researcher diagnostic copied from the normalized scene.",
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
 */
export function explainLegality(rawLegality, lane) {
  const legality = isRecord(rawLegality) ? rawLegality : {};
  const laneName = lane === 0 ? "Basic" : "Ultimate";
  const rawAvailable =
    lane === 0 ? legality.lane_0_available : legality.lane_1_available;
  if (typeof rawAvailable !== "boolean") {
    throw new TypeError(`${laneName} legality must be an exact boolean.`);
  }
  return descriptor(
    "legality",
    `legality:${lane}:${rawAvailable}`,
    `${laneName} Legality`,
    `${laneName} ability is ${rawAvailable ? "" : "not "}available this tick.`,
    [row("Status", rawAvailable ? "True" : "False")],
    [],
    { tone: rawAvailable ? "positive" : "warning" },
  );
}

/** @param {unknown} rawRoute */
export function explainPendingRoute(rawRoute) {
  const route = isRecord(rawRoute) ? rawRoute : {};
  const lane = integer(route.lane);
  const laneName = lane === 0 ? "Basic" : lane === 1 ? "Ultimate" : "Action";
  return descriptor(
    "pending-route",
    `pending:${text(route.source_public_agent_id) ?? "unknown"}:${text(route.target_public_agent_id) ?? "unknown"}:${lane ?? "unknown"}`,
    `${laneName} Action Route`,
    "Currently selected action intent; no physical path is implied.",
    [
      row("Source", publicAgentLabel(route.source_public_agent_id)),
      row("Selected Target", publicAgentLabel(route.target_public_agent_id)),
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
    `activation:${token.tokenId}:${text(source) ?? "source-unavailable"}:${text(target) ?? (redacted ? "target-redacted" : "source-local")}`,
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
    `net:${text(event.recipientPublicAgentId ?? event.recipient_public_agent_id) ?? "recipient-unavailable"}:${text(event.outcome) ?? "outcome-unavailable"}`,
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
