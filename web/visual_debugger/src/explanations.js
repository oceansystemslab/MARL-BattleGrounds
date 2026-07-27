import { formatDisplayNumber } from "./display.js";
import {
  classTokenFromId,
  resolveVisualToken,
  teamTokenFromId,
  ultimateTokenFromClassId,
} from "./vocabulary.js";

/**
 * Tooltip prose builders consume only allowlisted fields already present in an
 * authorized frame. They format and label those facts; they do not infer
 * legality, combat effects, visibility, or hidden endpoints.
 */

/** @typedef {Record<string, any>} JsonRecord */
/**
 * @typedef {{
 *   kind: string,
 *   id: string,
 *   title: string,
 *   details: string[],
 *   anchor: "element" | "pointer",
 * }} TooltipDescriptor
 */

/**
 * @param {unknown} value
 * @returns {value is JsonRecord}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * @param {unknown} value
 * @returns {number | null}
 */
function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * @param {unknown} value
 * @returns {number | null}
 */
function integer(value) {
  return Number.isInteger(value) ? Number(value) : null;
}

/**
 * @param {unknown} value
 */
function humanize(value) {
  return typeof value === "string" && value.trim()
    ? value
        .trim()
        .replaceAll("_", " ")
        .replace(/\b\w/g, (character) => character.toUpperCase())
    : "Unknown";
}

/**
 * @param {string} kind
 * @param {string} id
 * @param {string} title
 * @param {string[]} details
 * @param {"element" | "pointer"} [anchor]
 * @returns {TooltipDescriptor}
 */
function descriptor(kind, id, title, details, anchor = "element") {
  return { kind, id, title, details, anchor };
}

/**
 * @param {unknown} rawAgent
 * @param {{controlled?: boolean, selected?: boolean}} [selection]
 */
export function explainAgent(rawAgent, selection = {}) {
  const agent = isRecord(rawAgent) ? rawAgent : {};
  const slot = integer(agent.global_slot);
  const classToken = classTokenFromId(agent.class_id, agent);
  const teamToken = teamTokenFromId(agent.team_id, agent);
  const currentHealth = finiteNumber(agent.current_health);
  const maxHealth = finiteNumber(agent.max_health);
  const effectiveSpeed = finiteNumber(agent.effective_speed);
  const cooldown = integer(agent.ultimate_cooldown);
  const details = [
    teamToken.label,
    currentHealth === null || maxHealth === null
      ? "Health unavailable"
      : `Health ${formatDisplayNumber(currentHealth)} / ${formatDisplayNumber(maxHealth)}`,
    effectiveSpeed === null
      ? "Effective speed unavailable"
      : `Effective speed ${formatDisplayNumber(effectiveSpeed)}`,
    cooldown === null
      ? "Ultimate cooldown unavailable"
      : cooldown > 0
        ? `Ultimate cooldown ${cooldown} ${cooldown === 1 ? "tick" : "ticks"}`
        : "Ultimate ready",
  ];
  if (selection.controlled) {
    details.push("Controlled actor");
  }
  if (selection.selected) {
    details.push("Selected target");
  }
  return descriptor(
    "agent",
    `agent:${slot ?? "unknown"}`,
    `${slot === null ? "Unknown agent" : `id_${slot}`} · ${classToken.label}`,
    details,
  );
}

/**
 * @param {unknown} rawStatus
 * @param {number} slot
 */
export function explainStatus(rawStatus, slot) {
  const status = isRecord(rawStatus) ? rawStatus : {};
  const token = resolveVisualToken("status", status.token_id, status);
  const source = classTokenFromId(status.source_class_id);
  const duration = integer(status.duration);
  return descriptor("status", `status:${slot}:${token.tokenId}`, token.label, [
    token.accessibleName,
    `Source class ${source.label}`,
    duration === null
      ? "Duration unavailable"
      : `Duration ${duration} ${duration === 1 ? "tick" : "ticks"}`,
    `Recipient id_${slot}`,
  ]);
}

/**
 * @param {unknown} rawModifier
 * @param {number} slot
 */
export function explainModifier(rawModifier, slot) {
  const modifier = isRecord(rawModifier) ? rawModifier : {};
  const token = resolveVisualToken("modifier", modifier.token_id, modifier);
  const multiplier = finiteNumber(modifier.multiplier);
  return descriptor("modifier", `modifier:${slot}:${token.tokenId}`, token.label, [
    token.accessibleName,
    multiplier === null
      ? "Multiplier unavailable"
      : `Multiplier ×${formatDisplayNumber(multiplier)}`,
    `Recipient id_${slot}`,
  ]);
}

/**
 * Explain every item hidden behind one neutral overflow cell.
 *
 * @param {ReadonlyArray<unknown>} rawItems
 * @param {"status" | "modifier"} kind
 * @param {number} slot
 */
export function explainOverflow(rawItems, kind, slot) {
  const details = rawItems.map((rawItem) => {
    const item = isRecord(rawItem) ? rawItem : {};
    const token = resolveVisualToken(kind, item.token_id, item);
    if (kind === "status") {
      const source = classTokenFromId(item.source_class_id);
      const duration = integer(item.duration);
      return [
        token.label,
        token.accessibleName,
        `source ${source.label}`,
        duration === null
          ? "duration unavailable"
          : `${duration} ${duration === 1 ? "tick" : "ticks"}`,
      ].join(" · ");
    }
    const multiplier = finiteNumber(item.multiplier);
    return [
      token.label,
      token.accessibleName,
      multiplier === null
        ? "multiplier unavailable"
        : `×${formatDisplayNumber(multiplier)}`,
    ].join(" · ");
  });
  return descriptor(
    `${kind}-overflow`,
    `${kind}-overflow:${slot}`,
    `id_${slot} · ${rawItems.length} hidden ${kind === "status" ? "statuses" : "modifiers"}`,
    details.length === 0 ? ["No hidden items"] : details,
  );
}

/**
 * @param {unknown} rawAgent
 */
export function explainCooldown(rawAgent) {
  const agent = isRecord(rawAgent) ? rawAgent : {};
  const slot = integer(agent.global_slot);
  const ticks = integer(agent.ultimate_cooldown);
  const token = ultimateTokenFromClassId(agent.class_id);
  return descriptor(
    "cooldown",
    `cooldown:${slot ?? "unknown"}`,
    `${token.label} cooldown`,
    [
      ticks === null
        ? "Remaining duration unavailable"
        : `${ticks} ${ticks === 1 ? "tick" : "ticks"} remaining`,
      slot === null ? "Agent unavailable" : `Agent id_${slot}`,
    ],
  );
}

/**
 * @param {unknown} rawField
 */
export function explainAura(rawField) {
  const field = isRecord(rawField) ? rawField : {};
  const slot = integer(field.source_global_slot);
  const token = resolveVisualToken("modifier", field.token_id, field);
  const radius = finiteNumber(field.radius);
  return descriptor(
    "aura",
    `aura:${slot ?? "unknown"}:${token.tokenId}`,
    token.label,
    [
      token.accessibleName,
      radius === null
        ? "Radius unavailable"
        : `Public field radius ${formatDisplayNumber(radius)}`,
      slot === null ? "Source unavailable" : `Source id_${slot}`,
    ],
    "pointer",
  );
}

/**
 * @param {unknown} rawRange
 */
export function explainRange(rawRange) {
  const range = isRecord(rawRange) ? rawRange : {};
  const slot = integer(range.global_slot);
  const kind = typeof range.kind === "string" ? range.kind : "unknown";
  const radius = finiteNumber(range.radius);
  return descriptor(
    `range-${kind}`,
    `range:${slot ?? "unknown"}:${kind}`,
    `${humanize(kind)} range`,
    [
      radius === null ? "Radius unavailable" : `Radius ${formatDisplayNumber(radius)}`,
      slot === null ? "Owner unavailable" : `Owner id_${slot}`,
    ],
    "pointer",
  );
}

/**
 * @param {unknown} rawObstacle
 */
export function explainObstacle(rawObstacle) {
  const obstacle = isRecord(rawObstacle) ? rawObstacle : {};
  const obstacleId =
    typeof obstacle.obstacle_id === "string" ? obstacle.obstacle_id : "unknown";
  const details = [`Kind ${humanize(obstacle.kind)}`];
  for (const [label, rawValue] of [
    ["Radius", obstacle.radius],
    ["Width", obstacle.width],
    ["Height", obstacle.height],
  ]) {
    const value = finiteNumber(rawValue);
    if (value !== null) {
      details.push(`${label} ${formatDisplayNumber(value)}`);
    }
  }
  return descriptor(
    "obstacle",
    `obstacle:${obstacleId}`,
    `Obstacle ${obstacleId}`,
    details,
    "pointer",
  );
}

/**
 * @param {unknown} rawFact
 */
export function explainVisibility(rawFact) {
  const fact = isRecord(rawFact) ? rawFact : {};
  const observer = integer(fact.observer_global_slot);
  const candidate = integer(fact.candidate_global_slot);
  const visible = typeof fact.visible === "boolean" ? fact.visible : null;
  return descriptor(
    "visibility",
    `visibility:${observer ?? "unknown"}:${candidate ?? "unknown"}`,
    "Observer visibility",
    [
      observer === null ? "Observer unavailable" : `Observer id_${observer}`,
      candidate === null ? "Candidate unavailable" : `Candidate id_${candidate}`,
      visible === null
        ? "Visibility unavailable"
        : visible
          ? "Candidate visible"
          : "Candidate hidden",
      "Privileged researcher diagnostic",
    ],
  );
}

/**
 * @param {unknown} rawLegality
 * @param {0 | 1} lane
 */
export function explainLegality(rawLegality, lane) {
  const legality = isRecord(rawLegality) ? rawLegality : {};
  const targetSlot = integer(legality.target_global_slot);
  const laneName = lane === 0 ? "Basic" : "Ultimate";
  const available =
    lane === 0
      ? legality.lane_0_available === true
      : legality.lane_1_available === true;
  const armed = legality.armed_lane === lane;
  const details = [
    `Exact Python mask value ${available}`,
    armed ? "Currently armed" : "Not armed",
  ];
  if (armed && typeof legality.armed_pair_legal === "boolean") {
    details.push(`Staged pair legal ${legality.armed_pair_legal}`);
  }
  if (targetSlot !== null) {
    details.push(`Target id_${targetSlot}`);
  }
  return descriptor(
    "legality",
    `legality:${targetSlot ?? "unknown"}:${lane}`,
    `${laneName} legality`,
    details,
  );
}

/**
 * @param {unknown} rawRoute
 */
export function explainPendingRoute(rawRoute) {
  const route = isRecord(rawRoute) ? rawRoute : {};
  const source = integer(route.source_global_slot);
  const target = integer(route.target_global_slot);
  const lane = integer(route.lane);
  return descriptor(
    "pending-route",
    `pending:${source ?? "unknown"}:${target ?? "unknown"}:${lane ?? "unknown"}`,
    "Pending action route",
    [
      source === null ? "Source unavailable" : `Source id_${source}`,
      target === null ? "Target unavailable" : `Target id_${target}`,
      lane === 0 ? "Basic lane" : lane === 1 ? "Ultimate lane" : "Lane unavailable",
      `Exact staged pair legal ${route.legal === true}`,
    ],
    "pointer",
  );
}

/**
 * @param {unknown} rawEvent
 */
export function explainActivation(rawEvent) {
  const event = isRecord(rawEvent) ? rawEvent : {};
  const token = resolveVisualToken(
    "activation",
    event.tokenId ?? event.token_id,
    event,
  );
  const source = integer(event.sourceSlot ?? event.source_global_slot);
  const target = integer(event.targetSlot ?? event.target_global_slot);
  const details = [
    token.accessibleName,
    source === null ? "Source not spatially disclosed" : `Source id_${source}`,
  ];
  if (target !== null) {
    details.push(`Target id_${target}`);
  } else if (
    event.targetDisclosure === "redacted" ||
    event.target_disclosure === "redacted"
  ) {
    details.push("Target endpoint not disclosed in this view");
  } else {
    details.push("Source-local activation");
  }
  details.push("No per-source health amount is available");
  return descriptor(
    "activation",
    `activation:${event.eventId ?? event.event_id ?? "unknown"}`,
    token.label,
    details,
    "pointer",
  );
}

/**
 * @param {unknown} rawEvent
 */
export function explainNetHealth(rawEvent) {
  const event = isRecord(rawEvent) ? rawEvent : {};
  const recipient = integer(event.recipientSlot ?? event.recipient_global_slot);
  const delta = finiteNumber(event.netDelta ?? event.net_delta);
  const outcome =
    typeof event.outcome === "string" ? humanize(event.outcome) : "Unknown";
  return descriptor(
    "impact",
    `net:${event.eventId ?? event.event_id ?? "unknown"}`,
    `Recipient NET ${outcome}`,
    [
      recipient === null ? "Recipient unavailable" : `Recipient id_${recipient}`,
      delta === null ? "NET health change unavailable" : `NET ${formatNetDelta(delta)}`,
      "Recipient-level before/after outcome; not source attribution",
    ],
    "pointer",
  );
}

/**
 * Preserve the direction of a non-zero health delta that rounds below the
 * two-decimal display threshold without printing a misleading signed zero.
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
