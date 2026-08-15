import {
  authorizedPresentationAudience,
  authorizedPresentationIdentityRows,
  authorizedPresentationIncomingRows,
  authorizedPresentationInspectionState,
  authorizedPresentationSceneView,
  authorizedPresentationTechnicalFacts,
  authorizedPresentationTransitionRows,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { formatCompactDisplayNumber, formatDisplayNumber } from "./display.js";
import {
  explainAgent,
  explainLegality,
  explainModifier,
  explainPendingRoute,
  explainPovAgent,
  explainPovStatus,
  explainStatus,
} from "./explanations.js";
import { createSvgIcon } from "./icons.js";
import {
  createSemanticDescriptor,
  registerTooltipOwner,
  renderSemanticDescriptor,
} from "./tooltip.js";
import { classTokenFromId, resolveVisualToken, teamTokenFromId } from "./vocabulary.js";

/**
 * @typedef {{
 *   roster: HTMLElement,
 *   rosterCount: HTMLElement,
 *   selectionCard: HTMLElement,
 *   pendingHeading: HTMLElement,
 *   pendingCount: HTMLElement,
 *   pendingScope: HTMLElement,
 *   pendingCard: HTMLElement,
 *   acceptedCard: HTMLElement,
 *   acceptedAnnouncement: HTMLElement,
 *   eventFeed: HTMLElement,
 *   eventCount: HTMLElement,
 *   diagnosticsCard: HTMLElement,
 *   onCommand: (command: Record<string, unknown>) => void | Promise<void>,
 * }} DebuggerPanelBindings
 */

/**
 * @typedef {{
 *   busy?: boolean,
 *   shuttingDown?: boolean,
 *   resyncRequired?: boolean,
 *   offline?: boolean,
 * }} PanelInteractionState
 */

/**
 * @typedef {{
 *   element: HTMLElement,
 *   identityId: HTMLElement,
 *   identityClass: HTMLElement,
 *   health: HTMLElement,
 *   statuses: HTMLElement,
 *   modifiers: HTMLElement,
 *   targetButton: HTMLButtonElement,
 *   controlButton: HTMLButtonElement,
 * }} RosterRow
 */

/**
 * @typedef {{
 *   element: HTMLElement,
 *   count: HTMLElement,
 *   rows: HTMLElement,
 *   empty: HTMLElement,
 * }} RosterTeamGroup
 */

/**
 * @typedef {{
 *   element: HTMLElement,
 * }} PendingActionRow
 */

/**
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Neutral aggregate modifiers remain valid wire truth, but they are visual
 * noise. Filter exact ×1 rows only at this browser presentation boundary.
 *
 * @param {Record<string, any>} agent
 */
function agentForPresentation(agent) {
  const key = Array.isArray(agent.aura_modifiers)
    ? "aura_modifiers"
    : Array.isArray(agent.modifiers)
      ? "modifiers"
      : null;
  if (key === null) {
    return agent;
  }
  return {
    ...agent,
    [key]: agent[key].filter(
      (/** @type {unknown} */ modifier) =>
        !isRecord(modifier) || modifier.multiplier !== 1,
    ),
  };
}

/**
 * @param {unknown} value
 * @returns {any[]}
 */
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/**
 * @param {unknown} frame
 * @returns {Record<string, any> | null}
 */
function frameScene(frame) {
  if (!isRecord(frame)) {
    return null;
  }
  return isRecord(frame.scene) ? frame.scene : null;
}

export const DISCLOSURE_PANEL_IDS = Object.freeze([
  "command-deck",
  "roster-details",
  "agent-details",
  "pending-turn-details",
  "latest-transition-details",
  "events-details",
  "visual-key",
  "technical-frame-details",
]);

/**
 * Build the disclosure reset epoch without revisions, transition IDs, or
 * replay cursors. Ordinary authoritative refreshes therefore preserve native
 * details state; only scenario/artifact or audience authority changes reset it.
 *
 * @param {unknown} rawFrame
 */
export function panelDisclosureAuthorityKey(rawFrame) {
  const frame = isRecord(rawFrame) ? rawFrame : null;
  if (frame === null) {
    return null;
  }
  if (isAuthorizedPresentationFrame(frame)) {
    return [
      frame.viewer_mode,
      frame.session_id,
      frame.episode_id,
      frame.presentation_kind,
      frame.recipient_presentation_key ?? "researcher",
    ].join(":");
  }
  const scene = frameScene(frame);
  const replay = frame.viewer_mode === "replay";
  const reference = isRecord(frame.artifact_summary?.replay_reference)
    ? frame.artifact_summary.replay_reference
    : {};
  const scenario = isRecord(frame.scenario) ? frame.scenario : {};
  const episode = replay
    ? (reference.canonical_digest_sha256 ??
      reference.artifact_id ??
      frame.timeline_id ??
      "unknown-artifact")
    : (scenario.name ?? frame.scenario_name ?? "unknown-scenario");
  const audience =
    frame.replay_audience ?? scene?.audience ?? frame.view_mode ?? "unknown-audience";
  const recipient =
    audience === "researcher"
      ? "researcher"
      : (frame.pov_global_slot ??
        frame.selected_global_slot ??
        frame.public_agent_id ??
        asArray(scene?.agents)[0]?.public_agent_id ??
        "recipient");
  return `${replay ? "replay" : "live"}:${episode}:${audience}:${recipient}`;
}

/**
 * @param {string} panelId
 * @param {boolean} replay
 */
export function disclosurePanelInitiallyOpen(panelId, replay) {
  if (!DISCLOSURE_PANEL_IDS.includes(panelId)) {
    throw new RangeError(`Unknown disclosure panel ${panelId}.`);
  }
  return panelId === "roster-details" || (!replay && panelId === "command-deck");
}

/**
 * @param {unknown} frame
 * @returns {Record<string, any> | null}
 */
function frameEvents(frame) {
  if (!isRecord(frame)) {
    return null;
  }
  return isRecord(frame.event_batch) ? frame.event_batch : null;
}

/**
 * Build one exact same-root global-slot to public-ID join. A conflict between
 * scene and event-batch roots fails closed for that slot.
 *
 * @param {unknown} frame
 * @returns {ReadonlyMap<number, string>}
 */
export function publicAgentIdMap(frame) {
  const normalizedFrame = isRecord(frame) ? frame : {};
  const scene = isRecord(normalizedFrame.scene) ? normalizedFrame.scene : {};
  const batch = isRecord(normalizedFrame.event_batch)
    ? normalizedFrame.event_batch
    : {};
  /** @type {Map<number, string>} */
  const joined = new Map();
  const conflicts = new Set();
  /** @param {number} slot @param {unknown} rawPublicId */
  const accept = (slot, rawPublicId) => {
    const publicId =
      typeof rawPublicId === "string" && rawPublicId.trim() ? rawPublicId.trim() : null;
    if (!Number.isInteger(slot) || publicId === null || conflicts.has(slot)) {
      return;
    }
    const existing = joined.get(slot);
    if (existing !== undefined && existing !== publicId) {
      joined.delete(slot);
      conflicts.add(slot);
      return;
    }
    joined.set(slot, publicId);
  };
  for (const agent of asArray(scene.agents).filter(isRecord)) {
    accept(agent.global_slot, agent.public_agent_id);
  }
  for (const [slot, publicId] of asArray(
    batch.public_agent_id_by_global_slot,
  ).entries()) {
    accept(slot, publicId);
  }
  return joined;
}

/**
 * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} resolver
 * @param {unknown} slot
 * @returns {string | null}
 */
function resolvePublicAgentId(resolver, slot) {
  if (!Number.isInteger(slot)) {
    return null;
  }
  const value =
    typeof resolver === "function"
      ? resolver(Number(slot))
      : (resolver.get(Number(slot)) ?? null);
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/**
 * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} resolver
 * @param {unknown} slot
 */
function agentLabelForSlot(resolver, slot) {
  const publicId = resolvePublicAgentId(resolver, slot);
  return publicId === null ? "Agent ID unavailable" : `Agent ID ${publicId}`;
}

/**
 * Build finite help copy for one roster action. Registering this descriptor on
 * the button itself prevents focus/hover from inheriting the surrounding agent
 * card while leaving the native button action intact.
 *
 * @param {"target" | "control"} role
 * @param {unknown} publicAgentId
 * @param {"live" | "researcher_replay"} mode
 * @param {boolean} disabled
 */
export function rosterControlDescriptor(role, publicAgentId, mode, disabled) {
  if (role !== "target" && role !== "control") {
    throw new TypeError("roster control role must be target or control.");
  }
  if (mode !== "live" && mode !== "researcher_replay") {
    throw new TypeError("roster control mode must be live or researcher_replay.");
  }
  if (typeof disabled !== "boolean") {
    throw new TypeError("roster control disabled state must be a boolean.");
  }
  const identity =
    typeof publicAgentId === "string" && publicAgentId.trim()
      ? publicAgentId.trim()
      : "unavailable";
  const replay = mode === "researcher_replay";
  const title = replay
    ? role === "target"
      ? "Reference"
      : "POV actor"
    : role === "target"
      ? "Target"
      : "Control";
  const enabledSummary = replay
    ? role === "target"
      ? "Selects this agent as the replay reference for inspection and highlighting; it does not change the immutable range anchor."
      : "Selects this agent's recorded point of view for replay inspection."
    : role === "target"
      ? "Selects this agent as the target while editing the staged action."
      : "Selects this agent as the controlled actor for staged action editing.";
  return createSemanticDescriptor({
    kind: "control",
    id: `roster-control:${mode}:${role}:${identity}:${disabled ? "disabled" : "enabled"}`,
    title,
    tone: disabled ? "warning" : "information",
    accent: "none",
    summary: disabled ? "This control is currently unavailable." : enabledSummary,
    rows: [
      {
        label: "Agent",
        value: `Agent ID ${identity}`,
        metadata: { compact: true, full: true },
      },
      {
        label: "Availability",
        value: disabled ? "Unavailable" : "Available",
        metadata: { compact: true, full: true },
      },
    ],
    sections: [],
    metadata: { compact: true, full: false },
    anchor: "element",
  });
}

/**
 * Render the full projection into an existing persistent inspector container.
 * The caller owns pane visibility, close behavior, and focus return.
 *
 * @param {HTMLElement} container
 * @param {unknown} descriptor
 */
export function renderSemanticInspector(container, descriptor) {
  const title = htmlElement("span", "sr-only");
  const details = htmlElement("div", "semantic-inspector__details");
  container.replaceChildren(title, details);
  return renderSemanticDescriptor({
    descriptor,
    title,
    details,
    surface: "full",
  });
}

/**
 * @param {unknown} descriptor
 * @param {string} className
 */
function semanticPanelCard(descriptor, className) {
  const card = htmlElement("article", className);
  const title = htmlElement("h3");
  const details = htmlElement("div", "semantic-panel-card__details");
  card.append(title, details);
  renderSemanticDescriptor({ descriptor, title, details, surface: "full" });
  return card;
}

/**
 * @param {unknown} value
 */
function humanize(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

/**
 * @template {keyof HTMLElementTagNameMap} K
 * @param {K} tagName
 * @param {string | null} className
 * @param {string | null} text
 * @returns {HTMLElementTagNameMap[K]}
 */
function htmlElement(tagName, className = null, text = null) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== null) {
    element.textContent = text;
  }
  return element;
}

/**
 * @param {HTMLElement} container
 * @param {string} label
 * @param {unknown} value
 */
function addFact(container, label, value) {
  const fact = htmlElement("div", "fact");
  fact.append(
    htmlElement("span", null, label),
    htmlElement("strong", null, String(value)),
  );
  container.append(fact);
}

/**
 * Keep roster status values compact while leaving exact duration truth in the
 * owning chip's data, accessible label, and tooltip.
 *
 * @param {unknown} duration
 */
export function rosterStatusDurationLabel(duration) {
  return Number.isInteger(duration) ? formatCompactDisplayNumber(duration) : "?";
}

/**
 * Render every authorized status/modifier record without sorting, merging, or
 * deriving mechanics.
 *
 * @param {HTMLElement} container
 * @param {unknown[]} items
 * @param {"status" | "modifier"} kind
 * @param {string} emptyText
 * @param {Record<string, any>} recipient
 * @param {ReadonlyArray<unknown>} sourceAgents
 * @param {"researcher" | "agent_pov"} audience
 */
function renderFactTokens(
  container,
  items,
  kind,
  emptyText,
  recipient,
  sourceAgents,
  audience,
) {
  const nodes = [];
  const authorizedItems =
    audience === "agent_pov" && kind === "modifier"
      ? []
      : kind === "modifier"
        ? items.filter((item) => !isRecord(item) || item.multiplier !== 1)
        : items;
  for (const rawItem of authorizedItems) {
    const item = isRecord(rawItem) ? rawItem : {};
    const token = resolveVisualToken(
      kind,
      item.token_id,
      audience === "agent_pov" && kind === "status" ? undefined : item,
    );
    const value =
      kind === "status"
        ? `duration ${Number.isInteger(item.duration) ? item.duration : "unknown"}`
        : `multiplier ${formatDisplayNumber(item.multiplier)}`;
    const chip = htmlElement("span", `roster-fact-token roster-fact-token--${kind}`);
    chip.dataset.tokenId = token.tokenId;
    if (kind === "status") {
      const displayDuration = rosterStatusDurationLabel(item.duration);
      const sourceClass = classTokenFromId(item.source_class_id);
      const icon = createSvgIcon(container.ownerDocument, token.glyphKey, {
        className: "roster-fact-token__icon",
      });
      const durationValue = htmlElement(
        "span",
        "roster-fact-token__duration",
        String(displayDuration),
      );
      chip.dataset.icon = token.glyphKey;
      chip.dataset.sourceClass = sourceClass.cssKey;
      if (Number.isInteger(item.duration)) {
        chip.dataset.duration = String(item.duration);
        chip.dataset.visibleValueAbbreviated = String(
          String(displayDuration) !== String(item.duration),
        );
      }
      chip.append(icon, durationValue);
    } else {
      chip.textContent = `${token.shortLabel} ×${formatDisplayNumber(item.multiplier)}`;
    }
    if (kind === "modifier" && Number.isFinite(item.multiplier)) {
      chip.dataset.multiplier = String(item.multiplier);
    }
    chip.setAttribute("aria-label", `${token.accessibleName}, ${value}`);
    chip.tabIndex = 0;
    registerTooltipOwner(
      chip,
      kind === "status"
        ? audience === "agent_pov"
          ? explainPovStatus(item, recipient)
          : explainStatus(item, recipient, sourceAgents)
        : explainModifier(item, recipient),
    );
    nodes.push(chip);
  }
  if (nodes.length === 0) {
    nodes.push(htmlElement("span", "roster-fact-empty", emptyText));
  }
  container.replaceChildren(...nodes);
}

/**
 * @param {unknown} target
 * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} [resolver]
 */
function targetLabel(target, resolver = new Map()) {
  if (!isRecord(target)) {
    return "Undisclosed";
  }
  if (target.disclosure === "target_none") {
    return "None";
  }
  if (target.target_action === 0) {
    return "None";
  }
  if (
    Number.isInteger(target.target_action) &&
    typeof target.public_agent_id === "string"
  ) {
    if (Number.isInteger(target.global_slot)) {
      const joined = resolvePublicAgentId(resolver, target.global_slot);
      if (joined !== target.public_agent_id) {
        return "Agent ID unavailable";
      }
    }
    return `Agent ID ${target.public_agent_id} (action ${target.target_action})`;
  }
  if (target.disclosure === "public" && Number.isInteger(target.global_slot)) {
    const identity = resolvePublicAgentId(resolver, target.global_slot);
    return identity === null
      ? "Agent ID unavailable"
      : `Agent ID ${identity} (action ${target.target_action ?? "unavailable"})`;
  }
  return humanize(target.disclosure ?? "undisclosed");
}

/**
 * @param {unknown} lane
 */
function laneLabel(lane) {
  return lane === 0 ? "Basic (0/B)" : lane === 1 ? "Ultimate (1/U)" : "No combat";
}

/**
 * @param {unknown} moveAction
 */
function movementLabel(moveAction) {
  const labels = [
    "Stay",
    "North",
    "South",
    "East",
    "West",
    "Northeast",
    "Northwest",
    "Southeast",
    "Southwest",
  ];
  return Number.isInteger(moveAction) && labels[Number(moveAction)]
    ? `${labels[Number(moveAction)]} (${moveAction})`
    : "Undisclosed";
}

/**
 * Present the staged combat choice, accounting for the canonical target-none
 * row whose lane field remains an authoritative transport detail.
 *
 * @param {Record<string, any>} pending
 */
function pendingCombatLabel(pending) {
  const targetAction = Number(
    (isRecord(pending.target) ? pending.target.target_action : undefined) ??
      pending.target_action,
  );
  return targetAction === 0 && pending.armed_lane !== 1
    ? "No combat"
    : laneLabel(pending.armed_lane);
}

/**
 * Convert authoritative pending fields into polished display copy without
 * changing, inferring, or replacing the returned action tuple.
 *
 * @param {Record<string, any>} pending
 * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} [resolver]
 */
export function pendingActionDisplayFacts(pending, resolver = new Map()) {
  if (!isRecord(pending)) {
    throw new TypeError("pending action display facts require an object.");
  }
  const rawTarget = isRecord(pending.target) ? pending.target : {};
  const target =
    rawTarget.target_action === undefined && Number.isInteger(pending.target_action)
      ? { ...rawTarget, target_action: pending.target_action }
      : rawTarget;
  return Object.freeze({
    movement: `Movement · ${movementLabel(pending.move_action)} · ${availabilityLabel(pending.movement_mask_value)}`,
    target: `Target · ${targetLabel(target, resolver)}`,
    action: `Action · ${pendingCombatLabel(pending)}`,
    legality: `Legality · ${pendingPairMaskLabel(pending)}`,
  });
}

/**
 * @param {unknown} value
 */
function availabilityLabel(value) {
  return value === true ? "Available" : value === false ? "Unavailable" : "Undisclosed";
}

/**
 * @param {Record<string, any>} pending
 */
function pendingPairMaskLabel(pending) {
  const targetAction = Number(
    (isRecord(pending.target) ? pending.target.target_action : undefined) ??
      pending.target_action,
  );
  if (
    (targetAction === 0 && pending.armed_lane !== 1) ||
    (pending.armed_lane !== 0 && pending.armed_lane !== 1)
  ) {
    return "Not applicable";
  }
  return availabilityLabel(pending.pair_mask_value);
}

/**
 * @param {PendingActionRow} row
 * @param {Record<string, any>} pending
 * @param {boolean} controlled
 * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} resolver
 */
function updatePendingActionRow(row, pending, controlled, resolver) {
  const element = row.element;
  element.tabIndex = 0;
  for (const attribute of [
    "data-target-slot",
    "data-armed-lane",
    "data-pair-mask-value",
  ]) {
    element.removeAttribute(attribute);
  }
  element.dataset.actorSlot = String(pending.actor_global_slot);
  element.dataset.controlled = String(controlled);
  element.dataset.moveAction = String(pending.move_action ?? "");
  element.dataset.targetDisclosure = String(
    isRecord(pending.target) ? (pending.target.disclosure ?? "redacted") : "redacted",
  );
  element.dataset.movementMaskValue = String(pending.movement_mask_value === true);
  if (Number.isInteger(pending.target?.global_slot)) {
    element.dataset.targetSlot = String(pending.target.global_slot);
  }
  if (pending.armed_lane === 0 || pending.armed_lane === 1) {
    element.dataset.armedLane = String(pending.armed_lane);
  }
  if (typeof pending.pair_mask_value === "boolean") {
    element.dataset.pairMaskValue = String(pending.pair_mask_value);
  }
  if (controlled) {
    element.setAttribute("aria-current", "true");
  } else {
    element.removeAttribute("aria-current");
  }

  const heading = htmlElement("div", "pending-action-row__heading");
  const actorIdentity = agentLabelForSlot(resolver, pending.actor_global_slot);
  heading.append(
    htmlElement("strong", null, actorIdentity),
    htmlElement(
      "span",
      "pending-action-row__state",
      controlled ? "Editing now" : "Staged",
    ),
  );
  const summary = htmlElement(
    "p",
    "pending-action-row__summary",
    String(pending.summary ?? "Pending action"),
  );
  const displayFacts = pendingActionDisplayFacts(pending, resolver);
  const facts = htmlElement("div", "pending-action-row__facts");
  facts.append(
    htmlElement("span", "pending-action-chip", displayFacts.movement),
    htmlElement("span", "pending-action-chip", displayFacts.target),
    htmlElement("span", "pending-action-chip", displayFacts.action),
    htmlElement("span", "pending-action-chip", displayFacts.legality),
  );
  element.replaceChildren(heading, summary, facts);
  element.setAttribute(
    "aria-label",
    `Pending action for ${actorIdentity}: ${String(pending.summary ?? "unavailable")}`,
  );
  const target = isRecord(pending.target) ? pending.target : {};
  const sourcePublicId = resolvePublicAgentId(resolver, pending.actor_global_slot);
  const targetPublicId = Number.isInteger(target.global_slot)
    ? resolvePublicAgentId(resolver, target.global_slot)
    : typeof target.public_agent_id === "string"
      ? target.public_agent_id
      : null;
  registerTooltipOwner(
    element,
    explainPendingRoute({
      source_public_agent_id: sourcePublicId,
      target_public_agent_id: targetPublicId,
      lane: pending.armed_lane,
    }),
  );
}

/**
 * @param {PendingActionRow} row
 * @param {Record<string, any>} pending
 */
function updatePovPendingActionRow(row, pending) {
  const identity = String(pending.actor_public_agent_id ?? "unavailable");
  row.element.tabIndex = 0;
  const target = isRecord(pending.target) ? pending.target : {};
  row.element.removeAttribute("data-actor-slot");
  row.element.dataset.actorPublicAgentId = identity;
  row.element.dataset.targetAction = String(target.target_action ?? "");
  row.element.dataset.controlled = "true";
  row.element.dataset.movementMaskValue = String(pending.movement_mask_value === true);
  if (pending.armed_lane === 0 || pending.armed_lane === 1) {
    row.element.dataset.armedLane = String(pending.armed_lane);
  } else {
    row.element.removeAttribute("data-armed-lane");
  }
  const heading = htmlElement("div", "pending-action-row__heading");
  heading.append(
    htmlElement("strong", null, `Agent ID ${identity}`),
    htmlElement("span", "pending-action-row__state", "Editing now"),
  );
  const facts = htmlElement("div", "pending-action-row__facts");
  facts.append(
    htmlElement(
      "span",
      "pending-action-chip",
      `Movement · ${movementLabel(pending.move_action)} · ${availabilityLabel(pending.movement_mask_value)}`,
    ),
    htmlElement("span", "pending-action-chip", `Target · ${targetLabel(target)}`),
    htmlElement(
      "span",
      "pending-action-chip",
      `Action · ${pendingCombatLabel(pending)}`,
    ),
    htmlElement(
      "span",
      "pending-action-chip",
      `Legality · ${pendingPairMaskLabel(pending)}`,
    ),
  );
  row.element.replaceChildren(
    heading,
    htmlElement(
      "p",
      "pending-action-row__summary",
      String(pending.summary ?? "Pending actor-POV action"),
    ),
    facts,
  );
  row.element.setAttribute(
    "aria-label",
    `Pending action for Agent ID ${identity}: ${String(pending.summary ?? "unavailable")}`,
  );
  registerTooltipOwner(
    row.element,
    explainPendingRoute({
      source_public_agent_id: identity,
      target_public_agent_id: target.public_agent_id,
      lane: pending.armed_lane,
    }),
  );
}

/**
 * @param {HTMLElement} container
 * @param {string} label
 * @param {Record<string, any>} action
 * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} [resolver]
 */
function addActionTuple(container, label, action, resolver = new Map()) {
  const card = htmlElement("section", "action-tuple");
  card.dataset.kind = label.toLowerCase();
  card.append(
    htmlElement("h4", null, label),
    htmlElement(
      "p",
      "action-tuple__summary",
      String(action.summary ?? "No action summary"),
    ),
  );
  const facts = htmlElement("div", "action-tuple__facts");
  addFact(facts, "Move action", action.move_action ?? "—");
  addFact(facts, "Target", targetLabel(action.target, resolver));
  addFact(facts, "Combat lane", actionTupleCombatLabel(action));
  card.append(facts);
  container.append(card);
}

/**
 * @param {Record<string, any>} action
 */
export function actionTupleCombatLabel(action) {
  const target = isRecord(action.target) ? action.target : {};
  if (
    action.use_ultimate_action === 0 &&
    (target.disclosure === "target_none" || target.target_action === 0)
  ) {
    return "No combat";
  }
  return action.use_ultimate_action === 0
    ? "0/B · Basic"
    : action.use_ultimate_action === 1
      ? "1/U · Ultimate"
      : "Undisclosed";
}

/**
 * @param {HTMLElement} container
 * @param {Record<string, any> | null} latest
 * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} resolver
 * @returns {string | null}
 */
function renderLatestTransition(container, latest, resolver) {
  container.replaceChildren();
  container.removeAttribute("data-transition-id");
  if (!latest) {
    container.append(htmlElement("p", "empty-copy", "No transition yet."));
    return null;
  }
  if (
    typeof latest.pov_transition_id === "string" &&
    isRecord(latest.actor) &&
    typeof latest.actor.actor_public_agent_id === "string"
  ) {
    const actor = latest.actor;
    container.dataset.transitionId = latest.pov_transition_id;
    container.append(
      htmlElement(
        "p",
        "action-card__label",
        String(latest.label ?? "LATEST ACCEPTED RESULT"),
      ),
    );
    const metadata = htmlElement("div", "action-card__facts");
    addFact(metadata, "POV transition", latest.pov_transition_id);
    addFact(metadata, "Submission", humanize(latest.submission_kind ?? "unknown"));
    container.append(metadata);
    const result = htmlElement("section", "action-result");
    result.dataset.actorPublicAgentId = actor.actor_public_agent_id;
    result.dataset.combatResult = String(actor.combat_result ?? "undisclosed");
    result.append(htmlElement("h3", null, `Agent ID ${actor.actor_public_agent_id}`));
    const comparison = htmlElement("div", "action-result__comparison");
    if (isRecord(actor.submitted)) {
      addActionTuple(comparison, "Submitted", actor.submitted, resolver);
    }
    if (isRecord(actor.accepted)) {
      addActionTuple(comparison, "Accepted", actor.accepted, resolver);
    }
    const results = htmlElement("div", "action-card__facts");
    addFact(results, "Movement accepted", availabilityLabel(actor.movement_accepted));
    addFact(results, "Combat result", humanize(actor.combat_result ?? "undisclosed"));
    result.append(comparison, results);
    container.append(result);
    return `POV transition ${latest.pov_transition_id}. Agent ID ${actor.actor_public_agent_id}: ${String(actor.accepted?.summary ?? "accepted action unavailable")}; combat ${humanize(actor.combat_result ?? "undisclosed")}`;
  }
  container.dataset.transitionId = String(latest.transition_id);
  container.append(
    htmlElement(
      "p",
      "action-card__label",
      String(latest.label ?? "LATEST ACCEPTED RESULT"),
    ),
  );
  const metadata = htmlElement("div", "action-card__facts");
  addFact(metadata, "Transition", latest.transition_id ?? "—");
  addFact(metadata, "Submission", humanize(latest.submission_kind ?? "unknown"));
  container.append(metadata);

  const announcements = [];
  for (const rawActor of asArray(latest.actors)) {
    if (!isRecord(rawActor)) {
      continue;
    }
    const actor = htmlElement("section", "action-result");
    const actorIdentity = agentLabelForSlot(resolver, rawActor.actor_global_slot);
    actor.dataset.actorSlot = String(rawActor.actor_global_slot);
    actor.dataset.combatResult = String(rawActor.combat_result ?? "undisclosed");
    actor.append(htmlElement("h3", null, actorIdentity));
    const comparison = htmlElement("div", "action-result__comparison");
    if (isRecord(rawActor.submitted)) {
      addActionTuple(comparison, "Submitted", rawActor.submitted, resolver);
    }
    if (isRecord(rawActor.accepted)) {
      addActionTuple(comparison, "Accepted", rawActor.accepted, resolver);
    }
    const results = htmlElement("div", "action-card__facts");
    addFact(
      results,
      "Movement accepted",
      rawActor.movement_accepted === true
        ? "Yes"
        : rawActor.movement_accepted === false
          ? "No"
          : "Undisclosed",
    );
    addFact(results, "Movement mask", availabilityLabel(rawActor.movement_mask_value));
    addFact(results, "Pair mask", availabilityLabel(rawActor.pair_mask_value));
    addFact(
      results,
      "Combat result",
      humanize(rawActor.combat_result ?? "undisclosed"),
    );
    actor.append(comparison, results);
    container.append(actor);
    const acceptedSummary = isRecord(rawActor.accepted)
      ? rawActor.accepted.summary
      : null;
    announcements.push(
      `${actorIdentity}: ${String(acceptedSummary ?? "accepted action unavailable")}; combat ${humanize(rawActor.combat_result ?? "undisclosed")}`,
    );
  }
  return `Transition ${latest.transition_id}. ${announcements.join(". ")}`;
}

/**
 * @param {unknown} value
 */
function pointLabel(value) {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    !value.every((coordinate) => Number.isFinite(coordinate))
  ) {
    return "undisclosed";
  }
  return `(${value.map((coordinate) => formatDisplayNumber(coordinate)).join(", ")})`;
}

/**
 * @param {unknown} event
 * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} [resolver]
 */
export function eventSummary(event, resolver = new Map()) {
  if (!isRecord(event)) {
    return "Unknown event";
  }
  switch (event.event_type) {
    case "action_rejected":
      return `Action rejected · actor ${agentLabelForSlot(resolver, event.actor_global_slot)} · recorded component ${humanize(event.rejection_component ?? "unknown")}`;
    case "ability_activated": {
      const recipient = Number.isInteger(event.recipient_global_slot)
        ? agentLabelForSlot(resolver, event.recipient_global_slot)
        : "source-local";
      return `${humanize(event.ability_component ?? "ability")} activated · ${agentLabelForSlot(resolver, event.source_global_slot)} → ${recipient}`;
    }
    case "source_damage_output":
      return `Damage output · ${agentLabelForSlot(resolver, event.source_global_slot)} · raw ${formatDisplayNumber(event.raw_damage_output)} · source-modified ${formatDisplayNumber(event.source_modified_damage_output)} · recipient modifier ×${formatDisplayNumber(event.recipient_damage_modifier)} · Sorcerer’s Empowerment emitters ${asArray(event.mage_damage_aura_covering_emitter_global_slots).length} · Guardian’s Barrier emitters ${asArray(event.warrior_mitigation_aura_covering_emitter_global_slots).length}`;
    case "source_healing_output":
      return `Healing output · ${agentLabelForSlot(resolver, event.source_global_slot)} · raw ${formatDisplayNumber(event.raw_healing_output)} · source-modified ${formatDisplayNumber(event.source_modified_healing_output)} · recipient modifier ×${formatDisplayNumber(event.recipient_healing_modifier)} · aura emitter evidence not recorded`;
    case "recipient_health_resolution": {
      const delta = Number(event.realized_net_health_change);
      const signed = Number.isFinite(delta)
        ? `${delta >= 0 ? "+" : ""}${formatDisplayNumber(delta)}`
        : "undisclosed";
      return `Net combat health · ${agentLabelForSlot(resolver, event.recipient_global_slot)} · ${signed} · HP ${formatDisplayNumber(event.transition_start_health)} → ${formatDisplayNumber(event.health_after_combat_resolution)}`;
    }
    case "combat_countdown_reset":
      return `Combat countdown reset · ${agentLabelForSlot(resolver, event.agent_global_slot)}`;
    case "health_regenerated":
      return `Out-of-combat regeneration · ${agentLabelForSlot(resolver, event.agent_global_slot)} · +${formatDisplayNumber(event.actual_health_regenerated)}`;
    case "cooldown_started":
      return `Ultimate cooldown started · ${agentLabelForSlot(resolver, event.agent_global_slot)}`;
    case "cooldown_ready":
      return `Ultimate ready · ${agentLabelForSlot(resolver, event.agent_global_slot)}`;
    case "charge_phase_displacement":
      return `Charge phase displacement · ${agentLabelForSlot(resolver, event.agent_global_slot)} · ${pointLabel(event.start_anchor?.position)} → ${pointLabel(event.end_anchor?.position)}`;
    case "ordinary_movement_phase_displacement":
      return `Ordinary movement phase displacement · ${agentLabelForSlot(resolver, event.agent_global_slot)} · ${pointLabel(event.start_anchor?.position)} → ${pointLabel(event.end_anchor?.position)}`;
    case "agent_died":
      return `Agent died · ${agentLabelForSlot(resolver, event.recipient_global_slot)}`;
    case "lethal_damage_contribution":
      return `Positive lethal-damage contributor · ${agentLabelForSlot(resolver, event.source_global_slot)} → ${agentLabelForSlot(resolver, event.recipient_global_slot)} · ${formatDisplayNumber(event.attributed_death_damage)}`;
    case "status_aged_to_zero":
      return `${humanize(event.status_id ?? "status")} expired · ${agentLabelForSlot(resolver, event.recipient_global_slot)}`;
    case "status_broken_by_damage":
      return `${humanize(event.status_id ?? "status")} broken by damage · ${agentLabelForSlot(resolver, event.recipient_global_slot)}`;
    case "status_applied":
      return `${humanize(event.status_id ?? "status")} applied · ${agentLabelForSlot(resolver, event.source_global_slot)} → ${agentLabelForSlot(resolver, event.recipient_global_slot)}`;
    case "status_refreshed_or_extended":
      return `${humanize(event.status_id ?? "status")} refreshed or extended · ${agentLabelForSlot(resolver, event.recipient_global_slot)} · source agent not recorded`;
    case "status_cleared_by_new_death":
      return `${humanize(event.status_id ?? "status")} cleared by new death · ${agentLabelForSlot(resolver, event.recipient_global_slot)}`;
    case "spawn_shield_expired":
      return `Spawn shield expired · ${agentLabelForSlot(resolver, event.agent_global_slot)}`;
    case "respawn_wave_occurred":
      return `Respawn wave occurred · Team ${event.team_id}`;
    case "agent_respawned":
      return `Agent respawned · ${agentLabelForSlot(resolver, event.agent_global_slot)} · ${pointLabel(event.realized_successor_position)}`;
    case "own_action_outcome":
      return `Own action ${humanize(event.outcome ?? "outcome")}`;
    case "own_position_changed":
      return `Own position changed · ${pointLabel(event.start_position)} → ${pointLabel(event.successor_position)}`;
    case "own_health_changed":
      return `Own health changed · ${formatDisplayNumber(event.start_health)} → ${formatDisplayNumber(event.successor_health)}`;
    case "own_status_changed":
      return `Own status observation changed · ${asArray(event.changed_feature_indices).length} recorded feature${asArray(event.changed_feature_indices).length === 1 ? "" : "s"}`;
    case "own_cooldown_changed":
      return `Own cooldown changed · ${formatDisplayNumber(event.start_remaining_ticks)} → ${formatDisplayNumber(event.successor_remaining_ticks)}`;
    case "own_lifecycle_changed":
      return `Own lifecycle changed · ${event.successor_alive === true ? "alive" : "not alive"} · shield ${event.successor_spawn_shield_remaining_ticks ?? "?"}`;
    case "visible_body_observation_changed":
      return `${humanize(event.relation ?? "visible body")} observation row ${event.observation_row} changed`;
    case "episode_ended":
      return `Episode ended · ${event.terminated === true ? "terminated" : event.truncated === true ? "truncated" : "recorded end"}`;
    default:
      return `Unknown semantic event · ${String(event.event_type ?? "missing type")}`;
  }
}

/**
 * Project an event-feed row into the semantic explanation boundary. Canonical
 * event IDs remain exclusively on the technical feed node used for keyed DOM
 * reconciliation; neither this descriptor nor its accessible text reads them.
 *
 * @param {unknown} event
 * @param {number} eventOrdinal
 * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} [resolver]
 */
export function eventDescriptor(event, eventOrdinal, resolver = new Map()) {
  const normalizedEvent = isRecord(event) ? event : {};
  if (!Number.isInteger(eventOrdinal) || eventOrdinal < 0) {
    throw new TypeError("semantic event ordinal must be a non-negative integer.");
  }
  const eventType = String(normalizedEvent.event_type ?? "unknown");
  return createSemanticDescriptor({
    kind: "event",
    id: `event:${eventType}:${eventOrdinal}`,
    title: humanize(normalizedEvent.event_type ?? "Semantic event"),
    tone: "information",
    accent: "none",
    summary: eventSummary(normalizedEvent, resolver),
    rows: [],
    sections: [],
    metadata: { compact: true, full: true },
    anchor: "element",
  });
}

/**
 * Project the replay Technical Frame facts without crossing its audience
 * boundary. Recorded movement scale exists only on the researcher envelope;
 * POV/source-material roots never receive or derive it.
 *
 * @param {unknown} rawFrame
 * @returns {ReadonlyArray<Readonly<{label: string, value: unknown}>>}
 */
export function replayDiagnosticFacts(rawFrame) {
  const frame = isRecord(rawFrame) ? rawFrame : null;
  if (frame?.viewer_mode !== "replay") {
    return Object.freeze([]);
  }
  const facts = [
    { label: "Frame kind", value: frame.frame_kind ?? "unavailable" },
    { label: "Timeline", value: frame.timeline_id ?? "unavailable" },
    {
      label: "Cursor generation",
      value: frame.cursor?.cursor_generation ?? "unavailable",
    },
    {
      label: "Choreography generation",
      value: frame.cursor?.choreography_generation ?? "unavailable",
    },
    {
      label: "Metric report",
      value: frame.artifact_summary?.metric_report_availability ?? "unavailable",
    },
  ];
  if (
    frame.replay_audience === "researcher" &&
    typeof frame.recorded_ordinary_movement_distance_scale === "number" &&
    Number.isFinite(frame.recorded_ordinary_movement_distance_scale)
  ) {
    facts.push({
      label: "Recorded movement scale",
      value: frame.recorded_ordinary_movement_distance_scale.toFixed(2),
    });
  }
  return Object.freeze(facts.map((fact) => Object.freeze(fact)));
}

/**
 * Render the debugger's HTML inspection panels from an authoritative frame.
 *
 * Roster rows and their action buttons are keyed by global slot. A render
 * updates those nodes in place, so ordinary authoritative refreshes retain
 * both DOM identity and keyboard focus.
 */
export class DebuggerPanels {
  /**
   * @param {DebuggerPanelBindings} bindings
   */
  constructor({
    roster,
    rosterCount,
    selectionCard,
    pendingHeading,
    pendingCount,
    pendingScope,
    pendingCard,
    acceptedCard,
    acceptedAnnouncement,
    eventFeed,
    eventCount,
    diagnosticsCard,
    onCommand,
  }) {
    this.roster = roster;
    this.rosterCount = rosterCount;
    this.selectionCard = selectionCard;
    this.pendingHeading = pendingHeading;
    this.pendingCount = pendingCount;
    this.pendingScope = pendingScope;
    this.pendingCard = pendingCard;
    this.acceptedCard = acceptedCard;
    this.acceptedAnnouncement = acceptedAnnouncement;
    this.eventFeed = eventFeed;
    this.eventCount = eventCount;
    this.diagnosticsCard = diagnosticsCard;
    this.onCommand = onCommand;
    /** @type {Map<number | string, RosterRow>} */
    this.rosterRows = new Map();
    /** @type {Map<number, RosterTeamGroup>} */
    this.rosterTeamGroups = new Map();
    /** @type {Map<string, HTMLElement>} */
    this.eventRows = new Map();
    /** @type {Map<string, HTMLElement>} */
    this.diagnosticRows = new Map();
    /** @type {Map<number | string, PendingActionRow>} */
    this.pendingActionRows = new Map();
    /** @type {{role: "target" | "control", slot: number} | null} */
    this.pendingRosterFocus = null;
    /** @type {string | null} */
    this.lastAnnouncedTransitionKey = null;
    this.roster.replaceChildren();
    this.ensureRosterTeamGroup(1);
    this.ensureRosterTeamGroup(2);
    this.emptyEvents = htmlElement("li", "empty-copy", "No transition events.");
    this.pendingLabel = htmlElement("p", "action-card__label", "PENDING / WILL SUBMIT");
    this.pendingList = htmlElement("ol", "pending-action-list");
    this.emptyPending = htmlElement("li", "empty-copy", "No pending action.");
    this.pendingList.append(this.emptyPending);
    this.pendingCard.replaceChildren(this.pendingLabel, this.pendingList);
  }

  /**
   * @param {number} teamId
   * @returns {RosterTeamGroup}
   */
  ensureRosterTeamGroup(teamId) {
    const existing = this.rosterTeamGroups.get(teamId);
    if (existing) {
      return existing;
    }
    const team = teamTokenFromId(teamId);
    const element = htmlElement("section", "roster-team");
    element.dataset.teamId = String(teamId);
    element.dataset.team = team.cssKey;
    element.setAttribute("aria-label", team.accessibleName);
    const heading = htmlElement("div", "roster-team__heading");
    const title = htmlElement("h3", null, team.label);
    const count = htmlElement("span", "count-badge", "0 authorized");
    heading.append(title, count);
    const rows = htmlElement("div", "roster-team__rows");
    const empty = htmlElement("p", "empty-copy", "No authorized agents.");
    rows.append(empty);
    element.append(heading, rows);
    this.roster.append(element);
    const group = { element, count, rows, empty };
    this.rosterTeamGroups.set(teamId, group);
    return group;
  }

  /**
   * @param {number} globalSlot
   * @returns {RosterRow}
   */
  createRosterRow(globalSlot) {
    const element = htmlElement("article", "roster-row");
    element.tabIndex = 0;
    const summary = htmlElement("div");
    const identity = htmlElement("div", "roster-identity");
    const identityId = htmlElement("span", "roster-id");
    const identityClass = htmlElement("span", "roster-class");
    identity.append(identityId, identityClass);
    const health = htmlElement("div", "roster-health");
    const statuses = htmlElement("div", "roster-fact-list roster-statuses");
    statuses.setAttribute("aria-label", "Persistent statuses");
    const modifiers = htmlElement("div", "roster-fact-list roster-modifiers");
    modifiers.setAttribute("aria-label", "Exact effective modifiers");
    summary.append(identity, health, statuses, modifiers);

    const actions = htmlElement("div", "roster-actions");
    const targetButton = htmlElement("button", null, "Target");
    targetButton.type = "button";
    targetButton.dataset.role = "target";
    targetButton.dataset.slot = String(globalSlot);
    targetButton.addEventListener("click", () => {
      this.pendingRosterFocus = { role: "target", slot: globalSlot };
      void this.onCommand({
        command_type: "roster_selection",
        role: "target",
        global_slot: globalSlot,
      });
    });

    const controlButton = htmlElement("button", null, "Control");
    controlButton.type = "button";
    controlButton.dataset.role = "control";
    controlButton.dataset.slot = String(globalSlot);
    controlButton.addEventListener("click", () => {
      this.pendingRosterFocus = { role: "control", slot: globalSlot };
      void this.onCommand({
        command_type: "roster_selection",
        role: "control",
        global_slot: globalSlot,
      });
    });
    actions.append(targetButton, controlButton);
    element.append(summary, actions);

    return {
      element,
      identityId,
      identityClass,
      health,
      statuses,
      modifiers,
      targetButton,
      controlButton,
    };
  }

  /**
   * Create one presentation-keyed roster row. Agent POV rows deliberately
   * receive no command listener or slot-bearing data. Oracle command slots
   * remain confined to the two command buttons.
   *
   * @param {ReturnType<typeof authorizedPresentationIdentityRows>[number]} identity
   * @returns {RosterRow}
   */
  createAuthorizedRosterRow(identity) {
    const element = htmlElement("article", "roster-row");
    element.tabIndex = 0;
    element.dataset.presentationKey = identity.presentation_key;
    const summary = htmlElement("div");
    const identityContainer = htmlElement("div", "roster-identity");
    const identityId = htmlElement("span", "roster-id");
    const identityClass = htmlElement("span", "roster-class");
    identityContainer.append(identityId, identityClass);
    const health = htmlElement("div", "roster-health");
    const statuses = htmlElement("div", "roster-fact-list roster-statuses");
    statuses.setAttribute("aria-label", "Persistent statuses");
    const modifiers = htmlElement("div", "roster-fact-list roster-modifiers");
    modifiers.setAttribute("aria-label", "Exact effective modifiers");
    summary.append(identityContainer, health, statuses, modifiers);

    const actions = htmlElement("div", "roster-actions");
    const targetButton = htmlElement("button", null, "Target");
    targetButton.type = "button";
    targetButton.dataset.role = "target";
    const controlButton = htmlElement("button", null, "Control");
    controlButton.type = "button";
    controlButton.dataset.role = "control";
    if (Number.isInteger(identity.command_global_slot)) {
      const commandSlot = Number(identity.command_global_slot);
      targetButton.dataset.commandSlot = String(commandSlot);
      controlButton.dataset.commandSlot = String(commandSlot);
      targetButton.addEventListener("click", () => {
        void this.onCommand({
          command_type: "roster_selection",
          role: "target",
          global_slot: commandSlot,
        });
      });
      controlButton.addEventListener("click", () => {
        void this.onCommand({
          command_type: "roster_selection",
          role: "control",
          global_slot: commandSlot,
        });
      });
    } else {
      targetButton.hidden = true;
      controlButton.hidden = true;
      targetButton.disabled = true;
      controlButton.disabled = true;
    }
    actions.append(targetButton, controlButton);
    element.append(summary, actions);
    return {
      element,
      identityId,
      identityClass,
      health,
      statuses,
      modifiers,
      targetButton,
      controlButton,
    };
  }

  /**
   * @param {Record<string, any>} presentation
   * @param {boolean} disabled
   */
  renderAuthorizedRoster(presentation, disabled) {
    const audience = authorizedPresentationAudience(presentation);
    const identities = authorizedPresentationIdentityRows(presentation);
    const scene = authorizedPresentationSceneView(presentation);
    const mechanicsByClassId = new Map(
      asArray(scene?.class_mechanics)
        .filter(
          (mechanics) => isRecord(mechanics) && Number.isInteger(mechanics.class_id),
        )
        .map((mechanics) => [Number(mechanics.class_id), mechanics]),
    );
    const sourceAgents =
      audience === "researcher" ? identities.map(({ agent }) => agent) : [];
    const activeKeys = new Set(identities.map(({ display_key }) => display_key));
    for (const [key, row] of this.rosterRows) {
      if (typeof key !== "string" || !activeKeys.has(key)) {
        row.element.remove();
        this.rosterRows.delete(key);
      }
    }

    this.roster.removeAttribute("data-compact");
    this.rosterCount.textContent = `${identities.length} visible`;
    /** @type {Map<number, HTMLElement[]>} */
    const desiredByTeam = new Map();
    for (const identity of identities) {
      const agent = agentForPresentation(
        isRecord(scene)
          ? (asArray(scene.agents).find(
              (candidate) =>
                isRecord(candidate) &&
                candidate.presentation_key === identity.presentation_key,
            ) ?? identity.agent)
          : identity.agent,
      );
      let row = this.rosterRows.get(identity.display_key);
      if (!row) {
        row = this.createAuthorizedRosterRow(identity);
        this.rosterRows.set(identity.display_key, row);
      }
      const publicId = String(agent.public_agent_id);
      const classToken = classTokenFromId(agent.class_id);
      const teamToken = teamTokenFromId(agent.team_id);
      row.element.dataset.presentationKey = String(agent.presentation_key);
      row.element.dataset.teamId = String(agent.team_id);
      row.element.dataset.classId = String(agent.class_id);
      row.element.dataset.team = teamToken.cssKey;
      row.element.dataset.class = classToken.cssKey;
      row.element.setAttribute(
        "aria-label",
        `Agent ID ${publicId}, ${classToken.label}, ${teamToken.label}`,
      );
      registerTooltipOwner(
        row.element,
        audience === "researcher"
          ? explainAgent(
              agent,
              { controlled: false, selected: false },
              mechanicsByClassId.get(Number(agent.class_id)) ?? null,
              sourceAgents,
            )
          : explainPovAgent(agent, {
              controlled:
                agent.presentation_key === presentation.recipient_presentation_key,
              selected: false,
            }),
      );
      row.identityId.textContent = `Agent ID ${publicId}`;
      row.identityClass.textContent = `${classToken.label} · ${teamToken.label}`;
      row.health.textContent =
        `HP ${formatDisplayNumber(agent.current_health)} / ${formatDisplayNumber(agent.max_health ?? agent.maximum_health)}` +
        ` · cooldown ${agent.ultimate_cooldown_remaining ?? "—"}`;
      renderFactTokens(
        row.statuses,
        asArray(agent.statuses),
        "status",
        "No persistent statuses",
        agent,
        sourceAgents,
        audience ?? "agent_pov",
      );
      renderFactTokens(
        row.modifiers,
        audience === "researcher"
          ? asArray(agent.modifiers ?? agent.aura_modifiers)
          : [],
        "modifier",
        audience === "researcher"
          ? "No effective modifiers"
          : "Effective modifiers unavailable",
        agent,
        sourceAgents,
        audience ?? "agent_pov",
      );

      const commandable = Number.isInteger(identity.command_global_slot);
      const replay = presentation.viewer_mode === "replay";
      row.targetButton.hidden = !commandable;
      row.controlButton.hidden = !commandable;
      row.targetButton.disabled = disabled || !commandable;
      row.controlButton.disabled = disabled || !commandable;
      row.targetButton.textContent = replay ? "Reference" : "Target";
      row.controlButton.textContent = replay ? "POV actor" : "Control";
      const desired = desiredByTeam.get(Number(agent.team_id)) ?? [];
      desired.push(row.element);
      desiredByTeam.set(Number(agent.team_id), desired);
      this.ensureRosterTeamGroup(Number(agent.team_id));
    }
    for (const [teamId, group] of this.rosterTeamGroups) {
      const desired = desiredByTeam.get(teamId) ?? [];
      group.count.textContent = `${desired.length} authorized`;
      this.reconcileChildren(group.rows, desired.length > 0 ? desired : [group.empty]);
    }
  }

  /** @param {Record<string, any>} presentation */
  renderAuthorizedInspector(presentation) {
    const scene = authorizedPresentationSceneView(presentation);
    const inspectionState = authorizedPresentationInspectionState(presentation);
    const inspection = inspectionState.inspection;
    const audience = authorizedPresentationAudience(presentation);
    const agents = asArray(scene?.agents).filter(isRecord);
    const primary =
      (inspection &&
        agents.find(
          (agent) => agent.presentation_key === inspection.actor_presentation_key,
        )) ||
      agents.find(
        (agent) => agent.presentation_key === presentation.recipient_presentation_key,
      ) ||
      null;
    this.selectionCard.replaceChildren();
    if (primary === null) {
      this.selectionCard.append(
        htmlElement("p", "empty-copy", "No authorized agent details are available."),
      );
    } else {
      renderSemanticInspector(
        this.selectionCard,
        audience === "researcher"
          ? explainAgent(primary, {}, null, agents)
          : explainPovAgent(primary, {
              controlled: presentation.viewer_mode === "live",
            }),
      );
    }

    if (inspectionState.submission_scope === null) {
      this.pendingCard.removeAttribute("data-submission-scope");
    } else {
      this.pendingCard.dataset.submissionScope = inspectionState.submission_scope;
    }
    this.pendingCard.dataset.inspectionState = inspectionState.state_kind;
    this.pendingHeading.textContent = {
      live_editable: "Pending authorized draft",
      live_scripted: "Scripted playback inspection",
      replay_outgoing: "Recorded outgoing inspection",
      replay_none: "No outgoing inspection",
      unavailable: "Inspection unavailable",
    }[inspectionState.state_kind];
    this.pendingCount.textContent = inspection ? "1 actor" : "0 actors";
    this.pendingScope.textContent =
      inspectionState.state_kind === "live_scripted"
        ? "This live frame advances registered scripted actions and has no editable draft."
        : inspection
          ? "This panel shows only the separately authorized outgoing inspection."
          : "No outgoing inspection is authorized at this replay frame.";
    const action =
      inspection?.inspection_kind === "live_draft_action"
        ? inspection.draft_action
        : inspection?.inspection_kind === "replay_recorded_outgoing_action"
          ? inspection.accepted_action
          : null;
    this.pendingLabel.textContent =
      action === null || inspection === null
        ? "NO OUTGOING INSPECTION"
        : `${inspection.actor_public_agent_id} · move ${action.move_action} · target ${action.target_action} · ${inspection.inspection_kind === "live_draft_action" ? action.armed_lane : inspection.combat_lane}`;
    this.pendingList.replaceChildren(
      htmlElement(
        "li",
        action === null ? "empty-copy" : "pending-action-row",
        action === null
          ? "No pending action."
          : "Settled range, target, and legality overlays use this outgoing branch only.",
      ),
    );

    const transitionRows = authorizedPresentationTransitionRows(presentation);
    this.acceptedCard.replaceChildren();
    this.acceptedCard.removeAttribute("data-transition-id");
    if (transitionRows.length === 0) {
      this.acceptedCard.append(
        htmlElement("p", "empty-copy", "No incoming Submitted / Accepted row."),
      );
      this.acceptedAnnouncement.textContent = "";
    } else {
      if (typeof presentation.latest_transition?.incoming_transition_id === "string") {
        this.acceptedCard.dataset.transitionId =
          presentation.latest_transition.incoming_transition_id;
      }
      const list = htmlElement("ol", "accepted-action-list");
      for (const transition of transitionRows) {
        const item = htmlElement("li", "accepted-action-row");
        item.dataset.presentationKey = String(transition.actor_presentation_key);
        const submitted = transition.submitted_action;
        const accepted = transition.accepted_action;
        item.textContent =
          `Agent ID ${transition.actor_public_agent_id} · Submitted ${submitted.move_action}/${submitted.target_action}/${submitted.use_ultimate_action}` +
          ` · Accepted ${accepted.move_action}/${accepted.target_action}/${accepted.use_ultimate_action}`;
        list.append(item);
      }
      this.acceptedCard.append(list);
      this.acceptedAnnouncement.textContent = `${transitionRows.length} incoming action ${transitionRows.length === 1 ? "row" : "rows"}.`;
    }

    const facts = authorizedPresentationTechnicalFacts(presentation);
    this.diagnosticsCard.replaceChildren();
    for (const fact of facts) {
      addFact(this.diagnosticsCard, humanize(fact.label), fact.value ?? "None");
    }
  }

  /** @param {Record<string, any>} presentation */
  renderAuthorizedEvents(presentation) {
    const rows = authorizedPresentationIncomingRows(presentation);
    this.eventCount.textContent = String(rows.length);
    const activeIds = new Set(rows.map(({ id }) => id));
    for (const [eventId, row] of this.eventRows) {
      if (!activeIds.has(eventId)) {
        row.remove();
        this.eventRows.delete(eventId);
      }
    }
    if (rows.length === 0) {
      this.reconcileChildren(this.eventFeed, [this.emptyEvents]);
      return;
    }
    const desired = rows.map((event, ordinal) => {
      let item = this.eventRows.get(event.id);
      if (!item) {
        item = htmlElement("li", "event-item");
        item.tabIndex = 0;
        this.eventRows.set(event.id, item);
      }
      item.dataset.eventId = event.id;
      item.dataset.eventType = event.kind;
      item.dataset.eventVocabulary = event.vocabulary;
      const prefix =
        event.vocabulary === "observation_delta"
          ? "Observation delta"
          : event.vocabulary === "recipient_cue"
            ? "Recipient cue"
            : "Incoming event";
      item.textContent = `${prefix} · ${humanize(event.kind)}`;
      registerTooltipOwner(
        item,
        createSemanticDescriptor({
          kind: event.vocabulary,
          id: `${presentation.session_id}:${event.id}`,
          title: `${prefix} ${ordinal + 1}`,
          tone: "information",
          accent: "none",
          summary:
            event.vocabulary === "observation_delta"
              ? "A recipient-authorized observation change; no simulator cause is asserted."
              : "An exact incoming presentation fact.",
          rows: [
            {
              label: "Kind",
              value: humanize(event.kind),
              metadata: { compact: true, full: true },
            },
            {
              label: "Identity",
              value: event.id,
              metadata: { compact: true, full: true },
            },
          ],
          sections: [],
          metadata: { compact: true, full: true },
          anchor: "element",
        }),
      );
      return item;
    });
    this.reconcileChildren(this.eventFeed, desired);
  }

  /**
   * @param {RosterRow} row
   * @param {Record<string, any>} agent
   * @param {Record<string, any>} selection
   * @param {boolean} disabled
   * @param {string | null} replayAudience
   * @param {Record<string, any> | null} classMechanics
   * @param {ReadonlyArray<unknown>} sourceAgents
   * @param {"researcher" | "agent_pov"} audience
   */
  updateRosterRow(
    row,
    agent,
    selection,
    disabled,
    replayAudience,
    classMechanics,
    sourceAgents,
    audience,
  ) {
    const globalSlot = agent.global_slot;
    const publicAgentId =
      typeof agent.public_agent_id === "string" ? agent.public_agent_id : "unavailable";
    const classToken = classTokenFromId(agent.class_id);
    const teamToken = teamTokenFromId(agent.team_id);
    row.element.setAttribute(
      "aria-label",
      `Agent ID ${publicAgentId}, ${classToken.label}, ${teamToken.label}`,
    );
    row.element.dataset.slot = String(globalSlot);
    row.element.dataset.teamId = String(agent.team_id);
    row.element.dataset.classId = String(agent.class_id);
    row.element.dataset.team = teamToken.cssKey;
    row.element.dataset.class = classToken.cssKey;
    row.element.dataset.controlled = String(
      globalSlot === selection.controlled_global_slot,
    );
    row.element.dataset.selected = String(
      globalSlot === selection.selected_global_slot,
    );
    row.element.dataset.reference = String(
      replayAudience === "researcher" && globalSlot === selection.selected_global_slot,
    );
    registerTooltipOwner(
      row.element,
      audience === "agent_pov"
        ? explainPovAgent(agent, {
            controlled:
              replayAudience === null &&
              globalSlot === selection.controlled_global_slot,
            selected:
              replayAudience === null && globalSlot === selection.selected_global_slot,
          })
        : explainAgent(
            agentForPresentation(agent),
            {
              controlled:
                replayAudience === null &&
                globalSlot === selection.controlled_global_slot,
              selected:
                replayAudience === null &&
                globalSlot === selection.selected_global_slot,
              reference:
                replayAudience === "researcher" &&
                globalSlot === selection.selected_global_slot,
            },
            classMechanics,
            sourceAgents,
          ),
    );

    row.identityId.textContent = `Agent ID ${publicAgentId}`;
    row.identityClass.textContent = `${classToken.label} · ${teamToken.label}`;
    row.health.textContent =
      `HP ${formatDisplayNumber(agent.current_health)} / ${formatDisplayNumber(agent.max_health)}` +
      ` · cooldown ${agent.ultimate_cooldown_remaining ?? agent.ultimate_cooldown ?? "—"}`;

    row.statuses.hidden = false;
    row.modifiers.hidden = false;
    renderFactTokens(
      row.statuses,
      asArray(agent.statuses),
      "status",
      "No persistent statuses",
      agent,
      sourceAgents,
      audience,
    );
    const modifiersAvailable =
      audience === "researcher" &&
      Array.isArray(agent.modifiers ?? agent.aura_modifiers);
    renderFactTokens(
      row.modifiers,
      modifiersAvailable ? (agent.modifiers ?? agent.aura_modifiers) : [],
      "modifier",
      modifiersAvailable ? "No effective modifiers" : "Effective modifiers unavailable",
      agent,
      sourceAgents,
      audience,
    );

    const researcherReplay = replayAudience === "researcher";
    const replayIdentityOnly = replayAudience !== null && !researcherReplay;
    row.targetButton.hidden = replayIdentityOnly;
    row.controlButton.hidden = replayIdentityOnly;
    row.targetButton.textContent = researcherReplay ? "Reference" : "Target";
    row.controlButton.textContent = researcherReplay ? "POV actor" : "Control";
    row.targetButton.setAttribute(
      "aria-label",
      researcherReplay
        ? `Use Agent ID ${publicAgentId} as Reference`
        : `Target Agent ID ${publicAgentId}`,
    );
    row.targetButton.setAttribute(
      "aria-pressed",
      String(globalSlot === selection.selected_global_slot),
    );
    row.targetButton.disabled = disabled;
    row.controlButton.setAttribute(
      "aria-label",
      researcherReplay
        ? `Choose Agent ID ${publicAgentId} as POV actor`
        : `Control Agent ID ${publicAgentId}`,
    );
    row.controlButton.setAttribute(
      "aria-pressed",
      String(globalSlot === selection.controlled_global_slot),
    );
    row.controlButton.disabled = disabled;
    const controlMode = researcherReplay ? "researcher_replay" : "live";
    registerTooltipOwner(
      row.targetButton,
      rosterControlDescriptor("target", publicAgentId, controlMode, disabled),
      { inspectable: false },
    );
    registerTooltipOwner(
      row.controlButton,
      rosterControlDescriptor("control", publicAgentId, controlMode, disabled),
      { inspectable: false },
    );
  }

  /**
   * @returns {{
   *   role: "target" | "control",
   *   slot: number,
   *   element: Element,
   * } | null}
   */
  focusedRosterControl() {
    const active = document.activeElement;
    if (!(active instanceof Element) || !this.roster.contains(active)) {
      return null;
    }
    const role = active.getAttribute("data-role");
    const slot = Number(active.getAttribute("data-slot"));
    if ((role !== "target" && role !== "control") || !Number.isInteger(slot)) {
      return null;
    }
    return { role, slot, element: active };
  }

  /**
   * @param {HTMLElement} container
   * @param {HTMLElement[]} desired
   */
  reconcileChildren(container, desired) {
    const desiredSet = new Set(desired);
    for (const child of [...container.children]) {
      if (!desiredSet.has(/** @type {HTMLElement} */ (child))) {
        child.remove();
      }
    }

    for (let index = 0; index < desired.length; index += 1) {
      const child = container.children.item(index);
      const desiredChild = desired[index];
      if (child !== desiredChild) {
        container.insertBefore(desiredChild, child);
      }
    }
  }

  /**
   * @param {Record<string, any> | null} frame
   * @param {boolean} disabled
   * @param {boolean} retainDisabledFocus
   */
  renderRoster(frame, disabled, retainDisabledFocus) {
    const scene = frameScene(frame);
    const agents = asArray(scene?.agents)
      .filter((agent) => isRecord(agent) && Number.isInteger(agent.global_slot))
      .sort((left, right) => left.global_slot - right.global_slot);
    const selection = isRecord(scene?.selection) ? scene.selection : {};
    const replayAudience =
      frame?.viewer_mode === "replay" && typeof frame.replay_audience === "string"
        ? frame.replay_audience
        : null;
    const audience =
      scene?.audience === "agent_pov" ||
      frame?.frame_kind === "actor_pov_live_debugger" ||
      (replayAudience !== null && replayAudience !== "researcher")
        ? "agent_pov"
        : "researcher";
    const mechanicsByClassId = new Map(
      asArray(scene?.class_mechanics)
        .filter(
          (mechanics) => isRecord(mechanics) && Number.isInteger(mechanics.class_id),
        )
        .map((mechanics) => [Number(mechanics.class_id), mechanics]),
    );
    const sourceAgents = audience === "researcher" ? agents : [];
    const povIdentityOnly =
      frame?.frame_kind === "actor_pov_live_debugger" ||
      (replayAudience !== null && replayAudience !== "researcher");
    const focused = this.focusedRosterControl();
    if (focused) {
      this.pendingRosterFocus = {
        role: focused.role,
        slot: focused.slot,
      };
    }

    this.roster.removeAttribute("data-compact");
    this.rosterCount.textContent = `${agents.length} visible`;
    const activeSlots = new Set(agents.map((agent) => Number(agent.global_slot)));
    for (const [globalSlot, row] of this.rosterRows) {
      if (typeof globalSlot !== "number" || !activeSlots.has(globalSlot)) {
        row.element.remove();
        this.rosterRows.delete(globalSlot);
      }
    }

    /** @type {Map<number, HTMLElement[]>} */
    const desiredByTeam = new Map();
    for (const agent of agents) {
      const globalSlot = Number(agent.global_slot);
      const teamId = Number(agent.team_id);
      let row = this.rosterRows.get(globalSlot);
      if (!row) {
        row = this.createRosterRow(globalSlot);
        this.rosterRows.set(globalSlot, row);
      }
      this.updateRosterRow(
        row,
        agent,
        selection,
        disabled || povIdentityOnly,
        replayAudience,
        mechanicsByClassId.get(Number(agent.class_id)) ?? null,
        sourceAgents,
        audience,
      );
      const desired = desiredByTeam.get(teamId) ?? [];
      desired.push(row.element);
      desiredByTeam.set(teamId, desired);
      this.ensureRosterTeamGroup(teamId);
    }

    for (const [teamId, group] of this.rosterTeamGroups) {
      const desired = desiredByTeam.get(teamId) ?? [];
      group.count.textContent = `${desired.length} authorized`;
      this.reconcileChildren(group.rows, desired.length > 0 ? desired : [group.empty]);
    }

    if (this.pendingRosterFocus) {
      const row = this.rosterRows.get(this.pendingRosterFocus.slot);
      const control =
        this.pendingRosterFocus.role === "target"
          ? row?.targetButton
          : row?.controlButton;
      if (!control) {
        this.pendingRosterFocus = null;
      } else if (!control.disabled) {
        if (document.activeElement !== control) {
          control.focus({ preventScroll: true });
        }
        if (document.activeElement === control) {
          this.pendingRosterFocus = null;
        }
      } else if (!retainDisabledFocus) {
        this.pendingRosterFocus = null;
      }
    }
  }

  /**
   * @param {Record<string, any>} hud
   * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} resolver
   */
  renderPendingPlan(hud, resolver) {
    /** @type {"joint_turn" | "controlled_actor" | "scripted_playback"} */
    const scope =
      hud.pending_submission_scope === "joint_turn" ||
      hud.pending_submission_scope === "controlled_actor" ||
      hud.pending_submission_scope === "scripted_playback"
        ? hud.pending_submission_scope
        : "controlled_actor";
    if (
      typeof hud.controlled_public_agent_id === "string" &&
      isRecord(hud.pending_action) &&
      typeof hud.pending_action.actor_public_agent_id === "string"
    ) {
      const identity = hud.controlled_public_agent_id;
      this.pendingHeading.textContent =
        scope === "scripted_playback" ? "Inspection only" : "Pending controlled actor";
      this.pendingCount.textContent = "1 actor";
      this.pendingScope.textContent =
        scope === "scripted_playback"
          ? "This legacy scripted frame is read-only."
          : "Only this recipient-authorized actor action will be submitted.";
      this.pendingCard.dataset.submissionScope = scope;
      this.pendingCard.dataset.pendingCount = "1";
      this.pendingLabel.textContent = String(hud.pending_action.label);
      for (const [key, row] of this.pendingActionRows) {
        if (key !== identity) {
          row.element.remove();
          this.pendingActionRows.delete(key);
        }
      }
      let row = this.pendingActionRows.get(identity);
      if (!row) {
        row = { element: htmlElement("li", "pending-action-row") };
        this.pendingActionRows.set(identity, row);
      }
      updatePovPendingActionRow(row, hud.pending_action);
      this.reconcileChildren(this.pendingList, [row.element]);
      return;
    }
    let pendingActions = asArray(hud.pending_actions).filter(
      (pending) => isRecord(pending) && Number.isInteger(pending.actor_global_slot),
    );
    if (pendingActions.length === 0 && isRecord(hud.pending_action)) {
      pendingActions = [hud.pending_action];
    }
    const controlledSlot = Number(hud.controlled_global_slot);
    const scopeCopy = {
      joint_turn:
        "Every listed active actor will be packaged into one authoritative transition.",
      controlled_actor:
        "Only the controlled actor will be submitted from this agent-POV frame.",
      scripted_playback: "This legacy scripted frame is read-only.",
    };
    const heading = {
      joint_turn: "Pending joint turn",
      controlled_actor: "Pending controlled actor",
      scripted_playback: "Inspection only",
    };

    this.pendingHeading.textContent = heading[scope];
    this.pendingCount.textContent = `${pendingActions.length} ${
      pendingActions.length === 1 ? "actor" : "actors"
    }`;
    this.pendingScope.textContent = scopeCopy[scope];
    this.pendingCard.dataset.submissionScope = scope;
    this.pendingCard.dataset.pendingCount = String(pendingActions.length);
    const controlledPending =
      pendingActions.find(
        (pending) => Number(pending.actor_global_slot) === controlledSlot,
      ) ?? pendingActions[0];
    this.pendingLabel.textContent = String(
      controlledPending?.label ??
        (scope === "scripted_playback" ? "INSPECTION ONLY" : "PENDING / WILL SUBMIT"),
    );

    const activeSlots = new Set(
      pendingActions.map((pending) => Number(pending.actor_global_slot)),
    );
    for (const [actorSlot, row] of this.pendingActionRows) {
      if (typeof actorSlot !== "number" || !activeSlots.has(actorSlot)) {
        row.element.remove();
        this.pendingActionRows.delete(actorSlot);
      }
    }
    const desired = [];
    for (const pending of pendingActions) {
      const actorSlot = Number(pending.actor_global_slot);
      let row = this.pendingActionRows.get(actorSlot);
      if (!row) {
        row = { element: htmlElement("li", "pending-action-row") };
        this.pendingActionRows.set(actorSlot, row);
      }
      updatePendingActionRow(row, pending, actorSlot === controlledSlot, resolver);
      desired.push(row.element);
    }
    this.reconcileChildren(
      this.pendingList,
      desired.length > 0 ? desired : [this.emptyPending],
    );
  }

  /**
   * @param {Record<string, any> | null} frame
   * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} resolver
   */
  renderInspector(frame, resolver) {
    const scene = frameScene(frame);
    const hud = isRecord(frame?.hud) ? frame.hud : {};
    const selection = isRecord(scene?.selection) ? scene.selection : null;
    const legality = isRecord(scene?.selected_legality)
      ? scene.selected_legality
      : null;
    const pov = frame?.frame_kind === "actor_pov_live_debugger";
    const replayAudience =
      frame?.viewer_mode === "replay" && typeof frame.replay_audience === "string"
        ? frame.replay_audience
        : null;
    const agents = asArray(scene?.agents).filter(isRecord);
    const agentsBySlot = new Map(
      agents
        .filter((agent) => Number.isInteger(agent.global_slot))
        .map((agent) => [Number(agent.global_slot), agent]),
    );
    const classMechanicsById = new Map(
      asArray(scene?.class_mechanics)
        .filter(
          (mechanics) => isRecord(mechanics) && Number.isInteger(mechanics.class_id),
        )
        .map((mechanics) => [Number(mechanics.class_id), mechanics]),
    );
    const researcherAudience =
      scene?.audience === "researcher" &&
      (replayAudience === null || replayAudience === "researcher") &&
      !pov;
    const sourceAgents = researcherAudience ? agents : [];
    /** @param {unknown} agent */
    const mechanicsFor = (agent) =>
      researcherAudience && isRecord(agent)
        ? (classMechanicsById.get(Number(agent.class_id)) ?? null)
        : null;
    this.selectionCard.replaceChildren();
    const selectedSlot = selection?.selected_global_slot;
    const controlledSlot = selection?.controlled_global_slot;
    const primaryAgent =
      replayAudience === "researcher"
        ? Number.isInteger(selectedSlot)
          ? (agentsBySlot.get(Number(selectedSlot)) ?? null)
          : null
        : replayAudience !== null
          ? (agents.find(
              (agent) =>
                agent.global_slot === frame?.pov_global_slot ||
                agent.global_slot === frame?.selected_global_slot ||
                agent.public_agent_id === frame?.public_agent_id,
            ) ??
            agents[0] ??
            null)
          : pov
            ? (agents.find(
                (agent) => agent.public_agent_id === hud.controlled_public_agent_id,
              ) ??
              agents[0] ??
              null)
            : Number.isInteger(selectedSlot)
              ? (agentsBySlot.get(Number(selectedSlot)) ?? null)
              : Number.isInteger(controlledSlot)
                ? (agentsBySlot.get(Number(controlledSlot)) ?? null)
                : null;
    if (primaryAgent === null) {
      this.selectionCard.append(
        htmlElement(
          "p",
          "empty-copy",
          replayAudience === "researcher"
            ? "Choose an authorized replay agent to open its comprehensive details."
            : "No authorized agent details are available.",
        ),
      );
    } else {
      const descriptor = researcherAudience
        ? explainAgent(
            agentForPresentation(primaryAgent),
            {
              controlled:
                replayAudience === null && primaryAgent.global_slot === controlledSlot,
              selected:
                replayAudience === null && primaryAgent.global_slot === selectedSlot,
              reference:
                replayAudience === "researcher" &&
                primaryAgent.global_slot === selectedSlot,
            },
            mechanicsFor(primaryAgent),
            sourceAgents,
          )
        : explainPovAgent(primaryAgent, {
            controlled: replayAudience === null,
            selected: false,
          });
      renderSemanticInspector(this.selectionCard, descriptor);
      if (
        researcherAudience &&
        legality !== null &&
        primaryAgent.global_slot === selectedSlot
      ) {
        const selectedLegality = htmlElement("section", "selected-legality");
        selectedLegality.setAttribute(
          "aria-label",
          "Exact selected-target Basic and Ultimate legality",
        );
        selectedLegality.append(htmlElement("h3", null, "Current legality"));
        const facts = htmlElement("div", "selected-legality__facts");
        facts.append(
          semanticPanelCard(explainLegality(legality, 0), "selected-legality__lane"),
          semanticPanelCard(explainLegality(legality, 1), "selected-legality__lane"),
        );
        selectedLegality.append(facts);
        this.selectionCard.append(selectedLegality);
      }
    }
    const latestCandidate = hud.latest_transition ?? null;
    const latest = isRecord(latestCandidate) ? latestCandidate : null;
    this.renderPendingPlan(hud, resolver);
    const announcement = renderLatestTransition(this.acceptedCard, latest, resolver);
    if (latest && announcement) {
      const transitionKey = `${frame?.run_generation ?? "unknown"}:${latest.pov_transition_id ?? latest.transition_id}`;
      if (transitionKey !== this.lastAnnouncedTransitionKey) {
        this.lastAnnouncedTransitionKey = transitionKey;
        this.acceptedAnnouncement.textContent = announcement;
      }
    } else {
      this.lastAnnouncedTransitionKey = null;
      this.acceptedAnnouncement.textContent = "";
    }
    this.renderDiagnostics(frame, hud);
  }

  /**
   * @param {Record<string, any> | null} frame
   * @param {Record<string, any>} hud
   */
  renderDiagnostics(frame, hud) {
    const diagnosticsSection = this.diagnosticsCard.closest(".diagnostics");
    diagnosticsSection?.removeAttribute("hidden");
    if (frame?.viewer_mode === "replay") {
      const replayFacts = htmlElement("div", "diagnostics-card");
      for (const fact of replayDiagnosticFacts(frame)) {
        addFact(replayFacts, fact.label, fact.value);
      }
      this.diagnosticsCard.replaceChildren(replayFacts);
      return;
    }
    const facts = asArray(hud.diagnostics).filter(
      (fact) => isRecord(fact) && typeof fact.fact_id === "string",
    );
    const activeIds = new Set(facts.map((fact) => String(fact.fact_id)));
    for (const factId of this.diagnosticRows.keys()) {
      if (!activeIds.has(factId)) {
        this.diagnosticRows.delete(factId);
      }
    }
    const desired = [];
    for (const fact of facts) {
      const factId = String(fact.fact_id);
      let row = this.diagnosticRows.get(factId);
      if (!row) {
        row = htmlElement("div", "diagnostic-fact");
        row.dataset.factId = factId;
        this.diagnosticRows.set(factId, row);
      }
      row.dataset.technical = String(Boolean(fact.technical));
      row.replaceChildren(
        htmlElement("span", null, String(fact.label ?? factId)),
        htmlElement("strong", null, String(fact.value ?? "")),
      );
      desired.push(row);
    }
    this.reconcileChildren(
      this.diagnosticsCard,
      desired.length > 0
        ? desired
        : [htmlElement("p", "empty-copy", "No concise diagnostics are available.")],
    );
  }

  /**
   * @param {Record<string, any> | null} frame
   * @param {ReadonlyMap<number, string> | ((slot: number) => string | null)} resolver
   */
  renderEvents(frame, resolver) {
    const batch = frameEvents(frame);
    const events = asArray(batch?.events).filter(
      (event) => isRecord(event) && typeof event.event_id === "string",
    );
    this.eventCount.textContent = String(events.length);
    const activeIds = new Set(events.map((event) => String(event.event_id)));
    for (const [eventId, row] of this.eventRows) {
      if (!activeIds.has(eventId)) {
        row.remove();
        this.eventRows.delete(eventId);
      }
    }
    if (events.length === 0) {
      this.reconcileChildren(this.eventFeed, [this.emptyEvents]);
      return;
    }

    const desired = [];
    for (const [eventOrdinal, event] of events.entries()) {
      const eventId = String(event.event_id);
      let item = this.eventRows.get(eventId);
      if (!item) {
        item = htmlElement("li", "event-item");
        item.tabIndex = 0;
        this.eventRows.set(eventId, item);
      }
      for (const attribute of [
        "data-source-slot",
        "data-target-slot",
        "data-recipient-slot",
        "data-actor-slot",
      ]) {
        item.removeAttribute(attribute);
      }
      item.dataset.eventId = eventId;
      item.dataset.eventType = String(event.event_type ?? "unknown");
      if (Number.isInteger(event.source_global_slot)) {
        item.dataset.sourceSlot = String(event.source_global_slot);
      }
      if (Number.isInteger(event.target_global_slot)) {
        item.dataset.targetSlot = String(event.target_global_slot);
      }
      if (Number.isInteger(event.recipient_global_slot)) {
        item.dataset.recipientSlot = String(event.recipient_global_slot);
      }
      if (Number.isInteger(event.actor_global_slot)) {
        item.dataset.actorSlot = String(event.actor_global_slot);
      }
      const summary = eventSummary(event, resolver);
      item.textContent = summary;
      registerTooltipOwner(item, eventDescriptor(event, eventOrdinal, resolver));
      desired.push(item);
    }
    this.reconcileChildren(this.eventFeed, desired);
  }

  /**
   * @param {Record<string, any> | null} frame
   * @param {PanelInteractionState} interactionState
   */
  render(frame, interactionState = {}) {
    const disabled = Boolean(
      interactionState.busy ||
        interactionState.shuttingDown ||
        interactionState.resyncRequired ||
        interactionState.offline,
    );
    if (isAuthorizedPresentationFrame(frame)) {
      this.renderAuthorizedRoster(frame, disabled);
      this.renderAuthorizedInspector(frame);
      this.renderAuthorizedEvents(frame);
      return;
    }
    const resolver = publicAgentIdMap(frame);
    this.renderRoster(frame, disabled, Boolean(interactionState.busy));
    this.renderInspector(frame, resolver);
    this.renderEvents(frame, resolver);
  }
}
