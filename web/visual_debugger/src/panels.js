import { formatDisplayNumber } from "./display.js";
import {
  explainActivation,
  explainAgent,
  explainModifier,
  explainNetHealth,
  explainStatus,
} from "./explanations.js";
import { registerTooltipOwner } from "./tooltip.js";
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
  return isRecord(frame.scene)
    ? frame.scene
    : isRecord(frame.battlefield_scene)
      ? frame.battlefield_scene
      : null;
}

/**
 * @param {unknown} frame
 * @returns {Record<string, any> | null}
 */
function frameEvents(frame) {
  if (!isRecord(frame)) {
    return null;
  }
  return isRecord(frame.event_batch)
    ? frame.event_batch
    : isRecord(frame.visual_event_batch)
      ? frame.visual_event_batch
      : null;
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
 * @param {unknown} value
 * @param {string} emptyText
 */
function formatRecord(value, emptyText) {
  if (value === null || value === undefined) {
    return emptyText;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
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
 * Render every authorized status/modifier record without sorting, merging, or
 * deriving mechanics.
 *
 * @param {HTMLElement} container
 * @param {unknown[]} items
 * @param {"status" | "modifier"} kind
 * @param {string} emptyText
 * @param {number} globalSlot
 */
function renderFactTokens(container, items, kind, emptyText, globalSlot) {
  const nodes = [];
  for (const rawItem of items) {
    const item = isRecord(rawItem) ? rawItem : {};
    const token = resolveVisualToken(kind, item.token_id, item);
    const value =
      kind === "status"
        ? `duration ${Number.isInteger(item.duration) ? item.duration : "unknown"}`
        : `multiplier ${formatDisplayNumber(item.multiplier)}`;
    const chip = htmlElement(
      "span",
      `roster-fact-token roster-fact-token--${kind}`,
      kind === "status"
        ? `${token.shortLabel} ${Number.isInteger(item.duration) ? item.duration : "?"}`
        : `${token.shortLabel} ×${formatDisplayNumber(item.multiplier)}`,
    );
    chip.dataset.tokenId = token.tokenId;
    if (kind === "status" && Number.isInteger(item.duration)) {
      chip.dataset.duration = String(item.duration);
    }
    if (kind === "modifier" && Number.isFinite(item.multiplier)) {
      chip.dataset.multiplier = String(item.multiplier);
    }
    chip.setAttribute("aria-label", `${token.accessibleName}, ${value}`);
    chip.tabIndex = 0;
    registerTooltipOwner(
      chip,
      kind === "status"
        ? explainStatus(item, globalSlot)
        : explainModifier(item, globalSlot),
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
 */
function targetLabel(target) {
  if (!isRecord(target)) {
    return "Undisclosed";
  }
  if (target.disclosure === "target_none") {
    return "target-none";
  }
  if (target.disclosure === "public" && Number.isInteger(target.global_slot)) {
    return `id_${target.global_slot}`;
  }
  return humanize(target.disclosure ?? "undisclosed");
}

/**
 * @param {unknown} lane
 */
function laneLabel(lane) {
  return lane === 0 ? "0/B · Basic" : lane === 1 ? "1/U · Ultimate" : "Not armed";
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
  if (pending.armed_lane !== 0 && pending.armed_lane !== 1) {
    return "Not applicable";
  }
  return availabilityLabel(pending.pair_mask_value);
}

/**
 * @param {HTMLElement} container
 * @param {Record<string, any>} hud
 */
function addCandidateLegality(container, hud) {
  const candidates = asArray(hud.candidate_legalities).filter(isRecord);
  if (candidates.length === 0) {
    return;
  }
  const group = htmlElement("section", "candidate-legality");
  group.setAttribute("aria-label", "Exact controlled-actor candidate legality");
  group.append(htmlElement("h3", null, "All candidate lanes"));
  const rows = htmlElement("div", "candidate-legality__rows");
  for (const candidate of candidates) {
    const target = isRecord(candidate.target) ? candidate.target : {};
    const targetAction = Number(candidate.target_action);
    const hasPublicTargetSlot =
      target.disclosure === "public" && Number.isInteger(target.global_slot);
    const targetSlot = hasPublicTargetSlot ? Number(target.global_slot) : null;
    const targetLabel =
      target.disclosure === "target_none"
        ? "target-none"
        : targetSlot !== null
          ? `id_${targetSlot}`
          : "undisclosed";
    const lane0Available = Boolean(candidate.lane_0_available);
    const lane1Available = Boolean(candidate.lane_1_available);
    const basicAvailable = Boolean(candidate.basic_available);
    const ultimateAvailable = Boolean(candidate.ultimate_available);
    const row = htmlElement("div", "candidate-legality-row");
    row.dataset.targetAction = String(targetAction);
    row.dataset.lane0Available = String(lane0Available);
    row.dataset.lane1Available = String(lane1Available);
    row.dataset.basicAvailable = String(basicAvailable);
    row.dataset.ultimateAvailable = String(ultimateAvailable);
    if (targetSlot !== null) {
      row.dataset.targetSlot = String(targetSlot);
    }
    row.setAttribute(
      "aria-label",
      `${targetLabel}, target action ${targetAction}, Basic ${basicAvailable ? "available" : "unavailable"}, Ultimate ${ultimateAvailable ? "available" : "unavailable"}`,
    );
    row.append(
      htmlElement("strong", "candidate-legality-row__target", targetLabel),
      htmlElement(
        "span",
        "candidate-legality-row__lane",
        `Basic ${basicAvailable ? "Available" : "Unavailable"}`,
      ),
      htmlElement(
        "span",
        "candidate-legality-row__lane",
        `Ultimate ${ultimateAvailable ? "Available" : "Unavailable"}`,
      ),
    );
    rows.append(row);
  }
  group.append(rows);
  container.append(group);
}

/**
 * @param {HTMLElement} container
 * @param {Record<string, any> | null} agent
 * @param {"Controlled actor" | "Selected target"} role
 */
function addAgentComparison(container, agent, role) {
  const card = htmlElement("article", "comparison-agent");
  card.tabIndex = 0;
  card.dataset.role = role === "Controlled actor" ? "controlled" : "selected";
  card.append(htmlElement("h3", null, role));
  if (!agent) {
    card.append(
      htmlElement(
        "p",
        "empty-copy",
        role === "Selected target"
          ? "No target selected."
          : "Controlled actor unavailable.",
      ),
    );
    container.append(card);
    return;
  }
  const classToken = classTokenFromId(agent.class_id);
  const teamToken = teamTokenFromId(agent.team_id);
  card.dataset.slot = String(agent.global_slot);
  registerTooltipOwner(
    card,
    explainAgent(agent, {
      controlled: role === "Controlled actor",
      selected: role === "Selected target",
    }),
  );
  addFact(card, "Identity", `id_${agent.global_slot}`);
  addFact(card, "Class / team", `${classToken.label} · ${teamToken.label}`);
  addFact(
    card,
    "Health",
    `${formatDisplayNumber(agent.current_health)} / ${formatDisplayNumber(agent.max_health)}`,
  );
  addFact(card, "Ultimate cooldown", agent.ultimate_cooldown ?? "—");
  addFact(card, "Effective speed", formatDisplayNumber(agent.effective_speed));
  const statuses = htmlElement("div", "comparison-agent__facts");
  statuses.append(htmlElement("h4", null, "Statuses"));
  const statusTokens = htmlElement("div", "roster-fact-list");
  renderFactTokens(
    statusTokens,
    asArray(agent.statuses),
    "status",
    "No persistent statuses",
    Number(agent.global_slot),
  );
  statuses.append(statusTokens);
  const modifiers = htmlElement("div", "comparison-agent__facts");
  modifiers.append(htmlElement("h4", null, "Effective modifiers"));
  const modifierTokens = htmlElement("div", "roster-fact-list");
  renderFactTokens(
    modifierTokens,
    asArray(agent.modifiers),
    "modifier",
    "No effective modifiers",
    Number(agent.global_slot),
  );
  modifiers.append(modifierTokens);
  card.append(statuses, modifiers);
  container.append(card);
}

/**
 * @param {PendingActionRow} row
 * @param {Record<string, any>} pending
 * @param {boolean} controlled
 */
function updatePendingActionRow(row, pending, controlled) {
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
  heading.append(
    htmlElement("strong", null, `id_${pending.actor_global_slot}`),
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
  const facts = htmlElement("div", "pending-action-row__facts");
  facts.append(
    htmlElement(
      "span",
      "pending-action-chip",
      `Move ${pending.move_action ?? "—"} · ${availabilityLabel(pending.movement_mask_value)}`,
    ),
    htmlElement("span", "pending-action-chip", `Target ${targetLabel(pending.target)}`),
    htmlElement("span", "pending-action-chip", `Lane ${laneLabel(pending.armed_lane)}`),
    htmlElement("span", "pending-action-chip", `Pair ${pendingPairMaskLabel(pending)}`),
  );
  element.replaceChildren(heading, summary, facts);
  element.setAttribute(
    "aria-label",
    `Pending action for id_${pending.actor_global_slot}: ${String(pending.summary ?? "unavailable")}`,
  );
  registerTooltipOwner(element, {
    kind: "pending-route",
    id: `pending-row:${pending.actor_global_slot}`,
    title: `Pending action · id_${pending.actor_global_slot}`,
    details: [
      String(pending.summary ?? "Pending action unavailable"),
      `Movement mask ${availabilityLabel(pending.movement_mask_value)}`,
      `Target ${targetLabel(pending.target)}`,
      `Lane ${laneLabel(pending.armed_lane)}`,
      `Pair ${pendingPairMaskLabel(pending)}`,
    ],
    anchor: "element",
  });
}

/**
 * @param {HTMLElement} container
 * @param {string} label
 * @param {Record<string, any>} action
 */
function addActionTuple(container, label, action) {
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
  addFact(facts, "Target", targetLabel(action.target));
  addFact(
    facts,
    "Combat lane",
    action.use_ultimate_action === 0
      ? isRecord(action.target) && action.target.disclosure === "target_none"
        ? "No combat"
        : "0/B · Basic"
      : action.use_ultimate_action === 1
        ? "1/U · Ultimate"
        : "Undisclosed",
  );
  card.append(facts);
  container.append(card);
}

/**
 * @param {HTMLElement} container
 * @param {Record<string, any> | null} latest
 * @returns {string | null}
 */
function renderLatestTransition(container, latest) {
  container.replaceChildren();
  container.removeAttribute("data-transition-id");
  if (!latest) {
    container.append(htmlElement("p", "empty-copy", "No transition yet."));
    return null;
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
    actor.dataset.actorSlot = String(rawActor.actor_global_slot);
    actor.dataset.combatResult = String(rawActor.combat_result ?? "undisclosed");
    actor.append(htmlElement("h3", null, `Actor id_${rawActor.actor_global_slot}`));
    const comparison = htmlElement("div", "action-result__comparison");
    if (isRecord(rawActor.submitted)) {
      addActionTuple(comparison, "Submitted", rawActor.submitted);
    }
    if (isRecord(rawActor.accepted)) {
      addActionTuple(comparison, "Accepted", rawActor.accepted);
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
      `id_${rawActor.actor_global_slot}: ${String(acceptedSummary ?? "accepted action unavailable")}; combat ${humanize(rawActor.combat_result ?? "undisclosed")}`,
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
 */
function eventSummary(event) {
  if (!isRecord(event)) {
    return "Unknown event";
  }
  switch (event.event_type) {
    case "accepted_activation": {
      const token = resolveVisualToken("activation", event.token_id, event);
      const source = Number.isInteger(event.source_global_slot)
        ? `id_${event.source_global_slot}`
        : "undisclosed source";
      const target = Number.isInteger(event.target_global_slot)
        ? `id_${event.target_global_slot}`
        : event.target_disclosure === "target_none"
          ? "source-local"
          : humanize(event.target_disclosure ?? "undisclosed target");
      return `${token.label} · ${source} → ${target} · ${laneLabel(event.lane)}`;
    }
    case "net_health": {
      const recipient = Number.isInteger(event.recipient_global_slot)
        ? `id_${event.recipient_global_slot}`
        : "undisclosed recipient";
      const delta = Number(event.net_delta);
      const signed = Number.isFinite(delta)
        ? `${delta >= 0 ? "+" : ""}${formatDisplayNumber(delta)}`
        : "undisclosed";
      return `${recipient} · NET ${signed} · HP ${formatDisplayNumber(event.health_before)} → ${formatDisplayNumber(event.health_after)} · ${humanize(event.outcome ?? "unknown")}`;
    }
    case "charge_displacement":
      return `Charge displacement endpoints · id_${event.source_global_slot} toward id_${event.target_global_slot} · ${pointLabel(event.start)} → ${pointLabel(event.end)} · ${humanize(event.path_kind ?? "unknown")}`;
    case "status_lifecycle": {
      const status = resolveVisualToken("status", event.token_id, event);
      return `${status.label} · id_${event.recipient_global_slot} · ${humanize(event.change ?? "unknown")} · duration ${event.duration_before ?? "?"} → ${event.duration_after ?? "?"}`;
    }
    case "rejected_action": {
      const target = Number.isInteger(event.target_global_slot)
        ? `id_${event.target_global_slot}`
        : humanize(event.target_disclosure ?? "undisclosed target");
      return `Rejected ${humanize(event.component ?? "action")} · actor id_${event.actor_global_slot} · target ${target} · movement mask ${Number(Boolean(event.movement_mask_value))} · pair mask ${Number(Boolean(event.pair_mask_value))}`;
    }
    default:
      return `Unknown semantic event · ${String(event.event_type ?? "missing type")}`;
  }
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
    /** @type {Map<number, RosterRow>} */
    this.rosterRows = new Map();
    /** @type {Map<number, RosterTeamGroup>} */
    this.rosterTeamGroups = new Map();
    /** @type {Map<string, HTMLElement>} */
    this.eventRows = new Map();
    /** @type {Map<string, HTMLElement>} */
    this.diagnosticRows = new Map();
    /** @type {Map<number, PendingActionRow>} */
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
   * @param {RosterRow} row
   * @param {Record<string, any>} agent
   * @param {Record<string, any>} selection
   * @param {boolean} disabled
   * @param {boolean} compact
   */
  updateRosterRow(row, agent, selection, disabled, compact) {
    const globalSlot = agent.global_slot;
    const classToken = classTokenFromId(agent.class_id);
    const teamToken = teamTokenFromId(agent.team_id);
    row.element.setAttribute(
      "aria-label",
      `Agent id_${globalSlot}, ${classToken.label}, ${teamToken.label}`,
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
    row.element.dataset.compact = String(compact);
    registerTooltipOwner(
      row.element,
      explainAgent(agent, {
        controlled: globalSlot === selection.controlled_global_slot,
        selected: globalSlot === selection.selected_global_slot,
      }),
    );

    row.identityId.textContent = `id_${globalSlot}`;
    row.identityClass.textContent = `${classToken.label} · ${teamToken.label}`;
    row.health.textContent =
      `HP ${formatDisplayNumber(agent.current_health)} / ${formatDisplayNumber(agent.max_health)}` +
      ` · cooldown ${agent.ultimate_cooldown ?? "—"}`;

    row.statuses.hidden = compact;
    row.modifiers.hidden = compact;
    if (compact) {
      row.statuses.replaceChildren();
      row.modifiers.replaceChildren();
    } else {
      renderFactTokens(
        row.statuses,
        asArray(agent.statuses),
        "status",
        "No persistent statuses",
        Number(globalSlot),
      );
      renderFactTokens(
        row.modifiers,
        asArray(agent.modifiers),
        "modifier",
        "No effective modifiers",
        Number(globalSlot),
      );
    }

    row.targetButton.setAttribute("aria-label", `Target id_${globalSlot}`);
    row.targetButton.setAttribute(
      "aria-pressed",
      String(globalSlot === selection.selected_global_slot),
    );
    row.targetButton.disabled = disabled;
    row.controlButton.setAttribute("aria-label", `Control id_${globalSlot}`);
    row.controlButton.setAttribute(
      "aria-pressed",
      String(globalSlot === selection.controlled_global_slot),
    );
    row.controlButton.disabled = disabled;
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
   * @param {boolean} compact
   */
  renderRoster(frame, disabled, retainDisabledFocus, compact) {
    const scene = frameScene(frame);
    const agents = asArray(scene?.agents)
      .filter((agent) => isRecord(agent) && Number.isInteger(agent.global_slot))
      .sort((left, right) => left.global_slot - right.global_slot);
    const selection = isRecord(scene?.selection) ? scene.selection : {};
    const focused = this.focusedRosterControl();
    if (focused) {
      this.pendingRosterFocus = {
        role: focused.role,
        slot: focused.slot,
      };
    }

    this.roster.dataset.compact = String(compact);
    this.rosterCount.textContent = `${agents.length} visible`;
    const activeSlots = new Set(agents.map((agent) => Number(agent.global_slot)));
    for (const [globalSlot, row] of this.rosterRows) {
      if (!activeSlots.has(globalSlot)) {
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
      this.updateRosterRow(row, agent, selection, disabled, compact);
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
   */
  renderPendingPlan(hud) {
    /** @type {"joint_turn" | "controlled_actor" | "scripted_playback"} */
    const scope =
      hud.pending_submission_scope === "joint_turn" ||
      hud.pending_submission_scope === "controlled_actor" ||
      hud.pending_submission_scope === "scripted_playback"
        ? hud.pending_submission_scope
        : "controlled_actor";
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
      scripted_playback:
        "Inspection only. Press N to advance the registered scripted trajectory.",
    };
    const heading = {
      joint_turn: "Pending joint turn",
      controlled_actor: "Pending controlled actor",
      scripted_playback: "Scripted playback",
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
        (scope === "scripted_playback"
          ? "PLAYBACK / INSPECTION ONLY"
          : "PENDING / WILL SUBMIT"),
    );

    const activeSlots = new Set(
      pendingActions.map((pending) => Number(pending.actor_global_slot)),
    );
    for (const [actorSlot, row] of this.pendingActionRows) {
      if (!activeSlots.has(actorSlot)) {
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
      updatePendingActionRow(row, pending, actorSlot === controlledSlot);
      desired.push(row.element);
    }
    this.reconcileChildren(
      this.pendingList,
      desired.length > 0 ? desired : [this.emptyPending],
    );
  }

  /**
   * @param {Record<string, any> | null} frame
   */
  renderInspector(frame) {
    const scene = frameScene(frame);
    const hud = isRecord(frame?.hud) ? frame.hud : {};
    const preset =
      frame?.preset === "presentation" ||
      frame?.preset === "analysis" ||
      frame?.preset === "debug"
        ? frame.preset
        : "analysis";
    const selection = isRecord(scene?.selection) ? scene.selection : null;
    const legality = isRecord(scene?.selected_legality)
      ? scene.selected_legality
      : null;
    const presentation = preset === "presentation";
    const agents = asArray(scene?.agents).filter(isRecord);
    const agentsBySlot = new Map(
      agents
        .filter((agent) => Number.isInteger(agent.global_slot))
        .map((agent) => [Number(agent.global_slot), agent]),
    );
    const selectionSection = this.selectionCard.closest(".hud-section");
    selectionSection?.toggleAttribute("hidden", presentation);
    this.selectionCard.replaceChildren();
    if (presentation) {
      // Presentation keeps a compact exact roster plus pending/result/event
      // story; detailed actor comparison belongs to Analysis and Debug.
    } else if (!selection) {
      this.selectionCard.append(
        htmlElement("p", "empty-copy", "No selection facts received yet."),
      );
    } else {
      const comparison = htmlElement("div", "selection-comparison");
      addAgentComparison(
        comparison,
        agentsBySlot.get(Number(selection.controlled_global_slot)) ?? null,
        "Controlled actor",
      );
      addAgentComparison(
        comparison,
        Number.isInteger(selection.selected_global_slot)
          ? (agentsBySlot.get(Number(selection.selected_global_slot)) ?? null)
          : null,
        "Selected target",
      );
      this.selectionCard.append(comparison);
      if (legality && preset !== "presentation") {
        const selectedLegality = htmlElement("section", "selected-legality");
        selectedLegality.setAttribute(
          "aria-label",
          `Exact selected-target legality for id_${legality.target_global_slot}`,
        );
        selectedLegality.append(htmlElement("h3", null, "Selected target legality"));
        const facts = htmlElement("div", "selected-legality__facts");
        addFact(facts, "Target action", legality.target_action);
        addFact(facts, "Basic lane", availabilityLabel(legality.lane_0_available));
        addFact(facts, "Ultimate lane", availabilityLabel(legality.lane_1_available));
        addFact(facts, "Armed lane", laneLabel(legality.armed_lane));
        addFact(
          facts,
          "Armed pair",
          legality.armed_lane === 0 || legality.armed_lane === 1
            ? legality.armed_pair_legal
              ? "Legal"
              : "Illegal"
            : "Not applicable",
        );
        selectedLegality.append(facts);
        this.selectionCard.append(selectedLegality);
      }
    }

    if (preset === "debug") {
      addCandidateLegality(this.selectionCard, hud);
    }
    const latestCandidate = hud.latest_transition ?? null;
    const latest = isRecord(latestCandidate) ? latestCandidate : null;
    this.renderPendingPlan(hud);
    const announcement = renderLatestTransition(this.acceptedCard, latest);
    if (latest && announcement) {
      const transitionKey = `${frame?.run_generation ?? "unknown"}:${latest.transition_id}`;
      if (transitionKey !== this.lastAnnouncedTransitionKey) {
        this.lastAnnouncedTransitionKey = transitionKey;
        this.acceptedAnnouncement.textContent = announcement;
      }
    } else {
      this.lastAnnouncedTransitionKey = null;
      this.acceptedAnnouncement.textContent = "";
    }
    this.renderDiagnostics(frame, hud, preset);
  }

  /**
   * @param {Record<string, any> | null} frame
   * @param {Record<string, any>} hud
   * @param {"presentation" | "analysis" | "debug"} preset
   */
  renderDiagnostics(frame, hud, preset) {
    const diagnosticsSection = this.diagnosticsCard.closest(".diagnostics");
    diagnosticsSection?.toggleAttribute("hidden", preset === "presentation");
    if (preset === "debug") {
      const raw = htmlElement(
        "pre",
        "record-card technical-json",
        formatRecord(frame, "No frame received."),
      );
      this.diagnosticsCard.replaceChildren(raw);
      return;
    }
    if (preset === "presentation") {
      this.diagnosticsCard.replaceChildren();
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
   */
  renderEvents(frame) {
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
    for (const event of events) {
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
      item.textContent = eventSummary(event);
      if (event.event_type === "accepted_activation") {
        registerTooltipOwner(item, explainActivation(event));
      } else if (event.event_type === "net_health") {
        registerTooltipOwner(item, explainNetHealth(event));
      } else {
        registerTooltipOwner(item, {
          kind: "event",
          id: `event:${eventId}`,
          title: humanize(event.event_type ?? "Semantic event"),
          details: [eventSummary(event)],
          anchor: "element",
        });
      }
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
    const compactRoster = frame?.preset === "presentation";
    this.renderRoster(frame, disabled, Boolean(interactionState.busy), compactRoster);
    this.renderInspector(frame);
    this.renderEvents(frame);
  }
}
