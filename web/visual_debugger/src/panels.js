import {
  authorizedPresentationAudience,
  authorizedPresentationHasResearcherSpace,
  authorizedPresentationIdentityRows,
  authorizedPresentationLatestTransitionId,
  authorizedPresentationPendingJointActionRows,
  authorizedPresentationResearcherInspectionState,
  authorizedPresentationResearcherSceneView,
  authorizedPresentationSceneView,
  authorizedPresentationTechnicalFacts,
  authorizedPresentationTransitionRows,
  authorizedPresentationUpcomingTransitionRows,
  isAuthorizedPresentationFrame,
} from "./authorized-presentation-adapter.js";
import { formatCompactDisplayNumber, formatDisplayNumber } from "./display.js";
import {
  explainAgent,
  explainClassDocumentation,
  explainModifier,
  explainPovAgent,
  explainPovStatus,
  explainStatus,
  explainTechnicalFact,
} from "./explanations.js";
import { createSvgIcon } from "./icons.js";
import { registerTooltipOwner, renderSemanticDescriptor } from "./tooltip.js";
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
 *   diagnosticsCard: HTMLElement,
 *   onCommand: (
 *     command: Record<string, unknown>,
 *     context?: Readonly<{pointerOriginated: boolean}>,
 *   ) => void | Promise<void>,
 * }} DebuggerPanelBindings
 */

/**
 * @typedef {{
 *   busy?: boolean,
 *   shuttingDown?: boolean,
 *   resyncRequired?: boolean,
 *   offline?: boolean,
 *   activationDisabled?: boolean,
 *   localInspectedPresentationKey?: string | null,
 * }} PanelInteractionState
 */

/**
 * @typedef {{
 *   element: HTMLElement,
 *   primaryButton: HTMLButtonElement,
 *   identityId: HTMLElement,
 *   identityClass: HTMLElement,
 *   health: HTMLElement,
 *   statuses: HTMLElement,
 *   modifiers: HTMLElement,
 * }} AuthorizedRosterRow
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

export const DISCLOSURE_PANEL_IDS = Object.freeze([
  "command-deck",
  "roster-details",
  "agent-details",
  "pending-turn-details",
  "latest-transition-details",
  "visual-key",
  "technical-frame-details",
]);

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
 * Join one already-authorized inspected owner to exactly one serialized class
 * mechanics row. Neither class name nor numeric-looking values are coerced.
 *
 * @param {Record<string, any>} owner
 * @param {unknown} rawClassMechanics
 */
function exactOwnerClassMechanics(owner, rawClassMechanics) {
  if (
    !Array.isArray(rawClassMechanics) ||
    !Number.isSafeInteger(owner.class_id) ||
    typeof owner.class_name !== "string"
  ) {
    return null;
  }
  const matches = rawClassMechanics.filter(
    (candidate) =>
      isRecord(candidate) &&
      candidate.class_id === owner.class_id &&
      candidate.class_name === owner.class_name,
  );
  return matches.length === 1 ? matches[0] : null;
}

/**
 * Project the authorized Agent Details class-documentation surface without
 * consulting raw transport, outgoing target identity, or recipient fallbacks.
 * The selected actor owns the returned identity and class descriptor; command
 * and battlefield legality remain with their dedicated consumers.
 *
 * @param {unknown} presentation
 * @param {string | null | undefined} [localInspectedPresentationKey]
 * @returns {Readonly<Record<string, any>> | null}
 */
export function authorizedInspectorView(
  presentation,
  localInspectedPresentationKey = undefined,
) {
  if (!isAuthorizedPresentationFrame(presentation)) {
    return null;
  }
  const scene =
    presentation.viewer_mode === "replay" ||
    authorizedPresentationHasResearcherSpace(presentation)
      ? authorizedPresentationResearcherSceneView(presentation)
      : authorizedPresentationSceneView(presentation, localInspectedPresentationKey);
  const audience = authorizedPresentationAudience(presentation);
  if (scene === null || (audience !== "researcher" && audience !== "agent_pov")) {
    return null;
  }

  const agents = asArray(scene.agents).filter(isRecord);
  const selection = isRecord(scene.selection) ? scene.selection : {};
  const ownerKey =
    typeof selection.inspection_owner_presentation_key === "string"
      ? selection.inspection_owner_presentation_key
      : null;
  const owner =
    ownerKey === null
      ? null
      : (agents.find((agent) => agent.presentation_key === ownerKey) ?? null);
  const ownerMechanics =
    owner === null ? null : exactOwnerClassMechanics(owner, scene.class_mechanics);
  const ownerDescriptor =
    owner === null || ownerMechanics === null
      ? null
      : explainClassDocumentation(owner, ownerMechanics);
  const ownerClassAccent =
    owner === null ? null : classTokenFromId(owner.class_id).cssKey;

  return Object.freeze({
    title: "Comprehensive Agent Class Details",
    owner,
    owner_descriptor: ownerDescriptor,
    owner_class_accent: ownerClassAccent,
  });
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
 * @returns {HTMLElement}
 */
function addFact(container, label, value) {
  const fact = htmlElement("div", "fact");
  fact.append(
    htmlElement("span", null, label),
    htmlElement("strong", null, String(value)),
  );
  container.append(fact);
  return fact;
}

/**
 * @param {"Submitted" | "Accepted" | "Pending"} label
 * @param {Record<string, any>} action
 */
function authorizedTransitionTuple(label, action) {
  const tuple = htmlElement("section", "accepted-action-tuple");
  tuple.dataset.kind = label.toLowerCase();
  tuple.append(
    htmlElement("h4", null, label),
    htmlElement(
      "p",
      "accepted-action-tuple__value",
      `Move ${action.move_action} · Target ${action.target_action} · Ultimate ${action.use_ultimate_action}`,
    ),
  );
  return tuple;
}

/**
 * Render action rows with one shared grammar for pending, incoming, and
 * upcoming researcher-space panels.
 *
 * @param {ReadonlyArray<Record<string, any>>} rows
 * @param {boolean} [pending]
 */
function authorizedTransitionList(rows, pending = false) {
  const list = htmlElement("ol", "accepted-action-list");
  for (const transition of rows) {
    const item = htmlElement("li", "accepted-action-row");
    item.dataset.team = transition.actor_team;
    const title = htmlElement(
      "h3",
      "accepted-action-row__title",
      transition.actor_title,
    );
    title.dataset.class = transition.actor_accent;
    const comparison = htmlElement("div", "accepted-action-row__comparison");
    comparison.dataset.layout = pending ? "single" : "comparison";
    if (pending) {
      comparison.append(
        authorizedTransitionTuple("Pending", transition.pending_action),
      );
    } else {
      comparison.append(
        authorizedTransitionTuple("Submitted", transition.submitted_action),
        authorizedTransitionTuple("Accepted", transition.accepted_action),
      );
    }
    item.append(title, comparison);
    list.append(item);
  }
  return list;
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
    kind === "modifier"
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
 * Render the debugger's HTML inspection panels from an authoritative frame.
 *
 * Authorized roster rows are keyed by presentation identity. A render updates
 * those nodes in place, so ordinary authoritative refreshes retain DOM
 * identity and keyboard focus without exposing a raw transport fallback.
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
    this.diagnosticsCard = diagnosticsCard;
    this.onCommand = onCommand;
    /** @type {Map<string, AuthorizedRosterRow>} */
    this.rosterRows = new Map();
    /** @type {Map<number, RosterTeamGroup>} */
    this.rosterTeamGroups = new Map();
    /** @type {Map<string, RosterTeamGroup>} */
    this.rosterVisibilityGroups = new Map();
    /** @type {WeakMap<HTMLButtonElement, Readonly<Record<string, unknown>>>} */
    this.rosterActivationByButton = new WeakMap();
    /** @type {string | null} */
    this.lastAnnouncedTransitionKey = null;
    this.roster.replaceChildren();
    this.ensureRosterTeamGroup(1);
    this.ensureRosterTeamGroup(2);
    this.ensureRosterVisibilityGroup("visible");
    this.ensureRosterVisibilityGroup("not-visible");
  }

  /**
   * @param {"visible" | "not-visible"} visibility
   * @returns {RosterTeamGroup}
   */
  ensureRosterVisibilityGroup(visibility) {
    const existing = this.rosterVisibilityGroups.get(visibility);
    if (existing) {
      return existing;
    }
    const visible = visibility === "visible";
    const element = htmlElement("section", "roster-team roster-visibility");
    element.dataset.visibility = visibility;
    element.hidden = true;
    const heading = htmlElement("div", "roster-team__heading");
    const title = htmlElement("h3", null, visible ? "VISIBLE" : "NOT VISIBLE");
    const count = htmlElement("span", "count-badge", "0 agents");
    heading.append(title, count);
    const rows = htmlElement("div", "roster-team__rows");
    const empty = htmlElement(
      "p",
      "empty-copy",
      visible ? "No agents are visible." : "No agents are outside this snapshot.",
    );
    rows.append(empty);
    element.append(heading, rows);
    this.roster.append(element);
    const group = { element, count, rows, empty };
    this.rosterVisibilityGroups.set(visibility, group);
    return group;
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
   * Create one presentation-keyed native action. Scientific fact chips remain
   * siblings of the button so their independent tooltip interactions cannot
   * trigger agent activation. Main resolves the opaque key against the current
   * installed authority before choosing any local or network effect.
   *
   * @param {ReturnType<typeof authorizedPresentationIdentityRows>[number]} identity
   * @returns {AuthorizedRosterRow}
   */
  createAuthorizedRosterRow(identity) {
    const element = htmlElement("article", "roster-row roster-row--authorized");
    element.dataset.presentationKey = identity.presentation_key;
    const primaryButton = htmlElement("button", "roster-primary-action");
    primaryButton.type = "button";
    primaryButton.dataset.action = "activate-agent";
    primaryButton.dataset.presentationKey = identity.presentation_key;
    const summary = htmlElement("span", "roster-primary-action__summary");
    const identityContainer = htmlElement("span", "roster-identity");
    const identityId = htmlElement("span", "roster-id");
    const identityClass = htmlElement("span", "roster-class");
    identityContainer.append(identityId, identityClass);
    const health = htmlElement("span", "roster-health");
    summary.append(identityContainer, health);
    primaryButton.append(summary);
    primaryButton.addEventListener("click", (event) => {
      const command = this.rosterActivationByButton.get(primaryButton);
      if (command !== undefined) {
        void this.onCommand(command, {
          pointerOriginated: Number(event.detail) > 0,
        });
      }
    });

    const facts = htmlElement("div", "roster-row__facts");
    const statuses = htmlElement("div", "roster-fact-list roster-statuses");
    statuses.setAttribute("aria-label", "Persistent statuses");
    const modifiers = htmlElement("div", "roster-fact-list roster-modifiers");
    modifiers.setAttribute("aria-label", "Exact effective modifiers");
    facts.append(statuses, modifiers);
    element.append(primaryButton, facts);
    return {
      element,
      primaryButton,
      identityId,
      identityClass,
      health,
      statuses,
      modifiers,
    };
  }

  /**
   * @param {Record<string, any>} presentation
   * @param {boolean} disabled
   * @param {string | null | undefined} [localInspectedPresentationKey]
   */
  renderAuthorizedRoster(
    presentation,
    disabled,
    localInspectedPresentationKey = undefined,
  ) {
    const audience = authorizedPresentationAudience(presentation);
    const identities = authorizedPresentationIdentityRows(presentation);
    const researcherInspectionState =
      authorizedPresentationResearcherInspectionState(presentation);
    const globalAgentRoster =
      audience === "agent_pov" &&
      authorizedPresentationHasResearcherSpace(presentation);
    const visibilityGroupedRoster =
      globalAgentRoster && presentation.viewer_mode === "replay";
    const scene = globalAgentRoster
      ? authorizedPresentationResearcherSceneView(presentation)
      : authorizedPresentationSceneView(presentation, localInspectedPresentationKey);
    const rosterAudience = scene?.audience ?? audience;
    const selection = isRecord(scene?.selection) ? scene.selection : {};
    const sourceAgents =
      rosterAudience === "researcher" ? identities.map(({ agent }) => agent) : [];
    const activeKeys = new Set(identities.map(({ display_key }) => display_key));
    for (const [key, row] of this.rosterRows) {
      if (typeof key !== "string" || !activeKeys.has(key)) {
        row.element.remove();
        this.rosterRows.delete(key);
      }
    }

    const visibleCount = identities.filter(
      (identity) => identity.visible_in_snapshot,
    ).length;
    this.rosterCount.textContent = visibilityGroupedRoster
      ? `${identities.length} agents · ${visibleCount} visible · ${identities.length - visibleCount} not visible`
      : `${identities.length} ${identities.length === 1 ? "actor" : "actors"}`;
    /** @type {Map<number, HTMLElement[]>} */
    const desiredByTeam = new Map();
    /** @type {Map<string, HTMLElement[]>} */
    const desiredByVisibility = new Map();
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
      const candidateRow = this.rosterRows.get(identity.display_key);
      let row =
        candidateRow && "primaryButton" in candidateRow ? candidateRow : undefined;
      if (!row) {
        candidateRow?.element.remove();
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
      if (presentation.viewer_mode === "replay") {
        row.element.dataset.visibleInSnapshot = String(identity.visible_in_snapshot);
      } else {
        delete row.element.dataset.visibleInSnapshot;
      }
      row.element.removeAttribute("data-class");
      row.identityId.dataset.class = classToken.cssKey;
      row.element.setAttribute(
        "aria-label",
        `Agent ID ${publicId}, ${classToken.label}, ${teamToken.label}`,
      );
      registerTooltipOwner(
        row.primaryButton,
        rosterAudience === "researcher"
          ? explainAgent(agent)
          : explainPovAgent(agent, {
              controlled:
                agent.presentation_key === presentation.recipient_presentation_key,
              selected: false,
              inspected:
                selection.inspection_owner_presentation_key === agent.presentation_key,
            }),
        { inspectable: false },
      );
      row.primaryButton.dataset.presentationKey = String(agent.presentation_key);
      const activation = Number.isInteger(identity.command_global_slot)
        ? identity.activation_kind === "replay_pov_global"
          ? Object.freeze({
              command_type: "activate_replay_pov_agent",
              global_slot: identity.command_global_slot,
            })
          : identity.activation_kind === "live_pov_global" &&
              researcherInspectionState.state_kind === "live_editable"
            ? Object.freeze({
                command_type: "activate_live_pov_agent",
                global_slot: identity.command_global_slot,
              })
            : identity.activation_kind === "scene_agent"
              ? Object.freeze({
                  command_type: "activate_authorized_agent",
                  presentation_key: identity.presentation_key,
                })
              : null
        : identity.activation_kind === "scene_agent"
          ? Object.freeze({
              command_type: "activate_authorized_agent",
              presentation_key: identity.presentation_key,
            })
          : null;
      if (activation === null) {
        this.rosterActivationByButton.delete(row.primaryButton);
      } else {
        this.rosterActivationByButton.set(row.primaryButton, activation);
      }
      row.primaryButton.disabled = disabled || activation === null;
      const inspected =
        selection.inspection_owner_presentation_key === agent.presentation_key;
      row.primaryButton.setAttribute("aria-pressed", String(inspected));
      row.element.dataset.selected = String(inspected);
      row.primaryButton.setAttribute(
        "aria-label",
        rosterAudience === "researcher" &&
          presentation.viewer_mode === "live" &&
          researcherInspectionState.state_kind === "live_editable"
          ? `Control and inspect Agent ID ${publicId}`
          : `Inspect Agent ID ${publicId}`,
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
        rosterAudience ?? "agent_pov",
      );
      renderFactTokens(
        row.modifiers,
        asArray(agent.modifiers ?? agent.aura_modifiers),
        "modifier",
        "No effective modifiers",
        agent,
        sourceAgents,
        rosterAudience ?? "agent_pov",
      );
      if (visibilityGroupedRoster) {
        const visibility = identity.visible_in_snapshot ? "visible" : "not-visible";
        const desired = desiredByVisibility.get(visibility) ?? [];
        desired.push(row.element);
        desiredByVisibility.set(visibility, desired);
      } else {
        const desired = desiredByTeam.get(Number(agent.team_id)) ?? [];
        desired.push(row.element);
        desiredByTeam.set(Number(agent.team_id), desired);
        this.ensureRosterTeamGroup(Number(agent.team_id));
      }
    }
    for (const [teamId, group] of this.rosterTeamGroups) {
      group.element.hidden = visibilityGroupedRoster;
      const desired = desiredByTeam.get(teamId) ?? [];
      group.count.textContent = `${desired.length} authorized`;
      this.reconcileChildren(group.rows, desired.length > 0 ? desired : [group.empty]);
    }
    for (const [visibility, group] of this.rosterVisibilityGroups) {
      group.element.hidden = !visibilityGroupedRoster;
      const desired = desiredByVisibility.get(visibility) ?? [];
      group.count.textContent = `${desired.length} agents`;
      this.reconcileChildren(group.rows, desired.length > 0 ? desired : [group.empty]);
    }
  }

  /**
   * @param {Record<string, any>} presentation
   * @param {string | null | undefined} [localInspectedPresentationKey]
   */
  renderAuthorizedInspector(presentation, localInspectedPresentationKey = undefined) {
    const inspector = authorizedInspectorView(
      presentation,
      localInspectedPresentationKey,
    );
    const inspectionState =
      authorizedPresentationResearcherInspectionState(presentation);
    const replay = presentation.viewer_mode === "replay";
    const upcomingRows = replay
      ? authorizedPresentationUpcomingTransitionRows(presentation)
      : [];
    const pendingRows = replay
      ? []
      : authorizedPresentationPendingJointActionRows(presentation);
    this.selectionCard.replaceChildren();
    if (inspector === null || inspector.owner_descriptor === null) {
      this.selectionCard.append(
        htmlElement("p", "empty-copy", "No authorized agent details are available."),
      );
    } else {
      renderSemanticInspector(this.selectionCard, inspector.owner_descriptor);
    }

    if (inspectionState.submission_scope === null) {
      this.pendingCard.removeAttribute("data-submission-scope");
    } else {
      this.pendingCard.dataset.submissionScope = inspectionState.submission_scope;
    }
    this.pendingCard.dataset.inspectionState = inspectionState.state_kind;
    this.pendingHeading.textContent = {
      live_editable: "Pending Joint Action",
      live_scripted: "Scripted playback inspection",
      replay_outgoing: "Upcoming Transition",
      replay_none: "Upcoming Transition",
      unavailable: "Inspection unavailable",
    }[inspectionState.state_kind];
    this.pendingCount.hidden = replay;
    this.pendingScope.hidden = replay;
    this.pendingCount.textContent = `${pendingRows.length} ${pendingRows.length === 1 ? "actor" : "actors"}`;
    this.pendingScope.textContent =
      inspectionState.state_kind === "live_scripted"
        ? "This live frame advances registered scripted actions and has no editable draft."
        : inspectionState.state_kind === "live_editable"
          ? "This panel shows the complete researcher-space joint action staged for the next submission."
          : inspectionState.state_kind === "replay_outgoing"
            ? "This panel shows the authorized recorded actions out of the current frame."
            : inspectionState.state_kind === "replay_none"
              ? "No upcoming transition is available at this replay frame."
              : "Inspection is unavailable for this frame.";
    if (replay) {
      this.pendingCard.replaceChildren(
        upcomingRows.length > 0
          ? authorizedTransitionList(upcomingRows)
          : htmlElement("p", "empty-copy", "No upcoming transition is available."),
      );
    } else if (inspectionState.state_kind === "live_editable") {
      this.pendingCard.replaceChildren(
        pendingRows.length > 0
          ? authorizedTransitionList(pendingRows, true)
          : htmlElement("p", "empty-copy", "No pending joint action is available."),
      );
    } else {
      this.pendingCard.replaceChildren(
        htmlElement(
          "p",
          "empty-copy",
          inspectionState.state_kind === "live_scripted"
            ? "No editable joint action is available during scripted playback."
            : "No pending joint action is available.",
        ),
      );
    }

    const transitionRows = authorizedPresentationTransitionRows(presentation);
    this.acceptedCard.replaceChildren();
    if (transitionRows.length === 0) {
      this.acceptedAnnouncement.textContent = "";
      this.lastAnnouncedTransitionKey = null;
    } else {
      this.acceptedCard.append(authorizedTransitionList(transitionRows));
      const transitionId = authorizedPresentationLatestTransitionId(presentation);
      const sessionId = presentation.source?.source_session_id;
      const transitionKey =
        typeof sessionId === "string" && typeof transitionId === "string"
          ? `${sessionId}:${transitionId}`
          : null;
      if (transitionKey !== this.lastAnnouncedTransitionKey) {
        this.acceptedAnnouncement.textContent = `${transitionRows.length} Submitted / Accepted action ${transitionRows.length === 1 ? "row" : "rows"}.`;
        this.lastAnnouncedTransitionKey = transitionKey;
      }
    }

    const facts = authorizedPresentationTechnicalFacts(presentation);
    this.diagnosticsCard.replaceChildren();
    for (const fact of facts) {
      const owner = addFact(this.diagnosticsCard, fact.label, fact.value);
      owner.tabIndex = 0;
      owner.dataset.technicalFact = fact.id;
      registerTooltipOwner(owner, explainTechnicalFact(fact.id), {
        inspectable: false,
      });
    }
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
   * Clear every scientific panel when no branded presentation is installed.
   * Raw transport objects and forged lookalikes are deliberately unavailable;
   * they never become a fallback presentation source.
   */
  renderUnavailable() {
    for (const row of this.rosterRows.values()) {
      row.element.remove();
    }
    this.rosterRows.clear();
    for (const group of this.rosterTeamGroups.values()) {
      group.element.hidden = false;
      group.count.textContent = "0 authorized";
      this.reconcileChildren(group.rows, [group.empty]);
    }
    for (const group of this.rosterVisibilityGroups.values()) {
      group.element.hidden = true;
      group.count.textContent = "0 agents";
      this.reconcileChildren(group.rows, [group.empty]);
    }
    this.rosterCount.textContent = "0 actors";

    this.selectionCard.replaceChildren(
      htmlElement("p", "empty-copy", "No authorized agent details are available."),
    );
    this.pendingHeading.textContent = "Inspection unavailable";
    this.pendingCount.textContent = "0 actors";
    this.pendingScope.textContent = "Waiting for authorized action details.";
    this.pendingCard.removeAttribute("data-submission-scope");
    this.pendingCard.removeAttribute("data-inspection-state");
    this.pendingCard.removeAttribute("data-pending-count");
    this.pendingCard.replaceChildren(
      htmlElement("p", "empty-copy", "No authorized action details."),
    );

    this.acceptedCard.replaceChildren();
    this.acceptedAnnouncement.textContent = "";
    this.lastAnnouncedTransitionKey = null;

    this.diagnosticsCard.replaceChildren(
      htmlElement("p", "empty-copy", "Technical Frame unavailable."),
    );
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
        interactionState.offline ||
        interactionState.activationDisabled,
    );
    if (isAuthorizedPresentationFrame(frame)) {
      this.renderAuthorizedRoster(
        frame,
        disabled,
        interactionState.localInspectedPresentationKey,
      );
      this.renderAuthorizedInspector(
        frame,
        interactionState.localInspectedPresentationKey,
      );
      return;
    }
    this.renderUnavailable();
  }
}
