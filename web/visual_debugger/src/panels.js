/**
 * @file Render roster, agent details, action comparisons and frame facts.
 * Only installed, branded presentations supply scientific data. This module
 * updates DOM nodes and sends activation commands through the caller; it
 * does not step the simulator or infer facts missing from an authorized view.
 */
import { canonicalAgentIdentity } from "./agent-identity.js";
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
 *   activatedAgentPublicId?: string | null,
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
 * Return whether value is a non-null object other than an array. This
 * shallow check accepts class instances and does not validate their fields.
 *
 * @param {unknown} value
 * @returns {value is Record<string, any>}
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return an agent view with exact neutral multiplier rows removed. Prefer
 * aura_modifiers when it is an array, otherwise modifiers. Copy that array
 * and the outer agent; preserve other references. With neither array, return
 * the original agent. This display filter changes no recorded metric or state.
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
 * Return value unchanged when it is an array, otherwise a new empty array.
 * Elements are neither validated nor copied.
 *
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
 * Return the initial open state for a known DISCLOSURE_PANEL_IDS member.
 * Roster Details is open in both modes; Command Deck is also open when replay
 * is false. Every other known panel starts closed. Throw RangeError for an
 * unknown panelId. This does not read or update a user's saved panel state.
 *
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
 * Replace container's children with a hidden title and full details view.
 * Pass descriptor to the shared semantic renderer and return its result.
 * The caller owns visibility, closing and focus return. This replaces DOM
 * content but does not change descriptor or register a new data authority.
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
 * Return the sole class-mechanics row matching owner's exact numeric ID
 * and class name. Return null for a non-array bank, invalid owner identity,
 * no match or multiple matches. Do not coerce values or copy the matched row.
 * Both inputs must already belong to the same authorized presentation.
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
 * Build the selected agent's class-details view from a branded presentation.
 * Return null when presentation/audience/scene is unavailable. Replay and
 * researcher-space frames use the researcher scene; other frames use the
 * scene selected by localInspectedPresentationKey (undefined uses its default).
 * A nonempty activatedAgentPublicId selects that exact public identity;
 * otherwise use the scene's inspection owner. Missing or ambiguous class
 * mechanics leaves owner_descriptor null. Return a frozen outer record with
 * referenced owner data; this does not grant battlefield or command access.
 *
 * @param {unknown} presentation
 * @param {string | null | undefined} [localInspectedPresentationKey]
 * @param {string | null | undefined} [activatedAgentPublicId]
 * @returns {Readonly<Record<string, any>> | null}
 */
export function authorizedInspectorView(
  presentation,
  localInspectedPresentationKey = undefined,
  activatedAgentPublicId = undefined,
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
    typeof activatedAgentPublicId === "string" && activatedAgentPublicId.length > 0
      ? (agents.find((agent) => agent.public_agent_id === activatedAgentPublicId) ??
        null)
      : ownerKey === null
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
 * Create an unattached element of tagName. Set className only when truthy
 * and textContent when text is not null; both default to null. Return the
 * element. Text is inserted as text, not parsed as HTML.
 *
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
 * Append one labelled fact row to container and return its element.
 * Convert value with String and insert both fields as text. This performs no
 * scientific validation or tooltip registration.
 *
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
 * Create an unattached action card labelled Submitted, Accepted or Pending.
 * Read move_action, target_action and use_ultimate_action from action without
 * recomputing acceptance. Return the section; inputs must be authorized rows.
 *
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
 * Create an unattached ordered list from already authorized action rows.
 * With pending=false, show submitted and accepted tuples side by side; true
 * shows pending_action only. Preserve input order and actor labels. Return
 * the list without modifying rows or inferring missing action fields.
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
 * Return the compact display form of an integer duration, otherwise ?.
 * This accepts any JavaScript integer, including negative values; validation
 * of legal durations belongs upstream. Exact duration remains in chip data
 * and accessible text when renderFactTokens builds the chip.
 *
 * @param {unknown} duration
 */
export function rosterStatusDurationLabel(duration) {
  return Number.isInteger(duration) ? formatCompactDisplayNumber(duration) : "?";
}

/**
 * Replace container's children with status or modifier chips in input order.
 * Hide exact multiplier=1 modifier rows; show emptyText if no chips remain.
 * Use recipient and authorized sourceAgents only for explanation text.
 * agent_pov status chips use the restricted explanation route. Register
 * tooltips and keyboard focus on each chip; do not derive combat mechanics
 * or mutate the input records. kind and audience must be the declared values.
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
 * Keep debugger panel DOM nodes in sync with installed presentations.
 * Reuse roster rows by display identity to preserve focus where possible.
 * The caller supplies all DOM bindings and an activation command handler.
 * This class owns panel updates, not transport, simulation or authorization.
 * Call render for each installed frame; invalid frames clear scientific views.
 */
export class DebuggerPanels {
  /**
   * Store required DOM bindings, clear roster and create its team/visibility
   * groups. Initialize row caches and transition-announcement state. onCommand
   * receives activation records and pointer-origin context; returned promises
   * are not awaited here. Bindings must be real, compatible DOM elements.
   * Construction changes the supplied roster immediately and returns the instance.
   *
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
   * Return the cached visible or not-visible roster group, creating it if
   * needed. A new group starts hidden with an empty-state row and is appended
   * to this.roster. visibility must be one of the two declared strings. The
   * returned object and its DOM nodes are shared mutable view state.
   *
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
   * Return a cached group for numeric teamId or create and append it.
   * Use the display vocabulary for labels, including unknown IDs; this does not
   * validate team membership. The new group starts with zero authorized agents.
   * Return shared DOM references for later count and row updates.
   *
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
   * Create an unattached roster row for one authorized identity. Its button
   * looks up the current activation record at click time and calls onCommand;
   * status/modifier chips are siblings so their clicks do not activate the agent.
   * Return mutable DOM references. The caller fills values, stores activation
   * and inserts the row. This installs a listener but performs no network call.
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
   * Update roster rows, labels, activation state and groups for presentation.
   * disabled blocks all row activation; localInspectedPresentationKey defaults
   * to the scene's inspection choice. Researcher-space Agent POV groups the
   * complete researcher roster by visibility; other views group authorized rows
   * by team. Reuse rows by display key, remove stale ones and register tooltips.
   * This changes DOM/cache state only and assumes a branded presentation.
   *
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
    // Live Debugger and Replay Agent POV deliberately share the same complete
    // researcher roster; battlefield presentation and interaction stay fog-scoped.
    const visibilityGroupedRoster = globalAgentRoster;
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
      const displayIdentity = canonicalAgentIdentity(agent).publicIdentity;
      const classToken = classTokenFromId(agent.class_id);
      const teamToken = teamTokenFromId(agent.team_id);
      row.element.dataset.presentationKey = String(agent.presentation_key);
      row.element.dataset.teamId = String(agent.team_id);
      row.element.dataset.classId = String(agent.class_id);
      row.element.dataset.team = teamToken.cssKey;
      if (visibilityGroupedRoster) {
        row.element.dataset.visibleInSnapshot = String(identity.visible_in_snapshot);
      } else {
        delete row.element.dataset.visibleInSnapshot;
      }
      row.element.removeAttribute("data-class");
      row.identityId.dataset.class = classToken.cssKey;
      row.element.setAttribute(
        "aria-label",
        `${displayIdentity}, ${classToken.label}, ${teamToken.label}`,
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
              public_agent_id: publicId,
            })
          : identity.activation_kind === "live_pov_global" &&
              researcherInspectionState.state_kind === "live_editable"
            ? Object.freeze({
                command_type: "activate_live_pov_agent",
                global_slot: identity.command_global_slot,
                public_agent_id: publicId,
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
          ? `Control and inspect ${displayIdentity}`
          : `Inspect ${displayIdentity}`,
      );
      row.identityId.textContent = displayIdentity;
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
   * Refresh agent details, pending/upcoming actions, accepted actions and
   * technical facts from presentation. Optional local inspection selects the
   * scene; activatedAgentPublicId=null shows the activation prompt, while an
   * omitted value allows scene-owner selection. Replay shows upcoming recorded
   * actions; editable live frames show the staged joint action. Suppress repeated
   * announcements while the session/transition key stays unchanged; returning
   * to an earlier key can announce it again. Replace panel contents without changing
   * recorded facts or submitting actions; input must already be branded.
   *
   * @param {Record<string, any>} presentation
   * @param {string | null | undefined} [localInspectedPresentationKey]
   * @param {string | null | undefined} [activatedAgentPublicId]
   */
  renderAuthorizedInspector(
    presentation,
    localInspectedPresentationKey = undefined,
    activatedAgentPublicId = undefined,
  ) {
    const inspector = authorizedInspectorView(
      presentation,
      localInspectedPresentationKey,
      activatedAgentPublicId,
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
    if (activatedAgentPublicId === null) {
      this.selectionCard.append(
        htmlElement(
          "p",
          "empty-copy",
          "Activate an agent to inspect its comprehensive class details.",
        ),
      );
    } else if (inspector === null || inspector.owner_descriptor === null) {
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
   * Make container's element children match desired by identity and order.
   * Remove unlisted elements and move/insert existing desired nodes in place.
   * The caller supplies a duplicate-free array of elements. Text nodes are not
   * part of this reconciliation. Return nothing; existing element state survives.
   *
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
   * Clear roster caches and scientific panel content and show unavailable
   * messages. Reset the last announced transition and action-scope attributes.
   * Keep reusable group DOM and bindings. Return nothing; raw transport data
   * is never used as a fallback when an installed presentation is absent.
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
   * Render a branded frame, or clear scientific panels for any other value.
   * interactionState defaults to {}; busy, shutdown, resync, offline or explicit
   * activationDisabled flags disable roster actions. Pass optional inspection
   * and activated public identity to their owned views. This mutates panel DOM
   * and caches, returns nothing, and neither advances simulation nor sends a
   * command until a user later activates a row.
   *
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
        interactionState.activatedAgentPublicId ?? null,
      );
      return;
    }
    this.renderUnavailable();
  }
}
