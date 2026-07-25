import {
  DebuggerApiError,
  acquireCapabilityToken,
  acquireClientId,
  extractFrame,
  extractNotice,
  getCurrentFrame,
  postCommand,
} from "./api.js";
import { bindBattlefieldControls, keyboardCommand } from "./controls.js";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

/**
 * @param {string} id
 * @returns {any}
 */
function requiredElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Browser shell is missing required element #${id}.`);
  }
  return element;
}

const elements = {
  connectionStatus: requiredElement("connection-status"),
  audienceBadge: requiredElement("audience-badge"),
  terminalBadge: requiredElement("terminal-badge"),
  scenarioSelect: requiredElement("scenario-select"),
  viewSelect: requiredElement("view-select"),
  presetSelect: requiredElement("preset-select"),
  revisionValue: requiredElement("revision-value"),
  stepValue: requiredElement("step-value"),
  transitionValue: requiredElement("transition-value"),
  reconnectButton: requiredElement("reconnect-button"),
  helpButton: requiredElement("help-button"),
  exitButton: requiredElement("exit-button"),
  resetButton: requiredElement("reset-button"),
  notice: requiredElement("notice"),
  scenarioDescription: requiredElement("scenario-description"),
  battlefieldShell: requiredElement("battlefield-shell"),
  battlefield: requiredElement("battlefield"),
  battlefieldEmpty: requiredElement("battlefield-empty"),
  commandDeck: document.querySelector(".command-deck"),
  roster: requiredElement("roster"),
  rosterCount: requiredElement("roster-count"),
  selectionCard: requiredElement("selection-card"),
  pendingCard: requiredElement("pending-card"),
  acceptedCard: requiredElement("accepted-card"),
  eventFeed: requiredElement("event-feed"),
  eventCount: requiredElement("event-count"),
  diagnosticsCard: requiredElement("diagnostics-card"),
  helpDialog: requiredElement("help-dialog"),
};

/**
 * @type {{
 *   token: string | null,
 *   clientId: string,
 *   frame: Record<string, any> | null,
 *   busy: boolean,
 *   offline: boolean,
 *   resyncRequired: boolean,
 *   shuttingDown: boolean,
 *   notice: string | null,
 *   noticeLevel: string,
 *   mapHeight: number,
 * }}
 */
const state = {
  token: acquireCapabilityToken(),
  clientId: acquireClientId(),
  frame: null,
  busy: false,
  offline: false,
  resyncRequired: false,
  shuttingDown: false,
  notice: null,
  noticeLevel: "info",
  mapHeight: 1,
};

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
 * @param {unknown} value
 * @param {number} fallback
 */
function finiteNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * @param {unknown} value
 * @param {number} fallback
 */
function integer(value, fallback = 0) {
  return Number.isInteger(value) ? value : fallback;
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
 * @param {unknown} frame
 * @returns {Record<string, any>}
 */
function scenarioRecord(frame) {
  return isRecord(frame) && isRecord(frame.scenario) ? frame.scenario : {};
}

/**
 * @param {unknown} frame
 */
function scenarioName(frame) {
  const scenario = scenarioRecord(frame);
  const record = isRecord(frame) ? frame : {};
  return typeof scenario.name === "string"
    ? scenario.name
    : typeof record.scenario_name === "string"
      ? record.scenario_name
      : "";
}

/**
 * @param {unknown} frame
 */
function scenarioDescription(frame) {
  const scenario = scenarioRecord(frame);
  const record = isRecord(frame) ? frame : {};
  return typeof scenario.description === "string"
    ? scenario.description
    : typeof record.scenario_description === "string"
      ? record.scenario_description
      : "Authoritative debugger frame received.";
}

/**
 * @param {unknown} frame
 */
function isTerminal(frame) {
  const record = isRecord(frame) ? frame : {};
  if (typeof record.terminal === "boolean") {
    return record.terminal;
  }
  if (isRecord(record.terminal)) {
    return Boolean(record.terminal.terminated || record.terminal.truncated);
  }
  return Boolean(record.terminated || record.truncated);
}

function currentRevision() {
  return integer(state.frame?.revision, 0);
}

/**
 * @param {string} message
 * @param {string} level
 */
function setNotice(message, level = "info") {
  state.notice = message;
  state.noticeLevel = level;
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
 * @param {number} digits
 */
function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
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
 * @param {string} tagName
 * @param {Record<string, unknown>} attributes
 * @returns {SVGElement}
 */
function svgElement(tagName, attributes = {}) {
  const element = document.createElementNS(SVG_NAMESPACE, tagName);
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined) {
      element.setAttribute(name, String(value));
    }
  }
  return element;
}

/**
 * @param {unknown} point
 * @param {number} mapHeight
 */
function screenPoint(point, mapHeight) {
  const values = Array.isArray(point) ? point : [0, 0];
  return {
    x: finiteNumber(values[0]),
    y: mapHeight - finiteNumber(values[1]),
  };
}

function renderConnection() {
  let label = "Online";
  let status = "online";
  if (state.shuttingDown) {
    label = "Shutting down";
    status = "busy";
  } else if (state.busy) {
    label = "Command in flight";
    status = "busy";
  } else if (state.resyncRequired) {
    label = "Resync required";
    status = "offline";
  } else if (state.offline) {
    label = "Offline";
    status = "offline";
  } else if (!state.frame) {
    label = "Connecting";
    status = "loading";
  }
  elements.connectionStatus.textContent = label;
  elements.connectionStatus.dataset.state = status;

  elements.notice.hidden = !state.notice;
  elements.notice.textContent = state.notice ?? "";
  elements.notice.dataset.level = state.noticeLevel;
  elements.battlefieldShell.setAttribute("aria-busy", String(state.busy));
}

function renderSessionToolbar() {
  const frame = state.frame;
  const scene = frameScene(frame);
  const disabled =
    state.busy || !frame || state.shuttingDown || state.resyncRequired || state.offline;

  elements.revisionValue.textContent = frame ? String(frame.revision ?? "—") : "—";
  elements.stepValue.textContent = frame
    ? String(frame.simulator_step ?? frame.step ?? "—")
    : "—";
  elements.transitionValue.textContent = frame
    ? String(frame.transition_id ?? "—")
    : "—";

  elements.audienceBadge.textContent =
    typeof scene?.audience_badge === "string"
      ? scene.audience_badge
      : typeof frame?.view_mode === "string"
        ? humanize(frame.view_mode)
        : "View unavailable";

  const terminal = isTerminal(frame);
  elements.terminalBadge.hidden = !terminal;
  elements.terminalBadge.textContent = terminal
    ? "Terminal · submissions blocked by Python"
    : "Terminal";

  elements.scenarioDescription.textContent = frame
    ? scenarioDescription(frame)
    : "Waiting for the Python debugger service.";

  renderScenarioOptions(frame);
  const viewMode = frame?.view_mode === "agent_pov" ? "pov" : frame?.view_mode;
  if (typeof viewMode === "string") {
    elements.viewSelect.value = viewMode;
  }
  if (typeof frame?.preset === "string") {
    elements.presetSelect.value = frame.preset;
  }
  elements.scenarioSelect.disabled = disabled;
  elements.viewSelect.disabled = disabled;
  elements.presetSelect.disabled = disabled;
  elements.reconnectButton.disabled = state.busy || state.shuttingDown;
  elements.exitButton.disabled = disabled;
  elements.resetButton.disabled = disabled;
  if (elements.commandDeck) {
    const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
      elements.commandDeck.querySelectorAll("button[data-key]")
    );
    for (const button of buttons) {
      button.disabled = disabled;
    }
  }
}

/**
 * @param {Record<string, any> | null} frame
 */
function renderScenarioOptions(frame) {
  const currentName = scenarioName(frame);
  const available = asArray(frame?.available_scenarios);
  const normalized = available
    .map((entry) => {
      if (typeof entry === "string") {
        return { name: entry, title: entry };
      }
      if (isRecord(entry) && typeof entry.name === "string") {
        return {
          name: entry.name,
          title:
            typeof entry.title === "string" && entry.title.trim()
              ? entry.title
              : entry.name,
        };
      }
      return null;
    })
    .filter((entry) => entry !== null);

  if (currentName && !normalized.some((entry) => entry.name === currentName)) {
    normalized.unshift({ name: currentName, title: currentName });
  }

  elements.scenarioSelect.replaceChildren();
  if (normalized.length === 0) {
    const option = htmlElement("option", null, frame ? "No scenarios" : "Loading…");
    option.value = "";
    elements.scenarioSelect.append(option);
    return;
  }
  for (const entry of normalized) {
    const option = htmlElement("option", null, entry.title);
    option.value = entry.name;
    option.selected = entry.name === currentName;
    elements.scenarioSelect.append(option);
  }
}

/**
 * @param {SVGElement} parent
 * @param {any[]} records
 * @param {string} className
 * @param {string} tokenAttribute
 */
function appendCircleLayer(parent, records, className, tokenAttribute) {
  for (const record of records) {
    if (!isRecord(record)) {
      continue;
    }
    const center = screenPoint(record.center, state.mapHeight);
    const radius = finiteNumber(record.radius);
    if (radius <= 0) {
      continue;
    }
    const circle = svgElement("circle", {
      class: className,
      cx: center.x,
      cy: center.y,
      r: radius,
    });
    if (typeof record[tokenAttribute] === "string") {
      circle.dataset[tokenAttribute === "kind" ? "kind" : "token"] =
        record[tokenAttribute];
    }
    parent.append(circle);
  }
}

function renderBattlefield() {
  const scene = frameScene(state.frame);
  const map = isRecord(scene?.map) ? scene.map : null;
  const width = finiteNumber(map?.width);
  const height = finiteNumber(map?.height);
  elements.battlefield.replaceChildren();

  if (!scene || !map || width <= 0 || height <= 0) {
    elements.battlefield.removeAttribute("viewBox");
    elements.battlefieldEmpty.hidden = false;
    elements.battlefieldEmpty.textContent = state.offline
      ? "The local debugger service is unavailable. Commands are not being retried."
      : "No authorized battlefield scene was returned.";
    state.mapHeight = 1;
    return;
  }

  state.mapHeight = height;
  elements.battlefield.setAttribute("viewBox", `0 0 ${width} ${height}`);
  elements.battlefield.setAttribute(
    "aria-label",
    `${scene.audience_badge ?? "Debugger"} battlefield, ${width} by ${height}.`,
  );
  elements.battlefieldEmpty.hidden = true;

  elements.battlefield.append(
    svgElement("rect", {
      class: "map-boundary",
      x: 0,
      y: 0,
      width,
      height,
      rx: 0.18,
    }),
  );

  const auraLayer = svgElement("g", { "aria-hidden": "true" });
  appendCircleLayer(auraLayer, asArray(scene.aura_fields), "aura-field", "token_id");
  elements.battlefield.append(auraLayer);

  const rangeLayer = svgElement("g", { "aria-hidden": "true" });
  appendCircleLayer(rangeLayer, asArray(scene.ranges), "range-ring", "kind");
  elements.battlefield.append(rangeLayer);

  const route = isRecord(scene.pending_route) ? scene.pending_route : null;
  if (route) {
    const source = screenPoint(route.source_anchor, height);
    const target = screenPoint(route.target_anchor, height);
    const line = svgElement("line", {
      class: "pending-route",
      x1: source.x,
      y1: source.y,
      x2: target.x,
      y2: target.y,
    });
    line.dataset.lane = String(route.lane ?? 0);
    line.dataset.legal = String(Boolean(route.legal));
    elements.battlefield.append(line);
  }

  const obstacleLayer = svgElement("g", { "aria-label": "Map obstacles" });
  for (const obstacle of asArray(map.obstacles)) {
    if (!isRecord(obstacle)) {
      continue;
    }
    const center = screenPoint(obstacle.center, height);
    if (obstacle.kind === "pillar") {
      obstacleLayer.append(
        svgElement("circle", {
          class: "obstacle",
          cx: center.x,
          cy: center.y,
          r: finiteNumber(obstacle.radius),
          "aria-label": `Pillar ${obstacle.obstacle_id ?? ""}`,
        }),
      );
    } else if (obstacle.kind === "wall") {
      const wallWidth = finiteNumber(obstacle.width);
      const wallHeight = finiteNumber(obstacle.height);
      const thetaDegrees = (-finiteNumber(obstacle.theta) * 180) / Math.PI;
      obstacleLayer.append(
        svgElement("rect", {
          class: "obstacle",
          x: center.x - wallWidth / 2,
          y: center.y - wallHeight / 2,
          width: wallWidth,
          height: wallHeight,
          transform: `rotate(${thetaDegrees} ${center.x} ${center.y})`,
          "aria-label": `Wall ${obstacle.obstacle_id ?? ""}`,
        }),
      );
    }
  }
  elements.battlefield.append(obstacleLayer);

  const selection = isRecord(scene.selection) ? scene.selection : {};
  const controlled = selection.controlled_global_slot;
  const selected = selection.selected_global_slot;
  const agentLayer = svgElement("g", { "aria-label": "Authorized agents" });
  for (const agent of asArray(scene.agents)) {
    if (!isRecord(agent) || !Number.isInteger(agent.global_slot)) {
      continue;
    }
    const center = screenPoint(agent.position, height);
    const radius = finiteNumber(agent.radius, 0.5);
    const group = svgElement("g", {
      class: "agent",
      tabindex: "-1",
      role: "img",
      "aria-label": `id_${agent.global_slot}, class ${agent.class_id}, health ${formatNumber(agent.current_health)} of ${formatNumber(agent.max_health)}`,
    });
    group.dataset.slot = String(agent.global_slot);
    group.dataset.team = String(agent.team_id);
    group.dataset.alive = String(Boolean(agent.alive));

    if (agent.global_slot === controlled) {
      group.append(
        svgElement("circle", {
          class: "controlled-ring",
          cx: center.x,
          cy: center.y,
          r: radius + 0.25,
        }),
      );
    }
    if (agent.global_slot === selected) {
      group.append(
        svgElement("circle", {
          class: "selected-ring",
          cx: center.x,
          cy: center.y,
          r: radius + 0.42,
        }),
      );
    }

    group.append(
      svgElement("circle", {
        class: "agent-body",
        cx: center.x,
        cy: center.y,
        r: radius,
      }),
    );
    const healthRadius = radius * 0.82;
    const healthRatio = Math.max(
      0,
      Math.min(
        1,
        finiteNumber(agent.current_health) /
          Math.max(finiteNumber(agent.max_health, 1), Number.EPSILON),
      ),
    );
    group.append(
      svgElement("circle", {
        class: "agent-health-track",
        cx: center.x,
        cy: center.y,
        r: healthRadius,
      }),
    );
    group.append(
      svgElement("circle", {
        class: "agent-health",
        cx: center.x,
        cy: center.y,
        r: healthRadius,
        pathLength: 100,
        "stroke-dasharray": `${healthRatio * 100} ${100 - healthRatio * 100}`,
        transform: `rotate(-90 ${center.x} ${center.y})`,
      }),
    );
    const label = svgElement("text", {
      class: "agent-label",
      x: center.x,
      y: center.y,
    });
    label.textContent = `C${agent.class_id}`;
    group.append(label);
    agentLayer.append(group);
  }
  elements.battlefield.append(agentLayer);
}

function renderRoster() {
  const scene = frameScene(state.frame);
  const agents = asArray(scene?.agents)
    .filter((agent) => isRecord(agent) && Number.isInteger(agent.global_slot))
    .sort((left, right) => left.global_slot - right.global_slot);
  const selection = isRecord(scene?.selection) ? scene.selection : {};
  elements.roster.replaceChildren();
  elements.rosterCount.textContent = `${agents.length} visible`;

  if (agents.length === 0) {
    elements.roster.append(
      htmlElement("p", "empty-copy", "No authorized agents in this view."),
    );
    return;
  }

  for (const agent of agents) {
    const row = htmlElement("article", "roster-row");
    row.setAttribute("aria-label", `Agent id_${agent.global_slot}`);
    row.dataset.team = String(agent.team_id);
    row.dataset.controlled = String(
      agent.global_slot === selection.controlled_global_slot,
    );
    row.dataset.selected = String(agent.global_slot === selection.selected_global_slot);

    const summary = htmlElement("div");
    const identity = htmlElement("div", "roster-identity");
    identity.append(
      htmlElement("span", "roster-id", `id_${agent.global_slot}`),
      htmlElement("span", "roster-class", `Class ${agent.class_id}`),
    );
    const health = htmlElement(
      "div",
      "roster-health",
      `HP ${formatNumber(agent.current_health)} / ${formatNumber(agent.max_health)} · cooldown ${agent.ultimate_cooldown ?? "—"}`,
    );
    const statuses = asArray(agent.statuses);
    const statusLine = htmlElement(
      "div",
      "status-list",
      statuses.length
        ? statuses
            .map(
              (status) =>
                `${status.short_label ?? status.label ?? status.token_id} ${status.duration ?? ""}`,
            )
            .join(" · ")
        : "No persistent statuses",
    );
    summary.append(identity, health, statusLine);

    const actions = htmlElement("div", "roster-actions");
    const targetButton = htmlElement("button", null, "Target");
    targetButton.type = "button";
    targetButton.setAttribute("aria-label", `Target id_${agent.global_slot}`);
    targetButton.disabled =
      state.busy || state.shuttingDown || state.resyncRequired || state.offline;
    targetButton.addEventListener("click", () => {
      dispatchCommand({
        command_type: "roster_selection",
        role: "target",
        global_slot: agent.global_slot,
      });
    });
    const controlButton = htmlElement("button", null, "Control");
    controlButton.type = "button";
    controlButton.setAttribute("aria-label", `Control id_${agent.global_slot}`);
    controlButton.disabled =
      state.busy || state.shuttingDown || state.resyncRequired || state.offline;
    controlButton.addEventListener("click", () => {
      dispatchCommand({
        command_type: "roster_selection",
        role: "control",
        global_slot: agent.global_slot,
      });
    });
    actions.append(targetButton, controlButton);
    row.append(summary, actions);
    elements.roster.append(row);
  }
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

function renderInspector() {
  const frame = state.frame;
  const scene = frameScene(frame);
  const selection = isRecord(scene?.selection) ? scene.selection : null;
  const legality = isRecord(scene?.selected_legality) ? scene.selected_legality : null;
  elements.selectionCard.replaceChildren();
  if (!selection) {
    elements.selectionCard.append(
      htmlElement("p", "empty-copy", "No selection facts received yet."),
    );
  } else {
    addFact(
      elements.selectionCard,
      "Controlled",
      `id_${selection.controlled_global_slot}`,
    );
    addFact(
      elements.selectionCard,
      "Target",
      Number.isInteger(selection.selected_global_slot)
        ? `id_${selection.selected_global_slot}`
        : "target-none",
    );
    if (legality) {
      addFact(
        elements.selectionCard,
        "Basic lane",
        legality.lane_0_available ? "Available" : "Unavailable",
      );
      addFact(
        elements.selectionCard,
        "Ultimate lane",
        legality.lane_1_available ? "Available" : "Unavailable",
      );
    }
  }

  const hud = isRecord(frame?.hud) ? frame.hud : {};
  const pending =
    hud.pending_action ?? frame?.pending_action ?? scene?.pending_route ?? null;
  const accepted =
    hud.latest_accepted_action ??
    hud.accepted_action ??
    hud.latest_transition ??
    frame?.latest_accepted_action ??
    null;
  elements.pendingCard.textContent = formatRecord(pending, "No pending action.");
  elements.acceptedCard.textContent = formatRecord(accepted, "No transition yet.");
  elements.diagnosticsCard.textContent = formatRecord(frame, "No frame received.");
}

/**
 * @param {unknown} event
 */
function eventSummary(event) {
  if (!isRecord(event)) {
    return "Unknown event";
  }
  const name =
    event.event_type ??
    event.token_id ??
    event.status_token_id ??
    event.outcome ??
    event.component ??
    "semantic event";
  const source = Number.isInteger(event.source_global_slot)
    ? `id_${event.source_global_slot}`
    : null;
  const target = Number.isInteger(event.target_global_slot)
    ? `id_${event.target_global_slot}`
    : Number.isInteger(event.recipient_global_slot)
      ? `id_${event.recipient_global_slot}`
      : null;
  const direction =
    source && target ? `${source} → ${target}` : (source ?? target ?? "local");
  const delta =
    typeof event.net_delta === "number"
      ? ` · NET ${event.net_delta >= 0 ? "+" : ""}${formatNumber(event.net_delta, 2)}`
      : "";
  return `${humanize(name)} · ${direction}${delta}`;
}

function renderEvents() {
  const batch = frameEvents(state.frame);
  const events = asArray(batch?.events);
  elements.eventFeed.replaceChildren();
  elements.eventCount.textContent = String(events.length);
  if (events.length === 0) {
    elements.eventFeed.append(htmlElement("li", "empty-copy", "No transition events."));
    return;
  }
  for (const event of events) {
    const item = htmlElement("li", "event-item", eventSummary(event));
    if (isRecord(event) && typeof event.event_id === "string") {
      item.title = event.event_id;
    }
    elements.eventFeed.append(item);
  }
}

function render() {
  renderConnection();
  renderSessionToolbar();
  renderBattlefield();
  renderRoster();
  renderInspector();
  renderEvents();
}

/**
 * @param {Record<string, unknown>} command
 */
function commandRequest(command) {
  return {
    schema_version: 1,
    client_id: state.clientId,
    command_id: window.crypto.randomUUID(),
    base_revision: currentRevision(),
    command,
  };
}

/**
 * @param {Record<string, unknown>} command
 */
async function dispatchCommand(command) {
  if (state.busy || state.shuttingDown) {
    setNotice("A command is already in flight; no second command was sent.", "warning");
    renderConnection();
    return;
  }
  if (state.resyncRequired || state.offline) {
    setNotice(
      "Reconnect to install the latest authoritative frame before sending another command.",
      "warning",
    );
    renderConnection();
    return;
  }
  if (!state.frame) {
    setNotice(
      "No authoritative frame is available. Reconnect before sending commands.",
      "error",
    );
    renderConnection();
    return;
  }

  state.busy = true;
  state.offline = false;
  setNotice("Waiting for the authoritative Python response…", "info");
  render();

  try {
    const payload = await postCommand(state.token, commandRequest(command));
    const frame = extractFrame(payload);
    if (!frame) {
      throw new DebuggerApiError("Command response did not contain a debugger frame.");
    }
    state.frame = frame;
    state.offline = false;
    const notice = extractNotice(payload);
    setNotice(
      notice ??
        (payload?.result === "duplicate"
          ? "Duplicate command recognized; it was not applied again."
          : "Authoritative frame updated."),
      payload?.result === "duplicate" ? "warning" : "success",
    );
    if (command.command_type === "exit") {
      state.shuttingDown = true;
      setNotice("Exit accepted. The local debugger server is shutting down.", "info");
    }
  } catch (error) {
    if (error instanceof DebuggerApiError && error.status === 409) {
      const latest = extractFrame(error.payload);
      if (latest) {
        state.frame = latest;
      }
      state.offline = false;
      state.resyncRequired = false;
      const errorCode = isRecord(error.payload) ? error.payload.error_code : null;
      setNotice(
        errorCode === "command_id_conflict"
          ? "The service rejected a command-ID conflict. The latest frame was installed and nothing was retried."
          : "This tab was stale. The latest frame was installed; the command was not retried.",
        "warning",
      );
    } else {
      const status = error instanceof DebuggerApiError ? error.status : 0;
      state.offline = status === 0 || status === 401 || status === 403;
      state.resyncRequired = true;
      setNotice(
        status === 401 || status === 403
          ? "Debugger capability is invalid. Reopen the exact URL printed by the Python launcher."
          : error instanceof Error
            ? `${error.message} Reconnect before sending another command.`
            : "Debugger command failed. Reconnect before sending another command.",
        "error",
      );
    }
  } finally {
    state.busy = false;
    render();
  }
}

async function loadCurrentFrame() {
  if (state.busy || state.shuttingDown) {
    return;
  }
  state.busy = true;
  setNotice("Fetching the current authoritative frame…", "info");
  renderConnection();
  try {
    const payload = await getCurrentFrame(state.token);
    const frame = extractFrame(payload);
    if (!frame) {
      throw new DebuggerApiError("Frame response did not contain a debugger frame.");
    }
    state.frame = frame;
    state.offline = false;
    state.resyncRequired = false;
    setNotice(extractNotice(payload) ?? "Connected to the local debugger.", "success");
  } catch (error) {
    const status = error instanceof DebuggerApiError ? error.status : 0;
    state.offline = status === 0 || status === 401 || status === 403;
    state.resyncRequired = true;
    setNotice(
      status === 401 || status === 403
        ? "Debugger capability is invalid. Reopen the exact URL printed by the Python launcher."
        : error instanceof Error
          ? error.message
          : "Could not load debugger frame.",
      "error",
    );
  } finally {
    state.busy = false;
    render();
  }
}

bindBattlefieldControls({
  battlefield: elements.battlefield,
  toWorldPoint: ({ x, y }) => ({
    world_x: x,
    world_y: state.mapHeight - y,
  }),
  onCommand: dispatchCommand,
  onHelp: () => elements.helpDialog.showModal(),
  onReleaseFocus: () => {
    const firstCommand = /** @type {HTMLButtonElement | null} */ (
      elements.commandDeck?.querySelector("button:not([disabled])") ?? null
    );
    const focusTarget = firstCommand ?? elements.helpButton;
    focusTarget.focus({ preventScroll: true });
  },
});

elements.scenarioSelect.addEventListener("change", () => {
  if (!elements.scenarioSelect.value) {
    return;
  }
  dispatchCommand({
    command_type: "scenario_switch",
    scenario_name: elements.scenarioSelect.value,
  });
});

elements.viewSelect.addEventListener("change", () => {
  dispatchCommand({
    command_type: "set_view",
    view_mode: elements.viewSelect.value,
  });
});

elements.presetSelect.addEventListener("change", () => {
  dispatchCommand({
    command_type: "set_preset",
    preset: elements.presetSelect.value,
  });
});

elements.resetButton.addEventListener("click", () => {
  dispatchCommand({ command_type: "reset" });
});

elements.exitButton.addEventListener("click", () => {
  dispatchCommand({ command_type: "exit" });
});

elements.reconnectButton.addEventListener("click", loadCurrentFrame);

elements.helpButton.addEventListener("click", () => {
  elements.helpDialog.showModal();
});

if (elements.commandDeck) {
  const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
    elements.commandDeck.querySelectorAll("button[data-key]")
  );
  for (const button of buttons) {
    button.addEventListener("click", () => {
      dispatchCommand(
        keyboardCommand(button.dataset.key ?? "", {
          shiftKey: button.dataset.shift === "true",
        }),
      );
    });
  }
}

render();
loadCurrentFrame();
