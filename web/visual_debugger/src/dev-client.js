import {
  acquireCapabilityToken,
  DebuggerApiError,
  postAuthoringCommand,
} from "./api.js";
import {
  focusAuthoringProblemField,
  readAuthoringFieldEdit,
  renderAuthoringInspector,
} from "./authoring-inspector.js";
import {
  addAuthoringObstacle,
  authoringContentSnapshot,
  authoringKind,
  authoringObjects,
  cloneAuthoringValue,
  deleteAuthoringObstacle,
  duplicateAuthoringObstacle,
  mapContent,
  moveAuthoringObjectWithSnap,
  normalizeAuthoringProblems,
  reorderAuthoringObstacle,
  restoreAuthoringContent,
  selectedAuthoringObject,
  setAgentAlive,
  setAuthoringField,
  setScenarioTeamSize,
} from "./authoring-model.js";
import {
  authoringClientPointToWorld,
  authoringMapDimensions,
  normalizeAuthoringCamera,
  panAuthoringCamera,
  renderAuthoringSvg,
  zoomAuthoringCamera,
} from "./authoring-renderer.js";

const bootstrap = Reflect.get(globalThis, "__MARL_DEBUGGER_BOOTSTRAP__");
const AUTHORING_AGENT_DRAG_TYPE = "application/x-marl-authoring-agent";
if (
  bootstrap?.product_kind === "combat_debugger" &&
  bootstrap?.authoring_available === true
) {
  installDevClient();
}

/**
 * General Open resumes mutable work. Frozen candidates remain available as
 * scenario-creation sources and through the Combat Debugger loader.
 *
 * @param {ReadonlyArray<Record<string, any>>} assets
 * @param {"map" | "scenario"} kind
 */
export function openableDraftAssets(assets, kind) {
  return assets.filter(
    (asset) => asset.asset_kind === kind && asset.source_kind === "saved_draft",
  );
}

/**
 * Validation echoes are evidence about the submitted snapshot, not a newer
 * mutable document. Keeping the current reference prevents a delayed response
 * from erasing local edits.
 *
 * @param {any} currentDraft
 * @param {Readonly<Record<string, any>>} response
 */
export function draftAfterAuthoringResponse(currentDraft, response) {
  if (response.command_type === "validate" || !response.draft) {
    return currentDraft;
  }
  return cloneAuthoringValue(response.draft);
}

/** @param {Readonly<Record<string, any>>} asset */
export function debugScenarioOptionLabel(asset) {
  const identity =
    asset.source_kind === "candidate"
      ? `candidate ${asset.candidate_id}`
      : `saved ${asset.asset_id} revision ${asset.revision}`;
  const status =
    asset.source_kind === "candidate" ? "frozen · execution-valid" : "execution-valid";
  return `${asset.name} · ${identity} · ${asset.map_width} × ${asset.map_height} · ${status}`;
}

/**
 * Keep the two Combat selectors as projections of installed host authority.
 * A browser change is an intent only: the controls snap back immediately and
 * move only after a successor frame confirms the requested configuration.
 *
 * @param {{
 *   teamBController: {value: string, disabled: boolean},
 *   informationMode: {value: string, disabled: boolean},
 *   root: {dataset: Record<string, string | undefined>},
 *   emit: (configuration: Readonly<Record<string, string>>) => void,
 * }} bindings
 */
export function createCombatConfigurationController(bindings) {
  /** @type {Readonly<Record<string, string>> | null} */
  let authoritative = null;

  /** @param {unknown} value */
  function normalize(value) {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const candidate = /** @type {Record<string, unknown>} */ (value);
    if (
      (candidate.team_b_controller !== "manual" &&
        candidate.team_b_controller !== "scripted_tdm") ||
      (candidate.execution_information_mode !== "shared_obs" &&
        candidate.execution_information_mode !== "no_shared_obs")
    ) {
      return null;
    }
    return Object.freeze({
      team_b_controller: candidate.team_b_controller,
      execution_information_mode: candidate.execution_information_mode,
    });
  }

  function render() {
    const configuration = authoritative;
    bindings.teamBController.disabled = configuration === null;
    bindings.informationMode.disabled = configuration === null;
    if (configuration === null) {
      return;
    }
    bindings.teamBController.value = configuration.team_b_controller;
    bindings.informationMode.value = configuration.execution_information_mode;
    bindings.root.dataset.teamBController = configuration.team_b_controller;
    bindings.root.dataset.executionInformationMode =
      configuration.execution_information_mode;
  }

  return Object.freeze({
    /** @param {unknown} value */
    install(value) {
      const normalized = normalize(value);
      if (normalized === null) {
        return false;
      }
      authoritative = normalized;
      render();
      return true;
    },
    request() {
      const requested = normalize({
        team_b_controller: bindings.teamBController.value,
        execution_information_mode: bindings.informationMode.value,
      });
      render();
      if (
        requested === null ||
        authoritative === null ||
        (requested.team_b_controller === authoritative.team_b_controller &&
          requested.execution_information_mode ===
            authoritative.execution_information_mode)
      ) {
        return false;
      }
      bindings.emit(requested);
      return true;
    },
    render,
  });
}

function installDevClient() {
  /** @param {string} id */
  function required(id) {
    const candidate = document.getElementById(id);
    if (!candidate) {
      throw new Error(`DevClient shell is missing #${id}.`);
    }
    return candidate;
  }

  /** @type {any} */
  const elements = {
    nav: required("devclient-nav"),
    combatConfig: required("devclient-combat-config"),
    scenarioSelect: required("devclient-scenario-select"),
    scenarioLoad: required("devclient-scenario-load"),
    teamBController: required("devclient-team-b-controller"),
    informationMode: required("devclient-information-mode"),
    shell: required("authoring-shell"),
    eyebrow: required("authoring-eyebrow"),
    title: required("authoring-title"),
    newScenarioChoice: required("authoring-new-scenario-choice"),
    newScenarioMode: required("authoring-new-scenario-mode"),
    newButton: required("authoring-new"),
    openButton: required("authoring-open"),
    saveButton: required("authoring-save"),
    saveAsButton: required("authoring-save-as"),
    validateButton: required("authoring-validate"),
    freezeButton: required("authoring-freeze"),
    openDebugButton: required("authoring-open-debug"),
    palette: required("authoring-palette"),
    objectList: required("authoring-object-list"),
    objectCount: required("authoring-object-count"),
    canvas: required("authoring-canvas"),
    inspector: required("authoring-inspector-form"),
    undoButton: required("authoring-undo"),
    redoButton: required("authoring-redo"),
    duplicateButton: required("authoring-duplicate"),
    deleteButton: required("authoring-delete"),
    orderUpButton: required("authoring-order-up"),
    orderDownButton: required("authoring-order-down"),
    problemList: required("authoring-problem-list"),
    problemCount: required("authoring-problem-count"),
  };
  /** @type {any} */
  const state = {
    token: acquireCapabilityToken(),
    area: "combat",
    draft: null,
    selectedId: null,
    problems: [],
    assets: [],
    validation: null,
    catalog: null,
    past: [],
    future: [],
    camera: null,
    pointer: null,
    spacePressed: false,
    busy: false,
  };
  const combatConfiguration = createCombatConfigurationController({
    teamBController: elements.teamBController,
    informationMode: elements.informationMode,
    root: document.documentElement,
    emit: (configuration) => {
      document.dispatchEvent(
        new CustomEvent("marl-devclient-combat-configuration", {
          detail: configuration,
        }),
      );
    },
  });

  elements.nav.hidden = false;
  elements.combatConfig.hidden = false;
  document.documentElement.dataset.devclientArea = "combat";
  combatConfiguration.render();

  /** @param {Event} event @param {string} selector */
  function closest(event, selector) {
    return event.target instanceof Element ? event.target.closest(selector) : null;
  }

  /** @param {Event | null} [event] */
  function authoringInteractionBlocked(event = null) {
    if (!state.busy) {
      return false;
    }
    event?.preventDefault();
    return true;
  }

  /** @param {string} message */
  function showLocalError(message) {
    state.problems = [
      {
        severity: "error",
        stable_code: "browser-authoring-operation",
        message,
        object_id: state.selectedId,
        field_path: "browser",
      },
    ];
    renderProblems();
  }

  /** @param {Record<string, any>} response */
  function installResponse(response) {
    const nextDraft = draftAfterAuthoringResponse(state.draft, response);
    if (nextDraft !== state.draft) {
      const newDocument = ["new_map", "new_scenario", "open"].includes(
        response.command_type,
      );
      state.draft = nextDraft;
      if (newDocument) {
        state.selectedId = null;
        state.past = [];
        state.future = [];
        state.camera = null;
      }
    }
    if (response.command_type === "list" && Array.isArray(response.assets)) {
      state.assets = response.assets;
    }
    if (response.validation) {
      state.validation = response.validation;
    }
    state.catalog = response.catalog ?? state.catalog;
    state.problems = normalizeAuthoringProblems(
      response.validation?.problems ?? response.problems ?? [],
    );
    renderAll();
  }

  /** @param {Record<string, any>} command */
  async function send(command) {
    if (state.busy) {
      return null;
    }
    state.busy = true;
    renderAvailability();
    try {
      const response = await postAuthoringCommand(state.token, command);
      installResponse(response);
      return response;
    } catch (error) {
      showLocalError(
        error instanceof DebuggerApiError || error instanceof Error
          ? error.message
          : "The authoring command failed.",
      );
      return null;
    } finally {
      state.busy = false;
      renderAvailability();
    }
  }

  async function refreshAssets() {
    await send({ command_type: "list", asset_kind: "all" });
  }

  function notifyDebugSessionReplaced() {
    document.dispatchEvent(new CustomEvent("marl-devclient-debug-session-replaced"));
  }

  /** @param {Record<string, any>} source @param {boolean} returnToCombat */
  async function openInDebug(source, returnToCombat = false) {
    const response = await send({ command_type: "open_in_debug", source });
    if (!response?.ok) {
      return;
    }
    notifyDebugSessionReplaced();
    if (returnToCombat) {
      await selectArea("combat");
    }
  }

  /** @param {"maximum_obstacle_slots" | "fixed_grid_world_units" | "fixed_snap_world_units"} key */
  function catalogNumber(key) {
    const value = state.catalog?.[key];
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      throw new TypeError(`Authoring catalog is missing ${key}.`);
    }
    return value;
  }

  /** @param {Record<string, any>} asset */
  function persistedSource(asset) {
    return asset.source_kind === "candidate"
      ? {
          source_kind: "candidate",
          asset_kind: asset.asset_kind,
          candidate_id: asset.candidate_id,
        }
      : {
          source_kind: "saved_draft",
          asset_kind: asset.asset_kind,
          asset_id: asset.asset_id,
          revision: asset.revision,
        };
  }

  /** @param {Record<string, any>} asset */
  function debugScenarioSource(asset) {
    return asset.source_kind === "candidate"
      ? {
          source_kind: "candidate",
          candidate_id: asset.candidate_id,
        }
      : {
          source_kind: "saved_draft",
          asset_id: asset.asset_id,
          revision: asset.revision,
        };
  }

  /** @param {string} promptText @param {Record<string, any>[]} candidates */
  function chooseAsset(promptText, candidates) {
    if (candidates.length === 0) {
      showLocalError("No compatible saved assets are available.");
      return null;
    }
    /** @param {Record<string, any>} asset */
    const identity = (asset) =>
      asset.source_kind === "candidate" ? asset.candidate_id : asset.asset_id;
    const requested = window.prompt(promptText, identity(candidates[0]));
    return candidates.find((asset) => identity(asset) === requested) ?? null;
  }

  /** @param {"map" | "scenario"} kind */
  async function createDraft(kind) {
    if (state.busy) {
      return;
    }
    const defaultId = kind === "map" ? "untitled-map" : "untitled-scenario";
    const assetId = window.prompt(
      `${kind === "map" ? "Map" : "Scenario"} asset ID`,
      defaultId,
    );
    if (assetId === null) {
      return;
    }
    if (assetId.length > 64 || !/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/u.test(assetId)) {
      showLocalError(
        "Asset IDs use lowercase letters, digits, and internal hyphens only.",
      );
      return;
    }
    if (kind === "map") {
      await send({ command_type: "new_map", asset_id: assetId });
      return;
    }
    const creationMode = elements.newScenarioMode.value;
    if (
      !["blank", "copy_saved_map", "duplicate_saved_scenario"].includes(creationMode)
    ) {
      showLocalError("Choose how to create the new scenario.");
      return;
    }
    let source = null;
    if (creationMode !== "blank") {
      const sourceKind = creationMode === "copy_saved_map" ? "map" : "scenario";
      const asset = chooseAsset(
        `Saved ${sourceKind} asset ID or candidate digest`,
        state.assets.filter(
          (/** @type {any} */ candidate) => candidate.asset_kind === sourceKind,
        ),
      );
      if (asset === null) {
        return;
      }
      source = persistedSource(asset);
    }
    /** @type {Record<string, any>} */
    const command = {
      command_type: "new_scenario",
      asset_id: assetId,
      creation_mode: creationMode,
    };
    if (source !== null) {
      command.source = source;
    }
    await send(command);
  }

  /** @param {"combat" | "maps" | "scenarios"} area */
  async function selectArea(area) {
    if (state.busy) {
      return;
    }
    state.area = area;
    elements.newScenarioChoice.hidden = area !== "scenarios";
    document.documentElement.dataset.devclientArea =
      area === "combat" ? "combat" : "authoring";
    for (const button of elements.nav.querySelectorAll("[data-devclient-area]")) {
      button.setAttribute(
        "aria-current",
        button.getAttribute("data-devclient-area") === area ? "page" : "false",
      );
    }
    elements.shell.hidden = area === "combat";
    if (area === "combat") {
      await refreshAssets();
      return;
    }
    const kind = area === "maps" ? "map" : "scenario";
    if (state.draft === null || authoringKind(state.draft) !== kind) {
      await createDraft(kind);
    } else {
      renderAll();
    }
  }

  /** @param {Record<string, any>} next @param {Record<string, any>} [before] */
  function commit(next, before = state.draft) {
    if (state.busy || before === null) {
      return;
    }
    state.past.push(authoringContentSnapshot(before));
    state.past = state.past.slice(-50);
    state.future = [];
    state.draft = next;
    state.validation = null;
    renderAll();
    void validateDraft();
  }

  async function validateDraft() {
    if (state.draft !== null) {
      await send({ command_type: "validate", draft: state.draft });
    }
  }

  function renderScenarioOptions() {
    const selected = elements.scenarioSelect.value;
    elements.scenarioSelect.replaceChildren(new Option("Built-in arena", ""));
    for (const asset of state.assets.filter(
      (/** @type {any} */ candidate) =>
        candidate.asset_kind === "scenario" && candidate.execution_valid,
    )) {
      elements.scenarioSelect.append(
        new Option(
          debugScenarioOptionLabel(asset),
          JSON.stringify(debugScenarioSource(asset)),
        ),
      );
    }
    if (
      [...elements.scenarioSelect.options].some((option) => option.value === selected)
    ) {
      elements.scenarioSelect.value = selected;
    }
    elements.scenarioLoad.disabled = !elements.scenarioSelect.value;
  }

  function renderObjectList() {
    elements.objectList.replaceChildren();
    if (state.draft === null) {
      elements.objectCount.textContent = "0";
      return;
    }
    const objects = authoringObjects(state.draft);
    const rows = [
      { object_id: "", label: `${authoringKind(state.draft)} document` },
      ...objects,
    ];
    for (const object of rows) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.objectId = object.object_id;
      button.setAttribute(
        "aria-current",
        String((object.object_id || null) === state.selectedId),
      );
      button.textContent = object.label;
      if (object.kind === "agent") {
        button.draggable = true;
        button.dataset.authoringRosterAgent = "true";
      }
      button.setAttribute("aria-describedby", "authoring-object-list-help");
      const row = document.createElement("li");
      row.append(button);
      elements.objectList.append(row);
    }
    elements.objectCount.textContent = String(objects.length);
  }

  function renderProblems() {
    elements.problemList.replaceChildren();
    elements.problemCount.textContent = String(state.problems.length);
    if (state.problems.length === 0) {
      const row = document.createElement("li");
      row.className = "empty-copy";
      row.textContent = state.validation?.freeze_qualified
        ? "Freeze-qualified."
        : state.validation?.execution_valid
          ? "Execution-valid. Complete the study contract to freeze."
          : "Validate to inspect host-authoritative errors and warnings.";
      elements.problemList.append(row);
      return;
    }
    for (const problem of state.problems) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.severity = problem.severity;
      button.dataset.objectId = problem.object_id ?? "";
      button.dataset.fieldPath = problem.field_path;
      button.setAttribute("aria-describedby", "authoring-problem-list-help");
      button.textContent = `${problem.stable_code}: ${problem.message}`;
      const row = document.createElement("li");
      row.append(button);
      elements.problemList.append(row);
    }
  }

  function renderCanvas() {
    if (state.draft === null) {
      elements.canvas.replaceChildren();
      return;
    }
    const map = mapContent(state.draft);
    state.camera = renderAuthoringSvg(
      elements.canvas,
      state.draft,
      state.selectedId,
      normalizeAuthoringCamera(state.camera, map.width, map.height),
      catalogNumber("fixed_grid_world_units"),
      state.catalog,
    );
  }

  /** @param {number} clientX @param {number} clientY */
  function authoringWorldPoint(clientX, clientY) {
    if (state.draft === null) {
      return null;
    }
    const map = mapContent(state.draft);
    const dimensions = authoringMapDimensions(map.width, map.height);
    if (dimensions === null) {
      return null;
    }
    return authoringClientPointToWorld(
      elements.canvas.getBoundingClientRect(),
      state.camera,
      dimensions.height,
      clientX,
      clientY,
    );
  }

  function renderAvailability() {
    const selected =
      state.draft && selectedAuthoringObject(state.draft, state.selectedId);
    const obstacle = selected?.kind === "wall" || selected?.kind === "pillar";
    elements.shell.inert = state.busy;
    elements.shell.setAttribute("aria-busy", String(state.busy));
    for (const button of elements.nav.querySelectorAll("button")) {
      button.disabled = state.busy;
    }
    elements.scenarioSelect.disabled = state.busy;
    elements.scenarioLoad.disabled = state.busy || !elements.scenarioSelect.value;
    elements.newButton.disabled = state.busy;
    elements.openButton.disabled = state.busy;
    elements.saveButton.disabled = state.busy || !state.draft;
    elements.saveAsButton.disabled = state.busy || !state.draft;
    elements.validateButton.disabled = state.busy || !state.draft;
    elements.undoButton.disabled = state.busy || state.past.length === 0;
    elements.redoButton.disabled = state.busy || state.future.length === 0;
    elements.duplicateButton.disabled = state.busy || !obstacle;
    elements.deleteButton.disabled = state.busy || !obstacle;
    elements.orderUpButton.disabled = state.busy || !obstacle;
    elements.orderDownButton.disabled = state.busy || !obstacle;
    elements.freezeButton.disabled = state.busy || !state.validation?.freeze_qualified;
    elements.openDebugButton.disabled =
      state.busy ||
      !state.draft ||
      authoringKind(state.draft) !== "scenario" ||
      !state.validation?.execution_valid;
  }

  function renderAll() {
    renderScenarioOptions();
    if (state.draft) {
      const kind = authoringKind(state.draft);
      elements.eyebrow.textContent = kind === "map" ? "Map Author" : "Scenario Author";
      elements.title.textContent = state.draft.content.name;
    }
    renderObjectList();
    renderCanvas();
    renderAuthoringInspector(
      elements.inspector,
      state.draft,
      state.selectedId,
      state.catalog,
      state.validation,
    );
    renderProblems();
    renderAvailability();
  }

  elements.nav.addEventListener("click", (/** @type {Event} */ event) => {
    if (authoringInteractionBlocked(event)) {
      return;
    }
    const button = closest(event, "[data-devclient-area]");
    if (button) {
      const area = button.getAttribute("data-devclient-area");
      if (area === "combat" || area === "maps" || area === "scenarios") {
        void selectArea(area);
      }
    }
  });
  elements.scenarioSelect.addEventListener("change", () => {
    if (state.busy) {
      return;
    }
    elements.scenarioLoad.disabled = !elements.scenarioSelect.value;
  });
  elements.scenarioLoad.addEventListener("click", () => {
    if (!state.busy && elements.scenarioSelect.value) {
      void openInDebug(JSON.parse(elements.scenarioSelect.value));
    }
  });
  for (const selector of [elements.teamBController, elements.informationMode]) {
    selector.addEventListener("change", () => {
      combatConfiguration.request();
    });
  }
  document.addEventListener(
    "marl-devclient-combat-configuration-installed",
    (event) => {
      if (event instanceof CustomEvent) {
        combatConfiguration.install(event.detail);
      }
    },
  );

  elements.newButton.addEventListener(
    "click",
    () => void createDraft(state.area === "maps" ? "map" : "scenario"),
  );
  elements.openButton.addEventListener("click", async () => {
    if (state.busy) {
      return;
    }
    await refreshAssets();
    const kind = state.area === "maps" ? "map" : "scenario";
    const asset = chooseAsset(
      `Saved ${kind} asset ID`,
      openableDraftAssets(state.assets, kind),
    );
    if (asset) {
      await send({ command_type: "open", source: persistedSource(asset) });
    }
  });
  elements.saveButton.addEventListener("click", () => {
    if (!state.busy && state.draft) {
      void send({
        command_type: "save",
        draft: state.draft,
        expected_revision: state.draft.revision,
      });
    }
  });
  elements.saveAsButton.addEventListener("click", () => {
    if (state.busy || !state.draft) {
      return;
    }
    const assetId = window.prompt("New asset ID", `${state.draft.asset_id}-copy`);
    if (assetId) {
      void send({ command_type: "save_as", draft: state.draft, asset_id: assetId });
    }
  });
  elements.validateButton.addEventListener("click", () => {
    if (!state.busy) {
      void validateDraft();
    }
  });
  elements.freezeButton.addEventListener("click", () => {
    if (!state.busy && state.draft) {
      void send({ command_type: "freeze", draft: state.draft });
    }
  });
  elements.openDebugButton.addEventListener("click", async () => {
    if (state.busy || !state.draft) {
      return;
    }
    await openInDebug({ source_kind: "current_buffer", draft: state.draft }, true);
  });

  elements.palette.addEventListener("click", (/** @type {Event} */ event) => {
    if (authoringInteractionBlocked(event)) {
      return;
    }
    const button = closest(event, "[data-authoring-add]");
    if (!button || !state.draft) {
      return;
    }
    try {
      const obstacleKind = button.getAttribute("data-authoring-add");
      if (obstacleKind !== "wall" && obstacleKind !== "pillar") {
        return;
      }
      const result = addAuthoringObstacle(
        state.draft,
        obstacleKind,
        catalogNumber("maximum_obstacle_slots"),
      );
      state.selectedId = result.object_id;
      commit(result.draft);
    } catch (error) {
      showLocalError(
        error instanceof Error ? error.message : "Obstacle could not be added.",
      );
    }
  });
  elements.objectList.addEventListener("click", (/** @type {Event} */ event) => {
    if (authoringInteractionBlocked(event)) {
      return;
    }
    const button = closest(event, "[data-object-id]");
    if (button) {
      state.selectedId = button.getAttribute("data-object-id") || null;
      renderAll();
    }
  });
  elements.objectList.addEventListener(
    "dragstart",
    (/** @type {DragEvent} */ event) => {
      if (authoringInteractionBlocked(event) || !event.dataTransfer || !state.draft) {
        return;
      }
      const row = closest(event, '[data-authoring-roster-agent="true"]');
      const objectId = row?.getAttribute("data-object-id");
      const object = objectId ? selectedAuthoringObject(state.draft, objectId) : null;
      if (!objectId || object?.kind !== "agent") {
        event.preventDefault();
        return;
      }
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData(AUTHORING_AGENT_DRAG_TYPE, objectId);
      event.dataTransfer.setData("text/plain", objectId);
    },
  );
  elements.problemList.addEventListener("click", (/** @type {Event} */ event) => {
    if (authoringInteractionBlocked(event)) {
      return;
    }
    const button = closest(event, "[data-object-id]");
    if (button) {
      state.selectedId = button.getAttribute("data-object-id") || null;
      const fieldPath = button.getAttribute("data-field-path") ?? "";
      renderAll();
      focusAuthoringProblemField(elements.inspector, fieldPath);
    }
  });
  elements.inspector.addEventListener("change", (/** @type {Event} */ event) => {
    if (authoringInteractionBlocked(event)) {
      return;
    }
    const edit = readAuthoringFieldEdit(event.target);
    if (!edit || !state.draft) {
      return;
    }
    const alive = edit.path.at(-1) === "alive" && state.selectedId;
    const teamSize =
      edit.path.length === 2 &&
      edit.path[0] === "content" &&
      ["team_a_size", "team_b_size"].includes(edit.path[1]);
    commit(
      alive
        ? setAgentAlive(state.draft, state.selectedId, Boolean(edit.value))
        : teamSize
          ? setScenarioTeamSize(
              state.draft,
              edit.path[1] === "team_a_size" ? "A" : "B",
              edit.value,
              state.catalog,
            )
          : setAuthoringField(state.draft, edit.path, edit.value),
    );
  });

  elements.undoButton.addEventListener("click", () => {
    const previousContent = state.busy ? null : state.past.pop();
    if (previousContent && state.draft) {
      state.future.push(authoringContentSnapshot(state.draft));
      state.draft = restoreAuthoringContent(state.draft, previousContent);
      renderAll();
      void validateDraft();
    }
  });
  elements.redoButton.addEventListener("click", () => {
    const nextContent = state.busy ? null : state.future.pop();
    if (nextContent && state.draft) {
      state.past.push(authoringContentSnapshot(state.draft));
      state.draft = restoreAuthoringContent(state.draft, nextContent);
      renderAll();
      void validateDraft();
    }
  });
  elements.duplicateButton.addEventListener("click", () => {
    if (!state.busy && state.draft && state.selectedId) {
      const result = duplicateAuthoringObstacle(
        state.draft,
        state.selectedId,
        catalogNumber("maximum_obstacle_slots"),
        catalogNumber("fixed_snap_world_units"),
      );
      state.selectedId = result.object_id;
      commit(result.draft);
    }
  });
  elements.deleteButton.addEventListener("click", () => {
    if (!state.busy && state.draft && state.selectedId) {
      const next = deleteAuthoringObstacle(state.draft, state.selectedId);
      state.selectedId = null;
      commit(next);
    }
  });
  elements.orderUpButton.addEventListener("click", () => {
    if (!state.busy && state.draft && state.selectedId) {
      commit(reorderAuthoringObstacle(state.draft, state.selectedId, -1));
    }
  });
  elements.orderDownButton.addEventListener("click", () => {
    if (!state.busy && state.draft && state.selectedId) {
      commit(reorderAuthoringObstacle(state.draft, state.selectedId, 1));
    }
  });

  elements.canvas.addEventListener("dragover", (/** @type {DragEvent} */ event) => {
    if (
      !state.busy &&
      state.draft !== null &&
      authoringKind(state.draft) === "scenario"
    ) {
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "move";
      }
    }
  });
  elements.canvas.addEventListener("drop", (/** @type {DragEvent} */ event) => {
    if (authoringInteractionBlocked(event) || !state.draft || !event.dataTransfer) {
      return;
    }
    const objectId =
      event.dataTransfer.getData(AUTHORING_AGENT_DRAG_TYPE) ||
      event.dataTransfer.getData("text/plain");
    const object = selectedAuthoringObject(state.draft, objectId);
    const world = authoringWorldPoint(event.clientX, event.clientY);
    if (object?.kind !== "agent" || world === null) {
      return;
    }
    event.preventDefault();
    state.selectedId = objectId;
    commit(
      moveAuthoringObjectWithSnap(
        state.draft,
        objectId,
        world.x,
        world.y,
        catalogNumber("fixed_snap_world_units"),
        event.altKey,
      ),
    );
  });

  elements.canvas.addEventListener(
    "pointerdown",
    (/** @type {PointerEvent} */ event) => {
      if (authoringInteractionBlocked(event) || !state.draft) {
        return;
      }
      const object = closest(event, "[data-object-id]");
      if (event.button === 1 || (event.button === 0 && state.spacePressed)) {
        state.pointer = {
          kind: "pan",
          pointerId: event.pointerId,
          x: event.clientX,
          y: event.clientY,
        };
      } else if (event.button === 0 && object) {
        state.selectedId = object.getAttribute("data-object-id");
        state.pointer = {
          kind: "drag",
          pointerId: event.pointerId,
          objectId: state.selectedId,
          before: cloneAuthoringValue(state.draft),
        };
        renderAll();
      } else {
        return;
      }
      elements.canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
    },
  );
  elements.canvas.addEventListener(
    "pointermove",
    (/** @type {PointerEvent} */ event) => {
      if (
        state.busy ||
        !state.pointer ||
        state.pointer.pointerId !== event.pointerId ||
        !state.draft
      ) {
        return;
      }
      if (state.pointer.kind === "pan") {
        const bounds = elements.canvas.getBoundingClientRect();
        state.camera = panAuthoringCamera(
          state.camera,
          -((event.clientX - state.pointer.x) * state.camera.width) / bounds.width,
          -((event.clientY - state.pointer.y) * state.camera.height) / bounds.height,
        );
        state.pointer.x = event.clientX;
        state.pointer.y = event.clientY;
        renderCanvas();
        return;
      }
      const world = authoringWorldPoint(event.clientX, event.clientY);
      if (world === null) {
        return;
      }
      state.draft = moveAuthoringObjectWithSnap(
        state.draft,
        state.pointer.objectId,
        world.x,
        world.y,
        catalogNumber("fixed_snap_world_units"),
        event.altKey,
      );
      renderCanvas();
      renderAuthoringInspector(
        elements.inspector,
        state.draft,
        state.selectedId,
        state.catalog,
        state.validation,
      );
    },
  );
  /** @param {PointerEvent} event */
  function finishPointer(event) {
    if (!state.pointer || state.pointer.pointerId !== event.pointerId) {
      return;
    }
    const pointer = state.pointer;
    state.pointer = null;
    if (elements.canvas.hasPointerCapture(event.pointerId)) {
      elements.canvas.releasePointerCapture(event.pointerId);
    }
    if (!state.busy && pointer.kind === "drag" && state.draft) {
      commit(state.draft, pointer.before);
    }
  }
  elements.canvas.addEventListener("pointerup", finishPointer);
  elements.canvas.addEventListener("pointercancel", finishPointer);
  elements.canvas.addEventListener(
    "wheel",
    (/** @type {WheelEvent} */ event) => {
      if (authoringInteractionBlocked(event) || !state.draft) {
        return;
      }
      const map = mapContent(state.draft);
      const dimensions = authoringMapDimensions(map.width, map.height);
      const world = authoringWorldPoint(event.clientX, event.clientY);
      if (dimensions === null || world === null) {
        return;
      }
      state.camera = zoomAuthoringCamera(
        state.camera,
        dimensions.width,
        dimensions.height,
        world,
        event.deltaY > 0 ? 1.15 : 1 / 1.15,
      );
      renderCanvas();
      event.preventDefault();
    },
    { passive: false },
  );
  elements.canvas.addEventListener("keydown", (/** @type {KeyboardEvent} */ event) => {
    if (authoringInteractionBlocked(event)) {
      return;
    }
    if (event.key === " ") {
      state.spacePressed = true;
      event.preventDefault();
      return;
    }
    const object =
      state.draft && selectedAuthoringObject(state.draft, state.selectedId);
    const snapStep = catalogNumber("fixed_snap_world_units");
    const step = event.altKey ? snapStep / 5 : snapStep;
    /** @type {Record<string, number[]>} */
    const deltas = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    };
    const delta = deltas[event.key];
    if (object && delta) {
      commit(
        moveAuthoringObjectWithSnap(
          state.draft,
          state.selectedId,
          object.x + delta[0],
          object.y + delta[1],
          snapStep,
          true,
        ),
      );
      event.preventDefault();
    }
  });
  window.addEventListener("keyup", (/** @type {KeyboardEvent} */ event) => {
    if (event.key === " ") {
      state.spacePressed = false;
    }
  });

  renderAll();
  void refreshAssets();
}
