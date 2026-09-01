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
export function persistedAuthoringSource(asset) {
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

/** @param {Readonly<Record<string, any>>} asset */
export function savedDraftOptionLabel(asset) {
  return `${asset.name} · ${asset.asset_id} · revision ${asset.revision} · ${asset.map_width} × ${asset.map_height}`;
}

/**
 * @param {ReadonlyArray<Record<string, any>>} assets
 * @param {string} creationMode
 */
export function newScenarioSourceAssets(assets, creationMode) {
  const sourceKind =
    creationMode === "copy_saved_map"
      ? "map"
      : creationMode === "duplicate_saved_scenario"
        ? "scenario"
        : null;
  return sourceKind === null
    ? []
    : assets.filter((asset) => asset.asset_kind === sourceKind);
}

/** @param {Readonly<Record<string, any>>} asset */
export function authoringSourceOptionLabel(asset) {
  const lifecycle =
    asset.source_kind === "candidate"
      ? `frozen candidate ${asset.candidate_id}`
      : `saved ${asset.asset_id} revision ${asset.revision}`;
  const status = asset.execution_valid ? "execution-valid" : "needs validation fixes";
  return `${asset.name} · ${lifecycle} · ${asset.map_width} × ${asset.map_height} · ${status}`;
}

/** @param {Readonly<Record<string, any>>} asset */
export function debugAssetOptionLabel(asset) {
  const identity =
    asset.source_kind === "candidate"
      ? `candidate ${asset.candidate_id}`
      : `saved ${asset.asset_id} revision ${asset.revision}`;
  const status =
    asset.source_kind === "candidate" ? "frozen · execution-valid" : "execution-valid";
  return asset.asset_kind === "map"
    ? `Map preview · ${asset.name} · ${identity} · ${asset.map_width} × ${asset.map_height} · ${status} · default 5v5 TDM`
    : `Scenario · ${asset.name} · ${identity} · ${asset.map_width} × ${asset.map_height} · ${status}`;
}

/**
 * @param {Record<string, any> | null} draft
 * @param {Record<string, any> | null} baseline
 * @param {{candidateId: string, content: Record<string, any>} | null} frozen
 */
export function authoringPersistenceMessage(draft, baseline, frozen) {
  if (draft === null) {
    return "No authoring draft is open.";
  }
  const kind = authoringKind(draft);
  const collection = kind === "map" ? "maps" : "scenarios";
  const currentContent = JSON.stringify(draft.content);
  const savedContent = baseline === null ? null : JSON.stringify(baseline);
  const savedPath = `artifacts/dev_client/drafts/${collection}/${draft.asset_id}/r${draft.revision}.json`;
  let message;
  if (draft.revision === 0) {
    message = `Unsaved ${kind} draft`;
  } else if (savedContent !== currentContent) {
    message = `Unsaved changes · last saved ${kind} ${draft.asset_id} revision ${draft.revision}`;
  } else {
    message = `Saved ${kind} ${draft.asset_id} · revision ${draft.revision} · ${savedPath}`;
  }
  if (frozen === null) {
    return message;
  }
  const candidatePath = `artifacts/dev_client/candidates/${kind}-${frozen.candidateId}.json`;
  return JSON.stringify(frozen.content) === currentContent
    ? `Frozen candidate ${frozen.candidateId} · ${candidatePath}`
    : `${message} · Frozen candidate ${frozen.candidateId} · ${candidatePath} · preserves an earlier snapshot; current edits are not frozen`;
}

/**
 * Bind Freeze feedback to the exact browser snapshot submitted to the host.
 * The candidate content may be canonically float32-normalized, which does not
 * mean that the user edited the draft after freezing it.
 *
 * @param {Readonly<Record<string, any>> | null} response
 * @param {Record<string, any>} submittedContent
 */
export function frozenAuthoringRecord(response, submittedContent) {
  const candidateId = response?.candidate?.candidate_id;
  return response?.ok === true && typeof candidateId === "string"
    ? {
        candidateId,
        content: cloneAuthoringValue(submittedContent),
      }
    : null;
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
    persistenceStatus: required("authoring-persistence-status"),
    savedDraftChoice: required("authoring-saved-draft-choice"),
    savedDraftSelect: required("authoring-saved-draft-select"),
    newScenarioChoice: required("authoring-new-scenario-choice"),
    newScenarioMode: required("authoring-new-scenario-mode"),
    newScenarioSourceChoice: required("authoring-new-scenario-source-choice"),
    newScenarioSource: required("authoring-new-scenario-source"),
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
    resetButton: required("authoring-reset"),
    recenterButton: required("authoring-recenter"),
    undoButton: required("authoring-undo"),
    redoButton: required("authoring-redo"),
    duplicateButton: required("authoring-duplicate"),
    deleteButton: required("authoring-delete"),
    orderUpButton: required("authoring-order-up"),
    orderDownButton: required("authoring-order-down"),
    problemList: required("authoring-problem-list"),
    problemCount: required("authoring-problem-count"),
  };

  function createEditorState() {
    return {
      draft: null,
      baseline: null,
      selectedId: null,
      problems: [],
      validation: null,
      past: [],
      future: [],
      camera: null,
      openSourceValue: "",
      frozen: null,
    };
  }

  const editors = {
    maps: createEditorState(),
    scenarios: createEditorState(),
  };
  /** @type {any} */
  const state = {
    token: acquireCapabilityToken(),
    area: "combat",
    editors,
    editor: editors.maps,
    assets: [],
    catalog: null,
    pointer: null,
    spacePressed: false,
    busy: false,
    newScenarioSourceValues: {
      copy_saved_map: "",
      duplicate_saved_scenario: "",
    },
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
  function showLocalError(message, editor = state.editor) {
    editor.problems = [
      {
        severity: "error",
        stable_code: "browser-authoring-operation",
        message,
        object_id: editor.selectedId,
        field_path: "browser",
      },
    ];
    renderProblems();
  }

  /** @param {Record<string, any>} response */
  function installResponse(response, editor = state.editor) {
    const nextDraft = draftAfterAuthoringResponse(editor.draft, response);
    if (nextDraft !== editor.draft) {
      const newDocument = ["new_map", "new_scenario", "open"].includes(
        response.command_type,
      );
      editor.draft = nextDraft;
      if (newDocument) {
        editor.selectedId = null;
        editor.past = [];
        editor.future = [];
        editor.camera = null;
        editor.frozen = null;
      }
      if (
        newDocument ||
        response.command_type === "save" ||
        response.command_type === "save_as"
      ) {
        editor.baseline = authoringContentSnapshot(nextDraft);
      }
      if (
        response.command_type === "open" ||
        response.command_type === "save" ||
        response.command_type === "save_as"
      ) {
        editor.openSourceValue = JSON.stringify({
          source_kind: "saved_draft",
          asset_kind: authoringKind(nextDraft),
          asset_id: nextDraft.asset_id,
          revision: nextDraft.revision,
        });
      } else if (newDocument) {
        editor.openSourceValue = "";
      }
    }
    if (response.command_type === "list" && Array.isArray(response.assets)) {
      state.assets = response.assets;
    }
    if (response.validation) {
      editor.validation = response.validation;
    }
    state.catalog = response.catalog ?? state.catalog;
    if (
      response.validation ||
      response.ok === false ||
      (Array.isArray(response.problems) && response.problems.length > 0)
    ) {
      editor.problems = normalizeAuthoringProblems(
        response.validation?.problems ?? response.problems ?? [],
      );
    }
    renderAll();
  }

  /** @param {Record<string, any>} command */
  async function send(command, editor = state.editor) {
    if (state.busy) {
      return null;
    }
    state.busy = true;
    renderAvailability();
    try {
      const response = await postAuthoringCommand(state.token, command);
      installResponse(response, editor);
      return response;
    } catch (error) {
      showLocalError(
        error instanceof DebuggerApiError || error instanceof Error
          ? error.message
          : "The authoring command failed.",
        editor,
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

  /** @param {Record<string, any>} command */
  async function sendAndRefreshAssets(command) {
    const response = await send(command);
    if (response?.ok) {
      await refreshAssets();
    }
    return response;
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

  /** @param {string} promptText @param {string} defaultId */
  function promptAssetId(promptText, defaultId) {
    const requested = window.prompt(promptText, defaultId);
    if (requested === null) {
      return null;
    }
    if (
      requested.length > 64 ||
      !/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/u.test(requested)
    ) {
      showLocalError(
        "Asset IDs use lowercase letters, digits, and internal hyphens only.",
      );
      return null;
    }
    return requested;
  }

  /** @param {"map" | "scenario"} kind */
  async function createDraft(kind) {
    if (state.busy) {
      return;
    }
    if (kind === "map") {
      await send({ command_type: "new_map" });
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
      if (!elements.newScenarioSource.value) {
        showLocalError("Choose a saved or frozen source for the new scenario.");
        return;
      }
      source = JSON.parse(elements.newScenarioSource.value);
    }
    /** @type {Record<string, any>} */
    const command = {
      command_type: "new_scenario",
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
    elements.combatConfig.hidden = area !== "combat";
    elements.shell.hidden = area === "combat";
    if (area !== "combat") {
      state.editor = state.editors[area];
    }
    await refreshAssets();
    if (area === "combat") {
      return;
    }
    const kind = area === "maps" ? "map" : "scenario";
    if (state.editor.draft === null) {
      await createDraft(kind);
    } else {
      renderAll();
    }
  }

  /** @param {Record<string, any>} next @param {Record<string, any>} [before] */
  function commit(next, before = state.editor.draft) {
    if (state.busy || before === null) {
      return;
    }
    state.editor.past.push(authoringContentSnapshot(before));
    state.editor.past = state.editor.past.slice(-50);
    state.editor.future = [];
    state.editor.draft = next;
    state.editor.validation = null;
    renderAll();
    void validateDraft();
  }

  function recenterAuthoringView() {
    if (state.busy || state.editor.draft === null) {
      return;
    }
    state.editor.camera = null;
    renderCanvas();
  }

  function resetAuthoringDraft() {
    const editor = state.editor;
    if (state.busy || editor.draft === null || editor.baseline === null) {
      return;
    }
    editor.camera = null;
    if (JSON.stringify(editor.draft.content) === JSON.stringify(editor.baseline)) {
      renderCanvas();
      return;
    }
    editor.past.push(authoringContentSnapshot(editor.draft));
    editor.past = editor.past.slice(-50);
    editor.future = [];
    editor.draft = restoreAuthoringContent(editor.draft, editor.baseline);
    editor.selectedId = null;
    editor.validation = null;
    renderAll();
    void validateDraft(editor);
  }

  async function validateDraft(editor = state.editor) {
    if (editor.draft !== null) {
      await send({ command_type: "validate", draft: editor.draft }, editor);
    }
  }

  function renderCombatOptions() {
    const selected = elements.scenarioSelect.value;
    elements.scenarioSelect.replaceChildren(new Option("Built-in arena", ""));
    for (const asset of state.assets.filter(
      (/** @type {any} */ candidate) => candidate.execution_valid,
    )) {
      elements.scenarioSelect.append(
        new Option(
          debugAssetOptionLabel(asset),
          JSON.stringify(persistedAuthoringSource(asset)),
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

  function renderSavedDraftOptions() {
    const kind = state.area === "scenarios" ? "scenario" : "map";
    const assets = openableDraftAssets(state.assets, kind);
    elements.savedDraftSelect.replaceChildren(
      new Option(`No saved ${kind} drafts`, ""),
    );
    for (const asset of assets) {
      elements.savedDraftSelect.append(
        new Option(
          savedDraftOptionLabel(asset),
          JSON.stringify(persistedAuthoringSource(asset)),
        ),
      );
    }
    const retained = state.editor.openSourceValue;
    if (
      retained &&
      [...elements.savedDraftSelect.options].some((option) => option.value === retained)
    ) {
      elements.savedDraftSelect.value = retained;
    } else if (assets.length > 0) {
      elements.savedDraftSelect.selectedIndex = 1;
      state.editor.openSourceValue = elements.savedDraftSelect.value;
    } else {
      state.editor.openSourceValue = "";
    }
    elements.savedDraftSelect.setAttribute("aria-label", `Saved ${kind} draft`);
  }

  function renderNewScenarioSourceOptions() {
    const mode = elements.newScenarioMode.value;
    const candidates = newScenarioSourceAssets(state.assets, mode);
    const sourceRequired = mode !== "blank";
    elements.newScenarioSourceChoice.hidden =
      state.area !== "scenarios" || !sourceRequired;
    elements.newScenarioSource.replaceChildren(
      new Option(
        sourceRequired ? "No compatible source assets" : "No source required",
        "",
      ),
    );
    for (const asset of candidates) {
      elements.newScenarioSource.append(
        new Option(
          authoringSourceOptionLabel(asset),
          JSON.stringify(persistedAuthoringSource(asset)),
        ),
      );
    }
    if (!sourceRequired) {
      return;
    }
    const retained = state.newScenarioSourceValues[mode] ?? "";
    if (
      retained &&
      [...elements.newScenarioSource.options].some(
        (option) => option.value === retained,
      )
    ) {
      elements.newScenarioSource.value = retained;
    } else if (candidates.length > 0) {
      elements.newScenarioSource.selectedIndex = 1;
      state.newScenarioSourceValues[mode] = elements.newScenarioSource.value;
    } else {
      state.newScenarioSourceValues[mode] = "";
    }
  }

  function renderPersistenceStatus() {
    elements.persistenceStatus.textContent = authoringPersistenceMessage(
      state.editor.draft,
      state.editor.baseline,
      state.editor.frozen,
    );
  }

  function renderObjectList() {
    elements.objectList.replaceChildren();
    if (state.editor.draft === null) {
      elements.objectCount.textContent = "0";
      return;
    }
    const objects = authoringObjects(state.editor.draft);
    const rows = [
      { object_id: "", label: `${authoringKind(state.editor.draft)} document` },
      ...objects,
    ];
    for (const object of rows) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.objectId = object.object_id;
      button.setAttribute(
        "aria-current",
        String((object.object_id || null) === state.editor.selectedId),
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
    elements.problemCount.textContent = String(state.editor.problems.length);
    if (state.editor.problems.length === 0) {
      const row = document.createElement("li");
      row.className = "empty-copy";
      row.textContent = state.editor.validation?.freeze_qualified
        ? "Execution-valid and freeze-qualified."
        : state.editor.validation?.execution_valid
          ? "Execution-valid."
          : "Validate to inspect host-authoritative errors and warnings.";
      elements.problemList.append(row);
      return;
    }
    for (const problem of state.editor.problems) {
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
    if (state.editor.draft === null) {
      elements.canvas.replaceChildren();
      return;
    }
    const map = mapContent(state.editor.draft);
    state.editor.camera = renderAuthoringSvg(
      elements.canvas,
      state.editor.draft,
      state.editor.selectedId,
      normalizeAuthoringCamera(state.editor.camera, map.width, map.height),
      catalogNumber("fixed_grid_world_units"),
      state.catalog,
    );
  }

  /** @param {number} clientX @param {number} clientY */
  function authoringWorldPoint(clientX, clientY) {
    if (state.editor.draft === null) {
      return null;
    }
    const map = mapContent(state.editor.draft);
    const dimensions = authoringMapDimensions(map.width, map.height);
    if (dimensions === null) {
      return null;
    }
    return authoringClientPointToWorld(
      elements.canvas.getBoundingClientRect(),
      state.editor.camera,
      dimensions.height,
      clientX,
      clientY,
    );
  }

  function renderAvailability() {
    const selected =
      state.editor.draft &&
      selectedAuthoringObject(state.editor.draft, state.editor.selectedId);
    const obstacle = selected?.kind === "wall" || selected?.kind === "pillar";
    elements.shell.inert = state.busy;
    elements.shell.setAttribute("aria-busy", String(state.busy));
    for (const button of elements.nav.querySelectorAll("button")) {
      button.disabled = state.busy;
    }
    elements.scenarioSelect.disabled = state.busy;
    elements.scenarioLoad.disabled = state.busy || !elements.scenarioSelect.value;
    elements.savedDraftSelect.disabled = state.busy || !elements.savedDraftSelect.value;
    elements.newScenarioMode.disabled = state.busy;
    elements.newScenarioSource.disabled =
      state.busy ||
      elements.newScenarioMode.value === "blank" ||
      !elements.newScenarioSource.value;
    elements.newButton.disabled = state.busy;
    elements.openButton.disabled = state.busy || !elements.savedDraftSelect.value;
    elements.saveButton.disabled = state.busy || !state.editor.draft;
    elements.saveAsButton.disabled = state.busy || !state.editor.draft;
    elements.validateButton.disabled = state.busy || !state.editor.draft;
    elements.resetButton.disabled =
      state.busy || !state.editor.draft || !state.editor.baseline;
    elements.recenterButton.disabled = state.busy || !state.editor.draft;
    elements.undoButton.disabled = state.busy || state.editor.past.length === 0;
    elements.redoButton.disabled = state.busy || state.editor.future.length === 0;
    elements.duplicateButton.disabled = state.busy || !obstacle;
    elements.deleteButton.disabled = state.busy || !obstacle;
    elements.orderUpButton.disabled = state.busy || !obstacle;
    elements.orderDownButton.disabled = state.busy || !obstacle;
    elements.freezeButton.disabled =
      state.busy || !state.editor.validation?.freeze_qualified;
    elements.openDebugButton.disabled =
      state.busy || !state.editor.draft || !state.editor.validation?.execution_valid;
  }

  function renderAll() {
    renderCombatOptions();
    renderSavedDraftOptions();
    renderNewScenarioSourceOptions();
    if (state.editor.draft) {
      const kind = authoringKind(state.editor.draft);
      elements.eyebrow.textContent = kind === "map" ? "Map Author" : "Scenario Author";
      elements.title.textContent = state.editor.draft.content.name;
      elements.openDebugButton.textContent =
        kind === "map" ? "Preview Map in Debug" : "Open Scenario in Debug";
    }
    renderPersistenceStatus();
    renderObjectList();
    renderCanvas();
    renderAuthoringInspector(
      elements.inspector,
      state.editor.draft,
      state.editor.selectedId,
      state.catalog,
      state.editor.validation,
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
  elements.savedDraftSelect.addEventListener("change", () => {
    state.editor.openSourceValue = elements.savedDraftSelect.value;
    renderAvailability();
  });
  elements.newScenarioMode.addEventListener("change", () => {
    renderAll();
  });
  elements.newScenarioSource.addEventListener("change", () => {
    const mode = elements.newScenarioMode.value;
    if (mode === "copy_saved_map" || mode === "duplicate_saved_scenario") {
      state.newScenarioSourceValues[mode] = elements.newScenarioSource.value;
    }
    renderAvailability();
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
    if (state.busy || !elements.savedDraftSelect.value) {
      return;
    }
    await send({
      command_type: "open",
      source: JSON.parse(elements.savedDraftSelect.value),
    });
  });
  elements.saveButton.addEventListener("click", async () => {
    const draft = state.editor.draft;
    if (state.busy || !draft) {
      return;
    }
    if (draft.revision === 0) {
      const assetId = promptAssetId("New asset ID", draft.asset_id);
      if (assetId !== null) {
        await sendAndRefreshAssets({
          command_type: "save_as",
          draft,
          asset_id: assetId,
        });
      }
      return;
    }
    await sendAndRefreshAssets({
      command_type: "save",
      draft,
      expected_revision: draft.revision,
    });
  });
  elements.saveAsButton.addEventListener("click", async () => {
    const draft = state.editor.draft;
    if (state.busy || !draft) {
      return;
    }
    const assetId = promptAssetId("New asset ID", `${draft.asset_id}-copy`);
    if (assetId !== null) {
      await sendAndRefreshAssets({ command_type: "save_as", draft, asset_id: assetId });
    }
  });
  elements.validateButton.addEventListener("click", () => {
    if (!state.busy) {
      void validateDraft();
    }
  });
  elements.freezeButton.addEventListener("click", async () => {
    if (!state.busy && state.editor.draft) {
      const editor = state.editor;
      const submittedContent = authoringContentSnapshot(editor.draft);
      const response = await send({
        command_type: "freeze",
        draft: editor.draft,
      });
      const frozen = frozenAuthoringRecord(response, submittedContent);
      if (frozen !== null) {
        editor.frozen = frozen;
        renderAll();
        await refreshAssets();
      }
    }
  });
  elements.openDebugButton.addEventListener("click", async () => {
    if (state.busy || !state.editor.draft) {
      return;
    }
    await openInDebug(
      {
        source_kind: "current_buffer",
        asset_kind: authoringKind(state.editor.draft),
        draft: state.editor.draft,
      },
      true,
    );
  });

  elements.resetButton.addEventListener("click", resetAuthoringDraft);
  elements.recenterButton.addEventListener("click", recenterAuthoringView);

  elements.palette.addEventListener("click", (/** @type {Event} */ event) => {
    if (authoringInteractionBlocked(event)) {
      return;
    }
    const button = closest(event, "[data-authoring-add]");
    if (!button || !state.editor.draft) {
      return;
    }
    try {
      const obstacleKind = button.getAttribute("data-authoring-add");
      if (obstacleKind !== "wall" && obstacleKind !== "pillar") {
        return;
      }
      const result = addAuthoringObstacle(
        state.editor.draft,
        obstacleKind,
        catalogNumber("maximum_obstacle_slots"),
      );
      state.editor.selectedId = result.object_id;
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
      state.editor.selectedId = button.getAttribute("data-object-id") || null;
      renderAll();
    }
  });
  elements.objectList.addEventListener(
    "dragstart",
    (/** @type {DragEvent} */ event) => {
      if (
        authoringInteractionBlocked(event) ||
        !event.dataTransfer ||
        !state.editor.draft
      ) {
        return;
      }
      const row = closest(event, '[data-authoring-roster-agent="true"]');
      const objectId = row?.getAttribute("data-object-id");
      const object = objectId
        ? selectedAuthoringObject(state.editor.draft, objectId)
        : null;
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
      state.editor.selectedId = button.getAttribute("data-object-id") || null;
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
    if (!edit || !state.editor.draft) {
      return;
    }
    const alive = edit.path.at(-1) === "alive" && state.editor.selectedId;
    const teamSize =
      edit.path.length === 2 &&
      edit.path[0] === "content" &&
      ["team_a_size", "team_b_size"].includes(edit.path[1]);
    commit(
      alive
        ? setAgentAlive(
            state.editor.draft,
            state.editor.selectedId,
            Boolean(edit.value),
          )
        : teamSize
          ? setScenarioTeamSize(
              state.editor.draft,
              edit.path[1] === "team_a_size" ? "A" : "B",
              edit.value,
              state.catalog,
            )
          : setAuthoringField(state.editor.draft, edit.path, edit.value),
    );
  });

  elements.undoButton.addEventListener("click", () => {
    const previousContent = state.busy ? null : state.editor.past.pop();
    if (previousContent && state.editor.draft) {
      state.editor.future.push(authoringContentSnapshot(state.editor.draft));
      state.editor.draft = restoreAuthoringContent(state.editor.draft, previousContent);
      renderAll();
      void validateDraft();
    }
  });
  elements.redoButton.addEventListener("click", () => {
    const nextContent = state.busy ? null : state.editor.future.pop();
    if (nextContent && state.editor.draft) {
      state.editor.past.push(authoringContentSnapshot(state.editor.draft));
      state.editor.draft = restoreAuthoringContent(state.editor.draft, nextContent);
      renderAll();
      void validateDraft();
    }
  });
  elements.duplicateButton.addEventListener("click", () => {
    if (!state.busy && state.editor.draft && state.editor.selectedId) {
      const result = duplicateAuthoringObstacle(
        state.editor.draft,
        state.editor.selectedId,
        catalogNumber("maximum_obstacle_slots"),
        catalogNumber("fixed_snap_world_units"),
      );
      state.editor.selectedId = result.object_id;
      commit(result.draft);
    }
  });
  elements.deleteButton.addEventListener("click", () => {
    if (!state.busy && state.editor.draft && state.editor.selectedId) {
      const next = deleteAuthoringObstacle(state.editor.draft, state.editor.selectedId);
      state.editor.selectedId = null;
      commit(next);
    }
  });
  elements.orderUpButton.addEventListener("click", () => {
    if (!state.busy && state.editor.draft && state.editor.selectedId) {
      commit(reorderAuthoringObstacle(state.editor.draft, state.editor.selectedId, -1));
    }
  });
  elements.orderDownButton.addEventListener("click", () => {
    if (!state.busy && state.editor.draft && state.editor.selectedId) {
      commit(reorderAuthoringObstacle(state.editor.draft, state.editor.selectedId, 1));
    }
  });

  elements.canvas.addEventListener("dragover", (/** @type {DragEvent} */ event) => {
    if (
      !state.busy &&
      state.editor.draft !== null &&
      authoringKind(state.editor.draft) === "scenario"
    ) {
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "move";
      }
    }
  });
  elements.canvas.addEventListener("drop", (/** @type {DragEvent} */ event) => {
    if (
      authoringInteractionBlocked(event) ||
      !state.editor.draft ||
      !event.dataTransfer
    ) {
      return;
    }
    const objectId =
      event.dataTransfer.getData(AUTHORING_AGENT_DRAG_TYPE) ||
      event.dataTransfer.getData("text/plain");
    const object = selectedAuthoringObject(state.editor.draft, objectId);
    const world = authoringWorldPoint(event.clientX, event.clientY);
    if (object?.kind !== "agent" || world === null) {
      return;
    }
    event.preventDefault();
    state.editor.selectedId = objectId;
    commit(
      moveAuthoringObjectWithSnap(
        state.editor.draft,
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
      if (authoringInteractionBlocked(event) || !state.editor.draft) {
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
        state.editor.selectedId = object.getAttribute("data-object-id");
        state.pointer = {
          kind: "drag",
          pointerId: event.pointerId,
          objectId: state.editor.selectedId,
          before: cloneAuthoringValue(state.editor.draft),
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
        !state.editor.draft
      ) {
        return;
      }
      if (state.pointer.kind === "pan") {
        const bounds = elements.canvas.getBoundingClientRect();
        state.editor.camera = panAuthoringCamera(
          state.editor.camera,
          -((event.clientX - state.pointer.x) * state.editor.camera.width) /
            bounds.width,
          -((event.clientY - state.pointer.y) * state.editor.camera.height) /
            bounds.height,
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
      state.editor.draft = moveAuthoringObjectWithSnap(
        state.editor.draft,
        state.pointer.objectId,
        world.x,
        world.y,
        catalogNumber("fixed_snap_world_units"),
        event.altKey,
      );
      renderCanvas();
      renderAuthoringInspector(
        elements.inspector,
        state.editor.draft,
        state.editor.selectedId,
        state.catalog,
        state.editor.validation,
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
    if (!state.busy && pointer.kind === "drag" && state.editor.draft) {
      commit(state.editor.draft, pointer.before);
    }
  }
  elements.canvas.addEventListener("pointerup", finishPointer);
  elements.canvas.addEventListener("pointercancel", finishPointer);
  elements.canvas.addEventListener(
    "wheel",
    (/** @type {WheelEvent} */ event) => {
      event.preventDefault();
      if (authoringInteractionBlocked(event) || !state.editor.draft) {
        return;
      }
      const map = mapContent(state.editor.draft);
      const dimensions = authoringMapDimensions(map.width, map.height);
      const world = authoringWorldPoint(event.clientX, event.clientY);
      if (dimensions === null || world === null) {
        return;
      }
      state.editor.camera = zoomAuthoringCamera(
        state.editor.camera,
        dimensions.width,
        dimensions.height,
        world,
        event.deltaY > 0 ? 1.15 : 1 / 1.15,
      );
      renderCanvas();
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
      state.editor.draft &&
      selectedAuthoringObject(state.editor.draft, state.editor.selectedId);
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
          state.editor.draft,
          state.editor.selectedId,
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
  document.addEventListener("keydown", (/** @type {KeyboardEvent} */ event) => {
    const target = event.target;
    const editing =
      target instanceof Element &&
      target.closest("input, textarea, select, button, [contenteditable]") !== null;
    if (
      state.area === "combat" ||
      elements.shell.hidden ||
      editing ||
      document.querySelector("dialog[open]") !== null ||
      event.repeat ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey ||
      event.key.toLowerCase() !== "r"
    ) {
      return;
    }
    event.preventDefault();
    resetAuthoringDraft();
  });

  renderAll();
  void refreshAssets();
}
