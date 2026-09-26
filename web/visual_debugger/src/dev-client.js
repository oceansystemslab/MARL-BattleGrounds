/**
 * @file Install the DevClient map/scenario editor when the server enables authoring.
 * Local drafts, selection, camera, and bounded undo history belong to the browser;
 * validation, saved revisions, and live scenario replacement belong to the host.
 * This module binds DOM events and sends whole-draft commands through api.js.
 * Combat selectors reflect confirmed host configuration, not unconfirmed choices.
 * Failed Combat loads publish their error to the main client's existing notice.
 */
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
  renameAuthoringObstacleId,
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

const ASSET_ID_COLLATOR = new Intl.Collator("en", {
  numeric: true,
  sensitivity: "base",
});

/**
 * Return a sorted copy of assets using English numeric asset-ID order, then kind
 * and revision. The host discovery array and its entry objects are not changed.
 *
 * @param {ReadonlyArray<Record<string, any>>} assets
 */
export function orderedAuthoringAssets(assets) {
  return [...assets].sort((left, right) => {
    const identityOrder = ASSET_ID_COLLATOR.compare(left.asset_id, right.asset_id);
    if (identityOrder !== 0) {
      return identityOrder;
    }
    const kindOrder = String(left.asset_kind).localeCompare(String(right.asset_kind));
    return kindOrder !== 0 ? kindOrder : Number(left.revision) - Number(right.revision);
  });
}

/**
 * Return saved-draft assets of the requested map or scenario kind in display order.
 * Invalid-for-execution drafts remain openable for repair. No asset is loaded.
 *
 * @param {ReadonlyArray<Record<string, any>>} assets @param {"map" | "scenario"} kind
 */
export function openableDraftAssets(assets, kind) {
  return orderedAuthoringAssets(
    assets.filter(
      (asset) => asset.asset_kind === kind && asset.source_kind === "saved_draft",
    ),
  );
}

/**
 * Return execution-valid saved drafts in deterministic display order. These are
 * eligible launcher choices; listing them does not reset the live debugger.
 *
 * @param {ReadonlyArray<Record<string, any>>} assets
 */
export function debuggableAuthoringAssets(assets) {
  return orderedAuthoringAssets(
    assets.filter(
      (asset) => asset.source_kind === "saved_draft" && asset.execution_valid,
    ),
  );
}

/**
 * Choose the editor draft after a host response. Validation replies and replies
 * without a draft preserve the exact currentDraft reference so delayed validation
 * cannot erase edits. Other returned drafts are cloned before local editing.
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

/**
 * Build an explicit saved-draft source descriptor from asset kind, ID, and revision.
 * Throw TypeError for any source_kind other than saved_draft. Return a new object
 * without mutating the discovery row.
 *
 * @param {Readonly<Record<string, any>>} asset
 */
export function persistedAuthoringSource(asset) {
  if (asset.source_kind !== "saved_draft") {
    throw new TypeError("DevClient sources must identify a saved draft revision.");
  }
  return {
    source_kind: "saved_draft",
    asset_kind: asset.asset_kind,
    asset_id: asset.asset_id,
    revision: asset.revision,
  };
}

/**
 * Format asset ID, name, revision, and map dimensions for an open-draft option.
 * Dimensions use the host row's map units; this returns text only.
 *
 * @param {Readonly<Record<string, any>>} asset
 */
export function savedDraftOptionLabel(asset) {
  return `${asset.asset_id} · ${asset.name} · revision ${asset.revision} · ${asset.map_width} × ${asset.map_height}`;
}

/**
 * Return saved maps for copy_saved_map or saved scenarios for duplicate_saved_scenario.
 * Other creationMode values return an empty array. Keep deterministic display order
 * and allow drafts needing validation fixes so they can be edited.
 *
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
    : orderedAuthoringAssets(
        assets.filter(
          (asset) =>
            asset.source_kind === "saved_draft" && asset.asset_kind === sourceKind,
        ),
      );
}

/**
 * Format a source draft label with identity, dimensions, and execution-valid status.
 * This returns text and does not perform validation.
 *
 * @param {Readonly<Record<string, any>>} asset
 */
export function authoringSourceOptionLabel(asset) {
  const status = asset.execution_valid ? "execution-valid" : "needs validation fixes";
  return `${asset.asset_id} · ${asset.name} · revision ${asset.revision} · ${asset.map_width} × ${asset.map_height} · ${status}`;
}

/**
 * Format a saved asset's debug-launch label. Map choices describe a default 5v5 TDM
 * preview; scenario choices describe their authored scenario. This assumes the
 * caller already filtered execution-valid assets.
 *
 * @param {Readonly<Record<string, any>>} asset
 */
export function debugAssetOptionLabel(asset) {
  const identity = `${asset.asset_id} · revision ${asset.revision}`;
  return asset.asset_kind === "map"
    ? `${identity} · Map preview · ${asset.name} · ${asset.map_width} × ${asset.map_height} · execution-valid · default 5v5 TDM`
    : `${identity} · Scenario · ${asset.name} · ${asset.map_width} × ${asset.map_height} · execution-valid`;
}

/**
 * Describe whether draft is absent, new, changed, or equal to baseline content.
 * baseline is the last installed saved/new content snapshot, or null. Return a
 * status string including the saved path only for unchanged persisted content.
 * JSON serialization errors propagate; no file existence check occurs.
 *
 * @param {Record<string, any> | null} draft @param {Record<string, any> | null} baseline
 */
export function authoringPersistenceMessage(draft, baseline) {
  if (draft === null) {
    return "No authoring draft is open.";
  }
  const kind = authoringKind(draft);
  const collection = kind === "map" ? "maps" : "scenarios";
  const currentContent = JSON.stringify(draft.content);
  const savedContent = baseline === null ? null : JSON.stringify(baseline);
  const savedPath = `artifacts/dev_client/drafts/${collection}/${draft.asset_id}/r${draft.revision}.json`;
  if (draft.revision === 0) {
    return `Unsaved ${kind} draft`;
  }
  if (savedContent !== currentContent) {
    return `Unsaved changes · last saved ${kind} ${draft.asset_id} revision ${draft.revision}`;
  }
  return `Saved ${kind} ${draft.asset_id} · revision ${draft.revision} · ${savedPath}`;
}

/**
 * Preserve unrelated draft references. If source names the open saved asset, return
 * a cloned unsaved recovery draft with an untitled ID and revision zero. This
 * changes no server file and does not discard the open content.
 *
 * @param {Record<string, any> | null} draft
 * @param {Readonly<Record<string, any>>} source
 */
export function draftAfterSavedAssetDeletion(draft, source) {
  if (
    draft === null ||
    source.source_kind !== "saved_draft" ||
    source.asset_kind !== authoringKind(draft) ||
    source.asset_id !== draft.asset_id
  ) {
    return draft;
  }
  const recovery = cloneAuthoringValue(draft);
  recovery.asset_id =
    source.asset_kind === "map" ? "untitled_map" : "untitled_scenario";
  recovery.revision = 0;
  return recovery;
}

/**
 * Build confirmation text explaining that deleting asset removes all saved revisions.
 * Return text only; the caller owns confirmation and the host deletion command.
 *
 * @param {Readonly<Record<string, any>>} asset
 */
export function savedAssetDeletionPrompt(asset) {
  return `Permanently delete saved ${asset.asset_kind} "${asset.name}" (asset ID: ${asset.asset_id}) and all of its revisions? This cannot be undone.`;
}

/**
 * Return whether value is a nonempty lowercase snake_case string of at most
 * 64 characters. Only lowercase letters, digits, and single separators are allowed.
 *
 * @param {unknown} value
 */
export function isValidAuthoringAssetId(value) {
  return (
    typeof value === "string" &&
    value.length <= 64 &&
    /^[a-z0-9]+(?:_[a-z0-9]+)*$/u.test(value)
  );
}

/**
 * Create install/request/render operations around the supplied selector bindings.
 * Only confirmed host configuration becomes authoritative. A user request first
 * restores displayed confirmed values, then emits a valid changed intent. Return
 * a frozen controller; no HTTP request is sent directly. Both teams accept the
 * same five controllers: Manual, Random and the reactive ALPHA (reactive_tdm),
 * BETA (scenario_5) and GAMMA (tdm_gamma). Reactive controllers require SharedObs
 * whichever team uses them. reactiveControllerOptions holds both teams' ALPHA
 * options and scenarioControllerOptions holds both teams' BETA and GAMMA
 * options; all of them are enabled only under confirmed SharedObs. The
 * NoSharedObs option is disabled while either team uses a reactive controller.
 *
 * @param {{
 *   teamAController: {value: string, disabled: boolean},
 *   teamBController: {value: string, disabled: boolean},
 *   informationMode: {value: string, disabled: boolean},
 *   reactiveControllerOptions: {disabled: boolean}[],
 *   scenarioControllerOptions: {disabled: boolean}[],
 *   noSharedOption: {disabled: boolean},
 *   root: {dataset: Record<string, string | undefined>},
 *   emit: (configuration: Readonly<Record<string, string>>) => void,
 * }} bindings
 */
export function createCombatConfigurationController(bindings) {
  /** @type {Readonly<Record<string, string>> | null} */
  let authoritative = null;

  /**
   * Return whether value is a controller either team may use: manual,
   * random_valid, reactive_tdm, scenario_5 or tdm_gamma.
   *
   * @param {unknown} value
   */
  function isSupportedController(value) {
    return (
      value === "manual" || value === "random_valid" || isReactiveController(value)
    );
  }

  /**
   * Return whether value is a reactive controller that needs SharedObs:
   * reactive_tdm (ALPHA), scenario_5 (BETA) or tdm_gamma (GAMMA).
   *
   * @param {unknown} value
   */
  function isReactiveController(value) {
    return value === "reactive_tdm" || value === "scenario_5" || value === "tdm_gamma";
  }

  /**
   * Return a frozen supported controller/information-mode selection or null.
   * Reject unsupported combinations without changing controls or host state.
   *
   * @param {unknown} value
   */
  function normalize(value) {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return null;
    }
    const candidate = /** @type {Record<string, unknown>} */ (value);
    if (
      !isSupportedController(candidate.team_a_controller) ||
      !isSupportedController(candidate.team_b_controller) ||
      (candidate.execution_information_mode !== "shared_obs" &&
        candidate.execution_information_mode !== "no_shared_obs") ||
      ((isReactiveController(candidate.team_a_controller) ||
        isReactiveController(candidate.team_b_controller)) &&
        candidate.execution_information_mode !== "shared_obs")
    ) {
      return null;
    }
    return Object.freeze({
      team_a_controller: candidate.team_a_controller,
      team_b_controller: candidate.team_b_controller,
      execution_information_mode: candidate.execution_information_mode,
    });
  }

  /**
   * Project installed configuration into selector values, enabled options, and root
   * dataset fields. With no authority, disable selectors and retain no inferred config.
   */
  function render() {
    const configuration = authoritative;
    bindings.teamAController.disabled = configuration === null;
    bindings.teamBController.disabled = configuration === null;
    bindings.informationMode.disabled = configuration === null;
    for (const option of [
      ...bindings.reactiveControllerOptions,
      ...bindings.scenarioControllerOptions,
    ]) {
      option.disabled = configuration?.execution_information_mode !== "shared_obs";
    }
    bindings.noSharedOption.disabled =
      isReactiveController(configuration?.team_a_controller) ||
      isReactiveController(configuration?.team_b_controller);
    if (configuration === null) {
      return;
    }
    bindings.teamAController.value = configuration.team_a_controller;
    bindings.teamBController.value = configuration.team_b_controller;
    bindings.informationMode.value = configuration.execution_information_mode;
    bindings.root.dataset.teamAController = configuration.team_a_controller;
    bindings.root.dataset.teamBController = configuration.team_b_controller;
    bindings.root.dataset.executionInformationMode =
      configuration.execution_information_mode;
  }

  return Object.freeze({
    /**
     * Install a supported confirmed configuration and redraw selectors.
     * Return false without changes for invalid value, or true after installation.
     * @param {unknown} value
     */
    install(value) {
      const normalized = normalize(value);
      if (normalized === null) {
        return false;
      }
      authoritative = normalized;
      render();
      return true;
    },
    /**
     * Read selector intent, restore confirmed values, and emit a valid changed choice.
     * Return true only when emit is called. Callback errors propagate; this method
     * does not optimistically replace the installed configuration.
     */
    request() {
      const requested = normalize({
        team_a_controller: bindings.teamAController.value,
        team_b_controller: bindings.teamBController.value,
        execution_information_mode: bindings.informationMode.value,
      });
      render();
      if (
        requested === null ||
        authoritative === null ||
        (requested.team_a_controller === authoritative.team_a_controller &&
          requested.team_b_controller === authoritative.team_b_controller &&
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

/**
 * Bind the enabled DevClient DOM, independent map/scenario editor state, and events.
 * Acquire the tab token, initialize confirmed combat controls, and install local
 * editing, navigation, pointer, keyboard, save, validation, and host-handoff handlers.
 * Missing required shell elements throw Error. Returns undefined; listeners live
 * for this page lifetime and are installed only for an authoring-enabled live product.
 */
function installDevClient() {
  /**
   * Find the DOM element with id or throw Error naming the missing DevClient shell node.
   *
   * @param {string} id
   */
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
    teamAController: required("devclient-team-a-controller"),
    teamBController: required("devclient-team-b-controller"),
    informationMode: required("devclient-information-mode"),
    scenarioControllerOptions: [
      required("devclient-team-a-scenario-5-option"),
      required("devclient-team-a-tdm-gamma-option"),
      required("devclient-scenario-5-controller-option"),
      required("devclient-tdm-gamma-controller-option"),
    ],
    reactiveControllerOptions: [
      required("devclient-team-a-reactive-option"),
      required("devclient-team-b-reactive-option"),
    ],
    noSharedOption: required("devclient-no-shared-option"),
    shell: required("authoring-shell"),
    eyebrow: required("authoring-eyebrow"),
    title: required("authoring-title"),
    persistenceStatus: required("authoring-persistence-status"),
    savedDraftChoice: required("authoring-saved-draft-choice"),
    savedDraftSelect: required("authoring-saved-draft-select"),
    deleteSavedButton: required("authoring-delete-saved"),
    newScenarioChoice: required("authoring-new-scenario-choice"),
    newScenarioMode: required("authoring-new-scenario-mode"),
    newScenarioSourceChoice: required("authoring-new-scenario-source-choice"),
    newScenarioSource: required("authoring-new-scenario-source"),
    newButton: required("authoring-new"),
    openButton: required("authoring-open"),
    saveButton: required("authoring-save"),
    saveAsButton: required("authoring-save-as"),
    validateButton: required("authoring-validate"),
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

  /**
   * Return fresh empty draft, baseline, selection, validation, camera, and history state.
   * Map and scenario editors receive independent objects and arrays.
   */
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
    teamAController: elements.teamAController,
    teamBController: elements.teamBController,
    informationMode: elements.informationMode,
    scenarioControllerOptions: elements.scenarioControllerOptions,
    reactiveControllerOptions: elements.reactiveControllerOptions,
    noSharedOption: elements.noSharedOption,
    root: document.documentElement,
    /**
     * Publish the resolved combat configuration as a document event.
     * The configuration argument contains both controller choices and the execution
     * information mode. Listeners own applying/resetting the live session; this
     * callback only dispatches marl-devclient-combat-configuration synchronously.
     */
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

  /**
   * Return the nearest selector match from an Element event target, otherwise null.
   * The DOM is unchanged; invalid selector syntax can throw.
   *
   * @param {Event} event @param {string} selector
   */
  function closest(event, selector) {
    return event.target instanceof Element ? event.target.closest(selector) : null;
  }

  /**
   * Return whether an authoring request is busy. If blocked, prevent the optional
   * event's default action; event defaults to null. No command is queued.
   *
   * @param {Event | null} [event]
   */
  function authoringInteractionBlocked(event = null) {
    if (!state.busy) {
      return false;
    }
    event?.preventDefault();
    return true;
  }

  /**
   * Replace editor problems with one field-linked browser error and redraw the problem
   * list. editor defaults to the current editor; message is shown as text.
   *
   * @param {string} message
   */
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

  /**
   * Install a host reply into editor, which defaults to the current editor.
   * New/open drafts reset selection/history/camera; saved responses refresh the baseline.
   * Validation echoes preserve local draft identity. Update catalog/assets/problems
   * when supplied, then render all controls. Invalid model operations can throw.
   *
   * @param {Record<string, any>} response
   */
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

  /**
   * Send one authoring command for editor, defaulting to the active editor. Busy calls
   * resolve null. Otherwise fence interaction, install the response, and return it.
   * Errors become local problem text and a null result; the busy fence always clears.
   * The API helper never retries an uncertain POST.
   *
   * @param {Record<string, any>} command
   */
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

  /**
   * Send one all-assets list command and install its response. Resolve after completion;
   * failures are shown by send and do not replace existing discovery state.
   */
  async function refreshAssets() {
    await send({ command_type: "list", asset_kind: "all" });
  }

  /**
   * Send command and refresh asset discovery only if the response reports ok.
   * Return the original response or null; failed commands are not retried.
   *
   * @param {Record<string, any>} command
   */
  async function sendAndRefreshAssets(command) {
    const response = await send(command);
    if (response?.ok) {
      await refreshAssets();
    }
    return response;
  }

  /**
   * Dispatch the page event that tells the main client its live scenario was replaced.
   * This notification alone does not fetch or install a frame.
   */
  function notifyDebugSessionReplaced() {
    document.dispatchEvent(new CustomEvent("marl-devclient-debug-session-replaced"));
  }

  /**
   * Ask the host to load source into the live debugger. On success notify the main
   * client and optionally select Combat; returnToCombat defaults to false. Without
   * a successful reply, keep the current area and browser frame. The editor shows
   * its problems in authoring areas; Combat sends their text through
   * marl-devclient-debug-load-failed to the main client's notice owner. A failed
   * connection keeps its unknown-outcome warning and is never retried here.
   *
   * @param {Record<string, any>} source @param {boolean} returnToCombat
   */
  async function openInDebug(source, returnToCombat = false) {
    const response = await send({ command_type: "open_in_debug", source });
    if (!response?.ok) {
      if (state.area === "combat") {
        document.dispatchEvent(
          new CustomEvent("marl-devclient-debug-load-failed", {
            detail: {
              message:
                state.editor.problems
                  .map((/** @type {{message: string}} */ problem) => problem.message)
                  .join("\n") || "Could not load the selected match.",
            },
          }),
        );
      }
      return;
    }
    notifyDebugSessionReplaced();
    if (returnToCombat) {
      await selectArea("combat");
    }
  }

  /**
   * Read the requested positive finite numeric catalog field. Return its value or
   * throw TypeError naming the missing field; grid and snap values use map units.
   *
   * @param {"maximum_obstacle_slots" | "fixed_grid_world_units" | "fixed_snap_world_units"} key
   */
  function catalogNumber(key) {
    const value = state.catalog?.[key];
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      throw new TypeError(`Authoring catalog is missing ${key}.`);
    }
    return value;
  }

  /**
   * Prompt with promptText and defaultId. Return a valid lowercase asset ID or null
   * on cancellation/invalid text; invalid text also becomes a local error. Sends no request.
   *
   * @param {string} promptText @param {string} defaultId
   */
  function promptAssetId(promptText, defaultId) {
    const requested = window.prompt(promptText, defaultId);
    if (requested === null) {
      return null;
    }
    if (!isValidAuthoringAssetId(requested)) {
      showLocalError(
        "Asset IDs use lowercase snake_case: lowercase letters and digits separated by single underscores.",
      );
      return null;
    }
    return requested;
  }

  /**
   * Create a map or scenario through the host unless busy. Scenario mode selects blank,
   * copied saved map, or duplicated saved scenario and requires its explicit source.
   * Missing/invalid choices become local errors; saved-source JSON parse errors can
   * propagate. A successful response is installed by send.
   *
   * @param {"map" | "scenario"} kind
   */
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
        showLocalError("Choose a saved source for the new scenario.");
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

  /**
   * Switch between Combat, Maps, and Scenarios unless busy. Update navigation and
   * visibility, refresh assets, and create a draft when the selected editor is empty.
   * Each editor retains its own draft/history; switching does not save it.
   *
   * @param {"combat" | "maps" | "scenarios"} area
   */
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

  /**
   * Install next as a local edit and push before content into a 50-entry undo history.
   * before defaults to the current draft. Busy or null-before calls do nothing.
   * Clear redo/validation, redraw, and start host validation without awaiting it.
   *
   * @param {Record<string, any>} next @param {Record<string, any>} [before]
   */
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

  /**
   * Clear selectedId only when the restored draft no longer contains that object.
   * The draft and history are unchanged.
   */
  function clearStaleSelection() {
    if (
      state.editor.draft !== null &&
      state.editor.selectedId !== null &&
      selectedAuthoringObject(state.editor.draft, state.editor.selectedId) === null
    ) {
      state.editor.selectedId = null;
    }
  }

  /**
   * Clear the active draft's camera and redraw its canvas unless busy or no draft exists.
   * This changes only the browser view, not authored content.
   */
  function recenterAuthoringView() {
    if (state.busy || state.editor.draft === null) {
      return;
    }
    state.editor.camera = null;
    renderCanvas();
  }

  /**
   * Restore the active draft content from its baseline and recenter the view.
   * Changed content becomes one undoable edit, clears selection/redo/validation, and
   * starts host validation. Busy, missing-draft, or missing-baseline calls do nothing.
   */
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

  /**
   * Submit editor's full draft for host validation when present. editor defaults to
   * the current editor. Resolve after send; validation echoes cannot replace local content.
   */
  async function validateDraft(editor = state.editor) {
    if (editor.draft !== null) {
      await send({ command_type: "validate", draft: editor.draft }, editor);
    }
  }

  /**
   * Rebuild the debug-launch dropdown from valid saved assets, retaining its selected
   * source if still present. Update Load availability; send no host command.
   */
  function renderCombatOptions() {
    const selected = elements.scenarioSelect.value;
    elements.scenarioSelect.replaceChildren(new Option("Built-in arena", ""));
    for (const asset of debuggableAuthoringAssets(state.assets)) {
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

  /**
   * Rebuild saved-map or saved-scenario choices for the current area. Retain the open
   * source when possible, otherwise select the first available asset and update
   * editor.openSourceValue. Opening still requires an explicit command.
   */
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

  /**
   * Rebuild source choices for the selected scenario creation mode. Preserve a separate
   * selection per copy/duplicate mode and hide the source picker for blank creation.
   */
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

  /**
   * Show the active draft's saved/unsaved status from its current content and baseline.
   */
  function renderPersistenceStatus() {
    elements.persistenceStatus.textContent = authoringPersistenceMessage(
      state.editor.draft,
      state.editor.baseline,
    );
  }

  /**
   * Rebuild document/object buttons and count from the active draft. Mark current
   * selection and make agent rows draggable. An absent draft clears the list.
   */
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

  /**
   * Rebuild field-linked validation/error buttons. When no problems exist, show
   * execution-valid status or ask for host validation; this renderer validates nothing itself.
   */
  function renderProblems() {
    elements.problemList.replaceChildren();
    elements.problemCount.textContent = String(state.editor.problems.length);
    if (state.editor.problems.length === 0) {
      const row = document.createElement("li");
      row.className = "empty-copy";
      row.textContent = state.editor.validation?.execution_valid
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

  /**
   * Render the active draft with selected object, normalized camera, and host grid/catalog.
   * The Red Zone tint uses only the latest host validation of this draft; every edit
   * clears that validation, so no tint shows while a check is pending, and map
   * drafts, invalid drafts and depth 0 carry none. Store the resulting camera; an
   * absent draft clears the canvas. Missing required catalog numbers throw through
   * catalogNumber. No authored values are changed.
   */
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
      state.editor.validation?.red_zone ?? null,
    );
  }

  /**
   * Convert clientX/clientY browser pixels into map coordinates using the canvas bounds
   * and current camera. Return null without a draft or usable dimensions. No state is changed.
   *
   * @param {number} clientX @param {number} clientY
   */
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

  /**
   * Set busy/inert state and control availability from draft, selection, history, and
   * host validation. Only selected obstacles enable duplicate/delete/reorder controls.
   * Opening in Debug requires an execution-valid draft.
   */
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
    elements.deleteSavedButton.disabled =
      state.busy || !elements.savedDraftSelect.value;
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
    elements.openDebugButton.disabled =
      state.busy || !state.editor.draft || !state.editor.validation?.execution_valid;
  }

  /**
   * Refresh source choices, editor labels, persistence, objects, canvas, inspector,
   * problems, and availability from current local state. This performs DOM work only.
   */
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
  for (const selector of [
    elements.teamAController,
    elements.teamBController,
    elements.informationMode,
  ]) {
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
  elements.deleteSavedButton.addEventListener("click", async () => {
    if (state.busy || !elements.savedDraftSelect.value) {
      return;
    }
    const source = JSON.parse(elements.savedDraftSelect.value);
    const asset = state.assets.find(
      (/** @type {any} */ candidate) =>
        candidate.source_kind === "saved_draft" &&
        candidate.asset_kind === source.asset_kind &&
        candidate.asset_id === source.asset_id &&
        candidate.revision === source.revision,
    );
    if (!asset) {
      showLocalError("The selected saved asset is no longer available.");
      return;
    }
    if (!window.confirm(savedAssetDeletionPrompt(asset))) {
      return;
    }
    const editor = state.editor;
    const response = await send({ command_type: "delete", source }, editor);
    if (!response?.ok) {
      return;
    }
    state.assets = state.assets.filter(
      (/** @type {any} */ candidate) =>
        !(
          candidate.source_kind === "saved_draft" &&
          candidate.asset_kind === source.asset_kind &&
          candidate.asset_id === source.asset_id
        ),
    );
    const recovery = draftAfterSavedAssetDeletion(editor.draft, source);
    if (recovery !== editor.draft) {
      editor.draft = recovery;
      editor.baseline = authoringContentSnapshot(recovery);
      editor.openSourceValue = "";
    }
    renderAll();
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
    const assetId = promptAssetId("New asset ID", `${draft.asset_id}_copy`);
    if (assetId !== null) {
      await sendAndRefreshAssets({ command_type: "save_as", draft, asset_id: assetId });
    }
  });
  elements.validateButton.addEventListener("click", () => {
    if (!state.busy) {
      void validateDraft();
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
    const selected = selectedAuthoringObject(
      state.editor.draft,
      state.editor.selectedId,
    );
    if (
      edit.path.at(-1) === "object_id" &&
      (selected?.kind === "wall" || selected?.kind === "pillar")
    ) {
      try {
        const result = renameAuthoringObstacleId(
          state.editor.draft,
          selected.object_id,
          edit.value,
        );
        if (result.draft === state.editor.draft) {
          renderAll();
          return;
        }
        state.editor.selectedId = result.object_id;
        state.editor.problems = [];
        commit(result.draft);
      } catch (error) {
        showLocalError(
          error instanceof Error ? error.message : "Obstacle ID could not be changed.",
        );
        renderAll();
      }
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
      state.editor.validation = null;
      state.editor.problems = [];
      clearStaleSelection();
      renderAll();
      void validateDraft();
    }
  });
  elements.redoButton.addEventListener("click", () => {
    const nextContent = state.busy ? null : state.editor.future.pop();
    if (nextContent && state.editor.draft) {
      state.editor.past.push(authoringContentSnapshot(state.editor.draft));
      state.editor.draft = restoreAuthoringContent(state.editor.draft, nextContent);
      state.editor.validation = null;
      state.editor.problems = [];
      clearStaleSelection();
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
  /**
   * Finish only the matching active pointer gesture and release pointer capture.
   * A completed drag commits one undoable edit when not busy; panning only updates
   * the view. Both pointerup and pointercancel use this same behavior.
   *
   * @param {PointerEvent} event
   */
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
