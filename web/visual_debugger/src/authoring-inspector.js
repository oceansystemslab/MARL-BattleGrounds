import {
  authoringKind,
  mapContent,
  selectedAuthoringObject,
} from "./authoring-model.js";
import { formatDisplayNumber } from "./display.js";

/** @typedef {Record<string, any>} JsonRecord */

/** @param {string} label @param {readonly (string | number)[] | null} path @param {unknown} value @param {Record<string, any>} [options] */
function field(label, path, value, options = {}) {
  return { label, path, value, ...options };
}

/** @param {unknown} value @param {boolean} readonly */
export function authoringFieldDisplayValue(value, readonly = false) {
  return readonly && typeof value === "number"
    ? formatDisplayNumber(value)
    : String(value ?? "");
}

/** @param {unknown} value */
export function humanizeAuthoringIdentifier(value) {
  const normalized = String(value ?? "")
    .replaceAll("_", " ")
    .trim();
  return normalized
    ? `${normalized.charAt(0).toUpperCase()}${normalized.slice(1)}`
    : "";
}

/** @param {HTMLElement} owner @param {Record<string, any>} descriptor */
function appendField(owner, descriptor) {
  const label = document.createElement("label");
  label.className = "authoring-field";
  const caption = document.createElement("span");
  caption.textContent = descriptor.label;
  label.append(caption);

  /** @type {HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement} */
  let input;
  if (descriptor.options) {
    input = document.createElement("select");
    for (const item of descriptor.options) {
      input.append(new Option(item.label, item.value));
    }
    input.value = String(descriptor.value);
  } else if (descriptor.type === "textarea") {
    input = document.createElement("textarea");
    input.rows = 3;
    input.value = String(descriptor.value ?? "");
  } else {
    input = document.createElement("input");
    input.type = descriptor.type ?? "text";
    if (input.type === "checkbox") {
      input.checked = Boolean(descriptor.value);
    } else {
      input.value = authoringFieldDisplayValue(
        descriptor.value,
        Boolean(descriptor.readonly),
      );
    }
    if (descriptor.min !== undefined) {
      input.min = String(descriptor.min);
    }
    if (descriptor.max !== undefined) {
      input.max = String(descriptor.max);
    }
    if (descriptor.step !== undefined) {
      input.step = descriptor.step;
    }
  }
  if (descriptor.path) {
    input.dataset.authoringPath = JSON.stringify(descriptor.path);
  }
  input.setAttribute("aria-describedby", "authoring-inspector-help");
  input.disabled = Boolean(descriptor.readonly);
  label.append(input);
  owner.append(label);
}

/** @param {string} encodedPath @param {string} fieldPath */
export function authoringPathMatchesProblem(encodedPath, fieldPath) {
  /** @type {unknown} */
  let decoded;
  try {
    decoded = JSON.parse(encodedPath);
  } catch {
    return false;
  }
  if (!Array.isArray(decoded) || typeof fieldPath !== "string") {
    return false;
  }
  const authored = decoded.map(String);
  if (authored[0] === "content") {
    authored.shift();
  }
  const problem = fieldPath.split(".").filter(Boolean);
  const variants = [authored];
  if (authored[0] === "embedded_map") {
    variants.push(authored.slice(1));
  }
  return variants.some(
    (candidate) =>
      problem.length > 0 && problem.every((part, index) => candidate[index] === part),
  );
}

/** @param {{querySelectorAll(selectors: string): Iterable<any>}} form @param {string} fieldPath */
export function focusAuthoringProblemField(form, fieldPath) {
  for (const input of form.querySelectorAll("[data-authoring-path]")) {
    if (authoringPathMatchesProblem(input.dataset.authoringPath ?? "", fieldPath)) {
      input.focus();
      input.scrollIntoView?.({ block: "nearest" });
      return true;
    }
  }
  return false;
}

/** @param {HTMLElement} form @param {string} legendText @param {readonly Record<string, any>[]} fields */
function appendGroup(form, legendText, fields) {
  const fieldset = document.createElement("fieldset");
  fieldset.className = "authoring-fieldset";
  const legend = document.createElement("legend");
  legend.textContent = legendText;
  fieldset.append(legend);
  for (const descriptor of fields) {
    appendField(fieldset, descriptor);
  }
  form.append(fieldset);
}

const AGENT_TIMER_FIELDS = [
  "ultimate_cooldown_remaining",
  "spawn_shield_duration_remaining",
  "steps_until_out_of_combat",
  "warrior_charge_slow_duration",
  "hunter_basic_slow_duration",
  "rogue_poison_slow_duration",
  "warrior_charge_stun_duration",
  "hunter_trap_stun_duration",
  "rogue_poison_stun_duration",
  "rogue_poison_anti_heal_duration",
  "mage_burst_duration",
  "priest_blessing_of_freedom_duration",
];

/** @param {JsonRecord | null} catalog @param {string} className */
export function authoringClassMechanics(catalog, className) {
  if (!Array.isArray(catalog?.class_mechanics)) {
    return null;
  }
  const normalized = className.toLowerCase();
  return (
    catalog.class_mechanics.find(
      (/** @type {any} */ mechanics) =>
        typeof mechanics.class_name === "string" &&
        mechanics.class_name.toLowerCase() === normalized,
    ) ?? null
  );
}

/** @param {JsonRecord | null} catalog @param {unknown} classId */
export function authoringClassStatusMechanics(catalog, classId) {
  if (!Number.isInteger(classId) || !Array.isArray(catalog?.status_channels)) {
    return Object.freeze([]);
  }
  return Object.freeze(
    catalog.status_channels.filter(
      (/** @type {any} */ status) => status.source_class_id === classId,
    ),
  );
}

/** @param {JsonRecord | null} catalog @param {unknown} classId */
export function authoringClassAuraMechanics(catalog, classId) {
  if (!Number.isInteger(classId) || !Array.isArray(catalog?.aura_mechanics)) {
    return Object.freeze([]);
  }
  return Object.freeze(
    catalog.aura_mechanics.filter(
      (/** @type {any} */ aura) => aura.emitter_class_id === classId,
    ),
  );
}

/** @param {JsonRecord | null} catalog */
function classOptions(catalog) {
  if (!Array.isArray(catalog?.class_mechanics)) {
    return [];
  }
  return catalog.class_mechanics.map((/** @type {any} */ mechanics) => ({
    value: mechanics.class_name.toLowerCase(),
    label: mechanics.class_name,
  }));
}

/** @param {HTMLElement} form @param {JsonRecord} draft */
function renderDocument(form, draft) {
  const kind = authoringKind(draft);
  const content = draft.content;
  const map = mapContent(draft);
  const mapPath = kind === "map" ? ["content"] : ["content", "embedded_map"];
  const documentIdentityFields = [
    field("Name", ["content", "name"], content.name),
    field("Description", ["content", "description"], content.description, {
      type: "textarea",
    }),
  ];
  if (kind === "map") {
    appendGroup(form, "Map", [
      ...documentIdentityFields,
      field("Width", [...mapPath, "width"], map.width, {
        type: "number",
        min: 0,
        step: "any",
      }),
      field("Height", [...mapPath, "height"], map.height, {
        type: "number",
        min: 0,
        step: "any",
      }),
    ]);
    return;
  }
  appendGroup(form, "Scenario", [
    ...documentIdentityFields,
    field("Notes", ["content", "notes"], content.notes, { type: "textarea" }),
  ]);
  appendGroup(form, "Embedded map", [
    field("Map name", [...mapPath, "name"], map.name),
    field("Map description", [...mapPath, "description"], map.description, {
      type: "textarea",
    }),
    field("Width", [...mapPath, "width"], map.width, {
      type: "number",
      min: 0,
      step: "any",
    }),
    field("Height", [...mapPath, "height"], map.height, {
      type: "number",
      min: 0,
      step: "any",
    }),
  ]);
  appendGroup(form, "Roster", [
    field("Team A size", ["content", "team_a_size"], content.team_a_size, {
      type: "number",
      min: 1,
      max: 5,
      step: "1",
    }),
    field("Team B size", ["content", "team_b_size"], content.team_b_size, {
      type: "number",
      min: 1,
      max: 5,
      step: "1",
    }),
  ]);
  appendGroup(form, "TDM episode", [
    field(
      "Score threshold K",
      ["content", "task", "score_threshold"],
      content.task.score_threshold,
      { type: "number", min: 1, step: "1" },
    ),
    field("Max steps", ["content", "episode", "max_steps"], content.episode.max_steps, {
      type: "number",
      min: 1,
      step: "1",
    }),
    field(
      "Shield duration",
      ["content", "episode", "spawn_shield_duration_steps"],
      content.episode.spawn_shield_duration_steps,
      { type: "number", min: 0, step: "1" },
    ),
    field(
      "Shield speed",
      ["content", "episode", "spawn_shield_movement_speed"],
      content.episode.spawn_shield_movement_speed,
      { type: "number", min: 0, step: "any" },
    ),
    field(
      "Team A respawn period",
      ["content", "episode", "team_a_respawn_wave_period_steps"],
      content.episode.team_a_respawn_wave_period_steps,
      { type: "number", min: 1, step: "1" },
    ),
    field(
      "Team B respawn period",
      ["content", "episode", "team_b_respawn_wave_period_steps"],
      content.episode.team_b_respawn_wave_period_steps,
      { type: "number", min: 1, step: "1" },
    ),
  ]);
  appendGroup(form, "Current state", [
    field(
      "Step count",
      ["content", "global_state", "step_count"],
      content.global_state.step_count,
      { type: "number", min: 0, step: "1" },
    ),
    field(
      "Team A score",
      ["content", "global_state", "team_a_score"],
      content.global_state.team_a_score,
      { type: "number", min: 0, step: "1" },
    ),
    field(
      "Team B score",
      ["content", "global_state", "team_b_score"],
      content.global_state.team_b_score,
      { type: "number", min: 0, step: "1" },
    ),
    field(
      "Team A countdown",
      ["content", "global_state", "team_a_respawn_countdown"],
      content.global_state.team_a_respawn_countdown,
      { type: "number", min: 0, step: "1" },
    ),
    field(
      "Team B countdown",
      ["content", "global_state", "team_b_respawn_countdown"],
      content.global_state.team_b_respawn_countdown,
      { type: "number", min: 0, step: "1" },
    ),
  ]);
}

/** @param {HTMLElement} form @param {JsonRecord} draft @param {JsonRecord} object @param {JsonRecord | null} catalog @param {JsonRecord | null} validation */
function renderObject(form, draft, object, catalog, validation) {
  const map = mapContent(draft);
  const mapPath =
    authoringKind(draft) === "map" ? ["content"] : ["content", "embedded_map"];
  if (object.kind === "wall" || object.kind === "pillar") {
    const index = map.obstacles.findIndex(
      (/** @type {any} */ candidate) => candidate.object_id === object.object_id,
    );
    const path = [...mapPath, "obstacles", index];
    const fields = [
      field("Object ID", null, object.object_id, { readonly: true }),
      field("Center X", [...path, "center_x"], object.x, {
        type: "number",
        step: "any",
      }),
      field("Center Y", [...path, "center_y"], object.y, {
        type: "number",
        step: "any",
      }),
    ];
    if (object.kind === "wall") {
      fields.push(
        field("Width", [...path, "width"], object.obstacle.width, {
          type: "number",
          min: 0,
          step: "any",
        }),
        field("Height", [...path, "height"], object.obstacle.height, {
          type: "number",
          min: 0,
          step: "any",
        }),
        field(
          "Rotation (degrees)",
          [...path, "rotation_degrees"],
          object.obstacle.rotation_degrees,
          { type: "number", step: "any" },
        ),
      );
    } else {
      fields.push(
        field("Radius", [...path, "radius"], object.obstacle.radius, {
          type: "number",
          min: 0,
          step: "any",
        }),
      );
    }
    appendGroup(form, object.kind === "wall" ? "Wall" : "Pillar", fields);
    return;
  }
  if (object.kind === "spawn_pad") {
    const index = map.spawn_pads.findIndex(
      (/** @type {any} */ candidate) => candidate.object_id === object.object_id,
    );
    const path = [...mapPath, "spawn_pads", index, "position"];
    appendGroup(form, "Spawn pad", [
      field("Identity", null, `${object.pad.team}${object.pad.team_local_slot}`, {
        readonly: true,
      }),
      field("Center X", [...path, "x"], object.x, { type: "number", step: "any" }),
      field("Center Y", [...path, "y"], object.y, { type: "number", step: "any" }),
    ]);
    return;
  }
  const rosterIndex = draft.content.roster.findIndex(
    (/** @type {any} */ candidate) => candidate.object_id === object.object_id,
  );
  const stateIndex = draft.content.agent_states.findIndex(
    (/** @type {any} */ candidate) => candidate.object_id === object.object_id,
  );
  appendGroup(form, "Agent identity", [
    field("Team", null, object.roster.team, { readonly: true }),
    field("Team-local slot", null, object.roster.team_local_slot, { readonly: true }),
    field("Global slot", null, object.roster.global_slot, { readonly: true }),
    field(
      "Class",
      ["content", "roster", rosterIndex, "class_name"],
      object.roster.class_name,
      { options: classOptions(catalog) },
    ),
  ]);
  const statePath = ["content", "agent_states", stateIndex];
  const fields = [
    field("Center X", [...statePath, "position", "x"], object.x, {
      type: "number",
      step: "any",
    }),
    field("Center Y", [...statePath, "position", "y"], object.y, {
      type: "number",
      step: "any",
    }),
    field("Alive", [...statePath, "alive"], object.state.alive, { type: "checkbox" }),
    field(
      "Current health",
      [...statePath, "current_health"],
      object.state.current_health,
      { type: "number", min: 0, step: "any" },
    ),
    ...AGENT_TIMER_FIELDS.map((key) =>
      field(humanizeAuthoringIdentifier(key), [...statePath, key], object.state[key], {
        type: "number",
        min: 0,
        step: "1",
      }),
    ),
  ];
  appendGroup(form, "Initial state", fields);
  const mechanics = authoringClassMechanics(catalog, object.roster.class_name);
  if (mechanics) {
    const effectiveSpeed =
      validation?.effective_movement_speeds?.[object.roster.global_slot];
    const mechanicFields = [
      ["Maximum health", "maximum_health"],
      ["Body radius", "body_radius"],
      ["Base movement speed", "base_movement_speed"],
      ["Observation radius", "observation_radius"],
      ["Basic target mode", "basic_target_mode"],
      ["Basic interaction radius", "basic_interaction_radius"],
      ["Basic raw damage", "basic_raw_damage"],
      ["Basic raw healing", "basic_raw_healing"],
      ["Ultimate target mode", "ultimate_target_mode"],
      ["Ultimate interaction radius", "ultimate_interaction_radius"],
      ["Ultimate cooldown maximum", "ultimate_cooldown_steps"],
      ["Ultimate raw damage", "ultimate_raw_damage"],
      ["Ultimate raw healing", "ultimate_raw_healing"],
      ["Out-of-combat delay", "out_of_combat_delay_steps"],
      [
        "Recovery fraction per step",
        "out_of_combat_health_regeneration_fraction_per_step",
      ],
    ].map(([label, key]) =>
      field(
        label,
        null,
        typeof mechanics[key] === "string"
          ? humanizeAuthoringIdentifier(mechanics[key])
          : mechanics[key],
        { readonly: true },
      ),
    );
    mechanicFields.splice(
      3,
      0,
      field(
        "Current effective speed",
        null,
        Number.isFinite(effectiveSpeed) ? effectiveSpeed : "Validate to derive",
        { readonly: true },
      ),
    );
    appendGroup(form, "Class mechanics", mechanicFields);
    const statusMechanics = authoringClassStatusMechanics(catalog, mechanics.class_id);
    if (statusMechanics.length > 0) {
      appendGroup(
        form,
        "Status mechanics",
        statusMechanics.flatMap((status) => [
          field(
            `${humanizeAuthoringIdentifier(status.status_id)} duration`,
            null,
            status.duration_steps,
            { readonly: true },
          ),
          field(
            `${humanizeAuthoringIdentifier(status.status_id)} magnitude`,
            null,
            status.magnitude === null
              ? humanizeAuthoringIdentifier(status.magnitude_kind)
              : `${humanizeAuthoringIdentifier(status.magnitude_kind)}: ${formatDisplayNumber(status.magnitude)}`,
            { readonly: true },
          ),
        ]),
      );
    }
    const auraMechanics = authoringClassAuraMechanics(catalog, mechanics.class_id);
    if (auraMechanics.length > 0) {
      appendGroup(
        form,
        "Aura mechanics",
        auraMechanics.flatMap((aura) => [
          field(
            `${humanizeAuthoringIdentifier(aura.aura_id)} radius`,
            null,
            aura.radius,
            {
              readonly: true,
            },
          ),
          field(
            `${humanizeAuthoringIdentifier(aura.aura_id)} multiplier`,
            null,
            aura.per_emitter_multiplier,
            { readonly: true },
          ),
          field(
            `${humanizeAuthoringIdentifier(aura.aura_id)} clamp`,
            null,
            `${humanizeAuthoringIdentifier(aura.stacking_rule)}; ${humanizeAuthoringIdentifier(aura.clamp_kind)} ${formatDisplayNumber(aura.clamp_value)}`,
            { readonly: true },
          ),
        ]),
      );
    }
  }
  if (catalog) {
    appendGroup(form, "Product constants", [
      field("Movement scale", null, catalog.canonical_product_movement_scale, {
        readonly: true,
      }),
    ]);
  }
}

/** @param {HTMLElement} form @param {JsonRecord | null} draft @param {string | null} selectedId @param {JsonRecord | null} catalog @param {JsonRecord | null} validation */
export function renderAuthoringInspector(
  form,
  draft,
  selectedId,
  catalog = null,
  validation = null,
) {
  form.replaceChildren();
  if (draft === null) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = "Open or create a draft.";
    form.append(empty);
    return;
  }
  const selected = selectedAuthoringObject(draft, selectedId);
  if (selected === null) {
    renderDocument(form, draft);
  } else {
    renderObject(form, draft, selected, catalog, validation);
  }
}

/** @param {EventTarget | null} target */
export function readAuthoringFieldEdit(target) {
  if (
    !(target instanceof HTMLInputElement) &&
    !(target instanceof HTMLSelectElement) &&
    !(target instanceof HTMLTextAreaElement)
  ) {
    return null;
  }
  const encodedPath = target.dataset.authoringPath;
  if (!encodedPath) {
    return null;
  }
  const path = JSON.parse(encodedPath);
  /** @type {any} */
  let value = target.value;
  if (target instanceof HTMLInputElement && target.type === "checkbox") {
    value = target.checked;
  } else if (target instanceof HTMLInputElement && target.type === "number") {
    value = target.value === "" ? null : Number(target.value);
  }
  return { path, value };
}
