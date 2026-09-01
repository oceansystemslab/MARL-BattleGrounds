/** @param {unknown} value */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** @template T @param {T} value @returns {T} */
export function cloneAuthoringValue(value) {
  return structuredClone(value);
}

/**
 * History records only editable content. Persisted draft identity belongs to
 * the host and must survive undo/redo after a save advances the revision.
 *
 * @param {any} draft
 */
export function authoringContentSnapshot(draft) {
  authoringKind(draft);
  return cloneAuthoringValue(draft.content);
}

/** @param {any} draft @param {unknown} content */
export function restoreAuthoringContent(draft, content) {
  authoringKind(draft);
  const next = cloneAuthoringValue(draft);
  next.content = cloneAuthoringValue(content);
  mapContent(next);
  return next;
}

/** @param {unknown} value @param {string} label */
function finiteNumber(value, label) {
  const number = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(number)) {
    throw new TypeError(`${label} must be finite.`);
  }
  return number;
}

/** @param {any} draft */
export function authoringKind(draft) {
  if (!isRecord(draft) || !isRecord(draft.content)) {
    throw new TypeError("Authoring draft must contain content.");
  }
  if (draft.schema === "dev-map-draft@1") {
    return "map";
  }
  if (draft.schema === "dev-scenario-draft@1") {
    return "scenario";
  }
  throw new TypeError("Authoring draft schema is unsupported.");
}

/** @param {any} draft @returns {any} */
export function mapContent(draft) {
  const kind = authoringKind(draft);
  const content = /** @type {Record<string, any>} */ (draft.content);
  const map = kind === "map" ? content : content.embedded_map;
  if (
    !isRecord(map) ||
    !Array.isArray(map.obstacles) ||
    !Array.isArray(map.spawn_pads)
  ) {
    throw new TypeError("Authoring draft has invalid map content.");
  }
  return map;
}

/** @param {number} value @param {number} step @param {boolean} bypass */
export function snapAuthoringCoordinate(value, step, bypass = false) {
  const coordinate = finiteNumber(value, "coordinate");
  const snapStep = finiteNumber(step, "snap step");
  if (snapStep <= 0) {
    throw new RangeError("snap step must be positive.");
  }
  return bypass ? coordinate : Math.round(coordinate / snapStep) * snapStep;
}

/** @param {any} draft @returns {readonly any[]} */
export function authoringObjects(draft) {
  const map = mapContent(draft);
  const objects = [];
  if (authoringKind(draft) === "scenario") {
    const content = /** @type {Record<string, any>} */ (draft.content);
    if (!Array.isArray(content.roster) || !Array.isArray(content.agent_states)) {
      throw new TypeError("Scenario draft has invalid fixed-slot rows.");
    }
    for (let globalSlot = 0; globalSlot < content.roster.length; globalSlot += 1) {
      const roster = content.roster[globalSlot];
      const state = content.agent_states[globalSlot];
      const teamSize = roster.team === "A" ? content.team_a_size : content.team_b_size;
      const active = roster.team_local_slot <= teamSize;
      if (!active) {
        continue;
      }
      objects.push(
        Object.freeze({
          object_id: roster.object_id,
          kind: "agent",
          label: `${roster.team}${roster.team_local_slot} · ${roster.class_name}`,
          team: roster.team,
          global_slot: roster.global_slot,
          x: state.position.x,
          y: state.position.y,
          roster,
          state,
        }),
      );
    }
  }
  for (const obstacle of map.obstacles) {
    objects.push(
      Object.freeze({
        object_id: obstacle.object_id,
        kind: obstacle.kind,
        label: obstacle.object_id,
        x: obstacle.center_x,
        y: obstacle.center_y,
        obstacle,
      }),
    );
  }
  for (const pad of map.spawn_pads) {
    objects.push(
      Object.freeze({
        object_id: pad.object_id,
        kind: "spawn_pad",
        label: `${pad.team}${pad.team_local_slot} spawn pad`,
        team: pad.team,
        x: pad.position.x,
        y: pad.position.y,
        pad,
      }),
    );
  }
  return Object.freeze(objects);
}

/** @param {any} draft @param {string | null} objectId @returns {any} */
export function selectedAuthoringObject(draft, objectId) {
  if (objectId === null) {
    return null;
  }
  return (
    authoringObjects(draft).find((object) => object.object_id === objectId) ?? null
  );
}

/** @param {any} draft @param {string} objectId @param {number} x @param {number} y @returns {any} */
export function moveAuthoringObject(draft, objectId, x, y) {
  const next = cloneAuthoringValue(draft);
  const map = mapContent(next);
  const obstacle = map.obstacles.find(
    (/** @type {any} */ candidate) => candidate.object_id === objectId,
  );
  if (obstacle) {
    obstacle.center_x = finiteNumber(x, "x");
    obstacle.center_y = finiteNumber(y, "y");
    return next;
  }
  const pad = map.spawn_pads.find(
    (/** @type {any} */ candidate) => candidate.object_id === objectId,
  );
  if (pad) {
    pad.position.x = finiteNumber(x, "x");
    pad.position.y = finiteNumber(y, "y");
    return next;
  }
  if (authoringKind(next) === "scenario") {
    const state = next.content.agent_states.find(
      (/** @type {any} */ candidate) => candidate.object_id === objectId,
    );
    if (state) {
      state.position.x = finiteNumber(x, "x");
      state.position.y = finiteNumber(y, "y");
      return next;
    }
  }
  throw new RangeError(`Unknown authoring object ${objectId}.`);
}

/**
 * Apply the one authoring snap rule before moving an existing object. Pointer
 * drags and native roster-row drops deliberately share this path.
 *
 * @param {any} draft
 * @param {string} objectId
 * @param {number} x
 * @param {number} y
 * @param {number} step
 * @param {boolean} bypass
 */
export function moveAuthoringObjectWithSnap(
  draft,
  objectId,
  x,
  y,
  step,
  bypass = false,
) {
  return moveAuthoringObject(
    draft,
    objectId,
    snapAuthoringCoordinate(x, step, bypass),
    snapAuthoringCoordinate(y, step, bypass),
  );
}

/** @param {any} draft @param {readonly (string | number)[]} path @param {unknown} value @returns {any} */
export function setAuthoringField(draft, path, value) {
  if (!Array.isArray(path) || path.length === 0) {
    throw new TypeError("Authoring field path must be nonempty.");
  }
  const next = cloneAuthoringValue(draft);
  /** @type {any} */
  let owner = next;
  for (const key of path.slice(0, -1)) {
    if (!isRecord(owner) && !Array.isArray(owner)) {
      throw new TypeError("Authoring field path does not resolve.");
    }
    if (owner[key] === null) {
      owner[key] = {};
    }
    owner = owner[key];
  }
  const finalKey = path.at(-1);
  if ((!isRecord(owner) && !Array.isArray(owner)) || finalKey === undefined) {
    throw new TypeError("Authoring field path does not resolve.");
  }
  owner[finalKey] = value;
  return next;
}

/** @param {any} draft @param {"A" | "B"} team @param {number} size @param {Record<string, any>} catalog */
export function setScenarioTeamSize(draft, team, size, catalog) {
  if (authoringKind(draft) !== "scenario") {
    throw new TypeError("Only scenarios contain team rosters.");
  }
  if (!Number.isInteger(size) || size < 1 || size > 5) {
    throw new RangeError("Team size must be an integer from 1 through 5.");
  }
  if (team !== "A" && team !== "B") {
    throw new RangeError("Team must be A or B.");
  }
  if (!isRecord(catalog) || !Array.isArray(catalog.class_mechanics)) {
    throw new TypeError("The host mechanics catalog is unavailable.");
  }

  const next = cloneAuthoringValue(draft);
  const content = next.content;
  content[team === "A" ? "team_a_size" : "team_b_size"] = size;
  const defaultClasses = ["mage", "warrior", "hunter", "rogue", "priest"];
  for (let index = 0; index < content.roster.length; index += 1) {
    const roster = content.roster[index];
    if (roster.team !== team) {
      continue;
    }
    const state = content.agent_states[index];
    if (roster.team_local_slot > size) {
      roster.class_name = "not_applicable";
      state.position = { x: 0, y: 0 };
      state.alive = false;
      for (const key of Object.keys(state)) {
        if (typeof state[key] === "number") {
          state[key] = 0;
        }
      }
      continue;
    }
    if (roster.class_name !== "not_applicable") {
      continue;
    }
    const className = defaultClasses[roster.team_local_slot - 1];
    const mechanics = catalog.class_mechanics?.find(
      (/** @type {any} */ row) => row.class_name?.toLowerCase() === className,
    );
    const pad = content.embedded_map.spawn_pads.find(
      (/** @type {any} */ row) =>
        row.team === team && row.team_local_slot === roster.team_local_slot,
    );
    if (!mechanics || !pad) {
      throw new Error("The host catalog or embedded spawn pads are incomplete.");
    }
    roster.class_name = className;
    state.position = { ...pad.position };
    state.alive = true;
    state.current_health = mechanics.maximum_health;
  }
  return next;
}

/** @param {any} draft @param {string} objectId @param {boolean} alive @returns {any} */
export function setAgentAlive(draft, objectId, alive) {
  if (authoringKind(draft) !== "scenario") {
    throw new TypeError("Only scenarios contain agent lifecycle state.");
  }
  const next = cloneAuthoringValue(draft);
  const state = next.content.agent_states.find(
    (/** @type {any} */ candidate) => candidate.object_id === objectId,
  );
  if (!state) {
    throw new RangeError(`Unknown scenario agent ${objectId}.`);
  }
  state.alive = Boolean(alive);
  if (!alive) {
    state.current_health = 0;
    state.spawn_shield_duration_remaining = 0;
    state.steps_until_out_of_combat = 0;
    for (const key of Object.keys(state)) {
      if (key.endsWith("_duration") && key !== "ultimate_cooldown_remaining") {
        state[key] = 0;
      }
    }
  }
  return next;
}

/** @param {any} draft */
function nextObstacleObjectId(draft) {
  const map = mapContent(draft);
  const occupied = new Set([
    ...map.obstacles.map((/** @type {any} */ obstacle) => obstacle.object_id),
    ...map.spawn_pads.map((/** @type {any} */ pad) => pad.object_id),
    ...(authoringKind(draft) === "scenario"
      ? draft.content.roster.map((/** @type {any} */ row) => row.object_id)
      : []),
  ]);
  let ordinal = 0;
  while (occupied.has(`obstacle-${ordinal}`)) {
    ordinal += 1;
  }
  return `obstacle-${ordinal}`;
}

/** @param {any} draft @param {"wall" | "pillar"} kind @param {number} maximumObstacles */
export function addAuthoringObstacle(draft, kind, maximumObstacles) {
  const next = cloneAuthoringValue(draft);
  const map = mapContent(next);
  if (!Number.isInteger(maximumObstacles) || maximumObstacles <= 0) {
    throw new RangeError("maximum obstacles must be a positive integer.");
  }
  if (map.obstacles.length >= maximumObstacles) {
    throw new RangeError(`Maps support at most ${maximumObstacles} obstacles.`);
  }
  const objectId = nextObstacleObjectId(next);
  const common = {
    kind,
    object_id: objectId,
    center_x: map.width / 2,
    center_y: map.height / 2,
  };
  map.obstacles.push(
    kind === "wall"
      ? { ...common, width: 2, height: 1, rotation_degrees: 0 }
      : { ...common, radius: 0.75 },
  );
  return Object.freeze({ draft: next, object_id: objectId });
}

/** @param {any} draft @param {string} objectId @param {number} maximumObstacles @param {number} offset */
export function duplicateAuthoringObstacle(draft, objectId, maximumObstacles, offset) {
  const next = cloneAuthoringValue(draft);
  const map = mapContent(next);
  if (!Number.isInteger(maximumObstacles) || maximumObstacles <= 0) {
    throw new RangeError("maximum obstacles must be a positive integer.");
  }
  const duplicateOffset = finiteNumber(offset, "duplicate offset");
  if (duplicateOffset <= 0) {
    throw new RangeError("duplicate offset must be positive.");
  }
  if (map.obstacles.length >= maximumObstacles) {
    throw new RangeError(`Maps support at most ${maximumObstacles} obstacles.`);
  }
  const index = map.obstacles.findIndex(
    (/** @type {any} */ obstacle) => obstacle.object_id === objectId,
  );
  if (index < 0) {
    throw new TypeError("Only obstacles may be duplicated.");
  }
  const original = map.obstacles[index];
  const duplicateId = nextObstacleObjectId(next);
  const duplicate = cloneAuthoringValue(original);
  duplicate.object_id = duplicateId;
  duplicate.center_x += duplicateOffset;
  duplicate.center_y += duplicateOffset;
  map.obstacles.push(duplicate);
  return Object.freeze({ draft: next, object_id: duplicateId });
}

/** @param {any} draft @param {string} objectId @returns {any} */
export function deleteAuthoringObstacle(draft, objectId) {
  const next = cloneAuthoringValue(draft);
  const map = mapContent(next);
  const index = map.obstacles.findIndex(
    (/** @type {any} */ obstacle) => obstacle.object_id === objectId,
  );
  if (index < 0) {
    throw new TypeError("Spawn pads and agents cannot be deleted.");
  }
  map.obstacles.splice(index, 1);
  return next;
}

/** @param {any} draft @param {string} objectId @param {-1 | 1} direction @returns {any} */
export function reorderAuthoringObstacle(draft, objectId, direction) {
  const next = cloneAuthoringValue(draft);
  const obstacles = mapContent(next).obstacles;
  const index = obstacles.findIndex(
    (/** @type {any} */ obstacle) => obstacle.object_id === objectId,
  );
  const destination = index + direction;
  if (index < 0 || destination < 0 || destination >= obstacles.length) {
    return next;
  }
  const [obstacle] = obstacles.splice(index, 1);
  obstacles.splice(destination, 0, obstacle);
  return next;
}

/** @param {unknown} problems */
export function normalizeAuthoringProblems(problems) {
  if (!Array.isArray(problems)) {
    return Object.freeze([]);
  }
  return Object.freeze(
    problems
      .filter(
        (problem) =>
          isRecord(problem) &&
          (problem.severity === "error" || problem.severity === "warning") &&
          typeof problem.stable_code === "string" &&
          typeof problem.message === "string" &&
          typeof problem.field_path === "string",
      )
      .map((problem) => Object.freeze({ ...problem })),
  );
}
