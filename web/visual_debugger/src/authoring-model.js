/**
 * @file Edit browser-local map/scenario drafts before host validation and saving.
 * Most edits clone the draft and return a replacement; no helper writes files
 * or changes the simulator. Coordinates use map world units. Authored roster
 * team_local_slot values are 1–5, distinct from the public API's zero-based
 * team-local indices. The host owns physical/configuration validity.
 */
/**
 * Return whether value is a non-null object that is not an array. This
 * checks shape only, without prototype, field or schema validation.
 *
 * @param {unknown} value
 */
function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return structuredClone(value), preserving its supported data types and
 * removing shared nested references. The browser throws DataCloneError for
 * unsupported values such as functions. No transfer list or input mutation
 * is used. Callers supply plain draft data, not live DOM objects.
 *
 * @template T @param {T} value @returns {T}
 */
export function cloneAuthoringValue(value) {
  return structuredClone(value);
}

/**
 * Validate draft's recognized kind and return a deep copy of draft.content.
 * Exclude persisted draft identity/revision so undo cannot roll a later save
 * back to an older host identity. Invalid draft shape throws TypeError;
 * structured-clone errors propagate. No file or original object is changed.
 *
 * @param {any} draft
 */
export function authoringContentSnapshot(draft) {
  authoringKind(draft);
  return cloneAuthoringValue(draft.content);
}

/**
 * Return a cloned draft with a separate clone of content installed. Preserve
 * the draft's host identity/revision. Check its recognized kind and the
 * replacement map's basic arrays; invalid shape throws TypeError. This does
 * not fully validate geometry or save anything. Neither input is changed.
 *
 * @param {any} draft @param {unknown} content
 */
export function restoreAuthoringContent(draft, content) {
  authoringKind(draft);
  const next = cloneAuthoringValue(draft);
  next.content = cloneAuthoringValue(content);
  mapContent(next);
  return next;
}

/**
 * Return value as a finite number, using Number conversion for nonnumbers.
 * Throw TypeError with label if the converted number is nonfinite; native
 * conversion errors propagate. This does not constrain sign, range or units.
 *
 * @param {unknown} value @param {string} label
 */
function finiteNumber(value, label) {
  const number = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(number)) {
    throw new TypeError(`${label} must be finite.`);
  }
  return number;
}

/**
 * Return map or scenario for a recognized draft schema with object content.
 * Other values throw TypeError. The check recognizes dev-map-draft@1 and
 * both scenario versions: dev-scenario-draft@2, the version the host sends
 * for editing (it declares content.task.red_zone_depth), and the older
 * dev-scenario-draft@1, which has no depth. It does not validate their full
 * content or revision.
 *
 * @param {any} draft
 */
export function authoringKind(draft) {
  if (!isRecord(draft) || !isRecord(draft.content)) {
    throw new TypeError("Authoring draft must contain content.");
  }
  if (draft.schema === "dev-map-draft@1") {
    return "map";
  }
  if (
    draft.schema === "dev-scenario-draft@2" ||
    draft.schema === "dev-scenario-draft@1"
  ) {
    return "scenario";
  }
  throw new TypeError("Authoring draft schema is unsupported.");
}

/**
 * Return draft's map content object, retaining its original reference.
 * For scenarios, use content.embedded_map; for maps, use content itself.
 * Require recognized draft kind plus obstacle and spawn_pad arrays, otherwise
 * throw TypeError. Nested geometry is not checked and no copy is made.
 *
 * @param {any} draft @returns {any}
 */
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

/**
 * Return value on the nearest step-sized grid in world units. Both inputs
 * pass finiteNumber; step must stay positive even when bypass is true.
 * bypass defaults to false; true returns the unsnapped coordinate. Invalid
 * inputs throw TypeError/RangeError. Rounding uses JavaScript Math.round;
 * no map-bound or collision checks are performed.
 *
 * @param {number} value @param {number} step @param {boolean} bypass
 */
export function snapAuthoringCoordinate(value, step, bypass = false) {
  const coordinate = finiteNumber(value, "coordinate");
  const snapStep = finiteNumber(step, "snap step");
  if (snapStep <= 0) {
    throw new RangeError("snap step must be positive.");
  }
  return bypass ? coordinate : Math.round(coordinate / snapStep) * snapStep;
}

/**
 * List selectable draft objects in roster, obstacle, then spawn-pad order.
 *
 * draft must have valid basic map arrays; a scenario also needs roster and
 * agent_states arrays aligned by index. Include only roster rows within
 * their team's declared size. Return a frozen array of frozen view records
 * with object IDs, kind/label and world x/y. Nested roster/state/shape records
 * still reference draft content. Invalid basic shape throws TypeError;
 * complete row and physical validation belong to the host.
 *
 * @param {any} draft @returns {readonly any[]}
 */
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

/**
 * Return the first visible object in draft with object_id equal to objectId,
 * or null when not found. A null objectId returns null without inspecting the
 * draft. The projection uses authoringObjects and retains its nested source
 * references; draft validation errors propagate.
 *
 * @param {any} draft @param {string | null} objectId @returns {any}
 */
export function selectedAuthoringObject(draft, objectId) {
  if (objectId === null) {
    return null;
  }
  return (
    authoringObjects(draft).find((object) => object.object_id === objectId) ?? null
  );
}

/**
 * Return a deep-cloned draft with objectId moved to finite world x/y.
 * Search obstacles, spawn pads, then scenario agent state. Unknown IDs throw
 * RangeError; invalid map/coordinates throw TypeError. Do not snap, clamp or
 * check collisions. The input draft is unchanged; host validation later
 * decides whether the authored position is physically valid.
 *
 * @param {any} draft @param {string} objectId @param {number} x @param {number} y @returns {any}
 */
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
 * Snap x and y using step, then move objectId in a cloned draft.
 * bypass defaults to false and skips rounding only; step still must be
 * positive. Pointer drags and row drops share this path. Return the new draft
 * and propagate snap/object validation errors without editing the original.
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

/**
 * Return a cloned draft with one path assigned value. path is a nonempty
 * array of string/number keys. Each parent must resolve to an object/array;
 * a null parent is replaced by an empty object. Invalid paths throw TypeError.
 * The supplied value is assigned by reference, not cloned. This low-level
 * helper does not validate the draft schema, field meaning or physical rules.
 *
 * @param {any} draft @param {readonly (string | number)[]} path @param {unknown} value @returns {any}
 */
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

/**
 * Return a cloned scenario draft with team A or B resized to 1–5 agents.
 *
 * size must be an integer; catalog must contain host class_mechanics. Rows
 * beyond the size become inactive, dead and zeroed. Newly activated rows use
 * the class for their 1-based authored slot, the matching spawn pad and the
 * catalog maximum health. Existing active rows keep their authored values.
 * Invalid kind/catalog throws TypeError, invalid team/size RangeError, and
 * missing required mechanics/pads Error. This prepares a draft, not a fully
 * validated or saved scenario, and leaves the other input objects unchanged.
 *
 * @param {any} draft @param {"A" | "B"} team @param {number} size @param {Record<string, any>} catalog
 */
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

/**
 * Return a cloned scenario draft with objectId's alive flag set from alive.
 *
 * A false value also clears health, spawn shield, out-of-combat countdown
 * and fields whose names end in _duration. A true value does not restore
 * health or clear other timers; the author and host validation must make the
 * full start consistent. Non-scenarios throw TypeError; unknown agents throw
 * RangeError. No simulator death or respawn transition is executed.
 *
 * @param {any} draft @param {string} objectId @param {boolean} alive @returns {any}
 */
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

const OBJECT_ID_PATTERN = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$/u;

/**
 * Rename one obstacle in a cloned draft while retaining its geometry/order.
 *
 * requestedId is trimmed and must be 1–64 ASCII letters/digits/dot/underscore/
 * hyphen, beginning and ending with a letter/digit. Reject invalid names or
 * collisions with RangeError; nontext and non-obstacle oldId throw TypeError.
 * Return a frozen {draft, object_id} pair; an unchanged name reuses draft,
 * otherwise return its clone. Spawn-pad and roster identities are not renamed.
 *
 * @param {any} draft
 * @param {string} oldId
 * @param {unknown} requestedId
 */
export function renameAuthoringObstacleId(draft, oldId, requestedId) {
  if (typeof requestedId !== "string") {
    throw new TypeError("Obstacle ID must be text.");
  }
  const objectId = requestedId.trim();
  if (!OBJECT_ID_PATTERN.test(objectId)) {
    throw new RangeError(
      "Obstacle IDs must be 1–64 characters, start and end with a letter or number, and use only letters, numbers, periods, underscores, or hyphens.",
    );
  }

  const map = mapContent(draft);
  const obstacleIndex = map.obstacles.findIndex(
    (/** @type {any} */ obstacle) => obstacle.object_id === oldId,
  );
  if (obstacleIndex < 0) {
    throw new TypeError("Only obstacle IDs may be renamed.");
  }
  if (objectId === oldId) {
    return Object.freeze({ draft, object_id: objectId });
  }

  const occupied = new Set([
    ...map.obstacles.map((/** @type {any} */ obstacle) => obstacle.object_id),
    ...map.spawn_pads.map((/** @type {any} */ pad) => pad.object_id),
    ...(authoringKind(draft) === "scenario"
      ? draft.content.roster.map((/** @type {any} */ row) => row.object_id)
      : []),
  ]);
  if (occupied.has(objectId)) {
    throw new RangeError(`Authoring object ID ${objectId} is already in use.`);
  }

  const next = cloneAuthoringValue(draft);
  mapContent(next).obstacles[obstacleIndex].object_id = objectId;
  return Object.freeze({ draft: next, object_id: objectId });
}

/**
 * Return the first unused obstacle_N name starting at N=0. Check draft's
 * obstacle and spawn-pad IDs plus scenario roster IDs. This does not reserve
 * the name or mutate the draft; malformed draft errors come from mapContent.
 *
 * @param {any} draft
 */
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
  while (occupied.has(`obstacle_${ordinal}`)) {
    ordinal += 1;
  }
  return `obstacle_${ordinal}`;
}

/**
 * Append a default shape at the map center in a cloned draft.
 *
 * kind must be wall or pillar as supplied by the caller. A wall is 2-by-1
 * world units at zero rotation; the other branch uses radius 0.75. This helper
 * does not validate kind separately. maximumObstacles must be a positive
 * integer and exceed the current count, or throw RangeError. Return a frozen
 * {draft, object_id} pair using the first free obstacle_N ID. No geometry
 * validity check or save is performed.
 *
 * @param {any} draft @param {"wall" | "pillar"} kind @param {number} maximumObstacles
 */
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

/**
 * Copy one obstacle to a fresh ID in a cloned draft and offset both axes.
 *
 * objectId must identify an obstacle. maximumObstacles is a positive integer
 * and must leave room. offset must convert to a positive finite number in
 * world units. Invalid limits/offset throw RangeError or TypeError; an unknown
 * obstacle throws TypeError. Return a frozen {draft, object_id} pair. The copy
 * is appended, not inserted beside its source; no collision/bounds check occurs.
 *
 * @param {any} draft @param {string} objectId @param {number} maximumObstacles @param {number} offset
 */
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

/**
 * Remove objectId from a cloned draft's ordered obstacle array and return
 * the new draft. If it is not an obstacle, throw TypeError. Spawn pads and
 * agents are not deletable through this helper; the input stays unchanged.
 *
 * @param {any} draft @param {string} objectId @returns {any}
 */
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

/**
 * Move objectId one position in a cloned obstacle array. direction must be
 * -1 or 1; the caller owns that precondition. Return the clone unchanged when
 * the ID is absent or destination is out of range. Preserve every other
 * obstacle's relative order. This does not change geometry or save the draft.
 *
 * @param {any} draft @param {string} objectId @param {-1 | 1} direction @returns {any}
 */
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

/**
 * Return a frozen list of shallow-frozen valid problem records. problems
 * may be any input; a non-array returns an empty list. Keep rows with severity
 * error/warning and string stable_code, message and field_path. Other rows
 * are dropped. Extra fields are copied; nested values stay shared. This
 * filters display records, not physical validity or host error truth.
 *
 * @param {unknown} problems
 */
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
