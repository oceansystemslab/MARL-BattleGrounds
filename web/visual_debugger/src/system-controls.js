/**
 * @file Check the small, optional declared-System menu and pending-work envelope.
 * Names come from the launcher. This module never loads code or accepts URLs.
 * The service owns declaration membership, execution, cancellation and adoption.
 */

/** Return whether a value is a bounded System alias. @param {unknown} value */
export function isSystemController(value) {
  return typeof value === "string" && /^system:[a-z0-9_-]{1,64}$/u.test(value);
}

/** Return whether a controller needs SharedObs. @param {unknown} value */
export function requiresSharedObs(value) {
  return (
    (typeof value === "string" &&
      ["reactive_tdm", "scenario_5", "tdm_gamma"].includes(value)) ||
    isSystemController(value)
  );
}

/** Return whether either team may use this name.
 * @param {unknown} value
 * @returns {value is string}
 */
export function isTeamController(value) {
  return value === "manual" || value === "random_valid" || requiresSharedObs(value);
}

/**
 * Extend a live frame's exact keys only for present, checked optional fields.
 * @param {Iterable<string>} keys
 * @param {Record<string, any>} frame
 */
export function systemTransportKeys(keys, frame) {
  normalizeSystemControls(frame);
  return [
    ...keys,
    ...["system_choices", "system_operation"].filter((key) =>
      Object.hasOwn(frame, key),
    ),
  ].sort();
}

/**
 * Validate optional menu/status fields and return frozen copies, or no extra keys.
 * Unknown entry fields, duplicate names and malformed operation IDs are rejected.
 * @param {Record<string, any>} frame
 * @returns {Readonly<Record<string, any>>}
 */
export function normalizeSystemControls(frame) {
  /** @type {Record<string, any>} */
  const result = {};
  if (Object.hasOwn(frame, "system_choices")) {
    if (!Array.isArray(frame.system_choices))
      throw new TypeError("System choices must be a list.");
    const seen = new Set();
    result.system_choices = Object.freeze(
      frame.system_choices.map((choice) => {
        if (
          !choice ||
          typeof choice !== "object" ||
          Array.isArray(choice) ||
          Object.keys(choice).sort().join(",") !== "id,label" ||
          !isSystemController(choice.id) ||
          typeof choice.label !== "string" ||
          choice.label.length < 1 ||
          choice.label.length > 64 ||
          seen.has(choice.id)
        ) {
          throw new TypeError("System choice is invalid.");
        }
        seen.add(choice.id);
        return Object.freeze({ id: choice.id, label: choice.label });
      }),
    );
  }
  if (Object.hasOwn(frame, "system_operation")) {
    const operation = frame.system_operation;
    if (
      !operation ||
      typeof operation !== "object" ||
      Array.isArray(operation) ||
      Object.keys(operation).sort().join(",") !== "operation_id,state" ||
      typeof operation.operation_id !== "string" ||
      !/^[A-Za-z0-9_-]{1,128}$/u.test(operation.operation_id) ||
      !["loading", "thinking", "finishing", "cancelling", "failed"].includes(
        operation.state,
      )
    ) {
      throw new TypeError("System operation is invalid.");
    }
    result.system_operation = Object.freeze({ ...operation });
  }
  return Object.freeze(result);
}
