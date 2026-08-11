import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  authorizationContextKey,
  buildChoreographyPlan,
  CHOREOGRAPHY_PHASES,
  eventFingerprint,
  isSubmissionCommand,
  transitionEpochKey,
} from "../src/choreography-plan.js";
import {
  loadRendererFixture,
  syntheticDebuggerFrame,
} from "../e2e/support/renderer-fixture.js";

/**
 * @typedef {{
 *   worldToScreen: (point: readonly [number, number] | {x: number, y: number}) =>
 *     {x: number, y: number},
 *   worldLengthToScreen: (length: number) => number,
 *   viewportBounds: {
 *     left: number,
 *     top: number,
 *     right: number,
 *     bottom: number,
 *     width: number,
 *     height: number,
 *   },
 *   protectedRects: ReadonlyArray<{
 *     left: number,
 *     top: number,
 *     right: number,
 *     bottom: number,
 *     width: number,
 *     height: number,
 *   }>,
 * }} ProjectionSurface
 * @typedef {{
 *   grammar: Record<string, any>,
 *   routes: Record<string, any>,
 *   crowded: Record<string, any>,
 *   pov: Record<string, any>,
 * }} CanonicalFrames
 */

/** @type {ProjectionSurface} */
const surface = {
  worldToScreen: (point) => {
    const x = "x" in point ? point.x : Number(point[0]);
    const y = "y" in point ? point.y : Number(point[1]);
    return { x: x * 10, y: 120 - y * 10 };
  },
  worldLengthToScreen: (length) => length * 10,
  viewportBounds: {
    left: 0,
    top: 0,
    right: 200,
    bottom: 120,
    width: 200,
    height: 120,
  },
  protectedRects: [],
};

/** @type {Promise<CanonicalFrames> | undefined} */
let canonicalFramesPromise;

/** @returns {Promise<CanonicalFrames>} */
async function canonicalFrames() {
  if (!canonicalFramesPromise) {
    canonicalFramesPromise = (async () => {
      const grammar = syntheticDebuggerFrame(
        await loadRendererFixture("canonical_event_vocabulary"),
      );
      const routes = syntheticDebuggerFrame(
        await loadRendererFixture("route_collision"),
      );
      const crowded = syntheticDebuggerFrame(
        await loadRendererFixture("crowded_teamfight"),
      );
      const pov = syntheticDebuggerFrame(await loadRendererFixture("pov_redaction"));
      return { grammar, routes, crowded, pov };
    })();
  }
  return canonicalFramesPromise;
}

test("transition identity excludes revision while authorization tracks POV actor", async () => {
  const { grammar, pov } = await canonicalFrames();
  const revised = { ...grammar, revision: grammar.revision + 1 };
  assert.equal(transitionEpochKey(revised), transitionEpochKey(grammar));
  assert.equal(eventFingerprint(revised), eventFingerprint(grammar));
  assert.deepEqual(JSON.parse(authorizationContextKey(grammar) ?? "null"), [
    grammar.session_id,
    grammar.run_generation,
    "researcher",
    null,
  ]);
  assert.deepEqual(JSON.parse(authorizationContextKey(pov) ?? "null"), [
    pov.session_id,
    pov.run_generation,
    "agent_pov",
    pov.scene.self_actor.global_slot,
  ]);
  assert.notEqual(authorizationContextKey(grammar), authorizationContextKey(pov));
});

test("submission classification is presentation gating, not legality", () => {
  assert.equal(isSubmissionCommand({ command_type: "keyboard", key: "Enter" }), true);
  assert.equal(isSubmissionCommand({ command_type: "keyboard", key: "N" }), true);
  assert.equal(
    isSubmissionCommand({ command_type: "keyboard", key: "ArrowLeft" }),
    false,
  );
  assert.equal(
    isSubmissionCommand({ command_type: "actor_pov_target_action", target_action: 1 }),
    false,
  );
});

test("canonical V2 displacement and rejection selectors own transient color", async () => {
  const css = await readFile(new URL("../styles.css", import.meta.url), "utf8");
  assert.match(css, /data-event-type="charge_phase_displacement"/u);
  assert.match(css, /data-event-type="action_rejected"/u);
});

test("canonical fixture gives all 21 V2 events one exact ordered disposition", async () => {
  const { grammar } = await canonicalFrames();
  const plan = buildChoreographyPlan(grammar, surface);
  assert.ok(plan);
  assert.equal(plan.transitionId, grammar.incoming_transition_id);
  assert.equal(plan.events.length, 21);
  const sourceEvents = /** @type {Record<string, any>[]} */ (
    grammar.event_batch.events
  );
  assert.deepEqual(
    plan.events.map((event) => event.eventId),
    sourceEvents.map((event) => event.event_id),
  );
  assert.deepEqual(
    plan.events.map((event) => event.eventType),
    sourceEvents.map((event) => event.event_type),
  );
  assert.equal(
    plan.events.some((event) => event.kind === "unknown"),
    false,
  );
});

test("canonical V2 phases are non-overlapping in causal order", async () => {
  const { grammar } = await canonicalFrames();
  const plan = buildChoreographyPlan(grammar, surface);
  assert.ok(plan);
  const byType = new Map(plan.events.map((event) => [event.eventType, event]));
  const orderedTypes = [
    "action_rejected",
    "ability_activated",
    "recipient_health_resolution",
    "combat_countdown_reset",
    "cooldown_started",
    "charge_phase_displacement",
    "ordinary_movement_phase_displacement",
    "agent_died",
    "status_applied",
    "spawn_shield_expired",
    "respawn_wave_occurred",
    "agent_respawned",
  ];
  for (let index = 1; index < orderedTypes.length; index += 1) {
    const previous = byType.get(orderedTypes[index - 1]);
    const current = byType.get(orderedTypes[index]);
    assert.ok(previous);
    assert.ok(current);
    assert.ok(previous.phaseEnd <= current.phaseStart);
  }
  assert.equal(byType.get("charge_phase_displacement")?.phaseStart, 350);
  assert.equal(byType.get("charge_phase_displacement")?.persistent, true);
  assert.equal(byType.get("ordinary_movement_phase_displacement")?.phaseStart, 450);
  assert.equal(byType.get("ordinary_movement_phase_displacement")?.persistent, false);
  assert.equal(byType.get("agent_respawned")?.phaseEnd, CHOREOGRAPHY_PHASES.total);
});

test("route fixture preserves multiplicity and exact V2 event identities", async () => {
  const { routes } = await canonicalFrames();
  const plan = buildChoreographyPlan(routes, surface);
  assert.ok(plan);
  assert.equal(plan.events.length, 9);
  assert.ok(plan.events.every((event) => event.kind === "activation"));
  const sourceEvents = /** @type {Record<string, any>[]} */ (routes.event_batch.events);
  assert.deepEqual(
    plan.events.map((event) => event.eventId),
    sourceEvents.map((event) => event.event_id),
  );
  assert.equal(new Set(plan.events.map((event) => event.eventId)).size, 9);
  assert.ok(plan.events.every((event) => event.route));
  assert.equal(
    plan.events.filter((event) => event.sourceSlot === 1 && event.targetSlot === 6)
      .length,
    2,
  );
});

test("crowded fixture retains every health and status cue under pressure", async () => {
  const { crowded } = await canonicalFrames();
  const plan = buildChoreographyPlan(crowded, surface);
  assert.ok(plan);
  assert.equal(plan.events.length, 32);
  assert.equal(plan.events.filter((event) => event.kind === "net_health").length, 8);
  assert.equal(
    plan.events.filter((event) => event.kind === "status_lifecycle").length,
    12,
  );
  assert.equal(plan.events.filter((event) => event.kind === "activation").length, 10);
  assert.equal(
    plan.events.filter((event) => event.kind === "charge_displacement").length,
    2,
  );
});

test("non-coexisting activation and health phases do not reserve each other's geometry", async () => {
  const { grammar } = await canonicalFrames();
  const events = /** @type {Record<string, any>[]} */ (grammar.event_batch.events);
  const ability = {
    ...events.find((event) => event.event_type === "ability_activated"),
    source_anchor: { position: [150, 100] },
    recipient_anchor: { position: [100, 56] },
  };
  const health = {
    ...events.find((event) => event.event_type === "recipient_health_resolution"),
    recipient_anchor: { position: [100, 100] },
  };
  const phaseSurface = /** @type {ProjectionSurface} */ ({
    worldToScreen: (point) => ({
      x: "x" in point ? point.x : Number(point[0]),
      y: "y" in point ? point.y : Number(point[1]),
    }),
    worldLengthToScreen: (length) => length,
    viewportBounds: {
      left: 0,
      top: 0,
      right: 200,
      bottom: 200,
      width: 200,
      height: 200,
    },
    protectedRects: [],
  });
  /** @param {Record<string, any>[]} selectedEvents */
  const planFor = (selectedEvents) =>
    buildChoreographyPlan(
      {
        ...grammar,
        event_batch: { ...grammar.event_batch, events: selectedEvents },
      },
      phaseSurface,
    );
  const healthOnly = planFor([health]);
  const withActivation = planFor([ability, health]);
  assert.ok(healthOnly);
  assert.ok(withActivation);
  assert.deepEqual(
    withActivation.events.find((event) => event.kind === "net_health")?.cue,
    healthOnly.events.find((event) => event.kind === "net_health")?.cue,
  );
});

test("POV planning consumes only recipient-local cue identities", async () => {
  const { pov } = await canonicalFrames();
  const plan = buildChoreographyPlan(pov, surface);
  assert.ok(plan);
  assert.equal(plan.transitionId, pov.incoming_pov_transition_id);
  const sourceCues = /** @type {Record<string, any>[]} */ (
    pov.projection.incoming_cues
  );
  assert.deepEqual(
    plan.events.map((event) => event.eventId),
    sourceCues.map((cue) => cue.cue_id),
  );
  assert.ok(plan.events.every((event) => event.eventId.includes(":actor-pov:")));
  assert.ok(
    plan.events.every(
      (event) => !Object.hasOwn(event, "sourceGlobalSlot") && !event.targetSlot,
    ),
  );
  const spatialDeltas = plan.events.filter(
    (event) =>
      event.eventType === "own_position_changed" ||
      event.eventType === "own_health_changed",
  );
  assert.deepEqual(
    spatialDeltas.map((event) => event.eventType),
    ["own_position_changed", "own_health_changed"],
  );
  assert.ok(
    spatialDeltas.every(
      (event) =>
        event.phaseStart === CHOREOGRAPHY_PHASES.povSuccessorObservationStart &&
        event.phaseEnd === CHOREOGRAPHY_PHASES.total,
    ),
  );
  assert.equal(
    new Set(spatialDeltas.map((event) => event.phaseStart)).size,
    1,
    "adjacent POV deltas must not invent a causal order",
  );
});

test("numeric transition identities fail closed", async () => {
  const { grammar } = await canonicalFrames();
  assert.equal(
    buildChoreographyPlan(
      {
        ...grammar,
        incoming_transition_id: 1,
        transition_id: 1,
        event_batch: { ...grammar.event_batch, transition_id: 1 },
      },
      surface,
    ),
    null,
  );
});
