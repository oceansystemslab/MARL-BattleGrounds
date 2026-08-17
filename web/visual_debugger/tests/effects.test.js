import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import { explainChoreographyEvent } from "../src/choreography-painter.js";
import {
  buildChoreographyPlan,
  isSubmissionCommand,
} from "../src/choreography-plan.js";

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

/** @type {Promise<Record<string, any>> | undefined} */
let authorizedFixturePromise;

/** @returns {Promise<Record<string, any>>} */
async function authorizedFixture() {
  if (!authorizedFixturePromise) {
    authorizedFixturePromise = readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ).then(JSON.parse);
  }
  return authorizedFixturePromise;
}

test("strict authorized fixture preserves exact choreography identity", async () => {
  const fixture = await authorizedFixture();
  const rawPresentation = fixture.presentations.replay_oracle;
  const presentation = await normalizeAuthorizedPresentationFrameV1(rawPresentation);
  const serializedBefore = JSON.stringify(presentation);
  const plan = buildChoreographyPlan(presentation, surface);
  assert.ok(plan);
  assert.equal(plan.transitionId, rawPresentation.latest_events.incoming_transition_id);
  assert.deepEqual(
    plan.events.flatMap((event) => event.atomicEventIds ?? [event.eventId]),
    rawPresentation.latest_events.ordered_event_ids,
  );
  assert.equal(JSON.stringify(presentation), serializedBefore);
  assert.equal(buildChoreographyPlan(rawPresentation, surface), null);
  assert.equal(buildChoreographyPlan({ ...presentation }, surface), null);
  assert.equal(
    buildChoreographyPlan(fixture.pairs.live_oracle.transport, surface),
    null,
  );
  assert.equal(buildChoreographyPlan(null, surface), null);
  assert.equal(buildChoreographyPlan({}, surface), null);
});

test("transition identity excludes revision while authorization tracks POV actor", async () => {
  const fixture = await authorizedFixture();
  const rawOracle = fixture.presentations.replay_oracle;
  const rawPov = fixture.presentations.replay_no_shared_obs_agent_pov;
  const revisedRawOracle = structuredClone(rawOracle);
  revisedRawOracle.source.source_revision += 1;
  revisedRawOracle.source.source_authority_epoch += 1;
  const oracle = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(rawOracle),
    surface,
  );
  const revised = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(revisedRawOracle),
    surface,
  );
  const pov = buildChoreographyPlan(
    await normalizeAuthorizedPresentationFrameV1(rawPov),
    surface,
  );
  assert.ok(oracle);
  assert.ok(revised);
  assert.ok(pov);
  assert.equal(revised.epochKey, oracle.epochKey);
  assert.equal(revised.fingerprint, oracle.fingerprint);
  assert.notEqual(revised.authorizationKey, oracle.authorizationKey);
  assert.deepEqual(JSON.parse(oracle.authorizationKey), [
    rawOracle.source.source_session_id,
    rawOracle.source.source_authority_epoch,
    rawOracle.presentation_kind,
    null,
  ]);
  assert.deepEqual(JSON.parse(pov.authorizationKey), [
    rawPov.source.source_session_id,
    rawPov.source.source_authority_epoch,
    rawPov.presentation_kind,
    rawPov.authority.recipient_presentation_key,
  ]);
  const rejectedDiagnosticReplay = structuredClone(rawPov);
  rejectedDiagnosticReplay.authority.projection_basis = "shared_obs_source_material";
  assert.equal(buildChoreographyPlan(rejectedDiagnosticReplay, surface), null);
  await assert.rejects(
    normalizeAuthorizedPresentationFrameV1(rejectedDiagnosticReplay),
  );
  assert.notEqual(oracle.authorizationKey, pov.authorizationKey);
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

test("scene has one Analysis battlefield branch and owns durable shield hooks", async () => {
  const source = await readFile(new URL("../src/scene.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /debug-protected-zone/u);
  assert.doesNotMatch(source, /debug-visibility-cue/u);
  assert.doesNotMatch(source, /preset === "debug"/u);
  assert.doesNotMatch(source, /preset !== "presentation"/u);
  assert.doesNotMatch(source, /scene\.audience_badge/u);
  assert.match(source, /dataset\.preset = "analysis"/u);
  assert.match(
    source,
    /scene\.audience === "researcher" \? "Oracle View" : "Agent POV"/u,
  );
  assert.match(source, /showLegality: true/u);
  assert.match(source, /agent-spawn-shield__shell/u);
  assert.match(source, /agent-spawn-shield__ticks/u);
  assert.match(source, /combat-in-progress/u);
  assert.match(source, /agent-combat-state-icon/u);
  assert.match(source, /combat status IC, steps until OOC/u);
  assert.match(source, /combat status OOC/u);
  assert.match(source, /createSpawnShieldView\(agent, spawnShieldMechanics\)/u);
  assert.match(
    source,
    /registerTooltipOwner\(nodes\.shieldRoot, spawnShieldView\.descriptor\)/u,
  );
});

test("authorized regeneration owns a packed plus cue instead of the generic onion", async () => {
  const [planSource, painterSource, css] = await Promise.all([
    readFile(new URL("../src/choreography-plan.js", import.meta.url), "utf8"),
    readFile(new URL("../src/choreography-painter.js", import.meta.url), "utf8"),
    readFile(new URL("../styles.css", import.meta.url), "utf8"),
  ]);
  assert.match(
    planSource,
    /row\.kind === "health_regenerated" \? "regeneration" : "semantic_pulse"/u,
  );
  assert.match(
    planSource,
    /layoutOutcomeCues\(statusEvents, surface, "regeneration"\)/u,
  );
  assert.match(painterSource, /combat-regeneration__plus/u);
  assert.match(painterSource, /combat-regeneration__value/u);
  assert.doesNotMatch(painterSource, /event\.cueSemantic === "health_regenerated"/u);
  assert.match(css, /\.combat-regeneration__pulse/u);
  assert.match(css, /\.combat-regeneration__recipient-anchor/u);
  assert.doesNotMatch(css, /\.combat-semantic-pulse--health-regenerated/u);
});

test("authorized age plus application is one Reapplied plan row without changing atomics", async () => {
  const fixture = JSON.parse(
    await readFile(
      new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
      "utf8",
    ),
  );
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const applied = structuredClone(
    raw.latest_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "status_applied",
    ),
  );
  assert.ok(applied);
  const aged = structuredClone(applied);
  aged.event_kind = "status_aged_to_zero";
  delete aged.source_anchor;
  const events = [aged, applied].map((event, ordinal) => ({
    ...event,
    event_id: `${raw.latest_events.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
    phase_rank: 100,
  }));
  raw.latest_events.events = events;
  raw.latest_events.event_count = events.length;
  raw.latest_events.ordered_event_ids = events.map(({ event_id }) => event_id);
  raw.latest_events.ordered_event_kinds = events.map(({ event_kind }) => event_kind);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const incomingEventsBefore = JSON.stringify(frame.latest_events.events);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(JSON.stringify(frame.latest_events.events), incomingEventsBefore);
  assert.equal(plan.events.length, 1);
  assert.equal(plan.bounds.nodes, 30);
  assert.deepEqual(
    plan.events.map(({ eventId, eventType }) => [eventId, eventType]),
    [[events[1].event_id, events[1].event_kind]],
  );
  const [primaryApplication] = plan.events;
  assert.deepEqual(
    primaryApplication.atomicEventIds,
    events.map(({ event_id }) => event_id),
  );
  assert.deepEqual(primaryApplication.applicationEventIds, [events[1].event_id]);
  assert.equal(primaryApplication.presentationSuppressed, false);
  assert.equal(primaryApplication.spatial, true);
  assert.equal(primaryApplication.lifecycle, "reapplied");
  assert.equal(primaryApplication.lifecycleToken.label, "Reapplied");
  assert.doesNotMatch(
    `${primaryApplication.lifecycleToken.accessibleName} ${primaryApplication.lifecycleToken.glyphKey}`,
    /expir/iu,
  );
});

test("authorized break plus application is one Reapplied plan row without changing atomics", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const applied = structuredClone(
    raw.latest_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "status_applied",
    ),
  );
  assert.ok(applied);
  const broken = structuredClone(applied);
  broken.event_kind = "status_broken_by_damage";
  delete broken.source_anchor;
  const events = [broken, applied].map((event, ordinal) => ({
    ...event,
    event_id: `${raw.latest_events.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
    phase_rank: 100,
  }));
  raw.latest_events.events = events;
  raw.latest_events.event_count = events.length;
  raw.latest_events.ordered_event_ids = events.map(({ event_id }) => event_id);
  raw.latest_events.ordered_event_kinds = events.map(({ event_kind }) => event_kind);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(plan.events.length, 1);
  assert.deepEqual(
    plan.events[0]?.atomicEventIds,
    events.map(({ event_id }) => event_id),
  );
  assert.deepEqual(plan.events[0]?.applicationEventIds, [events[1].event_id]);
  assert.equal(plan.events[0]?.lifecycle, "trap_broken_and_reapplied");
  assert.equal(plan.events[0]?.lifecycleToken.label, "Broken, then reapplied");
});

test("authorized Trap and Charge routes retain exact public endpoint owners", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const agentsByKey = new Map(
    raw.current_endpoint.scene.agents.map(
      (/** @type {Record<string, any>} */ agent) => [agent.presentation_key, agent],
    ),
  );
  /** @param {number} classId @returns {Record<string, any>} */
  const trajectoryForClass = (classId) => {
    const trajectory = raw.latest_events.agent_phase_trajectories.find(
      (/** @type {Record<string, any>} */ trajectory) =>
        agentsByKey.get(trajectory.agent_presentation_key)?.class_id === classId,
    );
    assert.ok(trajectory);
    return trajectory;
  };
  /** @param {number} sourceClass @param {number} targetClass @param {number} ordinal */
  const ability = (sourceClass, targetClass, ordinal) => ({
    event_kind: "ability_activated",
    ability_component: "ultimate",
    event_id: `${raw.latest_events.incoming_transition_id}:event:${String(ordinal).padStart(4, "0")}`,
    ordinal,
    phase_rank: 20,
    source_anchor: structuredClone(trajectoryForClass(sourceClass).transition_start),
    recipient_anchor: structuredClone(trajectoryForClass(targetClass).transition_start),
  });
  const events = [ability(3, 1, 0), ability(2, 3, 1)];
  raw.latest_events.events = events;
  raw.latest_events.event_count = events.length;
  raw.latest_events.ordered_event_ids = events.map(({ event_id }) => event_id);
  raw.latest_events.ordered_event_kinds = events.map(({ event_kind }) => event_kind);
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.deepEqual(
    plan.events.map((event) => event.tokenId),
    ["hunter_trap", "warrior_charge"],
  );
  /** @type {ReadonlyArray<readonly [Record<string, any>, number]>} */
  const endpointCases = [
    [plan.events[0], 1],
    [plan.events[1], 3],
  ];
  for (const [event, targetClass] of endpointCases) {
    const targetTrajectory = trajectoryForClass(targetClass);
    assert.equal(event.targetPresentationKey, targetTrajectory.agent_presentation_key);
    assert.equal(event.targetPublicAgentId, targetTrajectory.agent_public_agent_id);
    assert.deepEqual(
      event.target,
      surface.worldToScreen(targetTrajectory.successor.position),
    );
    assert.ok(event.route);
    assert.ok(
      Math.abs(
        Math.hypot(
          event.route.end.x - event.target.x,
          event.route.end.y - event.target.y,
        ) - 8,
      ) < 1e-9,
    );
  }
});

test("status presentation preserves the valid five-source application maximum", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const appliedTemplate = structuredClone(
    raw.latest_events.events.find(
      (/** @type {Record<string, any>} */ event) =>
        event.event_kind === "status_applied",
    ),
  );
  assert.ok(appliedTemplate);
  const sourceTrajectories = /** @type {Record<string, any>[]} */ (
    raw.latest_events.agent_phase_trajectories
  );
  const applications = sourceTrajectories.map(
    (/** @type {Record<string, any>} */ trajectory, index) => ({
      ...structuredClone(appliedTemplate),
      event_id: `${raw.latest_events.incoming_transition_id}:event:${String(index).padStart(4, "0")}`,
      ordinal: index,
      source_anchor: structuredClone(trajectory.successor),
    }),
  );
  raw.latest_events.events = applications;
  raw.latest_events.event_count = applications.length;
  raw.latest_events.ordered_event_ids = applications.map(
    (/** @type {Record<string, any>} */ { event_id }) => event_id,
  );
  raw.latest_events.ordered_event_kinds = applications.map(
    (/** @type {Record<string, any>} */ { event_kind }) => event_kind,
  );
  const frame = await normalizeAuthorizedPresentationFrameV1(raw);
  const serializedBefore = JSON.stringify(frame);
  const plan = buildChoreographyPlan(frame, surface);
  assert.ok(plan);
  assert.equal(JSON.stringify(frame), serializedBefore);
  assert.equal(plan.events.length, 1);
  const applicationEventIds = applications.map(
    (/** @type {Record<string, any>} */ event) => event.event_id,
  );
  const [primary] = plan.events;
  assert.ok(primary);
  assert.deepEqual(primary.atomicEventIds, applicationEventIds);
  assert.deepEqual(primary.applicationEventIds, applicationEventIds);
  assert.deepEqual(
    primary.applicationSources.map(
      (/** @type {Record<string, any>} */ { eventId }) => eventId,
    ),
    applicationEventIds,
  );
  assert.deepEqual(
    primary.applicationSources.map(
      (/** @type {Record<string, any>} */ { sourcePublicAgentId }) =>
        sourcePublicAgentId,
    ),
    sourceTrajectories.map(
      (/** @type {Record<string, any>} */ { agent_public_agent_id }) =>
        agent_public_agent_id,
    ),
  );
  const explanation = explainChoreographyEvent(primary);
  assert.deepEqual(
    explanation.rows.map(({ label, value }) => [label, value]),
    [
      [
        "Application Sources",
        sourceTrajectories
          .map(
            (/** @type {Record<string, any>} */ { agent_public_agent_id }) =>
              `Agent ID ${agent_public_agent_id}`,
          )
          .join("; "),
      ],
      ["Recipient", `Agent ID ${applications[0].recipient_anchor.public_agent_id}`],
    ],
  );
});

test("authorized normalization rejects cross-transition status rows", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  const applied = raw.latest_events.events.find(
    (/** @type {Record<string, any>} */ { event_kind }) =>
      event_kind === "status_applied",
  );
  assert.ok(applied);
  applied.event_id = "different-transition:event:0003";
  raw.latest_events.ordered_event_ids[3] = applied.event_id;
  await assert.rejects(normalizeAuthorizedPresentationFrameV1(raw), TypeError);
});

test("standalone serialized status events keep exact presentation meanings", async () => {
  const fixture = await authorizedFixture();
  const expectedCases = /** @type {ReadonlyArray<readonly [string, string]>} */ ([
    ["status_aged_to_zero", "expired"],
    ["status_broken_by_damage", "trap_broken"],
    ["status_applied", "applied"],
    ["status_refreshed_or_extended", "refreshed"],
    ["status_cleared_by_new_death", "cleared_by_death"],
  ]);
  for (const [eventKind, lifecycle] of expectedCases) {
    const raw = structuredClone(fixture.presentations.replay_oracle);
    const event = structuredClone(
      raw.latest_events.events.find(
        (/** @type {Record<string, any>} */ { event_kind }) =>
          event_kind === "status_applied",
      ),
    );
    event.event_kind = eventKind;
    event.event_id = `${raw.latest_events.incoming_transition_id}:event:0000`;
    event.ordinal = 0;
    if (eventKind !== "status_applied") {
      delete event.source_anchor;
    }
    raw.latest_events.events = [event];
    raw.latest_events.event_count = 1;
    raw.latest_events.ordered_event_ids = [event.event_id];
    raw.latest_events.ordered_event_kinds = [event.event_kind];
    const frame = await normalizeAuthorizedPresentationFrameV1(raw);
    const plan = buildChoreographyPlan(frame, surface);
    assert.ok(plan);
    assert.equal(plan.events.length, 1);
    assert.deepEqual(
      [plan.events[0].eventType, plan.events[0].lifecycle],
      [eventKind, lifecycle],
    );
    assert.deepEqual(plan.events[0].atomicEventIds, [event.event_id]);
    assert.equal(plan.events[0].phaseStart, 0);
    assert.equal(plan.events[0].phaseEnd, 480);
  }
});

test("durable status countdown changes never synthesize browser lifecycle events", async () => {
  const fixture = await authorizedFixture();
  const raw = structuredClone(fixture.presentations.replay_oracle);
  raw.latest_events.events = [];
  raw.latest_events.event_count = 0;
  raw.latest_events.ordered_event_ids = [];
  raw.latest_events.ordered_event_kinds = [];
  const countdownOnly = await normalizeAuthorizedPresentationFrameV1(raw);
  const plan = buildChoreographyPlan(countdownOnly, surface);
  assert.ok(plan);
  assert.deepEqual(plan.events, []);
  assert.equal(plan.phases.total, 0);
});
