import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  actionTupleCombatLabel,
  authorizedInspectorView,
  authorizedOutgoingTargetDescriptor,
  DebuggerPanels,
  disclosurePanelInitiallyOpen,
  eventDescriptor,
  eventSummary,
  pendingActionDisplayFacts,
  publicAgentIdMap,
  replayDiagnosticFacts,
  rosterControlDescriptor,
  rosterStatusDurationLabel,
} from "../src/panels.js";
import { projectSemanticDescriptor, semanticDescriptorText } from "../src/tooltip.js";

const authorizedFixture = JSON.parse(
  await readFile(
    new URL("./fixtures/authorized-presentations-v1.json", import.meta.url),
    "utf8",
  ),
);

/** @param {string} kind */
async function normalizedPresentation(kind) {
  return await normalizeAuthorizedPresentationFrameV1(
    authorizedFixture.presentations[kind],
  );
}

/** @param {string} kind */
async function normalizedStateCase(kind) {
  return await normalizeAuthorizedPresentationFrameV1(
    authorizedFixture.state_cases[kind],
  );
}

/**
 * Minimal event-capable DOM for the roster constructor. It deliberately models
 * bubbling so a fact-chip click exposes any ancestor activation listener.
 */
function rosterDomHarness() {
  /** @type {WeakMap<object, Map<string, Array<(...args: any[]) => void>>>} */
  const listenerRegistry = new WeakMap();
  /** @param {any} node @param {string} type */
  const listenersFor = (node, type) => listenerRegistry.get(node)?.get(type) ?? [];
  /** @type {any} */
  let ownerDocument;

  /** @param {string} tagName */
  function createNode(tagName) {
    const attributes = new Map();
    const listeners = new Map();
    /** @type {any[]} */
    const children = [];
    Object.defineProperty(children, "item", {
      value: (/** @type {number} */ index) => children[index] ?? null,
    });
    /** @type {any} */
    const node = {
      ownerDocument,
      parentElement: null,
      tagName: tagName.toUpperCase(),
      className: "",
      textContent: "",
      dataset: {},
      children,
      disabled: false,
      hidden: false,
      tabIndex: tagName.toLowerCase() === "button" ? 0 : -1,
      type: "",
      getBoundingClientRect() {
        return { left: 0, top: 0, right: 1, bottom: 1, width: 1, height: 1 };
      },
      /** @param {string} name */
      hasAttribute(name) {
        return attributes.has(name);
      },
      /** @param {string} name */
      getAttribute(name) {
        return attributes.get(name) ?? null;
      },
      /** @param {string} name @param {unknown} value */
      setAttribute(name, value) {
        attributes.set(name, String(value));
        if (name === "class") {
          this.className = String(value);
        }
      },
      /** @param {string} name */
      removeAttribute(name) {
        attributes.delete(name);
        if (name === "data-class") {
          delete this.dataset.class;
        }
      },
      /** @param {string} type @param {(...args: any[]) => void} listener */
      addEventListener(type, listener) {
        const registered = listeners.get(type) ?? [];
        registered.push(listener);
        listeners.set(type, registered);
      },
      /** @param {...any} appended */
      append(...appended) {
        for (const child of appended) {
          child.remove?.();
          child.parentElement = this;
          children.push(child);
        }
      },
      /** @param {...any} replacements */
      replaceChildren(...replacements) {
        for (const child of children) {
          child.parentElement = null;
        }
        children.splice(0);
        this.append(...replacements);
      },
      /** @param {any} child @param {any} reference */
      insertBefore(child, reference) {
        child.remove?.();
        const index =
          reference === null ? children.length : children.indexOf(reference);
        child.parentElement = this;
        children.splice(index < 0 ? children.length : index, 0, child);
      },
      remove() {
        if (this.parentElement === null) {
          return;
        }
        const siblings = this.parentElement.children;
        const index = siblings.indexOf(this);
        if (index >= 0) {
          siblings.splice(index, 1);
        }
        this.parentElement = null;
      },
      /** @param {any} candidate */
      contains(candidate) {
        return (
          this === candidate || children.some((child) => child.contains(candidate))
        );
      },
      click() {
        if (this.disabled) {
          return;
        }
        const event = { type: "click", target: this };
        for (let current = this; current !== null; current = current.parentElement) {
          for (const listener of listenersFor(current, "click")) {
            listener.call(current, event);
          }
        }
      },
    };
    listenerRegistry.set(node, listeners);
    return node;
  }

  ownerDocument = {
    createElement: createNode,
    /** @param {string} _namespace @param {string} tagName */
    createElementNS(_namespace, tagName) {
      return createNode(tagName);
    },
  };
  /** @param {any} root */
  const descendants = (root) => [
    root,
    ...root.children.flatMap((/** @type {any} */ child) => descendants(child)),
  ];
  /** @param {any} root */
  const textTree = (root) =>
    descendants(root)
      .map((node) => node.textContent)
      .filter(Boolean)
      .join(" ");
  return { createNode, descendants, ownerDocument, textTree };
}

/** @param {Record<string, any>} descriptor @param {string} label */
function semanticRowValue(descriptor, label) {
  const found = descriptor.rows.find(
    (/** @type {Record<string, any>} */ row) => row.label === label,
  );
  assert.ok(found, `missing semantic row ${label}`);
  return found.value;
}

const PUBLIC_IDS = new Map([
  [0, "zero:<opaque>"],
  [1, "one/agent&x"],
  [3, "three.300"],
  [4, "four four"],
  [5, "five#five"],
  [7, "seven:semicolon;"],
]);

test("authorized inspector rejects raw and forged presentation roots", async () => {
  const raw = authorizedFixture.presentations.replay_oracle;
  const normalized = await normalizedPresentation("replay_oracle");
  const forged = Object.freeze({ ...normalized });

  assert.equal(authorizedInspectorView(raw), null);
  assert.equal(authorizedInspectorView(forged), null);
  assert.equal(authorizedInspectorView(null), null);
});

test("authorized replay inspector keeps current owner separate from outgoing target", async () => {
  const cases = [
    ["replay_oracle", "researcher"],
    ["replay_no_shared_obs_agent_pov", "agent_pov"],
    ["replay_shared_obs_agent_pov", "agent_pov"],
  ];
  const legalityDescriptorIds = new Set();

  for (const [kind, audience] of cases) {
    const presentation = await normalizedPresentation(kind);
    const inspector = authorizedInspectorView(presentation);
    assert.ok(inspector);
    assert.equal(inspector.title, "Comprehensive Agent Details");
    assert.equal(
      inspector.owner.presentation_key,
      presentation.inspection.actor_presentation_key,
    );
    assert.equal(
      inspector.owner.public_agent_id,
      presentation.inspection.actor_public_agent_id,
    );
    assert.equal(inspector.owner_descriptor.title, "Agent ID agent-slot-0 · Now");
    assert.equal(inspector.owner_class_accent, "mage");
    assert.equal(
      inspector.legality.owner_presentation_key,
      inspector.owner.presentation_key,
    );
    assert.equal(
      inspector.legality.owner_public_agent_id,
      inspector.owner.public_agent_id,
    );
    assert.equal(inspector.legality.target_kind, "no_target");
    assert.equal(inspector.legality.target_presentation_key, null);
    assert.equal(inspector.legality.target_public_agent_id, null);
    assert.equal(inspector.outgoing_target_descriptor.title, "Outgoing target");
    assert.equal(
      semanticRowValue(inspector.outgoing_target_descriptor, "Disclosure"),
      "No Target",
    );
    assert.equal(
      semanticRowValue(inspector.outgoing_target_descriptor, "Target"),
      "No target",
    );
    assert.equal(
      semanticRowValue(inspector.outgoing_target_descriptor, "Authorized label"),
      inspector.legality.target_display_name,
    );
    assert.notEqual(
      inspector.owner.public_agent_id,
      inspector.legality.target_public_agent_id,
    );
    assert.deepEqual(
      inspector.legality_cards.map(
        (/** @type {Record<string, any>} */ card) => card.heading,
      ),
      [
        "Basic Legality · Agent ID agent-slot-0",
        "Ultimate Legality · Agent ID agent-slot-0",
      ],
    );
    const targetAction = presentation.inspection.accepted_action.target_action;
    const exactRow =
      presentation.inspection.decision_mask.target_use_ultimate_joint_mask[
        targetAction
      ];
    assert.deepEqual(
      inspector.legality_cards.map((/** @type {Record<string, any>} */ card) => ({
        lane: card.lane,
        available: card.descriptor.rows[0].value === "True",
      })),
      [
        { lane: 0, available: exactRow[0] },
        { lane: 1, available: exactRow[1] },
      ],
    );
    for (const card of /** @type {ReadonlyArray<Record<string, any>>} */ (
      inspector.legality_cards
    )) {
      assert.equal(card.descriptor.title, card.heading);
      assert.equal(card.descriptor.id.includes(inspector.owner.presentation_key), true);
      legalityDescriptorIds.add(card.descriptor.id);
      assert.equal(Object.isFrozen(card.descriptor), true);
    }
    const auraAvailability = inspector.owner_descriptor.sections
      .flatMap((/** @type {Record<string, any>} */ section) => section.rows)
      .find(
        (/** @type {Record<string, any>} */ row) =>
          row.label === "Aggregate Aura Modifiers",
      )?.value;
    if (audience === "researcher") {
      assert.notEqual(auraAvailability, "Unavailable");
    } else {
      assert.equal(auraAvailability, "Unavailable");
      assert.equal(
        inspector.owner_descriptor.sections.some(
          (/** @type {Record<string, any>} */ section) =>
            section.title === "Exact Class Mechanics",
        ),
        false,
      );
    }
    assert.equal(Object.isFrozen(inspector), true);
    assert.equal(Object.isFrozen(inspector.legality_cards), true);
    assert.equal(
      inspector.legality_cards.every((/** @type {Record<string, any>} */ card) =>
        Object.isFrozen(card),
      ),
      true,
    );
  }
  assert.equal(legalityDescriptorIds.size, 6);
});

test("authorized replay inspector retains final selected owner without outgoing legality", async () => {
  for (const kind of [
    "replay_oracle_final_selected",
    "replay_no_shared_final",
    "replay_shared_final",
  ]) {
    const presentation = await normalizedStateCase(kind);
    const inspector = authorizedInspectorView(presentation);
    assert.ok(inspector);
    assert.equal(
      inspector.owner.presentation_key,
      presentation.action_axis.owner_presentation_key,
    );
    assert.equal(
      inspector.owner.public_agent_id,
      presentation.action_axis.owner_public_agent_id,
    );
    assert.equal(inspector.owner_descriptor.title, "Agent ID agent-slot-0 · Now");
    assert.equal(inspector.owner_class_accent, "mage");
    assert.equal(inspector.legality, null);
    assert.equal(inspector.outgoing_target_descriptor, null);
    assert.deepEqual(inspector.legality_cards, []);
  }
});

test("authorized live inspector keeps draft target vocabulary out of the outgoing card", async () => {
  const presentation = await normalizedPresentation("live_oracle");
  const inspector = authorizedInspectorView(presentation);
  assert.ok(inspector);
  assert.ok(inspector.legality);
  assert.equal(inspector.outgoing_target_descriptor, null);
});

test("authorized replay inspector leaves final unselected details absent", async () => {
  const presentation = await normalizedStateCase("replay_oracle_final_unselected");
  const inspector = authorizedInspectorView(presentation);
  assert.ok(inspector);
  assert.equal(inspector.title, "Comprehensive Agent Details");
  assert.equal(inspector.owner, null);
  assert.equal(inspector.owner_descriptor, null);
  assert.equal(inspector.owner_class_accent, null);
  assert.equal(inspector.legality, null);
  assert.equal(inspector.outgoing_target_descriptor, null);
  assert.deepEqual(inspector.legality_cards, []);
});

test("authorized roster exposes one native key-only action with isolated fact owners", async () => {
  const dom = rosterDomHarness();
  /** @type {Record<string, unknown>[]} */
  const commands = [];
  const binding = () => dom.createNode("div");
  const documentDescriptor = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: dom.ownerDocument,
  });

  try {
    const roster = binding();
    const panels = new DebuggerPanels({
      roster,
      rosterCount: binding(),
      selectionCard: binding(),
      pendingHeading: binding(),
      pendingCount: binding(),
      pendingScope: binding(),
      pendingCard: binding(),
      acceptedCard: binding(),
      acceptedAnnouncement: binding(),
      eventFeed: binding(),
      eventCount: binding(),
      diagnosticsCard: binding(),
      onCommand: (command) => {
        commands.push(command);
      },
    });
    const liveOracle = await normalizedPresentation("live_oracle");
    panels.renderAuthorizedRoster(liveOracle, false);
    const oracleRows = /** @type {Record<string, any>[]} */ (
      [...panels.rosterRows.values()].filter(
        (/** @type {Record<string, any>} */ row) => "primaryButton" in row,
      )
    );
    assert.equal(oracleRows.length, 5);

    for (const row of oracleRows) {
      const descendants = dom.descendants(row.element);
      const buttons = descendants.filter(
        (/** @type {Record<string, any>} */ node) => node.tagName === "BUTTON",
      );
      assert.deepEqual(buttons, [row.primaryButton]);
      assert.equal(row.element.tabIndex, -1);
      assert.equal(row.primaryButton.tabIndex, 0);
      assert.equal(row.primaryButton.type, "button");
      assert.deepEqual(row.primaryButton.dataset, {
        action: "activate-agent",
        presentationKey: row.element.dataset.presentationKey,
      });
      assert.equal(row.primaryButton.getAttribute("data-role"), null);
      assert.equal(Object.hasOwn(row.primaryButton.dataset, "role"), false);
      assert.equal(Object.hasOwn(row.primaryButton.dataset, "slot"), false);
      assert.equal(Object.hasOwn(row.primaryButton.dataset, "commandSlot"), false);
      assert.doesNotMatch(
        dom.textTree(row.element),
        /\b(?:Target|Control|Reference|POV actor)\b/u,
      );
      assert.equal(row.primaryButton.contains(row.statuses), false);
      assert.equal(row.primaryButton.contains(row.modifiers), false);
      const classOwners = descendants.filter(
        (/** @type {Record<string, any>} */ node) =>
          Object.hasOwn(node.dataset, "class"),
      );
      assert.deepEqual(classOwners, [row.identityId]);
      assert.match(row.element.dataset.team, /^team-[ab]$/u);
    }
    assert.deepEqual(
      new Set(oracleRows.map((row) => row.identityId.dataset.class)),
      new Set(["mage", "warrior", "hunter", "rogue", "priest"]),
    );

    oracleRows[0].primaryButton.click();
    assert.deepEqual(commands, [
      {
        command_type: "activate_authorized_agent",
        presentation_key: oracleRows[0].element.dataset.presentationKey,
      },
    ]);
    const factRow = oracleRows.find(
      (row) =>
        dom
          .descendants(row.element)
          .some((node) => node.className.includes("roster-fact-token--status")) &&
        dom
          .descendants(row.element)
          .some((node) => node.className.includes("roster-fact-token--modifier")),
    );
    assert.ok(factRow);
    const factNodes = dom.descendants(factRow.element);
    const statusOwner = factNodes.find((node) =>
      node.className.includes("roster-fact-token--status"),
    );
    const modifierOwner = factNodes.find((node) =>
      node.className.includes("roster-fact-token--modifier"),
    );
    assert.ok(statusOwner);
    assert.ok(modifierOwner);
    statusOwner.click();
    modifierOwner.click();
    assert.equal(commands.length, 1);

    const replayAgent = await normalizedPresentation("replay_no_shared_obs_agent_pov");
    panels.renderAuthorizedRoster(replayAgent, false);
    const agentRows = /** @type {Record<string, any>[]} */ (
      [...panels.rosterRows.values()].filter(
        (/** @type {Record<string, any>} */ row) => "primaryButton" in row,
      )
    );
    assert.equal(agentRows.length, 3);
    assert.equal(
      agentRows.every((row) => row.primaryButton.disabled === false),
      true,
    );
    agentRows[0].primaryButton.click();
    const latestCommand = commands.at(-1);
    assert.ok(latestCommand);
    assert.deepEqual(Object.keys(latestCommand).sort(), [
      "command_type",
      "presentation_key",
    ]);
    assert.equal(latestCommand.command_type, "activate_authorized_agent");
    panels.renderAuthorizedRoster(replayAgent, true);
    assert.equal(
      agentRows.every((row) => row.primaryButton.disabled === true),
      true,
    );
    const commandCount = commands.length;
    agentRows[0].primaryButton.click();
    assert.equal(commands.length, commandCount);
  } finally {
    if (documentDescriptor === undefined) {
      Reflect.deleteProperty(globalThis, "document");
    } else {
      Object.defineProperty(globalThis, "document", documentDescriptor);
    }
  }
});

test("authorized outgoing target descriptor preserves all three disclosure kinds", () => {
  const base = {
    owner_presentation_key: "pov_owner",
    owner_public_agent_id: "owner:opaque",
    target_action: 4,
  };
  const cases = [
    ["no_target", "No target", null, null, "No Target"],
    [
      "visible_authorized_agent",
      "Visible body",
      "pov_visible",
      "visible/opaque",
      "Visible Authorized Agent",
    ],
    [
      "axis_only_authorized_agent",
      "Axis-only body",
      null,
      "axis:opaque",
      "Axis Only Authorized Agent",
    ],
  ];
  for (const [kind, label, key, publicId, disclosure] of cases) {
    const descriptor = authorizedOutgoingTargetDescriptor({
      ...base,
      target_kind: kind,
      target_display_name: label,
      target_presentation_key: key,
      target_public_agent_id: publicId,
    });
    assert.ok(descriptor);
    assert.equal(semanticRowValue(descriptor, "Disclosure"), disclosure);
    assert.equal(
      semanticRowValue(descriptor, "Target"),
      kind === "no_target" ? "No target" : `Agent ID ${publicId}`,
    );
    assert.equal(semanticRowValue(descriptor, "Authorized label"), label);
    assert.equal(Object.isFrozen(descriptor), true);
  }
  assert.equal(
    authorizedOutgoingTargetDescriptor({
      ...base,
      target_kind: "axis_only_authorized_agent",
      target_display_name: "Poisoned axis",
      target_presentation_key: "forbidden-visible-key",
      target_public_agent_id: "axis:opaque",
    }),
    null,
  );
});

test("roster rendering has no legacy compact Presentation branch", async () => {
  const source = await readFile(new URL("../src/panels.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /compactRoster/u);
  assert.doesNotMatch(source, /dataset\.compact/u);
  assert.doesNotMatch(source, /frame\?\.preset === "presentation"/u);
  assert.match(source, /removeAttribute\("data-compact"\)/u);
});

test("authorized inspector copy separates live draft and replay outgoing epochs", async () => {
  const source = await readFile(new URL("../src/panels.js", import.meta.url), "utf8");
  assert.match(source, /authorized pending draft for the next submission/u);
  assert.match(source, /separately authorized recorded outgoing action/u);
  assert.match(source, /Draft range, target, and legality overlays/u);
  assert.match(source, /Recorded range, target, and legality overlays/u);
  assert.doesNotMatch(source, /Settled range, target, and legality overlays/u);
});

test("native disclosure defaults open operational live panels, roster, and events", () => {
  assert.equal(disclosurePanelInitiallyOpen("command-deck", false), true);
  assert.equal(disclosurePanelInitiallyOpen("command-deck", true), false);
  assert.equal(disclosurePanelInitiallyOpen("roster-details", false), true);
  assert.equal(disclosurePanelInitiallyOpen("roster-details", true), true);
  assert.equal(disclosurePanelInitiallyOpen("events-details", false), true);
  assert.equal(disclosurePanelInitiallyOpen("events-details", true), true);
  for (const id of [
    "agent-details",
    "pending-turn-details",
    "latest-transition-details",
    "visual-key",
    "technical-frame-details",
  ]) {
    assert.equal(disclosurePanelInitiallyOpen(id, false), false);
    assert.equal(disclosurePanelInitiallyOpen(id, true), false);
  }
  assert.throws(() => disclosurePanelInitiallyOpen("unknown", false), /Unknown/u);
});

test("replay diagnostics display recorded movement scale only for researchers", () => {
  const base = {
    viewer_mode: "replay",
    frame_kind: "researcher_replay_viewer",
    timeline_id: "artifact:timeline:researcher",
    cursor: { cursor_generation: 3, choreography_generation: 4 },
    artifact_summary: { metric_report_availability: "available" },
  };
  const researcher = replayDiagnosticFacts({
    ...base,
    replay_audience: "researcher",
    recorded_ordinary_movement_distance_scale: 0.375,
  });
  const pov = replayDiagnosticFacts({
    ...base,
    replay_audience: "actor_pov",
  });

  assert.deepEqual(researcher.at(-1), {
    label: "Recorded movement scale",
    value: "0.38",
  });
  assert.equal(
    pov.some((fact) => fact.label === "Recorded movement scale"),
    false,
  );
  assert.equal(Object.isFrozen(researcher), true);
  assert.deepEqual(replayDiagnosticFacts({ viewer_mode: "live" }), []);
});

test("roster buttons own audience- and availability-specific control help", () => {
  /** @type {Array<["target" | "control", "live" | "researcher_replay", boolean, string, string, string]>} */
  const cases = [
    ["target", "live", false, "Target", "staged action", "Available"],
    ["control", "live", true, "Control", "currently unavailable", "Unavailable"],
    [
      "target",
      "researcher_replay",
      false,
      "Reference",
      "does not change the immutable range anchor",
      "Available",
    ],
    ["control", "researcher_replay", false, "POV actor", "point of view", "Available"],
  ];
  for (const [role, mode, disabled, title, copy, availability] of cases) {
    const descriptor = rosterControlDescriptor(
      role,
      "arbitrary:<agent>&7",
      mode,
      disabled,
    );
    assert.equal(descriptor.title, title);
    assert.match(descriptor.summary, new RegExp(copy, "u"));
    assert.equal(descriptor.rows[0].value, "Agent ID arbitrary:<agent>&7");
    assert.equal(descriptor.rows[1].value, availability);
    assert.equal(descriptor.metadata.full, false);
  }
  assert.throws(
    () => rosterControlDescriptor("target", "agent", "live", /** @type {never} */ (1)),
    /boolean/u,
  );
});

test("same-root public-ID map joins scene and batch and fails closed on conflicts", () => {
  const joined = publicAgentIdMap({
    scene: {
      agents: [
        { global_slot: 1, public_agent_id: "one/agent&x" },
        { global_slot: 7, public_agent_id: "seven:semicolon;" },
        { global_slot: "4", public_agent_id: "coerced-slot-must-not-join" },
      ],
    },
    event_batch: {
      public_agent_id_by_global_slot: ["zero:<opaque>", "one/agent&x", "batch-two"],
    },
  });
  assert.deepEqual(
    [...joined],
    [
      [1, "one/agent&x"],
      [7, "seven:semicolon;"],
      [0, "zero:<opaque>"],
      [2, "batch-two"],
    ],
  );
  assert.equal(joined.has(4), false);

  const conflict = publicAgentIdMap({
    scene: { agents: [{ global_slot: 1, public_agent_id: "scene-one" }] },
    event_batch: {
      public_agent_id_by_global_slot: ["zero", "different-one"],
    },
  });
  assert.equal(conflict.has(1), false);
  assert.equal(conflict.get(0), "zero");
});

test("all 21 researcher event kinds use arbitrary public IDs and never slot labels", () => {
  const events = [
    {
      event_type: "action_rejected",
      actor_global_slot: 1,
      rejection_component: "movement",
    },
    {
      event_type: "ability_activated",
      source_global_slot: 1,
      recipient_global_slot: 7,
      ability_component: "warrior_charge",
    },
    {
      event_type: "source_damage_output",
      source_global_slot: 1,
      raw_damage_output: 10,
      source_modified_damage_output: 12,
      recipient_damage_modifier: 0.8,
      mage_damage_aura_covering_emitter_global_slots: [0, 5],
      warrior_mitigation_aura_covering_emitter_global_slots: [3],
    },
    {
      event_type: "source_healing_output",
      source_global_slot: 4,
      raw_healing_output: 10,
      source_modified_healing_output: 11,
      recipient_healing_modifier: 1,
    },
    {
      event_type: "recipient_health_resolution",
      recipient_global_slot: 7,
      realized_net_health_change: -3,
      transition_start_health: 10,
      health_after_combat_resolution: 7,
    },
    { event_type: "combat_countdown_reset", agent_global_slot: 1 },
    {
      event_type: "health_regenerated",
      agent_global_slot: 3,
      actual_health_regenerated: 1.25,
    },
    { event_type: "cooldown_started", agent_global_slot: 4 },
    { event_type: "cooldown_ready", agent_global_slot: 5 },
    {
      event_type: "charge_phase_displacement",
      agent_global_slot: 1,
      start_anchor: { position: [1, 2] },
      end_anchor: { position: [3, 4] },
    },
    {
      event_type: "ordinary_movement_phase_displacement",
      agent_global_slot: 7,
      start_anchor: { position: [4, 5] },
      end_anchor: { position: [6, 7] },
    },
    { event_type: "agent_died", recipient_global_slot: 7 },
    {
      event_type: "lethal_damage_contribution",
      source_global_slot: 1,
      recipient_global_slot: 7,
      attributed_death_damage: 4.5,
    },
    {
      event_type: "status_aged_to_zero",
      recipient_global_slot: 7,
      status_id: "hunter_basic_slow",
    },
    {
      event_type: "status_broken_by_damage",
      recipient_global_slot: 7,
      status_id: "warrior_charge_stun",
    },
    {
      event_type: "status_applied",
      source_global_slot: 1,
      recipient_global_slot: 7,
      status_id: "rogue_poison_slow",
    },
    {
      event_type: "status_refreshed_or_extended",
      recipient_global_slot: 7,
      status_id: "mage_burst",
    },
    {
      event_type: "status_cleared_by_new_death",
      recipient_global_slot: 7,
      status_id: "priest_freedom",
    },
    { event_type: "spawn_shield_expired", agent_global_slot: 3 },
    { event_type: "respawn_wave_occurred", team_id: 2 },
    {
      event_type: "agent_respawned",
      agent_global_slot: 5,
      realized_successor_position: [8, 9],
    },
  ];
  assert.equal(events.length, 21);
  const summaries = events.map((event) => eventSummary(event, PUBLIC_IDS));
  assert.equal(
    summaries.every((summary) => !summary.includes("id_")),
    true,
  );
  assert.equal(
    summaries
      .filter((_, index) => index !== 19)
      .every((summary) => !summary.includes("Agent ID unavailable")),
    true,
  );
  assert.match(summaries[0], /Agent ID one\/agent&x/u);
  assert.match(summaries[1], /Agent ID seven:semicolon;/u);
  assert.match(summaries[2], /Sorcerer’s Empowerment emitters 2/u);
  assert.match(summaries[2], /Guardian’s Barrier emitters 1/u);
  assert.doesNotMatch(summaries[2], /zero:<opaque>|five#five|three\.300/u);
  assert.match(summaries[3], /aura emitter evidence not recorded/u);
});

test("missing event joins fail closed instead of deriving a public ID from a slot", () => {
  const summary = eventSummary(
    { event_type: "agent_died", recipient_global_slot: 999 },
    PUBLIC_IDS,
  );
  assert.equal(summary, "Agent died · Agent ID unavailable");
  assert.doesNotMatch(summary, /999|id_/u);
});

test("event semantic descriptor and accessible text never consume canonical event IDs", () => {
  const technicalEventId = "event:canonical:<private>&9001";
  const event = {
    event_id: technicalEventId,
    event_type: "status_applied",
    status_id: "rogue_poison_slow",
    source_global_slot: 1,
    recipient_global_slot: 7,
  };
  const descriptor = eventDescriptor(event, 4, PUBLIC_IDS);
  const serialized = JSON.stringify(descriptor);
  const compactAccessibleText = semanticDescriptorText(
    projectSemanticDescriptor(descriptor, "compact"),
  ).join(" ");
  const fullAccessibleText = semanticDescriptorText(
    projectSemanticDescriptor(descriptor, "full"),
  ).join(" ");

  assert.equal(descriptor.id, "event:status_applied:4");
  assert.doesNotMatch(serialized, /event:canonical:<private>&9001/u);
  assert.doesNotMatch(compactAccessibleText, /event:canonical:<private>&9001/u);
  assert.doesNotMatch(fullAccessibleText, /event:canonical:<private>&9001/u);
  assert.match(compactAccessibleText, /Agent ID one\/agent&x/u);
  assert.match(fullAccessibleText, /Agent ID seven:semicolon;/u);
  assert.throws(() => eventDescriptor(event, -1, PUBLIC_IDS), /non-negative integer/u);
});

test("roster duration labels abbreviate only the human-facing extreme value", () => {
  assert.equal(rosterStatusDurationLabel(5), "5");
  assert.equal(rosterStatusDurationLabel(123456789), "123M");
  assert.equal(rosterStatusDurationLabel(null), "?");
});

test("pending no-combat copy hides transport vocabulary without dropping facts", () => {
  const facts = pendingActionDisplayFacts({
    move_action: 0,
    movement_mask_value: true,
    target_action: 0,
    target: { disclosure: "target_none", global_slot: null },
    armed_lane: 0,
    pair_mask_value: null,
  });

  assert.deepEqual(facts, {
    movement: "Movement · Stay (0) · Available",
    target: "Target · None",
    action: "Action · No combat",
    legality: "Legality · Not applicable",
  });
  assert.doesNotMatch(Object.values(facts).join(" "), /target-none|Lane 0\/B/u);
  assert.equal(Object.isFrozen(facts), true);
});

test("pending combat copy requires the same-root resolver for its public target", () => {
  const pending = {
    move_action: 5,
    movement_mask_value: false,
    target_action: 3,
    target: { disclosure: "public", global_slot: 7, target_action: 3 },
    armed_lane: 1,
    pair_mask_value: false,
  };
  assert.deepEqual(pendingActionDisplayFacts(pending, PUBLIC_IDS), {
    movement: "Movement · Northeast (5) · Unavailable",
    target: "Target · Agent ID seven:semicolon; (action 3)",
    action: "Action · Ultimate (1/U)",
    legality: "Legality · Unavailable",
  });
  assert.equal(
    pendingActionDisplayFacts(pending).target,
    "Target · Agent ID unavailable",
  );
});

test("pending combat copy retains the envelope target action", () => {
  const facts = pendingActionDisplayFacts(
    {
      move_action: 3,
      movement_mask_value: true,
      target_action: 6,
      target: { disclosure: "public", global_slot: 6 },
      armed_lane: null,
      pair_mask_value: null,
    },
    new Map([[6, "opaque-six"]]),
  );
  assert.equal(facts.target, "Target · Agent ID opaque-six (action 6)");
  assert.doesNotMatch(facts.target, /unavailable|id_6/u);
});

test("pending public ID and slot must agree when both are supplied", () => {
  const facts = pendingActionDisplayFacts(
    {
      move_action: 0,
      movement_mask_value: true,
      target: {
        disclosure: "public",
        global_slot: 7,
        target_action: 3,
        public_agent_id: "mismatched-public-id",
      },
      armed_lane: 0,
      pair_mask_value: true,
    },
    PUBLIC_IDS,
  );
  assert.equal(facts.target, "Target · Agent ID unavailable");
  assert.doesNotMatch(JSON.stringify(facts), /mismatched-public-id|id_7/u);
});

test("pending target-none retains an explicitly armed source-local Ultimate", () => {
  assert.deepEqual(
    pendingActionDisplayFacts({
      move_action: 0,
      movement_mask_value: true,
      target: { target_action: 0, public_agent_id: null },
      armed_lane: 1,
      pair_mask_value: true,
    }),
    {
      movement: "Movement · Stay (0) · Available",
      target: "Target · None",
      action: "Action · Ultimate (1/U)",
      legality: "Legality · Available",
    },
  );
});

test("actor-POV action tuples recognize the recipient-local no-target action", () => {
  assert.equal(
    actionTupleCombatLabel({
      target: { target_action: 0, public_agent_id: null },
      use_ultimate_action: 0,
    }),
    "No combat",
  );
  assert.equal(
    actionTupleCombatLabel({
      target: { target_action: 3, public_agent_id: "visible-target" },
      use_ultimate_action: 0,
    }),
    "0/B · Basic",
  );
  assert.equal(
    actionTupleCombatLabel({
      target: { target_action: 3, public_agent_id: "visible-target" },
      use_ultimate_action: 1,
    }),
    "1/U · Ultimate",
  );
  assert.equal(
    actionTupleCombatLabel({
      target: { target_action: 0, public_agent_id: null },
      use_ultimate_action: 1,
    }),
    "1/U · Ultimate",
  );
});
