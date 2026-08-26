import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeAuthorizedPresentationFrameV1 } from "../src/authorized-presentation-normalizer.js";
import {
  explainClassDocumentation,
  explainTechnicalFact,
} from "../src/explanations.js";
import {
  authorizedInspectorView,
  DebuggerPanels,
  disclosurePanelInitiallyOpen,
  rosterStatusDurationLabel,
} from "../src/panels.js";

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

/** @param {string} kind */
async function normalizedCompatibilityCase(kind) {
  return await normalizeAuthorizedPresentationFrameV1(
    authorizedFixture.compatibility_cases[kind],
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
    let tabIndex = tagName.toLowerCase() === "button" ? 0 : -1;
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
    Object.defineProperty(node, "tabIndex", {
      configurable: true,
      enumerable: true,
      get() {
        return tabIndex;
      },
      set(value) {
        tabIndex = Number(value);
        attributes.set("tabindex", String(tabIndex));
      },
    });
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

/** @param {Record<string, any>} descriptor @param {string} title */
function semanticSection(descriptor, title) {
  const found = descriptor.sections.find(
    (/** @type {Record<string, any>} */ section) => section.title === title,
  );
  assert.ok(found, `missing semantic section ${title}`);
  return found;
}

test("authorized inspector rejects raw and forged presentation roots", async () => {
  const raw = authorizedFixture.presentations.replay_oracle;
  const normalized = await normalizedPresentation("replay_oracle");
  const forged = Object.freeze({ ...normalized });

  assert.equal(authorizedInspectorView(raw), null);
  assert.equal(authorizedInspectorView(forged), null);
  assert.equal(authorizedInspectorView(null), null);
});

test("authorized replay inspector keeps researcher selection separate from transition panels", async () => {
  const cases = [
    "replay_oracle",
    "replay_no_shared_obs_agent_pov",
    "replay_shared_obs_agent_pov",
  ];
  const documentationBytes = new Set();

  for (const kind of cases) {
    const presentation = await normalizedPresentation(kind);
    const inspector = authorizedInspectorView(presentation);
    const researcher = presentation.authority.authority_kind === "agent_pov";
    const expectedOwner = researcher
      ? presentation.researcher_space.roster_agents.find(
          (/** @type {Record<string, any>} */ agent) =>
            agent.public_agent_id ===
            presentation.researcher_space.selected_public_agent_id,
        )
      : presentation.inspection;
    assert.ok(expectedOwner);
    assert.ok(inspector);
    assert.equal(inspector.title, "Comprehensive Agent Class Details");
    assert.equal(
      inspector.owner.presentation_key,
      expectedOwner.presentation_key ?? expectedOwner.actor_presentation_key,
    );
    assert.equal(
      inspector.owner.public_agent_id,
      expectedOwner.public_agent_id ?? expectedOwner.actor_public_agent_id,
    );
    assert.equal(
      inspector.owner_descriptor.title,
      "Agent ID agent-slot-0 · Mage · Team A",
    );
    assert.equal(inspector.owner_descriptor.summary, null);
    assert.deepEqual(inspector.owner_descriptor.rows, []);
    assert.deepEqual(
      inspector.owner_descriptor.sections.map(
        (/** @type {Record<string, any>} */ section) => section.title,
      ),
      ["Class Overview", "Authored Tactical Guide", "Class Mechanics"],
    );
    assert.equal(
      semanticSection(inspector.owner_descriptor, "Class Overview").summary,
      "The Mage is a fragile backline damage dealer that creates explosive ranged-damage windows with Burst and relies on allied protection to operate.",
    );
    assert.deepEqual(
      semanticSection(inspector.owner_descriptor, "Authored Tactical Guide").rows.map(
        (/** @type {Record<string, any>} */ row) => row.label,
      ),
      ["Role", "Primary Strength", "Primary Weakness", "Counters", "Countered By"],
    );
    assert.deepEqual(
      semanticSection(inspector.owner_descriptor, "Class Mechanics").rows.map(
        (/** @type {Record<string, any>} */ row) => row.label,
      ),
      [
        "Maximum Health",
        "Body Radius",
        "Base Movement Speed",
        "Observation Radius",
        "Basic Target",
        "Basic Ability Radius",
        "Basic Raw Damage",
        "Out-of-combat Delay",
        "Out-of-combat Regeneration",
        "Ultimate Name",
        "Ultimate Description",
        "Ultimate Target",
        "Ultimate Radius",
        "Ultimate Cooldown",
        "Passive Name",
        "Passive Description",
      ],
    );
    assert.doesNotMatch(
      JSON.stringify(inspector.owner_descriptor),
      /Current Health|Effective Speed|Ultimate Status|Combat Status|Steps until out of combat|Selection|Target legality|Transition/iu,
    );
    documentationBytes.add(JSON.stringify(inspector.owner_descriptor));
    assert.equal(inspector.owner_class_accent, "mage");
    assert.deepEqual(Object.keys(inspector), [
      "title",
      "owner",
      "owner_descriptor",
      "owner_class_accent",
    ]);
    assert.equal(Object.isFrozen(inspector), true);
  }
  assert.equal(documentationBytes.size, 1);
});

test("all five certified class cards use the exact documentation section contract", () => {
  const scene = authorizedFixture.presentations.replay_oracle.current_endpoint.scene;
  const expectedClasses = new Map([
    [1, "Mage"],
    [2, "Warrior"],
    [3, "Hunter"],
    [4, "Rogue"],
    [5, "Priest"],
  ]);
  const expectedDescriptions = new Map([
    [
      1,
      [
        "For 5 Ticks, Burst multiplies this Mage's outgoing damage by a factor of 1.5 (50% more damage dealt), beginning with the successor decision.",
        "An eligible unshielded Mage emits Sorcerer's Empowerment. Eligible unshielded same-team agents within a radius of 2, including the Mage, receive a 15% outgoing-damage increase per recorded emitter; overlapping emitters multiply up to 1.32.",
      ],
    ],
    [
      2,
      [
        "Charge moves the Warrior toward an enemy target during the Charge phase before ordinary movement. The accepted ultimate also applies 20 raw damage before source and recipient damage modifiers, 1 Tick of stun, and a 50% movement reduction (×0.5) for 5 Ticks.",
        "An eligible unshielded Warrior emits Guardian's Barrier. Eligible unshielded same-team agents within a radius of 2, including the Warrior, receive a 15% incoming-damage reduction per recorded emitter; overlapping emitters multiply down to 0.72.",
      ],
    ],
    [
      3,
      [
        "Freezing Trap applies 10 raw damage to an enemy target before source and recipient damage modifiers and applies a stun for 4 Ticks. Accepted positive raw damage ends an existing trap before any same-transition reapplication.",
        "Every accepted Hunter basic applies Serrated Arrows for 1 Tick, imposing a 15% movement reduction (×0.85). Later accepted Hunter basics refresh the remaining duration.",
      ],
    ],
    [
      4,
      [
        "Crippling Poison applies 36 raw damage to an enemy target before source and recipient damage modifiers, a stun for 1 Tick, a 50% movement reduction (×0.5) for 5 Ticks, and a 50% reduction (×0.5) to incoming healing and out-of-combat regeneration for 4 Ticks.",
        "This Rogue's base movement speed of 1.3 is the highest in the certified profile. After 3 Ticks without combat participation, it becomes eligible for the displayed Out-of-combat Regeneration on each transition tick.",
      ],
    ],
    [
      5,
      [
        "Holy Word: Salvation applies 200 raw healing to a same-team target before recipient healing modifiers and maximum-health clamping.",
        "Every accepted Priest basic applies Blessing of Freedom to its same-team target, including the Priest where same-team targeting permits it, for 1 Tick. Freedom limits how far slow effects can reduce ordinary movement, using a floor of 85% of base movement speed (×0.85); it does not override stun.",
      ],
    ],
  ]);
  for (const [classId, className] of expectedClasses) {
    const owner = scene.agents.find(
      (/** @type {Record<string, any>} */ agent) => agent.class_id === classId,
    );
    const mechanics = scene.class_mechanics.find(
      (/** @type {Record<string, any>} */ row) => row.class_id === classId,
    );
    assert.ok(owner);
    assert.ok(mechanics);
    const descriptor = explainClassDocumentation(owner, mechanics);
    assert.ok(descriptor);
    assert.equal(
      descriptor.title,
      `Agent ID ${owner.public_agent_id} · ${className} · ${owner.team_id === 1 ? "Team A" : "Team B"}`,
    );
    assert.equal(descriptor.summary, null);
    assert.deepEqual(
      descriptor.sections.map((section) => section.title),
      ["Class Overview", "Authored Tactical Guide", "Class Mechanics"],
    );
    assert.equal(semanticSection(descriptor, "Authored Tactical Guide").rows.length, 5);
    const mechanicsRows = semanticSection(descriptor, "Class Mechanics").rows;
    const mechanicsLabels = mechanicsRows.map(
      (/** @type {Record<string, any>} */ row) => row.label,
    );
    for (const label of [
      "Maximum Health",
      "Body Radius",
      "Base Movement Speed",
      "Observation Radius",
      "Basic Target",
      "Basic Ability Radius",
      "Out-of-combat Delay",
      "Out-of-combat Regeneration",
      "Ultimate Name",
      "Ultimate Description",
      "Ultimate Target",
      "Ultimate Radius",
      "Ultimate Cooldown",
      "Passive Name",
      "Passive Description",
    ]) {
      assert.equal(mechanicsLabels.includes(label), true, `${className}: ${label}`);
    }
    const descriptions = expectedDescriptions.get(classId);
    assert.ok(descriptions);
    assert.equal(
      mechanicsRows.find(
        (/** @type {Record<string, any>} */ row) =>
          row.label === "Ultimate Description",
      )?.value,
      descriptions[0],
    );
    assert.equal(
      mechanicsRows.find(
        (/** @type {Record<string, any>} */ row) => row.label === "Passive Description",
      )?.value,
      descriptions[1],
    );
    assert.doesNotMatch(JSON.stringify(descriptor), /\{\{|Unavailable/u);
    assert.equal(Object.isFrozen(descriptor), true);
  }
});

test("class documentation fails closed for unavailable, V1, missing, and mismatched joins", async () => {
  const scene = authorizedFixture.presentations.replay_oracle.current_endpoint.scene;
  const owner = scene.agents.find(
    (/** @type {Record<string, any>} */ agent) => agent.class_id === 1,
  );
  const mechanics = scene.class_mechanics.find(
    (/** @type {Record<string, any>} */ row) => row.class_id === 1,
  );
  assert.ok(owner);
  assert.ok(mechanics);

  const mechanicsOnlyCases = [
    {
      ...mechanics,
      documentation_profile: { availability_kind: "unavailable" },
    },
    {
      ...mechanics,
      documentation_profile: {
        availability_kind: "available",
        profile_id: "future-profile",
      },
    },
    {
      ...mechanics,
      status_mechanics: mechanics.status_mechanics.filter(
        (/** @type {Record<string, any>} */ status) =>
          status.status_id !== "mage_burst_damage_amplification",
      ),
    },
  ];
  const legacyMechanics = { ...mechanics };
  delete legacyMechanics.mechanics_version;
  delete legacyMechanics.documentation_profile;
  mechanicsOnlyCases.push(legacyMechanics);

  for (const candidate of mechanicsOnlyCases) {
    const descriptor = explainClassDocumentation(owner, candidate);
    assert.ok(descriptor);
    assert.deepEqual(
      descriptor.sections.map((section) => section.title),
      ["Class Mechanics"],
    );
    assert.doesNotMatch(
      JSON.stringify(descriptor),
      /Unavailable|Class Overview|Authored Tactical Guide|Ultimate Name|Ultimate Description|Passive Name|Passive Description/u,
    );
  }

  assert.equal(explainClassDocumentation({ ...owner, class_id: 2 }, mechanics), null);
  assert.equal(
    explainClassDocumentation(owner, { ...mechanics, class_name: "Warrior" }),
    null,
  );
  assert.equal(explainClassDocumentation(owner, null), null);

  const legacyPresentation = await normalizedCompatibilityCase("legacy_v1");
  const legacyInspector = authorizedInspectorView(legacyPresentation);
  assert.ok(legacyInspector?.owner_descriptor);
  assert.deepEqual(
    legacyInspector.owner_descriptor.sections.map(
      (/** @type {Record<string, any>} */ section) => section.title,
    ),
    ["Class Mechanics"],
  );
});

test("current and action state are byte-inert to certified class documentation", () => {
  const scene = authorizedFixture.presentations.replay_oracle.current_endpoint.scene;
  const owner = scene.agents.find(
    (/** @type {Record<string, any>} */ agent) => agent.class_id === 4,
  );
  const mechanics = scene.class_mechanics.find(
    (/** @type {Record<string, any>} */ row) => row.class_id === 4,
  );
  assert.ok(owner);
  assert.ok(mechanics);
  const baseline = explainClassDocumentation(owner, mechanics);
  const injected = explainClassDocumentation(
    {
      ...owner,
      current_health: -999,
      ultimate_cooldown_remaining: 999,
      statuses: [{ secret: "current-status" }],
      aura_modifiers: [{ secret: "current-aura" }],
      selected: true,
      target: "secret-target",
      legality: "secret-legality",
      transition: "secret-transition",
    },
    {
      ...mechanics,
      current_health: -999,
      ultimate_cooldown_remaining: 999,
      statuses: [{ secret: "current-status" }],
      selection: "secret-selection",
      target: "secret-target",
      legality: "secret-legality",
      transition: "secret-transition",
    },
  );
  assert.equal(JSON.stringify(injected), JSON.stringify(baseline));
  assert.doesNotMatch(
    JSON.stringify(injected),
    /current-|secret-|999|Selection|Target legality/iu,
  );
});

test("authorized replay inspector retains final selected owner without outgoing legality", async () => {
  for (const kind of [
    "replay_oracle_final_selected",
    "replay_no_shared_final",
    "replay_shared_final",
  ]) {
    const presentation = await normalizedStateCase(kind);
    const inspector = authorizedInspectorView(presentation);
    const researcher = presentation.authority.authority_kind === "agent_pov";
    const expectedOwner = researcher
      ? presentation.researcher_space.roster_agents.find(
          (/** @type {Record<string, any>} */ agent) =>
            agent.public_agent_id ===
            presentation.researcher_space.selected_public_agent_id,
        )
      : presentation.action_axis;
    assert.ok(expectedOwner);
    assert.ok(inspector);
    assert.equal(
      inspector.owner.presentation_key,
      expectedOwner.presentation_key ?? expectedOwner.owner_presentation_key,
    );
    assert.equal(
      inspector.owner.public_agent_id,
      expectedOwner.public_agent_id ?? expectedOwner.owner_public_agent_id,
    );
    assert.equal(
      inspector.owner_descriptor.title,
      "Agent ID agent-slot-0 · Mage · Team A",
    );
    assert.equal(inspector.owner_class_accent, "mage");
    assert.deepEqual(Object.keys(inspector), [
      "title",
      "owner",
      "owner_descriptor",
      "owner_class_accent",
    ]);
  }
});

test("authorized live inspector keeps battlefield geometry out of Agent panels", async () => {
  for (const kind of ["live_oracle", "live_no_shared_obs_agent_pov"]) {
    const presentation = await normalizedPresentation(kind);
    const inspector = authorizedInspectorView(presentation);
    assert.ok(inspector);
    assert.deepEqual(Object.keys(inspector), [
      "title",
      "owner",
      "owner_descriptor",
      "owner_class_accent",
    ]);
    if (kind === "live_no_shared_obs_agent_pov") {
      assert.equal(
        inspector.owner.public_agent_id,
        presentation.researcher_space.selected_public_agent_id,
      );
      assert.equal(JSON.stringify(inspector).includes('"position"'), false);
    }
  }
});

test("authorized replay inspector leaves final unselected details absent", async () => {
  const presentation = await normalizedStateCase("replay_oracle_final_unselected");
  const inspector = authorizedInspectorView(presentation);
  assert.ok(inspector);
  assert.equal(inspector.title, "Comprehensive Agent Class Details");
  assert.equal(inspector.owner, null);
  assert.equal(inspector.owner_descriptor, null);
  assert.equal(inspector.owner_class_accent, null);
  assert.deepEqual(Object.keys(inspector), [
    "title",
    "owner",
    "owner_descriptor",
    "owner_class_accent",
  ]);
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
    const selectionCard = binding();
    const pendingCard = binding();
    const pendingHeading = binding();
    const pendingCount = binding();
    const pendingScope = binding();
    const diagnosticsCard = binding();
    const acceptedCard = binding();
    const acceptedAnnouncement = binding();
    const rosterCount = binding();
    const panels = new DebuggerPanels({
      roster,
      rosterCount,
      selectionCard,
      pendingHeading,
      pendingCount,
      pendingScope,
      pendingCard,
      acceptedCard,
      acceptedAnnouncement,
      diagnosticsCard,
      onCommand: (command) => {
        commands.push(command);
      },
    });
    const liveOracle = await normalizedPresentation("live_oracle");
    panels.renderAuthorizedRoster(liveOracle, false);
    const liveOracleRosterCount = liveOracle.scene.agents.length;
    assert.equal(rosterCount.textContent, `${liveOracleRosterCount} actors`);
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

    const liveAgent = await normalizedPresentation("live_no_shared_obs_agent_pov");
    panels.renderAuthorizedInspector(liveAgent);
    assert.equal(pendingHeading.textContent, "Pending Joint Action");
    assert.equal(pendingScope.hidden, false);
    assert.equal(
      pendingScope.textContent,
      "This panel shows the complete researcher-space joint action staged for the next submission.",
    );
    assert.equal(
      pendingCount.textContent,
      `${liveAgent.researcher_space.pending_joint_action.action_rows.length} actors`,
    );
    const pendingRows = dom
      .descendants(pendingCard)
      .filter((node) => node.className === "accepted-action-row");
    assert.equal(
      pendingRows.length,
      liveAgent.researcher_space.pending_joint_action.action_rows.length,
    );
    assert.equal(
      pendingRows.every((row) => {
        const text = dom.textTree(row);
        const tuples = dom
          .descendants(row)
          .filter((node) => node.className === "accepted-action-tuple");
        const comparisons = dom
          .descendants(row)
          .filter((node) => node.className === "accepted-action-row__comparison");
        return (
          /^team-[ab]$/u.test(row.dataset.team) &&
          tuples.length === 1 &&
          tuples[0].dataset.kind === "pending" &&
          comparisons.length === 1 &&
          comparisons[0].dataset.layout === "single" &&
          text.includes("Pending") &&
          !/Submitted|Accepted|Current Legality|Upcoming Target/u.test(text)
        );
      }),
      true,
    );
    const scriptedRaw = structuredClone(
      authorizedFixture.presentations.live_no_shared_obs_agent_pov,
    );
    const scriptedInspection = {
      inspection_kind: "scripted_playback_inspection",
      submission_scope: "scripted_playback",
      editable_draft_available: false,
      advance_semantics: "registered_script_frame",
    };
    scriptedRaw.source.source_submission_scope = "scripted_playback";
    scriptedRaw.live_inspection.inspection = structuredClone(scriptedInspection);
    scriptedRaw.researcher_space.pending_inspection =
      structuredClone(scriptedInspection);
    scriptedRaw.researcher_space.pending_joint_action = null;
    const scriptedAgent = await normalizeAuthorizedPresentationFrameV1(scriptedRaw);
    panels.renderAuthorizedInspector(scriptedAgent);
    assert.equal(pendingHeading.textContent, "Scripted playback inspection");
    assert.equal(pendingCount.textContent, "0 actors");
    assert.equal(pendingCount.hidden, false);
    assert.equal(
      pendingScope.textContent,
      "This live frame advances registered scripted actions and has no editable draft.",
    );
    assert.equal(
      dom
        .descendants(pendingCard)
        .filter((node) => node.className === "accepted-action-row").length,
      0,
    );
    assert.equal(
      dom.textTree(pendingCard),
      "No editable joint action is available during scripted playback.",
    );
    panels.renderAuthorizedInspector(liveAgent);
    assert.equal(
      dom
        .descendants(acceptedCard)
        .filter((node) => node.className === "accepted-action-row").length,
      liveAgent.researcher_space.latest_transition.action_rows.length,
    );
    panels.renderAuthorizedRoster(liveAgent, false);
    const liveAgentRows = /** @type {Record<string, any>[]} */ (
      [...panels.rosterRows.values()].filter(
        (/** @type {Record<string, any>} */ row) => "primaryButton" in row,
      )
    );
    assert.equal(liveAgentRows.length, liveAgent.researcher_space.roster_agents.length);
    const liveAgentVisibleCount = liveAgentRows.filter(
      (row) => row.element.dataset.visibleInSnapshot === "true",
    ).length;
    assert.equal(
      rosterCount.textContent,
      `${liveAgentRows.length} agents · ${liveAgentVisibleCount} visible · ${liveAgentRows.length - liveAgentVisibleCount} not visible`,
    );
    assert.equal(
      [...panels.rosterTeamGroups.values()].every(
        (group) => group.element.hidden === true,
      ),
      true,
    );
    assert.equal(
      [...panels.rosterVisibilityGroups.values()].every(
        (group) => group.element.hidden === false,
      ),
      true,
    );
    assert.equal(
      liveAgentRows.every((row) =>
        ["true", "false"].includes(row.element.dataset.visibleInSnapshot),
      ),
      true,
    );
    assert.equal(
      liveAgentRows.every(
        (row) =>
          row.primaryButton.disabled === false &&
          !Object.hasOwn(row.primaryButton.dataset, "globalSlot") &&
          !Object.hasOwn(row.primaryButton.dataset, "commandSlot"),
      ),
      true,
    );
    const visibleLivePublicIds = new Set(
      liveAgent.scene.agents.map(
        (/** @type {Record<string, any>} */ agent) => agent.public_agent_id,
      ),
    );
    const hiddenLiveAgent = liveAgent.researcher_space.roster_agents.find(
      (/** @type {Record<string, any>} */ agent) =>
        !visibleLivePublicIds.has(agent.public_agent_id),
    );
    assert.ok(hiddenLiveAgent);
    const hiddenLiveRow = liveAgentRows.find(
      (row) => row.element.dataset.presentationKey === hiddenLiveAgent.presentation_key,
    );
    assert.ok(hiddenLiveRow);
    hiddenLiveRow.primaryButton.click();
    assert.deepEqual(commands.at(-1), {
      command_type: "activate_live_pov_agent",
      global_slot:
        (hiddenLiveRow.element.dataset.teamId === "1" ? 0 : 5) +
        Number(
          liveAgent.researcher_space.roster_agents.find(
            (/** @type {Record<string, any>} */ agent) =>
              agent.presentation_key === hiddenLiveRow.element.dataset.presentationKey,
          ).team_local_slot,
        ),
    });

    const replayAgent = await normalizedPresentation("replay_no_shared_obs_agent_pov");
    panels.renderAuthorizedInspector(replayAgent);
    assert.equal(pendingHeading.textContent, "Upcoming Transition");
    assert.equal(pendingScope.hidden, true);
    const upcomingRows = dom
      .descendants(pendingCard)
      .filter((node) => node.className === "accepted-action-row");
    assert.equal(upcomingRows.length, 5);
    assert.match(dom.textTree(upcomingRows[0]), /Submitted.*Accepted/u);
    assert.equal(
      upcomingRows.every((row) => /^team-[ab]$/u.test(row.dataset.team)),
      true,
    );
    assert.match(
      dom.textTree(selectionCard),
      /Class Overview.*Authored Tactical Guide.*Class Mechanics/u,
    );
    assert.equal(
      dom
        .descendants(selectionCard)
        .some((node) =>
          ["selected-outgoing-target", "selected-legality"].includes(node.className),
        ),
      false,
    );
    assert.equal(
      dom
        .descendants(pendingCard)
        .some((node) => node.className === "selected-outgoing-target"),
      false,
    );
    assert.equal(
      dom
        .descendants(pendingCard)
        .some((node) => node.className === "selected-legality"),
      false,
    );
    panels.renderAuthorizedRoster(replayAgent, false);
    const agentRows = /** @type {Record<string, any>[]} */ (
      [...panels.rosterRows.values()].filter(
        (/** @type {Record<string, any>} */ row) => "primaryButton" in row,
      )
    );
    assert.equal(agentRows.length, 5);
    assert.equal(
      rosterCount.textContent,
      `${agentRows.length} agents · ${agentRows.filter((row) => row.element.dataset.visibleInSnapshot === "true").length} visible · ${agentRows.filter((row) => row.element.dataset.visibleInSnapshot === "false").length} not visible`,
    );
    assert.match(dom.textTree(roster), /VISIBLE.*NOT VISIBLE/u);
    assert.equal(
      [...panels.rosterVisibilityGroups.values()].every(
        (group) => group.element.hidden === false,
      ),
      true,
    );
    const agentModifierOwners = agentRows.flatMap((row) =>
      dom
        .descendants(row.element)
        .filter((node) => node.className.includes("roster-fact-token--modifier")),
    );
    assert.deepEqual(agentModifierOwners.map((node) => node.dataset.tokenId).sort(), [
      "mage_amplification",
      "warrior_mitigation",
    ]);
    assert.deepEqual(
      agentModifierOwners.map((node) => node.dataset.multiplier).sort(),
      ["0.8500000238418579", "1.149999976158142"],
    );
    for (const modifierOwner of agentModifierOwners) {
      assert.doesNotMatch(
        modifierOwner.getAttribute("aria-label"),
        /source|agent-slot/iu,
      );
    }
    assert.equal(
      agentRows.every((row) => row.primaryButton.disabled === false),
      true,
    );
    agentRows[0].primaryButton.click();
    const latestCommand = /** @type {Record<string, any> | undefined} */ (
      commands.at(-1)
    );
    assert.ok(latestCommand);
    assert.deepEqual(Object.keys(latestCommand).sort(), [
      "command_type",
      "global_slot",
    ]);
    assert.equal(latestCommand.command_type, "activate_replay_pov_agent");
    assert.equal(latestCommand.global_slot, 0);
    panels.renderAuthorizedRoster(replayAgent, true);
    assert.equal(
      agentRows.every((row) => row.primaryButton.disabled === true),
      true,
    );
    const commandCount = commands.length;
    agentRows[0].primaryButton.click();
    assert.equal(commands.length, commandCount);

    const technicalCases = [
      [
        "presentation",
        "live_oracle",
        ["episode", "frame", "simulator_step", "incoming_transition"],
      ],
      [
        "presentation",
        "live_no_shared_obs_agent_pov",
        ["episode", "frame", "simulator_step", "incoming_transition"],
      ],
      [
        "presentation",
        "replay_oracle",
        [
          "artifact_digest_prefix",
          "frame",
          "simulator_step",
          "incoming_transition",
          "ordinary_movement_distance_scale",
        ],
      ],
      [
        "presentation",
        "replay_no_shared_obs_agent_pov",
        ["frame", "simulator_step", "incoming_transition"],
      ],
      [
        "presentation",
        "replay_shared_obs_agent_pov",
        ["frame", "simulator_step", "incoming_transition"],
      ],
      ["state", "live_oracle_frame_zero", ["episode", "frame", "simulator_step"]],
      ["state", "live_no_shared_frame_zero", ["episode", "frame", "simulator_step"]],
      [
        "state",
        "replay_oracle_frame_zero",
        [
          "artifact_digest_prefix",
          "frame",
          "simulator_step",
          "ordinary_movement_distance_scale",
        ],
      ],
      ["state", "replay_no_shared_frame_zero", ["frame", "simulator_step"]],
      ["state", "replay_shared_frame_zero", ["frame", "simulator_step"]],
    ];
    for (const [source, kind, expectedFactIds] of technicalCases) {
      const presentation =
        source === "presentation"
          ? await normalizedPresentation(String(kind))
          : await normalizedStateCase(String(kind));
      panels.renderAuthorizedInspector(presentation);
      const owners = dom
        .descendants(diagnosticsCard)
        .filter((node) => Object.hasOwn(node.dataset, "technicalFact"));
      assert.deepEqual(
        owners.map((owner) => owner.dataset.technicalFact),
        expectedFactIds,
        String(kind),
      );
      assert.equal(
        owners.every((owner) => {
          const help = explainTechnicalFact(owner.dataset.technicalFact);
          return (
            owner.tabIndex === 0 &&
            owner.hasAttribute("data-tooltip-owner") &&
            owner.getAttribute("aria-description") === help.summary &&
            dom.textTree(owner).trim().length > 0 &&
            !/undefined|null/u.test(dom.textTree(owner))
          );
        }),
        true,
        String(kind),
      );
    }

    const replayOracle = await normalizedPresentation("replay_oracle");
    panels.renderAuthorizedInspector(replayOracle);
    const transitionRows = dom
      .descendants(acceptedCard)
      .filter((node) => node.className === "accepted-action-row");
    assert.equal(transitionRows.length, 5);
    assert.equal(
      transitionRows.every((row) => {
        const text = dom.textTree(row);
        return (
          /Agent ID .+ · (?:Mage|Warrior|Hunter|Rogue|Priest) · Team [AB]/u.test(
            text,
          ) &&
          text.includes("Submitted") &&
          text.includes("Accepted") &&
          !/(?:mask|combat result|submission|revision|generation|transition)/iu.test(
            text.replace("Submitted", ""),
          )
        );
      }),
      true,
    );
    assert.equal(
      transitionRows.every((row) => {
        const titles = dom
          .descendants(row)
          .filter((node) => node.className === "accepted-action-row__title");
        return (
          titles.length === 1 &&
          ["mage", "warrior", "hunter", "rogue", "priest"].includes(
            titles[0].dataset.class,
          ) &&
          !Object.hasOwn(row.dataset, "presentationKey") &&
          !Object.hasOwn(row.dataset, "transitionId") &&
          /^team-[ab]$/u.test(row.dataset.team)
        );
      }),
      true,
    );
    assert.equal(Object.hasOwn(acceptedCard.dataset, "transitionId"), false);

    const frameZero = await normalizedStateCase("replay_oracle_frame_zero");
    panels.render(frameZero);
    assert.equal(acceptedCard.children.length, 0);
    assert.equal(acceptedAnnouncement.textContent, "");

    panels.render(replayOracle);
    assert.notEqual(panels.rosterRows.size, 0);
    panels.render({ ...replayOracle });
    assert.equal(panels.rosterRows.size, 0);
    assert.equal(acceptedCard.children.length, 0);
    assert.equal(acceptedAnnouncement.textContent, "");
    assert.doesNotMatch(
      [
        dom.textTree(roster),
        dom.textTree(selectionCard),
        dom.textTree(pendingCard),
        dom.textTree(acceptedCard),
        dom.textTree(diagnosticsCard),
      ].join(" "),
      /Submitted|Accepted|Technical Frame.*\d|Observation delta|Incoming event/u,
    );
  } finally {
    if (documentDescriptor === undefined) {
      Reflect.deleteProperty(globalThis, "document");
    } else {
      Object.defineProperty(globalThis, "document", documentDescriptor);
    }
  }
});

test("panel source has one branded render path and no raw fallbacks", async () => {
  const [source, styles] = await Promise.all([
    readFile(new URL("../src/panels.js", import.meta.url), "utf8"),
    readFile(new URL("../styles.css", import.meta.url), "utf8"),
  ]);
  for (const forbidden of [
    "compactRoster",
    "data-compact",
    "frameScene",
    "frameEvents",
    "publicAgentIdMap",
    "rosterControlDescriptor",
    "pendingActionDisplayFacts",
    "actionTupleCombatLabel",
    "eventSummary",
    "eventDescriptor",
    "replayDiagnosticFacts",
    "createRosterRow",
    "updateRosterRow",
    "renderRoster",
    "renderPendingPlan",
    "renderDiagnostics",
  ]) {
    assert.equal(source.includes(forbidden), false, forbidden);
  }
  assert.match(
    source,
    /if \(isAuthorizedPresentationFrame\(frame\)\)[\s\S]*this\.renderAuthorizedInspector\([\s\S]*return;\s*\}\s*this\.renderUnavailable\(\);/u,
  );
  assert.doesNotMatch(source, /renderAuthorizedEvents|eventFeed|eventCount/u);
  for (const forbiddenSelector of [
    ".roster-actions",
    ".action-result",
    ".action-tuple",
    ".pending-action-list",
    ".pending-action-row",
    ".pending-action-chip",
    ".diagnostic-fact",
    ".action-card__label",
    ".selected-outgoing-target",
    ".selected-legality",
    ".roster-row:focus-visible",
    ".event-item strong",
  ]) {
    assert.equal(styles.includes(forbiddenSelector), false, forbiddenSelector);
  }
});

test("authorized inspector copy separates live draft and replay upcoming epochs", async () => {
  const source = await readFile(new URL("../src/panels.js", import.meta.url), "utf8");
  assert.match(source, /complete researcher-space joint action staged/u);
  assert.match(source, /authorized recorded actions out of the current frame/u);
  assert.match(source, /Pending Joint Action/u);
  assert.match(source, /Upcoming Transition/u);
  assert.doesNotMatch(source, /Current Legality|Upcoming Target/u);
  assert.doesNotMatch(
    source,
    /Recorded outgoing inspection|No (?:authorized )?outgoing inspection/u,
  );
  assert.doesNotMatch(source, /Settled range, target, and legality overlays/u);
});

test("native disclosure defaults open only operational live panels and roster", () => {
  assert.equal(disclosurePanelInitiallyOpen("command-deck", false), true);
  assert.equal(disclosurePanelInitiallyOpen("command-deck", true), false);
  assert.equal(disclosurePanelInitiallyOpen("roster-details", false), true);
  assert.equal(disclosurePanelInitiallyOpen("roster-details", true), true);
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

test("roster duration labels abbreviate only the human-facing extreme value", () => {
  assert.equal(rosterStatusDurationLabel(5), "5");
  assert.equal(rosterStatusDurationLabel(123456789), "123M");
  assert.equal(rosterStatusDurationLabel(null), "?");
});
