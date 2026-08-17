import assert from "node:assert/strict";
import test from "node:test";

import { exactAuthorizedAgentIdentityV1 } from "../src/agent-identity.js";
import { authorizedSourceAttributionV1 } from "../src/source-attribution.js";

/**
 * @param {string} presentationKey
 * @param {string} publicAgentId
 * @param {1 | 2 | 3 | 4 | 5} classId
 * @param {1 | 2} teamId
 */
function agent(presentationKey, publicAgentId, classId, teamId) {
  const className = {
    1: "Mage",
    2: "Warrior",
    3: "Hunter",
    4: "Rogue",
    5: "Priest",
  }[classId];
  return {
    presentation_key: presentationKey,
    public_agent_id: publicAgentId,
    class_id: classId,
    class_name: className,
    team_id: teamId,
    position: [999, 999],
    global_slot: 999,
    hidden_event_id: "must-not-be-read",
  };
}

/** @param {string} presentationKey @param {string} publicAgentId */
function source(presentationKey, publicAgentId) {
  return {
    source_presentation_key: presentationKey,
    source_public_agent_id: publicAgentId,
  };
}

/**
 * @param {Partial<{
 *   attribution_kind: "direct" | "aggregate_aura" | "spawn_shield",
 *   audience: "researcher" | "agent_pov",
 *   direct_sources: unknown,
 *   authorized_agents: unknown,
 * }>} [overrides]
 */
function options(overrides = {}) {
  return {
    attribution_kind: "direct",
    audience: "researcher",
    direct_sources: [source("source:m1", "alpha/9001")],
    authorized_agents: [agent("source:m1", "alpha/9001", 1, 1)],
    ...overrides,
  };
}

test("one direct source uses the exact canonical public identity and no raw fields", () => {
  const result = authorizedSourceAttributionV1(options());
  assert.deepEqual(result, {
    state: "single",
    label: "Source",
    value: "Agent ID alpha/9001 · Mage · Team A",
    text: "Source: Agent ID alpha/9001 · Mage · Team A",
  });
  assert.equal(Object.isFrozen(result), true);
  assert.doesNotMatch(JSON.stringify(result), /source:m1|999|must-not-be-read/u);
});

test("multiple sources preserve serialized first occurrence and dedupe exact repeats", () => {
  const first = source("source:r4", "beta.17");
  const second = source("source:h3", "alpha/9001");
  const result = authorizedSourceAttributionV1(
    options({
      direct_sources: [first, second, { ...first }, { ...second }],
      authorized_agents: [
        agent("source:h3", "alpha/9001", 3, 1),
        agent("source:r4", "beta.17", 4, 2),
      ],
    }),
  );
  assert.deepEqual(result, {
    state: "multiple",
    label: "Sources",
    value: "Agent ID beta.17 · Rogue · Team B; Agent ID alpha/9001 · Hunter · Team A",
    text: "Sources: Agent ID beta.17 · Rogue · Team B; Agent ID alpha/9001 · Hunter · Team A",
  });
});

test("Oracle absence and every malformed or conflicting join fail closed", () => {
  const unavailable = {
    state: "unavailable",
    label: "Source",
    value: "Unavailable in this artifact",
    text: "Source unavailable in this artifact",
  };
  const cases = [
    options({ direct_sources: [] }),
    options({ direct_sources: null }),
    options({ authorized_agents: [] }),
    options({ direct_sources: [{ source_public_agent_id: "alpha/9001" }] }),
    options({
      direct_sources: [
        {
          ...source("source:m1", "alpha/9001"),
          event_id: "must-not-authorize",
        },
      ],
    }),
    options({
      direct_sources: [source("source:m1", "alpha/9001")],
      authorized_agents: [agent("different-key", "alpha/9001", 1, 1)],
    }),
    options({
      direct_sources: [source("source:m1", "alpha/9001")],
      authorized_agents: [agent("source:m1", "different-id", 1, 1)],
    }),
    options({
      direct_sources: [
        source("source:m1", "alpha/9001"),
        source("source:m1", "conflicting-id"),
      ],
    }),
    options({
      direct_sources: [
        source("source:m1", "alpha/9001"),
        source("conflicting-key", "alpha/9001"),
      ],
    }),
    options({
      authorized_agents: [
        agent("source:m1", "alpha/9001", 1, 1),
        agent("source:m1", "alpha/9001", 1, 1),
      ],
    }),
  ];
  for (const candidate of cases) {
    assert.deepEqual(authorizedSourceAttributionV1(candidate), unavailable);
  }
});

test("caller arrays must be dense plain data arrays and accessors are never read", () => {
  let reads = 0;
  /** @type {unknown[]} */
  const sourceGetter = [];
  Object.defineProperty(sourceGetter, "0", {
    enumerable: true,
    get() {
      reads += 1;
      return source("source:m1", "alpha/9001");
    },
  });
  sourceGetter.length = 1;
  /** @type {unknown[]} */
  const agentGetter = [];
  Object.defineProperty(agentGetter, "0", {
    enumerable: true,
    get() {
      reads += 1;
      return agent("source:m1", "alpha/9001", 1, 1);
    },
  });
  agentGetter.length = 1;

  const sparseSources = Array(1);
  const extraPropertySources = [source("source:m1", "alpha/9001")];
  Object.defineProperty(extraPropertySources, "extra", {
    value: "not an index",
  });
  const overriddenMapAgents = [agent("source:m1", "alpha/9001", 1, 1)];
  Object.defineProperty(overriddenMapAgents, "map", {
    value() {
      throw new Error("caller map must never run");
    },
  });
  const overriddenIteratorAgents = [agent("source:m1", "alpha/9001", 1, 1)];
  Object.defineProperty(overriddenIteratorAgents, Symbol.iterator, {
    value() {
      throw new Error("caller iterator must never run");
    },
  });
  const customPrototypeAgents = [agent("source:m1", "alpha/9001", 1, 1)];
  Object.setPrototypeOf(customPrototypeAgents, Object.create(Array.prototype));

  const malformedArrays = [
    options({ direct_sources: sourceGetter }),
    options({ authorized_agents: agentGetter }),
    options({ direct_sources: sparseSources }),
    options({ direct_sources: extraPropertySources }),
    options({ authorized_agents: overriddenMapAgents }),
    options({ authorized_agents: overriddenIteratorAgents }),
    options({ authorized_agents: customPrototypeAgents }),
  ];
  for (const candidate of malformedArrays) {
    assert.equal(authorizedSourceAttributionV1(candidate)?.state, "unavailable");
  }
  assert.equal(reads, 0);
});

test("every malformed agent fails closed, including valid-plus-malformed ambiguity", () => {
  const valid = agent("source:m1", "alpha/9001", 1, 1);
  const malformedDuplicate = { ...valid, class_id: 9 };
  assert.equal(
    authorizedSourceAttributionV1(
      options({ authorized_agents: [valid, malformedDuplicate] }),
    )?.state,
    "unavailable",
  );
  assert.equal(
    authorizedSourceAttributionV1(
      options({ authorized_agents: [valid, { unrelated: "malformed" }] }),
    )?.state,
    "unavailable",
  );
});

test("Agent POV is exactly redacted and hidden source mutations are byte-inert", () => {
  let nestedReads = 0;
  const hiddenSource = {};
  Object.defineProperty(hiddenSource, "source_presentation_key", {
    enumerable: true,
    get() {
      nestedReads += 1;
      return "secret-key";
    },
  });
  const hiddenAgent = {};
  Object.defineProperty(hiddenAgent, "presentation_key", {
    enumerable: true,
    get() {
      nestedReads += 1;
      return "secret-key";
    },
  });

  const expected = {
    state: "redacted",
    label: "Source",
    value: "Not disclosed in Agent POV",
    text: "Source not disclosed in Agent POV",
  };
  assert.deepEqual(
    authorizedSourceAttributionV1(
      options({
        audience: "agent_pov",
        direct_sources: [hiddenSource],
        authorized_agents: [hiddenAgent],
      }),
    ),
    expected,
  );
  assert.deepEqual(
    authorizedSourceAttributionV1(
      options({
        audience: "agent_pov",
        direct_sources: [{ hidden: "different secret" }],
        authorized_agents: [{ hidden: "different scene" }],
      }),
    ),
    expected,
  );
  assert.equal(nestedReads, 0);
});

test("aggregate aura deliberately has no source and Spawn Shield is excluded", () => {
  let nestedReads = 0;
  const hidden = {};
  Object.defineProperty(hidden, "source_public_agent_id", {
    enumerable: true,
    get() {
      nestedReads += 1;
      return "secret";
    },
  });
  for (const audience of /** @type {const} */ (["researcher", "agent_pov"])) {
    assert.equal(
      authorizedSourceAttributionV1(
        options({
          attribution_kind: "aggregate_aura",
          audience,
          direct_sources: [hidden],
          authorized_agents: [hidden],
        }),
      ),
      null,
    );
  }
  assert.equal(nestedReads, 0);
  assert.throws(
    () => authorizedSourceAttributionV1(options({ attribution_kind: "spawn_shield" })),
    /direct or aggregate_aura/u,
  );
});

test("class, proximity, event, and slot lookalikes never create attribution", () => {
  const lookalike = agent("unreferenced", "unreferenced-agent", 1, 1);
  const noEvidence = authorizedSourceAttributionV1(
    options({ direct_sources: [], authorized_agents: [lookalike] }),
  );
  assert.equal(noEvidence?.state, "unavailable");

  const legacyEvidence = {
    source_global_slot: 999,
    source_public_agent_id: "unreferenced-agent",
    source_class_id: 1,
    event_id: "event:lookalike",
    position: [999, 999],
  };
  const legacyResult = authorizedSourceAttributionV1(
    options({ direct_sources: [legacyEvidence], authorized_agents: [lookalike] }),
  );
  assert.equal(legacyResult?.state, "unavailable");
  assert.doesNotMatch(JSON.stringify(legacyResult), /unreferenced-agent|event|999/u);
});

test("malformed public identities never receive a partial canonical title", () => {
  const valid = agent("source:m1", "alpha/9001", 1, 1);
  const malformed = [
    { ...valid, class_id: 9 },
    { ...valid, team_id: 3 },
    { ...valid, public_agent_id: " alpha/9001" },
    { ...valid, public_agent_id: "alpha/9001\nforged" },
    { ...valid, presentation_key: "" },
    { ...valid, presentation_key: "x".repeat(513) },
    { ...valid, public_agent_id: "x".repeat(513) },
    { ...valid, class_id: "1" },
    Object.create({
      presentation_key: "source:m1",
      public_agent_id: "alpha/9001",
      class_id: 1,
      class_name: "Mage",
      team_id: 1,
    }),
  ];
  assert.equal(
    exactAuthorizedAgentIdentityV1(valid)?.title,
    "Agent ID alpha/9001 · Mage · Team A",
  );
  for (const identity of malformed) {
    assert.equal(exactAuthorizedAgentIdentityV1(identity), null);
    assert.equal(
      authorizedSourceAttributionV1(options({ authorized_agents: [identity] }))?.state,
      "unavailable",
    );
  }

  let ignoredAccessorReads = 0;
  const ignoredAccessor = { ...valid };
  Object.defineProperty(ignoredAccessor, "class_name", {
    enumerable: true,
    get() {
      ignoredAccessorReads += 1;
      return "Warrior";
    },
  });
  assert.equal(
    exactAuthorizedAgentIdentityV1(ignoredAccessor)?.title,
    "Agent ID alpha/9001 · Mage · Team A",
  );
  assert.equal(ignoredAccessorReads, 0);

  let requiredAccessorReads = 0;
  const requiredAccessor = { ...valid };
  Object.defineProperty(requiredAccessor, "team_id", {
    enumerable: true,
    get() {
      requiredAccessorReads += 1;
      return 1;
    },
  });
  assert.equal(exactAuthorizedAgentIdentityV1(requiredAccessor), null);
  assert.equal(requiredAccessorReads, 0);
});

test("the public API rejects loose option records and unsupported audiences", () => {
  assert.throws(
    () => authorizedSourceAttributionV1({ ...options(), extra: true }),
    /exact plain record/u,
  );
  assert.throws(
    () =>
      authorizedSourceAttributionV1(
        options({ audience: /** @type {any} */ ("oracle") }),
      ),
    /audience/u,
  );
  assert.throws(() => authorizedSourceAttributionV1(null), /exact plain record/u);
});
