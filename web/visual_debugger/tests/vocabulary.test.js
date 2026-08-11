import assert from "node:assert/strict";
import test from "node:test";

import { iconDefinition, KNOWN_GLYPH_KEYS } from "../src/icons.js";
import {
  activationImpactSemantic,
  ACTIVATION_TOKEN_IDS,
  CANONICAL_STATUS_ORDER,
  CLASS_TOKEN_IDS,
  classTokenFromId,
  LIFECYCLE_TOKEN_IDS,
  MODIFIER_TOKEN_IDS,
  resolveVisualToken,
  TEAM_TOKEN_IDS,
  teamTokenFromId,
  ultimateTokenFromClassId,
} from "../src/vocabulary.js";

const EXPECTED_CLASSES = ["mage", "warrior", "hunter", "rogue", "priest"];
const EXPECTED_TEAMS = ["team_a", "team_b"];
const EXPECTED_STATUSES = [
  "stun_warrior_charge",
  "stun_hunter_trap",
  "stun_rogue_poison",
  "slow_warrior_charge",
  "slow_hunter_basic",
  "slow_rogue_poison",
  "anti_heal_rogue_poison",
  "priest_freedom",
  "mage_burst",
];
const EXPECTED_ACTIVATIONS = [
  "basic_damage",
  "basic_heal",
  "holy_word",
  "mage_burst",
  "warrior_charge",
  "hunter_trap",
  "rogue_poison",
];
const EXPECTED_MODIFIERS = [
  "mage_amplification",
  "warrior_mitigation",
  "rogue_anti_heal",
  "priest_freedom",
  "mage_burst",
];
const EXPECTED_LIFECYCLE = [
  "applied",
  "refreshed",
  "decremented",
  "expired",
  "trap_broken",
  "cleared_unclassified",
  "trap_broken_and_reapplied",
];

test("display registries cover every current stable semantic token", () => {
  assert.deepEqual(CLASS_TOKEN_IDS, EXPECTED_CLASSES);
  assert.deepEqual(TEAM_TOKEN_IDS, EXPECTED_TEAMS);
  assert.deepEqual(CANONICAL_STATUS_ORDER, EXPECTED_STATUSES);
  assert.deepEqual(ACTIVATION_TOKEN_IDS, EXPECTED_ACTIVATIONS);
  assert.deepEqual(MODIFIER_TOKEN_IDS, EXPECTED_MODIFIERS);
  assert.deepEqual(LIFECYCLE_TOKEN_IDS, EXPECTED_LIFECYCLE);

  /** @type {Array<[("class" | "team" | "status" | "activation" | "modifier" | "lifecycle"), string[]]>} */
  const registries = [
    ["class", EXPECTED_CLASSES],
    ["team", EXPECTED_TEAMS],
    ["status", EXPECTED_STATUSES],
    ["activation", EXPECTED_ACTIVATIONS],
    ["modifier", EXPECTED_MODIFIERS],
    ["lifecycle", EXPECTED_LIFECYCLE],
  ];
  for (const [kind, tokenIds] of registries) {
    for (const tokenId of tokenIds) {
      const definition = resolveVisualToken(kind, tokenId);
      assert.equal(definition.tokenId, tokenId);
      assert.notEqual(definition.cssKey, "unknown");
      assert.ok(definition.label);
      assert.ok(definition.shortLabel);
      assert.ok(definition.accessibleName);
      assert.ok(definition.fallback);
      assert.notEqual(iconDefinition(definition.glyphKey).glyphKey, "unknown");
      assert.ok(Object.isFrozen(definition));
    }
  }
});

test("numeric class and team identities map only to display vocabulary", () => {
  assert.deepEqual(
    [1, 2, 3, 4, 5].map((classId) => classTokenFromId(classId).tokenId),
    EXPECTED_CLASSES,
  );
  assert.deepEqual(
    [1, 2].map((teamId) => teamTokenFromId(teamId).tokenId),
    EXPECTED_TEAMS,
  );
});

test("class glyphs use the canonical Hunter bow and Priest medic cross", () => {
  const hunter = iconDefinition("class-hunter");
  assert.deepEqual(
    hunter.primitives.map((primitive) => primitive.tag),
    ["path", "line", "line", "polyline"],
  );
  assert.deepEqual(hunter.primitives[1].attributes, {
    x1: 6.5,
    y1: 3.5,
    x2: 6.5,
    y2: 20.5,
  });
  assert.deepEqual(hunter.primitives[2].attributes, {
    x1: 3,
    y1: 12,
    x2: 21,
    y2: 12,
  });
  assert.equal(hunter.primitives[3].attributes.points, "17,8 21,12 17,16");

  const priest = iconDefinition("class-priest");
  assert.deepEqual(
    priest.primitives.map((primitive) => primitive.attributes),
    [
      {
        x: 9,
        y: 3,
        width: 6,
        height: 18,
        fill: "currentColor",
        stroke: "none",
      },
      {
        x: 3,
        y: 9,
        width: 18,
        height: 6,
        fill: "currentColor",
        stroke: "none",
      },
    ],
  );
});

test("durable control statuses share canonical glyphs without losing source identity", () => {
  const stuns = EXPECTED_STATUSES.slice(0, 3).map((tokenId) =>
    resolveVisualToken("status", tokenId),
  );
  assert.deepEqual(
    stuns.map((definition) => definition.glyphKey),
    ["status-stun", "status-stun", "status-stun"],
  );
  assert.equal(new Set(stuns.map((definition) => definition.cssKey)).size, 3);
  assert.equal(new Set(stuns.map((definition) => definition.accessibleName)).size, 3);
  assert.deepEqual(
    iconDefinition("status-stun").primitives.map((primitive) => primitive.attributes.d),
    ["m12 2 8 5v10l-8 5-8-5V7Z", "m8.5 8.5 7 7M15.5 8.5l-7 7"],
  );
  assert.equal(KNOWN_GLYPH_KEYS.includes("status-charge-stun"), false);
  assert.equal(KNOWN_GLYPH_KEYS.includes("status-trap"), false);
  assert.equal(KNOWN_GLYPH_KEYS.includes("status-poison-stun"), false);

  const slows = EXPECTED_STATUSES.slice(3, 6).map((tokenId) =>
    resolveVisualToken("status", tokenId),
  );
  assert.deepEqual(
    slows.map((definition) => definition.glyphKey),
    ["status-slow", "status-slow", "status-slow"],
  );
  assert.equal(new Set(slows.map((definition) => definition.cssKey)).size, 3);
  assert.equal(new Set(slows.map((definition) => definition.accessibleName)).size, 3);
  assert.deepEqual(
    iconDefinition("status-slow").primitives.map((primitive) => primitive.tag),
    ["path"],
  );
});

test("class-specific Ultimate tokens preserve activation vocabulary", () => {
  const ultimates = [1, 2, 3, 4, 5].map((classId) => ultimateTokenFromClassId(classId));
  assert.deepEqual(
    ultimates.map((definition) => definition.tokenId),
    ["mage_burst", "warrior_charge", "hunter_trap", "rogue_poison", "holy_word"],
  );
  assert.deepEqual(
    ultimates.map((definition) => definition.glyphKey),
    [
      "activation-burst",
      "activation-charge",
      "activation-trap",
      "activation-poison",
      "activation-holy-word",
    ],
  );
  assert.deepEqual(
    ultimates.map((definition) => definition.cssKey),
    ["mage-burst", "warrior-charge", "hunter-trap", "rogue-poison", "holy-word"],
  );
});

test("activation impact grammar is explicit, non-numeric, and fail-closed", () => {
  assert.deepEqual(
    EXPECTED_ACTIVATIONS.map((tokenId) => activationImpactSemantic(tokenId)),
    ["damage", "healing", "healing", "local", "damage", "damage", "damage"],
  );
  assert.equal(activationImpactSemantic("future_activation"), "neutral");
  assert.equal(activationImpactSemantic(null), "neutral");
});

test("payload prose wins without allowing payload-controlled glyph or CSS keys", () => {
  const definition = resolveVisualToken("status", "stun_hunter_trap", {
    label: "Authoritative trap label",
    short_label: "AUTH",
    accessible_name: "Authoritative accessible trap label",
    glyphKey: "external-image",
    cssKey: "payload-selector",
  });

  assert.equal(definition.label, "Authoritative trap label");
  assert.equal(definition.shortLabel, "AUTH");
  assert.equal(definition.accessibleName, "Authoritative accessible trap label");
  assert.equal(definition.glyphKey, "status-stun");
  assert.equal(definition.cssKey, "stun-hunter-trap");
});

test("unknown and malformed IDs use a stable non-injecting fallback", () => {
  const future = resolveVisualToken("status", "future/status:hover", {
    label: "Future status",
    accessible_name: "Future status supplied by Python",
  });
  assert.equal(future.tokenId, "future/status:hover");
  assert.equal(future.label, "Future status");
  assert.equal(future.accessibleName, "Future status supplied by Python");
  assert.equal(future.glyphKey, "unknown");
  assert.equal(future.cssKey, "unknown");
  assert.equal(future.fallback, "?");

  assert.equal(resolveVisualToken("status", null).tokenId, "unknown");
  assert.equal(resolveVisualToken("status", "__proto__").cssKey, "unknown");
  assert.equal(classTokenFromId(999).cssKey, "unknown");
  assert.equal(ultimateTokenFromClassId(999).cssKey, "unknown");
  assert.equal(ultimateTokenFromClassId("1").cssKey, "unknown");
  assert.equal(teamTokenFromId("1").cssKey, "unknown");
  assert.equal(iconDefinition("external-image").glyphKey, "unknown");
  assert.equal(iconDefinition("constructor").glyphKey, "unknown");
});

test("glyph definitions are frozen local SVG primitives without links", () => {
  assert.ok(KNOWN_GLYPH_KEYS.includes("unknown"));
  for (const glyphKey of KNOWN_GLYPH_KEYS) {
    const definition = iconDefinition(glyphKey);
    assert.ok(Object.isFrozen(definition));
    assert.ok(Object.isFrozen(definition.primitives));
    assert.ok(definition.primitives.length > 0);
    for (const primitive of definition.primitives) {
      assert.ok(["circle", "line", "path", "polyline", "rect"].includes(primitive.tag));
      assert.equal("href" in primitive.attributes, false);
      assert.equal("xlink:href" in primitive.attributes, false);
      assert.ok(Object.isFrozen(primitive));
      assert.ok(Object.isFrozen(primitive.attributes));
    }
  }
});
