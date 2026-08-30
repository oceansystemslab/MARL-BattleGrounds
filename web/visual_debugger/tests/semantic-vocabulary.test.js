import assert from "node:assert/strict";
import test from "node:test";

import { auraPresentation } from "../src/semantic-vocabulary.js";

test("aura vocabulary owns only the exact qualitative C2 copy", () => {
  const cases = [
    [
      "mage_damage_amplification",
      {
        fieldTitle: "Sorcerer's Empowerment · Mage Damage Amplification Aura",
        recipientTitle: "Sorcerer's Empowerment · Mage Damage Amplification Aura",
        fieldEffect:
          "This Mage radiates arcane magic, amplifying outgoing damage for eligible unshielded same-team agents in its radius, including itself.",
        aggregateEffect: "This agent benefits from authorized Mage aura coverage.",
        fieldEffectLabel: "Damage Amplification Effect",
        aggregateEffectLabel: "Aggregated Damage Amplification Effect",
        accent: "mage",
        effectKind: "damage_dealt",
      },
    ],
    [
      "warrior_damage_mitigation",
      {
        fieldTitle: "Guardian's Barrier · Warrior Damage Mitigation Aura",
        recipientTitle: "Guardian's Barrier · Warrior Damage Mitigation Aura",
        fieldEffect:
          "This Warrior emanates a defensive aura, mitigating incoming damage for eligible unshielded same-team agents in its radius, including itself.",
        aggregateEffect: "This agent benefits from authorized Warrior aura coverage.",
        fieldEffectLabel: "Damage Mitigation Effect",
        aggregateEffectLabel: "Aggregated Damage Mitigation Effect",
        accent: "warrior",
        effectKind: "damage_received",
      },
    ],
  ];
  for (const [auraId, expected] of cases) {
    const presentation = auraPresentation(auraId);
    assert.deepEqual(presentation, expected);
    assert.equal(Object.isFrozen(presentation), true);
  }
});

test("unknown aura inputs retain no class guide or source fallback", () => {
  for (const auraId of ["", "paladin_aura", null, 7]) {
    const presentation = auraPresentation(auraId);
    assert.equal(presentation.fieldTitle, "Recorded Aura Field");
    assert.equal(presentation.recipientTitle, "Recorded Aura");
    assert.equal(presentation.accent, "none");
    assert.equal(Object.hasOwn(presentation, "ultimateName"), false);
    assert.equal(Object.hasOwn(presentation, "source"), false);
  }
});
