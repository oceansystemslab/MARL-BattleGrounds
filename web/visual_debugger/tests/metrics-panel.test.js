import assert from "node:assert/strict";
import test from "node:test";
import { metricLabel, selectedMetricRows } from "../src/metrics-panel.js";

test("metric groups preserve actual class identities and order teams before numeric agents", () => {
  const rows = [
    {
      name: "agent_9_healing",
      family: "healing_done",
      scope: "agent",
      subjects: [9],
      subject: "Agent 9 · Mage",
      valid: false,
      value: null,
    },
    {
      name: "agent_2_healing",
      family: "healing_done",
      scope: "agent",
      subjects: [2],
      subject: "Agent 2 · Priest",
      valid: true,
      value: 12.5,
    },
    {
      name: "team_b_healing",
      family: "healing_done",
      scope: "team",
      subjects: [2],
      subject: "Team B",
      valid: true,
      value: 0,
    },
    {
      name: "team_a_healing",
      family: "healing_done",
      scope: "team",
      subjects: [1],
      subject: "Team A",
      valid: true,
      value: 12.5,
    },
    {
      name: "team_a_score",
      family: "priority",
      scope: "team",
      subjects: [1],
      subject: "Team A",
      valid: true,
      value: 2,
    },
  ];
  const before = structuredClone(rows);
  const selected = selectedMetricRows({ statistics: rows }, "healing_done");
  assert.deepEqual(
    selected.map((/** @type {Record<string, any>} */ row) => row.name),
    ["team_a_healing", "team_b_healing", "agent_2_healing", "agent_9_healing"],
  );
  assert.equal(selected[2].subject, "Agent 2 · Priest");
  assert.equal(selected[2].value, 12.5);
  assert.equal(selected[3].value, null);
  assert.deepEqual(rows, before);
  assert.deepEqual(selectedMetricRows({ statistics: rows }, "overview"), [rows[4]]);
  assert.equal(metricLabel("agent_steps"), "Agent Steps");
});
