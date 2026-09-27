/**
 * @file Check one whole-file browser profile, exact test coverage and safe
 * deterministic profile environments. Every spec runs once with normal setup;
 * the profile does not filter test titles or enable an isolated test slice.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  playwrightArgumentsForShard,
  playwrightEnvironmentForShard,
  validatedCiManifest,
  validatedEnvironment,
} from "../e2e/support/run-ci-shard.js";

const frontendRoot = fileURLToPath(new URL("..", import.meta.url));
const playwrightCli = path.join(frontendRoot, "node_modules/@playwright/test/cli.js");

/** @param {string[]} arguments_ @param {Record<string, string>} [environment] */
function collectedPlaywrightIds(arguments_, environment = {}) {
  const result = spawnSync(
    process.execPath,
    [
      playwrightCli,
      "test",
      ...arguments_,
      "--config",
      "playwright.config.js",
      "--list",
      "--reporter=json",
    ],
    {
      cwd: frontendRoot,
      encoding: "utf8",
      env: { ...process.env, ...environment },
      maxBuffer: 32 * 1024 * 1024,
    },
  );
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stderr || result.stdout);
  const report = JSON.parse(result.stdout);
  /** @type {string[]} */
  const ids = [];

  /** @param {unknown} value */
  function visitSuite(value) {
    assert.equal(typeof value, "object");
    assert.notEqual(value, null);
    const suite =
      /** @type {{suites?: unknown[], specs?: Array<{file?: string, line?: number, column?: number, title?: string, tests?: Array<{projectName?: string}>}>}} */ (
        value
      );
    for (const spec of suite.specs ?? []) {
      for (const case_ of spec.tests ?? []) {
        ids.push(
          [
            case_.projectName ?? "",
            spec.file ?? "",
            spec.line ?? 0,
            spec.column ?? 0,
            spec.title ?? "",
          ].join("|"),
        );
      }
    }
    for (const child of suite.suites ?? []) {
      visitSuite(child);
    }
  }

  for (const suite of report.suites ?? []) {
    visitSuite(suite);
  }
  return ids.sort();
}

test("CI browser manifest runs every file whole in one profile", () => {
  const manifest = validatedCiManifest();
  assert.equal(manifest.shards.length, 1);
  assert.ok(manifest.shards[0].files.length > 0);
  assert.deepEqual(Object.keys(manifest.shards[0]), ["files"]);
  assert.deepEqual(
    playwrightArgumentsForShard(manifest.shards[0]),
    manifest.shards[0].files.map((filename) => `e2e/${filename}`),
  );
});

test("CI browser profile environment is additive", () => {
  const shard = {
    files: ["authorized-presentation-install.spec.js"],
    env: { MARL_CP5_SLICE_5_ONLY: "1" },
  };
  assert.deepEqual(playwrightEnvironmentForShard(shard, { PATH: "/bin" }), {
    PATH: "/bin",
    MARL_CP5_SLICE_5_ONLY: "1",
  });
  assert.deepEqual(
    playwrightEnvironmentForShard(validatedCiManifest().shards[0], { PATH: "/bin" }),
    { PATH: "/bin" },
  );
});

test("CI browser profile environment rejects unsafe or empty settings", () => {
  for (const value of [
    {},
    { PATH: "/tmp" },
    { MARL_EMPTY: "" },
    { MARL_NON_STRING: 1 },
  ]) {
    assert.throws(
      () => validatedEnvironment(value, "profile env"),
      /environment object|nonempty MARL_\* string settings/u,
    );
  }
});

test("CI browser profiles are an exact disjoint cover of collected Playwright tests", () => {
  const manifest = validatedCiManifest();
  const allFiles = [...new Set(manifest.shards.flatMap((shard) => shard.files))]
    .sort()
    .map((filename) => `e2e/${filename}`);
  const complete = collectedPlaywrightIds(allFiles);
  const assignedByShard = manifest.shards.map((shard) =>
    collectedPlaywrightIds(playwrightArgumentsForShard(shard), shard.env),
  );
  assert.ok(complete.length > 0);
  assert.ok(assignedByShard.every((ids) => ids.length > 0));

  const assigned = assignedByShard.flat().sort();
  assert.equal(new Set(assigned).size, assigned.length);
  assert.deepEqual(assigned, complete);
});
