import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  playwrightArgumentsForShard,
  validatedCiManifest,
} from "../e2e/support/run-ci-shard.js";

const frontendRoot = fileURLToPath(new URL("..", import.meta.url));
const playwrightCli = path.join(frontendRoot, "node_modules/@playwright/test/cli.js");

/** @param {string[]} arguments_ */
function collectedPlaywrightIds(arguments_) {
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
      env: process.env,
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

test("CI browser manifest is nonempty, exact, and eight-way", () => {
  const manifest = validatedCiManifest();
  assert.equal(manifest.shards.length, 8);
  assert.ok(manifest.shards.every((shard) => shard.files.length > 0));
  assert.deepEqual(
    manifest.shards.slice(0, 2).map((shard) => shard.test_titles?.length),
    [4, 4],
  );
  assert.deepEqual(manifest.shards[2].files, [
    "authorized-presentation-renderer.spec.js",
    "resize.spec.js",
  ]);
});

test("CI browser profiles are an exact disjoint cover of collected Playwright tests", {
  timeout: 60_000,
}, () => {
  const manifest = validatedCiManifest();
  const allFiles = [...new Set(manifest.shards.flatMap((shard) => shard.files))]
    .sort()
    .map((filename) => `e2e/${filename}`);
  const complete = collectedPlaywrightIds(allFiles);
  const assignedByShard = manifest.shards.map((shard) =>
    collectedPlaywrightIds(playwrightArgumentsForShard(shard)),
  );
  assert.ok(complete.length > 0);
  assert.ok(assignedByShard.every((ids) => ids.length > 0));

  const assigned = assignedByShard.flat().sort();
  assert.equal(new Set(assigned).size, assigned.length);
  assert.deepEqual(assigned, complete);
});
