import assert from "node:assert/strict";
import test from "node:test";

import { validatedCiManifest } from "../e2e/support/run-ci-shard.js";

test("CI browser manifest is nonempty, exact, unique, and eight-way", () => {
  const manifest = validatedCiManifest();
  assert.equal(manifest.shards.length, 8);
  assert.ok(manifest.shards.every((shard) => shard.length > 0));
  const declared = manifest.shards.flat();
  assert.equal(new Set(declared).size, declared.length);
});
