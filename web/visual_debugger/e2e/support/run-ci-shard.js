import { spawnSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = fileURLToPath(new URL("../..", import.meta.url));
const e2eRoot = path.join(frontendRoot, "e2e");
const manifestPath = path.join(e2eRoot, "ci-shards.json");

/** @typedef {{schema_version: number, shards: string[][]}} CiManifest */

/** @returns {CiManifest} */
export function validatedCiManifest() {
  const value = /** @type {unknown} */ (JSON.parse(readFileSync(manifestPath, "utf8")));
  if (
    typeof value !== "object" ||
    value === null ||
    !("schema_version" in value) ||
    value.schema_version !== 1 ||
    !("shards" in value) ||
    !Array.isArray(value.shards) ||
    value.shards.length !== 8
  ) {
    throw new Error("e2e/ci-shards.json has an invalid root contract");
  }

  const shards = value.shards.map((entry, index) => {
    if (
      !Array.isArray(entry) ||
      entry.length === 0 ||
      entry.some((filename) => typeof filename !== "string")
    ) {
      throw new Error(`CI browser shard ${index + 1} must be a nonempty string list`);
    }
    return /** @type {string[]} */ (entry);
  });
  const declared = shards.flat();
  if (new Set(declared).size !== declared.length) {
    throw new Error("each browser spec must be declared exactly once");
  }

  const actual = readdirSync(e2eRoot)
    .filter((filename) => filename.endsWith(".spec.js"))
    .sort();
  if (JSON.stringify([...declared].sort()) !== JSON.stringify(actual)) {
    throw new Error("browser shard manifest must exactly cover e2e/*.spec.js");
  }
  return { schema_version: 1, shards };
}

/** @param {string} value */
function parseShard(value) {
  const match = /^(\d+)\/8$/u.exec(value);
  if (match === null) {
    throw new Error("browser shard must use N/8");
  }
  const index = Number.parseInt(match[1], 10);
  if (index < 1 || index > 8) {
    throw new Error("browser shard requires 1 <= N <= 8");
  }
  return index - 1;
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1] ?? "")) {
  const [rawShard, ...forwarded] = process.argv.slice(2);
  const shardIndex = parseShard(rawShard ?? "");
  const manifest = validatedCiManifest();
  const cli = path.join(frontendRoot, "node_modules/@playwright/test/cli.js");
  const files = manifest.shards[shardIndex].map((filename) => `e2e/${filename}`);
  const result = spawnSync(
    process.execPath,
    [cli, "test", ...files, "--config", "playwright.config.js", ...forwarded],
    { cwd: frontendRoot, env: process.env, stdio: "inherit" },
  );
  process.exit(result.status ?? 1);
}
