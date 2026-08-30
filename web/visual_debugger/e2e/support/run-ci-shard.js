import { spawnSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = fileURLToPath(new URL("../..", import.meta.url));
const e2eRoot = path.join(frontendRoot, "e2e");
const manifestPath = path.join(e2eRoot, "ci-shards.json");

/** @typedef {{files: string[], test_titles?: string[]}} CiShard */
/** @typedef {{schema_version: 2, shards: CiShard[]}} CiManifest */

/** @param {unknown} value @param {string} label */
function validatedNonemptyStrings(value, label) {
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.some((entry) => typeof entry !== "string" || entry.length === 0)
  ) {
    throw new Error(`${label} must be a nonempty string list`);
  }
  if (new Set(value).size !== value.length) {
    throw new Error(`${label} must not contain duplicates`);
  }
  return /** @type {string[]} */ (value);
}

/**
 * @param {string} directory
 * @param {string} [relativeDirectory]
 * @returns {string[]}
 */
function discoveredPlaywrightSpecFiles(directory, relativeDirectory = "") {
  const currentDirectory = path.join(directory, relativeDirectory);
  const discovered = [];
  for (const entry of readdirSync(currentDirectory, { withFileTypes: true })) {
    const relativePath = path.posix.join(relativeDirectory, entry.name);
    if (entry.isDirectory()) {
      discovered.push(...discoveredPlaywrightSpecFiles(directory, relativePath));
    } else if (entry.isFile() && entry.name.endsWith(".spec.js")) {
      discovered.push(relativePath);
    }
  }
  return discovered.sort();
}

/** @returns {CiManifest} */
export function validatedCiManifest() {
  const value = /** @type {unknown} */ (JSON.parse(readFileSync(manifestPath, "utf8")));
  if (
    typeof value !== "object" ||
    value === null ||
    !("schema_version" in value) ||
    value.schema_version !== 2 ||
    !("shards" in value) ||
    !Array.isArray(value.shards) ||
    value.shards.length !== 8
  ) {
    throw new Error("e2e/ci-shards.json has an invalid root contract");
  }

  const shards = value.shards.map((entry, index) => {
    if (typeof entry !== "object" || entry === null || !("files" in entry)) {
      throw new Error(`CI browser shard ${index + 1} must be an object with files`);
    }
    const files = validatedNonemptyStrings(
      entry.files,
      `CI browser shard ${index + 1} files`,
    );
    if (!("test_titles" in entry)) {
      return { files };
    }
    const testTitles = validatedNonemptyStrings(
      entry.test_titles,
      `CI browser shard ${index + 1} test_titles`,
    );
    if (files.length !== 1) {
      throw new Error(
        `CI browser shard ${index + 1} may select exact titles from only one file`,
      );
    }
    return { files, test_titles: testTitles };
  });

  const actual = discoveredPlaywrightSpecFiles(e2eRoot);
  const declared = [...new Set(shards.flatMap((shard) => shard.files))].sort();
  if (JSON.stringify(declared) !== JSON.stringify(actual)) {
    throw new Error("browser shard manifest must exactly cover e2e/**/*.spec.js");
  }

  for (const filename of actual) {
    const owners = shards.filter((shard) => shard.files.includes(filename));
    if (owners.length > 1 && owners.some((shard) => shard.test_titles === undefined)) {
      throw new Error(
        `browser spec ${filename} may repeat only through exact-title shards`,
      );
    }
    const selectedTitles = owners.flatMap((shard) => shard.test_titles ?? []);
    if (new Set(selectedTitles).size !== selectedTitles.length) {
      throw new Error(`browser spec ${filename} assigns one title more than once`);
    }
  }
  return { schema_version: 2, shards };
}

/** @param {string} value */
function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

/** @param {CiShard} shard */
export function playwrightArgumentsForShard(shard) {
  const args = shard.files.map((filename) => `e2e/${filename}`);
  if (shard.test_titles !== undefined) {
    const alternatives = shard.test_titles.map(escapeRegExp).join("|");
    args.push("--grep", `(?:^|\\s)(?:${alternatives})$`);
  }
  return args;
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
  const profileArguments = playwrightArgumentsForShard(manifest.shards[shardIndex]);
  const result = spawnSync(
    process.execPath,
    [
      cli,
      "test",
      ...profileArguments,
      "--config",
      "playwright.config.js",
      ...forwarded,
    ],
    { cwd: frontendRoot, env: process.env, stdio: "inherit" },
  );
  process.exit(result.status ?? 1);
}
