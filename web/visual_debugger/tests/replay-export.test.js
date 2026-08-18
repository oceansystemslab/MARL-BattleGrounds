import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { deflateSync } from "node:zlib";

import {
  joinTransportAndAuthorizedPresentationV1,
  normalizeAuthorizedPresentationFrameV1,
} from "../src/authorized-presentation-normalizer.js";
import {
  assertOpaqueReplayExportPixelsV1,
  buildReplayBattlefieldPngArtifactV1,
  buildReplayBattlefieldPngFilenameV1,
  canonicalReplayPngProvenanceV1,
  captureReplayBattlefieldPngV1,
  inspectReplayBattlefieldPngV1,
  projectReplayPngProvenanceV1,
  REPLAY_BATTLEFIELD_BACKGROUND_V1,
  REPLAY_PNG_PROVENANCE_KEYWORD,
  sanitizeReplayEpisodeForPngFilenameV1,
  sanitizeReplayRecipientForPngFilenameV1,
} from "../src/replay-export.js";
import {
  DEFAULT_VISUAL_FILTER_STATE,
  setVisualFilterEnabled,
  VISUAL_FILTER_IDS,
} from "../src/visual-filters.js";

const fixtureUrl = new URL(
  "./fixtures/authorized-presentations-v1.json",
  import.meta.url,
);
const sourceUrl = new URL("../src/replay-export.js", import.meta.url);
const fixture = JSON.parse(await readFile(fixtureUrl, "utf8"));
/** @type {Record<string, Readonly<Record<string, any>>>} */
const presentations = {};
for (const [name, presentation] of Object.entries(fixture.presentations)) {
  presentations[name] = await normalizeAuthorizedPresentationFrameV1(presentation);
}

const replayNames = Object.freeze([
  "replay_oracle",
  "replay_no_shared_obs_agent_pov",
  "replay_shared_obs_agent_pov",
]);
const liveNames = Object.freeze(["live_oracle", "live_no_shared_obs_agent_pov"]);

function deferred() {
  /** @type {(value?: any) => void} */
  let resolve = () => {};
  /** @type {(reason?: any) => void} */
  let reject = () => {};
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

/** @param {Readonly<Record<string, any>>} presentation */
function projectionOptions(presentation) {
  return {
    presentation,
    renderPolicy: "replay_static",
    cssWidth: 640,
    cssHeight: 360,
    showRanges: true,
    localInspectedPresentationKey:
      presentation.presentation_kind === "replay_oracle" ? undefined : null,
    visualFilters: DEFAULT_VISUAL_FILTER_STATE,
  };
}

/** @param {number} value */
function u32(value) {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setUint32(0, value, false);
  return bytes;
}

/** @param {Uint8Array} bytes */
function independentCrc32(bytes) {
  let value = 0xffffffff;
  for (const byte of bytes) {
    value ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value >>> 1) ^ (value & 1 ? 0xedb88320 : 0);
    }
  }
  return (value ^ 0xffffffff) >>> 0;
}

/** @param {readonly Uint8Array[]} pieces */
function joinBytes(pieces) {
  const result = new Uint8Array(
    pieces.reduce((total, piece) => total + piece.byteLength, 0),
  );
  let offset = 0;
  for (const piece of pieces) {
    result.set(piece, offset);
    offset += piece.byteLength;
  }
  return result;
}

/** @param {string} type @param {Uint8Array} data */
function chunk(type, data) {
  const typeBytes = new TextEncoder().encode(type);
  return joinBytes([
    u32(data.byteLength),
    typeBytes,
    data,
    u32(independentCrc32(joinBytes([typeBytes, data]))),
  ]);
}

/** @param {number} width @param {number} height */
function syntheticCanvasPng(width = 1280, height = 720) {
  const ihdr = new Uint8Array(13);
  ihdr.set(u32(width), 0);
  ihdr.set(u32(height), 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  const compressedPixel = new Uint8Array(
    deflateSync(new Uint8Array([0, 17, 24, 39, 255])),
  );
  return joinBytes([
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("tEXt", new TextEncoder().encode("Software\0synthetic-test")),
    chunk("IDAT", compressedPixel),
    chunk("IEND", new Uint8Array()),
  ]);
}

/** @param {Uint8Array} bytes */
function independentlyParseChunks(bytes) {
  assert.deepEqual([...bytes.slice(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  const chunks = [];
  let offset = 8;
  while (offset < bytes.byteLength) {
    const length = new DataView(
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength,
    ).getUint32(offset, false);
    const typeBytes = bytes.slice(offset + 4, offset + 8);
    const data = bytes.slice(offset + 8, offset + 8 + length);
    const expected = new DataView(
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength,
    ).getUint32(offset + 8 + length, false);
    assert.equal(independentCrc32(joinBytes([typeBytes, data])), expected);
    const end = offset + 12 + length;
    chunks.push({
      type: new TextDecoder().decode(typeBytes),
      data,
      raw: bytes.slice(offset, end),
    });
    offset = end;
  }
  assert.equal(offset, bytes.byteLength);
  return chunks;
}

/** @param {unknown} value @returns {string} */
function independentCanonicalJson(value) {
  if (value === null || typeof value !== "object") {
    const encoded = JSON.stringify(value);
    if (encoded === undefined) throw new TypeError("Unsupported JSON value.");
    return encoded;
  }
  if (Array.isArray(value)) {
    return `[${value.map(independentCanonicalJson).join(",")}]`;
  }
  const record = /** @type {Record<string, unknown>} */ (value);
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${independentCanonicalJson(record[key])}`)
    .join(",")}}`;
}

/** @param {unknown} value @returns {Set<string>} */
function recursiveObjectKeys(value) {
  const keys = new Set();
  if (!value || typeof value !== "object") return keys;
  for (const [key, child] of Object.entries(value)) {
    keys.add(key);
    for (const nested of recursiveObjectKeys(child)) keys.add(nested);
  }
  return keys;
}

test("provenance projector accepts exactly three branded replay leaves", () => {
  for (const name of replayNames) {
    const presentation = presentations[name];
    const provenance = projectReplayPngProvenanceV1(projectionOptions(presentation));
    assert.equal(provenance.presentation_kind, presentation.presentation_kind);
    assert.equal(provenance.product_kind, "replay_viewer");
    assert.equal(provenance.presentation.render_policy, "replay_static");
    assert.equal(provenance.presentation.scale_factor, 2);
    assert.equal(provenance.presentation.pixel_width, 1280);
    assert.equal(provenance.presentation.pixel_height, 720);
    assert.deepEqual(
      Object.keys(provenance.presentation.visual_filters),
      VISUAL_FILTER_IDS,
    );
    assert.equal(Object.isFrozen(provenance), true);
    assert.equal(Object.isFrozen(provenance.presentation), true);
    assert.equal(Object.isFrozen(provenance.presentation.visual_filters), true);
  }
  for (const name of liveNames) {
    assert.throws(
      () => projectReplayPngProvenanceV1(projectionOptions(presentations[name])),
      /only strict Replay Viewer leaves/u,
    );
  }
});

test("Oracle and Agent provenance use disjoint authority-safe source shapes", () => {
  const oracle = projectReplayPngProvenanceV1(
    projectionOptions(presentations.replay_oracle),
  );
  assert.deepEqual(oracle.authority, { audience: "oracle" });
  assert.deepEqual(Object.keys(oracle.source), [
    "episode_id",
    "authorized_endpoint_digest_sha256",
    "artifact_id",
    "replay_schema_version",
    "artifact_digest_sha256",
  ]);
  assert.equal(oracle.source.artifact_digest_sha256.length, 64);

  for (const name of replayNames.slice(1)) {
    const agent = projectReplayPngProvenanceV1(projectionOptions(presentations[name]));
    assert.deepEqual(Object.keys(agent.source), [
      "episode_id",
      "authorized_endpoint_digest_sha256",
    ]);
    assert.deepEqual(Object.keys(agent.authority), [
      "audience",
      "observation_mode",
      "recipient_public_agent_id",
    ]);
    const keys = recursiveObjectKeys(agent);
    for (const forbidden of [
      "artifact_id",
      "artifact_digest",
      "context_digest",
      "trajectory_content_digest",
      "session_id",
      "revision",
      "generation",
      "presentation_key",
      "timeline_id",
      "global_slot",
    ]) {
      assert.equal(keys.has(forbidden), false, forbidden);
    }
  }
});

test("projection rejects lookalikes, accessors, symbols, and extras", () => {
  const branded = presentations.replay_oracle;
  assert.throws(
    () =>
      projectReplayPngProvenanceV1({
        ...projectionOptions(branded),
        presentation: structuredClone(branded),
      }),
    /branded authorized presentation/u,
  );
  let accessorReads = 0;
  const accessor = { ...projectionOptions(branded) };
  Object.defineProperty(accessor, "presentation", {
    enumerable: true,
    get() {
      accessorReads += 1;
      return branded;
    },
  });
  assert.throws(() => projectReplayPngProvenanceV1(accessor), /enumerable data field/u);
  assert.equal(accessorReads, 0);
  assert.throws(
    () =>
      projectReplayPngProvenanceV1({
        ...projectionOptions(branded),
        unexpected: true,
      }),
    /unknown or missing fields/u,
  );
  const symbolic = /** @type {Record<PropertyKey, any>} */ ({
    ...projectionOptions(branded),
  });
  symbolic[Symbol("hidden")] = true;
  assert.throws(() => projectReplayPngProvenanceV1(symbolic), /symbol fields/u);
  const filterAccessor = { ...DEFAULT_VISUAL_FILTER_STATE };
  Object.defineProperty(filterAccessor, "aura_fields", {
    enumerable: true,
    get() {
      accessorReads += 1;
      return true;
    },
  });
  assert.throws(
    () =>
      projectReplayPngProvenanceV1({
        ...projectionOptions(branded),
        visualFilters: filterAccessor,
      }),
    /enumerable data field/u,
  );
  assert.equal(accessorReads, 0);
});

test("selection is resolved from the painted presentation key, never a caller label", () => {
  const presentation = presentations.replay_no_shared_obs_agent_pov;
  const selected = presentation.scene.agents[0];
  const provenance = projectReplayPngProvenanceV1({
    ...projectionOptions(presentation),
    localInspectedPresentationKey: selected.presentation_key,
  });
  assert.equal(
    provenance.presentation.selected_public_agent_id,
    selected.public_agent_id,
  );
  const relabelAttempt = projectReplayPngProvenanceV1({
    ...projectionOptions(presentation),
    localInspectedPresentationKey: selected.public_agent_id,
  });
  assert.equal(relabelAttempt.presentation.selected_public_agent_id, null);
  assert.equal(JSON.stringify(provenance).includes(selected.presentation_key), false);
});

test("canonical provenance recursively sorts keys and preserves exact UTF-8", () => {
  const original = projectReplayPngProvenanceV1(
    projectionOptions(presentations.replay_oracle),
  );
  const hostileUnicode = structuredClone(original);
  hostileUnicode.source.episode_id = "episódio-東京-π";
  const canonical = canonicalReplayPngProvenanceV1(hostileUnicode);
  assert.equal(canonical.json, independentCanonicalJson(canonical.provenance));
  assert.equal(new TextDecoder().decode(canonical.utf8), canonical.json);
  assert.match(canonical.json, /episódio-東京-π/u);
  assert.equal(canonical.json.endsWith("\n"), false);
  assert.deepEqual(Object.keys(JSON.parse(canonical.json)), [
    "authority",
    "frame",
    "presentation",
    "presentation_kind",
    "product_kind",
    "schema_id",
    "schema_version",
    "source",
  ]);

  const malformed = structuredClone(original);
  malformed.presentation.extra = true;
  assert.throws(
    () => canonicalReplayPngProvenanceV1(malformed),
    /unknown or missing fields/u,
  );
  const nonfinite = structuredClone(original);
  nonfinite.presentation.css_width = Number.NaN;
  assert.throws(() => canonicalReplayPngProvenanceV1(nonfinite), /safe integer/u);
});

test("safe PNG filenames are bounded ASCII and authority-disjoint", () => {
  assert.equal(sanitizeReplayEpisodeForPngFilenameV1("../../.."), "replay");
  assert.equal(sanitizeReplayRecipientForPngFilenameV1("💥東京"), "agent");
  assert.equal(
    sanitizeReplayRecipientForPngFilenameV1("..agent/alpha\\beta.."),
    "agent-alpha-beta",
  );
  assert.equal(
    sanitizeReplayEpisodeForPngFilenameV1("safe.marlbg-metrics.json"),
    "safe",
  );
  assert.equal(
    sanitizeReplayRecipientForPngFilenameV1(
      "safe.marlbg-metrics.json.marlbg-metrics.json",
    ),
    "safe",
  );
  assert.equal(sanitizeReplayEpisodeForPngFilenameV1("a".repeat(200)).length, 64);
  const oracle = buildReplayBattlefieldPngFilenameV1({
    episodeId: `.hidden/..\\C:\\tmp/episódio/${"x".repeat(200)}`,
    audience: "oracle",
    artifactDigestSha256: "a".repeat(64),
    recipientPublicAgentId: null,
    frameIndex: Number.MAX_SAFE_INTEGER,
    simulatorStepCount: Number.MAX_SAFE_INTEGER,
  });
  assert.match(
    oracle,
    /__oracle-aaaaaaaa__frame-9007199254740991__tick-9007199254740991__presentation\.png$/u,
  );
  assert.equal(/^[\x20-\x7e]+$/u.test(oracle), true);
  assert.ok(new TextEncoder().encode(oracle).byteLength <= 240);
  assert.equal(oracle.includes("/"), false);
  assert.equal(oracle.includes("\\"), false);

  const agent = buildReplayBattlefieldPngFilenameV1({
    episodeId: "episode",
    audience: "agent_pov",
    artifactDigestSha256: null,
    recipientPublicAgentId: "💥".repeat(100),
    frameIndex: 7,
    simulatorStepCount: 12,
  });
  assert.equal(
    agent,
    "episode__agent-pov-agent__frame-000007__tick-000012__presentation.png",
  );
  assert.doesNotMatch(agent, /aaaaaaaa/u);
  assert.throws(
    () =>
      buildReplayBattlefieldPngFilenameV1({
        episodeId: "episode",
        audience: "agent_pov",
        artifactDigestSha256: "a".repeat(64),
        recipientPublicAgentId: "agent",
        frameIndex: 0,
        simulatorStepCount: 0,
      }),
    /cannot contain an artifact digest/u,
  );
});

test("artifact builder inserts one canonical uncompressed iTXt after IHDR", async () => {
  const options = {
    ...projectionOptions(presentations.replay_oracle),
    canvasPngBytes: syntheticCanvasPng(),
  };
  const first = buildReplayBattlefieldPngArtifactV1(options);
  const second = buildReplayBattlefieldPngArtifactV1(options);
  assert.equal(Object.isFrozen(first), true);
  assert.equal(first.blob.type, "image/png");
  assert.equal(first.filename.endsWith("__presentation.png"), true);
  const firstBytes = new Uint8Array(await first.blob.arrayBuffer());
  const secondBytes = new Uint8Array(await second.blob.arrayBuffer());
  assert.deepEqual(firstBytes, secondBytes);
  assert.equal(first.byteLength, firstBytes.byteLength);

  const chunks = independentlyParseChunks(firstBytes);
  assert.deepEqual(
    chunks.map(({ type }) => type),
    ["IHDR", "iTXt", "tEXt", "IDAT", "IEND"],
  );
  assert.deepEqual(
    chunks.slice(2).map(({ raw }) => raw),
    independentlyParseChunks(options.canvasPngBytes)
      .slice(1)
      .map(({ raw }) => raw),
  );
  const keywordBytes = new TextEncoder().encode(REPLAY_PNG_PROVENANCE_KEYWORD);
  assert.deepEqual(chunks[1].data.slice(0, keywordBytes.byteLength), keywordBytes);
  assert.deepEqual(
    [...chunks[1].data.slice(keywordBytes.byteLength, keywordBytes.byteLength + 5)],
    [0, 0, 0, 0, 0],
  );

  const inspected = inspectReplayBattlefieldPngV1(firstBytes);
  assert.equal(inspected.width, 1280);
  assert.equal(inspected.height, 720);
  assert.equal(inspected.provenanceKeywordCount, 1);
  assert.equal(inspected.canonicalProvenanceJson, first.canonicalProvenanceJson);
  assert.deepEqual(inspected.provenance, first.provenance);
});

test("PNG validation rejects corruption, truncation, duplicates, and dimension drift", async () => {
  const baseOptions = {
    ...projectionOptions(presentations.replay_oracle),
    canvasPngBytes: syntheticCanvasPng(),
  };
  const corrupt = syntheticCanvasPng();
  corrupt[corrupt.length - 5] ^= 1;
  assert.throws(
    () =>
      buildReplayBattlefieldPngArtifactV1({
        ...baseOptions,
        canvasPngBytes: corrupt,
      }),
    /CRC is invalid/u,
  );
  assert.throws(
    () =>
      buildReplayBattlefieldPngArtifactV1({
        ...baseOptions,
        canvasPngBytes: syntheticCanvasPng().slice(0, -1),
      }),
    /out of bounds|truncated/u,
  );
  assert.throws(
    () =>
      buildReplayBattlefieldPngArtifactV1({
        ...baseOptions,
        canvasPngBytes: syntheticCanvasPng(1279, 720),
      }),
    /dimensions do not match/u,
  );
  const once = buildReplayBattlefieldPngArtifactV1(baseOptions);
  const onceBytes = new Uint8Array(await once.blob.arrayBuffer());
  assert.throws(
    () =>
      buildReplayBattlefieldPngArtifactV1({
        ...baseOptions,
        canvasPngBytes: onceBytes,
      }),
    /already contains replay provenance/u,
  );
  const originalChunks = independentlyParseChunks(syntheticCanvasPng());
  const keyword = new TextEncoder().encode(REPLAY_PNG_PROVENANCE_KEYWORD);
  /** @type {Array<[string, Uint8Array]>} */
  const injectedTextChunks = [
    ["tEXt", new TextEncoder().encode("forged")],
    ["zTXt", joinBytes([new Uint8Array([0]), new Uint8Array(deflateSync("forged"))])],
  ];
  for (const [type, suffix] of injectedTextChunks) {
    const injected = joinBytes([
      new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
      originalChunks[0].raw,
      chunk(type, joinBytes([keyword, new Uint8Array([0]), suffix])),
      ...originalChunks.slice(1).map(({ raw }) => raw),
    ]);
    assert.throws(
      () =>
        buildReplayBattlefieldPngArtifactV1({
          ...baseOptions,
          canvasPngBytes: injected,
        }),
      /already contains replay provenance/u,
      type,
    );
  }
  const trailing = joinBytes([syntheticCanvasPng(), new Uint8Array([0])]);
  assert.throws(
    () =>
      buildReplayBattlefieldPngArtifactV1({
        ...baseOptions,
        canvasPngBytes: trailing,
      }),
    /trailing bytes/u,
  );
});

test("Agent artifacts retain public provenance while excluding Oracle identity", async () => {
  for (const name of replayNames.slice(1)) {
    const filters = setVisualFilterEnabled(
      DEFAULT_VISUAL_FILTER_STATE,
      "damage_effects",
      false,
    );
    const artifact = buildReplayBattlefieldPngArtifactV1({
      ...projectionOptions(presentations[name]),
      showRanges: false,
      visualFilters: filters,
      canvasPngBytes: syntheticCanvasPng(),
    });
    const bytes = new Uint8Array(await artifact.blob.arrayBuffer());
    const inspected = inspectReplayBattlefieldPngV1(bytes);
    assert.equal(inspected.provenance.authority.audience, "agent_pov");
    assert.equal(inspected.provenance.presentation.show_ranges, false);
    assert.equal(
      inspected.provenance.presentation.visual_filters.damage_effects,
      false,
    );
    assert.equal(artifact.filename.includes("oracle"), false);
    assert.equal(artifact.filename.includes("cccccccc"), false);
    for (const forbidden of [
      "artifact_id",
      "artifact_digest_sha256",
      "source_artifact",
      "source_session",
      "presentation_key",
    ]) {
      assert.equal(artifact.canonicalProvenanceJson.includes(forbidden), false);
    }
  }
});

test("background and opacity proof bind the exact battlefield-only contract", () => {
  assert.deepEqual(REPLAY_BATTLEFIELD_BACKGROUND_V1, {
    baseColor: "#111827",
    gridColor: "rgba(42, 58, 84, 0.17)",
    gridLineCssPixels: 1,
    gridSpacingRem: 2,
  });
  assert.equal(Object.isFrozen(REPLAY_BATTLEFIELD_BACKGROUND_V1), true);
  assert.equal(
    assertOpaqueReplayExportPixelsV1(
      new Uint8Array([17, 24, 39, 255, 42, 58, 84, 255]),
    ),
    2,
  );
  assert.throws(
    () =>
      assertOpaqueReplayExportPixelsV1(
        new Uint8Array([17, 24, 39, 255, 42, 58, 84, 254]),
      ),
    /all be opaque/u,
  );
  assert.throws(
    () => assertOpaqueReplayExportPixelsV1(new Uint8Array([1, 2, 3])),
    /RGBA buffer/u,
  );
});

test("deferred font loading cannot relabel the detached export snapshot", async () => {
  const pair = fixture.continuity_pairs.shared_obs;
  const installedAuthority = await joinTransportAndAuthorizedPresentationV1(
    pair.transport,
    pair.presentation,
  );
  const ownerDocument = /** @type {any} */ ({});
  let cloneCount = 0;

  class FakeSvgElement {
    /** @param {string} localName */
    constructor(localName) {
      this.localName = localName;
      this.ownerDocument = ownerDocument;
      /** @type {FakeSvgElement[]} */
      this.children = [];
      /** @type {FakeSvgElement | null} */
      this.parentElement = null;
      /** @type {Record<string, string>} */
      this.dataset = {};
      this.clientWidth = 0;
      this.clientHeight = 0;
      this.textContent = "";
      /** @type {Map<string, string>} */
      this.attributeMap = new Map();
    }

    get id() {
      return this.getAttribute("id") ?? "";
    }

    /** @param {string} value */
    set id(value) {
      this.setAttribute("id", value);
    }

    get attributes() {
      return [...this.attributeMap].map(([name, value]) => ({ name, value }));
    }

    /** @param {string} name @param {unknown} value */
    setAttribute(name, value) {
      this.attributeMap.set(name, String(value));
    }

    /** @param {string} name */
    getAttribute(name) {
      return this.attributeMap.get(name) ?? null;
    }

    /** @param {string} name */
    removeAttribute(name) {
      this.attributeMap.delete(name);
    }

    /** @param {...FakeSvgElement} children */
    append(...children) {
      for (const child of children) child.parentElement = this;
      this.children.push(...children);
    }

    /** @param {...FakeSvgElement} children */
    prepend(...children) {
      for (const child of children) child.parentElement = this;
      this.children.unshift(...children);
    }

    cloneNode() {
      cloneCount += 1;
      const clone = new FakeSvgElement(this.localName);
      clone.dataset = { ...this.dataset };
      clone.clientWidth = this.clientWidth;
      clone.clientHeight = this.clientHeight;
      clone.textContent = this.textContent;
      clone.attributeMap = new Map(this.attributeMap);
      for (const child of this.children) clone.append(child.cloneNode());
      return clone;
    }

    /** @param {string} selector */
    querySelectorAll(selector) {
      assert.equal(selector, "*");
      /** @type {FakeSvgElement[]} */
      const descendants = [];
      const visit = (/** @type {FakeSvgElement} */ element) => {
        for (const child of element.children) {
          descendants.push(child);
          visit(child);
        }
      };
      visit(this);
      return descendants;
    }

    /** @param {string} selector */
    querySelector(selector) {
      return this.querySelectorAll("*").find(
        (element) => element.localName === selector,
      );
    }
  }

  class FakeImage {
    constructor() {
      /** @type {Map<string, () => void>} */
      this.listeners = new Map();
    }

    /** @param {string} type @param {() => void} listener */
    addEventListener(type, listener) {
      this.listeners.set(type, listener);
    }

    /** @param {string} value */
    set src(value) {
      assert.match(value, /^data:image\/svg\+xml;charset=utf-8,/u);
      serializedCapture = decodeURIComponent(
        value.slice("data:image/svg+xml;charset=utf-8,".length),
      );
      queueMicrotask(() => this.listeners.get("load")?.());
    }
  }

  class FakeXmlSerializer {
    /** @param {FakeSvgElement} value */
    serializeToString(value) {
      assert.equal(value.id, "battlefield");
      const elements = [value, ...value.querySelectorAll("*")];
      assert.equal(
        elements.some((element) => element.getAttribute("style") !== null),
        false,
      );
      const styleSheets = elements.filter((element) => element.localName === "style");
      assert.equal(styleSheets.length, 2);
      assert.equal(
        styleSheets.filter((style) =>
          style.textContent.includes(".replay-export-resolved-v1-0{"),
        ).length,
        1,
      );
      assert.equal(
        styleSheets
          .flatMap((style) => [
            ...(style.textContent.matchAll(/url\(([^)]+)\)/giu) ?? []),
          ])
          .filter((match) => String(match[1]).startsWith("data:font/woff2;base64,"))
          .length,
        2,
      );

      /** @param {FakeSvgElement} element @returns {string} */
      const serialize = (element) => {
        const attributes = element.attributes
          .map(({ name, value: attributeValue }) => `${name}="${attributeValue}"`)
          .join(" ");
        const prefix = attributes ? ` ${attributes}` : "";
        return `<${element.localName}${prefix}>${element.textContent}${element.children
          .map(serialize)
          .join("")}</${element.localName}>`;
      };
      return serialize(value);
    }
  }

  const battlefield = new FakeSvgElement("svg");
  battlefield.id = "battlefield";
  battlefield.dataset.renderPolicy = "replay_static";
  battlefield.clientWidth = 640;
  battlefield.clientHeight = 360;
  const shell = new FakeSvgElement("div");
  shell.id = "battlefield-shell";
  shell.setAttribute("aria-busy", "false");
  battlefield.parentElement = shell;
  const documentElement = new FakeSvgElement("html");
  ownerDocument.documentElement = documentElement;
  ownerDocument.fonts = { ready: Promise.resolve() };
  ownerDocument.defaultView = {
    /** @param {FakeSvgElement} element */
    getComputedStyle(element) {
      if (element === shell) {
        return {
          backgroundColor: "rgb(17, 24, 39)",
          backgroundImage:
            "linear-gradient(rgba(42, 58, 84, 0.17) 1px, rgba(0, 0, 0, 0) 1px), linear-gradient(90deg, rgba(42, 58, 84, 0.17) 1px, rgba(0, 0, 0, 0) 1px)",
          backgroundSize: "32px 32px",
          getPropertyValue() {
            return "";
          },
        };
      }
      if (element === documentElement) {
        return {
          fontSize: "16px",
          getPropertyValue() {
            return "";
          },
        };
      }
      return {
        /** @param {string} property */
        getPropertyValue(property) {
          if (property === "display") return "block";
          if (property === "font-family") return '"Atkinson Hyperlegible"';
          return "";
        },
      };
    },
  };
  ownerDocument.createElementNS = (
    /** @type {string} */ namespace,
    /** @type {string} */ localName,
  ) => {
    assert.equal(namespace, "http://www.w3.org/2000/svg");
    return new FakeSvgElement(localName);
  };
  ownerDocument.createElement = (/** @type {string} */ localName) => {
    assert.equal(localName, "canvas");
    const canvas = {
      width: 0,
      height: 0,
      getContext() {
        return {
          fillStyle: "",
          fillRect() {},
          drawImage() {},
          getImageData() {
            const data = new Uint8ClampedArray(canvas.width * canvas.height * 4);
            for (let offset = 3; offset < data.byteLength; offset += 4) {
              data[offset] = 255;
            }
            return { data };
          },
        };
      },
      /** @param {(blob: Blob | null) => void} callback */
      toBlob(callback) {
        callback(
          new Blob([syntheticCanvasPng(canvas.width, canvas.height)], {
            type: "image/png",
          }),
        );
      },
    };
    return canvas;
  };

  const previousSvg = Object.getOwnPropertyDescriptor(globalThis, "SVGSVGElement");
  const previousImage = Object.getOwnPropertyDescriptor(globalThis, "Image");
  const previousSerializer = Object.getOwnPropertyDescriptor(
    globalThis,
    "XMLSerializer",
  );
  const previousFetch = Object.getOwnPropertyDescriptor(globalThis, "fetch");
  let serializedCapture = "";
  Object.defineProperty(globalThis, "SVGSVGElement", {
    configurable: true,
    value: FakeSvgElement,
  });
  Object.defineProperty(globalThis, "Image", {
    configurable: true,
    value: FakeImage,
  });
  Object.defineProperty(globalThis, "XMLSerializer", {
    configurable: true,
    value: FakeXmlSerializer,
  });
  const fontGate = deferred();
  let fontRequests = 0;
  Object.defineProperty(globalThis, "fetch", {
    configurable: true,
    value: async (/** @type {string} */ route) => {
      assert.match(route, /AtkinsonHyperlegible-(?:Regular|Bold)\.woff2$/u);
      fontRequests += 1;
      await fontGate.promise;
      return {
        ok: true,
        status: 200,
        async arrayBuffer() {
          return new Uint8Array([1, 2, 3, fontRequests]).buffer;
        },
      };
    },
  });

  const visualFilters = { ...DEFAULT_VISUAL_FILTER_STATE };
  const firstAgent = installedAuthority.presentation.scene.agents[0];
  /** @type {Record<string, any>} */
  const captureOptions = {
    battlefield,
    installedAuthority,
    isCurrent: () => true,
    transportState: "SETTLED",
    renderPolicy: "replay_static",
    showRanges: true,
    localInspectedPresentationKey: firstAgent.presentation_key,
    visualFilters,
  };
  try {
    const pending = captureReplayBattlefieldPngV1(captureOptions);
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(cloneCount, 1);
    assert.equal(fontRequests, 2);

    visualFilters.damage_effects = false;
    captureOptions.localInspectedPresentationKey = null;
    captureOptions.showRanges = false;
    fontGate.resolve();
    const artifact = await pending;

    assert.equal(artifact.provenance.presentation.show_ranges, true);
    assert.equal(artifact.provenance.presentation.visual_filters.damage_effects, true);
    assert.equal(
      artifact.provenance.presentation.selected_public_agent_id,
      firstAgent.public_agent_id,
    );
    assert.equal(
      artifact.provenance.presentation.selected_public_agent_id.includes("pov_"),
      false,
    );
    assert.doesNotMatch(serializedCapture, /\sstyle=/iu);
    assert.match(serializedCapture, /class="replay-export-resolved-v1-0"/u);
    assert.match(serializedCapture, /\.replay-export-resolved-v1-0\{/u);
    assert.equal(
      [...serializedCapture.matchAll(/url\(data:font\/woff2;base64,/gu)].length,
      2,
    );
  } finally {
    /** @type {Array<[string, PropertyDescriptor | undefined]>} */
    const globalDescriptors = [
      ["SVGSVGElement", previousSvg],
      ["Image", previousImage],
      ["XMLSerializer", previousSerializer],
      ["fetch", previousFetch],
    ];
    for (const [name, descriptor] of globalDescriptors) {
      if (descriptor) {
        Object.defineProperty(globalThis, name, descriptor);
      } else {
        Reflect.deleteProperty(globalThis, name);
      }
    }
  }
});

test("resolved capture styles are CSP-clean and confined to the detached SVG", async () => {
  const source = await readFile(sourceUrl, "utf8");
  assert.doesNotMatch(source, /\.setAttribute\(\s*["']style["']/u);
  assert.match(source, /cloned\.setAttribute\("class", className\)/u);
  assert.match(source, /style\.textContent = rules\.join\(""\)/u);
  assert.match(source, /bundledFontUrlCount !== 2/u);
  assert.doesNotMatch(source, /document\.(?:head|body)\.append/u);
});

test("full capture owns a detached CSP-compatible snapshot and no transport API", async () => {
  const source = await readFile(sourceUrl, "utf8");
  assert.match(source, /battlefield\.cloneNode\(true\)/u);
  assert.match(source, /await battlefield\.ownerDocument\.fonts\.ready/u);
  assert.match(source, /data:image\/svg\+xml;charset=utf-8/u);
  assert.match(source, /AtkinsonHyperlegible-Regular\.woff2/u);
  assert.match(source, /AtkinsonHyperlegible-Bold\.woff2/u);
  assert.match(source, /getImageData\(0, 0, canvas\.width, canvas\.height\)/u);
  assert.match(source, /getContext\("2d", \{ alpha: false \}\)/u);
  assert.doesNotMatch(source, /URL\.createObjectURL|blob:image/u);
  assert.doesNotMatch(source, /serializedSvg/u);
  assert.doesNotMatch(
    source,
    /from "\.\/(?:api|main|replay-controls|presentation-install)\.js"/u,
  );
  assert.doesNotMatch(source, /postCommand|postReplayCommand|fetch\("\/api\//u);
  const fontReady = source.indexOf("await battlefield.ownerDocument.fonts.ready");
  const revalidation = source.indexOf("options.isCurrent()", fontReady);
  const clone = source.indexOf("detachBattlefieldSnapshot(", revalidation);
  const fontFetch = source.indexOf("await loadBundledFontBytes()", clone);
  assert.ok(fontReady < revalidation);
  assert.ok(revalidation < clone);
  assert.ok(clone < fontFetch);
  assert.equal(typeof captureReplayBattlefieldPngV1, "function");
});
