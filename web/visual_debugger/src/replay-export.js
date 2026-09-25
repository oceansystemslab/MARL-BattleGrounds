/**
 * @file Export the painted Replay Viewer battlefield as a self-contained PNG.
 * Capture a settled authorized frame, embed its canonical provenance and
 * retain the declared audience. Helpers validate metadata, PNG containers
 * and SVG resources; they do not prove image pixels came from a simulator.
 * Browser capture loads bundled fonts and uses SVG/Canvas, without saving
 * a file or binding UI events. The caller owns the eventual download.
 */
import { authorizedPresentationSceneView } from "./authorized-presentation-adapter.js";
import {
  isJoinedTransportAndAuthorizedPresentationV1,
  isNormalizedAuthorizedPresentationFrameV1,
} from "./authorized-presentation-normalizer.js";
import { VISUAL_FILTER_IDS } from "./visual-filters.js";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const PNG_MIME_TYPE = "image/png";
const PNG_SIGNATURE = Object.freeze([137, 80, 78, 71, 13, 10, 26, 10]);
const PNG_PROVENANCE_KEYWORD = "MARL-BattleGrounds Replay Provenance";
const PROVENANCE_SCHEMA_ID = "marl_battlegrounds.replay_battlefield_png_provenance";
// New exports write this provenance version. Version 2 added red_zone_floors.
const CURRENT_PROVENANCE_SCHEMA_VERSION = 2;
// The 19 filter IDs of version-1 provenance, frozen so saved PNGs stay readable
// when the live registry grows.
const REPLAY_PNG_VISUAL_FILTER_IDS_V1 = Object.freeze([
  "aura_fields",
  "aura_modifier_badges",
  "duration_status_badges",
  "spawn_shield",
  "target_selection_visuals",
  "basic_ability_effects",
  "ultimate_ability_effects",
  "regeneration_effects",
  "cooldown_effects",
  "status_application",
  "natural_status_expiry",
  "freezing_trap_break",
  "status_clear_on_death",
  "death_effects",
  "respawn_wave",
  "resurrection_effects",
  "spawn_shield_expiry",
  "scrolling_battle_text",
  "death_announcer",
]);
// Exact filter IDs each provenance version records; version 2 is the live registry.
const REPLAY_PNG_VISUAL_FILTER_IDS_BY_VERSION = Object.freeze(
  new Map([
    [1, REPLAY_PNG_VISUAL_FILTER_IDS_V1],
    [2, VISUAL_FILTER_IDS],
  ]),
);
const REPLAY_PRESENTATION_KINDS = new Set([
  "replay_oracle",
  "replay_no_shared_obs_agent_pov",
  "replay_shared_obs_agent_pov",
]);
const HEX_SHA256 = /^[0-9a-f]{64}$/u;
const PNG_CHUNK_TYPE = /^[A-Za-z]{4}$/u;
const SAFE_COMPONENT_RUN = /[^A-Za-z0-9._-]+/gu;
const REGULAR_FONT_ROUTE = "/assets/fonts/AtkinsonHyperlegible-Regular.woff2";
const BOLD_FONT_ROUTE = "/assets/fonts/AtkinsonHyperlegible-Bold.woff2";
const METRIC_REPORT_SUFFIX = ".marlbg-metrics.json";
const EXPORT_STYLE_PROPERTIES = Object.freeze([
  "alignment-baseline",
  "baseline-shift",
  "box-sizing",
  "clip-path",
  "color",
  "display",
  "dominant-baseline",
  "fill",
  "fill-opacity",
  "filter",
  "font-family",
  "font-feature-settings",
  "font-kerning",
  "font-size",
  "font-stretch",
  "font-style",
  "font-variant",
  "font-weight",
  "height",
  "isolation",
  "letter-spacing",
  "line-height",
  "marker-end",
  "marker-mid",
  "marker-start",
  "mix-blend-mode",
  "opacity",
  "overflow",
  "paint-order",
  "shape-rendering",
  "stroke",
  "stroke-dasharray",
  "stroke-dashoffset",
  "stroke-linecap",
  "stroke-linejoin",
  "stroke-miterlimit",
  "stroke-opacity",
  "stroke-width",
  "text-anchor",
  "text-decoration",
  "text-rendering",
  "text-transform",
  "transform",
  "transform-box",
  "transform-origin",
  "vector-effect",
  "visibility",
  "white-space",
  "width",
  "word-spacing",
]);

export const REPLAY_PNG_PROVENANCE_KEYWORD = PNG_PROVENANCE_KEYWORD;

export const REPLAY_BATTLEFIELD_BACKGROUND_V1 = Object.freeze({
  baseColor: "#111827",
  gridColor: "rgba(42, 58, 84, 0.17)",
  gridLineCssPixels: 1,
  gridSpacingRem: 2,
});

/** @type {Promise<Readonly<{regular: Uint8Array, bold: Uint8Array}>> | null} */
let bundledFontBytesPromise = null;

/**
 * Throw TypeError with message. Used for invalid export inputs and
 * container data; this helper never returns.
 *
 * @param {string} message @returns {never}
 */
function invalid(message) {
  throw new TypeError(message);
}

/**
 * Copy one plain object's own enumerable data fields into a new record
 * with no prototype. Accept Object.prototype or null; reject arrays, other
 * prototypes, symbol keys, accessors and nonenumerable fields with TypeError
 * labelled by label. Child values remain references. Proxy reflection can
 * still run traps; this is not a sandbox for arbitrary executable objects.
 *
 * @param {unknown} value
 * @param {string} label
 * @returns {Record<string, any>}
 */
function snapshotRecord(value, label) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid(`${label} must be a plain object.`);
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(`${label} must use a plain object prototype.`);
  }
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    invalid(`${label} must not contain symbol fields.`);
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  /** @type {Record<string, any>} */
  const snapshot = Object.create(null);
  for (const [key, descriptor] of Object.entries(descriptors)) {
    if (!descriptor.enumerable || !("value" in descriptor)) {
      invalid(`${label}.${key} must be an enumerable data field.`);
    }
    snapshot[key] = descriptor.value;
  }
  return snapshot;
}

/**
 * Require value's enumerable string keys to equal expected, independent
 * of order. Throw TypeError naming label for missing/extra keys; return
 * nothing otherwise. Call snapshotRecord first to reject accessors, symbols
 * and unexpected prototypes. Neither object nor expected is modified.
 *
 * @param {Record<string, any>} value
 * @param {readonly string[]} expected
 * @param {string} label
 */
function exactKeys(value, expected, label) {
  const actual = Object.keys(value).sort();
  const canonical = [...expected].sort();
  if (
    actual.length !== canonical.length ||
    actual.some((key, index) => key !== canonical[index])
  ) {
    invalid(`${label} has unknown or missing fields.`);
  }
}

/**
 * Return value if it is a string with length greater than zero. Whitespace
 * is allowed and not trimmed. Otherwise throw TypeError naming label.
 *
 * @param {unknown} value @param {string} label
 */
function nonEmptyString(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    invalid(`${label} must be a non-empty string.`);
  }
  return value;
}

/**
 * Return a numeric safe integer value at least zero. Reject strings,
 * booleans, fractions, infinities and out-of-range values with TypeError
 * naming label. Negative zero is accepted; no rounding is performed.
 *
 * @param {unknown} value @param {string} label
 */
function nonNegativeSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || Number(value) < 0) {
    invalid(`${label} must be a non-negative safe integer.`);
  }
  return Number(value);
}

/**
 * Return a numeric safe integer greater than zero, or throw TypeError
 * naming label. Apply the same strict numeric checks as
 * nonNegativeSafeInteger; do not coerce strings or round fractions.
 *
 * @param {unknown} value @param {string} label
 */
function positiveSafeInteger(value, label) {
  const normalized = nonNegativeSafeInteger(value, label);
  if (normalized === 0) {
    invalid(`${label} must be positive.`);
  }
  return normalized;
}

/**
 * Return value when it is exactly 64 lowercase hexadecimal characters;
 * otherwise throw TypeError naming label. This validates digest spelling,
 * not the content or authenticity of the object the digest claims to identify.
 *
 * @param {unknown} value @param {string} label
 */
function sha256(value, label) {
  if (typeof value !== "string" || !HEX_SHA256.test(value)) {
    invalid(`${label} must be a lowercase SHA-256 digest.`);
  }
  return value;
}

/**
 * Copy exactly the ids boolean fields into a frozen record.
 * ids is the filter ID list of one provenance version (default: the live
 * VISUAL_FILTER_IDS, which version 2 records). Throw TypeError for
 * missing/extra fields or nonboolean values. value should already be a
 * plain-data snapshot; this helper does not inspect descriptors. No default
 * filters are filled in.
 *
 * @param {Record<string, any>} value @param {readonly string[]} [ids]
 * @returns {Readonly<Record<string, boolean>>}
 */
function normalizeVisualFilters(value, ids = VISUAL_FILTER_IDS) {
  exactKeys(value, ids, "Replay export visual filters");
  /** @type {Record<string, boolean>} */
  const normalized = {};
  for (const id of ids) {
    if (typeof value[id] !== "boolean") {
      invalid(`Replay export visual filter ${id} must be boolean.`);
    }
    normalized[id] = value[id];
  }
  return Object.freeze(normalized);
}

/**
 * Validate a version-1 or version-2 Replay Viewer provenance object and return
 * a deeply frozen plain copy with the same version. The versions differ only in
 * visual_filters: version 1 holds the 19 IDs of REPLAY_PNG_VISUAL_FILTER_IDS_V1,
 * version 2 the 20 live IDs (adding red_zone_floors); a mixed form is rejected.
 * Require exact root/nested fields, audience-specific
 * source fields, consistent replay kind/observation mode, nonnegative frame
 * counts and an incoming transition exactly when frame_index is nonzero.
 * Require replay_static, scale factor 2, positive CSS sizes, matching doubled
 * pixel sizes and complete boolean filters. Reject invalid data with TypeError.
 * Hashes are checked for format; no replay file is opened or verified here.
 *
 * @param {unknown} value @returns {Readonly<Record<string, any>>}
 */
function normalizeProvenance(value) {
  const root = snapshotRecord(value, "Replay PNG provenance");
  exactKeys(
    root,
    [
      "authority",
      "frame",
      "presentation",
      "presentation_kind",
      "product_kind",
      "schema_id",
      "schema_version",
      "source",
    ],
    "Replay PNG provenance",
  );
  const filterIds = REPLAY_PNG_VISUAL_FILTER_IDS_BY_VERSION.get(root.schema_version);
  if (
    root.schema_id !== PROVENANCE_SCHEMA_ID ||
    filterIds === undefined ||
    root.product_kind !== "replay_viewer" ||
    !REPLAY_PRESENTATION_KINDS.has(root.presentation_kind)
  ) {
    invalid("Replay PNG provenance root identity is invalid.");
  }

  const authority = snapshotRecord(root.authority, "Replay PNG authority");
  const source = snapshotRecord(root.source, "Replay PNG source");
  const frame = snapshotRecord(root.frame, "Replay PNG frame");
  const presentation = snapshotRecord(root.presentation, "Replay PNG presentation");
  const audience = authority.audience;
  if (audience === "oracle") {
    exactKeys(authority, ["audience"], "Replay PNG Oracle authority");
    exactKeys(
      source,
      [
        "artifact_digest_sha256",
        "artifact_id",
        "authorized_endpoint_digest_sha256",
        "episode_id",
        "replay_schema_version",
      ],
      "Replay PNG Oracle source",
    );
    if (root.presentation_kind !== "replay_oracle") {
      invalid("Replay PNG Oracle authority has the wrong presentation leaf.");
    }
    nonEmptyString(source.artifact_id, "Replay PNG artifact ID");
    positiveSafeInteger(
      source.replay_schema_version,
      "Replay PNG replay schema version",
    );
    sha256(source.artifact_digest_sha256, "Replay PNG artifact digest");
  } else if (audience === "agent_pov") {
    exactKeys(
      authority,
      ["audience", "observation_mode", "recipient_public_agent_id"],
      "Replay PNG Agent POV authority",
    );
    exactKeys(
      source,
      ["authorized_endpoint_digest_sha256", "episode_id"],
      "Replay PNG Agent POV source",
    );
    if (
      root.presentation_kind !== "replay_no_shared_obs_agent_pov" &&
      root.presentation_kind !== "replay_shared_obs_agent_pov"
    ) {
      invalid("Replay PNG Agent POV authority has the wrong presentation leaf.");
    }
    if (
      (root.presentation_kind === "replay_no_shared_obs_agent_pov" &&
        authority.observation_mode !== "no_shared_obs") ||
      (root.presentation_kind === "replay_shared_obs_agent_pov" &&
        authority.observation_mode !== "shared_obs_visual_union")
    ) {
      invalid("Replay PNG Agent POV observation mode is inconsistent.");
    }
    nonEmptyString(
      authority.recipient_public_agent_id,
      "Replay PNG recipient public agent ID",
    );
  } else {
    invalid("Replay PNG authority audience is invalid.");
  }
  nonEmptyString(source.episode_id, "Replay PNG episode ID");
  sha256(
    source.authorized_endpoint_digest_sha256,
    "Replay PNG authorized endpoint digest",
  );

  exactKeys(
    frame,
    ["frame_index", "incoming_transition_id", "simulator_step_count"],
    "Replay PNG frame",
  );
  nonNegativeSafeInteger(frame.frame_index, "Replay PNG frame index");
  nonNegativeSafeInteger(frame.simulator_step_count, "Replay PNG simulator step count");
  if (
    frame.incoming_transition_id !== null &&
    (typeof frame.incoming_transition_id !== "string" ||
      frame.incoming_transition_id.length === 0)
  ) {
    invalid("Replay PNG incoming transition ID must be non-empty or null.");
  }
  if ((frame.frame_index === 0) !== (frame.incoming_transition_id === null)) {
    invalid("Replay PNG incoming transition identity is epoch-incoherent.");
  }

  exactKeys(
    presentation,
    [
      "css_height",
      "css_width",
      "pixel_height",
      "pixel_width",
      "render_policy",
      "scale_factor",
      "selected_public_agent_id",
      "show_ranges",
      "visual_filters",
    ],
    "Replay PNG presentation",
  );
  if (
    presentation.render_policy !== "replay_static" ||
    presentation.scale_factor !== 2 ||
    typeof presentation.show_ranges !== "boolean"
  ) {
    invalid("Replay PNG presentation policy is invalid.");
  }
  const cssWidth = positiveSafeInteger(presentation.css_width, "Replay PNG CSS width");
  const cssHeight = positiveSafeInteger(
    presentation.css_height,
    "Replay PNG CSS height",
  );
  if (
    presentation.pixel_width !== cssWidth * 2 ||
    presentation.pixel_height !== cssHeight * 2
  ) {
    invalid("Replay PNG pixel dimensions must be exactly 2x CSS dimensions.");
  }
  if (
    presentation.selected_public_agent_id !== null &&
    (typeof presentation.selected_public_agent_id !== "string" ||
      presentation.selected_public_agent_id.length === 0)
  ) {
    invalid("Replay PNG selected public agent ID must be non-empty or null.");
  }
  const filters = normalizeVisualFilters(
    snapshotRecord(presentation.visual_filters, "Replay PNG visual filters"),
    filterIds,
  );

  return deepFreeze({
    schema_id: PROVENANCE_SCHEMA_ID,
    schema_version: root.schema_version,
    product_kind: "replay_viewer",
    presentation_kind: root.presentation_kind,
    authority: { ...authority },
    source: { ...source },
    frame: { ...frame },
    presentation: {
      ...presentation,
      visual_filters: filters,
    },
  });
}

/**
 * Recursively freeze enumerable child values and value itself, returning
 * the same reference. Primitive values pass through. Call only with owned
 * acyclic plain JSON data; this does not copy data or detect cycles.
 *
 * @param {unknown} value
 * @returns {any}
 */
function deepFreeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

/**
 * Encode already validated JSON data with sorted object keys and no
 * extra whitespace. Preserve array order and finite number/string/boolean/
 * null values. Require dense plain arrays and plain data objects; unsupported
 * values or shapes throw TypeError. The caller provides acyclic data. This
 * returns text only and does not hash or mutate the input.
 *
 * @param {unknown} value
 * @returns {string}
 */
function canonicalJson(value) {
  if (value === null || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      invalid("Canonical replay provenance numbers must be finite.");
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    if (Object.getPrototypeOf(value) !== Array.prototype) {
      invalid("Canonical replay provenance arrays must be plain.");
    }
    const keys = Reflect.ownKeys(value);
    const expected = Array.from({ length: value.length }, (_, index) => String(index));
    expected.push("length");
    if (
      keys.length !== expected.length ||
      keys.some((key, index) => key !== expected[index])
    ) {
      invalid("Canonical replay provenance arrays must be dense and exact.");
    }
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  const record = snapshotRecord(value, "Canonical replay provenance object");
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
    .join(",")}}`;
}

/**
 * Validate provenance value and return its frozen normalized record,
 * canonical compact JSON text and new UTF-8 bytes. Object keys are sorted
 * recursively; no external identity check occurs. Invalid fields throw
 * TypeError. The outer result and provenance are frozen; the returned
 * Uint8Array is a caller-owned mutable buffer.
 *
 * @param {unknown} value
 * @returns {Readonly<{provenance: Readonly<Record<string, any>>, json: string, utf8: Uint8Array}>}
 */
export function canonicalReplayPngProvenanceV1(value) {
  const provenance = normalizeProvenance(value);
  const json = canonicalJson(provenance);
  return Object.freeze({
    provenance,
    json,
    utf8: new TextEncoder().encode(json),
  });
}

/**
 * Project export metadata from a branded strict Replay Viewer presentation.
 * value must contain exactly presentation, renderPolicy, cssWidth, cssHeight,
 * showRanges, visualFilters and localInspectedPresentationKey. Use
 * replay_static, positive integer CSS sizes and complete boolean filters.
 * The inspection key may be undefined, null or a nonempty string. Resolve
 * the painted selection through the authorized scene, then record 2x pixel
 * sizes. Oracle includes artifact references; Agent POV omits those fields.
 * Return deeply frozen validated provenance at the current version 2, which
 * records all 20 live filters; invalid or unresolved inputs throw TypeError.
 * Raw transport objects cannot substitute for the brand.
 *
 * @param {unknown} value
 * @returns {Readonly<Record<string, any>>}
 */
export function projectReplayPngProvenanceV1(value) {
  const options = snapshotRecord(value, "Replay PNG projection options");
  exactKeys(
    options,
    [
      "cssHeight",
      "cssWidth",
      "localInspectedPresentationKey",
      "presentation",
      "renderPolicy",
      "showRanges",
      "visualFilters",
    ],
    "Replay PNG projection options",
  );
  const presentation = options.presentation;
  if (!isNormalizedAuthorizedPresentationFrameV1(presentation)) {
    invalid("Replay PNG export requires a branded authorized presentation.");
  }
  if (
    presentation.viewer_mode !== "replay" ||
    presentation.product_kind !== "replay_viewer" ||
    !REPLAY_PRESENTATION_KINDS.has(presentation.presentation_kind)
  ) {
    invalid("Replay PNG export accepts only strict Replay Viewer leaves.");
  }
  if (options.renderPolicy !== "replay_static") {
    invalid("Replay PNG export requires replay_static presentation.");
  }
  const cssWidth = positiveSafeInteger(options.cssWidth, "Replay export CSS width");
  const cssHeight = positiveSafeInteger(options.cssHeight, "Replay export CSS height");
  if (typeof options.showRanges !== "boolean") {
    invalid("Replay export ranges state must be boolean.");
  }
  const visualFilters = normalizeVisualFilters(
    snapshotRecord(options.visualFilters, "Replay export visual filters"),
  );
  const localInspectedPresentationKey = options.localInspectedPresentationKey;
  if (
    localInspectedPresentationKey !== undefined &&
    localInspectedPresentationKey !== null &&
    (typeof localInspectedPresentationKey !== "string" ||
      localInspectedPresentationKey.length === 0)
  ) {
    invalid(
      "Replay export local inspection key must be non-empty, null, or undefined.",
    );
  }
  const paintedScene = authorizedPresentationSceneView(
    presentation,
    localInspectedPresentationKey,
  );
  if (paintedScene === null) {
    invalid("Replay export could not resolve the painted authorized scene.");
  }
  const selectedPresentationKey = paintedScene.selection.selected_presentation_key;
  const selectedAgent =
    selectedPresentationKey === null
      ? null
      : paintedScene.agents.find(
          (/** @type {Record<string, any>} */ agent) =>
            agent.presentation_key === selectedPresentationKey,
        );
  if (selectedPresentationKey !== null && selectedAgent === undefined) {
    invalid("Replay export painted selection does not resolve to an agent.");
  }
  const selectedPublicAgentId = selectedAgent?.public_agent_id ?? null;

  const oracle = presentation.presentation_kind === "replay_oracle";
  const authority = oracle
    ? { audience: "oracle" }
    : {
        audience: "agent_pov",
        observation_mode: presentation.authority.observation_mode,
        recipient_public_agent_id: presentation.authority.recipient_public_agent_id,
      };
  const source = oracle
    ? {
        episode_id: presentation.source.episode_id,
        authorized_endpoint_digest_sha256:
          presentation.source.source_authorized_endpoint_digest_sha256,
        artifact_id: presentation.source.source_artifact_id,
        replay_schema_version: presentation.source.source_replay_schema_version,
        artifact_digest_sha256: presentation.source.source_artifact_digest_sha256,
      }
    : {
        episode_id: presentation.source.episode_id,
        authorized_endpoint_digest_sha256:
          presentation.source.source_authorized_endpoint_digest_sha256,
      };
  const incomingTransitionId = oracle
    ? presentation.technical_frame.incoming_transition_id
    : presentation.technical_frame.incoming_recipient_transition_id;

  return normalizeProvenance({
    schema_id: PROVENANCE_SCHEMA_ID,
    schema_version: CURRENT_PROVENANCE_SCHEMA_VERSION,
    product_kind: "replay_viewer",
    presentation_kind: presentation.presentation_kind,
    authority,
    source,
    frame: {
      frame_index: presentation.frame_index,
      simulator_step_count: presentation.simulator_step_count,
      incoming_transition_id: incomingTransitionId,
    },
    presentation: {
      render_policy: "replay_static",
      scale_factor: 2,
      css_width: cssWidth,
      css_height: cssHeight,
      pixel_width: cssWidth * 2,
      pixel_height: cssHeight * 2,
      show_ranges: options.showRanges,
      selected_public_agent_id: selectedPublicAgentId,
      visual_filters: visualFilters,
    },
  });
}

/**
 * Return an ASCII-safe filename component from string value. Replace runs
 * outside letters/digits/dot/underscore/hyphen, trim edge punctuation, limit
 * to maximumLength and remove the metric-report suffix. Use fallback if
 * nothing remains. Nonstring values throw TypeError. Caller supplies the
 * positive length and safe fallback; neither is independently validated.
 *
 * @param {unknown} value @param {number} maximumLength @param {string} fallback
 */
function safeFilenameComponent(value, maximumLength, fallback) {
  if (typeof value !== "string") {
    invalid("Replay PNG filename components must be strings.");
  }
  let safe = value.replace(SAFE_COMPONENT_RUN, "-").replace(/^[._-]+|[._-]+$/gu, "");
  safe = safe.slice(0, maximumLength).replace(/^[._-]+|[._-]+$/gu, "");
  safe = safe.replaceAll(METRIC_REPORT_SUFFIX, "-").replace(/^[._-]+|[._-]+$/gu, "");
  return safe || fallback;
}

/**
 * Make a filename component from episode string value, limited to 64
 * characters. Use replay if sanitizing leaves it empty. Nonstring input
 * throws TypeError. The output is a display filename, not an episode identity.
 *
 * @param {unknown} value
 */
export function sanitizeReplayEpisodeForPngFilenameV1(value) {
  return safeFilenameComponent(value, 64, "replay");
}

/**
 * Make a filename component from recipient string value, limited to 64
 * characters. Use agent if sanitizing leaves it empty. Nonstring input
 * throws TypeError. Preserve the original identity separately in provenance.
 *
 * @param {unknown} value
 */
export function sanitizeReplayRecipientForPngFilenameV1(value) {
  return safeFilenameComponent(value, 64, "agent");
}

/**
 * Build an ASCII PNG filename from exactly episodeId, audience,
 * artifactDigestSha256, recipientPublicAgentId, frameIndex and
 * simulatorStepCount. Counts must be nonnegative safe integers and receive
 * at least six digits. Oracle requires a digest and null recipient; Agent
 * POV requires a recipient and null artifact digest. Sanitize identity
 * components and reject names over 240 bytes. Return the name only; invalid
 * fields throw TypeError and no file is written.
 *
 * @param {unknown} value
 * @returns {string}
 */
export function buildReplayBattlefieldPngFilenameV1(value) {
  const fields = snapshotRecord(value, "Replay PNG filename fields");
  exactKeys(
    fields,
    [
      "artifactDigestSha256",
      "audience",
      "episodeId",
      "frameIndex",
      "recipientPublicAgentId",
      "simulatorStepCount",
    ],
    "Replay PNG filename fields",
  );
  const episode = sanitizeReplayEpisodeForPngFilenameV1(fields.episodeId);
  const frame = String(
    nonNegativeSafeInteger(fields.frameIndex, "Replay PNG filename frame"),
  ).padStart(6, "0");
  const tick = String(
    nonNegativeSafeInteger(fields.simulatorStepCount, "Replay PNG filename tick"),
  ).padStart(6, "0");
  let authority;
  if (fields.audience === "oracle") {
    if (fields.recipientPublicAgentId !== null) {
      invalid("Oracle PNG filenames cannot contain a recipient.");
    }
    authority = `oracle-${sha256(
      fields.artifactDigestSha256,
      "Replay PNG filename artifact digest",
    ).slice(0, 8)}`;
  } else if (fields.audience === "agent_pov") {
    if (fields.artifactDigestSha256 !== null) {
      invalid("Agent POV PNG filenames cannot contain an artifact digest.");
    }
    authority = `agent-pov-${sanitizeReplayRecipientForPngFilenameV1(
      fields.recipientPublicAgentId,
    )}`;
  } else {
    invalid("Replay PNG filename audience is invalid.");
  }
  const filename = `${episode}__${authority}__frame-${frame}__tick-${tick}__presentation.png`;
  if (!/^[\x20-\x7e]+$/u.test(filename)) {
    invalid("Replay PNG filename must be ASCII.");
  }
  if (new TextEncoder().encode(filename).byteLength > 240) {
    invalid("Replay PNG filename exceeds 240 bytes.");
  }
  return filename;
}

/**
 * Return a new Uint8Array containing value's bytes. Accept ArrayBuffer
 * or a Uint8Array with exactly Uint8Array.prototype, rejecting other views
 * and subclasses with TypeError naming label. The copy isolates later input
 * mutation; allocation or detached-buffer errors can propagate.
 *
 * @param {unknown} value @param {string} label
 */
function snapshotBytes(value, label) {
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value.slice(0));
  }
  if (
    value instanceof Uint8Array &&
    Object.getPrototypeOf(value) === Uint8Array.prototype
  ) {
    return new Uint8Array(value);
  }
  invalid(`${label} must be an ArrayBuffer or exact Uint8Array.`);
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value >>> 1) ^ (value & 1 ? 0xedb88320 : 0);
    }
    table[index] = value >>> 0;
  }
  return table;
})();

/**
 * Return the unsigned PNG CRC-32 checksum of bytes. Read the buffer without
 * changing it, using the module's precomputed polynomial table. This checks
 * accidental corruption; it is not a cryptographic integrity proof.
 *
 * @param {Uint8Array} bytes
 */
function crc32(bytes) {
  let value = 0xffffffff;
  for (const byte of bytes) {
    value = CRC32_TABLE[(value ^ byte) & 0xff] ^ (value >>> 8);
  }
  return (value ^ 0xffffffff) >>> 0;
}

/**
 * Read an unsigned big-endian 32-bit integer at byte offset in bytes.
 * Respect the typed view's own start/length. Invalid offsets propagate
 * DataView RangeError; this function does not modify the buffer.
 *
 * @param {Uint8Array} bytes @param {number} offset
 */
function readUint32(bytes, offset) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(
    offset,
    false,
  );
}

/**
 * Return four new bytes encoding value as a big-endian unsigned integer.
 * DataView applies JavaScript's numeric conversion; callers must validate
 * the intended nonnegative 32-bit range before this helper.
 *
 * @param {number} value
 */
function uint32Bytes(value) {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setUint32(0, value, false);
  return bytes;
}

/**
 * Copy chunks in order into one new Uint8Array and return it. Preserve
 * every input buffer. Allocation fails normally if the total length exceeds
 * the platform's available memory or typed-array limits.
 *
 * @param {readonly Uint8Array[]} chunks
 */
function concatenateBytes(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const result = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

/**
 * Convert a small byte sequence to characters with the same code points.
 * No ASCII-range validation occurs. Callers use short PNG types/keywords;
 * large buffers may exceed JavaScript's spread-argument limit.
 *
 * @param {Uint8Array} bytes
 */
function ascii(bytes) {
  return String.fromCharCode(...bytes);
}

/**
 * @typedef {Readonly<{
 *   type: string,
 *   start: number,
 *   end: number,
 *   data: Uint8Array,
 *   raw: Uint8Array,
 * }>} ParsedPngChunk
 */

/**
 * Copy PNG bytes and check signature, chunk bounds/types/CRCs, required
 * IHDR/IDAT/IEND structure, no trailing bytes and supported IHDR fields.
 * Accept the byte inputs supported by snapshotBytes; invalid containers throw
 * TypeError. Return width, height, copied bytes and frozen chunk records with
 * copied data/raw buffers. It does not decompress IDAT, decode pixels or
 * implement every PNG semantic constraint. Returned typed buffers are mutable.
 *
 * @param {unknown} value
 */
function parsePng(value) {
  const bytes = snapshotBytes(value, "Replay PNG bytes");
  if (
    bytes.byteLength < PNG_SIGNATURE.length ||
    PNG_SIGNATURE.some((byte, index) => bytes[index] !== byte)
  ) {
    invalid("Replay PNG signature is invalid.");
  }
  /** @type {ParsedPngChunk[]} */
  const chunks = [];
  let offset = PNG_SIGNATURE.length;
  let ihdrCount = 0;
  let iendCount = 0;
  let idatCount = 0;
  while (offset < bytes.byteLength) {
    if (bytes.byteLength - offset < 12) {
      invalid("Replay PNG contains a truncated chunk header.");
    }
    const length = readUint32(bytes, offset);
    const end = offset + 12 + length;
    if (!Number.isSafeInteger(end) || end > bytes.byteLength) {
      invalid("Replay PNG chunk length is out of bounds.");
    }
    const typeBytes = bytes.slice(offset + 4, offset + 8);
    const type = ascii(typeBytes);
    if (!PNG_CHUNK_TYPE.test(type) || type[2] !== type[2].toUpperCase()) {
      invalid("Replay PNG chunk type is invalid.");
    }
    const data = bytes.slice(offset + 8, offset + 8 + length);
    const expectedCrc = readUint32(bytes, offset + 8 + length);
    const actualCrc = crc32(concatenateBytes([typeBytes, data]));
    if (actualCrc !== expectedCrc) {
      invalid(`Replay PNG ${type} chunk CRC is invalid.`);
    }
    if (type === "IHDR") ihdrCount += 1;
    if (type === "IDAT") idatCount += 1;
    if (type === "IEND") iendCount += 1;
    chunks.push(
      Object.freeze({
        type,
        start: offset,
        end,
        data,
        raw: bytes.slice(offset, end),
      }),
    );
    offset = end;
    if (type === "IEND" && offset !== bytes.byteLength) {
      invalid("Replay PNG has trailing bytes after IEND.");
    }
  }
  if (
    chunks.length < 3 ||
    chunks[0].type !== "IHDR" ||
    chunks.at(-1)?.type !== "IEND" ||
    ihdrCount !== 1 ||
    iendCount !== 1 ||
    idatCount === 0 ||
    chunks[0].data.byteLength !== 13 ||
    chunks.at(-1)?.data.byteLength !== 0
  ) {
    invalid("Replay PNG chunk structure is invalid.");
  }
  const ihdr = chunks[0].data;
  const width = readUint32(ihdr, 0);
  const height = readUint32(ihdr, 4);
  const bitDepth = ihdr[8];
  const colorType = ihdr[9];
  const allowedDepths = new Map([
    [0, new Set([1, 2, 4, 8, 16])],
    [2, new Set([8, 16])],
    [3, new Set([1, 2, 4, 8])],
    [4, new Set([8, 16])],
    [6, new Set([8, 16])],
  ]);
  if (
    width === 0 ||
    height === 0 ||
    !allowedDepths.get(colorType)?.has(bitDepth) ||
    ihdr[10] !== 0 ||
    ihdr[11] !== 0 ||
    (ihdr[12] !== 0 && ihdr[12] !== 1)
  ) {
    invalid("Replay PNG IHDR fields are invalid.");
  }
  return Object.freeze({ bytes, chunks: Object.freeze(chunks), width, height });
}

/**
 * Encode type and data into new PNG chunk bytes with length and CRC.
 * Require a four-letter type or throw TypeError. The caller owns data-size
 * and semantic chunk validity; this does not check reserved-bit rules or
 * copy scientific metadata beyond the requested bytes.
 *
 * @param {string} type @param {Uint8Array} data
 */
function pngChunk(type, data) {
  const typeBytes = new TextEncoder().encode(type);
  if (typeBytes.byteLength !== 4 || !PNG_CHUNK_TYPE.test(type)) {
    invalid("PNG output chunk type is invalid.");
  }
  return concatenateBytes([
    uint32Bytes(data.byteLength),
    typeBytes,
    data,
    uint32Bytes(crc32(concatenateBytes([typeBytes, data]))),
  ]);
}

/**
 * Return the ASCII text before data's first null byte when its length is
 * 1–79 and all characters are printable ASCII. Return null otherwise. This
 * checks the keyword prefix only, not a text chunk's remaining structure.
 *
 * @param {Uint8Array} data
 */
function pngKeyword(data) {
  const end = data.indexOf(0);
  if (end < 1 || end > 79) return null;
  const keywordBytes = data.slice(0, end);
  if (keywordBytes.some((byte) => byte < 32 || byte > 126)) return null;
  return ascii(keywordBytes);
}

/**
 * Return a new uncompressed iTXt chunk with the fixed provenance keyword,
 * empty language/translation fields and canonicalProvenanceJson as UTF-8.
 * The caller must validate and canonicalize the JSON first.
 *
 * @param {string} canonicalProvenanceJson
 */
function provenanceItxtChunk(canonicalProvenanceJson) {
  const keyword = new TextEncoder().encode(PNG_PROVENANCE_KEYWORD);
  const text = new TextEncoder().encode(canonicalProvenanceJson);
  return pngChunk("iTXt", concatenateBytes([keyword, new Uint8Array(5), text]));
}

/**
 * Inspect a PNG with exactly one provenance entry directly after IHDR.
 * Require an uncompressed, untranslated UTF-8 iTXt entry with canonical
 * version-1 or version-2 provenance, matching PNG dimensions and no duplicate keyword
 * in text chunks. Return frozen metadata, chunk types and provenance. Invalid
 * bytes/metadata throw TypeError. This verifies container consistency, not
 * the truth of the picture or the referenced simulator artifacts.
 *
 * @param {unknown} value
 */
export function inspectReplayBattlefieldPngV1(value) {
  const parsed = parsePng(value);
  const matches = parsed.chunks.filter(
    (chunk) =>
      (chunk.type === "iTXt" || chunk.type === "tEXt" || chunk.type === "zTXt") &&
      pngKeyword(chunk.data) === PNG_PROVENANCE_KEYWORD,
  );
  if (
    matches.length !== 1 ||
    matches[0].type !== "iTXt" ||
    parsed.chunks[1] !== matches[0]
  ) {
    invalid("Replay PNG requires one provenance iTXt immediately after IHDR.");
  }
  const chunk = matches[0];
  const textOffset = PNG_PROVENANCE_KEYWORD.length + 5;
  if (
    chunk.data.byteLength <= textOffset ||
    chunk.data[PNG_PROVENANCE_KEYWORD.length] !== 0 ||
    chunk.data[PNG_PROVENANCE_KEYWORD.length + 1] !== 0 ||
    chunk.data[PNG_PROVENANCE_KEYWORD.length + 2] !== 0 ||
    chunk.data[PNG_PROVENANCE_KEYWORD.length + 3] !== 0 ||
    chunk.data[PNG_PROVENANCE_KEYWORD.length + 4] !== 0
  ) {
    invalid("Replay PNG provenance iTXt must be uncompressed and untranslated.");
  }
  let json;
  try {
    json = new TextDecoder("utf-8", { fatal: true }).decode(
      chunk.data.slice(textOffset),
    );
  } catch {
    invalid("Replay PNG provenance iTXt is not valid UTF-8.");
  }
  let decoded;
  try {
    decoded = JSON.parse(json);
  } catch {
    invalid("Replay PNG provenance iTXt is not valid JSON.");
  }
  const canonical = canonicalReplayPngProvenanceV1(decoded);
  if (canonical.json !== json) {
    invalid("Replay PNG provenance iTXt is not canonical JSON.");
  }
  if (
    parsed.width !== canonical.provenance.presentation.pixel_width ||
    parsed.height !== canonical.provenance.presentation.pixel_height
  ) {
    invalid("Replay PNG dimensions disagree with provenance.");
  }
  return Object.freeze({
    width: parsed.width,
    height: parsed.height,
    chunkTypes: Object.freeze(parsed.chunks.map(({ type }) => type)),
    provenanceKeywordCount: matches.length,
    canonicalProvenanceJson: json,
    provenance: canonical.provenance,
  });
}

/**
 * Build a PNG artifact from trusted Canvas bytes and a branded replay view.
 * value contains exactly canvasPngBytes plus the seven projection options
 * documented by projectReplayPngProvenanceV1. Project provenance internally;
 * callers cannot pass replacement provenance. Return a frozen artifact with
 * Blob, filename, byteLength, schemaVersion, provenance and canonical JSON.
 * Invalid metadata, PNG dimensions or existing provenance throw TypeError.
 * The caller is responsible for the bytes actually depicting that view;
 * this helper cannot verify pixel meaning and does not start a download.
 *
 * @param {unknown} value
 */
export function buildReplayBattlefieldPngArtifactV1(value) {
  const options = snapshotRecord(value, "Replay PNG artifact options");
  exactKeys(
    options,
    [
      "canvasPngBytes",
      "cssHeight",
      "cssWidth",
      "localInspectedPresentationKey",
      "presentation",
      "renderPolicy",
      "showRanges",
      "visualFilters",
    ],
    "Replay PNG artifact options",
  );
  const provenance = projectReplayPngProvenanceV1({
    presentation: options.presentation,
    renderPolicy: options.renderPolicy,
    cssWidth: options.cssWidth,
    cssHeight: options.cssHeight,
    showRanges: options.showRanges,
    localInspectedPresentationKey: options.localInspectedPresentationKey,
    visualFilters: options.visualFilters,
  });
  return buildArtifactFromProvenance(options.canvasPngBytes, provenance);
}

/**
 * Insert validated provenance immediately after Canvas PNG's IHDR and
 * verify the resulting container. Reject mismatched dimensions or existing
 * provenance with TypeError. Preserve all original chunks otherwise. Return
 * a frozen artifact record with an image/png Blob, safe filename, byte count
 * and metadata. No pixel decoding or filesystem write occurs.
 *
 * @param {unknown} canvasPngBytes
 * @param {Readonly<Record<string, any>>} provenance
 */
function buildArtifactFromProvenance(canvasPngBytes, provenance) {
  const canonical = canonicalReplayPngProvenanceV1(provenance);
  const input = parsePng(canvasPngBytes);
  if (
    input.width !== provenance.presentation.pixel_width ||
    input.height !== provenance.presentation.pixel_height
  ) {
    invalid("Canvas PNG dimensions do not match the replay export snapshot.");
  }
  if (
    input.chunks.some(
      (chunk) =>
        (chunk.type === "iTXt" || chunk.type === "tEXt" || chunk.type === "zTXt") &&
        pngKeyword(chunk.data) === PNG_PROVENANCE_KEYWORD,
    )
  ) {
    invalid("Canvas PNG already contains replay provenance.");
  }
  const outputBytes = concatenateBytes([
    new Uint8Array(PNG_SIGNATURE),
    input.chunks[0].raw,
    provenanceItxtChunk(canonical.json),
    ...input.chunks.slice(1).map(({ raw }) => raw),
  ]);
  const inspected = inspectReplayBattlefieldPngV1(outputBytes);
  if (inspected.canonicalProvenanceJson !== canonical.json) {
    invalid("Replay PNG provenance did not survive container verification.");
  }
  const authority = provenance.authority;
  const filename = buildReplayBattlefieldPngFilenameV1({
    episodeId: provenance.source.episode_id,
    audience: authority.audience,
    artifactDigestSha256:
      authority.audience === "oracle" ? provenance.source.artifact_digest_sha256 : null,
    recipientPublicAgentId:
      authority.audience === "agent_pov" ? authority.recipient_public_agent_id : null,
    frameIndex: provenance.frame.frame_index,
    simulatorStepCount: provenance.frame.simulator_step_count,
  });
  const blob = new Blob([outputBytes], { type: PNG_MIME_TYPE });
  return Object.freeze({
    schemaVersion: 1,
    filename,
    blob,
    byteLength: outputBytes.byteLength,
    provenance,
    canonicalProvenanceJson: canonical.json,
  });
}

/**
 * Copy a nonempty packed RGBA byte buffer and require every alpha byte
 * to equal 255. Accept ArrayBuffer, exact Uint8Array or exact
 * Uint8ClampedArray. Return pixel count; wrong type, non-multiple-of-four
 * length or transparency throws TypeError. Dimensions/color meaning are
 * not checked, and the input remains unchanged.
 *
 * @param {unknown} value
 */
export function assertOpaqueReplayExportPixelsV1(value) {
  const pixels =
    value instanceof Uint8ClampedArray &&
    Object.getPrototypeOf(value) === Uint8ClampedArray.prototype
      ? new Uint8Array(value)
      : snapshotBytes(value, "Replay export RGBA pixels");
  if (pixels.byteLength === 0 || pixels.byteLength % 4 !== 0) {
    invalid("Replay export pixels must be a non-empty RGBA buffer.");
  }
  for (let offset = 3; offset < pixels.byteLength; offset += 4) {
    if (pixels[offset] !== 255) {
      invalid("Replay export pixels must all be opaque.");
    }
  }
  return pixels.byteLength / 4;
}

/**
 * Encode bytes as standard padded Base64 and return the string. Read
 * without mutation; output length is four characters per three-byte group.
 * Used for bundled font data URLs, not encryption or content validation.
 *
 * @param {Uint8Array} bytes
 */
function base64(bytes) {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let encoded = "";
  for (let offset = 0; offset < bytes.byteLength; offset += 3) {
    const first = bytes[offset];
    const second = bytes[offset + 1];
    const third = bytes[offset + 2];
    const triple = (first << 16) | ((second ?? 0) << 8) | (third ?? 0);
    encoded += alphabet[(triple >>> 18) & 63];
    encoded += alphabet[(triple >>> 12) & 63];
    encoded += second === undefined ? "=" : alphabet[(triple >>> 6) & 63];
    encoded += third === undefined ? "=" : alphabet[triple & 63];
  }
  return encoded;
}

/**
 * Fetch route with force-cache and same-origin credentials and return
 * new nonempty bytes. Reject non-success HTTP status or empty content with
 * Error; network failures propagate. The caller supplies a bundled font
 * route. This does not validate WOFF2 contents or a cryptographic digest.
 *
 * @param {string} route
 */
async function fetchFontBytes(route) {
  const response = await fetch(route, {
    cache: "force-cache",
    credentials: "same-origin",
  });
  if (!response.ok) {
    throw new Error(`Bundled replay-export font request failed (${response.status}).`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength === 0) {
    throw new Error("Bundled replay-export font is empty.");
  }
  return bytes;
}

/**
 * Load regular and bold bundled fonts together and cache their promise.
 * Return a frozen record of byte arrays, shared by later exports. On failure
 * clear the cache so a future call can retry, then propagate the error.
 * Callers must treat the returned buffers as read-only.
 */
async function loadBundledFontBytes() {
  if (bundledFontBytesPromise === null) {
    bundledFontBytesPromise = Promise.all([
      fetchFontBytes(REGULAR_FONT_ROUTE),
      fetchFontBytes(BOLD_FONT_ROUTE),
    ])
      .then(([regular, bold]) => Object.freeze({ regular, bold }))
      .catch((error) => {
        bundledFontBytesPromise = null;
        throw error;
      });
  }
  return bundledFontBytesPromise;
}

/**
 * Remove UI/style/class/event/accessibility attributes from detached
 * element and remove IDs other than battlefield. Reject linked-resource
 * attributes and nonlocal URL forms with TypeError. Mutate this element
 * only; children are handled by the caller. This is one export check, not
 * a general sanitizer for arbitrary SVG.
 *
 * @param {Element} element
 */
function stripTransientAndExternalAttributes(element) {
  for (const attribute of [...element.attributes]) {
    const name = attribute.name.toLowerCase();
    const value = attribute.value;
    if (
      name === "style" ||
      name === "class" ||
      name === "tabindex" ||
      name === "focusable" ||
      name.startsWith("aria-") ||
      name.startsWith("data-") ||
      name.startsWith("on")
    ) {
      element.removeAttribute(attribute.name);
      continue;
    }
    if (name === "href" || name === "xlink:href" || name === "src") {
      invalid("Replay export SVG cannot contain linked resources.");
    }
    if (/url\((?!\s*#[A-Za-z0-9_.:-]+\s*\))/iu.test(value)) {
      invalid("Replay export SVG cannot contain external URL attributes.");
    }
  }
  if (element.id && element.id !== "battlefield") {
    element.removeAttribute("id");
  }
}

/**
 * Extract url(...) values from CSS text, trimming whitespace and matching
 * outer quotes. Throw TypeError for mismatched outer quotes. Return strings
 * in occurrence order. This is a limited URL scanner, not a complete CSS
 * parser or a resource fetcher.
 *
 * @param {string} value
 */
function cssResourceUrls(value) {
  return [...value.matchAll(/url\(([^)]+)\)/giu)].map((match) => {
    const raw = String(match[1]).trim();
    const quote = raw[0];
    if (quote === '"' || quote === "'") {
      if (raw.at(-1) !== quote) {
        invalid("Replay export CSS resource URL has mismatched quotes.");
      }
      return raw.slice(1, -1);
    }
    return raw;
  });
}

/**
 * Return whether value is a #fragment using letters, digits, underscore,
 * dot, colon or hyphen. It does not check whether that SVG target exists.
 *
 * @param {string} value
 */
function isLocalFragmentUrl(value) {
  return /^#[A-Za-z0-9_.:-]+$/u.test(value);
}

/**
 * Return whether value has the accepted WOFF2 Base64 data-URL spelling.
 * This checks syntax only, not font bytes or whether the font came from the
 * bundled routes. The caller controls which resources are inserted.
 *
 * @param {string} value
 */
function isBundledFontDataUrl(value) {
  return /^data:font\/woff2;base64,[A-Za-z0-9+/]+=*$/u.test(value);
}

/**
 * Copy supported nonempty computed style properties into CSS declarations.
 * Permit only local-fragment URLs; reject other resources with TypeError.
 * Return semicolon-separated text in declared property order. Read computed
 * without mutation; unsupported properties are intentionally not exported.
 *
 * @param {CSSStyleDeclaration} computed
 */
function resolvedStyleDeclarations(computed) {
  const declarations = [];
  for (const property of EXPORT_STYLE_PROPERTIES) {
    const value = computed.getPropertyValue(property).trim();
    if (!value) continue;
    if (cssResourceUrls(value).some((url) => !isLocalFragmentUrl(url))) {
      invalid(`Replay export resolved style ${property} contains an external URL.`);
    }
    declarations.push(`${property}:${value}`);
  }
  return declarations.join(";");
}

/**
 * Prepend a style element containing rules to clone's first defs element.
 * Throw TypeError when defs is absent. Mutate the detached clone only; rules
 * must already be resolved and checked. This avoids inline style attributes
 * that the host page's content-security policy rejects.
 *
 * @param {SVGSVGElement} clone
 * @param {readonly string[]} rules
 */
function installResolvedStyleSheet(clone, rules) {
  const defs = clone.querySelector("defs");
  if (!defs) {
    invalid("Detached replay battlefield is missing its resource container.");
  }
  const style = clone.ownerDocument.createElementNS(SVG_NAMESPACE, "style");
  style.textContent = rules.join("");
  defs.prepend(style);
}

/**
 * Read shell's computed background and return grid spacing in CSS pixels.
 * Require the locked dark background and two linear-gradient occurrences
 * containing the expected color, opacity and 1px text. Check the first
 * background-size pair is square and 2rem within 0.01px. This is a limited
 * style check, not full gradient equivalence. Invalid styles throw TypeError.
 * view must belong to shell's document; this checks export appearance without
 * changing the live page.
 *
 * @param {Element} shell @param {Window} view
 */
function resolveGridSpacing(shell, view) {
  const shellStyle = view.getComputedStyle(shell);
  const base = shellStyle.backgroundColor.replaceAll(" ", "").toLowerCase();
  if (base !== "rgb(17,24,39)" && base !== "rgba(17,24,39,1)") {
    invalid("Replay battlefield background must resolve to #111827.");
  }
  const image = shellStyle.backgroundImage.replaceAll(" ", "").toLowerCase();
  const gradientCount = (image.match(/linear-gradient\(/gu) ?? []).length;
  if (
    gradientCount !== 2 ||
    !image.includes("42,58,84") ||
    !image.includes("0.17") ||
    !image.includes("1px")
  ) {
    invalid("Replay battlefield grid must resolve to the locked 1px grid.");
  }
  const firstSize = shellStyle.backgroundSize.split(",", 1)[0].trim();
  const match = /^(\d+(?:\.\d+)?)px\s+(\d+(?:\.\d+)?)px$/u.exec(firstSize);
  if (!match || Number(match[1]) !== Number(match[2])) {
    invalid("Replay battlefield grid spacing must resolve to equal pixels.");
  }
  const spacing = Number(match[1]);
  const rootFontSize = Number.parseFloat(
    view.getComputedStyle(shell.ownerDocument.documentElement).fontSize,
  );
  if (
    !Number.isFinite(spacing) ||
    spacing <= 0 ||
    !Number.isFinite(rootFontSize) ||
    Math.abs(spacing - rootFontSize * 2) > 0.01
  ) {
    invalid("Replay battlefield grid spacing must resolve from exactly 2rem.");
  }
  return spacing;
}

/**
 * Prepend defs and opaque/grid rectangles to detached clone. width and
 * height are CSS pixels; spacing is the validated square-grid spacing.
 * Use fixed colors and 1px lines. Return nothing and do not modify the live
 * battlefield. Call once per clone to avoid duplicate resource IDs.
 *
 * @param {SVGSVGElement} clone
 * @param {number} width
 * @param {number} height
 * @param {number} spacing
 */
function installBattlefieldBackdrop(clone, width, height, spacing) {
  const document = clone.ownerDocument;
  const defs = document.createElementNS(SVG_NAMESPACE, "defs");
  const horizontal = document.createElementNS(SVG_NAMESPACE, "pattern");
  horizontal.setAttribute("id", "replay-export-grid-horizontal-v1");
  horizontal.setAttribute("patternUnits", "userSpaceOnUse");
  horizontal.setAttribute("width", String(spacing));
  horizontal.setAttribute("height", String(spacing));
  const horizontalLine = document.createElementNS(SVG_NAMESPACE, "rect");
  horizontalLine.setAttribute("width", String(spacing));
  horizontalLine.setAttribute(
    "height",
    String(REPLAY_BATTLEFIELD_BACKGROUND_V1.gridLineCssPixels),
  );
  horizontalLine.setAttribute("fill", REPLAY_BATTLEFIELD_BACKGROUND_V1.gridColor);
  horizontal.append(horizontalLine);
  defs.append(horizontal);

  const vertical = document.createElementNS(SVG_NAMESPACE, "pattern");
  vertical.setAttribute("id", "replay-export-grid-vertical-v1");
  vertical.setAttribute("patternUnits", "userSpaceOnUse");
  vertical.setAttribute("width", String(spacing));
  vertical.setAttribute("height", String(spacing));
  const verticalLine = document.createElementNS(SVG_NAMESPACE, "rect");
  verticalLine.setAttribute(
    "width",
    String(REPLAY_BATTLEFIELD_BACKGROUND_V1.gridLineCssPixels),
  );
  verticalLine.setAttribute("height", String(spacing));
  verticalLine.setAttribute("fill", REPLAY_BATTLEFIELD_BACKGROUND_V1.gridColor);
  vertical.append(verticalLine);
  defs.append(vertical);

  const base = document.createElementNS(SVG_NAMESPACE, "rect");
  base.setAttribute("width", String(width));
  base.setAttribute("height", String(height));
  base.setAttribute("fill", REPLAY_BATTLEFIELD_BACKGROUND_V1.baseColor);
  const horizontalGrid = document.createElementNS(SVG_NAMESPACE, "rect");
  horizontalGrid.setAttribute("width", String(width));
  horizontalGrid.setAttribute("height", String(height));
  horizontalGrid.setAttribute("fill", "url(#replay-export-grid-horizontal-v1)");
  const verticalGrid = document.createElementNS(SVG_NAMESPACE, "rect");
  verticalGrid.setAttribute("width", String(width));
  verticalGrid.setAttribute("height", String(height));
  verticalGrid.setAttribute("fill", "url(#replay-export-grid-vertical-v1)");
  clone.prepend(defs, base, horizontalGrid, verticalGrid);
}

/**
 * Prepend regular/bold Atkinson font-face rules with embedded WOFF2 bytes
 * to clone's first defs. Throw TypeError if defs is missing. Mutate only the
 * detached SVG; fonts must be the trusted bundled buffers. No request or font
 * content validation happens inside this helper.
 *
 * @param {SVGSVGElement} clone
 * @param {Readonly<{regular: Uint8Array, bold: Uint8Array}>} fonts
 */
function installBundledFonts(clone, fonts) {
  const defs = clone.querySelector("defs");
  if (!defs) {
    invalid("Detached replay battlefield is missing its resource container.");
  }
  const style = clone.ownerDocument.createElementNS(SVG_NAMESPACE, "style");
  style.textContent = `@font-face{font-family:"Atkinson Hyperlegible";font-style:normal;font-weight:400;src:url(data:font/woff2;base64,${base64(fonts.regular)}) format("woff2")}@font-face{font-family:"Atkinson Hyperlegible";font-style:normal;font-weight:700;src:url(data:font/woff2;base64,${base64(fonts.bold)}) format("woff2")}`;
  defs.prepend(style);
}

/**
 * Check clone and descendants for permitted local-fragment resources and
 * exactly two WOFF2 data URLs in style text. Reject linked attributes, CSS
 * imports or other URLs with TypeError. Return nothing and do not modify DOM.
 * This validates the export builder's resource policy, not arbitrary SVG
 * behavior or the authenticity of font bytes.
 *
 * @param {SVGSVGElement} clone
 */
function assertSelfContainedSvgResources(clone) {
  let bundledFontUrlCount = 0;
  for (const element of [clone, ...clone.querySelectorAll("*")]) {
    for (const attribute of [...element.attributes]) {
      const name = attribute.name.toLowerCase();
      const value = attribute.value;
      if (name === "href" || name === "xlink:href" || name === "src") {
        invalid("Replay export SVG cannot contain linked resources.");
      }
      const urls = cssResourceUrls(value);
      if (urls.some((url) => !isLocalFragmentUrl(url))) {
        invalid("Replay export SVG attribute contains an external URL.");
      }
    }
    if (element.localName === "style") {
      const css = element.textContent ?? "";
      if (/@import\b/iu.test(css)) {
        invalid("Replay export SVG styles cannot import external resources.");
      }
      for (const url of cssResourceUrls(css)) {
        if (isLocalFragmentUrl(url)) continue;
        if (isBundledFontDataUrl(url)) {
          bundledFontUrlCount += 1;
          continue;
        }
        invalid("Replay export SVG style resource is not self-contained.");
      }
    }
  }
  if (bundledFontUrlCount !== 2) {
    invalid("Replay export SVG must contain exactly two bundled font resources.");
  }
}

/**
 * Synchronously clone battlefield and copy its supported computed styles.
 * Require a window and #battlefield-shell parent, matching clone structure
 * and the locked backdrop. width/height are validated CSS pixels. Strip
 * transient attributes, set dimensions/viewBox and insert backdrop/styles.
 * Return a detached mutable SVG; invalid structure/resources throw TypeError.
 * The live DOM is read only. After this call, asynchronous painting uses the
 * clone's appearance rather than rereading live styles.
 *
 * @param {SVGSVGElement} battlefield
 * @param {number} width
 * @param {number} height
 */
function detachBattlefieldSnapshot(battlefield, width, height) {
  const document = battlefield.ownerDocument;
  const view = document.defaultView;
  const shell = battlefield.parentElement;
  if (!view || !shell || shell.id !== "battlefield-shell") {
    invalid("Replay export requires #battlefield inside #battlefield-shell.");
  }
  const clone = /** @type {SVGSVGElement} */ (battlefield.cloneNode(true));
  const originals = [battlefield, ...battlefield.querySelectorAll("*")];
  const clones = [clone, ...clone.querySelectorAll("*")];
  if (originals.length !== clones.length) {
    invalid("Replay battlefield clone is structurally inconsistent.");
  }
  const resolvedStyleRules = [];
  for (let index = 0; index < originals.length; index += 1) {
    const original = originals[index];
    const cloned = clones[index];
    if (original.localName !== cloned.localName) {
      invalid("Replay battlefield clone element order is inconsistent.");
    }
    stripTransientAndExternalAttributes(cloned);
    const className = `replay-export-resolved-v1-${index}`;
    cloned.setAttribute("class", className);
    resolvedStyleRules.push(
      `.${className}{${resolvedStyleDeclarations(view.getComputedStyle(original))}}`,
    );
  }
  clone.setAttribute("id", "battlefield");
  clone.setAttribute("xmlns", SVG_NAMESPACE);
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
  installBattlefieldBackdrop(clone, width, height, resolveGridSpacing(shell, view));
  installResolvedStyleSheet(clone, resolvedStyleRules);
  return clone;
}

/**
 * Load serialized self-contained SVG through an encoded data URL and
 * resolve to an Image after its load event. Reject decoder failure with
 * Error. No object URL cleanup is needed. The caller must have checked SVG
 * resources before passing the string.
 *
 * @param {string} serialized
 */
async function decodeSvgImage(serialized) {
  const image = new Image();
  const loaded = new Promise((resolve, reject) => {
    image.addEventListener("load", resolve, { once: true });
    image.addEventListener(
      "error",
      () => reject(new Error("Self-contained replay SVG could not be decoded.")),
      { once: true },
    );
  });
  image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(serialized)}`;
  await loaded;
  return image;
}

/**
 * Ask canvas to encode an image/png Blob and return a promise for it.
 * Reject missing or wrong-type output with Error; Canvas security/encoding
 * failures may also reject. This does not save the Blob or alter the canvas.
 *
 * @param {HTMLCanvasElement} canvas
 */
function canvasPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!(blob instanceof Blob) || blob.type !== PNG_MIME_TYPE) {
        reject(new Error("Canvas did not produce a PNG Blob."));
        return;
      }
      resolve(blob);
    }, PNG_MIME_TYPE);
  });
}

/**
 * Capture one settled installed Replay Viewer frame as a PNG artifact.
 * value has exactly battlefield, installedAuthority, isCurrent,
 * localInspectedPresentationKey, renderPolicy, showRanges, transportState
 * and visualFilters. Require a branded joined authority, SETTLED transport,
 * replay_static render policy and a painted #battlefield SVG in a non-busy
 * #battlefield-shell. isCurrent() must return true after document fonts are
 * ready. Inspection/filter options follow projectReplayPngProvenanceV1.
 *
 * Snapshot the live SVG/styles synchronously, then load bundled fonts, decode
 * the detached SVG and paint an opaque Canvas at twice each CSS dimension.
 * Return the same Blob/filename/provenance artifact as the builder. Invalid
 * inputs throw TypeError; changed authority, loading, decoding and Canvas
 * failures reject the promise. The live frame can advance after its snapshot;
 * this export still describes that captured frame. No live DOM mutation,
 * event binding, automatic download or filesystem write occurs.
 *
 * @param {unknown} value
 */
export async function captureReplayBattlefieldPngV1(value) {
  const options = snapshotRecord(value, "Replay battlefield capture options");
  exactKeys(
    options,
    [
      "battlefield",
      "installedAuthority",
      "isCurrent",
      "localInspectedPresentationKey",
      "renderPolicy",
      "showRanges",
      "transportState",
      "visualFilters",
    ],
    "Replay battlefield capture options",
  );
  if (!isJoinedTransportAndAuthorizedPresentationV1(options.installedAuthority)) {
    invalid("Replay capture requires one coherent branded installed authority.");
  }
  if (typeof options.isCurrent !== "function") {
    invalid("Replay capture requires an authority revalidation callback.");
  }
  if (
    options.transportState !== "SETTLED" ||
    options.renderPolicy !== "replay_static"
  ) {
    invalid("Replay capture requires exact SETTLED/replay_static state.");
  }
  const battlefield = options.battlefield;
  if (
    typeof SVGSVGElement === "undefined" ||
    !(battlefield instanceof SVGSVGElement) ||
    battlefield.id !== "battlefield" ||
    battlefield.dataset.renderPolicy !== "replay_static"
  ) {
    invalid("Replay capture requires the painted #battlefield SVG.");
  }
  await battlefield.ownerDocument.fonts.ready;
  if (options.isCurrent() !== true) {
    throw new Error("Replay export authority changed before snapshot capture.");
  }
  if (
    battlefield.id !== "battlefield" ||
    battlefield.dataset.renderPolicy !== "replay_static"
  ) {
    invalid("Replay battlefield changed before snapshot capture.");
  }
  const shell = battlefield.parentElement;
  if (
    shell?.id !== "battlefield-shell" ||
    shell.getAttribute("aria-busy") !== "false"
  ) {
    invalid("Replay capture requires a non-busy battlefield shell.");
  }
  const cssWidth = positiveSafeInteger(
    battlefield.clientWidth,
    "Replay battlefield CSS width",
  );
  const cssHeight = positiveSafeInteger(
    battlefield.clientHeight,
    "Replay battlefield CSS height",
  );
  const detachedBattlefield = detachBattlefieldSnapshot(
    battlefield,
    cssWidth,
    cssHeight,
  );
  const provenance = projectReplayPngProvenanceV1({
    presentation: options.installedAuthority.presentation,
    renderPolicy: options.renderPolicy,
    cssWidth,
    cssHeight,
    showRanges: options.showRanges,
    localInspectedPresentationKey: options.localInspectedPresentationKey,
    visualFilters: options.visualFilters,
  });
  const fonts = await loadBundledFontBytes();
  installBundledFonts(detachedBattlefield, fonts);
  assertSelfContainedSvgResources(detachedBattlefield);
  const serialized = new XMLSerializer().serializeToString(detachedBattlefield);
  const image = await decodeSvgImage(serialized);
  const canvas = battlefield.ownerDocument.createElement("canvas");
  canvas.width = cssWidth * 2;
  canvas.height = cssHeight * 2;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) {
    throw new Error("Replay export requires a 2D Canvas context.");
  }
  context.fillStyle = REPLAY_BATTLEFIELD_BACKGROUND_V1.baseColor;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  assertOpaqueReplayExportPixelsV1(
    context.getImageData(0, 0, canvas.width, canvas.height).data,
  );
  const rawBlob = /** @type {Blob} */ (await canvasPngBlob(canvas));
  return buildArtifactFromProvenance(await rawBlob.arrayBuffer(), provenance);
}
