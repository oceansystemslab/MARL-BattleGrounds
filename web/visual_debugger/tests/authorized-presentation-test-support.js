/**
 * @file Build authorized presentation fixtures for browser unit tests from checked
 * Python records. Helpers preserve the same normalization and join boundaries used by
 * the client.
 */
import { createHash } from "node:crypto";

/** @param {unknown} value @returns {unknown} */
function sortedJsonValue(value) {
  if (Array.isArray(value)) return value.map(sortedJsonValue);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, sortedJsonValue(child)]),
  );
}

/** @param {Record<string, any>} presentation */
export function resealEmptyLocalOracleCorpseOverlay(presentation) {
  const overlay = presentation.local_oracle_corpse_overlay;
  if (!overlay) return;
  if (overlay.corpse_observations.length !== 0) {
    throw new TypeError("epoch-only test helper requires an empty corpse overlay");
  }
  overlay.source_authority_epoch = presentation.source.source_authority_epoch;
  const content = Object.fromEntries(
    Object.entries(overlay).filter(
      ([key]) => key !== "authorized_overlay_digest_sha256",
    ),
  );
  overlay.authorized_overlay_digest_sha256 = createHash("sha256")
    .update(JSON.stringify(sortedJsonValue(content)), "utf8")
    .digest("hex");
}
