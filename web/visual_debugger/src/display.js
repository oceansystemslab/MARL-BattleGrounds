/**
 * Format one finite human-facing number without changing the authoritative
 * value retained in frame records or data attributes.
 *
 * @param {unknown} value
 * @param {{minimumFractionDigits?: number, maximumFractionDigits?: number}} [options]
 * @returns {string}
 */
export function formatDisplayNumber(value, options = {}) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  const minimumFractionDigits = options.minimumFractionDigits ?? 0;
  const maximumFractionDigits = options.maximumFractionDigits ?? 2;
  if (
    !Number.isInteger(minimumFractionDigits) ||
    !Number.isInteger(maximumFractionDigits) ||
    minimumFractionDigits < 0 ||
    maximumFractionDigits > 2 ||
    minimumFractionDigits > maximumFractionDigits
  ) {
    throw new RangeError(
      "Display precision must use zero to two fractional digits with minimum not exceeding maximum.",
    );
  }

  const normalized = Object.is(value, -0) ? 0 : value;
  const fixed = normalized.toFixed(maximumFractionDigits);
  const unsignedFixed =
    Number(fixed) === 0 ? (0).toFixed(maximumFractionDigits) : fixed;
  const [integer, fraction = ""] = unsignedFixed.split(".");
  let retainedFraction = fraction;
  while (
    retainedFraction.length > minimumFractionDigits &&
    retainedFraction.endsWith("0")
  ) {
    retainedFraction = retainedFraction.slice(0, -1);
  }
  return retainedFraction ? `${integer}.${retainedFraction}` : integer;
}

/**
 * Produce a short, honest visual label for a value that cannot fit its compact
 * battlefield cell. The exact value remains in the owning cue's accessible
 * label, tooltip, and data attributes; this label is presentation-only.
 *
 * @param {unknown} value
 * @returns {string}
 */
export function formatCompactDisplayNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  const normalized = Object.is(value, -0) ? 0 : value;
  const magnitude = Math.abs(normalized);
  if (magnitude < 1_000) {
    return formatDisplayNumber(normalized);
  }

  const suffixes = ["K", "M", "B", "T", "P"];
  const exponent = Math.min(Math.floor(Math.log10(magnitude) / 3), suffixes.length);
  const scale = 1_000 ** exponent;
  const scaled = normalized / scale;
  const precision = Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2;
  const suffix = suffixes[exponent - 1] ?? "P+";
  return `${formatDisplayNumber(scaled, {
    maximumFractionDigits: precision,
  })}${suffix}`;
}
