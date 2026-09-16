/**
 * @file Format numbers for browser labels without changing recorded measurements.
 * Use formatDisplayNumber for ordinary values and formatCompactDisplayNumber
 * for small battlefield labels. These pure helpers return text; callers retain
 * the exact value for exports, tooltips and accessible labels.
 */
/**
 * Format a finite number with zero to two decimal places.
 *
 * value is the measurement to display. Non-numbers, NaN and infinities return
 * an em dash before precision options are checked. options defaults to {};
 * minimumFractionDigits defaults to 0 and maximumFractionDigits to 2. Both
 * must be integers with 0 <= minimum <= maximum <= 2, or a RangeError is thrown.
 * Trailing zeros are removed down to the minimum. Rounded negative zero is
 * shown as zero. The result is a string and the input is not changed.
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
 * Shorten a finite number with K, M, B, T or P when its magnitude reaches 1,000.
 *
 * value accepts any input; invalid or non-finite numbers return an em dash.
 * Smaller values use formatDisplayNumber. Larger values use a power-of-1,000
 * suffix and zero, one or two decimal places based on the scaled magnitude.
 * The suffix stops at P; very large values keep growing in that unit. Returns
 * text only. Callers must keep the exact number in the tooltip/export data.
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
