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
  const [integer, fraction = ""] = normalized.toFixed(maximumFractionDigits).split(".");
  let retainedFraction = fraction;
  while (
    retainedFraction.length > minimumFractionDigits &&
    retainedFraction.endsWith("0")
  ) {
    retainedFraction = retainedFraction.slice(0, -1);
  }
  return retainedFraction ? `${integer}.${retainedFraction}` : integer;
}
