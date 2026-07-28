import assert from "node:assert/strict";
import test from "node:test";

import { formatCompactDisplayNumber, formatDisplayNumber } from "../src/display.js";

test("human-facing numbers never exceed two decimal places", () => {
  assert.equal(formatDisplayNumber(12.3456), "12.35");
  assert.equal(formatDisplayNumber(12.3), "12.3");
  assert.equal(formatDisplayNumber(12), "12");
  assert.equal(formatDisplayNumber(-0), "0");
  assert.equal(formatDisplayNumber(-0.004), "0");
  assert.equal(formatDisplayNumber(Number.NaN), "—");
});

test("movement scale can request exactly two retained decimal places", () => {
  const fixedTwo = {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  };
  assert.equal(formatDisplayNumber(0.1, fixedTwo), "0.10");
  assert.equal(formatDisplayNumber(1, fixedTwo), "1.00");
  assert.equal(formatDisplayNumber(0.01, fixedTwo), "0.01");
  assert.equal(formatDisplayNumber(-0.004, fixedTwo), "0.00");
});

test("display precision cannot exceed the two-decimal product policy", () => {
  assert.throws(
    () => formatDisplayNumber(1.234, { maximumFractionDigits: 3 }),
    /zero to two fractional digits/,
  );
  assert.throws(
    () =>
      formatDisplayNumber(1.2, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 1,
      }),
    /minimum not exceeding maximum/,
  );
});

test("extreme compact labels remain readable while exact data stays external", () => {
  assert.equal(formatCompactDisplayNumber(123456789), "123M");
  assert.equal(formatCompactDisplayNumber(123456.789), "123K");
  assert.equal(formatCompactDisplayNumber(12_345), "12.3K");
  assert.equal(formatCompactDisplayNumber(-1_234), "-1.23K");
  assert.equal(formatCompactDisplayNumber(999), "999");
  assert.equal(formatCompactDisplayNumber(Number.POSITIVE_INFINITY), "—");
});
