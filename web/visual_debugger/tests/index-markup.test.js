import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const indexUrl = new URL("../index.html", import.meta.url);

test("static debugger IDs are unique", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const ids = [...markup.matchAll(/\bid="([^"]+)"/gu)].map((match) => match[1]);

  assert.equal(new Set(ids).size, ids.length);
});

test("every native disclosure has one named phrasing-content summary", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const detailsCount = [...markup.matchAll(/<details\b/gu)].length;
  const summaries = [...markup.matchAll(/<summary\b[^>]*>([\s\S]*?)<\/summary>/gu)];

  assert.equal(summaries.length, detailsCount);
  for (const [, content] of summaries) {
    assert.doesNotMatch(
      content,
      /<(?:address|article|aside|div|dl|fieldset|footer|form|h[1-6]|header|main|nav|ol|p|section|table|ul)\b/iu,
    );
    assert.notEqual(content.replace(/<[^>]+>/gu, "").trim(), "");
  }
});

test("bootstrap copy does not imply a configurable graphics rate", async () => {
  const markup = await readFile(indexUrl, "utf8");

  assert.match(markup, /id="motion-status"[^>]*>\s*Graphics\s*</u);
  assert.doesNotMatch(markup, /Graphics\s+1\.00×/u);
});
