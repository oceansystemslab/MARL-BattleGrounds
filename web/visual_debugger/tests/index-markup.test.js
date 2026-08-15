import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const indexUrl = new URL("../index.html", import.meta.url);
const productionSourceUrls = [
  new URL("../src/controls.js", import.meta.url),
  new URL("../src/main.js", import.meta.url),
  new URL("../src/scene.js", import.meta.url),
  new URL("../src/explanations.js", import.meta.url),
  new URL("../src/panels.js", import.meta.url),
];

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

test("product shell loads strict identity before the module and omits retired controls", async () => {
  const markup = await readFile(indexUrl, "utf8");

  const bootstrapIndex = markup.indexOf('<script src="/bootstrap.js"></script>');
  const moduleIndex = markup.indexOf(
    '<script type="module" src="/src/main.js"></script>',
  );
  assert.notEqual(bootstrapIndex, -1);
  assert.ok(bootstrapIndex < moduleIndex);
  assert.match(markup, /<title>MARL-BattleGrounds<\/title>/u);
  assert.match(markup, /<h1 id="app-title">MARL-BattleGrounds<\/h1>/u);
  assert.match(markup, /<option value="researcher">Oracle View<\/option>/u);

  for (const retiredId of [
    "scenario-select",
    "preset-select",
    "motion-pause-button",
    "motion-off-button",
    "motion-skip-button",
    "motion-status",
    "advance-script-button",
  ]) {
    assert.doesNotMatch(markup, new RegExp(`id="${retiredId}"`, "u"));
  }
  assert.doesNotMatch(markup, /class="replay-shortcuts"/u);
  assert.doesNotMatch(markup, /command-deck__shortcut/u);
  assert.doesNotMatch(markup, /<dt>(?:N|P|\[ \/ \])<\/dt>/u);
  assert.doesNotMatch(markup, /Previous scenario|Next scenario|Motion Off/u);
});

test("browser production paths omit retired navigation and privileged display copy", async () => {
  const [controls, main, scene, explanations, panels] = await Promise.all(
    productionSourceUrls.map((url) => readFile(url, "utf8")),
  );

  assert.doesNotMatch(main, /liveDebuggerFrameIsScripted|REMOVED_LIVE_KEYS/u);
  assert.doesNotMatch(main, /onPresentationKey|Manual submit unavailable/u);
  assert.match(
    main,
    /publicAudienceLabel\(authorizedPresentationAudience\(presentation\)\)/u,
  );
  assert.match(main, /return "Oracle View"/u);
  assert.match(main, /return "Agent POV"/u);
  assert.doesNotMatch(main, /SharedObs source material/u);
  assert.doesNotMatch(scene, /scene\.audience_badge/u);
  assert.match(scene, /"Oracle View" : "Agent POV"/u);

  for (const source of [controls, main, scene, explanations, panels]) {
    assert.doesNotMatch(
      source,
      /Privileged researcher|Press N|PLAYBACK \/ INSPECTION ONLY/u,
    );
  }
});
