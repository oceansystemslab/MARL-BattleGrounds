import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const indexUrl = new URL("../index.html", import.meta.url);
const stylesUrl = new URL("../styles.css", import.meta.url);
const productionSourceUrls = [
  new URL("../src/controls.js", import.meta.url),
  new URL("../src/main.js", import.meta.url),
  new URL("../src/scene.js", import.meta.url),
  new URL("../src/explanations.js", import.meta.url),
  new URL("../src/panels.js", import.meta.url),
];

/**
 * @param {string} markup
 * @param {string} id
 * @param {string} tagName
 */
function elementBody(markup, id, tagName) {
  const match = markup.match(
    new RegExp(`<${tagName}\\b[^>]*\\bid="${id}"[^>]*>([\\s\\S]*?)</${tagName}>`, "u"),
  );
  assert.ok(match, `#${id} must be a <${tagName}> element.`);
  return match[1];
}

/**
 * Return a direct stable disclosure body and prove it follows the native
 * summary without interposing another container.
 *
 * @param {string} markup
 * @param {string} panelId
 */
function disclosureBody(markup, panelId) {
  const details = elementBody(markup, panelId, "details");
  const bodyId = `${panelId}-body`;
  const directWrappers = [
    ...details.matchAll(
      new RegExp(
        `</summary>\\s*<div\\b(?=[^>]*\\bid="${bodyId}")(?=[^>]*\\bdata-disclosure-body="${panelId}")[^>]*>`,
        "gu",
      ),
    ),
  ];
  assert.match(details, /^\s*<summary\b/u, `#${panelId} summary must remain direct.`);
  assert.equal(
    directWrappers.length,
    1,
    `#${panelId} must have exactly one direct stable disclosure body.`,
  );

  const wrapper = directWrappers[0];
  const wrapperStart = wrapper.index + wrapper[0].lastIndexOf("<div");
  const divTag = /<\/?div\b[^>]*>/giu;
  divTag.lastIndex = wrapperStart;
  const opening = divTag.exec(details);
  assert.ok(opening, `#${bodyId} must be a div.`);

  let depth = 1;
  for (let tag = divTag.exec(details); tag; tag = divTag.exec(details)) {
    depth += tag[0].startsWith("</") ? -1 : 1;
    if (depth === 0) {
      return details.slice(opening.index + opening[0].length, tag.index);
    }
  }
  assert.fail(`#${bodyId} must close.`);
}

/** @param {string} markup */
function textContent(markup) {
  return markup
    .replace(/<[^>]+>/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
}

test("static debugger IDs are unique", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const ids = [...markup.matchAll(/\bid="([^"]+)"/gu)].map((match) => match[1]);

  assert.equal(new Set(ids).size, ids.length);
});

test("only Team B exposes the SharedObs scenario controller", async () => {
  const markup = await readFile(indexUrl, "utf8");
  for (const id of ["devclient-team-a-controller", "devclient-team-b-controller"]) {
    const select = elementBody(markup, id, "select");
    assert.deepEqual(
      [...select.matchAll(/<option value="([^"]+)"[^>]*>([^<]+)<\/option>/gu)].map(
        ([, value, label]) => [value, label],
      ),
      [
        ["manual", "Manual"],
        ["scripted_tdm", "Scripted TDM"],
        ["random_valid", "Random"],
        ...(id === "devclient-team-b-controller"
          ? [["scenario_1", "Reactive MRP Controller"]]
          : []),
      ],
    );
  }
  assert.match(markup, /Scenario controllers require SharedObs\./u);
});

test("replay Help names the exact arrow keys and Escape selection behavior", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const help = elementBody(markup, "help-dialog", "dialog");
  assert.match(help, /<dt>LEFT ARROW \/ RIGHT ARROW<\/dt>/u);
  assert.match(help, /<dd>Go to the previous \/ next captured tick<\/dd>/u);
  assert.match(help, /<dt>Escape<\/dt><dd>Clear the selected agent<\/dd>/u);
  assert.doesNotMatch(help, /<dt>Left \/ Right<\/dt>/u);
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

test("every CP3 disclosure keeps its scientific content in one stable direct body", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const panels = {
    "command-deck": [
      "command-controlled-actor",
      "stay-button",
      "command-target-select",
      "no-combat-button",
      "basic-button",
      "ultimate-button",
      "command-commit-title",
      "command-commit-summary",
      "submit-turn-button",
      "reset-button",
    ],
    "roster-details": ["roster"],
    "agent-details": ["selection-card"],
    "pending-turn-details": ["pending-scope", "pending-card"],
    "latest-transition-details": ["accepted-card", "accepted-announcement"],
    "visual-key": ["live-visual-key", "replay-visual-key"],
    "technical-frame-details": ["diagnostics-card"],
  };

  for (const [panelId, contentIds] of Object.entries(panels)) {
    const body = disclosureBody(markup, panelId);
    for (const contentId of contentIds) {
      assert.match(
        body,
        new RegExp(`\\bid="${contentId}"`, "u"),
        `#${contentId} must stay inside #${panelId}-body.`,
      );
      assert.equal(
        [...markup.matchAll(new RegExp(`\\bid="${contentId}"`, "gu"))].length,
        1,
        `#${contentId} must remain unique.`,
      );
    }
  }
});

test("shared shell uses the requested details and visual-filter controls without an event feed", async () => {
  const markup = await readFile(indexUrl, "utf8");

  assert.match(markup, />Comprehensive Agent Class Details</u);
  assert.match(markup, /id="accepted-heading">Latest Transition</u);
  assert.match(markup, /id="command-commit-title">Submit the staged joint turn</u);
  assert.match(markup, /id="submit-turn-button"[^>]*>\s*Submit joint turn\s*</u);
  assert.match(
    markup,
    /<button type="button" data-key="Tab">Next actor<\/button>[\s\S]*<button type="button" data-key="Tab" data-shift="true">/u,
  );
  assert.match(markup, /<summary>Visual Key<\/summary>/u);
  assert.match(markup, /id="visual-filter-count"[^>]*>19 enabled</u);
  assert.match(
    markup,
    /id="enable-all-visual-filters-button"[^>]*disabled[^>]*>\s*Enable All\s*<\/button>/u,
  );
  assert.match(
    markup,
    /id="disable-all-visual-filters-button"[^>]*>\s*Disable All\s*<\/button>/u,
  );
  assert.doesNotMatch(markup, /Restore All|Latest events/u);
  assert.doesNotMatch(markup, /<dt>Right click<\/dt>/u);
  for (const removedId of ["events-details", "event-count", "event-feed"]) {
    assert.doesNotMatch(markup, new RegExp(`id="${removedId}"`, "u"));
  }
});

test("current, upcoming, and latest-transition surfaces remain available in both products", async () => {
  const markup = await readFile(indexUrl, "utf8");
  const pendingOpeningTag = markup.match(
    /<details\b[^>]*\bid="pending-turn-details"[^>]*>/u,
  );
  assert.ok(pendingOpeningTag);
  assert.doesNotMatch(pendingOpeningTag[0], /\bdata-live-only\b/u);
  const latestTransitionOpeningTag = markup.match(
    /<details\b[^>]*\bid="latest-transition-details"[^>]*>/u,
  );
  assert.ok(latestTransitionOpeningTag);
  assert.doesNotMatch(latestTransitionOpeningTag[0], /\bdata-live-only\b/u);
  assert.doesNotMatch(latestTransitionOpeningTag[0], /\bdata-replay-only\b/u);
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
    "revision-value",
    "replay-incoming-value",
  ]) {
    assert.doesNotMatch(markup, new RegExp(`id="${retiredId}"`, "u"));
  }
  assert.doesNotMatch(markup, /class="replay-shortcuts"/u);
  assert.doesNotMatch(markup, /command-deck__shortcut/u);
  assert.doesNotMatch(markup, /<dt>(?:N|P|\[ \/ \])<\/dt>/u);
  assert.doesNotMatch(markup, /Previous scenario|Next scenario|Motion Off/u);
  for (const id of ["replay-completion-badge", "replay-processing-badge"]) {
    assert.match(
      markup,
      new RegExp(`<dd\\b(?=[^>]*\\bid="${id}")(?=[^>]*\\btabindex="0")[^>]*>`, "u"),
    );
  }
  assert.doesNotMatch(markup, /<dt>Incoming<\/dt>|>Revision\s*</u);
});

test("replay transport exposes the exact CP8 controls, rates, and truthful help", async () => {
  const [markup, styles] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(stylesUrl, "utf8"),
  ]);
  assert.match(
    markup,
    /id="replay-first-button"[^>]*aria-label="Start replay tick"[^>]*>Start<\/button>/u,
  );
  assert.match(
    markup,
    /id="replay-last-button"[^>]*aria-label="End replay tick"[^>]*>End<\/button>/u,
  );
  const rateOptions = elementBody(markup, "replay-playback-rate", "select");
  assert.deepEqual(
    [
      ...rateOptions.matchAll(
        /<option value="([^"]+)"(?: selected)?>([^<]+)<\/option>/gu,
      ),
    ].map(([, value, label]) => [value, label]),
    [
      ["0.25", "0.25×"],
      ["0.5", "0.50×"],
      ["0.75", "0.75×"],
      ["1", "1.00×"],
      ["1.25", "1.25×"],
      ["1.5", "1.50×"],
      ["1.75", "1.75×"],
      ["2", "2.00×"],
    ],
  );
  assert.match(markup, /id="replay-transport-status"[^>]*aria-live="polite"/u);
  assert.match(
    markup,
    /Frame slider<\/dt><dd>Preview without a request; commit one exact seek<\/dd>/u,
  );
  assert.match(
    markup,
    /Playback speed<\/dt><dd>Scale the complete replay presentation clock<\/dd>/u,
  );
  assert.doesNotMatch(markup, /Home \/ End|Shift\+Left|Shift\+Right|short debounce/u);
  assert.match(
    styles,
    /@media \(max-width: 70rem\)[\s\S]*\.replay-timeline__transport\s*\{\s*grid-template-columns: repeat\(7, minmax\(0, 1fr\)\);/u,
  );
});

test("replay artifact actions are accessible, fail closed, and CSP compatible", async () => {
  const [markup, styles] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(stylesUrl, "utf8"),
  ]);
  const actions = elementBody(markup, "replay-artifact-actions", "fieldset");
  const opening = markup.match(
    /<fieldset\b(?=[^>]*\bid="replay-artifact-actions")(?=[^>]*\bdata-replay-only)(?=[^>]*\bhidden)[^>]*>/u,
  );
  assert.ok(opening);
  for (const [id, label, helpId] of [
    ["replay-export-png-button", "Export PNG", "replay-export-png-help"],
    [
      "replay-download-metrics-button",
      "Download Metrics",
      "replay-download-metrics-help",
    ],
  ]) {
    assert.match(
      actions,
      new RegExp(
        `<button\\b(?=[^>]*\\bid="${id}")(?=[^>]*\\bdisabled)(?=[^>]*\\baria-describedby="${helpId}")[^>]*>${label}</button>`,
        "u",
      ),
    );
    assert.equal([...markup.matchAll(new RegExp(`\\bid="${id}"`, "gu"))].length, 1);
    assert.match(actions, new RegExp(`\\bid="${helpId}"`, "u"));
  }
  assert.match(
    markup,
    /Content-Security-Policy[\s\S]*img-src 'self' data:;[\s\S]*font-src 'self'/u,
  );
  assert.doesNotMatch(markup, /img-src[^;]*blob:/u);
  assert.match(
    styles,
    /\.replay-artifact-actions\s*\{[\s\S]*justify-content: flex-end;[\s\S]*border: 0;/u,
  );
  assert.match(
    styles,
    /@media \(max-width: 42rem\)[\s\S]*\.replay-artifact-actions\s*\{\s*justify-content: flex-start;/u,
  );
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
  assert.doesNotMatch(main, /fixed recipient|passive inspection|controlled_actor/iu);
  assert.doesNotMatch(panels, /fixed recipient|controlled_actor/iu);
  assert.doesNotMatch(scene, /scene\.audience_badge/u);
  assert.match(scene, /"Oracle View" : "Agent POV"/u);

  for (const source of [controls, main, scene, explanations, panels]) {
    assert.doesNotMatch(
      source,
      /Privileged researcher|Press N|PLAYBACK \/ INSPECTION ONLY/u,
    );
  }
});

test("battlefield support owns visible instructions, product controls, and minimal keys", async () => {
  const [markup, styles] = await Promise.all([
    readFile(indexUrl, "utf8"),
    readFile(stylesUrl, "utf8"),
  ]);
  const battlefieldShellStart = markup.indexOf('id="battlefield-shell"');
  const supportStart = markup.indexOf('<div class="battlefield-support">');
  const commandDeckStart = markup.indexOf('<details id="command-deck"');
  assert.notEqual(battlefieldShellStart, -1);
  assert.ok(battlefieldShellStart < supportStart);
  assert.ok(supportStart < commandDeckStart);

  assert.match(
    markup,
    /id="battlefield"[\s\S]*?aria-describedby="battlefield-instructions"/u,
  );
  assert.match(
    markup,
    /id="battlefield"[\s\S]*?role="img"[\s\S]*?tabindex="-1"[\s\S]*?aria-label="Battlefield unavailable until product identity is validated\."/u,
  );
  assert.doesNotMatch(
    markup,
    /id="battlefield"[\s\S]*?aria-label="Interactive battlefield/u,
  );
  assert.match(
    markup,
    /id="battlefield-instructions" class="battlefield-instructions"/u,
  );
  assert.match(markup, /Product-specific Battlefield instructions are loading\./u);
  assert.doesNotMatch(markup, /id="battlefield-instructions"[^>]*class="[^"]*sr-only/u);

  const utilities = elementBody(markup, "battlefield-utilities", "fieldset");
  const visualFilters = elementBody(markup, "visual-filters", "details");
  for (const id of ["live-ranges-button", "replay-ranges-button"]) {
    assert.doesNotMatch(utilities, new RegExp(`id="${id}"`, "u"));
    assert.match(visualFilters, new RegExp(`id="${id}"`, "u"));
    assert.equal([...markup.matchAll(new RegExp(`id="${id}"`, "gu"))].length, 1);
  }
  assert.match(utilities, /id="replay-clear-reference-button"/u);
  assert.equal([...markup.matchAll(/id="replay-clear-reference-button"/gu)].length, 1);
  assert.match(utilities, />Clear Selection<\/button>/u);
  assert.doesNotMatch(markup, /Clear Reference/u);

  /** @param {string} id */
  const labels = (id) =>
    [...elementBody(markup, id, "dl").matchAll(/<dt>([\s\S]*?)<\/dt>/gu)].map((match) =>
      textContent(match[1]),
    );
  assert.deepEqual(labels("live-visual-key"), [
    "Team A",
    "Team B",
    "Controlled",
    "Selected target",
  ]);
  assert.deepEqual(labels("replay-visual-key"), ["Team A", "Team B", "Selected agent"]);

  assert.match(
    styles,
    /html:not\(\[data-product-kind\]\) \[data-live-only\],[\s\S]*html:not\(\[data-product-kind\]\) \[data-replay-only\],[\s\S]*html:not\(\[data-product-kind\]\) #battlefield-instructions,[\s\S]*html\[data-product-kind="combat_debugger"\] \[data-replay-only\],[\s\S]*html\[data-product-kind="replay_viewer"\] \[data-live-only\]/u,
  );
});
