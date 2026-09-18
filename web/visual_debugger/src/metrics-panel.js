/**
 * @file Display the metric catalog and values supplied by Python. This module
 * orders rows using catalog keys, builds navigation/search displays and
 * registers metric tooltips. It does not compute measurements, change missing
 * values to zero or decide which scientific measurements apply.
 */
import { registerTooltipOwner } from "./tooltip.js";

const TOOLTIP_UNITS = Object.freeze({
  health: "Health points (HP)",
  steps: "Ticks",
  agent_steps: "Ticks added across agents",
  pair_steps: "Pair measurements",
  fraction: "Share: 1 means 100%",
  indicator: "1 means yes; 0 means no",
  ratio: "One number divided by another",
  distance: "Map distance units",
  score: "Points",
});

/**
 * Turn string value into a title by replacing underscores with spaces and
 * capitalizing word initials. Return text only; exact CSV names stay unchanged
 * in records and exports. The caller supplies a string.
 *
 * @param {string} value
 */
export function metricLabel(value) {
  return value.replaceAll("_", " ").replace(/\b\w/gu, (letter) => letter.toUpperCase());
}

/**
 * Return row's display section index: 0 for episode/team, 2 for recipient,
 * 3 for source-to-recipient, 4 for ally pairs, and 1 otherwise. Scope/role
 * fields come from the catalog; no record is changed or revalidated.
 *
 * @param {Record<string, any>} row
 */
function metricSection(row) {
  if (row.scope === "team" || row.scope === "episode") return 0;
  if (row.scope === "team_recipient" || row.subject_role === "recipient") return 2;
  if (row.scope === "source_recipient") return 3;
  if (row.scope === "ally_pair") return 4;
  return 1;
}

/**
 * Apply row.topic_text[topic] to a shallow row copy when present. All topics
 * may override description/subtitle; only the five named Ultimate topics may
 * also override label, numerator/denominator, guidance and missing_when.
 * Without topic text, return the original row. Inputs and metric values are
 * unchanged; nested data remains shared.
 *
 * @param {Record<string, any>} row @param {string} topic
 */
function metricText(row, topic) {
  const text = row.topic_text?.[topic];
  if (!text) return row;
  const fields = ["description", "subtitle"];
  // Only the five named ability topics can change their row and tooltip words.
  if (
    [
      "ultimate_mage",
      "ultimate_warrior",
      "ultimate_hunter",
      "ultimate_rogue",
      "ultimate_priest",
    ].includes(topic)
  ) {
    fields.push("label", "numerator", "denominator", "guidance", "missing_when");
  }
  const result = { ...row };
  for (const field of fields) {
    if (text[field] !== undefined) result[field] = text[field];
  }
  return result;
}

/**
 * Select summary.statistics for the exact topic/view location. Omit rows
 * whose applicable field is false, keep missing values, and sort by Python's
 * topic_order tuple then row.order. Return a new array with topic wording
 * applied. The caller supplies a valid catalog summary; no independent
 * metric calculation, shape validation or mutation occurs.
 *
 * @param {Record<string, any>} summary @param {string} topic @param {string} view
 */
export function selectedMetricRows(summary, topic, view) {
  const rows = summary.statistics.filter(
    (/** @type {Record<string, any>} */ row) =>
      row.applicable !== false &&
      row.locations.some(
        (/** @type {{topic: string, view: string}} */ location) =>
          location.topic === topic && location.view === view,
      ),
  );
  // Python owns topic order, including related rows and their section order.
  return rows
    .sort(
      (
        /** @type {Record<string, any>} */ first,
        /** @type {Record<string, any>} */ second,
      ) => {
        const a = first.topic_order[topic];
        const b = second.topic_order[topic];
        for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
          const difference = (a[index] ?? 0) - (b[index] ?? 0);
          if (difference) return difference;
        }
        return first.order - second.order;
      },
    )
    .map((/** @type {Record<string, any>} */ row) => metricText(row, topic));
}

/**
 * Update topic selection and view selects from the supplied topics catalog.
 *
 * selection groups topics by section; view shows the selected topic's views.
 * Retain existing valid choices, otherwise select each list's first choice.
 * Cache JSON inventories in element datasets to avoid rebuilding equal lists.
 * Hide viewField for a single view. topics and each views list must be
 * nonempty/valid; an unresolved topic throws TypeError. Return undefined and
 * change only the DOM, not the catalog.
 *
 * @param {HTMLSelectElement} selection
 * @param {HTMLSelectElement} view
 * @param {HTMLElement} viewField
 * @param {Record<string, any>[]} topics
 */
export function renderMetricNavigation(selection, view, viewField, topics) {
  const inventory = JSON.stringify(topics);
  if (selection.dataset.inventory !== inventory) {
    const selected = selection.value;
    /** @type {Map<string, HTMLOptGroupElement>} */
    const sections = new Map();
    for (const topic of topics) {
      let section = sections.get(topic.section);
      if (!section) {
        section = document.createElement("optgroup");
        section.label = topic.section;
        sections.set(topic.section, section);
      }
      section.append(new Option(topic.label, topic.name));
    }
    selection.replaceChildren(...sections.values());
    selection.value = topics.some((topic) => topic.name === selected)
      ? selected
      : topics[0].name;
    selection.dataset.inventory = inventory;
  }
  const topic = topics.find((item) => item.name === selection.value);
  if (!topic) throw new TypeError("Unknown measurement topic.");
  const views = JSON.stringify(topic.views);
  if (view.dataset.inventory !== views) {
    const selected = view.value;
    view.replaceChildren(
      ...topic.views.map(
        (/** @type {{name: string, label: string}} */ item) =>
          new Option(item.label, item.name),
      ),
    );
    view.value = topic.views.some(
      (/** @type {{name: string}} */ item) => item.name === selected,
    )
      ? selected
      : topic.views[0].name;
    view.dataset.inventory = views;
  }
  viewField.hidden = topic.views.length === 1;
}

export {
  buildMetricSearchIndex,
  findMeasurements,
  searchMeasurements,
} from "./metric-search.js";

/**
 * Return row's primary topic label, adding its view label when needed.
 * topics must contain primary_topic and primary_view. An unknown topic
 * throws TypeError; a missing required view may also fail. No lookup fallback
 * or input mutation occurs.
 *
 * @param {Record<string, any>} row @param {Record<string, any>[]} topics
 */
function primaryLocationLabel(row, topics) {
  const topic = topics.find((item) => item.name === row.primary_topic);
  if (!topic) throw new TypeError("Unknown primary measurement topic.");
  const view = topic.views.find(
    (/** @type {{name: string}} */ item) => item.name === row.primary_view,
  );
  return topic.views.length === 1 ? topic.label : `${topic.label}: ${view.label}`;
}

/**
 * Replace container with the first limit matches as accessible result buttons.
 *
 * matches is already ranked; topics resolves each primary location. limit is
 * a caller-supplied nonnegative slice bound. Clicking a button calls select
 * with the original row. Show applicability notices without dropping results.
 * Return undefined; create DOM/listeners but do not compute or alter values.
 *
 * @param {HTMLElement} container
 * @param {Record<string, any>[]} matches
 * @param {Record<string, any>[]} topics
 * @param {number} limit
 * @param {(row: Record<string, any>) => void} select
 */
export function renderMetricSearchResults(container, matches, topics, limit, select) {
  const list = document.createElement("ol");
  list.className = "metric-search-results";
  list.setAttribute("role", "presentation");
  for (const row of matches.slice(0, limit)) {
    const item = document.createElement("li");
    item.setAttribute("role", "presentation");
    const button = document.createElement("button");
    button.type = "button";
    button.id = `metric-search-option-${row.order}`;
    button.tabIndex = -1;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", "false");
    button.dataset.measurement = row.name;
    button.append(
      element("strong", row.label),
      element("span", row.subject),
      element("code", row.name),
      element("small", primaryLocationLabel(row, topics)),
    );
    if (!row.applicable)
      button.append(element("small", "Not applicable to this roster"));
    button.addEventListener("click", () => select(row));
    item.append(button);
    list.append(item);
  }
  container.replaceChildren(list);
}

/**
 * Show and focus a metric definition in container. row uses primary-topic
 * wording; topics supplies its location label. Include guidance, units and
 * not-applicable explanation. Return undefined and replace DOM children.
 * The owning search flow decides when this definition view is appropriate;
 * invalid catalog references may throw.
 *
 * @param {HTMLElement} container @param {Record<string, any>} row @param {Record<string, any>[]} topics
 */
export function renderMetricDefinition(container, row, topics) {
  row = metricText(row, row.primary_topic);
  const title = element("strong", row.label);
  const fields = document.createElement("dl");
  for (const { label, value } of metricTooltipRows(row)) {
    fields.append(element("dt", label), element("dd", value));
  }
  container.replaceChildren(
    title,
    element("p", row.subject),
    element("p", row.description),
    element("p", row.not_applicable_reason ?? "Not applicable to this roster."),
    element("p", `Primary location: ${primaryLocationLabel(row, topics)}.`),
    fields,
  );
  container.hidden = false;
  container.focus();
}

/**
 * Build new label/value/metadata records for row's metric tooltip. Include
 * exact CSV name, units, directed From/To identities, a fraction's numerator
 * and denominator, guidance and missing-value rule. Team IDs and global
 * agent IDs come from the catalog subjects; no identities or values are
 * inferred from the current scene. Return a mutable array without editing row.
 *
 * @param {Record<string, any>} row
 */
export function metricTooltipRows(row) {
  const directed = row.scope === "source_recipient" || row.scope === "team_recipient";
  const rows = [
    ["CSV Column", row.name],
    [
      "Unit",
      TOOLTIP_UNITS[/** @type {keyof typeof TOOLTIP_UNITS} */ (row.unit)] ??
        metricLabel(row.unit),
    ],
  ];
  if (directed) {
    rows.push(
      [
        "From",
        row.scope === "team_recipient"
          ? `Team ${row.subjects[0] === 1 ? "A" : "B"}`
          : `Agent ID ${row.subjects[0]}`,
      ],
      ["To", `Agent ID ${row.subjects[1]}`],
    );
  }
  if (row.denominator) {
    rows.push(["Numerator", row.numerator], ["Denominator", row.denominator]);
  }
  rows.push(["How to Read It", row.guidance], ["Blank When", row.missing_when]);
  return rows.map(([label, value]) => ({
    label,
    value,
    metadata: { compact: true, full: true },
  }));
}

/**
 * Create a detached page-document element with tag and literal textContent.
 * Return the node; no HTML parsing or attachment occurs. Callers supply the
 * trusted local tag and display text.
 *
 * @param {string} tag @param {string} text
 */
function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

/**
 * Replace container with the selected metric table and update description.
 *
 * topicName/view select rows from summary; the catalog controls row order and
 * section placement. Format valid values in en-GB with up to four decimals
 * for fractions/ratios, two otherwise. Preserve exact values in data attributes.
 * Invalid measurements show an em dash and missing-value help, never zero.
 * Register tooltips on new measure labels. Return undefined; mutate only DOM.
 * The caller supplies a valid summary/topic and owns fetching/exporting data.
 *
 * @param {HTMLElement} container
 * @param {string} topicName
 * @param {string} view
 * @param {Record<string, any>} summary
 * @param {HTMLElement} description
 */
export function renderMetricRows(container, topicName, view, summary, description) {
  const rows = selectedMetricRows(summary, topicName, view);
  const available = rows.filter(
    (/** @type {Record<string, any>} */ row) => row.valid,
  ).length;
  const topic = summary.topics.find(
    (/** @type {Record<string, string>} */ item) => item.name === topicName,
  );
  description.textContent = `${topic.description} ${available} of ${rows.length} measurements available.`;
  const table = document.createElement("table");
  table.className = "metric-table";
  const head = document.createElement("thead");
  const headings = document.createElement("tr");
  for (const label of ["Subject", "Measure", "Value"]) {
    const heading = element("th", label);
    heading.setAttribute("scope", "col");
    headings.append(heading);
  }
  head.append(headings);
  table.append(head);
  const body = document.createElement("tbody");
  body.tabIndex = 0;
  body.setAttribute("aria-label", "Scrollable metric values");
  let previousSection = -1;
  for (const row of rows) {
    const section = metricSection(row);
    if (section !== previousSection) {
      const headingRow = document.createElement("tr");
      headingRow.className = "metric-section";
      const heading = element(
        "th",
        [
          "Episode and Team Totals",
          "Agent Details",
          "Recipient Totals",
          "Source-to-Recipient Details",
          "Ally Pair Details",
        ][section],
      );
      heading.setAttribute("colspan", "3");
      headingRow.append(heading);
      body.append(headingRow);
      previousSection = section;
    }
    const tr = document.createElement("tr");
    tr.dataset.metric = row.name;
    const subject = element("td", row.subject);
    const measure = element("td", "");
    const label = element("span", row.label);
    label.tabIndex = 0;
    label.className = "metric-measure";
    const subtitle = row.subtitle ?? row.status;
    registerTooltipOwner(label, {
      kind: "metric",
      tone: "neutral",
      accent: "none",
      id: `metric:${row.name}`,
      title: row.label,
      summary: subtitle ? `${subtitle}. ${row.description}` : row.description,
      rows: metricTooltipRows(row),
      sections: [],
      metadata: { compact: true, full: true },
      anchor: "element",
    });
    measure.append(label);
    if (subtitle) {
      const status = element("div", subtitle);
      status.className = "metric-condition";
      measure.append(status);
    }
    measure.append(
      element("small", row.unit === "health" ? "HP" : metricLabel(row.unit)),
    );
    const value = element(
      "td",
      row.valid
        ? row.value.toLocaleString("en-GB", {
            maximumFractionDigits:
              row.unit === "fraction" || row.unit === "ratio" ? 4 : 2,
          })
        : "—",
    );
    value.className = "metric-value";
    value.dataset.value = row.valid ? String(row.value) : "";
    if (!row.valid)
      value.setAttribute(
        "aria-label",
        `Unavailable. Possible reasons: ${row.missing_when}`,
      );
    tr.append(subject, measure, value);
    body.append(tr);
  }
  if (rows.length === 0) {
    const empty = document.createElement("tr");
    const message = element("td", "No applicable measurements for this roster.");
    message.setAttribute("colspan", "3");
    empty.append(message);
    body.append(empty);
  }
  table.append(body);
  container.replaceChildren(table);
}
