/** Show the catalog's words and numbers. Python decides which rows apply. */
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

/** @param {string} value */
export function metricLabel(value) {
  return value.replaceAll("_", " ").replace(/\b\w/gu, (letter) => letter.toUpperCase());
}

/** @param {Record<string, any>} row */
function metricSection(row) {
  if (row.scope === "team" || row.scope === "episode") return 0;
  if (row.scope === "team_recipient" || row.subject_role === "recipient") return 2;
  if (row.scope === "source_recipient") return 3;
  if (row.scope === "ally_pair") return 4;
  return 1;
}

/** @param {Record<string, any>} row @param {string} topic */
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

/** @param {Record<string, any>} summary @param {string} topic @param {string} view */
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

/** @param {Record<string, any>} row @param {Record<string, any>[]} topics */
function primaryLocationLabel(row, topics) {
  const topic = topics.find((item) => item.name === row.primary_topic);
  if (!topic) throw new TypeError("Unknown primary measurement topic.");
  const view = topic.views.find(
    (/** @type {{name: string}} */ item) => item.name === row.primary_view,
  );
  return topic.views.length === 1 ? topic.label : `${topic.label}: ${view.label}`;
}

/**
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

/** @param {HTMLElement} container @param {Record<string, any>} row @param {Record<string, any>[]} topics */
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

/** @param {Record<string, any>} row */
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

/** @param {string} tag @param {string} text */
function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

/**
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
