/** Catalog-driven presentation. Numerical values and applicability belong to Python. */
import { registerTooltipOwner } from "./tooltip.js";

/** @param {string} value */
export function metricLabel(value) {
  return value.replaceAll("_", " ").replace(/\b\w/gu, (letter) => letter.toUpperCase());
}

/** @param {Record<string, any>} summary @param {string} selection */
export function selectedMetricRows(summary, selection) {
  const rows = summary.statistics.filter((/** @type {Record<string, any>} */ row) =>
    selection === "overview"
      ? row.family === "priority" && ["team", "episode"].includes(row.scope)
      : row.family === selection,
  );
  // Keep teams together before agent detail, with numerical slot ordering.
  const rank = (/** @type {Record<string, any>} */ row) =>
    row.scope === "team" ? 0 : row.scope === "episode" ? 1 : 2;
  return rows.sort(
    (
      /** @type {Record<string, any>} */ first,
      /** @type {Record<string, any>} */ second,
    ) =>
      rank(first) - rank(second) ||
      (first.subjects[0] ?? 0) - (second.subjects[0] ?? 0) ||
      (first.subjects[1] ?? 0) - (second.subjects[1] ?? 0),
  );
}

/** @param {string} tag @param {string} text */
function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

/**
 * @param {HTMLElement} container
 * @param {HTMLSelectElement} selection
 * @param {Record<string, any>} summary
 * @param {HTMLElement} description
 */
export function renderMetricRows(container, selection, summary, description) {
  const families = summary.families;
  const inventory = families
    .map((/** @type {Record<string, string>} */ family) => family.name)
    .join("|");
  if (selection.dataset.inventory !== inventory) {
    const selected = selection.value;
    selection.replaceChildren(
      ...families.map(
        (/** @type {Record<string, string>} */ family) =>
          new Option(family.label, family.name),
      ),
    );
    selection.value = families.some(
      (/** @type {Record<string, string>} */ family) => family.name === selected,
    )
      ? selected
      : "overview";
    selection.dataset.inventory = inventory;
  }
  const rows = selectedMetricRows(summary, selection.value);
  const available = rows.filter(
    (/** @type {Record<string, any>} */ row) => row.valid,
  ).length;
  const family = families.find(
    (/** @type {Record<string, string>} */ item) => item.name === selection.value,
  );
  description.textContent = `${family.description} ${available} of ${rows.length} measurements available.`;
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
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.dataset.metric = row.name;
    const subject = element("td", row.subject);
    const measure = element("td", "");
    const label = element("span", row.label);
    label.tabIndex = 0;
    label.className = "metric-measure";
    const direction =
      row.direction === "higher"
        ? "Higher Is Better"
        : row.direction === "lower"
          ? "Lower Is Better"
          : "No Preferred Direction";
    registerTooltipOwner(label, {
      kind: "metric",
      tone: "neutral",
      accent: "none",
      id: `metric:${row.name}`,
      title: row.label,
      summary: row.status ? `${row.status}. ${row.description}` : row.description,
      rows: [
        {
          label: "Unit",
          value: metricLabel(row.unit),
          metadata: { compact: true, full: true },
        },
        {
          label: "Interpretation",
          value: direction,
          metadata: { compact: true, full: true },
        },
        {
          label: "Unavailable When",
          value: row.missing_when,
          metadata: { compact: true, full: true },
        },
      ],
      sections: [],
      metadata: { compact: true, full: true },
      anchor: "element",
    });
    measure.append(label);
    if (row.status) {
      const status = element("div", row.status);
      status.className = "metric-condition";
      measure.append(status);
    }
    measure.append(
      element("small", row.unit === "health" ? "HP" : metricLabel(row.unit)),
    );
    const value = element(
      "td",
      row.valid ? row.value.toLocaleString("en-GB", { maximumFractionDigits: 4 }) : "—",
    );
    value.className = "metric-value";
    value.dataset.value = row.valid ? String(row.value) : "";
    if (!row.valid)
      value.setAttribute("aria-label", `Unavailable. ${row.missing_when}`);
    tr.append(subject, measure, value);
    body.append(tr);
  }
  table.append(body);
  container.replaceChildren(table);
}
