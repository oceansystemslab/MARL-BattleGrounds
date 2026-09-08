/** Human-readable presentation of host-computed metric rows. No metric formulas. */
const CLASSES = ["", "Mage", "Warrior", "Hunter", "Rogue", "Priest"];

/** @param {string} value */
export function metricLabel(value) {
  return value
    .replace(/^marlbg\./u, "")
    .replace(/\.v[0-9]+$/u, "")
    .replaceAll("_", " ")
    .split(".")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" · ");
}

/** @param {unknown} value */
function number(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-GB", { maximumFractionDigits: 4 })
    : "—";
}

/** @param {Record<string, any>} subject */
function subjectLabel(subject) {
  const team = subject.team_id === 1 ? "Team A" : "Team B";
  switch (subject.subject_type) {
    case "episode":
      return "Episode";
    case "team":
      return team;
    case "team_class":
      return `${team} · ${CLASSES[subject.class_id] ?? "Class"}`;
    case "agent":
      return `Agent ${subject.global_slot < 5 ? "A" : "B"}${(subject.global_slot % 5) + 1}`;
    case "agent_pair":
      return `Slots ${subject.primary_global_slot} → ${subject.secondary_global_slot}`;
    default:
      return "Unavailable";
  }
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
 */
export function renderMetricRows(container, selection, summary) {
  const ids = [
    ...new Set(
      summary.statistics.map((/** @type {Record<string, any>} */ row) =>
        String(row.metric_id),
      ),
    ),
  ].sort();
  const inventory = ids.join("|");
  if (selection.dataset.inventory !== inventory) {
    const selected = selection.value;
    selection.replaceChildren(
      new Option("Team overview", "overview"),
      ...ids.map((id) => new Option(metricLabel(id), id)),
    );
    selection.value = ids.includes(selected) ? selected : "overview";
    selection.dataset.inventory = inventory;
  }
  const rows = summary.statistics.filter((/** @type {Record<string, any>} */ row) =>
    selection.value === "overview"
      ? row.metric_id.startsWith("marlbg.task.") &&
        ["team", "episode"].includes(row.subject.subject_type)
      : row.metric_id === selection.value,
  );
  const table = document.createElement("table");
  table.className = "metric-table";
  const head = document.createElement("thead");
  const headings = document.createElement("tr");
  for (const label of ["Subject / measure", "Value", "Exposure"])
    headings.append(element("th", label));
  head.append(headings);
  table.append(head);
  const body = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    const label = document.createElement("td");
    label.append(element("strong", subjectLabel(row.subject)));
    const dimensions = row.dimensions
      .map(
        (/** @type {Record<string, string>} */ item) =>
          `${item.name.replaceAll("_", " ")}: ${item.value.replaceAll("_", " ")}`,
      )
      .join(" · ");
    label.append(
      element(
        "div",
        [metricLabel(row.component_name), dimensions].filter(Boolean).join(" · "),
      ),
    );
    if (selection.value === "overview")
      label.prepend(element("div", metricLabel(row.metric_id)));
    const value = document.createElement("td");
    const pending =
      summary.completion === null &&
      row.completion_scope === "complete_episode" &&
      row.result_status === "insufficient_data";
    const status = pending
      ? "Pending"
      : row.result_status === "defined"
        ? ""
        : metricLabel(row.result_status);
    value.append(element("div", status || number(row.display_value)));
    value.append(element("small", row.status_reason ?? row.units));
    tr.append(label, value, element("td", number(row.exposure)));
    body.append(tr);
  }
  table.append(body);
  container.replaceChildren(table);
}
