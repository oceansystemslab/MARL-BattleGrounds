import { isAuthorizedPresentationFrame } from "./authorized-presentation-adapter.js";

/**
 * Human-facing match facts retain the installed source epoch. No score, winner,
 * policy identity, or task mode is inferred from battlefield paint.
 *
 * @param {unknown} presentation
 */
export function matchSummaryView(presentation) {
  const match = isAuthorizedPresentationFrame(presentation)
    ? presentation.match_summary
    : null;
  if (!match) return null;
  const tdm = match.task_mode === 1;
  return Object.freeze({
    task: tdm ? "TDM" : "Combat diagnostic",
    taskMode: match.task_mode,
    teams: match.teams.map(
      (/** @type {Record<string, any>} */ team, /** @type {number} */ index) =>
        Object.freeze({
          teamId: team.team_id,
          label: `Team ${index === 0 ? "A (Blue)" : "B (Red)"}`,
          name: team.display_name,
          score: tdm ? `${match.scores[index]}/${match.score_threshold}` : null,
          result:
            match.outcome === "draw"
              ? "DRAW"
              : match.outcome === "team_a_win"
                ? index === 0
                  ? "VICTORY"
                  : "DEFEAT"
                : match.outcome === "team_b_win"
                  ? index === 1
                    ? "VICTORY"
                    : "DEFEAT"
                  : null,
          provenance: [...team.policy_ids, ...team.checkpoint_digests].join("\n"),
        }),
    ),
  });
}

/**
 * @param {{root: HTMLElement, task: HTMLElement, teams: HTMLElement[], taskSelect: HTMLSelectElement}} elements
 * @param {unknown} presentation
 */
export function renderMatchSummary(elements, presentation) {
  const summary = matchSummaryView(presentation);
  elements.root.hidden = summary === null;
  if (summary === null) {
    elements.task.textContent = "Task unavailable";
    for (const team of elements.teams) team.replaceChildren();
    elements.taskSelect.disabled = true;
    return;
  }
  elements.task.textContent = `Task mode: ${summary.task}`;
  elements.taskSelect.disabled = summary.taskMode !== 1;
  for (const [index, team] of summary.teams.entries()) {
    const root = elements.teams[index];
    const title = root.ownerDocument.createElement("strong");
    title.className = "match-scoreboard__team-label";
    title.textContent = team.label;
    const name = root.ownerDocument.createElement("span");
    name.className = "match-scoreboard__policy";
    name.textContent = team.name;
    name.title = team.provenance;
    root.replaceChildren(title, name);
    if (team.score !== null) {
      const score = root.ownerDocument.createElement("strong");
      score.className = "match-scoreboard__score";
      score.textContent = `Score: ${team.score}`;
      root.append(score);
    }
    if (team.result !== null) {
      const result = root.ownerDocument.createElement("strong");
      result.className = `match-scoreboard__result match-scoreboard__result--${team.result.toLowerCase()}`;
      result.textContent = team.result;
      root.append(result);
    }
  }
}
