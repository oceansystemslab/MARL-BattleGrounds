/**
 * @file Build and paint match labels from an installed authorized presentation.
 * The scoreboard uses the supplied task, map, team and result facts. It does
 * not infer scores or winners from agents drawn on the battlefield.
 */
import { isAuthorizedPresentationFrame } from "./authorized-presentation-adapter.js";

/**
 * Read a normalized match's display side, using the historical layout if absent.
 * teamId is 1 or 2. This reads recorded metadata, never current actor positions.
 * @param {Record<string, any> | null | undefined} match
 * @param {number} teamId
 * @returns {"left" | "right"}
 */
export function matchTeamSide(match, teamId) {
  return match?.teams?.[teamId - 1]?.display_side ?? (teamId === 1 ? "left" : "right");
}

/**
 * Build labels from a recognized presentation's match_summary, or return null.
 *
 * presentation may be any input. Unrecognized frames and missing summaries
 * produce null. Recognized records supply the map name/split, Team A/B names,
 * policy/checkpoint references and outcome. TDM adds score/threshold text;
 * combat diagnostics omit scores. The returned outer record and team records
 * are frozen; the teams array is newly allocated but not frozen. No DOM or
 * input record is changed. Normalization owns the validity of nested facts.
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
    map: match.map
      ? `${match.map.display_name}${match.map.split ? ` (${match.map.split[0].toUpperCase()}${match.map.split.slice(1)} Map)` : ""}`
      : null,
    teams: match.teams.map(
      (/** @type {Record<string, any>} */ team, /** @type {number} */ index) =>
        Object.freeze({
          teamId: team.team_id,
          side: matchTeamSide(match, team.team_id),
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
 * Replace the supplied scoreboard DOM with the current authorized match labels.
 *
 * elements supplies root, task text, two team containers in left/right order and
 * the task selector. presentation is passed to matchSummaryView. If unavailable,
 * hide the root, clear team contents and disable the selector. Otherwise show
 * task/map text, team names, TDM scores and any final result; policy references
 * become name tooltips. This mutates those elements and returns undefined.
 * The caller must supply matching containers; DOM shape is not validated.
 *
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
  elements.task.textContent = `Task mode: ${summary.task}${summary.map ? ` · Map: ${summary.map}` : ""}`;
  elements.taskSelect.disabled = summary.taskMode !== 1;
  for (const team of summary.teams) {
    const root = elements.teams[team.side === "left" ? 0 : 1];
    root.dataset.team = String(team.teamId);
    root.classList.toggle("match-scoreboard__team--a", team.teamId === 1);
    root.classList.toggle("match-scoreboard__team--b", team.teamId === 2);
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
