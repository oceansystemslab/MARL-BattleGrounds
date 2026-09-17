"""Reduce complete tournament game measurements into participant summaries.

This host-only module reads narrow final rows. It performs no simulation, rating
fit, JAX import or full-report expansion. The tournament runner first checks the
physical spawn pairs; this module binds that evidence to the exact rows being
summarized. Every played game has equal descriptive weight.
"""

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

HEADLINE_COLUMNS = (
    "run_id",
    "system_id",
    "system_name",
    "games_played",
    "completed_pairs",
    "episode_length_min",
    "episode_length_max",
    "episode_length_median",
    "episode_length_mean",
    "episode_length_std",
    "total_steps_played",
    "cumulative_policy_return",
    "mean_policy_return",
    "score_difference_min",
    "score_difference_max",
    "score_difference_median",
    "score_difference_mean",
    "score_difference_std",
    "total_score_difference",
    "kills_min",
    "kills_max",
    "kills_median",
    "kills_mean",
    "kills_std",
    "total_kills",
    "deaths_min",
    "deaths_max",
    "deaths_median",
    "deaths_mean",
    "deaths_std",
    "total_deaths",
    "kd_ratio",
)


def content_digest(value: object) -> str:
    """Hash JSON evidence using the recording format's exact canonical encoding.

    Accept JSON-compatible host values only. Sorted keys, compact separators and
    a final newline match recorded identities. Nonfinite values raise ValueError.
    No input is changed and no file is opened.
    """
    return sha256(
        (
            json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    ).hexdigest()


def match_digest(matches: Sequence[Mapping[str, object]]) -> str:
    """Identify final rows in deterministic origin order, independent of arrival.

    Rows must contain JSON-compatible values and phase, pass_id and episode_id.
    Missing fields or nonfinite numbers fail instead of changing missing cells.
    """
    ordered = sorted(
        matches,
        key=lambda row: (
            str(row["phase"]),
            str(row["pass_id"]),
            int(cast(int, row["episode_id"])),
        ),
    )
    return content_digest(ordered)


def validate_evidence(
    matches: Sequence[Mapping[str, object]], evidence: Mapping[str, Any]
) -> None:
    """Check that prepared physical-pair evidence belongs to these exact rows.

    The private runner owns actual configuration and participant verification.
    This check prevents later consumers from mixing its result with another
    population, schedule or set of measurements. It does not turn an arbitrary
    caller-provided ID into physical proof. Invalid joins raise ValueError.
    """
    if (
        evidence.get("version") != 1
        or evidence.get("protocol") != "fixed-team-spawn-v1"
    ):
        raise ValueError("headline metrics require verified fixed-team spawn pairs")
    if evidence.get("match_digest") != match_digest(matches):
        raise ValueError("tournament evidence does not match the supplied game rows")
    games = cast(Mapping[str, Any], evidence.get("games"))
    if not isinstance(games, Mapping) or len(games) != len(matches):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("tournament evidence must cover every game exactly once")
    schedule = {str(row["episode_id"]): row for row in evidence["schedule"]}
    if len(schedule) != len(games) or set(schedule) != set(games):
        raise ValueError("tournament schedule and game evidence coverage differ")
    seen: set[str] = set()
    for row in matches:
        key = str(row["episode_id"])
        if key in seen or key not in games:
            raise ValueError("duplicate or missing tournament game evidence")
        seen.add(key)
        game = games[key]
        if any(
            game.get(field) != row.get(field)
            for field in ("run_id", "phase", "pass_id", "episode_id")
        ):
            raise ValueError(
                "tournament game origin differs from its recorded evidence"
            )
        declaration = schedule[key]
        if any(
            game.get(field) != declaration.get(field)
            for field in (
                "block_id",
                "source_config_id",
                "resolved_config_id",
                "spawn_locations",
            )
        ):
            raise ValueError("tournament game evidence differs from its schedule")
        pass_key = json.dumps((row["phase"], row["pass_id"]), separators=(",", ":"))
        entry = evidence["passes"].get(pass_key)
        if entry is None:
            raise ValueError("tournament evidence has no originating pass")
        episode = entry.get("episodes", {}).get(key, {})
        owners = episode.get("system_ids", entry.get("system_ids", {}))
        for team in ("team_a", "team_b"):
            identifier = game.get(team + "_system_id")
            if (
                identifier != owners.get(team)
                or identifier not in evidence["systems"]
                or evidence["systems"][identifier].get("name")
                != row.get(team + "_policy")
            ):
                raise ValueError(
                    "tournament participant evidence differs from its recorded owner"
                )
        if game.get("comparison_kind") != "verified_spawn_pair":
            raise ValueError(
                "custom or unverified games cannot complete headline metrics"
            )
    if evidence.get("schedule_digest") != content_digest(evidence.get("schedule")):
        raise ValueError("tournament schedule differs from its prepared evidence")


def _integer(row: Mapping[str, object], field: str) -> int:
    """Read one required exact integer without rounding a stored float."""
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"required tournament {field} must be an exact integer")
    return int(value)


def _measurement(row: Mapping[str, object], field: str) -> float:
    """Read one finite measurement; missing values never become zero."""
    value = row.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
    ):
        raise ValueError(f"headline metrics require every game's {field}")
    return float(value)


def _distribution(values: Sequence[int | float], prefix: str) -> dict[str, int | float]:
    """Return exact endpoints/median and float64 mean/population spread.

    Integer middle pairs are added as Python integers before division. Floating
    measurements retain their stored precision. Every value enters every field.
    """
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    array = np.asarray(values, np.float64)
    return {
        prefix + "_min": ordered[0],
        prefix + "_max": ordered[-1],
        prefix + "_median": median,
        prefix + "_mean": float(array.mean()),
        prefix + "_std": float(array.std(ddof=0)),
    }


def summarize_headlines(
    matches: Sequence[Mapping[str, object]],
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Summarize every complete entrant from verified priority/full game rows.

    Parameters
    ----------
    matches : sequence of mappings
        Complete final match rows with exact integer lengths/scores and finite
        return, kill, death and score-difference measurements. Each game appears
        once and keeps its original recorded Team A/B ownership.
    evidence : mapping
        The private runner's version-1 verified population, schedule, physical
        configuration and registration evidence bound to these exact rows.

    Returns
    -------
    tuple[dict, ...]
        One HEADLINE_COLUMNS row per System ID, sorted by ID. Medians are exact,
        standard deviations use divisor N, and zero deaths gives kd_ratio=None.
        Step totals count a shared game for each participant, so their population
        sum is twice the unique environment transitions. Scores can include
        authored points; kills remain observed events. Ratings are not changed.

    Raises
    ------
    ValueError
        Coverage, evidence, ownership or required measurements are inconsistent.

    Notes
    -----
    Call only for priority/full tournaments after physical qualification. None
    mode must skip this function even if selected full reports happen to exist.
    Arrays retain O(number of games) narrow values, never full-report vectors.
    """
    validate_evidence(matches, evidence)
    systems = evidence["systems"]
    population = evidence["population_system_ids"]
    if (
        not population
        or len(set(population)) != len(population)
        or any(i not in systems for i in population)
    ):
        raise ValueError("headline population requires distinct registered systems")
    data: dict[str, dict[str, list[int | float]]] = {
        identifier: {
            key: []
            for key in (
                "episode_length",
                "score_difference",
                "return",
                "kills",
                "deaths",
            )
        }
        for identifier in population
    }
    pairs: dict[str, dict[int, list[int]]] = {
        identifier: defaultdict(list) for identifier in population
    }
    run_ids = {row["run_id"] for row in matches}
    if len(run_ids) != 1:
        raise ValueError("headline rows must belong to one resolved tournament run")
    for row in sorted(
        matches,
        key=lambda r: (
            str(r["phase"]),
            str(r["pass_id"]),
            int(cast(int, r["episode_id"])),
        ),
    ):
        game = evidence["games"][str(row["episode_id"])]
        length = _integer(row, "episode_length")
        score_a, score_b = (
            _integer(row, "team_a_score"),
            _integer(row, "team_b_score"),
        )
        if length <= 0 or min(score_a, score_b) < 0:
            raise ValueError("completed game length and scores must be valid")
        difference = score_a - score_b
        if _measurement(row, "score_difference") != float(np.float32(difference)):
            raise ValueError(
                "score_difference does not match the exact recorded scores"
            )
        owners = (game["team_a_system_id"], game["team_b_system_id"])
        if owners[0] == owners[1] or any(owner not in data for owner in owners):
            raise ValueError(
                "game owners differ from the complete tournament population"
            )
        for team, identifier in enumerate(owners):
            entry = data[identifier]
            entry["episode_length"].append(length)
            entry["score_difference"].append(difference if team == 0 else -difference)
            prefix = "team_a_" if team == 0 else "team_b_"
            for field in ("return", "kills", "deaths"):
                value = _measurement(row, prefix + field)
                if field != "return" and value < 0:
                    raise ValueError("observed kills and deaths cannot be negative")
                entry[field].append(value)
            pairs[identifier][game["block_id"]].append(game["spawn_locations"])
    rows: list[dict[str, Any]] = []
    for identifier in sorted(population):
        entry = data[identifier]
        lengths = entry["episode_length"]
        if not lengths or any(
            sorted(choices) != [0, 1] for choices in pairs[identifier].values()
        ):
            raise ValueError("every entrant needs complete verified spawn pairs")
        total_kills = float(np.sum(entry["kills"], dtype=np.float64))
        total_deaths = float(np.sum(entry["deaths"], dtype=np.float64))
        row = {
            "run_id": next(iter(run_ids)),
            "system_id": identifier,
            "system_name": systems[identifier]["name"],
            "games_played": len(lengths),
            "completed_pairs": len(pairs[identifier]),
            **_distribution(lengths, "episode_length"),
            "total_steps_played": sum(lengths),
            "cumulative_policy_return": float(
                np.sum(entry["return"], dtype=np.float64)
            ),
            "mean_policy_return": float(np.mean(entry["return"], dtype=np.float64)),
            **_distribution(entry["score_difference"], "score_difference"),
            "total_score_difference": sum(entry["score_difference"]),
            **_distribution(entry["kills"], "kills"),
            "total_kills": total_kills,
            **_distribution(entry["deaths"], "deaths"),
            "total_deaths": total_deaths,
            "kd_ratio": total_kills / total_deaths if total_deaths else None,
        }
        rows.append({name: row[name] for name in HEADLINE_COLUMNS})
    return tuple(rows)
