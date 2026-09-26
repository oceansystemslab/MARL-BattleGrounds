"""Freeze a declared validation field's members before refitting or testing them.

Tournament setup binds the rule and field before any game. Selection reads the
existing checked results and ratings; it never loads a controller or plays a
replacement game. The ordinary tournament writer remains the only game writer.
These tools do not approve an official release or establish learning quality.
"""

from __future__ import annotations

# The shared result and writer owners keep record checks and durable JSON writes.
# pyright: reportPrivateUsage=false
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from marl_battlegrounds.evaluation.tournament_config import (
    canonical_json,
    read_config_json,
    snapshot_identity,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.results import (
        SavedResults,
        TournamentResult,
    )

Row = dict[str, Any]
_FORMAT = "marlbg-initial-population"
_POLICY = "require-complete-field"


def _digest(value: object) -> str:
    """Hash finite canonical JSON without mutating its contents."""
    return sha256(canonical_json(value)).hexdigest()


def _copy(value: Mapping[str, Any]) -> Row:
    """Copy a finite JSON mapping, rejecting non-JSON values and keys."""
    return cast(Row, json.loads(canonical_json(dict(value))))


def read_selection_request(value: str | Path | Mapping[str, Any]) -> Row:
    """Read a small declaration, resolving its population path from its file.

    Paths name UTF-8 JSON objects. Mapping paths use the current directory.
    No model or numerical backend is loaded. Shape and field checks belong to
    binding, when the actual entrant names and conditions are known.
    """
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        result, base = read_config_json(path), path.parent
    elif isinstance(cast(object, value), Mapping):
        result, base = _copy(value), Path.cwd()
    else:
        raise TypeError("selection must be a JSON path or mapping")
    if "population" in result:
        if not isinstance(result["population"], str) or not result["population"]:
            raise ValueError("selection.population must name a population JSON file")
        result["population"] = str(
            (base / Path(result["population"]).expanduser()).resolve()
        )
    return result


def _validation_rule(request: Mapping[str, Any], names: Sequence[str]) -> Row:
    """Normalize one complete-field rule without inventing omitted entrants."""
    allowed = {"stage", "size", "entrant_order", "failure_policy"}
    if set(request) - allowed:
        raise ValueError(
            "Unknown validation selection fields: "
            + ", ".join(sorted(set(request) - allowed))
        )
    size = request.get("size")
    if type(size) is not int or not 2 <= size <= len(names):
        raise ValueError(
            "selection.size must be an integer from 2 through the candidate count"
        )
    order = request.get("entrant_order", list(names))
    if not isinstance(order, list) or any(
        not isinstance(item, str) for item in cast(list[object], order)
    ):
        raise ValueError("selection.entrant_order must list each candidate name once")
    order = cast(list[str], order)
    if (
        len(order) != len(names)
        or len(set(order)) != len(order)
        or set(order) != set(names)
    ):
        raise ValueError("selection.entrant_order must list each candidate name once")
    if request.get("failure_policy", _POLICY) != _POLICY:
        raise ValueError(
            "selection.failure_policy must be require-complete-field; "
            "failed games stay visible"
        )
    if request.get("stage", "validation") != "validation":
        raise ValueError("This selection rule requires the validation stage")
    return {
        "stage": "validation",
        "size": size,
        "entrant_order": list(order),
        "failure_policy": _POLICY,
    }


def _field_id(config: Mapping[str, Any]) -> str:
    """Bind the full field, rules, seeds and assets without a circular selection ID."""
    return snapshot_identity(
        {key: value for key, value in config.items() if key != "selection"}
    )


def _population_id(value: Mapping[str, Any]) -> str:
    """Identify immutable membership and its full selection evidence."""
    return _digest({key: item for key, item in value.items() if key != "population_id"})


def read_population(path: str | Path) -> Row:
    """Read and check a frozen complete population, without reading its models.

    An incomplete decision cannot open a test stage. The content hash detects
    accidental changes; it is not an external signature or official approval.
    """
    result = read_config_json(path)
    _check_population(result)
    return result


def _check_population(value: Mapping[str, Any]) -> None:
    """Check a complete decision's identity and unique selected descriptors."""
    if (
        value.get("format") != _FORMAT
        or value.get("version") != 1
        or value.get("status") != "complete"
    ):
        raise ValueError("Test selection requires a complete frozen population")
    required = {
        "format",
        "version",
        "status",
        "selection_id",
        "declaration",
        "source",
        "candidates",
        "members",
        "rankings",
        "scheduled_games",
        "completed_games",
        "missing_games",
        "source_status",
        "sampling_evidence",
        "population_id",
    }
    if not required <= value.keys() or set(value) - required - {"failures"}:
        raise ValueError("Frozen population fields are incomplete or unsupported")
    if value.get("population_id") != _population_id(value):
        raise ValueError("Frozen population content differs from population_id")
    members = value.get("members")
    if (
        not isinstance(members, list)
        or len(cast(list[object], members)) < 2
        or any(not isinstance(row, dict) for row in cast(list[object], members))
    ):
        raise ValueError(
            "Frozen population must contain at least two member descriptors"
        )
    members = cast(list[Row], members)
    names = [row.get("name") for row in members]
    ids = [row.get("entrant_id") for row in members]
    if (
        any(not isinstance(item, str) or not item for item in names + ids)
        or len(set(names)) != len(names)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Frozen population member identities must be distinct")
    declaration = value.get("declaration", {})
    if not isinstance(declaration, dict) or declaration.get("size") != len(members):
        raise ValueError("Frozen population size differs from its declaration")
    candidates, rankings = value["candidates"], value["rankings"]
    if (
        not isinstance(candidates, list)
        or not isinstance(rankings, list)
        or any(
            not isinstance(row, dict)
            for row in [*cast(list[object], candidates), *cast(list[object], rankings)]
        )
    ):
        raise ValueError(
            "Frozen population needs its complete candidate and rating rows"
        )
    try:
        selected, ordered = rank_population(
            cast(list[Row], candidates),
            cast(list[Row], rankings),
            cast(Row, declaration),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Frozen population has invalid selection evidence") from error
    if members != selected or rankings != ordered:
        raise ValueError("Frozen members differ from their declared selection evidence")
    if (
        type(value["scheduled_games"]) is not int
        or value["scheduled_games"] <= 0
        or value["completed_games"] != value["scheduled_games"]
        or value["missing_games"]
        or value["source_status"] != "complete"
    ):
        raise ValueError("Frozen population does not record a complete source field")


def _members_match(
    participants: Sequence[Mapping[str, Any]], population: Mapping[str, Any]
) -> None:
    """Compare method identities, ignoring optional old ratings and file locations."""
    expected = {row["name"]: row for row in population["members"]}
    if {row["name"] for row in participants} != set(expected):
        raise ValueError(
            "Test field must contain exactly the frozen population members"
        )
    for row in participants:
        saved = expected[row["name"]]
        if any(
            row.get(key) != saved.get(key) for key in ("entrant_id", "controller_id")
        ):
            raise ValueError(
                "Test entrant identity differs from the frozen population: "
                + row["name"]
            )


def bind_selection(
    request: str | Path | Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    entrant_order: Sequence[str] | None = None,
) -> Row:
    """Bind a validation rule or a previously frozen test field before games.

    request is a short declaration. config is the prepared v2 field without its
    selection binding. entrant_order supplies the input order when the rule omits
    its explicit tie order. Return finite JSON ready for the descriptor. Test
    declarations name a population file, copied into the binding so resume does
    not depend on that file's continued location. Invalid stages or fields fail.
    """
    raw = read_selection_request(request)
    names = [row["name"] for row in config["participants"]]
    population = None
    if raw.get("stage", "validation") == "validation":
        declaration = _validation_rule(
            raw, names if entrant_order is None else entrant_order
        )
    else:
        if set(raw) != {"stage", "population"} or raw["stage"] != "test":
            raise ValueError("Test selection accepts only stage='test' and population")
        population = read_population(raw["population"])
        declaration = {"stage": "test", "population_id": population["population_id"]}
    result = {
        "version": 1,
        "field_id": _field_id(config),
        "declaration": declaration,
        "population": population,
    }
    result["selection_id"] = _digest(result)
    validate_bound_selection({**config, "selection": result})
    return result


def validate_bound_selection(config: Mapping[str, Any]) -> None:
    """Validate optional v2 selection content without files, imports or games.

    The full descriptor already checks actual entrant/config/schedule assets.
    This owner checks the pre-game rule, equal weights and declared map split.
    Unrelated tournaments without selection remain unchanged.
    """
    if "selection" not in config:
        return
    bound = config["selection"]
    if config["version"] != 2 or not isinstance(bound, dict):
        raise ValueError("Initial selection needs a bound version 2 tournament")
    bound = cast(Row, bound)
    if (
        set(bound)
        != {"version", "field_id", "declaration", "population", "selection_id"}
        or type(bound["version"]) is not int
        or bound["version"] != 1
    ):
        raise ValueError("Unsupported selection binding")
    if bound["field_id"] != _field_id(config) or bound["selection_id"] != _digest(
        {key: value for key, value in bound.items() if key != "selection_id"}
    ):
        raise ValueError("Selection binding differs from the declared field")
    declaration = bound["declaration"]
    if not isinstance(declaration, dict):
        raise ValueError("Selection declaration must be an object")
    declaration = cast(Row, declaration)
    stage = declaration.get("stage")
    split = "validation" if stage == "validation" else "test"
    if stage not in {"validation", "test"} or any(
        row["registered_map"] is None or row["split"] != split
        for row in config["conditions"]["map_sources"]
    ):
        raise ValueError(
            "Selection games require registered maps in their declared "
            "validation or test map split"
        )
    if config["record_sources"]:
        raise ValueError(
            "Selection stages start separate records; "
            "earlier games cannot bind a new rule"
        )
    if stage == "validation":
        expected = _validation_rule(
            declaration, [row["name"] for row in config["participants"]]
        )
        if declaration != expected or bound["population"] is not None:
            raise ValueError("Validation selection binding is inconsistent")
        weights = config["analysis"]["opponent_weights"]
        if weights is not None and len(set(weights.values())) != 1:
            raise ValueError(
                "Initial population selection requires equal opponent weights"
            )
    else:
        population = bound["population"]
        if not isinstance(population, dict):
            raise ValueError("Test selection needs its frozen population")
        population = cast(Row, population)
        _check_population(population)
        if declaration != {
            "stage": "test",
            "population_id": population["population_id"],
        }:
            raise ValueError("Test declaration differs from its frozen population")
        _members_match(config["participants"], population)


def check_selection_plan(
    config: Mapping[str, Any],
    games: Sequence[Mapping[str, Any]],
    *,
    challenger_id: str | None,
    budget: Mapping[str, Any],
) -> None:
    """Reject post-declaration population, budget or reused-game substitutions."""
    if "selection" not in config:
        return
    validate_bound_selection(config)
    if challenger_id is not None:
        raise ValueError(
            "A selection field is frozen; "
            "declare the challenger as a candidate before games"
        )
    if (
        budget["resolved_games_per_opponent"]
        != config["conditions"]["games_per_opponent"]
    ):
        raise ValueError("A selection field keeps its predeclared game budget")
    if any(
        game.get("origin") is not None or game.get("prior_origin") is not None
        for game in games
    ):
        raise ValueError(
            "Selection stages cannot relabel earlier games as predeclared evidence"
        )


def rank_population(
    participants: Sequence[Mapping[str, Any]],
    rankings: Sequence[Mapping[str, Any]],
    declaration: Mapping[str, Any],
) -> tuple[list[Row], list[Row]]:
    """Rank a complete field by Elo, expected score, then its declared name order.

    Return copied selected descriptors and all ordered rating rows. Exact saved
    floating-point ties use the next key; no rounding or hidden tolerance changes
    membership. Missing, duplicate or nonfinite ratings raise ValueError. This
    pure helper never fits, drops weak entrants, mutates inputs or writes files.
    """
    names = [row["name"] for row in participants]
    rule = _validation_rule(declaration, names)
    by_name = {row["name"]: row for row in participants}
    rows = {row.get("policy"): row for row in rankings}
    if (
        len(by_name) != len(names)
        or len(rows) != len(rankings)
        or set(rows) != set(names)
    ):
        raise ValueError("Ratings must cover the complete candidate field exactly once")
    for row in rows.values():
        for key in ("elo", "expected_score"):
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(
                    "Complete finite Elo and expected score "
                    "are required for every candidate"
                )
    order = {name: index for index, name in enumerate(rule["entrant_order"])}
    ranked = sorted(
        rows.values(),
        key=lambda row: (-row["elo"], -row["expected_score"], order[row["policy"]]),
    )
    return (
        [_copy(by_name[row["policy"]]) for row in ranked[: rule["size"]]],
        [_copy(row) for row in ranked],
    )


def select_initial_population(
    result: str | Path | TournamentResult | SavedResults,
    *,
    output_dir: str | Path,
    declaration: str | Path | Mapping[str, Any] | None = None,
) -> Row:
    """Freeze top-N membership from a predeclared validation tournament.

    Parameters
    ----------
    result : path, TournamentResult or SavedResults
        Complete or interrupted v2 field produced with a validation selection
        declaration. Saved reads check the writer's configuration and row hashes.
    output_dir : str or Path
        Exact new or empty directory outside the source run. Writes population.json
        once, then refit.json. Existing decisions and source games stay untouched.
    declaration : path, mapping or None, default=None
        Optional assertion of the rule already bound before games. It cannot add
        or replace a rule. File paths resolve from that declaration file.

    Returns
    -------
    dict
        Immutable decision contents plus the separate refit report when complete.
        status='incomplete' retains missing game IDs and no selected members.
        status='complete' freezes members and population_id before refitting.
        Refit failure is visible in refit.status and never changes membership.

    Raises
    ------
    ValueError
        Unbound or changed declaration, test-stage input, invalid records, unsafe
        output folder or incomplete/invalid rating coverage.
    OSError
        A saved file or the new decision cannot be read or written.

    Notes
    -----
    Host-only result reading and rating refit; no controller or game is executed.
    Whole-field ratings are reused from the existing checked summary. A selected
    subset uses the same joint Elo fitter on its retained validation games. This
    answers a different ranking question and cannot change membership. Unknown
    determinism stays unknown; repeated games do not become independent samples.
    This is a research tool, not maintainer admission or official eligibility.
    """
    from marl_battlegrounds.evaluation.canonical_results import CanonicalView
    from marl_battlegrounds.evaluation.results import (
        CanonicalTournamentResult,
        SavedResults,
        load_results,
    )
    from marl_battlegrounds.evaluation.run_writer import _atomic_json
    from marl_battlegrounds.evaluation.tournament_records import origin_key
    from marl_battlegrounds.evaluation.tournament_reuse import analysis_schedule
    from marl_battlegrounds.evaluation.tournament_statistics import summarize_tournament

    loaded = load_results(result) if isinstance(result, (str, Path)) else result
    if not isinstance(
        cast(object, loaded), (CanonicalTournamentResult, SavedResults)
    ) or not isinstance(loaded._view, CanonicalView):
        raise ValueError(
            "Initial selection needs a configured tournament "
            "with its saved pre-game declaration"
        )
    view = loaded._view
    records, config = view.records, view.records.config
    validate_bound_selection(config)
    bound = config.get("selection")
    if bound is None or bound["declaration"]["stage"] != "validation":
        raise ValueError(
            "Initial selection needs a validation declaration bound before games"
        )
    if declaration is not None:
        supplied = _validation_rule(
            read_selection_request(declaration),
            [row["name"] for row in config["participants"]],
        )
        if supplied != bound["declaration"]:
            raise ValueError(
                "Supplied selection declaration differs from the saved pre-game rule"
            )
    check_selection_plan(
        config,
        records.games,
        challenger_id=view.manifest["tournament_reuse"].get("challenger_id"),
        budget=view.manifest["tournament_reuse"]["budget"],
    )
    destination = Path(output_dir).expanduser().resolve()
    if loaded.run_dir is not None and (
        destination == loaded.run_dir or destination.is_relative_to(loaded.run_dir)
    ):
        raise ValueError(
            "Write the population decision outside the original run folder"
        )
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError(
            "output_dir must be a new or empty directory; "
            "use a new folder for each decision"
        )
    missing = [
        game["logical_game_id"] for game in records.games if not records.completed(game)
    ]
    summary = view.manifest.get("tournament_summary", {})
    complete = not missing and loaded.status == "complete"
    decision: Row = {
        "format": _FORMAT,
        "version": 1,
        "status": "complete" if complete else "incomplete",
        "selection_id": bound["selection_id"],
        "declaration": _copy(bound["declaration"]),
        "source": {
            "run_id": view.manifest["run_id"],
            "config_id": config["snapshot_id"],
            "games_sha256": view.manifest["tournament_reuse"]["games_sha256"],
            "summary_digest": summary.get("digest"),
        },
        "candidates": [_copy(row) for row in config["participants"]],
        "members": [],
        "rankings": [],
        "scheduled_games": len(records.games),
        "completed_games": len(records.games) - len(missing),
        "missing_games": missing,
        "source_status": loaded.status,
        "failures": [
            {
                "phase": row.get("phase"),
                "pass_id": row.get("pass_id"),
                "reason": row["result_state"].get("reason"),
            }
            for row in view.manifest.get("passes", {}).values()
            if row.get("result_state", {}).get("status") == "failed"
        ],
        "sampling_evidence": _copy(summary.get("metadata", {})),
    }
    if complete:
        rows = list(view.iter_rows("tournament_rankings", 128))
        members, ranked = rank_population(
            config["participants"], rows, bound["declaration"]
        )
        decision.update(members=members, rankings=ranked)
        decision["population_id"] = _population_id(decision)
    destination.mkdir(parents=True, exist_ok=True)
    reservation = destination / ".selection.lock"
    with reservation.open("x"):
        pass
    try:
        if any(path != reservation for path in destination.iterdir()):
            raise ValueError("output_dir became occupied while preparing the decision")
        _atomic_json(destination / "population.json", decision)
        if not complete:
            return decision
        names = {row["entrant_id"]: row["name"] for row in decision["members"]}
        selected_games = tuple(
            game
            for game in records.games
            if game["team_a"] in names and game["team_b"] in names
        )
        schedule = analysis_schedule(selected_games, names)
        refit: Row
        if len(names) == len(config["participants"]):
            refit = {
                "status": "complete",
                "reused_full_field": True,
                "rankings": decision["rankings"],
                "metadata": decision["sampling_evidence"],
            }
        else:
            outcomes = {
                origin_key(row): row["outcome"]
                for batch in records.iter_rows(
                    "match_results.csv",
                    game_ids=[game["logical_game_id"] for game in selected_games],
                    columns=("run_id", "phase", "pass_id", "episode_id", "outcome"),
                )
                for row in batch
            }
            try:
                stats = summarize_tournament(
                    schedule,
                    {
                        game["logical_game_id"]: outcomes[
                            origin_key(records.origin(game))
                        ]
                        for game in selected_games
                    },
                    seed=config["analysis"]["bootstrap_seed"],
                    method_sampling={
                        name: decision["sampling_evidence"]
                        .get("method_sampling", {})
                        .get(
                            name,
                            {
                                "determinism": "unknown",
                                "basis": "No saved sampling facts",
                            },
                        )
                        for name in names.values()
                    },
                )
                refit = {
                    "status": "complete",
                    "reused_full_field": False,
                    "rankings": list(stats.tournament_results),
                    "metadata": stats.metadata,
                }
            except RuntimeError as error:
                refit = {"status": "failed", "reason": str(error), "rankings": []}
        refit["population_id"] = decision["population_id"]
        _atomic_json(destination / "refit.json", refit)
        return {**decision, "refit": refit}
    finally:
        reservation.unlink()
