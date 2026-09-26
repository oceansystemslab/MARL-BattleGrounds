"""Check declared population selection and its separate saved test stage.

Pure host cases check rating ties, complete fields and exact entrant identities.
Tiny CPU games check predeclared validation, saved-only selection, frozen members,
changed refit order, interruption, resume and test-map gating. Synthetic short
runs are software checks, not official entrants or evidence of learned skill.
"""

# Fault hooks inspect saved boundaries without replacing the public runner.
# pyright: reportPrivateUsage=false

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation import tournament_statistics
from marl_battlegrounds.evaluation.results import CanonicalTournamentResult
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.evaluation.tournament_config import canonical_json
from marl_battlegrounds.evaluation.tournament_schedule import TournamentMatch

_ENTRANTS = ["random", "tdm-alpha", "tdm-beta"]
_ORDER = ["tdm-beta", "tdm-alpha", "random"]


def _declaration(*, size: int = 2) -> dict[str, Any]:
    return {
        "size": size,
        "entrant_order": list(_ORDER),
        "failure_policy": "require-complete-field",
    }


def _field(*, selection: object = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "entrants": list(_ENTRANTS),
        "maps": [42, 43, 44, 45, 46],
        "games_per_opponent": 10,
        "max_steps": 1,
        "seed": 93,
        "metrics": "none",
    }
    if selection is not None:
        value["selection"] = selection
    return value


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _bytes(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _unexpected(*args: object, **kwargs: object) -> None:
    raise AssertionError("Saved selection must not load methods or play games")


def _participants(names: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "entrant_id": f"entrant-{name}",
            "controller_id": f"controller-{name}",
        }
        for name in names
    ]


@pytest.mark.parametrize("size", [2, 3])
def test_rating_then_score_then_declared_order_choose_exact_members(size: int) -> None:
    module = import_module("marl_battlegrounds.evaluation.population_selection")
    participants = _participants(["first", "second", "third"])
    rows = [
        {"policy": "second", "elo": 1200.0, "expected_score": 0.7},
        {"policy": "first", "elo": 1201.0, "expected_score": 0.1},
        {"policy": "third", "elo": 1200.0, "expected_score": 0.7},
    ]
    declaration = {
        "size": size,
        "entrant_order": ["third", "second", "first"],
        "failure_policy": "require-complete-field",
    }
    before = copy.deepcopy((participants, rows, declaration))
    members, rankings = module.rank_population(participants, rows, declaration)
    expected = [participants[0], participants[2], participants[1]]
    assert members == expected[:size]
    assert [row["policy"] for row in rankings] == ["first", "third", "second"]
    assert (participants, rows, declaration) == before


def test_expected_score_breaks_equal_rating_before_entrant_order() -> None:
    module = import_module("marl_battlegrounds.evaluation.population_selection")
    participants = _participants(["weak", "better", "best"])
    rankings = [
        {"policy": "weak", "elo": 1200.0, "expected_score": 0.05},
        {"policy": "better", "elo": 1200.0, "expected_score": 0.4},
        {"policy": "best", "elo": 1200.0, "expected_score": 0.8},
    ]
    declaration = {
        "size": 3,
        "entrant_order": ["weak", "better", "best"],
        "failure_policy": "require-complete-field",
    }
    members, ordered = module.rank_population(participants, rankings, declaration)
    assert [row["name"] for row in members] == ["best", "better", "weak"]
    assert members[-1] == participants[0]
    assert [row["policy"] for row in ordered] == ["best", "better", "weak"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "unknown", "nan", "infinity"])
def test_ranking_cannot_silently_drop_or_invent_candidate_evidence(bad: str) -> None:
    module = import_module("marl_battlegrounds.evaluation.population_selection")
    participants = _participants(_ENTRANTS)
    rows = [
        {"policy": name, "elo": 1200.0, "expected_score": 0.5} for name in _ENTRANTS
    ]
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows[-1] = dict(rows[0])
    elif bad == "unknown":
        rows[-1]["policy"] = "unlisted"
    elif bad == "nan":
        rows[-1]["elo"] = float("nan")
    else:
        rows[-1]["expected_score"] = float("inf")
    with pytest.raises(ValueError):
        module.rank_population(participants, rows, _declaration())


@pytest.fixture(scope="module")
def completed_field(
    tmp_path_factory: pytest.TempPathFactory,
) -> CanonicalTournamentResult:
    output = tmp_path_factory.mktemp("population-field")
    original_write = RunWriter.write
    observed: list[str] = []

    def inspect_before_game(writer: RunWriter, info: EpisodeInfo) -> None:
        if not observed:
            saved_config = _json(writer.run_dir / "tournament_config.json")
            manifest = _json(writer.run_dir / "run_details.json")
            binding = saved_config["selection"]
            assert binding["declaration"] == {"stage": "validation", **_declaration()}
            assert binding["field_id"] and binding["selection_id"]
            assert binding["population"] is None
            assert manifest["details"]["canonical_config"] == saved_config
            observed.append(binding["selection_id"])
        original_write(writer, info)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RunWriter, "write", inspect_before_game)
        result = marl_bgs.run_tournament(
            config=_field(selection=_declaration()),
            output_dir=output,
            num_envs=2,
            chunk_size=1,
        )
    assert observed == [
        result.metadata["canonical_config"]["selection"]["selection_id"]
    ]
    assert result.run_dir is not None
    assert isinstance(result, CanonicalTournamentResult)
    assert len(result.matches) == 30
    assert {row["outcome"] for row in result.matches} == {3}
    return result


def test_selection_uses_saved_full_field_and_freezes_before_changed_refit(
    completed_field: CanonicalTournamentResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("marl_battlegrounds.evaluation.population_selection")
    assert completed_field.run_dir is not None
    before = _bytes(completed_field.run_dir)
    output = tmp_path / "decision"
    fit = tournament_statistics.summarize_tournament
    calls: list[set[str]] = []
    frozen: list[bytes] = []

    def reverse_refit(
        schedule: Sequence[TournamentMatch],
        outcomes: Mapping[int, int],
        *,
        seed: int = 0,
        opponent_weights: Mapping[str, float] | None = None,
        method_sampling: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tournament_statistics.TournamentStatistics:
        names = {name for game in schedule for name in (game.team_a, game.team_b)}
        calls.append(names)
        assert names == set(_ORDER[:2])
        decision_path = output / "population.json"
        frozen.append(decision_path.read_bytes())
        decision = _json(decision_path)
        assert decision["status"] == "complete"
        assert [row["name"] for row in decision["members"]] == _ORDER[:2]
        result = fit(
            schedule,
            outcomes,
            seed=seed,
            opponent_weights=opponent_weights,
            method_sampling=method_sampling,
        )
        return replace(
            result,
            tournament_results=tuple(
                next(row for row in result.tournament_results if row["policy"] == name)
                for name in reversed(_ORDER[:2])
            ),
        )

    monkeypatch.setattr(tournament_statistics, "summarize_tournament", reverse_refit)
    if hasattr(module, "summarize_tournament"):
        monkeypatch.setattr(module, "summarize_tournament", reverse_refit)
    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.canonical"),
        "_active_pair",
        _unexpected,
    )
    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "_evaluate_tournament_episodes",
        _unexpected,
    )
    decision = marl_bgs.select_initial_population(
        completed_field.run_dir, output_dir=output
    )
    assert calls == [set(_ORDER[:2])]
    assert decision["status"] == "complete"
    assert [row["name"] for row in decision["members"]] == _ORDER[:2]
    assert len(decision["rankings"]) == 3
    assert decision["scheduled_games"] == decision["completed_games"] == 30
    assert decision["missing_games"] == []
    assert decision["failures"] == []
    assert decision["population_id"]
    assert (output / "population.json").read_bytes() == frozen[0]
    assert [row["policy"] for row in _json(output / "refit.json")["rankings"]] == list(
        reversed(_ORDER[:2])
    )
    original = {
        row["name"]: row
        for row in completed_field.metadata["canonical_config"]["participants"]
    }
    assert decision["members"] == [original[name] for name in _ORDER[:2]]
    assert _bytes(completed_field.run_dir) == before


@pytest.mark.parametrize("route", ["result", "saved", "path"])
def test_result_routes_preserve_frozen_selection_identity(
    completed_field: CanonicalTournamentResult, tmp_path: Path, route: str
) -> None:
    assert completed_field.run_dir is not None
    result: Any = completed_field
    if route == "saved":
        result = marl_bgs.load_results(completed_field.run_dir)
    elif route == "path":
        result = completed_field.run_dir
    supplied = tmp_path / "declaration.json"
    supplied.write_text(json.dumps(_declaration()))
    if route == "saved":
        (tmp_path / "decision").mkdir()
    decision = marl_bgs.select_initial_population(
        result, output_dir=tmp_path / "decision", declaration=supplied
    )
    saved = _json(tmp_path / "decision/population.json")
    assert decision["population_id"] == saved["population_id"]
    assert decision["selection_id"] == saved["selection_id"]
    assert [row["name"] for row in saved["members"]] == _ORDER[:2]
    assert saved["declaration"] == {"stage": "validation", **_declaration()}


def test_changed_declaration_and_original_output_paths_fail_without_writes(
    completed_field: CanonicalTournamentResult, tmp_path: Path
) -> None:
    assert completed_field.run_dir is not None
    before = _bytes(completed_field.run_dir)
    output = tmp_path / "changed"
    with pytest.raises(ValueError):
        marl_bgs.select_initial_population(
            completed_field,
            output_dir=output,
            declaration=_declaration(size=3),
        )
    assert not output.exists()
    for destination in (completed_field.run_dir, completed_field.run_dir / "decision"):
        with pytest.raises(ValueError):
            marl_bgs.select_initial_population(completed_field, output_dir=destination)
    assert _bytes(completed_field.run_dir) == before


def test_existing_decision_is_never_overwritten(
    completed_field: CanonicalTournamentResult, tmp_path: Path
) -> None:
    output = tmp_path / "decision"
    marl_bgs.select_initial_population(completed_field, output_dir=output)
    before = _bytes(output)
    with pytest.raises(ValueError):
        marl_bgs.select_initial_population(completed_field, output_dir=output)
    assert _bytes(output) == before


def test_refit_failure_keeps_the_already_frozen_population(
    completed_field: CanonicalTournamentResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("marl_battlegrounds.evaluation.population_selection")
    output = tmp_path / "decision"

    def fail_refit(*args: object, **kwargs: object) -> None:
        assert _json(output / "population.json")["status"] == "complete"
        raise RuntimeError("injected selected-field refit failure")

    monkeypatch.setattr(tournament_statistics, "summarize_tournament", fail_refit)
    if hasattr(module, "summarize_tournament"):
        monkeypatch.setattr(module, "summarize_tournament", fail_refit)
    returned = marl_bgs.select_initial_population(completed_field, output_dir=output)
    assert returned["refit"]["status"] == "failed"
    assert "selected-field refit failure" in returned["refit"]["reason"]
    assert _json(output / "refit.json") == returned["refit"]
    decision = _json(output / "population.json")
    assert decision["status"] == "complete"
    assert decision["population_id"]
    assert [row["name"] for row in decision["members"]] == _ORDER[:2]


def test_unbound_completed_games_cannot_be_given_a_late_selection_rule(
    tmp_path: Path,
) -> None:
    result = marl_bgs.run_tournament(
        ["random", "tdm-alpha"],
        maps=[42],
        games_per_opponent=2,
        max_steps=1,
        metrics="none",
        num_envs=2,
        chunk_size=1,
        output_dir=tmp_path / "unbound",
    )
    assert result.run_dir is not None
    before = _bytes(result.run_dir)
    declaration = {
        "size": 2,
        "entrant_order": ["random", "tdm-alpha"],
        "failure_policy": "require-complete-field",
    }
    output = tmp_path / "late"
    with pytest.raises(ValueError):
        marl_bgs.select_initial_population(
            result, output_dir=output, declaration=declaration
        )
    assert not output.exists()
    assert _bytes(result.run_dir) == before


def test_interrupted_field_stays_incomplete_without_dropping_failed_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = RunWriter.write

    def stop_after_first(writer: RunWriter, info: EpisodeInfo) -> None:
        original(writer, info)
        writer.flush()
        raise RuntimeError("injected selection-field interruption")

    monkeypatch.setattr(RunWriter, "write", stop_after_first)
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="selection-field interruption"):
        marl_bgs.run_tournament(
            config=_field(selection=_declaration()),
            output_dir=output,
            num_envs=1,
            chunk_size=1,
        )
    run_dir = next(output.iterdir())
    before = _bytes(run_dir)
    decision = marl_bgs.select_initial_population(
        run_dir, output_dir=tmp_path / "incomplete"
    )
    assert decision["status"] == "incomplete"
    assert decision["members"] == []
    assert not decision.get("population_id")
    assert decision["scheduled_games"] == 30
    assert 0 < decision["completed_games"] < 30
    assert len(decision["missing_games"]) == 30 - decision["completed_games"]
    assert decision["declaration"]["entrant_order"] == _ORDER
    saved_passes = _json(run_dir / "run_details.json")["passes"]
    failed_passes = {
        (row["phase"], row["pass_id"])
        for row in saved_passes.values()
        if row.get("result_state", {}).get("status") == "failed"
    }
    assert failed_passes
    assert {
        (row["phase"], row["pass_id"]) for row in decision["failures"]
    } == failed_passes
    assert all(
        "selection-field interruption" in row["reason"] for row in decision["failures"]
    )
    assert not (tmp_path / "incomplete/refit.json").exists()
    assert _bytes(run_dir) == before


@pytest.mark.parametrize(
    "change",
    [
        {"size": 0},
        {"size": 1},
        {"size": 4},
        {"entrant_order": ["random", "tdm-alpha"]},
        {"entrant_order": ["random", "random", "tdm-beta"]},
        {"failure_policy": "drop-failed-entrants"},
    ],
)
def test_invalid_declarations_fail_before_game_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "_evaluate_tournament_episodes",
        _unexpected,
    )
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        marl_bgs.run_tournament(
            config=_field(selection={**_declaration(), **change}), output_dir=output
        )
    assert not output.exists() or list(output.iterdir()) == []


@pytest.mark.parametrize("maps", [[47], [42, 47], [41]])
def test_selection_validation_rejects_test_or_training_maps_before_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, maps: list[int]
) -> None:
    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "_evaluate_tournament_episodes",
        _unexpected,
    )
    config = _field(selection=_declaration())
    config["maps"] = maps
    config["games_per_opponent"] = 2 * len(maps)
    output = tmp_path / "wrong-maps"
    with pytest.raises(ValueError):
        marl_bgs.run_tournament(config=config, output_dir=output)
    assert not output.exists() or list(output.iterdir()) == []


def test_matching_resume_keeps_binding_and_changed_rule_refuses_before_actions(
    completed_field: CanonicalTournamentResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert completed_field.run_dir is not None
    before = _bytes(completed_field.run_dir)
    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.canonical"),
        "_active_pair",
        _unexpected,
    )
    monkeypatch.setattr(tournament_statistics, "summarize_tournament", _unexpected)
    resumed = marl_bgs.run_tournament(
        config=_field(selection=_declaration()),
        resume_from=completed_field.run_dir,
    )
    assert resumed.matches == completed_field.matches
    assert (
        resumed.metadata["canonical_config"]["selection"]
        == completed_field.metadata["canonical_config"]["selection"]
    )
    with pytest.raises(ValueError):
        marl_bgs.run_tournament(
            config=_field(selection=_declaration(size=3)),
            resume_from=completed_field.run_dir,
        )
    assert _bytes(completed_field.run_dir) == before


def test_selecting_whole_field_reuses_its_fit_without_a_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration = {
        "size": 2,
        "entrant_order": ["random", "tdm-alpha"],
        "failure_policy": "require-complete-field",
    }
    config = {
        "entrants": ["random", "tdm-alpha"],
        "maps": [42],
        "games_per_opponent": 2,
        "max_steps": 1,
        "selection": declaration,
    }
    result = marl_bgs.run_tournament(
        config=config,
        output_dir=tmp_path / "field",
        num_envs=2,
        chunk_size=1,
    )
    monkeypatch.setattr(tournament_statistics, "summarize_tournament", _unexpected)
    decision = marl_bgs.select_initial_population(
        result, output_dir=tmp_path / "decision"
    )
    assert decision["status"] == "complete"
    assert decision["refit"]["reused_full_field"] is True
    assert decision["refit"]["rankings"] == decision["rankings"]


def test_frozen_members_open_separate_test_records_and_resume_without_source_file(
    completed_field: CanonicalTournamentResult, tmp_path: Path
) -> None:
    assert completed_field.run_dir is not None
    before = _bytes(completed_field.run_dir)
    decision_dir = tmp_path / "decision"
    decision = marl_bgs.select_initial_population(
        completed_field, output_dir=decision_dir
    )
    config_dir = tmp_path / "test-config"
    config_dir.mkdir()
    config_path = config_dir / "field.json"
    config_path.write_text(
        json.dumps(
            {
                "entrants": _ORDER[:2],
                "maps": [47],
                "games_per_opponent": 2,
                "max_steps": 1,
                "selection": {
                    "stage": "test",
                    "population": "../decision/population.json",
                },
            }
        )
    )
    tested = marl_bgs.run_tournament(
        config=config_path,
        output_dir=tmp_path / "test-results",
        num_envs=2,
        chunk_size=1,
    )
    assert tested.run_dir is not None and tested.run_dir != completed_field.run_dir
    assert len(tested.matches) == 2
    assert {row["map_id"] for row in tested.matches} == {47}
    assert {row["map_id"] for row in completed_field.matches} == set(range(42, 47))
    binding = tested.metadata["canonical_config"]["selection"]
    assert binding["declaration"]["population_id"] == decision["population_id"]
    assert binding["population"]["members"] == decision["members"]
    with pytest.raises(ValueError):
        marl_bgs.select_initial_population(tested, output_dir=tmp_path / "late-test")
    (decision_dir / "population.json").rename(decision_dir / "moved-population.json")
    test_before = _bytes(tested.run_dir)
    resumed = marl_bgs.run_tournament(resume_from=tested.run_dir)
    assert resumed.matches == tested.matches
    assert _bytes(tested.run_dir) == test_before
    assert _bytes(completed_field.run_dir) == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "incomplete",
        "changed",
        "extra-entrant",
        "validation-map",
        "rehashed-members",
        "rehashed-rankings",
        "rehashed-counts",
    ],
)
def test_test_stage_requires_complete_exact_frozen_members_before_games(
    completed_field: CanonicalTournamentResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    decision_dir = tmp_path / "decision"
    decision = marl_bgs.select_initial_population(
        completed_field, output_dir=decision_dir
    )
    path = decision_dir / "population.json"
    if fault == "missing":
        path = tmp_path / "not-frozen.json"
    elif fault == "incomplete":
        changed = _json(path)
        changed["status"] = "incomplete"
        path.write_text(json.dumps(changed))
    elif fault == "changed":
        changed = _json(path)
        changed["members"][0]["controller_id"] = "different-controller"
        path.write_text(json.dumps(changed))
    elif fault.startswith("rehashed-"):
        changed = _json(path)
        if fault == "rehashed-members":
            changed["members"][0] = next(
                row for row in changed["candidates"] if row["name"] == "random"
            )
        elif fault == "rehashed-rankings":
            changed["rankings"].pop()
        else:
            changed["completed_games"] -= 1
        changed["population_id"] = sha256(
            canonical_json(
                {key: value for key, value in changed.items() if key != "population_id"}
            )
        ).hexdigest()
        path.write_text(json.dumps(changed))
    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "_evaluate_tournament_episodes",
        _unexpected,
    )
    names = [row["name"] for row in decision["members"]]
    if fault == "extra-entrant":
        names = list(_ENTRANTS)
    output = tmp_path / "invalid-test"
    maps = [42] if fault == "validation-map" else [47]
    with pytest.raises((ValueError, OSError)):
        marl_bgs.run_tournament(
            config={
                "entrants": names,
                "maps": maps,
                "games_per_opponent": 2,
                "max_steps": 1,
                "selection": {"stage": "test", "population": str(path)},
            },
            output_dir=output,
        )
    assert not output.exists() or list(output.iterdir()) == []


def test_unequal_opponent_weights_cannot_change_the_declared_selection_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "_evaluate_tournament_episodes",
        _unexpected,
    )
    config = _field(selection=_declaration())
    config["opponent_weights"] = dict(zip(_ENTRANTS, [1.0, 2.0, 1.0], strict=True))
    output = tmp_path / "weighted-field"
    with pytest.raises(ValueError, match="equal opponent weights"):
        marl_bgs.run_tournament(config=config, output_dir=output)
    assert not output.exists() or list(output.iterdir()) == []


@pytest.mark.parametrize(
    "stage,actual_map,valid_map", [("validation", 47, 42), ("test", 42, 47)]
)
def test_full_descriptor_cannot_relabel_unregistered_source_for_selection(
    completed_field: CanonicalTournamentResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    actual_map: int,
    valid_map: int,
) -> None:
    from hashlib import sha256

    from marl_battlegrounds.evaluation.population_selection import bind_selection
    from marl_battlegrounds.evaluation.tournament_config import (
        canonical_json,
        load_tournament_config,
        snapshot_identity,
    )
    from marl_battlegrounds.evaluation.tournament_inputs import (
        prepare_tournament_inputs,
    )

    entrants = list(_ENTRANTS)
    request: dict[str, Any] = _declaration()
    if stage == "test":
        decision = marl_bgs.select_initial_population(
            completed_field, output_dir=tmp_path / "population"
        )
        entrants = [row["controller"]["reference"] for row in decision["members"]]
        request = {
            "stage": "test",
            "population": str(tmp_path / "population/population.json"),
        }
    valid, _, _ = prepare_tournament_inputs(
        entrants, maps=[valid_map], games_per_opponent=2, max_steps=1
    )
    changed, _, _ = prepare_tournament_inputs(
        entrants, maps=[actual_map], games_per_opponent=2, max_steps=1
    )
    bound = bind_selection(request, valid)
    source = changed["conditions"]["map_sources"][0]
    assert source["registered_map"]["map_id"] == actual_map
    source.update(split=stage, registered_map=None)
    changed["snapshot_id"] = snapshot_identity(changed)
    # Ordinary custom fields still permit unregistered source metadata.
    load_tournament_config(changed)
    bound["field_id"] = snapshot_identity(changed)
    bound["selection_id"] = sha256(
        canonical_json(
            {key: value for key, value in bound.items() if key != "selection_id"}
        )
    ).hexdigest()
    changed["selection"] = bound
    changed["snapshot_id"] = snapshot_identity(changed)
    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.canonical"),
        "_active_pair",
        _unexpected,
    )
    output = tmp_path / "forbidden-stage"
    with pytest.raises(ValueError, match="registered maps"):
        marl_bgs.run_tournament(config=changed, output_dir=output)
    assert not output.exists()
