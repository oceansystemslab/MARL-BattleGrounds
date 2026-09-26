"""Check fresh tournament normalization without playing or inventing games.

Real method registrations, map content and source identities become checked
inline metadata. Tests preserve ordinary schedule IDs, paired roots, reference
loading count, immutable numerical bindings, live-System stream boundaries and
honest unknown evidence. Short declarations retain caller values and path bases.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from marl_battlegrounds.evaluation import tournament_inputs as inputs
from marl_battlegrounds.evaluation.evaluation_conditions import OMITTED, Omitted
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.recording_identity import policy_description
from marl_battlegrounds.evaluation.tournament_assets import (
    AssetVerifier,
    controller_content_identity,
    verify_tournament_environment,
)
from marl_battlegrounds.evaluation.tournament_config import (
    CONFIG_FORMAT,
    canonical_json,
)
from marl_battlegrounds.evaluation.tournament_headlines import content_digest
from marl_battlegrounds.evaluation.tournament_reuse import ReusePlan, resolve_reuse_plan
from marl_battlegrounds.evaluation.tournament_schedule import build_tournament_schedule


@pytest.mark.parametrize(
    "first,second,expected",
    ((OMITTED, OMITTED, OMITTED), (4, OMITTED, 4), (OMITTED, 4, 4), (4, 4, 4)),
)
def test_budget_aliases_preserve_omission_and_equal_values(
    first: int | Omitted, second: int | Omitted, expected: int | Omitted
) -> None:
    result = inputs.resolve_tournament_budget(first, second)
    assert (
        isinstance(result, Omitted)
        if isinstance(expected, Omitted)
        else result == expected
    )


@pytest.mark.parametrize(
    "first,second",
    ((4, 6), (True, OMITTED), (0, OMITTED), (4.0, OMITTED), (OMITTED, None)),
)
def test_invalid_budget_aliases_fail(first: object, second: object) -> None:
    with pytest.raises(ValueError):
        inputs.resolve_tournament_budget(cast(int, first), cast(int, second))


def test_short_declaration_resolves_paths_and_keeps_factories_unopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actors = tmp_path / "actor:folder"
    actors.mkdir()
    (tmp_path / "random").mkdir()
    source = {
        "entrants": ["actor:folder", "random", "not_imported:make"],
        "output_dir": "results",
        "maps": [47],
        "games_per_opponent": 2,
    }
    path = tmp_path / "field.json"
    path.write_text(json.dumps(source))
    monkeypatch.chdir(tmp_path.parent)
    result = inputs.read_short_tournament_config(path)
    assert result is not None
    raw, base = result
    assert raw["entrants"] == [str(actors), "random", "not_imported:make"]
    assert raw["output_dir"] == str(tmp_path / "results")
    assert base == tmp_path
    assert json.loads(path.read_text()) == source


def test_mapping_inputs_are_copied_and_full_descriptors_are_left_to_their_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = {"entrants": ["random", "tdm-alpha"], "maps": [47]}
    result = inputs.read_short_tournament_config(source)
    assert result is not None
    raw, base = result
    raw["maps"].append(48)
    assert source["maps"] == [47]
    assert base == tmp_path
    assert inputs.read_short_tournament_config({"format": CONFIG_FORMAT}) is None


@pytest.mark.parametrize(
    "extra",
    (
        {"num_envs": 32},
        {"chunk_size": 16},
        {"typo": 1},
        {"entrants": "random,tdm-alpha"},
        {"entrants": [None]},
    ),
)
def test_short_declaration_rejects_unsupported_fields_or_entrant_shapes(
    extra: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        inputs.read_short_tournament_config(
            {"entrants": ["random", "tdm-alpha"], **extra}
        )


def test_native_schedule_keeps_original_games_and_honest_registration() -> None:
    descriptor, bindings, metadata = inputs.prepare_tournament_inputs(
        ["tdm-alpha", "random"],
        maps=[48, 47],
        games_per_opponent=4,
        seed=97,
        max_steps=2,
    )
    verifier = AssetVerifier(descriptor)
    verify_tournament_environment(descriptor, verifier, execution=True)
    plan = resolve_reuse_plan(descriptor, {}, asset_reader=verifier.read_json)
    expected = build_tournament_schedule(
        ["tdm-alpha", "random"], [48, 47], episodes_per_pair=4
    )
    names = {row["entrant_id"]: row["name"] for row in descriptor["participants"]}
    assert len(plan.games) == len(expected) == 4
    for game, original in zip(plan.games, expected, strict=True):
        assert (
            game["logical_game_id"],
            game["pair_id"],
            game["execution"]["episode_id"],
            game["execution"]["seed_id"],
            game["map_id"],
            game["spawn_locations"],
        ) == (
            original.episode_id,
            original.block_id,
            original.episode_id,
            original.seed_id,
            original.map_id,
            original.spawn_locations,
        )
        assert names[game["team_a"]] == original.team_a
        assert names[game["team_b"]] == original.team_b
        assert game["execution"]["root_seed"] == 97
        assert game["execution"]["action_stream_version"] == "episode-fold-in-v1"
        assert game["origin"] is None and game["prior_origin"] is None
    assert metadata["seeds"] == [97]
    assert metadata["schedule_digest"] == content_digest(metadata["schedule"])
    assert metadata["input_references"] == {
        "tdm-alpha": "tdm-alpha",
        "random": "random",
    }
    assert metadata["policies"] == [
        policy_description(
            cast(Policy, method),
            method.variables,
            cast(Policy, method).initial_carry,
            include_digests=True,
        )
        for method in sorted(bindings.values(), key=lambda method: method.name)
    ]
    assert all("inline" in asset for asset in descriptor["assets"].values())
    for participant in descriptor["participants"]:
        _, known = controller_content_identity(participant["controller"], verifier)
        assert not known
        assert all(
            value is None for value in participant["controller"]["content"].values()
        )
    assert canonical_json(descriptor)


def test_factory_is_loaded_once_and_frozen_values_are_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = replace(
        policy("random"),
        name="research",
        variables={"weight": np.asarray([1.0], dtype=np.float32)},
    )
    calls: list[str] = []
    freezes: list[str] = []
    load = inputs.load_method
    freeze = inputs.freeze_evaluation_method

    def loaded(reference: str) -> System | Policy | str:
        calls.append(reference)
        return original if reference == "research:make" else load(reference)

    def frozen(method: System | Policy | str) -> System | Policy:
        result = freeze(method)
        freezes.append(result.name)
        return result

    monkeypatch.setattr(inputs, "load_method", loaded)
    monkeypatch.setattr(inputs, "freeze_evaluation_method", frozen)
    descriptor, bindings, _ = inputs.prepare_tournament_inputs(
        ["research:make", "tdm-alpha"], maps=[47], games_per_opponent=2, max_steps=2
    )
    assert calls == ["research:make", "tdm-alpha"]
    assert freezes == ["research", "tdm-alpha"]
    participant = next(
        row for row in descriptor["participants"] if row["name"] == "research"
    )
    resident = bindings[participant["entrant_id"]]
    assert resident is not original
    np.asarray(original.variables["weight"])[0] = 9.0
    np.testing.assert_array_equal(resident.variables["weight"], [1.0])
    assert participant["controller"]["reference"] == "research:make"


def test_explicit_roots_split_each_map_budget_without_more_games() -> None:
    descriptor, _, metadata = inputs.prepare_tournament_inputs(
        ["random", "tdm-alpha"],
        maps=[47, 48],
        games_per_opponent=8,
        seeds=[7, 19],
        max_steps=2,
    )
    verifier = AssetVerifier(descriptor)
    plan = resolve_reuse_plan(descriptor, {}, asset_reader=verifier.read_json)
    assert len(plan.games) == 8
    counts = Counter(
        (row["map_id"], row["execution"]["root_seed"]) for row in plan.games
    )
    assert counts == {(47, 7): 2, (47, 19): 2, (48, 7): 2, (48, 19): 2}
    assert metadata["seeds"] == [7, 19]
    assert all(
        row["extension"] == "none" for row in plan.schedule_manifest["execution_groups"]
    )


def test_live_system_pairs_and_challenger_keep_the_right_streams() -> None:
    live = replace(shared_policy(policy("random")), name="live-system")
    challenger = replace(shared_policy(policy("random")), name="challenger")
    descriptor, bindings, metadata = inputs.prepare_tournament_inputs(
        [live, "tdm-alpha", "tdm-beta"],
        maps=[47],
        games_per_opponent=2,
        challenger=challenger,
        max_steps=2,
    )
    verifier = AssetVerifier(descriptor)
    kinds = {
        identifier: "policy" if isinstance(method, Policy) else "system"
        for identifier, method in bindings.items()
    }
    kinds["challenger-id"] = "system"
    plan = resolve_reuse_plan(
        descriptor,
        {},
        asset_reader=verifier.read_json,
        challenger_id="challenger-id",
        participant_kinds=kinds,
    )
    names = {row["entrant_id"]: row["name"] for row in descriptor["participants"]}
    live_id = next(
        identifier
        for identifier, method in bindings.items()
        if method.name == "live-system"
    )
    live_row = next(
        row for row in descriptor["participants"] if row["entrant_id"] == live_id
    )
    assert live_row["controller"]["factory"] == "unknown:requires_supplied_system"
    assert metadata["input_references"]["live-system"] is None
    assert len(plan.games) == 12
    for game in plan.games:
        is_challenger = game["team_a"] == "challenger-id"
        system_pair = is_challenger or "live-system" in (
            names[game["team_a"]],
            names[game["team_b"]],
        )
        expected = "evaluation-systems-v1" if system_pair else "episode-fold-in-v1"
        assert game["execution"]["action_stream_version"] == expected
        if is_challenger:
            assert game["logical_game_id"] > 6
            assert game["execution"]["seed_id"] > 3
            assert game["team_b"] in names


@pytest.mark.parametrize(
    "kwargs",
    (
        {"seeds": [1, 1]},
        {"seeds": [True]},
        {"seeds": []},
        {"seeds": [1, 2], "games_per_opponent": 2},
        {"maps": [47, 47]},
        {"seed": -1},
        {"opponent_weights": {"random": 1.0}},
    ),
)
def test_invalid_schedule_inputs_fail_without_output(
    kwargs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    arguments: dict[str, Any] = {
        "maps": [47],
        "games_per_opponent": 2,
        "max_steps": 2,
        **kwargs,
    }
    with pytest.raises(ValueError):
        inputs.prepare_tournament_inputs(["random", "tdm-alpha"], **arguments)
    assert not list(tmp_path.iterdir())


def test_numpy_integer_roots_keep_the_supported_python_boundary() -> None:
    descriptor, _, metadata = inputs.prepare_tournament_inputs(
        ["random", "tdm-alpha"],
        maps=[47],
        games_per_opponent=4,
        seed=cast(int, np.uint32(97)),
        seeds=cast(list[int], [np.uint32(7), np.int64(19)]),
        max_steps=2,
    )
    assert metadata["seed"] == 97 and type(metadata["seed"]) is int
    assert metadata["seeds"] == [7, 19]
    assert all(type(value) is int for value in metadata["seeds"])
    assert descriptor["analysis"]["bootstrap_seed"] == 97
    assert canonical_json(descriptor)


def test_rosters_and_red_zone_rules_use_actual_configurations() -> None:
    rosters = {"team_a": ["mage"], "team_b": ["priest", "warrior"]}
    descriptor, _, metadata = inputs.prepare_tournament_inputs(
        ["random", "tdm-alpha"],
        maps=[47],
        games_per_opponent=2,
        rosters=cast(Any, rosters),
        score_threshold=3,
        max_steps=2,
        red_zone_depth=0.0,
    )
    assert descriptor["conditions"]["rosters"] == rosters
    assert metadata["rosters"] == rosters
    assert metadata["red_zone_depth"] == 0.0
    source = descriptor["conditions"]["map_sources"][0]
    content = AssetVerifier(descriptor).read_json(source["source_config_asset"])
    assert isinstance(content, dict)
    assert content["team_deathmatch_red_zone_depth"] == 0.0


def test_one_saved_template_binds_policy_or_system_without_changing_coordinates() -> (
    None
):
    descriptor, bindings, _ = inputs.prepare_tournament_inputs(
        ["random", "tdm-alpha"], maps=[47], games_per_opponent=2, max_steps=2
    )
    before = canonical_json(descriptor)
    verifier = AssetVerifier(descriptor)
    kinds = {identifier: "policy" for identifier in bindings}
    plans: list[ReusePlan] = []
    for kind in ("policy", "system"):
        plan = resolve_reuse_plan(
            descriptor,
            {},
            asset_reader=verifier.read_json,
            challenger_id="challenger",
            participant_kinds={**kinds, "challenger": kind},
        )
        repeated = resolve_reuse_plan(
            descriptor,
            {},
            asset_reader=verifier.read_json,
            challenger_id="challenger",
            participant_kinds={**kinds, "challenger": kind},
        )
        assert plan == repeated
        incumbent_groups = {
            game["execution"]["group_id"]
            for game in plan.games
            if game["team_a"] != "challenger"
        }
        companion_groups = {
            game["execution"]["group_id"]
            for game in plan.games
            if game["team_a"] == "challenger"
        }
        assert not incumbent_groups & companion_groups
        for game in plan.games:
            expected = (
                "evaluation-systems-v1"
                if game["team_a"] == "challenger" and kind == "system"
                else "episode-fold-in-v1"
            )
            assert game["execution"]["action_stream_version"] == expected
            assert game["execution"]["initialization_stream_version"] == expected
        plans.append(plan)
    assert plans[0].games[:2] == plans[1].games[:2]
    for left, right in zip(plans[0].games[2:], plans[1].games[2:], strict=True):
        for key in left.keys() - {"execution"}:
            assert left[key] == right[key]
        for key in left["execution"].keys() - {
            "action_stream_version",
            "initialization_stream_version",
        }:
            assert left["execution"][key] == right["execution"][key]
    assert canonical_json(descriptor) == before


@pytest.mark.parametrize("kind", (None, "unknown"))
def test_deferred_companion_requires_checked_actual_kinds(kind: str | None) -> None:
    descriptor, bindings, _ = inputs.prepare_tournament_inputs(
        ["random", "tdm-alpha"], maps=[47], games_per_opponent=2, max_steps=2
    )
    verifier = AssetVerifier(descriptor)
    kinds = (
        None
        if kind is None
        else {**dict.fromkeys(bindings, "policy"), "challenger": kind}
    )
    with pytest.raises(ValueError, match="checked Policy/System kinds"):
        resolve_reuse_plan(
            descriptor,
            {},
            asset_reader=verifier.read_json,
            challenger_id="challenger",
            participant_kinds=kinds,
        )


def test_unmarked_template_keeps_its_fixed_stream_contract() -> None:
    descriptor, bindings, _ = inputs.prepare_tournament_inputs(
        ["random", "tdm-alpha"], maps=[47], games_per_opponent=2, max_steps=2
    )
    manifest = descriptor["assets"]["schedule"]["inline"]
    manifest.pop("companion_stream_rule")
    inputs._inline(descriptor["assets"], "schedule", manifest, "schedule")
    verifier = AssetVerifier(descriptor)
    plan = resolve_reuse_plan(
        descriptor,
        {},
        asset_reader=verifier.read_json,
        challenger_id="challenger",
        participant_kinds={**dict.fromkeys(bindings, "policy"), "challenger": "system"},
    )
    assert all(
        game["execution"]["action_stream_version"] == "episode-fold-in-v1"
        for game in plan.games
    )


def test_deferred_companion_cannot_change_an_incumbent_execution_group() -> None:
    descriptor, bindings, _ = inputs.prepare_tournament_inputs(
        ["random", "tdm-alpha"], maps=[47], games_per_opponent=2, max_steps=2
    )
    incumbent_group = descriptor["assets"]["games"]["inline"][0]["execution"][
        "group_id"
    ]
    companion = descriptor["assets"]["challenger-games"]["inline"]
    for game in companion:
        game["execution"]["group_id"] = incumbent_group
    inputs._inline(descriptor["assets"], "challenger-games", companion, "schedule")
    verifier = AssetVerifier(descriptor)
    with pytest.raises(ValueError, match="separate declared groups"):
        resolve_reuse_plan(
            descriptor,
            {},
            asset_reader=verifier.read_json,
            challenger_id="challenger",
            participant_kinds={
                **dict.fromkeys(bindings, "policy"),
                "challenger": "system",
            },
        )


@pytest.mark.parametrize(
    "field,declared,supplied",
    (
        ("maps", [47], (48,)),
        ("maps", [47], ()),
        ("full_metrics_episodes", [1], (2,)),
        ("replay_episodes", [1], (2,)),
    ),
)
def test_short_explicit_selection_conflicts_fail_before_loading_or_output(
    field: str,
    declared: list[int],
    supplied: tuple[int, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.evaluation.tournament import run_tournament

    def forbidden(reference: str) -> None:
        raise AssertionError("Conflicting settings must fail before loading")

    monkeypatch.setattr(inputs, "load_method", forbidden)
    config = {
        "entrants": ["random", "tdm-alpha"],
        "maps": [47],
        "games_per_opponent": 2,
        "max_steps": 1,
        field: declared,
    }
    explicit: dict[str, Any] = {field: supplied}
    with pytest.raises(ValueError, match=field + " conflicts"):
        run_tournament(config=config, output_dir=tmp_path / "runs", **explicit)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["full_metrics_episodes", "replay_episodes"])
@pytest.mark.parametrize("source", ["explicit", "config"])
@pytest.mark.parametrize("invalid", [True, 1.5, "1", 0, 2**31])
def test_short_selection_invalid_ids_fail_before_loading_or_output(
    field: str,
    source: str,
    invalid: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.evaluation.tournament import run_tournament

    def forbidden(reference: str) -> None:
        raise AssertionError("Invalid captures must fail before loading")

    monkeypatch.setattr(inputs, "load_method", forbidden)
    config: dict[str, Any] = {
        "entrants": ["random", "tdm-alpha"],
        "maps": [47],
        "games_per_opponent": 2,
        "max_steps": 1,
        field: [1],
    }
    explicit: dict[str, Any] = {}
    if source == "config":
        config[field] = [invalid]
    else:
        explicit[field] = (value for value in [invalid])
    with pytest.raises(ValueError, match=field + " must contain positive int32"):
        run_tournament(config=config, output_dir=tmp_path / "runs", **explicit)
    assert not list(tmp_path.iterdir())


def test_short_matching_map_id_preserves_stale_description_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation.tournament import run_tournament
    from marl_battlegrounds.tasks import list_tdm_maps

    def forbidden(reference: str) -> None:
        raise AssertionError("A stale map must fail before loading")

    monkeypatch.setattr(inputs, "load_method", forbidden)
    stale = list_tdm_maps()[47].model_copy(update={"resource_sha256": "0" * 64})
    config = {
        "entrants": ["random", "tdm-alpha"],
        "maps": [47],
        "games_per_opponent": 2,
        "max_steps": 1,
    }
    with pytest.raises(
        ValueError, match=r"choose the map again with list_tdm_maps\(\)"
    ):
        run_tournament(config=config, maps=(stale,), output_dir=tmp_path / "runs")
    assert not list(tmp_path.iterdir())
