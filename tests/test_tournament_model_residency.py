"""Check one loaded instance per entrant without taking provider ownership.

Configured and public tournaments reuse reference-loaded methods across pairs,
keep distinct host entrants separate and leave caller objects in their context.
Input preparation copies numerical values once, keeps opaque objects intact and
passes through loader errors. Completed resume does not reload played entrants.
These CPU checks make no memory-reclamation or GPU-throughput claim.
"""

# pyright: reportPrivateUsage=false

import json
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import get_ident
from typing import Any, Never, cast

import jax
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from canonical_record_fixtures import build_record_bundle
from marl_battlegrounds import _cli
from marl_battlegrounds.evaluation import canonical, tournament_inputs
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    policy,
)
from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier
from marl_battlegrounds.policies.actor import ActorAction


@dataclass
class Model:
    name: str
    closes: int = 0

    def close(self) -> None:
        self.closes += 1


def _event(phase: str) -> None:
    path = os.environ.get("MARL_BG_TEST_FACTORY_LOG")
    if path:
        with Path(path).open("a") as stream:
            stream.write(
                json.dumps({"phase": phase, "pid": os.getpid(), "thread": get_ident()})
                + "\n"
            )


def _host_apply(
    variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: jax.Array
) -> SystemOutput:
    del variables, keys
    _event("apply")
    zero = np.zeros(inputs.active_mask.shape, dtype=np.int32)
    action = ActorAction(*(cast(jax.Array, zero) for _ in range(3)))
    return SystemOutput(action, memory)


def make_host() -> System:
    _event("construct")
    return System("local-host", _host_apply, execution="host")


def test_canonical_borrows_exact_objects_and_does_not_merge_unknown_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    borrowed = cast(System, Model("borrowed"))
    participants = {
        name: {
            "name": name,
            "controller_id": "unknown",
            "controller": {"kind": "factory", "factory": "same:factory"},
        }
        for name in ("borrowed", "first", "second")
    }
    loads: list[str] = []
    loaded: list[Model] = []

    def load(participant: Mapping[str, Any], _: AssetVerifier) -> System:
        name = participant["name"]
        loads.append(name)
        model = Model(name)
        loaded.append(model)
        return cast(System, model)

    monkeypatch.setattr(canonical, "load_tournament_controller", load)
    verifier = AssetVerifier({"assets": {}})
    cache: dict[tuple[str, bytes, str], System | Policy] = {}
    for other in ("first", "second", "first"):
        with canonical._active_pair(
            "borrowed",
            other,
            participants,
            verifier,
            None,
            None,
            {"borrowed": borrowed},
            residents=cache,
        ) as pair:
            assert pair[0] is borrowed
            assert pair[1].name == other
    with canonical._active_pair(
        "first", "second", participants, verifier, None, None, residents=cache
    ) as pair:
        assert pair[0] is not pair[1]
    assert loads == ["first", "second"]
    assert all(model.closes == 0 for model in [cast(Model, borrowed), *loaded])


def test_failed_pair_loading_keeps_previous_model_and_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participants = {name: {"controller": {"name": name}} for name in ("a", "b")}
    first = cast(System, Model("a"))
    calls: list[str] = []
    failure = RuntimeError("Provider could not load")

    def load(participant: Mapping[str, Any], _: AssetVerifier) -> System:
        name = participant["controller"]["name"]
        calls.append(name)
        if name == "b":
            raise failure
        return first

    monkeypatch.setattr(canonical, "load_tournament_controller", load)
    cache: dict[tuple[str, bytes, str], System | Policy] = {}
    with (
        pytest.raises(RuntimeError) as raised,
        canonical._active_pair(
            "a",
            "b",
            participants,
            AssetVerifier({"assets": {}}),
            None,
            None,
            residents=cache,
        ),
    ):
        pytest.fail("A failed load returned a pair")
    assert raised.value is failure
    assert calls == ["a", "b"]
    assert list(cache.values()) == [first]
    assert cast(Model, first).closes == 0


def test_configured_tournament_builds_each_factory_once_for_all_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_record_bundle(tmp_path / "inputs", entrants=3, maps=1, max_steps=1)
    loads: list[str] = []
    loader = canonical.load_tournament_controller

    def counted_load(
        participant: Mapping[str, Any], verifier: AssetVerifier
    ) -> System | Policy:
        loads.append(participant["entrant_id"])
        return loader(participant, verifier)

    monkeypatch.setattr(canonical, "load_tournament_controller", counted_load)
    result = canonical._run_resolved_tournament(
        bundle["config"],
        rerun_existing=True,
        num_envs=2,
        chunk_size=1,
        output_dir=tmp_path / "out",
    )
    expected = [row["entrant_id"] for row in bundle["config"]["participants"]]
    assert result.status == "complete"
    assert result.executed_games == result.metadata["executed_this_call"] == 6
    assert Counter(loads) == Counter(expected)
    assert {row["episode_id"] for row in result.matches} == {
        game["execution"]["episode_id"] for game in bundle["games"]
    }
    resumed = canonical._run_resolved_tournament(
        bundle["config"], rerun_existing=True, resume_from=result.run_dir
    )
    assert resumed.metadata["executed_this_call"] == 0
    assert resumed.matches == result.matches
    assert Counter(loads) == Counter(expected)


def test_public_host_factory_is_reused_on_the_calling_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = tmp_path / "factory-events.jsonl"
    monkeypatch.setenv("MARL_BG_TEST_FACTORY_LOG", str(events))
    result = marl_bgs.run_tournament(
        ["tests.test_tournament_model_residency:make_host", "random", "tdm-alpha"],
        maps=[12],
        games_per_opponent=2,
        max_steps=2,
        num_envs=2,
        chunk_size=1,
        replay_episodes=(1, 2),
        output_dir=tmp_path / "runs",
    )
    records = [json.loads(line) for line in events.read_text().splitlines()]
    assert sum(row["phase"] == "construct" for row in records) == 1
    assert any(row["phase"] == "apply" for row in records)
    assert {(row["pid"], row["thread"]) for row in records} == {
        (os.getpid(), get_ident())
    }
    assert len(result.matches) == 6
    assert result.run_dir is not None
    assert len(list((result.run_dir / "replays").glob("*.json"))) == 2


@pytest.mark.parametrize("command", ["tournament", "canonical"])
def test_cli_passes_unloaded_references_and_fixed_default(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs: object) -> None:
        calls.append(kwargs)

    def forbidden(value: object) -> None:
        raise AssertionError(f"Parser loaded a method: {value}")

    def output(result: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr(
        marl_bgs,
        "run_tournament" if command == "tournament" else "run_canonical_tournament",
        run,
    )
    monkeypatch.setattr(_cli, "_load_method", forbidden)
    monkeypatch.setattr(_cli, "_print_result", output)
    reference = (
        ["--entrants", "trusted:factory,random"]
        if command == "tournament"
        else ["--system", "trusted:factory"]
    )
    assert _cli.main([command, *reference]) == 0
    assert len(calls) == 1
    assert calls[0]["num_envs"] == 128
    assert (
        calls[0]["policies"] == ("trusted:factory", "random")
        if command == "tournament"
        else calls[0]["system"] == "trusted:factory"
    )


def test_live_host_opaque_state_is_not_loaded_moved_or_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Provider:
        def close(self) -> None:
            calls.append("close")

        def __reduce__(self) -> Never:
            raise AssertionError("Opaque state must not be serialized")

    provider = Provider()

    def apply(*args: object) -> Never:
        calls.append("apply")
        raise AssertionError("Preparation must not choose actions")

    def init(*args: object) -> Never:
        calls.append("init")
        raise AssertionError("Preparation must not initialize memory")

    original = System(
        "host", apply, variables={"provider": provider}, init=init, execution="host"
    )

    def load(reference: str) -> Policy:
        raise AssertionError(f"Live method must not be loaded: {reference}")

    monkeypatch.setattr(tournament_inputs, "load_method", load)
    frozen = tournament_inputs._prepare_tournament_method(original)
    assert isinstance(frozen, System)
    assert frozen.variables["provider"] is provider
    assert frozen.apply is original.apply
    assert frozen.init is original.init
    assert calls == []


def test_preparation_keeps_a_detached_numerical_snapshot() -> None:
    values = np.asarray([1.0, 2.0], dtype=np.float32)
    original = replace(policy("random"), variables={"weights": values})
    frozen = tournament_inputs._prepare_tournament_method(original)
    values[0] = 9.0
    np.testing.assert_array_equal(frozen.variables["weights"], [1.0, 2.0])


def test_reference_loader_failure_keeps_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("Provider could not load")

    def load(reference: str) -> Policy:
        raise failure

    monkeypatch.setattr(tournament_inputs, "load_method", load)
    with pytest.raises(RuntimeError) as raised:
        tournament_inputs._prepare_tournament_method("research:make")
    assert raised.value is failure


@pytest.mark.parametrize("route", ["list", "config", "canonical"])
@pytest.mark.parametrize("by_reference", [False, True])
@pytest.mark.parametrize("method_kind", ["system", "policy"])
def test_public_challenger_copies_host_parameters_once(
    monkeypatch: pytest.MonkeyPatch, route: str, by_reference: bool, method_kind: str
) -> None:
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation import canonical

    config, _, _ = tournament_inputs.prepare_tournament_inputs(
        ["random", "tdm-alpha"], maps=[12], games_per_opponent=2, max_steps=1
    )
    values = np.arange(8, dtype=np.float32)

    def apply(*args: object) -> Never:
        raise AssertionError("This preparation check must not play games")

    live = (
        System("live", apply, variables={"weights": values}, execution="host")
        if method_kind == "system"
        else replace(
            policy("random"),
            name="live",
            apply=apply,
            variables={"weights": values},
            execution="host",
        )
    )
    snapshots: list[System | Policy] = []
    freeze = tournament_inputs.freeze_evaluation_method

    def tracked(method: System | Policy) -> System | Policy:
        result = freeze(method)
        if method.name == "live":
            snapshots.append(result)
        return result

    class ReachedIdentityError(Exception):
        pass

    def identity(method: System | Policy) -> Never:
        assert method is snapshots[0]
        assert len(snapshots) == 1
        assert not np.shares_memory(method.variables["weights"], values)
        np.testing.assert_array_equal(method.variables["weights"], values)
        raise ReachedIdentityError

    def load(reference: str) -> System | Policy:
        return live if reference == "research:challenger" else policy(reference)

    monkeypatch.setattr(tournament_inputs, "load_method", load)
    monkeypatch.setattr(tournament_inputs, "freeze_evaluation_method", tracked)
    monkeypatch.setattr(canonical, "_local_challenger", identity)
    challenger = "research:challenger" if by_reference else live
    with pytest.raises(ReachedIdentityError):
        if route == "list":
            marl_bgs.run_tournament(
                ["random", "tdm-alpha"],
                challenger=challenger,
                maps=[12],
                games_per_opponent=2,
                max_steps=1,
            )
        elif route == "config":
            marl_bgs.run_tournament(config=config, challenger=challenger)
        else:

            def selected(*args: object, **kwargs: object) -> dict[str, object]:
                return config

            monkeypatch.setattr(canonical, "resolve_tournament_config", selected)
            marl_bgs.run_canonical_tournament(challenger, config=config)
    assert len(snapshots) == 1
