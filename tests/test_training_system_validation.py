"""Check any-System validation, retained host clients and exact panel selection.

Built-ins, independent policies, mixtures, exported actors, JAX methods, host
methods and import factories share M8 registrations. A local text-JSON client
stands in for a language-model service. The real short-horizon CPU evaluator
proves 32-lane padded execution, complete saved rows and no extra provider calls
after a task completes. No network client is used. Existing tests keep the old
panel/task hashes and selection covered. A PQN-VDN export is validated against
a panel holding a host-execution System; its summary names the method and its
optimizer count. A panel chosen from a ranking saves the ranking's Red Zone
depth (read from its recorded configurations, 5.0 or 6.0) with evidence schema
2. load_panel keeps that ranking evidence while admitting actors for validation
at a separately declared depth. Setup-order tests supply the source horizon
without preparing full content, then stop at learner initialization. They check
frozen identities before allocation and no output before roster preflight.
Continuation binding tests complete real short CPU training and one-step M8
validation, then resume with retained clients. They omit inherited candidate
discovery; separate H300 tests verify inherited games and parent ranking. Their
runtime identity and recording provenance stay fixed while other packet files
change, so these checks do not qualify source compatibility.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.baselines.ppo import initialize_ppo
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    SystemInput,
    independent_policies,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
    policy_description,
)
from marl_battlegrounds.evaluation.results import TournamentResult
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput
from marl_battlegrounds.training import (
    _content,
    analysis,
    checkpoints,
    runner,
    validation,
)


def _no_content(**kwargs: object) -> SimpleNamespace:
    # History spacing reads the horizon before the learner setup sentinel.
    del kwargs
    return SimpleNamespace(source_configs=SimpleNamespace(max_steps=(300,)))


def _jax_apply(
    variables: object, memory: object, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, object]:
    del variables, keys
    zero = jnp.zeros((inputs.valid.shape[0], 5), jnp.int32)
    return ActorAction(zero, zero, zero), memory


class Client:
    def __init__(self) -> None:
        self.calls: list[np.ndarray[Any, Any]] = []

    def apply(
        self, variables: object, memory: object, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, object]:
        del variables, keys
        self.calls.append(np.asarray(inputs.valid).copy())
        zero = jnp.zeros((inputs.valid.shape[0], 5), jnp.int32)
        return ActorAction(zero, zero, zero), memory


def _mixture_policy(
    variables: object,
    memory: Array,
    actor: ActorInput,
    mask: ActionMask,
    key: Array,
) -> tuple[ActorAction, Array]:
    del variables
    sampled, _ = policy("random").apply((), (), actor, mask, key)
    moving = (actor.observation.self_ally_index + memory) % 2 == 0
    action = ActorAction(*(jnp.where(moving, head, 0) for head in sampled))
    return action, memory + 1


class FakeLanguageClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def complete(self, prompt: str) -> str:
        request = json.loads(prompt)
        self.requests.append(request)
        return json.dumps(
            {
                "move": request["legal_moves"][-1],
                "target": request["legal_target_ultimate"][0][0],
                "ultimate": request["legal_target_ultimate"][0][1],
            }
        )


class LanguageMethod:
    def __init__(self, client: FakeLanguageClient) -> None:
        self.client = client
        self.valid_counts: list[int] = []

    def apply(
        self, variables: object, memory: object, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, object]:
        del variables, keys
        valid = np.asarray(inputs.valid)
        self.valid_counts.append(int(valid.sum()))
        heads = np.zeros((3, len(valid), 5), dtype=np.int32)
        moves = np.asarray(inputs.action_mask.move_mask)
        joint = np.asarray(inputs.action_mask.select_target_use_ultimate_joint_mask)
        own = np.asarray(inputs.actors.observation.self_features)
        for lane in np.flatnonzero(valid):
            for slot in range(5):
                prompt = json.dumps(
                    {
                        "own_observation": own[lane, slot].tolist(),
                        "legal_moves": np.flatnonzero(moves[lane, slot]).tolist(),
                        "legal_target_ultimate": np.argwhere(
                            joint[lane, slot]
                        ).tolist(),
                    }
                )
                answer = json.loads(self.client.complete(prompt))
                heads[:, lane, slot] = (
                    answer["move"],
                    answer["target"],
                    answer["ultimate"],
                )
        return ActorAction(*(jnp.asarray(head) for head in heads)), memory


def _task(panel: validation.FrozenPanel, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    return validation.panel_task_description(
        checkpoint_id="checkpoint",
        actor_digest="actor",
        env_steps=32,
        panel=panel,
        purpose="routine",
        seed_pairs=1,
        **kwargs,
    )


def test_builtin_and_system_panel_uses_m8_identity_without_fake_actor_metadata(
    tmp_path: Path,
) -> None:
    panel = validation.create_panel(
        opponents=("tdm-alpha", System("Jax", _jax_apply)), output_dir=tmp_path
    )
    content = json.loads(panel.path.read_text())
    assert panel.schema_version == 2 and panel.qualified
    assert all(
        "checkpoint_id" not in row and "actor_digest" not in row
        for row in content["members"]
    )
    alpha = policy("tdm-alpha")
    expected = normalize_system_registration(
        policy_description(
            alpha, alpha.variables, alpha.initial_carry, include_digests=True
        ),
        phase="validation",
    )
    assert panel.members[0].registration_id == expected[0]
    assert panel.members[0].registration == expected[1]
    with pytest.raises(ValueError, match="live-only"):
        validation.load_panel(panel.path)
    restored = validation.load_panel(panel.path, bindings=panel.methods)
    assert restored.digest == panel.digest
    with pytest.raises(ValueError, match="frozen M8 identity"):
        validation.load_panel(panel.path, bindings=("tdm-beta", panel.methods[1]))


def test_factory_runs_once_and_retained_clients_are_not_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module("marl_battlegrounds._method_loading")
    client = Client()
    host = System("Host", client.apply, execution="host")
    calls: list[str] = []

    def factory(reference: str) -> System:
        calls.append(reference)
        return host

    monkeypatch.setattr(module, "load_method", factory)
    panel = validation.create_panel(
        opponents=("my_methods:factory",), output_dir=tmp_path
    )
    assert calls == ["my_methods:factory"]
    loaded = validation.load_panel(panel.path, bindings=panel.methods)
    assert calls == ["my_methods:factory"]
    assert loaded.methods[0].apply.__self__ is client
    assert client.calls == []


def test_roots_are_frozen_fresh_and_independent_by_opponent(tmp_path: Path) -> None:
    panel = validation.create_panel(
        opponents=("tdm-alpha", "tdm-beta"),
        output_dir=tmp_path,
        roots={"routine": 100, "confirmation": 200},
    )
    task = _task(panel)
    assert panel.roots == {"routine": 100, "initialization": 100, "confirmation": 200}
    assert task["members"][0]["root"] != task["members"][1]["root"]
    again = _task(panel)
    assert task == again
    for purpose in ("confirmation", "assessment"):
        with pytest.raises(ValueError, match="fresh root"):
            validation.panel_task_description(
                checkpoint_id="c",
                actor_digest="a",
                env_steps=32,
                panel=panel,
                purpose=purpose,
                seed_pairs=1,
                root_seed=100,
            )
    fresh = _task(panel, root_seed=300)
    assert fresh["task_id"] != task["task_id"]
    assert {row["root"] for row in fresh["members"]}.isdisjoint(
        row["root"] for row in task["members"]
    )
    with pytest.raises(ValueError, match="fresh root"):
        validation.create_panel(
            opponents=("tdm-alpha",),
            output_dir=tmp_path / "bad",
            roots={"routine": 100, "confirmation": 100},
        )


def test_native_score_then_kill_difference_and_legacy_ties() -> None:
    base = {"panel_digest": "panel", "complete": True, "purpose": "confirmation"}
    early = {
        **base,
        "checkpoint_id": "early",
        "env_steps": 32,
        "score": 0.5,
        "mean_kill_difference": -2,
    }
    later = {
        **base,
        "checkpoint_id": "later",
        "env_steps": 64,
        "score": 0.5,
        "mean_kill_difference": 3,
    }
    assert (
        analysis.select_checkpoint((early, later), rule="saved")["checkpoint_id"]
        == "early"
    )
    new = [{**row, "selection_schema_version": 2} for row in (early, later)]
    assert analysis.select_checkpoint(new, rule="saved")["checkpoint_id"] == "later"
    assert (
        analysis.select_checkpoint((new[0], {**new[1], "score": 0.49}), rule="saved")[
            "checkpoint_id"
        ]
        == "early"
    )
    rows = [
        {
            "map_id": 42,
            "seed_id": i + 10 * o,
            "opponent": opponent,
            "spawn_locations": end,
            "system_game_score": float(i),
            "team_a_score": 4 + o,
            "team_b_score": 2,
        }
        for o, opponent in enumerate(("A", "B"))
        for i in (0, 1)
        for end in (0, 1)
    ]
    result = analysis.summarize_validation(
        rows, maps=(42,), opponents=("A", "B"), seed_pairs=2, independent_opponents=True
    )
    assert result["score"] == 0.5 and result["mean_kill_difference"] == 2.5
    assert result["independent_blocks"] == 4
    assert len(result["opponents"]) == 2
    assert (
        analysis.summarize_validation(
            list(reversed(rows)),
            maps=(42,),
            opponents=("A", "B"),
            seed_pairs=2,
            independent_opponents=True,
        )
        == result
    )


def test_json_opponents_roundtrip_and_conflicting_declarations() -> None:
    config = runner.TrainConfig(validation_opponents=("tdm-alpha",))
    assert runner.config_from_dict(runner.config_to_dict(config)) == config
    with pytest.raises(ValueError, match="Declare"):
        replace(config, validation_panel="panel.json")
    for value in ("actor", ".", "..", "relative/actor"):
        with pytest.raises(ValueError, match="absolute"):
            replace(config, validation_opponents=(value,))


def test_live_host_validation_keeps_32_lanes_and_reuses_completed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = Client()
    panel = validation.create_panel(
        opponents=(System("Host", client.apply, execution="host"),),
        output_dir=tmp_path / "panel",
    )

    def identity(path: object) -> dict[str, Any]:
        del path
        return {"checkpoint_id": "checkpoint", "actor_digest": "actor", "env_steps": 32}

    def actor(path: object) -> Policy:
        del path
        return policy("random")

    monkeypatch.setattr(validation, "_artifact", identity)
    monkeypatch.setattr(checkpoints, "load_system", actor)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )

    provenance = capture_recording_provenance(num_envs=32)

    def fixed_provenance(**kwargs: object) -> dict[str, object]:
        del kwargs
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    original = evaluator.evaluate
    original_verify = evaluator._verify_evaluation

    def short(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original(*args, max_steps=1, **kwargs)

    def short_verify(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_verify(*args, max_steps=1, **kwargs)

    monkeypatch.setattr(evaluator, "evaluate", short)
    monkeypatch.setattr(evaluator, "_verify_evaluation", short_verify)
    result = validation.validate_checkpoint(
        "unused", panel, output_dir=tmp_path / "task", seed_pairs=1, num_envs=32
    )
    assert result["games"] == 10 and result["complete"]
    assert client.calls and all(
        mask.shape == (32,) and mask.any() for mask in client.calls
    )
    assert all(mask.sum() <= 10 for mask in client.calls)
    before = len(client.calls)
    assert (
        validation.validate_checkpoint(
            "unused", panel, output_dir=tmp_path / "task", seed_pairs=1, num_envs=32
        )
        == result
    )
    assert len(client.calls) == before
    from copy import deepcopy

    from marl_battlegrounds.evaluation.run_writer import RunWriter

    saved = {
        path: path.read_bytes()
        for path in (tmp_path / "task").rglob("*")
        if path.is_file()
    }

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Cached-pass verification must not open a writer")

    monkeypatch.setattr(RunWriter, "__init__", forbidden)
    for bad_actor, bad_opponent, change_root in (
        (policy("tdm-beta"), panel.methods[0], False),
        (policy("random"), policy("tdm-alpha"), False),
        (policy("random"), panel.methods[0], True),
    ):
        task = deepcopy(result)
        if change_root:
            task["members"][0]["root"] = (task["members"][0]["root"] + 1) % 2**32
        with pytest.raises(ValueError):
            validation._verify_panel_pass(
                Path(result["pass_paths"][0]),
                bad_actor,
                bad_opponent,
                task=task,
                member_index=0,
                num_envs=32,
            )
    assert all(path.read_bytes() == data for path, data in saved.items())
    assert len(client.calls) == before


def test_a_pqn_export_is_validated_against_a_host_system_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.baselines import pqn

    network = pqn.initialize_pqn(jax.random.key(7), planned_learning_blocks=1).network
    export = checkpoints.export_system(
        network,
        tmp_path / "actor",
        metadata={
            "run_id": "pqn-validation",
            "seed": 7,
            "env_steps": 40,
            "checkpoint_id": "d" * 64,
            "optimizer_steps": 2,
        },
        spawn_frame="left",
        method="pqn_vdn",
    )
    client = Client()
    panel = validation.create_panel(
        opponents=(System("Host", client.apply, execution="host"),),
        output_dir=tmp_path / "panel",
    )
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    original = evaluator.evaluate
    original_verify = evaluator._verify_evaluation

    def short(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original(*args, max_steps=1, **kwargs)

    def short_verify(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_verify(*args, max_steps=1, **kwargs)

    monkeypatch.setattr(evaluator, "evaluate", short)
    monkeypatch.setattr(evaluator, "_verify_evaluation", short_verify)
    result = validation.validate_checkpoint(
        export, panel, output_dir=tmp_path / "task", seed_pairs=1, num_envs=32
    )
    assert result["complete"] and result["games"] == 10
    assert result["method"] == "pqn_vdn" and result["optimizer_steps"] == 2
    assert (
        result["actor_digest"] == checkpoints.artifact_identity(export)["actor_digest"]
    )
    assert client.calls


@pytest.mark.parametrize("method", ("mappo", "ff_ippo"))
def test_independent_mixture_export_and_fake_language_client_complete_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )

    actor = checkpoints.export_system(
        initialize_ppo(jax.random.key(89), method=method).actor_params,
        tmp_path / "actor",
        metadata={
            "run_id": "system-validation",
            "seed": 89,
            "env_steps": 32,
            "checkpoint_id": "c" * 64,
        },
        spawn_frame="left",
        method=method,
    )
    client = FakeLanguageClient()
    language = LanguageMethod(client)
    independent = independent_policies(
        tuple(
            policy(name)
            for name in ("random", "tdm-alpha", "tdm-beta", "random", "tdm-alpha")
        )
    )
    mixture = shared_policy(
        Policy("Mixture", _mixture_policy, initial_carry=jnp.int32(0))
    )
    panel = validation.create_panel(
        opponents=(
            independent,
            mixture,
            str(actor),
            System("Language", language.apply, execution="host"),
        ),
        output_dir=tmp_path / "panel",
    )
    restored = validation.load_panel(panel.path, bindings=panel.methods)
    assert restored.digest == panel.digest
    assert restored.methods[-1].apply.__self__.client is client
    assert panel.members[2].reference == str(actor)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance(num_envs=32)

    def fixed_provenance(**kwargs: object) -> dict[str, object]:
        del kwargs
        return provenance

    original = evaluator.evaluate
    original_verify = evaluator._verify_evaluation

    def short(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original(*args, max_steps=1, **kwargs)

    def short_verify(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_verify(*args, max_steps=1, **kwargs)

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    monkeypatch.setattr(evaluator, "evaluate", short)
    monkeypatch.setattr(evaluator, "_verify_evaluation", short_verify)
    summary = validation.validate_checkpoint(
        actor, restored, output_dir=tmp_path / "task", seed_pairs=1, num_envs=32
    )
    assert summary["complete"] and summary["games"] == 40
    assert {row["opponent"] for row in summary["opponents"]} == {
        member.name for member in panel.members
    }
    assert language.valid_counts == [10]
    assert len(client.requests) == 50
    assert all(
        set(request) == {"own_observation", "legal_moves", "legal_target_ultimate"}
        for request in client.requests
    )
    assert (
        validation.validate_checkpoint(
            actor, restored, output_dir=tmp_path / "task", seed_pairs=1, num_envs=32
        )
        == summary
    )
    assert len(client.requests) == 50


def _ranking_fixture(
    panel: validation.FrozenPanel, red_zone_depth: float = 5.0
) -> TournamentResult:
    from dataclasses import asdict

    from marl_battlegrounds.evaluation.evaluate import normalize_episode_specs
    from marl_battlegrounds.evaluation.evaluation_conditions import config_record
    from marl_battlegrounds.evaluation.tournament_schedule import (
        build_tournament_schedule,
    )
    from marl_battlegrounds.tasks import _swap_spawn_banks, canonical_tournament_rosters

    names = tuple(member.name for member in panel.members)
    participants = {member.name: member.registration_id for member in panel.members}
    systems = {member.registration_id: member.registration for member in panel.members}
    roster_a, roster_b = canonical_tournament_rosters()
    specs = normalize_episode_specs(
        validation.VALIDATION_MAPS,
        5,
        roster_a,
        roster_b,
        20,
        300,
        red_zone_depth=red_zone_depth,
    )
    configurations: dict[str, Any] = {}
    sources: dict[int, tuple[str, str]] = {}
    for spec in specs:
        source_id, source = config_record(spec.env_config)
        swapped_id, swapped = config_record(_swap_spawn_banks(spec.env_config))
        configurations.update({source_id: source, swapped_id: swapped})
        assert spec.map_id is not None
        sources[spec.map_id] = source_id, swapped_id
    schedule = tuple(
        replace(
            match,
            source_config_id=sources[match.map_id][0],
            resolved_config_id=sources[match.map_id][cast(int, match.spawn_locations)],
        )
        for match in build_tournament_schedule(
            names, validation.VALIDATION_MAPS, episodes_per_pair=10
        )
    )
    passes: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for match in schedule:
        pass_id = f"{match.team_a}-{match.team_b}"
        entry = passes.setdefault(
            json.dumps(("tournament", pass_id), separators=(",", ":")),
            {
                "phase": "tournament",
                "pass_id": pass_id,
                "episodes": {},
                "details": {},
                "system_ids": {
                    "team_a": participants[match.team_a],
                    "team_b": participants[match.team_b],
                },
                "completed_episode_ids": [],
                "result_state": {"version": 1, "status": "complete", "reason": None},
            },
        )
        entry["episodes"][str(match.episode_id)] = {
            "configuration_digest": match.resolved_config_id,
            "source_config_id": match.source_config_id,
            "spawn_locations": match.spawn_locations,
            "comparison_kind": "verified_spawn_pair",
            "seed_id": match.seed_id,
            "map_id": match.map_id,
            "initial_state_digest": None,
        }
        entry["completed_episode_ids"].append(match.episode_id)
        rows.append(
            {
                "run_id": "fixture",
                "phase": "tournament",
                "pass_id": pass_id,
                "episode_id": match.episode_id,
                "seed_id": match.seed_id,
                "map_id": match.map_id,
                "block_id": match.block_id,
                "bootstrap_group": match.bootstrap_group,
                "team_a_policy": match.team_a,
                "team_b_policy": match.team_b,
                "config_id": match.resolved_config_id,
                "outcome": 3,
                "episode_length": 300,
                "team_a_score": 0,
                "team_b_score": 0,
            }
        )
    metadata = {
        "run_id": "fixture",
        "pairing_protocol": "fixed-team-spawn-v1",
        "map_ids": list(validation.VALIDATION_MAPS),
        "score_threshold": 20,
        "max_steps": 300,
        "participants": participants,
        "systems": systems,
        "configurations": configurations,
        "passes": passes,
        "configuration_ids_by_map": {
            str(key): value[0] for key, value in sources.items()
        },
        "schedule": [asdict(match) for match in schedule],
        "schedule_digest": "fixed-schedule",
    }
    ratings = tuple({"policy": name, "elo": 1200.0} for name in names)
    return TournamentResult(tuple(rows), ratings, (), (), {}, (), metadata, None)


def test_ranked_panel_checks_actual_pool_maps_pairs_and_stable_ties(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copy import deepcopy

    pool = validation.create_panel(
        opponents=("tdm-alpha", "tdm-beta"), output_dir=tmp_path / "pool"
    )
    ranking = _ranking_fixture(pool)
    selected = validation.create_panel(
        opponents=pool.methods,
        ranking=ranking,
        size=1,
        output_dir=tmp_path / "selected",
    )
    assert selected.members[0].registration_id == min(
        member.registration_id for member in pool.members
    )
    # The ranking's depth is read from its recorded configurations and saved.
    evidence = json.loads(selected.path.read_text())["ranking_evidence"]
    assert evidence["schema_version"] == 2 and evidence["red_zone_depth"] == 5.0
    assert (
        validation.load_panel(
            selected.path, bindings=selected.methods, red_zone_depth=5.0
        )
        == selected
    )
    before_panel = selected.path.read_bytes()
    assert (
        validation.load_panel(
            selected.path, bindings=selected.methods, red_zone_depth=6.0
        )
        == selected
    )
    assert selected.path.read_bytes() == before_panel

    def identity(path: object) -> dict[str, Any]:
        del path
        return {"checkpoint_id": "checkpoint", "actor_digest": "actor", "env_steps": 32}

    def actor(path: object) -> Policy:
        del path
        return policy("random")

    monkeypatch.setattr(validation, "_artifact", identity)
    monkeypatch.setattr(checkpoints, "load_system", actor)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    original_evaluate = evaluator.evaluate

    def short(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_evaluate(*args, max_steps=1, **kwargs)

    monkeypatch.setattr(evaluator, "evaluate", short)
    result = validation.validate_checkpoint(
        "unused",
        selected,
        output_dir=tmp_path / "foreign-rule-task",
        red_zone_depth=6.0,
        seed_pairs=1,
        num_envs=2,
    )
    assert result["complete"] and result["red_zone_depth"] == 6.0
    assert result["panel_digest"] == selected.digest
    assert selected.path.read_bytes() == before_panel
    deeper = validation.create_panel(
        opponents=pool.methods,
        ranking=_ranking_fixture(pool, red_zone_depth=6.0),
        size=1,
        output_dir=tmp_path / "deeper",
    )
    deeper_evidence = json.loads(deeper.path.read_text())["ranking_evidence"]
    assert deeper_evidence["red_zone_depth"] == 6.0
    assert (
        validation.load_panel(deeper.path, bindings=deeper.methods, red_zone_depth=6.0)
        == deeper
    )
    before = deepcopy(ranking.metadata)
    ranking.metadata["map_ids"] = [0]
    with pytest.raises(ValueError, match="development maps"):
        validation.create_panel(
            opponents=pool.methods,
            ranking=ranking,
            size=1,
            output_dir=tmp_path / "wrong-map",
        )
    ranking.metadata.update(before)
    ranking.metadata["participants"] = {"tdm-alpha": pool.members[0].registration_id}
    with pytest.raises(ValueError, match="entire declared pool"):
        validation.create_panel(
            opponents=pool.methods,
            ranking=ranking,
            size=1,
            output_dir=tmp_path / "incomplete",
        )


def test_live_system_panel_is_frozen_before_learner_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import learner

    class SetupReachedError(Exception):
        pass

    prepared: list[tuple[validation.FrozenPanel, dict[str, Any]]] = []
    original_prepare = validation._prepare_system_panel
    client = Client()
    values = np.asarray([1.0], dtype=np.float32)
    method = System("Live", client.apply, variables=values, execution="host")

    def prepare(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        result = original_prepare(*args, **kwargs)
        prepared.append(result)
        return result

    def begin(**kwargs: object) -> Any:  # noqa: ANN401
        del kwargs
        assert len(prepared) == 1
        panel, content = prepared[0]
        assert panel.digest == content["panel_digest"]
        np.testing.assert_array_equal(panel.methods[0].variables, [1.0])
        values[0] = 7.0
        np.testing.assert_array_equal(panel.methods[0].variables, [1.0])
        assert not (tmp_path / "run").exists()
        raise SetupReachedError

    monkeypatch.setattr(validation, "_prepare_system_panel", prepare)
    monkeypatch.setattr(learner, "init_learner", begin)
    monkeypatch.setattr(_content, "prepare_training_content", _no_content)
    config = runner.TrainConfig(purpose="demonstration")
    with pytest.raises(SetupReachedError):
        runner.train(
            config, output_dir=tmp_path / "run", validation_opponents=(method,)
        )
    panel, saved = prepared[0]
    assert saved["schema_version"] == 2 and saved["members"][0]["reference"] is None
    assert panel.methods[0].apply.__self__ is client
    assert client.calls == [] and not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="qualified frozen validation panel"):
        runner.train(config, output_dir=tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_direct_panel_checks_slot_actor_before_learner_or_writer_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.training import learner

    checked: list[Path] = []
    actor = tmp_path / "slot-actor"

    class SetupReachedError(Exception):
        pass

    def begin(**kwargs: object) -> Any:  # noqa: ANN401
        del kwargs
        assert checked == [actor]
        raise SetupReachedError

    def no_writer(*args: object, **kwargs: object) -> None:
        pytest.fail("Slot preflight must not open a writer")

    monkeypatch.setattr(learner, "init_learner", begin)
    monkeypatch.setattr(_content, "prepare_training_content", _no_content)
    monkeypatch.setattr(RunWriter, "__init__", no_writer)
    client = Client()
    method = System("Live", client.apply, execution="host")
    config = runner.TrainConfig(slot_diagnostic=True, slot_diagnostic_actor=str(actor))
    with pytest.raises(ValueError, match="frozen validation panel"):
        runner.train(config, output_dir=tmp_path / "missing-panel")
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.train(
            config,
            output_dir=tmp_path / "missing-actor",
            validation_opponents=(method,),
        )

    def identity(path: str | Path) -> dict[str, object]:
        checked.append(Path(path))
        return {"actor_digest": "checked-by-checkpoint-owner"}

    monkeypatch.setattr(checkpoints, "artifact_identity", identity)
    with pytest.raises(SetupReachedError):
        runner.train(
            config, output_dir=tmp_path / "accepted", validation_opponents=(method,)
        )
    assert checked == [actor] and client.calls == []


def test_resume_can_rebind_the_same_client_from_a_saved_config_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import learner

    class SetupReachedError(Exception):
        pass

    def begin(**kwargs: object) -> Any:  # noqa: ANN401
        del kwargs
        raise SetupReachedError

    monkeypatch.setattr(learner, "init_learner", begin)
    monkeypatch.setattr(_content, "prepare_training_content", _no_content)
    config = runner.TrainConfig(validation_opponents=("my_methods:factory",))
    client = Client()
    method = System("Live", client.apply, execution="host")
    with pytest.raises(ValueError, match="Declare validation_opponents once"):
        runner.train(
            config, output_dir=tmp_path / "new", validation_opponents=(method,)
        )
    run = tmp_path / "saved"
    panel = validation.create_panel(
        opponents=(method,), output_dir=run / "validation_panel"
    )
    before = panel.path.read_bytes()
    (run / "run_details.json").write_text("{}")
    checkpoint = run / "checkpoints" / ("c" * 64)

    def details(path: object) -> dict[str, object]:
        del path
        return {
            "kind": "learner",
            "schemas": checkpoints.checkpoint_schemas(),
            "metadata": {"config": runner.config_to_dict(config)},
            "collection": {},
        }

    def saved_config(saved: object) -> dict[str, Any]:
        del saved
        return runner.config_to_dict(config)

    monkeypatch.setattr(checkpoints, "read_checkpoint_details", details)
    monkeypatch.setattr(checkpoints, "saved_training_config", saved_config)
    with pytest.raises(ValueError, match="frozen M8 identity"):
        runner.train(resume_from=checkpoint, validation_opponents=("tdm-alpha",))
    with pytest.raises(SetupReachedError):
        runner.train(resume_from=checkpoint, validation_opponents=(method,))
    assert panel.path.read_bytes() == before and client.calls == []


def test_extension_separates_runtime_bindings_without_mutating_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = Client()
    method = System("Future", client.apply, execution="host")
    parent = System("Parent", Client().apply, execution="host")
    bindings = [method]
    future = {
        "panel": str(tmp_path / "future/panel.json"),
        "bindings": bindings,
        "env_steps": [12],
    }
    changes: dict[str, object] = {"validation": future}
    captured: dict[str, Any] = {}
    marker = object()

    def train(**kwargs: Any) -> Any:  # noqa: ANN401
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(runner, "_train", train)
    result = runner.extend_training(
        tmp_path / "parent/checkpoints/id",
        additional_env_steps=4,
        output_dir=tmp_path / "child",
        changes=changes,
        validation_opponents=(parent,),
    )
    assert result is marker
    assert captured["validation_opponents"] == (parent,)
    assert captured["extension"]["validation_bindings"] is bindings
    assert captured["extension"]["changes"] == {
        "validation": {"panel": str(tmp_path / "future/panel.json"), "env_steps": [12]}
    }
    assert changes["validation"] is future and future["bindings"] is bindings
    assert future["env_steps"] == [12] and bindings[0] is method
    assert not (tmp_path / "child").exists()
    for invalid, error, match in (
        ({"bindings": bindings}, ValueError, "requires a declared future panel"),
        ({"panel": "unused", "bindings": "bad"}, TypeError, "sequence of methods"),
    ):
        captured.clear()
        with pytest.raises(error, match=match):
            runner.extend_training(
                tmp_path / "parent/checkpoints/id",
                additional_env_steps=4,
                output_dir=tmp_path / "child",
                changes={"validation": invalid},
            )
        assert not captured


@pytest.mark.parametrize("parent_reference", (False, True), ids=("live", "factory"))
def test_live_panel_continuation_rebinds_clients_and_keeps_json_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_reference: bool,
) -> None:
    import hashlib

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.training import _selection_evidence

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    original_evaluate = evaluator.evaluate
    original_verify = evaluator._verify_evaluation

    def short(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_evaluate(*args, max_steps=1, **kwargs)

    def short_verify(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_verify(*args, max_steps=1, **kwargs)

    def no_candidates(checkpoint: Path, *, declaration: object) -> list[object]:
        return []

    # This checks client wiring with real saved one-step M8 games. The separate
    # H300 continuation test owns inherited-game verification and parent ranking.
    monkeypatch.setattr(evaluator, "evaluate", short)
    monkeypatch.setattr(evaluator, "_verify_evaluation", short_verify)
    monkeypatch.setattr(
        _selection_evidence, "freeze_inherited_candidates", no_candidates
    )
    client = Client()
    method = System("Parent Live", client.apply, execution="host")
    wrong = System("Wrong Live", Client().apply, execution="host")
    factory_calls: list[str] = []

    def factory(reference: str) -> System:
        factory_calls.append(reference)
        assert reference == "local_fixture:parent"
        return method

    if parent_reference:
        monkeypatch.setattr(
            import_module("marl_battlegrounds._method_loading"), "load_method", factory
        )
    config = runner.TrainConfig(
        method="ff_ippo",
        num_envs=4,
        total_env_steps=8,
        keep_past=0,
        seed=817,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        checkpoint_interval_updates=1,
        validation_fractions=(1.0,),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        validation_opponents=("local_fixture:parent",) if parent_reference else None,
        metrics="none",
        verbose=False,
    )
    parent = runner.train(
        config,
        output_dir=tmp_path / "parent",
        validation_opponents=None if parent_reference else (method,),
    )
    assert client.calls and all(
        mask.shape == (config.num_envs,) for mask in client.calls
    )
    parent_panel = parent.run_dir / "validation_panel/panel.json"
    saved_panel = json.loads(parent_panel.read_text())
    assert saved_panel["members"][0]["reference"] == (
        "local_fixture:parent" if parent_reference else None
    )
    assert factory_calls == (["local_fixture:parent"] if parent_reference else [])

    def files(root: Path) -> dict[str, str]:
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    def latest(root: Path) -> Path:
        pointer = json.loads((root / "latest_checkpoint.json").read_text())
        return root / pointer["relative_path"]

    before = files(parent.run_dir)
    parent_checkpoint = latest(parent.run_dir)
    calls_before = len(client.calls)
    if not parent_reference:
        with pytest.raises(ValueError, match="live-only"):
            runner.extend_training(
                parent_checkpoint,
                additional_env_steps=4,
                output_dir=tmp_path / "missing-parent-binding",
            )
        assert not (tmp_path / "missing-parent-binding").exists()
    with pytest.raises(ValueError, match="frozen M8 identity"):
        runner.extend_training(
            parent_checkpoint,
            additional_env_steps=4,
            output_dir=tmp_path / "wrong-parent-binding",
            validation_opponents=(wrong,),
        )
    assert not (tmp_path / "wrong-parent-binding").exists()
    assert len(client.calls) == calls_before and files(parent.run_dir) == before
    changes: dict[str, object] = {"validation": {"env_steps": [12]}}
    child = runner.extend_training(
        parent_checkpoint,
        additional_env_steps=4,
        output_dir=tmp_path / "child",
        changes=changes,
        validation_opponents=(method,),
    )
    assert changes == {"validation": {"env_steps": [12]}}
    assert len(client.calls) > calls_before
    child_details = json.loads((child.run_dir / "run_details.json").read_text())
    assert child_details["config"]["validation_panel"] == str(parent_panel)
    assert child_details["config"]["validation_opponents"] is None
    assert child_details["continuation"]["changes"] == changes
    calls_before = len(client.calls)
    restored = runner.train(
        resume_from=latest(child.run_dir), validation_opponents=(method,)
    )
    assert restored.final_actor == child.final_actor
    assert restored.selected_actor == child.selected_actor
    assert len(client.calls) == calls_before
    assert files(parent.run_dir) == before
    assert factory_calls == (["local_fixture:parent"] if parent_reference else [])
    if parent_reference:
        return

    future_client = Client()
    future_method = System("Future Live", future_client.apply, execution="host")
    future_panel = validation.create_panel(
        opponents=(future_method,), output_dir=tmp_path / "future-panel"
    )
    future_bindings = [future_method]
    future_changes: dict[str, object] = {
        "panel": str(future_panel.path),
        "bindings": future_bindings,
        "env_steps": [12],
    }
    request: dict[str, object] = {"validation": future_changes}
    with pytest.raises(ValueError, match="frozen M8 identity"):
        runner.extend_training(
            parent_checkpoint,
            additional_env_steps=4,
            output_dir=tmp_path / "wrong-future-binding",
            validation_opponents=(method,),
            changes={
                "validation": {"panel": str(future_panel.path), "bindings": [method]}
            },
        )
    assert not (tmp_path / "wrong-future-binding").exists()
    assert len(client.calls) == calls_before and not future_client.calls
    preflight_names: list[list[str]] = []
    checked_panels: list[str] = []
    original_preflight = validation.prepare_validation_teams
    original_purpose = runner._check_panel_purpose

    def preflight(
        candidate: System | Policy,
        opponents: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        if not (tmp_path / "future-child").exists():
            preflight_names.append([member.name for member in opponents])
        return original_preflight(candidate, opponents, **kwargs)

    def check_purpose(
        config: runner.TrainConfig, panel: validation.FrozenPanel | None
    ) -> None:
        assert panel is not None
        checked_panels.append(panel.digest)
        original_purpose(config, panel)

    monkeypatch.setattr(validation, "prepare_validation_teams", preflight)
    monkeypatch.setattr(runner, "_check_panel_purpose", check_purpose)
    future = runner.extend_training(
        parent_checkpoint,
        additional_env_steps=4,
        output_dir=tmp_path / "future-child",
        validation_opponents=(method,),
        changes=request,
    )
    assert preflight_names == [["Future Live"]]
    assert checked_panels == [saved_panel["panel_digest"], future_panel.digest]
    assert future_client.calls and len(client.calls) == calls_before
    assert request["validation"] is future_changes
    assert (
        future_changes["bindings"] is future_bindings
        and future_bindings[0] is future_method
    )
    assert future_changes["env_steps"] == [12]
    future_details = json.loads((future.run_dir / "run_details.json").read_text())
    assert future_details["config"]["validation_panel"] == str(future_panel.path)
    assert future_details["continuation"]["changes"]["validation"] == {
        "panel": str(future_panel.path),
        "env_steps": [12],
    }
    assert '"bindings"' not in (future.run_dir / "run_details.json").read_text()
    assert '"calls"' not in (future.run_dir / "run_details.json").read_text()
    future_calls = len(future_client.calls)
    resumed = runner.train(
        resume_from=latest(future.run_dir), validation_opponents=(future_method,)
    )
    assert (
        resumed.final_actor == future.final_actor
        and resumed.selected_actor == future.selected_actor
    )
    assert (
        len(future_client.calls) == future_calls and len(client.calls) == calls_before
    )
    assert files(parent.run_dir) == before


@pytest.mark.parametrize("kind", ("jax", "policy", "host", "mixed", "factory"))
def test_live_candidates_validate_custom_rosters_and_reuse_exact_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from marl_battlegrounds.evaluation.policy_execution import team
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.evaluation.results import load_results

    calls: list[np.ndarray[Any, Any]] = []

    def act(
        variables: object, memory: object, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, object]:
        del variables, keys
        calls.append(np.asarray(inputs.controlled_mask).copy())
        zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
        return ActorAction(zero, zero, zero), memory

    candidate: System | Policy | str
    host = System("Candidate", act, execution="host")
    if kind == "policy":
        candidate = Policy("Candidate", _mixture_policy, initial_carry=jnp.int32(0))
    elif kind in ("host", "mixed", "factory"):
        candidate = team("random", host, slots=[[0], [1]]) if kind == "mixed" else host
    else:
        candidate = System("Candidate", _jax_apply)
    panel = validation.create_panel(opponents=["random"], output_dir=tmp_path / "panel")
    made: list[str] = []
    if kind == "factory":
        loader = import_module("marl_battlegrounds._method_loading")
        original_load = loader.load_method

        def load(reference: str) -> System | Policy:
            if reference == "researcher:build":
                made.append(reference)
                return host
            return original_load(reference)

        monkeypatch.setattr(loader, "load_method", load)
        candidate = "researcher:build"
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    original, original_verify = evaluator.evaluate, evaluator._verify_evaluation
    provenance = capture_recording_provenance(num_envs=2)

    def fixed_provenance(**kwargs: object) -> dict[str, object]:
        del kwargs
        return provenance

    def short(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original(*args, max_steps=2, **kwargs)

    def short_verify(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_verify(*args, max_steps=2, **kwargs)

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    monkeypatch.setattr(evaluator, "evaluate", short)
    monkeypatch.setattr(evaluator, "_verify_evaluation", short_verify)
    options: dict[str, Any] = dict(
        output_dir=tmp_path / "task",
        seed_pairs=1,
        num_envs=2,
        maps=[42],
        system_roster=["mage", "mage"],
        opponent_roster=["warrior"],
    )
    original_roster = options["system_roster"]

    def change_caller_list(event: dict[str, Any]) -> None:
        if event["event"] == "validation_task":
            original_roster[:] = ["rogue"]
            event["system_roster"][:] = ["priest"]
            event["maps"][:] = [43]

    result = validation.validate_checkpoint(
        candidate, panel, event_callback=change_caller_list, **options
    )
    options["system_roster"] = ["mage", "mage"]
    assert result["complete"] and result["games"] == 2
    assert result["schema_version"] == 5
    assert result["system_id"] and result["system"]
    assert not {"checkpoint_id", "actor_digest", "env_steps"}.intersection(result)
    assert result["maps"] == [42]
    assert result["system_roster"] == ["mage", "mage"]
    saved = load_results(result["pass_paths"][0], phase="validation")
    rows = saved.table("episodes")
    saved_pass = next(iter(saved.metadata["passes"].values()))
    assert {row["spawn_locations"] for row in saved_pass["episodes"].values()} == {0, 1}
    assert saved_pass["system_ids"]["team_a"] == result["system_id"]
    assert all(value == result["system"]["name"] for value in rows["team_a_policy"])
    if kind == "mixed":
        assert calls
        assert all(
            np.array_equal(mask, [[False, True, False, False, False]] * 2)
            for mask in calls
        )
    before = len(calls)
    assert validation.validate_checkpoint(candidate, panel, **options) == result
    assert len(calls) == before
    if kind == "factory":
        assert len(made) == 2
    files = {
        path: path.read_bytes()
        for path in (tmp_path / "task").rglob("*")
        if path.is_file()
    }
    for changed in (
        {"maps": [43]},
        {"system_roster": ["mage", "warrior"]},
        {"opponent_roster": ["rogue"]},
    ):
        changed_options = options.copy()
        changed_options.update(changed)
        with pytest.raises(ValueError, match="different scientific conditions"):
            validation.validate_checkpoint(candidate, panel, **changed_options)
    with pytest.raises(ValueError, match="different scientific conditions"):
        validation.validate_checkpoint(policy("tdm-beta"), panel, **options)
    assert all(path.read_bytes() == data for path, data in files.items())
    assert len(calls) == before


@pytest.mark.parametrize(
    "change", ({"maps": []}, {"maps": [42, 42]}, {"system_roster": []})
)
def test_invalid_custom_validation_conditions_create_no_output(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    panel = validation.create_panel(opponents=["random"], output_dir=tmp_path / "panel")
    output = tmp_path / "task"
    with pytest.raises(ValueError):
        validation.validate_checkpoint("random", panel, output_dir=output, **change)
    assert not output.exists()


def test_duplicate_panel_labels_preserve_names_and_avoid_numbered_name_collisions(
    tmp_path: Path,
) -> None:
    methods = [
        System("Same", _jax_apply, variables=jnp.float32(value)) for value in (1, 2)
    ]
    methods.append(System("Same (2)", _jax_apply))
    panel = validation.create_panel(opponents=methods, output_dir=tmp_path / "panel")
    assert [member.name for member in panel.members] == ["Same", "Same (3)", "Same (2)"]
    assert [method.name for method in panel.methods] == ["Same", "Same", "Same (2)"]
    assert len({member.registration_id for member in panel.members}) == 3
    loaded = validation.load_panel(panel.path, bindings=methods)
    assert loaded.members == panel.members


def test_late_incompatible_panel_roster_fails_before_any_output_or_provider(
    tmp_path: Path,
) -> None:
    client = Client()
    short = independent_policies((policy("random"),))
    panel = validation.create_panel(
        opponents=[System("Host", client.apply, execution="host"), short],
        output_dir=tmp_path / "panel",
    )
    events: list[dict[str, Any]] = []
    output = tmp_path / "task"
    with pytest.raises(ValueError):
        validation.validate_checkpoint(
            "random",
            panel,
            output_dir=output,
            maps=[42],
            system_roster=["mage"],
            opponent_roster=["warrior", "warrior"],
            event_callback=events.append,
        )
    assert client.calls == [] and events == []
    assert not output.exists()


def test_builtin_candidate_precedes_a_same_named_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "random").mkdir()
    panel = validation.create_panel(
        opponents=["tdm-alpha"], output_dir=tmp_path / "panel"
    )

    def stop_at_task(
        directory: Path, content: dict[str, Any], callback: object
    ) -> dict[str, Any]:
        del directory, callback
        assert content["system"]["name"] == "random"
        raise RuntimeError("candidate resolved")

    monkeypatch.setattr(validation, "_task", stop_at_task)
    with pytest.raises(RuntimeError, match="candidate resolved"):
        validation.validate_checkpoint("random", panel, output_dir=tmp_path / "task")
    assert not (tmp_path / "task").exists()


def test_validation_folder_names_are_clear_and_never_replace_other_paths(
    tmp_path: Path,
) -> None:
    first = validation._pass_directory(tmp_path, 0, "Same / Name")
    second = validation._pass_directory(tmp_path, 1, "Same / Name")
    assert first.name == "opponent_00_same_name"
    assert second.name == "opponent_01_same_name" and first != second
    legacy = tmp_path / "opponent-0"
    legacy.mkdir()
    (legacy / "keep.txt").write_text("saved")
    assert validation._pass_directory(tmp_path, 0, "Same / Name") == legacy
    first.mkdir()
    with pytest.raises(ValueError, match="two folders"):
        validation._pass_directory(tmp_path, 0, "Same / Name")
    assert (legacy / "keep.txt").read_text() == "saved"
    second.symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError, match="plain directory"):
        validation._pass_directory(tmp_path, 1, "Same / Name")


@pytest.mark.parametrize("saved_actor", (False, True))
def test_deployed_partner_panel_preserves_candidate_identity_and_saved_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_actor: bool
) -> None:
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.evaluation.results import load_results
    from marl_battlegrounds.training import _selection_evidence

    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance(num_envs=2)

    def fixed_provenance(**kwargs: object) -> dict[str, object]:
        del kwargs
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    calls: list[np.ndarray[Any, Any]] = []

    def act(
        variables: object, memory: object, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, object]:
        del variables, keys
        calls.append(np.asarray(inputs.controlled_mask).copy())
        zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
        return ActorAction(zero, zero, zero), memory

    candidate: System | Path = System("Researcher", act, execution="host")
    if saved_actor:
        candidate = checkpoints.export_system(
            initialize_ppo(jax.random.key(42), method="ff_ippo").actor_params,
            tmp_path / "actor",
            metadata={
                "run_id": "partner-validation",
                "seed": 42,
                "env_steps": 0,
                "checkpoint_id": "c" * 64,
            },
            spawn_frame="left",
            method="ff_ippo",
        )
    else:
        evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
        original, original_verify = evaluator.evaluate, evaluator._verify_evaluation

        def short(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            return original(*args, max_steps=2, **kwargs)

        def short_verify(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            return original_verify(*args, max_steps=2, **kwargs)

        monkeypatch.setattr(evaluator, "evaluate", short)
        monkeypatch.setattr(evaluator, "_verify_evaluation", short_verify)
    panel = validation.create_panel(opponents=["random"], output_dir=tmp_path / "panel")
    partners = {
        "Training Partner": policy("random"),
        "New Partner": policy("tdm-alpha"),
    }
    bindings, declaration = validation.freeze_validation_partners(
        partners,
        learner_slots=[0],
        partner_labels={"Training Partner": "familiar", "New Partner": "held_out"},
    )
    options: dict[str, Any] = dict(
        output_dir=tmp_path / "task",
        seed_pairs=1,
        num_envs=2,
        maps=[42],
        system_roster=["mage", "priest"],
        opponent_roster=["warrior"],
        partners=bindings,
        learner_slots=[0],
        partner_labels={"Training Partner": "familiar", "New Partner": "held_out"},
    )
    result = validation.validate_checkpoint(candidate, panel, **options)
    assert result["games"] == 4 and result["independent_blocks"] == 1
    assert result["members"][0]["root"] == result["members"][1]["root"]
    assert [
        (row["name"], row["label"], row["games"]) for row in result["partner_results"]
    ] == [("Training Partner", "familiar", 2), ("New Partner", "held_out", 2)]
    if saved_actor:
        assert isinstance(candidate, Path)
        assert result["checkpoint_id"] == "c" * 64 and result["env_steps"] == 0
        assert (
            result["actor_digest"]
            == checkpoints.artifact_identity(candidate)["actor_digest"]
        )
    else:
        assert result["system"]["name"] == "Researcher"
        assert not {"checkpoint_id", "actor_digest", "env_steps"}.intersection(result)
        assert calls and all(
            np.array_equal(mask, [[True, False, False, False, False]] * 2)
            for mask in calls
        )
    for index, path in enumerate(result["pass_paths"]):
        saved = load_results(path, phase="validation")
        entry = next(iter(saved.metadata["passes"].values()))
        evidence = json.loads((Path(path).parent / "sampling_facts.json").read_text())
        assert evidence["deployment"]["partner"] == declaration["partners"][index]
        assert evidence["system_ids"] == entry["system_ids"]
        assert Path(path).parent.parent.name.startswith(f"partner_{index:02d}_")
    before = len(calls)
    assert validation.validate_checkpoint(candidate, panel, **options) == result
    assert len(calls) == before
    sidecar = Path(result["pass_paths"][-1]).parent / "sampling_facts.json"
    original_sidecar = sidecar.read_bytes()
    sidecar.unlink()
    with pytest.raises(ValueError, match="missing constituent evidence"):
        validation.validate_checkpoint(candidate, panel, **options)
    sidecar.write_bytes(original_sidecar)
    assert len(calls) == before
    changed = dict(
        options,
        partner_labels={"Training Partner": "unknown", "New Partner": "held_out"},
    )
    with pytest.raises(ValueError, match="different scientific conditions"):
        validation.validate_checkpoint(candidate, panel, **changed)
    if saved_actor:
        assert isinstance(candidate, Path)
        task = json.loads((tmp_path / "task" / "task.json").read_text())
        reduced, rows = _selection_evidence._panel_evidence(
            tmp_path / "task" / "validation_summary.json",
            result,
            task,
            validation._artifact(candidate),
            panel,
            _selection_evidence._Snapshot(),
        )
        assert reduced == result and len(rows) == 4
        sidecar = Path(result["pass_paths"][1]).parent / "sampling_facts.json"
        evidence = json.loads(sidecar.read_text())
        evidence["deployment"]["learner_slots"] = [1]
        sidecar.write_text(json.dumps(evidence))
        with pytest.raises(ValueError, match="frozen constituents"):
            _selection_evidence._panel_evidence(
                tmp_path / "task" / "validation_summary.json",
                result,
                task,
                validation._artifact(candidate),
                panel,
                _selection_evidence._Snapshot(),
            )


def test_deployed_partner_preflight_checks_every_member_before_files_or_calls(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.policy_execution import team

    client = Client()
    actor = System("Actor", client.apply, execution="host")
    partners = {"First": policy("random"), "Last": team("random", slots=[[0]])}
    panel = validation.create_panel(opponents=["random"], output_dir=tmp_path / "panel")
    with pytest.raises(ValueError):
        validation.validate_checkpoint(
            actor,
            panel,
            output_dir=tmp_path / "task",
            partners=partners,
            learner_slots=[0],
            system_roster=["mage", "priest"],
            opponent_roster=["warrior"],
            maps=[42],
        )
    assert not client.calls and not (tmp_path / "task").exists()
    with pytest.raises(ValueError, match="no active learner"):
        validation.prepare_validation_teams(
            actor,
            ["random"],
            partners={"Partner": policy("random")},
            learner_slots=[1],
            system_roster=["mage"],
            maps=[42],
        )


@pytest.mark.parametrize("with_partner", (False, True))
def test_custom_random_deployment_verifies_actual_saved_rows_and_declared_conditions(
    tmp_path: Path, with_partner: bool
) -> None:
    actor = checkpoints.export_system(
        initialize_ppo(jax.random.key(11), method="ff_ippo").actor_params,
        tmp_path / "actor",
        metadata={
            "run_id": "random-team",
            "seed": 11,
            "env_steps": 0,
            "checkpoint_id": "d" * 64,
        },
        spawn_frame="left",
        method="ff_ippo",
    )
    options: dict[str, Any] = dict(
        maps=[42], system_roster=["mage", "priest"], opponent_roster=["warrior"]
    )
    declaration = options.copy()
    if with_partner:
        bindings, declared = validation.freeze_validation_partners(
            {"Training Partner": policy("random")},
            learner_slots=[0],
            partner_labels={"Training Partner": "familiar"},
        )
        declaration.update(declared)
        options.update(
            partners=bindings,
            learner_slots=[0],
            partner_labels={"Training Partner": "familiar"},
        )
    result = validation.validate_random(
        actor, output_dir=tmp_path / "task", num_envs=2, seed_pairs=1, **options
    )
    capture = {
        **result,
        "actor_path": str(actor),
        "summary_path": str(tmp_path / "task" / "validation_summary.json"),
        "reference_path": None,
        "reused_initialization": False,
        "elapsed_seconds": 0.0,
        "training_seconds": 0.0,
        "wall_seconds": 0.0,
    }
    verify: dict[str, Any] = dict(
        actor_digest=result["actor_digest"], seed_pairs=1, deployment=declaration
    )
    assert validation.verify_random_result(capture, **verify) == result
    reference = tmp_path / "original.json"
    reference.write_text(json.dumps(capture))
    assert validation.read_random_initialization(reference, **verify) == capture
    changed = dict(declaration, system_roster=["priest", "mage"])
    with pytest.raises(ValueError, match="task or summary differs"):
        validation.verify_random_result(capture, **dict(verify, deployment=changed))
    if with_partner:
        from marl_battlegrounds.training import _selection_evidence

        task = json.loads((tmp_path / "task" / "task.json").read_text())
        _, rows = _selection_evidence._panel_evidence(
            tmp_path / "task" / "validation_summary.json",
            result,
            task,
            validation._artifact(actor),
            validation._random_panel(),
            _selection_evidence._Snapshot(),
        )
        statistics = analysis._screen_statistics(rows, result, summary=result)
        assert statistics["score"] == result["score"]


def test_fresh_partner_run_records_native_actor_validation_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.training_continuation_helpers import fixed_source

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.training import _selection_evidence

    fixed_source(monkeypatch, "ff_ippo")
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance(num_envs=2)

    def fixed_provenance(**kwargs: object) -> dict[str, object]:
        del kwargs
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    monkeypatch.setattr(validation, "VALIDATION_MAPS", (42,))
    config = runner.TrainConfig(
        method="ff_ippo",
        num_envs=2,
        total_env_steps=4,
        keep_past=0,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        curriculum=[
            {
                "share": 1.0,
                "maps": [0],
                "rosters": {"system": ["mage", "priest"], "opponent": ["warrior"]},
            }
        ],
        learner_slots=(0,),
        partners={"Training Partner": 1.0},
        validation_partners={"New Partner": "tdm-alpha"},
        validation_partner_labels={"New Partner": "held_out"},
        validation_opponents=("random",),
        validation_fractions=(1.0,),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        checkpoint_interval_updates=1,
        metrics="none",
        verbose=False,
    )
    result = runner.train(
        config, partners={"Training Partner": "random"}, output_dir=tmp_path / "run"
    )
    saved = checkpoints.read_checkpoint_details(result.final_checkpoint)
    metadata = saved["metadata"]
    assert metadata.get("continuation") is None
    assert (
        metadata["validation_declaration"]["deployment"]
        == metadata["validation_deployment"]
    )
    records = (
        metadata["host_state"]["routine_results"]
        + metadata["host_state"]["confirmation_results"]
    )
    assert records and all(
        row["system_roster"] == ["mage", "priest"] for row in records
    )
    assert all(
        [part["label"] for part in row["partner_results"]] == ["familiar", "held_out"]
        for row in records
    )
    evidence = _selection_evidence.read_run_evidence(result.run_dir)
    assert evidence["status"] == "complete" and evidence["records"]
    for row in evidence["records"]:
        assert row["checkpoint_id"] in evidence["actors"]
        assert (
            row["actor_digest"]
            == evidence["actors"][row["checkpoint_id"]]["actor_digest"]
        )
    resumed = runner.train(resume_from=result.final_checkpoint)
    assert resumed.selected_actor == result.selected_actor
    assert resumed.final_checkpoint == result.final_checkpoint


def test_runner_rejects_custom_conditions_with_historical_panel_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.evaluation.run_writer import RunWriter

    historical = validation.FrozenPanel(
        path=tmp_path / "historical_panel.json",
        digest="old-panel",
        members=(),
        qualified=True,
        schema_version=1,
    )

    def load(*args: object, **kwargs: object) -> validation.FrozenPanel:
        del args, kwargs
        return historical

    def no_writer(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid historical deployment must fail before opening a writer")

    monkeypatch.setattr(validation, "load_panel", load)
    monkeypatch.setattr(RunWriter, "__init__", no_writer)
    config = runner.TrainConfig(
        method="ff_ippo",
        num_envs=2,
        total_env_steps=4,
        keep_past=0,
        curriculum=[{"share": 1.0, "maps": [0], "team_size": 2}],
        ppo=PPOConfig(rollout_length=2, epochs=1),
        validation_panel=str(historical.path),
        metrics="none",
        verbose=False,
    )
    with pytest.raises(ValueError, match="panel made with opponents="):
        runner.train(config, output_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_public_validation_reuses_program_for_changed_same_shape_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from time import perf_counter

    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    program = evaluator._jax_system_chunk
    chunk_seconds: list[float] = []
    final_positions: list[np.ndarray[Any, Any]] = []
    final_moves: list[list[int]] = []

    def timed_chunk(*args: object, **kwargs: object) -> object:
        started = perf_counter()
        output = program(*args, **kwargs)
        jax.block_until_ready(output)
        chunk_seconds.append(perf_counter() - started)
        state = output[0].state.core_state
        final_positions.append(np.asarray(state.agent_positions))
        final_moves.append(
            np.asarray(state.previous_timestep_move_actions)[:, 0].tolist()
        )
        return output

    def weighted_actor(
        variables: object, memory: object, inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, object]:
        del keys
        moves = inputs.action_mask.move_mask
        last_legal = jnp.argmax(
            jnp.where(moves, jnp.arange(moves.shape[-1]), -1), axis=-1
        )
        selected = jnp.where(cast(Array, variables) > 0, last_legal, 0)
        zero = jnp.zeros_like(selected)
        return ActorAction(selected, zero, zero), memory

    monkeypatch.setattr(evaluator, "_jax_system_chunk", timed_chunk)
    started = perf_counter()
    panel = validation.create_panel(
        opponents=[System("Idle Opponent", _jax_apply)],
        output_dir=tmp_path / "panel",
    )
    panel_seconds = perf_counter() - started
    candidate = System("Changing Actor", weighted_actor, variables=jnp.float32(0))
    partner = System("Fixed Partner", _jax_apply)
    measurements: list[dict[str, object]] = []
    summaries: list[dict[str, Any]] = []
    for index in range(2):
        chunk_seconds.clear()
        compiled_before = program._cache_size()
        started = perf_counter()
        summary = validation.validate_checkpoint(
            replace(candidate, variables=jnp.float32(index)),
            panel,
            output_dir=tmp_path / f"candidate_{index}",
            maps=[42],
            system_roster=["mage", "mage"],
            opponent_roster=["warrior"],
            partners={"Partner": partner},
            learner_slots=[0],
            seed_pairs=1,
            num_envs=2,
            chunk_size=300,
        )
        wall_seconds = perf_counter() - started
        added_programs = program._cache_size() - compiled_before
        assert summary["complete"] and summary["games"] == 2
        assert chunk_seconds
        assert added_programs == (1 if index == 0 else 0)
        summaries.append(summary)
        measurements.append(
            {
                "weight": index,
                "new_outer_programs": added_programs,
                "chunk_calls": len(chunk_seconds),
                "last_owned_moves": final_moves[-1],
                "chunk_compile_and_execution_seconds": sum(chunk_seconds),
                "outside_chunk_seconds": wall_seconds - sum(chunk_seconds),
                "total_seconds": wall_seconds,
                "saved_bytes": sum(
                    path.stat().st_size
                    for path in (tmp_path / f"candidate_{index}").rglob("*")
                    if path.is_file()
                ),
            }
        )
    assert summaries[0]["task_id"] != summaries[1]["task_id"]
    assert summaries[0]["system_id"] != summaries[1]["system_id"]
    assert not np.array_equal(final_positions[0][:, 0], final_positions[-1][:, 0])
    report = {"panel_seconds": panel_seconds, "calls": measurements}
    (tmp_path / "validation_reuse.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def test_nested_composition_execution_key_hashes_each_leaf_once() -> None:
    from marl_battlegrounds.evaluation.policy_execution import team
    from marl_battlegrounds.evaluation.system_evaluation import (
        prepare_evaluation_system,
    )

    calls: list[int] = []

    class Hook:
        def __hash__(self) -> int:
            calls.append(1)
            return 91

        def __call__(self, *_args: object) -> object:
            raise AssertionError("Building an execution key called the method")

    actor = System("Actor", Hook(), variables=jnp.float32(0))
    first = prepare_evaluation_system(team(team(team(actor))))[0]
    second = prepare_evaluation_system(
        team(team(team(replace(actor, variables=jnp.float32(1)))))
    )[0]
    calls.clear()
    first_hash = hash(first)
    assert len(calls) == 1
    calls.clear()
    assert first == second and first_hash == hash(second)
    assert len(calls) == 1
    changed = replace(second, reset=None)
    assert changed != first


@pytest.mark.parametrize("invalid_role", ("partner", "opponent"))
def test_runner_panel_preflight_allows_a_corrected_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_role: str
) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.evaluation.policy_execution import team

    class SetupCompleteError(Exception):
        pass

    published: list[validation.FrozenPanel] = []
    original_publish = validation._publish_system_panel

    def publish(panel: validation.FrozenPanel, content: dict[str, Any]) -> None:
        original_publish(panel, content)
        published.append(panel)
        raise SetupCompleteError

    monkeypatch.setattr(validation, "_publish_system_panel", publish)
    client = Client()
    compatible = System("Live", client.apply, execution="host")
    incompatible = team("random", slots=[[0]])
    partner_case = invalid_role == "partner"
    config = runner.TrainConfig(
        method="ff_ippo",
        num_envs=2,
        total_env_steps=4,
        keep_past=0,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        learner_slots=(0,) if partner_case else None,
        partners={"Familiar": 1.0} if partner_case else None,
        validation_partner_labels={"Held Out": "held_out"} if partner_case else None,
        metrics="none",
        verbose=False,
    )
    training_partners = {"Familiar": "random"} if partner_case else None
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        runner.train(
            config,
            output_dir=output,
            partners=training_partners,
            validation_partners={"Held Out": incompatible} if partner_case else None,
            validation_opponents=(compatible if partner_case else incompatible,),
        )
    assert not output.exists() and not published and not client.calls
    with pytest.raises(SetupCompleteError):
        runner.train(
            config,
            output_dir=output,
            partners=training_partners,
            validation_partners={"Held Out": policy("random")}
            if partner_case
            else None,
            validation_opponents=(compatible,),
        )
    assert len(published) == 1 and published[0].path.is_file()
    assert published[0].methods[0].apply.__self__ is client
    assert not client.calls
