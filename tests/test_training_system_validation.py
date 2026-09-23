"""Check any-System validation, retained host clients and exact panel selection.

Built-ins, independent policies, mixtures, exported actors, JAX methods, host
methods and import factories share M8 registrations. A local text-JSON client
stands in for a language-model service. The real short-horizon CPU evaluator
proves 32-lane padded execution, complete saved rows and no extra provider calls
after a task completes. No network client is used. Existing tests keep the old
panel/task hashes and selection covered.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path
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
from marl_battlegrounds.training import analysis, checkpoints, runner, validation


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
    assert analysis.select_checkpoint((early, later))["checkpoint_id"] == "early"
    new = [{**row, "selection_schema_version": 2} for row in (early, later)]
    assert analysis.select_checkpoint(new)["checkpoint_id"] == "later"
    assert (
        analysis.select_checkpoint((new[0], {**new[1], "score": 0.49}))["checkpoint_id"]
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


def test_independent_mixture_export_and_fake_language_client_complete_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )

    actor = checkpoints.export_system(
        initialize_ppo(jax.random.key(89)).actor_params,
        tmp_path / "actor",
        metadata={
            "run_id": "system-validation",
            "seed": 89,
            "env_steps": 32,
            "checkpoint_id": "c" * 64,
        },
        spawn_frame="left",
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


def _ranking_fixture(panel: validation.FrozenPanel) -> TournamentResult:
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
        validation.VALIDATION_MAPS, 5, roster_a, roster_b, 20, 300
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

    def begin(**kwargs: object) -> Any:  # noqa: ANN401
        del kwargs
        raise SetupReachedError

    monkeypatch.setattr(learner, "init_learner", begin)
    client = Client()
    method = System("Live", client.apply, execution="host")
    config = runner.TrainConfig(purpose="demonstration")
    with pytest.raises(SetupReachedError):
        runner.train(
            config, output_dir=tmp_path / "run", validation_opponents=(method,)
        )
    path = tmp_path / "run/validation_panel/panel.json"
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == 2 and saved["members"][0]["reference"] is None
    assert (
        validation.load_panel(path, bindings=(method,)).methods[0].apply.__self__
        is client
    )
    assert client.calls == []
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

    def details(path: object) -> dict[str, str]:
        del path
        return {"kind": "learner"}

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
