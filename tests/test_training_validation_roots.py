"""Check fresh Random validation roots without running a learning experiment.

The real validation and M8 schedule builders pass roots to a stubbed numerical
executor. Known game rows cover task hashes, native options, strict saved-result
checks, and unchanged shared initialization. The stub supplies a real frozen
Policy and its shared registration so sampling facts keep their identity checks.
CPU JAX key derivation checks the
routine and confirmation schedules use distinct reset/game/policy start keys.
A new Random record carries red_zone_depth 5.0 in its schema 3 task and M8
options and verifies only at that depth. The same games rewritten as a record
saved before the Red Zone rule (schema 1 task, no depth option, kills read from
scores) verify, and are reusable as a shared initialization, only when the
caller passes red_zone_depth=None.
"""

# Tests inspect the validation owner's strict evidence boundaries.
# pyright: reportPrivateUsage=false

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.evaluation.policy_execution import Policy, policy
from marl_battlegrounds.evaluation.recording_identity import tree_digest
from marl_battlegrounds.training import checkpoints, validation


def _task(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    return validation.validation_task_description(
        checkpoint_id="initial-checkpoint",
        actor_digest="fixed-inference",
        env_steps=0,
        panel_digest="random-diagnostic-v1",
        purpose="random",
        seed_pairs=4,
        members=(("Random", "builtin-random"),),
        **overrides,
    )


def test_default_task_hash_is_identical_to_historical_description() -> None:
    historical = {
        "schema_version": 1,
        "checkpoint_id": "initial-checkpoint",
        "actor_digest": "fixed-inference",
        "env_steps": 0,
        "panel_digest": "random-diagnostic-v1",
        "purpose": "random",
        "seed_pairs": 4,
        "maps": [42, 43, 44, 45, 46],
        "root": 19_043_001,
        "members": [{"name": "Random", "actor_digest": "builtin-random"}],
    }
    digest = hashlib.sha256(
        json.dumps(historical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected = {**historical, "task_id": digest}
    assert _task() == _task(root_seed=19_043_001) == expected
    changed = _task(root_seed=19_044_791)
    assert changed["task_id"] != expected["task_id"]
    assert changed["root"] == 19_044_791
    assert {k: v for k, v in changed.items() if k not in ("root", "task_id")} == {
        k: v for k, v in expected.items() if k not in ("root", "task_id")
    }


@pytest.mark.parametrize("root", [-1, 2**32, True, False, 1.5, "19044791", None])
def test_invalid_root_rejects_before_actor_or_output_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root: Any,  # noqa: ANN401
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Invalid roots must reject before inspecting an actor")

    monkeypatch.setattr(validation, "_artifact", forbidden)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="root_seed"):
        validation.validate_random("missing-actor", output_dir=output, root_seed=root)
    with pytest.raises(ValueError, match="root_seed"):
        validation.verify_random_result(
            {}, actor_digest="fixed", seed_pairs=4, root_seed=root
        )
    assert not output.exists()


@pytest.mark.parametrize("root", [0, 2**32 - 1])
def test_root_uint32_endpoints_are_valid(root: int) -> None:
    assert _task(root_seed=root)["root"] == root


def test_panel_roots_stay_owned_by_the_existing_protocol() -> None:
    for purpose, root in (
        ("routine", 19_043_002),
        ("initialization", 19_043_002),
        ("confirmation", 19_043_003),
    ):
        options: dict[str, Any] = dict(
            checkpoint_id="checkpoint",
            actor_digest="actor",
            env_steps=0,
            panel_digest="panel",
            purpose=purpose,
            seed_pairs=10,
            members=(("Halfway", "halfway"), ("Final", "final")),
        )
        assert validation.validation_task_description(**options)["root"] == root
        with pytest.raises(ValueError, match="Random checks only"):
            validation.validation_task_description(**options, root_seed=19_044_791)


@pytest.fixture
def fake_m8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    results = import_module("marl_battlegrounds.evaluation.results")
    actor_path = tmp_path / "actor"
    actor_path.mkdir()
    actor = replace(
        policy("random"), name="Fixed Test Actor", checkpoint="fixed-inference"
    )
    identity = {
        "checkpoint_id": "initial-checkpoint",
        "actor_digest": "fixed-inference",
        "weight_digest": tree_digest(actor.variables),
        "env_steps": 0,
        "run_id": "original-run",
        "seed": 42,
    }
    calls: list[dict[str, Any]] = []
    saved: dict[Path, SimpleNamespace] = {}
    game_rows: dict[Path, list[dict[str, Any]]] = {}

    def execute(_actor: object, opponent: object, specs: Any, **options: Any) -> Any:  # noqa: ANN401
        assert opponent == "random"
        directory = Path(options["output_dir"]) / "known-m8-run"
        directory.mkdir(parents=True)
        (directory / "run_details.json").write_text("{}")
        contract = deepcopy(options["contract"])
        contract["source_choices"] = [
            {"map_id": spec.map_id} for spec in options["source_choices"]
        ]
        contract = json.loads(json.dumps(contract))
        methods = {
            "team_a": validation._method_snapshot(_actor),
            "team_b": validation._method_snapshot(opponent),
        }
        entry = {
            "system_ids": {team: method[1] for team, method in methods.items()},
            "policies": {team: method[2] for team, method in methods.items()},
            "details": {
                "evaluation_contract": contract,
                "phase": "validation",
                "pass_id": options["pass_id"],
                "seed": options["seed"],
                "num_episodes": len(specs),
                "metrics": "priority",
                "chunk_size": options["chunk_size"],
            },
        }
        game_rows[directory] = [
            {
                "map_id": spec.map_id,
                "seed_id": spec.random_seed_id,
                "spawn_locations": spec.spawn_locations,
                "opponent": "Random",
                "system_game_score": 0.5,
                "episode_length": 300,
                "team_a_score": 3,
                "team_b_score": 1,
            }
            for spec in specs
        ]
        saved[directory] = SimpleNamespace(
            status="complete", metadata={"passes": {"pass": entry}}
        )
        calls.append({"seed": options["seed"], "specs": specs, "directory": directory})
        return SimpleNamespace(run_dir=directory)

    def run_pass(request: dict[str, Any], **_kwargs: object) -> Path:
        return validation._execute_request(request)

    def artifact(_path: str | Path) -> dict[str, Any]:
        return identity.copy()

    def load_system(_path: str | Path) -> Policy:
        return actor

    def load_results(path: str | Path, **_kwargs: object) -> SimpleNamespace:
        return saved[Path(path)]

    def rows(path: str | Path, **_kwargs: object) -> list[dict[str, Any]]:
        return game_rows[Path(path)]

    monkeypatch.setattr(validation, "_artifact", artifact)
    monkeypatch.setattr(checkpoints, "load_system", load_system)
    monkeypatch.setattr(validation, "_run_pass", run_pass)
    monkeypatch.setattr(evaluator, "_run_evaluation", execute)
    monkeypatch.setattr(results, "load_results", load_results)
    monkeypatch.setattr(validation, "_rows", rows)
    return {"actor": actor_path, "calls": calls, "saved": saved, "root": tmp_path}


def _capture(fake_m8: dict[str, Any], root_seed: int, pairs: int = 4) -> dict[str, Any]:
    directory = fake_m8["root"] / f"root-{root_seed}-{pairs}"
    result = validation.validate_random(
        fake_m8["actor"], output_dir=directory, root_seed=root_seed, seed_pairs=pairs
    )
    return {
        **result,
        "actor_path": str(fake_m8["actor"]),
        "summary_path": str(directory / "validation_summary.json"),
        "reference_path": None,
        "reused_initialization": False,
        "elapsed_seconds": 0.0,
        "training_seconds": 0.0,
        "wall_seconds": 0.0,
    }


def test_confirmation_root_reaches_native_m8_and_verifies_saved_evidence(
    fake_m8: dict[str, Any],
) -> None:
    routine = _capture(fake_m8, 19_043_001)
    confirmation = _capture(fake_m8, 19_044_791, pairs=20)
    assert [call["seed"] for call in fake_m8["calls"]] == [19_043_001, 19_044_791]
    assert [len(call["specs"]) for call in fake_m8["calls"]] == [40, 200]
    assert routine["task_id"] != confirmation["task_id"]
    assert (
        validation.verify_random_result(
            routine, actor_digest="fixed-inference", seed_pairs=4
        )["root"]
        == 19_043_001
    )
    assert (
        validation.verify_random_result(
            confirmation,
            actor_digest="fixed-inference",
            seed_pairs=20,
            root_seed=19_044_791,
        )["root"]
        == 19_044_791
    )
    with pytest.raises(ValueError, match="summary or task"):
        validation.verify_random_result(
            confirmation, actor_digest="fixed-inference", seed_pairs=20
        )
    before = {
        path: path.read_bytes() for path in fake_m8["root"].rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError, match="different scientific"):
        validation.validate_random(
            fake_m8["actor"],
            output_dir=Path(routine["summary_path"]).parent,
            seed_pairs=4,
            root_seed=19_044_791,
        )
    assert len(fake_m8["calls"]) == 2
    assert all(path.read_bytes() == content for path, content in before.items())
    entry = next(
        iter(
            fake_m8["saved"][fake_m8["calls"][-1]["directory"]]
            .metadata["passes"]
            .values()
        )
    )
    entry["details"]["evaluation_contract"]["options"]["seed"] = 19_043_001
    with pytest.raises(ValueError, match="native evaluation settings"):
        validation.verify_random_result(
            confirmation,
            actor_digest="fixed-inference",
            seed_pairs=20,
            root_seed=19_044_791,
        )


def test_saved_random_sampling_bounds_rebuild_from_bound_pass_facts(
    fake_m8: dict[str, Any],
) -> None:
    record = _capture(fake_m8, 19_043_001, pairs=1)
    assert record["ci_low"] is record["ci_high"] is None
    assert record["conditional_ci_low"] == record["conditional_ci_high"] == 0.5
    assert record["independent_blocks"] == record["declared_blocks"] == 5
    assert "stratum" in record["interval_status"]
    before = {
        path: path.read_bytes() for path in fake_m8["root"].rglob("*") if path.is_file()
    }
    verified = validation.verify_random_result(
        record, actor_digest="fixed-inference", seed_pairs=1
    )
    assert verified["sampling_evidence"] == record["sampling_evidence"]
    assert verified["ci_low"] is verified["ci_high"] is None
    assert all(path.read_bytes() == content for path, content in before.items())
    facts_path = Path(record["pass_paths"][0]).parent / "sampling_facts.json"
    facts = json.loads(facts_path.read_text())
    facts["system_ids"]["team_a"] = "different-actor"
    facts_path.write_text(json.dumps(facts))
    with pytest.raises(ValueError, match="Sampling facts"):
        validation.verify_random_result(
            record, actor_digest="fixed-inference", seed_pairs=1
        )


def test_routine_and_confirmation_game_start_keys_do_not_overlap(
    fake_m8: dict[str, Any],
) -> None:
    from marl_battlegrounds.evaluation.evaluate import episode_keys

    _capture(fake_m8, 19_043_001)
    _capture(fake_m8, 19_044_791, pairs=20)
    key_sets = []
    for call in fake_m8["calls"]:
        specs = call["specs"]
        seed_ids = jnp.asarray(sorted({s.random_seed_id for s in specs}), jnp.uint32)
        zeros = jnp.zeros_like(seed_ids)
        keys: set[tuple[int, ...]] = set()
        for stream in (0, 1, 2, 3):
            actual = episode_keys(jax.random.key(call["seed"]), seed_ids, zeros, stream)
            keys.update(
                tuple(map(int, row)) for row in np.asarray(jax.random.key_data(actual))
            )
        assert len(keys) == len(seed_ids) * 4
        assert {s.map_id for s in specs} == set(range(42, 47))
        for seed_id in map(int, seed_ids):
            pair = [s for s in specs if s.random_seed_id == seed_id]
            assert len(pair) == 2
            assert {s.spawn_locations for s in pair} == {0, 1}
            assert pair[0].map_id == pair[1].map_id
        key_sets.append(keys)
    assert not key_sets[0].intersection(key_sets[1])


def test_shared_initialization_keeps_original_identity_and_expected_root(
    fake_m8: dict[str, Any],
) -> None:
    original = _capture(fake_m8, 19_044_791, pairs=20)
    reference = fake_m8["root"] / "shared-initialization.json"
    reference.write_text(json.dumps(original))
    read = validation.read_random_initialization(
        reference, actor_digest="fixed-inference", seed_pairs=20, root_seed=19_044_791
    )
    assert read == original
    reused = {
        **original,
        "reused_initialization": True,
        "reference_path": str(reference),
    }
    verified = validation.verify_random_result(
        reused, actor_digest="fixed-inference", seed_pairs=20, root_seed=19_044_791
    )
    for field in ("task_id", "checkpoint_id", "actor_digest", "pass_paths"):
        assert verified[field] == original[field]
    with pytest.raises(ValueError, match="summary or task"):
        validation.verify_random_result(
            reused, actor_digest="fixed-inference", seed_pairs=20
        )
    with pytest.raises(ValueError, match="summary or task"):
        validation.read_random_initialization(
            reference, actor_digest="fixed-inference", seed_pairs=20
        )
    altered = {**reused, "checkpoint_id": "pretend-new-checkpoint"}
    with pytest.raises(ValueError, match="changed or chains"):
        validation.verify_random_result(
            altered, actor_digest="fixed-inference", seed_pairs=20, root_seed=19_044_791
        )


def test_random_records_verify_only_under_their_own_scoring_rule(
    fake_m8: dict[str, Any],
) -> None:
    from marl_battlegrounds.training.analysis import summarize_validation

    current = _capture(fake_m8, 19_043_001)
    assert current["schema_version"] == 3 and current["red_zone_depth"] == 5.0
    entry = next(
        iter(
            fake_m8["saved"][fake_m8["calls"][-1]["directory"]]
            .metadata["passes"]
            .values()
        )
    )
    assert entry["details"]["evaluation_contract"]["options"]["red_zone_depth"] == 5.0
    options: dict[str, Any] = {"actor_digest": "fixed-inference", "seed_pairs": 4}
    assert validation.verify_random_result(current, **options)["red_zone_depth"] == 5.0
    for depth in (None, 6.0):
        with pytest.raises(ValueError, match="summary or task"):
            validation.verify_random_result(current, **options, red_zone_depth=depth)
    # Rewrite the same games as a record saved before the rule: a schema 1
    # task, no depth option and kills read from scores.
    legacy_task = _task()
    assert legacy_task["schema_version"] == 1 and "red_zone_depth" not in legacy_task
    entry["details"]["evaluation_contract"]["options"].pop("red_zone_depth")
    entry["details"]["pass_id"] = validation.validation_pass_id(
        legacy_task["task_id"], "Random"
    )
    rows = validation._rows(
        Path(current["pass_paths"][0]), pass_id="unused-by-fake", opponent="Random"
    )
    legacy_summary = {
        **legacy_task,
        **summarize_validation(
            rows, maps=validation.VALIDATION_MAPS, opponents=("Random",), seed_pairs=4
        ),
        "pass_paths": current["pass_paths"],
    }
    directory = Path(current["summary_path"]).parent
    (directory / "task.json").write_text(json.dumps(legacy_task))
    (directory / "validation_summary.json").write_text(json.dumps(legacy_summary))
    legacy = {
        **legacy_summary,
        **{key: current[key] for key in validation._RANDOM_CAPTURE_FIELDS},
    }
    assert (
        validation.verify_random_result(legacy, **options, red_zone_depth=None)
        == legacy_summary
    )
    with pytest.raises(ValueError, match="summary or task"):
        validation.verify_random_result(legacy, **options)
    reference = fake_m8["root"] / "legacy-initialization.json"
    reference.write_text(json.dumps(legacy))
    assert (
        validation.read_random_initialization(reference, **options, red_zone_depth=None)
        == legacy
    )
    with pytest.raises(ValueError, match="summary or task"):
        validation.read_random_initialization(reference, **options)


def _continuation_panel(tmp_path: Path) -> validation.FrozenPanel:
    return validation.FrozenPanel(
        tmp_path / "panel.json",
        "panel",
        (validation.PanelMember("Opponent", registration_id="opponent"),),
        True,
        schema_version=2,
        roots={"routine": 11, "initialization": 12, "confirmation": 13},
    )


def _continuation_config(total: int) -> dict[str, Any]:
    return {
        "method": "mappo",
        "num_envs": 2,
        "total_env_steps": total,
        "ppo": {"rollout_length": 4},
        "validation_fractions": (0.5, 1.0),
        "routine_seed_pairs": 2,
        "confirmation_seed_pairs": 3,
        "red_zone_depth": 5.0,
    }


def test_continuation_validation_keeps_absolute_points_and_partial_block_origin(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    from marl_battlegrounds.training._continuation_schedules import LearnerContinuation

    panel = _continuation_panel(tmp_path)
    parent = validation.run_validation_declaration(_continuation_config(10), panel)
    context = {
        "start_env_steps": 10,
        "parent_validation_declaration": parent,
        "learner": {
            "schema_version": 1,
            **asdict(LearnerContinuation("mappo", 5, 2, 2)),
        },
        "changes": {
            "validation": {"env_steps": [11, 17, 20], "roots": {"routine": 21}}
        },
    }
    config = _continuation_config(20)
    declared = validation.run_validation_declaration(
        config, panel, continuation=context
    )
    assert declared["points"] == [
        {"requested_steps": [11, 17], "env_steps": 18, "update_index": 3},
        {"requested_steps": [20], "env_steps": 20, "update_index": 4},
    ]
    assert declared["roots"] == {
        "routine": 21,
        "initialization": 12,
        "confirmation": 13,
    }
    task = validation.declared_panel_task(
        declared,
        panel,
        checkpoint_id="actor",
        actor_digest="weights",
        env_steps=18,
        purpose="routine",
    )
    assert task["root"] == 21 and task["seed_pairs"] == 2
    assert (
        validation.saved_validation_declaration(
            {
                "config": config,
                "continuation": context,
                "validation_declaration": declared,
            },
            panel,
        )
        == declared
    )
    altered = {**declared, "roots": {**declared["roots"], "routine": 22}}
    with pytest.raises(ValueError, match="frozen inputs"):
        validation.saved_validation_declaration(
            {
                "config": config,
                "continuation": context,
                "validation_declaration": altered,
            },
            panel,
        )
    context["changes"] = {"validation": {"env_steps": []}}
    final_only = validation.run_validation_declaration(
        config, panel, continuation=context
    )
    assert [point["env_steps"] for point in final_only["points"]] == [20]


@pytest.mark.parametrize("points", [[10], [21], [19, 18], [True]])
def test_continuation_validation_rejects_invalid_future_targets(
    tmp_path: Path,
    points: list[int],
) -> None:
    from dataclasses import asdict

    from marl_battlegrounds.training._continuation_schedules import LearnerContinuation

    panel = _continuation_panel(tmp_path)
    config = _continuation_config(20)
    context = {
        "start_env_steps": 10,
        "parent_validation_declaration": validation.run_validation_declaration(
            _continuation_config(10), panel
        ),
        "learner": {
            "schema_version": 1,
            **asdict(LearnerContinuation("mappo", 5, 2, 2)),
        },
        "changes": {"validation": {"env_steps": points}},
    }
    with pytest.raises(ValueError):
        validation.run_validation_declaration(config, panel, continuation=context)


def test_schema_one_continuation_roots_stay_fixed(tmp_path: Path) -> None:
    panel = validation.FrozenPanel(
        tmp_path / "panel.json",
        "old-panel",
        (
            validation.PanelMember("Halfway", actor_digest="one"),
            validation.PanelMember("Final", actor_digest="two"),
        ),
        True,
    )
    declaration = validation.run_validation_declaration(_continuation_config(20), panel)
    task = validation.declared_panel_task(
        declaration,
        panel,
        checkpoint_id="actor",
        actor_digest="weights",
        env_steps=20,
        purpose="confirmation",
    )
    assert task["root"] == 19_043_003
    declaration["roots"]["confirmation"] = 99
    with pytest.raises(ValueError, match="Historical panels"):
        validation.declared_panel_task(
            declaration,
            panel,
            checkpoint_id="actor",
            actor_digest="weights",
            env_steps=20,
            purpose="confirmation",
        )
