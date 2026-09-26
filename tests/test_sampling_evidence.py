"""Check sampling claims without mistaking repeated outcomes for independence.

Declared deterministic conditions provide no random sampling units. Unknown
methods remain unknown. Known stochastic groups stay whole even when captured
actions or trajectories repeat. Native methods are identified by actual hooks,
not display labels, and malformed saved evidence fails before use.
"""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
from marl_battlegrounds.evaluation.sampling_evidence import (
    combine_sampling_facts,
    method_sampling_fact,
    summarize_sampling_evidence,
    validate_sampling_evidence,
)


def test_repeated_stochastic_trajectories_do_not_remove_sampling_units() -> None:
    evidence = summarize_sampling_evidence(
        [1, 2, 3, 4],
        scheduled_games=6,
        sampling_units={1: "pair-a", 2: "pair-a", 3: "pair-b", 4: "pair-b"},
        determinism="stochastic",
        basis="Independent paired keys under fixed actors",
        trajectory_digests={game: "a" * 64 for game in range(1, 5)},
        action_digests={game: "b" * 64 for game in range(1, 5)},
    )
    assert evidence["scheduled_games"] == 6
    assert evidence["completed_games"] == 4
    assert evidence["supported_independent_sampling_units"] == 2
    assert evidence["interval_status"] == "Available"
    assert evidence["trajectory_observations"]["repeated_games"] == 3
    assert evidence["action_observations"]["repeated_games"] == 3
    assert validate_sampling_evidence(evidence, completed_games=4) == evidence


@pytest.mark.parametrize("kind, expected", [("deterministic", 0), ("unknown", None)])
def test_fixed_repetition_and_unknown_sampling_never_claim_independent_games(
    kind: str, expected: int | None
) -> None:
    evidence = summarize_sampling_evidence(
        [1, 2, 3, 4],
        scheduled_games=4,
        sampling_units={game: game for game in range(1, 5)},
        determinism=kind,  # pyright: ignore[reportArgumentType]
        basis="Declared fixed method and deterministic initial conditions",
    )
    assert evidence["declared_sampling_units"] == 4
    assert evidence["supported_independent_sampling_units"] == expected
    assert evidence["interval_status"].startswith("Unavailable:")
    assert evidence["trajectory_observations"]["distinct_digests"] is None
    assert validate_sampling_evidence(evidence) == evidence


def test_game_ids_and_whole_group_coverage_are_checked() -> None:
    with pytest.raises(ValueError, match="unique"):
        summarize_sampling_evidence([1, 1], scheduled_games=2)
    with pytest.raises(ValueError, match="every completed game"):
        summarize_sampling_evidence([1, 2], scheduled_games=2, sampling_units={1: 1})
    with pytest.raises(ValueError, match="unknown completed game"):
        summarize_sampling_evidence(
            [1], scheduled_games=1, action_digests={2: "a" * 64}
        )
    with pytest.raises(ValueError, match="source or declaration"):
        summarize_sampling_evidence([1], scheduled_games=1, determinism="deterministic")
    mixed = summarize_sampling_evidence(
        [1, 2, 3, 4],
        scheduled_games=4,
        sampling_units={1: None, 2: None, 3: "random", 4: "random"},
        determinism="stochastic",
        basis="Two fixed games plus one random paired block",
    )
    assert mixed["supported_independent_sampling_units"] == 1
    assert mixed["interval_status"] == "Unavailable: Too few independent sampling units"


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"completed_games": 5},
        {"supported_independent_sampling_units": 4},
        {"interval_status": "Available"},
        {"determinism": "A provider said nothing"},
        {"basis": None},
        {
            "action_observations": {
                "recorded_games": 2,
                "distinct_digests": 1,
                "repeated_games": 3,
            }
        },
    ],
)
def test_inconsistent_saved_sampling_metadata_is_refused(
    changes: dict[str, object],
) -> None:
    evidence = summarize_sampling_evidence(
        [1, 2], scheduled_games=2, determinism="deterministic", basis="Fixed setup"
    )
    altered = deepcopy(evidence)
    altered.update(changes)
    with pytest.raises(ValueError):
        validate_sampling_evidence(altered, completed_games=2)


def test_method_facts_use_real_hooks_and_frozen_exploration_not_names() -> None:
    from marl_battlegrounds.baselines import ppo, pqn, qmix
    from marl_battlegrounds.evaluation.policy_execution import policy, shared_policy

    alpha = policy("tdm-alpha")
    assert method_sampling_fact(alpha)["determinism"] == "deterministic"
    assert method_sampling_fact(shared_policy(alpha))["determinism"] == "deterministic"
    assert (
        method_sampling_fact(replace(policy("random"), name="tdm-alpha"))["determinism"]
        == "stochastic"
    )
    assert (
        method_sampling_fact(replace(alpha, execution="host"))["determinism"]
        == "unknown"
    )
    for method in ("mappo", "ippo", "ff_mappo", "ff_ippo"):
        assert (
            method_sampling_fact(ppo.make_ppo_system({}, method=method))["determinism"]
            == "stochastic"
        )

    def unknown_apply(*args: object) -> NoReturn:
        raise AssertionError("Classification must not call the unknown method")

    for greedy, exploring in (
        (
            qmix.make_qmix_system({}, epsilon=0.0),
            qmix.make_qmix_system({}, epsilon=0.1),
        ),
        (
            pqn.make_pqn_system(pqn.PQNInferenceVariables({}, {}), epsilon=0.0),
            pqn.make_pqn_system(pqn.PQNInferenceVariables({}, {}), epsilon=0.1),
        ),
    ):
        assert method_sampling_fact(greedy)["determinism"] == "deterministic"
        assert method_sampling_fact(exploring)["determinism"] == "stochastic"
        unknown = replace(greedy, apply=unknown_apply, name="An unrelated label")
        assert method_sampling_fact(unknown)["determinism"] == "unknown"
    assert (
        combine_sampling_facts(
            [
                {"determinism": "deterministic", "basis": "Fixed native method"},
                {"determinism": "unknown", "basis": "Opaque provider"},
            ]
        )["determinism"]
        == "unknown"
    )


def test_validation_sampling_is_bound_to_saved_pass_and_preserves_old_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib import import_module

    from marl_battlegrounds.evaluation.policy_execution import Policy, policy
    from marl_battlegrounds.training import analysis, checkpoints, validation

    panel = validation.create_panel(
        opponents=("tdm-alpha",), output_dir=tmp_path / "panel"
    )

    def artifact(path: str | Path) -> dict[str, Any]:
        return {
            "checkpoint_id": "checked",
            "actor_digest": "actor",
            "env_steps": 32,
        }

    def load_system(path: str | Path) -> Policy:
        return policy("tdm-beta")

    monkeypatch.setattr(validation, "_artifact", artifact)
    monkeypatch.setattr(checkpoints, "load_system", load_system)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    evaluate, verify = evaluator.evaluate, evaluator._verify_evaluation
    provenance = evaluator.capture_recording_provenance(num_envs=2)

    def fixed_provenance(*, num_envs: int) -> dict[str, Any]:
        return deepcopy(provenance)

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)

    def short(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return evaluate(*args, max_steps=1, **kwargs)

    def short_verify(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return verify(*args, max_steps=1, **kwargs)

    monkeypatch.setattr(evaluator, "evaluate", short)
    monkeypatch.setattr(evaluator, "_verify_evaluation", short_verify)
    folder = tmp_path / "task"
    result = validation.validate_checkpoint(
        "unused", panel, output_dir=folder, seed_pairs=2, num_envs=2
    )
    assert result["sampling_evidence"]["supported_independent_sampling_units"] == 0
    assert result["ci_low"] is result["ci_high"] is None
    path = Path(result["pass_paths"][0])
    pass_id = validation.validation_pass_id(result["task_id"], panel.members[0].name)
    sidecar = path.parent / "sampling_facts.json"
    saved = sidecar.read_bytes()
    fact = validation._read_pass_sampling(
        path, task_id=result["task_id"], pass_id=pass_id
    )
    assert fact["determinism"] == "deterministic"

    altered = json.loads(saved)
    altered["system_ids"]["team_a"] = "changed-registration"
    sidecar.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="registrations"):
        validation._read_pass_sampling(path, task_id=result["task_id"], pass_id=pass_id)
    sidecar.write_bytes(saved)

    raw = validation._rows(
        path, pass_id=pass_id, opponent=panel.members[0].name, kills=True
    )
    historical = {
        **json.loads((folder / "task.json").read_text()),
        **analysis.summarize_validation(
            raw,
            maps=validation.VALIDATION_MAPS,
            opponents=(panel.members[0].name,),
            seed_pairs=2,
            independent_opponents=True,
            actual_kills=True,
        ),
        "pass_paths": result["pass_paths"],
    }
    summary = folder / "validation_summary.json"
    original_bytes = (json.dumps(historical, separators=(",", ":")) + "\n").encode()
    summary.write_bytes(original_bytes)
    resumed = validation.validate_checkpoint(
        "unused", panel, output_dir=folder, seed_pairs=2, num_envs=2
    )
    assert resumed == historical
    assert summary.read_bytes() == original_bytes

    summary.unlink()
    sidecar.unlink()
    reanalyzed = validation.validate_checkpoint(
        "unused", panel, output_dir=folder, seed_pairs=2, num_envs=2
    )
    assert reanalyzed["sampling_evidence"]["determinism"] == "Determinism Unknown"
    assert (
        reanalyzed["sampling_evidence"]["supported_independent_sampling_units"] is None
    )
    assert reanalyzed["ci_low"] is reanalyzed["ci_high"] is None
    assert not sidecar.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"task_id": "different-task"},
        {"pass_id": "different-pass"},
        {"system_ids": {"team_a": "other", "team_b": "b"}},
        {"methods": {"team_a": None, "team_b": None}},
    ],
)
def test_pass_sampling_rejects_malformed_or_misbound_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    from marl_battlegrounds.evaluation import results
    from marl_battlegrounds.training import validation

    saved = {
        "schema_version": 1,
        "task_id": "task",
        "pass_id": "pass",
        "system_ids": {"team_a": "a", "team_b": "b"},
        "methods": {
            team: {"determinism": "deterministic", "basis": "Checked native hook"}
            for team in ("team_a", "team_b")
        },
    }

    def saved_results(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            metadata={"passes": {"pass": {"system_ids": saved["system_ids"]}}}
        )

    monkeypatch.setattr(results, "load_results", saved_results)
    (tmp_path / "sampling_facts.json").write_text(json.dumps({**saved, **changes}))
    with pytest.raises(ValueError, match=r"Sampling facts|Method sampling facts"):
        validation._read_pass_sampling(tmp_path / "run", task_id="task", pass_id="pass")
