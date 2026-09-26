"""Check the finite MAPPO study without running a learner or a GPU.

These host tests check fixed eight-recipe and historical adaptive budgets,
three-seed eligibility, fixed run order, source/clock binding and the complete
unattended phase sequence.
Fake workers retain the ordinary job records so resume, numerical failure,
infrastructure failure and assessment separation are checked through the real
controller. Process shutdown remains owned by the separately tested launcher.
Tiny CPU actor exports check frozen identities without training or playing
games. Another valid export at the selected path must fail before any game
writer is created or recovered. Returned summaries cannot hide a different
actor, and final exports must match their completed learner descriptions.
Fresh processes check CPU metadata placement, restoration after errors, lazy
status and the separate CPU controller and GPU worker environments.
The declaration pins red_zone_depth 0.0 (the study's one-point scoring), every
training config and assessment request carries it, assessment plays at the
request's depth, and a summary at another depth is rejected unpublished.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from marl_battlegrounds.training import search
from marl_battlegrounds.training._run_io import atomic_json


@pytest.fixture
def declared() -> dict[str, Any]:
    result = search.declaration()
    result.pop("fixed_discovery_steps")
    result.update(recipes=search.recipes(), tiers=[12, 10, 8], target_seconds=43_200)
    return result


def _costs(seconds_per_transition: float = 1 / 30_000) -> dict[str, Any]:
    return {
        "recipes": {
            recipe["recipe_id"]: {
                "seconds_per_transition": seconds_per_transition,
                "setup_seconds": 30.0,
                "save_seconds": 0.2,
                "report_seconds": 0.5,
                "run_bytes": 1_000_000,
                "checkpoint_bytes": 300_000,
                "checkpoint_metadata_bytes": 10_000,
                "actor_bytes": 50_000,
                "log_bytes_per_update": 1000,
            }
            for recipe in search.recipes()
        },
        "validation_seconds": {
            "alpha40": 5.0,
            "alpha80": 8.0,
            "alpha100": 10.0,
            "alpha200": 15.0,
            "alpha400": 25.0,
            "beta200": 20.0,
        },
        "validation_setup_seconds": 5.0,
        "assessment_seconds": {"alpha": 35.0, "beta": 30.0},
        "validation_bytes": {
            name: games * 1000
            for name, games in (
                ("alpha40", 40),
                ("alpha80", 80),
                ("alpha100", 100),
                ("alpha200", 200),
                ("alpha400", 400),
                ("beta200", 200),
            )
        },
    }


def test_declaration_keeps_all_main_levers_and_three_fresh_seed_blocks(
    declared: dict[str, Any],
) -> None:
    declared = search.declaration()
    assert [row["recipe_id"] for row in declared["recipes"]] == [
        f"c{i:02}" for i in range(8)
    ]
    assert declared["tiers"] == [8]
    assert declared["fixed_discovery_steps"] == 20_054_016
    assert declared["target_seconds"] == declared["hard_stop_seconds"] == 50_400
    assert declared["numerical_stop_seconds"] == 50_100
    assert len(declared["discovery_seeds"]) == len(declared["finalist_seeds"]) == 3
    assert not set(declared["discovery_seeds"]) & set(declared["finalist_seeds"])
    base = declared["base_config"]
    assert base["ppo"]["value_normalization"] is True
    assert base["ppo"]["spawn_frame"] == "left"
    assert base["pinned_opponent"] == "tdm-alpha"
    assert base["ppo"]["critic_lr"] == 0.00025
    assert base["validation_fractions"] == [0.25, 0.5, 0.75, 1.0]
    assert base["random_diagnostic_seed_pairs"] is None
    # The frozen study keeps one point per death, not the new 5.0 default.
    assert base["red_zone_depth"] == 0.0


@pytest.mark.parametrize("cost", [1 / 30_000, 1 / 22_000, 1 / 20_000])
def test_fixed_eight_budget_does_not_expand_into_spare_time(cost: float) -> None:
    declaration = search.declaration()
    budgets = search.resolve_budgets(declaration, _costs(cost))
    assert budgets["tier"] == 8
    assert budgets["discovery_steps"] == 20_054_016
    assert budgets["finalist_steps"] == 40_108_032
    assert len(budgets["discovery_order"]) == 24
    assert budgets["training_transitions"] == 721_944_576
    assert budgets["reserved_seconds"] <= 50_400
    assert {row["recipe_id"] for row in budgets["discovery_order"]} == {
        f"c{i:02}" for i in range(8)
    }
    with pytest.raises(ValueError, match="eight-recipe"):
        search.resolve_budgets(declaration, _costs(1 / 10_000))


@pytest.mark.parametrize(
    "steps", [None, True, 20_054_016.0, -1, 19_922_944, 20_054_017]
)
def test_invalid_fixed_experience_is_rejected(steps: object) -> None:
    declaration = search.declaration()
    declaration["fixed_discovery_steps"] = steps
    with pytest.raises(ValueError, match="fixed_discovery_steps"):
        search.resolve_budgets(declaration, _costs())


@pytest.mark.parametrize(
    "cost, tier", [(1 / 30_000, 12), (1 / 25_000, 10), (1 / 22_000, 8)]
)
def test_timing_selects_widest_feasible_tier_then_largest_budget(
    declared: dict[str, Any],
    cost: float,
    tier: int,
) -> None:
    before = random.getstate()
    budgets = search.resolve_budgets(declared, _costs(cost))
    assert budgets["tier"] == tier
    assert random.getstate() == before
    assert budgets["discovery_steps"] >= 20_000_000
    assert budgets["discovery_steps"] % 131_072 == 0
    assert budgets["finalist_steps"] == 2 * budgets["discovery_steps"]
    assert budgets["reserved_seconds"] <= 43_200
    assert budgets["maximum_validation_games"] == 1500 * tier + 9600
    assert (
        budgets["training_transitions"] == (3 * tier + 12) * budgets["discovery_steps"]
    )
    larger = {**declared, "minimum_steps": budgets["discovery_steps"] + 131_072}
    try:
        next_result = search.resolve_budgets(larger, _costs(cost))
    except ValueError:
        assert tier == 8
    else:
        assert next_result["tier"] < tier
    for seed in declared["discovery_seeds"]:
        block = [row for row in budgets["discovery_order"] if row["seed"] == seed]
        assert len(block) == tier
        assert {row["recipe_id"] for row in block} == {f"c{i:02}" for i in range(tier)}
    assert search.resolve_budgets(declared, _costs(cost)) == budgets


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_bad_or_infeasible_costs_cannot_launch_a_smaller_unapproved_study(
    declared: dict[str, Any],
    value: object,
) -> None:
    costs = _costs()
    costs["recipes"]["c00"]["seconds_per_transition"] = value
    with pytest.raises(ValueError, match="measurement"):
        search.resolve_budgets(declared, costs)
    with pytest.raises(ValueError, match="eight-recipe"):
        search.resolve_budgets(declared, _costs(1 / 10_000))


def test_rewards_do_not_enter_budget_choice_and_configuration_is_copied(
    declared: dict[str, Any],
) -> None:
    costs = _costs()
    expected = search.resolve_budgets(declared, costs)
    costs["rewards"] = {"c00": 1e30, "c11": -1e30}
    actual = search.resolve_budgets(declared, costs)
    assert actual["discovery_steps"] == expected["discovery_steps"]
    assert actual["tier"] == expected["tier"]
    original = copy.deepcopy(declared)
    config = search.run_config(
        declared, declared["recipes"][5], seed=91, steps=131_072, panel="/panel"
    )
    assert config["pinned_opponent_share"] == 0.3
    assert config["ppo"]["actor_lr"] == 0.00025
    assert declared == original
    for length in (32, 64):
        assert all(
            (expected["discovery_steps"] * part // 4) % (512 * length) == 0
            for part in (1, 2, 3, 4)
        )


def test_storage_forecast_includes_retained_checkpoints_and_recorded_games(
    declared: dict[str, Any],
) -> None:
    costs = _costs()
    expected = search.resolve_budgets(declared, costs)
    more_checkpoints = copy.deepcopy(costs)
    more_checkpoints["recipes"]["c00"]["checkpoint_bytes"] *= 2
    more_games = copy.deepcopy(costs)
    more_games["validation_bytes"]["alpha400"] *= 10
    for changed in (more_checkpoints, more_games):
        budget = search.resolve_budgets(declared, changed)
        assert budget["discovery_steps"] == expected["discovery_steps"]
        assert budget["minimum_free_bytes"] > expected["minimum_free_bytes"]
    assert expected["minimum_free_bytes"] >= (
        1.25 * expected["projected_retained_bytes"] + 2_000_000_000 - 2
    )


def test_forecast_charges_validation_setup_once_and_assessment_cold_workers(
    declared: dict[str, Any],
) -> None:
    setup = _costs()
    setup["validation_setup_seconds"] += 5
    repeated = _costs()
    repeated["validation_seconds"]["alpha40"] += 1
    repeated["validation_seconds"]["alpha80"] += 1
    first, second = (
        search.resolve_budgets(declared, setup),
        search.resolve_budgets(declared, repeated),
    )
    for field in ("tier", "discovery_steps", "forecast_seconds", "reserved_seconds"):
        assert first[field] == second[field]
    costs = _costs()
    expected = search.resolve_budgets(declared, costs)
    costs["validation_seconds"]["alpha400"] *= 100
    costs["validation_seconds"]["beta200"] *= 100
    result = search.resolve_budgets(declared, costs)
    assert result["forecast_seconds"] == expected["forecast_seconds"]
    costs["assessment_seconds"]["alpha"] *= 5
    result = search.resolve_budgets(declared, costs)
    assert result["discovery_steps"] < expected["discovery_steps"]


def _row(
    identifier: str, seed: int, score: float, kills: float = 0.0, steps: int = 100
) -> dict[str, Any]:
    return {
        "recipe_id": identifier,
        "seed": seed,
        "complete": True,
        "score": score,
        "mean_kill_difference": kills,
        "env_steps": steps,
    }


def test_recipe_ranking_uses_all_seeds_and_exact_declared_tiebreaks() -> None:
    seeds = (1, 2, 3)
    rows = [
        _row(identifier, seed, score, kills, steps)
        for identifier, score, kills, steps in (
            ("a", 0.5, 5.0, 100),
            ("b", 0.5001, -20.0, 100),
            ("c", 0.5, 6.0, 100),
            ("d", 0.5, 6.0, 50),
        )
        for seed in seeds
    ]
    assert [
        row["recipe_id"]
        for row in search.rank_recipes(
            rows, identifiers=("a", "b", "c", "d"), seeds=seeds
        )
    ] == ["b", "d", "c", "a"]
    rows[3]["complete"] = False
    assert "b" not in {
        row["recipe_id"]
        for row in search.rank_recipes(
            rows, identifiers=("a", "b", "c", "d"), seeds=seeds
        )
    }
    with pytest.raises(ValueError, match="duplicate"):
        search.rank_recipes(
            [*rows, rows[0]], identifiers=("a", "b", "c", "d"), seeds=seeds
        )
    assert search.rank_recipes(rows[:2], identifiers=("a",), seeds=seeds) == []


def _package(
    tmp_path: Path, declared: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> Path:
    root = tmp_path / "package"
    root.mkdir()
    costs = _costs()
    budgets = search.resolve_budgets(declared, costs)
    records: tuple[tuple[str, dict[str, Any]], ...] = (
        ("declaration.json", declared),
        ("calibration.json", costs),
        ("budgets.json", budgets),
        (
            "qualified.json",
            {
                "budget_digest": search._digest(budgets),
                "timing_digest": search._digest(costs),
                "source_files_digest": search._digest({}),
            },
        ),
        (search._MANIFEST, {"origin": {"files": {}}}),
    )
    for name, value in records:
        atomic_json(root / name, value)
    started = time.time()
    clock = {
        "started_at_seconds": started,
        "numerical_deadline": started + declared["numerical_stop_seconds"],
        "hard_deadline": started + declared["hard_stop_seconds"],
        "budget_digest": search._digest(budgets),
    }
    atomic_json(root / "launch_time.json", clock)
    atomic_json(
        root / "study.json",
        {"schema_version": 1, "status": "starting", "jobs": {}, **clock},
    )

    def verified(root: Path, *, full: bool = True) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(search, "verify_package", verified)

    def actor_identity(path: str | Path) -> dict[str, Any]:
        return search._read(Path(path) / "identity.json")

    def checkpoint_description(path: str | Path) -> dict[str, Any]:
        return search._read(Path(path))

    monkeypatch.setattr(search, "_actor_identity", actor_identity)
    from marl_battlegrounds.training import checkpoints

    monkeypatch.setattr(
        checkpoints, "read_checkpoint_description", checkpoint_description
    )
    return root


def _fake_worker(
    calls: list[str],
    *,
    numerical_failure: str | None = None,
    infrastructure_failure: str | None = None,
) -> Callable[..., None]:
    def run(
        root: Path, job: Path, *, mode: str, deadline: float, resume: bool = False
    ) -> None:
        assert deadline > time.time()
        calls.append(job.name)
        if job.name == infrastructure_failure:
            raise OSError("Saved game writer failed")
        worker = {"pid": 42, "start_ticks": "fake"}
        atomic_json(
            job / "process.json",
            {"state": "exited", "exit_code": 0, "trainer": worker},
        )
        if job.name == numerical_failure:
            atomic_json(
                job / "failure.json",
                {
                    "kind": "numerical",
                    "error": "finite guard",
                    "config_digest": search._digest(search._read(job / "config.json")),
                    "worker": worker,
                },
            )
            raise RuntimeError("Rejected numerical update")
        if mode == "assessment":
            frozen = search._read(root / "final_selection.json")
            assert frozen["winner"] == "c01"
            request = search._read(job / "request.json")
            assert request["red_zone_depth"] == 0.0
            # Reverse the assessment ordering: it must never change selection.
            atomic_json(
                job / "result.json",
                {
                    **request,
                    **{
                        key: request["expected_actor"][key]
                        for key in ("actor_digest", "checkpoint_id", "env_steps")
                    },
                    "complete": True,
                    "score": 1.0 if request["recipe_id"] == "c00" else 0.0,
                    "request_digest": search._digest(request),
                },
            )
            return
        case = search._read(job / "case.json")
        run_dir = job / "run"
        run_dir.mkdir()
        score = 0.9 if case["recipe_id"] == "c01" else 0.5
        selected = {
            **_row(case["recipe_id"], case["seed"], score),
            "env_steps": case["steps"] // 2,
            "checkpoint_id": "selected",
            "actor_digest": "selected-weights",
        }
        config = search._read(job / "config.json")
        assert config["red_zone_depth"] == 0.0
        for label, steps in (
            ("selected", selected["env_steps"]),
            ("final", case["steps"]),
        ):
            actor_dir = run_dir / "actors" / label
            actor_dir.mkdir(parents=True)
            atomic_json(
                actor_dir / "identity.json",
                {
                    "actor_digest": f"{label}-weights",
                    "weight_digest": f"{label}-weights",
                    "checkpoint_id": label,
                    "artifact_id": f"{label}-artifact",
                    "run_id": case["name"],
                    "seed": case["seed"],
                    "env_steps": steps,
                    "input_scale": config["ppo"]["input_scale"],
                    "spawn_frame": config["ppo"]["spawn_frame"],
                },
            )
        boundary = run_dir / "final-checkpoint.json"
        atomic_json(
            boundary,
            {
                "kind": "learner",
                "checkpoint_id": "final",
                "actor_digest": "final-weights",
                "metadata": {"run_id": case["name"], "config": config},
                "counters": {"env_steps": case["steps"]},
            },
        )
        atomic_json(run_dir / "selection.json", selected)
        atomic_json(
            run_dir / "status.json",
            {
                "status": "complete",
                "run_id": case["name"],
                "env_steps": case["steps"],
                "selected_actor": str(run_dir / "actors/selected"),
                "final_actor": str(run_dir / "actors/final"),
                "latest_checkpoint": str(boundary),
            },
        )

    return run


@pytest.mark.parametrize("fixed_eight", [False, True])
def test_fake_workers_complete_all_phases_and_resume_without_new_games(
    tmp_path: Path,
    declared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    fixed_eight: bool,
) -> None:
    if fixed_eight:
        declared = search.declaration()
    root = _package(tmp_path, declared, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(search, "_run_job", _fake_worker(calls))
    result = search.execute_search(root)
    assert result["status"] == "complete"
    expected_jobs = 42 if fixed_eight else 54
    assert len(calls) == expected_jobs
    selection = search._read(root / "final_selection.json")
    assert selection["winner"] == "c01"
    assert len(selection["actors"]) == 6
    assert {row["seed"] for row in selection["actors"]} == set(
        declared["finalist_seeds"]
    )
    assert (root / "reports/assessment.csv").is_file()
    assert search.execute_search(root)["status"] == "complete"
    assert len(calls) == expected_jobs
    assert search._read(root / "final_selection.json") == selection


def test_numerical_failure_is_visible_and_never_averaged_as_a_success(
    tmp_path: Path,
    declared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _package(tmp_path, declared, monkeypatch)
    failed = "discovery-c11-s19046101"
    calls: list[str] = []
    monkeypatch.setattr(
        search, "_run_job", _fake_worker(calls, numerical_failure=failed)
    )
    assert search.execute_search(root)["status"] == "complete"
    assert search._read(root / "jobs" / failed / "result.json")["complete"] is False
    ranking = search._read(root / "discovery_selection.json")["ranking"]
    assert "c11" not in {row["recipe_id"] for row in ranking}
    assert calls.count(failed) == 1


def test_infrastructure_failure_stops_before_selection_and_can_resume(
    tmp_path: Path,
    declared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _package(tmp_path, declared, monkeypatch)
    order = search._read(root / "budgets.json")["discovery_order"]
    calls: list[str] = []
    monkeypatch.setattr(
        search, "_run_job", _fake_worker(calls, infrastructure_failure=order[1]["name"])
    )
    with pytest.raises(OSError, match="writer"):
        search.execute_search(root)
    assert len(calls) == 2
    assert search._read(root / "study.json")["status"] == "incomplete"
    assert not (root / "final_selection.json").exists()
    original_clock = search._read(root / "launch_time.json")
    monkeypatch.setattr(search, "_run_job", _fake_worker(calls))
    assert search.execute_search(root)["status"] == "complete"
    assert calls.count(order[0]["name"]) == 1
    assert search._read(root / "launch_time.json") == original_clock


def test_resume_preserves_numerical_failure_before_result_publication(
    tmp_path: Path,
    declared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _package(tmp_path, declared, monkeypatch)
    calls: list[str] = []
    failed = "discovery-c11-s19046101"
    worker = _fake_worker(calls, numerical_failure=failed)

    def interrupted(
        root: Path, job: Path, *, mode: str, deadline: float, resume: bool = False
    ) -> None:
        try:
            worker(root, job, mode=mode, deadline=deadline, resume=resume)
        except RuntimeError as error:
            raise KeyboardInterrupt(
                "Controller interrupted before publication"
            ) from error

    monkeypatch.setattr(search, "_run_job", interrupted)
    with pytest.raises(KeyboardInterrupt):
        search.execute_search(root)
    job = root / "jobs" / failed
    assert (job / "failure.json").is_file()
    assert not (job / "result.json").exists()
    monkeypatch.setattr(search, "_run_job", _fake_worker(calls))
    assert search.execute_search(root)["status"] == "complete"
    assert calls.count(failed) == 1
    assert search._read(job / "result.json")["complete"] is False
    failure = search._read(job / "failure.json")
    failure["worker"]["pid"] += 1
    atomic_json(job / "failure.json", failure)
    with pytest.raises(ValueError, match="config and worker"):
        search._numerical_failure(job)


def test_changed_or_expired_clock_cannot_start_a_worker(
    tmp_path: Path,
    declared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _package(tmp_path, declared, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(search, "_run_job", _fake_worker(calls))
    study = search._read(root / "study.json")
    study["numerical_deadline"] += 1
    atomic_json(root / "study.json", study)
    with pytest.raises(ValueError, match="clock"):
        search.execute_search(root)
    clock = search._read(root / "launch_time.json")
    for key in ("started_at_seconds", "numerical_deadline", "hard_deadline"):
        clock[key] -= 100_000
    atomic_json(root / "launch_time.json", clock)
    atomic_json(root / "study.json", {"status": "starting", "jobs": {}, **clock})
    assert search.execute_search(root)["status"] == "incomplete"
    assert calls == []


def test_changed_timing_or_budget_is_refused_before_work(
    tmp_path: Path,
    declared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _package(tmp_path, declared, monkeypatch)
    budgets = search._read(root / "budgets.json")
    budgets["discovery_steps"] += 131_072
    atomic_json(root / "budgets.json", budgets)
    with pytest.raises(ValueError, match="budgets"):
        search._check_budget(root)


def test_source_and_panel_hashes_are_checked_before_launch(tmp_path: Path) -> None:
    root = tmp_path
    (root / "panels").mkdir()
    atomic_json(root / "declaration.json", {})
    atomic_json(
        root / search._MANIFEST,
        {
            "schema_version": 1,
            "package_path": str(root),
            "declaration_sha256": "wrong",
            "panels": {},
            "scripts": {},
        },
    )
    with pytest.raises(ValueError, match="declaration"):
        search.verify_package(root, full=False)


def test_status_neither_imports_jax_nor_changes_package_files(tmp_path: Path) -> None:
    atomic_json(tmp_path / "study.json", {"status": "starting"})
    before = (tmp_path / "study.json").read_bytes()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(search.__file__).parents[2])
    environment["JAX_PLATFORMS"] = "cuda,cpu"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys; from pathlib import Path; "
            "from marl_battlegrounds.training.search import main, search_status; "
            "search_status(Path(sys.argv[1])); "
            "assert main(['status', sys.argv[1]]) == 0; "
            "assert os.environ['JAX_PLATFORMS'] == 'cuda,cpu'; "
            "assert 'jax' not in sys.modules",
            str(tmp_path),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "study.json").read_bytes() == before


def test_stop_signals_only_verified_live_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "process.json"
    atomic_json(path, {"process": {"pid": 123, "start_ticks": "different"}})

    def live(root: Path) -> list[Path]:
        return [path]

    def dead(record: dict[str, Any]) -> bool:
        return False

    monkeypatch.setattr(search, "_live_records", live)
    monkeypatch.setattr(search._launch, "_alive", dead)
    calls: list[int] = []

    def killed(pid: int, number: int) -> None:
        calls.append(pid)

    monkeypatch.setattr(search.os, "kill", killed)
    assert search.stop_search(tmp_path)["stop_requested_pids"] == []
    assert calls == []


@pytest.fixture(scope="module")
def actor_exports(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    import jax
    import numpy as np
    import numpy.typing as npt

    from marl_battlegrounds.baselines.ppo import initialize_ppo
    from marl_battlegrounds.training.checkpoints import export_system

    root = tmp_path_factory.mktemp("search-actors")
    template = jax.eval_shape(initialize_ppo, jax.random.key(0)).actor_params
    for index, name in enumerate(("first", "second")):

        def filled(
            leaf: jax.ShapeDtypeStruct, value: float = index * 0.01
        ) -> npt.NDArray[np.float32]:
            return cast(
                npt.NDArray[np.float32], np.full(leaf.shape, value, dtype=leaf.dtype)
            )

        variables = jax.tree.map(filled, template)
        export_system(
            variables,
            root / name,
            metadata={
                "run_id": "frozen-run",
                "seed": 123,
                "env_steps": 32768,
                "checkpoint_id": "a" * 64,
            },
            input_scale=0.01,
            spawn_frame="left",
        )
    return root / "first", root / "second"


def _assessment_request(path: Path) -> dict[str, Any]:
    return {
        "actor_path": str(path),
        "expected_actor": search._actor_identity(path),
        "opponent": "alpha",
        "seed_pairs": 40,
        "root_seed": 19046700,
        "red_zone_depth": 0.0,
    }


def test_assessment_rejects_swapped_valid_export_before_writer_recovery(
    tmp_path: Path,
    actor_exports: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.training import validation

    job = tmp_path / "job"
    job.mkdir()
    actor = tmp_path / "selected"
    shutil.copytree(actor_exports[0], actor)
    request = _assessment_request(actor)
    atomic_json(job / "request.json", request)
    actor.rename(tmp_path / "original")
    shutil.copytree(actor_exports[1], actor)
    assert (
        search._actor_identity(actor)["actor_digest"]
        != request["expected_actor"]["actor_digest"]
    )

    def forbidden(*args: object, **kwargs: object) -> dict[str, Any]:
        pytest.fail("Changed actor reached validation and writer recovery")

    monkeypatch.setattr(validation, "validate_checkpoint", forbidden)
    before = {str(path): path.read_bytes() for path in job.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="actor differs from the frozen selection"):
        search._assess(tmp_path, job)
    after = {str(path): path.read_bytes() for path in job.rglob("*") if path.is_file()}
    assert before == after
    assert not (job / "validation").exists()
    assert not (job / "result.json").exists()


@pytest.mark.parametrize("field", ["actor_digest", "checkpoint_id", "env_steps"])
def test_assessment_rejects_returned_identity_before_result_publication(
    tmp_path: Path,
    actor_exports: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    from marl_battlegrounds.training import validation

    job = tmp_path / "job"
    job.mkdir()
    request = _assessment_request(actor_exports[0])
    atomic_json(job / "request.json", request)

    def mismatch(*args: object, **kwargs: object) -> dict[str, Any]:
        return {"complete": True, **request["expected_actor"], field: "different"}

    monkeypatch.setattr(validation, "validate_checkpoint", mismatch)
    with pytest.raises(ValueError, match="summary differs from the frozen actor"):
        search._assess(tmp_path, job)
    assert not (job / "result.json").exists()


@pytest.mark.parametrize("returned", [0.0, 5.0])
def test_assessment_plays_and_checks_the_declared_red_zone_depth(
    tmp_path: Path,
    actor_exports: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    returned: float,
) -> None:
    from marl_battlegrounds.training import validation

    job = tmp_path / "job"
    job.mkdir()
    request = _assessment_request(actor_exports[0])
    atomic_json(job / "request.json", request)
    calls: list[dict[str, Any]] = []

    def assessed(*args: object, **kwargs: object) -> dict[str, Any]:
        calls.append(dict(kwargs))
        return {
            "complete": True,
            **request["expected_actor"],
            "red_zone_depth": returned,
        }

    monkeypatch.setattr(validation, "validate_checkpoint", assessed)
    if returned != request["red_zone_depth"]:
        with pytest.raises(ValueError, match="summary differs"):
            search._assess(tmp_path, job)
        assert not (job / "result.json").exists()
    else:
        search._assess(tmp_path, job)
        assert search._read(job / "result.json")["red_zone_depth"] == 0.0
    assert [call["red_zone_depth"] for call in calls] == [0.0]


@pytest.mark.parametrize("export", ["selected", "final"])
def test_selected_row_rejects_changed_export_identity(
    tmp_path: Path,
    declared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    export: str,
) -> None:
    root = _package(tmp_path, declared, monkeypatch)
    monkeypatch.setattr(search, "_run_job", _fake_worker([]))
    case = search._read(root / "budgets.json")["discovery_order"][0]
    study = search._read(root / "study.json")
    result = search._case(root, study, declared, case)
    assert result["actor_identity"]["checkpoint_id"] == "selected"
    assert result["final_actor_identity"]["checkpoint_id"] == "final"
    identity_path = (
        root / "jobs" / case["name"] / "run/actors" / export / "identity.json"
    )
    identity = search._read(identity_path)
    identity.update(actor_digest="different", weight_digest="different")
    atomic_json(identity_path, identity)
    with pytest.raises(ValueError, match="actor differs"):
        search._selected_row(root / "jobs" / case["name"], case)


@pytest.mark.parametrize("action", ["supervise", "job"])
def test_cli_reserves_complete_cleanup_before_declared_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    deadline = time.time() + 100
    atomic_json(tmp_path / "launch_time.json", {"hard_deadline": deadline})
    atomic_json(tmp_path / search._MANIFEST, {"gpu": {"uuid": "GPU-test"}})
    calls: list[dict[str, Any]] = []

    def supervise(package: Path, command: list[str], **kwargs: object) -> int:
        calls.append(dict(kwargs))
        return 0

    monkeypatch.setattr(search._launch, "supervise_command", supervise)
    arguments = [action, str(tmp_path)]
    if action == "job":
        arguments += [
            "--job",
            str(tmp_path),
            "--mode",
            "assessment",
            "--deadline",
            str(deadline),
        ]
    assert search.main(arguments) == 0
    grace = (
        search._launch._NESTED_STOP_TIMEOUT_SECONDS if action == "supervise" else None
    )
    assert calls[0]["deadline_at"] == deadline - search._launch.cleanup_reserve_seconds(
        stop_grace_seconds=grace
    )
    assert calls[0].get("stop_grace_seconds") == grace
    environment = calls[0]["env"]
    assert environment["JAX_PLATFORMS"] == (
        "cpu" if action == "supervise" else "cuda,cpu"
    )
    assert environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    if action == "job":
        assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        assert environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.85"
        assert environment["JAX_COMPILATION_CACHE_DIR"] == str(
            tmp_path / "compilation-cache"
        )


@pytest.mark.parametrize("options_first", [False, True])
def test_dedicated_cli_selects_cpu_for_host_actions_only(
    tmp_path: Path, options_first: bool
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(search.__file__).parents[2])
    environment["JAX_PLATFORMS"] = "cuda,cpu"
    script = """
import os, runpy, sys
module = 'marl_battlegrounds.training.search'
sys.argv = ['search', 'worker', sys.argv[1]]
try:
    runpy.run_module(module, run_name='__main__')
except SystemExit as error:
    assert error.code == 2
else:
    raise AssertionError('Missing worker arguments were accepted')
assert os.environ['JAX_PLATFORMS'] == 'cuda,cpu'
root = sys.argv[2]
options = ['--repository', root]
options_first = __OPTIONS_FIRST__
sys.argv = ['search', *(options if options_first else []), 'status', root,
            *(options if not options_first else [])]
try:
    runpy.run_module(module, run_name='__main__')
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('The module did not return its command status')
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert 'jax' not in sys.modules
""".replace("__OPTIONS_FIRST__", repr(options_first))
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "operation", ["declaration", "identity", "identity_error", "selected_error"]
)
def test_metadata_imports_use_cpu_and_restore_caller_default(
    tmp_path: Path,
    actor_exports: tuple[Path, Path],
    declared: dict[str, Any],
    operation: str,
) -> None:
    actor = search._actor_identity(actor_exports[0])
    case = {"seed": 123, "steps": 32768, "recipe_id": "c00", "name": "metadata-proof"}
    (tmp_path / "run").mkdir()
    atomic_json(
        tmp_path / "run/status.json",
        {
            "status": "complete",
            "env_steps": 32768,
            "run_id": "frozen-run",
            "selected_actor": str(actor_exports[0]),
            "final_actor": str(actor_exports[0]),
            "latest_checkpoint": str(tmp_path / "missing-checkpoint"),
        },
    )
    atomic_json(tmp_path / "run/selection.json", {**actor, "complete": True})
    atomic_json(tmp_path / "config.json", declared["base_config"])
    atomic_json(
        tmp_path / "expected.json",
        {"actor": actor, "declaration": search.declaration(), "case": case},
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(search.__file__).parents[2])
    environment["JAX_PLATFORMS"] = "cpu"
    script = """
import builtins, json, os, sys
from pathlib import Path
from marl_battlegrounds.training import search
assert 'jax' not in sys.modules
import jax
root, operation = Path(sys.argv[1]), sys.argv[2]
expected = json.loads((root / 'expected.json').read_text())
before_default = jax.config.jax_default_device
before_environment = dict(os.environ)
original_import = builtins.__import__
seen = []
def checked_import(name, *args, **kwargs):
    if name in ('marl_battlegrounds.baselines.ppo',
                'marl_battlegrounds.training.checkpoints') and name not in sys.modules:
        assert jax.config.jax_default_device is not None, name
        assert jax.config.jax_default_device.platform == 'cpu', name
        seen.append(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
try:
    if operation == 'declaration':
        assert json.loads(json.dumps(search.declaration())) == expected['declaration']
    elif operation == 'identity':
        assert search._actor_identity(sys.argv[3]) == expected['actor']
    elif operation == 'identity_error':
        search._actor_identity(root / 'missing-actor')
    else:
        search._selected_row(root, expected['case'])
except (ValueError, OSError):
    assert operation.endswith('_error')
else:
    assert not operation.endswith('_error')
finally:
    builtins.__import__ = original_import
assert seen
assert jax.config.jax_default_device is before_default
assert dict(os.environ) == before_environment
from marl_battlegrounds.policies.input import MOVE_MIRROR
assert MOVE_MIRROR.device.platform == 'cpu'
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), operation, str(actor_exports[0])],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("boundary", ["study_supervisor", "job_supervisor"])
def test_parent_process_boundaries_use_cpu_without_changing_caller(
    tmp_path: Path,
    declared: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    root = _package(tmp_path, declared, monkeypatch)
    (root / "logs").mkdir()
    job = root / "calibration/c00"
    job.mkdir(parents=True)
    atomic_json(job / "process.json", {"state": "exited", "exit_code": 0})
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    before = dict(os.environ)
    calls: list[dict[str, Any]] = []

    class Child:
        pid = os.getpid()

        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append(dict(kwargs))

        def wait(self, *, timeout: float) -> int:
            return 0

    monkeypatch.setattr(search.subprocess, "Popen", Child)
    if boundary == "study_supervisor":
        search.start_search(root, resume=True)
        assert calls[0]["start_new_session"] is True
    else:
        search._run_job(root, job, mode="calibration", deadline=time.time() + 60)
    assert len(calls) == 1
    assert calls[0]["env"]["JAX_PLATFORMS"] == "cpu"
    assert calls[0]["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert dict(os.environ) == before


def test_search_shared_launch_keeps_historical_clock_bytes_and_deadlines(
    tmp_path: Path, declared: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package(tmp_path, declared, monkeypatch)
    (root / "logs").mkdir()
    (root / "study.json").unlink()
    clock_bytes = (root / "launch_time.json").read_bytes()
    clock = search._read(root / "launch_time.json")
    calls: list[dict[str, Any]] = []

    def spawned(
        path: Path,
        command: list[str],
        *,
        log_path: Path,
        mode: str,
        env: dict[str, str],
        cwd: Path,
    ) -> dict[str, Any]:
        calls.append({"root": path, "command": command, "mode": mode, "env": env})
        assert log_path == root / "logs/experiment.log"
        assert cwd == root / "source"
        with pytest.raises(BlockingIOError), search._launch.launch_lock(root):
            pytest.fail("The caller must keep the launch lock until publication")
        return {"state": "starting", "mode": mode}

    monkeypatch.setattr(search._launch, "spawn_detached", spawned)
    assert search.start_search(root) == {"state": "starting", "mode": "start"}
    assert (root / "launch_time.json").read_bytes() == clock_bytes
    assert all(
        search._read(root / "study.json")[key] == value for key, value in clock.items()
    )
    assert search.start_search(root, resume=True)["mode"] == "resume"
    assert (root / "launch_time.json").read_bytes() == clock_bytes
    assert [call["mode"] for call in calls] == ["start", "resume"]
    assert all(call["env"]["JAX_PLATFORMS"] == "cpu" for call in calls)
