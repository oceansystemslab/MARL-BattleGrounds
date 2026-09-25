"""Check exact validation schedules, frozen panels and resumed M8 game tasks.

CPU artifact fixtures use untrained MAPPO only. A short-horizon fixture exercises
the real writer/evaluator recovery path; it is not an admitted learning trial.
No validation result may alter learner state or change a frozen artifact identity.
The one boundary helper, ``_collection_boundary``, gives PQN-VDN's offset
boundaries: before its initial rounds W = H + T, multiples of T capped at W;
after W, W + j * T capped at the total, with the collection-block ordinal. At
B32, T128, H4 the boundaries are 4,096, 4,224, 8,320 and 12,416 transitions,
never 8,192 or 12,288; fractions that round to one boundary merge, the final
partial boundary is kept, and explicit save steps are valid only on a
boundary. With no initial rounds it equals the earlier rule
``min(total, ceil(q / (B * T)) * B * T)`` and its update count over a grid.

Panel member names: two exports of one method both load under the method's
name, so a System panel given their two paths refuses them with the
"distinct names" error and writes no panel.json. Two ``module:function``
factories in this file, each loading one export with load_system and
returning a renamed copy, are accepted: panel.json keeps the two references
and names, and load_panel runs both factories again and restores the same
names, identities and digest.

Red Zone depth: a task description built without a depth keeps its original
bytes and ID (schema 1, and schema 2 for System panels); a depth builds schema
3 (or 4 for System panels, selection schema still 2) that records it, so depths
0, 5 and 6 and the old layout give four different task IDs, and a bad depth is
refused; a depth wider than the validation maps is refused before any file is
written. validate_checkpoint's requests carry the depth (default 5.0), its
result reports points and recorded kills, and the same folder under another
depth is refused before any game. A short real pass at depth 6.0 saves 6.0 in
every game configuration for the panel/Random and slot routes, and after one
warm pass, fresh passes at depths 5.0 and 0.0 reuse the compiled game program
(no new compilation for a new depth). load_panel admits a directly frozen
panel at every depth, a ranked panel only at its ranked float32 depth
(unversioned ranking evidence means 0.0), and refuses an unknown evidence
version. mean_kill_difference and the selection tiebreak use
recorded kills, not points (two points for one Red Zone kill), and selection
refuses to mix Red Zone scoring rules.
"""

import itertools
import json
import math
import os
import shutil
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import jax
import pytest

from marl_battlegrounds.baselines.ppo import initialize_ppo
from marl_battlegrounds.evaluation.policy_execution import System
from marl_battlegrounds.evaluation.recording_context import capture_recording_provenance
from marl_battlegrounds.evaluation.recording_identity import tree_digest
from marl_battlegrounds.evaluation.results import load_results
from marl_battlegrounds.training.analysis import (
    _candidates,
    select_checkpoint,
    summarize_validation,
)
from marl_battlegrounds.training.checkpoints import (
    artifact_identity,
    export_system,
    load_system,
)
from marl_battlegrounds.training.validation import (
    _collection_boundary,
    _panel_digest,
    _pending,
    _rows,
    _run_pass,
    _saved_run,
    create_panel,
    evaluation_backend,
    load_panel,
    make_slot_diagnostic_schedule,
    panel_task_description,
    resolve_validation_schedule,
    validate_checkpoint,
    validate_random,
    validation_task_description,
)

# Tests deliberately exercise the package's internal task ownership boundary.
# pyright: reportPrivateUsage=false


@pytest.fixture(scope="module")
def actors(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("validation-actors")
    weights = initialize_ppo(jax.random.key(88)).actor_params
    paths: list[Path] = []
    for index, steps in enumerate((4, 8)):
        paths.append(
            export_system(
                weights,
                root / f"actor-{index}",
                metadata={
                    "run_id": "development",
                    "seed": 88,
                    "env_steps": steps,
                    "checkpoint_id": str(index + 1) * 64,
                },
                spawn_frame="world",
            )
        )
    return paths[0], paths[1]


def _diagnostic(identity: dict[str, Any]) -> dict[str, Any]:
    task = {
        "schema_version": 1,
        "checkpoint_id": identity["metadata"]["checkpoint_id"],
        "actor_digest": identity["actor_digest"],
        "env_steps": identity["env_steps"],
        "panel_digest": "random-diagnostic-v1",
        "purpose": "random",
        "seed_pairs": 10,
        "maps": [42, 43, 44, 45, 46],
        "root": 19_043_001,
        "members": [{"name": "Random", "actor_digest": "builtin-random"}],
    }
    return {
        **task,
        "task_id": sha256(
            json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "complete": True,
        "games": 100,
        "independent_blocks": 50,
        "score": 0.75,
        "ci_low": 0.5,
        "ci_high": 0.9,
        "bootstrap_draws": 2000,
        "bootstrap_seed": 19_044_001,
        "pass_paths": ["/original/results/need/not/move"],
        "cells": [
            {
                "map_id": map_id,
                "opponent": "Random",
                "games": 20,
                "seed_pairs": 10,
                "score": 0.75,
                "wins": 15,
                "draws": 0,
                "losses": 5,
                "mean_team_a_score": 2,
                "mean_team_b_score": 1,
                "mean_episode_length": 10,
            }
            for map_id in range(42, 47)
        ],
    }


def test_validation_thresholds_round_to_updates_and_keep_exact_partial_end() -> None:
    points = resolve_validation_schedule(10_000_000, 32)
    assert points[0].env_steps == points[0].update_index == 0
    assert [point.env_steps for point in points[1:]] == [
        1_003_520,
        2_002_944,
        3_002_368,
        4_001_792,
        5_001_216,
        6_000_640,
        7_000_064,
        8_003_584,
        9_003_008,
        10_000_000,
    ]
    assert points[-1].update_index == 2442
    assert points[-1].requested_steps == (10_000_000,)
    tiny = resolve_validation_schedule(128, 4, 128)
    assert len(tiny) == 2
    assert tiny[-1].env_steps == 128 and tiny[-1].update_index == 1
    assert len(tiny[-1].requested_steps) == 10
    assert resolve_validation_schedule(128, 4, 128, fractions=())[-1].env_steps == 128


def test_pqn_offset_boundaries_follow_initial_rounds_then_normal_blocks() -> None:
    def boundary(requested: int, total: int = 10_000_000) -> tuple[int, int]:
        return _collection_boundary(
            requested,
            total_env_steps=total,
            num_envs=32,
            rollout_length=128,
            initial_rounds=132,
        )

    assert boundary(0) == (0, 0)
    assert [boundary(q) for q in (1, 4096, 4097, 4224)] == [
        (4096, 1),
        (4096, 1),
        (4224, 2),
        (4224, 2),
    ]
    assert [boundary(q) for q in (4225, 8192, 8320, 8321, 12288)] == [
        (8320, 3),
        (8320, 3),
        (8320, 3),
        (12416, 4),
        (12416, 4),
    ]
    # 312,500 rounds: 132 initial rounds, then 2,441 blocks, the last partial.
    assert boundary(9_999_999) == (10_000_000, 2443)
    assert boundary(10_000_000) == (10_000_000, 2443)
    for step in (0, 4096, 4224, 8320, 12416, 10_000_000):
        assert boundary(step)[0] == step
    for step in (8192, 12288, 4200, 9_999_968):
        assert boundary(step)[0] != step
    points = resolve_validation_schedule(
        16_640, 32, 128, fractions=(0.25, 0.253, 0.5), initial_rounds=132
    )
    assert [(p.env_steps, p.update_index) for p in points] == [
        (0, 0),
        (4224, 2),
        (8320, 3),
        (16_640, 6),
    ]
    assert points[1].requested_steps == (4160, 4210)
    with pytest.raises(ValueError, match="below the total rounds"):
        boundary(1, total=132 * 32)
    with pytest.raises(ValueError):
        boundary(True)  # pyright: ignore[reportArgumentType]


def test_boundary_without_initial_rounds_matches_the_earlier_rule() -> None:
    fractions = (0.1, 0.25, 1 / 3, 0.5, 0.7, 0.999, 1.0)
    for total_rounds, batch, length in itertools.product(
        (1, 2, 7, 64, 129, 1000, 4097), (1, 2, 4, 32), (1, 3, 4, 128)
    ):
        total = total_rounds * batch
        block = batch * length
        for requested in {
            0,
            1,
            total - 1,
            total,
            *(math.ceil(total * f) for f in fractions),
        }:
            if requested < 0:
                continue
            expected = min(total, math.ceil(requested / block) * block)
            assert _collection_boundary(
                requested, total_env_steps=total, num_envs=batch, rollout_length=length
            ) == (expected, math.ceil(expected / block))


@pytest.mark.parametrize("values", ((127, 4), (True, 4), (128, 0)))
def test_invalid_total_or_batch_is_rejected(values: tuple[Any, Any]) -> None:
    with pytest.raises(ValueError):
        resolve_validation_schedule(*values)


@pytest.mark.parametrize("fractions", ((0.1, 0.1), (0.9, 0.1), (float("nan"),), (0.0,)))
def test_invalid_fraction_schedule_is_rejected(fractions: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        resolve_validation_schedule(128, 4, fractions=fractions)


@pytest.mark.parametrize("pending", (0, 1, 31, 32, 33))
def test_backend_routing_never_starts_a_small_gpu_batch(pending: int) -> None:
    assert evaluation_backend(pending, "gpu") == (
        "none" if pending == 0 else "cpu" if pending < 32 else "gpu"
    )
    assert evaluation_backend(pending, "cpu") == ("none" if pending == 0 else "cpu")


def test_panel_binds_lineage_digests_and_gate_without_requiring_distinct_weights(
    actors: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    identities = [artifact_identity(path) for path in actors]
    diagnostics = [_diagnostic(row) for row in identities]
    panel = create_panel(*actors, output_dir=tmp_path, diagnostics=diagnostics)
    assert panel.qualified
    assert [row.name for row in panel.members] == ["Halfway", "Final"]
    assert panel.members[0].checkpoint_id == "1" * 64
    stored = json.loads(panel.path.read_text())["qualification_evidence"]
    assert [row["task_id"] for row in stored] == [row["task_id"] for row in diagnostics]
    assert all("pass_paths" not in row for row in stored)
    assert load_panel(tmp_path) == panel
    assert create_panel(*actors, output_dir=tmp_path, diagnostics=diagnostics) == panel
    invalid = [{**row, "score": 0.5} for row in diagnostics]
    with pytest.raises(ValueError, match="usefulness"):
        create_panel(*actors, output_dir=tmp_path / "failed", diagnostics=invalid)
    test_panel = create_panel(*actors, output_dir=tmp_path / "test", test_only=True)
    assert not test_panel.qualified
    changed = json.loads(panel.path.read_text())
    changed["members"][0]["actor_digest"] = "bad"
    panel.path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="digest"):
        load_panel(panel.path)


@pytest.mark.parametrize(
    "changed",
    (
        "nan",
        "infinity",
        "above_one",
        "small_budget",
        "root",
        "missing_map",
        "duplicate_map",
        "wrong_checkpoint",
        "wrong_actor",
        "counts",
        "cell_nan",
        "team_nan",
        "task_id",
        "missing_evidence",
    ),
)
def test_panel_rejects_incomplete_nonfinite_or_mismatched_qualification(
    changed: str,
    actors: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    diagnostics = [_diagnostic(artifact_identity(path)) for path in actors]
    result = deepcopy(diagnostics[0])
    if changed in ("nan", "infinity", "above_one"):
        result["score"] = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "above_one": 1.1,
        }[changed]
    elif changed == "small_budget":
        result.update(seed_pairs=1, games=10, independent_blocks=5)
    elif changed == "root":
        result["root"] += 1
    elif changed == "missing_map":
        result["cells"].pop()
    elif changed == "duplicate_map":
        result["cells"][-1] = result["cells"][0]
    elif changed == "wrong_checkpoint":
        result["checkpoint_id"] = "a" * 64
    elif changed == "wrong_actor":
        result["actor_digest"] = "b" * 64
    elif changed == "counts":
        result["cells"][0]["wins"] = 14
    elif changed == "cell_nan":
        result["cells"][0]["score"] = float("nan")
    elif changed == "team_nan":
        result["cells"][0]["mean_team_a_score"] = float("nan")
    elif changed == "task_id":
        result["task_id"] = "c" * 64
    else:
        diagnostics.pop()
    if diagnostics:
        diagnostics[0] = result
    with pytest.raises(ValueError):
        create_panel(*actors, output_dir=tmp_path, diagnostics=diagnostics)


def test_relocated_member_paths_keep_panel_scientific_identity(
    actors: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    panel = create_panel(
        *actors,
        output_dir=tmp_path / "original",
        diagnostics=[_diagnostic(artifact_identity(path)) for path in actors],
    )
    data = json.loads(panel.path.read_text())
    relocated = tmp_path / "copied"
    relocated.mkdir()
    for row, path in zip(data["members"], actors, strict=True):
        shutil.copytree(path, relocated / path.name)
        row["path"] = path.name
    (relocated / "panel.json").write_text(json.dumps(data))
    copied = load_panel(relocated)
    assert copied.digest == panel.digest and copied.qualified
    assert all(
        member.path is not None and member.path.parent == relocated
        for member in copied.members
    )


# The panel imports these factories by name, possibly as a second copy of this
# module, so each export path travels in an environment variable.
_EARLY_EXPORT = "MARL_BG_TEST_PANEL_EARLY_EXPORT"
_LATE_EXPORT = "MARL_BG_TEST_PANEL_LATE_EXPORT"


def renamed_early_export() -> System:
    return replace(load_system(os.environ[_EARLY_EXPORT]), name="MAPPO early")


def renamed_late_export() -> System:
    return replace(load_system(os.environ[_LATE_EXPORT]), name="MAPPO late")


def test_panel_accepts_same_method_exports_renamed_by_factories(
    actors: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_EARLY_EXPORT, str(actors[0]))
    monkeypatch.setenv(_LATE_EXPORT, str(actors[1]))
    references = [
        f"tests.test_training_validation:{factory.__name__}"
        for factory in (renamed_early_export, renamed_late_export)
    ]
    names = ["MAPPO early", "MAPPO late"]
    panel = create_panel(opponents=references, output_dir=tmp_path)
    assert [member.name for member in panel.members] == names
    assert [member.reference for member in panel.members] == references
    saved = json.loads(panel.path.read_text())["members"]
    assert [row["name"] for row in saved] == names
    assert [row["reference"] for row in saved] == references
    reloaded = load_panel(tmp_path)
    assert reloaded == panel
    assert [method.name for method in reloaded.methods] == names
    assert all(
        isinstance(method, System) and method is not original
        for method, original in zip(reloaded.methods, panel.methods, strict=True)
    )


def test_panel_refuses_two_unrenamed_exports_of_one_method(
    actors: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    assert [load_system(path).name for path in actors] == ["Recurrent MAPPO"] * 2
    with pytest.raises(ValueError, match="distinct names"):
        create_panel(opponents=[str(path) for path in actors], output_dir=tmp_path)
    assert not (tmp_path / "panel.json").exists()


@pytest.mark.parametrize("kind", ("panel", "slot"))
def test_real_cpu_tail_resume_uses_all_saved_games_and_keeps_frozen_weights(
    kind: str,
    actors: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    original_evaluate = evaluator.evaluate

    def short_horizon(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_evaluate(*args, **kwargs, max_steps=1)

    monkeypatch.setattr(evaluator, "evaluate", short_horizon)
    if kind == "slot":
        from marl_battlegrounds.tasks import (
            canonical_tournament_rosters,
            make_standard_team_deathmatch_config,
        )

        validation = import_module("marl_battlegrounds.training.validation")

        def short_schedule(
            *, maps: list[int], seed_blocks: int, red_zone_depth: float
        ) -> object:
            team_a, team_b = canonical_tournament_rosters()
            return make_slot_diagnostic_schedule(
                maps=maps,
                seed_blocks=seed_blocks,
                configs=tuple(
                    make_standard_team_deathmatch_config(
                        map_id=value,
                        max_steps=1,
                        team_a_roster=team_a,
                        team_b_roster=team_b,
                        red_zone_depth=red_zone_depth,
                    )
                    for value in maps
                ),
            )

        monkeypatch.setattr(validation, "make_slot_diagnostic_schedule", short_schedule)
    original_chunk = evaluator._jax_system_chunk
    calls = 0

    def interrupted(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Deliberate interruption after the first saved chunk")
        return original_chunk(*args, **kwargs)

    monkeypatch.setattr(evaluator, "_jax_system_chunk", interrupted)
    identity = artifact_identity(actors[0])
    request = {
        "kind": kind,
        "task_id": "task",
        "actor": str(actors[0]),
        "actor_digest": identity["actor_digest"],
        "opponent": "random",
        "opponent_digest": "builtin-random",
        "output_parent": str(tmp_path / "pass"),
        "pass_id": "fixed",
        "maps": [42],
        "seed_pairs": 2,
        "total_games": 4,
        "root": 19_043_001,
        "num_envs": 2,
        "chunk_size": 1,
        "focal_team": 0,
        "red_zone_depth": 6.0,
    }
    events: list[dict[str, Any]] = []
    before = artifact_identity(actors[0])
    with pytest.raises(RuntimeError, match="Deliberate interruption"):
        _run_pass(request, event_callback=events.append)
    run_dir = _saved_run(tmp_path / "pass", "fixed")
    assert run_dir is not None
    assert _pending(run_dir, pass_id="fixed", total=4) == 2
    monkeypatch.setattr(evaluator, "_jax_system_chunk", original_chunk)
    restored = _run_pass(request, event_callback=events.append)
    assert restored == run_dir
    rows = _rows(restored, pass_id="fixed", opponent="Random", kills=True)
    assert len(rows) == 4
    assert {row["episode_id"] for row in rows} == (
        {1, 2, 3, 4} if kind == "panel" else {1, 2, 5, 6}
    )
    assert all(row["system_game_score"] == 0.5 for row in rows)
    assert all(row["team_a_kills"] == row["team_b_kills"] == 0 for row in rows)
    # The request's depth reaches every saved game configuration.
    saved = load_results(restored, phase="validation", pass_id="fixed").metadata
    assert {
        content["team_deathmatch_red_zone_depth"]
        for content in saved["configurations"].values()
    } == {6.0}
    assert _pending(restored, pass_id="fixed", total=4) == 0
    count = len(events)
    assert _run_pass(request, event_callback=events.append) == restored
    assert len(events) == count
    assert artifact_identity(actors[0]) == before
    assert [
        row["pending_games"] for row in events if row["event"] == "evaluation_segment"
    ] == [4, 2]
    if kind == "panel":
        # After one fresh pass warms the programs, other depths with the same
        # shapes reuse them: the depth is a dynamic value, not a static one.
        def fresh(name: str, depth: float) -> int:
            changed = {
                **request,
                "output_parent": str(tmp_path / name),
                "pass_id": name,
                "red_zone_depth": depth,
            }
            _run_pass(changed, event_callback=None)
            return original_chunk._cache_size()

        warmed = fresh("warm-6", 6.0)
        assert fresh("new-5", 5.0) == fresh("new-0", 0.0) == warmed > 0


def test_loaded_actor_contains_only_the_original_actor_tree(
    actors: tuple[Path, Path],
) -> None:
    from marl_battlegrounds.training.checkpoints import load_system

    system = load_system(actors[0])
    assert tree_digest(system.variables) == artifact_identity(actors[0])["actor_digest"]


@pytest.mark.parametrize(
    "purpose,pairs,games,root",
    (
        ("routine", 10, 200, 19_043_002),
        ("initialization", 10, 200, 19_043_002),
        ("confirmation", 50, 1000, 19_043_003),
    ),
)
def test_default_tasks_bind_exact_population_weights_and_distinct_roots(
    purpose: str,
    pairs: int,
    games: int,
    root: int,
    actors: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = import_module("marl_battlegrounds.training.validation")
    panel = create_panel(*actors, output_dir=tmp_path / "panel", test_only=True)
    requests: list[dict[str, Any]] = []

    def run_pass(request: dict[str, Any], *, event_callback: object) -> Path:
        requests.append(request)
        return Path(request["output_parent"])

    def known_rows(
        path: Path, *, pass_id: str, opponent: str, kills: bool
    ) -> list[dict[str, Any]]:
        assert kills
        return [
            {
                "map_id": map_id,
                "seed_id": index * pairs + block + 1,
                "opponent": opponent,
                "spawn_locations": end,
                "system_game_score": 0.5,
                "team_a_score": 2,
                "team_b_score": 2,
                "team_a_kills": 1,
                "team_b_kills": 1,
            }
            for index, map_id in enumerate((42, 43, 44, 45, 46))
            for block in range(pairs)
            for end in (0, 1)
        ]

    monkeypatch.setattr(validation, "_run_pass", run_pass)
    monkeypatch.setattr(validation, "_rows", known_rows)
    result = validate_checkpoint(
        actors[0], panel, output_dir=tmp_path / "task", purpose=purpose
    )
    assert result["games"] == games and result["score"] == 0.5
    assert result["checkpoint_id"] == "1" * 64
    assert result["actor_digest"] == artifact_identity(actors[0])["actor_digest"]
    assert result["panel_digest"] == panel.digest
    assert result["schema_version"] == 3 and result["red_zone_depth"] == 5.0
    assert all(
        cell["mean_team_a_score"] == 2 and cell["mean_team_a_kills"] == 1
        for cell in result["cells"]
    )
    assert len(requests) == 2
    assert len({request["pass_id"] for request in requests}) == 2
    for request in requests:
        assert request["total_games"] == games // 2
        assert request["root"] == root
        assert request["num_envs"] == 32 and request["chunk_size"] == 128
        assert request["maps"] == [42, 43, 44, 45, 46]
        assert request["red_zone_depth"] == 5.0
    # The same folder under another depth is a different task and is refused.
    with pytest.raises(ValueError, match="different scientific conditions"):
        validate_checkpoint(
            actors[0],
            panel,
            output_dir=tmp_path / "task",
            purpose=purpose,
            red_zone_depth=6.0,
        )
    assert len(requests) == 2


@pytest.mark.parametrize("pending", (1, 31, 32, 33))
def test_gpu_tail_launches_only_a_cpu_worker_below_32(
    pending: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation = import_module("marl_battlegrounds.training.validation")
    run = tmp_path / "saved"
    run.mkdir()
    (run / "run_details.json").write_text(json.dumps({"passes": {}}))
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    def saved_run(parent: Path, pass_id: str) -> Path:
        return run

    def remaining(directory: Path, *, pass_id: str, total: int) -> int:
        return pending

    calls: list[dict[str, Any]] = []

    def execute(request: dict[str, Any]) -> Path:
        calls.append({"route": "local", "request": request})
        return run

    def worker(command: list[str], *, env: dict[str, str], check: bool) -> None:
        calls.append(
            {"route": "worker", "command": command, "env": env, "check": check}
        )

    monkeypatch.setattr(validation, "_saved_run", saved_run)
    monkeypatch.setattr(validation, "_pending", remaining)
    monkeypatch.setattr(validation, "_execute_request", execute)
    monkeypatch.setattr(validation.subprocess, "run", worker)
    request = {
        "task_id": "fixed",
        "pass_id": "fixed",
        "total_games": 100,
        "num_envs": 32,
        "output_parent": str(tmp_path / "pass"),
    }
    events: list[dict[str, Any]] = []
    assert _run_pass(request, event_callback=events.append) == run
    assert len(calls) == 1
    assert calls[0]["route"] == ("worker" if pending < 32 else "local")
    assert events[0]["backend"] == ("cpu" if pending < 32 else "gpu")
    assert events[0]["num_envs"] == min(32, pending)
    if pending < 32:
        call = calls[0]
        assert call["env"]["JAX_PLATFORMS"] == "cpu"
        assert call["command"][1:4] == ["-m", validation.__name__, "--worker"]
        assert json.loads(Path(call["command"][4]).read_text()) == request
        assert call["check"] is True


def test_task_descriptions_record_the_depth_and_keep_legacy_bytes(
    actors: tuple[Path, Path], tmp_path: Path
) -> None:
    identity = artifact_identity(actors[0])
    options: dict[str, Any] = dict(
        checkpoint_id=identity["metadata"]["checkpoint_id"],
        actor_digest=identity["actor_digest"],
        env_steps=identity["env_steps"],
        panel_digest="random-diagnostic-v1",
        purpose="random",
        seed_pairs=10,
        members=(("Random", "builtin-random"),),
    )
    legacy = validation_task_description(**options)
    assert legacy == {
        key: value for key, value in _diagnostic(identity).items() if key in legacy
    }
    assert legacy["schema_version"] == 1 and "red_zone_depth" not in legacy
    fresh = {
        depth: validation_task_description(**options, red_zone_depth=depth)
        for depth in (0.0, 5.0, 6.0)
    }
    for depth, task in fresh.items():
        assert task["schema_version"] == 3 and task["red_zone_depth"] == depth
    assert len({legacy["task_id"], *(row["task_id"] for row in fresh.values())}) == 4
    panel = create_panel(opponents=[str(actors[0])], output_dir=tmp_path / "panel")
    panel_options: dict[str, Any] = dict(
        checkpoint_id=identity["metadata"]["checkpoint_id"],
        actor_digest=identity["actor_digest"],
        env_steps=identity["env_steps"],
        panel=panel,
        purpose="routine",
        seed_pairs=4,
    )
    old = panel_task_description(**panel_options)
    assert old["schema_version"] == 2 and "red_zone_depth" not in old
    new = {
        depth: panel_task_description(**panel_options, red_zone_depth=depth)
        for depth in (0.0, 5.0, 6.0)
    }
    for depth, task in new.items():
        assert task["schema_version"] == 4 and task["red_zone_depth"] == depth
        assert task["selection_schema_version"] == 2
        assert {
            key: value
            for key, value in task.items()
            if key not in ("schema_version", "red_zone_depth", "task_id")
        } == {
            key: value
            for key, value in old.items()
            if key not in ("schema_version", "task_id")
        }
    assert len({old["task_id"], *(row["task_id"] for row in new.values())}) == 4
    for bad, error in ((5, TypeError), (-1.0, ValueError), (math.nan, ValueError)):
        with pytest.raises(error):
            validation_task_description(**options, red_zone_depth=bad)
        with pytest.raises(error):
            panel_task_description(**panel_options, red_zone_depth=bad)
    # A depth wider than the 20-unit validation maps is refused before any
    # validation file (such as task.json) is written, so the folder stays usable.
    wide = tmp_path / "too-wide"
    with pytest.raises(ValueError, match="must not exceed map_width"):
        validate_random(actors[0], output_dir=wide, seed_pairs=1, red_zone_depth=25.0)
    assert not wide.exists()


def test_ranked_panel_is_admitted_only_at_its_ranked_depth(
    actors: tuple[Path, Path], tmp_path: Path
) -> None:
    direct = create_panel(opponents=[str(actors[0])], output_dir=tmp_path / "direct")
    for depth in (0.0, 5.0, 6.0):
        assert load_panel(direct.path, red_zone_depth=depth) == direct
    content = json.loads(direct.path.read_text())
    evidence: dict[str, Any] = {
        "schedule_digest": "s",
        "evidence_digest": "e",
        "ratings": [],
        "size": 1,
    }

    def write(name: str, ranking: dict[str, Any]) -> Path:
        ranked = {**content, "ranking_evidence": ranking}
        ranked["panel_digest"] = _panel_digest(ranked)
        (tmp_path / name).mkdir()
        (tmp_path / name / "panel.json").write_text(json.dumps(ranked))
        return tmp_path / name

    # A ranking saved before the Red Zone rule scored one point per death.
    for folder, ranked_depth in (
        (write("before-rule", evidence), 0.0),
        (
            write("ranked", {**evidence, "schema_version": 2, "red_zone_depth": 6.0}),
            6.0,
        ),
    ):
        admitted = load_panel(folder, red_zone_depth=ranked_depth)
        assert (
            admitted.digest
            == json.loads((folder / "panel.json").read_text())["panel_digest"]
        )
        assert load_panel(folder).digest == admitted.digest
        with pytest.raises(ValueError, match="ranked with red_zone_depth"):
            load_panel(folder, red_zone_depth=5.0)
    # Depths are compared as the float32 values the game configs store.
    assert load_panel(tmp_path / "ranked", red_zone_depth=6.0000001).digest
    unknown = write("unknown", {**evidence, "schema_version": 3, "red_zone_depth": 5.0})
    with pytest.raises(ValueError, match="ranking evidence"):
        load_panel(unknown, red_zone_depth=5.0)


def test_kill_difference_and_selection_use_recorded_kills_not_points() -> None:
    def summary(
        points: int, kills: int, *, actual_kills: bool = True
    ) -> dict[str, Any]:
        rows = [
            {
                "map_id": 42,
                "seed_id": 1,
                "opponent": "Alpha",
                "spawn_locations": end,
                "system_game_score": 1.0,
                "team_a_score": points,
                "team_b_score": 0,
                "team_a_kills": kills,
                "team_b_kills": 0,
            }
            for end in (0, 1)
        ]
        return summarize_validation(
            rows,
            maps=[42],
            opponents=["Alpha"],
            seed_pairs=1,
            independent_opponents=True,
            actual_kills=actual_kills,
        )

    # One Red Zone kill: two points, one kill.
    red_zone = summary(2, 1)
    assert red_zone["mean_kill_difference"] == 1.0
    assert red_zone["cells"][0]["mean_team_a_score"] == 2.0
    assert red_zone["cells"][0]["mean_team_a_kills"] == 1.0
    legacy = summary(2, 1, actual_kills=False)
    assert legacy["mean_kill_difference"] == 2.0
    assert "mean_team_a_kills" not in legacy["cells"][0]
    base = {
        "complete": True,
        "purpose": "confirmation",
        "panel_digest": "panel",
        "selection_schema_version": 2,
        "red_zone_depth": 5.0,
        "score": 1.0,
        "env_steps": 64,
    }
    more_points = {
        **base,
        "checkpoint_id": "a",
        "mean_kill_difference": summary(4, 2)["mean_kill_difference"],
    }
    more_kills = {
        **base,
        "checkpoint_id": "b",
        "mean_kill_difference": summary(3, 3)["mean_kill_difference"],
    }
    assert select_checkpoint([more_points, more_kills])["checkpoint_id"] == "b"
    for other in (
        {**more_kills, "red_zone_depth": 0.0},
        {key: value for key, value in more_kills.items() if key != "red_zone_depth"},
    ):
        with pytest.raises(ValueError, match="Red Zone scoring rules"):
            _candidates([more_points, other])
