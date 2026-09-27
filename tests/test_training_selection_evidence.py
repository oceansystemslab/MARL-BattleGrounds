"""Check saved-only selection evidence, using small signed files and known game rows.

Actor payload hashes and learner descriptions use the real checkpoint readers.
The M8 table reader is replaced with fixed completed games, so these tests check
joins and rejection boundaries without playing games or constructing a model.
They cover original and fresh confirmations, historical rules, actor corruption,
pruned parent payloads, incomplete runs, saved score/identity changes, sampling
facts bound to exact passes, and files changed during inspection. A separate
retained real-run check covers M8 CSV reads.
"""

# pyright: reportPrivateUsage=false

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from marl_battlegrounds.evaluation import results
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.evaluation.run_writer import _json_bytes
from marl_battlegrounds.training import _selection_evidence as evidence
from marl_battlegrounds.training import checkpoints, selection, validation
from marl_battlegrounds.training.analysis import summarize_validation


def _write(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, sort_keys=True, allow_nan=False))


def _signed(content: dict[str, Any]) -> dict[str, Any]:
    return {
        **content,
        "checkpoint_id": hashlib.sha256(checkpoints._json_bytes(content)).hexdigest(),
    }


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Saved evidence must not load a model, factory or learner")

    for name in ("load_system", "restore_checkpoint", "_restore_arrays"):
        monkeypatch.setattr(checkpoints, name, forbidden)
    monkeypatch.setattr(validation, "load_panel", forbidden)

    def load(path: str | Path, **_kwargs: object) -> SimpleNamespace:
        data = json.loads((Path(path) / "run_details.json").read_text())
        return SimpleNamespace(status="complete", metadata=data)

    def rows(path: Path, **_kwargs: object) -> list[dict[str, Any]]:
        return json.loads((path / "rows.json").read_text())

    monkeypatch.setattr(results, "load_results", load)
    monkeypatch.setattr(validation, "_rows", rows)
    return {"root": tmp_path / "run", "external": tmp_path / "fresh"}


def _make_run(
    package: dict[str, Any],
    *,
    historical: bool = False,
    sampling: str | None = None,
    total_env_steps: int = 32,
) -> dict[str, Any]:
    root = package["root"]
    config: dict[str, Any] = {
        "method": "mappo",
        "seed": 7,
        "total_env_steps": total_env_steps,
        "routine_seed_pairs": 2,
        "confirmation_seed_pairs": 2,
        "ppo": {},
    }
    if not historical:
        config["red_zone_depth"] = 5.0
    schemas = checkpoints.checkpoint_schemas("mappo")
    if historical:
        schemas = {**schemas, "actor_input": 1, "training_state": 1}
    registration_id, registration = normalize_system_registration(
        {"name": "Opponent", "checkpoint": "b" * 64, "variables_frozen": True},
        phase="validation",
    )
    panel_content: dict[str, Any]
    if historical:
        panel_content = {
            "schema_version": 1,
            "provisional": True,
            "qualified": False,
            "inference": "Sampled masked categorical",
            "members": [
                {
                    "name": name,
                    "path": "unused",
                    "actor_digest": "b" * 64,
                    "checkpoint_id": str(index + 1) * 64,
                    "run_id": "opponent-run",
                    "seed": 10,
                    "env_steps": (index + 1) * 32,
                }
                for index, name in enumerate(("Halfway", "Final"))
            ],
        }
    else:
        panel_content = {
            "schema_version": 2,
            "selection_schema_version": 2,
            "roots": validation._panel_roots(None),
            "ranking_evidence": None,
            "members": [
                {
                    "name": "Opponent",
                    "reference": "unused:factory",
                    "registration_id": registration_id,
                    "registration": registration,
                }
            ],
        }
    panel_content["panel_digest"] = validation._panel_digest(panel_content)
    run = {
        "schema_version": 1,
        "run_id": "training-run",
        "config": config,
        "schemas": schemas,
        "source": {"id": "original-source"},
        "dependencies": {"version": "original"},
        "panel_digest": panel_content["panel_digest"],
    }
    parent = _signed(
        {
            "schema_version": 1,
            "kind": "learner",
            "schemas": schemas,
            "metadata": {
                **{k: run[k] for k in ("run_id", "config", "source", "dependencies")},
                "attempt_id": "first",
                "parent_checkpoint": None,
            },
            "actor_digest": "a" * 64,
            "actor_layout": [],
            "files": {},
            "collection": {},
            "layout": [],
            "recording_token": None,
            "counters": {"env_steps": 32, "updates": 1},
        }
    )
    identifier = parent["checkpoint_id"]
    actor_path = root / "actors" / identifier
    payload = b"A small opaque payload; this reader never reconstructs arrays."
    (actor_path / "actor").mkdir(parents=True)
    (actor_path / "actor" / "payload").write_bytes(payload)
    actor = _signed(
        {
            "schema_version": 1,
            "kind": "actor",
            "schemas": schemas,
            "metadata": {
                "run_id": run["run_id"],
                "seed": 7,
                "env_steps": 32,
                "checkpoint_id": identifier,
            },
            "actor_digest": "a" * 64,
            "actor_layout": [],
            "input_scale": 1.0,
            "files": checkpoints._inventory(actor_path),
        }
    )
    _write(actor_path / "actor_details.json", actor)
    _write(root / "checkpoints" / identifier / "checkpoint_details.json", parent)
    _write(root / "run_details.json", run)
    _write(
        root / "status.json",
        {"run_id": run["run_id"], "status": "complete", "final_actor": str(actor_path)},
    )
    panel_path = root / "validation_panel" / "panel.json"
    _write(panel_path, panel_content)
    panel = evidence._panel(panel_path, evidence._Snapshot())
    package.update(
        run=run,
        panel=panel,
        identifier=identifier,
        actor_path=actor_path,
        historical=historical,
    )
    summary = _make_task(package, sampling=sampling)
    _write(root / "validation_results.json", [summary])
    return package


def _make_task(
    package: dict[str, Any],
    *,
    purpose: str = "routine",
    root_seed: int | None = None,
    seed_pairs: int = 2,
    external: bool = False,
    sampling: str | None = None,
) -> dict[str, Any]:
    panel = package["panel"]
    identifier = package["identifier"]
    depth = None if package["historical"] else 5.0
    task = validation.panel_task_description(
        checkpoint_id=identifier,
        actor_digest="a" * 64,
        env_steps=32,
        panel=panel,
        purpose=purpose,
        seed_pairs=seed_pairs,
        root_seed=root_seed,
        red_zone_depth=depth,
    )
    target = (
        package["external"]
        if external
        else package["root"] / "validation" / f"{purpose}-{identifier}"
    )
    _write(target / "task.json", task)
    all_rows: list[dict[str, Any]] = []
    paths: list[str] = []
    facts: dict[str, dict[str, Any]] = {}
    for index, member in enumerate(panel.members):
        directory = target / f"opponent-{index}" / "m8-run"
        paths.append(str(directory))
        pass_id = validation.validation_pass_id(task["task_id"], member.name)
        focal = {"name": "Actor", "checkpoint": "a" * 64, "variables_frozen": True}
        other = {"name": "Opponent", "checkpoint": "b" * 64, "variables_frozen": True}
        registrations = {
            team: normalize_system_registration(value, phase="validation")[0]
            for team, value in (("team_a", focal), ("team_b", other))
        }
        root_seed = (
            task["root"]
            if panel.schema_version == 1
            else task["members"][index]["root"]
        )
        options: dict[str, Any] = {
            "seed": root_seed,
            "spawn_mode": "paired",
            "score_threshold": 20,
            "max_steps": 300,
            "metrics": "priority",
            "save_replays": 0,
            "system_roster": None,
            "opponent_roster": None,
            "full_metrics_episodes": [],
            "replay_episodes": [],
        }
        if depth is not None:
            options["red_zone_depth"] = depth
        configs: dict[str, Any] = {}
        sources: list[dict[str, Any]] = []
        episodes: dict[str, Any] = {}
        game_rows: list[dict[str, Any]] = []
        for map_index, map_id in enumerate(validation.VALIDATION_MAPS):
            config: dict[str, Any] = {
                "map_width": 20.0,
                "map_height": 10.0,
                "team_spawn_pad_positions": [[[0.5, 1.0]], [[19.5, 1.0]]],
                "team_deathmatch_score_threshold": 20,
                "max_steps": 300,
                "task_mode": 1,
                "agent_profile": {
                    "class_ids": [1, 2, 3, 4, 5] * 2,
                    "active_mask": [True] * 10,
                },
            }
            if depth is not None:
                config["team_deathmatch_red_zone_depth"] = depth
            source_id = hashlib.sha256(_json_bytes(config)).hexdigest()
            sources.append({"map_id": map_id, "source_config_id": source_id})
            configs[source_id] = config
            for pair in range(seed_pairs):
                for end in range(2):
                    actual = dict(config)
                    if end:
                        actual["team_spawn_pad_positions"] = config[
                            "team_spawn_pad_positions"
                        ][::-1]
                    config_id = hashlib.sha256(_json_bytes(actual)).hexdigest()
                    configs[config_id] = actual
                    seed_id = pair * len(validation.VALIDATION_MAPS) + map_index + 1
                    episode_id = (seed_id - 1) * 2 + end + 1
                    episodes[str(episode_id)] = {
                        "episode_id": episode_id,
                        "seed_id": seed_id,
                        "expected_horizon": 300,
                        "initial_state_digest": None,
                        "map_id": map_id,
                        "spawn_locations": end,
                        "source_config_id": source_id,
                        "configuration_digest": config_id,
                    }
                    game_rows.append(
                        {
                            "episode_id": episode_id,
                            "map_id": map_id,
                            "seed_id": seed_id,
                            "spawn_locations": end,
                            "opponent": member.name,
                            "system_game_score": 0.5,
                            "episode_length": 300,
                            "team_a_score": 3,
                            "team_b_score": 1,
                            **(
                                {"team_a_kills": 2, "team_b_kills": 1}
                                if depth is not None
                                else {}
                            ),
                        }
                    )
        entry: dict[str, Any] = {
            "policies": {"team_a": focal, "team_b": other},
            "system_ids": registrations,
            "episodes": episodes,
            "details": {
                "evaluation_contract": {
                    "version": 1,
                    "schedule_kind": "generated",
                    "map_selection": "explicit",
                    "spawn_mode": "paired",
                    "options": options,
                    "source_choices": sources,
                },
                "phase": "validation",
                "pass_id": pass_id,
                "seed": root_seed,
                "num_episodes": len(validation.VALIDATION_MAPS) * seed_pairs * 2,
                "metrics": "priority",
            },
        }
        _write(
            directory / "run_details.json",
            {
                "passes": {pass_id: entry},
                "configurations": configs,
                "tables": {"rows.json": {}},
            },
        )
        if sampling is not None:
            fact = {"determinism": sampling, "basis": "Declared test method fact"}
            _write(
                directory.parent / "sampling_facts.json",
                {
                    "schema_version": 1,
                    "task_id": task["task_id"],
                    "pass_id": pass_id,
                    "system_ids": registrations,
                    "methods": {"team_a": fact, "team_b": fact},
                },
            )
            facts[member.name] = fact
        _write(directory / "rows.json", game_rows)
        all_rows.extend(game_rows)
    summary: dict[str, Any] = {
        **task,
        **summarize_validation(
            all_rows,
            maps=validation.VALIDATION_MAPS,
            opponents=[member.name for member in panel.members],
            seed_pairs=seed_pairs,
            independent_opponents=panel.schema_version == 2,
            actual_kills=depth is not None,
            sampling_evidence=validation.validation_sampling_evidence(
                all_rows,
                facts=facts,
                scheduled_games=len(validation.VALIDATION_MAPS)
                * seed_pairs
                * 2
                * len(panel.members),
                independent_opponents=panel.schema_version == 2,
            )
            if sampling is not None
            else None,
        ),
        "pass_paths": paths,
    }
    _write(target / "validation_summary.json", summary)
    return summary


@pytest.mark.parametrize("historical", [False, True])
def test_saved_records_join_without_models_or_pruned_parent_payloads(
    package: dict[str, Any],
    historical: bool,
) -> None:
    data = _make_run(package, historical=historical)
    root = data["root"]
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = evidence.read_run_evidence(root)
    assert result["status"] == "complete"
    assert result["final_checkpoint_id"] == data["identifier"]
    actor = result["actors"][data["identifier"]]
    assert actor["artifact_id"] != actor["checkpoint_id"]
    assert result["panel"].methods == ()
    assert result["records"][0]["mean_kill_difference"] == (2.0 if historical else 1.0)
    assert ("red_zone_depth" in result["records"][0]) is not historical
    assert all(path.read_bytes() == content for path, content in before.items())
    assert all(Path(row["path"]) in before for row in result["source_files"])


@pytest.mark.parametrize(
    "change", ["score", "root", "actor", "opponent", "seed", "config"]
)
def test_changed_saved_evidence_rejects(package: dict[str, Any], change: str) -> None:
    data = _make_run(package)
    root = data["root"]
    rows = json.loads((root / "validation_results.json").read_text())
    summary_path = (
        root
        / "validation"
        / f"routine-{data['identifier']}"
        / "validation_summary.json"
    )
    if change in ("score", "root"):
        rows[0]["score" if change == "score" else "root"] = (
            0.9 if change == "score" else 333
        )
        _write(root / "validation_results.json", rows)
        _write(summary_path, rows[0])
    elif change == "actor":
        (data["actor_path"] / "actor" / "payload").write_bytes(b"changed")
    else:
        path = Path(rows[0]["pass_paths"][0]) / "run_details.json"
        manifest = json.loads(path.read_text())
        entry = next(iter(manifest["passes"].values()))
        if change == "opponent":
            entry["policies"]["team_b"]["checkpoint"] = "c" * 64
        elif change == "seed":
            entry["details"]["seed"] += 1
        else:
            next(iter(manifest["configurations"].values()))["max_steps"] = 301
        _write(path, manifest)
    with pytest.raises(ValueError):
        evidence.read_run_evidence(root)


def test_explicit_confirmation_root_does_not_authorize_routine_override(
    package: dict[str, Any],
) -> None:
    data = _make_run(package)
    _make_task(data, purpose="confirmation", root_seed=19_044_791, external=True)
    path = data["external"] / "validation_summary.json"
    with pytest.raises(ValueError, match="declared identity"):
        evidence.read_run_evidence(data["root"], confirmation_results=[path])
    result = evidence.read_run_evidence(
        data["root"], confirmation_results=[path], confirmation_root=19_044_791
    )
    assert [row["purpose"] for row in result["records"]] == ["routine", "confirmation"]
    assert result["records"][0]["root"] == data["panel"].roots["routine"]
    assert result["records"][1]["root"] == 19_044_791
    with pytest.raises(ValueError, match="repeats"):
        evidence.read_run_evidence(
            data["root"],
            confirmation_results=[path, path],
            confirmation_root=19_044_791,
        )


def test_failed_run_and_missing_work_remain_visible(package: dict[str, Any]) -> None:
    data = _make_run(package)
    root = data["root"]
    _write(root / "status.json", {"status": "failed", "reason": "Declared timeout"})
    summary = (
        root
        / "validation"
        / f"routine-{data['identifier']}"
        / "validation_summary.json"
    )
    summary.unlink()
    result = evidence.read_run_evidence(root)
    assert result["status"] == "failed"
    assert result["reason"] == "Declared timeout"
    assert result["failures"]
    assert result["records"] == []
    assert result["final_checkpoint_id"] is None
    assert evidence.read_run_evidence(root / "missing")["status"] == "missing"


def test_snapshot_refuses_a_file_changed_during_read(
    package: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _make_run(package)
    original = evidence._record

    def change_after_read(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        result = original(*args, **kwargs)
        path = data["root"] / "run_details.json"
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(evidence, "_record", change_after_read)
    with pytest.raises(ValueError, match="changed while reading"):
        evidence.read_run_evidence(data["root"])


@pytest.mark.parametrize("sampling", ["deterministic", "stochastic", "unknown"])
def test_sampling_facts_are_verified_and_hashed(
    package: dict[str, Any], sampling: str
) -> None:
    data = _make_run(package, sampling=sampling)
    result = evidence.read_run_evidence(data["root"])
    record = result["records"][0]
    facts = record["sampling_evidence"]
    assert facts["completed_games"] == facts["scheduled_games"] == 20
    assert (
        facts["supported_independent_sampling_units"]
        == {
            "deterministic": 0,
            "stochastic": 10,
            "unknown": None,
        }[sampling]
    )
    assert (record["ci_low"] is not None) is (sampling == "stochastic")
    source = Path(record["pass_paths"][0]).parent / "sampling_facts.json"
    assert {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    } in result["source_files"]


@pytest.mark.parametrize("change", ["task", "pass", "method", "fact", "missing"])
def test_changed_sampling_facts_cannot_support_saved_intervals(
    package: dict[str, Any], change: str
) -> None:
    data = _make_run(package, sampling="stochastic")
    summary = json.loads((data["root"] / "validation_results.json").read_text())[0]
    path = Path(summary["pass_paths"][0]).parent / "sampling_facts.json"
    facts = json.loads(path.read_text())
    if change == "missing":
        path.unlink()
    else:
        if change == "task":
            facts["task_id"] = "e" * 64
        elif change == "pass":
            facts["pass_id"] = "another-pass"
        elif change == "method":
            facts["system_ids"]["team_b"] = "f" * 64
        else:
            facts["methods"]["team_b"]["determinism"] = "deterministic"
            facts["methods"]["team_a"]["determinism"] = "deterministic"
        _write(path, facts)
    with pytest.raises(ValueError, match=r"Sampling facts|Saved sampling evidence"):
        evidence.read_run_evidence(data["root"])


@pytest.mark.parametrize("target", ["schedule", "rows"])
def test_recorded_games_must_use_the_declared_generated_seeds(
    package: dict[str, Any], target: str
) -> None:
    data = _make_run(package)
    summary = json.loads((data["root"] / "validation_results.json").read_text())[0]
    directory = Path(summary["pass_paths"][0])
    path = directory / ("run_details.json" if target == "schedule" else "rows.json")
    content = json.loads(path.read_text())
    if target == "schedule":
        entry = next(iter(content["passes"].values()))
        for episode in entry["episodes"].values():
            episode["seed_id"] += 100
    else:
        for row in content:
            row["seed_id"] += 100
    _write(path, content)
    with pytest.raises(ValueError, match="generated schedule"):
        evidence.read_run_evidence(data["root"])


@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.5, "3"])
def test_confirmation_pair_count_requires_a_positive_plain_integer(
    package: dict[str, Any], bad: object
) -> None:
    with pytest.raises(ValueError, match="positive plain integer"):
        evidence.read_run_evidence(
            package["root"],
            confirmation_seed_pairs=bad,  # pyright: ignore[reportArgumentType]
        )


def test_fresh_confirmation_count_keeps_original_task_counts(
    package: dict[str, Any],
) -> None:
    data = _make_run(package)
    root = data["root"]
    originals = json.loads((root / "validation_results.json").read_text())
    originals.append(_make_task(data, purpose="confirmation"))
    _write(root / "validation_results.json", originals)
    _make_task(
        data,
        purpose="confirmation",
        root_seed=19_044_791,
        seed_pairs=3,
        external=True,
    )
    path = data["external"] / "validation_summary.json"
    with pytest.raises(ValueError, match="declared identity"):
        evidence.read_run_evidence(
            root, confirmation_results=[path], confirmation_root=19_044_791
        )
    result = evidence.read_run_evidence(
        root,
        confirmation_results=[path],
        confirmation_root=19_044_791,
        confirmation_seed_pairs=3,
    )
    assert [row["seed_pairs"] for row in result["records"]] == [2, 2, 3]
    assert [row["games"] for row in result["records"]] == [20, 20, 30]
    assert [row["root"] for row in result["records"]] == [
        data["panel"].roots["routine"],
        data["panel"].roots["confirmation"],
        19_044_791,
    ]
    originals[0] = _make_task(data, seed_pairs=3)
    _write(root / "validation_results.json", originals)
    with pytest.raises(ValueError, match="declared identity"):
        evidence.read_run_evidence(
            root,
            confirmation_results=[path],
            confirmation_root=19_044_791,
            confirmation_seed_pairs=3,
        )


def test_reselection_finishes_with_its_declared_confirmation_pair_count(
    package: dict[str, Any],
) -> None:
    data = _make_run(package)
    root = data["root"]
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    declaration = {
        "name": "Three Pair Confirmation",
        "confirmation_root": 19_044_791,
        "confirmation_seed_pairs": 3,
    }
    pending_dir = root.parent / "pending"
    pending = selection.reselect_checkpoint(
        [root], declaration=declaration, output_dir=pending_dir
    )
    assert pending["status"] == "needs_confirmation"
    assert pending["runs"][0]["winner"] is None
    request = pending["runs"][0]["needs_confirmation"][0]
    assert request["arguments"]["seed_pairs"] == 3
    pending_path = pending_dir / "selection_decision.json"
    pending_bytes = pending_path.read_bytes()
    data["external"] = Path(request["arguments"]["output_dir"])
    summary = _make_task(
        data,
        purpose="confirmation",
        root_seed=request["arguments"]["root_seed"],
        seed_pairs=request["arguments"]["seed_pairs"],
        external=True,
    )
    assert all(summary[key] == value for key, value in request["expected_task"].items())
    completed_dir = root.parent / "completed"
    completed = selection.reselect_checkpoint(
        [root],
        declaration={
            **declaration,
            "previous_decision": str(pending_path),
            "confirmation_results": [str(data["external"] / "validation_summary.json")],
        },
        output_dir=completed_dir,
    )
    winner = completed["runs"][0]["winner"]
    assert completed["status"] == "complete"
    assert completed["previous_decision_id"] == pending["decision_id"]
    assert winner["seed_pairs"] == 3 and winner["games"] == 30
    assert winner["checkpoint_id"] == data["identifier"]
    assert winner["artifact_id"] != winner["checkpoint_id"]
    assert winner["actor_digest"] == "a" * 64
    assert pending_path.read_bytes() == pending_bytes
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (
        selection.read_selection_decision(completed_dir / "selection_decision.json")
        == completed
    )


def _child_validation(data: dict[str, Any], **changes: object) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "panel_digest": data["panel"].digest,
        "panel_schema_version": data["panel"].schema_version,
        "roots": dict(data["panel"].roots),
        "red_zone_depth": 5.0,
        "routine_seed_pairs": 2,
        "confirmation_seed_pairs": 2,
        "inherit_parent_candidates": True,
        "allow_different_roots": False,
        **changes,
    }


def test_inherited_candidate_keeps_original_owner_without_parent_log_copy(
    package: dict[str, Any],
) -> None:
    data = _make_run(package)
    root = data["root"]
    boundary = root / "checkpoints" / data["identifier"]
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    declaration = _child_validation(data)
    references = evidence.freeze_inherited_candidates(boundary, declaration=declaration)
    checked = evidence.read_inherited_candidates(references, declaration=declaration)
    assert checked["actors"][data["identifier"]]["actor_path"] == str(
        data["actor_path"]
    )
    assert checked["actors"][data["identifier"]]["run_id"] == "training-run"
    assert checked["records"][0]["checkpoint_id"] == data["identifier"]
    assert checked["records"][0]["root"] == declaration["roots"]["routine"]
    assert checked["used_roots"] == [declaration["roots"]["routine"]]
    assert all(
        Path(item["path"]).name not in {"validation_results.json", "status.json"}
        for item in checked["source_files"]
    )
    assert before == {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    _write(root / "validation_results.json", [])
    _write(root / "status.json", {"status": "incomplete"})
    assert (
        evidence.read_inherited_candidates(references, declaration=declaration)
        == checked
    )


def test_inherited_candidate_corruption_and_changed_panel_are_distinct(
    package: dict[str, Any],
) -> None:
    data = _make_run(package)
    boundary = data["root"] / "checkpoints" / data["identifier"]
    declaration = _child_validation(data)
    references = evidence.freeze_inherited_candidates(boundary, declaration=declaration)
    changed_panel = _child_validation(data, panel_digest="different-panel")
    excluded = evidence.read_inherited_candidates(references, declaration=changed_panel)
    assert not excluded["records"] and not excluded["actors"]
    assert excluded["used_roots"] == [declaration["roots"]["routine"]]
    path = Path(references[0]["tasks"][0]["summary_path"])
    row = json.loads(path.read_text())
    row["score"] = 0.17
    _write(path, row)
    with pytest.raises(ValueError, match="Inherited source changed"):
        evidence.read_inherited_candidates(references, declaration=changed_panel)


def test_inherited_root_difference_needs_permission_and_fresh_confirmation(
    package: dict[str, Any],
) -> None:
    data = _make_run(package)
    boundary = data["root"] / "checkpoints" / data["identifier"]
    declaration = _child_validation(data)
    references = evidence.freeze_inherited_candidates(boundary, declaration=declaration)
    changed = _child_validation(data, roots={**declaration["roots"], "routine": 17})
    with pytest.raises(ValueError, match="Different inherited roots"):
        evidence.read_inherited_candidates(references, declaration=changed)
    changed["allow_different_roots"] = True
    assert evidence.read_inherited_candidates(references, declaration=changed)[
        "records"
    ]
    changed["roots"] = {
        "routine": 17,
        "initialization": 18,
        "confirmation": declaration["roots"]["routine"],
    }
    with pytest.raises(ValueError, match="every routine"):
        evidence.read_inherited_candidates(references, declaration=changed)


def test_inherited_reference_cannot_insert_an_unbound_task(
    package: dict[str, Any],
) -> None:
    data = _make_run(package)
    boundary = data["root"] / "checkpoints" / data["identifier"]
    declaration = _child_validation(data)
    references = evidence.freeze_inherited_candidates(boundary, declaration=declaration)
    references[0]["tasks"][0]["checkpoint_id"] = "f" * 64
    with pytest.raises(ValueError, match="outside the chosen active ancestry"):
        evidence.read_inherited_candidates(references, declaration=declaration)


def test_fork_before_parent_end_excludes_later_confirmation_work(
    package: dict[str, Any],
) -> None:
    data = _make_run(package, total_env_steps=64)
    root = data["root"]
    original = json.loads((root / "validation_results.json").read_text())
    original.append(_make_task(data, purpose="confirmation"))
    _write(root / "validation_results.json", original)
    references = evidence.freeze_inherited_candidates(
        root / "checkpoints" / data["identifier"], declaration=_child_validation(data)
    )
    result = evidence.read_inherited_candidates(
        references, declaration=_child_validation(data)
    )
    assert [row["purpose"] for row in result["records"]] == ["routine"]


def test_repeated_lineage_reuses_only_completed_call_local_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = validation.FrozenPanel(
        tmp_path / "panel.json",
        "panel",
        (),
        True,
        schema_version=2,
        roots={"routine": 11, "initialization": 12, "confirmation": 13},
    )
    declared = _child_validation({"panel": panel})
    references: list[dict[str, Any]] = []
    contexts: dict[
        Path,
        tuple[
            Path, dict[str, Any], dict[str, Any], dict[str, Any], validation.FrozenPanel
        ],
    ] = {}
    all_sources: dict[Path, str] = {}
    for index in range(5):
        root = tmp_path / str(index)
        identifier = f"{index:064x}"
        checkpoint = root / "checkpoints" / f"boundary-{index}"
        metadata = root / "run_details.json"
        path = root / "validation" / f"routine-{identifier}" / "validation_summary.json"
        _write(metadata, {"index": index})
        task = {
            "checkpoint_id": identifier,
            "task_id": f"task-{index}",
            "purpose": "routine",
            "root": 11,
            "panel_digest": "panel",
            "red_zone_depth": 5.0,
            "selection_schema_version": 2,
            "maps": list(validation.VALIDATION_MAPS),
        }
        _write(path, task)
        all_sources.update(
            {source: evidence._hash(source) for source in (metadata, path)}
        )
        run = {
            "run_id": str(index),
            "config": {"total_env_steps": 4},
            "continuation": {"inherited_candidates": list(references)},
            "validation_declaration": declared,
        }
        contexts[checkpoint] = (
            root,
            run,
            {"checkpoint_id": checkpoint.name, "counters": {"env_steps": 4}},
            {identifier: {}},
            panel,
        )
        references.append(
            {
                "schema_version": 1,
                "parent_checkpoint": str(checkpoint),
                "parent_checkpoint_id": checkpoint.name,
                "run_dir": str(root),
                "run_id": str(index),
                "tasks": [
                    {
                        "summary_path": str(path),
                        "checkpoint_id": identifier,
                        "task_id": task["task_id"],
                        "purpose": "routine",
                    }
                ],
                "source_files": [
                    {"path": str(source), "sha256": digest}
                    for source, digest in sorted(all_sources.items())
                ],
            }
        )
    calls = 0

    def context(
        checkpoint: Path, snapshot: evidence._Snapshot
    ) -> tuple[
        Path, dict[str, Any], dict[str, Any], dict[str, Any], validation.FrozenPanel
    ]:
        nonlocal calls
        calls += 1
        result = contexts[checkpoint]
        snapshot.track(result[0] / "run_details.json")
        return result

    def record(
        path: Path,
        original: object,
        run: object,
        panel: object,
        actors: dict[str, Any],
        root: Path,
        snapshot: evidence._Snapshot,
        confirmation_root: object,
    ) -> dict[str, Any]:
        value = dict(snapshot.read(path))
        actors[value["checkpoint_id"]] = {
            "actor_path": str(root / "actors" / value["checkpoint_id"])
        }
        return value

    monkeypatch.setattr(evidence, "_reference_context", context)
    monkeypatch.setattr(evidence, "_record", record)
    implementation = evidence.read_inherited_candidates

    def without_memo(
        refs: Sequence[Mapping[str, Any]],
        *,
        declaration: Mapping[str, Any],
        _active: frozenset[str] = frozenset(),
        _verified: dict[str, tuple[dict[str, Any], frozenset[str]]] | None = None,
    ) -> dict[str, Any]:
        fresh: dict[str, tuple[dict[str, Any], frozenset[str]]] = {}
        result = implementation(
            refs, declaration=declaration, _active=_active, _verified=fresh
        )
        if _verified is not None:
            _verified.update(fresh)
        return result

    monkeypatch.setattr(evidence, "read_inherited_candidates", without_memo)
    before = without_memo(references, declaration=declared)
    uncached_calls = calls
    calls = 0
    monkeypatch.setattr(evidence, "read_inherited_candidates", implementation)
    after = implementation(references, declaration=declared)
    assert after == before
    assert uncached_calls == 31 and calls == 15
    print(
        f"Five generations: context reads {uncached_calls} -> {calls}; "
        f"all {len(all_sources)} source files retained"
    )
    first = tmp_path / "0/run_details.json"
    _write(first, {"index": "changed"})
    with pytest.raises(ValueError, match="Inherited source changed"):
        implementation(references, declaration=declared)
    _write(first, {"index": 0})
    contexts[Path(references[-1]["parent_checkpoint"])][1]["continuation"][
        "inherited_candidates"
    ] = [references[-1]]
    with pytest.raises(ValueError, match="cyclic"):
        implementation(references, declaration=declared)


def test_inherited_reference_rejects_boolean_schema(package: dict[str, Any]) -> None:
    data = _make_run(package)
    declaration = _child_validation(data)
    references = evidence.freeze_inherited_candidates(
        data["root"] / "checkpoints" / data["identifier"], declaration=declaration
    )
    references[0]["schema_version"] = True
    with pytest.raises(ValueError, match="Invalid inherited candidate reference"):
        evidence.read_inherited_candidates(references, declaration=declaration)
